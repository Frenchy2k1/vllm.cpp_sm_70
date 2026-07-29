// Ported from: vllm/csrc/libtorch_stable/cache_kernels.cu @ e24d1b24
//   (reshape_and_cache_flash — the "auto"/contiguous-heads write semantics
//    only; the NHD cache layout is FlashAttentionBackend::get_kv_cache_shape's
//    (num_blocks, 2, block_size, num_kv_heads, head_size), NOT the HND cpu_attn
//    layout — see .agents/discipline.md and the M1.6 Task-2 layout trap note).
#include <cstdint>
#include <cstring>

#include "vt/dtype.h"
#include "vt/ops.h"

namespace vt::cpu {
namespace {

// Write each new per-token K/V into the paged NHD cache at its slot id. The
// "auto" cache path is a raw element copy (cache dtype == k/v dtype); we copy
// bytes so f32/f16/bf16 are all bit-exact (upstream KV_T == CACHE_T in auto).
//
// CRITICAL (M1.6 Task-2): the destination is indexed from the TENSOR STRIDES,
// not from k_cache.shape. get_kv_cache_shape hands us ONE (num_blocks, 2,
// block_size, H, D) allocation and K/V are its two dim-1 unbind slices, so the
// block stride is 2*bs*H*D (NOT bs*H*D) and K/V are NON-contiguous rank-4 views.
// This mirrors pinned csrc/libtorch_stable/cache_kernels.cu @ e24d1b24:
//   host reshape_and_cache_flash (~L797-801): key_stride = key.stride(0);
//     block_stride = key_cache.stride(0); page_stride = key_cache.stride(1);
//     head_stride = key_cache.stride(2);
//   kernel reshape_and_cache_flash_kernel (~L337-347): key_src = key +
//     token_idx*key_stride; key_dst = key_cache + block_idx*block_stride +
//     block_offset*page_stride; is_contiguous_heads = (head_stride == head_size).
// The wrapper guarantees head_stride == head_size && elem stride == 1 (the NHD
// unbind slice), so the per-token page is one dense run of n_elems inside the
// block — the is_contiguous_heads fast path, i.e. a single memcpy per token.
void ReshapeAndCacheKernel(Queue&, const Tensor& k, const Tensor& v, Tensor& k_cache,
                           Tensor& v_cache, const Tensor& slot_mapping) {
  const int64_t num_slots = slot_mapping.shape[0];
  const int64_t block_size = k_cache.shape[1];
  const int64_t num_kv_heads = k_cache.shape[2];
  const int64_t head_size = k_cache.shape[3];
  const int64_t n_elems = num_kv_heads * head_size;  // one token's page (NHD)
  // Destination strides come from the tensors (unbind-slice aware), each cache
  // with ITS OWN strides. Source token stride comes from k/v.stride(0); the
  // per-token [H, D] payload is packed (input k/v are contiguous rows).
  const int64_t k_block_stride = k_cache.stride[0];
  const int64_t k_page_stride = k_cache.stride[1];
  const int64_t v_block_stride = v_cache.stride[0];
  const int64_t v_page_stride = v_cache.stride[1];
  const int64_t k_tok_stride = k.stride[0];
  const int64_t v_tok_stride = v.stride[0];
  const size_t elem = SizeOf(k.dtype);

  const int64_t* slots = slot_mapping.Ptr<int64_t>();
  const auto* ksrc = static_cast<const uint8_t*>(k.data);
  const auto* vsrc = static_cast<const uint8_t*>(v.data);
  auto* kdst = static_cast<uint8_t*>(k_cache.data);
  auto* vdst = static_cast<uint8_t*>(v_cache.data);
  const size_t bytes = static_cast<size_t>(n_elems) * elem;

  for (int64_t t = 0; t < num_slots; ++t) {
    const int64_t slot = slots[t];
    if (slot < 0) continue;  // padded token → skip (upstream NOTE: slot can be -1)
    const int64_t block = slot / block_size;
    const int64_t offset = slot % block_size;
    const int64_t kdst_off = block * k_block_stride + offset * k_page_stride;  // elements
    const int64_t vdst_off = block * v_block_stride + offset * v_page_stride;
    const int64_t ksrc_off = t * k_tok_stride;
    const int64_t vsrc_off = t * v_tok_stride;
    std::memcpy(kdst + static_cast<size_t>(kdst_off) * elem,
                ksrc + static_cast<size_t>(ksrc_off) * elem, bytes);
    std::memcpy(vdst + static_cast<size_t>(vdst_off) * elem,
                vsrc + static_cast<size_t>(vsrc_off) * elem, bytes);
  }
}

// MLA cache write — the CPU REFERENCE for vt::ConcatAndCacheMla, ported 1:1 from
// vllm/csrc/libtorch_stable/cache_kernels.cu:401-442
// (`concat_and_cache_mla_kernel`) @ e24d1b24. Upstream is literally two strided
// `copy(...)` lambda calls into one destination entry:
//   copy(kv_c, kv_cache, kv_c_stride, block_stride, kv_lora_rank, 0);       (:440)
//   copy(k_pe, kv_cache, k_pe_stride, block_stride, pe_dim, kv_lora_rank);  (:441)
// with dst_idx = block_idx*block_stride + block_offset*entry_stride + i + offset
// (`:431-433`) — note upstream passes `block_stride` as the SOURCE-side
// dst_stride argument and never uses it, the real destination stride pair is
// (block_stride, entry_stride). We reproduce the destination arithmetic exactly.
//
// The "auto" path (kv_dt == kAuto, `:435-436`) is a raw element copy, so we copy
// BYTES: f32/f16/bf16 are all bit-exact. The fp8 `scaled_convert` branch
// (`:437-440`) is out of scope and is refused by the op wrapper's dtype check.
// This is a CONCATENATION, not two writes to two tensors: the latent occupies
// columns [0, kv_lora_rank) of the single 576-wide entry and the decoupled rope
// part occupies [kv_lora_rank, kv_lora_rank + pe_dim).
void ConcatAndCacheMlaKernel(Queue&, const Tensor& kv_c, const Tensor& k_pe, Tensor& kv_cache,
                             const Tensor& slot_mapping) {
  const int64_t num_slots = slot_mapping.shape[0];
  const int64_t block_size = kv_cache.shape[1];
  const int64_t kv_lora_rank = kv_c.shape[1];
  const int64_t pe_dim = k_pe.shape[1];
  // Destination strides from the TENSOR (upstream `:882-884`), so a strided
  // cache view works without a copy.
  const int64_t block_stride = kv_cache.stride[0];
  const int64_t entry_stride = kv_cache.stride[1];
  const int64_t kv_c_stride = kv_c.stride[0];
  const int64_t k_pe_stride = k_pe.stride[0];
  const size_t elem = SizeOf(kv_c.dtype);

  const int64_t* slots = slot_mapping.Ptr<int64_t>();
  const auto* csrc = static_cast<const uint8_t*>(kv_c.data);
  const auto* psrc = static_cast<const uint8_t*>(k_pe.data);
  auto* dst = static_cast<uint8_t*>(kv_cache.data);

  for (int64_t t = 0; t < num_slots; ++t) {
    const int64_t slot = slots[t];
    if (slot < 0) continue;  // padded token → skip (upstream `:419-422`)
    const int64_t block = slot / block_size;
    const int64_t offset = slot % block_size;
    const int64_t entry = block * block_stride + offset * entry_stride;  // elements
    std::memcpy(dst + static_cast<size_t>(entry) * elem,
                csrc + static_cast<size_t>(t * kv_c_stride) * elem,
                static_cast<size_t>(kv_lora_rank) * elem);
    std::memcpy(dst + static_cast<size_t>(entry + kv_lora_rank) * elem,
                psrc + static_cast<size_t>(t * k_pe_stride) * elem,
                static_cast<size_t>(pe_dim) * elem);
  }
}

// Load one source K/V element as f32 (the model dtype: f32/f16/bf16).
float LoadSrcF32(const Tensor& t, int64_t elem_off) {
  switch (t.dtype) {
    case DType::kF32: return t.Ptr<float>()[elem_off];
    case DType::kF16: return F16ToF32(t.Ptr<uint16_t>()[elem_off]);
    case DType::kBF16: return BF16ToF32(t.Ptr<uint16_t>()[elem_off]);
    default: VT_CHECK(false, "reshape_and_cache_fp8 LoadSrcF32: unsupported source dtype");
      return 0.0f;
  }
}

// fp8 KV STORE — the fp8 sibling of ReshapeAndCacheKernel. Ported from the fp8
// branch of vllm reshape_and_cache_flash_kernel (cache_kernels.cu:314-401) +
// CopyWithScaleOp (:241-252): each element is stored as Quantize(hp / scale)
// (the fp8::scaled_convert scale convention, quant_utils.cuh:296-308). The cache
// pages are 1-byte fp8 (kI8), so the destination element size is 1; the strides
// are the same NHD unbind-slice arithmetic as the auto path. Per-tensor k/v
// scales (BaseKVCacheMethod, kv_cache.py:108-191); per-head is a later brick.
void ReshapeAndCacheFp8Kernel(Queue&, const Tensor& k, const Tensor& v, Tensor& k_cache,
                              Tensor& v_cache, const Tensor& slot_mapping,
                              Fp8KVCacheDataType kind, float k_scale, float v_scale) {
  VT_CHECK(kind == Fp8KVCacheDataType::kFp8E4M3,
           "reshape_and_cache_fp8: CPU kernel implements fp8_e4m3 only");
  const int64_t num_slots = slot_mapping.shape[0];
  const int64_t block_size = k_cache.shape[1];
  const int64_t num_kv_heads = k_cache.shape[2];
  const int64_t head_size = k_cache.shape[3];
  const int64_t n_elems = num_kv_heads * head_size;  // one token's page (NHD)
  const int64_t k_block_stride = k_cache.stride[0];
  const int64_t k_page_stride = k_cache.stride[1];
  const int64_t v_block_stride = v_cache.stride[0];
  const int64_t v_page_stride = v_cache.stride[1];
  const int64_t k_tok_stride = k.stride[0];
  const int64_t v_tok_stride = v.stride[0];

  const int64_t* slots = slot_mapping.Ptr<int64_t>();
  auto* kdst = k_cache.Ptr<uint8_t>();  // fp8 bytes (kI8)
  auto* vdst = v_cache.Ptr<uint8_t>();

  for (int64_t t = 0; t < num_slots; ++t) {
    const int64_t slot = slots[t];
    if (slot < 0) continue;  // padded token → skip (upstream: slot can be -1)
    const int64_t block = slot / block_size;
    const int64_t offset = slot % block_size;
    const int64_t kdst_base = block * k_block_stride + offset * k_page_stride;  // bytes/elems
    const int64_t vdst_base = block * v_block_stride + offset * v_page_stride;
    const int64_t ksrc_base = t * k_tok_stride;
    const int64_t vsrc_base = t * v_tok_stride;
    for (int64_t e = 0; e < n_elems; ++e) {
      kdst[kdst_base + e] = StoreKvFp8E4M3(LoadSrcF32(k, ksrc_base + e), k_scale);
      vdst[vdst_base + e] = StoreKvFp8E4M3(LoadSrcF32(v, vsrc_base + e), v_scale);
    }
  }
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kReshapeAndCache, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<ReshapeAndCacheFn>(&ReshapeAndCacheKernel)));
    RegisterOp(
        OpId::kReshapeAndCacheFp8, DeviceType::kCPU,
        reinterpret_cast<void*>(static_cast<ReshapeAndCacheFp8Fn>(&ReshapeAndCacheFp8Kernel)));
    RegisterOp(
        OpId::kConcatAndCacheMla, DeviceType::kCPU,
        reinterpret_cast<void*>(static_cast<ConcatAndCacheMlaFn>(&ConcatAndCacheMlaKernel)));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
