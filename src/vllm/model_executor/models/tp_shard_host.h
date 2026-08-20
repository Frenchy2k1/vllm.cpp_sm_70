// Per-rank host-shard helpers for the tp>1 routes of the dense model families
// (BACKEND-DISTRIBUTED-TP breadth wave; the SM70 2xV100 gate-proven
// primitives from src/vt/cuda/nccl_communicator.cu). Each helper runs the
// EXACT distributed math the corresponding qwen3_5 family already gates:
//   * the dense SwiGLU MLP: each group rank owns I/W of the intermediate,
//     per-token AllReduceSum of the [H] partial (vt_cuda_mlp_shard_run(_bf16));
//   * the paged GQA attention: KV-head-split caches + num/den AllReduceSum
//     over the group (vt_cuda_attn_kv_shard_paged_run);
// and returns the FULL result on the calling rank (exit identical on every
// rank), so the ordinary per-rank residual / o_proj / lm_head compute stays
// the replicated tp1-math — the exact "exit-all-reduced" discipline the
// qwen3_5 dense/MoE forwards gate on.
//
// These helpers are reached ONLY from call sites guarded on tp_size()>1, so a
// null tp — the single-GPU, byte-identical path — NEVER reaches them.
#pragma once
#ifndef VT_NCCL

#include <stdexcept>
#include <string>

namespace vllm {
namespace shard_host {

// A non-NCCL build cannot reach the group primitives; refuse by name rather
// than pretending (never silent tp1).
inline void TpNcclAbsent(const char* family) {
  throw std::runtime_error(std::string(family) +
                           ": tp>1 requires an NCCL build (VLLM_CPP_NCCL)");
}

}  // namespace shard_host
}  // namespace vllm

#else  // VT_NCCL

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"  // Dev/DBuf/ResidentWeight
#include "vllm/model_executor/models/qwen3_5_weights.h"    // OwnedTensor
#include "vllm/v1/attention/backend.h"                     // CommonAttentionMetadata

// The group primitives (src/vt/cuda/nccl_communicator.cu, VT_NCCL-only).
// Declarations mirror qwen3_5.cpp verbatim; the implementations live in the
// transport TU and are linked by the same seam the SM70 wave ran.
extern "C" int vt_cuda_mlp_shard_run(int O, int H, int I, const float* x,
                                     const float* gate, const float* up,
                                     const float* down, float* out);
extern "C" int vt_cuda_mlp_shard_run_bf16(int O, int H, int I,
                                          const uint16_t* x16, const uint16_t* gu16,
                                          const uint16_t* dn16, float* out);
extern "C" int vt_cuda_attn_kv_shard_paged_run(int T, int Hq, int Hkv, int D,
                                               int block, int bt_cols, int num_blocks,
                                               int kv_dtype, const float* q,
                                               const float* kv_new,
                                               const int64_t* slot_mapping,
                                               const int32_t* block_table,
                                               const int32_t* seq_lens,
                                               void* kv0, float* out);

namespace vllm {
namespace shard_host {

using dense_attn::DBuf;
using dense_attn::Dev;
using dense_attn::ResidentWeight;
using v1::CommonAttentionMetadata;
using vt::DType;
using vt::Tensor;

namespace {

// Copy a device tensor to host f32: bf16 upcasts bit-exactly, f32 verbatim.
void HostToF32(Dev d, const Tensor& src, std::vector<float>& outv) {
  if (src.dtype == DType::kBF16) {
    std::vector<uint16_t> tmp(outv.size());
    d.b.Copy(d.q, tmp.data(), src.data, tmp.size() * sizeof(uint16_t));
    for (size_t i = 0; i < outv.size(); ++i) outv[i] = vt::BF16ToF32(tmp[i]);
  } else {
    d.b.Copy(d.q, outv.data(), src.data, outv.size() * sizeof(float));
  }
}

// Raw bytes of a (possibly host-released) weight: the resident copy is read
// through the device Tensor, the host copy otherwise — the qwen3_5
// DenseMlpBlock tp>1 host_data discipline.
const uint8_t* WeightHostBytes(Dev d, const OwnedTensor& w) {
  if (w.HasHostBytes())
    return reinterpret_cast<const uint8_t*>(w.bytes.data());
  return reinterpret_cast<const uint8_t*>(ResidentWeight(d, w).data);
}

}  // namespace

// <family> paged attention at tp>1 — PURE-DECODE steps only (the same gate
// qwen3_5's FullAttnBlockPaged applies). `q3` is the FULL [T,Hq,Dh] query,
// `kw`/`vw` this step's k/v already in the cache dtype (bf16 or f32 — what
// the tp1 ReshapeAndCache would have stored); the primitive partitions the KV
// cache heads across the group, performs the reduced softmax attention and
// returns the FULL [T,Hq,Dh] attention in f32 — the identical tensor every
// tp1 PagedAttention would produce. Callers cast to their activation dtype
// and run the usual full-width o_proj.
inline DBuf TpPagedAttentionHost(Dev& d, const char* family, const Tensor& q3,
                                 const Tensor& kw, const Tensor& vw,
                                 const PagedKvCache& kv,
                                 const CommonAttentionMetadata& meta,
                                 int64_t T, int64_t Hq, int64_t Hkv, int64_t Dh) {
  if (meta.num_actual_tokens != meta.num_reqs || T != meta.num_reqs)
    throw std::runtime_error(std::string(family) +
                             " paged attention: tp>1 per-rank paged-KV decode is "
                             "landed for pure-decode steps only (T == num_reqs); "
                             "prefill+mixed with tp is a pending wave — refusing "
                             "to run tp1 math");
  const int64_t qn = T * Hq * Dh, vn = T * Hkv * Dh;
  std::vector<float> qh((size_t)qn), kh((size_t)vn), vh((size_t)vn),
      ohh((size_t)qn, 0.f);
  HostToF32(d, q3, qh);
  HostToF32(d, kw, kh);
  HostToF32(d, vw, vh);
  // One host interleaved [T, 2*Hkv*D] buffer: the first T*Hkv*D floats are the
  // K plane (k0,v0,k1,v1), the last T*Hkv*D the V plane at T*2*Hkv*D.
  const size_t kb = (size_t)T * 2 * Hkv * Dh;
  std::vector<float> kv_interp(2 * kb);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t h = 0; h < Hkv; ++h) {
      const size_t s = ((size_t)t * Hkv + h) * Dh;
      const size_t dst = ((size_t)t * 2 * Hkv + h) * Dh;
      for (int64_t md = 0; md < Dh; ++md) {
        kv_interp[dst + (size_t)md] = kh[s + (size_t)md];
        kv_interp[kb + dst + (size_t)md] = vh[s + (size_t)md];
      }
    }
  d.b.Synchronize(d.q);
  const int rc = vt_cuda_attn_kv_shard_paged_run(
      (int)T, (int)Hq, (int)Hkv, (int)Dh, (int)kv.block_size,
      (int)meta.block_table_num_cols, (int)kv.num_blocks,
      kv.dtype == DType::kBF16 ? 0 : 1, qh.data(), kv_interp.data(),
      meta.slot_mapping.data(), meta.block_table_tensor.data(),
      meta.seq_lens.data(), kv.data, ohh.data());
  if (rc == 2)
    throw std::runtime_error(std::string(family) +
                             " paged attention: tp>1 requires >=2 GPUs (fault loudly)");
  if (rc != 0)
    throw std::runtime_error(std::string(family) +
                             " paged attention: tp>1 per-rank paged KV shard failed (rc " +
                             std::to_string(rc) + ")");
  DBuf attn_tp(d, DType::kF32, {T, Hq, Dh});
  d.b.Copy(d.q, attn_tp.ptr(), ohh.data(), (size_t)qn * sizeof(float));
  d.b.Synchronize(d.q);
  return attn_tp;
}

// <family> dense SwiGLU MLP at tp>1. `gate_up` is the raw-NK merged [2I,H]
// bf16 owner (rows [0,I) = gate, [I,2I) = up), `down` the raw-NK [H,I] bf16
// owner — the layout every dense family here loads and the tp1 path consumes.
// Runs the group sharded silu-gate MLP per token (each rank owns I/W of the
// intermediate; the group AllReduceSum yields the reduced [H] on rank 0) and
// returns the reduced [T,H] bf16 — token-identical to tp1.
inline DBuf TpSwiGluHost(Dev d, const char* family, const OwnedTensor& gate_up,
                         const OwnedTensor& down, const Tensor& dh,
                         int64_t T, int64_t H, int64_t I) {
  DBuf red(d, DType::kBF16, {T, H});
  std::vector<uint16_t> hid((size_t)T * H);
  d.b.Copy(d.q, hid.data(), (const void*)dh.data,
           (size_t)T * H * sizeof(uint16_t));
  d.b.Synchronize(d.q);
  const auto* gu_b = WeightHostBytes(d, gate_up);  // RawNK [2I, H]
  const auto* dn_b = WeightHostBytes(d, down);     // RawNK [H, I]
  std::vector<uint8_t> host((size_t)T * H * 2);
  for (int64_t t = 0; t < T; ++t) {
    std::vector<float> o((size_t)H);
    const int rc = vt_cuda_mlp_shard_run_bf16(
        (int)H, (int)H, (int)I,
        reinterpret_cast<const uint16_t*>(hid.data() + (size_t)t * H),
        reinterpret_cast<const uint16_t*>(gu_b),
        reinterpret_cast<const uint16_t*>(dn_b), o.data());
    if (rc == 2)
      throw std::runtime_error(std::string(family) +
                               " MLP: tp>1 requires >=2 GPUs (fault loudly)");
    if (rc != 0)
      throw std::runtime_error(std::string(family) +
                               " MLP: tp>1 shard mismatched host reference");
    auto* row16 = reinterpret_cast<uint16_t*>(host.data() + (size_t)t * H * 2);
    for (int64_t h = 0; h < H; ++h) row16[(size_t)h] = vt::F32ToBF16(o[(size_t)h]);
  }
  d.b.Copy(d.q, red.ptr(), (const void*)host.data(),
           (size_t)T * H * sizeof(uint16_t));
  d.b.Synchronize(d.q);
  return red;
}

}  // namespace shard_host
}  // namespace vllm

#endif  // VT_NCCL