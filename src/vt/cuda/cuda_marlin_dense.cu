// vllm.cpp — DENSE Marlin NVFP4/MXFP4 W4A16 GEMM drop-in (vt::Tensor launcher).
//
// Torch-free host launcher for the vendored DENSE Marlin kernel (src/vt/cuda/
// marlin/libtorch_stable/quantization/marlin/, a 1:1 lift of vLLM's dense
// marlin.cu @ 555967922). It mirrors the a16 (weight-only) branch of vLLM's
// `marlin_gemm` (marlin.cu:545): b_type=kFE2M1f + s_type=kFE4M3fn (NVFP4) or
// s_type=kFE8M0fnu (MXFP4), bf16 activation/output, no act-order/zero-point/bias.
// All those irrelevant branches are dropped; the compute call into
// marlin::marlin_mm is the verbatim vendored dispatcher (marlin_mm_dense.cu).
//
// This is the BYTE-PRESERVING replacement for the single-expert MoE-marlin route
// the dense E=1 projections use today (dense_nvfp4_gemm.h): the DENSE kernel's
// direct-A, tile-per-CTA grid + its own dense fp32 C_tmp reduce ARE vLLM's own
// numerics, so it does not incur the one-bf16-ULP shift the MoE par regrouping
// does (row QUANT-CT-MXFP4-MARLIN-STRUCT / #50 / #54).
//
// Weights MUST be pre-repacked into Marlin's interleaved layout with processed
// fp8 block scales + per-tensor global scale — the SAME resident the MoE route
// builds (dense_nvfp4_gemm.h BuildMarlinDenseResident); the repack permute is
// vLLM's shared marlin_permute for both dense and MoE.
//
// Isolated TU (heavy templated kernel). Gated by VT_MARLIN_NVFP4.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "core/scalar_type.hpp"
#include "libtorch_stable/quantization/marlin/marlin_mm_dense.h"

#include "vt/cuda/graph_safe_scratch.h"
#include "vt/ops.h"

namespace vt::cuda {
namespace {

// max_thread_n from the vendored marlin.cuh:28 (dense C_tmp reduce scratch upper
// bound; vLLM marlin.cu:716 sizes c_tmp as sms * max_m_block_size * max_thread_n).
constexpr int kMarlinMaxThreadN = 256;

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: marlin_dense: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

// Persistent per-stream C_tmp workspace pool (VT_MARLIN_WS_POOL, default ON).
// Same rationale as cuda_moe_marlin.cu: vLLM allocates c_tmp per call through
// PyTorch's CACHING allocator (a cheap pool hit, not a raw cudaMalloc); a raw
// per-GEMM cudaMallocAsync/cudaFreeAsync serializes on the forward host thread
// and is a steady-state decode idle. This mirrors the caching allocator with a
// grown-on-demand per-stream buffer. c_tmp is scratch the kernel fully writes
// before it reads (vLLM uses new_empty; no zero-on-entry invariant), so reuse is
// race-free under the forward's single-stream ordering. The RETIRE-not-free on
// regrow keeps a captured decode graph's baked c_tmp pointer valid across a later
// larger forward (graph_safe_scratch.h). VT_MARLIN_WS_POOL=0 restores per-call.
bool MarlinWsPoolEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_MARLIN_WS_POOL");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

float* EnsureCtmp(cudaStream_t s, size_t bytes) {
  struct Scratch {
    void* p = nullptr;
    size_t cap = 0;
  };
  static std::mutex mu;
  static std::unordered_map<cudaStream_t, Scratch> pool;
  std::lock_guard<std::mutex> lk(mu);
  Scratch& sc = pool[s];
  if (bytes > sc.cap) {
    RetireGraphScratch(sc.p);
    Check(cudaMallocAsync(&sc.p, bytes, s), "cudaMallocAsync c_tmp (pool)");
    sc.cap = bytes;
  }
  return static_cast<float*>(sc.p);
}

// vt::MarlinDenseGemm registered kernel.
void MarlinDenseGemmKernelCuda(Queue& q, Tensor& c, const Tensor& a, const Tensor& b_q_weight,
                               const Tensor& b_scales, const Tensor& global_scale,
                               Tensor& workspace, const MarlinDenseArgs& args) {
  cudaStream_t s = AsStream(q);
  const int dev = q.device.index;

  // NVFP4 W4A16, bf16 activation/output; OR MXFP4 W4A16 when args.mxfp4 (E8M0
  // scales => s_type kFE8M0fnu, group_size 32 => group_blocks 2, NO global scale).
  const vllm::ScalarType a_type = vllm::kBFloat16;
  const vllm::ScalarType b_type = vllm::kFE2M1f;
  const vllm::ScalarType c_type = vllm::kBFloat16;
  const vllm::ScalarType s_type = args.mxfp4 ? vllm::kFE8M0fnu : vllm::kFE4M3fn;

  const int size_m = args.size_m;
  const int size_n = args.size_n;
  const int size_k = args.size_k;
  const int group_size = args.group_size;  // 16 (nvfp4) or 32 (mxfp4)
  const int num_groups = size_k / group_size;
  // MXFP4 has NO global scale — the dense kernel reads global_scale_ptr only under
  // (b_type==kFE2M1f && s_type==kFE4M3fn), so pass nullptr on the mxfp4 path.
  void* global_scale_ptr = args.mxfp4 ? nullptr : global_scale.data;

  int sms = -1;
  Check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev),
        "cudaDeviceGetAttribute(sms)");

  // C_tmp for the fp32 global reduce (use_fp32_reduce && !use_atomic_add). Size
  // per vLLM marlin.cu:713-716: sms * min(ceil(size_m/16)*16, 64) * max_thread_n.
  const bool use_atomic_add = false;
  const bool use_fp32_reduce = true;
  int max_m_block_size = (size_m + 16 - 1) / 16 * 16;
  if (max_m_block_size > 64) max_m_block_size = 64;
  const int64_t c_tmp_elems =
      static_cast<int64_t>(sms) * max_m_block_size * kMarlinMaxThreadN;
  const size_t c_tmp_bytes = static_cast<size_t>(c_tmp_elems) * sizeof(float);
  float* c_tmp = nullptr;
  bool c_tmp_pooled = false;
  if (MarlinWsPoolEnabled()) {
    c_tmp = EnsureCtmp(s, c_tmp_bytes);  // persistent per-stream, reused
    c_tmp_pooled = true;
  } else {
    Check(cudaMallocAsync(&c_tmp, c_tmp_bytes, s), "cudaMallocAsync c_tmp");
  }

  // lda = A.stride(0) = size_k (a is [size_m, size_k] contiguous, row-major).
  marlin::marlin_mm(
      a.data, b_q_weight.data, c.data, c_tmp, /*b_bias=*/nullptr, /*a_s=*/nullptr,
      b_scales.data, global_scale_ptr, /*zp=*/nullptr, /*g_idx=*/nullptr, /*perm=*/nullptr,
      /*a_tmp=*/nullptr, size_m, size_n, size_k, /*lda=*/size_k, workspace.data, a_type, b_type,
      c_type, s_type, /*has_bias=*/false, /*has_act_order=*/false, /*is_k_full=*/true,
      /*has_zp=*/false, num_groups, group_size, dev, s, /*thread_k=*/-1, /*thread_n=*/-1, sms,
      use_atomic_add, use_fp32_reduce, /*is_zp_float=*/false);

  if (c_tmp && !c_tmp_pooled) Check(cudaFreeAsync(c_tmp, s), "cudaFreeAsync c_tmp");
  Check(cudaGetLastError(), "marlin_dense marlin_mm launch");
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMarlinDenseGemm, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<MarlinDenseGemmFn>(&MarlinDenseGemmKernelCuda)));
  }
};
Registrar g_registrar;

}  // namespace
}  // namespace vt::cuda
