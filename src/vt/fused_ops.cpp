#include "vt/fused_ops.h"

#include <stdexcept>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

#if defined(VLLM_CPP_HIP)
#include "vt/rocm/rocm_gelu_mul_sep.h"
#include "vt/rocm/rocm_gemma4_expert_geglu.h"
#include "vt/rocm/rocm_matmul_batch.h"
#include "vt/rocm/rocm_rmsnorm_plus_add.h"
#endif

namespace vt {

void RmsNormPlusAdd(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                    const Tensor& addend, const RmsNormArgs& args) {
#if defined(VLLM_CPP_HIP)
  if (q.device.type == DeviceType::kROCM) {
    rocm::RmsNormPlusAddRocm(q, out, x, w, addend, args);
    return;
  }
#endif
  // Composed reference (CPU / non-ROCm): out = rmsnorm(x) + addend via tmp in out.
  // Use out as scratch for rmsnorm then add — requires out != addend alias.
  RmsNorm(q, out, x, w, args);
  Add(q, out, out, addend);
}

void DualRmsNormPlusRes(Queue& q, Tensor& out, const Tensor& x1, const Tensor& w1,
                        const Tensor& x2, const Tensor& w2, const Tensor& w3,
                        const Tensor& residual, const RmsNormArgs& args) {
#if defined(VLLM_CPP_HIP)
  if (q.device.type == DeviceType::kROCM) {
    rocm::DualRmsNormPlusResRocm(q, out, x1, w1, x2, w2, w3, residual, args);
    return;
  }
#endif
  // Slow but correct host-side compose using existing ops (allocates temps on device).
  Tensor n1 = x1;  // shape clone without owning — fall back to sequential RmsNorm+Add
  // Prefer throw on non-ROCm discrete GPUs without a known-good compose path.
  if (q.device.type != DeviceType::kCPU) {
    throw std::runtime_error("vt::DualRmsNormPlusRes: non-ROCm GPU path not registered");
  }
  (void)n1;
  (void)out;
  (void)w1;
  (void)x2;
  (void)w2;
  (void)w3;
  (void)residual;
  (void)args;
  throw std::runtime_error("vt::DualRmsNormPlusRes: CPU compose not yet wired");
}

void GeluMulSeparate(Queue& q, void* out, const void* gate, const void* up, int64_t n,
                     DType dtype) {
#if defined(VLLM_CPP_HIP)
  if (q.device.type == DeviceType::kROCM) {
    rocm::GeluMulSeparateRocm(q, out, gate, up, n, dtype);
    return;
  }
#endif
  // Composed reference (CPU / non-ROCm), same shape as RmsNormPlusAdd above.
  //
  // This is NOT an optional fast path: gemma4.cpp's `ple > 0` block calls this
  // from the SHARED forward, so throwing here aborted Gemma-4 on its first
  // layer on every non-ROCm backend (issue #377). vt::GeluAndMul computes
  // exactly this math -- gelu_tanh(gate) * up -- but wants the two halves
  // adjacent as one [rows, 2D] tensor, so the compose stages them into a
  // temporary [1, 2n] laid out as [gate | up] and runs the shipped kernel on
  // it. Whatever GeluAndMul does, this matches it by construction, on every
  // backend that registers it.
  if (n <= 0) return;
  Backend& b = GetBackend(q.device.type);
  const size_t elem = SizeOf(dtype);
  const size_t half = static_cast<size_t>(n) * elem;
  void* tmp = b.Alloc(2 * half);
  if (tmp == nullptr) {
    throw std::runtime_error("vt::GeluMulSeparate: temporary allocation failed");
  }
  try {
    b.Copy(q, tmp, gate, half);
    b.Copy(q, static_cast<char*>(tmp) + half, up, half);
    Tensor tin = Tensor::Contiguous(tmp, dtype, q.device, {1, 2 * n});
    Tensor tout = Tensor::Contiguous(out, dtype, q.device, {1, n});
    GeluAndMul(q, tout, tin);
    // The temporary is read by work queued on `q`, so it must outlive that
    // work: on an async backend the Free below would otherwise race the
    // kernel. Synchronous backends (CPU) no-op this.
    b.Synchronize(q);
  } catch (...) {
    b.Free(tmp);
    throw;
  }
  b.Free(tmp);
}

void MatmulBTAlphaBeta(Queue& q, void* out, const void* a, const void* b, int M, int N, int K,
                       float alpha, float beta, DType dtype) {
#if defined(VLLM_CPP_HIP)
  if (q.device.type == DeviceType::kROCM) {
    rocm::MatmulBTAlphaBetaRocm(q, out, a, b, M, N, K, alpha, beta, dtype);
    return;
  }
#endif
  (void)q;
  (void)out;
  (void)a;
  (void)b;
  (void)M;
  (void)N;
  (void)K;
  (void)alpha;
  (void)beta;
  (void)dtype;
  throw std::runtime_error("vt::MatmulBTAlphaBeta: ROCm-only in this build");
}

void MatmulBTFp8Channel(Queue& q, void* out, const void* a, const void* b_fp8,
                        const void* scale_bf16, int M, int N, int K, float alpha, float beta) {
#if defined(VLLM_CPP_HIP)
  if (q.device.type == DeviceType::kROCM) {
    rocm::MatmulBTFp8ChannelRocm(q, out, a, b_fp8, scale_bf16, M, N, K, alpha, beta);
    return;
  }
#endif
  (void)q;
  (void)out;
  (void)a;
  (void)b_fp8;
  (void)scale_bf16;
  (void)M;
  (void)N;
  (void)K;
  (void)alpha;
  (void)beta;
  throw std::runtime_error("vt::MatmulBTFp8Channel: ROCm-only in this build");
}

bool ExpertGeGLUBf16TopKM1(Queue& q, void* ysum, const void* x, const void* const* w_gu,
                           const void* const* w_dn, const float* wts, int G, int I, int H) {
#if defined(VLLM_CPP_HIP)
  if (q.device.type == DeviceType::kROCM) {
    return rocm::ExpertGeGLUBf16TopKM1Rocm(q, ysum, x, w_gu, w_dn, wts, G, I, H);
  }
#endif
  (void)q;
  (void)ysum;
  (void)x;
  (void)w_gu;
  (void)w_dn;
  (void)wts;
  (void)G;
  (void)I;
  (void)H;
  return false;
}

}  // namespace vt
