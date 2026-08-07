// ROCm RmsNorm — the ONE op the W0 skeleton registers (BACKEND-ROCM, W0).
//
// **UNBUILT.** Never compiled: no AMD GPU on the authoring machine. See the
// header of rocm_backend.hip.
//
// PORTED FROM: src/vt/cuda/cuda_ops.cu:96-126 `RmsNormRowKernel` (one block per
// row, shared-memory f32 tree reduction), whose own upstream counterpart is
// vLLM csrc/layernorm_kernels.cu `rms_norm_kernel` / `fused_add_rms_norm_kernel`.
// This is what a hipify pass produces for that kernel: __nv_bfloat16 becomes
// __hip_bfloat16, the intrinsics change spelling, the algorithm does not change
// at all. Diff the two files side by side — that is the point of writing it out.
//
// WHY THIS OP AND NOT A SIMPLER ONE. tests/vt/test_backend_cross_device.cpp runs
// against every registered non-CPU backend and SKIPS ops a device has not
// registered, comparing the rest against the CPU backend at NMSE <= 5e-4 from
// the same binary on the same inputs. So registering RmsNorm here means a
// contributor gets a real numeric gate on their own silicon without writing a
// test. RmsNorm specifically because it is a CROSS-LANE REDUCTION, which is
// where a hipified kernel can actually be wrong — AMD's wavefront is 64 lanes
// where NVIDIA's warp is 32, so a kernel that quietly assumed 32 breaks here.
// This one does not assume: the reduction is a __syncthreads() tree over shared
// memory with no warp-level primitives, so it is wavefront-width agnostic. An
// elementwise op like Add would have passed the same gate while proving nothing.
//
// NOT PORTED, deliberately: the VT_RMSNORM_DECODE_FAST paths
// (cuda_ops.cu:421-437). They exist to be BIT-IDENTICAL to this kernel while
// being faster on NVIDIA, they are tuned against sm_121a, and W0 is not a speed
// milestone. RmsNormQuantFp8 and the gated variants are likewise out of scope.
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include "vt/ops.h"

namespace vt::rocm {
namespace {

// Block width, from cuda_ops.cu:24. Kept at 256 rather than raised to a multiple
// of AMD's 64-lane wavefront: 256 is already 4 whole wavefronts, and matching the
// CUDA kernel's block width keeps the REDUCTION ORDER identical, which is what
// keeps the two backends' outputs comparable at the NMSE bar.
constexpr int kBlock = 256;

__device__ inline float Load(const float* p, int64_t i) { return p[i]; }
__device__ inline float Load(const __hip_bfloat16* p, int64_t i) {
  return __bfloat162float(p[i]);
}
__device__ inline void Store(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void Store(__hip_bfloat16* p, int64_t i, float v) {
  p[i] = __float2bfloat16(v);  // round-to-nearest-even, as the CUDA path and the host do
}

// Round-trip a f32 through the residual store dtype so the variance squares the
// SAME rounded value that is written back — mirrors vLLM's fused_add_rms_norm,
// whose bf16 residual add rounds to model dtype before the f32 variance.
// Identity for a f32 residual. cuda_ops.cu:85-93.
template <typename Tres>
__device__ inline float ResRound(float v);
template <>
__device__ inline float ResRound<float>(float v) {
  return v;
}
template <>
__device__ inline float ResRound<__hip_bfloat16>(float v) {
  return __bfloat162float(__float2bfloat16(v));
}

template <typename Tin, typename Tout, typename Tres>
__global__ void RmsNormRowKernel(Tout* out, const Tin* x, const Tin* w, Tres* residual, int64_t h,
                                 float eps, bool gemma) {
  const int64_t row = blockIdx.x;
  const Tin* xrow = x + row * h;
  Tout* orow = out + row * h;
  Tres* rrow = residual == nullptr ? nullptr : residual + row * h;

  __shared__ float partial[kBlock];
  float acc = 0.0f;
  for (int64_t j = threadIdx.x; j < h; j += kBlock) {
    float v = Load(xrow, j);
    if (rrow != nullptr) {
      v = ResRound<Tres>(v + Load(rrow, j));  // new residual stream: f32 add, round to Tres
      Store(rrow, j, v);                      // updated in place
    }
    acc += v * v;
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) partial[threadIdx.x] += partial[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = 1.0f / sqrtf(partial[0] / static_cast<float>(h) + eps);
  for (int64_t j = threadIdx.x; j < h; j += kBlock) {
    const float v = rrow != nullptr ? Load(rrow, j) : Load(xrow, j);
    float wj = Load(w, j);
    if (gemma) wj += 1.0f;
    Store(orow, j, v * inv * wj);
  }
}

template <typename Tin, typename Tout>
void LaunchRmsNormRes(hipStream_t s, Tensor& out, const Tensor& x, const Tensor& w,
                      const RmsNormArgs& args, Tensor* residual, unsigned rows, int64_t h) {
  if (residual != nullptr && residual->dtype == DType::kBF16) {
    RmsNormRowKernel<Tin, Tout, __hip_bfloat16><<<rows, kBlock, 0, s>>>(
        out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tin>(), residual->Ptr<__hip_bfloat16>(), h, args.eps,
        args.gemma);
  } else {
    float* res = residual == nullptr ? nullptr : residual->Ptr<float>();
    RmsNormRowKernel<Tin, Tout, float><<<rows, kBlock, 0, s>>>(
        out.Ptr<Tout>(), x.Ptr<Tin>(), w.Ptr<Tin>(), res, h, args.eps, args.gemma);
  }
}

template <typename Tin>
void LaunchRmsNorm(hipStream_t s, Tensor& out, const Tensor& x, const Tensor& w,
                   const RmsNormArgs& args, Tensor* residual) {
  const int64_t t = x.shape[0], h = x.shape[1];
  if (t == 0 || h == 0) return;
  const unsigned rows = static_cast<unsigned>(t);
  switch (out.dtype) {
    case DType::kF32:
      LaunchRmsNormRes<Tin, float>(s, out, x, w, args, residual, rows, h);
      break;
    case DType::kBF16:
      LaunchRmsNormRes<Tin, __hip_bfloat16>(s, out, x, w, args, residual, rows, h);
      break;
    default:
      VT_CHECK(false, "rocm rmsnorm: unsupported out dtype (f32/bf16 only)");
  }
}

}  // namespace

// The registered RmsNormFn (include/vt/ops.h:810). Signature is the shared vt
// one, unchanged — that is the whole contract a backend has to meet.
void RmsNormKernelRocm(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                       const RmsNormArgs& args, Tensor* residual) {
  VT_CHECK(w.dtype == x.dtype, "rocm rmsnorm: weight dtype must match x");
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  switch (x.dtype) {
    case DType::kF32:
      LaunchRmsNorm<float>(s, out, x, w, args, residual);
      break;
    case DType::kBF16:
      LaunchRmsNorm<__hip_bfloat16>(s, out, x, w, args, residual);
      break;
    default:
      VT_CHECK(false, "rocm rmsnorm: unsupported input dtype (f32/bf16 only)");
  }
}

}  // namespace vt::rocm
