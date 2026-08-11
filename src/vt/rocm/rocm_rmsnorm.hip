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
#include "vt/rocm/rocm_device_bind.h"

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

// out = rmsnorm(x, w) + addend  — Gemma-4 residual join (NOT residual+=x then norm).
template <typename T>
__global__ void RmsNormPlusAddKernel(T* out, const T* x, const T* w, const T* add, int64_t h,
                                     float eps, bool gemma) {
  const int64_t row = blockIdx.x;
  const T* xrow = x + row * h;
  const T* arow = add + row * h;
  T* orow = out + row * h;
  __shared__ float partial[kBlock];
  float acc = 0.0f;
  for (int64_t j = threadIdx.x; j < h; j += kBlock) {
    const float v = Load(xrow, j);
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
    float wj = Load(w, j);
    if (gemma) wj += 1.0f;
    Store(orow, j, Load(xrow, j) * inv * wj + Load(arow, j));
  }
}

}  // namespace

// The registered RmsNormFn (include/vt/ops.h:810). Signature is the shared vt
// one, unchanged — that is the whole contract a backend has to meet.
void RmsNormKernelRocm(Queue& q,
  // bind
 Tensor& out, const Tensor& x, const Tensor& w,
                       const RmsNormArgs& args, Tensor* residual) {
  EnsureQueueDevice(q);
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

// out = rmsnorm(x, w) + addend
void RmsNormPlusAddRocm(Queue& q,
  // bind
 Tensor& out, const Tensor& x, const Tensor& w,
                        const Tensor& addend, const RmsNormArgs& args) {
  EnsureQueueDevice(q);
  VT_CHECK(w.dtype == x.dtype && out.dtype == x.dtype && addend.dtype == x.dtype,
           "rocm rmsnorm+add: dtype match");
  VT_CHECK(x.rank >= 2 && out.shape[0] == x.shape[0] && out.shape[1] == x.shape[1] &&
               addend.shape[0] == x.shape[0] && addend.shape[1] == x.shape[1],
           "rocm rmsnorm+add: shape");
  const int64_t t = x.shape[0], h = x.shape[1];
  if (t == 0 || h == 0) return;
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  if (x.dtype == DType::kBF16) {
    RmsNormPlusAddKernel<__hip_bfloat16><<<static_cast<unsigned>(t), kBlock, 0, s>>>(
        out.Ptr<__hip_bfloat16>(), x.Ptr<__hip_bfloat16>(), w.Ptr<__hip_bfloat16>(),
        addend.Ptr<__hip_bfloat16>(), h, args.eps, args.gemma);
  } else if (x.dtype == DType::kF32) {
    RmsNormPlusAddKernel<float><<<static_cast<unsigned>(t), kBlock, 0, s>>>(
        out.Ptr<float>(), x.Ptr<float>(), w.Ptr<float>(), addend.Ptr<float>(), h, args.eps,
        args.gemma);
  } else {
    VT_CHECK(false, "rocm rmsnorm+add dtype");
  }
}

// out = rmsnorm(rmsnorm(x1,w1) + rmsnorm(x2,w2), w3) + residual
// Gemma-4 MoE post-FF: dual expert streams + join residual.
template <typename T>
__global__ void DualRmsNormPlusResKernel(T* out, const T* x1, const T* w1, const T* x2,
                                         const T* w2, const T* w3, const T* residual, int64_t h,
                                         float eps1, float eps2, float eps3, bool gemma) {
  const int64_t row = blockIdx.x;
  const T* x1r = x1 + row * h;
  const T* x2r = x2 + row * h;
  const T* rr = residual + row * h;
  T* orow = out + row * h;
  __shared__ float p1[kBlock];
  __shared__ float p2[kBlock];
  __shared__ float p3[kBlock];

  float a1 = 0.f, a2 = 0.f;
  for (int64_t j = threadIdx.x; j < h; j += kBlock) {
    const float v1 = Load(x1r, j);
    const float v2 = Load(x2r, j);
    a1 += v1 * v1;
    a2 += v2 * v2;
  }
  p1[threadIdx.x] = a1;
  p2[threadIdx.x] = a2;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) {
      p1[threadIdx.x] += p1[threadIdx.x + s];
      p2[threadIdx.x] += p2[threadIdx.x + s];
    }
    __syncthreads();
  }
  const float inv1 = 1.f / sqrtf(p1[0] / static_cast<float>(h) + eps1);
  const float inv2 = 1.f / sqrtf(p2[0] / static_cast<float>(h) + eps2);

  // Second pass: build mid = n1+n2 and accumulate variance of mid.
  // Use shared? too large for H=2816. Recompute n1,n2 per thread.
  float a3 = 0.f;
  for (int64_t j = threadIdx.x; j < h; j += kBlock) {
    float w1j = Load(w1, j);
    float w2j = Load(w2, j);
    if (gemma) {
      w1j += 1.f;
      w2j += 1.f;
    }
    const float mid = Load(x1r, j) * inv1 * w1j + Load(x2r, j) * inv2 * w2j;
    a3 += mid * mid;
  }
  p3[threadIdx.x] = a3;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) p3[threadIdx.x] += p3[threadIdx.x + s];
    __syncthreads();
  }
  const float inv3 = 1.f / sqrtf(p3[0] / static_cast<float>(h) + eps3);
  for (int64_t j = threadIdx.x; j < h; j += kBlock) {
    float w1j = Load(w1, j);
    float w2j = Load(w2, j);
    float w3j = Load(w3, j);
    if (gemma) {
      w1j += 1.f;
      w2j += 1.f;
      w3j += 1.f;
    }
    const float mid = Load(x1r, j) * inv1 * w1j + Load(x2r, j) * inv2 * w2j;
    Store(orow, j, mid * inv3 * w3j + Load(rr, j));
  }
}

void DualRmsNormPlusResRocm(Queue& q,
  // bind
 Tensor& out, const Tensor& x1, const Tensor& w1,
                            const Tensor& x2, const Tensor& w2, const Tensor& w3,
                            const Tensor& residual, const RmsNormArgs& args) {
  EnsureQueueDevice(q);
  VT_CHECK(out.dtype == x1.dtype && x1.dtype == x2.dtype && x1.dtype == residual.dtype,
           "dual rms dtype");
  const int64_t t = x1.shape[0], h = x1.shape[1];
  if (t == 0 || h == 0) return;
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  // plain RMS (no gemma +1) for Gemma-4 layer norms
  const bool gemma = args.gemma;
  if (x1.dtype == DType::kBF16) {
    DualRmsNormPlusResKernel<__hip_bfloat16><<<static_cast<unsigned>(t), kBlock, 0, s>>>(
        out.Ptr<__hip_bfloat16>(), x1.Ptr<__hip_bfloat16>(), w1.Ptr<__hip_bfloat16>(),
        x2.Ptr<__hip_bfloat16>(), w2.Ptr<__hip_bfloat16>(), w3.Ptr<__hip_bfloat16>(),
        residual.Ptr<__hip_bfloat16>(), h, args.eps, args.eps, args.eps, gemma);
  } else if (x1.dtype == DType::kF32) {
    DualRmsNormPlusResKernel<float><<<static_cast<unsigned>(t), kBlock, 0, s>>>(
        out.Ptr<float>(), x1.Ptr<float>(), w1.Ptr<float>(), x2.Ptr<float>(), w2.Ptr<float>(),
        w3.Ptr<float>(), residual.Ptr<float>(), h, args.eps, args.eps, args.eps, gemma);
  } else {
    VT_CHECK(false, "dual rms dtype");
  }
}

}  // namespace vt::rocm
