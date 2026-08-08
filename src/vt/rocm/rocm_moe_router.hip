// ROCm ungrouped MoE router: softmax + greedy top-k (lowest-index tie-break).
#include <hip/hip_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/ops.h"

namespace vt::rocm {
namespace {

void CheckHip(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("vt rocm moe_router: ") + what + ": " +
                             hipGetErrorString(err));
  }
}

// One block per token. E <= 512, K <= 64. Shared: scores[E] + mask work in registers.
__global__ void MoeRouterTopKUngroupedKernel(const float* __restrict__ logits,
                                             float* __restrict__ weights,
                                             int32_t* __restrict__ indices, int T, int E,
                                             int K, int renormalize, float scale) {
  const int t = static_cast<int>(blockIdx.x);
  if (t >= T) return;
  extern __shared__ float smem[];
  float* scores = smem;  // [E]

  const float* row = logits + static_cast<size_t>(t) * static_cast<size_t>(E);
  // Load + max
  float mx = -INFINITY;
  for (int j = static_cast<int>(threadIdx.x); j < E; j += static_cast<int>(blockDim.x)) {
    const float v = row[j];
    scores[j] = v;
    mx = fmaxf(mx, v);
  }
  // block reduce max
  __shared__ float red[256];
  red[threadIdx.x] = mx;
  __syncthreads();
  for (int s = static_cast<int>(blockDim.x) / 2; s > 0; s >>= 1) {
    if (static_cast<int>(threadIdx.x) < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  mx = red[0];

  // softmax
  float sum = 0.f;
  for (int j = static_cast<int>(threadIdx.x); j < E; j += static_cast<int>(blockDim.x)) {
    const float e = expf(scores[j] - mx);
    scores[j] = e;
    sum += e;
  }
  red[threadIdx.x] = sum;
  __syncthreads();
  for (int s = static_cast<int>(blockDim.x) / 2; s > 0; s >>= 1) {
    if (static_cast<int>(threadIdx.x) < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  sum = red[0];
  const float inv = (sum > 0.f) ? (1.f / sum) : 0.f;
  for (int j = static_cast<int>(threadIdx.x); j < E; j += static_cast<int>(blockDim.x)) {
    float p = scores[j] * inv;
    if (!isfinite(p)) p = 0.f;
    scores[j] = p;
  }
  __syncthreads();

  // Greedy top-k with strict > (lowest index wins ties). Thread 0 only — E small.
  if (threadIdx.x == 0) {
    float denom = 0.f;
    float* wout = weights + static_cast<size_t>(t) * static_cast<size_t>(K);
    int32_t* iout = indices + static_cast<size_t>(t) * static_cast<size_t>(K);
    for (int ki = 0; ki < K; ++ki) {
      int best = 0;
      float best_v = -INFINITY;
      for (int j = 0; j < E; ++j) {
        const float v = scores[j];
        if (v > best_v) {
          best_v = v;
          best = j;
        }
      }
      const float w = scores[best];
      scores[best] = -INFINITY;
      wout[ki] = w;
      iout[ki] = best;
      denom += w;
    }
    if (renormalize) {
      if (!(denom > 0.f)) denom = 1.f;
      for (int ki = 0; ki < K; ++ki) wout[ki] /= denom;
    }
    if (scale != 1.f) {
      for (int ki = 0; ki < K; ++ki) wout[ki] *= scale;
    }
  }
}

}  // namespace

void MoeRouterTopKKernelRocm(Queue& q, Tensor& weights, Tensor& indices, const Tensor& logits,
                             const MoeRouterTopKArgs& args, const Tensor* bias) {
  // Grouped / bias path: fall back via host (rare for Gemma4).
  if (args.num_expert_group > 0 || bias != nullptr ||
      args.scoring_func != MoeScoringFunc::kSoftmax) {
    throw std::runtime_error(
        "vt rocm: MoeRouterTopK only implements ungrouped softmax without bias; "
        "use CPU backend for grouped/bias routers");
  }
  if (logits.dtype != DType::kF32 || weights.dtype != DType::kF32 ||
      indices.dtype != DType::kI32) {
    throw std::runtime_error("vt rocm: MoeRouterTopK dtypes must be f32/f32/i32");
  }
  const int T = static_cast<int>(logits.shape[0]);
  const int E = static_cast<int>(logits.shape[1]);
  const int K = args.top_k;
  if (T == 0 || K == 0) return;
  if (E > 512 || K > 64) {
    throw std::runtime_error("vt rocm: MoeRouterTopK E>512 or K>64 not supported");
  }
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  const int threads = 256;
  const size_t shmem = static_cast<size_t>(E) * sizeof(float);
  hipLaunchKernelGGL(MoeRouterTopKUngroupedKernel, dim3(T), dim3(threads), shmem, s,
                     static_cast<const float*>(logits.data), static_cast<float*>(weights.data),
                     static_cast<int32_t*>(indices.data), T, E, K, args.renormalize ? 1 : 0,
                     args.routed_scaling_factor);
  CheckHip(hipGetLastError(), "MoeRouterTopK launch");
}

}  // namespace vt::rocm
