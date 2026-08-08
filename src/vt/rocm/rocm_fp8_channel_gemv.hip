// FP8 E4M3 weight + BF16 per-row channel scale × BF16 activation GEMV (M=1).
// y[n] = alpha * scale[n] * sum_k x[k]*f8(W[n,k]) + beta * y[n]
// Opt-in via VT_GEMMA4_FP8_NATIVE path in gemma4 MoE.
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include "vt/device.h"
#include "vt/dtype.h"

namespace vt::rocm {
namespace {

// IEEE fp8-e4m3fn (matches vllm::F8E4M3ToF32)
__device__ inline float F8E4M3ToF32(uint8_t byte) {
  const uint32_t sign = static_cast<uint32_t>(byte >> 7) & 0x1U;
  const uint32_t exp = static_cast<uint32_t>(byte >> 3) & 0xFU;
  const uint32_t mant = static_cast<uint32_t>(byte) & 0x7U;
  const float sm = sign ? -1.0f : 1.0f;
  if (exp == 0xFU && mant == 0x7U) return 0.f;  // NaN → 0 in matmul
  if (exp == 0U) return sm * (static_cast<float>(mant) * (1.0f / 512.0f));
  const float mantissa = 1.0f + static_cast<float>(mant) * (1.0f / 8.0f);
  return sm * ldexpf(mantissa, static_cast<int>(exp) - 7);
}

// x in LDS; each thread owns output rows
__global__ void Fp8ChannelGemvKernel(__hip_bfloat16* __restrict__ y,
                                     const __hip_bfloat16* __restrict__ x,
                                     const uint8_t* __restrict__ W,  // [N,K]
                                     const __hip_bfloat16* __restrict__ scale,  // [N]
                                     int N, int K, float alpha, float beta) {
  extern __shared__ float smem[];
  float* x_cache = smem;
  for (int k = static_cast<int>(threadIdx.x); k < K; k += static_cast<int>(blockDim.x)) {
    x_cache[k] = __bfloat162float(x[k]);
  }
  __syncthreads();

  for (int n = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); n < N;
       n += static_cast<int>(gridDim.x * blockDim.x)) {
    const uint8_t* wrow = W + static_cast<int64_t>(n) * static_cast<int64_t>(K);
    float acc = 0.f;
    int k = 0;
    for (; k + 3 < K; k += 4) {
      acc += x_cache[k] * F8E4M3ToF32(wrow[k]);
      acc += x_cache[k + 1] * F8E4M3ToF32(wrow[k + 1]);
      acc += x_cache[k + 2] * F8E4M3ToF32(wrow[k + 2]);
      acc += x_cache[k + 3] * F8E4M3ToF32(wrow[k + 3]);
    }
    for (; k < K; ++k) acc += x_cache[k] * F8E4M3ToF32(wrow[k]);
    const float s = __bfloat162float(scale[n]);
    float v = alpha * s * acc;
    if (beta != 0.f) v += beta * __bfloat162float(y[n]);
    y[n] = __float2bfloat16(v);
  }
}

}  // namespace

// out[1,N] bf16, a[1,K] bf16, b_fp8[N,K], scale_bf16[N]
void MatmulBTFp8ChannelRocm(Queue& q, void* out, const void* a, const void* b_fp8,
                            const void* scale_bf16, int M, int N, int K, float alpha,
                            float beta) {
  if (M != 1 || N <= 0 || K <= 0) return;
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  constexpr int kBlock = 256;
  const int grid = (N + kBlock - 1) / kBlock;
  const size_t shmem = static_cast<size_t>(K) * sizeof(float);
  if (shmem > 48 * 1024) return;  // caller must fall back
  Fp8ChannelGemvKernel<<<grid, kBlock, shmem, s>>>(
      static_cast<__hip_bfloat16*>(out), static_cast<const __hip_bfloat16*>(a),
      static_cast<const uint8_t*>(b_fp8), static_cast<const __hip_bfloat16*>(scale_bf16), N, K,
      alpha, beta);
}

}  // namespace vt::rocm
