// SPDX-License-Identifier: Apache-2.0
// sm70 NVFP4 W4A16 decode GEMM (BACKEND-CUDA-SM070, Phase 2, brick A: SIMT).
//
// CUDA kernels ported (Apache-2.0) from dnv2003/v100-skinny
// (kernels/skinny_kernels.cu): the "skinny SIMT" Volta FP4 decoder path that
// reaches register bandwidth on W4A16 NVFP4 (stock Marlin fallback on this
// arch is ~12% of memcpy ceiling; this band measures ~69%). The E2M1 decoder
// derives from TurboMind/LMDeploy cvt_f16x8_e2m1 (Apache-2.0; 1Cat-vLLM
// csrc/sm70_turbomind/lmdeploy quantization.h).
//
// Bit layout kept verbatim (compressed-tensors NVFP4): packed E2M1 nibbles
// codes[n][k/2] (n-major; nibble (k&1) of byte k>>1), one fp8-e4m3 group
// scale byte scales[n][k/16], one global f32 scale folded in-kernel.
//
// Dispatch (BACKEND-CUDA-ARCH-ADDITIVITY): this TU registers ONE tactic for
// TacticFamily::kSm70Nvfp4W4a16; the launcher's portable path is untouched.
// Current band: SIMT decode M <= 3. QPN (M 4-16) and WMMA (M 17-64) are the
// next brick; otherwise Launch returns false => caller falls back.
//
// Self-check contract (inherited from skinny): the first two eager calls are
// cross-checked against a CPU fp32 reference on the LIVE arguments; a mismatch
// disables the tactic for the process lifetime (Launch returns false ->
// portable fallback), so a regression can never silently corrupt output.
// VT_SM70_SELFCHECK=0 disables the check for benchmark runs.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "vt/cuda/cuda_arch_tactics.h"
#include "vt/cuda/cuda_device_caps.h"

namespace vt::cuda {
namespace {

// ---------------------------------------------------------------------------
// Device-side decode helpers.
// ---------------------------------------------------------------------------

// fp8-e4m3 group scale byte -> half (both lanes identical). The 0x5C00
// multiplier carries the e4m3->e5m10 expansion, bit-identical to upstream.
__device__ __forceinline__ __half2 fp8e4m3_to_half2(unsigned char b) {
  const unsigned short hb =
      (((unsigned short)b & 0x80u) << 8) | (((unsigned short)b & 0x7Fu) << 7);
  const __half hs = __hmul(__ushort_as_half(hb), __ushort_as_half(0x5C00));
  return __halves2half2(hs, hs);
}

// XOR swizzle on the low 3 bits of a k-pair index; conflict-free for the SIMT
// read pattern (lane-groups sharing a bank base differ in p>>5).
__device__ __forceinline__ int swz(int p) { return (p & ~7) | ((p ^ (p >> 5)) & 7); }

// Stage 8 activation halves as four half2 pairs (adjacent-k pairing).
__device__ __forceinline__ void stage_pairs(__half2* dst, int lo, const uint4& v) {
  const unsigned* r = reinterpret_cast<const unsigned*>(&v);
  unsigned o[4] = {__byte_perm(r[0], r[2], 0x5410),
                   __byte_perm(r[0], r[2], 0x7632),
                   __byte_perm(r[1], r[3], 0x5410),
                   __byte_perm(r[1], r[3], 0x7632)};
#pragma unroll
  for (int j = 0; j < 4; j++) dst[swz(lo + j)] = *reinterpret_cast<__half2*>(&o[j]);
}

// Dequant 8 E2M1 codes (one 32-bit word = 8 nibbles) -> four half2 pairs.
// `sc` already carries the folded global scale and the 2^14 re-bias.
__device__ __forceinline__ void dequant8_k2(unsigned q, __half2 sc, __half2 out[4]) {
  constexpr unsigned S = 0x80008000u, EM = 0x0E000E00u;
  unsigned v0 = ((q << 12) & S) | ((q << 9) & EM);
  unsigned v1 = ((q << 8) & S) | ((q << 5) & EM);
  unsigned v2 = ((q << 4) & S) | ((q << 1) & EM);
  unsigned v3 = (q & S) | ((q >> 3) & EM);
  out[0] = __hmul2(*reinterpret_cast<__half2*>(&v0), sc);
  out[1] = __hmul2(*reinterpret_cast<__half2*>(&v1), sc);
  out[2] = __hmul2(*reinterpret_cast<__half2*>(&v2), sc);
  out[3] = __hmul2(*reinterpret_cast<__half2*>(&v3), sc);
}

// ---------------------------------------------------------------------------
// SIMT kernel (verbatim port; one output row per warp, M rows per block).
// KC: tile k width (512 codes = 32 scale groups, one per lane).
// ---------------------------------------------------------------------------
template <int M, int KC>
__global__ void Sm70Nvfp4SimtDecode(const uint8_t* __restrict__ codes,
                                    const uint8_t* __restrict__ scales,
                                    const __half* __restrict__ x,
                                    __half* __restrict__ y, int N, int K,
                                    float gscale) {
  constexpr int P2 = KC / 2;
  extern __shared__ char smem_raw[];
  __half2* xs = reinterpret_cast<__half2*>(smem_raw);  // [M][P2]

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = blockIdx.x * 8 + warp;
  const uint8_t* crow = codes + (size_t)n * (K >> 1);
  const uint8_t* srow = scales + (size_t)n * (K >> 4);
  const __half2 gm2 = __float2half2_rn(gscale * 16384.f);  // TM re-bias fold

  float accf[M];
#pragma unroll
  for (int m = 0; m < M; m++) accf[m] = 0.f;

  int k0 = 0;
  for (; k0 + KC <= K; k0 += KC) {
    __syncthreads();
    for (int idx = threadIdx.x; idx < M * (KC / 8); idx += blockDim.x) {
      const int m = idx / (KC / 8), j4 = idx % (KC / 8);
      const uint4 v = *reinterpret_cast<const uint4*>(x + (size_t)m * K + k0 + j4 * 8);
      stage_pairs(xs + m * P2, j4 * 4, v);
    }
    __syncthreads();
#pragma unroll
    for (int t = 0; t < KC / 512; t++) {
      const int s = lane + 32 * t;  // one 16-code scale segment per lane
      const uint2 q2 = *reinterpret_cast<const uint2*>(crow + (k0 >> 1) + s * 8);
      const __half2 sc = __hmul2(fp8e4m3_to_half2(srow[(k0 >> 4) + s]), gm2);
      __half2 acch[M];
#pragma unroll
      for (int m = 0; m < M; m++)
        acch[m] = __halves2half2(__ushort_as_half(0), __ushort_as_half(0));
#pragma unroll
      for (int w = 0; w < 2; w++) {
        __half2 w8[4];
        dequant8_k2(w == 0 ? q2.x : q2.y, sc, w8);
#pragma unroll
        for (int pi = 0; pi < 4; pi++) {
          const int ps = swz(s * 8 + w * 4 + pi);
#pragma unroll
          for (int m = 0; m < M; m++) {
            const __half2 xv = xs[m * P2 + ps];
            acch[m] = __hfma2(w8[pi], xv, acch[m]);
          }
        }
      }
#pragma unroll
      for (int m = 0; m < M; m++) {
        const float2 f = __half22float2(acch[m]);
        accf[m] += f.x + f.y;
      }
    }
    __syncthreads();
  }

  // Tail: K % KC (any multiple of 16; runtime segment bound).
  const int tail = K - k0;
  if (tail > 0) {
    const int copies = (tail + 7) / 8;
    __syncthreads();
    for (int idx = threadIdx.x; idx < M * copies; idx += blockDim.x) {
      const int m = idx / copies, j4 = idx % copies;
      const uint4 v = *reinterpret_cast<const uint4*>(x + (size_t)m * K + k0 + j4 * 8);
      stage_pairs(xs + m * P2, j4 * 4, v);
    }
    __syncthreads();
    const int nseg = tail >> 4;
    for (int s = lane; s < nseg; s += 32) {
      const uint2 q2 = *reinterpret_cast<const uint2*>(crow + (k0 >> 1) + s * 8);
      const __half2 sc = __hmul2(fp8e4m3_to_half2(srow[(k0 >> 4) + s]), gm2);
      __half2 acch[M];
#pragma unroll
      for (int m = 0; m < M; m++)
        acch[m] = __halves2half2(__ushort_as_half(0), __ushort_as_half(0));
#pragma unroll
      for (int w = 0; w < 2; w++) {
        __half2 w8[4];
        dequant8_k2(w == 0 ? q2.x : q2.y, sc, w8);
#pragma unroll
        for (int pi = 0; pi < 4; pi++) {
          const int ps = swz(s * 8 + w * 4 + pi);
#pragma unroll
          for (int m = 0; m < M; m++) {
            const __half2 xv = xs[m * P2 + ps];
            acch[m] = __hfma2(w8[pi], xv, acch[m]);
          }
        }
      }
#pragma unroll
      for (int m = 0; m < M; m++) {
        const float2 f = __half22float2(acch[m]);
        accf[m] += f.x + f.y;
      }
    }
  }

#pragma unroll
  for (int m = 0; m < M; m++) {
    float v = accf[m];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
    if (lane == 0) y[(size_t)m * N + n] = __float2half(v);
  }
}

// ---------------------------------------------------------------------------
// Host side helpers (bit-exact against the device arithmetic).
// ---------------------------------------------------------------------------

// One E2M1 code (nibble) -> f32 with the DEVICE bit layout:
//   sign << 12 | exp << 10 | mant << 9 (see dequant8_k2 lane math), re-bias
//   left to the scale fold (the /16384 that gm2 encodes).
float E2m1ToFloat(unsigned char c) {
  const unsigned short hb = (unsigned short)(((c & 0x8u) << 12) | ((c & 0x7u) << 9));
  return __half2float(__ushort_as_half(hb)) / 16384.0f;
}

// The exact device weight: code_half * (fp8scale16 * (gscale*16384)16)10.
float DeviceWeight(unsigned char c, unsigned char scale, float gscale) {
  const __half fp8h = __hmul(__ushort_as_half(
      (unsigned short)(((unsigned short)(scale & 0x80u) << 8) | ((unsigned short)(scale & 0x7Fu) << 7))),
      __ushort_as_half(0x5C00));
  const __half gm = __float2half(gscale * 16384.0f);
  const __half wsc = __hmul(fp8h, gm);
  return __half2float(__hmul(__float2half(E2m1ToFloat(c)), wsc));
}

std::vector<float> CpuReference(int m, int n, int k, const std::vector<__half>& x,
                               const std::vector<uint8_t>& codes,
                               const std::vector<uint8_t>& scales, float gscale) {
  std::vector<float> out((size_t)m * n, 0.f);
  for (int mm = 0; mm < m; ++mm) {
    for (int nn = 0; nn < n; ++nn) {
      double acc = 0.0;
      for (int kk = 0; kk < k; ++kk) {
        const unsigned char c =
            (codes[(size_t)nn * (k >> 1) + (kk >> 1)] >> ((kk & 1) ? 4 : 0)) & 15;
        const float w = DeviceWeight(c, scales[(size_t)nn * (k >> 4) + (kk >> 4)], gscale);
        acc += (double)__half2float(x[(size_t)mm * k + kk]) * (double)w;
      }
      out[(size_t)mm * n + nn] = (float)acc;
    }
  }
  return out;
}

struct SelfCheckState {
  int remaining = 2;
  bool disabled = false;
};

// Live self-check on the kernel's own first two calls. Returns false when the
// tactic must be permanently disabled (fallback contract).
bool RunSelfCheck(const Sm70Nvfp4W4a16Args& args, SelfCheckState& state) {
  if (state.disabled) return false;
  if (state.remaining <= 0) return true;
  state.remaining--;

  const int m = static_cast<int>(args.m), n = static_cast<int>(args.n), k = static_cast<int>(args.k);
  if (m <= 0 || m > 3 || n <= 0 || k <= 0 || (size_t)m * n * k > (1 << 22)) {
    return !state.disabled;  // shape too large to host-check; trust the band
  }

  std::vector<__half> x((size_t)m * k);
  std::vector<__half> y((size_t)m * n);
  std::vector<uint8_t> codes((size_t)n * (k >> 1));
  std::vector<uint8_t> scales((size_t)n * (k >> 4));
  cudaMemcpyAsync(x.data(), args.x, x.size() * sizeof(__half), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(codes.data(), args.codes, codes.size(), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(scales.data(), args.scales, scales.size(), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(y.data(), args.out, y.size() * sizeof(__half), cudaMemcpyDeviceToHost);
  cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(args.stream));

  const std::vector<float> ref = CpuReference(m, n, k, x, codes, scales, args.gscale);
  double err = 0.0, norm = 0.0;
  for (size_t i = 0; i < ref.size(); i++) {
    err = std::max(err, std::abs((double)__half2float(y[i]) - (double)ref[i]));
    norm = std::max(norm, std::abs((double)ref[i]));
  }
  if (err > 2.0e-2 * std::max(1.0, norm)) {
    state.disabled = true;
  }
  return !state.disabled;
}

// ---------------------------------------------------------------------------
// The registered tactic.
// ---------------------------------------------------------------------------
bool Sm70Nvfp4Supports(const DeviceCaps& caps) {
  return caps.valid && caps.sm_major == 7 && caps.sm_minor == 0;
}

bool LaunchSm70Nvfp4W4a16(const DeviceCaps&, void* argsPtr) {
  auto& args = *static_cast<Sm70Nvfp4W4a16Args*>(argsPtr);
  const int m = static_cast<int>(args.m), n = static_cast<int>(args.n), k = static_cast<int>(args.k);
  if (m <= 0 || n <= 0 || k <= 0 || (k & 15) || (n & 7)) return false;
  if (m > 3) return false;  // QPN (M 4-16) / WMMA (M 17-64): next brick

  static SelfCheckState self;
  const char* sk = std::getenv("VT_SM70_SELFCHECK");
  const bool check_enabled = !(sk && sk[0] == '0');
  if (check_enabled && !RunSelfCheck(args, self)) return false;

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(args.stream);
  auto* y = static_cast<__half*>(args.out);
  auto* x = static_cast<const __half*>(args.x);
  auto* codes = static_cast<const uint8_t*>(args.codes);
  auto* scales = static_cast<const uint8_t*>(args.scales);
  constexpr int KC = 512;
  const int smem = 3 * (KC / 2) * (int)sizeof(__half2);  // max M = 3
  switch (m) {
    case 1:
      Sm70Nvfp4SimtDecode<1, KC><<<n / 8, 256, smem, stream>>>(codes, scales, x, y, n, k, args.gscale);
      break;
    case 2:
      Sm70Nvfp4SimtDecode<2, KC><<<n / 8, 256, smem, stream>>>(codes, scales, x, y, n, k, args.gscale);
      break;
    case 3:
    default:
      Sm70Nvfp4SimtDecode<3, KC><<<n / 8, 256, smem, stream>>>(codes, scales, x, y, n, k, args.gscale);
      break;
  }
  return cudaGetLastError() == cudaSuccess;
}

struct Registrar {
  Registrar() {
    const ArchTactic tactic = {"nvfp4-w4a16/sm70-simt", &Sm70Nvfp4Supports, &LaunchSm70Nvfp4W4a16};
    RegisterArchTactic(TacticFamily::kSm70Nvfp4W4a16, tactic);
  }
};
static Registrar g_registrar;

}  // namespace
}  // namespace vt::cuda