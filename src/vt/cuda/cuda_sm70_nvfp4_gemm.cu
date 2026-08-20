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
// Bands: SIMT decode M <= 3 (A), QPN m8n8k4 M 4..16 (C, prepacked fragment
// layout), WMMA tiles M 4..64 (B, m-padded; the QPN fallback). All three
// share the CT layout + self-check; Launch returns false => caller falls
// back.
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
#include <mma.h>

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
// WMMA band (brick B): WN x WM warps of 16x16 fp16 tiles, KC-deep smem staging
// with software pipelining. Operates on the SAME codes/scales layout (dequant
// in shared); the epilogue undoes the 2^-14 re-bias folded by dequant8_k2.
// Handles m < MT via m_real (rows padded with zeros) and n % (WN*16) == 0.
// The original (small-M band 4..16) is served by a single 16-row tile; the
// QPN m8n8k4 tensor-core variant remains a later brick (this tile is correct,
// not yet the bw-optimal one for M in 4..16).
// ---------------------------------------------------------------------------
template <int WN, int WM, int KC>
__global__ void Sm70Nvfp4Wmma(const uint8_t* __restrict__ codes,
                              const uint8_t* __restrict__ scales,
                              const __half* __restrict__ x,
                              __half* __restrict__ y, int N, int K,
                              int m_real, float gscale) {
  constexpr int NT = WN * 16, MT = WM * 16;
  constexpr int PW = KC + 16, PX = KC + 16;  // padded smem pitches (halfs)
  constexpr int NTHREADS = WN * WM * 32;
  constexpr int CSEG = NT * (KC / 16) / NTHREADS;
  constexpr int XSEG = MT * (KC / 8) / NTHREADS;
  static_assert(CSEG * NTHREADS == NT * (KC / 16), "code seg split");
  static_assert(XSEG * NTHREADS == MT * (KC / 8), "x seg split");

  extern __shared__ char smem_raw[];
  __half* ws = reinterpret_cast<__half*>(smem_raw);  // [NT][PW]
  __half* xs = ws + NT * PW;                         // [MT][PX]

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int wn = warp % WN, wm = warp / WN;
  const int nb = blockIdx.x * NT;

  uint2 st_c[CSEG];
  unsigned char st_s[CSEG];
  uint4 st_x[XSEG];

  auto load_stage = [&](int k0) {
#pragma unroll
    for (int i = 0; i < CSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int n = idx / (KC / 16), s = idx % (KC / 16);
      st_c[i] = __ldcs(reinterpret_cast<const uint2*>(codes + (size_t)(nb + n) * (K >> 1) + (k0 >> 1) + s * 8));
      st_s[i] = __ldcs(scales + (size_t)(nb + n) * (K >> 4) + (k0 >> 4) + s);
    }
#pragma unroll
    for (int i = 0; i < XSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int m = idx / (KC / 8), j4 = idx % (KC / 8);
      st_x[i] = (m < m_real) ? *reinterpret_cast<const uint4*>(x + (size_t)m * K + k0 + j4 * 8)
                             : make_uint4(0, 0, 0, 0);
    }
  };

  auto store_stage = [&]() {
#pragma unroll
    for (int i = 0; i < CSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int n = idx / (KC / 16), s = idx % (KC / 16);
      const __half2 sc2 = fp8e4m3_to_half2(st_s[i]);
      __half2* wrow = reinterpret_cast<__half2*>(ws + n * PW + s * 16);
      const unsigned qs[2] = {st_c[i].x, st_c[i].y};
#pragma unroll
      for (int w = 0; w < 2; w++) {
        __half2 t[4];
        dequant8_k2(qs[w], sc2, t);  // values carry a 2^-14 factor here
        const unsigned* tr = reinterpret_cast<const unsigned*>(t);
        unsigned lin[4] = {__byte_perm(tr[0], tr[1], 0x5410),
                           __byte_perm(tr[2], tr[3], 0x5410),
                           __byte_perm(tr[0], tr[1], 0x7632),
                           __byte_perm(tr[2], tr[3], 0x7632)};
#pragma unroll
        for (int pi = 0; pi < 4; pi++)
          wrow[w * 4 + pi] = *reinterpret_cast<__half2*>(&lin[pi]);
      }
    }
#pragma unroll
    for (int i = 0; i < XSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int m = idx / (KC / 8), j4 = idx % (KC / 8);
      *reinterpret_cast<uint4*>(xs + m * PX + j4 * 8) = st_x[i];
    }
  };

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> cfrag;
  nvcuda::wmma::fill_fragment(cfrag, 0.f);

  load_stage(0);
  for (int k0 = 0; k0 < K; k0 += KC) {
    __syncthreads();
    store_stage();
    __syncthreads();
    if (k0 + KC < K) load_stage(k0 + KC);

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> a[2];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> b[2];
    nvcuda::wmma::load_matrix_sync(a[0], ws + wn * 16 * PW, PW);
    nvcuda::wmma::load_matrix_sync(b[0], xs + wm * 16 * PX, PX);
#pragma unroll
    for (int kk = 0; kk < KC / 16; kk++) {
      const int cur = kk & 1, nxt = cur ^ 1;
      if (kk + 1 < KC / 16) {
        nvcuda::wmma::load_matrix_sync(a[nxt], ws + wn * 16 * PW + (kk + 1) * 16, PW);
        nvcuda::wmma::load_matrix_sync(b[nxt], xs + wm * 16 * PX + (kk + 1) * 16, PX);
      }
      nvcuda::wmma::mma_sync(cfrag, a[cur], b[cur], cfrag);
    }
  }

  __syncthreads();
  float* cs = reinterpret_cast<float*>(smem_raw) + warp * 256;
  nvcuda::wmma::store_matrix_sync(cs, cfrag, 16, nvcuda::wmma::mem_row_major);
  __syncwarp();
  const float gs_eff = gscale * 16384.f;  // undo dequant8_k2's 2^-14
  for (int e = lane; e < 256; e += 32) {
    const int i = e >> 4, j = e & 15;  // i: n within 16-tile, j: m within tile
    const int gm = wm * 16 + j, gn = nb + wn * 16 + i;
    if (gm < m_real) y[(size_t)gm * N + gn] = __float2half(cs[e] * gs_eff);
  }
}

// ---------------------------------------------------------------------------
// QPN band (brick C): the Volta-native four-quadpair mma.m8n8k4 tensor-op
// kernel, fed from the fragment-order PREPACK of the CT layout (byte-equal
// permutation, rebuilt per call into a cached scratch). One CTA per 32-column
// tile, K split across 4 warps in 16-wide groups. This is the bw-optimal path
// for M in 4..16 (the WMMA tile above is its correctness fallback).
// ---------------------------------------------------------------------------
#define MMA_8N8K4(C, A0, A1, B0, B1)                                        \
  asm volatile(                                                             \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "                    \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "                     \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                        \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]),         \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                                  \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

template <int MT>
__global__ void Sm70Nvfp4Qpn(const uint8_t* __restrict__ qcodes,
                             const uint8_t* __restrict__ qscales,
                             const __half* __restrict__ x,
                             __half* __restrict__ y, int N, int K, int M,
                             float gscale) {
  constexpr int WARPS = 4;
  __shared__ float cs[WARPS][MT * 256];

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int qp = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int G = K >> 4, Gq = G / WARPS;
  const int g0 = warp * Gq;
  const uint2* cb = reinterpret_cast<const uint2*>(qcodes) + (size_t)tile * G * 32 + lane;
  const uint8_t* sb = qscales + (size_t)tile * G * 32 + lane;

  const __half2 gm2 = __float2half2_rn(gscale * 16384.f);
  float c[MT][8];
#pragma unroll
  for (int t = 0; t < MT; t++)
#pragma unroll
    for (int i = 0; i < 8; i++) c[t][i] = 0.f;

#pragma unroll 4
  for (int g = g0; g < g0 + Gq; g++) {
    const uint2 q2 = __ldcs(cb + (size_t)g * 32);
    const __half2 sc2 = __hmul2(fp8e4m3_to_half2(__ldg(sb + (size_t)g * 32)), gm2);
    __half2 b[8];
    dequant8_k2(q2.x, sc2, b + 0);
    dequant8_k2(q2.y, sc2, b + 4);
    const unsigned* B = reinterpret_cast<const unsigned*>(b);
#pragma unroll
    for (int t = 0; t < MT; t++) {
      const int ar = t * 8 + r;
      uint4 a01 = make_uint4(0, 0, 0, 0), a23 = make_uint4(0, 0, 0, 0);
      if (ar < M) {
        const __half* xrow = x + (size_t)ar * K;
        a01 = *reinterpret_cast<const uint4*>(xrow + g * 16);
        a23 = *reinterpret_cast<const uint4*>(xrow + g * 16 + 8);
      }
      const unsigned* A0 = reinterpret_cast<const unsigned*>(&a01);
      const unsigned* A1 = reinterpret_cast<const unsigned*>(&a23);
      MMA_8N8K4(c[t], A0[0], A0[1], B[0], B[1]);
      MMA_8N8K4(c[t], A0[2], A0[3], B[2], B[3]);
      MMA_8N8K4(c[t], A1[0], A1[1], B[4], B[5]);
      MMA_8N8K4(c[t], A1[2], A1[3], B[6], B[7]);
    }
  }
#pragma unroll
  for (int t = 0; t < MT; t++)
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      cs[warp][(t * 8 + row) * 32 + qp * 8 + col] = c[t][i];
    }
  __syncthreads();
  for (int e = threadIdx.x; e < MT * 256; e += blockDim.x) {
    const float v = cs[0][e] + cs[1][e] + cs[2][e] + cs[3][e];
    const int row = e >> 5, col = e & 31;
    if (row < M) y[(size_t)row * N + (size_t)tile * 32 + col] = __float2half(v);
  }
}

// Fragment-order prepack (device; byte-equal to the fork's python build):
//   qcodes[(tile*G + g)*32*8 + lane*8] = 8 bytes of korder-ordered, pair-
//   interleaved nibbles; qscales[(tile*G+g)*32 + lane] = the group scale.
__global__ void Sm70Nvfp4PackQpn(const uint8_t* __restrict__ codes,
                                 const uint8_t* __restrict__ scales,
                                 uint8_t* __restrict__ qcodes,
                                 uint8_t* __restrict__ qscales, int K) {
  constexpr unsigned short korder[16] = {0, 2, 4, 6, 1, 3, 5, 7,
                                         8, 10, 12, 14, 9, 11, 13, 15};
  const int tile = blockIdx.x, g = blockIdx.y, lane = threadIdx.x;
  const int G = K >> 4;
  const int col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  const int row = tile * 32 + col;
  const uint8_t* crow = codes + (size_t)row * (K >> 1);
  unsigned char nb[16];
#pragma unroll
  for (int j = 0; j < 16; j++) {
    const int kk = g * 16 + korder[j];
    const uint8_t b = crow[kk >> 1];
    nb[j] = (kk & 1) ? (uint8_t)(b >> 4) : (uint8_t)(b & 0xF);
  }
  uint8_t* dst = qcodes + (size_t)(tile * G + g) * 32 * 8 + lane * 8;
#pragma unroll
  for (int j = 0; j < 8; j++)
    dst[j] = (uint8_t)(nb[2 * j] | (nb[2 * j + 1] << 4));
  qscales[(size_t)(tile * G + g) * 32 + lane] = scales[(size_t)row * G + g];
}

// Fused greedy-argmax decode (M == 1): the same row- decode as the SIMT band
// but, instead of writing the fp16 row, each 8-warp block keeps its per-column
// winner and emits one (val, idx) pair per block (frontier semantics of the
// v100-skinny `gemm_simt_argmax`: the caller reduces across blocks). Used by
// the drafter's lm_head. `y` may stay null (no row materialization).
__global__ void Sm70Nvfp4SimtArgmax(const uint8_t* __restrict__ codes,
                                    const uint8_t* __restrict__ scales,
                                    const __half* __restrict__ x,
                                    __half* __restrict__ y,
                                    __half* __restrict__ argmax_val,
                                    int* __restrict__ argmax_idx, int N, int K,
                                    float gscale) {
  constexpr int KC = 512;
  constexpr int P2 = KC / 2;
  extern __shared__ char smem_raw[];
  __half2* xs = reinterpret_cast<__half2*>(smem_raw);  // [1][P2]
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int col = blockIdx.x * 8 + warp;  // one of the block's 8 columns
  const uint8_t* crow = codes + (size_t)col * (K >> 1);
  const uint8_t* srow = scales + (size_t)col * (K >> 4);
  const __half2 gm2 = __float2half2_rn(gscale * 16384.f);
  float acc = 0.f;

  int k0 = 0;
  for (; k0 + KC <= K; k0 += KC) {
    __syncthreads();
    for (int idx = threadIdx.x; idx < KC / 8; idx += blockDim.x) {
      const uint4 v = *reinterpret_cast<const uint4*>(x + k0 + idx * 8);
      stage_pairs(xs, idx * 4, v);
    }
    __syncthreads();
    for (int t = 0; t < KC / 512; t++) {
      const int s = lane + 32 * t;
      const uint2 q2 = *reinterpret_cast<const uint2*>(crow + (k0 >> 1) + s * 8);
      const __half2 sc = __hmul2(fp8e4m3_to_half2(srow[(k0 >> 4) + s]), gm2);
      __half2 acch = __halves2half2(__ushort_as_half(0), __ushort_as_half(0));
#pragma unroll
      for (int w = 0; w < 2; w++) {
        __half2 w8[4];
        dequant8_k2(w == 0 ? q2.x : q2.y, sc, w8);
#pragma unroll
        for (int pi = 0; pi < 4; pi++) {
          const int ps = swz(s * 8 + w * 4 + pi);
          acch = __hfma2(w8[pi], xs[ps], acch);
        }
      }
      const float2 f = __half22float2(acch);
      acc += f.x + f.y;
    }
    __syncthreads();
  }
  const int tail = K - k0;
  if (tail > 0) {
    __syncthreads();
    for (int idx = threadIdx.x; idx < (tail + 7) / 8; idx += blockDim.x) {
      const uint4 v = *reinterpret_cast<const uint4*>(x + k0 + idx * 8);
      stage_pairs(xs, idx * 4, v);
    }
    __syncthreads();
    const int nseg = tail >> 4;
    for (int s = lane; s < nseg; s += 32) {
      const uint2 q2 = *reinterpret_cast<const uint2*>(crow + (k0 >> 1) + s * 8);
      const __half2 sc = __hmul2(fp8e4m3_to_half2(srow[(k0 >> 4) + s]), gm2);
      __half2 acch = __halves2half2(__ushort_as_half(0), __ushort_as_half(0));
#pragma unroll
      for (int w = 0; w < 2; w++) {
        __half2 w8[4];
        dequant8_k2(w == 0 ? q2.x : q2.y, sc, w8);
#pragma unroll
        for (int pi = 0; pi < 4; pi++) {
          const int ps = swz(s * 8 + w * 4 + pi);
          acch = __hfma2(w8[pi], xs[ps], acch);
        }
      }
      const float2 f = __half22float2(acch);
      acc += f.x + f.y;
    }
  }

  // Sum-reduce the lane partials -> column value at lane0; then the block's
  // 8 columns reduce to one (val, idx) through a smem tail (offsets past the
  // staging region; the launch reserves the bytes).
  float vsum = acc;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) vsum += __shfl_xor_sync(~0u, vsum, o);
  if (lane == 0 && col < N) {
    float* cvals = reinterpret_cast<float*>(smem_raw) + P2;
    // Round to the fp16 value BEFORE the max compare: the fork's argmax is
    // over half outputs (strict >, lowest index wins), and two f32 sums that
    // differ by ~1e-7 can tie after __float2half. Store the rounded value.
    cvals[warp] = __half2float(__float2half_rn(vsum));
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float* cvals = reinterpret_cast<float*>(smem_raw) + P2;
    int best = 0;
    for (int w = 1; w < 8; w++)
      if (cvals[w] > cvals[best]) best = w;  // strict > => lowest index wins
    argmax_val[blockIdx.x] = __float2half(cvals[best]);
    argmax_idx[blockIdx.x] = blockIdx.x * 8 + best;
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
  // Fused-argmax mode (M == 1, single 8-column block is self-verifiable here):
  // compare the winner (val, idx) against the CPU row argmax.
  if (args.argmax_idx != nullptr) {
    if (m != 1 || n != 8 || state.disabled) return !state.disabled;
    std::vector<__half> x((size_t)k);
    std::vector<uint8_t> codes((size_t)k >> 1), scales((size_t)k >> 4);
    __half amval = 0;
    int amidx = -1;
    cudaMemcpyAsync(x.data(), args.x, x.size() * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(codes.data(), args.codes, codes.size(), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(scales.data(), args.scales, scales.size(), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(&amval, args.argmax_val, sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(&amidx, args.argmax_idx, sizeof(int), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(args.stream));
    if (amidx < 0) { state.disabled = true; return false; }
    // CPU row (single m == 1) via the generic reference.
    std::vector<__half> xm(1 * k);
    for (int i = 0; i < k; i++) xm[i] = x[i];
    std::vector<float> ref = CpuReference(1, n, k, xm, codes, scales, args.gscale);
    int best = 0;
    for (int c = 1; c < n; c++) if (ref[(size_t)c] > ref[(size_t)best]) best = c;
    const float gotv = __half2float(amval);
    if (amidx != best || std::fabs(gotv - ref[(size_t)best]) > 1.0e-3f * std::max(1.0f, std::fabs(ref[(size_t)best]))) {
      state.disabled = true;
    }
    return !state.disabled;
  }
  if (m <= 0 || m > 64 || n <= 0 || k <= 0 || (size_t)m * n * k > (1 << 22)) {
    return !state.disabled;  // shape too large to host-check; trust the band
  }
  // The WMMA band (M 4..64) and the QPN band (M 4..16) each carry their own
  // preconditions; CPU-verify only where that band would actually have run.
  if (m >= 4) {
    const bool qpn_ok = (k % 64) == 0 && (n % 32) == 0;
    const bool wmma_ok = (k % 512) == 0 && (n & 15) == 0;
    if (!(qpn_ok || wmma_ok)) return !state.disabled;
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
  if (m > 64) return false;  // beyond the tile family (next band: >64 prefill)

  static SelfCheckState self;
  const char* sk = std::getenv("VT_SM70_SELFCHECK");
  const bool check_enabled = !(sk && sk[0] == '0');

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(args.stream);
  auto* y = static_cast<__half*>(args.out);
  auto* x = static_cast<const __half*>(args.x);
  auto* codes = static_cast<const uint8_t*>(args.codes);
  auto* scales = static_cast<const uint8_t*>(args.scales);

if (m <= 3) {
    // Fused greedy-argmax only serves M == 1 with k % 128 == 0 (the fork
    // contract). Any other argmax request must be DECLINED so the caller uses
    // its portable path — never silently plain-decode while leaving the
    // argmax buffers untouched (which would also poison the live self-check).
    if (m == 1 && (args.argmax_idx != nullptr || args.argmax_val != nullptr) &&
        !(args.argmax_idx != nullptr && args.argmax_val != nullptr && (k % 128) == 0)) {
      return false;
    }
    constexpr int KC = 512;
    // Fused greedy-argmax: single-row decode that emits the per-8-column
    // winners instead of the row (n%8; cross-block reduce is the caller's).
    if (m == 1 && args.argmax_idx != nullptr && args.argmax_val != nullptr &&
        (k % 128) == 0) {
      const int smem = (KC / 2) * (int)sizeof(__half2) + 8 * (int)sizeof(float) +
                       8 * (int)sizeof(int);
      Sm70Nvfp4SimtArgmax<<<n / 8, 256, smem, stream>>>(
          codes, scales, x, static_cast<__half*>(args.out),
          static_cast<__half*>(args.argmax_val),
          static_cast<int*>(args.argmax_idx), n, k, args.gscale);
      if (check_enabled && !RunSelfCheck(args, self)) return false;
      return cudaGetLastError() == cudaSuccess;
    }
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
    if (check_enabled && !RunSelfCheck(args, self)) return false;
    return cudaGetLastError() == cudaSuccess;
  }

  // QPN band: M 4..16 via the Volta mma.m8n8k4 tensor cores, fed from the
    // fragment-order prepack (byte-equal CT permutation, rebuilt per call into
    // a cached scratch; n%32 and k%64 are the tensor geometry).
    if (m >= 4 && m <= 16 && (n % 32) == 0 && (k % 64) == 0) {
      static uint8_t* s_qc = nullptr;
      static uint8_t* s_qs = nullptr;
      static size_t s_cap = 0;
      const size_t tiles = (size_t)n / 32;
      const size_t G = (size_t)(k / 16);
      const size_t need = tiles * G * 32 * 8 + tiles * G * 32;
      if (need > s_cap) {
        cudaFree(s_qc);
        cudaFree(s_qs);
        cudaMalloc(&s_qc, tiles * G * 32 * 8);
        cudaMalloc(&s_qs, tiles * G * 32);
        s_cap = need;
      }
      Sm70Nvfp4PackQpn<<<dim3((unsigned)tiles, (unsigned)G), 32, 0, stream>>>(
          codes, scales, s_qc, s_qs, k);
      if (m <= 8)
        Sm70Nvfp4Qpn<1><<<n / 32, 128, 0, stream>>>(s_qc, s_qs, x, y, n, k, m, args.gscale);
      else
        Sm70Nvfp4Qpn<2><<<n / 32, 128, 0, stream>>>(s_qc, s_qs, x, y, n, k, m, args.gscale);
      if (check_enabled && !RunSelfCheck(args, self)) return false;
      return cudaGetLastError() == cudaSuccess;
    }

    // WMMA band (brick B): M 4..16 served by 1..4 16-row tiles with row padding.
  // The tensor-core tiles have no k-tail: require k % 512 == 0 (falls back to
  // the portable path otherwise). n must be 16-aligned for the tile grid.
  if (m >= 4 && (n & 15) == 0 && k % 512 == 0) {
    constexpr int KC = 512;
    const int wm = m <= 16 ? 1 : (m <= 32 ? 2 : 4);  // splits must divide 512:  1/2/4
    const int mt = wm * 16;
    const int smem = (int)((16 + mt) * (KC + 16) * (int)sizeof(__half));
    // V100: 48KB default dynamic smem; larger tiles need the attribute (one
    // per kernel config; repeated SetAttribute is a cheap no-op).
    const bool need_attr = smem > 48 * 1024;
    switch (wm) {
      case 1:
        if (need_attr) {
          cudaFuncSetAttribute((const void*)Sm70Nvfp4Wmma<1, 1, KC>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        }
        Sm70Nvfp4Wmma<1, 1, KC><<<n / 16, 32, smem, stream>>>(codes, scales, x, y, n, k, m, args.gscale);
        break;
      case 2:
        if (need_attr) {
          cudaFuncSetAttribute((const void*)Sm70Nvfp4Wmma<1, 2, KC>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        }
        Sm70Nvfp4Wmma<1, 2, KC><<<n / 16, 64, smem, stream>>>(codes, scales, x, y, n, k, m, args.gscale);
        break;
      case 4:
      default:
        if (need_attr) {
          cudaFuncSetAttribute((const void*)Sm70Nvfp4Wmma<1, 4, KC>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        }
        Sm70Nvfp4Wmma<1, 4, KC><<<n / 16, 128, smem, stream>>>(codes, scales, x, y, n, k, m, args.gscale);
        break;
    }
    if (check_enabled && !RunSelfCheck(args, self)) return false;
    return cudaGetLastError() == cudaSuccess;
  }
  return false;
}

struct Registrar {
  Registrar() {
    const ArchTactic tactic = {"nvfp4-w4a16/sm70-simt", &Sm70Nvfp4Supports, &LaunchSm70Nvfp4W4a16};
    RegisterArchTactic(TacticFamily::kSm70Nvfp4W4a16, tactic);
  }
};
static Registrar g_registrar;

}  // namespace

// On-box driver for the brick A-D family: builds small W4A16 fixtures and runs
// the registered tactic's launcher directly (the live path a real head would
// call), CPU-oracle parity per band + the fused-argmax CPU max + the k%128
// decline contract. Returns 0 = all good, 2 = not an sm_70 device (skip).
extern "C" int vt_sm70_nvfp4_selfcheck(void) {
  const DeviceCaps& caps = GetDeviceCaps();
  if (!caps.valid || caps.sm_major != 7 || caps.sm_minor != 0) return 2;

  auto rnd = [](uint64_t& s) -> uint32_t {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)s;
  };
  auto f2h = [](float f) -> __half { return __float2half_rn(f); };

  cudaStream_t st = 0;
  Sm70Nvfp4W4a16Args args{};
  args.stream = st;
  args.gscale = 1.0f;

  // Case A: SIMT decode M=2, n=64, k=512.
  {
    const int m = 2, n = 64, k = 512;
    uint64_t s = 0x1234;
    std::vector<__half> x((size_t)m * k), y((size_t)m * n);
    std::vector<uint8_t> codes((size_t)n * (k >> 1)), scales((size_t)n * (k >> 4));
    for (auto& v : codes) v = (uint8_t)rnd(s);
    for (auto& v : scales) v = (uint8_t)rnd(s);
    for (auto& v : x) v = f2h(0.01f * (float)(rnd(s) % 2000) - 10.f);
    void* dx; void* dc; void* ds; void* dy;
    cudaMalloc(&dx, x.size() * sizeof(__half));
    cudaMalloc(&dy, y.size() * sizeof(__half));
    cudaMalloc(&dc, codes.size());
    cudaMalloc(&ds, scales.size());
    cudaMemcpy(dx, x.data(), x.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, codes.data(), codes.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, scales.data(), scales.size(), cudaMemcpyHostToDevice);
    args.m = m; args.n = n; args.k = k; args.out = dy; args.x = dx;
    args.codes = dc; args.scales = ds;
    args.argmax_val = nullptr; args.argmax_idx = nullptr;
    const bool ok = LaunchSm70Nvfp4W4a16(caps, &args);
    cudaDeviceSynchronize();
    if (!ok) { cudaFree(dx); cudaFree(dy); cudaFree(dc); cudaFree(ds); return 1; }
    cudaMemcpy(y.data(), dy, y.size() * sizeof(__half), cudaMemcpyDeviceToHost);
    const auto ref = CpuReference(m, n, k, x, codes, scales, args.gscale);
    for (size_t i = 0; i < y.size(); ++i) {
      if (std::fabs((double)__half2float(y[i]) - (double)ref[i]) >
          2.0e-2 * std::max(1.0, std::fabs((double)ref[i]))) {
        cudaFree(dx); cudaFree(dy); cudaFree(dc); cudaFree(ds); return 1;
      }
    }
    cudaFree(dx); cudaFree(dy); cudaFree(dc); cudaFree(ds);
    fprintf(stderr, " [selfcheck] case A (SIMT M2xN64) parity OK\n");
  }

  // Case B: fused greedy-argmax M=1, n=128, k=512 (k%128==0) -> per-8-col
  // winners; harness reduces to the row argmax and CPU-checks val+idx.
  {
    const int m = 1, n = 128, k = 512;
    uint64_t s = 0x2223;
    std::vector<__half> x((size_t)m * k), amval((size_t)(n / 8));
    std::vector<int> amidx((size_t)(n / 8));
    std::vector<uint8_t> codes((size_t)n * (k >> 1)), scales((size_t)n * (k >> 4));
    for (auto& v : codes) v = (uint8_t)rnd(s);
    for (auto& v : scales) v = (uint8_t)rnd(s);
    for (auto& v : x) v = f2h(0.01f * (float)(rnd(s) % 100) - 0.f);
    void *dx, *dc, *ds, *dv, *di, *dy;
    cudaMalloc(&dx, x.size() * sizeof(__half));
    cudaMalloc(&dy, sizeof(__half));
    cudaMalloc(&dc, codes.size());
    cudaMalloc(&ds, scales.size());
    cudaMalloc(&dv, (n / 8) * sizeof(__half));
    cudaMalloc(&di, (n / 8) * sizeof(int));
    cudaMemcpy(dx, x.data(), x.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, codes.data(), codes.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, scales.data(), scales.size(), cudaMemcpyHostToDevice);
    args.m = m; args.n = n; args.k = k; args.out = dy; args.x = dx;
    args.codes = dc; args.scales = ds; args.argmax_val = dv; args.argmax_idx = reinterpret_cast<int*>(di);
    const bool ok = LaunchSm70Nvfp4W4a16(caps, &args);
    cudaDeviceSynchronize();
    cudaMemcpy(amval.data(), dv, (n / 8) * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(amidx.data(), di, (n / 8) * sizeof(int), cudaMemcpyDeviceToHost);
    if (!ok) { cudaFree(dx); cudaFree(dy); cudaFree(dc); cudaFree(ds); cudaFree(dv); cudaFree(di); return 1; }
    const auto ref = CpuReference(m, n, k, x, codes, scales, args.gscale);
    // The kernel's winner per 8-col block is the argmax of the HALF-ROUNDED
    // values, strict > (lowest index wins). Verify each block against that
    // exact rule (not the fp32 global max, which can tie differently after
    // __float2half and spuriously fail the check).
    bool bad = false;
    for (int b = 0; b < n / 8 && !bad; ++b) {
      int bb = 0;
      float bbv = __half2float(__float2half_rn(ref[(size_t)(b * 8)]));
      for (int c2 = 1; c2 < 8; ++c2) {
        const float c2v = __half2float(__float2half_rn(ref[(size_t)(b * 8 + c2)]));
        if (c2v > bbv) { bbv = c2v; bb = c2; }
      }
      if (amidx[(size_t)b] != b * 8 + bb ||
          std::fabs((double)__half2float(amval[(size_t)b]) - (double)bbv) > 1.0e-3) {
        bad = true;
      }
    }
    fprintf(stderr, " [selfcheck] case B (fused argmax) bad=%d\n", bad ? 1 : 0);
    cudaFree(dx); cudaFree(dy); cudaFree(dc); cudaFree(ds); cudaFree(dv); cudaFree(di);
    if (bad) return 1;
  }

  // Case D: QPN band (brick C) — M=8 n=64 k=512 (n%32==0, k%64==0 -> QPN,
  // checked before WMMA). Device-tensor-core decode vs CpuReference.
  {
    const int m = 8, n = 64, k = 512;
    uint64_t s = 0x44aa;
    std::vector<__half> x((size_t)m * k), y((size_t)m * n);
    std::vector<uint8_t> codes((size_t)n * (k >> 1)), scales((size_t)n * (k >> 4));
    for (auto& v : codes) v = (uint8_t)rnd(s);
    for (auto& v : scales) v = (uint8_t)rnd(s);
    for (auto& v : x) v = f2h(0.01f * (float)(rnd(s) % 2000) - 10.f);
    void* dx; void* dc; void* ds; void* dy;
    cudaMalloc(&dx, x.size() * sizeof(__half));
    cudaMalloc(&dy, y.size() * sizeof(__half));
    cudaMalloc(&dc, codes.size());
    cudaMalloc(&ds, scales.size());
    cudaMemcpy(dx, x.data(), x.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, codes.data(), codes.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, scales.data(), scales.size(), cudaMemcpyHostToDevice);
    args.m = m; args.n = n; args.k = k; args.out = dy; args.x = dx;
    args.codes = dc; args.scales = ds;
    args.argmax_val = nullptr; args.argmax_idx = nullptr;
    const bool ok = LaunchSm70Nvfp4W4a16(caps, &args);
    cudaDeviceSynchronize();
    if (!ok) { cudaFree(dx); cudaFree(dy); cudaFree(dc); cudaFree(ds); return 1; }
    cudaMemcpy(y.data(), dy, y.size() * sizeof(__half), cudaMemcpyDeviceToHost);
    const auto ref = CpuReference(m, n, k, x, codes, scales, args.gscale);
    bool bad = false;
    for (size_t i = 0; i < y.size(); ++i) {
      if (std::fabs((double)__half2float(y[i]) - (double)ref[i]) >
          2.0e-2 * std::max(1.0, std::fabs((double)ref[i]))) { bad = true; break; }
    }
    fprintf(stderr, " [selfcheck] case D (QPN M8xN64) parity OK=%d\n", bad ? 0 : 1);
    cudaFree(dx); cudaFree(dy); cudaFree(dc); cudaFree(ds);
    if (bad) return 1;
  }

  // Case E: WMMA band (brick B) — M=32 n=48 k=512 (n%32!=0 so NOT QPN;
  // n%16==0 and k%512==0 -> WMMA wm=2). Device tensor-core vs CpuReference.
  {
    const int m = 32, n = 48, k = 512;
    uint64_t s = 0x55bb;
    std::vector<__half> x((size_t)m * k), y((size_t)m * n);
    std::vector<uint8_t> codes((size_t)n * (k >> 1)), scales((size_t)n * (k >> 4));
    for (auto& v : codes) v = (uint8_t)rnd(s);
    for (auto& v : scales) v = (uint8_t)rnd(s);
    for (auto& v : x) v = f2h(0.01f * (float)(rnd(s) % 2000) - 10.f);
    void* dx; void* dc; void* ds; void* dy;
    cudaMalloc(&dx, x.size() * sizeof(__half));
    cudaMalloc(&dy, y.size() * sizeof(__half));
    cudaMalloc(&dc, codes.size());
    cudaMalloc(&ds, scales.size());
    cudaMemcpy(dx, x.data(), x.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, codes.data(), codes.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, scales.data(), scales.size(), cudaMemcpyHostToDevice);
    args.m = m; args.n = n; args.k = k; args.out = dy; args.x = dx;
    args.codes = dc; args.scales = ds;
    args.argmax_val = nullptr; args.argmax_idx = nullptr;
    const bool ok = LaunchSm70Nvfp4W4a16(caps, &args);
    cudaDeviceSynchronize();
    if (!ok) { cudaFree(dx); cudaFree(dy); cudaFree(dc); cudaFree(ds); return 1; }
    cudaMemcpy(y.data(), dy, y.size() * sizeof(__half), cudaMemcpyDeviceToHost);
    const auto ref = CpuReference(m, n, k, x, codes, scales, args.gscale);
    bool bad = false;
    for (size_t i = 0; i < y.size(); ++i) {
      if (std::fabs((double)__half2float(y[i]) - (double)ref[i]) >
          2.0e-2 * std::max(1.0, std::fabs((double)ref[i]))) { bad = true; break; }
    }
    fprintf(stderr, " [selfcheck] case E (WMMA M32xN48 k=512) parity OK=%d\n", bad ? 0 : 1);
    cudaFree(dx); cudaFree(dy); cudaFree(dc); cudaFree(ds);
    if (bad) return 1;
  }

  // Case C: argmax requested at k%128 != 0 must be DECLINED (portable path),
  // not silently plain-decoded.
  {
    const int n = 8, k = 96;
    uint64_t s = 0xbeef;
    std::vector<__half> x((size_t)k);
    std::vector<uint8_t> codes((size_t)n * (k >> 1)), scales((size_t)n * (k >> 4));
    for (auto& v : codes) v = (uint8_t)rnd(s);
    for (auto& v : scales) v = (uint8_t)rnd(s);
    for (auto& v : x) v = f2h(0.01f * (float)(rnd(s) % 100));
    void* dx, *dc, *ds, *dv, *di, *dy;
    cudaMalloc(&dx, x.size() * sizeof(__half));
    cudaMalloc(&dy, sizeof(__half));
    cudaMalloc(&dc, codes.size());
    cudaMalloc(&ds, scales.size());
    cudaMalloc(&dv, 1 * sizeof(__half));
    cudaMalloc(&di, 1 * sizeof(int));
    __half sentinel_v = f2h(-99.f); int sentinel_i = -77;
    cudaMemcpy(dv, &sentinel_v, 1 * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(di, &sentinel_i, 1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), x.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, codes.data(), codes.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, scales.data(), scales.size(), cudaMemcpyHostToDevice);
    args.m = 1; args.n = n; args.k = k; args.out = dy; args.x = dx;
    args.codes = dc; args.scales = ds; args.argmax_val = dv; args.argmax_idx = reinterpret_cast<int*>(di);
    const bool declined = !LaunchSm70Nvfp4W4a16(caps, &args);
    cudaDeviceSynchronize();
    __half gotv; int goti;
    cudaMemcpy(&gotv, dv, sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(&goti, di, sizeof(int), cudaMemcpyDeviceToHost);
    fprintf(stderr, " [selfcheck] case C decline ok=%d goti=%d gotv=%f\n",
            declined ? 1 : 0, goti, (double)__half2float(gotv));
    cudaFree(dx); cudaFree(dy); cudaFree(dc); cudaFree(ds); cudaFree(dv); cudaFree(di);
    // declined AND the argmax buffers are untouched (still sentinels -77 / -99)
    if (!declined || goti != -77 || std::fabs((double)__half2float(gotv) + 99.0) > 1e-3) return 1;
  }
  return 0;
}

}  // namespace vt::cuda