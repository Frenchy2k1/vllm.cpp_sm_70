// CUDA keep-quant GGUF k-quant GEMM (MMVQ-style) — the kCUDA provider for
// `OpId::kMatmulBTQuant` (QUANT-GGUF-CIQ-GEMM-CUDA). Runs the DeepSeek-V4 routed
// experts / MLA projections ON THE GPU with the weights kept COMPRESSED in the
// unified pool (no bf16 expansion): quantize the activation tile to Q8_K on the
// GPU, then integer-dot it against the compressed k-quant weight blocks
// (dequant-in-kernel via the block scales / codebook), exactly as vLLM /
// llama.cpp / ds4 do.
//
// GROUNDING (AGENTS.md: mirror the source, cite file:line on both sides).
// This is NOT a copy of llama.cpp's CUDA MMQ/MMVQ path: that path quantizes the
// activation to Q8_1 (32-wide) and uses its own Q8_1-based `vec_dot`s
// (ggml-cuda/mmvq.cu + vecdotq.cuh), so it would NOT reproduce OUR landed CPU
// keep-quant reference, which follows ggml's CPU tier (Q8_K activation). The
// ORACLE this kernel must match is our CPU `kMatmulBTQuant`:
//   src/vt/cpu/cpu_quant_gemm.cpp        MatmulBTQuantKernel   (the GEMM wiring)
//   src/vt/cpu/cpu_quant_dot.cpp         VecDot{Q2_K,Q3_K,Q4_K,Q5_K,Q6_K,
//                                        IQ2_XXS,IQ3_XXS}Q8_K  (the per-block dot)
//   src/vt/cpu/cpu_quant_act.cpp         QuantizeRowQ8_K       (the activation quant)
// which are themselves 1:1 ports of llama.cpp @ 237ad9b96
//   ggml/src/ggml-cpu/quants.c:514/:566/:645/:720/:800/:855/:999  (the vec_dots)
//   ggml/src/ggml-quants.c:2696                                    (quantize_row_q8_K)
//   ggml/src/ggml-cpu/ggml-cpu.c:1245-1443                         (mul_mat wiring)
// The device numeric helpers (DF16ToF32 / DBF16ToF32 / DF32ToBF16 / DNearestInt)
// are bit-exact ports of src/vt/dtype.cpp + cpu_quant_act.cpp so the Q8_K
// activation bytes — and therefore the whole INTEGER dot — are IDENTICAL to the
// CPU reference. Only the per-super-block float scale sum is reassociated (warp
// reduction vs the CPU's sequential add), so the gate is: INTEGER core bit-exact,
// final scale within the same NMSE band `test_ops_quant_dot` uses (5e-4).
//
// COVERAGE. The seven Q8_K-family encodings (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K,
// IQ2_XXS, IQ3_XXS — all dot against a Q8_K activation) run natively on the GPU;
// DeepSeek-V4's experts are IQ2_XXS / IQ3_XXS / Q2_K. The two legacy Q8_0-
// activation encodings (Q4_0, Q8_0) fall back to the CPU keep-quant kernel over
// the SAME unified-memory tensors (correct, just not GPU-accelerated) — nothing
// in the DeepSeek-V4 vehicle uses them.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "vt/cpu/cpu_quant_blocks.h"        // vt::cpu::Block* struct mirror (single source)
#include "vt/cuda/cuda_quant_iq_tables.cuh"  // d_iq2xxs_grid / d_iq3xxs_grid / d_ksigns / d_kmask
#include "vt/cuda/graph_safe_scratch.h"      // RetireGraphScratch (cudagraph-safe grow-only)
#include "vt/ops.h"
#include "vt/quant.h"

namespace vt::cuda {
namespace {

using vt::cpu::BlockIQ2_XXS;
using vt::cpu::BlockIQ3_XXS;
using vt::cpu::BlockQ2_K;
using vt::cpu::BlockQ3_K;
using vt::cpu::BlockQ4_K;
using vt::cpu::BlockQ5_K;
using vt::cpu::BlockQ6_K;
using vt::cpu::BlockQ8_K;
using vt::cpu::kQK_K;  // 256

void CheckCuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: matmul_bt_quant: ") + what +
                             ": " + cudaGetErrorString(err));
  }
}

// --- device numeric helpers — bit-exact ports of src/vt/dtype.cpp -------------
__device__ inline float DF16ToF32(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  if (exp == 0x1F) return __int_as_float(sign | 0x7F800000 | (mant << 13));
  if (exp == 0) {
    if (mant == 0) return __int_as_float(sign);
    int shift = 0;
    while ((mant & 0x400) == 0) {
      mant <<= 1;
      ++shift;
    }
    mant &= 0x3FF;
    return __int_as_float(sign | ((113 - shift) << 23) | (mant << 13));
  }
  return __int_as_float(sign | ((exp + 112) << 23) | (mant << 13));
}

__device__ inline float DBF16ToF32(uint16_t b) {
  return __int_as_float(static_cast<uint32_t>(b) << 16);
}

__device__ inline uint16_t DF32ToBF16(float f) {
  uint32_t u = __float_as_int(f);
  if ((u & 0x7F800000) == 0x7F800000 && (u & 0x7FFFFF)) {
    return static_cast<uint16_t>((u >> 16) | 0x0040);
  }
  uint32_t rounding = 0x7FFF + ((u >> 16) & 1);
  return static_cast<uint16_t>((u + rounding) >> 16);
}

// cpu_quant_act.cpp NearestInt (ggml-quants.c:563) — magic-constant round-to-even.
__device__ inline int DNearestInt(float fval) {
  float val = fval + 12582912.0f;
  int i = __float_as_int(val);
  return (i & 0x007fffff) - 0x00400000;
}

// Load one activation element (dtype-decoded, exactly like cpu LoadActF32).
enum class ActDT : int { kF32 = 0, kF16 = 1, kBF16 = 2 };

__device__ inline float DLoadAct(const void* base, ActDT dt, int64_t idx) {
  switch (dt) {
    case ActDT::kF32: return static_cast<const float*>(base)[idx];
    case ActDT::kF16: return DF16ToF32(static_cast<const uint16_t*>(base)[idx]);
    default: return DBF16ToF32(static_cast<const uint16_t*>(base)[idx]);
  }
}

// ---------------------------------------------------------------------------
// GPU activation quantizer — one thread per Q8_K super-block (256 elements).
// Bit-exact port of QuantizeRowQ8_K (cpu_quant_act.cpp / ggml-quants.c:2696).
// Scratch layout: row i is `nsb` contiguous BlockQ8_K; block (i,sb) sits at
// scratch[(i*nsb + sb)]. The per-row activation stride `a_rs` is in ELEMENTS.
// ---------------------------------------------------------------------------
__global__ void QuantizeQ8KKernel(BlockQ8_K* __restrict__ scratch,
                                  const void* __restrict__ a, ActDT adt,
                                  int64_t a_rs, int64_t m, int64_t nsb) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = m * nsb;
  if (t >= total) return;
  const int64_t i = t / nsb;   // activation row
  const int64_t sb = t % nsb;  // super-block within the row
  const int64_t elem0 = i * a_rs + sb * kQK_K;

  float mx = 0.0f;
  float amax = 0.0f;
  for (int j = 0; j < kQK_K; ++j) {
    const float ax = fabsf(DLoadAct(a, adt, elem0 + j));
    if (ax > amax) {
      amax = ax;
      mx = DLoadAct(a, adt, elem0 + j);
    }
  }
  BlockQ8_K& y = scratch[t];
  if (amax == 0.0f) {
    y.d = 0.0f;
    for (int j = 0; j < kQK_K; ++j) y.qs[j] = 0;
    for (int g = 0; g < kQK_K / 16; ++g) y.bsums[g] = 0;
    return;
  }
  const float iscale = -127.0f / mx;
  for (int j = 0; j < kQK_K; ++j) {
    const int v = DNearestInt(iscale * DLoadAct(a, adt, elem0 + j));
    y.qs[j] = static_cast<int8_t>(v < 127 ? v : 127);
  }
  for (int g = 0; g < kQK_K / 16; ++g) {
    int sum = 0;
    for (int ii = 0; ii < 16; ++ii) sum += y.qs[g * 16 + ii];
    y.bsums[g] = static_cast<int16_t>(sum);
  }
  y.d = 1.0f / iscale;
}

// ---------------------------------------------------------------------------
// Per-super-block dot kernels. Each returns the block's float contribution with
// the INTEGER core computed bit-identically to the CPU vec_dot; the type's final
// constant magnitude factor (0.125 iq2 / 0.25 iq3 / 1 otherwise) is folded once
// at the very end (matching the CPU `*s = factor * sumf`).
// ---------------------------------------------------------------------------

// cpu_quant_dot.cpp VecDotIQ2_XXSQ8_K (quants.c:855) — one super-block.
__device__ inline float DotIQ2XXS(const BlockIQ2_XXS* xb, const BlockQ8_K* yb) {
  const float d = DF16ToF32(xb->d) * yb->d;
  const uint16_t* qs = xb->qs;
  const int8_t* q8 = yb->qs;
  int32_t bsum = 0;
  for (int ib32 = 0; ib32 < kQK_K / 32; ++ib32) {
    const uint32_t a0 = static_cast<uint32_t>(qs[4 * ib32 + 0]) |
                        (static_cast<uint32_t>(qs[4 * ib32 + 1]) << 16);
    const uint32_t a1 = static_cast<uint32_t>(qs[4 * ib32 + 2]) |
                        (static_cast<uint32_t>(qs[4 * ib32 + 3]) << 16);
    const uint32_t ls = 2 * (a1 >> 28) + 1;
    int32_t sumi = 0;
    for (int l = 0; l < 4; ++l) {
      const uint64_t grid = d_iq2xxs_grid[(a0 >> (8 * l)) & 0xff];
      const uint8_t signs = d_ksigns_iq2xs[(a1 >> (7 * l)) & 127];
      for (int j = 0; j < 8; ++j) {
        const int g = static_cast<int>((grid >> (8 * j)) & 0xff);
        sumi += g * q8[j] * ((signs & d_kmask_iq2xs[j]) ? -1 : 1);
      }
      q8 += 8;
    }
    bsum += sumi * static_cast<int32_t>(ls);
  }
  return d * bsum;  // final *0.125 applied after the warp reduction
}

// cpu_quant_dot.cpp VecDotIQ3_XXSQ8_K (quants.c:999) — one super-block.
__device__ inline float DotIQ3XXS(const BlockIQ3_XXS* xb, const BlockQ8_K* yb) {
  const float d = DF16ToF32(xb->d) * yb->d;
  const uint8_t* q3 = xb->qs;
  const uint8_t* gas = xb->qs + kQK_K / 4;
  const int8_t* q8 = yb->qs;
  int32_t bsum = 0;
  for (int ib32 = 0; ib32 < kQK_K / 32; ++ib32) {
    uint32_t a32;
    memcpy(&a32, gas, sizeof(uint32_t));
    gas += sizeof(uint32_t);
    const uint32_t ls = 2 * (a32 >> 28) + 1;
    int32_t sumi = 0;
    for (int l = 0; l < 4; ++l) {
      const uint32_t g1 = d_iq3xxs_grid[q3[2 * l + 0]];
      const uint32_t g2 = d_iq3xxs_grid[q3[2 * l + 1]];
      const uint8_t signs = d_ksigns_iq2xs[(a32 >> (7 * l)) & 127];
      for (int j = 0; j < 4; ++j) {
        const int b1 = static_cast<int>((g1 >> (8 * j)) & 0xff);
        const int b2 = static_cast<int>((g2 >> (8 * j)) & 0xff);
        sumi += b1 * q8[j + 0] * ((signs & d_kmask_iq2xs[j + 0]) ? -1 : 1);
        sumi += b2 * q8[j + 4] * ((signs & d_kmask_iq2xs[j + 4]) ? -1 : 1);
      }
      q8 += 8;
    }
    q3 += 8;
    bsum += sumi * static_cast<int32_t>(ls);
  }
  return d * bsum;  // final *0.25 applied after the warp reduction
}

// cpu_quant_dot.cpp VecDotQ2_KQ8_K (quants.c:514) — one super-block.
__device__ inline float DotQ2K(const BlockQ2_K* xb, const BlockQ8_K* yb) {
  const uint8_t* q2 = xb->qs;
  const int8_t* q8 = yb->qs;
  const uint8_t* sc = xb->scales;
  int summs = 0;
  for (int j = 0; j < 16; ++j) summs += yb->bsums[j] * (sc[j] >> 4);
  const float dall = yb->d * DF16ToF32(xb->d);
  const float dmin = yb->d * DF16ToF32(xb->dmin);
  int isum = 0;
  int is = 0;
  for (int k = 0; k < kQK_K / 128; ++k) {
    int shift = 0;
    for (int j = 0; j < 4; ++j) {
      int d = sc[is++] & 0xF;
      int isuml = 0;
      for (int l = 0; l < 16; ++l)
        isuml += q8[k * 128 + j * 32 + l] * ((q2[k * 32 + l] >> shift) & 3);
      isum += d * isuml;
      d = sc[is++] & 0xF;
      isuml = 0;
      for (int l = 16; l < 32; ++l)
        isuml += q8[k * 128 + j * 32 + l] * ((q2[k * 32 + l] >> shift) & 3);
      isum += d * isuml;
      shift += 2;
    }
  }
  return dall * isum - dmin * summs;
}

// cpu_quant_dot.cpp VecDotQ3_KQ8_K (quants.c:566) — one super-block.
__device__ inline float DotQ3K(const BlockQ3_K* xb, const BlockQ8_K* yb) {
  const uint32_t kmask1 = 0x03030303;
  const uint32_t kmask2 = 0x0f0f0f0f;
  const uint8_t* hm = xb->hmask;
  const int8_t* q8 = yb->qs;
  int8_t aux8[kQK_K];
  int8_t* a = aux8;
  const uint8_t* q3 = xb->qs;
  uint8_t m = 1;
  for (int jj = 0; jj < kQK_K; jj += 128) {
    for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
    for (int l = 0; l < 32; ++l) a[l] = static_cast<int8_t>(a[l] - ((hm[l] & m) ? 0 : 4));
    a += 32; m = static_cast<uint8_t>(m << 1);
    for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
    for (int l = 0; l < 32; ++l) a[l] = static_cast<int8_t>(a[l] - ((hm[l] & m) ? 0 : 4));
    a += 32; m = static_cast<uint8_t>(m << 1);
    for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
    for (int l = 0; l < 32; ++l) a[l] = static_cast<int8_t>(a[l] - ((hm[l] & m) ? 0 : 4));
    a += 32; m = static_cast<uint8_t>(m << 1);
    for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
    for (int l = 0; l < 32; ++l) a[l] = static_cast<int8_t>(a[l] - ((hm[l] & m) ? 0 : 4));
    a += 32; m = static_cast<uint8_t>(m << 1);
    q3 += 32;
  }
  uint32_t auxs[4];
  memcpy(auxs, xb->scales, 12);
  const int8_t* scales = reinterpret_cast<const int8_t*>(auxs);
  uint32_t tmp = auxs[2];
  auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
  auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
  auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
  auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
  a = aux8;
  const int8_t* q8p = q8;
  int32_t aux32[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int j = 0; j < kQK_K / 16; ++j) {
    for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * (q8p[l] * a[l]);
    q8p += 8; a += 8;
    for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * (q8p[l] * a[l]);
    q8p += 8; a += 8;
  }
  const float d = DF16ToF32(xb->d) * yb->d;
  int isum = 0;
  for (int l = 0; l < 8; ++l) isum += aux32[l];
  return d * isum;
}

// cpu_quant_dot.cpp VecDotQ4_KQ8_K (quants.c:645) — one super-block.
__device__ inline float DotQ4K(const BlockQ4_K* xb, const BlockQ8_K* yb) {
  const uint32_t kmask1 = 0x3f3f3f3f;
  const uint32_t kmask2 = 0x0f0f0f0f;
  const uint32_t kmask3 = 0x03030303;
  const uint8_t* q4 = xb->qs;
  const int8_t* q8 = yb->qs;
  int8_t aux8[kQK_K];
  int8_t* a = aux8;
  for (int j = 0; j < kQK_K / 64; ++j) {
    for (int l = 0; l < 32; ++l) a[l] = static_cast<int8_t>(q4[l] & 0xF);
    a += 32;
    for (int l = 0; l < 32; ++l) a[l] = static_cast<int8_t>(q4[l] >> 4);
    a += 32;
    q4 += 32;
  }
  uint32_t utmp[4];
  memcpy(utmp, xb->scales, 12);
  utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
  const uint32_t uaux = utmp[1] & kmask1;
  utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
  utmp[2] = uaux;
  utmp[0] &= kmask1;
  const uint8_t* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
  const uint8_t* mins = reinterpret_cast<const uint8_t*>(&utmp[2]);
  int sumi = 0;
  for (int j = 0; j < kQK_K / 16; ++j) sumi += yb->bsums[j] * mins[j / 2];
  a = aux8;
  int is = 0;
  const int8_t* q8p = q8;
  int32_t aux32[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int j = 0; j < kQK_K / 32; ++j) {
    const int32_t scale = scales[is++];
    for (int pass = 0; pass < 4; ++pass) {
      for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8p[l] * a[l]);
      q8p += 8; a += 8;
    }
  }
  const float d = DF16ToF32(xb->d) * yb->d;
  int isum = 0;
  for (int l = 0; l < 8; ++l) isum += aux32[l];
  const float dmin = DF16ToF32(xb->dmin) * yb->d;
  return d * isum - dmin * sumi;
}

// cpu_quant_dot.cpp VecDotQ5_KQ8_K (quants.c:720) — one super-block.
__device__ inline float DotQ5K(const BlockQ5_K* xb, const BlockQ8_K* yb) {
  const uint32_t kmask1 = 0x3f3f3f3f;
  const uint32_t kmask2 = 0x0f0f0f0f;
  const uint32_t kmask3 = 0x03030303;
  const uint8_t* q4 = xb->qs;
  const uint8_t* hm = xb->qh;
  const int8_t* q8 = yb->qs;
  int8_t aux8[kQK_K];
  int8_t* a = aux8;
  uint8_t m = 1;
  for (int j = 0; j < kQK_K / 64; ++j) {
    for (int l = 0; l < 32; ++l) a[l] = static_cast<int8_t>(q4[l] & 0xF);
    for (int l = 0; l < 32; ++l) a[l] = static_cast<int8_t>(a[l] + ((hm[l] & m) ? 16 : 0));
    a += 32; m = static_cast<uint8_t>(m << 1);
    for (int l = 0; l < 32; ++l) a[l] = static_cast<int8_t>(q4[l] >> 4);
    for (int l = 0; l < 32; ++l) a[l] = static_cast<int8_t>(a[l] + ((hm[l] & m) ? 16 : 0));
    a += 32; m = static_cast<uint8_t>(m << 1);
    q4 += 32;
  }
  uint32_t utmp[4];
  memcpy(utmp, xb->scales, 12);
  utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
  const uint32_t uaux = utmp[1] & kmask1;
  utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
  utmp[2] = uaux;
  utmp[0] &= kmask1;
  const uint8_t* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
  const uint8_t* mins = reinterpret_cast<const uint8_t*>(&utmp[2]);
  int sumi = 0;
  for (int j = 0; j < kQK_K / 16; ++j) sumi += yb->bsums[j] * mins[j / 2];
  a = aux8;
  int is = 0;
  const int8_t* q8p = q8;
  int32_t aux32[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int j = 0; j < kQK_K / 32; ++j) {
    const int32_t scale = scales[is++];
    for (int pass = 0; pass < 4; ++pass) {
      for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8p[l] * a[l]);
      q8p += 8; a += 8;
    }
  }
  const float d = DF16ToF32(xb->d) * yb->d;
  int isum = 0;
  for (int l = 0; l < 8; ++l) isum += aux32[l];
  const float dmin = DF16ToF32(xb->dmin) * yb->d;
  return d * isum - dmin * sumi;
}

// cpu_quant_dot.cpp VecDotQ6_KQ8_K (quants.c:800) — one super-block.
__device__ inline float DotQ6K(const BlockQ6_K* xb, const BlockQ8_K* yb) {
  const uint8_t* q4 = xb->ql;
  const uint8_t* qh = xb->qh;
  const int8_t* q8 = yb->qs;
  int8_t aux8[kQK_K];
  int8_t* a = aux8;
  for (int j = 0; j < kQK_K; j += 128) {
    for (int l = 0; l < 32; ++l) {
      a[l + 0] = static_cast<int8_t>(
          static_cast<int8_t>((q4[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
      a[l + 32] = static_cast<int8_t>(
          static_cast<int8_t>((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
      a[l + 64] = static_cast<int8_t>(
          static_cast<int8_t>((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
      a[l + 96] = static_cast<int8_t>(
          static_cast<int8_t>((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
    }
    a += 128; q4 += 64; qh += 32;
  }
  a = aux8;
  const int8_t* q8p = q8;
  int is = 0;
  int32_t aux32[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int j = 0; j < kQK_K / 16; ++j) {
    const int scale = xb->scales[is++];
    for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8p[l] * a[l]);
    q8p += 8; a += 8;
    for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8p[l] * a[l]);
    q8p += 8; a += 8;
  }
  const float d = DF16ToF32(xb->d) * yb->d;
  int isum = 0;
  for (int l = 0; l < 8; ++l) isum += aux32[l];
  return d * isum;
}

// The supported Q8_K-family encodings, as small integer tags for the templated
// GEMM. Kept in sync with `IsCudaKeepQuantSupported` below.
enum class WType : int {
  kIQ2_XXS = 0,
  kIQ3_XXS = 1,
  kQ2_K = 2,
  kQ3_K = 3,
  kQ4_K = 4,
  kQ5_K = 5,
  kQ6_K = 6,
};

template <WType W>
__device__ inline float DotSuperblock(const void* w_sb, const BlockQ8_K* a_sb);
template <>
__device__ inline float DotSuperblock<WType::kIQ2_XXS>(const void* w, const BlockQ8_K* a) {
  return DotIQ2XXS(static_cast<const BlockIQ2_XXS*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kIQ3_XXS>(const void* w, const BlockQ8_K* a) {
  return DotIQ3XXS(static_cast<const BlockIQ3_XXS*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kQ2_K>(const void* w, const BlockQ8_K* a) {
  return DotQ2K(static_cast<const BlockQ2_K*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kQ3_K>(const void* w, const BlockQ8_K* a) {
  return DotQ3K(static_cast<const BlockQ3_K*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kQ4_K>(const void* w, const BlockQ8_K* a) {
  return DotQ4K(static_cast<const BlockQ4_K*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kQ5_K>(const void* w, const BlockQ8_K* a) {
  return DotQ5K(static_cast<const BlockQ5_K*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kQ6_K>(const void* w, const BlockQ8_K* a) {
  return DotQ6K(static_cast<const BlockQ6_K*>(w), a);
}

template <WType W>
__device__ constexpr float FinalFactor() {
  return W == WType::kIQ2_XXS ? 0.125f : (W == WType::kIQ3_XXS ? 0.25f : 1.0f);
}

// ---------------------------------------------------------------------------
// The MMVQ-style GEMM: one WARP per output element (i,j). The 32 lanes split the
// K super-blocks (lane `w` handles sb = w, w+32, ...), each computing the exact
// integer core + per-block float scale, then a warp reduction sums the partials.
// out[i,j] = FinalFactor * sum_sb DotSuperblock(weight_row_j[sb], act_row_i[sb]).
// Determinism note: the integer core is order-independent (exact); only the
// float scale sum is reassociated (warp tree vs CPU sequential) — within NMSE.
// ---------------------------------------------------------------------------
template <WType W, typename OutT>
__global__ void QuantDotGemmKernel(OutT* __restrict__ out,
                                   const uint8_t* __restrict__ weight,
                                   const BlockQ8_K* __restrict__ act, int64_t m,
                                   int64_t n, int64_t nsb, size_t w_row_bytes,
                                   size_t w_block_bytes) {
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (warp >= m * n) return;  // uniform across the whole warp (idx independent of lane)
  const int64_t i = warp / n;
  const int64_t j = warp % n;
  const int lane = threadIdx.x;

  const uint8_t* w_row = weight + static_cast<size_t>(j) * w_row_bytes;
  const BlockQ8_K* a_row = act + i * nsb;

  float partial = 0.0f;
  for (int64_t sb = lane; sb < nsb; sb += 32) {
    const void* w_sb = w_row + static_cast<size_t>(sb) * w_block_bytes;
    partial += DotSuperblock<W>(w_sb, a_row + sb);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    partial += __shfl_down_sync(0xffffffffu, partial, off);

  if (lane == 0) {
    const float v = FinalFactor<W>() * partial;
    if constexpr (sizeof(OutT) == 4) {
      out[i * n + j] = v;
    } else {
      out[i * n + j] = DF32ToBF16(v);  // OutT == uint16_t (bf16)
    }
  }
}

// GROUPED variant (kMatmulBTQuantGrouped): warp per (p, n); the weight row is
// selected by the per-group expert index — row (expert_ids[p]*N + n) of the
// stacked [E*N,K] block weight. Same integer-dot core as QuantDotGemmKernel; the
// ONLY difference is the weight-row index, so it is numerically identical to the
// per-expert kMatmulBTQuant. Collapses the DeepSeek-V4 MoE's per-expert matvecs
// into one launch with P*N warps of parallelism (higher GB10 occupancy at T=1).
template <WType W, typename OutT>
__global__ void QuantDotGemmGroupedKernel(OutT* __restrict__ out,
                                          const uint8_t* __restrict__ weight,
                                          const BlockQ8_K* __restrict__ act,
                                          const int32_t* __restrict__ expert_ids,
                                          int64_t P, int64_t n, int64_t nsb,
                                          size_t w_row_bytes, size_t w_block_bytes) {
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (warp >= P * n) return;
  const int64_t p = warp / n;
  const int64_t j = warp % n;
  const int lane = threadIdx.x;

  const int64_t e = expert_ids[p];
  const uint8_t* w_row = weight + static_cast<size_t>(e * n + j) * w_row_bytes;
  const BlockQ8_K* a_row = act + p * nsb;

  float partial = 0.0f;
  for (int64_t sb = lane; sb < nsb; sb += 32) {
    const void* w_sb = w_row + static_cast<size_t>(sb) * w_block_bytes;
    partial += DotSuperblock<W>(w_sb, a_row + sb);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    partial += __shfl_down_sync(0xffffffffu, partial, off);

  if (lane == 0) {
    const float v = FinalFactor<W>() * partial;
    if constexpr (sizeof(OutT) == 4) {
      out[p * n + j] = v;
    } else {
      out[p * n + j] = DF32ToBF16(v);
    }
  }
}

// --- per-stream grow-only Q8_K activation scratch (cudagraph-safe) -----------
struct StreamScratch {
  void* buf = nullptr;
  size_t bytes = 0;
};
StreamScratch& ScratchFor(cudaStream_t s) {
  static std::mutex mu;
  static std::unordered_map<cudaStream_t, StreamScratch> m;
  std::lock_guard<std::mutex> lk(mu);
  return m[s];
}
void* EnsureScratch(size_t need, cudaStream_t s) {
  StreamScratch& sc = ScratchFor(s);
  if (need > sc.bytes) {
    // Retire (never free) the old block: a captured decode graph may have baked
    // this pointer — freeing it would dangle on replay. See graph_safe_scratch.h.
    RetireGraphScratch(sc.buf);
    CheckCuda(cudaMallocAsync(&sc.buf, need, s), "cudaMallocAsync q8_K act scratch");
    sc.bytes = need;
  }
  return sc.buf;
}

bool IsCudaKeepQuantSupported(DType dt, WType* out) {
  switch (dt) {
    case DType::kIQ2_XXS: *out = WType::kIQ2_XXS; return true;
    case DType::kIQ3_XXS: *out = WType::kIQ3_XXS; return true;
    case DType::kQ2_K: *out = WType::kQ2_K; return true;
    case DType::kQ3_K: *out = WType::kQ3_K; return true;
    case DType::kQ4_K: *out = WType::kQ4_K; return true;
    case DType::kQ5_K: *out = WType::kQ5_K; return true;
    case DType::kQ6_K: *out = WType::kQ6_K; return true;
    default: return false;  // Q4_0 / Q8_0 (Q8_0-activation) -> CPU fallback
  }
}

template <WType W>
void LaunchGemm(Tensor& out, const uint8_t* weight, const BlockQ8_K* act,
                int64_t m, int64_t n, int64_t nsb, size_t w_row_bytes,
                size_t w_block_bytes, cudaStream_t s) {
  constexpr int kWarpsPerBlock = 4;
  dim3 block(32, kWarpsPerBlock);
  const int64_t warps = m * n;
  const int64_t grid = (warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
  if (out.dtype == DType::kF32) {
    QuantDotGemmKernel<W, float><<<static_cast<unsigned>(grid), block, 0, s>>>(
        static_cast<float*>(out.data), weight, act, m, n, nsb, w_row_bytes,
        w_block_bytes);
  } else {
    QuantDotGemmKernel<W, uint16_t><<<static_cast<unsigned>(grid), block, 0, s>>>(
        static_cast<uint16_t*>(out.data), weight, act, m, n, nsb, w_row_bytes,
        w_block_bytes);
  }
}

// The kCUDA provider for OpId::kMatmulBTQuant. Validation already done by
// vt::MatmulBTQuant (ops.cpp) before dispatch; this mirrors the CPU kernel's
// contract: b is [N,K] block-quant, a is [M,K] f32/bf16 (row-packed), out [M,N].
void MatmulBTQuantKernelCuda(Queue& q, Tensor& out, const Tensor& a,
                             const Tensor& b) {
  cudaStream_t s = static_cast<cudaStream_t>(q.handle);
  const int64_t m = a.shape[0];
  const int64_t k = a.shape[1];
  const int64_t n = b.shape[0];
  if (m == 0 || n == 0) return;

  WType w{};
  if (!IsCudaKeepQuantSupported(b.dtype, &w)) {
    // Legacy Q8_0-activation encodings (Q4_0/Q8_0) — not used by DeepSeek-V4.
    // Run the CPU keep-quant kernel over the SAME unified-memory tensors: drain
    // the stream first so any GPU-produced activation is visible to the host.
    CheckCuda(cudaStreamSynchronize(s), "keepquant CPU-fallback drain");
    reinterpret_cast<MatmulFn>(GetOp(OpId::kMatmulBTQuant, DeviceType::kCPU))(q, out, a, b);
    return;
  }

  // K must be a whole number of Q8_K super-blocks (256). vt::MatmulBTQuant has
  // already checked K % BlockElems(weight) == 0; the K-quants ARE 256-blocked,
  // so this is the same fact — assert defensively.
  if (k % kQK_K != 0) {
    throw std::runtime_error(
        "vt cuda: matmul_bt_quant: K must be a whole number of 256-element "
        "Q8_K super-blocks");
  }
  const int64_t nsb = k / kQK_K;
  const size_t w_block_bytes = static_cast<size_t>(BlockBytes(b.dtype));
  const size_t w_row_bytes = static_cast<size_t>(nsb) * w_block_bytes;

  // 1. Quantize the M activation rows to Q8_K on the GPU (ggml-cpu.c:1313-1349
  //    "src1 -> wdata", done once per GEMM). Scratch is per-stream, grow-only.
  const size_t act_bytes = static_cast<size_t>(m) * static_cast<size_t>(nsb) *
                           sizeof(BlockQ8_K);
  BlockQ8_K* act = static_cast<BlockQ8_K*>(EnsureScratch(act_bytes, s));

  ActDT adt = a.dtype == DType::kF32 ? ActDT::kF32
              : a.dtype == DType::kF16 ? ActDT::kF16
                                       : ActDT::kBF16;
  const int64_t total_sb = m * nsb;
  {
    constexpr int kQBlock = 128;
    const int64_t grid = (total_sb + kQBlock - 1) / kQBlock;
    QuantizeQ8KKernel<<<static_cast<unsigned>(grid), kQBlock, 0, s>>>(
        act, a.data, adt, a.stride[0], m, nsb);
  }

  // 2. The integer dot GEMM (one warp per output), dequant-in-kernel.
  const uint8_t* weight = static_cast<const uint8_t*>(b.data);
  switch (w) {
    case WType::kIQ2_XXS: LaunchGemm<WType::kIQ2_XXS>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kIQ3_XXS: LaunchGemm<WType::kIQ3_XXS>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kQ2_K: LaunchGemm<WType::kQ2_K>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kQ3_K: LaunchGemm<WType::kQ3_K>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kQ4_K: LaunchGemm<WType::kQ4_K>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kQ5_K: LaunchGemm<WType::kQ5_K>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kQ6_K: LaunchGemm<WType::kQ6_K>(out, weight, act, m, n, nsb, w_row_bytes, w_block_bytes, s); break;
  }
  CheckCuda(cudaGetLastError(), "matmul_bt_quant launch");
}

template <WType W>
void LaunchGroupedGemm(Tensor& out, const uint8_t* weight, const BlockQ8_K* act,
                       const int32_t* expert_ids, int64_t P, int64_t n, int64_t nsb,
                       size_t w_row_bytes, size_t w_block_bytes, cudaStream_t s) {
  constexpr int kWarpsPerBlock = 4;
  dim3 block(32, kWarpsPerBlock);
  const int64_t warps = P * n;
  const int64_t grid = (warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
  if (out.dtype == DType::kF32) {
    QuantDotGemmGroupedKernel<W, float><<<static_cast<unsigned>(grid), block, 0, s>>>(
        static_cast<float*>(out.data), weight, act, expert_ids, P, n, nsb, w_row_bytes,
        w_block_bytes);
  } else {
    QuantDotGemmGroupedKernel<W, uint16_t><<<static_cast<unsigned>(grid), block, 0, s>>>(
        static_cast<uint16_t*>(out.data), weight, act, expert_ids, P, n, nsb, w_row_bytes,
        w_block_bytes);
  }
}

// The kCUDA provider for OpId::kMatmulBTQuantGrouped. out[P,N], act[P,K],
// weight[E*N,K] block-quant, expert_ids[P] i32 (unified memory). Validation done
// by vt::MatmulBTQuantGrouped. Quantizes the P activation rows to Q8_K once, then
// one grouped-kernel launch computes every (p,n) output — the expert-batched
// analog of MatmulBTQuantKernelCuda.
void MatmulBTQuantGroupedKernelCuda(Queue& q, Tensor& out, const Tensor& act,
                                    const Tensor& weight, const Tensor& expert_ids) {
  cudaStream_t s = static_cast<cudaStream_t>(q.handle);
  const int64_t P = out.shape[0];
  const int64_t n = out.shape[1];
  const int64_t k = act.shape[1];
  if (P == 0 || n == 0) return;

  WType w{};
  if (!IsCudaKeepQuantSupported(weight.dtype, &w)) {
    CheckCuda(cudaStreamSynchronize(s), "keepquant-grouped CPU-fallback drain");
    reinterpret_cast<MatmulBTQuantGroupedFn>(
        GetOp(OpId::kMatmulBTQuantGrouped, DeviceType::kCPU))(q, out, act, weight, expert_ids);
    return;
  }
  if (k % kQK_K != 0) {
    throw std::runtime_error(
        "vt cuda: matmul_bt_quant_grouped: K must be a whole number of 256-element "
        "Q8_K super-blocks");
  }
  const int64_t nsb = k / kQK_K;
  const size_t w_block_bytes = static_cast<size_t>(BlockBytes(weight.dtype));
  const size_t w_row_bytes = static_cast<size_t>(nsb) * w_block_bytes;

  // Quantize the P activation rows to Q8_K (per-stream grow-only scratch).
  const size_t act_bytes = static_cast<size_t>(P) * static_cast<size_t>(nsb) * sizeof(BlockQ8_K);
  BlockQ8_K* qact = static_cast<BlockQ8_K*>(EnsureScratch(act_bytes, s));
  ActDT adt = act.dtype == DType::kF32 ? ActDT::kF32
              : act.dtype == DType::kF16 ? ActDT::kF16
                                         : ActDT::kBF16;
  {
    constexpr int kQBlock = 128;
    const int64_t total_sb = P * nsb;
    const int64_t grid = (total_sb + kQBlock - 1) / kQBlock;
    QuantizeQ8KKernel<<<static_cast<unsigned>(grid), kQBlock, 0, s>>>(
        qact, act.data, adt, act.stride[0], P, nsb);
  }

  const uint8_t* wt = static_cast<const uint8_t*>(weight.data);
  const int32_t* eids = static_cast<const int32_t*>(expert_ids.data);
  switch (w) {
    case WType::kIQ2_XXS: LaunchGroupedGemm<WType::kIQ2_XXS>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kIQ3_XXS: LaunchGroupedGemm<WType::kIQ3_XXS>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kQ2_K: LaunchGroupedGemm<WType::kQ2_K>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kQ3_K: LaunchGroupedGemm<WType::kQ3_K>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kQ4_K: LaunchGroupedGemm<WType::kQ4_K>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kQ5_K: LaunchGroupedGemm<WType::kQ5_K>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, s); break;
    case WType::kQ6_K: LaunchGroupedGemm<WType::kQ6_K>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, s); break;
  }
  CheckCuda(cudaGetLastError(), "matmul_bt_quant_grouped launch");
}

// Registers the CUDA keep-quant GEMM during static init (table fill only, no
// CUDA calls — same rationale as cuda_matmul.cu Registrar). This makes
// vt::OpRegistered(kMatmulBTQuant, kCUDA) TRUE, which flips the GGUF loader's
// keep-quant default ON on a CUDA device (gguf_keep_quant.cpp
// GgufQuantComputeAvailable) so DeepSeek-V4's experts/MLA GEMMs dispatch here
// instead of the unified-memory CPU reference tier.
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMatmulBTQuant, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<MatmulFn>(&MatmulBTQuantKernelCuda)));
    RegisterOp(OpId::kMatmulBTQuantGrouped, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<MatmulBTQuantGroupedFn>(&MatmulBTQuantGroupedKernelCuda)));
  }
} registrar;

}  // namespace
}  // namespace vt::cuda
