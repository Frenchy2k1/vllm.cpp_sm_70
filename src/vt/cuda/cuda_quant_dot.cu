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
using vt::cpu::BlockQ8_0;
using vt::cpu::BlockQ8_K;
using vt::cpu::kQK8_0;  // 32
using vt::cpu::kQK_K;   // 256

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

// dtype.cpp F32ToF16 — bit-exact port (round-to-nearest-even, subnormals, inf/nan).
// Used only for the Q8_0-activation scale `y.d` (the round-trip F32ToF16→F16ToF32 the
// CPU Q8_0 vec_dot applies); the integer core is scale-independent, so the whole Q8_0
// INTEGER dot stays bit-identical to the CPU reference.
__device__ inline uint16_t DF32ToF16(float f) {
  uint32_t u = __float_as_uint(f);
  uint16_t sign = static_cast<uint16_t>((u >> 16) & 0x8000);
  int32_t exp = static_cast<int32_t>((u >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = u & 0x7FFFFF;
  if (((u >> 23) & 0xFF) == 0xFF)
    return static_cast<uint16_t>(sign | 0x7C00 | (mant ? 0x200 | (mant >> 13) : 0));
  if (exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00);
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant |= 0x800000;
    uint32_t shift = static_cast<uint32_t>(14 - exp);
    uint32_t half = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1);
    uint32_t mid = 1u << (shift - 1);
    if (rem > mid || (rem == mid && (half & 1))) ++half;
    return static_cast<uint16_t>(sign | half);
  }
  uint32_t half = static_cast<uint32_t>(exp << 10) | (mant >> 13);
  uint32_t rem = mant & 0x1FFF;
  if (rem > 0x1000 || (rem == 0x1000 && (half & 1))) ++half;
  return static_cast<uint16_t>(sign | half);
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
// Brick 1 (last-mile): __dp4a vectorized-dequant matvec, ported from llama.cpp
// ggml-cuda/vecdotq.cuh:920-928 (`vec_dot_iq2_xxs_q8_1`) + ds4 `dev_iq2_dp4a_8`
// (ds4_cuda.cu:16147). The per-element `g*q8[j]*sign` branch → SIMD sign-apply
// (__vcmpne4/__vsub4) + `__dp4a` (4 int8 products/instr). BIT-IDENTICAL integer core:
// __dp4a is an EXACT int32 accumulation of int8 products (order-independent), the
// grid bytes are ≤~43 so ±g fits int8, and d_kmask_iq2xs[j]==1<<j so the packed
// sign masks 0x08040201 / 0x80402010 select the same per-byte sign as the scalar.
// The per-block ls fold + final *0.125 (after the warp reduction) are unchanged.
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
      const uint32_t* grid =
          reinterpret_cast<const uint32_t*>(&d_iq2xxs_grid[(a0 >> (8 * l)) & 0xff]);
      // Broadcast the 8-bit sign pattern to 4 bytes (UNSIGNED — signed *0x01010101
      // overflows int for signs≥128 = UB), then per-byte 0xff mask where bit b/b+4 set.
      const unsigned sbc =
          static_cast<unsigned>(d_ksigns_iq2xs[(a1 >> (7 * l)) & 127]) * 0x01010101u;
      const int slo = __vcmpne4(static_cast<int>(sbc & 0x08040201u), 0);  // bytes 0-3
      const int shi = __vcmpne4(static_cast<int>(sbc & 0x80402010u), 0);  // bytes 4-7
      const int glo = __vsub4(static_cast<int>(grid[0]) ^ slo, slo);  // ±grid bytes 0-3
      const int ghi = __vsub4(static_cast<int>(grid[1]) ^ shi, shi);  // ±grid bytes 4-7
      sumi = __dp4a(glo, *reinterpret_cast<const int*>(q8 + 0), sumi);
      sumi = __dp4a(ghi, *reinterpret_cast<const int*>(q8 + 4), sumi);
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
// Brick 1 (last-mile): __dp4a vectorized-dequant, ported from llama.cpp
// ggml-cuda/vecdotq.cuh:329-354 (`vec_dot_q2_K_q8_1`) + ds4 `dev_dot_q2_16`
// (ds4_cuda.cu:16158). The scalar per-element `q8 * ((q2>>shift)&3)` → `__dp4a`
// on the 0x03030303-masked 2-bit packs. BIT-IDENTICAL: `(word>>shift)&0x03030303`
// per byte == `(byte>>shift)&3` (the cross-byte bits land in bits 6-7, masked off),
// __dp4a is exact int32; the summs (min) term + dall/dmin fold are unchanged. All
// int reads are 4-aligned (Q2_K block=84 B ÷4, qs@16 ÷4; the Q8_K activation ÷4).
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
    const uint8_t* q2k = q2 + k * 32;
    const int8_t* q8k = q8 + k * 128;
    int shift = 0;
    for (int j = 0; j < 4; ++j) {
      int sl = 0, sh = 0;
      for (int l = 0; l < 16; l += 4) {
        const int v = (*reinterpret_cast<const int*>(q2k + l) >> shift) & 0x03030303;
        sl = __dp4a(v, *reinterpret_cast<const int*>(q8k + j * 32 + l), sl);
      }
      isum += (sc[is++] & 0xF) * sl;
      for (int l = 16; l < 32; l += 4) {
        const int v = (*reinterpret_cast<const int*>(q2k + l) >> shift) & 0x03030303;
        sh = __dp4a(v, *reinterpret_cast<const int*>(q8k + j * 32 + l), sh);
      }
      isum += (sc[is++] & 0xF) * sh;
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
                                          size_t w_row_bytes, size_t w_block_bytes,
                                          bool bcast) {
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (warp >= P * n) return;
  const int64_t p = warp / n;
  const int64_t j = warp % n;
  const int lane = threadIdx.x;

  const int64_t e = expert_ids[p];
  const uint8_t* w_row = weight + static_cast<size_t>(e * n + j) * w_row_bytes;
  // Broadcast activation: the routed gate/up share ONE quantized hidden (all P
  // experts see the SAME x), so a 1-row Q8_K feeds every p — bit-identical to the
  // per-row path (identical input ⇒ identical Q8_K ⇒ identical integer dot).
  const BlockQ8_K* a_row = act + (bcast ? 0 : p) * nsb;

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

// FUSED gate+up+silu grouped kernel — the ds4 `moe_gate_up_mid` epilogue
// (ds4_cuda.cu moe_gate_up_mid_decode_lut_qwarp32_kernel:17127). ONE warp per
// (p,j) computes BOTH the gate dot (gate_w[e,j]·xq) AND the up dot (up_w[e,j]·xq)
// against the SAME broadcast Q8_K activation, then writes the clamped-SwiGLU
// product adown[p*n+j] = silu(min(gate,limit)) · clamp(up,±limit). This collapses
// the resident-decode routed-MoE's {gate grouped-GEMM + up grouped-GEMM + topk×2
// AsyncCopyF + topk ClampedSwiGLU} into ONE launch and NEVER writes the gate/up
// intermediates to HBM (they stay in registers). BIT-IDENTICAL to that chain: the
// SAME DotSuperblock integer core, the SAME 32-lane warp-tree reduce, the SAME
// FinalFactor, and the SAME ClampedSwiGLUKernel formula with alpha=1,beta=0
// (cuda_deepseek_v4.cu:612-619, Sig(x)=1/(1+e^-x)). The route weight is NOT folded
// here — it stays in moe_combine (post-down), preserving the down-GEMM's exact
// input bytes → the whole change is a pure launch-count + HBM-traffic fusion.
template <WType W>
__global__ void QuantDotGemmGroupedFusedSwiGLUKernel(float* __restrict__ out,
                                                     const uint8_t* __restrict__ gate_w,
                                                     const uint8_t* __restrict__ up_w,
                                                     const BlockQ8_K* __restrict__ act,
                                                     const int32_t* __restrict__ expert_ids,
                                                     int64_t P, int64_t n, int64_t nsb,
                                                     size_t w_row_bytes, size_t w_block_bytes,
                                                     float limit, bool bcast) {
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (warp >= P * n) return;
  const int64_t p = warp / n;
  const int64_t j = warp % n;
  const int lane = threadIdx.x;

  const int64_t e = expert_ids[p];
  const uint8_t* g_row = gate_w + static_cast<size_t>(e * n + j) * w_row_bytes;
  const uint8_t* u_row = up_w + static_cast<size_t>(e * n + j) * w_row_bytes;
  const BlockQ8_K* a_row = act + (bcast ? 0 : p) * nsb;

  float pg = 0.0f, pu = 0.0f;
  for (int64_t sb = lane; sb < nsb; sb += 32) {
    const void* gw_sb = g_row + static_cast<size_t>(sb) * w_block_bytes;
    const void* uw_sb = u_row + static_cast<size_t>(sb) * w_block_bytes;
    pg += DotSuperblock<W>(gw_sb, a_row + sb);
    pu += DotSuperblock<W>(uw_sb, a_row + sb);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    pg += __shfl_down_sync(0xffffffffu, pg, off);
    pu += __shfl_down_sync(0xffffffffu, pu, off);
  }
  if (lane == 0) {
    // == ClampedSwiGLUKernel(gate_up, mi, limit, alpha=1, beta=0): out[i] =
    //    gate·Sig(gate)·up, gate=min(g,limit), up=clamp(u,±limit). Bit-identical.
    const float gate = fminf(FinalFactor<W>() * pg, limit);
    const float up = fminf(fmaxf(FinalFactor<W>() * pu, -limit), limit);
    out[p * n + j] = gate * (1.0f / (1.0f + expf(-gate))) * up;
  }
}

template <WType W>
void LaunchGroupedFusedSwiGLU(Tensor& out, const uint8_t* gate_w, const uint8_t* up_w,
                              const BlockQ8_K* act, const int32_t* expert_ids, int64_t P,
                              int64_t n, int64_t nsb, size_t w_row_bytes, size_t w_block_bytes,
                              float limit, bool bcast, cudaStream_t s) {
  constexpr int kWarpsPerBlock = 4;
  dim3 block(32, kWarpsPerBlock);
  const int64_t warps = P * n;
  const int64_t grid = (warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
  QuantDotGemmGroupedFusedSwiGLUKernel<W><<<static_cast<unsigned>(grid), block, 0, s>>>(
      static_cast<float*>(out.data), gate_w, up_w, act, expert_ids, P, n, nsb, w_row_bytes,
      w_block_bytes, limit, bcast);
}

// ===========================================================================
// Q8_0 keep-quant GEMM — the DeepSeek-V4 MLA projections / o-LoRA / shared
// experts / lm_head (the "AProjQ8/SExpQ8/OutQ8" weights) run ON THE GPU instead
// of the CPU keep-quant fallback (which drained the stream + made the decode
// step uncapturable). Q8_0 is a LEGACY (32-element, single-fp16-scale) encoding
// whose CPU vec_dot pairs it with a Q8_0 ACTIVATION (not the K-quants' Q8_K), so
// this is a self-contained path: quantize the activation to Q8_0 on the GPU, then
// the Q8_0×Q8_0 integer dot. ORACLE = our CPU reference:
//   cpu_quant_act.cpp QuantizeRowQ8_0 (ggml-quants.c quantize_row_q8_0) — the quant
//   cpu_quant_dot.cpp VecDotQ8_0Q8_0  (quants.c:400)                    — the dot
// The INTEGER core (Σ x.qs·y.qs per 32-block) is bit-identical; only the per-block
// float scale sum is reassociated (warp tree vs CPU sequential) — the same near-tie
// band the K-quant path is gated at (NMSE 5e-4).
// ---------------------------------------------------------------------------
// GPU Q8_0 activation quantizer — one thread per 32-element block. Bit-exact port
// of QuantizeRowQ8_0 (ternary amax, d=amax/127, y.d=F32ToF16(d), qs=roundf(x·id)).
__global__ void QuantizeQ8_0Kernel(BlockQ8_0* __restrict__ scratch,
                                   const void* __restrict__ a, ActDT adt, int64_t a_rs,
                                   int64_t m, int64_t nb) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= m * nb) return;
  const int64_t i = t / nb;   // activation row
  const int64_t b = t % nb;   // 32-block within the row
  const int64_t elem0 = i * a_rs + b * kQK8_0;
  float amax = 0.0f;
  for (int j = 0; j < kQK8_0; ++j) {
    const float av = fabsf(DLoadAct(a, adt, elem0 + j));
    amax = amax > av ? amax : av;  // ternary MAX (matches CPU, NaN-propagating)
  }
  BlockQ8_0& y = scratch[t];
  const float d = amax / 127.0f;
  const float id = d != 0.0f ? 1.0f / d : 0.0f;
  y.d = DF32ToF16(d);
  for (int j = 0; j < kQK8_0; ++j) {
    const float x0 = DLoadAct(a, adt, elem0 + j) * id;
    y.qs[j] = static_cast<int8_t>(roundf(x0));  // round half away from zero (== std::roundf)
  }
}

// Brick 3 (last-mile): read a 4-byte int from a 2-BYTE-aligned int8 stream. The Q8_0
// block `qs` starts at offset 2 in the 34-byte block (uint16 d + 32×int8), so — unlike
// the 4-byte-aligned Q8_K qs the Brick-1 IQ2/Q2_K path int-loads directly — a naked
// int32 load here would be MIS-ALIGNED (UB / fault on half the blocks). Bit-exact port
// of llama.cpp `ggml-cuda/common.cuh:get_int_b2` (two uint16 loads, little-endian) — the
// reconstructed byte pattern is identical to a valid int32 load, so __dp4a extracts the
// same signed int8 lanes as the scalar `(int)qs[p]`.
__device__ __forceinline__ int GetIntB2(const int8_t* qs, int i32) {
  const uint16_t* x16 = reinterpret_cast<const uint16_t*>(qs);
  return static_cast<int>(x16[2 * i32 + 0]) | (static_cast<int>(x16[2 * i32 + 1]) << 16);
}

// Q8_0×Q8_0 GEMM: one warp per output (i,j); lane `w` handles blocks b=w,w+32,…,
// each a 32-element integer dot scaled by f16(wd)·f16(ad); warp-reduce the partials.
// Brick 3: the per-block dot reads the 32 int8 as 8 int32 (`GetIntB2`, coalesced 2-byte
// loads) + 8 `__dp4a` (mirrors llama.cpp `ggml-cuda/vecdotq.cuh:vec_dot_q8_0_q8_1_impl`
// + `VDR_Q8_0_Q8_1_MMVQ`) instead of 32 scattered int8 loads + scalar MACs. BIT-IDENTICAL
// (__dp4a = exact int32 accumulation; integer sumi unchanged; the f16-scale fold unchanged).
template <typename OutT>
__global__ void QuantDotGemmQ8_0Kernel(OutT* __restrict__ out,
                                       const uint8_t* __restrict__ weight,
                                       const BlockQ8_0* __restrict__ act, int64_t m, int64_t n,
                                       int64_t nb, size_t w_row_bytes) {
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (warp >= m * n) return;
  const int64_t i = warp / n;
  const int64_t j = warp % n;
  const int lane = threadIdx.x;
  const uint8_t* w_row = weight + static_cast<size_t>(j) * w_row_bytes;
  const BlockQ8_0* a_row = act + i * nb;
  float partial = 0.0f;
  for (int64_t b = lane; b < nb; b += 32) {
    const BlockQ8_0* wb = reinterpret_cast<const BlockQ8_0*>(w_row + static_cast<size_t>(b) *
                                                                          sizeof(BlockQ8_0));
    const BlockQ8_0* ab = a_row + b;
    int sumi = 0;
#pragma unroll
    for (int k = 0; k < kQK8_0 / 4; ++k) sumi = __dp4a(GetIntB2(wb->qs, k), GetIntB2(ab->qs, k), sumi);
    partial += sumi * (DF16ToF32(wb->d) * DF16ToF32(ab->d));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) partial += __shfl_down_sync(0xffffffffu, partial, off);
  if (lane == 0) {
    if constexpr (sizeof(OutT) == 4) out[i * n + j] = partial;
    else out[i * n + j] = DF32ToBF16(partial);
  }
}

// Brick 4 (last-mile): Q8_0 GEMM over the CUDA COALESCED-LOAD layout (RepackQ8_0Cuda).
// The weight tensor is deinterleaved into two contiguous sections — qs `[nblk*32]`
// (16-byte-aligned per block) then scales `[nblk]` uint16 (nblk = n*nb). Global block
// index for output row j, block b = j*nb + b, so a warp lane reads its 32 int8 via TWO
// aligned `int4` (128-bit) loads instead of the in-place block's 2-byte reads — the
// coalesced-load lever. BIT-IDENTICAL: same int8 + f16 scale values (byte permutation),
// same 8×__dp4a integer dot as QuantDotGemmQ8_0Kernel. The activation stays the plain
// Q8_0 scratch (small, reused — no repack needed).
template <typename OutT>
__global__ void QuantDotGemmQ8_0AlignedKernel(OutT* __restrict__ out,
                                              const uint8_t* __restrict__ weight,
                                              const BlockQ8_0* __restrict__ act, int64_t m,
                                              int64_t n, int64_t nb) {
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (warp >= m * n) return;
  const int64_t i = warp / n;
  const int64_t j = warp % n;
  const int lane = threadIdx.x;
  const int8_t* qs_base = reinterpret_cast<const int8_t*>(weight);
  const uint16_t* d_base =
      reinterpret_cast<const uint16_t*>(weight + static_cast<size_t>(n) * nb * kQK8_0);
  const BlockQ8_0* a_row = act + i * nb;
  float partial = 0.0f;
  for (int64_t b = lane; b < nb; b += 32) {
    const int64_t gi = j * nb + b;  // global block index in the deinterleaved weight
    const int4* wq = reinterpret_cast<const int4*>(qs_base + static_cast<size_t>(gi) * kQK8_0);
    const int4 w0 = wq[0];  // aligned 128-bit loads (qs at gi*32, 16-byte aligned)
    const int4 w1 = wq[1];
    const int8_t* aq = a_row[b].qs;
    int sumi = 0;
    sumi = __dp4a(w0.x, GetIntB2(aq, 0), sumi);
    sumi = __dp4a(w0.y, GetIntB2(aq, 1), sumi);
    sumi = __dp4a(w0.z, GetIntB2(aq, 2), sumi);
    sumi = __dp4a(w0.w, GetIntB2(aq, 3), sumi);
    sumi = __dp4a(w1.x, GetIntB2(aq, 4), sumi);
    sumi = __dp4a(w1.y, GetIntB2(aq, 5), sumi);
    sumi = __dp4a(w1.z, GetIntB2(aq, 6), sumi);
    sumi = __dp4a(w1.w, GetIntB2(aq, 7), sumi);
    partial += sumi * (DF16ToF32(d_base[gi]) * DF16ToF32(a_row[b].d));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) partial += __shfl_down_sync(0xffffffffu, partial, off);
  if (lane == 0) {
    if constexpr (sizeof(OutT) == 4) out[i * n + j] = partial;
    else out[i * n + j] = DF32ToBF16(partial);
  }
}

// GROUPED Q8_0 variant (weight row = expert_ids[p]*n + j). Same dot core.
template <typename OutT>
__global__ void QuantDotGemmGroupedQ8_0Kernel(OutT* __restrict__ out,
                                              const uint8_t* __restrict__ weight,
                                              const BlockQ8_0* __restrict__ act,
                                              const int32_t* __restrict__ expert_ids, int64_t P,
                                              int64_t n, int64_t nb, size_t w_row_bytes,
                                              bool bcast) {
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (warp >= P * n) return;
  const int64_t p = warp / n;
  const int64_t j = warp % n;
  const int lane = threadIdx.x;
  const int64_t e = expert_ids[p];
  const uint8_t* w_row = weight + static_cast<size_t>(e * n + j) * w_row_bytes;
  const BlockQ8_0* a_row = act + (bcast ? 0 : p) * nb;
  float partial = 0.0f;
  for (int64_t b = lane; b < nb; b += 32) {
    const BlockQ8_0* wb = reinterpret_cast<const BlockQ8_0*>(w_row + static_cast<size_t>(b) *
                                                                          sizeof(BlockQ8_0));
    const BlockQ8_0* ab = a_row + b;
    int sumi = 0;  // Brick 3: 8×__dp4a over GetIntB2 (see QuantDotGemmQ8_0Kernel) — bit-identical
#pragma unroll
    for (int k = 0; k < kQK8_0 / 4; ++k) sumi = __dp4a(GetIntB2(wb->qs, k), GetIntB2(ab->qs, k), sumi);
    partial += sumi * (DF16ToF32(wb->d) * DF16ToF32(ab->d));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) partial += __shfl_down_sync(0xffffffffu, partial, off);
  if (lane == 0) {
    if constexpr (sizeof(OutT) == 4) out[p * n + j] = partial;
    else out[p * n + j] = DF32ToBF16(partial);
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

inline ActDT ActDtOf(DType dt) {
  return dt == DType::kF32 ? ActDT::kF32 : dt == DType::kF16 ? ActDT::kF16 : ActDT::kBF16;
}

// Q8_0 keep-quant GEMM (single). Quantize the m activation rows to Q8_0 on the GPU
// (per-stream grow-only scratch, shared with the Q8_K path — sequential GEMMs), then
// one warp-per-output integer dot. NO CPU fallback, NO stream sync ⇒ capturable.
void MatmulQ8_0Cuda(Tensor& out, const Tensor& a, const Tensor& b, cudaStream_t s) {
  const int64_t m = a.shape[0], k = a.shape[1], n = b.shape[0];
  if (m == 0 || n == 0) return;
  if (k % kQK8_0 != 0)
    throw std::runtime_error("vt cuda: matmul_bt_quant Q8_0: K must be a multiple of 32");
  const int64_t nb = k / kQK8_0;
  const size_t w_row_bytes = static_cast<size_t>(nb) * sizeof(BlockQ8_0);
  const size_t act_bytes = static_cast<size_t>(m) * static_cast<size_t>(nb) * sizeof(BlockQ8_0);
  BlockQ8_0* act = static_cast<BlockQ8_0*>(EnsureScratch(act_bytes, s));
  {
    constexpr int kQBlock = 128;
    const int64_t grid = (m * nb + kQBlock - 1) / kQBlock;
    QuantizeQ8_0Kernel<<<static_cast<unsigned>(grid), kQBlock, 0, s>>>(
        act, a.data, ActDtOf(a.dtype), a.stride[0], m, nb);
  }
  constexpr int kWarpsPerBlock = 4;
  dim3 block(32, kWarpsPerBlock);
  const int64_t grid = (m * n + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const uint8_t* w = static_cast<const uint8_t*>(b.data);
  if (b.q8_0_aligned) {  // Brick 4: coalesced-load layout (RepackQ8_0Cuda) — aligned int4 loads
    if (out.dtype == DType::kF32)
      QuantDotGemmQ8_0AlignedKernel<float><<<static_cast<unsigned>(grid), block, 0, s>>>(
          static_cast<float*>(out.data), w, act, m, n, nb);
    else
      QuantDotGemmQ8_0AlignedKernel<uint16_t><<<static_cast<unsigned>(grid), block, 0, s>>>(
          static_cast<uint16_t*>(out.data), w, act, m, n, nb);
  } else if (out.dtype == DType::kF32)
    QuantDotGemmQ8_0Kernel<float><<<static_cast<unsigned>(grid), block, 0, s>>>(
        static_cast<float*>(out.data), w, act, m, n, nb, w_row_bytes);
  else
    QuantDotGemmQ8_0Kernel<uint16_t><<<static_cast<unsigned>(grid), block, 0, s>>>(
        static_cast<uint16_t*>(out.data), w, act, m, n, nb, w_row_bytes);
  CheckCuda(cudaGetLastError(), "matmul_bt_quant Q8_0 launch");
}

// Q8_0 keep-quant GEMM (grouped: weight row = expert_ids[p]*n + j).
void MatmulQ8_0GroupedCuda(Tensor& out, const Tensor& act_t, const Tensor& weight,
                           const Tensor& expert_ids, cudaStream_t s) {
  const int64_t P = out.shape[0], n = out.shape[1], k = act_t.shape[1];
  if (P == 0 || n == 0) return;
  if (k % kQK8_0 != 0)
    throw std::runtime_error("vt cuda: matmul_bt_quant_grouped Q8_0: K must be a multiple of 32");
  const int64_t nb = k / kQK8_0;
  const int64_t Pa = act_t.shape[0];  // broadcast when 1 row feeds P>1 experts (preq-reuse)
  const bool bcast = (Pa == 1 && P > 1);
  const size_t w_row_bytes = static_cast<size_t>(nb) * sizeof(BlockQ8_0);
  const size_t act_bytes = static_cast<size_t>(Pa) * static_cast<size_t>(nb) * sizeof(BlockQ8_0);
  BlockQ8_0* qact = static_cast<BlockQ8_0*>(EnsureScratch(act_bytes, s));
  {
    constexpr int kQBlock = 128;
    const int64_t grid = (Pa * nb + kQBlock - 1) / kQBlock;
    QuantizeQ8_0Kernel<<<static_cast<unsigned>(grid), kQBlock, 0, s>>>(
        qact, act_t.data, ActDtOf(act_t.dtype), act_t.stride[0], Pa, nb);
  }
  constexpr int kWarpsPerBlock = 4;
  dim3 block(32, kWarpsPerBlock);
  const int64_t grid = (P * n + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const uint8_t* w = static_cast<const uint8_t*>(weight.data);
  const int32_t* eids = static_cast<const int32_t*>(expert_ids.data);
  if (out.dtype == DType::kF32)
    QuantDotGemmGroupedQ8_0Kernel<float><<<static_cast<unsigned>(grid), block, 0, s>>>(
        static_cast<float*>(out.data), w, qact, eids, P, n, nb, w_row_bytes, bcast);
  else
    QuantDotGemmGroupedQ8_0Kernel<uint16_t><<<static_cast<unsigned>(grid), block, 0, s>>>(
        static_cast<uint16_t*>(out.data), w, qact, eids, P, n, nb, w_row_bytes, bcast);
  CheckCuda(cudaGetLastError(), "matmul_bt_quant_grouped Q8_0 launch");
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

  // Q8_0 (32-block, Q8_0-activation) runs its own on-GPU path — the DeepSeek-V4
  // MLA/o-LoRA/shared-expert/lm_head weights. No CPU fallback, no stream sync.
  if (b.dtype == DType::kQ8_0) {
    MatmulQ8_0Cuda(out, a, b, s);
    return;
  }

  WType w{};
  if (!IsCudaKeepQuantSupported(b.dtype, &w)) {
    // Q4_0 (the only remaining Q8_0-activation encoding) — not used by DeepSeek-V4.
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
                       size_t w_row_bytes, size_t w_block_bytes, bool bcast, cudaStream_t s) {
  constexpr int kWarpsPerBlock = 4;
  dim3 block(32, kWarpsPerBlock);
  const int64_t warps = P * n;
  const int64_t grid = (warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
  if (out.dtype == DType::kF32) {
    QuantDotGemmGroupedKernel<W, float><<<static_cast<unsigned>(grid), block, 0, s>>>(
        static_cast<float*>(out.data), weight, act, expert_ids, P, n, nsb, w_row_bytes,
        w_block_bytes, bcast);
  } else {
    QuantDotGemmGroupedKernel<W, uint16_t><<<static_cast<unsigned>(grid), block, 0, s>>>(
        static_cast<uint16_t*>(out.data), weight, act, expert_ids, P, n, nsb, w_row_bytes,
        w_block_bytes, bcast);
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

  if (weight.dtype == DType::kQ8_0) {  // on-GPU Q8_0 grouped path (no CPU sync)
    MatmulQ8_0GroupedCuda(out, act, weight, expert_ids, s);
    return;
  }

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

  // Broadcast activation (preq-reuse): when act has ONE row but P>1 outputs, the
  // routed experts all share the SAME hidden — quantize it ONCE and let every p read
  // Q8_K row 0. Eliminates the topk-fold redundant re-quant (was the bulk of the
  // QuantizeQ8K time) and the caller's xrep copy. Bit-identical (§Brick 2).
  const int64_t Pa = act.shape[0];
  const bool bcast = (Pa == 1 && P > 1);

  // Quantize the Pa activation rows to Q8_K (per-stream grow-only scratch).
  const size_t act_bytes = static_cast<size_t>(Pa) * static_cast<size_t>(nsb) * sizeof(BlockQ8_K);
  BlockQ8_K* qact = static_cast<BlockQ8_K*>(EnsureScratch(act_bytes, s));
  ActDT adt = act.dtype == DType::kF32 ? ActDT::kF32
              : act.dtype == DType::kF16 ? ActDT::kF16
                                         : ActDT::kBF16;
  {
    constexpr int kQBlock = 128;
    const int64_t total_sb = Pa * nsb;
    const int64_t grid = (total_sb + kQBlock - 1) / kQBlock;
    QuantizeQ8KKernel<<<static_cast<unsigned>(grid), kQBlock, 0, s>>>(
        qact, act.data, adt, act.stride[0], Pa, nsb);
  }

  const uint8_t* wt = static_cast<const uint8_t*>(weight.data);
  const int32_t* eids = static_cast<const int32_t*>(expert_ids.data);
  switch (w) {
    case WType::kIQ2_XXS: LaunchGroupedGemm<WType::kIQ2_XXS>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
    case WType::kIQ3_XXS: LaunchGroupedGemm<WType::kIQ3_XXS>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
    case WType::kQ2_K: LaunchGroupedGemm<WType::kQ2_K>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
    case WType::kQ3_K: LaunchGroupedGemm<WType::kQ3_K>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
    case WType::kQ4_K: LaunchGroupedGemm<WType::kQ4_K>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
    case WType::kQ5_K: LaunchGroupedGemm<WType::kQ5_K>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
    case WType::kQ6_K: LaunchGroupedGemm<WType::kQ6_K>(out, wt, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, bcast, s); break;
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

// DeepSeek-V4 resident-decode fused routed-MoE gate+up+SwiGLU (external linkage,
// called from cuda_deepseek_v4.cu's MoeDeviceKernels wrapper — same CUDA library).
// out[P,n] adown, act[Pa,K] (Pa==1 broadcast), gate_w/up_w[E*n,K] block-quant (same
// dtype), expert_ids[P] i32. Quantizes act to Q8_K ONCE (grow-only per-stream
// scratch, identical to MatmulBTQuantGroupedKernelCuda), then one fused launch.
// Bit-identical to the two grouped GEMMs + ClampedSwiGLU it replaces.
void MoeGateUpSwiGLUGroupedCuda(Queue& q, Tensor& out, const Tensor& act, const Tensor& gate_w,
                                const Tensor& up_w, const Tensor& expert_ids, float limit) {
  cudaStream_t s = static_cast<cudaStream_t>(q.handle);
  const int64_t P = out.shape[0];
  const int64_t n = out.shape[1];
  const int64_t k = act.shape[1];
  if (P == 0 || n == 0) return;

  WType w{}, wu{};
  if (!IsCudaKeepQuantSupported(gate_w.dtype, &w) || !IsCudaKeepQuantSupported(up_w.dtype, &wu) ||
      w != wu) {
    throw std::runtime_error(
        "vt cuda: moe_gate_up_swiglu: gate/up must be the SAME CUDA keep-quant dtype");
  }
  if (k % kQK_K != 0) {
    throw std::runtime_error(
        "vt cuda: moe_gate_up_swiglu: K must be a whole number of 256-element Q8_K super-blocks");
  }
  const int64_t nsb = k / kQK_K;
  const size_t w_block_bytes = static_cast<size_t>(BlockBytes(gate_w.dtype));
  const size_t w_row_bytes = static_cast<size_t>(nsb) * w_block_bytes;

  const int64_t Pa = act.shape[0];
  const bool bcast = (Pa == 1 && P > 1);
  const size_t act_bytes = static_cast<size_t>(Pa) * static_cast<size_t>(nsb) * sizeof(BlockQ8_K);
  BlockQ8_K* qact = static_cast<BlockQ8_K*>(EnsureScratch(act_bytes, s));
  ActDT adt = act.dtype == DType::kF32 ? ActDT::kF32
              : act.dtype == DType::kF16 ? ActDT::kF16
                                         : ActDT::kBF16;
  {
    constexpr int kQBlock = 128;
    const int64_t total_sb = Pa * nsb;
    const int64_t grid = (total_sb + kQBlock - 1) / kQBlock;
    QuantizeQ8KKernel<<<static_cast<unsigned>(grid), kQBlock, 0, s>>>(qact, act.data, adt,
                                                                      act.stride[0], Pa, nsb);
  }

  const uint8_t* gw = static_cast<const uint8_t*>(gate_w.data);
  const uint8_t* uw = static_cast<const uint8_t*>(up_w.data);
  const int32_t* eids = static_cast<const int32_t*>(expert_ids.data);
  switch (w) {
    case WType::kIQ2_XXS: LaunchGroupedFusedSwiGLU<WType::kIQ2_XXS>(out, gw, uw, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, limit, bcast, s); break;
    case WType::kIQ3_XXS: LaunchGroupedFusedSwiGLU<WType::kIQ3_XXS>(out, gw, uw, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, limit, bcast, s); break;
    case WType::kQ2_K: LaunchGroupedFusedSwiGLU<WType::kQ2_K>(out, gw, uw, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, limit, bcast, s); break;
    case WType::kQ3_K: LaunchGroupedFusedSwiGLU<WType::kQ3_K>(out, gw, uw, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, limit, bcast, s); break;
    case WType::kQ4_K: LaunchGroupedFusedSwiGLU<WType::kQ4_K>(out, gw, uw, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, limit, bcast, s); break;
    case WType::kQ5_K: LaunchGroupedFusedSwiGLU<WType::kQ5_K>(out, gw, uw, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, limit, bcast, s); break;
    case WType::kQ6_K: LaunchGroupedFusedSwiGLU<WType::kQ6_K>(out, gw, uw, qact, eids, P, n, nsb, w_row_bytes, w_block_bytes, limit, bcast, s); break;
  }
  CheckCuda(cudaGetLastError(), "moe_gate_up_swiglu launch");
}

}  // namespace vt::cuda
