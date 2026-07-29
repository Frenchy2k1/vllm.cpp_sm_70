// Block-quant `to_float` decoders — the `dequantize_row_*` kernels ported
// byte-for-byte from llama.cpp @ 237ad9b96:
//   ggml/src/ggml-common.h  (block_q2_K/q4_0/q8_0/q3_K/q4_K/q5_K/q6_K +
//                            block_iq2_xxs layouts; iq2xxs_grid/ksigns tables)
//   ggml/src/ggml-quants.c  (dequantize_row_q4_0:401, _q8_0:495, _q2_K:903,
//                            _q3_K:1247, _q4_K:1471, _q5_K:1673, _q6_K:1881,
//                            _iq2_xxs:2416, get_scale_min_k4:822)
//
// These moved here VERBATIM from the GGUF loader
// (src/vllm/model_executor/model_loader/gguf_dequant.cpp), which now delegates:
// vt:: is the lower layer and the compute-in-quant GEMM's generic fallback
// needs the same decode, so a single implementation serves both the loader
// oracle and `vt::MatmulBTQuant`. Numerics are unchanged by construction (same
// code, same order, same `-ffp-contract=off` pinning) and
// tests/vllm/test_gguf_dequant.cpp gates that.
#include <cstring>

#include "vt/quant.h"
#include "vt/dtype.h"

namespace vt::cpu {
namespace {

// Read a little-endian ggml_half (f16) at byte pointer `p` and widen to f32.
// (Aligned load is not guaranteed for mmap'd block bytes, so memcpy.)
float ReadF16(const uint8_t* p) {
  uint16_t h = 0;
  std::memcpy(&h, p, sizeof(h));
  return vt::F16ToF32(h);
}

// get_scale_min_k4 (ggml-quants.c:822): unpack the j-th 6-bit scale `d` and
// 6-bit min `m` from a Q4_K/Q5_K block's packed scales[12]. j in 0..7.
void GetScaleMinK4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = static_cast<uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
    *m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
  }
}

// --- Per-type dequant (one full row = nb blocks). Each mirrors the matching
// dequantize_row_* in ggml-quants.c; byte offsets follow the ggml-common.h
// struct layouts. `y` is written in order (numel outputs). ---

// block_q4_0 = { f16 d; u8 qs[16]; }  (18 bytes)   dequantize_row_q4_0:401
void DequantQ4_0(const uint8_t* data, int64_t nb, float* y) {
  constexpr int qk = 32;
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* blk = data + i * 18;
    const float d = ReadF16(blk);
    const uint8_t* qs = blk + 2;
    for (int j = 0; j < qk / 2; ++j) {
      const int x0 = (qs[j] & 0x0F) - 8;
      const int x1 = (qs[j] >> 4) - 8;
      y[i * qk + j + 0] = x0 * d;
      y[i * qk + j + qk / 2] = x1 * d;
    }
  }
}

// block_q8_0 = { f16 d; i8 qs[32]; }  (34 bytes)   dequantize_row_q8_0:495
void DequantQ8_0(const uint8_t* data, int64_t nb, float* y) {
  constexpr int qk = 32;
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* blk = data + i * 34;
    const float d = ReadF16(blk);
    const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
    for (int j = 0; j < qk; ++j) {
      y[i * qk + j] = qs[j] * d;
    }
  }
}

// block_q3_K = { u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d; } (110 bytes)
// dequantize_row_q3_K:1247. The 3-bit quant = 2 low bits (qs) + 1 high bit
// (hmask, inverted: absent bit -> -4) times the 6-bit scale (-32 biased).
void DequantQ3_K(const uint8_t* data, int64_t nb, float* y) {
  constexpr int qk = 256;
  const uint32_t kmask1 = 0x03030303;
  const uint32_t kmask2 = 0x0f0f0f0f;
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* blk = data + i * 110;
    const uint8_t* hm = blk;         // hmask[32]
    const uint8_t* q = blk + 32;     // qs[64]
    const uint8_t* sc_raw = blk + 96;  // scales[12]
    const float d_all = ReadF16(blk + 108);

    // Scale unpack: 12 packed bytes -> 16 6-bit scales in int8 view of aux.
    uint32_t aux[4];
    std::memcpy(aux, sc_raw, 12);
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t* scales = reinterpret_cast<const int8_t*>(aux);

    int is = 0;
    uint8_t m = 1;
    for (int n = 0; n < qk; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        float dl = d_all * (scales[is++] - 32);
        for (int l = 0; l < 16; ++l) {
          *y++ = dl * (static_cast<int8_t>((q[l + 0] >> shift) & 3) -
                       ((hm[l + 0] & m) ? 0 : 4));
        }
        dl = d_all * (scales[is++] - 32);
        for (int l = 0; l < 16; ++l) {
          *y++ = dl * (static_cast<int8_t>((q[l + 16] >> shift) & 3) -
                       ((hm[l + 16] & m) ? 0 : 4));
        }
        shift += 2;
        m = static_cast<uint8_t>(m << 1);
      }
      q += 32;
    }
  }
}

// block_q4_K = { f16 d; f16 dmin; u8 scales[12]; u8 qs[128]; } (144 bytes)
// dequantize_row_q4_K:1471. y = d*sc*(nibble) - dmin*m over 8 sub-blocks of 32.
void DequantQ4_K(const uint8_t* data, int64_t nb, float* y) {
  constexpr int qk = 256;
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* blk = data + i * 144;
    const float d = ReadF16(blk);
    const float min = ReadF16(blk + 2);
    const uint8_t* scales = blk + 4;
    const uint8_t* q = blk + 16;  // qs[128]

    int is = 0;
    uint8_t sc = 0;
    uint8_t mm = 0;
    for (int j = 0; j < qk; j += 64) {
      GetScaleMinK4(is + 0, scales, &sc, &mm);
      const float d1 = d * sc;
      const float m1 = min * mm;
      GetScaleMinK4(is + 1, scales, &sc, &mm);
      const float d2 = d * sc;
      const float m2 = min * mm;
      for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
      for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
      q += 32;
      is += 2;
    }
  }
}

// block_q5_K = { f16 d; f16 dmin; u8 scales[12]; u8 qh[32]; u8 qs[128]; }
// (176 bytes) dequantize_row_q5_K:1673. Like Q4_K plus the 5th (high) bit from
// qh: bit u1 for the low nibbles, u2 for the high nibbles (both <<=2 per pair).
void DequantQ5_K(const uint8_t* data, int64_t nb, float* y) {
  constexpr int qk = 256;
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* blk = data + i * 176;
    const float d = ReadF16(blk);
    const float min = ReadF16(blk + 2);
    const uint8_t* scales = blk + 4;
    const uint8_t* qh = blk + 16;   // qh[32]
    const uint8_t* ql = blk + 48;   // qs[128]

    int is = 0;
    uint8_t sc = 0;
    uint8_t mm = 0;
    uint8_t u1 = 1;
    uint8_t u2 = 2;
    for (int j = 0; j < qk; j += 64) {
      GetScaleMinK4(is + 0, scales, &sc, &mm);
      const float d1 = d * sc;
      const float m1 = min * mm;
      GetScaleMinK4(is + 1, scales, &sc, &mm);
      const float d2 = d * sc;
      const float m2 = min * mm;
      for (int l = 0; l < 32; ++l)
        *y++ = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
      for (int l = 0; l < 32; ++l)
        *y++ = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
      ql += 32;
      is += 2;
      u1 = static_cast<uint8_t>(u1 << 2);
      u2 = static_cast<uint8_t>(u2 << 2);
    }
  }
}

// block_q6_K = { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d; } (210 bytes)
// dequantize_row_q6_K:1881. 6-bit quant = 4 low bits (ql) + 2 high bits (qh),
// -32 biased, times an 8-bit (int8) scale. 16 blocks of 16.
void DequantQ6_K(const uint8_t* data, int64_t nb, float* y) {
  constexpr int qk = 256;
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* blk = data + i * 210;
    const uint8_t* ql = blk;         // ql[128]
    const uint8_t* qh = blk + 128;   // qh[64]
    const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192);  // scales[16]
    const float d = ReadF16(blk + 208);

    for (int n = 0; n < qk; n += 128) {
      for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        const int8_t q1 = static_cast<int8_t>(
            (ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const int8_t q2 = static_cast<int8_t>(
            (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const int8_t q3 = static_cast<int8_t>(
            (ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const int8_t q4 = static_cast<int8_t>(
            (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        y[l + 0] = d * sc[is + 0] * q1;
        y[l + 32] = d * sc[is + 2] * q2;
        y[l + 64] = d * sc[is + 4] * q3;
        y[l + 96] = d * sc[is + 6] * q4;
      }
      y += 128;
      ql += 64;
      qh += 32;
      sc += 8;
    }
  }
}

// block_q8_K = { f32 d; i8 qs[256]; i16 bsums[16]; } (292 bytes)
// dequantize_row_q8_K (ggml-quants.c). Q8_K is the K-quant ACTIVATION type; it
// never appears in a GGUF file, but the decoder completes the table and lets
// the activation-quant round trip be unit-tested in G2.
void DequantQ8_K(const uint8_t* data, int64_t nb, float* y) {
  constexpr int qk = 256;
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* blk = data + i * 292;
    float d;
    std::memcpy(&d, blk, sizeof(d));
    const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 4);
    for (int j = 0; j < qk; ++j) *y++ = d * qs[j];
  }
}

// --- IQ2_XXS codebook + sign tables, ported byte-for-byte from llama.cpp @
// 237ad9b96 ggml/src/ggml-common.h: kmask_iq2xs:499, ksigns_iq2xs:503,
// iq2xxs_grid:550. The 256-entry grid is the 2-bit codebook: each u64 packs 8
// grid bytes (little-endian) that a 3-bit-ish index selects; ksigns_iq2xs maps a
// 7-bit selector to the per-lane sign byte, masked by kmask_iq2xs. ---
constexpr uint8_t kKmaskIq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

constexpr uint8_t kKsignsIq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255
};

constexpr uint64_t kIq2xxsGrid[256] = {
    0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL,
    0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x08080808082b0808ULL,
    0x08080808082b082bULL, 0x08080808082b2b08ULL, 0x08080808082b2b2bULL, 0x0808080819080819ULL,
    0x0808080819081908ULL, 0x0808080819190808ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL,
    0x08080808192b1908ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b082b2bULL,
    0x080808082b2b082bULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x0808081908190808ULL,
    0x0808081908191919ULL, 0x0808081919080808ULL, 0x080808192b081908ULL, 0x080808192b192b08ULL,
    0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b082b082bULL, 0x0808082b2b08082bULL,
    0x0808190808080819ULL, 0x0808190808081908ULL, 0x0808190808190808ULL, 0x08081908082b0819ULL,
    0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819082b08ULL,
    0x08081908192b0808ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL,
    0x080819082b2b1908ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908082b08ULL,
    0x08081919082b0808ULL, 0x080819191908192bULL, 0x08081919192b2b19ULL, 0x080819192b080808ULL,
    0x080819192b190819ULL, 0x0808192b08082b19ULL, 0x0808192b08190808ULL, 0x0808192b19080808ULL,
    0x0808192b2b081908ULL, 0x0808192b2b2b1908ULL, 0x08082b0808080808ULL, 0x08082b0808081919ULL,
    0x08082b0808082b08ULL, 0x08082b0808191908ULL, 0x08082b08082b2b08ULL, 0x08082b0819080819ULL,
    0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b081919082bULL, 0x08082b082b082b08ULL,
    0x08082b1908081908ULL, 0x08082b1919080808ULL, 0x08082b2b0808082bULL, 0x08082b2b08191908ULL,
    0x0819080808080819ULL, 0x0819080808081908ULL, 0x0819080808190808ULL, 0x08190808082b0819ULL,
    0x0819080819080808ULL, 0x08190808192b0808ULL, 0x081908082b081908ULL, 0x081908082b190808ULL,
    0x081908082b191919ULL, 0x0819081908080808ULL, 0x0819081908082b08ULL, 0x08190819082b0808ULL,
    0x0819081919190808ULL, 0x0819081919192b2bULL, 0x081908192b080808ULL, 0x0819082b082b1908ULL,
    0x0819082b19081919ULL, 0x0819190808080808ULL, 0x0819190808082b08ULL, 0x08191908082b0808ULL,
    0x08191908082b1919ULL, 0x0819190819082b19ULL, 0x081919082b080808ULL, 0x0819191908192b08ULL,
    0x08191919192b082bULL, 0x0819192b08080808ULL, 0x0819192b0819192bULL, 0x08192b0808080819ULL,
    0x08192b0808081908ULL, 0x08192b0808190808ULL, 0x08192b0819080808ULL, 0x08192b082b080819ULL,
    0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b192b2b0808ULL, 0x08192b2b19190819ULL,
    0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808082b2bULL, 0x082b080819081908ULL,
    0x082b0808192b0819ULL, 0x082b08082b080808ULL, 0x082b08082b08082bULL, 0x082b0819082b2b19ULL,
    0x082b081919082b08ULL, 0x082b082b08080808ULL, 0x082b082b0808082bULL, 0x082b190808080819ULL,
    0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b190819080808ULL, 0x082b19081919192bULL,
    0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b1919192b1908ULL, 0x082b192b2b190808ULL,
    0x082b2b0808082b08ULL, 0x082b2b08082b0808ULL, 0x082b2b082b191908ULL, 0x082b2b2b19081908ULL,
    0x1908080808080819ULL, 0x1908080808081908ULL, 0x1908080808190808ULL, 0x1908080808192b08ULL,
    0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x1908080819080808ULL, 0x1908080819082b08ULL,
    0x190808081919192bULL, 0x19080808192b0808ULL, 0x190808082b080819ULL, 0x190808082b081908ULL,
    0x190808082b190808ULL, 0x1908081908080808ULL, 0x19080819082b0808ULL, 0x19080819192b0819ULL,
    0x190808192b080808ULL, 0x190808192b081919ULL, 0x1908082b08080819ULL, 0x1908082b08190808ULL,
    0x1908082b19082b08ULL, 0x1908082b1919192bULL, 0x1908082b192b2b08ULL, 0x1908190808080808ULL,
    0x1908190808082b08ULL, 0x19081908082b0808ULL, 0x190819082b080808ULL, 0x190819082b192b19ULL,
    0x190819190819082bULL, 0x19081919082b1908ULL, 0x1908192b08080808ULL, 0x19082b0808080819ULL,
    0x19082b0808081908ULL, 0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL,
    0x19082b1908080808ULL, 0x19082b1919192b08ULL, 0x19082b19192b0819ULL, 0x19082b192b08082bULL,
    0x19082b2b19081919ULL, 0x19082b2b2b190808ULL, 0x1919080808080808ULL, 0x1919080808082b08ULL,
    0x1919080808190819ULL, 0x1919080808192b19ULL, 0x19190808082b0808ULL, 0x191908082b080808ULL,
    0x191908082b082b08ULL, 0x1919081908081908ULL, 0x191908191908082bULL, 0x191908192b2b1908ULL,
    0x1919082b2b190819ULL, 0x191919082b190808ULL, 0x191919082b19082bULL, 0x1919191908082b2bULL,
    0x1919192b08080819ULL, 0x1919192b19191908ULL, 0x19192b0808080808ULL, 0x19192b0808190819ULL,
    0x19192b0808192b19ULL, 0x19192b08192b1908ULL, 0x19192b1919080808ULL, 0x19192b2b08082b08ULL,
    0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL, 0x192b0808192b2b08ULL,
    0x192b081908080808ULL, 0x192b081919191919ULL, 0x192b082b08192b08ULL, 0x192b082b192b0808ULL,
    0x192b190808080808ULL, 0x192b190808081919ULL, 0x192b191908190808ULL, 0x192b19190819082bULL,
    0x192b19192b081908ULL, 0x192b2b081908082bULL, 0x2b08080808080808ULL, 0x2b0808080808082bULL,
    0x2b08080808082b2bULL, 0x2b08080819080819ULL, 0x2b0808082b08082bULL, 0x2b08081908081908ULL,
    0x2b08081908192b08ULL, 0x2b08081919080808ULL, 0x2b08082b08190819ULL, 0x2b08190808080819ULL,
    0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b08190808191919ULL, 0x2b08190819080808ULL,
    0x2b081908192b0808ULL, 0x2b08191908080808ULL, 0x2b0819191908192bULL, 0x2b0819192b191908ULL,
    0x2b08192b08082b19ULL, 0x2b08192b19080808ULL, 0x2b08192b192b0808ULL, 0x2b082b080808082bULL,
    0x2b082b1908081908ULL, 0x2b082b2b08190819ULL, 0x2b19080808081908ULL, 0x2b19080808190808ULL,
    0x2b190808082b1908ULL, 0x2b19080819080808ULL, 0x2b1908082b2b0819ULL, 0x2b1908190819192bULL,
    0x2b1908192b080808ULL, 0x2b19082b19081919ULL, 0x2b19190808080808ULL, 0x2b191908082b082bULL,
    0x2b19190819081908ULL, 0x2b19191919190819ULL, 0x2b192b082b080819ULL, 0x2b192b19082b0808ULL,
    0x2b2b08080808082bULL, 0x2b2b080819190808ULL, 0x2b2b08082b081919ULL, 0x2b2b081908082b19ULL,
    0x2b2b082b08080808ULL, 0x2b2b190808192b08ULL, 0x2b2b2b0819190808ULL, 0x2b2b2b1908081908ULL,
};

// block_q2_K = { u8 scales[16]; u8 qs[64]; f16 d; f16 dmin; } (84 bytes)
// dequantize_row_q2_K:903. 2-bit quant (qs, shift 0/2/4/6) times a 4-bit
// per-16 sub-scale (low nibble of scales[]) minus a 4-bit sub-min (high
// nibble), both scaled by the f16 super-block d / dmin.
void DequantQ2_K(const uint8_t* data, int64_t nb, float* y) {
  constexpr int qk = 256;
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* blk = data + i * 84;
    const uint8_t* scales = blk;       // scales[16]
    const uint8_t* q = blk + 16;       // qs[64]
    const float d = ReadF16(blk + 80);
    const float min = ReadF16(blk + 82);

    int is = 0;
    for (int n = 0; n < qk; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        uint8_t sc = scales[is++];
        float dl = d * (sc & 0xF);
        float ml = min * (sc >> 4);
        for (int l = 0; l < 16; ++l)
          *y++ = dl * static_cast<int8_t>((q[l] >> shift) & 3) - ml;
        sc = scales[is++];
        dl = d * (sc & 0xF);
        ml = min * (sc >> 4);
        for (int l = 0; l < 16; ++l)
          *y++ = dl * static_cast<int8_t>((q[l + 16] >> shift) & 3) - ml;
        shift += 2;
      }
      q += 32;
    }
  }
}

// block_iq2_xxs = { f16 d; u16 qs[32]; } (66 bytes) dequantize_row_iq2_xxs:2416.
// Codebook decode: each 32-element sub-block reads two u32 from qs -- aux32[0]
// holds four 8-bit grid indices, aux32[1] holds four 7-bit sign selectors plus a
// 4-bit block scale in its top nibble (db = d*(0.5 + (aux32[1]>>28))*0.25). The
// eight grid bytes are looked up in kIq2xxsGrid and sign-flipped per
// kKsignsIq2xs & kKmaskIq2xs.
void DequantIQ2_XXS(const uint8_t* data, int64_t nb, float* y) {
  constexpr int qk = 256;
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* blk = data + i * 66;
    const float d = ReadF16(blk);
    const uint8_t* qs = blk + 2;       // u16 qs[32], little-endian bytes
    for (int ib32 = 0; ib32 < qk / 32; ++ib32) {
      uint32_t aux32[2];
      std::memcpy(aux32, qs + 8 * ib32, 2 * sizeof(uint32_t));
      const uint8_t* aux8 = reinterpret_cast<const uint8_t*>(aux32);
      const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
      for (int l = 0; l < 4; ++l) {
        const uint8_t* grid =
            reinterpret_cast<const uint8_t*>(kIq2xxsGrid + aux8[l]);
        const uint8_t signs = kKsignsIq2xs[(aux32[1] >> (7 * l)) & 127];
        for (int j = 0; j < 8; ++j)
          y[j] = db * grid[j] * ((signs & kKmaskIq2xs[j]) ? -1.f : 1.f);
        y += 8;
      }
    }
  }
}

// Adapt a whole-row `(data, nb, y)` decoder to upstream's
// `ggml_to_float_t(x, y, k)` shape.
template <void (*Kernel)(const uint8_t*, int64_t, float*), int64_t kBlockElems>
void ToFloatAdapter(const void* x, float* y, int64_t k) {
  VT_CHECK(k % kBlockElems == 0,
           "block to_float: element count is not a whole number of blocks");
  Kernel(static_cast<const uint8_t*>(x), k / kBlockElems, y);
}

}  // namespace

ToFloatFn BlockToFloat(DType dtype) {
  switch (dtype) {
    case DType::kQ4_0: return &ToFloatAdapter<&DequantQ4_0, 32>;
    case DType::kQ8_0: return &ToFloatAdapter<&DequantQ8_0, 32>;
    case DType::kQ2_K: return &ToFloatAdapter<&DequantQ2_K, 256>;
    case DType::kQ3_K: return &ToFloatAdapter<&DequantQ3_K, 256>;
    case DType::kQ4_K: return &ToFloatAdapter<&DequantQ4_K, 256>;
    case DType::kQ5_K: return &ToFloatAdapter<&DequantQ5_K, 256>;
    case DType::kQ6_K: return &ToFloatAdapter<&DequantQ6_K, 256>;
    case DType::kQ8_K: return &ToFloatAdapter<&DequantQ8_K, 256>;
    case DType::kIQ2_XXS: return &ToFloatAdapter<&DequantIQ2_XXS, 256>;
    default: return nullptr;
  }
}

}  // namespace vt::cpu
