// Unit gate for the compressed-tensors MXFP4 (`mxfp4-pack-quantized`) weight
// dequant (mxfp4_dequant.{h,cpp}).
//
// Golden: tests/quantization/reference_mxfp4.py:28-117 `dq_mxfp4_torch`
//   (e8m0_to_half = 2^(byte-127); upcast_fp4_to_fp16_or_bf16; per-32 multiply)
//   @ pin 555967922. Re-expressed here as `RefDqMxfp4` in double precision and
//   as literal hand cases. RED-first: the E8M0-vs-fp8 and group-32-vs-16 traps
//   are pinned so a wrong-scale port provably fails.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "vllm/model_executor/model_loader/mxfp4_dequant.h"
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"  // F8E4M3ToF32, kE2M1Lut
#include "vt/dtype.h"

using vllm::DequantMxfp4ToBf16;
using vllm::DequantMxfp4ToF32;
using vllm::E8M0ToF32;
using vllm::F8E4M3ToF32;
using vllm::kE2M1Lut;
using vllm::kMxfp4GroupSize;

namespace {

// Independent double-precision port of dq_mxfp4_torch. group_size 32, no global.
void RefDqMxfp4(const std::vector<uint8_t>& packed,
                const std::vector<uint8_t>& scale, int64_t out_dim,
                int64_t in_dim, std::vector<double>* out) {
  const int64_t packed_cols = in_dim / 2;
  const int64_t groups = in_dim / kMxfp4GroupSize;
  out->assign(static_cast<size_t>(out_dim * in_dim), 0.0);
  for (int64_t o = 0; o < out_dim; ++o) {
    for (int64_t g = 0; g < groups; ++g) {
      // e8m0_to_half: 2 ** (byte - 127).
      const double s = std::pow(2.0, static_cast<int>(scale[o * groups + g]) - 127);
      for (int64_t j = 0; j < kMxfp4GroupSize / 2; ++j) {
        const uint8_t b = packed[o * packed_cols + g * (kMxfp4GroupSize / 2) + j];
        const uint8_t lo = b & 0x0FU;
        const uint8_t hi = b >> 4;
        const double lov = kE2M1Lut[lo & 0x7U] * ((lo & 0x8U) ? -1.0 : 1.0);
        const double hiv = kE2M1Lut[hi & 0x7U] * ((hi & 0x8U) ? -1.0 : 1.0);
        const int64_t base = o * in_dim + g * kMxfp4GroupSize + 2 * j;
        (*out)[base] = lov * s;
        (*out)[base + 1] = hiv * s;
      }
    }
  }
}

}  // namespace

// --- E8M0 scale decode: byte = biased exponent, bias 127, value = 2^(byte-127).
// reference_mxfp4.py:31-34. This is the whole point vs NVFP4's fp8-e4m3 scale. ---
TEST_CASE("E8M0ToF32 known bytes") {
  CHECK(E8M0ToF32(0x7F) == doctest::Approx(1.0F));     // 2^0
  CHECK(E8M0ToF32(0x80) == doctest::Approx(2.0F));     // 2^1
  CHECK(E8M0ToF32(0x81) == doctest::Approx(4.0F));     // 2^2
  CHECK(E8M0ToF32(0x7E) == doctest::Approx(0.5F));     // 2^-1
  CHECK(E8M0ToF32(0x00) == doctest::Approx(std::ldexp(1.0F, -127)));  // 2^-127
  CHECK(E8M0ToF32(0x87) == doctest::Approx(256.0F));   // 2^8
  CHECK(std::isnan(E8M0ToF32(0xFF)));                  // OCP E8M0 NaN

  // RED trap: E8M0 and fp8-e4m3 (the NVFP4 scale codec) DISAGREE on these bytes.
  // A port that reused F8E4M3ToF32 (byte 0x80 -> -0.0, 0x7F -> NaN) would break.
  CHECK(E8M0ToF32(0x80) != doctest::Approx(F8E4M3ToF32(0x80)));  // 2.0 vs -0.0
  CHECK(std::isnan(F8E4M3ToF32(0x7F)));               // fp8 NaN where E8M0 = 1.0
}

// --- Hand-computed block: one row, one 32-element group (16 packed bytes, one
// E8M0 scale byte). scale 0x80 -> 2^1 = 2.0. out[i] = e2m1(nibble_i) * 2.0. ---
TEST_CASE("DequantMxfp4 hand-computed group of 32") {
  std::vector<uint8_t> packed(16, 0x00);
  // elem 0 = +0.5 (0x1), elem 1 = +6.0 (0x7)  -> byte0 = 0x71
  packed[0] = 0x71;
  // elem 2 = -6.0 (0xF), elem 3 = +0.0 (0x0)  -> byte1 = 0x0F
  packed[1] = 0x0F;
  // elem 4 = +1.0 (0x2), elem 5 = -1.0 (0xA)  -> byte2 = 0xA2
  packed[2] = 0xA2;
  // elem 6 = +2.0 (0x4), elem 7 = -3.0 (0xD)  -> byte3 = 0xD4
  packed[3] = 0xD4;
  std::vector<uint8_t> scale = {0x80};  // 2.0

  std::vector<uint16_t> out_bf16(32, 0xFFFF);
  DequantMxfp4ToBf16(packed.data(), scale.data(), /*out_dim=*/1,
                     /*in_dim=*/32, out_bf16.data());

  const float expected[8] = {1.0F, 12.0F, -12.0F, 0.0F, 2.0F, -2.0F, 4.0F, -6.0F};
  for (int i = 0; i < 8; ++i)
    CHECK(vt::BF16ToF32(out_bf16[i]) == doctest::Approx(expected[i]));
  for (int i = 8; i < 32; ++i)
    CHECK(vt::BF16ToF32(out_bf16[i]) == doctest::Approx(0.0F));

  // f32 emitter agrees, and (power-of-two scale) bf16 is exact here.
  std::vector<float> out_f32(32, -1.0F);
  DequantMxfp4ToF32(packed.data(), scale.data(), 1, 32, out_f32.data());
  for (int i = 0; i < 8; ++i)
    CHECK(out_f32[i] == doctest::Approx(expected[i]));
}

// --- RED-first wrong-scale trap: a fractional-exponent E8M0 byte forces the
// group scale to a value fp8-e4m3 decoding would NOT produce, so a scale-codec
// bug is caught. scale 0x7E -> 2^-1 = 0.5; the same byte via fp8-e4m3 = 448/... .
TEST_CASE("DequantMxfp4 E8M0 scale is 2^(b-127), not fp8") {
  std::vector<uint8_t> packed(16, 0x00);
  packed[0] = 0x06;  // elem0 = +4.0 (nibble 0x6)
  std::vector<uint8_t> scale = {0x7E};  // E8M0 -> 0.5

  std::vector<float> out(32, -1.0F);
  DequantMxfp4ToF32(packed.data(), scale.data(), 1, 32, out.data());

  CHECK(out[0] == doctest::Approx(2.0F));                        // 4.0 * 0.5
  // The wrong (fp8-e4m3) reading of 0x7E is 448 -> would give 1792; assert not.
  CHECK(out[0] != doctest::Approx(4.0F * F8E4M3ToF32(0x7E)));
}

// --- Multi-row / multi-group: exercises row + group offset arithmetic with a
// distinct scale per group, so a stride or group-size (16 vs 32) bug surfaces. ---
TEST_CASE("DequantMxfp4 two rows two groups") {
  const int64_t in_dim = 64;              // 2 groups of 32
  const int64_t packed_cols = in_dim / 2;  // 32
  const int64_t groups = in_dim / kMxfp4GroupSize;  // 2

  std::vector<uint8_t> packed(2 * packed_cols, 0x00);
  packed[0] = 0x04;                       // row0 g0 elem0 = +2.0
  packed[16] = 0x02;                      // row0 g1 elem0 (elem 32) = +1.0
  packed[packed_cols + 0] = 0x06;         // row1 g0 elem0 = +4.0

  std::vector<uint8_t> scale(2 * groups, 0x00);
  scale[0] = 0x7F;  // row0 g0 = 1.0
  scale[1] = 0x80;  // row0 g1 = 2.0
  scale[2] = 0x81;  // row1 g0 = 4.0
  scale[3] = 0x7F;  // row1 g1 = 1.0

  std::vector<float> out(2 * in_dim, -1.0F);
  DequantMxfp4ToF32(packed.data(), scale.data(), 2, in_dim, out.data());

  CHECK(out[0] == doctest::Approx(2.0F));         // 2.0 * 1.0
  CHECK(out[32] == doctest::Approx(2.0F));        // 1.0 * 2.0
  CHECK(out[in_dim + 0] == doctest::Approx(16.0F));  // 4.0 * 4.0

  // Cross-check the whole tensor vs the double-precision golden.
  std::vector<double> ref;
  RefDqMxfp4(packed, scale, 2, in_dim, &ref);
  for (size_t i = 0; i < out.size(); ++i)
    CHECK(out[i] == doctest::Approx(static_cast<float>(ref[i])));
}

// --- Randomized vs the double-precision golden (both emitters). E8M0 scales are
// kept in a finite range so no overflow; bf16 must equal f32 (power-of-two mul).
TEST_CASE("DequantMxfp4 randomized vs double-precision reference") {
  std::mt19937 rng(20260728u);
  std::uniform_int_distribution<int> byte(0, 255);
  // exponents in [-8, 8] around bias 127 keep products well inside bf16 range.
  std::uniform_int_distribution<int> e8m0(127 - 8, 127 + 8);

  const int64_t out_dim = 5;
  const int64_t in_dim = 96;  // 3 groups of 32
  const int64_t groups = in_dim / kMxfp4GroupSize;
  std::vector<uint8_t> packed(out_dim * in_dim / 2);
  std::vector<uint8_t> scale(out_dim * groups);
  for (auto& p : packed) p = static_cast<uint8_t>(byte(rng));
  for (auto& s : scale) s = static_cast<uint8_t>(e8m0(rng));

  std::vector<double> ref;
  RefDqMxfp4(packed, scale, out_dim, in_dim, &ref);

  std::vector<float> f32(out_dim * in_dim, 0.0F);
  DequantMxfp4ToF32(packed.data(), scale.data(), out_dim, in_dim, f32.data());
  std::vector<uint16_t> bf16(out_dim * in_dim, 0);
  DequantMxfp4ToBf16(packed.data(), scale.data(), out_dim, in_dim, bf16.data());

  for (size_t i = 0; i < f32.size(); ++i) {
    CHECK(f32[i] == doctest::Approx(static_cast<float>(ref[i])));
    // Power-of-two scale => bf16 store is exact for these in-range values.
    CHECK(vt::BF16ToF32(bf16[i]) == f32[i]);
  }
}
