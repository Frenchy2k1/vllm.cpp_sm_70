// Ports the executable spec of vLLM's AWQ/GPTQ INT4 dequant:
//   tests/kernels/quantization/test_awq_triton.py::awq_dequantize_torch @ 555967922
//     (the reference (iweights - zeros) * scales with reverse-AWQ order)
//   csrc/libtorch_stable/quantization/gptq/qdq_4.cuh dequant_4bit_8_gptq +
//     q_gemm.cu:201-202 zero_offset (GPTQv1 vs v2) @ 555967922
//
// Two-layer gate (mirrors test_nvfp4_dequant.cpp): (1) hand-computed known
// packed int32s with exact-integer expected bf16 give an INDEPENDENT arithmetic
// oracle and pin the bit order / zero convention; (2) a randomized roundtrip
// packs nibbles through an INDEPENDENT reference packer (mirroring the vLLM
// layout) and checks the production unpack+dequant against a DOUBLE-precision
// reference, so any packing-axis / group-index / order bug surfaces.
#include <doctest/doctest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "vllm/model_executor/model_loader/awq_gptq_dequant.h"
#include "vt/dtype.h"

using vllm::DequantAwq4ToBf16;
using vllm::DequantGptq4ToBf16;
using vllm::kAwqReverseOrder;

namespace {

// --- Independent reference packers (mirror the vLLM on-disk layout) ---

// AWQ weight/zero packer: [rows][N] nibbles -> int32 [rows][N/8], packed along
// N with reverse-AWQ order (auto_awq.py:77). Output element `col` lands at
// nibble kAwqReverseOrder[col%8] of int32 col/8.
std::vector<int32_t> PackAwqAlongN(const std::vector<std::vector<int>>& nib,
                                   int64_t rows, int64_t n) {
  std::vector<int32_t> out(rows * (n / 8), 0);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < n; ++c) {
      const int64_t pc = c / 8;
      const int shift = kAwqReverseOrder[c % 8] * 4;
      out[r * (n / 8) + pc] |=
          static_cast<int32_t>((nib[r][c] & 0xF) << shift);
    }
  }
  return out;
}

// GPTQ weight packer: [K][N] nibbles -> int32 [K/8][N], packed along K with
// standard order (quant_utils.gptq_pack == pack_rows). Row k lands at nibble
// k%8 of int32 row k/8.
std::vector<int32_t> PackGptqWeightAlongK(const std::vector<std::vector<int>>& nib,
                                          int64_t k, int64_t n) {
  std::vector<int32_t> out((k / 8) * n, 0);
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t c = 0; c < n; ++c) {
      const int shift = static_cast<int>(row % 8) * 4;
      out[(row / 8) * n + c] |=
          static_cast<int32_t>((nib[row][c] & 0xF) << shift);
    }
  }
  return out;
}

// GPTQ zero packer: [G][N] nibbles -> int32 [G][N/8], packed along N with
// STANDARD order. Column n lands at nibble n%8 of int32 n/8.
std::vector<int32_t> PackGptqZerosAlongN(const std::vector<std::vector<int>>& nib,
                                         int64_t groups, int64_t n) {
  std::vector<int32_t> out(groups * (n / 8), 0);
  for (int64_t g = 0; g < groups; ++g) {
    for (int64_t c = 0; c < n; ++c) {
      const int shift = static_cast<int>(c % 8) * 4;
      out[g * (n / 8) + c / 8] |=
          static_cast<int32_t>((nib[g][c] & 0xF) << shift);
    }
  }
  return out;
}

}  // namespace

// --- AWQ hand-computed: one row, one 8-wide group. Independent-arithmetic
// oracle for the reverse-AWQ order and (w - z) * s. col7 = 15 forces a set MSB
// on the packed int32 so a sign-extension unpack bug would surface. ---
TEST_CASE("DequantAwq4ToBf16 hand-computed reverse order + sign safety") {
  // desired output cols: 5,3,0,15,8,1,2,15
  // reverse[e]: e->pos {0:0,1:4,2:1,3:5,4:2,5:6,6:3,7:7}
  //  pos0=col0=5 pos1=col2=0 pos2=col4=8 pos3=col6=2
  //  pos4=col1=3 pos5=col3=15 pos6=col5=1 pos7=col7=15  => 0xF1F32805
  std::vector<int32_t> qweight = {static_cast<int32_t>(0xF1F32805)};
  std::vector<int32_t> qzeros = {static_cast<int32_t>(0x22222222)};  // z=2 all cols
  std::vector<float> scales = {1.0F, 2.0F, 4.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};

  std::vector<uint16_t> out(8, 0xFFFF);
  DequantAwq4ToBf16(qweight.data(), scales.data(), qzeros.data(), /*k=*/1,
                    /*n=*/8, /*group_size=*/1, out.data());

  const float expected[8] = {
      (5 - 2) * 1.0F,   // 3
      (3 - 2) * 2.0F,   // 2
      (0 - 2) * 4.0F,   // -8
      (15 - 2) * 1.0F,  // 13
      (8 - 2) * 1.0F,   // 6
      (1 - 2) * 1.0F,   // -1
      (2 - 2) * 1.0F,   // 0
      (15 - 2) * 1.0F,  // 13
  };
  for (int i = 0; i < 8; ++i) {
    CHECK(vt::BF16ToF32(out[i]) == doctest::Approx(expected[i]));
  }
}

// --- GPTQ hand-computed: 8 K-rows x 8 cols (N%8==0, qzeros packed along N),
// standard K-packing; column 0 is the meaningful one (rows 1..7 all zero).
// Verifies the zero_offset convention for BOTH v1 (offset 1) and v2 (offset 0). ---
TEST_CASE("DequantGptq4ToBf16 hand-computed zero_offset v1/v2") {
  // column 0 rows k0..k7 = 5,3,0,15,8,1,2,7 -> standard K-pack = 0x7218F035;
  // columns 1..7 all zero. qweight is [K/8=1][N=8].
  std::vector<int32_t> qweight(8, 0);
  qweight[0] = static_cast<int32_t>(0x7218F035);
  // qzeros [num_groups=1][N/8=1]: col0 zero at standard nibble 0 = 3, cols 1..7 = 0.
  std::vector<int32_t> qzeros = {0x00000003};
  std::vector<float> scales(8, 1.0F);  // [1][8]
  const int rows[8] = {5, 3, 0, 15, 8, 1, 2, 7};

  SUBCASE("v1 (zero_offset=1): eff_zero = 3 + 1 = 4") {
    std::vector<uint16_t> out(8 * 8, 0xFFFF);
    DequantGptq4ToBf16(qweight.data(), scales.data(), qzeros.data(),
                       /*g_idx=*/nullptr, /*k=*/8, /*n=*/8, /*group_size=*/8,
                       /*zero_offset=*/1, out.data());
    for (int k = 0; k < 8; ++k) {
      CHECK(vt::BF16ToF32(out[k * 8 + 0]) ==
            doctest::Approx(static_cast<float>(rows[k] - 4)));
    }
  }
  SUBCASE("v2 (zero_offset=0): eff_zero = 3") {
    std::vector<uint16_t> out(8 * 8, 0xFFFF);
    DequantGptq4ToBf16(qweight.data(), scales.data(), qzeros.data(),
                       /*g_idx=*/nullptr, /*k=*/8, /*n=*/8, /*group_size=*/8,
                       /*zero_offset=*/0, out.data());
    for (int k = 0; k < 8; ++k) {
      CHECK(vt::BF16ToF32(out[k * 8 + 0]) ==
            doctest::Approx(static_cast<float>(rows[k] - 3)));
    }
  }
}

// --- GPTQ act-order: g_idx remaps rows to groups so a row-stride/group-index
// bug surfaces. K=8, N=8, 2 groups; g_idx sends even rows to group 0, odd to
// group 1, each group with a distinct zero, and confirms scale/zero indexing
// follows g_idx not k/group_size. ---
TEST_CASE("DequantGptq4ToBf16 act-order g_idx group selection") {
  const int64_t k = 8, n = 8, groups = 2;
  std::vector<std::vector<int>> wn(k, std::vector<int>(n, 0));
  for (int64_t r = 0; r < k; ++r)
    for (int64_t c = 0; c < n; ++c) wn[r][c] = static_cast<int>((r + c) % 16);
  auto qweight = PackGptqWeightAlongK(wn, k, n);

  std::vector<std::vector<int>> zn(groups, std::vector<int>(n, 0));
  for (int64_t c = 0; c < n; ++c) {
    zn[0][c] = 1;  // group 0 zero
    zn[1][c] = 4;  // group 1 zero
  }
  auto qzeros = PackGptqZerosAlongN(zn, groups, n);
  std::vector<float> scales(groups * n);
  for (int64_t g = 0; g < groups; ++g)
    for (int64_t c = 0; c < n; ++c) scales[g * n + c] = (g == 0) ? 1.0F : 2.0F;

  // even rows -> group 0, odd rows -> group 1 (a permutation groupsize=1 could
  // never produce).
  std::vector<int32_t> g_idx(k);
  for (int64_t r = 0; r < k; ++r) g_idx[r] = static_cast<int32_t>(r % 2);

  std::vector<uint16_t> out(k * n, 0xFFFF);
  DequantGptq4ToBf16(qweight.data(), scales.data(), qzeros.data(), g_idx.data(),
                     k, n, /*group_size=*/1, /*zero_offset=*/0, out.data());

  for (int64_t r = 0; r < k; ++r) {
    const int64_t g = r % 2;
    const int zval = (g == 0) ? 1 : 4;
    const float s = (g == 0) ? 1.0F : 2.0F;
    for (int64_t c = 0; c < n; ++c) {
      const float exp = (static_cast<float>(wn[r][c]) - zval) * s;
      CHECK(vt::BF16ToF32(out[r * n + c]) == doctest::Approx(exp));
    }
  }
}

// --- Randomized roundtrip vs a DOUBLE-precision reference. Nibbles/zeros/scales
// are drawn at random, packed through the independent reference packers, and the
// production dequant is compared to a double-precision (w - z) * s. Exercises
// multi-group row/col offset arithmetic that the hand cases do not. ---
TEST_CASE("DequantAwq4ToBf16 randomized roundtrip vs double reference") {
  std::mt19937 rng(0xA0B1C2D3U);
  std::uniform_int_distribution<int> nib(0, 15);
  std::uniform_real_distribution<float> sdist(-2.0F, 2.0F);

  const int64_t k = 64, n = 32, group_size = 16;
  const int64_t groups = k / group_size;

  std::vector<std::vector<int>> wn(k, std::vector<int>(n));
  for (auto& row : wn)
    for (auto& v : row) v = nib(rng);
  std::vector<std::vector<int>> zn(groups, std::vector<int>(n));
  for (auto& row : zn)
    for (auto& v : row) v = nib(rng);
  std::vector<float> scales(groups * n);
  for (auto& s : scales) s = sdist(rng);

  auto qweight = PackAwqAlongN(wn, k, n);
  auto qzeros = PackAwqAlongN(zn, groups, n);

  std::vector<uint16_t> out(k * n, 0xFFFF);
  DequantAwq4ToBf16(qweight.data(), scales.data(), qzeros.data(), k, n,
                    group_size, out.data());

  for (int64_t r = 0; r < k; ++r) {
    const int64_t g = r / group_size;
    for (int64_t c = 0; c < n; ++c) {
      const double ref =
          (static_cast<double>(wn[r][c]) - zn[g][c]) *
          static_cast<double>(scales[g * n + c]);
      // bf16 has ~8 mantissa bits: independent double ref within bf16 rounding.
      CHECK(vt::BF16ToF32(out[r * n + c]) ==
            doctest::Approx(ref).epsilon(0.01));
    }
  }
}

TEST_CASE("DequantGptq4ToBf16 randomized roundtrip vs double reference") {
  std::mt19937 rng(0x1234ABCDU);
  std::uniform_int_distribution<int> nib(0, 15);
  std::uniform_real_distribution<float> sdist(-2.0F, 2.0F);

  const int64_t k = 64, n = 32, group_size = 16;
  const int64_t groups = k / group_size;
  const int zero_offset = 1;  // classic AutoGPTQ

  std::vector<std::vector<int>> wn(k, std::vector<int>(n));
  for (auto& row : wn)
    for (auto& v : row) v = nib(rng);
  std::vector<std::vector<int>> zn(groups, std::vector<int>(n));
  for (auto& row : zn)
    for (auto& v : row) v = nib(rng);
  std::vector<float> scales(groups * n);
  for (auto& s : scales) s = sdist(rng);

  auto qweight = PackGptqWeightAlongK(wn, k, n);
  auto qzeros = PackGptqZerosAlongN(zn, groups, n);

  std::vector<uint16_t> out(k * n, 0xFFFF);
  DequantGptq4ToBf16(qweight.data(), scales.data(), qzeros.data(),
                     /*g_idx=*/nullptr, k, n, group_size, zero_offset,
                     out.data());

  for (int64_t r = 0; r < k; ++r) {
    const int64_t g = r / group_size;
    for (int64_t c = 0; c < n; ++c) {
      const double ref =
          (static_cast<double>(wn[r][c]) - (zn[g][c] + zero_offset)) *
          static_cast<double>(scales[g * n + c]);
      CHECK(vt::BF16ToF32(out[r * n + c]) ==
            doctest::Approx(ref).epsilon(0.01));
    }
  }
}

// --- Guard rails: null / mis-sized inputs abort. ---
TEST_CASE("DequantAwq4/Gptq4 argument validation") {
  std::vector<int32_t> qw = {0};
  std::vector<int32_t> qz = {0};
  std::vector<float> sc(8, 1.0F);
  std::vector<uint16_t> out(8, 0);
  CHECK_THROWS(DequantAwq4ToBf16(nullptr, sc.data(), qz.data(), 1, 8, 1, out.data()));
  CHECK_THROWS(DequantAwq4ToBf16(qw.data(), sc.data(), qz.data(), 1, 7, 1, out.data()));  // n%8
  CHECK_THROWS(DequantAwq4ToBf16(qw.data(), sc.data(), qz.data(), 1, 8, 3, out.data()));  // G!|k
  CHECK_THROWS(DequantGptq4ToBf16(qw.data(), sc.data(), qz.data(), nullptr, 4, 8, 4, 1,
                                  out.data()));  // k%8
  CHECK_THROWS(DequantGptq4ToBf16(qw.data(), sc.data(), qz.data(), nullptr, 8, 8, 8, 2,
                                  out.data()));  // zero_offset
}
