// Ported from:
//   vllm/model_executor/layers/quantization/awq_triton.py (awq_dequantize_kernel) @ 555967922
//   csrc/libtorch_stable/quantization/gptq/qdq_4.cuh (dequant_4bit_8_gptq) @ 555967922
//   csrc/libtorch_stable/quantization/gptq/q_gemm.cu:201-202 (zero_offset) @ 555967922
#include "vllm/model_executor/model_loader/awq_gptq_dequant.h"

#include "vt/dtype.h"

namespace vllm {

namespace {

// Nibble e (0..7) of an int32, read as uint32 so a set sign bit cannot leak
// into the shifted value.
inline uint32_t Nibble(int32_t packed, int shift) {
  return (static_cast<uint32_t>(packed) >> shift) & 0xFU;
}

}  // namespace

void DequantAwq4ToBf16(const int32_t* qweight, const float* scales,
                       const int32_t* qzeros, int64_t k, int64_t n,
                       int64_t group_size, uint16_t* out_bf16) {
  VT_CHECK(qweight != nullptr, "awq dequant: qweight is null");
  VT_CHECK(scales != nullptr, "awq dequant: scales is null");
  VT_CHECK(qzeros != nullptr, "awq dequant: qzeros is null");
  VT_CHECK(out_bf16 != nullptr, "awq dequant: output buffer is null");
  VT_CHECK(k > 0 && n > 0, "awq dequant: non-positive dimension");
  VT_CHECK(n % kAwqGptqPackFactor == 0, "awq dequant: n must be a multiple of 8");
  VT_CHECK(group_size > 0 && k % group_size == 0,
           "awq dequant: group_size must be positive and divide k");

  const int64_t packed_n = n / kAwqGptqPackFactor;  // int32 cols of qweight/qzeros

  for (int64_t row = 0; row < k; ++row) {
    const int64_t g = row / group_size;  // group index along K
    const int32_t* w_row = qweight + row * packed_n;
    const int32_t* z_row = qzeros + g * packed_n;
    const float* s_row = scales + g * n;
    uint16_t* o_row = out_bf16 + row * n;

    for (int64_t col = 0; col < n; ++col) {
      const int64_t pc = col / kAwqGptqPackFactor;     // which int32
      const int elem = static_cast<int>(col % kAwqGptqPackFactor);
      // AWQ order: output element `elem` reads nibble kAwqReverseOrder[elem].
      const int shift = kAwqReverseOrder[elem] * 4;

      const uint32_t w = Nibble(w_row[pc], shift);
      const uint32_t z = Nibble(z_row[pc], shift);
      // (w - z) * scale; subtraction is signed and can go negative.
      const float val =
          (static_cast<float>(static_cast<int>(w) - static_cast<int>(z))) *
          s_row[col];
      o_row[col] = vt::F32ToBF16(val);
    }
  }
}

void DequantGptq4ToBf16(const int32_t* qweight, const float* scales,
                        const int32_t* qzeros, const int32_t* g_idx, int64_t k,
                        int64_t n, int64_t group_size, int zero_offset,
                        uint16_t* out_bf16) {
  VT_CHECK(qweight != nullptr, "gptq dequant: qweight is null");
  VT_CHECK(scales != nullptr, "gptq dequant: scales is null");
  VT_CHECK(qzeros != nullptr, "gptq dequant: qzeros is null");
  VT_CHECK(out_bf16 != nullptr, "gptq dequant: output buffer is null");
  VT_CHECK(k > 0 && n > 0, "gptq dequant: non-positive dimension");
  VT_CHECK(k % kAwqGptqPackFactor == 0, "gptq dequant: k must be a multiple of 8");
  VT_CHECK(n % kAwqGptqPackFactor == 0, "gptq dequant: n must be a multiple of 8");
  VT_CHECK(group_size > 0 && k % group_size == 0,
           "gptq dequant: group_size must be positive and divide k");
  VT_CHECK(zero_offset == 0 || zero_offset == 1,
           "gptq dequant: zero_offset must be 0 (v2) or 1 (v1)");

  const int64_t packed_n = n / kAwqGptqPackFactor;  // int32 cols of qzeros

  for (int64_t row = 0; row < k; ++row) {
    // Group along K: act-order g_idx when present, else contiguous k/G.
    const int64_t g =
        g_idx != nullptr ? static_cast<int64_t>(g_idx[row]) : row / group_size;
    // qweight is packed along K: row `row` lives in int32 row row/8, nibble row%8.
    const int64_t w_prow = row / kAwqGptqPackFactor;
    const int w_shift = static_cast<int>(row % kAwqGptqPackFactor) * 4;
    const int32_t* w_row = qweight + w_prow * n;
    const int32_t* z_row = qzeros + g * packed_n;
    const float* s_row = scales + g * n;
    uint16_t* o_row = out_bf16 + row * n;

    for (int64_t col = 0; col < n; ++col) {
      const uint32_t w = Nibble(w_row[col], w_shift);
      // qzeros packed along N, standard order: col -> int32 col/8, nibble col%8.
      const int64_t pc = col / kAwqGptqPackFactor;
      const int z_shift = static_cast<int>(col % kAwqGptqPackFactor) * 4;
      const uint32_t z = Nibble(z_row[pc], z_shift);
      // effective zero = stored + zero_offset; w[k,n] = (w - eff_z) * scale.
      const int eff_z = static_cast<int>(z) + zero_offset;
      const float val =
          (static_cast<float>(static_cast<int>(w) - eff_z)) * s_row[col];
      o_row[col] = vt::F32ToBF16(val);
    }
  }
}

}  // namespace vllm
