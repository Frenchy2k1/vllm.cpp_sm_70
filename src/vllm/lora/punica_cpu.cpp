// Punica-style batched LoRA apply — CPU implementation.
//
// UPSTREAM (ported FROM; ${VLLM_SOURCE} @ 555967922 / vLLM 0.26.0.dev0):
//   vllm/lora/ops/torch_ops/lora_ops.py:24-128    bgmv_shrink/expand/expand_slice
//   vllm/lora/punica_wrapper/punica_cpu.py:265-312 add_lora_linear
//   vllm/lora/lora_weights.py:36-96               optimize / create_dummy
//   vllm/lora/layers/base_linear.py:100-238       create/set/reset/apply
//
// Portable float math (the CPU brick per .agents/specs/lora-adapter.md). The
// einsum "bi,boi->bo" (lora_ops.py:35,78,123) is, per token t with slot s:
//   out[t, o] = sum_i x[t, i] * W_stacked[s][o, i]
// with W_stacked[s] contiguous row-major [o_dim, i_dim].
#include "vllm/lora/punica.h"

#include <cstddef>

#include "vllm/lora/lora_weights.h"

namespace vllm {
namespace lora {

namespace {

// One token's o = W[s] @ x_row (W[s] row-major [o_dim, i_dim]). Accumulates in
// double for numeric stability, matching vLLM's float32 punica buffer intent.
inline void MatVecRow(const float* w_slot, const float* x_row, int64_t o_dim,
                      int64_t i_dim, double scale, bool add, float* out_row,
                      int64_t out_stride) {
  for (int64_t o = 0; o < o_dim; ++o) {
    const float* w_o = w_slot + o * i_dim;
    double acc = 0.0;
    for (int64_t i = 0; i < i_dim; ++i) {
      acc += static_cast<double>(w_o[i]) * static_cast<double>(x_row[i]);
    }
    acc *= scale;
    float* dst = out_row + o * out_stride;
    *dst = add ? static_cast<float>(*dst + acc) : static_cast<float>(acc);
  }
}

}  // namespace

// ---------------------------------------------------------------------------

void BgmvShrink(const float* x, int64_t T, int64_t in_dim,
                const float* a_stacked, int64_t num_slots, int64_t rank,
                const int32_t* indices, double scaling, float* out) {
  // lora_ops.py:67-80 — selected_loras = a[idx] ([rank, in]); out = scaling *
  // einsum. OVERWRITE for active tokens; idx < 0 skipped (out pre-zeroed).
  const int64_t slot_stride = rank * in_dim;
  for (int64_t t = 0; t < T; ++t) {
    const int32_t s = indices[t];
    if (s < 0 || s >= num_slots) continue;
    MatVecRow(a_stacked + s * slot_stride, x + t * in_dim, /*o_dim=*/rank,
              /*i_dim=*/in_dim, scaling, /*add=*/false, out + t * rank,
              /*out_stride=*/1);
  }
}

void BgmvExpandSlice(const float* buffer, int64_t T, int64_t rank,
                     const float* b_stacked, int64_t num_slots, int64_t out_dim,
                     const int32_t* indices, int64_t y_width,
                     int64_t slice_offset, int64_t slice_size, bool add_inputs,
                     float* y) {
  // lora_ops.py:110-128 — selected_loras = b[idx] ([out, rank]); y window
  // [off:off+slice] += / = einsum. slice_size == out_dim for a plain linear.
  const int64_t slot_stride = out_dim * rank;
  const int64_t n = slice_size < out_dim ? slice_size : out_dim;
  for (int64_t t = 0; t < T; ++t) {
    const int32_t s = indices[t];
    if (s < 0 || s >= num_slots) continue;
    MatVecRow(b_stacked + s * slot_stride, buffer + t * rank, /*o_dim=*/n,
              /*i_dim=*/rank, /*scale=*/1.0, add_inputs,
              y + t * y_width + slice_offset, /*out_stride=*/1);
  }
}

void BgmvExpand(const float* buffer, int64_t T, int64_t rank,
                const float* b_stacked, int64_t num_slots, int64_t out_dim,
                const int32_t* indices, bool add_inputs, float* y) {
  // lora_ops.py:24 — full-width expand at offset 0 (y_width == out_dim).
  BgmvExpandSlice(buffer, T, rank, b_stacked, num_slots, out_dim, indices,
                  /*y_width=*/out_dim, /*slice_offset=*/0, /*slice_size=*/out_dim,
                  add_inputs, y);
}

void AddLoraLinear(float* y, const float* x, int64_t T, int64_t in_dim,
                   int64_t out_dim, const float* a_stacked,
                   const float* b_stacked, int64_t num_slots, int64_t rank,
                   const int32_t* indices, double scaling) {
  // punica_cpu.py:265-312 — buffer = shrink(x); y += expand(buffer). Buffer is
  // zero-initialized so idx < 0 tokens contribute nothing through the expand.
  std::vector<float> buffer(static_cast<size_t>(T * rank), 0.0f);
  BgmvShrink(x, T, in_dim, a_stacked, num_slots, rank, indices, scaling,
             buffer.data());
  BgmvExpandSlice(buffer.data(), T, rank, b_stacked, num_slots, out_dim, indices,
                  /*y_width=*/out_dim, /*slice_offset=*/0, /*slice_size=*/out_dim,
                  /*add_inputs=*/true, y);
}

// ---------------------------------------------------------------------------
// LoRALayerWeights

LoRALayerWeights& LoRALayerWeights::Optimize() {
  // lora_weights.py:36-42 — fold scaling into lora_b; no-op when scaling == 1.
  if (scaling == 1.0) return *this;
  for (float& v : lora_b) v = static_cast<float>(v * scaling);
  scaling = 1.0;
  return *this;
}

LoRALayerWeights LoRALayerWeights::CreateDummy(const std::string& module_name,
                                               int input_dim, int output_dim,
                                               int rank) {
  // lora_weights.py:72-96 — zero a[rank,input] / b[output,rank], alpha == 1.
  LoRALayerWeights w;
  w.module_name = module_name;
  w.rank = rank;
  w.lora_alpha = 1;
  w.input_dim = input_dim;
  w.output_dim = output_dim;
  w.lora_a.assign(static_cast<size_t>(rank) * input_dim, 0.0f);
  w.lora_b.assign(static_cast<size_t>(output_dim) * rank, 0.0f);
  w.scaling = rank != 0 ? 1.0 / rank : 1.0;
  return w;
}

// ---------------------------------------------------------------------------
// LoRALinear

void LoRALinear::ResetLora(int64_t index) {
  // base_linear.py:153-156 — zero slot `index` in both stacked tensors.
  const int64_t a_stride = max_rank_ * input_size_;
  const int64_t b_stride = output_size_ * max_rank_;
  float* a = a_stacked_.data() + index * a_stride;
  float* b = b_stacked_.data() + index * b_stride;
  for (int64_t i = 0; i < a_stride; ++i) a[i] = 0.0f;
  for (int64_t i = 0; i < b_stride; ++i) b[i] = 0.0f;
  scaling_[static_cast<size_t>(index)] = 1.0;
}

void LoRALinear::SetLora(int64_t index, const LoRALayerWeights& lora) {
  // base_linear.py:158-184 — reset then copy one adapter into slot `index`.
  // The stacked slot rows are max_rank_ wide (a) / max_rank_ deep (b); the
  // adapter's rank may be smaller, so we copy into the leading sub-block and
  // leave the zero padding. Scaling is FOLDED into b (manager optimize path),
  // so the batched apply then uses scale 1.0.
  ResetLora(index);
  const int64_t a_stride = max_rank_ * input_size_;
  const int64_t b_stride = output_size_ * max_rank_;
  float* a = a_stacked_.data() + index * a_stride;
  float* b = b_stacked_.data() + index * b_stride;
  const int64_t r = lora.rank;
  // a: [r, input_size_] -> a_stacked slot [max_rank_, input_size_], rows 0..r.
  for (int64_t rr = 0; rr < r; ++rr) {
    const float* src = lora.lora_a.data() + rr * lora.input_dim;
    float* dst = a + rr * input_size_;
    for (int64_t i = 0; i < input_size_; ++i) dst[i] = src[i];
  }
  // b: [output_size_, r] * scaling -> b_stacked slot [output_size_, max_rank_].
  const double scale = lora.scaling;
  for (int64_t o = 0; o < output_size_; ++o) {
    const float* src = lora.lora_b.data() + o * r;
    float* dst = b + o * max_rank_;
    for (int64_t rr = 0; rr < r; ++rr) {
      dst[rr] = static_cast<float>(static_cast<double>(src[rr]) * scale);
    }
  }
  scaling_[static_cast<size_t>(index)] = lora.scaling;
}

void LoRALinear::ApplyLoraToOutput(float* y, const float* x, int64_t T,
                                   const int32_t* indices) const {
  // base_linear.py:227-229 — add_lora_linear with scale 1.0 (scaling folded).
  AddLoraLinear(y, x, T, input_size_, output_size_, a_stacked_.data(),
                b_stacked_.data(), num_slots_, max_rank_, indices,
                /*scaling=*/1.0);
}

}  // namespace lora
}  // namespace vllm
