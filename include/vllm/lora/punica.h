// Punica-style batched LoRA apply (CPU brick).
//
// UPSTREAM (ported FROM, ground-every-impl rule; ${VLLM_SOURCE} @ 555967922/
// vLLM 0.26.0.dev0):
//   vllm/lora/ops/torch_ops/lora_ops.py:24-128   bgmv_shrink / bgmv_expand /
//                                                 bgmv_expand_slice
//   vllm/lora/punica_wrapper/punica_cpu.py:147-312  _apply_shrink/_apply_expand,
//                                                 add_shrink/add_expand,
//                                                 add_lora_linear
//   vllm/lora/layers/base_linear.py:100-238       create_lora_weights (stacked
//                                                 slots), set_lora, reset_lora,
//                                                 apply -> _apply_lora_to_output
//
// The punica apply is a batched two-GEMM sandwich around a base linear output:
//   buffer[T, rank]  = scaling * (x @ lora_aᵀ)        (SHRINK, per token's slot)
//   y[T, output]    += buffer @ lora_bᵀ               (EXPAND, add onto base)
// Each token selects its adapter by an integer slot index; index < 0 means "no
// adapter" and contributes nothing (mirrors the triton kernel early-exit at
// lora_id == -1, base_linear.py:257-259, and the CPU test references in
// tests/lora/test_punica_ops.py:36-74) — NOT the naive torch w[-1] wrap.
//
// STACKED WEIGHT LAYOUT (base_linear.py:129-151, with the "1" layer dim
// collapsed for the one-slot-per-layer CPU brick, per lora-adapter.md):
//   lora_a_stacked : [num_slots, rank,   input ] row-major, contiguous per slot
//   lora_b_stacked : [num_slots, output, rank  ] row-major, contiguous per slot
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/lora/lora_weights.h"

namespace vllm {
namespace lora {

// bgmv_shrink (lora_ops.py:67). For each token t with slot s = indices[t] >= 0:
//   out[t, :rank] = scaling * (x[t, :in] @ a_stacked[s]ᵀ)     (OVERWRITE)
// Tokens with s < 0 are left untouched (skip). `out` is [T, rank] row-major and
// must be pre-zeroed by the caller for the skip semantics to hold.
//   x         : [T, in_dim] row-major
//   a_stacked : [num_slots, rank, in_dim] row-major
void BgmvShrink(const float* x, int64_t T, int64_t in_dim,
                const float* a_stacked, int64_t num_slots, int64_t rank,
                const int32_t* indices, double scaling, float* out);

// bgmv_expand_slice (lora_ops.py:110). For each token t with slot s = indices[t]
// >= 0, writes an `slice_size`-wide window at column `slice_offset` of y:
//   y[t, off : off+slice] (+)= buffer[t, :rank] @ b_stacked[s]ᵀ
// add_inputs selects += (true) vs = (false). Tokens with s < 0 are skipped.
//   buffer    : [T, rank] row-major
//   b_stacked : [num_slots, out_dim, rank] row-major (out_dim == slice_size)
//   y         : [T, y_width] row-major
void BgmvExpandSlice(const float* buffer, int64_t T, int64_t rank,
                     const float* b_stacked, int64_t num_slots, int64_t out_dim,
                     const int32_t* indices, int64_t y_width,
                     int64_t slice_offset, int64_t slice_size, bool add_inputs,
                     float* y);

// bgmv_expand (lora_ops.py:24) == expand_slice over the full width at offset 0.
void BgmvExpand(const float* buffer, int64_t T, int64_t rank,
                const float* b_stacked, int64_t num_slots, int64_t out_dim,
                const int32_t* indices, bool add_inputs, float* y);

// add_lora_linear (punica_cpu.py:265), single-slice form: the full delta apply
// for one linear. Zeros an internal [T, rank] buffer, SHRINKs into it, then
// EXPAND-adds onto `y` in place:
//   y[t] += scaling * (x[t] @ a_stacked[s]ᵀ) @ b_stacked[s]ᵀ    for s >= 0
// `y` is [T, out_dim] row-major and already holds the base linear output.
void AddLoraLinear(float* y, const float* x, int64_t T, int64_t in_dim,
                   int64_t out_dim, const float* a_stacked,
                   const float* b_stacked, int64_t num_slots, int64_t rank,
                   const int32_t* indices, double scaling);

// ---------------------------------------------------------------------------
// Multi-slice apply (a PACKED base layer — qkv_proj, gate_up_proj — carries one
// low-rank pair per output window, so the punica sandwich runs once per slice).

// add_shrink (punica_cpu.py:166-195):
//   for i in range(n_slices): buffers[i] = scale * (x @ lora_a_stacked[i]^T)
// `buffers` is [n_slices, T, rank] contiguous and must be pre-zeroed (skip
// semantics for indices < 0). `a_stacked[i]` is [num_slots, rank_a, in_dim];
// `rank_a` is `rank` except on the fully-sharded (S-LoRA) path, where it is
// rank/tp_size — hence the explicit parameter.
void AddShrink(float* buffers, const float* x, int64_t T, int64_t in_dim,
               const std::vector<const float*>& a_stacked, int64_t num_slots,
               int64_t rank_a, int64_t buffer_rank, const int32_t* indices,
               double scale);

// add_expand (punica_cpu.py:197-236):
//   offset = offset_start
//   for i in range(n_slices):
//       y[:, offset : offset+output_slices[i]] (+)= buffers[i] @ lora_b_stacked[i]^T
//       offset += output_slices[i]
// `b_stacked[i]` is [num_slots, output_slices[i], rank].
void AddExpand(float* y, int64_t y_width, const float* buffers, int64_t T,
               int64_t rank, const std::vector<const float*>& b_stacked,
               const std::vector<int64_t>& output_slices, int64_t num_slots,
               const int32_t* indices, int64_t offset_start, bool add_inputs);

// add_lora_linear (punica_cpu.py:265-312), multi-slice form: allocate the
// n_slices zeroed [T, rank] float buffers, AddShrink into them, then AddExpand
// onto `y` with add_inputs=True. `y` is [T, sum(output_slices)] and already
// holds the base linear output.
void AddLoraLinear(float* y, const float* x, int64_t T, int64_t in_dim,
                   const std::vector<const float*>& a_stacked,
                   const std::vector<const float*>& b_stacked,
                   const std::vector<int64_t>& output_slices, int64_t num_slots,
                   int64_t rank_a, int64_t rank, const int32_t* indices,
                   double scale);

// add_lora_embedding (punica_cpu.py:238-263): the embedding LoRA needs only the
// EXPAND half — its shrink is an embedding-table lookup, not a GEMM.
//   y += x @ lora_b_stacked^T           (x is the gathered [T, rank] lora_a rows)
void AddLoraEmbedding(float* y, const float* x, int64_t T, int64_t rank,
                      const float* b_stacked, int64_t num_slots,
                      int64_t embed_dim, const int32_t* indices,
                      bool add_inputs);

// add_lora_logits (punica_cpu.py:314-351): the same shrink/expand sandwich as a
// single-slice linear, but indexed by the SAMPLER indices (one per request)
// rather than the per-token indices. The wrapper always uses bgmv
// (punica_cpu.py:348). Producing `sampler_indices` from a LoRAMapping is
// convert_mapping's job (W3 in .agents/specs/lora-adapter.md); here the caller
// passes the array.
void AddLoraLogits(float* y, int64_t y_width, const float* x, int64_t T,
                   int64_t hidden, int64_t vocab, const float* a_stacked,
                   const float* b_stacked, int64_t num_slots, int64_t rank,
                   const int32_t* sampler_indices, double scale);

// LoRALinear — the ReplicatedLinear (n_slices == 1) LoRA-wrapped layer
// (base_linear.py:70). Owns `num_slots` stacked adapter slots and applies the
// batched delta onto a base linear output. Mirrors create_lora_weights /
// set_lora / reset_lora / _apply_lora_to_output for the single-slice case.
class LoRALinear {
 public:
  // create_lora_weights (base_linear.py:100): allocate zeroed stacked slots.
  LoRALinear(int64_t num_slots, int64_t max_rank, int64_t input_size,
             int64_t output_size)
      : num_slots_(num_slots),
        max_rank_(max_rank),
        input_size_(input_size),
        output_size_(output_size),
        a_stacked_(static_cast<size_t>(num_slots * max_rank * input_size), 0.0f),
        b_stacked_(static_cast<size_t>(num_slots * output_size * max_rank),
                   0.0f),
        scaling_(num_slots, 1.0) {}

  int64_t num_slots() const { return num_slots_; }
  int64_t max_rank() const { return max_rank_; }
  int64_t input_size() const { return input_size_; }
  int64_t output_size() const { return output_size_; }

  // reset_lora (base_linear.py:153): zero slot `index` (a + b + scaling).
  void ResetLora(int64_t index);

  // set_lora (base_linear.py:158): copy one adapter into slot `index`. The
  // adapter's rank may be <= max_rank_ (zero-padded); dims must match. Mirrors
  // vLLM's runtime, where the manager optimize()s each adapter so scaling is
  // FOLDED into lora_b before the slot copy (lora_weights.py:36-42): we store
  // `scaling * lora_b` into b_stacked and the original scaling into scaling_
  // (introspection). This is why one scalar scale (1.0) then covers a mixed
  // batch where each token's slot carries its own baked-in scaling.
  void SetLora(int64_t index, const LoRALayerWeights& lora);

  // _apply_lora_to_output (base_linear.py:215-238): add the LoRA delta onto `y`
  // (the base linear output, [T, output_size]) for the batch's per-token slots.
  // Calls AddLoraLinear with scale 1.0 exactly like base_linear.py:227-229,
  // since SetLora already folded each slot's scaling into b_stacked. `indices`
  // is [T]; index < 0 => no adapter for that token (base output unchanged).
  void ApplyLoraToOutput(float* y, const float* x, int64_t T,
                         const int32_t* indices) const;

  const std::vector<float>& a_stacked() const { return a_stacked_; }
  const std::vector<float>& b_stacked() const { return b_stacked_; }
  double slot_scaling(int64_t index) const { return scaling_[static_cast<size_t>(index)]; }

 private:
  int64_t num_slots_;
  int64_t max_rank_;
  int64_t input_size_;
  int64_t output_size_;
  std::vector<float> a_stacked_;  // [num_slots, max_rank, input_size]
  std::vector<float> b_stacked_;  // [num_slots, output_size, max_rank]
  std::vector<double> scaling_;   // per-slot scaling
};

}  // namespace lora
}  // namespace vllm
