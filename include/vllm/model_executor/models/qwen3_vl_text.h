// Qwen3-VL text-backbone helpers (M2b/M2c) — the pieces that fork the plain
// Qwen3-dense text path for the vision-conditioned decode. Pure ADDITIVE TU: it
// does NOT touch the shared dense forward / model runner / registry, so the text
// SACRED gates are byte-identical by construction (like the M2a tower).
//
// Ported from vllm/model_executor/models/qwen3_vl.py @ e24d1b24:
//   _get_mrope_input_positions (:2567), _iter_mm_grid_hw (:2482) — get_rope_index;
//   _compute_deepstack_embeds (:2761) + Qwen3LLMModel.forward deepstack (:1589);
//   vllm/model_executor/models/utils.py::_merge_multimodal_embeddings (:524).
//
// The three deterministic contracts here are host, index-math functions (the
// production device scatter reuses the same position list): they are unit-gated
// bit/near-exact against the dumped vLLM 0.25.0 reference
// (scripts/mm/m2b_text_ref_dump.py). The 3-section MRoPE APPLICATION itself is
// the existing vt::RopeFromCache mrope path (positions [3,T] + mrope_section),
// gated separately in the same test against MRotaryEmbedding.forward_native.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace vllm::multimodal {

// One image placeholder occurrence in the (already placeholder-expanded) prompt.
struct MmImageSpan {
  int64_t offset;                 // index of the first image token in input_ids
  std::array<int64_t, 3> grid_thw;  // (t, h, w) patch grid (t == 1 for images)
};

// get_rope_index (single-image / text): compute the MRoPE 3-D position ids
// [3, T] (row-major, row 0 = temporal, 1 = height, 2 = width). Text tokens get
// sequential positions broadcast to all 3 rows; image tokens get (t,h,w) grid
// positions offset past the running max. `spatial_merge_size` divides h,w to the
// LLM grid. Mirrors _get_mrope_input_positions for the image path (video/EVS and
// multi-image lumping are deferred to M3 and rejected here). Returns a flat
// vector of length 3*T. `mrope_position_delta` (== max+1 - T) is written out if
// non-null (used to continue positions during decode).
std::vector<int32_t> Qwen3VLGetRopeIndex(const std::vector<int32_t>& input_ids,
                                         const std::vector<MmImageSpan>& images,
                                         int64_t spatial_merge_size,
                                         int64_t* mrope_position_delta = nullptr);

// _merge_multimodal_embeddings (masked scatter): overwrite the rows of
// `inputs_embeds` [T, H] at the visual-token positions (mask true) with the
// consecutive rows of `mm_embeds` [N, H]. `mask` has length T with exactly N
// true entries. In place on `inputs_embeds`.
void Qwen3VLMergeMultimodal(std::vector<float>& inputs_embeds, int64_t T, int64_t H,
                            const std::vector<float>& mm_embeds,
                            const std::vector<bool>& mask);

// _compute_deepstack_embeds scatter: given the tower's per-visual-token multiscale
// features `multiscale` [N, L*H] and the visual mask, produce the decoder-injection
// tensor [L, T, H] (row-major, L = num deepstack levels): out[l][t] = the l-th
// H-slice of the k-th visual token's multiscale row when token t is the k-th
// visual token, zero elsewhere. Mirrors the zero-init + masked-scatter +
// view(T,L,H).permute(1,0,2) of _compute_deepstack_embeds.
std::vector<float> Qwen3VLComputeDeepstack(const std::vector<float>& multiscale,
                                           int64_t N, int64_t L, int64_t H,
                                           const std::vector<bool>& mask, int64_t T);

}  // namespace vllm::multimodal
