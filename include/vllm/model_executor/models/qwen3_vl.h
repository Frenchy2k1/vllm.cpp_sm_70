// Qwen3-VL (`Qwen3VLForConditionalGeneration`) — M2c e2e image->text wire-up.
//
// The FORKED VL decode: it reuses the landed Qwen3-dense text backbone
// (Qwen3DenseWeights / dense_attn AttnBlock machinery) but forks the forward on
// three points that the plain-dense path does NOT do:
//   1. inputs_embeds path — embed text ids, then masked-scatter the vision tower's
//      merger output (Qwen3VLMergeMultimodal) into the image-placeholder rows
//      before the first decoder layer (instead of the pure embed-from-ids path).
//   2. 3-section MRoPE positions — vt::RopeFromCache over positions [3,T] +
//      mrope_section=[24,20,20] interleaved (the existing mrope path; NOT a new
//      kernel), instead of 1-D RoPE.
//   3. DeepStack injection — add the tower's 3 multiscale merger outputs
//      (Qwen3VLComputeDeepstack -> [L,T,H]) to the hidden stream after decoder
//      layers 0/1/2 (qwen3_vl.py:1589-1594).
//
// Everything else (per-head q/k RMSNorm, paged FA2, SwiGLU MLP, tied lm_head) is
// the byte-identical landed dense path. The vision tower is the M2a
// Qwen3VLVisionForward; the merge/rope-index/deepstack index math are the M2b
// host helpers (qwen3_vl_text.h). This TU is ADDITIVE — it does NOT touch the
// shared dense forward / model runner / registry, so the text SACRED gates are
// byte-identical by construction.
//
// Ported from vllm/model_executor/models/qwen3_vl.py @ e24d1b24:
//   Qwen3VLForConditionalGeneration.load_weights (:2905), get_input_embeddings /
//   forward (:2843), Qwen3LLMModel.forward deepstack (:1589-1594);
//   _get_mrope_input_positions (:2567); the language_model.* / visual.* remap.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/qwen3.h"             // Qwen3DenseWeights, PagedKvCache
#include "vllm/model_executor/models/qwen3_vl_vision.h"    // Qwen3VLVisionWeights/Config
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"

namespace vllm {

class SafetensorsFile;

// The full Qwen3-VL model weights: the plain-dense text backbone (bf16, tied
// lm_head) under the `model.language_model.*` prefix + the M2a vision tower under
// `model.visual.*`.
struct Qwen3VLWeights {
  Qwen3DenseWeights text;
  multimodal::Qwen3VLVisionWeights vision;
  multimodal::Qwen3VLVisionConfig vision_cfg;
};

// Load `Qwen3VLForConditionalGeneration` (Qwen3-VL-4B, BF16) safetensors. The
// text half remaps `model.language_model.*` onto the landed Qwen3-dense loader
// helpers (LoadMergedBf16RawNK etc.); the vision half loads `model.visual.*` into
// the M2a tower weights (bf16 widened to f32, matching the M2a dump). `config` is
// the text_config-resolved HfConfig (hidden 2560, 36 layers, 32 heads, head_dim
// 128, kv 8, vocab 151936, rope_theta 5e6, tied).
Qwen3VLWeights LoadQwen3VLWeights(const std::vector<SafetensorsFile>& shards,
                                  const HfConfig& config);

// Single-image, single-sequence GREEDY image->text generation (the M2c gate
// driver). Runs the FULL forked forward: embed(prompt_ids) + merge(mm_embeds) ->
// MRoPE prefill with DeepStack inject at layers 0/1/2 -> greedy argmax -> paged
// decode continuation (MRoPE decode positions, no deepstack). Returns the
// generated token ids (length <= max_new_tokens; stops on eos_token_id).
//
// prompt_ids : the placeholder-expanded model input ids (image_token_id repeated
//              N times at the image span).
// mm_main    : the tower merger output [N, H_text] (== tower_out[:, :out_hidden]),
//              host f32; scattered into the image rows.
// mm_deepstack: the tower multiscale output [N, L*H_text] (== tower_out[:,
//              out_hidden:]), host f32; L = num deepstack levels (3).
// grid_thw   : the LLM-grid source (t,h,w) for get_rope_index.
std::vector<int32_t> Qwen3VLGenerateGreedy(
    const std::vector<int32_t>& prompt_ids, const std::vector<float>& mm_main,
    const std::vector<float>& mm_deepstack, int64_t num_deepstack_levels,
    const std::array<int64_t, 3>& grid_thw, int32_t image_token_id,
    int32_t eos_token_id, const Qwen3VLWeights& weights, const HfConfig& config,
    vt::Queue& queue, int max_new_tokens);

}  // namespace vllm
