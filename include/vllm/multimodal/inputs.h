// Multimodal input containers — C++ mirror of vllm/multimodal/inputs.py.
//
// Ported from: vllm/multimodal/inputs.py (MultiModalKwargs / MultiModalInputs)
// @ vLLM e24d1b24. This is the M1 input-pipeline surface: the processed,
// per-modality feature tensors + grid metadata that the (M2) vision tower will
// consume, the placeholder-EXPANDED prompt ids, and the per-item mm-hashes.
//
// INERT-WHEN-OFF: every field defaults empty; a text-only request carries an
// empty MultiModalInputs and every downstream mm hook is a no-op.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vllm::multimodal {

// Processed features for ONE image item — the Qwen3-VL image branch of
// MultiModalKwargs (vllm/multimodal/inputs.py). `pixel_values` is the flattened
// patch matrix [num_patches, patch_feature_dim]; `image_grid_thw` = [t, h, w]
// (post-merge grid is h/merge x w/merge). vLLM casts mm_kwargs to the model
// dtype (bf16) in processing/context.py::call_hf_processor -> _postprocess_output;
// we keep BOTH the pre-cast float32 (exact HF-processor output) and the bf16
// production bytes (what the encoder consumes) so the parity gate can check both.
struct ImageKwargs {
  std::vector<float> pixel_values_f32;      // [num_patches * patch_feature_dim]
  std::vector<uint16_t> pixel_values_bf16;  // round-to-nearest-even of the above
  int64_t num_patches = 0;
  int64_t patch_feature_dim = 0;
  std::array<int64_t, 3> image_grid_thw{0, 0, 0};  // [grid_t, grid_h, grid_w]

  bool empty() const { return num_patches == 0; }
};

// One multimodal placeholder occupied in the prompt id stream — the C++ analogue
// of MultiModalFeatureSpec (vllm/multimodal/inputs.py). Carried on Request so the
// scheduler/encoder-cache seam can budget/allocate/free per item WITHOUT the
// text path ever seeing a non-empty vector.
struct MultiModalFeatureSpec {
  std::string mm_hash;                     // MultiModalHasher digest (hex)
  std::string modality = "image";          // "image" (M1); "video"/"audio" later
  int offset = 0;                          // start index in the expanded prompt
  int length = 0;                          // number of placeholder tokens (N)
  std::shared_ptr<ImageKwargs> data;       // encoder input (opaque until M2)
};

// The processed bundle returned by the mm input pipeline for a single prompt —
// mirror of MultiModalInputs (prompt_token_ids + mm_kwargs + mm_hashes +
// mm_placeholders). Empty for text-only prompts.
struct MultiModalInputs {
  std::vector<int32_t> prompt_token_ids;             // placeholder-EXPANDED ids
  std::vector<MultiModalFeatureSpec> mm_features;    // one per placeholder item

  bool empty() const { return mm_features.empty(); }
};

}  // namespace vllm::multimodal
