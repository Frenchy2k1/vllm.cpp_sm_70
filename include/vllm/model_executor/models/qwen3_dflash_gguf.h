// DFlash draft loading from a `dflash`-arch GGUF (`SPEC-DFLASH-GGUF`).
#ifndef VLLM_MODEL_EXECUTOR_MODELS_QWEN3_DFLASH_GGUF_H_
#define VLLM_MODEL_EXECUTOR_MODELS_QWEN3_DFLASH_GGUF_H_

#include <cstdint>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/qwen3_dflash.h"
#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

// Build the draft's HfConfig from a `dflash` GGUF's metadata, the GGUF
// counterpart of MakeDflashDraftConfig's config.json read.
//
// Two conventions this undoes, BOTH invisible to shape checks:
//   * `dflash.target_layers` is stored +1-offset by llama.cpp's converter, so
//     the rebuilt `dflash_config.target_layer_ids` subtracts one;
//   * the mask token arrives on the STANDARD `tokenizer.ggml.mask_token_id`
//     key, not a dflash-specific one.
// `vocab_size` is left 0: a DFlash draft carries no vocab key and no
// embed/lm_head tensors because it SHARES the target's.
HfConfig MakeDflashGgufConfig(const GgufFile& gguf);

// Load the draft's weights from the same file. Norms are read RAW - the
// `DFlashModel` converter class does NOT inherit the Qwen3Next `(w + 1)` norm
// shift, so unlike the trunk and the MTP head these must not be un-shifted.
// embed_tokens / lm_head are NOT read here; they come from the target.
Qwen3DFlashWeights LoadQwen3DFlashFromGguf(const GgufFile& gguf,
                                           const HfConfig& config,
                                           int64_t num_taps,
                                           int32_t mask_token_id);

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_MODELS_QWEN3_DFLASH_GGUF_H_
