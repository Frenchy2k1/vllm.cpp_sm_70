// Gemma-4 text backbone (`Gemma4ForConditionalGeneration`) registry TU —
// MODEL-GEMMA4 G1. Self-registers "Gemma4ForConditionalGeneration" via
// REGISTER_VLLM_MODEL and owns the arch entry points (config hook, KV-cache spec,
// LoadedModel subclass + factory). Mirrors the gemma3_registry.cpp seam (new TU +
// one in-TU REGISTER line -> ZERO shared-array edit). The engine's HfConfig
// loader descends into `text_config` (hf_config.cpp:103-113), so the typed fields
// + config.raw are the text sub-dict of the mm wrapper config.
//
// G1 SCOPE: the text backbone only. The vision (SigLIP) + audio (USM-Conformer)
// towers are G2/G3 (skipped by the weight loader). See gemma4-multimodal.md.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/gemma4.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits (shared carrier)
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py _ModelInfo: text generation via the Gemma-4 language_model stack.
// supports_multimodal stays false for G1 (the towers are not built yet); the
// arch is the mm-wrapper name because the checkpoint carries no bare text row.
inline constexpr ModelInfo kGemma4Info{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

class Gemma4LoadedModel final : public LoadedModel {
 public:
  Gemma4LoadedModel(const ModelRegistration& registration, Gemma4Weights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}
  const Gemma4Weights& weights() const { return weights_; }

 private:
  Gemma4Weights weights_;
};

std::unique_ptr<LoadedModel> LoadGemma4ForConditionalGeneration(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture Gemma4ForConditionalGeneration does not support "
        "GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<Gemma4LoadedModel>(
      registration,
      LoadGemma4ForConditionalGenerationWeights(*source.safetensors, config));
}

void PrepareGemma4ForConditionalGeneration(LoadedModel& model,
                                           const HfConfig& config,
                                           vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardGemma4ForConditionalGeneration(
    LoadedModel& model, const ModelForwardInput& input) {
  const auto& gemma = static_cast<Gemma4LoadedModel&>(model);
  const Gemma4Weights& weights = gemma.weights();
  if (input.gather_logits) {
    return Gemma4Model::ForwardDevice(input.token_ids, input.positions,
                                      input.attn_meta, input.attn_kv, weights,
                                      input.config, input.queue,
                                      input.logits_indices);
  }
  return HostLogits(
      Gemma4Model::Forward(input.token_ids, input.positions, input.attn_meta,
                           input.attn_kv, weights, input.config, input.queue,
                           input.logits_indices),
      input.config.vocab_size);
}

const ModelFactory kGemma4Factory{
    .parse_config = &ParseGemma4ForConditionalGenerationConfig,
    .load_weights = &LoadGemma4ForConditionalGeneration,
    .prepare = &PrepareGemma4ForConditionalGeneration,
    .forward = &ForwardGemma4ForConditionalGeneration,
    .make_kv_cache = &MakeGemma4ForConditionalGenerationKVCache,
    .is_dense_model = true,
};

}  // namespace

void ParseGemma4ForConditionalGenerationConfig(const HfConfig& config) {
  (void)config;  // Gemma-4 scalars are read from config.raw by loader/forward.
}

v1::KVCacheConfig MakeGemma4ForConditionalGenerationKVCache(const HfConfig& config,
                                                            int block_size,
                                                            int num_blocks) {
  // TRUE topology: TWO head dims — sliding layers head_dim=256, full layers
  // global_head_dim=512, both num_key_value_heads=2, plus YOCO KV-sharing on the
  // last num_kv_shared_layers. The current runner allocates ONE uniform KV
  // head_dim per non-GDN layer (runner.cpp:600-646), so it cannot yet consume a
  // per-layer/per-group head_dim. Until that shared-path change lands, this emits
  // the single-group uniform spec the runner accepts (at the sliding head_dim);
  // the forward's per-layer VT_CHECK(kv.head_size == Dh) makes the mismatch on
  // full-attention layers an explicit, honest failure rather than a silent wrong
  // answer. THIS IS THE G1->G-next e2e blocker (see gemma4.h / gemma4-multimodal.md).
  const int num_kv_heads = static_cast<int>(config.num_key_value_heads);
  const int head_dim = static_cast<int>(config.head_dim);  // 256 (sliding)

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<v1::FullAttentionSpec>(
          block_size, num_kv_heads, head_dim, v1::ResolveKvCacheDType()));
  return kv;
}

REGISTER_VLLM_MODEL(gemma4, "Gemma4ForConditionalGeneration", kGemma4Factory,
                    kGemma4Info)

}  // namespace vllm
