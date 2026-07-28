// Gemma-4 text backbone (`Gemma4ForConditionalGeneration`) registry TU —
// MODEL-GEMMA4 G1. Self-registers "Gemma4ForConditionalGeneration" via
// REGISTER_VLLM_MODEL and owns the arch entry points (config hook, KV-cache spec,
// LoadedModel subclass + factory). Mirrors the gemma3_registry.cpp seam (new TU +
// one in-TU REGISTER line -> ZERO shared-array edit). The engine's HfConfig
// loader descends into `text_config` for the TYPED fields (hf_config.cpp:103-113),
// but keeps `config.raw` as the FULL config.json (hf_config.cpp:414). Gemma-4's
// per-arch scalars (global_head_dim, layer_types, hidden_size_per_layer_input,
// rope_parameters, num_kv_shared_layers) are nested under raw["text_config"] in
// the mm wrapper, so every raw read here / in gemma4.cpp / gemma4_weights.cpp
// goes through a `TextCfg`/text_config view (top-level for a plain config).
//
// G1 SCOPE: the text backbone only. The vision (SigLIP) + audio (USM-Conformer)
// towers are G2/G3 (skipped by the weight loader). See gemma4-multimodal.md.
#include "vllm/model_executor/models/model_registry.h"

#include <algorithm>
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
  (void)config;  // Gemma-4 scalars are read from raw["text_config"] by loader/forward.
}

v1::KVCacheConfig MakeGemma4ForConditionalGenerationKVCache(const HfConfig& config,
                                                            int block_size,
                                                            int num_blocks) {
  // TRUE topology (G1b LANDED): TWO head dims — sliding layers head_dim=256,
  // full_attention layers global_head_dim=512, both num_key_value_heads=2, plus
  // YOCO KV-sharing on the last num_kv_shared_layers. The runner now consumes a
  // PER-LAYER attention spec (`per_layer_attn_specs`, runner.cpp G1b), so each
  // non-GDN layer allocates + views its own head_dim. The single group below
  // still drives the (head_dim-independent) block table / KV manager; its spec
  // carries the LARGER head_dim as an honest per-page upper bound.
  // HfConfig::raw is the FULL config.json (hf_config.cpp:414); Gemma-4's
  // global_head_dim / layer_types are nested under `text_config` in the mm
  // wrapper, so resolve that view (top-level for a plain config).
  const nlohmann::json& raw =
      (config.raw.contains("text_config") && config.raw.at("text_config").is_object())
          ? config.raw.at("text_config")
          : config.raw;
  const int num_kv_heads = static_cast<int>(config.num_key_value_heads);
  const int head_dim_sliding = static_cast<int>(config.head_dim);  // 256
  int head_dim_full = head_dim_sliding;                            // 512
  if (const auto it = raw.find("global_head_dim");
      it != raw.end() && it->is_number_integer()) {
    head_dim_full = it->get<int>();
  }
  const vt::DType kv_dtype = v1::ResolveKvCacheDType();
  const int64_t L = config.num_hidden_layers;

  // Per-layer full_attention flag from config.layer_types (gemma4.py:441-489);
  // a missing/short array defaults a layer to sliding (head_dim_sliding). This
  // mirrors gemma4_weights.cpp::MakeTopo (allocation only needs head_dim; YOCO
  // kv_target is handled in the forward, not here — G1c would DEDUP the shared
  // layers' unused buffers).
  std::vector<bool> is_full(static_cast<size_t>(L), false);
  if (const auto it = raw.find("layer_types");
      it != raw.end() && it->is_array()) {
    for (int64_t l = 0; l < L && static_cast<size_t>(l) < it->size(); ++l) {
      is_full[static_cast<size_t>(l)] =
          it->at(static_cast<size_t>(l)).is_string() &&
          it->at(static_cast<size_t>(l)).get<std::string>() == "full_attention";
    }
  }

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<v1::FullAttentionSpec>(
          block_size, num_kv_heads, std::max(head_dim_sliding, head_dim_full),
          kv_dtype));
  kv.per_layer_attn_specs.reserve(static_cast<size_t>(L));
  for (int64_t l = 0; l < L; ++l) {
    const int hd = is_full[static_cast<size_t>(l)] ? head_dim_full
                                                   : head_dim_sliding;
    kv.per_layer_attn_specs.push_back(std::make_shared<v1::FullAttentionSpec>(
        block_size, num_kv_heads, hd, kv_dtype));
  }
  return kv;
}

REGISTER_VLLM_MODEL(gemma4, "Gemma4ForConditionalGeneration", kGemma4Factory,
                    kGemma4Info)

}  // namespace vllm
