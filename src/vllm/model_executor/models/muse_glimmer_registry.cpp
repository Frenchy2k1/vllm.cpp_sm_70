// Muse Glimmer registry TU — the ADDITIVE self-registration seam (W0). Follows
// the kimi_k3_registry.cpp / deepseek_v4_registry.cpp seam exactly: a NEW
// translation unit with REGISTER_VLLM_MODEL lines and ZERO edit to any shared
// array.
//
// Upstream registers BOTH architecture strings onto the SAME class
// (registry.py @ vllm#51655): `MuseGlimmerForCausalLM` -> muse_glimmer
// and `MuseGlimmerForConditionalGeneration` -> `MuseGlimmerForCausalLM`. We mirror
// that: one factory, two registered names, so a text-only and a multimodal
// checkpoint both RESOLVE.
//
// The arch RESOLVES and parses config + accounts the structural name map. It does
// NOT forward: MuseGlimmerModel REFUSES-by-name. The model-matrix row stays SPIKE.
// Muse Glimmer is beyond the pinned oracle (555967922) and is anchored to the OPEN
// vllm#51655 — see porting-inventory §9 deviation 16 and specs/muse-glimmer.md §0.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <utility>

#include "vllm/model_executor/models/muse_glimmer.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits carrier
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"

namespace vllm {
namespace {

// registry.py _ModelInfo for Muse Glimmer: text generation, NOT hybrid (the whole
// tower is dense attention — the iRoPE split is sliding vs full, both attention),
// multimodal (the perception encoder covers image AND video).
inline constexpr ModelInfo kMuseGlimmerInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = true,
    .score_type = "bi-encoder",
};

class MuseGlimmerLoadedModel final : public LoadedModel {
 public:
  MuseGlimmerLoadedModel(const ModelRegistration& registration,
                         MuseGlimmerWeights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}
  const MuseGlimmerWeights& weights() const { return weights_; }

 private:
  MuseGlimmerWeights weights_;
};

std::unique_ptr<LoadedModel> LoadMuseGlimmer(const ModelRegistration& registration,
                                             const HfConfig& config,
                                             const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture MuseGlimmerForConditionalGeneration does not support "
        "GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<MuseGlimmerLoadedModel>(
      registration,
      LoadMuseGlimmerForConditionalGenerationWeights(*source.safetensors, config));
}

void PrepareMuseGlimmer(LoadedModel& model, const HfConfig& config,
                        vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardMuseGlimmer(LoadedModel& model,
                                 const ModelForwardInput& input) {
  auto& mg = static_cast<MuseGlimmerLoadedModel&>(model);
  const MuseGlimmerWeights& weights = mg.weights();
  if (input.gather_logits) {
    return MuseGlimmerModel::ForwardDevice(input.token_ids, input.positions,
                                           input.attn_meta, input.attn_kv, weights,
                                           input.queue, input.logits_indices);
  }
  return HostLogits(
      MuseGlimmerModel::Forward(input.token_ids, input.positions, input.attn_meta,
                                input.attn_kv, weights, input.queue,
                                input.logits_indices),
      weights.params.text.vocab_size);
}

const ModelFactory kMuseGlimmerFactory{
    .parse_config = &ParseMuseGlimmerConfig,
    .load_weights = &LoadMuseGlimmer,
    .prepare = &PrepareMuseGlimmer,
    .forward = &ForwardMuseGlimmer,
    .make_kv_cache = &MakeMuseGlimmerKVCache,
    .is_dense_model = true,
};

}  // namespace

v1::KVCacheConfig MakeMuseGlimmerKVCache(const HfConfig& config, int block_size,
                                         int num_blocks) {
  // W0 PLACEHOLDER. The TRUE topology is heterogeneous: `no_rope_layers[i] == 1`
  // layers are SLIDING-window and `== 0` layers are FULL attention
  // (muse_glimmer.py:1167-1168). The KV GEOMETRY is uniform across both (same
  // num_kv_heads and head_dim), so only the window differs; the per-layer spec
  // seam Gemma-4 landed can express that split, and it lands with the W1 forward.
  // Emitting one full-attention group here keeps the arch resolvable and is never
  // exercised, because the forward REFUSES-by-name.
  const MuseGlimmerParams p = ParseMuseGlimmerParams(config);
  const int num_kv_heads = static_cast<int>(p.text.num_key_value_heads);
  const int head_dim = static_cast<int>(p.text.head_dim);

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<v1::FullAttentionSpec>(block_size, num_kv_heads, head_dim,
                                              v1::ResolveKvCacheDType()));
  return kv;
}

REGISTER_VLLM_MODEL(muse_glimmer, "MuseGlimmerForCausalLM", kMuseGlimmerFactory,
                    kMuseGlimmerInfo)
REGISTER_VLLM_MODEL(muse_glimmer_mm, "MuseGlimmerForConditionalGeneration",
                    kMuseGlimmerFactory, kMuseGlimmerInfo)

}  // namespace vllm
