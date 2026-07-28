// Weight loader for the `Gemma4ForConditionalGeneration` text backbone
// (unsloth/gemma-4-E4B-it, BF16) — MODEL-GEMMA4 G1. Loads the language_model
// stack into Gemma4Weights (gemma4.h) via the shared dense_weight_loaders.h
// helpers, VERIFIED against the checkpoint's safetensors header (2130 tensors;
// the 336 language_model.* tensors mapped below, mm towers skipped).
//
// Grounding: vllm/model_executor/models/gemma4.py — load_weights (:1625-1728)
// strips the `language_model.` prefix (:1644); packed_modules_mapping
// (:1536-1546) qkv_proj<-[q,k,v]_proj, gate_up_proj<-[gate,up]_proj;
// tie_word_embeddings (:1566-1567) — lm_head aliases embed_tokens, checkpoint has
// no lm_head.weight; skip mm weights audio_tower/vision_tower/embed_audio/
// embed_vision (:1716-1723).
//
// Name map (unsloth/gemma-4-E4B-it, wrapper prefix `model.language_model.`):
//   model.language_model.embed_tokens.weight              -> embed_tokens [V,H]
//   model.language_model.embed_tokens_per_layer.weight    -> embed_tokens_per_layer [V, ple*L]
//   model.language_model.per_layer_model_projection.weight-> per_layer_model_projection [ple*L, H]
//   model.language_model.per_layer_projection_norm.weight -> per_layer_projection_norm [ple]
//   model.language_model.norm.weight                      -> final_norm [H]
//   ...layers.N.input_layernorm.weight                    -> input_layernorm [H]
//   ...layers.N.post_attention_layernorm.weight           -> post_attention_ln [H]
//   ...layers.N.pre_feedforward_layernorm.weight          -> pre_feedforward_ln [H]
//   ...layers.N.post_feedforward_layernorm.weight         -> post_feedforward_ln [H]
//   ...layers.N.post_per_layer_input_norm.weight          -> post_per_layer_input_norm [H]
//   ...layers.N.per_layer_input_gate.weight               -> per_layer_input_gate [ple,H]
//   ...layers.N.per_layer_projection.weight               -> per_layer_projection [H,ple]
//   ...layers.N.layer_scalar                              -> layer_scalar [1]
//   ...layers.N.self_attn.{q,k,v}_proj.weight             -> merged qkv_proj (raw-NK)
//   ...layers.N.self_attn.o_proj.weight                   -> o_proj (raw-NK)
//   ...layers.N.self_attn.{q,k}_norm.weight               -> q_norm/k_norm [Dh]
//   ...layers.N.mlp.{gate,up}_proj.weight                 -> merged gate_up_proj (raw-NK)
//   ...layers.N.mlp.down_proj.weight                      -> down_proj (raw-NK)
#include "vllm/model_executor/models/gemma4.h"

#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

using dense_loaders::LoadBf16Direct;
using dense_loaders::LoadBf16Transposed;
using dense_loaders::LoadMergedBf16RawNK;

bool RawBool(const nlohmann::json& doc, const char* key, bool fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_boolean()) return fallback;
  return it->get<bool>();
}
int64_t RawInt(const nlohmann::json& doc, const char* key, int64_t fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_number_integer()) return fallback;
  return it->get<int64_t>();
}

// Per-layer attention topology derived from the config (gemma4.py:441-489,
// :572-593). full_attention layers use global_head_dim; the last
// num_kv_shared_layers reuse the last non-shared layer of the same type's K/V.
struct LayerTopo {
  std::vector<bool> is_full;
  std::vector<bool> is_shared;
  std::vector<int64_t> head_dim;
  std::vector<int64_t> kv_target;  // -1 for self, else source layer index
};

LayerTopo MakeTopo(const HfConfig& cfg) {
  const int64_t L = cfg.num_hidden_layers;
  const int64_t head_dim_sliding = cfg.head_dim;  // text_config head_dim (256)
  const int64_t head_dim_full =
      RawInt(cfg.raw, "global_head_dim", cfg.head_dim);  // 512
  const int64_t num_shared = RawInt(cfg.raw, "num_kv_shared_layers", 0);
  const int64_t first_shared = L - num_shared;

  LayerTopo t;
  t.is_full.assign(static_cast<size_t>(L), false);
  const auto it = cfg.raw.find("layer_types");
  if (it != cfg.raw.end() && it->is_array()) {
    for (int64_t l = 0; l < L && static_cast<size_t>(l) < it->size(); ++l)
      t.is_full[static_cast<size_t>(l)] =
          it->at(static_cast<size_t>(l)).is_string() &&
          it->at(static_cast<size_t>(l)).get<std::string>() == "full_attention";
  }
  t.is_shared.assign(static_cast<size_t>(L), false);
  t.head_dim.assign(static_cast<size_t>(L), head_dim_sliding);
  t.kv_target.assign(static_cast<size_t>(L), -1);
  for (int64_t l = 0; l < L; ++l) {
    const bool full = t.is_full[static_cast<size_t>(l)];
    t.head_dim[static_cast<size_t>(l)] = full ? head_dim_full : head_dim_sliding;
    if (num_shared > 0 && l >= first_shared) {
      t.is_shared[static_cast<size_t>(l)] = true;
      // Last non-shared layer (< first_shared) of the same attention type.
      int64_t target = -1;
      for (int64_t p = first_shared - 1; p >= 0; --p) {
        if (t.is_full[static_cast<size_t>(p)] == full) {
          target = p;
          break;
        }
      }
      t.kv_target[static_cast<size_t>(l)] = target;
    }
  }
  return t;
}

Gemma4LayerWeights LoadGemma4Layer(const TensorResolver& get, int64_t layer,
                                   const LayerTopo& topo) {
  const std::string base =
      "model.language_model.layers." + std::to_string(layer) + ".";
  const std::string sa = base + "self_attn.";
  const std::string mlp = base + "mlp.";

  Gemma4LayerWeights w;
  w.is_full_attention = topo.is_full[static_cast<size_t>(layer)];
  w.is_kv_shared = topo.is_shared[static_cast<size_t>(layer)];
  w.head_dim = topo.head_dim[static_cast<size_t>(layer)];
  w.kv_target_layer = topo.kv_target[static_cast<size_t>(layer)];

  // Four PLAIN RMSNorm weights (gemma4.py:632-641).
  w.input_layernorm = LoadBf16Direct(get, base + "input_layernorm.weight");
  w.post_attention_layernorm =
      LoadBf16Direct(get, base + "post_attention_layernorm.weight");
  w.pre_feedforward_layernorm =
      LoadBf16Direct(get, base + "pre_feedforward_layernorm.weight");
  w.post_feedforward_layernorm =
      LoadBf16Direct(get, base + "post_feedforward_layernorm.weight");

  // PLE per-layer components (gemma4.py:680-707).
  w.per_layer_input_gate =
      LoadMergedBf16RawNK(get, {base + "per_layer_input_gate.weight"});
  w.per_layer_projection =
      LoadMergedBf16RawNK(get, {base + "per_layer_projection.weight"});
  w.post_per_layer_input_norm =
      LoadBf16Direct(get, base + "post_per_layer_input_norm.weight");
  w.layer_scalar = LoadBf16Direct(get, base + "layer_scalar");

  // QKVParallelLinear merged (q,k,v output-row order), o_proj. No bias
  // (attention_bias=false). Shared layers still carry q/k/v_proj in the
  // checkpoint (verified); the forward reuses the target layer's K/V and
  // discards this layer's K/V, but we load them for a complete, verifiable map.
  w.attn.qkv_proj = LoadMergedBf16RawNK(
      get, {sa + "q_proj.weight", sa + "k_proj.weight", sa + "v_proj.weight"});
  w.attn.o_proj = LoadMergedBf16RawNK(get, {sa + "o_proj.weight"});
  // Per-head PLAIN Q/K RMSNorm (gemma4.py:434-435). V-norm is weight-less.
  w.attn.q_norm = LoadBf16Direct(get, sa + "q_norm.weight");
  w.attn.k_norm = LoadBf16Direct(get, sa + "k_norm.weight");

  // GeGLU MLP (gemma4.py:234-247): merged gate_up (gate,up), then down.
  w.mlp.gate_up_proj =
      LoadMergedBf16RawNK(get, {mlp + "gate_proj.weight", mlp + "up_proj.weight"});
  w.mlp.down_proj = LoadMergedBf16RawNK(get, {mlp + "down_proj.weight"});
  return w;
}

}  // namespace

Gemma4Weights LoadGemma4ForConditionalGenerationWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "gemma4: tensor not found: " + name);
    return it->second->Get(name);
  };

  VT_CHECK(config.num_hidden_layers > 0,
           "gemma4: num_hidden_layers must be positive");

  const LayerTopo topo = MakeTopo(config);

  Gemma4Weights w;
  // Gemma-4 ties embeddings by default (text_config.tie_word_embeddings=true).
  w.tie_word_embeddings = RawBool(config.raw, "tie_word_embeddings", true);

  w.embed_tokens = LoadBf16Direct(get, "model.language_model.embed_tokens.weight");
  w.embed_tokens_per_layer =
      LoadBf16Direct(get, "model.language_model.embed_tokens_per_layer.weight");
  w.per_layer_model_projection = LoadMergedBf16RawNK(
      get, {"model.language_model.per_layer_model_projection.weight"});
  w.per_layer_projection_norm =
      LoadBf16Direct(get, "model.language_model.per_layer_projection_norm.weight");
  w.final_norm = LoadBf16Direct(get, "model.language_model.norm.weight");
  if (!w.tie_word_embeddings) {
    w.lm_head = LoadBf16Transposed(get, "lm_head.weight");
  }

  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    w.layers.push_back(LoadGemma4Layer(get, l, topo));
  return w;
}

}  // namespace vllm
