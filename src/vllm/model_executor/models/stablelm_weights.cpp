// Weight loader for StableLM (`StableLmForCausalLM`, stabilityai/stablelm-2-1_6b,
// BF16). Merges the checkpoint's separate q/k/v -> qkv_proj and gate/up ->
// gate_up_proj (WeightsMapper orig_to_new_stacked, stablelm.py:279-289), loads the
// nn.LayerNorm weight+bias pairs, the optional merged qkv bias (`use_qkv_bias`),
// the untied lm_head, and precomputes the plain partial-rope cos/sin cache.
//
// Reuses the shared dense_weight_loaders.h helpers (the exact BF16 copy/merge
// routines used by the Qwen3/OPT/phi3 loaders): LoadBf16Direct,
// LoadBf16Transposed, LoadMergedBf16RawNK, LoadMergedBf16Vector.
#include "vllm/model_executor/models/stablelm.h"

#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/layers/rotary_embedding/base.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

using dense_loaders::LoadBf16Direct;
using dense_loaders::LoadBf16Transposed;
using dense_loaders::LoadMergedBf16RawNK;
using dense_loaders::LoadMergedBf16Vector;
using dense_loaders::MakeOwned;

bool RawBool(const nlohmann::json& doc, const char* key, bool fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_boolean()) return fallback;
  return it->get<bool>();
}

StablelmLayerWeights LoadStablelmLayer(const TensorResolver& get, int64_t layer,
                                       bool use_qkv_bias) {
  const std::string base = "model.layers." + std::to_string(layer) + ".";
  const std::string sa = base + "self_attn.";
  const std::string mlp = base + "mlp.";

  StablelmLayerWeights w;
  w.input_layernorm = LoadBf16Direct(get, base + "input_layernorm.weight");
  w.input_layernorm_bias = LoadBf16Direct(get, base + "input_layernorm.bias");
  w.post_attention_layernorm =
      LoadBf16Direct(get, base + "post_attention_layernorm.weight");
  w.post_attention_layernorm_bias =
      LoadBf16Direct(get, base + "post_attention_layernorm.bias");

  // q/k/v ship SEPARATE (WeightsMapper merges them): concat into one raw-NK qkv.
  w.attn.qkv_proj = LoadMergedBf16RawNK(
      get, {sa + "q_proj.weight", sa + "k_proj.weight", sa + "v_proj.weight"});
  if (use_qkv_bias) {
    w.attn.qkv_bias = LoadMergedBf16Vector(
        get, {sa + "q_proj.bias", sa + "k_proj.bias", sa + "v_proj.bias"});
  }
  w.attn.o_proj = LoadMergedBf16RawNK(get, {sa + "o_proj.weight"});

  // gate/up ship SEPARATE: concat in [gate, up] order, then down_proj.
  w.mlp.gate_up_proj =
      LoadMergedBf16RawNK(get, {mlp + "gate_proj.weight", mlp + "up_proj.weight"});
  w.mlp.down_proj = LoadMergedBf16RawNK(get, {mlp + "down_proj.weight"});
  return w;
}

// Build the plain partial-rope cos/sin cache (bf16 [max_pos, rotary_dim], [cos|sin]
// halves, indexed by REAL position). stablelm-2 uses rope_type "default" with
// partial_rotary_factor 0.25 -> rotary_dim = int(head_dim*0.25). Mirrors phi3's
// default branch (RotaryEmbedding, is_neox_style=true).
OwnedTensor BuildStablelmRopeCache(const HfConfig& config) {
  const int64_t head_dim = config.head_dim;
  const int64_t rotary_dim = config.rotary_dim;
  const int64_t max_pos = config.max_position_embeddings;
  const RopeParameters& rp = config.rope_parameters;
  VT_CHECK(rotary_dim > 0 && rotary_dim <= head_dim,
           "stablelm: rotary_dim must be in (0, head_dim]");
  VT_CHECK(max_pos > 0, "stablelm: max_position_embeddings must be positive");
  VT_CHECK(rp.rope_type == "default",
           "stablelm: only default rope is supported (got '" + rp.rope_type + "')");

  RotaryEmbedding rope(head_dim, rotary_dim, max_pos, rp.rope_theta,
                       /*is_neox_style=*/true, vt::DType::kBF16);
  const vt::Tensor cache = rope.cos_sin_cache();  // bf16 [max_pos, rotary_dim]
  VT_CHECK(cache.rank == 2 && cache.shape[1] == rotary_dim,
           "stablelm: rope cache shape mismatch");
  VT_CHECK(cache.shape[0] >= max_pos, "stablelm: rope cache too short");

  OwnedTensor out = MakeOwned(vt::DType::kBF16, {max_pos, rotary_dim});
  std::memcpy(out.bytes.data(), cache.data, out.bytes.size());
  return out;
}

}  // namespace

StablelmWeights LoadStableLmForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "stablelm: tensor not found: " + name);
    return it->second->Get(name);
  };

  VT_CHECK(config.num_hidden_layers > 0,
           "stablelm: num_hidden_layers must be positive");

  StablelmWeights w;
  w.tie_word_embeddings = RawBool(config.raw, "tie_word_embeddings", false);
  w.use_qkv_bias = StablelmUseQkvBias(config);

  w.embed_tokens = LoadBf16Direct(get, "model.embed_tokens.weight");
  w.final_norm = LoadBf16Direct(get, "model.norm.weight");
  w.final_norm_bias = LoadBf16Direct(get, "model.norm.bias");
  if (!w.tie_word_embeddings) {
    w.lm_head = LoadBf16Transposed(get, "lm_head.weight");
  }

  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    w.layers.push_back(LoadStablelmLayer(get, l, w.use_qkv_bias));

  w.rope_cos_sin = BuildStablelmRopeCache(config);
  return w;
}

}  // namespace vllm
