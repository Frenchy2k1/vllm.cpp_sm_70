// Weight loader for Command-R / Cohere (`CohereForCausalLM`, bf16). Merges the
// checkpoint's separate q/k/v -> qkv_proj and gate/up -> gate_up_proj
// (WeightsMapper orig_to_new_stacked, commandr.py:343-350), loads the weight-only
// Cohere LayerNorms (NO .bias), and precomputes the plain full-width GPT-J rope
// cos/sin cache. Embeddings are ALWAYS tied (commandr.py:372 asserts it) so lm_head
// is never loaded. NO biases anywhere.
//
// Reuses the shared dense_weight_loaders.h helpers (the exact BF16 copy/merge
// routines used by the Qwen3/OPT/stablelm loaders): LoadBf16Direct,
// LoadMergedBf16RawNK.
#include "vllm/model_executor/models/commandr.h"

#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "vllm/model_executor/layers/rotary_embedding/base.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

using dense_loaders::LoadBf16Direct;
using dense_loaders::LoadMergedBf16RawNK;
using dense_loaders::MakeOwned;

CommandrLayerWeights LoadCommandrLayer(const TensorResolver& get, int64_t layer) {
  const std::string base = "model.layers." + std::to_string(layer) + ".";
  const std::string sa = base + "self_attn.";
  const std::string mlp = base + "mlp.";

  CommandrLayerWeights w;
  // ONE weight-only LayerNorm per layer (no .bias, no post_attention_layernorm).
  w.input_layernorm = LoadBf16Direct(get, base + "input_layernorm.weight");

  // q/k/v ship SEPARATE (WeightsMapper merges them): concat into one raw-NK qkv.
  // NO biases (QKVParallelLinear/RowParallelLinear bias=False).
  w.attn.qkv_proj = LoadMergedBf16RawNK(
      get, {sa + "q_proj.weight", sa + "k_proj.weight", sa + "v_proj.weight"});
  w.attn.o_proj = LoadMergedBf16RawNK(get, {sa + "o_proj.weight"});

  // gate/up ship SEPARATE: concat in [gate, up] order, then down_proj.
  w.mlp.gate_up_proj =
      LoadMergedBf16RawNK(get, {mlp + "gate_proj.weight", mlp + "up_proj.weight"});
  w.mlp.down_proj = LoadMergedBf16RawNK(get, {mlp + "down_proj.weight"});
  return w;
}

// Build the plain full-width GPT-J rope cos/sin cache (bf16 [max_pos, head_dim],
// [cos|sin] halves, indexed by REAL position). Cohere uses full rotary
// (rotary_dim == head_dim) with rope_theta from config. The cos/sin CACHE is
// identical for NeoX vs GPT-J (only the apply differs) — the is_neox_style=False
// GPT-J rotation happens in the forward's RopeFromCache call. Mirrors
// get_rope(head_dim, ..., is_neox_style=False) at commandr.py:174-179.
OwnedTensor BuildCommandrRopeCache(const HfConfig& config) {
  const int64_t head_dim = config.head_dim;
  const int64_t max_pos = config.max_position_embeddings;
  const RopeParameters& rp = config.rope_parameters;
  VT_CHECK(head_dim > 0, "commandr: head_dim must be positive");
  VT_CHECK(max_pos > 0, "commandr: max_position_embeddings must be positive");
  VT_CHECK(rp.rope_type == "default",
           "commandr: only default rope is supported (got '" + rp.rope_type + "')");

  RotaryEmbedding rope(head_dim, /*rotary_dim=*/head_dim, max_pos, rp.rope_theta,
                       /*is_neox_style=*/false, vt::DType::kBF16);
  const vt::Tensor cache = rope.cos_sin_cache();  // bf16 [max_pos, head_dim]
  VT_CHECK(cache.rank == 2 && cache.shape[1] == head_dim,
           "commandr: rope cache shape mismatch");
  VT_CHECK(cache.shape[0] >= max_pos, "commandr: rope cache too short");

  OwnedTensor out = MakeOwned(vt::DType::kBF16, {max_pos, head_dim});
  std::memcpy(out.bytes.data(), cache.data, out.bytes.size());
  return out;
}

}  // namespace

CommandrWeights LoadCohereForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "commandr: tensor not found: " + name);
    return it->second->Get(name);
  };

  VT_CHECK(config.num_hidden_layers > 0,
           "commandr: num_hidden_layers must be positive");

  CommandrWeights w;
  w.logit_scale = CommandrLogitScale(config);

  w.embed_tokens = LoadBf16Direct(get, "model.embed_tokens.weight");
  w.final_norm = LoadBf16Direct(get, "model.norm.weight");  // weight-only

  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    w.layers.push_back(LoadCommandrLayer(get, l));

  w.rope_cos_sin = BuildCommandrRopeCache(config);
  return w;
}

}  // namespace vllm
