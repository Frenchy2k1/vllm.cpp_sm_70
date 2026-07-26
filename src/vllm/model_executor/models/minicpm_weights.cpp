// Weight loader for MiniCPM (`MiniCPMForCausalLM`, MiniCPM-2B-sft-bf16, BF16). Loads
// the checkpoint safetensors into the SHARED dense container (Qwen3DenseWeights,
// qwen3.h) via the SHARED dense_weight_loaders.h helpers. The name map is IDENTICAL
// to Llama (MiniCPM is Llama + scalar deltas, which live in config, not the
// weights): separate q/k/v merged to one qkv_proj, gate/up merged to one
// gate_up_proj, tied lm_head, NO qk-norm, NO biases.
//
// Grounding: vllm/model_executor/models/minicpm.py @ e24d1b24 —
//   - MiniCPMAttention: qkv_proj (QKVParallelLinear, bias=False) split [q,k,v]; NO
//     q_norm/k_norm; o_proj (RowParallelLinear, bias=False).
//   - MiniCPMMLP: merged gate_up_proj -> SiluAndMul -> down_proj (bias=False).
//   - packed_modules_mapping: q/k/v_proj -> qkv_proj, gate/up_proj -> gate_up_proj.
//   - tie_word_embeddings: AutoWeightsLoader skip_prefixes=["lm_head."]; the loader
//     skips the checkpoint lm_head.weight (MiniCPM-2B ties embeddings).
#include "vllm/model_executor/models/minicpm.h"

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

Qwen3DenseLayerWeights LoadMiniCPMLayer(const TensorResolver& get, int64_t layer) {
  const std::string base = "model.layers." + std::to_string(layer) + ".";
  const std::string sa = base + "self_attn.";
  const std::string mlp = base + "mlp.";

  Qwen3DenseLayerWeights w;
  w.input_layernorm = LoadBf16Direct(get, base + "input_layernorm.weight");
  w.post_attention_layernorm =
      LoadBf16Direct(get, base + "post_attention_layernorm.weight");

  // QKVParallelLinear: one merged owner in exact [q,k,v] output-row order, raw-NK.
  w.attn.qkv_proj = LoadMergedBf16RawNK(
      get, {sa + "q_proj.weight", sa + "k_proj.weight", sa + "v_proj.weight"});
  w.attn.o_proj = LoadMergedBf16RawNK(get, {sa + "o_proj.weight"});
  // MiniCPM has NO per-head q/k RMSNorm (q_norm/k_norm stay EMPTY) and NO biases.

  w.mlp.gate_up_proj = LoadMergedBf16RawNK(
      get, {mlp + "gate_proj.weight", mlp + "up_proj.weight"});
  w.mlp.down_proj = LoadMergedBf16RawNK(get, {mlp + "down_proj.weight"});
  return w;
}

}  // namespace

MiniCPMWeights LoadMiniCPMForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "minicpm: tensor not found: " + name);
    return it->second->Get(name);
  };

  VT_CHECK(config.num_hidden_layers > 0,
           "minicpm: num_hidden_layers must be positive");

  MiniCPMWeights w;
  // MiniCPM-2B ties embeddings (config has no tie_word_embeddings; the HF
  // MiniCPMConfig default and the AutoWeightsLoader skip_prefixes both tie). If the
  // checkpoint DOES carry a distinct lm_head.weight it is loaded (untied).
  const bool has_lm_head = where.count("lm_head.weight") > 0;
  w.tie_word_embeddings =
      RawBool(config.raw, "tie_word_embeddings", true) || !has_lm_head;

  w.embed_tokens = LoadBf16Direct(get, "model.embed_tokens.weight");
  w.final_norm = LoadBf16Direct(get, "model.norm.weight");
  if (!w.tie_word_embeddings) {
    w.lm_head = LoadBf16Transposed(get, "lm_head.weight");
  }

  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    w.layers.push_back(LoadMiniCPMLayer(get, l));
  return w;
}

}  // namespace vllm
