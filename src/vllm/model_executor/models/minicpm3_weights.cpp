// MiniCPM3 config resolution + BF16 safetensors weight loader. Header
// (include/vllm/model_executor/models/minicpm3.h) carries the full port map;
// this TU implements the config parse, the checkpoint name map, the load-time
// kv_b_proj -> W_UK/W_UV absorption (mla::AbsorbKvBProjBf16, shared with
// DeepSeek-V2), and the LongRoPE cos/sin cache.
//
// Name map (MiniCPM3-4B, plain `model.layers.` prefix):
//   model.embed_tokens.weight                            -> embed_tokens [V,H]
//   model.norm.weight                                    -> final_norm [H]
//   (tie_word_embeddings TRUE -> no lm_head.weight)
//   model.layers.N.input_layernorm.weight                -> input_layernorm [H]
//   model.layers.N.post_attention_layernorm.weight       -> post_attention_ln [H]
//   -- MLA attention (minicpm3.py:85-119) --
//   model.layers.N.self_attn.q_a_proj.weight       ]     -> fused_qkv_a_proj
//   model.layers.N.self_attn.kv_a_proj_with_mqa.weight ] (raw-NK, MERGED)
//   model.layers.N.self_attn.q_a_layernorm.weight        -> q_a_layernorm [ql]
//   model.layers.N.self_attn.q_b_proj.weight             -> q_b_proj (raw-NK)
//   model.layers.N.self_attn.kv_a_layernorm.weight       -> kv_a_layernorm [L]
//   model.layers.N.self_attn.kv_b_proj.weight            -> kv_b_proj (raw-NK)
//                                                        -> ALSO w_uk_t / w_uv
//   model.layers.N.self_attn.o_proj.weight               -> o_proj (raw-NK)
//   -- dense SwiGLU MLP (no MoE) --
//   model.layers.N.mlp.{gate,up}_proj.weight             -> mlp.gate_up (merged)
//   model.layers.N.mlp.down_proj.weight                  -> mlp.down (raw-NK)
#include "vllm/model_executor/models/minicpm3.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
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
using dense_loaders::MakeOwned;

// --- raw config.json readers (MiniCPM scalars are flat top-level fields) ---
const nlohmann::json* Field(const nlohmann::json& doc, const char* key) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return nullptr;
  return &(*it);
}
int64_t RawInt(const nlohmann::json& doc, const char* key, int64_t fallback) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_number()) ? f->get<int64_t>() : fallback;
}
double RawDouble(const nlohmann::json& doc, const char* key, double fallback) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_number()) ? f->get<double>() : fallback;
}

// The MLA `q_lora_rank is not None` loader branch (minicpm3.py always has q_lora).
// `fused_qkv_a_proj` FUSES q_a_proj + kv_a_proj_with_mqa row blocks, and the
// load-time absorption (mla::AbsorbKvBProjBf16, mla_attention.py:875-962) is done
// here — the identical transform DeepSeek-V2's loader runs.
MiniCPM3MlaWeights LoadMlaLayer(const TensorResolver& get, const std::string& sa,
                               const mla::MlaBlockDims& d) {
  MiniCPM3MlaWeights w;
  w.fused_qkv_a_proj = LoadMergedBf16RawNK(
      get, {sa + "q_a_proj.weight", sa + "kv_a_proj_with_mqa.weight"});
  w.q_a_layernorm = LoadBf16Direct(get, sa + "q_a_layernorm.weight");
  w.q_b_proj = LoadMergedBf16RawNK(get, {sa + "q_b_proj.weight"});
  w.kv_a_layernorm = LoadBf16Direct(get, sa + "kv_a_layernorm.weight");
  w.kv_b_proj = LoadMergedBf16RawNK(get, {sa + "kv_b_proj.weight"});
  w.o_proj = LoadMergedBf16RawNK(get, {sa + "o_proj.weight"});

  const int64_t N = d.num_heads, P = d.qk_nope_head_dim;
  const int64_t V = d.v_head_dim, L = d.kv_lora_rank;
  VT_CHECK(w.kv_b_proj.rank == 2 && w.kv_b_proj.shape[0] == N * (P + V) &&
               w.kv_b_proj.shape[1] == L,
           "minicpm3: kv_b_proj must be [num_heads*(qk_nope+v), kv_lora_rank]");
  const mla::AbsorbedKvBProj ab = mla::AbsorbKvBProjBf16(
      reinterpret_cast<const uint16_t*>(w.kv_b_proj.bytes.data()), d);
  w.w_uk_t = MakeOwned(vt::DType::kBF16, {N, P, L});
  std::memcpy(w.w_uk_t.bytes.data(), ab.w_uk_t.data(),
              ab.w_uk_t.size() * sizeof(uint16_t));
  w.w_uv = MakeOwned(vt::DType::kBF16, {N, L, V});
  std::memcpy(w.w_uv.bytes.data(), ab.w_uv.data(), ab.w_uv.size() * sizeof(uint16_t));
  return w;
}

Qwen3DenseMlpWeights LoadDenseMlp(const TensorResolver& get,
                                  const std::string& mlp) {
  Qwen3DenseMlpWeights m;
  m.gate_up_proj =
      LoadMergedBf16RawNK(get, {mlp + "gate_proj.weight", mlp + "up_proj.weight"});
  m.down_proj = LoadMergedBf16RawNK(get, {mlp + "down_proj.weight"});
  return m;
}

}  // namespace

MiniCPM3Params ParseMiniCPM3Params(const HfConfig& config) {
  const nlohmann::json& doc = config.raw;
  MiniCPM3Params p;
  p.hidden_size = config.hidden_size;
  p.num_hidden_layers = config.num_hidden_layers;
  p.vocab_size = config.vocab_size;
  p.intermediate_size = config.intermediate_size;
  p.rms_norm_eps = static_cast<float>(config.rms_norm_eps);
  // MiniCPM3Config defaults tie_word_embeddings TRUE (MiniCPM3-4B ships no
  // lm_head.weight); honor an explicit override if present.
  {
    const nlohmann::json* f = Field(doc, "tie_word_embeddings");
    p.tie_word_embeddings = (f != nullptr && f->is_boolean()) ? f->get<bool>() : true;
  }
  p.max_position_embeddings =
      config.max_position_embeddings > 0 ? config.max_position_embeddings : 32768;

  // The three MiniCPM scalar deltas (minicpm.py:588-640). dim_model_base defaults
  // to hidden_size (an identity scale_width) when absent, matching MiniCPM.
  p.scale_emb = RawDouble(doc, "scale_emb", 1.0);
  p.scale_depth = RawDouble(doc, "scale_depth", 1.0);
  p.dim_model_base =
      RawDouble(doc, "dim_model_base", static_cast<double>(p.hidden_size));

  // --- MLA geometry (minicpm3.py:52-134) ---
  mla::MlaBlockDims& d = p.mla;
  d.hidden_size = config.hidden_size;
  d.num_heads = config.num_attention_heads;
  d.qk_nope_head_dim = RawInt(doc, "qk_nope_head_dim", 0);
  d.qk_rope_head_dim = RawInt(doc, "qk_rope_head_dim", 0);
  // v_head_dim is a MiniCPM3Config default (64) when absent from config.json.
  d.v_head_dim = RawInt(doc, "v_head_dim", d.qk_nope_head_dim);
  d.kv_lora_rank = RawInt(doc, "kv_lora_rank", 0);
  d.q_lora_rank = RawInt(doc, "q_lora_rank", 0);
  d.rms_norm_eps = p.rms_norm_eps;
  // MiniCPM3 takes get_rope's DEFAULT is_neox_style=True (minicpm3.py:121-125).
  d.is_neox_style = true;
  // Plain qk_head_dim**-0.5 (minicpm3.py:82); LongRoPE contributes NO mscale
  // correction here (scale <= 1 -> mscale 1.0), so unlike DeepSeek there is no
  // mscale^2 factor.
  d.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(d.qk_head_dim())));
  if (d.q_lora_rank <= 0) {
    throw std::runtime_error(
        "MiniCPM3ForCausalLM config: q_lora_rank must be > 0 (MiniCPM3 always uses "
        "the low-rank query path, minicpm3.py:85-95)");
  }
  if (d.qk_nope_head_dim <= 0 || d.qk_rope_head_dim <= 0 || d.v_head_dim <= 0 ||
      d.kv_lora_rank <= 0) {
    throw std::runtime_error(
        "MiniCPM3ForCausalLM config: qk_nope_head_dim, qk_rope_head_dim, "
        "v_head_dim and kv_lora_rank must all be > 0");
  }
  d.Validate();

  // --- LongRoPE (phi3_long_rope_scaled_rope.py) ---
  const RopeParameters& rp = config.rope_parameters;
  if (rp.rope_type != "longrope") {
    throw std::runtime_error(
        "MiniCPM3ForCausalLM config: expected rope_scaling type 'longrope' "
        "(minicpm3.py:121-125 over the shipped rope_scaling); got '" +
        rp.rope_type + "'");
  }
  p.rope_base = config.rope_theta;
  p.rope_original_max_position_embeddings =
      rp.original_max_position_embeddings.value_or(p.max_position_embeddings);
  // `scale = max_position_embeddings / original_max_position_embeddings`; the
  // cos/sin mscale is 1.0 when scale <= 1 (phi3_long_rope:67-77). MiniCPM3-4B has
  // max_pos == original (32768) -> mscale 1.0 and use_long_rope FALSE (the SHORT
  // cache indexed directly by position).
  const double scale = static_cast<double>(p.max_position_embeddings) /
                       static_cast<double>(p.rope_original_max_position_embeddings);
  if (scale > 1.0) {
    // The long regime needs the offset-indexed LONG cache; the 4B vehicle never
    // hits it. Refuse loudly rather than silently mis-index (follow-up: add the
    // use_long_rope offset path + short/long cache selection).
    throw std::runtime_error(
        "MiniCPM3ForCausalLM config: max_position_embeddings > "
        "original_max_position_embeddings selects the LongRoPE long-cache regime "
        "(phi3_long_rope:54-90), which is not implemented for this vehicle");
  }
  p.rope_mscale = 1.0;
  // use_long_rope FALSE -> the SHORT cache (short_factor), rows ==
  // original_max_position_embeddings, indexed by position.
  p.rope_factor = rp.short_factor;
  p.rope_cache_rows = p.rope_original_max_position_embeddings;
  const int64_t half = d.qk_rope_head_dim / 2;
  if (static_cast<int64_t>(p.rope_factor.size()) != half) {
    throw std::runtime_error(
        "MiniCPM3ForCausalLM config: LongRoPE short_factor length (" +
        std::to_string(p.rope_factor.size()) + ") must equal qk_rope_head_dim/2 (" +
        std::to_string(half) + ")");
  }
  return p;
}

// phi3_long_rope_scaled_rope.py:97-123 (short cache, mscale 1.0). NEOX cache:
// [cos(rot/2) | sin(rot/2)] per row, exactly what vt::RopeFromCache reads.
std::vector<float> BuildMiniCPM3RopeCosSinCache(const MiniCPM3Params& p) {
  const int64_t rot = p.mla.qk_rope_head_dim;
  const int64_t half = rot / 2;
  const int64_t rows = p.rope_cache_rows;
  std::vector<double> inv_freq(static_cast<size_t>(half));
  for (int64_t i = 0; i < half; ++i) {
    const double denom = p.rope_factor[static_cast<size_t>(i)] *
                         std::pow(p.rope_base,
                                  static_cast<double>(2 * i) / static_cast<double>(rot));
    inv_freq[static_cast<size_t>(i)] = 1.0 / denom;
  }
  std::vector<float> cache(static_cast<size_t>(rows) * static_cast<size_t>(rot));
  for (int64_t pos = 0; pos < rows; ++pos) {
    float* row = cache.data() + static_cast<size_t>(pos) * static_cast<size_t>(rot);
    for (int64_t i = 0; i < half; ++i) {
      const double ang = static_cast<double>(pos) * inv_freq[static_cast<size_t>(i)];
      row[static_cast<size_t>(i)] =
          static_cast<float>(std::cos(ang) * p.rope_mscale);
      row[static_cast<size_t>(half + i)] =
          static_cast<float>(std::sin(ang) * p.rope_mscale);
    }
  }
  return cache;
}

MiniCPM3Weights LoadMiniCPM3ForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "minicpm3: tensor not found: " + name);
    return it->second->Get(name);
  };

  MiniCPM3Weights w;
  w.params = ParseMiniCPM3Params(config);
  const MiniCPM3Params& p = w.params;
  VT_CHECK(p.num_hidden_layers > 0, "minicpm3: num_hidden_layers must be positive");

  w.embed_tokens = LoadBf16Direct(get, "model.embed_tokens.weight");
  w.final_norm = LoadBf16Direct(get, "model.norm.weight");
  if (!p.tie_word_embeddings)
    w.lm_head = LoadBf16Transposed(get, "lm_head.weight");

  // ONE shared LongRoPE [cos|sin] cache for every layer, in bf16 (the forward
  // dtype), so vt::RopeFromCache reads it directly.
  {
    const int64_t rot = p.mla.qk_rope_head_dim;
    const std::vector<float> cache = BuildMiniCPM3RopeCosSinCache(p);
    w.rope_cos_sin_cache = MakeOwned(vt::DType::kBF16, {p.rope_cache_rows, rot});
    auto* dst = reinterpret_cast<uint16_t*>(w.rope_cos_sin_cache.bytes.data());
    for (size_t i = 0; i < cache.size(); ++i) dst[i] = vt::F32ToBF16(cache[i]);
  }

  w.layers.reserve(static_cast<size_t>(p.num_hidden_layers));
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const std::string base = "model.layers." + std::to_string(l) + ".";
    MiniCPM3LayerWeights lw;
    lw.input_layernorm = LoadBf16Direct(get, base + "input_layernorm.weight");
    lw.post_attention_layernorm =
        LoadBf16Direct(get, base + "post_attention_layernorm.weight");
    lw.attn = LoadMlaLayer(get, base + "self_attn.", p.mla);
    lw.mlp = LoadDenseMlp(get, base + "mlp.");
    w.layers.push_back(std::move(lw));
  }
  return w;
}

}  // namespace vllm
