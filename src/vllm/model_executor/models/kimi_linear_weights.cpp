// Kimi-Linear config resolution + checkpoint weight name-map + loader (W1). This
// TU implements `ParseKimiLinearParams` (standalone config descent over the typed
// HfConfig + `config.raw`, unit-testable), the pure `EnumerateKimiLinearTensors`
// name-map (grounded 1:1 in the pinned `kimi_linear.py` + `kimi_gdn_linear_attn.py`
// AND verified vs the shipped safetensors index), and
// `LoadKimiLinearForCausalLMWeights` which validates coverage + shapes and throws
// BY NAME (never silent zeros) on the first missing/mis-shaped tensor.
//
// ─── CHECKPOINT NAME MAP (pinned kimi_linear.py, VERIFIED vs the HF index) ──────
//   MODEL LEVEL      model.embed_tokens.weight, model.norm.weight,
//                    lm_head.weight (absent iff tie_word_embeddings)
//   PER LAYER N      model.layers.N.input_layernorm.weight,
//                    .post_attention_layernorm.weight
//     KDA layer (is_kda_layer, kimi_gdn_linear_attn.py:120-226):
//       .self_attn.{q_proj,k_proj,v_proj,f_a_proj,f_b_proj,b_proj,g_a_proj,
//                   g_b_proj,o_proj,q_conv1d,k_conv1d,v_conv1d}.weight
//                   + .self_attn.{dt_bias,A_log,o_norm.weight}
//     MLA layer (KimiMLAAttention, kimi_linear.py:217-248; q_lora_rank==null =>
//                direct q_proj, mla_use_nope => rotary_emb=None):
//       .self_attn.{q_proj,kv_a_proj_with_mqa,kv_a_layernorm,kv_b_proj,o_proj}.weight
//     MLP dispatch (kimi_linear.py:328-347):
//       MoE layer:  .block_sparse_moe.gate.weight,
//                   .block_sparse_moe.gate.e_score_correction_bias,
//                   .block_sparse_moe.shared_experts.{gate,up,down}_proj.weight,
//                   .block_sparse_moe.experts.E.{w1,w2,w3}.weight  (E in [0,num_experts))
//       dense layer (first_k_dense_replace): .mlp.{gate,up,down}_proj.weight
//
// NOTE the MoE module is `block_sparse_moe` (kimi_linear.py:334-339 registers it
// first; `self.mlp = self.block_sparse_moe` is only an alias, deduplicated away by
// named_parameters), NOT `mlp` — this is the concrete correction vs the DERIVED K3
// `EnumerateKimiK3TextBackboneTensors`, confirmed against the real index.
#include "vllm/model_executor/models/kimi_linear.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"  // dense_loaders::LoadBf16Direct/MakeOwned
#include "vllm/platforms/interface.h"                          // platforms::GetPlatform (is_cpu)
#include "vt/backend.h"                                        // vt::GetBackend (device staging)
#include "vt/dtype.h"

namespace vllm {
namespace {

// --- raw json readers over the top-level standalone config doc ---
const nlohmann::json* Field(const nlohmann::json& doc, const char* key) {
  if (!doc.is_object()) return nullptr;
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) return nullptr;
  return &(*it);
}
const nlohmann::json* Object(const nlohmann::json& doc, const char* key) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_object()) ? f : nullptr;
}
int64_t RawInt(const nlohmann::json& doc, const char* key, int64_t fallback) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_number()) ? f->get<int64_t>() : fallback;
}
double RawDouble(const nlohmann::json& doc, const char* key, double fallback) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_number()) ? f->get<double>() : fallback;
}
bool RawBool(const nlohmann::json& doc, const char* key, bool fallback) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_boolean()) ? f->get<bool>() : fallback;
}
std::string RawString(const nlohmann::json& doc, const char* key,
                      const std::string& fallback) {
  const nlohmann::json* f = Field(doc, key);
  return (f != nullptr && f->is_string()) ? f->get<std::string>() : fallback;
}
std::vector<int64_t> RawIntArray(const nlohmann::json& doc, const char* key) {
  std::vector<int64_t> out;
  const nlohmann::json* f = Field(doc, key);
  if (f != nullptr && f->is_array())
    for (const auto& v : *f)
      if (v.is_number()) out.push_back(v.get<int64_t>());
  return out;
}

// One enumerated checkpoint tensor: its name and (optionally) its expected shape.
// An EMPTY shape means "presence-only" — the tensors whose stored rank/layout the
// config does not unambiguously pin (the fp32 short convs `*_conv1d.weight`, whose
// Conv1d weight is unsqueezed at load, kimi_gdn_linear_attn.py:196-198, and the
// per-head `A_log`, stored broadcast — kimi_gdn_linear_attn.py:200-203).
struct TensorSpec {
  std::string name;
  std::vector<int64_t> shape;  // empty => presence-only
};

std::vector<TensorSpec> EnumerateSpecs(const KimiLinearParams& p) {
  std::vector<TensorSpec> t;
  const int64_t H = p.hidden_size;
  const int64_t kda_proj = p.kda_num_heads * p.kda_head_dim;
  const int64_t mla_qk_head = p.qk_nope_head_dim + p.qk_rope_head_dim;
  const int64_t shared_inter = p.moe_intermediate_size * p.num_shared_experts;

  // --- model level ---
  t.push_back({"model.embed_tokens.weight", {p.vocab_size, H}});
  t.push_back({"model.norm.weight", {H}});
  if (!p.tie_word_embeddings)
    t.push_back({"lm_head.weight", {p.vocab_size, H}});

  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const std::string b = "model.layers." + std::to_string(l) + ".";
    t.push_back({b + "input_layernorm.weight", {H}});
    t.push_back({b + "post_attention_layernorm.weight", {H}});

    const std::string a = b + "self_attn.";
    if (p.is_kda_layer(l)) {
      // KDA (kimi_gdn_linear_attn.py:120-226).
      t.push_back({a + "q_proj.weight", {kda_proj, H}});
      t.push_back({a + "k_proj.weight", {kda_proj, H}});
      t.push_back({a + "v_proj.weight", {kda_proj, H}});
      t.push_back({a + "f_a_proj.weight", {p.kda_head_dim, H}});
      t.push_back({a + "f_b_proj.weight", {kda_proj, p.kda_head_dim}});
      t.push_back({a + "b_proj.weight", {p.kda_num_heads, H}});
      t.push_back({a + "g_a_proj.weight", {p.kda_head_dim, H}});
      t.push_back({a + "g_b_proj.weight", {kda_proj, p.kda_head_dim}});
      t.push_back({a + "o_proj.weight", {H, kda_proj}});
      t.push_back({a + "q_conv1d.weight", {}});  // fp32, unsqueezed at load
      t.push_back({a + "k_conv1d.weight", {}});
      t.push_back({a + "v_conv1d.weight", {}});
      t.push_back({a + "dt_bias", {kda_proj}});
      t.push_back({a + "A_log", {}});  // per-head, broadcast layout
      t.push_back({a + "o_norm.weight", {p.kda_head_dim}});
    } else {
      // NoPE MLA (KimiMLAAttention, kimi_linear.py:217-248). q_lora_rank==null =>
      // direct q_proj; no q_a_proj/q_a_layernorm/q_b_proj branch.
      t.push_back(
          {a + "q_proj.weight", {p.num_attention_heads * mla_qk_head, H}});
      t.push_back({a + "kv_a_proj_with_mqa.weight",
                   {p.kv_lora_rank + p.qk_rope_head_dim, H}});
      t.push_back({a + "kv_a_layernorm.weight", {p.kv_lora_rank}});
      t.push_back(
          {a + "kv_b_proj.weight",
           {p.num_attention_heads * (p.qk_nope_head_dim + p.v_head_dim),
            p.kv_lora_rank}});
      t.push_back(
          {a + "o_proj.weight", {H, p.num_attention_heads * p.v_head_dim}});
    }

    if (p.is_moe_layer(l)) {
      const std::string m = b + "block_sparse_moe.";
      t.push_back({m + "gate.weight", {p.num_experts, H}});
      t.push_back({m + "gate.e_score_correction_bias", {p.num_experts}});
      if (p.num_shared_experts > 0) {
        t.push_back(
            {m + "shared_experts.gate_proj.weight", {shared_inter, H}});
        t.push_back({m + "shared_experts.up_proj.weight", {shared_inter, H}});
        t.push_back(
            {m + "shared_experts.down_proj.weight", {H, shared_inter}});
      }
      for (int64_t e = 0; e < p.num_experts; ++e) {
        const std::string ep = m + "experts." + std::to_string(e) + ".";
        t.push_back({ep + "w1.weight", {p.moe_intermediate_size, H}});
        t.push_back({ep + "w2.weight", {H, p.moe_intermediate_size}});
        t.push_back({ep + "w3.weight", {p.moe_intermediate_size, H}});
      }
    } else {
      const std::string m = b + "mlp.";
      t.push_back({m + "gate_proj.weight", {p.intermediate_size, H}});
      t.push_back({m + "up_proj.weight", {p.intermediate_size, H}});
      t.push_back({m + "down_proj.weight", {H, p.intermediate_size}});
    }
  }
  return t;
}

std::string ShapeStr(const std::vector<int64_t>& s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); ++i) {
    if (i != 0) out += ",";
    out += std::to_string(s[i]);
  }
  return out + "]";
}

// ─── HOST (float) MATERIALIZATION — the CPU reference-forward weights (W2) ──────
using HaveMap = std::unordered_map<std::string, const StTensor*>;

int64_t NumEl(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

// Decode one enumerated checkpoint tensor to row-major f32. Handles the two dtypes
// the shipped Kimi-Linear checkpoint uses (BF16 weights, F32 conv/dt_bias/A_log).
std::vector<float> ReadFloat(const HaveMap& have, const std::string& name) {
  const auto it = have.find(name);
  VT_CHECK(it != have.end(),
           "kimi-linear materialize: missing tensor '" + name + "'");
  const StTensor& t = *it->second;
  const int64_t n = NumEl(t.shape);
  std::vector<float> out(static_cast<size_t>(n), 0.0f);
  if (t.dtype == "F32") {
    std::memcpy(out.data(), t.data, static_cast<size_t>(n) * 4);
  } else if (t.dtype == "BF16") {
    const uint8_t* p = t.data;
    for (int64_t i = 0; i < n; ++i) {
      uint16_t bits = static_cast<uint16_t>(p[i * 2]) |
                      (static_cast<uint16_t>(p[i * 2 + 1]) << 8);
      uint32_t u = static_cast<uint32_t>(bits) << 16;
      std::memcpy(&out[static_cast<size_t>(i)], &u, 4);
    }
  } else {
    VT_CHECK(false, "kimi-linear materialize: tensor '" + name + "' has dtype '" +
                        t.dtype + "' (only BF16/F32 supported in the CPU reference)");
  }
  return out;
}

KimiLinearHostWeights MaterializeHost(const KimiLinearParams& p,
                                      const HaveMap& have) {
  KimiLinearHostWeights h;
  h.embed_tokens = ReadFloat(have, "model.embed_tokens.weight");
  h.final_norm = ReadFloat(have, "model.norm.weight");
  h.lm_head = p.tie_word_embeddings ? h.embed_tokens
                                    : ReadFloat(have, "lm_head.weight");

  h.layers.resize(static_cast<size_t>(p.num_hidden_layers));
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    KimiLinearLayerHostWeights& lw = h.layers[static_cast<size_t>(l)];
    const std::string b = "model.layers." + std::to_string(l) + ".";
    lw.input_layernorm = ReadFloat(have, b + "input_layernorm.weight");
    lw.post_attention_layernorm = ReadFloat(have, b + "post_attention_layernorm.weight");

    const std::string a = b + "self_attn.";
    lw.is_kda = p.is_kda_layer(l);
    if (lw.is_kda) {
      KdaLayerHostWeights& k = lw.kda;
      k.q_proj = ReadFloat(have, a + "q_proj.weight");
      k.k_proj = ReadFloat(have, a + "k_proj.weight");
      k.v_proj = ReadFloat(have, a + "v_proj.weight");
      k.f_a_proj = ReadFloat(have, a + "f_a_proj.weight");
      k.f_b_proj = ReadFloat(have, a + "f_b_proj.weight");
      k.b_proj = ReadFloat(have, a + "b_proj.weight");
      k.g_a_proj = ReadFloat(have, a + "g_a_proj.weight");
      k.g_b_proj = ReadFloat(have, a + "g_b_proj.weight");
      k.o_proj = ReadFloat(have, a + "o_proj.weight");
      // conv1d.weight is stored [proj,1,K]; the middle 1 flattens away to [proj,K].
      k.q_conv = ReadFloat(have, a + "q_conv1d.weight");
      k.k_conv = ReadFloat(have, a + "k_conv1d.weight");
      k.v_conv = ReadFloat(have, a + "v_conv1d.weight");
      k.dt_bias = ReadFloat(have, a + "dt_bias");
      k.a_log = ReadFloat(have, a + "A_log");
      k.o_norm = ReadFloat(have, a + "o_norm.weight");
    } else {
      MlaLayerHostWeights& m = lw.mla;
      m.q_proj = ReadFloat(have, a + "q_proj.weight");
      m.kv_a_proj_with_mqa = ReadFloat(have, a + "kv_a_proj_with_mqa.weight");
      m.kv_a_layernorm = ReadFloat(have, a + "kv_a_layernorm.weight");
      m.kv_b_proj = ReadFloat(have, a + "kv_b_proj.weight");
      m.o_proj = ReadFloat(have, a + "o_proj.weight");
    }

    lw.is_moe = p.is_moe_layer(l);
    if (lw.is_moe) {
      MoeHostWeights& mo = lw.moe;
      const std::string mp = b + "block_sparse_moe.";
      mo.gate = ReadFloat(have, mp + "gate.weight");
      mo.e_score_correction_bias = ReadFloat(have, mp + "gate.e_score_correction_bias");
      mo.has_shared = p.num_shared_experts > 0;
      if (mo.has_shared) {
        mo.shared.gate_proj = ReadFloat(have, mp + "shared_experts.gate_proj.weight");
        mo.shared.up_proj = ReadFloat(have, mp + "shared_experts.up_proj.weight");
        mo.shared.down_proj = ReadFloat(have, mp + "shared_experts.down_proj.weight");
      }
      mo.experts.resize(static_cast<size_t>(p.num_experts));
      for (int64_t e = 0; e < p.num_experts; ++e) {
        const std::string ep = mp + "experts." + std::to_string(e) + ".";
        MlpHostWeights& ex = mo.experts[static_cast<size_t>(e)];
        ex.gate_proj = ReadFloat(have, ep + "w1.weight");  // gate == w1
        ex.down_proj = ReadFloat(have, ep + "w2.weight");  // down == w2
        ex.up_proj = ReadFloat(have, ep + "w3.weight");    // up   == w3
      }
    } else {
      const std::string mp = b + "mlp.";
      lw.dense.gate_proj = ReadFloat(have, mp + "gate_proj.weight");
      lw.dense.up_proj = ReadFloat(have, mp + "up_proj.weight");
      lw.dense.down_proj = ReadFloat(have, mp + "down_proj.weight");
    }
  }
  h.materialized = true;
  return h;
}

// ─── bf16-RESIDENT materialization helpers (§13) ───────────────────────────────
// Decode one enumerated checkpoint tensor to row-major f32 THROUGH a TensorResolver
// (the bf16-resident load path indexes shards by resolver, not the HaveMap the f32
// reference uses), releasing the mmap source pages after the copy (shard-release per
// tensor). Handles the two dtypes the small vectors carry (BF16 norms/conv/gate;
// F32 dt_bias/A_log). Used for the tiny f32-kept vectors only — the large matmul
// weights load bf16-verbatim via dense_loaders::LoadBf16Direct.
std::vector<float> ReadFloatVec(const TensorResolver& get, const std::string& name) {
  const StTensor& t = get(name);
  const int64_t n = NumEl(t.shape);
  std::vector<float> out(static_cast<size_t>(n), 0.0f);
  if (t.dtype == "F32") {
    std::memcpy(out.data(), t.data, static_cast<size_t>(n) * 4);
  } else if (t.dtype == "BF16") {
    const uint8_t* p = t.data;
    for (int64_t i = 0; i < n; ++i) {
      const uint16_t bits = static_cast<uint16_t>(p[i * 2]) |
                            (static_cast<uint16_t>(p[i * 2 + 1]) << 8);
      const uint32_t u = static_cast<uint32_t>(bits) << 16;
      std::memcpy(&out[static_cast<size_t>(i)], &u, 4);
    }
  } else {
    VT_CHECK(false, "kimi-linear resident: tensor '" + name + "' has dtype '" +
                        t.dtype + "' (only BF16/F32 supported for the f32-kept vectors)");
  }
  MaybeReleaseSourcePages(t.data, t.nbytes);
  return out;
}

// Round a host f32 vector to a flat bf16 OwnedTensor (the tiny-config gate builds the
// resident weights from the f32 host weights). The device forward always views it
// with an explicit {N,K} shape, so a flat 1-D store is sufficient.
OwnedTensor Bf16OwnedFromF32(const std::vector<float>& v) {
  OwnedTensor o =
      dense_loaders::MakeOwned(vt::DType::kBF16, {static_cast<int64_t>(v.size())});
  auto* dst = reinterpret_cast<uint16_t*>(o.bytes.data());
  for (size_t i = 0; i < v.size(); ++i) dst[i] = vt::F32ToBF16(v[i]);
  return o;
}

}  // namespace

KimiLinearParams ParseKimiLinearParams(const HfConfig& config) {
  const nlohmann::json& raw = config.raw;
  KimiLinearParams p;

  // --- shared geometry (LoadHfConfig lifted the standard scalars into typed
  // fields; fall back to raw for a synthetically-constructed config) ---
  p.hidden_size =
      config.hidden_size > 0 ? config.hidden_size : RawInt(raw, "hidden_size", 0);
  p.num_hidden_layers = config.num_hidden_layers > 0
                            ? config.num_hidden_layers
                            : RawInt(raw, "num_hidden_layers", 0);
  p.vocab_size =
      config.vocab_size > 0 ? config.vocab_size : RawInt(raw, "vocab_size", 0);
  p.num_attention_heads = config.num_attention_heads > 0
                              ? config.num_attention_heads
                              : RawInt(raw, "num_attention_heads", 0);
  p.num_key_value_heads = config.num_key_value_heads > 0
                              ? config.num_key_value_heads
                              : RawInt(raw, "num_key_value_heads",
                                       p.num_attention_heads);
  if (p.num_key_value_heads == 0) p.num_key_value_heads = p.num_attention_heads;
  p.head_dim = config.head_dim > 0 ? config.head_dim : RawInt(raw, "head_dim", 0);
  if (p.head_dim == 0 && p.num_attention_heads > 0)
    p.head_dim = p.hidden_size / p.num_attention_heads;
  p.intermediate_size = config.intermediate_size > 0
                            ? config.intermediate_size
                            : RawInt(raw, "intermediate_size", 0);
  p.rms_norm_eps = static_cast<float>(RawDouble(raw, "rms_norm_eps", 1e-5));
  p.tie_word_embeddings = RawBool(raw, "tie_word_embeddings", false);
  p.max_position_embeddings = RawInt(raw, "max_position_embeddings", 0);
  p.num_nextn_predict_layers = RawInt(raw, "num_nextn_predict_layers", 0);
  p.rope_theta = RawDouble(raw, "rope_theta", 10000.0);

  // --- MLA geometry (kimi_linear.py:118-127; NoPE, no-q-lora) ---
  p.kv_lora_rank = RawInt(raw, "kv_lora_rank", 0);
  p.q_lora_rank = RawInt(raw, "q_lora_rank", 0);  // 0 == null
  p.qk_nope_head_dim = RawInt(raw, "qk_nope_head_dim", 0);
  p.qk_rope_head_dim = RawInt(raw, "qk_rope_head_dim", 0);
  p.v_head_dim = RawInt(raw, "v_head_dim", 0);
  p.mla_use_nope = RawBool(raw, "mla_use_nope", false);

  // --- MoE (kimi_linear.py:104-168; upstream key num_experts_per_token) ---
  p.num_experts = RawInt(raw, "num_experts", 0);
  p.num_experts_per_token = RawInt(raw, "num_experts_per_token", 0);
  p.num_shared_experts = RawInt(raw, "num_shared_experts", 0);
  p.moe_intermediate_size = RawInt(raw, "moe_intermediate_size", 0);
  p.first_k_dense_replace = RawInt(raw, "first_k_dense_replace", 0);
  p.moe_layer_freq = RawInt(raw, "moe_layer_freq", 1);
  p.routed_scaling_factor = RawDouble(raw, "routed_scaling_factor", 1.0);
  p.moe_renormalize = RawBool(raw, "moe_renormalize", true);
  p.num_expert_group = RawInt(raw, "num_expert_group", 1);
  p.topk_group = RawInt(raw, "topk_group", 1);
  p.use_grouped_topk = RawBool(raw, "use_grouped_topk", true);
  p.moe_router_activation_func =
      RawString(raw, "moe_router_activation_func", "sigmoid");

  // --- KDA linear_attn_config (kimi_gdn_linear_attn.py:110-118) ---
  if (const nlohmann::json* lac = Object(raw, "linear_attn_config")) {
    p.has_linear_attn_config = true;
    p.kda_layers = RawIntArray(*lac, "kda_layers");
    p.full_attn_layers = RawIntArray(*lac, "full_attn_layers");
    p.kda_num_heads = RawInt(*lac, "num_heads", 0);
    p.kda_head_dim = RawInt(*lac, "head_dim", 0);
    p.kda_short_conv_kernel_size = RawInt(*lac, "short_conv_kernel_size", 0);
  }

  // --- validation (throw a precise message on anything unrepresentable) ---
  VT_CHECK(p.hidden_size > 0, "kimi-linear: hidden_size must be positive");
  VT_CHECK(p.num_hidden_layers > 0,
           "kimi-linear: num_hidden_layers must be positive");
  VT_CHECK(p.vocab_size > 0, "kimi-linear: vocab_size must be positive");
  VT_CHECK(p.num_attention_heads > 0,
           "kimi-linear: num_attention_heads must be positive");
  VT_CHECK(p.num_experts > 0,
           "kimi-linear: num_experts must be positive (Kimi-Linear is a MoE arch, "
           "kimi_linear.py:130)");
  VT_CHECK(p.num_experts_per_token > 0,
           "kimi-linear: num_experts_per_token must be positive");
  VT_CHECK(p.kv_lora_rank > 0,
           "kimi-linear: kv_lora_rank must be positive (MLA latent)");
  VT_CHECK(p.qk_nope_head_dim > 0 && p.qk_rope_head_dim > 0 && p.v_head_dim > 0,
           "kimi-linear: MLA qk_nope_head_dim / qk_rope_head_dim / v_head_dim must "
           "be positive");
  // kimi_linear.py:214 hard-asserts use_nope is True; :215 asserts q_lora_rank is
  // None. This bring-up serves ONLY the shipped NoPE/no-q-lora Kimi-Linear.
  VT_CHECK(p.mla_use_nope,
           "kimi-linear: mla_use_nope must be true — KimiMLAAttention asserts "
           "use_nope (kimi_linear.py:214); a positional MLA variant is unsupported");
  VT_CHECK(p.q_lora_rank == 0,
           "kimi-linear: q_lora_rank must be null/0 — KimiMLAAttention asserts "
           "q_lora_rank is None (kimi_linear.py:215); the q-LoRA branch is the K3 "
           "path, not Kimi-Linear");
  VT_CHECK(p.has_linear_attn_config,
           "kimi-linear: linear_attn_config is required (KDA hybrid); kda_layers/"
           "full_attn_layers select the per-layer KDA vs MLA split "
           "(kimi_linear.py:105-108)");
  VT_CHECK(!p.kda_layers.empty(),
           "kimi-linear: linear_attn_config.kda_layers must be non-empty "
           "(is_kda_layer needs the set, kimi_linear.py:144-148)");
  VT_CHECK(p.kda_num_heads > 0 && p.kda_head_dim > 0 &&
               p.kda_short_conv_kernel_size > 0,
           "kimi-linear: linear_attn_config num_heads / head_dim / "
           "short_conv_kernel_size must be positive");
  VT_CHECK(p.moe_router_activation_func == "sigmoid" ||
               p.moe_router_activation_func == "softmax",
           "kimi-linear: moe_router_activation_func must be 'sigmoid' or "
           "'softmax'; got '" + p.moe_router_activation_func + "'");
  return p;
}

void ParseKimiLinearConfig(const HfConfig& config) {
  // The resolve itself IS the validation (throws on every unsupported field).
  (void)ParseKimiLinearParams(config);
}

bool KimiLinearParams::is_kda_layer(int64_t layer_idx) const {
  // Mirrors KimiLinearConfig.is_kda_layer: (layer_idx + 1) in kda_layers
  // (kimi_linear.py:144-148).
  return std::find(kda_layers.begin(), kda_layers.end(), layer_idx + 1) !=
         kda_layers.end();
}

bool KimiLinearParams::is_moe_layer(int64_t layer_idx) const {
  // kimi_linear.py:328-333: is_moe && layer_idx >= first_k_dense_replace &&
  // layer_idx % moe_layer_freq == 0. num_experts>0 => is_moe (kimi_linear.py:130).
  const int64_t freq = moe_layer_freq > 0 ? moe_layer_freq : 1;
  return num_experts > 0 && layer_idx >= first_k_dense_replace &&
         (layer_idx % freq) == 0;
}

std::vector<std::string> EnumerateKimiLinearTensors(const KimiLinearParams& p) {
  std::vector<std::string> names;
  for (const TensorSpec& s : EnumerateSpecs(p)) names.push_back(s.name);
  return names;
}

KimiLinearWeights LoadKimiLinearForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  const KimiLinearParams p = ParseKimiLinearParams(config);

  // Index every shard tensor once for O(1) presence + shape lookup.
  std::unordered_map<std::string, const StTensor*> have;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names())
      have.emplace(name, &shard.Get(name));

  const std::vector<TensorSpec> expected = EnumerateSpecs(p);
  int64_t accounted = 0;
  for (const TensorSpec& s : expected) {
    const auto it = have.find(s.name);
    // THROW BY NAME on a missing tensor — never a silent zero.
    VT_CHECK(it != have.end(),
             "kimi-linear loader: missing checkpoint tensor '" + s.name +
                 "' (expected by the KimiLinearForCausalLM name-map; "
                 "kimi_linear.py:460-554)");
    if (!s.shape.empty()) {
      const std::vector<int64_t>& got = it->second->shape;
      VT_CHECK(got == s.shape,
               "kimi-linear loader: tensor '" + s.name + "' has shape " +
                   ShapeStr(got) + " but the config implies " + ShapeStr(s.shape));
    }
    ++accounted;
  }

  KimiLinearWeights w;
  w.params = p;
  w.enumerated_tensors = static_cast<int64_t>(expected.size());
  w.accounted_tensors = accounted;
  // W2 CPU-reference lane: materialize the host float weights the reference forward
  // composes (bf16/f32 -> f32). Coverage/shapes already validated above, so every
  // ReadFloat resolves. The DEVICE staging (ResidentWeight, absorbed MLA bmm forms,
  // grouped-MoE slabs) is the born-on-runner W6/W7 residual.
  w.host = MaterializeHost(p, have);
  return w;
}

// ─── bf16-RESIDENT loader + builder + stager (§13) ─────────────────────────────
void StageKimiResidentBf16(vt::Queue& queue, const OwnedTensor& w) {
  if (w.d_dev || !w.HasHostBytes()) return;  // already staged / nothing to stage
  // CPU: the device forward aliases the host bytes directly (host-pointer aliasing is
  // a CPU property), so keep them — never release the sole copy.
  if (vllm::platforms::GetPlatform(queue.device.type).is_cpu()) return;
  vt::Backend& b = vt::GetBackend(queue.device.type);
  const size_t nb = w.bytes.size();
  void* p = b.Alloc(nb);            // cudaMalloc'd device memory — no ATS penalty
  b.Copy(queue, p, w.bytes.data(), nb);
  b.Synchronize(queue);            // the H2D must finish before the host mirror frees
  vt::Backend* bk = &b;
  w.d_dev = std::shared_ptr<void>(p, [bk](void* pp) { bk->Free(pp); });
  w.ReleaseHost();                 // free the host bf16 bytes — d_dev is authoritative
}

KimiLinearWeights LoadKimiLinearResidentBf16Weights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config,
    vt::Queue* stage_queue) {
  const KimiLinearParams p = ParseKimiLinearParams(config);

  // name -> shard resolver (mirror laguna_weights.cpp:398).
  auto where =
      std::make_shared<std::unordered_map<std::string, const SafetensorsFile*>>();
  for (const SafetensorsFile& s : shards)
    for (const std::string& n : s.Names()) (*where)[n] = &s;
  const TensorResolver get =
      [where](const std::string& name) -> const StTensor& {
    auto it = where->find(name);
    VT_CHECK(it != where->end(),
             "kimi-linear resident loader: missing checkpoint tensor '" + name +
                 "' (expected by the KimiLinearForCausalLM name-map)");
    return it->second->Get(name);
  };

  // Large matmul weight: bf16-verbatim OwnedTensor, then (on CUDA) staged to d_dev +
  // ReleaseHost so only ONE tensor's host bytes are live at a time (pool math §13).
  auto big = [&](const std::string& name) -> OwnedTensor {
    OwnedTensor o = dense_loaders::LoadBf16Direct(get, name);  // copies bf16, drops src pages
    if (stage_queue != nullptr) StageKimiResidentBf16(*stage_queue, o);
    return o;
  };
  // Small vector kept host f32 (ReadFloatVec releases the source pages).
  auto vecf = [&](const std::string& name) { return ReadFloatVec(get, name); };

  KimiLinearResidentWeights r;
  r.resident = true;
  r.tie_word_embeddings = p.tie_word_embeddings;
  r.embed_tokens = big("model.embed_tokens.weight");
  r.final_norm = vecf("model.norm.weight");
  if (!p.tie_word_embeddings) r.lm_head = big("lm_head.weight");

  r.layers.resize(static_cast<size_t>(p.num_hidden_layers));
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    KimiLinearLayerResidentWeights& lw = r.layers[static_cast<size_t>(l)];
    const std::string b = "model.layers." + std::to_string(l) + ".";
    lw.input_layernorm = vecf(b + "input_layernorm.weight");
    lw.post_attention_layernorm = vecf(b + "post_attention_layernorm.weight");

    const std::string a = b + "self_attn.";
    lw.is_kda = p.is_kda_layer(l);
    if (lw.is_kda) {
      KdaResidentWeights& k = lw.kda;
      k.q_proj = big(a + "q_proj.weight");
      k.k_proj = big(a + "k_proj.weight");
      k.v_proj = big(a + "v_proj.weight");
      k.f_a_proj = big(a + "f_a_proj.weight");
      k.f_b_proj = big(a + "f_b_proj.weight");
      k.b_proj = big(a + "b_proj.weight");
      k.g_a_proj = big(a + "g_a_proj.weight");
      k.g_b_proj = big(a + "g_b_proj.weight");
      k.o_proj = big(a + "o_proj.weight");
      k.q_conv = vecf(a + "q_conv1d.weight");
      k.k_conv = vecf(a + "k_conv1d.weight");
      k.v_conv = vecf(a + "v_conv1d.weight");
      k.dt_bias = vecf(a + "dt_bias");
      k.a_log = vecf(a + "A_log");
      k.o_norm = vecf(a + "o_norm.weight");
    } else {
      MlaResidentWeights& m = lw.mla;
      m.q_proj = big(a + "q_proj.weight");
      m.kv_a_proj_with_mqa = big(a + "kv_a_proj_with_mqa.weight");
      m.kv_a_layernorm = vecf(a + "kv_a_layernorm.weight");
      m.kv_b_proj = big(a + "kv_b_proj.weight");
      m.o_proj = big(a + "o_proj.weight");
    }

    lw.is_moe = p.is_moe_layer(l);
    if (lw.is_moe) {
      MoeResidentWeights& mo = lw.moe;
      const std::string mp = b + "block_sparse_moe.";
      mo.gate = big(mp + "gate.weight");
      mo.e_score_correction_bias = vecf(mp + "gate.e_score_correction_bias");
      mo.has_shared = p.num_shared_experts > 0;
      if (mo.has_shared) {
        mo.shared.gate_proj = big(mp + "shared_experts.gate_proj.weight");
        mo.shared.up_proj = big(mp + "shared_experts.up_proj.weight");
        mo.shared.down_proj = big(mp + "shared_experts.down_proj.weight");
      }
      mo.experts.resize(static_cast<size_t>(p.num_experts));
      for (int64_t e = 0; e < p.num_experts; ++e) {
        const std::string ep = mp + "experts." + std::to_string(e) + ".";
        MlpResidentWeights& ex = mo.experts[static_cast<size_t>(e)];
        ex.gate_proj = big(ep + "w1.weight");  // gate == w1
        ex.down_proj = big(ep + "w2.weight");  // down == w2
        ex.up_proj = big(ep + "w3.weight");    // up   == w3
      }
    } else {
      const std::string mp = b + "mlp.";
      lw.dense.gate_proj = big(mp + "gate_proj.weight");
      lw.dense.up_proj = big(mp + "up_proj.weight");
      lw.dense.down_proj = big(mp + "down_proj.weight");
    }
  }

  KimiLinearWeights w;
  w.params = p;
  w.enumerated_tensors = static_cast<int64_t>(EnumerateSpecs(p).size());
  w.accounted_tensors = w.enumerated_tensors;  // the resolver throws BY NAME on a miss
  w.resident = std::move(r);
  // host stays UNMATERIALIZED on the resident path (183 GiB f32 would OOM the pool).
  return w;
}

KimiLinearResidentWeights BuildKimiResidentFromHost(const KimiLinearHostWeights& host,
                                                    const KimiLinearParams& p) {
  KimiLinearResidentWeights r;
  r.resident = true;
  r.tie_word_embeddings = p.tie_word_embeddings;
  r.embed_tokens = Bf16OwnedFromF32(host.embed_tokens);
  r.final_norm = host.final_norm;
  if (!p.tie_word_embeddings && !host.lm_head.empty())
    r.lm_head = Bf16OwnedFromF32(host.lm_head);

  r.layers.resize(host.layers.size());
  for (size_t l = 0; l < host.layers.size(); ++l) {
    const KimiLinearLayerHostWeights& hl = host.layers[l];
    KimiLinearLayerResidentWeights& rl = r.layers[l];
    rl.input_layernorm = hl.input_layernorm;
    rl.post_attention_layernorm = hl.post_attention_layernorm;
    rl.is_kda = hl.is_kda;
    if (hl.is_kda) {
      const KdaLayerHostWeights& hk = hl.kda;
      KdaResidentWeights& rk = rl.kda;
      rk.q_proj = Bf16OwnedFromF32(hk.q_proj);
      rk.k_proj = Bf16OwnedFromF32(hk.k_proj);
      rk.v_proj = Bf16OwnedFromF32(hk.v_proj);
      rk.f_a_proj = Bf16OwnedFromF32(hk.f_a_proj);
      rk.f_b_proj = Bf16OwnedFromF32(hk.f_b_proj);
      rk.b_proj = Bf16OwnedFromF32(hk.b_proj);
      rk.g_a_proj = Bf16OwnedFromF32(hk.g_a_proj);
      rk.g_b_proj = Bf16OwnedFromF32(hk.g_b_proj);
      rk.o_proj = Bf16OwnedFromF32(hk.o_proj);
      rk.q_conv = hk.q_conv;
      rk.k_conv = hk.k_conv;
      rk.v_conv = hk.v_conv;
      rk.dt_bias = hk.dt_bias;
      rk.a_log = hk.a_log;
      rk.o_norm = hk.o_norm;
    } else {
      const MlaLayerHostWeights& hm = hl.mla;
      MlaResidentWeights& rm = rl.mla;
      rm.q_proj = Bf16OwnedFromF32(hm.q_proj);
      rm.kv_a_proj_with_mqa = Bf16OwnedFromF32(hm.kv_a_proj_with_mqa);
      rm.kv_b_proj = Bf16OwnedFromF32(hm.kv_b_proj);
      rm.o_proj = Bf16OwnedFromF32(hm.o_proj);
      rm.kv_a_layernorm = hm.kv_a_layernorm;
    }
    rl.is_moe = hl.is_moe;
    if (hl.is_moe) {
      const MoeHostWeights& hmo = hl.moe;
      MoeResidentWeights& rmo = rl.moe;
      rmo.gate = Bf16OwnedFromF32(hmo.gate);
      rmo.e_score_correction_bias = hmo.e_score_correction_bias;
      rmo.has_shared = hmo.has_shared;
      if (hmo.has_shared) {
        rmo.shared.gate_proj = Bf16OwnedFromF32(hmo.shared.gate_proj);
        rmo.shared.up_proj = Bf16OwnedFromF32(hmo.shared.up_proj);
        rmo.shared.down_proj = Bf16OwnedFromF32(hmo.shared.down_proj);
      }
      rmo.experts.resize(hmo.experts.size());
      for (size_t e = 0; e < hmo.experts.size(); ++e) {
        rmo.experts[e].gate_proj = Bf16OwnedFromF32(hmo.experts[e].gate_proj);
        rmo.experts[e].up_proj = Bf16OwnedFromF32(hmo.experts[e].up_proj);
        rmo.experts[e].down_proj = Bf16OwnedFromF32(hmo.experts[e].down_proj);
      }
    } else {
      rl.dense.gate_proj = Bf16OwnedFromF32(hl.dense.gate_proj);
      rl.dense.up_proj = Bf16OwnedFromF32(hl.dense.up_proj);
      rl.dense.down_proj = Bf16OwnedFromF32(hl.dense.down_proj);
    }
  }
  return r;
}

}  // namespace vllm
