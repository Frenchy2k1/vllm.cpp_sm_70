// Kimi K3 config resolution + text-backbone structural name-map (W2/W5). This TU
// implements `ParseKimiK3Params` (config-descent over the wrapper's `text_config`
// / `vision_config` / `quantization_config`, unit-testable), the pure
// `EnumerateKimiK3TextBackboneTensors` structural map (grounded 1:1 in the pinned
// `kimi_linear.py` + `kimi_gdn_linear_attn.py`), and the
// `LoadKimiK3ForConditionalGenerationWeights` accounting entry.
//
// ─── DERIVE-AND-SHIP HONESTY ─────────────────────────────────────────────────
// A REAL Kimi-K3 checkpoint is MXFP4 (`mxfp4-pack-quantized`) and multimodal, and
// K3 is beyond the pinned oracle (555967922). This loader therefore THROWS
// NOT-YET-BUILDABLE on a real checkpoint: (a) MXFP4 weight materialization is the
// SHARED DeepSeek-V4 MegaMoE MXFP4 scope (`CLAIM-DEEPSEEK-V4-*`) — not written
// twice here; (b) the K3 multimodal-wrapper weight PREFIX + the MoonViT-V2 vision
// tower are post-pin UNCONFIRMED (W7). What IS verifiable + tested is the
// STRUCTURE of the 93-layer KDA/MLA hybrid + 896-expert MoE text backbone,
// grounded in the pinned `kimi_linear.py`. A green build is DERIVED, not execution.
//
// ─── TEXT-BACKBONE NAME MAP (pinned kimi_linear.py, DERIVED for K3 scale) ──────
//   MODEL LEVEL       {p}model.embed_tokens.weight, {p}model.norm.weight,
//                     {p}lm_head.weight (absent if tie_word_embeddings)
//   PER LAYER N       {p}model.layers.N.input_layernorm.weight,
//                     .post_attention_layernorm.weight
//     KDA layer (is_kda_layer, kimi_gdn_linear_attn.py:102-226):
//       .self_attn.{q_proj,k_proj,v_proj,f_a_proj,f_b_proj,b_proj,g_a_proj,
//                   g_b_proj,o_proj,q_conv1d,k_conv1d,v_conv1d}.weight
//                   + .self_attn.{dt_bias,A_log,o_norm.weight}
//     MLA layer (KimiMLAAttention, kimi_linear.py:217-248):
//       q_lora_rank==0 (KimiLinear default): .self_attn.q_proj.weight
//       q_lora_rank >0 (K3 = 1536, DeepSeek-V3 q-LoRA branch):
//                     .self_attn.{q_a_proj,q_a_layernorm.weight? -> q_a_layernorm}.weight,
//                     .self_attn.q_b_proj.weight
//       common:       .self_attn.{kv_a_proj_with_mqa,kv_a_layernorm,kv_b_proj,
//                                 o_proj}.weight
//     MLP dispatch (kimi_linear.py:328-347):
//       MoE layer:    .mlp.gate.weight, .mlp.gate.e_score_correction_bias,
//                     .mlp.shared_experts.{gate_proj,up_proj,down_proj}.weight,
//                     .mlp.experts.E.{w1,w2,w3}.weight   (E in [0, num_experts))
//       dense layer:  .mlp.{gate_proj,up_proj,down_proj}.weight
#include "vllm/model_executor/models/kimi_k3.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// --- raw json readers over an arbitrary object (nested text/vision configs) ---
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

bool ContainsInsensitive(const std::string& hay, const char* needle) {
  std::string h = hay;
  std::transform(h.begin(), h.end(), h.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return h.find(needle) != std::string::npos;
}

}  // namespace

KimiK3Params ParseKimiK3Params(const HfConfig& config) {
  const nlohmann::json& raw = config.raw;
  // For the multimodal wrapper the text fields nest under `text_config`; for a
  // bare KimiLinear config `text` aliases the top-level doc.
  const nlohmann::json* text_obj = Object(raw, "text_config");
  const nlohmann::json& text = text_obj != nullptr ? *text_obj : raw;

  KimiK3Params p;
  KimiK3TextParams& t = p.text;

  // --- shared geometry (LoadHfConfig already lifted these from text_config) ---
  t.hidden_size =
      config.hidden_size > 0 ? config.hidden_size : RawInt(text, "hidden_size", 0);
  t.num_hidden_layers = config.num_hidden_layers > 0
                            ? config.num_hidden_layers
                            : RawInt(text, "num_hidden_layers", 0);
  t.vocab_size =
      config.vocab_size > 0 ? config.vocab_size : RawInt(text, "vocab_size", 0);
  t.num_attention_heads = config.num_attention_heads > 0
                              ? config.num_attention_heads
                              : RawInt(text, "num_attention_heads", 0);
  t.intermediate_size = RawInt(text, "intermediate_size", 0);
  t.rms_norm_eps = static_cast<float>(RawDouble(text, "rms_norm_eps", 1e-6));
  t.tie_word_embeddings = RawBool(text, "tie_word_embeddings", false);
  t.max_position_embeddings = RawInt(text, "max_position_embeddings", 0);
  t.num_nextn_predict_layers = RawInt(text, "num_nextn_predict_layers", 0);
  t.rope_theta = RawDouble(text, "rope_theta", 10000.0);

  // --- MLA geometry (kimi_linear.py:119-127) ---
  t.kv_lora_rank = RawInt(text, "kv_lora_rank", 0);
  t.q_lora_rank = RawInt(text, "q_lora_rank", 0);
  t.qk_nope_head_dim = RawInt(text, "qk_nope_head_dim", 0);
  t.qk_rope_head_dim = RawInt(text, "qk_rope_head_dim", 0);
  t.v_head_dim = RawInt(text, "v_head_dim", 0);
  t.mla_use_nope = RawBool(text, "mla_use_nope", false);

  // --- MoE (kimi_linear.py:104-168; note upstream key num_experts_per_token) ---
  t.num_experts = RawInt(text, "num_experts", 0);
  t.num_experts_per_token = RawInt(text, "num_experts_per_token", 0);
  t.num_shared_experts = RawInt(text, "num_shared_experts", 0);
  t.moe_intermediate_size = RawInt(text, "moe_intermediate_size", 0);
  t.first_k_dense_replace = RawInt(text, "first_k_dense_replace", 0);
  t.moe_layer_freq = RawInt(text, "moe_layer_freq", 1);
  t.routed_scaling_factor = RawDouble(text, "routed_scaling_factor", 1.0);
  t.moe_renormalize = RawBool(text, "moe_renormalize", true);
  t.num_expert_group = RawInt(text, "num_expert_group", 1);
  t.topk_group = RawInt(text, "topk_group", 1);
  t.use_grouped_topk = RawBool(text, "use_grouped_topk", true);
  t.moe_router_activation_func =
      RawString(text, "moe_router_activation_func", "sigmoid");

  // --- KDA linear_attn_config (kimi_gdn_linear_attn.py:109-119) ---
  if (const nlohmann::json* lac = Object(text, "linear_attn_config")) {
    t.has_linear_attn_config = true;
    t.kda_layers = RawIntArray(*lac, "kda_layers");
    t.full_attn_layers = RawIntArray(*lac, "full_attn_layers");
    t.kda_num_heads = RawInt(*lac, "num_heads", 0);
    t.kda_head_dim = RawInt(*lac, "head_dim", 0);
    t.kda_short_conv_kernel_size = RawInt(*lac, "short_conv_kernel_size", 0);
  }

  // --- vision_config (MoonViT-V2) — PARTIAL (W7) ---
  if (const nlohmann::json* vc = Object(raw, "vision_config")) {
    p.vision.present = true;
    p.vision.patch_size = RawInt(*vc, "patch_size", 0);
    p.vision.num_hidden_layers = RawInt(*vc, "num_hidden_layers", 0);
    p.vision.hidden_size = RawInt(*vc, "hidden_size", 0);
  }

  // --- quantization_config (compressed-tensors mxfp4-pack-quantized) ---
  if (const nlohmann::json* qc = Object(raw, "quantization_config")) {
    p.quant_method = RawString(*qc, "quant_method", "");
    // compressed-tensors names the packing via `format` (and/or a per-group
    // "type"). Detect MXFP4 leniently from either field.
    p.quant_format = RawString(*qc, "format", "");
    p.is_mxfp4 = ContainsInsensitive(p.quant_method, "mxfp4") ||
                 ContainsInsensitive(p.quant_format, "mxfp4");
  }

  // --- validation (throw with a precise message on anything unrepresentable) ---
  VT_CHECK(t.hidden_size > 0, "kimi-k3: text_config.hidden_size must be positive");
  VT_CHECK(t.num_hidden_layers > 0,
           "kimi-k3: text_config.num_hidden_layers must be positive");
  VT_CHECK(t.vocab_size > 0, "kimi-k3: text_config.vocab_size must be positive");
  VT_CHECK(t.num_experts > 0,
           "kimi-k3: text_config.num_experts must be positive (K3 is a MoE arch)");
  VT_CHECK(t.num_experts_per_token > 0,
           "kimi-k3: text_config.num_experts_per_token must be positive");
  VT_CHECK(t.kv_lora_rank > 0,
           "kimi-k3: text_config.kv_lora_rank must be positive (MLA latent)");
  VT_CHECK(t.qk_nope_head_dim > 0 && t.qk_rope_head_dim > 0,
           "kimi-k3: MLA qk_nope_head_dim / qk_rope_head_dim must be positive");
  VT_CHECK(t.has_linear_attn_config,
           "kimi-k3: text_config.linear_attn_config is required (KDA hybrid); "
           "kda_layers/full_attn_layers select the per-layer KDA vs MLA split "
           "(kimi_linear.py:105-108)");
  VT_CHECK(!t.kda_layers.empty(),
           "kimi-k3: linear_attn_config.kda_layers must be non-empty (K3 is a "
           "KDA/MLA hybrid; is_kda_layer needs the set, kimi_linear.py:144-148)");
  // kimi_linear.py:96 asserts the router activation is softmax or sigmoid.
  VT_CHECK(t.moe_router_activation_func == "sigmoid" ||
               t.moe_router_activation_func == "softmax",
           "kimi-k3: moe_router_activation_func must be 'sigmoid' or 'softmax'; "
           "got '" + t.moe_router_activation_func + "'");
  return p;
}

void ParseKimiK3Config(const HfConfig& config) {
  // The resolve itself IS the validation (throws on every unsupported field).
  (void)ParseKimiK3Params(config);
}

bool KimiK3TextParams::is_kda_layer(int64_t layer_idx) const {
  // Mirrors KimiLinearConfig.is_kda_layer: (layer_idx + 1) in kda_layers
  // (kimi_linear.py:144-148).
  return std::find(kda_layers.begin(), kda_layers.end(), layer_idx + 1) !=
         kda_layers.end();
}

bool KimiK3TextParams::is_moe_layer(int64_t layer_idx) const {
  // kimi_linear.py:328-333: is_moe && layer_idx >= first_k_dense_replace &&
  // layer_idx % moe_layer_freq == 0. num_experts>0 => is_moe (kimi_linear.py:130).
  const int64_t freq = moe_layer_freq > 0 ? moe_layer_freq : 1;
  return num_experts > 0 && layer_idx >= first_k_dense_replace &&
         (layer_idx % freq) == 0;
}

std::vector<std::string> EnumerateKimiK3TextBackboneTensors(
    const KimiK3TextParams& t, const std::string& prefix) {
  std::vector<std::string> names;
  const std::string p = prefix;  // wrapper language-model prefix ("" = standalone)

  // --- model level ---
  names.push_back(p + "model.embed_tokens.weight");
  names.push_back(p + "model.norm.weight");
  if (!t.tie_word_embeddings) names.push_back(p + "lm_head.weight");

  for (int64_t l = 0; l < t.num_hidden_layers; ++l) {
    const std::string b = p + "model.layers." + std::to_string(l) + ".";
    names.push_back(b + "input_layernorm.weight");
    names.push_back(b + "post_attention_layernorm.weight");

    const std::string a = b + "self_attn.";
    if (t.is_kda_layer(l)) {
      // KDA (kimi_gdn_linear_attn.py:120-226).
      for (const char* w :
           {"q_proj", "k_proj", "v_proj", "f_a_proj", "f_b_proj", "b_proj",
            "g_a_proj", "g_b_proj", "o_proj", "q_conv1d", "k_conv1d", "v_conv1d"})
        names.push_back(a + w + ".weight");
      names.push_back(a + "dt_bias");
      names.push_back(a + "A_log");
      names.push_back(a + "o_norm.weight");
    } else {
      // MLA (KimiMLAAttention, kimi_linear.py:217-248). K3 q_lora_rank=1536 uses
      // the DeepSeek-V3 q-LoRA branch; KimiLinear default (q_lora_rank None) uses
      // the direct q_proj.
      if (t.q_lora_rank > 0) {
        names.push_back(a + "q_a_proj.weight");
        names.push_back(a + "q_a_layernorm.weight");
        names.push_back(a + "q_b_proj.weight");
      } else {
        names.push_back(a + "q_proj.weight");
      }
      names.push_back(a + "kv_a_proj_with_mqa.weight");
      names.push_back(a + "kv_a_layernorm.weight");
      names.push_back(a + "kv_b_proj.weight");
      names.push_back(a + "o_proj.weight");
    }

    const std::string m = b + "mlp.";
    if (t.is_moe_layer(l)) {
      names.push_back(m + "gate.weight");
      // noaux_tc e_score_correction_bias (kimi_linear.py:138).
      names.push_back(m + "gate.e_score_correction_bias");
      if (t.num_shared_experts > 0)
        for (const char* w : {"gate_proj", "up_proj", "down_proj"})
          names.push_back(m + "shared_experts." + w + ".weight");
      // Checkpoint expert names w1/w2/w3 (fused_moe_make_expert_params_mapping,
      // kimi_linear.py:469-475).
      for (int64_t e = 0; e < t.num_experts; ++e) {
        const std::string ep = m + "experts." + std::to_string(e) + ".";
        for (const char* w : {"w1", "w2", "w3"}) names.push_back(ep + w + ".weight");
      }
    } else {
      for (const char* w : {"gate_proj", "up_proj", "down_proj"})
        names.push_back(m + w + ".weight");
    }
  }
  return names;
}

KimiK3Weights LoadKimiK3ForConditionalGenerationWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  const KimiK3Params p = ParseKimiK3Params(config);

  // NOT-YET-BUILDABLE #1 — MXFP4 weights are the SHARED DeepSeek-V4 MegaMoE MXFP4
  // scope. Refuse rather than write a second MXFP4 kernel.
  VT_CHECK(!p.is_mxfp4,
           "kimi-k3 loader: MXFP4 (mxfp4-pack-quantized, compressed-tensors) "
           "weight materialization is NOT-YET-BUILDABLE in this lane — it is the "
           "SHARED DeepSeek-V4 MegaMoE MXFP4 scope (CLAIM-DEEPSEEK-V4-*, "
           "quantization-matrix MXFP4 row). A real Kimi-K3 checkpoint is MXFP4, so "
           "this is the expected honest state. See .agents/specs/kimi-k3.md §W3.");

  // The structural text-backbone map (grounded in the pinned kimi_linear.py).
  const std::vector<std::string> expected =
      EnumerateKimiK3TextBackboneTensors(p.text, /*prefix=*/"");

  std::unordered_set<std::string> have;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) have.insert(name);

  int64_t accounted = 0;
  for (const std::string& name : expected)
    if (have.count(name) != 0) ++accounted;

  // NOT-YET-BUILDABLE #2 — the K3 multimodal-wrapper weight PREFIX and the
  // MoonViT-V2 vision tower are post-pin UNCONFIRMED (W7 / pin advance). This
  // accounting is over the standalone-KimiLinear names only; device materialization
  // of the towers is not wired. A forward refuses regardless (kimi_k3.cpp).

  KimiK3Weights w;
  w.params = p;
  w.enumerated_tensors = static_cast<int64_t>(expected.size());
  w.accounted_tensors = accounted;
  return w;
}

}  // namespace vllm
