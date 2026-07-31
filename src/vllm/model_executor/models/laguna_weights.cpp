// Laguna-S-2.1 config parse + name-map + (scaffolded) weight loaders. W1/W2.
//
// This TU owns three testable, CPU-buildable pieces:
//   1. ParseLagunaParams — resolves + validates every consumed field from the
//      HfConfig (typed + `raw`), including the NESTED per-layer-type
//      `rope_parameters` (the OLMo-3 gate hazard; laguna handles it explicitly)
//      and the per-layer VARIABLE Q-head array.
//   2. The Laguna GGUF `blk.N.*` name-map + the UD-Q4_K_XL per-tensor quant-mix
//      enumeration (pure string / metadata helpers; the byte-exact per-tensor
//      types are CONFIRMED against the real GGUF header at W4 when a checkpoint is
//      fetched — this increment records the EXPECTED unsloth "XL" mix).
//   3. LoadLaguna{ForCausalLMWeights,FromGguf} — parse params + build the seam,
//      then VT_CHECK(false, W3) on the actual device materialization (the towers
//      compose LANDED reuse; nothing here invents a kernel). Mirrors the ds4
//      loader philosophy: RESOLVE + account, DEFER materialize.
//
// Ground truth: poolside/Laguna-S-2.1/config.json (VERIFIED 2026-07-30, scope
// spec §1) + vllm/model_executor/models/laguna.py (MIRROR-vLLM) + the llama.cpp
// Poolside-fork `laguna` branch (GGUF name-map authority, scope spec §3).
#include "vllm/model_executor/models/laguna.h"

#include <cmath>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// --- raw (nlohmann::json) readers, mirroring deepseek_v4_weights.cpp:84-100 ---
int64_t RawInt(const nlohmann::json& doc, const char* key, int64_t fallback) {
  auto it = doc.find(key);
  return (it != doc.end() && it->is_number()) ? it->get<int64_t>() : fallback;
}
double RawDouble(const nlohmann::json& doc, const char* key, double fallback) {
  auto it = doc.find(key);
  return (it != doc.end() && it->is_number()) ? it->get<double>() : fallback;
}
bool RawBool(const nlohmann::json& doc, const char* key, bool fallback) {
  auto it = doc.find(key);
  return (it != doc.end() && it->is_boolean()) ? it->get<bool>() : fallback;
}
std::string RawString(const nlohmann::json& doc, const char* key,
                      const std::string& fallback) {
  auto it = doc.find(key);
  return (it != doc.end() && it->is_string()) ? it->get<std::string>() : fallback;
}
std::vector<int64_t> RawIntArray(const nlohmann::json& doc, const char* key) {
  std::vector<int64_t> out;
  auto it = doc.find(key);
  if (it != doc.end() && it->is_array())
    for (const auto& v : *it)
      if (v.is_number()) out.push_back(v.get<int64_t>());
  return out;
}

// Nested per-layer-type rope block reader. laguna.py walks
// `rope_parameters[layer_type]` (falling back to "full_attention"); we read the
// two known blocks explicitly. This is the field that KeyError-aborted OLMo-3 on
// an older transformers — laguna handles it, and our pin ships transformers 5.14.1.
const nlohmann::json* RopeBlock(const nlohmann::json& doc, const char* layer_type) {
  auto rp = doc.find("rope_parameters");
  if (rp == doc.end() || !rp->is_object()) return nullptr;
  auto lt = rp->find(layer_type);
  if (lt != rp->end() && lt->is_object()) return &*lt;
  return nullptr;
}

}  // namespace

LagunaParams ParseLagunaParams(const HfConfig& config) {
  const nlohmann::json& raw = config.raw;
  LagunaParams p;

  // --- shared geometry ---
  p.hidden_size = config.hidden_size > 0 ? config.hidden_size
                                         : RawInt(raw, "hidden_size", 0);
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
                              : RawInt(raw, "num_key_value_heads", 0);
  p.head_dim = config.head_dim > 0 ? config.head_dim : RawInt(raw, "head_dim", 0);
  p.intermediate_size = config.intermediate_size > 0
                            ? config.intermediate_size
                            : RawInt(raw, "intermediate_size", 0);
  p.rms_norm_eps = static_cast<float>(RawDouble(raw, "rms_norm_eps", 1e-6));
  p.tie_word_embeddings = RawBool(raw, "tie_word_embeddings", false);
  p.max_position_embeddings = config.max_position_embeddings > 0
                                  ? config.max_position_embeddings
                                  : RawInt(raw, "max_position_embeddings", 0);

  // --- interleaved attention ---
  p.sliding_window = config.sliding_window.value_or(RawInt(raw, "sliding_window", 0));
  p.layer_types = config.layer_types;  // HfConfig already parses the string array
  if (p.layer_types.empty()) {
    // Synthesize the 1:3 global:sliding pattern (layer % 4 == 0 -> global).
    for (int64_t l = 0; l < p.num_hidden_layers; ++l)
      p.layer_types.emplace_back(l % 4 == 0 ? "full_attention" : "sliding_attention");
  }
  p.num_attention_heads_per_layer =
      RawIntArray(raw, "num_attention_heads_per_layer");
  if (p.num_attention_heads_per_layer.empty()) {
    // Fall back to the 1:3 head pattern (global=base, sliding=1.5x base).
    for (int64_t l = 0; l < p.num_hidden_layers; ++l)
      p.num_attention_heads_per_layer.push_back(
          p.IsGlobalLayer(l) ? p.num_attention_heads
                             : p.num_attention_heads * 3 / 2);
  }

  // --- per-head softplus output gate ---
  p.per_head_output_gate = RawString(raw, "gating", "per-head") == "per-head";

  // --- MoE ---
  p.num_experts =
      config.num_experts > 0 ? config.num_experts : RawInt(raw, "num_experts", 0);
  p.num_experts_per_tok = config.num_experts_per_tok > 0
                              ? config.num_experts_per_tok
                              : RawInt(raw, "num_experts_per_tok", 0);
  p.moe_intermediate_size = config.moe_intermediate_size > 0
                                ? config.moe_intermediate_size
                                : RawInt(raw, "moe_intermediate_size", 0);
  p.shared_expert_intermediate_size =
      RawInt(raw, "shared_expert_intermediate_size", 0);
  p.norm_topk_prob = RawBool(raw, "norm_topk_prob", true);
  p.moe_routed_scaling_factor =
      static_cast<float>(RawDouble(raw, "moe_routed_scaling_factor", 1.0));
  // Router: sigmoid noaux_tc + e_score_correction_bias, UNGROUPED. laguna.py sets
  // use_grouped_topk=False + scoring_func="sigmoid" (the DeepSeek-V3 aux-loss-free
  // router MINUS the group step). The marketing "softplus router" is a misnomer;
  // softplus lives only in the attention out-gate.
  p.use_grouped_topk = false;
  p.has_e_score_correction_bias = true;
  p.mlp_only_layers = RawIntArray(raw, "mlp_only_layers");
  if (p.mlp_only_layers.empty()) p.mlp_only_layers = {0};  // layer 0 dense default

  // --- dual per-layer RoPE (nested rope_parameters) ---
  if (const nlohmann::json* full = RopeBlock(raw, "full_attention")) {
    p.rope_theta_full = RawDouble(*full, "rope_theta", 500000.0);
    // config.json (HF safetensors) uses factor 128 (1M ctx); the UD-Q4_K GGUF
    // uses factor 32 (262144 ctx). The GGUF loader OVERRIDES these from the GGUF
    // KV (laguna.rope.scaling.*) so the same-quant gate matches llama.cpp; this
    // typed path is the safetensors/NVFP4-oracle fallback.
    p.yarn_factor = RawDouble(*full, "factor", 32.0);
    p.yarn_orig_max_pos =
        RawInt(*full, "original_max_position_embeddings", 8192);
    p.yarn_beta_fast = RawDouble(*full, "beta_fast", 32.0);
    p.yarn_beta_slow = RawDouble(*full, "beta_slow", 1.0);
    // HF ships a precomputed `attention_factor` (== the llama.cpp mscale formula
    // output). Back out the raw yarn_attn_factor so LagunaYarnMscale reproduces it
    // for BOTH paths; if absent, default 1.0 (the GGUF value).
    if (const double af = RawDouble(*full, "attention_factor", -1.0); af > 0.0) {
      const double base = p.yarn_factor > 1.0 ? 1.0 + 0.1 * std::log(p.yarn_factor) : 1.0;
      p.yarn_attn_factor = base > 0.0 ? af / base : 1.0;
    } else {
      p.yarn_attn_factor = 1.0;
    }
    p.partial_rotary_factor_full = RawDouble(*full, "partial_rotary_factor", 0.5);
  }
  p.rotary_dim_full = static_cast<int64_t>(p.head_dim * p.partial_rotary_factor_full);
  if (const nlohmann::json* slide = RopeBlock(raw, "sliding_attention")) {
    p.rope_theta_sliding = RawDouble(*slide, "rope_theta", 10000.0);
    const double pr = RawDouble(*slide, "partial_rotary_factor", 1.0);
    p.rotary_dim_sliding = static_cast<int64_t>(p.head_dim * pr);
  } else {
    p.rotary_dim_sliding = p.head_dim;
  }

  // --- invariants ---
  VT_CHECK(p.hidden_size > 0 && p.num_hidden_layers > 0 && p.vocab_size > 0,
           "laguna: missing core geometry (hidden/layers/vocab)");
  VT_CHECK(p.num_key_value_heads > 0 && p.head_dim > 0,
           "laguna: missing GQA geometry (kv_heads/head_dim)");
  VT_CHECK(p.num_experts > 0 && p.num_experts_per_tok > 0,
           "laguna: missing MoE geometry (num_experts/top_k)");
  VT_CHECK(static_cast<int64_t>(p.layer_types.size()) == p.num_hidden_layers,
           "laguna: layer_types length must equal num_hidden_layers");
  VT_CHECK(static_cast<int64_t>(p.num_attention_heads_per_layer.size()) ==
               p.num_hidden_layers,
           "laguna: num_attention_heads_per_layer length must equal num_hidden_layers");
  VT_CHECK(p.rotary_dim_full > 0 && p.rotary_dim_full <= p.head_dim,
           "laguna: partial rotary_dim (full-attn YaRN) out of range");
  return p;
}

void ParseLagunaConfig(const HfConfig& config) {
  // The resolve itself IS the validation (throws on every unsupported field).
  (void)ParseLagunaParams(config);
}

// ─── GGUF name-map + UD-Q4_K_XL quant-mix (llama.cpp Poolside-fork `laguna`) ──
//
// The Laguna `blk.N.*` tensor names, mirroring the ds4 blk.N.* map + the standard
// llama.cpp MoE naming. Exposed as pure helpers so a name-map coverage checker
// (the check-dsv4-gguf-namemap.py pattern) can be extended to laguna WITHOUT a
// download. The per-tensor quant TYPE is read from the GGUF header at W4; the
// EXPECTED unsloth UD-Q4_K_XL "XL" mix is recorded in the table below.
std::string LagunaGgufAttnName(int64_t layer, const char* proj) {
  // proj in {attn_norm, attn_q, attn_k, attn_v, attn_output, attn_gate,
  //          attn_q_norm, attn_k_norm}. VERIFIED W4 from the real GGUF headers.
  return "blk." + std::to_string(layer) + "." + proj + ".weight";
}
std::string LagunaGgufMoeName(int64_t layer, const char* which) {
  // which in {ffn_norm, ffn_gate_inp, ffn_gate_exps, ffn_up_exps, ffn_down_exps,
  //   ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp} (+ exp_probs_b.bias, no
  //   ".weight" suffix). Dense layer 0: ffn_gate/ffn_up/ffn_down. VERIFIED W4.
  return "blk." + std::to_string(layer) + "." + which + ".weight";
}

// The VERIFIED UD-Q4_K_XL per-role quant mix (unsloth dynamic "XL"), read W4
// 2026-07-31 from the real GGUF tensor-info headers (814 tensors, split.count=3).
// Every type is ALREADY decoded in-tree (Q4_K/Q5_K/Q6_K/Q8_0 — cuda_quant_dot.cu +
// cpu_quant_dot.cpp) => ZERO new decode kernel for ANY tensor in the mix.
//   token_embd / output(lm_head)      -> Q6_K/Q8_0 (accuracy-critical)
//   attn_q/k/v/output/gate            -> Q8_0 (VERIFIED — all attn linears Q8_0)
//   attn_q_norm/k_norm/attn_norm      -> F32 (per-head QK-RMSNorm + pre-attn norm)
//   ffn_gate_inp (router)             -> F32 [H,256]; exp_probs_b.bias -> F32 [256]
//   ffn_{gate,up}_exps (256 experts)  -> Q4_K (the bulk)
//   ffn_down_exps                     -> Q5_K (UD "richer down-proj" rule)
//   ffn_{gate,up,down}_shexp (shared) -> Q8_0
//   ffn_norm                          -> F32
// GGUF ne-order is [in, out] (reverse of torch [out, in]); OwnGgufQuantBlocks +
// vt::MatmulBT consume the on-disk [N,K] block layout with no transpose.

LagunaWeights LoadLagunaForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  (void)shards;
  LagunaWeights w;
  w.params = ParseLagunaParams(config);
  // TODO(W3): materialize the BF16 towers + build the dual RoPE caches
  //   (BuildOlmo3YarnCache for full-attn YaRN-64; plain get_rope for sliding-128).
  //   Safetensors BF16 is 235 GiB — MEMORY-INFEASIBLE on one GB10; this path is
  //   for the NVFP4 behavior oracle + structural completeness only.
  VT_CHECK(false,
           "laguna: safetensors weight materialization is a W3 residual "
           "(structural bring-up + name-map only this increment). The GGUF "
           "UD-Q4_K_XL keep-quant path (LoadLagunaFromGguf) is the single-GB10 "
           "vehicle. See .agents/specs/laguna-s21-w1w2-2026-07-30.md.");
  return w;
}

LagunaWeights LoadLagunaFromGguf(const GgufFile& gguf, const HfConfig& config) {
  (void)gguf;
  LagunaWeights w;
  w.params = ParseLagunaParams(config);
  // TODO(W3): wire the ds4 gguf_keep_quant tower materialization + the Laguna
  //   blk.N.* name-map (LagunaGgufAttnName/LagunaGgufMoeName above). The 256
  //   routed-expert Q4_K blocks stay COMPRESSED (OwnGgufQuantBlocks — the memory
  //   enabler); the small norm/router/gate/embed tensors dequant. ZERO new decode
  //   kernel (Q4_K/Q5_K/Q6_K/Q8_0 all landed). Then extend the name-map coverage
  //   checker to laguna.
  VT_CHECK(false,
           "laguna: GGUF keep-quant tower materialization is a W3 residual "
           "(name-map + quant-mix scaffolded this increment; no new decode "
           "kernel needed). See .agents/specs/laguna-s21-w1w2-2026-07-30.md.");
  return w;
}

}  // namespace vllm
