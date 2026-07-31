// Laguna-S-2.1 (`LagunaForCausalLM` / `model_type=laguna`) — the ADDITIVE model
// TU skeleton for the Poolside Laguna bring-up (W1/W2). This header defines the
// config parse + the (scaffolded) weight layout + the forward / KV-cache seams;
// the genuinely-NEW primitives (per-head softplus attention output gate, the
// UNGROUPED sigmoid-noaux router, the per-layer VARIABLE Q-head runner wiring)
// are STUBBED with precise `TODO(W3/W4)` port markers and a `VT_CHECK(false, ...)`
// forward so the TU BUILDS but a forward LOUDLY reports the pending brick — never
// a silent wrong answer. The full forward + the strict dual-oracle gate are NAMED
// residuals (W3/W4). See `.agents/specs/laguna-s21-scope-2026-07-30.md` +
// `.agents/specs/laguna-s21-w1w2-2026-07-30.md`.
//
// ─── ~85-90% REUSE, ZERO new compute kernel (the extensibility payoff) ────────
// Every heavy component Laguna needs already exists in-tree from a prior model.
// The forward COMPOSES the reuse; only three small HOST ops are genuinely new.
//
//   COMPONENT                         REUSE SOURCE (file:line ported FROM)
//   ─────────────────────────────────────────────────────────────────────────
//   Q4_K/Q5_K/Q6_K/Q8_0 keep-quant    src/vt/cuda/cuda_quant_dot.cu DotQ4K /
//     decode  (feared "big kernel")     DotSuperblock<kQ4_K>; CPU cpu_quant_dot.cpp
//                                       VecDotQ4_K — ALREADY LANDED for ds4. ZERO new.
//   GGUF keep-quant load + name-map    src/vllm/model_executor/model_loader/
//                                       gguf_keep_quant.cpp + deepseek_v4_weights.cpp
//                                       LoadDeepseekV4FromGguf (the blk.N.* mirror);
//                                       qwen3_5_gguf_weights.cpp (dequant small tensors)
//   MoE: sigmoid noaux_tc +            src/vllm/model_executor/models/
//     e_score_correction_bias +         deepseek_v2.cpp:340-365 (RunMoeBlock, noaux
//     shared expert + routed_scaling    sigmoid + bias) + deepseek_v2_weights.cpp:186-320
//                                       (has_e_score_correction_bias, routed_scaling).
//                                       NEW = the UNGROUPED variant (drop ds2's group step).
//   Interleaved sliding-window(512)    src/vllm/model_executor/models/gemma3.cpp +
//     attn (1:3 global:sliding)          gemma3_registry.cpp:103-121 (layer_types /
//                                       is_sliding, window masked at the kernel).
//   Dual per-layer RoPE (YaRN full-    src/vllm/model_executor/models/
//     attn / plain sliding)              olmo2_weights.cpp:198-217 BuildOlmo3YarnCache
//                                       (get_rope yarn) — build TWO caches, select by
//                                       layer_types.
//   Partial rotary (full-attn dim 64)  src/vllm/model_executor/models/phi_weights.cpp
//                                       (rotary_dim = head_dim * partial_rotary_factor).
//   Per-head SOFTPLUS attn out-gate    NEW small op. Softplusf exists
//                                       (gemma4_audio.cpp:20). g_proj: hidden->num_heads,
//                                       softplus fp32, broadcast over head_dim.
//   Variable per-layer Q-head count    NEW runner wiring, extends the Gemma-4
//     (48 global / 72 sliding)           heterogeneous-per-layer-KV path (G1b, task #148).
//   RMSNorm / SwiGLU dense MLP (layer  shared vt::RmsNorm + MlpGateUpMethodBase +
//     0) / GQA / embed / untied head    FusedChain(kFusedAddRmsNorm). Config-drive only.
//
// ─── ORACLE (W1 decision) ────────────────────────────────────────────────────
// vLLM has a NATIVE `vllm/model_executor/models/laguna.py` (landed before our pin
// 555967922 / 0.26.0.dev0), so MIRROR-vLLM applies and the pinned oracle is
// EXPECTED to construct + run the config (config-constructs check per
// oracle-gateability-model-runs-not-config-constructs). Dual-oracle plan (ds4
// pattern): (a) vLLM on poolside/Laguna-S-2.1-NVFP4/-FP8 (fits GB10 119 GiB; BF16
// 235 GiB does NOT) for a behavior/coherence golden; (b) llama.cpp Poolside-fork
// `laguna` branch on the SAME UD-Q4_K GGUF for the token-exact greedy gate. No
// 73 GB download this increment — that is the W4 follow-on.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"          // PagedKvCache, ForwardLogits
#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"  // CommonAttentionMetadata
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"

namespace vllm {

class SafetensorsFile;
class GgufFile;

// Every Laguna config field the loader/forward consume, resolved ONCE from the
// HfConfig. Laguna keys (`num_experts`, `moe_routed_scaling_factor`,
// `num_attention_heads_per_layer`, the nested `rope_parameters`, ...) are mostly
// NOT on the typed HfConfig struct, so most are read from `config.raw`. Values in
// the comments are the shipped `poolside/Laguna-S-2.1/config.json` (VERIFIED
// 2026-07-30, see the scope spec §1).
struct LagunaParams {
  // --- shared geometry ---
  int64_t hidden_size = 0;          // 3072
  int64_t num_hidden_layers = 0;    // 48
  int64_t vocab_size = 0;           // 100352
  int64_t num_attention_heads = 0;  // 48 (BASE / global layers)
  int64_t num_key_value_heads = 0;  // 8  (GQA)
  int64_t head_dim = 0;             // 128
  int64_t intermediate_size = 0;    // 12288 (dense MLP, layer 0 only)
  float rms_norm_eps = 1e-6f;       // 1e-6
  bool tie_word_embeddings = false; // false (untied lm_head)
  int64_t max_position_embeddings = 0;  // 1048576 (1M)

  // --- interleaved attention (1:3 global:sliding) ---
  int64_t sliding_window = 0;  // 512
  // Per-layer attention regime: "full_attention" (global) / "sliding_attention".
  // 48 entries, 1:3 pattern -> 12 global + 36 sliding.
  std::vector<std::string> layer_types;
  // Per-layer Q-head COUNT (global=48, sliding=72). KV heads = 8 both, head_dim
  // 128 both -> GQA group 6 (global) / 9 (sliding). This is the riskiest bit:
  // the runner does per-layer KV head_dim (Gemma-4) but not per-layer Q-head COUNT.
  std::vector<int64_t> num_attention_heads_per_layer;
  bool IsGlobalLayer(int64_t l) const {
    return l < static_cast<int64_t>(layer_types.size()) &&
           layer_types[static_cast<size_t>(l)] == "full_attention";
  }
  // Per-layer VARIABLE Q-head wiring (the riskiest bit). KV heads + head_dim are
  // uniform; only the Q-head COUNT (and thus the GQA group + q_proj width) vary.
  int64_t QHeadsForLayer(int64_t l) const {
    if (l >= 0 && l < static_cast<int64_t>(num_attention_heads_per_layer.size()))
      return num_attention_heads_per_layer[static_cast<size_t>(l)];
    return num_attention_heads;  // fallback = base (global) count
  }
  int64_t QDimForLayer(int64_t l) const { return QHeadsForLayer(l) * head_dim; }
  int64_t GqaGroupForLayer(int64_t l) const {
    return num_key_value_heads > 0 ? QHeadsForLayer(l) / num_key_value_heads : 0;
  }
  // The dual per-layer RoPE regime + finite window are keyed off layer type.
  int64_t RotaryDimForLayer(int64_t l) const {
    return IsGlobalLayer(l) ? rotary_dim_full : rotary_dim_sliding;
  }
  int64_t WindowForLayer(int64_t l) const {
    return IsGlobalLayer(l) ? 0 : sliding_window;  // 0 => full causal
  }

  // --- per-head softplus attention OUTPUT gate (all layers) ---
  // gate = softplus(g_proj(x).float()).type_as(attn); broadcast over head_dim.
  // g_proj: [num_heads_of_layer, hidden]. NEW small op (Softplusf exists).
  bool per_head_output_gate = true;  // config `gating == "per-head"`

  // --- per-head QK-RMSNorm (VERIFIED W4 from the real GGUF) ---------------------
  // The GGUF ships `blk.N.attn_q_norm.weight` / `attn_k_norm.weight`, both F32
  // [head_dim] — Laguna applies an RMSNorm to EACH head's q and k vector before
  // RoPE (the Qwen3/OLMo-2 QK-norm pattern). The W1-W3 scope MISSED this (there is
  // no `qk_layernorm` flag in config.json; it surfaced only in the GGUF tensor
  // map), so the W3 forward omitted it. VERIFIED 2026-07-31 by dumping the GGUF
  // tensor-info headers (blk.*.attn_q_norm/attn_k_norm F32 [128]).
  bool has_qk_norm = true;

  // --- MoE (layers 1..47; layer 0 is dense, mlp_only_layers=[0]) ---
  int64_t num_experts = 0;             // 256
  int64_t num_experts_per_tok = 0;     // 10 (top-10)
  int64_t moe_intermediate_size = 0;   // 1024
  int64_t shared_expert_intermediate_size = 0;  // 1024 (1 shared expert)
  bool norm_topk_prob = true;          // renormalize the top-k weights
  float moe_routed_scaling_factor = 2.5f;  // applied to the routed OUTPUT
  // Router = sigmoid noaux_tc + e_score_correction_bias (DeepSeek-V3 style, NOT
  // softplus — softplus lives ONLY in the attention out-gate above). The NEW
  // wiring vs ds2/glm4 is the UNGROUPED variant (use_grouped_topk == False).
  bool use_grouped_topk = false;
  bool has_e_score_correction_bias = true;
  // mlp_only_layers[]: dense-MLP layer indices (everything else is MoE). = {0}.
  std::vector<int64_t> mlp_only_layers;
  bool IsDenseLayer(int64_t l) const {
    for (int64_t d : mlp_only_layers)
      if (d == l) return true;
    return false;
  }

  // --- dual per-layer RoPE (by layer-type) ---
  // full_attention: YaRN — theta 500000, factor 128, orig_max_pos 8192,
  //   beta_fast 32 / beta_slow 1, attention_factor(mscale) 1.4852030263919618,
  //   partial_rotary_factor 0.5 -> rotary_dim 64 of 128.
  double rope_theta_full = 500000.0;
  // YaRN `factor` = ctx_len / orig_ctx. The unsloth UD-Q4_K GGUF is built for a
  // 262144 context (factor 32 = 262144/8192), NOT the HF config.json 1M context
  // (factor 128). For the same-quant token-exact gate vs llama.cpp on THIS GGUF,
  // the GGUF value (32) is authoritative — VERIFIED W4 from laguna.rope.scaling.*.
  double yarn_factor = 32.0;
  int64_t yarn_orig_max_pos = 8192;
  double yarn_beta_fast = 32.0;
  double yarn_beta_slow = 1.0;
  // llama.cpp `yarn_attn_factor` (default 1.0). The FULL mscale llama.cpp applies is
  //   mscale = yarn_attn_factor * (1 + 0.1*ln(factor))     [rope_yarn, ext_factor!=0]
  // which reproduces HF's precomputed `attention_factor` too (factor 128 -> 1.48520,
  // factor 32 -> 1.34657). See LagunaYarnMscale. VERIFIED W4 (laguna.rope.scaling.
  // yarn_attn_factor = 1.0).
  double yarn_attn_factor = 1.0;
  double partial_rotary_factor_full = 0.5;  // rotary_dim_full = 64
  int64_t rotary_dim_full = 64;
  // sliding_attention: plain RoPE — theta 10000, full 128-dim rotary.
  double rope_theta_sliding = 10000.0;
  int64_t rotary_dim_sliding = 128;
};

// Resolve + validate LagunaParams from an HfConfig (throws on unsupported).
LagunaParams ParseLagunaParams(const HfConfig& config);

// Per-family config hook (registry `parse_config`).
void ParseLagunaConfig(const HfConfig& config);

// ─── Weight layout (SCAFFOLD) ────────────────────────────────────────────────
// One Laguna self-attention block. q_proj is per-layer VARIABLE width
// (num_heads_of_layer * head_dim); k/v are fixed (num_key_value_heads * head_dim).
// g_proj is the per-head softplus output gate (num_heads_of_layer, hidden).
struct LagunaAttnWeights {
  OwnedTensor q_proj;   // raw-NK [num_heads_l*Dh, H]   (GGUF blk.N.attn_q)
  OwnedTensor k_proj;   // raw-NK [Hkv*Dh, H]           (GGUF blk.N.attn_k)
  OwnedTensor v_proj;   // raw-NK [Hkv*Dh, H]           (GGUF blk.N.attn_v)
  OwnedTensor o_proj;   // raw-NK [H, num_heads_l*Dh]   (GGUF blk.N.attn_output)
  OwnedTensor g_proj;   // raw-NK [num_heads_l, H]  (per-head softplus out-gate; GGUF attn_gate)
  OwnedTensor q_norm;   // f32 [head_dim]  per-head QK-RMSNorm (GGUF blk.N.attn_q_norm) — VERIFIED W4
  OwnedTensor k_norm;   // f32 [head_dim]  per-head QK-RMSNorm (GGUF blk.N.attn_k_norm) — VERIFIED W4
};

// Laguna dense SwiGLU MLP (layer 0 only). silu(gate)*up -> down (separate tensors).
struct LagunaMlpWeights {
  // GGUF ships SEPARATE dense FFN tensors (blk.0.ffn_gate/up/down.weight); the
  // W1-W3 scaffold assumed a merged gate_up — corrected to the real name-map (W4).
  OwnedTensor gate_proj;  // raw-NK [I, H]  (ffn_gate)
  OwnedTensor up_proj;    // raw-NK [I, H]  (ffn_up)
  OwnedTensor down_proj;  // raw-NK [H, I]  (ffn_down)
};

// Laguna MoE block (layers 1..47): sigmoid noaux_tc router (+ e_score_correction_
// bias) UNGROUPED top-10 of 256 + 1 shared expert, routed_scaling 2.5 on output.
struct LagunaMoeWeights {
  OwnedTensor router;                 // raw-NK [num_experts, H]  (GGUF ffn_gate_inp, F32)
  OwnedTensor e_score_correction_bias;  // f32 [num_experts]      (GGUF exp_probs_b.bias)
  // 256 routed experts + 1 shared expert. The GGUF ships SEPARATE gate/up tensors
  // (VERIFIED W4: ffn_gate_exps Q4_K [E,moe_I,H] + ffn_up_exps Q4_K + ffn_down_exps
  // Q5_K); the W1-W3 scaffold assumed a MERGED gate_up — corrected here. The
  // keep-quant expert slabs stay COMPRESSED for the GGUF path (ds4 kStackedExpert),
  // consumed per-selected-expert via a row-slice keep-quant GEMM.
  OwnedTensor experts_gate;   // grouped [E, moe_I, H]  (ffn_gate_exps, Q4_K)
  OwnedTensor experts_up;     // grouped [E, moe_I, H]  (ffn_up_exps,   Q4_K)
  OwnedTensor experts_down;   // grouped [E, H, moe_I]  (ffn_down_exps, Q5_K)
  OwnedTensor shared_gate;    // [moe_I, H]  (ffn_gate_shexp, Q8_0)
  OwnedTensor shared_up;      // [moe_I, H]  (ffn_up_shexp,   Q8_0)
  OwnedTensor shared_down;    // [H, moe_I]  (ffn_down_shexp, Q8_0)
};

// One Laguna decoder layer.
struct LagunaLayerWeights {
  OwnedTensor input_norm;       // [H]  pre-attn RMSNorm
  OwnedTensor post_attn_norm;   // [H]  pre-MLP/MoE RMSNorm
  LagunaAttnWeights attn;
  bool is_dense = false;        // layer 0
  LagunaMlpWeights mlp;         // valid iff is_dense
  LagunaMoeWeights moe;         // valid iff !is_dense
};

// The full Laguna model weights (SCAFFOLD — accounting + name-map only this
// increment; device materialization is a named W3 residual).
struct LagunaWeights {
  LagunaParams params;
  OwnedTensor embed;    // [V, H]
  OwnedTensor norm;     // [H]  final RMSNorm
  OwnedTensor lm_head;  // [V, H]  untied
  // Dual per-layer RoPE caches (OLMo-3 BuildOlmo3YarnCache pattern): one YaRN
  // (full-attn, 64-dim) + one plain (sliding, 128-dim). Built at load.
  OwnedTensor rope_cos_sin_yarn_full;  // bf16 [rows, 64]
  OwnedTensor rope_cos_sin_plain_slide;  // bf16 [rows, 128]
  std::vector<LagunaLayerWeights> layers;
  int64_t accounted_tensors = 0;  // W2 accounting-pass count
};

// Safetensors loader (BF16 checkpoint; MEMORY-INFEASIBLE on one GB10 at 235 GiB,
// present for structural completeness + the NVFP4 behavior path). Encodes the
// Laguna name-map + the W2 accounting pass.
LagunaWeights LoadLagunaForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

// GGUF keep-quant loader (the single-GB10 vehicle: unsloth/Laguna-S-2.1-GGUF
// UD-Q4_K_XL, ~73.4 GiB). Reuses the ds4 gguf_keep_quant path + a Laguna blk.N.*
// name-map; the MoE expert blocks stay COMPRESSED (Q4_K/Q5_K/Q6_K/Q8_0 all
// already decoded — ZERO new kernel). Materialization is a named W3 residual.
LagunaWeights LoadLagunaFromGguf(const GgufFile& gguf, const HfConfig& config);

// The Laguna forward. STUB (W3/W4): composes the reuse (variable-Q-head GQA +
// interleaved sliding-window mask + dual per-layer RoPE + per-head softplus
// out-gate; dense MLP at layer 0; ungrouped sigmoid-noaux MoE at layers 1..47;
// untied lm_head). Both entrypoints VT_CHECK(false, ...) so a forward LOUDLY
// reports the pending brick.
class LagunaModel {
 public:
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const LagunaWeights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});

  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const LagunaWeights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});
};

// KV-cache spec builder. One FULL-ATTENTION KV group over all layers; the
// interleaved sliding-window layers are masked at the attention kernel (window
// 512), NOT by a smaller SlidingWindowSpec cache — the gemma3 topology the
// shape-agnostic runner already handles (gemma3_registry.cpp:103-121).
v1::KVCacheConfig MakeLagunaKVCache(const HfConfig& config, int block_size,
                                    int num_blocks);

}  // namespace vllm
