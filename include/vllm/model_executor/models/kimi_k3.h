// Kimi K3 (`KimiK3ForConditionalGeneration`) — the ADDITIVE model TU skeleton for
// the Kimi-K3 structural bring-up (`CLAIM-KIMI-K3-W2-W5`, DERIVE-AND-SHIP). This
// header defines the nested-config parse (`text_config` = the KimiLinear KDA+MLA
// +MoE hybrid, `vision_config` = MoonViT-V2), the text-backbone structural weight
// map (grounded 1:1 in the pinned `kimi_linear.py`), and the forward/KV-cache
// seams. The forward REFUSES-by-name (`VT_CHECK(false, ...)`) exactly like
// `deepseek_v4.{h,cpp}` — the TU BUILDS and the config/loader structure is
// unit-testable, but a forward LOUDLY reports the pending brick rather than
// returning a silent wrong answer.
//
// ─── DERIVE-AND-SHIP HONESTY (up front) ──────────────────────────────────────
// Kimi K3 is 2.8T params / ~1.56 TB (MXFP4) and does NOT fit ONE GB10 (119 GiB,
// off ~12×), AND `KimiK3ForConditionalGeneration` is NOT registered in the pinned
// vLLM oracle (`555967922`, 0.26.0.dev0 — K3 released 2026-07-27, after the pin).
// So there is NO on-box e2e oracle golden, like the beyond-vLLM CUDA-arch bricks.
// The arch scalars below are DERIVED from the HF `config.json` fetch + the literal
// text backbone class `KimiLinearForCausalLM`; AttnRes, MXFP8 activations, the
// exact ViT size, and the K3 multimodal-wrapper weight PREFIX are report-only /
// post-pin UNCONFIRMED (see `.agents/specs/kimi-k3.md` §0, §6). A green CPU build
// is DERIVED+BUILD-VERIFIED, NOT execution evidence.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides, @ pin 555967922) ────────
//   OURS                             <-  UPSTREAM
//   KimiK3TextParams                 <-  transformers_utils/configs/kimi_linear.py:11-148
//                                        (KimiLinearConfig; is_mla/is_moe/is_kda_layer)
//                                        + the fetched moonshotai/Kimi-K3 config.json
//                                        text_config scalars (DERIVED, not byte-verified)
//   KimiK3VisionParams               <-  fetched config.json vision_config (MoonViT-V2) —
//                                        PARTIAL, W7 residual (kimi_k25_vit.py reference)
//   ParseKimiK3Params                <-  configs/kimi_linear.py:56-148 (field descent)
//   EnumerateKimiK3TextBackboneTensors <- kimi_linear.py:288-378 (KimiDecoderLayer),
//                                        :180-277 (KimiMLAAttention), :104-168 (KimiMoE),
//                                        :460-554 (load_weights) + kimi_gdn_linear_attn.py
//                                        :102-226 (KDA q/k/v/f_a/f_b/b/g_a/g_b/conv/o_norm)
//   LoadKimiK3ForConditionalGenerationWeights <- kimi_linear.py:641-646 (AutoWeightsLoader)
//   MakeKimiK3KVCache                <-  kimi_linear.py:600-633 (kda_state) — STUB (hybrid)
//   KimiK3Model::Forward             <-  kimi_linear.py:426-458, 587-598 — REFUSE-by-name
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"          // PagedKvCache, ForwardLogits
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"  // CommonAttentionMetadata
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"

namespace vllm {

class SafetensorsFile;

// The KimiLinear text backbone K3 wraps. Fields resolve from the wrapper's
// nested `text_config` (LoadHfConfig already lifts the shared scalars — hidden_size
// etc. — into the typed HfConfig; the KimiLinear-specific keys live in
// `config.raw["text_config"]`). Values in comments are the DERIVED K3 scale
// (config.json fetch), NOT byte-verified.
struct KimiK3TextParams {
  // --- shared geometry ---
  int64_t hidden_size = 0;          // 7168
  int64_t num_hidden_layers = 0;    // 93
  int64_t vocab_size = 0;           // 163840
  int64_t num_attention_heads = 0;  // 96
  int64_t intermediate_size = 0;    // dense-MLP width (first_k_dense_replace layers)
  float rms_norm_eps = 1e-6f;
  bool tie_word_embeddings = false;
  int64_t max_position_embeddings = 0;
  int64_t num_nextn_predict_layers = 0;  // MTP tail; loader SKIPS `mtp.*`
  double rope_theta = 10000.0;

  // --- MLA geometry (DeepSeek-V2/V3 dims — the geometry we already landed) ---
  // kimi_linear.py:119-127 is_mla; K3 = kv_lora 512 / q_lora 1536 / nope 128 /
  // rope 64. NOTE: the pinned KimiLinear asserts q_lora_rank is None
  // (kimi_linear.py:215) and mla_use_nope True (:214); K3's q_lora_rank=1536 takes
  // the DeepSeek-V3 q-LoRA branch (q_a_proj/q_a_layernorm/q_b_proj) our MLA supports.
  int64_t kv_lora_rank = 0;      // 512
  int64_t q_lora_rank = 0;       // 1536 (0 => direct q_proj, KimiLinear default)
  int64_t qk_nope_head_dim = 0;  // 128
  int64_t qk_rope_head_dim = 0;  // 64
  int64_t v_head_dim = 0;        // 128
  bool mla_use_nope = false;

  // --- MoE (kimi_linear.py:104-168) — noaux_tc-family sigmoid router ---
  int64_t num_experts = 0;            // 896
  int64_t num_experts_per_token = 0;  // 16
  int64_t num_shared_experts = 0;     // 2
  int64_t moe_intermediate_size = 0;  // 3072
  int64_t first_k_dense_replace = 0;
  int64_t moe_layer_freq = 1;
  double routed_scaling_factor = 1.0;
  bool moe_renormalize = true;
  int64_t num_expert_group = 1;
  int64_t topk_group = 1;
  bool use_grouped_topk = true;
  // "sigmoid" | "softmax" (kimi_linear.py:96 assert). K3 = sigmoid (noaux_tc).
  std::string moe_router_activation_func = "sigmoid";

  // --- KDA (linear_attn_config, kimi_gdn_linear_attn.py:109-119) ---
  // is_kda_layer(l) == (l+1) in kda_layers (kimi_linear.py:144-148). K3 = 69 KDA +
  // 24 MLA of 93. num_heads 96 / head_dim 128 / short_conv 4.
  std::vector<int64_t> kda_layers;
  std::vector<int64_t> full_attn_layers;
  int64_t kda_num_heads = 0;               // 96
  int64_t kda_head_dim = 0;                // 128
  int64_t kda_short_conv_kernel_size = 0;  // 4
  bool has_linear_attn_config = false;

  // (l+1) in kda_layers; mirrors KimiLinearConfig.is_kda_layer (kimi_linear.py:144).
  bool is_kda_layer(int64_t layer_idx) const;
  // is_moe && layer_idx >= first_k_dense_replace && layer_idx % moe_layer_freq == 0
  // (kimi_linear.py:328-333). num_experts>0 => this is a MoE arch (is_moe :130).
  bool is_moe_layer(int64_t layer_idx) const;
};

// MoonViT-V2 tower — PARTIAL. Genuinely-NEW vision tower (W7 residual), scoped
// from the K2.5 tower `kimi_k25_vit.py` + our Qwen3-VL tower; the real forward is
// in the post-pin `modeling_kimi_k3.py` (NOT available). Only the scalars needed
// to RESOLVE the config are parsed; the tower load/forward is NOT-YET-BUILDABLE.
struct KimiK3VisionParams {
  bool present = false;
  int64_t patch_size = 0;        // 14
  int64_t num_hidden_layers = 0; // 27
  int64_t hidden_size = 0;
};

// Every Kimi-K3 config field the loader/forward consume, resolved ONCE from the
// wrapper HfConfig. `text` descends `text_config`; `vision` descends
// `vision_config`; the quant fields descend `quantization_config`.
struct KimiK3Params {
  KimiK3TextParams text;
  KimiK3VisionParams vision;

  // compressed-tensors quant. K3 ships `mxfp4-pack-quantized` (num_bits 4, group
  // 32, e8m0). MXFP4 is a NEW quant path for us (we have NVFP4 group-16 only);
  // it is SHARED with the DeepSeek-V4 MegaMoE MXFP4 scope — NOT implemented here.
  // When set, the loader throws NOT-YET-BUILDABLE citing that shared row.
  std::string quant_method;   // "compressed-tensors" (or empty)
  std::string quant_format;   // "mxfp4-pack-quantized" (or empty)
  bool is_mxfp4 = false;
};

// Resolve + validate KimiK3Params from a wrapper HfConfig. Pure/host —
// unit-testable without a checkpoint (config-descent gate). Throws with a precise
// message on a missing required field or a value this bring-up cannot represent.
KimiK3Params ParseKimiK3Params(const HfConfig& config);

// Per-family config hook (registry `parse_config`): resolves + validates and
// throws on anything unsupported.
void ParseKimiK3Config(const HfConfig& config);

// The structural weight name-map of the KimiLinear TEXT backbone (grounded 1:1 in
// the pinned `kimi_linear.py` + `kimi_gdn_linear_attn.py`), prefix-parameterized.
// `prefix` is the wrapper's language-model prefix; default "" == the standalone
// `KimiLinearForCausalLM` names (the VERIFIABLE, pinned-source-grounded part).
//
// HONESTY: the K3 MULTIMODAL-WRAPPER prefix + the MoonViT-V2 vision-tower tensors
// are post-pin UNCONFIRMED and are NOT enumerated here — that is a NOT-YET-BUILDABLE
// residual (W7 / pin advance). This enumeration is the DERIVED, testable structure
// of the 93-layer KDA/MLA hybrid + 896-expert MoE, NOT a claim that a real K3
// checkpoint loads.
std::vector<std::string> EnumerateKimiK3TextBackboneTensors(
    const KimiK3TextParams& text, const std::string& prefix = "");

// Whole Kimi-K3 weights. STRUCTURAL SCAFFOLDING: carries the resolved params + the
// loader's accounting result. Heavy tensor MATERIALIZATION (MXFP4 experts, the KDA
// low-rank-decay towers, the MoonViT-V2 tower) is the named NOT-YET-BUILDABLE
// residual — see the loader.
struct KimiK3Weights {
  KimiK3Params params{};
  // How many enumerated text-backbone tensors were present in the shards.
  int64_t accounted_tensors = 0;
  // Total enumerated text-backbone tensors (structural size of the hybrid).
  int64_t enumerated_tensors = 0;
};

// Load `KimiK3ForConditionalGeneration` safetensors. NOT-YET-BUILDABLE for a REAL
// K3 checkpoint: it is MXFP4 (throws, deferring to the shared MXFP4 row) and its
// multimodal-wrapper prefix + vision tower are post-pin unconfirmed. The structural
// text-backbone accounting pass over a (hypothetical bf16) checkpoint is exercised
// for build/structure verification only.
KimiK3Weights LoadKimiK3ForConditionalGenerationWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

// The Kimi-K3 forward. REFUSE-by-name (both entrypoints VT_CHECK(false, ...)): the
// scaled 93-layer KDA/MLA hybrid + 896-expert MoE forward composes primitives that
// are NOT wired here — the KDA kernel delta (Kimi-Linear row), MXFP4 (shared
// DeepSeek-V4 row), and MoonViT-V2 (W7). A forward LOUDLY reports the pending
// brick. See `.agents/specs/kimi-k3.md` §5 (W4/W5/W7).
class KimiK3Model {
 public:
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const KimiK3Weights& weights,
      vt::Queue& queue, const std::vector<int32_t>& logits_indices = {},
      const vllm::TensorParallel* tp = nullptr);

  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const KimiK3Weights& weights,
      vt::Queue& queue, const std::vector<int32_t>& logits_indices = {},
      const vllm::TensorParallel* tp = nullptr);
};

// KV-cache spec builder. STUB (hybrid residual): K3's TRUE KV topology is a
// MULTI-group hybrid — MLA latent-KV for the 24 full-attn layers + KDA conv/
// recurrent state for the 69 linear-attn layers (kimi_linear.py:600-633,
// kda_state). That hybrid geometry is not represented here; we emit ONE placeholder
// MLA group sized to the compressed latent + rope so the arch RESOLVES. The real
// hybrid topology defers to the Kimi-Linear KDA row. Never exercised (forward
// refuses).
v1::KVCacheConfig MakeKimiK3KVCache(const HfConfig& config, int block_size,
                                    int num_blocks);

}  // namespace vllm
