// Gemma-4 text backbone (`Gemma4ForConditionalGeneration`, the language_model
// stack of `unsloth/gemma-4-E4B-it`) — MODEL-GEMMA4 G1. A NEW-primitive dense
// decoder that reuses the Gemma vocabulary but diverges from Gemma-2/3 in FIVE
// load-bearing ways (all grounded in vllm/model_executor/models/gemma4.py @
// 555967922 / 0.26.0.dev0, cross-checked against gemma4.py @ e24d1b24):
//
//   1. PLAIN RMSNorm, NOT GemmaRMSNorm. gemma4.py imports
//      `from vllm.model_executor.layers.layernorm import RMSNorm` (:45) and uses
//      RMSNorm(...) at EVERY norm site — so the weight is applied as `x_norm * w`
//      (NOT the Gemma `(1+w)` offset gemma2/3 use). vt::RmsNormArgs{eps,false}.
//   2. Per-Layer Embeddings (PLE). A second embedding table
//      `embed_tokens_per_layer` [vocab, hidden_size_per_layer_input*num_layers]
//      (gemma4.py:999-1004), a `per_layer_model_projection` [total_ple, hidden]
//      (:1015-1023), a shared `per_layer_projection_norm` (:1025-1028), and per
//      decoder layer a gate/projection/norm (:680-704) that fuses a per-token,
//      per-layer embedding into the residual AFTER the MLP block (:753-761).
//   3. YOCO KV-sharing. The last `num_kv_shared_layers` (18 of 42 for E4B) do
//      NOT cache their own K/V; they attend to the last non-shared layer of the
//      same attention type (gemma4.py:463-489, forward :535-548). For E4B every
//      shared sliding layer targets layer 22, every shared full layer targets 23.
//   4. Heterogeneous head_dim. full_attention layers use global_head_dim=512,
//      sliding_attention layers head_dim=256 (gemma4.py:572-578), with a common
//      num_key_value_heads=2 and scaling=1.0 (Q/K norms carry the scale, :408-411).
//   5. Proportional partial-RoPE on full layers. rope_type "proportional"
//      (gemma4_rope.py) with partial_rotary_factor=0.25 → rotary_dim=128 of a 512
//      head, inv_freq denominator = head_dim (not rotary_dim), non-rotated pairs
//      zero-padded to identity; sliding layers use standard rope (theta 1e4,
//      full 256 rotary). Plus GeGLU MLP (gelu_pytorch_tanh), a sqrt(hidden) embed
//      normalizer, a per-layer learned `layer_scalar` [1] multiply (:707,:765),
//      tied lm_head, and a final logit soft-cap = 30.0.
//
//   E4B disables the harder Gemma-4 primitives: enable_moe_block=false (NO
//   Gemma-4 MoE router / per_expert_scale), attention_k_eq_v=false (NO k_eq_v),
//   use_double_wide_mlp=false. Those stay OUT of this bring-up (recorded in the
//   spec as the follow-on for the ≥12B MoE Gemma-4 checkpoints).
//
// Config (unsloth/gemma-4-E4B-it text_config): hidden 2560, 42L, GQA 8/2,
// head_dim 256 / global_head_dim 512, intermediate 10240,
// hidden_size_per_layer_input 256, num_kv_shared_layers 18, sliding_window 512,
// final_logit_softcapping 30.0, vocab 262144, tie_word_embeddings true.
//
// Numeric contract mirrors the bf16 dense path (dense_attn_block.h): the residual
// stream, all GEMMs, norms, RoPE, flash attention and MLP flow bf16 per-op; paged
// KV bf16. NOTE (honest, G1): the PLE projection combine is accumulated in f32 in
// vLLM (bf16*f32-scalar promotes); we round per-op in bf16 — a named to-verify
// nuance for the strict gate.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"          // PagedKvCache, ForwardLogits
#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor
#include "vllm/model_executor/models/gemma4_moe.h"       // MoE AWQ layer weights
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"  // CommonAttentionMetadata
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"

namespace vllm {

class SafetensorsFile;

// One Gemma-4 self-attention block (gemma4.py::Gemma4Attention). Merged QKV
// (q,k,v_proj), o_proj, per-head Q/K RMSNorm (plain, NOT 1+w) on head_dim, plus a
// WEIGHT-LESS V-norm (has_weight=false, no checkpoint tensor). No bias
// (attention_bias=false). head_dim is per-layer (256 sliding / 512 full). RAW-NK.
struct Gemma4AttnWeights {
  OwnedTensor qkv_proj;  // bf16 raw-NK [Hq*Dh + 2*Hkv*Dh, H] (rows q|k|v)
  OwnedTensor o_proj;    // bf16 raw-NK [H, Hq*Dh]
  OwnedTensor q_norm;    // bf16 [Dh]  (per-head plain RMSNorm)
  OwnedTensor k_norm;    // bf16 [Dh]
  // v_norm has_weight=false → no tensor; the V-norm is a pure normalization.
};

// Gemma-4 GeGLU MLP (gemma4.py::Gemma4MLP): merged gate_up -> GeluAndMul(tanh) ->
// down. Raw-NK. (E4B: intermediate 10240, single width — no double-wide MLP.)
struct Gemma4MlpWeights {
  OwnedTensor gate_up_proj;  // bf16 raw-NK [2*I, H] (rows gate|up)
  OwnedTensor down_proj;     // bf16 raw-NK [H, I]
};

// One Gemma-4 decoder layer (gemma4.py::Gemma4DecoderLayer). Four PLAIN RMSNorms
// (input/post_attention/pre_feedforward/post_feedforward) in a
// standalone-norm-then-explicit-add residual layout (NOT the gemma2/3 fused
// add-norm sandwich), the PLE gate/projection/norm, and the per-layer scalar.
struct Gemma4LayerWeights {
  OwnedTensor input_layernorm;             // bf16 [H]  (standalone plain)
  OwnedTensor post_attention_layernorm;    // bf16 [H]
  OwnedTensor pre_feedforward_layernorm;   // bf16 [H]
  OwnedTensor post_feedforward_layernorm;  // bf16 [H]
  // PLE per-layer components (gemma4.py:680-704).
  OwnedTensor per_layer_input_gate;   // bf16 raw-NK [ple_dim, H]  (Linear H->256)
  OwnedTensor per_layer_projection;   // bf16 raw-NK [H, ple_dim]  (Linear 256->H)
  OwnedTensor post_per_layer_input_norm;  // bf16 [H]  (plain RMSNorm)
  OwnedTensor layer_scalar;               // bf16 [1]  (learned per-layer scalar)
  Gemma4AttnWeights attn;
  Gemma4MlpWeights mlp;
  // Parallel MoE (26B-A4B): empty when enable_moe_block=false (12B dense).
  Gemma4MoeLayerWeights moe;
  bool is_full_attention = false;  // layer_type == "full_attention"
  bool is_kv_shared = false;       // layer_idx >= num_layers - num_kv_shared_layers
  bool k_eq_v = false;             // no v_proj; V shares K (attention_k_eq_v)
  int64_t head_dim = 0;            // 512 full / 256 sliding
  int64_t num_kv_heads = 0;        // may differ full vs sliding (global vs local GQA)
  int64_t kv_target_layer = -1;    // for shared layers: source of K/V (-1 = self)
};

// Whole Gemma-4 text-model weights. tie_word_embeddings defaults TRUE (lm_head
// aliases embed_tokens; the checkpoint has no lm_head.weight). PLE tables live at
// model level (gemma4.py:986-1063).
struct Gemma4Weights {
  bool tie_word_embeddings = true;
  OwnedTensor embed_tokens;             // bf16 [vocab, H]  (scaled by sqrt(H) at use)
  OwnedTensor embed_tokens_per_layer;   // bf16 [vocab, ple_dim*num_layers]
  OwnedTensor per_layer_model_projection;  // bf16 raw-NK [ple_dim*num_layers, H]
  OwnedTensor per_layer_projection_norm;   // bf16 [ple_dim]  (plain RMSNorm)
  OwnedTensor final_norm;               // bf16 [H]  (model.norm, plain RMSNorm)
  OwnedTensor lm_head;                  // bf16 [H, vocab] Matmul-B; EMPTY when tied
  std::vector<Gemma4LayerWeights> layers;
  // Keeps safetensors mmaps alive for borrowed fused expert tensors (26B MoE).
  std::shared_ptr<const void> shards_keepalive;
};

// Dense / small loads (shards need not outlive weights).
Gemma4Weights LoadGemma4ForConditionalGenerationWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

// MoE BF16: experts mmap-borrowed; pass FromSafetensorsOwned shared_ptr.
Gemma4Weights LoadGemma4ForConditionalGenerationWeightsOwned(
    std::shared_ptr<const std::vector<SafetensorsFile>> shards,
    const HfConfig& config);

// The Gemma-4 text backbone forward. Per decoder layer (gemma4.py:709-767):
//   r=h; input_layernorm(h) -> attn(Q/K/V plain-norm, V weight-less norm, dual
//   proportional/standard RoPE, per-layer head_dim, YOCO KV-share) ->
//   post_attention_layernorm -> h = attn_n + r; r=h;
//   pre_feedforward_layernorm -> GeGLU MLP -> post_feedforward_layernorm ->
//   h = mlp_n + r; PLE(gate=gelu(gate_lin(h)); h += pln(proj(gate*ple_in)));
//   h *= layer_scalar. Then model.norm (plain) -> tied lm_head -> soft-cap 30.
// The token embeddings are scaled by sqrt(hidden) (bf16) before layer 0, and the
// PLE inputs are precomputed once (gemma4.py:848-898). Returns [n_out, vocab] f32.
//
// G1 HONEST STATUS: the forward is grounded + compiles, but the strict 32/32
// end-to-end gate is BLOCKED on the runner allocating one uniform KV head_dim
// (runner.cpp:600-646) — Gemma-4's per-layer 256/512 heads are not representable
// without a shared-path change to attn_kv_ construction. See gemma4-multimodal.md.
class Gemma4Model {
 public:
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Gemma4Weights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});

  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Gemma4Weights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});

  // CLAIM-GEMMA4-MM-E2E: one MULTIMODAL forward. The hidden stream starts from the
  // already-merged host bf16 inputs_embeds [T*H] (text rows sqrt(H)-scaled, image/
  // audio soft-token rows the embed_vision/embed_audio projector output), and the
  // PLE embed_tokens_per_layer lookup uses `ple_token_ids` [T] (mm rows masked to
  // 0 + the vocab_size_per_layer_input range mask) — gemma4_mm.py:1962-1973 +
  // gemma4.py get_per_layer_inputs/project_per_layer_inputs (:848-928). Positions
  // are Gemma-4's standard 1-D positions (NO MRoPE, NO DeepStack). Returns
  // [n_out, vocab] host f32 (n_out = |logits_indices|, 1 for the last row).
  static std::vector<float> ForwardMm(
      const std::vector<uint16_t>& inputs_embeds_bf16,
      const std::vector<int32_t>& ple_token_ids,
      const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Gemma4Weights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});
};

// ── CLAIM-GEMMA4-MM-E2E: the registered engine mm-forward driver + adapters ──

// Registered-path greedy IMAGE->text generation for Gemma-4: EVERY decoder step is
// driven through ModelRegistry::Forward(model, input) with input.mm carrying the
// merged inputs_embeds + Gemma-4 PLE-masked ids (positions are the 1-D
// ModelForwardInput::positions) — i.e. it exercises the ENGINE registered
// mm-forward (ForwardGemma4ForConditionalGeneration's mm branch). Mirrors
// Qwen3VLGenerateGreedyViaRegistry (the Qwen3-VL fold): the prefill builds the
// merged embeds (embed(prompt_ids)*sqrt(H) with `mm_projected` scattered into the
// image-placeholder rows), then greedy-decodes through the registered forward.
//
// prompt_ids  : placeholder-expanded model input ids (image_token_id repeated N
//               times at the image span).
// mm_projected: the vision embed_vision projector output [N, text_hidden] (== the
//               masked-scatter merge input), host f32, bf16-rounded before merge.
// image_token_id / eos_token_id : the Gemma-4 <image> soft-token id + eos.
// `model` MUST be the registered Gemma-4 LoadedModel loaded from `weights`.
// `out_margins` (optional): per generated token, the greedy top1-top2 logit margin
// — used by the e2e gate to classify a divergence as a bf16 near-tie (small margin)
// vs a structural error (large margin), per the ratified near-tie methodology.
std::vector<int32_t> Gemma4GenerateGreedyViaRegistry(
    LoadedModel& model, const std::vector<int32_t>& prompt_ids,
    const std::vector<float>& mm_projected, int32_t image_token_id,
    int32_t eos_token_id, const Gemma4Weights& weights, const HfConfig& config,
    vt::Queue& queue, int max_new_tokens,
    std::vector<float>* out_margins = nullptr);

// Wrap already-loaded Gemma-4 weights in the registered LoadedModel so a caller
// that owns the weights (the mm e2e gate) can drive ModelRegistry::Forward without
// re-reading the checkpoint. `Make` OWNS the moved weights; `Borrow` does NOT own
// `weights` (it must outlive the model — used so the driver and the model share ONE
// Gemma4Weights on the unified-memory box). Mirrors Make/BorrowQwen3VLLoadedModel.
std::unique_ptr<LoadedModel> MakeGemma4LoadedModel(Gemma4Weights weights);
std::unique_ptr<LoadedModel> BorrowGemma4LoadedModel(const Gemma4Weights& weights);

// Per-family config hook (mirrors ParseGemma3ForCausalLMConfig). The engine's
// HfConfig loader already descends into `text_config` (hf_config.cpp:103-113), so
// the typed fields + config.raw are the text sub-dict; the Gemma-4-specific
// scalars (layer_types, rope_parameters, hidden_size_per_layer_input,
// num_kv_shared_layers, global_head_dim, final_logit_softcapping) are read from
// config.raw by the loader/forward. No-op hook.
void ParseGemma4ForConditionalGenerationConfig(const HfConfig& config);

// KV-cache spec builder. Emits TWO groups reflecting the true topology: sliding
// layers (head_dim 256) and full-attention layers (head_dim 512). NOTE: the
// current runner reads a single uniform head_dim, so consuming a two-head-dim
// spec is the named G-next shared-path change; this builder documents the intent.
v1::KVCacheConfig MakeGemma4ForConditionalGenerationKVCache(const HfConfig& config,
                                                            int block_size,
                                                            int num_blocks);

}  // namespace vllm
