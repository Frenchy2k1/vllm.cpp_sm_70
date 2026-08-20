// MiniCPM3 (`MiniCPM3ForCausalLM`, OpenBMB MiniCPM3-4B) — the MiniCPM dense
// bring-up (three scalar deltas, ZERO new kernel) with its ATTENTION swapped
// from GQA to DeepSeek-style **MLA** (multi-head latent attention). It REUSES,
// verbatim, two landed subsystems:
//   * the MiniCPM scalar wiring (scale_emb / scale_depth / dim_model_base) —
//     MiniCPM3ForCausalLM subclasses MiniCPMForCausalLM (minicpm3.py:224-233),
//     MiniCPM3Model subclasses MiniCPMModel (:207-221) and MiniCPM3DecoderLayer
//     subclasses MiniCPMDecoderLayer (:186-204), overriding ONLY the attention
//     module — so scale_emb / the scaled residual add / dim_model_base logit
//     scale are inherited UNCHANGED from MiniCPM; and
//   * the DeepSeek-V2 MLA attention block (mla::ForwardMlaAttentionBlock,
//     mla::MlaBlockWeights, mla::MlaBlockDims, the load-time kv_b_proj
//     absorption) — threaded with MiniCPM3's config dims.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides, @ pin e24d1b24) ───────
//   OURS                       <-  UPSTREAM
//   MiniCPM3Params             <-  vllm/model_executor/models/minicpm3.py:52-134
//                                  `MiniCPM3Attention.__init__` (the MLA geometry)
//                                  + minicpm.py:588-640 (the 3 scalars, inherited)
//   MiniCPM3MlaWeights         <-  minicpm3.py:85-119 (q_a_proj/q_a_layernorm/
//                                  q_b_proj, kv_a_proj_with_mqa/kv_a_layernorm/
//                                  kv_b_proj, o_proj — q_lora is ALWAYS present)
//   BuildMiniCPM3RopeCosSinCache <- phi3_long_rope_scaled_rope.py:97-123
//                                  (LongRoPE inv_freq rescale + mscale; the short
//                                  cache, since max_model_len ==
//                                  original_max_position_embeddings ⇒ mscale 1.0)
//   MiniCPM3Model::Forward     <-  minicpm3.py:136-183 (MLA forward) wrapped by
//                                  minicpm.py:378-394 decoder + :430-450 model +
//                                  :588-640 causal-LM (scalars)
//
// ─── THE THREE MLA DELTAS vs the landed DeepSeek-V2 path ────────────────────
// 1. `is_neox_style=True`. DeepSeek builds the decoupled rotary with
//    is_neox_style=False (adjacent-pair GPT-J); MiniCPM3 takes get_rope's DEFAULT
//    is_neox_style=True (neox half-split, minicpm3.py:121-125, no is_neox_style
//    arg). Threaded through the shared `mla::MlaBlockDims::is_neox_style` (default
//    false — DeepSeek unchanged).
// 2. LongRoPE, not YaRN. MiniCPM3-4B ships rope_scaling type "longrope" with a
//    per-frequency-pair rescale (phi3_long_rope). Because max_position_embeddings
//    == original_max_position_embeddings (32768), the LongRoPE scale is 1.0 ⇒
//    mscale 1.0 and the SHORT cache is selected; the attention softmax scale is
//    therefore the plain qk_head_dim**-0.5 with NO mscale correction (contrast
//    DeepSeek's mscale^2). A dedicated small cache builder replaces
//    BuildDeepseekRopeCosSinCache.
// 3. q_lora is ALWAYS present (q_lora_rank 768), so only the fused_qkv_a_proj
//    branch of the MLA block runs; the direct-q_proj branch is never taken.
// Everything else — the absorbed decode MQA, the materialized-MHA prefill, the
// chunked-context loop, the MLA KV spec — is the landed DeepSeek-V2 MLA code,
// unmodified, driven by MiniCPM3's dims.
#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/mla_attention.h"    // MlaBlockDims/Weights
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3.h"             // Qwen3DenseMlpWeights
#include "vllm/model_executor/models/qwen3_5.h"           // PagedKvCache, ForwardLogits
#include "vllm/model_executor/models/qwen3_5_weights.h"   // OwnedTensor
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"                    // CommonAttentionMetadata
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"

namespace vllm {

class SafetensorsFile;

// Every MiniCPM3 config field the loader/forward consume, resolved ONCE. The
// MiniCPM3-4B config.json (via trust_remote_code MiniCPM3Config defaults):
//   hidden 2560, 62 layers, 40 heads, qk_nope 64, qk_rope 32, v 64,
//   kv_lora 256, q_lora 768, vocab 73448, intermediate 6400, rms_eps 1e-5,
//   tie_word_embeddings TRUE, rope_theta 10000, max_pos 32768,
//   rope_scaling type "longrope" (short==long factor, orig_max_pos 32768),
//   scale_emb 12, scale_depth 1.4, dim_model_base 256.
struct MiniCPM3Params {
  int64_t hidden_size = 0;
  int64_t num_hidden_layers = 0;
  int64_t vocab_size = 0;
  int64_t intermediate_size = 0;
  float rms_norm_eps = 1e-6f;
  bool tie_word_embeddings = true;
  int64_t max_position_embeddings = 0;

  // The three MiniCPM scalar deltas (minicpm.py:588-640), inherited unchanged.
  double scale_emb = 1.0;
  double scale_depth = 1.0;
  double dim_model_base = 0.0;

  // MLA geometry (minicpm3.py:52-134). q_lora_rank is ALWAYS > 0 here.
  mla::MlaBlockDims mla{};

  // LongRoPE (phi3_long_rope_scaled_rope.py). `rope_factor` has qk_rope_head_dim/2
  // entries; `rope_mscale` is the (position-independent) cos/sin multiplier —
  // 1.0 whenever max_position_embeddings <= original_max_position_embeddings.
  double rope_base = 10000.0;
  int64_t rope_original_max_position_embeddings = 0;
  int64_t rope_cache_rows = 0;  // rows to build (== max_position_embeddings)
  double rope_mscale = 1.0;
  std::vector<double> rope_factor;

  // scale_depth / sqrt(num_hidden_layers) (minicpm.py:384-386,392-393).
  double residual_scale() const {
    return scale_depth / std::sqrt(static_cast<double>(num_hidden_layers));
  }
  // hidden_size / dim_model_base (minicpm.py:604) — the pre-lm_head logit divisor.
  double scale_width() const {
    return dim_model_base > 0.0 ? static_cast<double>(hidden_size) / dim_model_base
                                : 1.0;
  }
};

MiniCPM3Params ParseMiniCPM3Params(const HfConfig& config);

// `MiniCPM3Attention` weights (minicpm3.py:85-119). q_lora is ALWAYS present, so
// only the fused_qkv_a_proj branch exists. All 2-D projections are RAW-NK
// ([out_features, in_features]) for vt::MatmulBT. `fused_qkv_a_proj` FUSES the
// two checkpoint tensors q_a_proj + kv_a_proj_with_mqa in row order
// [q_lora_rank | kv_lora_rank | qk_rope_head_dim], exactly the layout the shared
// mla::MlaBlockWeights::fused_qkv_a_proj documents (MiniCPM3 has no upstream
// packed_modules_mapping entry for it — the two are separate modules there — but
// the MLA block consumes the fused owner, and fusing at load time is a pure
// concat that changes no numerics).
struct MiniCPM3MlaWeights {
  OwnedTensor fused_qkv_a_proj;  // [q_lora + kv_lora + qk_rope, H] raw-NK
  OwnedTensor q_a_layernorm;     // [q_lora]
  OwnedTensor q_b_proj;          // [N*qk_head_dim, q_lora] raw-NK
  OwnedTensor kv_a_layernorm;    // [kv_lora]
  OwnedTensor kv_b_proj;         // [N*(qk_nope+v), kv_lora] raw-NK (prefill path)
  OwnedTensor w_uk_t;            // [N, qk_nope, kv_lora] (absorbed decode)
  OwnedTensor w_uv;              // [N, kv_lora, v]       (absorbed decode)
  OwnedTensor o_proj;            // [H, N*v] raw-NK
};

// One MiniCPM3 decoder layer (minicpm3.py MiniCPM3DecoderLayer, inheriting
// MiniCPMDecoderLayer.forward). Standard (non-gemma) RMSNorms; dense SwiGLU MLP
// (no MoE — MiniCPM3 has no expert layers).
struct MiniCPM3LayerWeights {
  OwnedTensor input_layernorm;           // [H]
  OwnedTensor post_attention_layernorm;  // [H]
  MiniCPM3MlaWeights attn;
  Qwen3DenseMlpWeights mlp;              // gate_up + down (bf16 raw-NK)
};

// Whole MiniCPM3 text-model weights + resolved params. MiniCPM3-4B ties
// embeddings (no lm_head.weight).
struct MiniCPM3Weights {
  MiniCPM3Params params{};
  OwnedTensor embed_tokens;  // bf16 [vocab, H]
  OwnedTensor final_norm;    // bf16 [H]
  OwnedTensor lm_head;       // bf16 [H, vocab] Matmul-B; EMPTY when tied
  OwnedTensor rope_cos_sin_cache;  // bf16 [rows, qk_rope_head_dim] ([cos|sin])
  std::vector<MiniCPM3LayerWeights> layers;
};

// `Phi3LongRoPEScaledRotaryEmbedding._compute_cos_sin_cache`
// (phi3_long_rope_scaled_rope.py:97-123): inv_freq[i] = 1/(factor[i] *
// base**(2i/rotary_dim)) for i in [0, rotary_dim/2); cache row `pos` is
// [cos(pos*inv_freq)*mscale | sin(pos*inv_freq)*mscale], the [cos|sin] layout
// vt::RopeFromCache expects (NEOX application; only the pairing differs from
// BuildDeepseekRopeCosSinCache, not the cache contents). f32 intermediates.
std::vector<float> BuildMiniCPM3RopeCosSinCache(const MiniCPM3Params& p);

// Load `MiniCPM3ForCausalLM` bf16 safetensors into MiniCPM3Weights, INCLUDING the
// load-time kv_b_proj -> W_UK/W_UV absorption (mla::AbsorbKvBProjBf16) and the
// LongRoPE cos/sin cache.
MiniCPM3Weights LoadMiniCPM3ForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

// The MiniCPM3 forward: embed*scale_emb -> N (input_norm -> MLA block ->
// scaled residual add -> post_norm -> SwiGLU -> scaled residual add) ->
// final_norm -> /scale_width -> tied lm_head. `attn_kv` carries ONE MLA cache per
// layer, viewed [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim].
class MiniCPM3Model {
 public:
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const MiniCPM3Weights& weights,
      vt::Queue& queue, const std::vector<int32_t>& logits_indices = {},
      const vllm::TensorParallel* tp = nullptr);

  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const MiniCPM3Weights& weights,
      vt::Queue& queue, const std::vector<int32_t>& logits_indices = {},
      const vllm::TensorParallel* tp = nullptr);
};

// Per-family config hook (the registry `parse_config`): resolves + validates
// MiniCPM3Params and throws on anything unsupported.
void ParseMiniCPM3ForCausalLMConfig(const HfConfig& config);

// KV-cache spec builder: exactly ONE **MLA** attention group (1 head,
// kv_lora_rank + qk_rope_head_dim wide, NO separate V), identical topology to
// DeepSeek-V2's.
v1::KVCacheConfig MakeMiniCPM3ForCausalLMKVCache(const HfConfig& config,
                                                 int block_size, int num_blocks);

}  // namespace vllm
