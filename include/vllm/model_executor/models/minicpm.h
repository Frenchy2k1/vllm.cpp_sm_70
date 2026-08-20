// MiniCPM (`MiniCPMForCausalLM`, OpenBMB MiniCPM-2B) — a ZERO-NEW-KERNEL dense
// bring-up: the Llama dense forward plus THREE scalar multipliers derived from the
// MiniCPM config. (Upstream: vllm/model_executor/models/minicpm.py @ e24d1b24;
// config openbmb/MiniCPM-2B-sft-bf16/config.json — a standard dense transformer:
// NO GDN, the dense (num_experts==0) branch, standard (non-gemma) RMSNorm, MHA
// 36/36, head_dim 64, SiLU-SwiGLU, tied lm_head, rope_scaling null -> plain NeoX
// rope, one full-attention KV group.)
//
// MiniCPM IS the Llama forward with exactly three scalar deltas, all read from
// config (a Llama checkpoint with all three at their identity would be
// byte-identical):
//   1. scale_emb (minicpm.py:441-443, MiniCPMModel.embed_input_ids):
//      hidden = embed(ids) * config.scale_emb.  (MiniCPM-2B: scale_emb=12.)
//   2. scale_depth (minicpm.py:384-386,392-393, MiniCPMDecoderLayer.forward):
//      each sublayer output is scaled by (scale_depth / sqrt(num_hidden_layers))
//      BEFORE the residual add: h = residual + sublayer(h) * mult.  So the fused
//      add+RMSNorm form (residual += hidden) is NOT usable — the norm is a
//      STANDALONE RMSNorm and the scaled add is a separate MulScalar+Add.
//      (MiniCPM-2B: scale_depth=1.4, num_hidden_layers=40 -> mult=1.4/sqrt(40).)
//   3. dim_model_base (minicpm.py:604,633,640): hidden divided by
//      scale_width = hidden_size / dim_model_base BEFORE lm_head (logit scaling).
//      (MiniCPM-2B: hidden_size=2304, dim_model_base=256 -> scale_width=9.0.)
//
// The attention softmax scale is the STANDARD head_dim**-0.5 (minicpm.py:266 —
// MiniCPM has NO custom attention multiplier, unlike Granite). Everything else —
// merged QKV (loaded from separate q/k/v like Llama), MHA, SwiGLU MLP, standard
// RMSNorm, plain NeoX RoPE, tied lm_head, the full-attention paged path — is
// REUSED from the shared dense substrate (dense_attn_block.h glue,
// dense_weight_loaders.h). The self-attention block + decoder layer are written
// FRESH (like Granite/OLMo-2/OPT) because dense_attn::AttnBlock hard-codes the
// fused residual add that MiniCPM's scale_depth violates. See
// .agents/specs/sweep-recent-dense-batch.md §0.2 row 4.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3.h"  // shared dense weights container
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"  // CommonAttentionMetadata
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"

namespace vllm {

class SafetensorsFile;

// MiniCPM reuses the shared dense weight container (Qwen3DenseWeights): input/post
// RMSNorm + merged qkv/gate_up + tied-or-untied lm_head, with q_norm/k_norm EMPTY
// (MiniCPM has no qk-norm) and qkv_bias EMPTY (attention_bias=false). The three
// scalar deltas live in HfConfig, not the weights.
using MiniCPMWeights = Qwen3DenseWeights;

// Load `MiniCPMForCausalLM` (MiniCPM-2B-sft-bf16, BF16) safetensors into the shared
// dense container. Name map is IDENTICAL to Llama (flat): model.embed_tokens.weight,
// model.norm.weight, lm_head.weight (SKIPPED when tie_word_embeddings), per layer
// model.layers.N.{input_layernorm,post_attention_layernorm}.weight,
// .self_attn.{q,k,v,o}_proj.weight, .mlp.{gate,up,down}_proj.weight. q/k/v merged
// into one qkv_proj and gate/up into one gate_up_proj (vLLM packed_modules_mapping).
// Reuses the shared dense_weight_loaders.h helpers. Text path only.
MiniCPMWeights LoadMiniCPMForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

// The MiniCPM dense forward. Per decoder layer (minicpm.py::MiniCPMDecoderLayer):
//   residual = h
//   h = input_layernorm(h)                     # STANDALONE (non-gemma) RMSNorm
//   h = self_attn(h)                           # scale = head_dim**-0.5
//   h = residual + h * (scale_depth/sqrt(L))   # scaled residual add
//   residual = h
//   h = post_attention_layernorm(h)            # STANDALONE RMSNorm
//   h = mlp(h)                                 # SiLU-SwiGLU
//   h = residual + h * (scale_depth/sqrt(L))
// Model: h = embed(ids) * scale_emb -> N layers -> final RMSNorm ->
// (h / scale_width) -> lm_head(h). bf16 residual stream (mirrors vLLM's per-op bf16
// stores); qkv/rope/KV/attn/mlp flow bf16; the standalone norms + scaled adds
// compute in f32 and round to bf16. Returns [n_out, vocab] f32 logits.
class MiniCPMModel {
 public:
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const MiniCPMWeights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {},
      const vllm::TensorParallel* tp = nullptr);

  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const MiniCPMWeights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {},
      const vllm::TensorParallel* tp = nullptr);
};

// Per-family config hook (mirrors ParseGraniteForCausalLMConfig). LoadHfConfig
// materializes the typed fields; the three MiniCPM scalars (scale_emb, scale_depth,
// dim_model_base) are read from config.raw by the forward. Validates plain
// (non-partial) NeoX rope and rejects the config-gated MoE variant (num_experts>0).
void ParseMiniCPMForCausalLMConfig(const HfConfig& config);

// KV-cache spec builder: exactly ONE full-attention KV group, NO MambaSpec/GDN
// (MiniCPM dense is pure full-attention). Identical topology to Llama's.
v1::KVCacheConfig MakeMiniCPMForCausalLMKVCache(const HfConfig& config,
                                                int block_size, int num_blocks);

}  // namespace vllm
