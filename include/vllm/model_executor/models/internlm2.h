// InternLM2 (`InternLM2ForCausalLM`, internlm2-chat-1_8b / internlm2_5-*) — a
// near-additive dense bring-up: the Llama dense forward VERBATIM, with the ONE
// delta living entirely in the weight loader (ZERO new compute kernel).
// (Upstream: vllm/model_executor/models/internlm2.py @ e24d1b24; config
// internlm/internlm2-chat-1_8b/config.json — a standard dense transformer: NO
// GDN, NO MoE, standard (non-gemma) RMSNorm, GQA 16/8, head_dim 128, SwiGLU,
// UNTIED lm_head (`output.weight`), NeoX RoPE theta 1e6, one full-attention KV
// group.)
//
// InternLM2 IS the Llama/Qwen3-dense forward (Qwen3DenseModel) with exactly one
// structural difference, and it is a LOADER-ONLY layout difference, not a compute
// one:
//
//   The fused `wqkv` weight (QKVParallelLinear, internlm2.py:126-135) packs
//   q/k/v INTERLEAVED BY KV-GROUP, not as the plain [q|k|v] concat every other
//   dense family uses. vLLM's InternLM2Attention.split_qkv (internlm2.py:158-176)
//   reshapes the projection output to [seq, num_kv_heads, key_value_groups+2,
//   head_dim] and splits the group-slot dim into [key_value_groups, 1, 1] -> q,
//   k, v. So each kv-group's rows are laid out {g query heads, 1 key head, 1
//   value head} before the next kv-group. The loader (internlm2_weights.cpp)
//   DE-INTERLEAVES `wqkv` at LOAD into the merged [q|k|v]-row qkv_proj the shared
//   dense AttnBlock already consumes, so the forward reuses the landed GQA path
//   UNCHANGED. The interleave order is load-bearing — a plain [q|k|v] split
//   scrambles heads and produces a ROOT divergence the SACRED gate catches.
//
// Everything else is REUSED: merged qkv GEMM + QkvSplit, NeoX RoPE (theta 1e6;
// the config's rope_scaling type "dynamic" is IDENTITY for contexts within
// max_position_embeddings because max_trained_positions defaults to max_position,
// so vLLM's DynamicNTKScalingRotaryEmbedding leaves the base unchanged —
// dynamic_ntk_scaling_rope.py:53-70), 1/sqrt(head_dim) attention scale, standard
// fused add+RMSNorm residual, SwiGLU MLP, the full-attention paged path. NO
// per-head qk-norm (q_norm/k_norm stay EMPTY), NO biases (bias=false). The name
// map differs (InternLM2 uses model.tok_embeddings / attention.wqkv /
// attention.wo / feed_forward.w1|w3|w2 / attention_norm / ffn_norm / output).
// See .agents/specs/sweep-recent-dense-batch.md §0.2 row 5.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/qwen3.h"  // shared dense weights + forward
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/kv_cache_interface.h"

namespace vllm {

class SafetensorsFile;

// InternLM2 reuses the shared dense weight container and forward. q_norm/k_norm
// stay EMPTY (InternLM2 has no qk-norm), qkv_bias stays EMPTY (bias=false),
// tie_word_embeddings comes from the checkpoint config (false for internlm2 chat
// checkpoints -> the separate `output.weight` lm_head is loaded). Named aliases
// keep the InternLM2 TUs self-documenting.
using InternLM2Weights = Qwen3DenseWeights;
using InternLM2Model = Qwen3DenseModel;

// Load `InternLM2ForCausalLM` (internlm2-chat-1_8b, BF16) safetensors into the
// shared dense container. Name map (InternLM2 flat, no multimodal prefix):
//   model.tok_embeddings.weight                        -> embed_tokens [V,H]
//   model.norm.weight                                  -> final_norm [H]
//   output.weight                                      -> lm_head (SKIPPED tied)
//   model.layers.N.attention_norm.weight               -> input_layernorm [H]
//   model.layers.N.ffn_norm.weight                     -> post_attention_ln [H]
//   model.layers.N.attention.wqkv.weight   -> DE-INTERLEAVED -> merged qkv_proj
//   model.layers.N.attention.wo.weight                 -> o_proj (raw-NK)
//   model.layers.N.feed_forward.w1.weight  + .w3.weight-> merged gate_up_proj
//   model.layers.N.feed_forward.w2.weight              -> down_proj (raw-NK)
// The fused `wqkv` is de-interleaved from the [num_kv_heads, kv_groups+2,
// head_dim, hidden] group-major layout into the plain [q|k|v]-row merged
// qkv_proj (raw-NK) the shared AttnBlock expects. Reuses the shared
// dense_weight_loaders.h helpers for every other projection. Text path only.
InternLM2Weights LoadInternLM2ForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

// Per-family config hook (mirrors ParseLlamaForCausalLMConfig). LoadHfConfig
// materializes + validates every consumed field (including the rope_scaling
// "dynamic" dictionary). Validates full (non-partial) NeoX rope.
void ParseInternLM2ForCausalLMConfig(const HfConfig& config);

// KV-cache spec builder: exactly ONE full-attention KV group, NO MambaSpec/GDN
// (InternLM2 is pure full-attention). Identical topology to Llama's.
v1::KVCacheConfig MakeInternLM2ForCausalLMKVCache(const HfConfig& config,
                                                  int block_size, int num_blocks);

}  // namespace vllm
