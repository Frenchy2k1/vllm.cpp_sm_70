// DFlash draft model (`DFlashDraftModel` -> DFlashQwen3ForCausalLM /
// DFlashQwen3Model) for block-diffusion speculative decoding. SPEC-DFLASH D2,
// DF-DRAFT-MODEL.
//
// Ported from vllm/model_executor/models/qwen3_dflash.py @ 555967922
// (vLLM 0.26.0.dev0): DFlashQwen3Attention (:149-263), DFlashQwen3DecoderLayer
// (:266-342), DFlashQwen3Model (:345-661), DFlashQwen3ForCausalLM (:664-855).
//
// The draft is a PLAIN Qwen3-dense decoder (NOT the Qwen3.5 gated full-attention):
// per-layer q/k/v/o proj + per-head q_norm/k_norm RMSNorm, standard (non-gemma)
// input/post RMSNorm, NeoX RoPE (theta 1e7), SwiGLU gate/up/down MLP, NO GDN, NO
// MoE, NO attention output gate. The z-lab/Qwen3.6-27B-DFlash card: 5 layers
// (config.layer_types = 4x sliding_attention window 2048 + 1x full_attention),
// hidden 5120, 32 q-heads / 8 kv-heads / head_dim 128, vocab 248320,
// mask_token_id 248070, target_layer_ids [1,16,31,46,61] (5 aux taps, fc input
// 5120x5 -> 5120).
//
// The ONE genuinely new brick is the attention: the FULL-attention layer attends
// BIDIRECTIONALLY (non-causal) across the uniform (1+k) query block; the SWA
// layers are causal within their window. This routes through the new
// vt::DFlashBlockAttention primitive (ops.h) rather than the causal
// vt::PagedAttention every other model uses. Per-layer causality is resolved from
// config exactly as vLLM _resolve_layer_attention (:86-146): a layer is causal iff
// it is a sliding_attention layer (unless dflash_config.causal overrides).
//
// Context-KV precompute (qwen3_dflash.py:548-619 precompute_and_store_context_kv)
// and prepare_dflash_inputs are D3 (DF-DRAFT-KV-PREP); this header/cpp owns the
// draft model, the fc combine, the mask embedding, the block forward, and the
// loader. The D2 isolation gate exercises the CONTEXT-FREE block forward (the
// query block attends only to itself) which fully exercises the new non-causal
// primitive + the per-layer routing + fc + mask-embed + logits.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/models/qwen3.h"             // Qwen3DenseMlpWeights-style ops
#include "vllm/model_executor/models/qwen3_5.h"           // ForwardLogits
#include "vllm/model_executor/models/qwen3_5_weights.h"   // OwnedTensor, TensorResolver
#include "vllm/transformers_utils/hf_config.h"
#include "vt/device.h"
#include "vt/tensor.h"

namespace vllm {

class SafetensorsFile;

// Per-layer attention mode resolved from the draft config, mirroring
// _resolve_layer_attention (qwen3_dflash.py:86-146). `causal` is false for a
// full-attention layer (BIDIRECTIONAL in-block) and true for a sliding-window
// layer (causal within `sliding_window`).
struct Qwen3DFlashLayerAttnMode {
  bool causal = false;
  int64_t sliding_window = 0;  // >0 for SWA layers; 0 for full layers
};

// One DFlash draft decoder layer: input/post standard RMSNorm + plain Qwen3
// attention (merged qkv, per-head q/k norm, NeoX RoPE) + SwiGLU MLP. Weights are
// kept in the on-disk torch-Linear [N=out,K=in] orientation (nk=true) for
// vt::MatmulBT, exactly like the Qwen3-dense loader.
struct Qwen3DFlashLayerWeights {
  OwnedTensor input_layernorm;           // bf16 [H]
  OwnedTensor post_attention_layernorm;  // bf16 [H]
  OwnedTensor qkv_proj;  // bf16 raw-NK [Hq*Dh + 2*Hkv*Dh, H] (rows q|k|v), nk
  OwnedTensor o_proj;    // bf16 raw-NK [H, Hq*Dh], nk
  OwnedTensor q_norm;    // bf16 [head_dim] per-head RMSNorm
  OwnedTensor k_norm;    // bf16 [head_dim]
  OwnedTensor gate_up_proj;  // bf16 raw-NK [2*I, H] (rows gate|up), nk
  OwnedTensor down_proj;     // bf16 raw-NK [H, I], nk
  Qwen3DFlashLayerAttnMode attn_mode;
};

// Whole DFlash draft weights. The draft owns its OWN embed_tokens and lm_head
// (unlike the MTP head, which shares the target's) plus the fc aux-combine, the
// hidden_norm (applied to combined target features before the context-KV proj,
// D3), the final norm, and an optional dedicated mask embedding.
struct Qwen3DFlashWeights {
  OwnedTensor embed_tokens;  // bf16 [vocab, H] (embed lookup)
  OwnedTensor fc;            // bf16 raw-NK [H, H*num_taps], nk (combine_hidden_states)
  OwnedTensor hidden_norm;   // bf16 [H] (normed before context-KV proj, D3)
  OwnedTensor final_norm;    // bf16 [H] (model.norm)
  OwnedTensor lm_head;       // bf16 raw-NK [draft_vocab, H], nk
  // Optional dedicated mask embedding [H] substituted at mask_token_id
  // (has_separate_mask_embedding). Empty for the z-lab 27B (in-vocab mask token).
  OwnedTensor mask_embedding;
  std::vector<Qwen3DFlashLayerWeights> layers;
  int64_t num_taps = 0;         // len(target_layer_ids); fc input = H*num_taps
  int32_t mask_token_id = -1;   // dflash_config.mask_token_id (248070 for 27B)
  int64_t draft_vocab_size = 0;
};

// Load the z-lab DFlash draft checkpoint. The on-disk names follow vLLM's
// DFlashQwen3Model.load_weights + hf_to_vllm_mapper (qwen3_dflash.py:347-356,
// 772-855): per layer `model.layers.N.self_attn.{q,k,v,o}_proj.weight`,
// `.self_attn.{q,k}_norm.weight`, `.mlp.{gate,up,down}_proj.weight`,
// `.{input_layernorm,post_attention_layernorm}.weight`; top-level
// `model.embed_tokens.weight`, `model.fc.weight`, `model.hidden_norm.weight`,
// `model.norm.weight`, `lm_head.weight`. q/k/v are concatenated into one qkv_proj
// and gate/up into one gate_up_proj (the vLLM stacked mapping). All draft tensors
// are BF16. `num_taps`/`mask_token_id`/`draft_vocab_size` come from the resolved
// draft config.
Qwen3DFlashWeights LoadQwen3DFlash(const TensorResolver& get, const HfConfig& config,
                                   int64_t num_taps, int32_t mask_token_id);
Qwen3DFlashWeights LoadQwen3DFlash(const std::vector<SafetensorsFile>& shards,
                                   const HfConfig& config, int64_t num_taps,
                                   int32_t mask_token_id);

// Resolve the per-layer attention modes from config.layer_types (and the optional
// dflash_config overrides). Exposed for the loader + tests. Mirrors
// _resolve_layer_attention (qwen3_dflash.py:86-146).
std::vector<Qwen3DFlashLayerAttnMode> ResolveQwen3DFlashAttnModes(const HfConfig& config);

// The DFlash draft forward. D2 owns the CONTEXT-FREE block forward (the isolation
// gate): each request's uniform (1+k) query block attends only to itself through
// vt::DFlashBlockAttention (non-causal full / causal SWA per layer). D3 extends
// this to attend over pre-inserted context K/V.
class Qwen3DFlashModel {
 public:
  // The fc aux-combine (combine_hidden_states, qwen3_dflash.py:750-770): a bias-
  // free Linear [H*num_taps]->[H] over the D1 multi-tap `[T, H*num_taps]` output
  // (column order = ascending target_layer_ids). Returns [T,H] bf16 as a device
  // buffer's host download for parity checks. This is the combined feature that
  // D3 will normalize (hidden_norm) and project into the context KV cache.
  static std::vector<float> CombineAuxFeatures(const std::vector<float>& aux_features,
                                               int64_t T, const Qwen3DFlashWeights& weights,
                                               const HfConfig& config, vt::Queue& queue);

  // CONTEXT-FREE block forward -> [T, draft_vocab] f32 logits (the D2 isolation
  // gate). `input_ids` are the mask-block token ids (anchor + k mask_token_id per
  // request, T = num_reqs*(1+k)); `positions` the intra-context positions; `cu`
  // the per-request block boundaries (length num_reqs+1). Each block attends only
  // to itself: full-attention layers BIDIRECTIONAL, SWA layers causal-in-window.
  // Mask slots embed via embed_tokens[mask_token_id] (or the dedicated
  // mask_embedding when present), mirroring embed_input_ids (:432-438).
  static std::vector<float> ForwardBlockLogits(
      const std::vector<int32_t>& input_ids, const std::vector<int32_t>& positions,
      const std::vector<int32_t>& cu, const Qwen3DFlashWeights& weights,
      const HfConfig& config, vt::Queue& queue);
};

}  // namespace vllm
