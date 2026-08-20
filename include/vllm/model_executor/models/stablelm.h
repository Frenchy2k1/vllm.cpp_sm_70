// StableLM (`StableLmForCausalLM`, stabilityai/stablelm-2-1_6b) — a
// ZERO-NEW-KERNEL dense bring-up. It is the OPT pre-LN residual structure
// (nn.LayerNorm with weight AND bias, explicit residual add — NOT the fused
// add+RMSNorm) crossed with the phi3 GQA+partial-NeoX-RoPE+SwiGLU attention/MLP
// path. Every op REUSES a landed primitive:
//
//   * nn.LayerNorm (mean+variance, weight+bias) — vt::LayerNorm, landed for OPT.
//   * partial NeoX RoPE (rotary_dim = int(head_dim * partial_rotary_factor) <
//     head_dim) from a precomputed plain cos/sin cache — vt::RopeFromCache, the
//     same partial path phi3/GLM-4 use.
//   * optional merged qkv bias (`use_qkv_bias`, stablelm-2 sets it True) —
//     vt::Add row-broadcast, the same biased-projection path as OPT.
//   * SiLU-SwiGLU MLP (merged gate_up -> SiluAndMul -> down) — vt::SiluAndMul.
//
// Upstream: vllm/model_executor/models/stablelm.py @ e24d1b24
//   - StablelmDecoderLayer (:176-215): pre-norm, SEPARATE input_layernorm /
//     post_attention_layernorm (both nn.LayerNorm), explicit `residual + x`.
//   - StablelmAttention (:92-174): merged qkv (bias = getattr use_qkv_bias),
//     o_proj (NO bias), get_rope(head_dim, rope_parameters) => partial NeoX,
//     scaling = head_dim ** -0.5.
//   - StablelmMLP (:60-89): MergedColumnParallelLinear gate_up -> SiluAndMul ->
//     down_proj, all bias-free.
//   - StableLMEpochModel (:218-274): embed_tokens -> N layers -> final nn.LayerNorm.
//   - StablelmForCausalLM (:277-357): ParallelLMHead (tie_word_embeddings optional;
//     stablelm-2-1_6b is UNTIED), LogitsProcessor(vocab_size). WeightsMapper
//     merges q/k/v -> qkv_proj and gate/up -> gate_up_proj.
//   - config stabilityai/stablelm-2-1_6b/config.json: hidden 2048, 32 heads
//     (MHA: 32 kv heads), 24 layers, intermediate 5632, head_dim 64,
//     partial_rotary_factor 0.25 (rotary_dim 16), use_qkv_bias True,
//     layer_norm_eps 1e-5, rope_theta 10000 (default), tie_word_embeddings False.
//
// The forward is written FRESH (like OPT/phi3, NOT dense_attn::AttnBlock which
// hard-codes fused add+RMSNorm pre-norm + Qwen per-head qk-norm): StableLM's
// non-fused LayerNorm residual and merged-qkv-bias cannot ride that block. It
// reuses all the shared device glue (Dev/DBuf/ResidentWeight/KvSlice/StepInputs).
// See .agents/specs/sweep-recent-dense-batch.md §0.2 row 3.
#pragma once

#include <cstdint>
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

// StableLM attention weights (stablelm.py::StablelmAttention). Merged q|k|v in
// the on-disk torch-Linear [N=out, K=in] orientation (nk=true) for vt::MatmulBT.
// `qkv_bias` is present only when config `use_qkv_bias` is true (stablelm-2 sets
// it). o_proj is ALWAYS bias-free (RowParallelLinear bias=False, stablelm.py:139).
struct StablelmAttnWeights {
  OwnedTensor qkv_proj;  // bf16 raw-NK [Hq*Dh + 2*Hkv*Dh, H] (rows q|k|v), nk
  OwnedTensor qkv_bias;  // bf16 [Hq*Dh + 2*Hkv*Dh]; EMPTY unless use_qkv_bias
  OwnedTensor o_proj;    // bf16 raw-NK [H, Hq*Dh], nk
};

// StableLM SwiGLU MLP (stablelm.py::StablelmMLP): merged gate_up -> SiluAndMul ->
// down_proj, all bias-free. Raw-NK like the attention projections.
struct StablelmMlpWeights {
  OwnedTensor gate_up_proj;  // bf16 raw-NK [2*I, H] (rows gate|up), nk
  OwnedTensor down_proj;     // bf16 raw-NK [H, I], nk
};

// One StableLM decoder layer (stablelm.py::StablelmDecoderLayer). Both norms are
// nn.LayerNorm carrying weight AND bias.
struct StablelmLayerWeights {
  OwnedTensor input_layernorm;                // bf16 [H]
  OwnedTensor input_layernorm_bias;           // bf16 [H]
  OwnedTensor post_attention_layernorm;       // bf16 [H]
  OwnedTensor post_attention_layernorm_bias;  // bf16 [H]
  StablelmAttnWeights attn;
  StablelmMlpWeights mlp;
};

// Whole StableLM text-model weights (stablelm.py::StableLMEpochModel +
// StablelmForCausalLM). `final_norm` is the decoder-level nn.LayerNorm (weight+
// bias). When `tie_word_embeddings` is true, `lm_head` is EMPTY and aliases
// embed_tokens (stablelm-2-1_6b is UNTIED, so lm_head is populated). The rope
// cache is the plain per-position cos/sin cache [P, rotary_dim] (rope_type
// default), indexed by REAL position and applied to the leading rotary_dim.
struct StablelmWeights {
  bool tie_word_embeddings = false;
  bool use_qkv_bias = false;
  OwnedTensor embed_tokens;      // bf16 [vocab, H]
  OwnedTensor final_norm;        // bf16 [H]
  OwnedTensor final_norm_bias;   // bf16 [H]
  OwnedTensor lm_head;           // bf16 [H, vocab] Matmul-B; EMPTY when tied
  std::vector<StablelmLayerWeights> layers;
  OwnedTensor rope_cos_sin;      // bf16 [P, rotary_dim]; row p = angle for pos p
};

// nn.LayerNorm eps: vLLM uses `getattr(config, "norm_eps",
// getattr(config, "layer_norm_eps", 1e-05))` (stablelm.py:187,240). Read from
// HfConfig::raw so no shared config POD is touched.
float StablelmLayerNormEps(const HfConfig& config);

// Whether the checkpoint carries q/k/v projection bias (`use_qkv_bias`,
// stablelm.py:117). stablelm-2-1_6b sets it True.
bool StablelmUseQkvBias(const HfConfig& config);

// Load `StableLmForCausalLM` (stablelm-2-1_6b, BF16) safetensors + build the plain
// partial-rope cos/sin cache. On-disk names (WeightsMapper merges q/k/v->qkv_proj,
// gate/up->gate_up_proj): model.embed_tokens.weight, model.norm.{weight,bias},
// lm_head.weight (SKIPPED when tie_word_embeddings), and per layer
// model.layers.N.{input_layernorm,post_attention_layernorm}.{weight,bias},
// .self_attn.{q,k,v}_proj.{weight[,bias]}, .self_attn.o_proj.weight,
// .mlp.{gate,up,down}_proj.weight. Text path only. Reuses dense_weight_loaders.h.
StablelmWeights LoadStableLmForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

// The StableLM dense forward (pre-norm, non-fused LayerNorm residual). Per layer:
//   residual = h; h = LayerNorm(h, w_in, b_in) -> qkv_proj (+bias) -> split
//   -> partial NeoX RoPE from the plain cache (real positions) -> FA2 causal paged
//   attention (scale 1/sqrt(Dh)) -> o_proj -> h = residual + attn ;
//   residual = h; h = LayerNorm(h, w_post, b_post) -> gate_up -> SiluAndMul ->
//   down -> h = residual + mlp. Then final LayerNorm -> lm_head. bf16 stream;
//   LayerNorm accumulates mean/variance in f32. Returns [n_out, vocab] f32.
class StablelmModel {
 public:
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const StablelmWeights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {},
      const vllm::TensorParallel* tp = nullptr);

  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const StablelmWeights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {},
      const vllm::TensorParallel* tp = nullptr);
};

// Per-family config hook. Validates the partial-rotary invariant (rotary_dim in
// (0, head_dim], default rope) and MHA/GQA head divisibility.
void ParseStableLmForCausalLMConfig(const HfConfig& config);

// KV-cache spec builder: exactly ONE full-attention KV group.
v1::KVCacheConfig MakeStableLmForCausalLMKVCache(const HfConfig& config,
                                                 int block_size, int num_blocks);

}  // namespace vllm
