// Command-R / Cohere (`CohereForCausalLM`, e.g. CohereForAI/c4ai-command-r-v01,
// aya-expanse-8b) — a ZERO-NEW-KERNEL dense bring-up. It is a GPT-J-style
// PARALLEL-residual decoder: ONE nn.LayerNorm per layer feeds BOTH the attention
// and the MLP, and their outputs are summed back onto the residual together
// (unlike Llama's two sequential pre-norms). Every op REUSES a landed primitive:
//
//   * Cohere `LayerNorm` (mean-centred, weight-only, NO learnable bias) —
//     vt::LayerNorm called with a NULL bias pointer (the CUDA/CPU kernels already
//     branch on bias==nullptr). Mean-subtract + rsqrt(var+eps) + weight scale,
//     f32 accumulation, single bf16 rounding on store — bit-mirrors commandr.py's
//     layer_norm_func (:65-73).
//   * GPT-J (is_neox_style=False) full-width RoPE from a precomputed plain cos/sin
//     cache — vt::RopeFromCache with is_neox_style=false (the same adjacent-pair
//     path DeepSeek's decoupled RoPE uses). rotary_dim = head_dim (full rotary).
//   * SiLU-SwiGLU MLP (merged gate_up -> SiluAndMul -> down) — vt::SiluAndMul.
//     CohereMLP hard-codes SiluAndMul regardless of config.hidden_act.
//   * logit_scale scalar (logits *= config.logit_scale before sampling) —
//     folded into the lm_head epilogue via vt::MulScalar.
//   * tied embeddings (config.tie_word_embeddings asserted True) — lm_head aliases
//     embed_tokens (vt::MatmulBT), same as the tied path in stablelm/internlm2.
//
// Upstream: vllm/model_executor/models/commandr.py @ e24d1b24
//   - LayerNorm (:76-87) weight-only, layer_norm_func (:65-73) mean-centred f32.
//   - CohereMLP (:90-122): MergedColumnParallelLinear gate_up (bias=False) ->
//     SiluAndMul -> down_proj (bias=False).
//   - CohereAttention (:125-231): QKVParallelLinear (bias=False), o_proj
//     (bias=False), get_rope(head_dim, ..., is_neox_style=False), scaling =
//     head_dim**-0.5; use_qk_norm defaults False for CohereForCausalLM (True is
//     the Cohere2 arch, out of scope here). v1 always applies rope (:224-225).
//   - CohereDecoderLayer.forward (:257-273): residual = h;
//     h = input_layernorm(h); h_attn = self_attn(h); h_mlp = mlp(h);
//     h = residual + h_attn + h_mlp  (PARALLEL residual, ONE norm per layer).
//   - CohereModel (:277-339): embed_tokens -> N layers -> final LayerNorm.
//   - CohereForCausalLM (:342-417): assert config.tie_word_embeddings (:372);
//     LogitsProcessor(vocab_size, scale=config.logit_scale) (:376); compute_logits
//     applies the embed_tokens weight (:404-413); no biases anywhere.
//
// The forward is written FRESH (not dense_attn::AttnBlock, which hard-codes the
// fused add+RMSNorm sequential pre-norm): Command-R's weight-only LayerNorm,
// GPT-J rotary and single-norm parallel residual cannot ride that block. It
// reuses all the shared device glue (Dev/DBuf/ResidentWeight/KvSlice/StepInputs).
// See .agents/specs/sweep-recent-dense-batch.md §0.2 row 6.
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

// Command-R attention weights (commandr.py::CohereAttention). Merged q|k|v in the
// on-disk torch-Linear [N=out, K=in] orientation (nk=true) for vt::MatmulBT. NO
// biases anywhere (QKVParallelLinear/RowParallelLinear bias=False, commandr.py
// :160,168). qk-norm is a Cohere2-only feature and is rejected at config parse.
struct CommandrAttnWeights {
  OwnedTensor qkv_proj;  // bf16 raw-NK [Hq*Dh + 2*Hkv*Dh, H] (rows q|k|v), nk
  OwnedTensor o_proj;    // bf16 raw-NK [H, Hq*Dh], nk
};

// Command-R SwiGLU MLP (commandr.py::CohereMLP): merged gate_up -> SiluAndMul ->
// down_proj, all bias-free. Raw-NK like the attention projections.
struct CommandrMlpWeights {
  OwnedTensor gate_up_proj;  // bf16 raw-NK [2*I, H] (rows gate|up), nk
  OwnedTensor down_proj;     // bf16 raw-NK [H, I], nk
};

// One Command-R decoder layer (commandr.py::CohereDecoderLayer). Exactly ONE
// nn.LayerNorm (weight-only, no bias) feeds BOTH attn and mlp (parallel residual).
struct CommandrLayerWeights {
  OwnedTensor input_layernorm;  // bf16 [H]; NO bias (Cohere LayerNorm is weight-only)
  CommandrAttnWeights attn;
  CommandrMlpWeights mlp;
};

// Whole Command-R text-model weights (commandr.py::CohereModel + CohereForCausalLM).
// `final_norm` is the decoder-level Cohere LayerNorm (weight-only). Embeddings are
// ALWAYS tied (config.tie_word_embeddings asserted True, commandr.py:372) so there
// is no separate lm_head — the logits GEMM reuses embed_tokens. The rope cache is
// the plain per-position cos/sin cache [P, head_dim] (full rotary), indexed by REAL
// position and applied GPT-J style (is_neox_style=False).
struct CommandrWeights {
  double logit_scale = 1.0;    // logits *= logit_scale (commandr.py:376)
  OwnedTensor embed_tokens;    // bf16 [vocab, H]; also the (tied) lm_head
  OwnedTensor final_norm;      // bf16 [H]; weight-only Cohere LayerNorm
  std::vector<CommandrLayerWeights> layers;
  OwnedTensor rope_cos_sin;    // bf16 [P, head_dim]; row p = angle for pos p
};

// Cohere LayerNorm eps: config.layer_norm_eps (commandr.py:301, default 1e-5).
// Read from HfConfig::raw so no shared config POD is touched.
float CommandrLayerNormEps(const HfConfig& config);

// logit_scale scalar (commandr.py:376). Read from HfConfig::raw; default 1.0.
double CommandrLogitScale(const HfConfig& config);

// Load `CohereForCausalLM` (bf16) safetensors + build the plain full-width GPT-J
// rope cos/sin cache. On-disk names (WeightsMapper merges q/k/v->qkv_proj,
// gate/up->gate_up_proj, commandr.py:343-350): model.embed_tokens.weight,
// model.norm.weight (weight-only, no .bias), and per layer
// model.layers.N.input_layernorm.weight, .self_attn.{q,k,v,o}_proj.weight,
// .mlp.{gate,up,down}_proj.weight. lm_head is SKIPPED (tied). Text path only.
// Reuses dense_weight_loaders.h.
CommandrWeights LoadCohereForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

// The Command-R dense forward (GPT-J parallel residual, weight-only LayerNorm).
// Per layer: residual = h; n = LayerNorm(h, w_in) [no bias]; attn = attn(n);
//   mlp = mlp(n); h = residual + attn + mlp. Then final LayerNorm -> tied lm_head
//   -> logits *= logit_scale. bf16 stream; LayerNorm accumulates mean/variance in
//   f32. Returns [n_out, vocab] f32 logits.
class CommandrModel {
 public:
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const CommandrWeights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {},
      const vllm::TensorParallel* tp = nullptr);

  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const CommandrWeights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {},
      const vllm::TensorParallel* tp = nullptr);
};

// Per-family config hook. Validates full-rotary head_dim, MHA/GQA head
// divisibility, tied embeddings, and rejects the Cohere2-only use_qk_norm /
// sliding_window (a separate arch).
void ParseCohereForCausalLMConfig(const HfConfig& config);

// KV-cache spec builder: exactly ONE full-attention KV group.
v1::KVCacheConfig MakeCohereForCausalLMKVCache(const HfConfig& config,
                                               int block_size, int num_blocks);

}  // namespace vllm
