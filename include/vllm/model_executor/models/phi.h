// Phi-1 / Phi-1.5 / Phi-2 (`PhiForCausalLM`, microsoft/phi-2) — a ZERO-NEW-KERNEL
// dense bring-up. This is the OLDER Phi architecture, DISTINCT from the landed
// `Phi3ForCausalLM` (phi3.h): Phi-1/2 is a GPT-J-style PARALLEL-residual decoder
// with nn.LayerNorm (weight AND bias), q/k/v/dense projection BIASES, partial NeoX
// RoPE, a NON-gated GELU MLP (fc1 -> gelu_new -> fc2, both biased), a final
// nn.LayerNorm (weight+bias) and an UNTIED lm_head that carries a BIAS. Every op
// REUSES a landed primitive:
//
//   * nn.LayerNorm (mean+variance, weight+bias) — vt::LayerNorm, landed for OPT.
//   * PARALLEL residual (ONE input_layernorm feeds BOTH attn and mlp, their
//     outputs summed back onto the residual together) — the wiring landed for
//     Command-R (commandr.cpp RunLayer).
//   * biased q/k/v/dense projections — vt::Add row-broadcast, the OPT BiasedProj
//     path (RowParallelLinear/ColumnParallelLinear default bias=True).
//   * partial NeoX RoPE (rotary_dim = int(head_dim * partial_rotary_factor) <
//     head_dim) from a precomputed plain cos/sin cache — vt::RopeFromCache, the
//     same partial path phi3/StableLM/GLM-4 use.
//   * NON-gated NewGELU MLP (fc1 + bias -> gelu_new -> fc2 + bias) — vt::GeluTanh,
//     the unary tanh-approx GELU landed for the Qwen3-VL vision tower. gelu_new
//     (activation.py:516-519) is 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))),
//     bit-identical to vt::GeluTanh's formula — NO new kernel.
//   * untied lm_head with a per-vocab bias — vt::Matmul + vt::Add(f32 logits,
//     bf16 bias broadcast).
//
// Upstream: vllm/model_executor/models/phi.py @ e24d1b24
//   - PhiLayer.forward (:189-202): residual = h; h = input_layernorm(h);
//     attn_out = self_attn(h); ffn_out = mlp(h); h = attn_out + ffn_out + residual.
//     ONE nn.LayerNorm feeds BOTH attn AND mlp (parallel residual).
//   - PhiAttention (:77-136): qkv_proj (bias=True) -> chunk -> rotary_emb (partial
//     NeoX, get_rope(head_size, rope_parameters)) -> attn(scale head_size**-0.5)
//     -> dense (RowParallelLinear, bias=True default).
//   - PhiMLP (:139-169): fc1 (ColumnParallelLinear, bias default True) ->
//     get_act_fn(hidden_act) [gelu_new] -> fc2 (RowParallelLinear, bias default
//     True). n_inner defaults to 4*hidden_size when absent.
//   - PhiModel (:206-257): embed_tokens -> N PhiLayers -> final_layernorm
//     (nn.LayerNorm weight+bias).
//   - PhiForCausalLM (:260-325): asserts NOT tie_word_embeddings; ParallelLMHead
//     (bias=True); logits = logits_processor(lm_head, h, lm_head.bias). WeightsMapper
//     merges q/k/v -> qkv_proj (orig_to_new_stacked).
//   - config microsoft/phi-2/config.json: hidden 2560, 32 heads (MHA: 32 kv heads),
//     32 layers, intermediate 10240, head_dim 80, partial_rotary_factor 0.4
//     (rotary_dim 32), layer_norm_eps 1e-5, rope_theta 10000, hidden_act gelu_new,
//     tie_word_embeddings False, qkv/dense/fc1/fc2/lm_head all biased.
//
// The forward is written FRESH (like OPT/Command-R/StableLM, NOT dense_attn::
// AttnBlock which hard-codes fused add+RMSNorm pre-norm + Qwen per-head qk-norm):
// Phi's parallel-residual LayerNorm block, biased projections and non-gated MLP
// cannot ride that block. It reuses all the shared device glue
// (Dev/DBuf/ResidentWeight/KvSlice/StepInputs). See
// .agents/specs/sweep-recent-dense-batch.md §0.2 row 7.
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

// Phi attention weights (phi.py::PhiAttention). Merged q|k|v in the on-disk
// torch-Linear [N=out, K=in] orientation (nk=true) for vt::MatmulBT. q/k/v ALWAYS
// carry bias (`bias=True`, phi.py:98) and `dense` (o_proj) ALWAYS carries bias
// (RowParallelLinear default bias=True, phi.py:102-107).
struct PhiAttnWeights {
  OwnedTensor qkv_proj;  // bf16 raw-NK [Hq*Dh + 2*Hkv*Dh, H] (rows q|k|v), nk
  OwnedTensor qkv_bias;  // bf16 [Hq*Dh + 2*Hkv*Dh]
  OwnedTensor dense;     // bf16 raw-NK [H, Hq*Dh], nk (the o_proj)
  OwnedTensor dense_bias;  // bf16 [H]
};

// Phi NON-gated GELU MLP (phi.py::PhiMLP): fc1 (+bias) -> gelu_new -> fc2 (+bias).
// NOT a SwiGLU: there is no gate, and no merged gate_up. Raw-NK like attention.
struct PhiMlpWeights {
  OwnedTensor fc1;       // bf16 raw-NK [ffn, H], nk
  OwnedTensor fc1_bias;  // bf16 [ffn]
  OwnedTensor fc2;       // bf16 raw-NK [H, ffn], nk
  OwnedTensor fc2_bias;  // bf16 [H]
};

// One Phi decoder layer (phi.py::PhiLayer). ONE nn.LayerNorm (weight+bias) feeds
// BOTH attention and MLP (parallel residual).
struct PhiLayerWeights {
  OwnedTensor input_layernorm;       // bf16 [H]
  OwnedTensor input_layernorm_bias;  // bf16 [H]
  PhiAttnWeights attn;
  PhiMlpWeights mlp;
};

// Whole Phi text-model weights (phi.py::PhiModel + PhiForCausalLM). `final_norm`
// is the decoder-level nn.LayerNorm (weight+bias). lm_head is UNTIED and carries
// a per-vocab bias (phi.py asserts NOT tie_word_embeddings). The rope cache is the
// plain per-position cos/sin cache [P, rotary_dim] (rope_type default), indexed by
// REAL position and applied to the leading rotary_dim.
struct PhiWeights {
  OwnedTensor embed_tokens;      // bf16 [vocab, H]
  OwnedTensor final_norm;        // bf16 [H]
  OwnedTensor final_norm_bias;   // bf16 [H]
  OwnedTensor lm_head;           // bf16 [H, vocab] Matmul-B (untied)
  OwnedTensor lm_head_bias;      // bf16 [vocab]
  std::vector<PhiLayerWeights> layers;
  OwnedTensor rope_cos_sin;      // bf16 [P, rotary_dim]; row p = angle for pos p
};

// nn.LayerNorm eps: `config.layer_norm_eps` (phi.py:182,225). Read from
// HfConfig::raw so no shared config POD is touched.
float PhiLayerNormEps(const HfConfig& config);

// The Phi MLP inner width: `n_inner` when present, else 4 * hidden_size
// (phi.py:148-149). microsoft/phi-2 leaves n_inner absent -> 4*2560 == 10240
// (== the config's intermediate_size).
int64_t PhiFfnDim(const HfConfig& config);

// Load `PhiForCausalLM` (microsoft/phi-2, BF16) safetensors + build the plain
// partial-rope cos/sin cache. On-disk names (WeightsMapper merges q/k/v->qkv_proj):
// model.embed_tokens.weight, model.final_layernorm.{weight,bias},
// lm_head.{weight,bias}, and per layer model.layers.N.input_layernorm.{weight,bias},
// .self_attn.{q,k,v}_proj.{weight,bias}, .self_attn.dense.{weight,bias},
// .mlp.{fc1,fc2}.{weight,bias}. Text path only. Reuses dense_weight_loaders.h.
PhiWeights LoadPhiForCausalLMWeights(const std::vector<SafetensorsFile>& shards,
                                     const HfConfig& config);

// The Phi dense forward (parallel-residual LayerNorm block). Per layer:
//   residual = h; n = LayerNorm(h, w_in, b_in);
//   attn = dense( PagedAttn( partialNeoXRoPE( split( qkv_proj(n) + qkv_bias ) ) ) )
//          + dense_bias;
//   mlp  = fc2( gelu_new( fc1(n) + fc1_bias ) ) + fc2_bias;
//   h = residual + attn + mlp.
// Then final LayerNorm -> untied lm_head + lm_head_bias. bf16 stream; LayerNorm
// accumulates mean/variance in f32. Returns [n_out, vocab] f32.
class PhiModel {
 public:
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const PhiWeights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {},
      const vllm::TensorParallel* tp = nullptr);

  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const PhiWeights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {},
      const vllm::TensorParallel* tp = nullptr);
};

// Per-family config hook. Validates the partial-rotary invariant (rotary_dim in
// (0, head_dim], default rope), MHA/GQA head divisibility, and untied embeddings.
void ParsePhiForCausalLMConfig(const HfConfig& config);

// KV-cache spec builder: exactly ONE full-attention KV group.
v1::KVCacheConfig MakePhiForCausalLMKVCache(const HfConfig& config, int block_size,
                                            int num_blocks);

}  // namespace vllm
