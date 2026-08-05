// Kimi-Linear-48B-A3B (`KimiLinearForCausalLM`) — the ADDITIVE W1 model TU for the
// Kimi-Linear structural bring-up (`CLAIM-KIMI-LINEAR-W1`, spike
// `.agents/specs/kimi-linear.md`). This header defines the standalone config
// parse (`ParseKimiLinearParams` over the hybrid KDA/NoPE-MLA + DeepSeek-style-MoE
// decoder), the checkpoint weight name-map (`EnumerateKimiLinearTensors`, grounded
// 1:1 in the pinned `kimi_linear.py` + the REAL HF safetensors index), the loader
// (throws BY NAME on any missing tensor), the heterogeneous KV-cache spec builder,
// and the forward/registry seams. W1 SCOPE: registry + config + loader scaffolding
// so the forward (W3-W6) can start; the forward REFUSES-by-name (`VT_CHECK(false)`,
// exactly like `deepseek_v4.{h,cpp}` / `kimi_k3.{h,cpp}`) — the TU BUILDS and the
// config/loader structure is unit-testable, but a forward LOUDLY reports the
// pending brick rather than returning a silent wrong answer.
//
// Unlike its 2.8T sibling Kimi-K3 (`kimi_k3.{h,cpp}`, DERIVE-AND-SHIP,
// MXFP4-refusing, ~12x over one GB10), Kimi-Linear-48B-A3B is plain bf16, FITS one
// GB10 (48.9B ~ 91.5 GiB, 0.77x the 119 GiB pool), and IS registered in the pinned
// vLLM oracle (`555967922`) which constructs + serves it — so it earns a REAL e2e
// SACRED token gate (spike §4/§8). K3's `EnumerateKimiK3TextBackboneTensors` is a
// nested wrapper's DERIVED map; THIS enumeration is the standalone 27-layer map
// VERIFIED against the shipped `moonshotai/Kimi-Linear-48B-A3B-Instruct` index
// (in particular the MoE block is `block_sparse_moe.*`, not `mlp.*`).
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides, @ pin 555967922) ─────────
//   OURS                        <-  UPSTREAM (vllm/, @ 0.26.0.dev0)
//   KimiLinearParams            <-  transformers_utils/configs/kimi_linear.py:11-148
//                                   (KimiLinearConfig; is_mla/is_moe/is_linear_attn/
//                                    is_kda_layer) + the fetched authoritative
//                                    moonshotai/Kimi-Linear-48B-A3B config.json
//   ParseKimiLinearParams       <-  configs/kimi_linear.py:56-148 (field descent +
//                                   the use_nope / q_lora_rank asserts at
//                                   kimi_linear.py:214-215)
//   EnumerateKimiLinearTensors  <-  kimi_linear.py:64-101 (KimiMLP), :104-177
//                                   (KimiMoE block_sparse_moe), :180-285
//                                   (KimiMLAAttention), :288-378 (KimiDecoderLayer),
//                                   :460-554 (load_weights) + kimi_gdn_linear_attn.py
//                                   :102-226 (KDA q/k/v/f_a/f_b/b/g_a/g_b/conv/
//                                   A_log/dt_bias/o_norm/o_proj)
//   LoadKimiLinearForCausalLMWeights <- kimi_linear.py:641-646 (AutoWeightsLoader)
//   MakeKimiLinearKVCache       <-  mamba_utils.py:274-294 (kda_state_shape) +
//                                   :130-137 (kda_state_dtype) for the KDA layers;
//                                   layers/mla.py latent-KV for the full-attn layers
//   KimiLinearModel::Forward    <-  kimi_linear.py:426-458 — REFUSE-by-name (W1)
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"  // PagedKvCache, ForwardLogits
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"  // CommonAttentionMetadata
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"

namespace vllm {

class SafetensorsFile;

// Every Kimi-Linear config field the loader / KV-cache builder / (future) forward
// consume, resolved ONCE from the standalone HfConfig. Unlike K3 there is NO
// `text_config` nesting: `KimiLinearForCausalLM` is the top-level architecture, so
// the shared scalars are read from the typed HfConfig (LoadHfConfig materialized
// them) and the KimiLinear-specific keys from `config.raw`.
struct KimiLinearParams {
  // --- shared geometry (config.json fetch 2026-08-05) ---
  int64_t hidden_size = 0;          // 2304
  int64_t num_hidden_layers = 0;    // 27
  int64_t vocab_size = 0;           // 163840
  int64_t num_attention_heads = 0;  // 32
  int64_t num_key_value_heads = 0;  // 32 (absent -> num_attention_heads)
  int64_t head_dim = 0;             // 72 (= 2304/32; unused by the MLA layers)
  int64_t intermediate_size = 0;    // 9216 (dense-MLP width, layer 0)
  float rms_norm_eps = 1e-5f;
  bool tie_word_embeddings = false;
  int64_t max_position_embeddings = 0;
  int64_t num_nextn_predict_layers = 0;  // 0 => NO MTP head in this checkpoint
  double rope_theta = 10000.0;

  // --- MLA geometry (NoPE, no-q-lora — kimi_linear.py:180-285) ---
  // kimi_linear.py:214-215 hard-asserts use_nope is True AND q_lora_rank is None:
  // the full-attn layers are position-encoding-free MLA (rotary_emb=None, :253).
  int64_t kv_lora_rank = 0;      // 512
  int64_t q_lora_rank = 0;       // 0 == null (the KimiLinear branch: direct q_proj)
  int64_t qk_nope_head_dim = 0;  // 128
  int64_t qk_rope_head_dim = 0;  // 64
  int64_t v_head_dim = 0;        // 128
  bool mla_use_nope = false;     // true

  // --- MoE (DeepSeek-style sigmoid noaux_tc — kimi_linear.py:104-177) ---
  int64_t num_experts = 0;            // 256
  int64_t num_experts_per_token = 0;  // 8
  int64_t num_shared_experts = 0;     // 1
  int64_t moe_intermediate_size = 0;  // 1024
  int64_t first_k_dense_replace = 0;  // 1 (layer 0 dense)
  int64_t moe_layer_freq = 1;
  double routed_scaling_factor = 1.0;  // 2.446
  bool moe_renormalize = true;
  int64_t num_expert_group = 1;  // trivial grouping (1)
  int64_t topk_group = 1;
  bool use_grouped_topk = true;
  // "sigmoid" | "softmax" (kimi_linear.py:96 assert). Kimi-Linear = sigmoid.
  std::string moe_router_activation_func = "sigmoid";

  // --- KDA (linear_attn_config — kimi_gdn_linear_attn.py:110-118) ---
  // is_kda_layer(l) == (l+1) in kda_layers (kimi_linear.py:144-148). 20 KDA + 7 MLA
  // of 27. num_heads 32 / head_dim 128 / short_conv 4.
  std::vector<int64_t> kda_layers;
  std::vector<int64_t> full_attn_layers;
  int64_t kda_num_heads = 0;               // 32
  int64_t kda_head_dim = 0;                // 128
  int64_t kda_short_conv_kernel_size = 0;  // 4
  bool has_linear_attn_config = false;

  // (l+1) in kda_layers; mirrors KimiLinearConfig.is_kda_layer (kimi_linear.py:144).
  bool is_kda_layer(int64_t layer_idx) const;
  // is_moe && layer_idx >= first_k_dense_replace && layer_idx % moe_layer_freq == 0
  // (kimi_linear.py:328-333). num_experts>0 => is_moe (kimi_linear.py:130).
  bool is_moe_layer(int64_t layer_idx) const;
  // The compressed-latent MLA page width: kv_lora_rank + qk_rope_head_dim (= 576).
  // No factor 2, no separate V (MLAAttentionSpec — kv_cache_interface.h:189-238).
  int64_t mla_head_size() const { return kv_lora_rank + qk_rope_head_dim; }
  // The KDA short-conv projection dim: proj_size + 2*proj_k_size = 3*num_heads*
  // head_dim (num_k_heads==num_heads) — mamba_utils.py:288-291. = 12288.
  int64_t kda_conv_dim() const {
    return 3 * kda_num_heads * kda_head_dim;
  }
};

// Resolve + validate KimiLinearParams from a standalone HfConfig. Pure/host —
// unit-testable without a checkpoint (config-descent gate). Throws with a precise
// message on a missing required field or a value this bring-up cannot represent
// (a non-NoPE MLA, a q-LoRA query branch, a missing linear_attn_config, a
// non-sigmoid/softmax router).
KimiLinearParams ParseKimiLinearParams(const HfConfig& config);

// Per-family config hook (registry `parse_config`): resolves + validates and
// throws on anything unsupported. The resolve IS the validation.
void ParseKimiLinearConfig(const HfConfig& config);

// The standalone checkpoint weight name-map of `KimiLinearForCausalLM`, grounded
// 1:1 in the pinned `kimi_linear.py` + `kimi_gdn_linear_attn.py` AND verified
// against the shipped moonshotai/Kimi-Linear-48B-A3B safetensors index. Per layer:
// KDA (is_kda_layer) vs NoPE-MLA self_attn, and dense KimiMLP (`mlp.*`) vs
// DeepSeek-style MoE (`block_sparse_moe.*`) MLP.
std::vector<std::string> EnumerateKimiLinearTensors(const KimiLinearParams& p);

// Whole Kimi-Linear weights. W1 STRUCTURAL SCAFFOLDING: carries the resolved
// params + the loader's coverage result. Heavy tensor MATERIALIZATION (device
// staging of the KDA conv/decay towers, the 256 MoE experts, the MLA projections)
// is the W2+ residual; the loader here validates the name-map coverage + shapes.
struct KimiLinearWeights {
  KimiLinearParams params{};
  // Total enumerated checkpoint tensors (structural size of the 27-layer hybrid).
  int64_t enumerated_tensors = 0;
  // How many enumerated tensors were present in the shards. Equals
  // `enumerated_tensors` on a successful load (the loader throws BY NAME on the
  // first missing tensor, so a partial checkpoint never returns silently).
  int64_t accounted_tensors = 0;
};

// Load `KimiLinearForCausalLM` safetensors. Throws BY NAME (never silent zeros) on
// the FIRST enumerated tensor absent from the shards, and on a rank/shape mismatch
// for the tensors whose geometry is unambiguous from the config (the 2-D
// projections + the norms). W1 SCAFFOLDING: this proves the name-map + shapes; the
// device materialization is W2+ (the forward still REFUSES-by-name).
KimiLinearWeights LoadKimiLinearForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

// The Kimi-Linear forward. REFUSE-by-name (both entrypoints VT_CHECK(false, ...)):
// the 27-layer KDA/NoPE-MLA hybrid + 256-expert MoE forward composes primitives
// that are NOT wired in W1 — the KDA device kernel (W3), the NoPE-MLA routing (W4),
// the MoE assembly (W5), and the het-KV forward wiring (W6). A forward LOUDLY
// reports the pending brick. See `.agents/specs/kimi-linear.md` §5.
class KimiLinearModel {
 public:
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const KimiLinearWeights& weights,
      vt::Queue& queue, const std::vector<int32_t>& logits_indices = {});

  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const KimiLinearWeights& weights,
      vt::Queue& queue, const std::vector<int32_t>& logits_indices = {});
};

// KV-cache spec builder. The HETEROGENEOUS per-layer topology (spike §3): ONE MLA
// latent-KV group for the 7 full-attn layers (kv_lora_rank+qk_rope_head_dim wide,
// num_kv_heads==1, no separate V) + ONE KDA/GDN recurrent-state MambaSpec group for
// the 20 KDA layers (conv-state + recurrent-state). W1 DECLARES the shapes/routing;
// the runner wiring is W6. Mirrors the qwen3_5 GDN-hybrid two-group pattern
// (qwen3_5_common.cpp:65-105) with an MLA group in place of the full-attention one.
v1::KVCacheConfig MakeKimiLinearKVCache(const HfConfig& config, int block_size,
                                        int num_blocks);

}  // namespace vllm
