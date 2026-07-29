// DeepSeek-V4-Flash (`DeepseekV4ForCausalLM`) — the ADDITIVE model TU skeleton
// for the DeepSeek-V4 bring-up campaign (`CLAIM-DEEPSEEK-V4-IMPL`, W1/W2). This
// header defines the config parse + the checkpoint-verified weight layout + the
// forward/KV-cache seams; the genuinely-NEW primitives (Manifold Hyper-
// Connections, the DSA Lightning-Indexer + Compressor, the 512-wide MLA geometry
// with grouped output-LoRA, sqrtsoftplus/hash MoE) are STUBBED with precise
// `TODO(W3-W8)` port markers and a `VT_CHECK(false, ...)` forward so the TU
// BUILDS but a forward loudly reports the pending brick — never a silent wrong
// answer. The full forward + the strict gate are NAMED residuals (W3-W8), see
// `.agents/specs/deepseek-v4-flash.md` §5.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides, @ pin 555967922 / 0.26.0.dev0) ──
//   OURS                             <-  UPSTREAM (all under vllm/models/deepseek_v4/)
//   DeepseekV4Params                 <-  transformers_utils/configs/deepseek_v4.py
//                                        + the shipped nvidia/DeepSeek-V4-Flash-NVFP4
//                                        config.json (arch scalars)
//   ParseDeepseekV4Params            <-  nvidia/model.py:512-757 (DeepseekV4MoE),
//                                        attention.py:122-315 (DeepseekV4Attention),
//                                        nvidia/model.py:794-957 (DecoderLayer + MHC)
//   LoadDeepseekV4ForCausalLMWeights <-  nvidia/model.py:1150-1310 (load_weights),
//                                        :1316-1360 (_make_deepseek_v4_weights_mapper)
//   MakeDeepseekV4KVCache            <-  attention.py:626 (get_kv_cache_spec) — STUB
//   DeepseekV4Model::Forward         <-  nvidia/model.py:1080-1148 — STUB (W3-W8)
//
// ─── LOADER VERIFIED vs the REAL checkpoint header (2026-07-28, no download) ────
// `nvidia/DeepSeek-V4-Flash-NVFP4` model.safetensors.index.json (135,235 tensors,
// 46 shards) + shard-2 safetensors header via HTTP range. Confirmed name map +
// dtypes/shapes (see deepseek_v4_weights.cpp for the full table). HW-FIT REVERSAL
// recorded there: index `total_size` = 168,266,793,544 B = **156.7 GiB**, so this
// NVFP4 checkpoint does NOT fit ONE GB10's 119 GiB unified pool — the W0 spike's
// "~83 GiB fits" estimate was wrong (only the 256 routed experts are NVFP4; the
// MLA/shared-expert linears are FP8 block, + NVFP4 double-scale overhead). W1
// (single-GB10 oracle run) is therefore MEMORY-INFEASIBLE, not merely disk-blocked.
#pragma once

#include <cstdint>
#include <string>
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

// Every DeepSeek-V4-Flash config field the loader/forward consume, resolved ONCE
// from the HfConfig. DeepSeek keys (`n_routed_experts`, `hc_mult`, `o_groups`,
// `compress_ratios`, ...) are NOT on the typed HfConfig struct, so most are read
// from `config.raw`. Values below are the shipped
// `nvidia/DeepSeek-V4-Flash-NVFP4` config.json (VERIFIED 2026-07-28).
struct DeepseekV4Params {
  // --- shared geometry ---
  int64_t hidden_size = 0;          // 4096
  int64_t num_hidden_layers = 0;    // 43
  int64_t vocab_size = 0;           // 129280
  int64_t num_attention_heads = 0;  // 64
  int64_t num_key_value_heads = 0;  // 1 (MLA: single latent)
  float rms_norm_eps = 1e-6f;       // 1e-6
  bool tie_word_embeddings = false;
  int64_t max_position_embeddings = 0;  // 1048576
  int64_t num_nextn_predict_layers = 0;  // 1 (MTP tail; loader SKIPS `mtp.*`)

  // --- 512-wide MLA (attention.py) — NEW geometry vs V2/V3 (448 NoPE + 64 RoPE) ---
  int64_t head_dim = 0;          // 512  (= 448 v/nope + 64 rope, wq_b out = heads*512)
  int64_t qk_rope_head_dim = 0;  // 64
  int64_t q_lora_rank = 0;       // 1024
  int64_t o_lora_rank = 0;       // 1024  (grouped OUTPUT LoRA — NEW, no V2/V3 analogue)
  int64_t o_groups = 0;          // 8
  int64_t sliding_window = 0;    // 128  (per-head attention sinks + SWA — NEW)
  double rope_theta = 10000.0;
  double compress_rope_theta = 160000.0;  // dual theta for compressed layers

  // --- MoE (nvidia/model.py:512-757) ---
  int64_t n_routed_experts = 0;      // 256
  int64_t num_experts_per_tok = 0;   // 6
  int64_t moe_intermediate_size = 0; // 2048
  int64_t n_shared_experts = 0;      // 1 (FP8 block, NOT NVFP4)
  bool norm_topk_prob = true;
  double routed_scaling_factor = 1.0;  // 1.5
  double swiglu_limit = 0.0;           // 10.0 (clamped SwiGLU — NEW)
  // "sqrtsoftplus" (NEW — not sigmoid/softmax) + noaux_tc e_score_correction_bias.
  std::string scoring_func = "sqrtsoftplus";
  std::string topk_method = "noaux_tc";
  // First `num_hash_layers` layers replace the learned gate with a `tid2eid`
  // token-id -> expert-id HASH lookup (NEW). Verified: layers 0,1,2 carry
  // `ffn.gate.tid2eid` and NO `ffn.gate.bias`.
  int64_t num_hash_layers = 0;  // 3
  // "fp4" (MXFP4/NVFP4 experts) | "fp8" (block experts). NVFP4 vehicle = "fp4".
  std::string expert_dtype = "fp4";

  // --- Manifold/Markov Hyper-Connections (MHC) — NEW topology, no eager ref ---
  int64_t hc_mult = 0;           // 4  (residual stream -> [T, hc_mult, H])
  int64_t hc_sinkhorn_iters = 0; // 20
  double hc_eps = 1e-6;

  // --- DSA Lightning-Indexer + Compressor (attention.py:689-857) — NEW ---
  int64_t index_head_dim = 0;  // 128
  int64_t index_n_heads = 0;   // 64
  int64_t index_topk = 0;      // 512
  // Per-layer compress ratio: 4 => indexed (has DSA indexer + compressor@4),
  // 128 => compressor@128 only, 0 => neither (verified: layers 0,1 and the last
  // carry 0). Length == num_hidden_layers.
  std::vector<int64_t> compress_ratios;

  bool is_hash_layer(int64_t layer) const { return layer < num_hash_layers; }
  int64_t compress_ratio(int64_t layer) const {
    return (layer >= 0 && static_cast<size_t>(layer) < compress_ratios.size())
               ? compress_ratios[static_cast<size_t>(layer)]
               : 0;
  }
  bool has_compressor(int64_t layer) const { return compress_ratio(layer) != 0; }
  bool has_indexer(int64_t layer) const { return compress_ratio(layer) == 4; }
};

// Resolve + validate DeepseekV4Params from a HfConfig. Pure/host — unit-testable
// without a checkpoint (config-descent gate). Throws with a precise message on a
// missing required field or a value this bring-up cannot represent.
DeepseekV4Params ParseDeepseekV4Params(const HfConfig& config);

// ─── W7 host-float weight tower (the tiny-config CPU forward assembly) ────────
// The W7 forward composes the landed host-reference primitives on the portable
// CPU path at a SMALL synthetic config (the fixed-config 167B does not fit ONE
// GB10, so the real e2e run is the multi-Spark W8 gate — see this header top).
// These structs hold the plain-float weights the CPU forward consumes; they are
// populated by the STRUCTURAL unit gate (test_deepseek_v4_forward.cpp) at a tiny
// shape, NOT by the real checkpoint loader — the FP8-block + NVFP4 tower
// MATERIALIZATION into this layout is the named W2b residual. All tensors row-major
// fp32 unless noted.
struct DeepseekV4LayerHostWeights {
  // MHC mixing (nvidia/model.py:820-865): hc_attn/hc_ffn fn [(2+hc)*hc, hc*H],
  // base [(2+hc)*hc], scale [3]; the attn/ffn RMSNorms folded into the pre-mix.
  std::vector<float> attn_norm_weight;  // [H]
  std::vector<float> ffn_norm_weight;   // [H]
  std::vector<float> hc_attn_fn, hc_attn_base, hc_attn_scale;
  std::vector<float> hc_ffn_fn, hc_ffn_base, hc_ffn_scale;
  // 512-wide MLA (attention.py): q down/up, kv down, per-branch RMSNorms,
  // per-head attention sink, grouped OUTPUT-LoRA wo_a (bmm) + wo_b.
  std::vector<float> wq_a;            // [q_lora_rank, H]
  std::vector<float> q_norm_weight;   // [q_lora_rank]
  std::vector<float> wq_b;            // [n_heads*head_dim, q_lora_rank]
  std::vector<float> wkv;             // [head_dim, H]
  std::vector<float> kv_norm_weight;  // [head_dim]
  std::vector<float> attn_sink;       // [n_heads]
  std::vector<float> wo_a;            // [n_groups, o_lora_rank, in_per_group]
  std::vector<float> wo_b;            // [H, n_groups*o_lora_rank]
  // DSA Lightning-Indexer (indexer layers only; empty otherwise).
  std::vector<float> idx_wq;     // [index_n_heads*index_head_dim, H]
  std::vector<float> idx_wk;     // [index_head_dim, H]
  std::vector<float> idx_wproj;  // [index_n_heads, H]
  // DSA compressor (compressor layers only; empty otherwise).
  std::vector<float> comp_wgate;        // [head_dim, H]  (produces the pool score)
  std::vector<float> comp_ape;          // [compress_ratio, head_dim]
  std::vector<float> comp_norm_weight;  // [head_dim]
  // MoE router: learned gate + (non-hash) noaux_tc bias OR (hash) tid2eid table.
  std::vector<float> gate_weight;  // [n_routed_experts, H]
  std::vector<float> gate_bias;    // [n_routed_experts]  (non-hash layers)
  std::vector<int32_t> tid2eid;    // [vocab, num_experts_per_tok] (hash layers)
  // Shared + routed experts (clamped SwiGLU). Routed stored flat over experts.
  std::vector<float> shared_w1, shared_w3;  // [moe_inter, H]
  std::vector<float> shared_w2;             // [H, moe_inter]
  std::vector<float> exp_w1, exp_w3;        // [n_experts, moe_inter, H]
  std::vector<float> exp_w2;                // [n_experts, H, moe_inter]
};
struct DeepseekV4HostWeights {
  std::vector<float> embed;              // [vocab, H]
  std::vector<float> lm_head;            // [vocab, H]  (untied)
  std::vector<float> final_norm_weight;  // [H]
  std::vector<float> hc_head_fn;         // [hc, hc*H]
  std::vector<float> hc_head_base;       // [hc]
  float hc_head_scale = 0.0f;            // scalar (hc_head_scale[0] upstream)
  std::vector<DeepseekV4LayerHostWeights> layers;
};

// Whole DeepSeek-V4 weights. W1/W2 SCAFFOLDING: carries the resolved params + the
// loader's accounting result. Heavy tensor MATERIALIZATION (FP8-block MLA linears,
// NVFP4 grouped experts, MHC mixing matrices, DSA indexer/compressor) is the named
// W2b residual — the loader ACCOUNTS for every checkpoint tensor but does not yet
// upload the quantized towers (TODO markers in deepseek_v4_weights.cpp). The W7
// tiny-config CPU forward runs off `host` when `has_host_weights` is set.
struct DeepseekV4Weights {
  DeepseekV4Params params{};
  // Accounting from the verified name-map pass (W2 gate): how many checkpoint
  // tensors the loader recognized / would consume. Diagnostic only.
  int64_t accounted_tensors = 0;
  // W7 host-float tower for the tiny-config CPU forward composition. NOT filled by
  // LoadDeepseekV4ForCausalLMWeights (that is the W2b residual); the structural
  // gate populates it directly. `Forward` VT_CHECKs this present.
  DeepseekV4HostWeights host{};
  bool has_host_weights = false;
  // TODO(W2b): the materialized quantized towers (FP8-block MLA linears + NVFP4
  // grouped experts) land as device OwnedTensors; the device forward (W7-device)
  // reads those, this host tower is the portable-CPU composition oracle.
};

// W7 structural-gate knobs. A deliberately-miswired interleave MUST change the
// output (RED-first) — the structural gate proves each lever is load-bearing.
enum class V4Miswire {
  kNone,              // the faithful interleave
  kAllLayersGated,    // route hash layers by learned top-k (ignore tid2eid)
  kSkipFinalMhcPost,  // skip the final residual fold before the head collapse
  kNoAttnSink,        // drop the per-head attention sink (plain softmax)
};

// Structural facts the W7 forward records for the composition gate (proves the
// interleave RAN as designed at tiny shape — not a numerical parity claim).
struct V4ForwardTrace {
  int64_t hc_mult = 0, hidden = 0, num_tokens = 0;
  int64_t residual_stream_elems = 0;  // == num_tokens*hc*hidden (proves [T,hc,H])
  std::vector<int> layer_is_hash;          // per layer: config says hash-routed
  std::vector<int> layer_hash_routed;      // per layer: the router took the hash branch
  std::vector<int> layer_is_indexer;       // per layer: DSA Lightning-Indexer present
  std::vector<int> layer_indexer_selected; // per layer: #keys the last query selected
  std::vector<int> layer_compressor_ran;   // per layer: the DSA compressor pooled KV
};

// The W7 tiny-config CPU forward: compose the landed host primitives (MHC pre/post
// + Sinkhorn, DSA indexer select + compressor + fp8_ds_mla KV, 512-wide MLA with
// sinks + grouped output-LoRA, sqrtsoftplus/hash MoE with clamped SwiGLU) into an
// end-to-end logits producer. Returns flat row-major logits for the `logits_indices`
// rows (all rows if empty). `miswire` deliberately breaks the interleave for the
// RED-first structural gate; `trace` (optional) records structural facts.
// Grounding: vllm/models/deepseek_v4/nvidia/model.py:1080-1148 (DeepseekV4Model.forward)
// + :866-957 (DeepseekV4DecoderLayer.forward).
std::vector<float> DeepseekV4ForwardHost(
    const DeepseekV4HostWeights& hw, const DeepseekV4Params& p,
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const std::vector<int32_t>& logits_indices = {},
    V4Miswire miswire = V4Miswire::kNone, V4ForwardTrace* trace = nullptr);

// Load `DeepseekV4ForCausalLM` safetensors into DeepseekV4Weights. Encodes the
// checkpoint name-map VERIFIED against the real header (deepseek_v4_weights.cpp)
// and performs the W2 accounting pass (throws on a missing expected tensor). The
// device materialization of the quantized towers is a NAMED W2b residual.
DeepseekV4Weights LoadDeepseekV4ForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

// The DeepSeek-V4 forward. STUB (W3-W8): composes the 512-wide MLA block + DSA
// indexer/compressor + MHC hyper-connections + sqrtsoftplus/hash MoE, none of
// which are ported yet — both entrypoints VT_CHECK(false, ...) so a forward
// LOUDLY reports the pending brick. See `.agents/specs/deepseek-v4-flash.md` §5.
class DeepseekV4Model {
 public:
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const DeepseekV4Weights& weights,
      vt::Queue& queue, const std::vector<int32_t>& logits_indices = {});

  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const DeepseekV4Weights& weights,
      vt::Queue& queue, const std::vector<int32_t>& logits_indices = {});
};

// Per-family config hook (registry `parse_config`): resolves + validates
// DeepseekV4Params and throws on anything unsupported.
void ParseDeepseekV4Config(const HfConfig& config);

// KV-cache spec builder. STUB (W3): V4 uses the fp8_ds_mla UE8M0 576B-paged latent
// KV (attention.py:89, :626) + the DSA indexer/compressor caches — a NEW geometry
// not yet representable. Emits a placeholder MLA spec sized to the compressed
// latent so the arch RESOLVES; the true multi-cache topology is a named W3 residual.
v1::KVCacheConfig MakeDeepseekV4KVCache(const HfConfig& config, int block_size,
                                        int num_blocks);

}  // namespace vllm
