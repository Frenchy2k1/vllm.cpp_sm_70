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

// ─── HOST (float) MATERIALIZED WEIGHTS — the CPU reference forward (W2-W6) ─────
// The CPU reference forward composes the landed host primitives (`vllm::kimi_kda`
// for the KDA deltas, a materialized-MHA MLA reference, a sigmoid-`noaux_tc` MoE
// reference) into the whole 27-layer hybrid so the ONLY remaining correctness step
// is the e2e SACRED token golden on GB10 (spike §4/§8, W0/W7). These are plain
// row-major float arrays (bf16/f32 checkpoint tensors decoded to f32); the DEVICE
// weights (ResidentWeight staging, the absorbed W_UK/W_UV, the grouped-MoE slabs)
// are the born-on-the-runner W6/W7 residual, kept refuse-by-name in ForwardDevice.
// All matrices are torch `[out_features, in_features]` row-major (checkpoint layout).

// One KDA (Kimi Delta Attention) layer's materialized weights
// (kimi_gdn_linear_attn.py:120-226).
struct KdaLayerHostWeights {
  std::vector<float> q_proj, k_proj, v_proj;   // [num_heads*head_dim, hidden]
  std::vector<float> f_a_proj;                 // [head_dim, hidden]     (rank down)
  std::vector<float> f_b_proj;                 // [num_heads*head_dim, head_dim]
  std::vector<float> b_proj;                   // [num_heads, hidden]    (beta)
  std::vector<float> g_a_proj;                 // [head_dim, hidden]     (gate down)
  std::vector<float> g_b_proj;                 // [num_heads*head_dim, head_dim]
  std::vector<float> o_proj;                   // [hidden, num_heads*head_dim]
  std::vector<float> q_conv, k_conv, v_conv;   // [num_heads*head_dim, kernel_size]
  std::vector<float> dt_bias;                  // [num_heads*head_dim]   (may be empty)
  std::vector<float> a_log;                    // [num_heads]
  std::vector<float> o_norm;                   // [head_dim]  (FusedRMSNormGated weight)
};

// One NoPE-MLA layer's materialized weights (KimiMLAAttention, kimi_linear.py:217-248;
// q_lora_rank==null => direct q_proj, mla_use_nope => rotary_emb=None).
struct MlaLayerHostWeights {
  std::vector<float> q_proj;              // [num_heads*(qk_nope+qk_rope), hidden]
  std::vector<float> kv_a_proj_with_mqa;  // [kv_lora+qk_rope, hidden]
  std::vector<float> kv_a_layernorm;      // [kv_lora]
  std::vector<float> kv_b_proj;           // [num_heads*(qk_nope+v), kv_lora]
  std::vector<float> o_proj;              // [hidden, num_heads*v]
};

// A gated (SwiGLU) MLP — the dense layer-0 MLP AND each shared/routed expert.
// SiluAndMul: silu(gate @ x) * (up @ x) -> down @ (...).
struct MlpHostWeights {
  std::vector<float> gate_proj;  // [inter, hidden]   (== expert w1)
  std::vector<float> up_proj;    // [inter, hidden]   (== expert w3)
  std::vector<float> down_proj;  // [hidden, inter]   (== expert w2)
};

// One MoE block (KimiMoE, kimi_linear.py:104-177): the sigmoid `noaux_tc` router
// gate + its `e_score_correction_bias`, an optional shared expert, and the routed
// experts. Router weights (`w1`/`w2`/`w3`) map to gate/down/up
// (fused_moe_make_expert_params_mapping, kimi_linear.py:469-475).
struct MoeHostWeights {
  std::vector<float> gate;                     // [num_experts, hidden]
  std::vector<float> e_score_correction_bias;  // [num_experts]
  bool has_shared = false;
  MlpHostWeights shared;                       // shared expert (moe_inter*num_shared)
  std::vector<MlpHostWeights> experts;         // [num_experts]
};

struct KimiLinearLayerHostWeights {
  std::vector<float> input_layernorm;           // [hidden]
  std::vector<float> post_attention_layernorm;  // [hidden]
  bool is_kda = false;
  KdaLayerHostWeights kda;  // populated iff is_kda
  MlaLayerHostWeights mla;  // populated iff !is_kda
  bool is_moe = false;
  MlpHostWeights dense;   // populated iff !is_moe (dense layer-0)
  MoeHostWeights moe;     // populated iff is_moe
};

struct KimiLinearHostWeights {
  bool materialized = false;
  std::vector<float> embed_tokens;  // [vocab, hidden]
  std::vector<float> final_norm;    // [hidden]
  std::vector<float> lm_head;       // [vocab, hidden]  (== embed if tied)
  std::vector<KimiLinearLayerHostWeights> layers;
};

// ─── bf16-RESIDENT WEIGHTS — the full-model device forward (§13) ────────────────
// The full 48.9B model is 91.5 GiB bf16; materializing it to f32
// (KimiLinearHostWeights) is ~183 GiB and OOM-reboots the 119 GiB unified pool. The
// resident path keeps every LARGE matmul weight as a bf16 `OwnedTensor` (cudaMalloc'd
// `d_dev`, NO ATS penalty — the gb10-weight-residency lever) consumed via the
// (bf16,bf16)->f32 `vt::MatmulBT` (vLLM's own numerics, best token-exact chance vs the
// bf16 oracle golden), and keeps only the TINY norm/scale/conv/router vectors as host
// f32 (ReadF32 from bf16 at load; < 0.1 GiB total, aliased into device kernels via the
// GB10 unified pool). Activations stay f32 (the residual stream; the two host-fallback
// islands — KDA recurrence, NoPE-MLA softmax — need f32) with a per-GEMM bf16 cast.
// STAGED per tensor at load then `ReleaseHost` (only ONE tensor's host bytes live at a
// time; LOAD peak ~91.5 GiB device + <1 GiB host). Loaded by
// `LoadKimiLinearResidentBf16Weights` (NEVER `MaterializeHost`); the tiny-config unit
// gate builds it from the f32 host weights (`BuildKimiResidentFromHost`). Mirrors the
// laguna_weights.cpp / gemma_weights.cpp OwnedTensor loaders + laguna.cpp:125-139
// residency + the Laguna cast-GEMM (laguna.cpp:1939-1946).
struct KdaResidentWeights {
  OwnedTensor q_proj, k_proj, v_proj;  // bf16 [num_heads*head_dim, hidden]
  OwnedTensor f_a_proj;                // bf16 [head_dim, hidden]
  OwnedTensor f_b_proj;                // bf16 [num_heads*head_dim, head_dim]
  OwnedTensor b_proj;                  // bf16 [num_heads, hidden]
  OwnedTensor g_a_proj;                // bf16 [head_dim, hidden]
  OwnedTensor g_b_proj;                // bf16 [num_heads*head_dim, head_dim]
  OwnedTensor o_proj;                  // bf16 [hidden, num_heads*head_dim]
  std::vector<float> q_conv, k_conv, v_conv;  // f32 [proj, K]  (short conv; small)
  std::vector<float> dt_bias;          // f32 [proj]      (host recurrence input)
  std::vector<float> a_log;            // f32 [num_heads] (host recurrence input)
  std::vector<float> o_norm;           // f32 [head_dim]  (RmsNormGated weight)
};
struct MlaResidentWeights {
  OwnedTensor q_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj;  // bf16
  std::vector<float> kv_a_layernorm;   // f32 [kv_lora]
  // ROW 7 (§20.3c) — the ABSORBED decode forms for the paged-FA2 NoPE-MLA arm
  // (mla::ForwardMlaAttentionBlock): W_UK_T [nah, qk_nope, kv_lora] and W_UV
  // [nah, kv_lora, v_head] bf16, computed from kv_b_proj at LOAD time
  // (mla::AbsorbKvBProjBf16 — the host bf16 bytes are released after staging, so
  // absorption cannot be deferred). Populated by BOTH resident builders; empty
  // only on a pre-fold checkpoint of the struct (the paged-FA2 arm refuses).
  OwnedTensor w_uk_t, w_uv;
};
struct MlpResidentWeights {
  OwnedTensor gate_proj, up_proj, down_proj;  // bf16
};
struct MoeResidentWeights {
  // bf16 [num_experts, hidden]: the router gate goes through the SAME (bf16,bf16)->f32
  // GemmBf16 as vLLM's bf16 router — a f32 gate could flip near-tie expert selections
  // in the 256-expert top-8 MoE (the primary token-divergence risk vs the oracle).
  OwnedTensor gate;
  std::vector<float> e_score_correction_bias;  // f32 [num_experts] (added to scores)
  bool has_shared = false;
  MlpResidentWeights shared;
  std::vector<MlpResidentWeights> experts;
};
struct KimiLinearLayerResidentWeights {
  std::vector<float> input_layernorm;           // f32 [hidden]
  std::vector<float> post_attention_layernorm;  // f32 [hidden]
  bool is_kda = false;
  KdaResidentWeights kda;   // populated iff is_kda
  MlaResidentWeights mla;   // populated iff !is_kda
  bool is_moe = false;
  MlpResidentWeights dense; // populated iff !is_moe (dense layer-0)
  MoeResidentWeights moe;   // populated iff is_moe
};
struct KimiLinearResidentWeights {
  bool resident = false;
  OwnedTensor embed_tokens;       // bf16 [vocab, hidden]
  std::vector<float> final_norm;  // f32 [hidden]
  OwnedTensor lm_head;            // bf16 [vocab, hidden] (Empty() => tied, use embed)
  bool tie_word_embeddings = false;
  std::vector<KimiLinearLayerResidentWeights> layers;
};

// Whole Kimi-Linear weights. W1 landed the resolved params + the loader's coverage
// result; W2 (the CPU reference lane) also MATERIALIZES the host float weights that
// the reference forward composes. The bf16-RESIDENT lane (§13) populates `resident`
// instead of `host` for the full-model device forward (host f32 would OOM).
struct KimiLinearWeights {
  KimiLinearParams params{};
  // Total enumerated checkpoint tensors (structural size of the 27-layer hybrid).
  int64_t enumerated_tensors = 0;
  // How many enumerated tensors were present in the shards. Equals
  // `enumerated_tensors` on a successful load (the loader throws BY NAME on the
  // first missing tensor, so a partial checkpoint never returns silently).
  int64_t accounted_tensors = 0;
  // The host-materialized float weights the CPU reference forward composes (W2, tiny
  // config only for the full model — 183 GiB f32 OOMs the pool).
  KimiLinearHostWeights host{};
  // The bf16-resident weights the full-model device forward composes (§13). Set by
  // LoadKimiLinearResidentBf16Weights (full model) or BuildKimiResidentFromHost (tiny
  // gate); host stays unmaterialized on the full-model resident path.
  KimiLinearResidentWeights resident{};
};

// ─── PAGED-INCREMENTAL DECODE STATE (§18 lever e) ──────────────────────────────
// The persistent per-sequence state that turns the O(n²) full-recompute vehicle
// (ForwardDeviceCompute re-runs [0..prompt+t] every step — the 4.24 tok/s rate)
// into vLLM's paged-incremental decode: chunk/recurrent-PREFILL the prompt ONCE,
// capturing the KDA recurrent state + short-conv taps per KDA layer and the paged
// latent-KV per NoPE-MLA layer, then RECURRENT-decode each token from the CARRIED
// state (mirrors kimi_gdn_linear_attn.py:233-268 prefill=chunk / decode=recurrent).
// Kept host-resident f32 (GB10 unified pool: the up/download is a cheap memcpy) so
// it survives across the per-token forward calls byte-identically. Sizes at the real
// 48.9B config: KDA state 20 layers × 32 heads × 128×128 f32 = 40 MiB, conv taps
// 20 × 3 × 12288 × 3 f32 ≈ 8 MiB, MLA latent-KV 7 layers × T × (8192+64) f32 (≈2 MiB
// at T≈36) — well inside the §13 budget.
struct KimiKdaLayerCache {
  // Short-conv taps [proj*(K-1)] for q/k/v (the last K-1 token values); the
  // CausalConv1dFwd final-state carry (mamba conv decode). Empty => fresh zeros.
  std::vector<float> conv_q, conv_k, conv_v;
  // The gated-delta recurrent state [nh*hd*hd] f32 (KdaGatedDeltaRule state in/out).
  std::vector<float> recurrent;
};
struct KimiMlaLayerCache {
  // Growing paged latent-KV: per cached token, kv[kvw]=nah*(qk_nope+v_head_dim)
  // (per-head k_nope|v) and kpe[qr] (the shared rope key). Appended one row per
  // token in prefill (all prompt tokens) and per decode step.
  std::vector<float> kv;
  std::vector<float> kpe;
};
struct KimiDecodeCache {
  std::vector<KimiKdaLayerCache> kda;  // one per KDA layer (in KDA-layer order)
  std::vector<KimiMlaLayerCache> mla;  // one per NoPE-MLA layer (in MLA-layer order)
  int64_t seq_len = 0;                 // tokens absorbed so far (prompt + generated)
  bool prefilled = false;              // ForwardPrefillIncremental ran
};

// Load `KimiLinearForCausalLM` safetensors. Throws BY NAME (never silent zeros) on
// the FIRST enumerated tensor absent from the shards, and on a rank/shape mismatch
// for the tensors whose geometry is unambiguous from the config (the 2-D
// projections + the norms). W1 SCAFFOLDING: this proves the name-map + shapes; the
// device materialization is W2+ (the forward still REFUSES-by-name).
KimiLinearWeights LoadKimiLinearForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

// ─── bf16-RESIDENT loader (§13) — the full-model device-forward weights ─────────
// Load `KimiLinearForCausalLM` as bf16-resident weights (NEVER the 183 GiB f32
// MaterializeHost): every large matmul weight via `dense_loaders::LoadBf16Direct`
// into an `OwnedTensor`, the tiny norm/scale/conv/router vectors decoded to host f32.
// When `stage_queue` is a CUDA queue, each large tensor is STAGED to `d_dev`
// (cudaMalloc + one H2D) and `ReleaseHost`'d immediately after its copy, so the LOAD
// peak holds only ONE tensor's host bytes on top of the growing ~91.5 GiB device
// residency (the pool-math constraint, §13). When `stage_queue` is null (or CPU) the
// host bf16 bytes are kept and aliased on demand (the CPU tiny-config path). Throws
// BY NAME on the first missing tensor (same coverage/shape checks as the f32 loader).
KimiLinearWeights LoadKimiLinearResidentBf16Weights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config,
    vt::Queue* stage_queue);

// Build the bf16-resident weights from already-materialized host f32 weights (the
// tiny-config unit gate: round each large matmul weight f32->bf16 into an OwnedTensor,
// copy the tiny vectors verbatim as host f32). This exercises the SAME bf16 device
// forward the full model uses, against the independent f32 host reference.
KimiLinearResidentWeights BuildKimiResidentFromHost(const KimiLinearHostWeights& host,
                                                    const KimiLinearParams& p);

// Stage a bf16 OwnedTensor to a CUDA device-resident d_dev copy (cudaMalloc + one
// H2D, byte-exact) then release its host bytes — the per-tensor stage-then-release
// the resident loader interleaves with each LoadBf16Direct. No-op on CPU (the host
// bytes are aliased directly) and when already staged / host already released.
void StageKimiResidentBf16(vt::Queue& queue, const OwnedTensor& w);

// ─── PER-OP CPU REFERENCE FORWARDS (W2-W6, kimi_linear_forward.cpp) ────────────
// Each is a full-sequence, fresh-state reference over the host float weights,
// grounded 1:1 in the pinned vLLM forward + composed from the already-gated
// `vllm::kimi_kda` host refs. They are the per-op gates (test_kimi_linear_forward):
// the layer output must equal a hand-composition of the parts (a wiring proof), and
// the whole forward decodes coherently. The DEVICE (born-on-runner) forms are W6/W7.

// One KDA linear-attention layer (kimi_gdn_linear_attn.py:233-268 forward +
// :390-441 recurrence). `hidden_normed` is [num_tokens, hidden] already through the
// input RMSNorm. Composes: q/k/v proj -> 3 silu short convs -> per-head q/k L2-norm
// -> the gated-delta recurrence (`fused_recurrent_kda`, kda gate g = -exp(A_log)*
// softplus(f_b(f_a(x))+dt_bias), scale head_dim**-0.5) -> FusedRMSNormGated(., g2)
// -> o_proj. Returns [num_tokens, hidden].
std::vector<float> KimiKdaLayerForward(const KdaLayerHostWeights& w,
                                       const std::vector<float>& hidden_normed,
                                       const KimiLinearParams& p, int64_t num_tokens);

// One NoPE-MLA full-attention layer (KimiMLAAttention, kimi_linear.py:180-285;
// scaling = qk_head_dim**-0.5, kimi_linear.py:212). The MATERIALIZED-MHA reference
// (mla_attention.h "the UNABSORBED materialized-MHA reference"): q_proj, kv_a ->
// kv_a_layernorm(latent) -> kv_b -> per-head causal softmax over the cache with NO
// RoPE (rotary_emb=None) -> o_proj. Returns [num_tokens, hidden].
std::vector<float> KimiNoPEMlaLayerForward(const MlaLayerHostWeights& w,
                                           const std::vector<float>& hidden_normed,
                                           const KimiLinearParams& p,
                                           int64_t num_tokens);

// The sigmoid `noaux_tc` top-k routing (grouped_topk_router.py:106-161, the trivial
// `num_expert_group=1`/`topk_group=1` case): scores=sigmoid(gate@x); select top-k on
// scores+e_score_correction_bias; weight from the UNBIASED scores; renormalize then
// scale by routed_scaling_factor. `ids`/`weights` are row-major [num_tokens, top_k].
struct KimiMoeRouting {
  std::vector<int32_t> ids;
  std::vector<float> weights;
};
KimiMoeRouting KimiMoeRoute(const MoeHostWeights& w,
                            const std::vector<float>& hidden_normed,
                            const KimiLinearParams& p, int64_t num_tokens);

// The whole MoE block: router (above) -> shared expert (always) + routed experts.
// Returns [num_tokens, hidden].
std::vector<float> KimiMoeBlockForward(const MoeHostWeights& w,
                                       const std::vector<float>& hidden_normed,
                                       const KimiLinearParams& p, int64_t num_tokens);

// The dense layer-0 SwiGLU MLP (KimiMLP, kimi_linear.py:64-101). Returns
// [num_tokens, hidden].
std::vector<float> KimiDenseMlpForward(const MlpHostWeights& w,
                                       const std::vector<float>& hidden_normed,
                                       const KimiLinearParams& p, int64_t num_tokens);

// Greedy-decode `num_new` tokens from `prompt`, advancing the (recomputed) KDA
// recurrent + MLA latent context each step (the CPU reference decode driver). The
// device incremental-cache decode is the born-on-runner W6/W7 residual. Returns the
// `num_new` generated token ids.
std::vector<int32_t> KimiLinearGreedyDecode(const KimiLinearHostWeights& host,
                                            const KimiLinearParams& p,
                                            const std::vector<int32_t>& prompt,
                                            int num_new);

// ─── PER-OP DEVICE-COMPUTE PRIMITIVES (W7, kimi_linear_device.cpp) ──────────────
// The REAL DBuf-resident device-compute lane — each primitive routes the whole
// per-layer/per-block computation through the shared vt:: device ops on POOLED
// DBufs (embed/GEMMs/norms/convs/L2Norm/gated-norm/MoE router+combine), with two
// documented HOST-FALLBACK islands where no portable device op yet expresses the
// KDA-specific numerics (the per-k-channel gated-delta RECURRENCE + its exp/
// softplus decay gate — vt::GdnDecode carries only a per-HEAD scalar decay, ops.h
// GdnPrefill/GdnDecode "g/beta[T,Hv]") and the NoPE-MLA attention CORE (the device
// path is mla::ForwardMlaAttentionBlock over the runner's paged het-KV + the
// load-time W_UK/W_UV absorption — the born-on-runner residual). These islands are
// the W7-speed residuals; every OTHER op is genuine on-device vt:: dispatch.
//
// Because the CPU backend executes the SAME vt:: dispatch (a pooled DBuf is a
// device buffer on CPU too, and ResidentWeight ALIASES the host weight bytes on
// CPU), running these on a CPU queue and matching the W2 host f32 reference is a
// REAL wiring gate — the residual/vt-op/routing plumbing the GPU will run, proven
// on CPU. Activations are f32 DBufs (the reference holds f32), so the match is
// tight (f32-accumulation-order only). GPU numerics (bf16 activations for vLLM
// parity, the GDN Triton-AOT decode cubins, the paged het-KV, the grouped-MoE
// slabs) stay a NAMED pending — box down. Each returns the host [num_tokens,*]
// result of the device compute so the per-op gate can compare to the W2 reference.
std::vector<float> KimiKdaLayerForwardDevice(const KdaLayerHostWeights& w,
                                             const std::vector<float>& hidden_normed,
                                             const KimiLinearParams& p,
                                             int64_t num_tokens, vt::Queue& queue);
std::vector<float> KimiNoPEMlaLayerForwardDevice(const MlaLayerHostWeights& w,
                                                 const std::vector<float>& hidden_normed,
                                                 const KimiLinearParams& p,
                                                 int64_t num_tokens, vt::Queue& queue);
std::vector<float> KimiMoeBlockForwardDevice(const MoeHostWeights& w,
                                             const std::vector<float>& hidden_normed,
                                             const KimiLinearParams& p,
                                             int64_t num_tokens, vt::Queue& queue);
// Device MLA attention CORE only (VT_KIMI_DEVICE_MLA path: pad-V + vt::Attention),
// host-in / host-out — the RED-first CPU gate for the NoPE-MLA device attention wiring.
std::vector<float> KimiMlaAttnCoreDevice(const std::vector<float>& q_host,
                                         const std::vector<float>& kv_host,
                                         const std::vector<float>& kpe_host,
                                         const KimiLinearParams& p, int64_t num_tokens,
                                         vt::Queue& queue);
std::vector<float> KimiDenseMlpForwardDevice(const MlpHostWeights& w,
                                             const std::vector<float>& hidden_normed,
                                             const KimiLinearParams& p,
                                             int64_t num_tokens, vt::Queue& queue);

// Opt-in switch (VT_KIMI_DEVICE_COMPUTE, default OFF) that makes the runner
// `ForwardDevice` seam route through `ForwardDeviceCompute` (the real device
// compute) instead of the W6 host-reference compose. Default-OFF keeps the
// CPU-VERIFIED host-ref-compose seam as the production runner path until the
// device compute is GPU-verified against the SACRED oracle; the flag exists so the
// device compute CAN be exercised as the runner path for that verification when
// the box is up. Mirrors the project's gated-default-OFF in-flight convention.
bool KimiDeviceComputeEnabled();

// The Kimi-Linear forward.
//   Forward (host, the `!gather_logits` reference path): a REAL CPU reference over
//     the host float weights — the whole 27-layer KDA/NoPE-MLA hybrid + 256-expert
//     MoE, composed from the per-op reference forwards above, so the ONLY remaining
//     correctness step is the e2e SACRED token golden on GB10 (spike §8).
//   ForwardDevice (the DEFAULT `gather_logits` production/runner path, W6): the
//     born-on-the-runner SEAM — it composes the [rows,vocab] logits via `Forward`
//     and returns them DEVICE-RESIDENT (a pooled `DBuf`, `ForwardLogits.on_device()
//     ==true` on CPU and CUDA), so the runner's on-GPU sampler consumes them with NO
//     host logit download (the third MUST-route seam). The device-COMPUTE lane —
//     routing KDA through the GDN device family, the NoPE-MLA layers through
//     `mla::ForwardMlaAttentionBlock`, and the MoE through the DeepSeek-V2 grouped
//     GEMM over the paged het-KV caches — is the GPU-verify-pending W7 residual
//     (kimi_linear.cpp documents the full reuse-wiring plan). The row stays SPIKE
//     until the device compute + the e2e SACRED golden land on GB10.
class KimiLinearModel {
 public:
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const KimiLinearWeights& weights,
      vt::Queue& queue, const std::vector<int32_t>& logits_indices = {},
      const vllm::TensorParallel* tp = nullptr);

  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const KimiLinearWeights& weights,
      vt::Queue& queue, const std::vector<int32_t>& logits_indices = {},
      const vllm::TensorParallel* tp = nullptr);

  // W7 — the REAL DBuf-resident device COMPUTE forward. Composes the whole
  // 27-layer KDA/NoPE-MLA + 256-expert-MoE hybrid over POOLED f32 DBufs through
  // the shared vt:: device ops (embed -> per-layer FusedChain add+RMSNorm ->
  // KDA/NoPE-MLA self-attn -> FusedChain -> dense/MoE MLP -> final FusedChain ->
  // lm_head), returning DEVICE-RESIDENT [rows,vocab] f32 logits (the same pooled-
  // DBuf carrier ForwardDevice returns). The two documented host-fallback islands
  // (the KDA per-k-channel recurrence + decay gate; the NoPE-MLA attention core)
  // are the W7-speed residuals. CPU-gated against the W2 host reference
  // (KimiLinearModel::Forward) — see the per-op primitives above. GPU numerics are
  // a NAMED pending. Reachable as the runner path via VT_KIMI_DEVICE_COMPUTE=1.
  static ForwardLogits ForwardDeviceCompute(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const KimiLinearWeights& weights,
      vt::Queue& queue, const std::vector<int32_t>& logits_indices = {},
      const vllm::TensorParallel* tp = nullptr);

  // ─── PAGED-INCREMENTAL DECODE (§18 lever e) ──────────────────────────────────
  // The paged-incremental twin of ForwardDeviceCompute: instead of re-running the
  // whole [0..prompt+t] sequence every step (O(n²), the 4.24 tok/s vehicle), it
  // PREFILLS the prompt ONCE (capturing the KDA recurrent+conv state per KDA layer
  // and the latent-KV per NoPE-MLA layer into `cache`), then decodes ONE token per
  // step from the CARRIED state — killing the recompute (vLLM's decode regime).
  //
  // ForwardPrefillIncremental: runs the `prompt` tokens through the bf16 device
  // forward, filling `cache` (sizes the per-layer caches, appends the prompt's KV,
  // captures the KDA states) and returns the DEVICE-RESIDENT last-token logits (via
  // `logits_indices`, request order). Uses the CHUNKED KDA prefill (vt::KdaChunk
  // Prefill — vLLM's prefill path) when VT_KIMI_DEVICE_KDA_CHUNK=1, else the
  // recurrence (byte-exact vs ForwardDeviceCompute's device-KDA path — the
  // token-identity gate).
  static ForwardLogits ForwardPrefillIncremental(
      const std::vector<int32_t>& prompt, const std::vector<int32_t>& positions,
      const KimiLinearWeights& weights, vt::Queue& queue, KimiDecodeCache& cache,
      const std::vector<int32_t>& logits_indices = {},
      const vllm::TensorParallel* tp = nullptr);

  // ForwardDecodeStepIncremental: advances ONE token from the carried `cache` — the
  // KDA layers via the recurrence (vt::KdaGatedDeltaRule, T==1) from the carried
  // state; the NoPE-MLA layers via a causal softmax of the 1 query over the carried
  // latent-KV (query_len=1, key_len=cache.seq_len+1). Returns the DEVICE-RESIDENT
  // [1,vocab] logits of the new token. `cache.seq_len` advances by one.
  static ForwardLogits ForwardDecodeStepIncremental(
      int32_t token, int64_t position, const KimiLinearWeights& weights,
      vt::Queue& queue, KimiDecodeCache& cache,
      const vllm::TensorParallel* tp = nullptr);

  // ─── ROW 7 — THE SHARED-PAGED-RUNNER FOLD (§20.3, ARCH-ONE-SURFACE req 4) ────
  // The born-on-the-runner PRODUCTION forward: the whole 27-layer hybrid over the
  // runner's OWN paged state — the KDA conv+recurrent state lives in the runner's
  // MambaSpec `gdn_state` group keyed by `gdn_meta.non_spec_state_indices_tensor`
  // (mirror vLLM kimi_gdn_linear_attn._forward's (conv_state, recurrent_state)
  // slot handling), and the NoPE-MLA latent-KV lives in the paged `attn_kv` MLA
  // group written through vt::ConcatAndCacheMla at `attn_meta.slot_mapping`.
  // Prefill = vt::KdaChunkPrefill (vLLM's prompt path; recurrence when the
  // request continues an existing state); decode = vt::KdaGatedDeltaRule (T==1).
  // Batched decode-first (nd decodes then np prefills, the GDN builder's
  // segmentation); byte-for-byte the DeviceForwardBodyBf16Incremental per-token
  // compute with the per-sequence host KimiDecodeCache replaced by the paged
  // groups — which is what makes the CLI-incremental battery the fold-identity
  // reference. Returns DEVICE-RESIDENT [rows,vocab] logits for the on-GPU
  // sampler (the third MUST-route seam).
  static ForwardLogits ForwardPaged(const ModelForwardInput& input,
                                    const KimiLinearWeights& weights,
                                    const vllm::TensorParallel* tp = nullptr);
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
