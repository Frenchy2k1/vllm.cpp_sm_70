// Muse Glimmer (`MuseGlimmerForConditionalGeneration`) — the ADDITIVE model TU
// skeleton for the Muse Glimmer structural bring-up (`CLAIM-MUSE-GLIMMER-W0`).
// This header defines the config parse (flat/nested normalization + the iRoPE
// layer mask + the dual query-pre-scale schema), the structural weight name map
// for both checkpoint conventions, and the forward / KV-cache seams. The forward
// REFUSES-by-name (`VT_CHECK(false, ...)`) exactly like `kimi_k3.{h,cpp}` and
// `deepseek_v4.{h,cpp}` — the TU BUILDS and the config/name-map structure is
// unit-testable, but a forward LOUDLY reports the pending brick rather than
// returning a silent wrong answer.
//
// ─── OFF-PIN HONESTY (up front) ──────────────────────────────────────────────
// Meta released Muse Glimmer on 2026-08-08, AFTER the parity pin `555967922`
// (2026-07-26). There is no `muse_glimmer` code at the pin, and none on vLLM
// `main` either: the ONLY upstream implementation is the still-OPEN, approved-
// but-CI-red PR vllm#51655 at head `075d645af`. Every `file:line` below points at
// that BRANCH HEAD, not at the pin — a deliberate exception on explicit developer
// direction, recorded as porting-inventory §9 deviation 16.
//
// Consequently the pinned oracle CANNOT load this model, there is no honest
// throughput denominator, and **no speed axis is claimable** for Muse Glimmer
// until #51655 merges and the pin advances. Correctness gates against the HF
// reference instead. A green CPU build is STRUCTURE, not execution evidence.
//
// ─── WHAT THIS IS A PORT OF (file:line, @ vllm#51655 head 075d645af) ──────────
//   OURS                              <-  UPSTREAM
//   MuseGlimmerTextParams             <-  transformers_utils/configs/muse_glimmer.py:29-126
//                                         (MuseGlimmerTextConfig)
//   MuseGlimmerVisionParams           <-  configs/muse_glimmer.py:129-176
//                                         (MuseGlimmerVisionConfig)
//   ParseMuseGlimmerParams            <-  configs/muse_glimmer.py:186-305 (the flat ->
//                                         canonical normalization) + :20-26
//                                         (_default_no_rope_layers)
//   ResolveMuseGlimmerQueryPreScale   <-  models/muse_glimmer.py:472-517
//   MuseGlimmerUseQkNorm/OutputGate   <-  models/muse_glimmer.py:456-469
//   EnumerateMuseGlimmerTensors       <-  models/muse_glimmer.py:1271-1300 (text),
//                                         :692-739 + :651-668 (vision tower),
//                                         :1036-1044 (adapter), :1440-1486 (wrapper)
//   NormalizeMuseGlimmerWeightName    <-  models/muse_glimmer.py:1389-1425
//                                         (hf_to_vllm_mapper)
//   MuseGlimmerModel::Forward         <-  models/muse_glimmer.py:1304-1345,
//                                         :1604-1613 — REFUSE-by-name
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

// Which checkpoint naming convention a weight name came from. The discriminator
// is the PREFIX, and it is load-bearing: the two families give the sandwich norms
// DIFFERENT MEANINGS (see `NormalizeMuseGlimmerWeightName`).
enum class MuseGlimmerCheckpointConvention {
  kCanonical,  // `model.language_model.layers.N.*` — norms already correct
  kLegacyGuac,  // `model.layers.N.*` — `post_attention_layernorm` is really the
                // PRE-feedforward norm; `post_attn_norm` is the true post-attn one
};

// The Muse Glimmer text tower. Values in comments are the released
// `meta-models/Muse-Glimmer-30B` scale.
struct MuseGlimmerTextParams {
  int64_t vocab_size = 0;             // 202048
  int64_t hidden_size = 0;            // 6656
  int64_t intermediate_size = 0;
  int64_t num_hidden_layers = 0;      // 52
  int64_t num_attention_heads = 0;    // 32
  int64_t num_key_value_heads = 0;    // 2  (GQA 16:1)
  int64_t head_dim = 0;               // 128
  int64_t max_position_embeddings = 0;  // 131072
  int64_t sliding_window = 0;
  double rope_theta = 500000.0;
  bool tie_word_embeddings = false;

  // Sandwich norms: pre-norms use `rms_norm_eps`, post-norms use the separate,
  // typically SMALLER `post_norm_eps` (muse_glimmer.py:1236-1247). Both norms
  // carry a baked `+1` weight offset and are computed in fp32.
  float rms_norm_eps = 1e-6f;
  float post_norm_eps = 1e-6f;

  // iRoPE mask, one entry per layer (configs/muse_glimmer.py:20-26, 108-112):
  //   0 => NoPE   AND full attention
  //   1 => RoPE   AND sliding-window attention
  // Default is NoPE every 4th layer counted BACKWARD from the last layer.
  std::vector<int64_t> no_rope_layers;

  // Post-QK-norm query pre-scale, already normalized across the two config
  // schemas (see `ResolveMuseGlimmerQueryPreScale`). The softmax scaling stays
  // `head_dim ** -0.5` and is NOT folded in here.
  double scale_query_by = 1.0;

  // Both default to TRUE. The modular schema OMITS them, so they read as null
  // and only an explicit `false` disables them (muse_glimmer.py:456-469).
  bool use_qk_norm = true;
  bool use_attn_output_gate = true;

  bool normalize_tok_embeddings = false;
  double output_multiplier = 1.0;
  double final_logit_softcapping = 0.0;  // 0 == none
  std::string hidden_activation = "silu";
};

// The perception encoder. Parsed in full so the arch RESOLVES and the tower's
// tensor names are enumerable; the tower forward is a W3 residual.
struct MuseGlimmerVisionParams {
  bool present = false;
  int64_t patch_size = 14;
  int64_t pos_emb_height = 32;
  int64_t pos_emb_width = 32;
  int64_t num_attention_heads = 16;
  int64_t num_hidden_layers = 50;
  int64_t hidden_size = 1536;
  int64_t intermediate_size = 8960;
  int64_t merge_kernel_size = 2;
  int64_t output_dim = 6144;
  int64_t patch_temporal = 2;
  int64_t adapter_dim = 4096;
  float layer_norm_eps = 1e-5f;
  // "full_attention" / "sliding_attention" per layer. Upstream's vision default
  // is full every 4th layer AND on the last layer (configs/muse_glimmer.py:168-176)
  // — note that is a DIFFERENT rule from the text tower's backward-counted mask.
  std::vector<std::string> layer_types;
};

struct MuseGlimmerParams {
  MuseGlimmerTextParams text;
  MuseGlimmerVisionParams vision;
  int64_t image_token_id = 200092;
  int64_t video_token_id = 200091;
};

// Resolve + validate MuseGlimmerParams from an HfConfig. Pure/host —
// unit-testable without a checkpoint. Accepts BOTH the canonical nested layout
// (`text_config` / `vision_config`) and the older FLAT layout, normalizing the
// latter (configs/muse_glimmer.py:186-305). That normalization is not cosmetic:
// without it a flat config deserializes to an ALL-DEFAULT text config, silently
// ignoring every checkpoint value and building a wrong-shaped model with no
// error. Throws with a precise message on a missing required field.
MuseGlimmerParams ParseMuseGlimmerParams(const HfConfig& config);

// Per-family config hook (registry `parse_config`).
void ParseMuseGlimmerConfig(const HfConfig& config);

// The post-QK-norm query pre-scale, normalized across the two config schemas
// (muse_glimmer.py:472-517). HF native modeling ships the RAW `qk_scale_factor`
// (~43.784 at head_dim 128) and folds `1/sqrt(head_dim)` itself; the modular
// `text_config` ships the value ALREADY folded (~3.87). Both must yield the same
// ~3.87. Upstream disambiguates by MAGNITUDE against `sqrt(head_dim)`, and an
// explicit `scale_query_by` wins outright. Getting this wrong scales every query
// by ~11.3x, so it is exercised RED-first from both schemas.
double ResolveMuseGlimmerQueryPreScale(double qk_scale_factor,
                                       bool has_qk_scale_factor,
                                       double explicit_scale_query_by,
                                       bool has_explicit_scale, int64_t head_dim);

// The default iRoPE mask when the checkpoint omits `no_rope_layers`:
// NoPE every 4th layer counted BACKWARD from the last (configs/muse_glimmer.py:20-26).
std::vector<int64_t> DefaultMuseGlimmerNoRopeLayers(int64_t num_hidden_layers);

// Normalize ONE checkpoint weight name to our canonical internal name, mirroring
// upstream's `hf_to_vllm_mapper` (muse_glimmer.py:1389-1425). Returns false when
// the name is dropped outright (upstream maps `model.rotary_emb.` to None).
//
// TWO ordering hazards are load-bearing here and are covered by tests:
//   1. `.self_attn.gate_proj` -> `.self_attn.output_gate_proj` MUST be applied
//      before any `.gate_proj` -> `.gate_up_proj` MLP stacking rule. The
//      attention OUTPUT GATE and the MLP gate share a suffix; folding the former
//      into `gate_up_proj` would silently corrupt both.
//   2. For a kLegacyGuac checkpoint the sandwich norms must be renamed in the
//      right order: legacy `post_attention_layernorm` is really the
//      PRE-feedforward norm and must be renamed FIRST, before legacy
//      `post_attn_norm` becomes `post_attention_layernorm` — otherwise the
//      second rule's output is re-captured by the first and the two norms swap.
bool NormalizeMuseGlimmerWeightName(const std::string& name, std::string* out);

// Which convention a raw checkpoint name belongs to, by PREFIX
// (muse_glimmer.py:1379-1381).
MuseGlimmerCheckpointConvention MuseGlimmerConventionOf(const std::string& name);

// The structural weight name map, in our canonical post-normalization names.
// Grounded 1:1 in the module tree at vllm#51655 head `075d645af`. Weightless
// modules (`embed_norm`, the per-head `qk_norm`, `perception_emb_norm`) contribute
// NO tensor, which is why the counts are smaller than a Gemma-shaped tower's.
std::vector<std::string> EnumerateMuseGlimmerTensors(const MuseGlimmerParams& params);

// Whole Muse Glimmer weights. STRUCTURAL SCAFFOLDING for W0: carries the resolved
// params + the loader's accounting result. Tensor MATERIALIZATION is the named W1+
// residual.
struct MuseGlimmerWeights {
  MuseGlimmerParams params{};
  int64_t accounted_tensors = 0;
  int64_t enumerated_tensors = 0;
};

// Load `MuseGlimmerForConditionalGeneration` safetensors. W0 performs the
// STRUCTURAL accounting pass only; it does not materialize tensors.
MuseGlimmerWeights LoadMuseGlimmerForConditionalGenerationWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

// The Muse Glimmer forward. REFUSE-by-name (both entrypoints VT_CHECK(false)):
// the sandwich-norm + iRoPE + gated-attention text tower (W1) and the perception
// encoder (W3) are not wired here. A forward LOUDLY reports the pending brick.
// See `.agents/specs/muse-glimmer.md` §3.
class MuseGlimmerModel {
 public:
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const MuseGlimmerWeights& weights,
      vt::Queue& queue, const std::vector<int32_t>& logits_indices = {});

  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const MuseGlimmerWeights& weights,
      vt::Queue& queue, const std::vector<int32_t>& logits_indices = {});
};

// KV-cache spec builder. W0 PLACEHOLDER: the TRUE topology is heterogeneous —
// the RoPE layers are sliding-window and the NoPE layers are full attention
// (muse_glimmer.py:1167-1168) — which the per-layer spec seam Gemma-4 landed can
// represent. W0 emits ONE full-attention group sized to the uniform GQA geometry
// so the arch RESOLVES; the heterogeneous split lands with the W1 forward. Never
// exercised — the forward REFUSES.
v1::KVCacheConfig MakeMuseGlimmerKVCache(const HfConfig& config, int block_size,
                                         int num_blocks);

}  // namespace vllm
