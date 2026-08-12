// DSpark draft model (`Qwen3DSparkModel` / `Qwen3DSparkForCausalLM`) for
// semi-autoregressive BLOCK speculative decoding. SPEC-DSPARK W2.
//
// Ported from vllm/model_executor/models/qwen3_dspark.py @ 555967922
// (vLLM 0.26.0.dev0): DSparkMarkovHead (:36-67), Qwen3DSparkModel (:70-92),
// Qwen3DSparkForCausalLM (:95-185) — specifically compute_draft_logits (:132),
// map_draft_to_target (:137), markov_embed (:143), markov_bias (:146).
//
// DSpark IS the DFlash draft plus one small head. Upstream says so structurally:
// `Qwen3DSparkModel(DFlashQwen3Model)` and
// `Qwen3DSparkForCausalLM(DFlashQwen3ForCausalLM)` — the parallel backbone (the
// 5-layer plain Qwen3 decoder, the fc aux-combine, the mask/noise embedding, the
// context-KV precompute and the non-causal in-block attention) is INHERITED
// UNCHANGED. So `Qwen3DSparkWeights` COMPOSES our landed `Qwen3DFlashWeights`
// rather than re-declaring it, and every backbone forward
// (Qwen3DFlashModel::ForwardBlockLogits*) is reused as-is on `.backbone`.
//
// What DSpark adds:
//
//   * the MARKOV HEAD — a low-rank transition bias. `markov_w1` embeds the
//     PREVIOUSLY SAMPLED token (TARGET vocab, [vocab_size, markov_rank]) and
//     `markov_w2` projects that r-dim embedding to a DRAFT-vocab logit bias
//     ([draft_vocab_size, markov_rank]). The speculator adds it to the backbone's
//     base logits at each sequential step, which is what re-introduces the
//     intra-block dependency a purely parallel block draft cannot have.
//     `markov_rank` is 256 in every shipped checkpoint.
//
//   * the REDUCED DRAFT VOCAB — when `draft_vocab_size < vocab_size` the
//     checkpoint ships a `d2t` table and a sampled draft id maps to the target
//     vocab as `draft_id + d2t[draft_id]` (upstream map_draft_to_target :137-141;
//     the table is stored as the OFFSET, not the absolute id). Both RedHatAI
//     gate-model drafts are draft_vocab_size 32000 against a 248320 target vocab;
//     the deepseek-ai Qwen3 drafts are full-vocab, where the map is the identity.
//
// NOT here: the sequential sampling loop itself (W4, spec_decode/dspark/), the
// checkpoint loader (W3), and the confidence head — upstream deliberately does
// not wire the confidence head into inference (qwen3_dspark.py:173, "not wired
// into inference yet; skip its weights"), so neither do we.
#pragma once

#include <cstdint>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/qwen3_dflash.h"  // Qwen3DFlashWeights (the backbone)
#include "vllm/transformers_utils/hf_config.h"
#include "vt/device.h"

namespace vllm {

// DSpark draft weights: the inherited DFlash backbone + the Markov head + the
// optional draft->target vocab map.
struct Qwen3DSparkWeights {
  // The inherited DFlashQwen3Model backbone (embed/fc/hidden_norm/final_norm/
  // lm_head/layers). Every Qwen3DFlashModel:: forward takes this directly.
  Qwen3DFlashWeights backbone;

  // markov_w1: VocabParallelEmbedding(vocab_size, markov_rank) — bf16
  // [vocab_size, markov_rank], gathered by the previously sampled TARGET id.
  OwnedTensor markov_w1;
  // markov_w2: ParallelLMHead(draft_vocab_size, markov_rank) — bf16 raw-NK
  // [draft_vocab_size, markov_rank], nk (applied through the logits processor,
  // i.e. a MatmulBT like lm_head).
  OwnedTensor markov_w2;

  // d2t (`draft_id_to_target_id`): the per-draft-id OFFSET such that
  // target_id = draft_id + d2t[draft_id]. EMPTY when the draft is full-vocab
  // (upstream sets the parameter to None and map_draft_to_target is the
  // identity, qwen3_dspark.py:118-127,137-141).
  std::vector<int32_t> draft_id_to_target_id;

  // W8 follow-up: markov_w1 rows GATHERED into DRAFT-vocab order, i.e.
  // markov_w1_draft[j] == markov_w1[j + d2t[j]]. Built once at load when a d2t
  // table exists; EMPTY when the draft is full-vocab (then markov_w1 already is
  // draft-indexed and is used directly).
  //
  // Why: the sequential loop needs the NEXT step's Markov embedding of the token
  // it just sampled. The argmax produces a DRAFT id on device, but markov_w1 is
  // indexed by TARGET vocab, so mapping it forced a device->host round trip EVERY
  // step (k syncs per draft step). Indexing this table with the raw argmax lets
  // `prev` stay on device for the whole chain; the draft->target mapping then
  // happens ONCE, on the [num_reqs, k] ids at the end.
  OwnedTensor markov_w1_draft;

  int64_t markov_rank = 0;
  int64_t vocab_size = 0;        // TARGET vocab == markov_w1 rows
  int64_t draft_vocab_size = 0;  // == backbone.draft_vocab_size == markov_w2 rows
  // LogitsProcessor scale (`logit_scale`, qwen3_dspark.py:110-117). 1.0 in every
  // shipped DSpark checkpoint; applied identically to the base logits and the
  // Markov bias because upstream routes BOTH through the same LogitsProcessor.
  float logit_scale = 1.0f;
};

// Load a DSpark draft checkpoint: the inherited DFlash backbone plus the Markov
// head plus the optional d2t map. Mirrors Qwen3DSparkForCausalLM.load_weights
// (qwen3_dspark.py:149-185).
//
// On-disk names, verified against BOTH published layouts (2026-08-09):
// deepseek-ai/dspark_qwen3_4b_block7 (64 tensors) and
// RedHatAI/Qwen3.6-35B-A3B-speculator.dspark (66 tensors) ship the SAME,
// unprefixed spelling — `layers.N.*`, `embed_tokens.weight`, `fc.weight`,
// `hidden_norm.weight`, `norm.weight`, `lm_head.weight`,
// `markov_head.markov_w{1,2}.weight`, and, for a reduced draft vocab, `d2t`
// (I64 [draft_vocab], the OFFSET table) + `t2d` (BOOL, training-only). Upstream
// prepends "model." to everything except lm_head and d2t (:157-159), and our
// backbone resolver already tries bare-then-"model."-prefixed, so both
// conventions load.
//
// SKIPPED exactly as upstream skips them (:171-176): `t2d` (training-only),
// `mask_embedding` (an unused placeholder — DSpark masks via the vocab row), and
// `confidence_head.*` ("not wired into inference yet"). `embed_tokens` / `lm_head`
// are OPTIONAL: when the checkpoint omits them the draft shares the target's
// (load_dspark_model :56-73), and the caller supplies them.
Qwen3DSparkWeights LoadQwen3DSpark(const TensorResolver& get, const HfConfig& config,
                                   int64_t num_taps, int32_t mask_token_id);
Qwen3DSparkWeights LoadQwen3DSpark(const std::vector<SafetensorsFile>& shards,
                                   const HfConfig& config, int64_t num_taps,
                                   int32_t mask_token_id);

class Qwen3DSparkModel {
 public:
  // markov_embed (qwen3_dspark.py:143-144 -> DSparkMarkovHead.embed :61-63):
  // gather `markov_w1[token_ids]`. `prev_token_ids` are TARGET-vocab ids (the
  // anchor token on step 0, then each step's sampled token). Returns [B, r] f32.
  static std::vector<float> MarkovEmbed(const std::vector<int32_t>& prev_token_ids,
                                        const Qwen3DSparkWeights& weights,
                                        vt::Queue& queue);

  // markov_bias (qwen3_dspark.py:146-147 -> DSparkMarkovHead.bias :65-67):
  // `logits_processor(markov_w2, markov_embed)` — a [B, r] x [draft_vocab, r]^T
  // MatmulBT scaled by logit_scale. Returns [B, draft_vocab] f32, the bias added
  // to the backbone's base draft logits at this sequential step.
  static std::vector<float> MarkovBias(const std::vector<float>& markov_embed,
                                       int64_t num_rows,
                                       const Qwen3DSparkWeights& weights,
                                       vt::Queue& queue);

  // The composition the speculator actually runs per step:
  // bias = markov_bias(markov_embed(prev)). Returns [B, draft_vocab] f32.
  static std::vector<float> MarkovBiasForTokens(const std::vector<int32_t>& prev_token_ids,
                                                const Qwen3DSparkWeights& weights,
                                                vt::Queue& queue);

  // map_draft_to_target (qwen3_dspark.py:137-141): identity for a full-vocab
  // draft, else `draft_id + d2t[draft_id]`. Throws on an out-of-range id, which
  // a correct sampler cannot produce (argmax over draft_vocab columns).
  static int32_t MapDraftToTarget(int32_t draft_id, const Qwen3DSparkWeights& weights);

  // W7 (#436): the sequential Markov loop with the per-step bias AND the argmax
  // kept ON DEVICE, instead of downloading [B, draft_vocab] f32 and scanning it
  // on the host once per step.
  //
  // MEASURED motivation: `[spec-phase] backbone=27.44ms sample=10.50ms` on the
  // 27B lane, i.e. 28% of the draft step spent on host-side sampling, because
  // the host path downloads k x [B, 248320] f32 of bias and runs k host argmaxes
  // over the full vocab. Upstream keeps the whole draft step on device under one
  // captured graph (dspark/speculator.py:22-24); this is spec risk R5.
  //
  // TOKEN-IDENTICAL to the host path by construction: `vt::GreedyArgmax` uses
  // the same LOWEST-INDEX tie-break as the host scan, step i reads the same base
  // row (r*nqpr + first_sample_offset + i), and `prev` is still the d2t-mapped
  // TARGET id. Only the k per-step [B, V] downloads and host scans disappear;
  // the [B] sampled ids still come back each step, because the sequential
  // dependency is resolved on the host — a few bytes instead of a megabyte.
  static std::vector<std::vector<int32_t>> SampleSequentialDevice(
      const std::vector<float>& block_logits, const std::vector<int32_t>& anchor_ids,
      int64_t num_query_per_req, int64_t first_sample_offset, int64_t num_spec,
      const Qwen3DSparkWeights& weights, vt::Queue& queue);

  // Whether SampleSequentialDevice can serve this checkpoint. Refuses a non-unit
  // `logit_scale`, because the host path scales the BIAS ONLY and reproducing
  // that asymmetry on device would need a scale op this composition avoids;
  // every shipped DSpark checkpoint has 1.0, and the caller falls back.
  static bool CanSampleOnDevice(const Qwen3DSparkWeights& weights);

  // ---- W3: checkpoint layouts ---------------------------------------------
  //
  // DSpark drafts ship in TWO config layouts with the SAME tensor layout
  // (verified against both published checkpoints, 2026-08-09):
  //   * NATIVE (deepseek-ai/dspark_qwen3_{4,8,14}b_block7): a flat Qwen3 config
  //     with block_size / markov_rank / mask_token_id / target_layer_ids.
  //   * SPECULATORS (RedHatAI/*.dspark, our gate-model drafts): the backbone
  //     fields nested under `transformer_layer_config`, the DSpark fields on the
  //     outer doc, plus `speculators_config` carrying k.
  // Tensors are identical either way: unprefixed `layers.N.*`, `embed_tokens`,
  // `fc`, `hidden_norm`, `norm`, `lm_head`, `markov_head.markov_w{1,2}`, and
  // (reduced-vocab only) `d2t` / `t2d`.

  // True when `doc` is a speculators-format DSpark config (base.py:52-56).
  static bool IsSpeculatorsDsparkConfig(const nlohmann::json& doc);

  // Speculators -> native config translation. Mirrors
  // extract_transformers_pre_trained_config (base.py:47-64): start from
  // `transformer_layer_config` (REQUIRED, :90-95) and apply update_dspark
  // (algos.py:133-178) — architectures, sample_from_anchor (default FALSE on
  // this path, unlike the speculator's own getattr default of True for native
  // configs), `eagle_aux_hidden_state_layer_ids`, the `i - 1` target_layer_ids,
  // and the copied DSpark keys. NOTE (faithful mirror, not our omission):
  // update_dspark does NOT consume `sliding_window_non_causal` — only the DFlash
  // updater does (algos.py:127-131) — so a DSpark draft's per-layer causality
  // comes from `layer_types` alone, exactly as upstream resolves it.
  static nlohmann::json TranslateSpeculatorsDsparkConfig(const nlohmann::json& doc);

  // k for a speculators checkpoint: proposal_methods[0].speculative_tokens
  // (base.py:113-136).
  static int SpeculatorsNumSpeculativeTokens(const nlohmann::json& doc);

  // Resolve `markov_rank` / `draft_vocab_size` / `vocab_size` / `logit_scale`
  // from the draft HF config, mirroring Qwen3DSparkForCausalLM.__init__
  // (:97-127): draft_vocab_size defaults to vocab_size when the key is absent
  // (:99-100), and `markov_rank` is REQUIRED (upstream reads config.markov_rank
  // unguarded at :86, so a checkpoint without it is not a DSpark draft).
  static void ResolveDsparkDims(const HfConfig& config, Qwen3DSparkWeights& weights);
};

}  // namespace vllm
