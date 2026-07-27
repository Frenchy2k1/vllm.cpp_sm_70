// Ported from: vllm/config/speculative.py @ e24d1b24
//
// Scope (SPEC-MTP I2, scheduler-half): the T0 SUBSET of SpeculativeConfig the V1
// Scheduler actually reads — the method string, the resolved
// num_speculative_tokens, and the predicates that derive num_lookahead_tokens
// (scheduler.py:275-292). This is the config seam that turns the spec-decode
// scheduler plumbing ON: with no SpeculativeConfig (the production default), the
// scheduler keeps num_lookahead_tokens == 0 and every spec path is inert, so the
// scheduler/engine/input-batch behavior is byte-identical to the pre-spec engine.
//
// The full upstream resolution (draft ModelConfig, quantization, method
// auto-detection from the checkpoint's architectures, dynamic SD schedule,
// disable-by-batch-size, ROCm/attention-backend gating) is DEFERRED to the
// model-loader claims (MODEL-SPEC-qwen3-5-mtp-*); this header carries only what
// the scheduler consumes. The MTP method-resolution rule mirrored here is
// speculative.py:480-489 (model_type qwen3_5|qwen3_5_moe -> method "mtp",
// n_predict = mtp_num_hidden_layers) + speculative.py:865-875 (default
// num_speculative_tokens = n_predict when the user gives no explicit k).
#ifndef VLLM_CONFIG_SPECULATIVE_H_
#define VLLM_CONFIG_SPECULATIVE_H_

#include <optional>
#include <stdexcept>
#include <string>

namespace vllm {

// SpeculativeConfig (T0 scheduler subset). Value type; the Scheduler ctor reads
// it (upstream vllm_config.speculative_config).
struct SpeculativeConfig {
  // The speculative method (speculative.py `method`): "mtp", "eagle", "eagle3",
  // "dflash", "dspark", "draft_model", ... Empty string = unset (never
  // constructed on the default path).
  std::string method;

  // num_speculative_tokens (k). Upstream `int | None`; std::nullopt mirrors the
  // "not given -> default to n_predict" resolution done in ResolveMtp / the
  // upstream __post_init__ (speculative.py:865-875).
  std::optional<int> num_speculative_tokens = std::nullopt;

  // SPEC-DFLASH D5: the DFlash draft model path (the `model` key of
  // `--speculative-config`, e.g. "z-lab/Qwen3.6-27B-DFlash" or a local snapshot
  // dir). Unlike MTP (whose draft `mtp.*` tensors live inside the target
  // checkpoint), the DFlash draft is a SEPARATE checkpoint the loader opens; this
  // carries its path. Empty/unset for MTP and non-spec.
  std::optional<std::string> draft_model_path = std::nullopt;

  // n_predict: the draft head depth (MTP: mtp_num_hidden_layers, = 1 for both
  // gate checkpoints). Upstream reads it off the draft model's hf_config
  // (speculative.py:482-486,860-863). 0 when not an n_predict-style method.
  int n_predict = 0;

  // SPEC-NGRAM (ROAD-V1-D3): the n-gram proposer window (speculative.py:157-161
  // prompt_lookup_max/prompt_lookup_min). Only meaningful for method=="ngram";
  // the draft-free matcher looks for a suffix ngram of length in
  // [prompt_lookup_min, prompt_lookup_max]. Defaulted to 5/5 by ResolveNgram when
  // the user gives neither (speculative.py:737-742). std::nullopt for non-ngram.
  std::optional<int> prompt_lookup_min = std::nullopt;
  std::optional<int> prompt_lookup_max = std::nullopt;

  // ResolveMtp: build the scheduler-facing SpeculativeConfig for a Qwen3.5 MTP
  // checkpoint. Mirrors speculative.py:480-489 (method "mtp",
  // n_predict = mtp_num_hidden_layers) + :865-875 (default k = n_predict). A user
  // override for k must be a positive multiple of n_predict for MTP-module reuse
  // (speculative.py:869-875); we mirror the divisibility check.
  static SpeculativeConfig ResolveMtp(
      int mtp_num_hidden_layers,
      std::optional<int> user_num_speculative_tokens = std::nullopt) {
    SpeculativeConfig cfg;
    cfg.method = "mtp";
    cfg.n_predict = mtp_num_hidden_layers;
    if (user_num_speculative_tokens.has_value()) {
      const int k = *user_num_speculative_tokens;
      if (mtp_num_hidden_layers > 0 && k > mtp_num_hidden_layers &&
          k % mtp_num_hidden_layers != 0) {
        throw std::invalid_argument(
            "num_speculative_tokens must be divisible by n_predict "
            "(mtp_num_hidden_layers) for MTP module reuse");
      }
      cfg.num_speculative_tokens = k;
    } else {
      // Default to the head depth (speculative.py:867-868).
      cfg.num_speculative_tokens = mtp_num_hidden_layers;
    }
    return cfg;
  }

  // ResolveDflash: build the scheduler-facing SpeculativeConfig for a z-lab DFlash
  // draft (SPEC-DFLASH D4). Mirrors speculative.py method resolution for "dflash"
  // (:1172 use_dflash) + the num_speculative_tokens = block-derived k the loader
  // reads off the draft's dflash_config. Unlike MTP there is no n_predict-module
  // divisibility constraint (the block drafter is non-autoregressive); k is taken
  // as given (the CLI value, or the draft's block-1 default resolved by the loader).
  // n_predict stays 0 (not an n_predict-style method); the extra scheduler
  // lookahead slot comes from use_dflash()/NumLookaheadTokens() (already coded).
  static SpeculativeConfig ResolveDflash(int num_speculative_tokens_k) {
    SpeculativeConfig cfg;
    cfg.method = "dflash";
    cfg.n_predict = 0;
    cfg.num_speculative_tokens = num_speculative_tokens_k;
    return cfg;
  }

  // ResolveNgram: build the scheduler-facing SpeculativeConfig for the draft-free
  // n-gram proposer (SPEC-NGRAM, ROAD-V1-D3). Mirrors speculative.py:734-762 —
  // num_speculative_tokens (k) is REQUIRED (:1224-1234), and the prompt_lookup
  // window defaults to 5/5 when the user gives neither, or fills the missing bound
  // from the given one (:737-758). There is NO draft model and NO n_predict-module
  // constraint. NumLookaheadTokens() returns 0 for ngram (the scheduler reserves no
  // extra lookahead slots, scheduler.py:250-265 — none of the ngram branches fire),
  // exactly like upstream; the drafts are verified via the scheduled spec tokens.
  static SpeculativeConfig ResolveNgram(int k, std::optional<int> user_min,
                                        std::optional<int> user_max) {
    SpeculativeConfig cfg;
    cfg.method = "ngram";
    cfg.n_predict = 0;
    if (k <= 0) {
      throw std::invalid_argument(
          "speculative-config: method \"ngram\" requires a positive "
          "num_speculative_tokens");
    }
    cfg.num_speculative_tokens = k;
    // Default resolution (speculative.py:737-758).
    if (!user_min.has_value() && !user_max.has_value()) {
      cfg.prompt_lookup_min = 5;
      cfg.prompt_lookup_max = 5;
    } else if (!user_min.has_value()) {
      cfg.prompt_lookup_min = user_max;
      cfg.prompt_lookup_max = user_max;
    } else if (!user_max.has_value()) {
      cfg.prompt_lookup_min = user_min;
      cfg.prompt_lookup_max = user_min;
    } else {
      cfg.prompt_lookup_min = user_min;
      cfg.prompt_lookup_max = user_max;
    }
    if (*cfg.prompt_lookup_min < 1 || *cfg.prompt_lookup_max < 1) {
      throw std::invalid_argument(
          "speculative-config: prompt_lookup_min/max must be >= 1");
    }
    if (*cfg.prompt_lookup_min > *cfg.prompt_lookup_max) {
      throw std::invalid_argument(
          "speculative-config: prompt_lookup_min must be <= prompt_lookup_max");
    }
    return cfg;
  }

  // num_speculative_tokens resolved to a concrete k (falls back to n_predict).
  int ResolvedNumSpeculativeTokens() const {
    return num_speculative_tokens.value_or(n_predict);
  }

  // use_ngram (speculative.py method=="ngram"): the draft-FREE proposer. It is NOT
  // a target-hidden-state method, so it is deliberately absent from use_eagle()
  // and contributes 0 to NumLookaheadTokens() (scheduler.py:250-265).
  bool use_ngram() const { return method == "ngram"; }

  // use_eagle (speculative.py:1163-1170): true for the target-hidden-state
  // methods — eagle/eagle3/mtp/dflash/dspark. This is the predicate that sets the
  // scheduler's num_lookahead_tokens = k for MTP (scheduler.py:284-286).
  bool use_eagle() const {
    return method == "eagle" || method == "eagle3" || method == "mtp" ||
           method == "dflash" || method == "dspark";
  }

  // uses_draft_model (speculative.py:1195): true only for the "draft_model"
  // method (a separate small model). Also sets num_lookahead_tokens = k
  // (scheduler.py:287-288).
  bool uses_draft_model() const { return method == "draft_model"; }

  // use_dflash (speculative.py:1172): DFlash needs one EXTRA lookahead slot
  // (scheduler.py:289-292, num_lookahead = k + 1).
  bool use_dflash() const { return method == "dflash"; }

  // NumLookaheadTokens: the scheduler's num_lookahead_tokens for this config
  // (scheduler.py:275-292). This is the value threaded into allocate_slots so the
  // verify slots are reserved ahead of time. 0 for a method the scheduler does
  // not look ahead for.
  int NumLookaheadTokens() const {
    const int k = ResolvedNumSpeculativeTokens();
    if (use_dflash()) {
      return k + 1;  // extra slot for the in-fill last-sampled query.
    }
    if (use_eagle() || uses_draft_model()) {
      return k;
    }
    return 0;
  }
};

// Parse vLLM's `--speculative-config` JSON (SPEC-MTP I5d). Mirrors the subset of
// vllm/engine/arg_utils.py speculative-config handling the CLI needs: the
// `method` string and the optional `num_speculative_tokens`. "mtp" and "dflash"
// (SPEC-DFLASH D4) are supported at this pin; any other method throws. The
// returned config has n_predict == 0 — the loader resolves it from the model's
// mtp_num_hidden_layers via SpeculativeConfig::ResolveMtp (MTP) or the draft's
// dflash_config via ResolveDflash (DFlash) once the HF config is known. Throws
// std::invalid_argument on a malformed document / unknown method.
SpeculativeConfig ParseSpeculativeConfigJson(const std::string& json_text);

}  // namespace vllm

#endif  // VLLM_CONFIG_SPECULATIVE_H_
