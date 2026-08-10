// SPEC-MTP I5d: `--speculative-config` JSON parsing. Mirrors the CLI subset of
// vllm/engine/arg_utils.py + vllm/config/speculative.py that the entrypoint
// needs; the full method auto-detection / draft-model resolution stays in the
// loader (see SpeculativeConfig::ResolveMtp).
#include "vllm/config/speculative.h"

#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace vllm {

SpeculativeConfig ParseSpeculativeConfigJson(const std::string& json_text) {
  nlohmann::json doc;
  try {
    doc = nlohmann::json::parse(json_text);
  } catch (const nlohmann::json::exception& e) {
    throw std::invalid_argument(std::string("speculative-config: invalid JSON: ") +
                                e.what());
  }
  if (!doc.is_object()) {
    throw std::invalid_argument("speculative-config: expected a JSON object");
  }

  SpeculativeConfig cfg;
  // method (vllm/config/speculative.py `method`). Required for this CLI path;
  // upstream can auto-detect it from the draft checkpoint, but the entrypoint
  // that reaches here always passes an explicit method.
  if (!doc.contains("method") || !doc.at("method").is_string()) {
    throw std::invalid_argument(
        "speculative-config: a string \"method\" is required");
  }
  cfg.method = doc.at("method").get<std::string>();
  // SPEC-DFLASH D4: accept "dflash" alongside "mtp". SPEC-NGRAM (ROAD-V1-D3):
  // accept "ngram" — the draft-free proposer. SPEC-DRAFT-MODEL: accept
  // "draft_model" — the classic model-agnostic SEPARATE draft model
  // (speculative.py:684 default; uses_draft_model :1195; runner
  // gpu_model_runner.py:604-609). "mtp"/"dflash" are draft-hidden-state methods;
  // the loader resolves the concrete draft (MTP head vs the z-lab DFlash
  // checkpoint) and the block-derived k from the model config. "ngram" needs no
  // draft model; "draft_model" needs a separate `model` checkpoint (below).
  // Any other method is still rejected at this pin.
  // SPEC-DSPARK W1: accept "dspark" — the semi-autoregressive BLOCK drafter that
  // extends DFlash with a Markov logit-bias head (speculative.py:62,310;
  // dspark/speculator.py:37). Like DFlash it names a SEPARATE draft checkpoint,
  // so it requires the `model` key below.
  if (cfg.method != "mtp" && cfg.method != "dflash" && cfg.method != "ngram" &&
      cfg.method != "draft_model" && cfg.method != "dspark") {
    throw std::invalid_argument(
        "speculative-config: only methods \"mtp\", \"dflash\", \"dspark\", "
        "\"ngram\" and \"draft_model\" are supported at this pin (got \"" +
        cfg.method + "\")");
  }

  // num_speculative_tokens (k). Optional; the loader defaults it to n_predict
  // (mtp_num_hidden_layers) via ResolveMtp when absent (speculative.py:865-875).
  if (doc.contains("num_speculative_tokens") &&
      !doc.at("num_speculative_tokens").is_null()) {
    const nlohmann::json& k = doc.at("num_speculative_tokens");
    if (!k.is_number_integer() || k.get<int>() <= 0) {
      throw std::invalid_argument(
          "speculative-config: num_speculative_tokens must be a positive integer");
    }
    cfg.num_speculative_tokens = k.get<int>();
  }
  // SPEC-NGRAM (ROAD-V1-D3): the n-gram proposer window (speculative.py:157-161).
  // Optional; ResolveNgram defaults both to 5 when absent. Ignored for mtp/dflash.
  auto parse_lookup = [&](const char* key) -> std::optional<int> {
    if (doc.contains(key) && !doc.at(key).is_null()) {
      const nlohmann::json& v = doc.at(key);
      if (!v.is_number_integer() || v.get<int>() < 1) {
        throw std::invalid_argument(std::string("speculative-config: ") + key +
                                    " must be an integer >= 1");
      }
      return v.get<int>();
    }
    return std::nullopt;
  };
  cfg.prompt_lookup_min = parse_lookup("prompt_lookup_min");
  cfg.prompt_lookup_max = parse_lookup("prompt_lookup_max");
  if (cfg.method == "ngram" && !cfg.num_speculative_tokens.has_value()) {
    throw std::invalid_argument(
        "speculative-config: method \"ngram\" requires \"num_speculative_tokens\" "
        "(speculative.py:1224-1234)");
  }

  // SPEC-DFLASH D5: the DFlash draft is a SEPARATE checkpoint (unlike MTP's
  // in-target mtp.* tensors), so `--speculative-config` carries a `model` key
  // (vllm/config/speculative.py `model`) pointing at the z-lab draft. Required
  // for dflash; ignored (and not required) for mtp.
  if (doc.contains("model") && doc.at("model").is_string()) {
    cfg.draft_model_path = doc.at("model").get<std::string>();
  }
  if (cfg.method == "dflash" && !cfg.draft_model_path.has_value()) {
    throw std::invalid_argument(
        "speculative-config: method \"dflash\" requires a \"model\" key naming "
        "the DFlash draft checkpoint (path or HF repo id)");
  }
  // SPEC-DSPARK W1: the in-scope DSpark drafts (Qwen3 + Gemma4 families) are
  // SEPARATE checkpoints exactly like DFlash — e.g.
  // deepseek-ai/dspark_qwen3_4b_block7 (speculative.py:875). The DeepSeek-V4
  // variant that ships its draft inside the target checkpoint
  // (speculative.py:706-709) is out of scope for this row (hardware-blocked), so
  // the key is required here.
  if (cfg.method == "dspark" && !cfg.draft_model_path.has_value()) {
    throw std::invalid_argument(
        "speculative-config: method \"dspark\" requires a \"model\" key naming "
        "the DSpark draft checkpoint (path or HF repo id)");
  }
  // SPEC-DRAFT-MODEL: the generic draft model is a SEPARATE standalone
  // checkpoint, so it likewise requires the `model` key (speculative.py:692-701;
  // a separate `draft_model_config` is built from it). num_speculative_tokens (k)
  // is also required for a draft model — there is no head depth to default from
  // (speculative.py has no n_predict for "draft_model").
  if (cfg.method == "draft_model") {
    if (!cfg.draft_model_path.has_value()) {
      throw std::invalid_argument(
          "speculative-config: method \"draft_model\" requires a \"model\" key "
          "naming the separate draft checkpoint (path or HF repo id)");
    }
    if (!cfg.num_speculative_tokens.has_value()) {
      throw std::invalid_argument(
          "speculative-config: method \"draft_model\" requires "
          "\"num_speculative_tokens\"");
    }
  }
  // n_predict stays 0 here: the model loader resolves it from the checkpoint's
  // mtp_num_hidden_layers and re-runs ResolveMtp with this user k.
  return cfg;
}

}  // namespace vllm
