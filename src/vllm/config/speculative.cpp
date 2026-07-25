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
  if (cfg.method != "mtp") {
    throw std::invalid_argument(
        "speculative-config: only method \"mtp\" is supported at this pin (got \"" +
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
  // n_predict stays 0 here: the model loader resolves it from the checkpoint's
  // mtp_num_hidden_layers and re-runs ResolveMtp with this user k.
  return cfg;
}

}  // namespace vllm
