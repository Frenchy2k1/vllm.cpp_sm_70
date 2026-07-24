// Ported from: vllm/config/kv_transfer.py:92-121 @ e24d1b24
#include "vllm/config/kv_transfer.h"

#include <stdexcept>

#include <nlohmann/json.hpp>

namespace vllm {

std::optional<KVRole> parse_kv_role(const std::string& s) {
  if (s == "kv_producer") return KVRole::kProducer;
  if (s == "kv_consumer") return KVRole::kConsumer;
  if (s == "kv_both") return KVRole::kBoth;
  return std::nullopt;
}

const char* kv_role_str(KVRole role) {
  switch (role) {
    case KVRole::kProducer:
      return "kv_producer";
    case KVRole::kConsumer:
      return "kv_consumer";
    case KVRole::kBoth:
      return "kv_both";
  }
  return "";
}

std::optional<KVLoadFailurePolicy> parse_kv_load_failure_policy(
    const std::string& s) {
  if (s == "recompute") return KVLoadFailurePolicy::kRecompute;
  if (s == "fail") return KVLoadFailurePolicy::kFail;
  return std::nullopt;
}

const char* kv_load_failure_policy_str(KVLoadFailurePolicy policy) {
  switch (policy) {
    case KVLoadFailurePolicy::kRecompute:
      return "recompute";
    case KVLoadFailurePolicy::kFail:
      return "fail";
  }
  return "";
}

// kv_transfer.py:92-106. kv_role is REQUIRED whenever kv_connector is set; a
// missing engine_id is filled (upstream uses uuid4 — we use a deterministic
// placeholder so an unconfigured id is still valid and reproducible).
void KVTransferConfig::Validate() {
  if (!engine_id.has_value() || engine_id->empty()) {
    engine_id = "vllm-cpp-engine";
  }
  if (kv_connector.has_value() && !kv_role.has_value()) {
    throw std::invalid_argument(
        "Please specify kv_role when kv_connector is set (supported roles: "
        "kv_producer, kv_consumer, kv_both).");
  }
}

// See the header. Mirrors vLLM's --kv-transfer-config JSON payload 1:1.
KVTransferConfig ParseKVTransferConfigJson(const std::string& json_text) {
  nlohmann::json doc;
  try {
    doc = nlohmann::json::parse(json_text);
  } catch (const nlohmann::json::exception& e) {
    throw std::invalid_argument(std::string("kv-transfer-config is not valid "
                                            "JSON: ") +
                                e.what());
  }
  if (!doc.is_object()) {
    throw std::invalid_argument(
        "kv-transfer-config must be a JSON object, e.g. "
        "{\"kv_connector\": \"LMCacheConnector\", \"kv_role\": \"kv_both\"}");
  }

  KVTransferConfig cfg;
  for (auto it = doc.begin(); it != doc.end(); ++it) {
    const std::string& key = it.key();
    const nlohmann::json& value = it.value();
    if (key == "kv_connector" || key == "engine_id") {
      if (!value.is_string()) {
        throw std::invalid_argument("kv-transfer-config: \"" + key +
                                    "\" must be a string");
      }
      if (key == "kv_connector") {
        cfg.kv_connector = value.get<std::string>();
      } else {
        cfg.engine_id = value.get<std::string>();
      }
    } else if (key == "kv_role") {
      if (!value.is_string()) {
        throw std::invalid_argument(
            "kv-transfer-config: \"kv_role\" must be a string");
      }
      const std::string role = value.get<std::string>();
      std::optional<KVRole> parsed = parse_kv_role(role);
      if (!parsed.has_value()) {
        throw std::invalid_argument(
            "kv-transfer-config: unknown \"kv_role\" \"" + role +
            "\" (supported: kv_producer, kv_consumer, kv_both)");
      }
      cfg.kv_role = *parsed;
    } else if (key == "kv_load_failure_policy") {
      if (!value.is_string()) {
        throw std::invalid_argument(
            "kv-transfer-config: \"kv_load_failure_policy\" must be a string");
      }
      const std::string policy = value.get<std::string>();
      std::optional<KVLoadFailurePolicy> parsed =
          parse_kv_load_failure_policy(policy);
      if (!parsed.has_value()) {
        throw std::invalid_argument(
            "kv-transfer-config: unknown \"kv_load_failure_policy\" \"" +
            policy + "\" (supported: fail, recompute)");
      }
      cfg.kv_load_failure_policy = *parsed;
    } else if (key == "kv_connector_extra_config") {
      if (!value.is_object()) {
        throw std::invalid_argument(
            "kv-transfer-config: \"kv_connector_extra_config\" must be a JSON "
            "object of scalar values");
      }
      for (auto e = value.begin(); e != value.end(); ++e) {
        const nlohmann::json& v = e.value();
        if (v.is_string()) {
          cfg.kv_connector_extra_config[e.key()] = v.get<std::string>();
        } else if (v.is_number() || v.is_boolean()) {
          // Stringified in JSON spelling: the connectors read these back with
          // extra_int / extra_bool, which accept "8" / "true".
          cfg.kv_connector_extra_config[e.key()] = v.dump();
        } else {
          throw std::invalid_argument(
              "kv-transfer-config: kv_connector_extra_config[\"" + e.key() +
              "\"] must be a string, number or boolean");
        }
      }
    } else {
      throw std::invalid_argument(
          "kv-transfer-config: unknown key \"" + key +
          "\" (supported: kv_connector, kv_role, engine_id, "
          "kv_load_failure_policy, kv_connector_extra_config)");
    }
  }
  cfg.Validate();
  return cfg;
}

}  // namespace vllm
