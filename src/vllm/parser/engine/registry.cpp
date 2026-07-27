// Ported from: vllm/parser/engine/registered_adapters.py @ 555967922.
#include "vllm/parser/engine/registry.h"

#include "vllm/parser/engine/configs.h"

namespace vllm::parser::engine {

std::optional<ParserEngineConfig> get_engine_config(const std::string& name) {
  // seed_oss shares the qwen3 grammar with different wrapper tokens
  // (seed_oss.py: SeedOssParser(Qwen3Parser)).
  if (name == "qwen3") return qwen3_config(true, "qwen3");
  if (name == "seed_oss") {
    return qwen3_config(true, "seed_oss", "<seed:think>", "</seed:think>",
                        "<seed:tool_call>", "</seed:tool_call>");
  }
  if (name == "kimi_k2") return kimi_k2_config(true);
  if (name == "minimax_m2") return minimax_m2_config();
  if (name == "glm47_moe") return glm47_moe_config(true);
  if (name == "deepseek_v4") return deepseek_v4_config(true);
  if (name == "deepseek_v32") return deepseek_v32_config();
  if (name == "nemotron_v3") return nemotron_v3_config(true);
  if (name == "gemma4") return gemma4_config();
  if (name == "inkling") return inkling_config();
  return std::nullopt;
}

bool is_engine_backed(const std::string& name) {
  return get_engine_config(name).has_value();
}

std::vector<std::string> engine_backed_names() {
  return {"qwen3",       "seed_oss",    "kimi_k2",      "minimax_m2",
          "glm47_moe",   "deepseek_v4", "deepseek_v32", "nemotron_v3",
          "gemma4",      "inkling"};
}

}  // namespace vllm::parser::engine
