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
  return std::nullopt;
}

bool is_engine_backed(const std::string& name) {
  return get_engine_config(name).has_value();
}

std::vector<std::string> engine_backed_names() {
  return {"qwen3", "seed_oss", "kimi_k2"};
}

}  // namespace vllm::parser::engine
