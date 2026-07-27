// Ported from: vllm/parser/parser_manager.py + registered_adapters.py @
// 555967922 (vLLM 0.26.0.dev0).
#include "vllm/parser/parser_manager.h"

#include "vllm/parser/engine/configs.h"
#include "vllm/parser/kimi_k2.h"

namespace vllm::parser {

std::unique_ptr<engine::ParserEngine> get_parser_engine(
    const std::string& name, bool thinking,
    const engine::EngineTokenizer* tokenizer) {
  // qwen3 + seed_oss share the qwen3 grammar (the arg_converter lives in the
  // config); no per-family method overrides are needed for the assembly, so a
  // base ParserEngine over the right config suffices (qwen3.py / seed_oss.py).
  if (name == "qwen3") {
    return std::make_unique<engine::ParserEngine>(
        engine::qwen3_config(thinking, "qwen3"), tokenizer);
  }
  if (name == "seed_oss") {
    return std::make_unique<engine::ParserEngine>(
        engine::qwen3_config(thinking, "seed_oss", "<seed:think>",
                             "</seed:think>", "<seed:tool_call>",
                             "</seed:tool_call>"),
        tokenizer);
  }
  // kimi_k2 needs the native-header name/id parsing overrides (kimi_k2.py).
  if (name == "kimi_k2") {
    return std::make_unique<KimiK2Parser>(thinking, tokenizer);
  }
  return nullptr;
}

}  // namespace vllm::parser
