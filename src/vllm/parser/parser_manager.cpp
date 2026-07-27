// Ported from: vllm/parser/parser_manager.py + registered_adapters.py @
// 555967922 (vLLM 0.26.0.dev0).
#include "vllm/parser/parser_manager.h"

#include "vllm/parser/engine/configs.h"
#include "vllm/parser/glm47_moe.h"
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
  // minimax_m2 / deepseek_v4 / deepseek_v32 / nemotron_v3: the arg_converter and
  // all assembly behavior live in the config, so a base ParserEngine over the
  // right config suffices (the deepseek subclasses' _convert_args wrapper-unwrap
  // degenerates to the config converter under the no-tool-schema model — same
  // residual as _fix_arg_types; see parser_engine.h).
  if (name == "minimax_m2") {
    return std::make_unique<engine::ParserEngine>(engine::minimax_m2_config(),
                                                  tokenizer);
  }
  if (name == "deepseek_v4") {
    return std::make_unique<engine::ParserEngine>(
        engine::deepseek_v4_config(thinking), tokenizer);
  }
  if (name == "deepseek_v32") {
    return std::make_unique<engine::ParserEngine>(engine::deepseek_v32_config(),
                                                  tokenizer);
  }
  if (name == "nemotron_v3") {
    return std::make_unique<engine::ParserEngine>(
        engine::nemotron_v3_config(thinking), tokenizer);
  }
  // glm47_moe strips the function name via the emit_name_delta / handle_tool_end
  // hooks (glm47_moe.py).
  if (name == "glm47_moe") {
    return std::make_unique<Glm47MoeParser>(thinking, tokenizer);
  }
  return nullptr;
}

}  // namespace vllm::parser
