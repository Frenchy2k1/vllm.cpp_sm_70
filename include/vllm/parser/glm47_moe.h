// Ported from: vllm/parser/glm47_moe.py @ 555967922 (vLLM 0.26.0.dev0).
//
// Glm47MoeParser: the assembly-layer ParserEngine subclass for GLM-4.7's
// <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call> tool
// format. The only assembly divergence from the base engine is that the function
// name is .strip()'d (glm47_moe.py:198 _emit_name_delta, :203 _handle_tool_end)
// via the existing emit_name_delta / handle_tool_end virtual hooks. The grammar,
// reasoning plumbing, and regex arg-converter live in glm47_moe_config().
#pragma once

#include <string>
#include <vector>

#include "vllm/parser/engine/configs.h"
#include "vllm/parser/engine/parser_engine.h"

namespace vllm::parser {

class Glm47MoeParser : public engine::ParserEngine {
 public:
  explicit Glm47MoeParser(bool thinking = true,
                          const engine::EngineTokenizer* tokenizer = nullptr)
      : engine::ParserEngine(engine::glm47_moe_config(thinking), tokenizer) {}

 protected:
  // glm47_moe.py:198 _emit_name_delta — name.strip() before the base emit.
  void emit_name_delta(int idx,
                       std::vector<engine::oai::DeltaToolCall>& deltas,
                       const std::optional<std::string>& name) override;
  // glm47_moe.py:203 _handle_tool_end — strip the slot name before the base end.
  void handle_tool_end(const engine::SemanticEvent& event,
                       std::vector<engine::oai::DeltaToolCall>& deltas) override;
};

}  // namespace vllm::parser
