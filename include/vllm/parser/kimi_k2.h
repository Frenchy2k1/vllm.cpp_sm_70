// Ported from: vllm/parser/kimi_k2.py @ 555967922 (vLLM 0.26.0.dev0).
//
// KimiK2Parser: the assembly-layer ParserEngine subclass for the Kimi K2 native
// tool-call format. The tool header before <|tool_call_argument_begin|> is
// Kimi's native id "functions.<name>:<idx>"; the function name is the final
// component before ":N". Overrides the name/arg/end hooks to parse that header
// and to stream the JSON argument body verbatim.
#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "vllm/parser/engine/configs.h"
#include "vllm/parser/engine/parser_engine.h"

namespace vllm::parser {

class KimiK2Parser : public engine::ParserEngine {
 public:
  explicit KimiK2Parser(bool thinking = true,
                        const engine::EngineTokenizer* tokenizer = nullptr)
      : engine::ParserEngine(engine::kimi_k2_config(thinking), tokenizer) {}

  // kimi_k2.py:178 _extract_tool_id_and_name.
  static std::pair<std::optional<std::string>, std::optional<std::string>>
  extract_tool_id_and_name(const std::optional<std::string>& header);

 protected:
  void emit_name_delta(int idx,
                       std::vector<engine::oai::DeltaToolCall>& deltas,
                       const std::optional<std::string>& name) override;
  void handle_tool_end(const engine::SemanticEvent& event,
                       std::vector<engine::oai::DeltaToolCall>& deltas) override;
  void handle_arg_chunk(const engine::SemanticEvent& event,
                        std::vector<engine::oai::DeltaToolCall>& deltas) override;
  std::string extract_args_json(const std::string& raw_args,
                                const std::string& func_name) override;
};

}  // namespace vllm::parser
