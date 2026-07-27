// Ported from: vllm/parser/inkling.py @ 555967922 (vLLM 0.26.0.dev0).
//
// InklingParser: the assembly-layer ParserEngine subclass for Inkling's typed
// content blocks (<|content_thinking|>/<|content_text|>/<|content_invoke_tool_json|>).
// Two assembly-core divergences from the base engine:
//   * _extract_args_value (inkling.py:402) unwraps the Inkling "args" wrapper key
//     in the non-streaming name-from-args path — modelled here as an override of
//     the args_wrapper_keys() hook that prepends "args" to the base key list.
//   * _single_pass_parse (inkling.py:376) flushes a trailing text block that
//     follows a tool-call block (the base engine defers it).
// The JSON-span arg carver lives in inkling_config() (inkling_arg_converter).
#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "vllm/parser/engine/configs.h"
#include "vllm/parser/engine/parser_engine.h"

namespace vllm::parser {

class InklingParser : public engine::ParserEngine {
 public:
  explicit InklingParser(const engine::EngineTokenizer* tokenizer = nullptr)
      : engine::ParserEngine(engine::inkling_config(), tokenizer) {}

 protected:
  // inkling.py:402 _extract_args_value — Inkling wraps arguments under "args".
  std::vector<std::string> args_wrapper_keys() const override {
    return {"args", "arguments", "parameters"};
  }

  // inkling.py:376 _single_pass_parse — flush trailing text after a tool block.
  std::pair<std::optional<std::string>, std::optional<std::string>>
  single_pass_parse(const std::string& text,
                    std::optional<engine::ParserState> initial) override;
};

}  // namespace vllm::parser
