// Ported from: vllm/reasoning/deepseek_v3_reasoning_parser.py @ 555967922
#include "vllm/entrypoints/openai/reasoning_parsers/deepseek_v3.h"

#include "vllm/entrypoints/openai/reasoning_parsers/deepseek_r1.h"
#include "vllm/entrypoints/openai/reasoning_parsers/identity.h"

namespace vllm::entrypoints::openai {

DeepSeekV3ReasoningParser::DeepSeekV3ReasoningParser(bool thinking) {
  // deepseek_v3_reasoning_parser.py:34-38.
  if (thinking) {
    parser_ = std::make_unique<DeepSeekR1ReasoningParser>();
  } else {
    parser_ = std::make_unique<IdentityReasoningParser>();
  }
}

ExtractedReasoning DeepSeekV3ReasoningParser::extract_reasoning(
    const std::string& model_output, const ChatCompletionRequest& request) {
  // deepseek_v3_reasoning_parser.py:59.
  return parser_->extract_reasoning(model_output, request);
}

std::optional<DeltaMessage>
DeepSeekV3ReasoningParser::extract_reasoning_streaming(
    const std::string& previous_text, const std::string& current_text,
    const std::string& delta_text, const ChatCompletionRequest& request) {
  // deepseek_v3_reasoning_parser.py:64.
  return parser_->extract_reasoning_streaming(previous_text, current_text,
                                              delta_text, request);
}

bool DeepSeekV3ReasoningParser::is_reasoning_end(const std::string& text) const {
  // deepseek_v3_reasoning_parser.py:48.
  return parser_->is_reasoning_end(text);
}

}  // namespace vllm::entrypoints::openai
