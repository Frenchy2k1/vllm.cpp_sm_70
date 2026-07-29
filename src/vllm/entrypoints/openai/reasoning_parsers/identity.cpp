// Ported from: vllm/reasoning/identity_reasoning_parser.py @ 555967922
#include "vllm/entrypoints/openai/reasoning_parsers/identity.h"

namespace vllm::entrypoints::openai {

ExtractedReasoning IdentityReasoningParser::extract_reasoning(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  // identity_reasoning_parser.py:73 — (None, model_output). The whole output is
  // content; reasoning is absent (nullopt, distinct from an empty string).
  ExtractedReasoning out;
  out.reasoning = std::nullopt;
  out.content = model_output;
  return out;
}

std::optional<DeltaMessage> IdentityReasoningParser::extract_reasoning_streaming(
    const std::string& /*previous_text*/, const std::string& /*current_text*/,
    const std::string& delta_text, const ChatCompletionRequest& /*request*/) {
  // identity_reasoning_parser.py:64 — `if delta_text: return DeltaMessage(...)`.
  if (delta_text.empty()) {
    return std::nullopt;
  }
  DeltaMessage msg;
  msg.content = delta_text;
  return msg;
}

bool IdentityReasoningParser::is_reasoning_end(
    const std::string& /*text*/) const {
  // identity_reasoning_parser.py:41 — always True.
  return true;
}

}  // namespace vllm::entrypoints::openai
