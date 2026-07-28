// Ported from: vllm/reasoning/identity_reasoning_parser.py @ 555967922
// (IdentityReasoningParser). The passthrough parser: it never separates
// reasoning — the ENTIRE model output is content, reasoning is always absent.
// It is the non-thinking delegate of DeepSeekV3ReasoningParser (and the base
// case for any "reasoning disabled by the template" path). Not registered under
// its own --reasoning-parser name upstream (used internally + in tests); we
// mirror that (no factory name) and expose the class for composition.
#pragma once

#include <optional>
#include <string>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/reasoning_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class IdentityReasoningParser final : public ReasoningParser {
 public:
  // identity_reasoning_parser.py:68 — returns (None, model_output).
  ExtractedReasoning extract_reasoning(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // identity_reasoning_parser.py:54 — wrap a non-empty delta as content, else
  // None (nothing to emit yet).
  std::optional<DeltaMessage> extract_reasoning_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  // identity_reasoning_parser.py:41 — reasoning is never "in progress", so the
  // structured-output gate is always open.
  bool is_reasoning_end(const std::string& text) const override;
};

}  // namespace vllm::entrypoints::openai
