// Ported from: vllm/reasoning/deepseek_v3_reasoning_parser.py @ 555967922
// (DeepSeekV3ReasoningParser + DeepSeekV3ReasoningWithThinkingParser).
//
// DeepSeek V3.1 is a HYBRID-thinking model: the chat template decides at request
// build time whether the turn is a thinking turn (chat_template_kwargs.thinking /
// enable_thinking). The V3 reasoning parser therefore DELEGATES at construction:
//   thinking == True  -> DeepSeekR1ReasoningParser  (<think>…</think> split)
//   thinking == False -> IdentityReasoningParser    (passthrough, no split)
// and forwards every contract method to the chosen inner parser
// (deepseek_v3_reasoning_parser.py:34-80).
//
// DEVIATION (documented, same class as the dropped-tokenizer deviation on the
// seam): upstream reads `thinking` from the runtime `chat_template_kwargs`, but
// our name-only get_reasoning_parser() factory does not thread request-time
// kwargs. We mirror the DEFAULT construction: `deepseek_v3` -> thinking=False ->
// Identity (upstream default `thinking=False`); `holo2`
// (DeepSeekV3ReasoningWithThinkingParser) -> thinking=True -> R1 (upstream:44,
// which defaults thinking/enable_thinking to True when both are unset). Threading
// request-time chat_template_kwargs is a follow-on (W4 in specs/reasoning-parsers.md).
#pragma once

#include <memory>
#include <optional>
#include <string>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/reasoning_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class DeepSeekV3ReasoningParser : public ReasoningParser {
 public:
  // thinking selects the inner parser (deepseek_v3_reasoning_parser.py:34).
  explicit DeepSeekV3ReasoningParser(bool thinking);

  ExtractedReasoning extract_reasoning(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  std::optional<DeltaMessage> extract_reasoning_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  bool is_reasoning_end(const std::string& text) const override;

 protected:
  // deepseek_v3_reasoning_parser.py:34 (`self._parser`).
  std::unique_ptr<ReasoningParser> parser_;
};

// deepseek_v3_reasoning_parser.py:83 — the `holo2`-registered variant that
// defaults to thinking mode (thinking=True when the template leaves it unset).
class DeepSeekV3ReasoningWithThinkingParser final
    : public DeepSeekV3ReasoningParser {
 public:
  DeepSeekV3ReasoningWithThinkingParser() : DeepSeekV3ReasoningParser(true) {}
};

}  // namespace vllm::entrypoints::openai
