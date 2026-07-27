// Ported from: vllm/parser/inkling.py @ 555967922 (vLLM 0.26.0.dev0).
#include "vllm/parser/inkling.h"

namespace vllm::parser {

std::pair<std::optional<std::string>, std::optional<std::string>>
InklingParser::single_pass_parse(const std::string& text,
                                 std::optional<engine::ParserState> initial) {
  auto [reasoning, content] =
      engine::ParserEngine::single_pass_parse(text, initial);
  // inkling.py:386 — the engine defers content that follows tool-call events in
  // a single pass; Inkling allows text blocks after tool blocks, so flush it.
  if (!deferred_content_.empty()) {
    std::string trailing = deferred_content_;
    deferred_content_.clear();
    std::string combined = (content ? *content : std::string()) + trailing;
    content = strip_content_whitespace(combined, last_extracted_.tools_called);
    // Mirror the rebuilt ExtractedToolCallInformation (content updated).
    last_extracted_.content = content;
  }
  return {reasoning, content};
}

}  // namespace vllm::parser
