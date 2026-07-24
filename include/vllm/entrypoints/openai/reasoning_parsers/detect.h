// ORIGINAL packaging-layer component (no upstream mirror): reasoning-parser
// auto-detection from the chat template, the sibling of the tool-parser
// marker table (tool_parsers/detect.h) and the same llama.cpp common/chat.cpp
// sniffing idea. Upstream vLLM selects reasoning parsers only by explicit
// --reasoning-parser name; the C-ABI consumer (LocalAI) needs zero-config.
#ifndef VLLM_ENTRYPOINTS_OPENAI_REASONING_PARSERS_DETECT_H_
#define VLLM_ENTRYPOINTS_OPENAI_REASONING_PARSERS_DETECT_H_

#include <cstddef>
#include <string>

namespace vllm::entrypoints::openai {

// One ordered detection row: a chat-template tell-tale substring and the
// reasoning parser it selects. First match wins (more specific first).
struct ReasoningParserMarker {
  const char* parser;
  const char* template_marker;
};

// Detect the reasoning parser from a chat template. Returns "" when no marker
// matches: unlike tool detection (which falls back to hermes because a hermes
// parser is a no-op on tool-free output), a reasoning parser actively splits
// text, so the safe no-detection default is DISABLED.
std::string DetectReasoningParser(const std::string& chat_template);

// Testable overload + built-in table accessor (mirrors tool_parsers/detect.h).
std::string DetectReasoningParser(const std::string& chat_template,
                                  const ReasoningParserMarker* table,
                                  std::size_t count);
const ReasoningParserMarker* ReasoningParserMarkerTable(std::size_t* out_count);

// Resolve the `--reasoning-parser` selection, the sibling of
// ResolveToolParserName (same "auto"/"none" vocabulary, same loud rejection):
//   "" / "none" -> "" (reasoning extraction DISABLED). The server's flag
//                  defaults to "none", which is exactly what it hardcoded
//                  before the flag existed, so an invocation without
//                  --reasoning-parser is unchanged.
//   "auto"      -> DetectReasoningParser(chat_template) (may itself return "",
//                  meaning no marker matched and reasoning stays off).
//   otherwise   -> that name, validated against reasoning_parser_names().
// Throws std::invalid_argument listing the registered names on an unknown one.
std::string ResolveReasoningParserName(const std::string& requested,
                                       const std::string& chat_template);

}  // namespace vllm::entrypoints::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_REASONING_PARSERS_DETECT_H_
