// Ported from: vllm/tool_parsers/muse_glimmer_tool_parser.py @ 075d645af
// (vLLM PR #51655 head — deliberately NOT the parity pin; see
// .agents/specs/muse-glimmer.md §0 and .agents/porting-inventory.md §9
// deviation 16.)
//
// Muse Glimmer emits tool calls as XML-ish ATEM markup inside channel-scoped
// messages:
//
//   <|start|>assistant to=<tool>.<fn><|message|>
//     <atem:function_calls>
//     <atem:invoke name="tool.fn">
//     <atem:parameter name="arg">value</atem:parameter>
//     </atem:invoke>
//     </atem:function_calls><|eom|>
//
// Channel scoping is the whole game: an `<atem:invoke>` echoed inside a
// `to=self` reasoning block or a `to=user` final answer must NOT become a tool
// call. Upstream SEGMENTS the output into messages and SELECTS the tool-channel
// bodies rather than SUBTRACTING reasoning spans with regexes, because on a
// truncated turn subtraction can delete a valid tool call. This port keeps that
// structure verbatim.
//
// DEVIATIONS from the upstream file:
//
//  1. `adjust_request` (upstream:206) and `supports_required_and_named = False`
//     (upstream:192) have no analogue: the C++ ToolParser seam carries no
//     request-mutation hook, no `skip_special_tokens`, and no
//     named/required-tool_choice fast path that would need opting out of.
//  2. Upstream logs four warnings (unframed ATEM, unknown tool name, truncated
//     tool call, and the two `logger.exception` paths). This seam has no logger;
//     the conditions are preserved exactly, the logging is not.
//  3. `_normalize_name` binds a tool name by its trailing segment when — and
//     only when — that is unambiguous. See the comment on `normalize_name`; this
//     is the one place where upstream's CODE and upstream's own TEST disagree.
//  4. STREAMING GAINS THE UNFRAMED-CONTENT FALLBACK that upstream only has in
//     the non-streaming `_extract_content` (upstream:366-374). Upstream's
//     `DelegatingParser.parse_delta` hands the tool parser the RAW framed text;
//     our seam's `ShapeChatDelta` runs the reasoning parser first, so the text
//     reaching the tool parser has already had its framing classified away.
//     Without the fallback a `to=user` answer would be silently dropped — the
//     exact failure upstream's own docstring warns about (upstream:432-441).
#pragma once

#include <optional>
#include <set>
#include <string>
#include <vector>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class MuseGlimmerToolParser final : public ToolParser {
 public:
  MuseGlimmerToolParser() = default;

  // muse_glimmer_tool_parser.py:378 (extract_tool_calls).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // muse_glimmer_tool_parser.py:422 (extract_tool_calls_streaming).
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  // muse_glimmer_tool_parser.py:313 (_normalize_name). Map an emitted ATEM
  // invoke name back onto a name the client actually registered.
  //
  //  - exact match, or no registered tools               -> unchanged
  //  - `x.x` where `x` is registered (the shipped chat
  //    template renders a bare `get_weather` as the valid
  //    recipient `get_weather.*`, so the model emits
  //    `get_weather.get_weather`)                         -> `x`
  //  - the emitted TRAILING SEGMENT matches exactly one
  //    registered tool AND that tool is itself bare       -> that tool
  //  - anything else                                      -> unchanged
  //
  // The third rule is where upstream's code and upstream's test disagree.
  // `_normalize_name` implements only the first two and its docstring argues
  // that trailing-segment matching is unsafe: an emitted `weather.get` against a
  // registered `{calendar.get}` has a unique leaf match and would dispatch the
  // WRONG tool. But
  // tests/tool_use/test_muse_glimmer_toolname_normalize.py::
  // test_trailing_segment_unambiguous asserts that an emitted `foo.get_weather`
  // against a registered bare `get_weather` DOES bind. Restricting the leaf rule
  // to registered BARE names satisfies every one of the six ported assertions
  // AND upstream's stated safety invariant: `weather.get` vs `{calendar.get}`
  // is left alone, because `calendar.get` is not bare. Reported upstream-facing
  // in the W7 findings.
  static std::string normalize_name(const std::string& emitted,
                                    const std::set<std::string>& registered);

  // muse_glimmer_tool_parser.py:241 (_tool_channel_text). Bodies of the messages
  // addressed to a TOOL, joined with "\n"; falls back to the whole text when the
  // framing never arrived at all.
  static std::string tool_channel_text(const std::string& text);

 private:
  // Streaming cursors (upstream:202-204). One parser instance per request, so
  // instance state is per-stream.
  std::size_t streamed_content_len_ = 0;
  std::size_t streamed_reasoning_len_ = 0;
  std::size_t emitted_tool_calls_ = 0;
};

}  // namespace vllm::entrypoints::openai
