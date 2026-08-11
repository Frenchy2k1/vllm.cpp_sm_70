// Tests for the Muse Glimmer ATEM reasoning parser and its handoff to the tool
// parser. (vllm/reasoning/muse_glimmer_reasoning_parser.py @ 075d645af — vLLM
// PR #51655 head, NOT the parity pin. See .agents/specs/muse-glimmer.md §0.)
//
// PORTS, case for case:
//   - tests/tool_use/test_muse_glimmer_reasoning_handoff.py  (4 cases)
//   - tests/tool_use/test_muse_glimmer_streaming.py          (6 cases)
//   - tests/tool_use/test_muse_glimmer_parse_delta.py        (5 cases)
//
// HARNESS ADAPTATION (Python -> C++). Read this before changing an assertion.
//
//  A. Upstream constructs both parsers with `cls.__new__(cls)` to dodge an
//     `__init__` that wants a tokenizer. For the REASONING parser that is fatal:
//     `extract_reasoning_streaming` reads `self._emitted_reasoning`, which only
//     `__init__` sets, so every case in test_muse_glimmer_streaming.py raises
//     AttributeError before it can assert anything. This seam's ReasoningParser
//     has a default ctor and members with in-class initialisers, so the parsers
//     are simply constructed and the cases actually run.
//
//  B. THE STREAMING CASES DRIVE THE SERVING SEAM, NOT THE PARSERS IN ISOLATION.
//     Upstream's `_stream` helper feeds each parser the raw text separately and
//     then asserts that no framing token appears in what the REASONING parser
//     returned as `.content`. Against upstream's own implementation that
//     assertion cannot hold: the reasoning parser deliberately returns the whole
//     tool channel (framing included) as `.content` on the transition delta
//     (muse_glimmer_reasoning_parser.py:293-300) — that IS the handoff, and
//     `DelegatingParser.parse_delta` consumes it and replaces the DeltaMessage
//     with the tool parser's before anything reaches the client. Driving the
//     parsers bare makes an internal channel look like client-visible output.
//     So these cases run through `ShapeChatDelta`
//     (entrypoints/openai/serving_chat.cpp), this seam's analogue of
//     `parse_delta`: reasoning parser first, its content span routed into the
//     tool parser, the tool parser's DeltaMessage returned. Every upstream
//     ASSERTION is preserved verbatim; only the thing being observed moves from
//     "what one parser returned" to "what the client is sent", which is what the
//     assertions were always about.
//
//  C. test_muse_glimmer_parse_delta.py is SKIPPED upstream unless a real
//     Muse Glimmer checkpoint is on disk (it drives real token ids through a real
//     tokenizer). We have no checkpoint and no weights, so its cases run on the
//     same ShapeChatDelta harness fed ONE CHARACTER at a time — strictly harsher
//     than token-at-a-time, since it splits every marker at every offset.
//
//  D. `_Req.tools = None` (upstream) leaves this seam's `ToolsEnabled` false, so
//     the tool parser would never be driven at all. The cases that assert about
//     TOOL CALLS therefore register the tool the fixture calls; the cases that
//     assert about content/reasoning keep the upstream empty-tools fixture.
//
//  E. `include_reasoning=False` (parse_delta.py:146) is suppressed by
//     `DelegatingParser.parse_delta` upstream, not by the parser. This seam's
//     legacy ShapeChatDelta path has no such gate (a pre-existing seam property,
//     out of W7 scope), so the harness applies the suppression where parse_delta
//     applies it. The parser itself is unchanged.
#include <doctest/doctest.h>

#include <algorithm>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/reasoning_parsers/abstract.h"
#include "vllm/entrypoints/openai/reasoning_parsers/muse_glimmer.h"
#include "vllm/entrypoints/openai/serving_chat.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/muse_glimmer.h"

using namespace vllm::entrypoints::openai;
using json = nlohmann::json;

namespace {

// test_muse_glimmer_streaming.py:20 (_FRAMING) — any token that must NEVER
// appear in surfaced reasoning/content.
const std::vector<std::string>& Framing() {
  static const std::vector<std::string> f = {
      "<|start|>", "<|message|>", "<|eom|>",      "<|eot|>",
      "to=self",   "to=user",     "to=read.read", "<atem:"};
  return f;
}

ChatCompletionRequest Req() { return ChatCompletionRequest{}; }

ChatCompletionRequest ReqTools(const std::vector<std::string>& names) {
  ChatCompletionRequest r;
  std::vector<ChatCompletionToolsParam> tools;
  for (const std::string& n : names) {
    ChatCompletionToolsParam t;
    t.type = "function";
    t.function.name = n;
    tools.push_back(t);
  }
  r.tools = std::move(tools);
  return r;
}

struct StreamResult {
  std::string reasoning;
  std::string content;
  std::vector<DeltaToolCall> tool_calls;
};

// test_muse_glimmer_streaming.py:31 (_stream), rerouted per adaptation B.
StreamResult Stream(const std::string& raw, std::size_t chunk,
                    const ChatCompletionRequest& request) {
  MuseGlimmerReasoningParser reasoning_parser;
  MuseGlimmerToolParser tool_parser;
  StreamResult out;
  std::string prev;
  for (std::size_t i = 0; i < raw.size(); i += chunk) {
    const std::string cur = raw.substr(0, std::min(raw.size(), i + chunk));
    const std::string delta = cur.substr(prev.size());
    const std::optional<DeltaMessage> dm = ShapeChatDelta(
        prev, cur, delta, request, &tool_parser, &reasoning_parser);
    if (dm.has_value()) {
      // Adaptation E: the serving-layer include_reasoning gate.
      if (dm->reasoning.has_value() && request.include_reasoning) {
        out.reasoning += *dm->reasoning;
      }
      if (dm->content.has_value()) out.content += *dm->content;
      if (dm->tool_calls.has_value()) {
        for (const DeltaToolCall& tc : *dm->tool_calls) {
          out.tool_calls.push_back(tc);
        }
      }
    }
    prev = cur;
  }
  return out;
}

void CheckNoFraming(const std::string& text) {
  for (const std::string& f : Framing()) {
    CAPTURE(f);
    CAPTURE(text);
    CHECK(text.find(f) == std::string::npos);
  }
}

// test_muse_glimmer_streaming.py:66 (RAW_TOOLCALL) — captured raw framing.
const char* kRawToolCall =
    " to=self<|message|>I should read the hostname file to answer.<|eom|>"
    "<|start|>assistant to=read.read<|message|>"
    "<atem:function_calls>\n<atem:invoke name=\"read.read\">\n"
    "<atem:parameter name=\"path\">/etc/hostname</atem:parameter>\n"
    "</atem:invoke>\n</atem:function_calls>";

// test_muse_glimmer_streaming.py:106 (RAW_ANSWER).
const char* kRawAnswer =
    " to=self<|message|>Think about it.<|eom|>"
    "<|start|>assistant to=user<|message|>The answer is 42.<|eot|>";

// test_muse_glimmer_streaming.py:122 (RAW_TRUNCATED) — no closing <|eom|>.
const char* kRawTruncated =
    " to=self<|message|>Maybe I should call "
    "<atem:function_calls>\n<atem:invoke name=\"read.read\">\n"
    "<atem:parameter name=\"path\">/etc/hostname</atem:parameter>\n"
    "</atem:invoke>\n</atem:function_calls> but wait";

// test_muse_glimmer_streaming.py:75 (_check_toolcall_stream).
void CheckToolCallStream(std::size_t chunk) {
  CAPTURE(chunk);
  const StreamResult r = Stream(kRawToolCall, chunk, ReqTools({"read.read"}));
  // (a) no framing token leaks into reasoning or content
  CheckNoFraming(r.reasoning);
  CheckNoFraming(r.content);
  // (b) exactly one tool_call with the right name + args
  REQUIRE(r.tool_calls.size() == 1);
  REQUIRE(r.tool_calls[0].function.name.has_value());
  CHECK(*r.tool_calls[0].function.name == "read.read");
  REQUIRE(r.tool_calls[0].function.arguments.has_value());
  CHECK(json::parse(*r.tool_calls[0].function.arguments) ==
        json({{"path", "/etc/hostname"}}));
  CHECK(r.tool_calls[0].index == 0);
  REQUIRE(r.tool_calls[0].type.has_value());
  CHECK(*r.tool_calls[0].type == "function");
  REQUIRE(r.tool_calls[0].id.has_value());
  CHECK_FALSE(r.tool_calls[0].id->empty());
  // (c) reasoning captured separately and clean
  CHECK(r.reasoning == "I should read the hostname file to answer.");
  CHECK(r.content.empty());
}

}  // namespace

// ---------------------------------------------------------------------------
// tests/tool_use/test_muse_glimmer_reasoning_handoff.py
// ---------------------------------------------------------------------------

TEST_CASE("muse_glimmer reasoning: reasoning + tool call hands the channel over") {
  // reasoning_handoff.py:15 (Case 1) — the regression: a reasoning+tool-call
  // turn must NOT return content=None, or the tool parser starves.
  const std::string raw =
      " to=self<|message|>Let me call the tool.<|eom|>"
      "<|start|>assistant to=weather.get<|message|>"
      "<atem:function_calls>\n<atem:invoke name=\"weather.get\">\n"
      "<atem:parameter name=\"city\">Paris</atem:parameter>\n"
      "</atem:invoke>\n</atem:function_calls>";
  MuseGlimmerReasoningParser r;
  const ExtractedReasoning split = r.extract_reasoning(raw, Req());
  REQUIRE(split.reasoning.has_value());
  CHECK(*split.reasoning == "Let me call the tool.");
  REQUIRE(split.content.has_value());
  CHECK(split.content->find("<atem:invoke") != std::string::npos);

  MuseGlimmerToolParser t;
  const ExtractedToolCallInformation out =
      t.extract_tool_calls(*split.content, Req());
  REQUIRE(out.tools_called);
  REQUIRE(out.tool_calls.size() == 1);
  CHECK(out.tool_calls[0].function.name == "weather.get");
}

TEST_CASE("muse_glimmer reasoning: reasoning + user answer") {
  // reasoning_handoff.py:31 (Case 2).
  const std::string raw =
      " to=self<|message|>thinking<|eom|>"
      "<|start|>assistant to=user<|message|>The answer is 42.<|eot|>";
  MuseGlimmerReasoningParser r;
  const ExtractedReasoning split = r.extract_reasoning(raw, Req());
  REQUIRE(split.reasoning.has_value());
  CHECK(*split.reasoning == "thinking");
  REQUIRE(split.content.has_value());
  CHECK(*split.content == "The answer is 42.");

  MuseGlimmerToolParser t;
  CHECK_FALSE(t.extract_tool_calls(*split.content, Req()).tools_called);
}

TEST_CASE("muse_glimmer reasoning: plain content, no framing at all") {
  // reasoning_handoff.py:43 (Case 3).
  MuseGlimmerReasoningParser r;
  const ExtractedReasoning split =
      r.extract_reasoning("Just a direct answer.", Req());
  CHECK_FALSE(split.reasoning.has_value());
  REQUIRE(split.content.has_value());
  CHECK(*split.content == "Just a direct answer.");
}

TEST_CASE("muse_glimmer reasoning: reasoning + parallel tool calls") {
  // reasoning_handoff.py:49 (Case 4).
  const std::string raw =
      " to=self<|message|>need two calls<|eom|>"
      "<|start|>assistant to=math.add<|message|>"
      "<atem:function_calls>\n<atem:invoke name=\"math.add\">\n"
      "<atem:parameter name=\"a\">1</atem:parameter>\n</atem:invoke>\n"
      "</atem:function_calls><|eom|>"
      "<|start|>assistant to=math.mul<|message|>"
      "<atem:function_calls>\n<atem:invoke name=\"math.mul\">\n"
      "<atem:parameter name=\"a\">3</atem:parameter>\n</atem:invoke>\n"
      "</atem:function_calls><|eot|>";
  MuseGlimmerReasoningParser r;
  const ExtractedReasoning split = r.extract_reasoning(raw, Req());
  REQUIRE(split.reasoning.has_value());
  CHECK(*split.reasoning == "need two calls");
  REQUIRE(split.content.has_value());

  MuseGlimmerToolParser t;
  const ExtractedToolCallInformation out =
      t.extract_tool_calls(*split.content, Req());
  REQUIRE(out.tool_calls.size() == 2);
  CHECK(out.tool_calls[0].function.name == "math.add");
  CHECK(out.tool_calls[1].function.name == "math.mul");
}

TEST_CASE("muse_glimmer reasoning: a header-less channel switch does not eat "
          "the tool call") {
  // NOT one of the five upstream modules' cases, but the guarantee
  // muse_glimmer_reasoning_parser.py:46-62 exists for and documents: the model
  // sometimes leaves the analysis channel WITHOUT emitting <|eom|>, writing a
  // bare `to=<tool><|message|>` header instead ("observed deterministically for
  // a call with EMPTY arguments on a tool that has optional parameters").
  // Both unterminated-reasoning patterns must therefore stop at the next channel
  // header. An unbounded `...$` version consumes the real tool call along with
  // the reasoning: is_reasoning_end never fires, the tool parser is never
  // invoked, and the ENTIRE generation is dropped — empty reasoning, empty
  // content, no tool call.
  const std::string raw =
      " to=self<|message|>I will just call it. "
      "to=read.read<|message|>"
      "<atem:function_calls>\n<atem:invoke name=\"read.read\">\n"
      "</atem:invoke>\n</atem:function_calls>";
  MuseGlimmerReasoningParser r;
  CHECK(r.is_reasoning_end(raw));
  const ExtractedReasoning split = r.extract_reasoning(raw, Req());
  REQUIRE(split.reasoning.has_value());
  CHECK(split.reasoning->find("I will just call it.") != std::string::npos);
  REQUIRE(split.content.has_value());
  CHECK(split.content->find("<atem:invoke") != std::string::npos);

  MuseGlimmerToolParser t;
  const ExtractedToolCallInformation out =
      t.extract_tool_calls(*split.content, Req());
  REQUIRE(out.tools_called);
  REQUIRE(out.tool_calls.size() == 1);
  CHECK(out.tool_calls[0].function.name == "read.read");
  CHECK(out.tool_calls[0].function.arguments == "{}");
}

// ---------------------------------------------------------------------------
// tests/tool_use/test_muse_glimmer_streaming.py
// ---------------------------------------------------------------------------

TEST_CASE("muse_glimmer streaming: tool call, 3-char chunks") {
  CheckToolCallStream(3);  // streaming.py:92
}

TEST_CASE("muse_glimmer streaming: tool call, one char at a time") {
  CheckToolCallStream(1);  // streaming.py:96 — worst case, mid-marker deltas
}

TEST_CASE("muse_glimmer streaming: tool call, 17-char chunks") {
  CheckToolCallStream(17);  // streaming.py:101
}

TEST_CASE("muse_glimmer streaming: reasoning then a to=user answer") {
  // streaming.py:112.
  const StreamResult r = Stream(kRawAnswer, 3, Req());
  CheckNoFraming(r.reasoning);
  CheckNoFraming(r.content);
  CHECK(r.reasoning == "Think about it.");
  CHECK(r.content == "The answer is 42.");
  CHECK(r.tool_calls.empty());
}

TEST_CASE("muse_glimmer streaming: a to=user answer survives WITH tools armed") {
  // NOT an upstream case — it covers deviation 4 in tool_parsers/muse_glimmer.h.
  // Upstream's RAW_ANSWER case runs with no registered tools, so upstream never
  // streams a prose answer THROUGH the tool parser. In this seam that is the
  // normal shape of a tools-armed request the model chooses not to call a tool
  // on, and without the unframed-content fallback the answer is dropped on the
  // floor (the tool parser owns every delta once content is being forwarded).
  const StreamResult r = Stream(kRawAnswer, 3, ReqTools({"read.read"}));
  CheckNoFraming(r.reasoning);
  CheckNoFraming(r.content);
  CHECK(r.reasoning == "Think about it.");
  CHECK(r.content == "The answer is 42.");
  CHECK(r.tool_calls.empty());
}

TEST_CASE("muse_glimmer streaming: truncated CoT yields no non-streaming call") {
  // streaming.py:130 — the reasoning half (the tool half lives in the
  // tool_parsers sibling file).
  MuseGlimmerReasoningParser r;
  const ExtractedReasoning split = r.extract_reasoning(kRawTruncated, Req());
  REQUIRE(split.reasoning.has_value());
  CHECK(split.reasoning->find("Maybe I should call") != std::string::npos);
}

TEST_CASE("muse_glimmer streaming: truncated CoT yields no streaming call") {
  // streaming.py:138. Adaptation D: the tool is registered so the tool parser
  // is actually driven; the point is that the handoff never fires.
  const StreamResult r = Stream(kRawTruncated, 3, ReqTools({"read.read"}));
  CHECK(r.tool_calls.empty());
}

// ---------------------------------------------------------------------------
// tests/tool_use/test_muse_glimmer_parse_delta.py (adaptation C: char-at-a-time
// through ShapeChatDelta instead of token-at-a-time through parse_delta)
// ---------------------------------------------------------------------------

TEST_CASE("muse_glimmer parse_delta: no tools, reasoning then answer") {
  // parse_delta.py:92 — the core regression: the reasoning phase must stay
  // active for a no-tools turn.
  const StreamResult r = Stream(
      " to=self<|message|>Let me think step by step about the sum.<|eom|>"
      "<|start|>assistant to=user<|message|>The answer is 42.<|eot|>",
      1, Req());
  CheckNoFraming(r.content);
  CHECK(r.reasoning == "Let me think step by step about the sum.");
  CHECK(r.content == "The answer is 42.");
  CHECK(r.tool_calls.empty());
}

TEST_CASE("muse_glimmer parse_delta: content only") {
  // parse_delta.py:105.
  const StreamResult r =
      Stream(" to=user<|message|>Just a direct answer.<|eot|>", 1, Req());
  CheckNoFraming(r.content);
  CHECK(r.content == "Just a direct answer.");
  CHECK(r.tool_calls.empty());
}

TEST_CASE("muse_glimmer parse_delta: tool call") {
  // parse_delta.py:114.
  const StreamResult r = Stream(
      " to=self<|message|>I should read the hostname.<|eom|>"
      "<|start|>assistant to=read.read<|message|>"
      "<atem:function_calls>\n<atem:invoke name=\"read.read\">\n"
      "<atem:parameter name=\"path\">/etc/hostname</atem:parameter>\n"
      "</atem:invoke>\n</atem:function_calls>",
      1, ReqTools({"read.read"}));
  CheckNoFraming(r.content);
  CHECK(r.reasoning == "I should read the hostname.");
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].index == 0);
  REQUIRE(r.tool_calls[0].function.name.has_value());
  CHECK(*r.tool_calls[0].function.name == "read.read");
  REQUIRE(r.tool_calls[0].function.arguments.has_value());
  CHECK(json::parse(*r.tool_calls[0].function.arguments) ==
        json({{"path", "/etc/hostname"}}));
}

TEST_CASE("muse_glimmer parse_delta: a contemplated invoke is not a tool call") {
  // parse_delta.py:131.
  const StreamResult r = Stream(kRawTruncated, 1, ReqTools({"read.read"}));
  CheckNoFraming(r.content);
  CHECK(r.tool_calls.empty());
  CHECK(r.reasoning.find("Maybe I should call") != std::string::npos);
}

TEST_CASE("muse_glimmer parse_delta: reasoning suppressed when not requested") {
  // parse_delta.py:146. Adaptation E.
  ChatCompletionRequest req;
  req.include_reasoning = false;
  const StreamResult r = Stream(
      " to=self<|message|>secret thoughts<|eom|>"
      "<|start|>assistant to=user<|message|>Public answer.<|eot|>",
      1, req);
  CHECK(r.reasoning.empty());
  CHECK(r.content == "Public answer.");
  CheckNoFraming(r.content);
}

// ---------------------------------------------------------------------------
// Registry + is_reasoning_end (this seam's text form of the token-ID method).
// ---------------------------------------------------------------------------

TEST_CASE("muse_glimmer reasoning: the name resolves from the registry") {
  CHECK(get_reasoning_parser("muse_glimmer") != nullptr);
}

TEST_CASE("muse_glimmer reasoning: is_reasoning_end only fires on a real tool "
          "channel") {
  // muse_glimmer_reasoning_parser.py:129 — a to=user answer is NOT the end of
  // reasoning, and an invoke echoed inside the CoT never flips the phase.
  MuseGlimmerReasoningParser r;
  CHECK_FALSE(r.is_reasoning_end(" to=self<|message|>thinking"));
  CHECK_FALSE(r.is_reasoning_end(
      " to=self<|message|>thinking<|eom|>"
      "<|start|>assistant to=user<|message|>The answer is 42.<|eot|>"));
  CHECK_FALSE(r.is_reasoning_end(kRawTruncated));
  CHECK(r.is_reasoning_end(kRawToolCall));
}
