// Tests for the Muse Glimmer ATEM tool-call parser.
// (vllm/tool_parsers/muse_glimmer_tool_parser.py @ 075d645af — vLLM PR #51655
// head, NOT the parity pin. See .agents/specs/muse-glimmer.md §0.)
//
// PORTS, case for case, with the upstream fixtures byte-for-byte:
//   - tests/tool_use/test_muse_glimmer_tool_parser.py        (5 cases)
//   - tests/tool_use/test_muse_glimmer_toolname_normalize.py (6 cases)
//   - tests/tool_use/test_muse_glimmer_streaming.py::
//       test_truncated_cot_no_toolcall_nonstreaming — the tool-parser half
//       (its reasoning-parser half lives in the reasoning_parsers sibling file,
//       together with the rest of that module's cases)
//
// HARNESS ADAPTATION (Python -> C++), the whole list:
//   - Upstream builds its parser with `MuseGlimmerToolParser.__new__(cls)` to
//     skip an `__init__` that needs a tokenizer. This seam's ToolParser has a
//     default ctor and no tokenizer (tool_parsers/abstract.h), so the parser is
//     simply constructed. NOTE: that `__new__` trick is also why the upstream
//     STREAMING module cannot run at all — the reasoning parser it constructs
//     the same way raises AttributeError on the instance cursors `__init__`
//     would have set. Constructing normally is what makes those cases runnable.
//   - Upstream passes `None` for the request on the no-tools cases; the C++
//     signature takes a const ref, so a default-constructed request (no `tools`)
//     stands in — identical input to `_registered_names`.
//   - `SimpleNamespace(tools=[SimpleNamespace(function=...)])` becomes a real
//     ChatCompletionToolsParam list.
#include <doctest/doctest.h>

#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/muse_glimmer.h"

using namespace vllm::entrypoints::openai;
using json = nlohmann::json;

namespace {

// test_muse_glimmer_tool_parser.py:7 — the parser under test.
ChatCompletionRequest Req() { return ChatCompletionRequest{}; }

// test_muse_glimmer_toolname_normalize.py:15 (_req).
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

// test_muse_glimmer_toolname_normalize.py:21 (_call).
std::string Call(const std::string& name) {
  return "<|start|>assistant to=" + name + "<|message|>" +
         "<atem:function_calls>\n<atem:invoke name=\"" + name + "\">\n" +
         "<atem:parameter name=\"city\">Paris</atem:parameter>\n" +
         "</atem:invoke>\n</atem:function_calls>";
}

}  // namespace

// ---------------------------------------------------------------------------
// tests/tool_use/test_muse_glimmer_tool_parser.py
// ---------------------------------------------------------------------------

TEST_CASE("muse_glimmer tool: single tool call after reasoning") {
  // test_muse_glimmer_tool_parser.py:18 (Case 1).
  const std::string o1 =
      "to=self<|message|>Let me check the weather.<|eom|>"
      "<|start|>assistant to=weather.get<|message|>"
      "<atem:function_calls>\n<atem:invoke name=\"weather.get\">\n"
      "<atem:parameter name=\"city\">Paris</atem:parameter>\n"
      "<atem:parameter name=\"units\">celsius</atem:parameter>\n"
      "</atem:invoke>\n</atem:function_calls><|eot|>";
  MuseGlimmerToolParser p;
  const ExtractedToolCallInformation out = p.extract_tool_calls(o1, Req());
  REQUIRE(out.tools_called);
  REQUIRE(out.tool_calls.size() == 1);
  CHECK(out.tool_calls[0].function.name == "weather.get");
  CHECK(json::parse(out.tool_calls[0].function.arguments) ==
        json({{"city", "Paris"}, {"units", "celsius"}}));
}

TEST_CASE("muse_glimmer tool: parallel calls in <|eom|>-separated messages") {
  // test_muse_glimmer_tool_parser.py:34 (Case 2).
  const std::string o2 =
      "<|start|>assistant to=math.add<|message|>"
      "<atem:function_calls>\n<atem:invoke name=\"math.add\">\n"
      "<atem:parameter name=\"a\">1</atem:parameter>\n"
      "<atem:parameter name=\"b\">2</atem:parameter>\n"
      "</atem:invoke>\n</atem:function_calls><|eom|>"
      "<|start|>assistant to=math.mul<|message|>"
      "<atem:function_calls>\n<atem:invoke name=\"math.mul\">\n"
      "<atem:parameter name=\"a\">3</atem:parameter>\n"
      "<atem:parameter name=\"b\">4</atem:parameter>\n"
      "</atem:invoke>\n</atem:function_calls><|eot|>";
  MuseGlimmerToolParser p;
  const ExtractedToolCallInformation out = p.extract_tool_calls(o2, Req());
  REQUIRE(out.tools_called);
  REQUIRE(out.tool_calls.size() == 2);
  CHECK(out.tool_calls[0].function.name == "math.add");
  CHECK(out.tool_calls[1].function.name == "math.mul");
  // JSON-typed values decode to ints (x-parser: json, allow_non_json: True).
  CHECK(json::parse(out.tool_calls[0].function.arguments) ==
        json({{"a", 1}, {"b", 2}}));
}

TEST_CASE("muse_glimmer tool: an invoke echoed inside reasoning is NOT a call") {
  // test_muse_glimmer_tool_parser.py:53 (Case 3) — channel scoping.
  const std::string o3 =
      "to=self<|message|>I could call <atem:invoke name=\"evil.fn\">"
      "<atem:parameter name=\"x\">1</atem:parameter></atem:invoke> but I will "
      "not.<|eom|>"
      "<|start|>assistant to=user<|message|>The answer is 42.<|eot|>";
  MuseGlimmerToolParser p;
  const ExtractedToolCallInformation out = p.extract_tool_calls(o3, Req());
  CHECK_FALSE(out.tools_called);  // channel scoping failed - echoed invoke parsed!
  REQUIRE(out.content.has_value());
  CHECK(*out.content == "The answer is 42.");
}

TEST_CASE("muse_glimmer tool: plain answer calls no tools") {
  // test_muse_glimmer_tool_parser.py:64 (Case 4).
  MuseGlimmerToolParser p;
  const ExtractedToolCallInformation out =
      p.extract_tool_calls("to=user<|message|>Just a plain answer.<|eot|>", Req());
  CHECK_FALSE(out.tools_called);
}

TEST_CASE("muse_glimmer tool: object / array / bool parameter values") {
  // test_muse_glimmer_tool_parser.py:71 (Case 5).
  const std::string o5 =
      "<|start|>assistant to=api.call<|message|>"
      "<atem:function_calls>\n<atem:invoke name=\"api.call\">\n"
      "<atem:parameter name=\"payload\">{\"nested\": [1, 2, 3]}</atem:parameter>\n"
      "<atem:parameter name=\"flag\">true</atem:parameter>\n"
      "</atem:invoke>\n</atem:function_calls><|eot|>";
  MuseGlimmerToolParser p;
  const ExtractedToolCallInformation out = p.extract_tool_calls(o5, Req());
  REQUIRE(out.tool_calls.size() == 1);
  CHECK(json::parse(out.tool_calls[0].function.arguments) ==
        json({{"payload", {{"nested", {1, 2, 3}}}}, {"flag", true}}));
}

// ---------------------------------------------------------------------------
// tests/tool_use/test_muse_glimmer_toolname_normalize.py
// ---------------------------------------------------------------------------

TEST_CASE("muse_glimmer toolname: a doubled bare name collapses") {
  // test_muse_glimmer_toolname_normalize.py:30.
  MuseGlimmerToolParser p;
  const ExtractedToolCallInformation out = p.extract_tool_calls(
      Call("get_weather.get_weather"), ReqTools({"get_weather"}));
  REQUIRE(out.tools_called);
  REQUIRE(out.tool_calls.size() == 1);
  CHECK(out.tool_calls[0].function.name == "get_weather");
}

TEST_CASE("muse_glimmer toolname: a namespaced name is preserved") {
  // test_muse_glimmer_toolname_normalize.py:36.
  MuseGlimmerToolParser p;
  const ExtractedToolCallInformation out =
      p.extract_tool_calls(Call("weather.get"), ReqTools({"weather.get"}));
  REQUIRE(out.tool_calls.size() == 1);
  CHECK(out.tool_calls[0].function.name == "weather.get");
}

TEST_CASE("muse_glimmer toolname: an unambiguous trailing segment binds") {
  // test_muse_glimmer_toolname_normalize.py:41 — emitted foo.get_weather,
  // registered bare get_weather -> bind. This is the assertion upstream's OWN
  // _normalize_name does not satisfy (its docstring argues leaf matching is
  // unsafe); see muse_glimmer.h for how both are honoured.
  MuseGlimmerToolParser p;
  const ExtractedToolCallInformation out =
      p.extract_tool_calls(Call("foo.get_weather"), ReqTools({"get_weather"}));
  REQUIRE(out.tool_calls.size() == 1);
  CHECK(out.tool_calls[0].function.name == "get_weather");
}

TEST_CASE("muse_glimmer toolname: an ambiguous trailing segment is left alone") {
  // test_muse_glimmer_toolname_normalize.py:47 — two registered tools share the
  // leaf 'get', so the emitted name must NOT be rewritten.
  MuseGlimmerToolParser p;
  const ExtractedToolCallInformation out = p.extract_tool_calls(
      Call("x.get"), ReqTools({"weather.get", "time.get"}));
  REQUIRE(out.tool_calls.size() == 1);
  CHECK(out.tool_calls[0].function.name == "x.get");
}

TEST_CASE("muse_glimmer toolname: no registered tools means pass-through") {
  // test_muse_glimmer_toolname_normalize.py:53.
  MuseGlimmerToolParser p;
  const ExtractedToolCallInformation out =
      p.extract_tool_calls(Call("get_weather.get_weather"), Req());
  REQUIRE(out.tool_calls.size() == 1);
  CHECK(out.tool_calls[0].function.name == "get_weather.get_weather");
}

TEST_CASE("muse_glimmer toolname: an exact match is kept") {
  // test_muse_glimmer_toolname_normalize.py:58.
  MuseGlimmerToolParser p;
  const ExtractedToolCallInformation out =
      p.extract_tool_calls(Call("get_weather"), ReqTools({"get_weather"}));
  REQUIRE(out.tool_calls.size() == 1);
  CHECK(out.tool_calls[0].function.name == "get_weather");
}

TEST_CASE("muse_glimmer toolname: the leaf rule never rewrites onto a "
          "NAMESPACED tool") {
  // NOT an upstream case. Guards the safety invariant upstream's docstring
  // states (muse_glimmer_tool_parser.py:324-326): an emitted `weather.get`
  // against a registered `{calendar.get}` has a UNIQUE leaf match and must
  // still be left alone, or the wrong tool is dispatched.
  MuseGlimmerToolParser p;
  const ExtractedToolCallInformation out =
      p.extract_tool_calls(Call("weather.get"), ReqTools({"calendar.get"}));
  REQUIRE(out.tool_calls.size() == 1);
  CHECK(out.tool_calls[0].function.name == "weather.get");
}

TEST_CASE("muse_glimmer toolname: a leaf shared by a bare AND a namespaced tool "
          "is ambiguous") {
  // NOT an upstream case. Guards the OTHER half of the leaf rule: upstream's
  // `x.get` fixture is blocked by the bare-target requirement alone, so without
  // this case the UNIQUENESS requirement is never exercised. With both
  // `get_weather` and `ns.get_weather` registered, an emitted `foo.get_weather`
  // does not identify one tool and must stand unchanged.
  MuseGlimmerToolParser p;
  const ExtractedToolCallInformation out = p.extract_tool_calls(
      Call("foo.get_weather"), ReqTools({"get_weather", "ns.get_weather"}));
  REQUIRE(out.tool_calls.size() == 1);
  CHECK(out.tool_calls[0].function.name == "foo.get_weather");
}

// ---------------------------------------------------------------------------
// tests/tool_use/test_muse_glimmer_streaming.py — the tool-parser half of
// test_truncated_cot_no_toolcall_nonstreaming (streaming.py:130).
// ---------------------------------------------------------------------------

TEST_CASE("muse_glimmer tool: a truncated CoT invoke is not a tool call") {
  // streaming.py:122 (RAW_TRUNCATED) — NO closing <|eom|>.
  const std::string raw =
      " to=self<|message|>Maybe I should call "
      "<atem:function_calls>\n<atem:invoke name=\"read.read\">\n"
      "<atem:parameter name=\"path\">/etc/hostname</atem:parameter>\n"
      "</atem:invoke>\n</atem:function_calls> but wait";
  MuseGlimmerToolParser p;
  const ExtractedToolCallInformation out = p.extract_tool_calls(raw, Req());
  CHECK_FALSE(out.tools_called);
  CHECK(out.tool_calls.empty());
}

// ---------------------------------------------------------------------------
// Registry — the seam-level obligation that comes with a new parser name.
// ---------------------------------------------------------------------------

TEST_CASE("muse_glimmer tool: the name resolves from the registry") {
  CHECK(get_tool_parser("muse_glimmer") != nullptr);
}
