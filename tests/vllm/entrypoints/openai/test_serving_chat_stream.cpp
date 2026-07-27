// Exact-parity gate for the SERVING-SSE dispatch swap (ROAD-V1-C8
// TOOLS-STREAMING-PARSER — serving row): the engine-backed tool-call parser
// (qwen3 / seed_oss / kimi_k2) driven through the OpenAI /v1/chat/completions
// streaming path, one SSE chunk per emitted delta.
//
// Ported/gated against vLLM 0.26.0.dev0 (@ 555967922):
//   vllm/entrypoints/openai/chat_completion/serving.py:607 (parser.parse_delta
//   drive) + :688-693 (finish_reason "tool_calls" flip) +
//   vllm/parser/parser_manager.py:76 (ParserManager.get_parser dispatch).
//
// The parser is a PURE FUNCTION of the streamed (delta_text, delta_token_ids), so
// the SSE stream is exactly gateable: for a FIXED delta stream, our serving path
// must emit the SAME `data:` chunk SEQUENCE vLLM's chat_completion_stream_
// generator emits — the leading role frame, one chunk per emitted delta (a
// withheld null delta yields no chunk), each carrying the same
// delta.{reasoning,content,tool_calls.function.{name,arguments}} payload, and the
// terminal finish_reason with the tool_calls flip. The delta payloads in the
// golden .inc were captured directly from the pinned oracle
// (tools/parity/dump_serving_chat_stream.py). The SSE ENVELOPE serialization is
// the production `ChatCompletionStreamResponse` to_json (already gated against
// vLLM in test_openai_serving); here both the expected (golden) and actual
// (production ShapeChatDeltaEngine) sides frame through that same serializer, so
// the assertion discriminates on the parser-derived payload + framing cadence —
// exactly what the dispatch swap adds.
//
// Tool-call ids use the deterministic factory (chatcmpl-tool-<idx>; kimi_k2
// functions.<name>:<idx>) on both sides so ids compare exactly; production keeps
// the random-uuid factory.
#include <doctest/doctest.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/serving_chat.h"
#include "vllm/parser/engine/parser_engine.h"
#include "vllm/parser/parser_manager.h"

using vllm::parser::get_parser_engine;
namespace oai = vllm::entrypoints::openai;

namespace {

// Golden record types (populated by the generated .inc).
struct SSTC {
  int index;
  std::optional<std::string> id;
  std::optional<std::string> type;
  std::optional<std::string> name;
  std::optional<std::string> arguments;
};
struct SSDelta {
  std::optional<std::string> role;
  std::optional<std::string> content;
  std::optional<std::string> reasoning;
  std::vector<SSTC> tool_calls;
};
struct SSChunk {
  bool role_frame;
  SSDelta delta;
  std::optional<std::string> finish_reason;
};
struct SSScenario {
  std::string name;
  std::string cfg;
  bool thinking;
  bool include_reasoning;
  std::vector<std::string> deltas;
  std::vector<SSChunk> chunks;
};

#include "test_serving_chat_stream_goldens.inc"

// Fixed envelope constants — the structural (non-parser) fields. Their
// serialization parity is covered by test_openai_serving; here they are held
// constant on both sides so the comparison isolates the parser payload.
constexpr const char* kId = "chatcmpl-test";
constexpr int64_t kCreated = 0;
constexpr const char* kModel = "test-model";

// The deterministic id factory matching the dump harness stub.
vllm::parser::engine::ParserEngine::IdFactory det_id_factory() {
  return [](const std::string& id_type, const std::string& func_name,
            int idx) -> std::string {
    if (id_type == "kimi_k2")
      return "functions." + func_name + ":" + std::to_string(idx);
    return "chatcmpl-tool-" + std::to_string(idx);
  };
}

// Frame one (delta, finish_reason) into the production `data: {json}\n\n` SSE
// chunk, exactly as create_chat_completion does.
std::string frame(const oai::DeltaMessage& delta,
                  const std::optional<std::string>& finish) {
  oai::ChatCompletionResponseStreamChoice choice;
  choice.index = 0;
  choice.delta = delta;
  choice.finish_reason = finish;
  oai::ChatCompletionStreamResponse resp;
  resp.id = kId;
  resp.created = kCreated;
  resp.model = kModel;
  resp.choices.push_back(std::move(choice));
  return "data: " + nlohmann::json(resp).dump() + "\n\n";
}

// The role frame (delta = {role:"assistant", content:""}).
std::string role_frame() {
  oai::DeltaMessage d;
  d.role = "assistant";
  d.content = "";
  return frame(d, std::nullopt);
}

// A ChatCompletionRequest carrying tools (so ToolsEnabled) + include_reasoning,
// exactly what the serving path projects onto the engine ParserRequest.
oai::ChatCompletionRequest make_request(bool include_reasoning) {
  oai::ChatCompletionRequest req;
  oai::ChatCompletionToolsParam tool;
  tool.function.name = "get_weather";
  req.tools = std::vector<oai::ChatCompletionToolsParam>{tool};
  req.include_reasoning = include_reasoning;
  return req;
}

// EXPECTED: render the oracle-captured golden chunk sequence into `data:` lines.
std::vector<std::string> expected_chunks(const SSScenario& s) {
  std::vector<std::string> out;
  for (const SSChunk& c : s.chunks) {
    if (c.role_frame) {
      out.push_back(role_frame());
      continue;
    }
    oai::DeltaMessage d;
    d.role = c.delta.role;
    d.content = c.delta.content;
    d.reasoning = c.delta.reasoning;
    if (!c.delta.tool_calls.empty()) {
      std::vector<oai::DeltaToolCall> tcs;
      for (const SSTC& g : c.delta.tool_calls) {
        oai::DeltaToolCall t;
        t.index = g.index;
        t.id = g.id;
        t.type = g.type;
        t.function.name = g.name;
        t.function.arguments = g.arguments;
        tcs.push_back(std::move(t));
      }
      d.tool_calls = std::move(tcs);
    }
    out.push_back(frame(d, c.finish_reason));
  }
  return out;
}

// ACTUAL: drive the SERVING dispatch (get_parser_engine == MakeParserEngine's
// selection) + the production ShapeChatDeltaEngine over the fixed delta stream,
// framing exactly as create_chat_completion's streaming loop (role frame, skip a
// withheld null delta, terminal finish_reason with the tool_calls flip).
std::vector<std::string> actual_chunks(const SSScenario& s) {
  auto parser = get_parser_engine(s.cfg, s.thinking, /*tokenizer=*/nullptr);
  REQUIRE_MESSAGE(parser != nullptr, "no engine parser for cfg " << s.cfg);
  parser->set_id_factory(det_id_factory());
  const oai::ChatCompletionRequest req = make_request(s.include_reasoning);

  std::vector<std::string> out;
  out.push_back(role_frame());
  bool tools_streamed = false;
  for (std::size_t i = 0; i < s.deltas.size(); ++i) {
    const bool finished = (i + 1 == s.deltas.size());
    std::optional<oai::DeltaMessage> delta = oai::ShapeChatDeltaEngine(
        parser.get(), s.deltas[i], /*delta_token_ids=*/{}, req, finished);
    if (delta.has_value() && delta->tool_calls.has_value() &&
        !delta->tool_calls->empty()) {
      tools_streamed = true;
    }
    if (!delta.has_value()) {
      if (!finished) continue;  // withheld: no chunk
      delta = oai::DeltaMessage{};
    }
    std::optional<std::string> finish;
    if (finished) {
      finish = tools_streamed ? std::optional<std::string>("tool_calls")
                              : std::optional<std::string>("stop");
    }
    out.push_back(frame(*delta, finish));
  }
  return out;
}

}  // namespace

TEST_CASE("serving SSE: engine-backed parser stream parity, chunk-for-chunk") {
  for (const SSScenario& s : kServingStreamGoldens) {
    CAPTURE(s.name);
    const std::vector<std::string> want = expected_chunks(s);
    const std::vector<std::string> got = actual_chunks(s);
    REQUIRE_MESSAGE(got.size() == want.size(),
                    s.name << " chunk count " << got.size() << " != "
                           << want.size());
    for (std::size_t i = 0; i < want.size(); ++i) {
      CHECK_MESSAGE(got[i] == want[i],
                    s.name << " chunk[" << i << "]\n  want=" << want[i]
                           << "  got=" << got[i]);
    }
  }
}

TEST_CASE("serving SSE: name-selected dispatch (engine-backed vs legacy seam)") {
  // The 0.26 dispatch rule (parser_manager.py:76): an engine-backed name
  // resolves through get_parser_engine; every other name falls to the legacy
  // tool_parsers seam (get_parser_engine returns null, so MakeToolParser handles
  // it and the default no-tool-parser path is untouched).
  CHECK(get_parser_engine("qwen3") != nullptr);
  CHECK(get_parser_engine("seed_oss") != nullptr);
  CHECK(get_parser_engine("kimi_k2") != nullptr);
  CHECK(get_parser_engine("hermes") == nullptr);
  CHECK(get_parser_engine("mistral") == nullptr);
  CHECK(get_parser_engine("") == nullptr);
}
