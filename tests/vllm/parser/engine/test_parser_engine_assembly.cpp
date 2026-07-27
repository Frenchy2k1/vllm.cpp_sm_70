// Exact-parity gate for the unified parser ASSEMBLY layer (ROAD-V1-C8
// TOOLS-STREAMING-PARSER-ASSEMBLY).
//
// Ported/gated against vLLM 0.26.0.dev0 (@ 555967922):
//   vllm/parser/engine/parser_engine.py (ParserEngine — event -> DeltaMessage
//   assembly + one-shot extract_tool_calls), vllm/parser/qwen3.py,
//   vllm/parser/seed_oss.py, vllm/parser/kimi_k2.py.
//
// The assembly layer is a PURE FUNCTION of the (delta_text, delta_token_ids)
// stream, so it gates EXACTLY like the engine core: our per-delta DeltaMessage
// sequence AND the one-shot ExtractedToolCallInformation must equal, field-for-
// field, what the vLLM 0.26 Python assembly emits for the identical stream. The
// goldens in the .inc were captured directly from the pinned oracle (see
// tools/parity/dump_parser_engine_assembly.py). The tool-call id for qwen3's
// "random" id_type is made DETERMINISTIC on both sides (chatcmpl-tool-<idx>) so
// ids compare exactly; production keeps the random uuid factory.
//
// Coverage: qwen3 reasoning + XML tool call (whole-delta AND char-by-char
// cadence), reasoning-suppressed (include_reasoning=false), thinking-off plain
// content, two consecutive tool calls (tool_index increment), an unfinished
// tool call flushed by finish(), seed_oss wrapper-token variant, and kimi_k2
// JSON args with the held-back top-level brace (char-by-char AND whole-delta).
#include <doctest/doctest.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/parser/engine/parser_engine.h"
#include "vllm/parser/parser_manager.h"

using vllm::parser::engine::ParserRequest;
using vllm::parser::get_parser_engine;
namespace oai = vllm::entrypoints::openai;

namespace {

// Golden record types (populated by the generated .inc).
struct GTC {
  int index;
  std::optional<std::string> id;
  std::optional<std::string> type;
  std::optional<std::string> name;
  std::optional<std::string> arguments;
};
struct GDelta {
  bool present;
  std::optional<std::string> content;
  std::optional<std::string> reasoning;
  std::vector<GTC> tool_calls;
};
struct GXTC {
  std::string id;
  std::string name;
  std::string arguments;
};
struct GExtract {
  bool tools_called;
  std::optional<std::string> content;
  std::vector<GXTC> tool_calls;
};
struct GScenario {
  std::string name;
  std::string cfg;
  bool thinking;
  bool include_reasoning;
  std::vector<std::string> deltas;
  std::vector<GDelta> stream;
  GExtract extract;
};

#include "test_parser_engine_assembly_goldens.inc"

// Deterministic id factory matching the dump harness monkeypatch.
vllm::parser::engine::ParserEngine::IdFactory det_id_factory() {
  return [](const std::string& id_type, const std::string& func_name,
            int idx) -> std::string {
    if (id_type == "kimi_k2")
      return "functions." + func_name + ":" + std::to_string(idx);
    return "chatcmpl-tool-" + std::to_string(idx);
  };
}

std::unique_ptr<vllm::parser::engine::ParserEngine> make(const GScenario& s) {
  auto p = get_parser_engine(s.cfg, s.thinking, /*tokenizer=*/nullptr);
  REQUIRE_MESSAGE(p != nullptr, "no engine parser for cfg " << s.cfg);
  p->set_id_factory(det_id_factory());
  return p;
}

void check_tc(const std::string& where, const GTC& want,
              const oai::DeltaToolCall& got) {
  CHECK_MESSAGE(got.index == want.index, where << " tc index");
  CHECK_MESSAGE(got.id == want.id, where << " tc id");
  CHECK_MESSAGE(got.type == want.type, where << " tc type");
  CHECK_MESSAGE(got.function.name == want.name, where << " tc name");
  CHECK_MESSAGE(got.function.arguments == want.arguments, where << " tc args");
}

void check_delta(const std::string& where, const GDelta& want,
                 const std::optional<oai::DeltaMessage>& got) {
  CHECK_MESSAGE(got.has_value() == want.present, where << " present");
  if (!got.has_value() || !want.present) return;
  CHECK_MESSAGE(got->content == want.content, where << " content");
  CHECK_MESSAGE(got->reasoning == want.reasoning, where << " reasoning");
  const std::size_t n = got->tool_calls ? got->tool_calls->size() : 0;
  CHECK_MESSAGE(n == want.tool_calls.size(), where << " tool_calls size");
  if (n != want.tool_calls.size()) return;
  for (std::size_t i = 0; i < n; ++i)
    check_tc(where + " tc[" + std::to_string(i) + "]", want.tool_calls[i],
             (*got->tool_calls)[i]);
}

}  // namespace

TEST_CASE("parser assembly: streaming DeltaMessage parity, event-for-event") {
  for (const GScenario& s : kAssemblyGoldens) {
    CAPTURE(s.name);
    auto parser = make(s);
    ParserRequest req;
    req.include_reasoning = s.include_reasoning;
    REQUIRE(s.stream.size() == s.deltas.size());
    for (std::size_t i = 0; i < s.deltas.size(); ++i) {
      const bool finished = (i + 1 == s.deltas.size());
      std::optional<oai::DeltaMessage> got =
          parser->parse_delta(s.deltas[i], {}, req, finished);
      check_delta(s.name + " delta[" + std::to_string(i) + "]", s.stream[i],
                  got);
    }
  }
}

TEST_CASE("parser assembly: one-shot extract_tool_calls parity, field-for-field") {
  for (const GScenario& s : kAssemblyGoldens) {
    CAPTURE(s.name);
    auto parser = make(s);
    ParserRequest req;
    req.include_reasoning = s.include_reasoning;
    std::string full;
    for (const auto& d : s.deltas) full += d;

    oai::ExtractedToolCallInformation info =
        parser->extract_tool_calls(full, req);

    const std::string where = s.name + " extract";
    CHECK_MESSAGE(info.tools_called == s.extract.tools_called,
                  where << " tools_called");
    CHECK_MESSAGE(info.content == s.extract.content, where << " content");
    CHECK_MESSAGE(info.tool_calls.size() == s.extract.tool_calls.size(),
                  where << " tool_calls size");
    if (info.tool_calls.size() != s.extract.tool_calls.size()) continue;
    for (std::size_t i = 0; i < info.tool_calls.size(); ++i) {
      const std::string w = where + " tc[" + std::to_string(i) + "]";
      CHECK_MESSAGE(info.tool_calls[i].id == s.extract.tool_calls[i].id,
                    w << " id");
      CHECK_MESSAGE(info.tool_calls[i].function.name ==
                        s.extract.tool_calls[i].name,
                    w << " name");
      CHECK_MESSAGE(info.tool_calls[i].function.arguments ==
                        s.extract.tool_calls[i].arguments,
                    w << " arguments");
    }
  }
}
