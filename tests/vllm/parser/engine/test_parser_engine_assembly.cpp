// Exact-parity gate for the unified parser ASSEMBLY layer (ROAD-V1-C8
// TOOLS-STREAMING-PARSER-ASSEMBLY).
//
// Ported/gated against vLLM 0.26.0.dev0 (@ 555967922):
//   vllm/parser/engine/parser_engine.py (ParserEngine — event -> DeltaMessage
//   assembly + one-shot extract_tool_calls), vllm/parser/qwen3.py,
//   vllm/parser/seed_oss.py, vllm/parser/kimi_k2.py, and the ROAD-V1-C8 engine
//   families vllm/parser/{minimax_m2,glm47_moe,deepseek_v4,deepseek_v32,
//   nemotron_v3}.py.
//
// ROAD-V1-C8 coverage (goldens 10-19, whole-delta AND char-by-char per family):
//   minimax_m2 (<invoke>/<parameter> XML, held-back JSON arg diffs),
//   glm47_moe (<arg_key>/<arg_value>, function-name .strip() via the
//   Glm47MoeParser hook), deepseek_v4 (<think> + DSML tool_calls, string=
//   "true|false" typed-value coercion), deepseek_v32 (DSML function_calls,
//   no reasoning), nemotron_v3 (qwen3 grammar, strip_trailing_reasoning).
//   gemma4 + inkling are DEFERRED (need unported assembly hooks — see
//   .agents/specs/parser-assembly-c8.md).
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

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
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
// Non-streaming parse() golden: (reasoning, content, tool_calls). The parse()
// tuple carries FunctionCall{name, arguments} (no id).
struct GParse {
  std::optional<std::string> reasoning;
  std::optional<std::string> content;
  bool has_tool_calls;
  std::vector<std::pair<std::string, std::string>> tool_calls;  // {name, args}
};
struct GScenario {
  std::string name;
  std::string cfg;
  bool thinking;
  bool include_reasoning;
  std::vector<std::string> deltas;
  std::vector<GDelta> stream;
  GExtract extract;
  bool check_parse;
  GParse parse;
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

// gemma4's _preprocess_feed only injects the <|channel> opener when the channel
// markers resolve to non-None token ids, so gemma4 needs a tokenizer whose vocab
// carries them. Mirrors the Python MockTok vocab used to capture the goldens
// (only non-None-ness matters; the ids are never fed — gate streams carry none).
class GemmaMockTok : public vllm::parser::engine::EngineTokenizer {
 public:
  GemmaMockTok() {
    vocab_ = {{"<|channel>", 1040},
              {"<channel|>", 1041},
              {"<|tool_call>", 1042},
              {"<tool_call|>", 1043}};
    for (const auto& [t, i] : vocab_) rev_[i] = t;
  }
  std::string decode(int token_id) const override {
    auto it = rev_.find(token_id);
    return it == rev_.end() ? std::string() : it->second;
  }
  const std::map<std::string, int>& get_vocab() const override { return vocab_; }

 private:
  std::map<std::string, int> vocab_;
  std::map<int, std::string> rev_;
};

std::unique_ptr<vllm::parser::engine::ParserEngine> make(const GScenario& s) {
  static const GemmaMockTok gemma_tok;
  const vllm::parser::engine::EngineTokenizer* tok =
      (s.cfg == "gemma4") ? &gemma_tok : nullptr;
  auto p = get_parser_engine(s.cfg, s.thinking, tok);
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

// Non-streaming parse() parity. Checked for the gemma4/inkling scenarios only
// (check_parse) — this is the ONLY path that reaches inkling's _single_pass_parse
// trailing-text flush (extract_tool_calls flushes via finish_streaming instead).
TEST_CASE("parser assembly: non-streaming parse() parity (gemma4/inkling seams)") {
  for (const GScenario& s : kAssemblyGoldens) {
    if (!s.check_parse) continue;
    CAPTURE(s.name);
    auto parser = make(s);
    ParserRequest req;
    req.include_reasoning = s.include_reasoning;
    std::string full;
    for (const auto& d : s.deltas) full += d;

    auto [reasoning, content, tool_calls] = parser->parse(full, req);

    const std::string where = s.name + " parse";
    CHECK_MESSAGE(reasoning == s.parse.reasoning, where << " reasoning");
    CHECK_MESSAGE(content == s.parse.content, where << " content");
    CHECK_MESSAGE(tool_calls.has_value() == s.parse.has_tool_calls,
                  where << " has_tool_calls");
    if (tool_calls.has_value() != s.parse.has_tool_calls) continue;
    if (!tool_calls.has_value()) continue;
    CHECK_MESSAGE(tool_calls->size() == s.parse.tool_calls.size(),
                  where << " tool_calls size");
    if (tool_calls->size() != s.parse.tool_calls.size()) continue;
    for (std::size_t i = 0; i < tool_calls->size(); ++i) {
      const std::string w = where + " tc[" + std::to_string(i) + "]";
      CHECK_MESSAGE((*tool_calls)[i].name == s.parse.tool_calls[i].first,
                    w << " name");
      CHECK_MESSAGE((*tool_calls)[i].arguments == s.parse.tool_calls[i].second,
                    w << " arguments");
    }
  }
}
