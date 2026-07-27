// Exact-parity gate for the unified streaming parser engine.
//
// Ported/gated against vLLM 0.26.0.dev0 (@ 555967922):
//   vllm/parser/engine/streaming_parser_engine.py (StreamingParserEngine),
//   vllm/parser/engine/token_id_scanner.py, vllm/parser/engine/incremental_lexer.py,
//   driven by the vllm/parser/qwen3.py + vllm/parser/kimi_k2.py configs.
//
// A parser engine is a PURE FUNCTION of the (delta_text, delta_token_ids)
// stream -> SemanticEvent sequence, so it is gated EXACTLY: the C++ engine's
// emitted event sequence must equal, event-for-event, the sequence the vLLM
// 0.26 Python engine emits for the identical delta stream. The expected
// sequences in the .inc were captured directly from the upstream engine (see
// its provenance header). RED-first: a wrong tool-call boundary, a dropped
// argument delta, or a mis-split reasoning chunk changes the sequence and fails.
//
// Coverage: reasoning -> content -> XML tool call (qwen3, tool_args_json=false);
// char-by-char vs whole-string cadence (streaming stability); thinking-disabled
// plain content; two consecutive tool calls (tool_index increment); an
// unfinished tool call flushed by finish(); JSON argument streaming with the
// held-back top-level brace (kimi_k2, tool_args_json=true); and a token-ID
// driven stream where special terminals arrive via delta_token_ids through the
// TokenIDScanner + a mock vocab.
#include <doctest/doctest.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "vllm/parser/engine/configs.h"
#include "vllm/parser/engine/events.h"
#include "vllm/parser/engine/registry.h"
#include "vllm/parser/engine/streaming_parser_engine.h"

using namespace vllm::parser::engine;
using E = EventType;

namespace {

struct ExpectedEvent {
  EventType type;
  std::string value;
  int tool_index;
};

struct Scenario {
  std::string name;
  std::string cfg_tag;
  bool use_tokenizer;
  std::vector<std::pair<std::string, std::vector<int>>> deltas;
  std::vector<ExpectedEvent> expected;
};

// Fixed mock tokenizer for the token-ID scenario: single-token decode + a
// vocab whose keys are the terminal token texts. Exposes no special-token
// table (all_special returns false), so the drop machinery stays off — exactly
// mirroring the Python MockTok used to capture the goldens.
class MockTokenizer : public EngineTokenizer {
 public:
  MockTokenizer() {
    vocab_ = {
        {"<think>", 1001},
        {"</think>", 1002},
        {"<|tool_calls_section_begin|>", 1003},
        {"<|tool_calls_section_end|>", 1004},
        {"<|tool_call_begin|>", 1005},
        {"<|tool_call_end|>", 1006},
        {"<|tool_call_argument_begin|>", 1007},
    };
    for (const auto& [text, id] : vocab_) rev_[id] = text;
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

ParserEngineConfig build_config(const std::string& tag) {
  if (tag == "qwen3_think") return qwen3_config(true);
  if (tag == "qwen3_nothink") return qwen3_config(false);
  if (tag == "kimi_think") return kimi_k2_config(true);
  FAIL("unknown config tag: " << tag);
  return {};
}

std::vector<Scenario> make_scenarios() {
  std::vector<Scenario> scenarios;
#include "test_streaming_parser_engine_goldens.inc"
  return scenarios;
}

}  // namespace

TEST_CASE("streaming parser engine matches vLLM 0.26 event stream exactly") {
  MockTokenizer mock;
  for (const Scenario& s : make_scenarios()) {
    CAPTURE(s.name);
    ParserEngineConfig cfg = build_config(s.cfg_tag);
    const EngineTokenizer* tok = s.use_tokenizer ? &mock : nullptr;
    StreamingParserEngine engine(std::move(cfg), tok);

    std::vector<SemanticEvent> got;
    for (const auto& [dt, ids] : s.deltas) {
      auto ev = engine.feed(dt, ids);
      got.insert(got.end(), ev.begin(), ev.end());
    }
    auto fin = engine.finish();
    got.insert(got.end(), fin.begin(), fin.end());

    REQUIRE(got.size() == s.expected.size());
    for (size_t i = 0; i < got.size(); ++i) {
      CAPTURE(i);
      CAPTURE(event_type_name(got[i].type));
      CAPTURE(event_type_name(s.expected[i].type));
      CAPTURE(got[i].value);
      CAPTURE(s.expected[i].value);
      CHECK(got[i].type == s.expected[i].type);
      CHECK(got[i].value == s.expected[i].value);
      CHECK(got[i].tool_index == s.expected[i].tool_index);
    }
  }
}

// The unified registry binds engine-backed format names to their declarative
// config (registered_adapters.py). seed_oss reuses the qwen3 grammar with
// different wrapper tokens (seed_oss.py: SeedOssParser(Qwen3Parser)).
TEST_CASE("engine registry exposes the engine-backed families") {
  CHECK(is_engine_backed("qwen3"));
  CHECK(is_engine_backed("seed_oss"));
  CHECK(is_engine_backed("kimi_k2"));
  CHECK_FALSE(is_engine_backed("hermes"));  // legacy per-family parser, not engine-backed
  CHECK_FALSE(get_engine_config("does_not_exist").has_value());

  auto seed = get_engine_config("seed_oss");
  REQUIRE(seed.has_value());
  CHECK(seed->terminals.at("THINK_START") == "<seed:think>");
  CHECK(seed->terminals.at("TOOL_START") == "<seed:tool_call>");
  CHECK(seed->tool_args_json == false);

  auto kimi = get_engine_config("kimi_k2");
  REQUIRE(kimi.has_value());
  CHECK(kimi->tool_args_json == true);
}
