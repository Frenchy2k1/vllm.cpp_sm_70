// Ports tests/reasoning/test_deepseekv3_reasoning_parser.py @ 555967922
// (names "deepseek_v3" + "holo2"; the thinking-gated delegate over
// DeepSeekR1ReasoningParser / IdentityReasoningParser).
//
// Upstream test_parser_selection asserts the INNER parser TYPE from
// chat_template_kwargs.thinking. The text-only seam is name-only, so we assert
// the equivalent BEHAVIOR at the two default construction points:
//   - thinking=False (name "deepseek_v3") -> Identity: passthrough, NEVER splits.
//   - thinking=True  (name "holo2")       -> R1: splits <think>…</think>.
// This is the RED-first boundary — miswiring "deepseek_v3" to R1 makes the
// identity-passthrough CHECKs fail on the <think>… input.
//
// test_deepseek_v4_reasoning_parser_alias is an ENGINE-backed adapter
// (DeepSeekV4ParserReasoningAdapter over the streaming parser engine, W3 in
// specs/reasoning-parsers.md); it is not part of this text brick — see the
// SKIP note below.
#include <doctest/doctest.h>

#include <optional>
#include <string>
#include <vector>

#include "reasoning_test_utils.h"
#include "vllm/entrypoints/openai/reasoning_parsers/abstract.h"
#include "vllm/entrypoints/openai/reasoning_parsers/deepseek_v3.h"
#include "vllm/entrypoints/openai/reasoning_parsers/identity.h"

using namespace vllm::entrypoints::openai;
using vllm::entrypoints::openai::reasoning_test::Extracted;
using vllm::entrypoints::openai::reasoning_test::RunExtraction;

namespace {
using Opt = std::optional<std::string>;
}  // namespace

TEST_CASE("deepseek_v3: registered names resolve") {
  CHECK(get_reasoning_parser("deepseek_v3") != nullptr);
  CHECK(get_reasoning_parser("holo2") != nullptr);
}

// Mirrors test_parser_selection[(True, DeepSeekR1), (False, Identity)] by
// behavior. thinking=False -> passthrough; thinking=True -> <think> split.
TEST_CASE("deepseek_v3: thinking selects the inner parser (behavior)") {
  const std::string think_in = "<think>reasoning here</think>the answer";

  // thinking=False (deepseek_v3 default) -> Identity passthrough.
  {
    auto p = get_reasoning_parser("deepseek_v3");
    REQUIRE(p != nullptr);
    const Extracted ns =
        RunExtraction(*p, {think_in}, /*streaming=*/false);
    CHECK(ns.reasoning == std::nullopt);
    CHECK(ns.content == Opt(think_in));  // whole output, unsplit
  }

  // thinking=True (holo2) -> DeepSeekR1 split.
  {
    auto p = get_reasoning_parser("holo2");
    REQUIRE(p != nullptr);
    const Extracted ns = RunExtraction(*p, {think_in}, /*streaming=*/false);
    CHECK(ns.reasoning == Opt("reasoning here"));
    CHECK(ns.content == Opt("the answer"));
  }

  // Same selection reachable via the public ctor (test_parser_selection form).
  {
    DeepSeekV3ReasoningParser off(false);
    const ExtractedReasoning r =
        off.extract_reasoning(think_in, ChatCompletionRequest{});
    CHECK(r.reasoning == std::nullopt);
    CHECK(r.content == Opt(think_in));

    DeepSeekV3ReasoningParser on(true);
    const ExtractedReasoning r2 =
        on.extract_reasoning(think_in, ChatCompletionRequest{});
    CHECK(r2.reasoning == Opt("reasoning here"));
    CHECK(r2.content == Opt("the answer"));
  }
}

// Ports test_identity_reasoning_parser_basic: (None, model_output) non-stream;
// content-wrap streaming; empty delta -> None; is_reasoning_end always True.
TEST_CASE("deepseek_v3: identity delegate passthrough contract") {
  IdentityReasoningParser id;

  // extract_reasoning -> (None, model_output).
  const ExtractedReasoning r =
      id.extract_reasoning("This is some output", ChatCompletionRequest{});
  CHECK(r.reasoning == std::nullopt);
  CHECK(r.content == Opt("This is some output"));

  // is_reasoning_end always True (gate always open).
  CHECK(id.is_reasoning_end("This is some output") == true);
  CHECK(id.is_reasoning_end("<think>still thinking") == true);

  // Streaming wraps a non-empty delta as content.
  const std::optional<DeltaMessage> d = id.extract_reasoning_streaming(
      "", "Hello world", "Hello world", ChatCompletionRequest{});
  REQUIRE(d.has_value());
  CHECK(d->content == Opt("Hello world"));
  CHECK(d->reasoning == std::nullopt);

  // Empty delta -> None (nothing to emit).
  const std::optional<DeltaMessage> none = id.extract_reasoning_streaming(
      "Hello world", "Hello world", "", ChatCompletionRequest{});
  CHECK(!none.has_value());

  // Streaming reconstruction over a <think> stream stays passthrough content.
  const Extracted acc = RunExtraction(
      id, {"<think>", "reasoning", "</think>", "answer"}, /*streaming=*/true);
  CHECK(acc.reasoning == std::nullopt);
  CHECK(acc.content == Opt("<think>reasoning</think>answer"));
}

// SKIPPED — test_deepseek_v4_reasoning_parser_alias: "deepseek_v4" is an
// engine-backed reasoning adapter (DeepSeekV4ParserReasoningAdapter over the
// TOOLS-STREAMING-PARSER engine), deferred to W3 in specs/reasoning-parsers.md.
TEST_CASE("deepseek_v3: v4 alias is engine-backed (W3)" *
          doctest::skip(true)) {
  CHECK(get_reasoning_parser("deepseek_v4") == nullptr);  // not a text parser
}
