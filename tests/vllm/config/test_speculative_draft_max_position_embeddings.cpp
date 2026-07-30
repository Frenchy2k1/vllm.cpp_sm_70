// Ported from (test-porting protocol .agents/test-porting.md):
//   * vllm/tests/config/test_speculative_draft_max_position_embeddings.py
//     @ 32e657e68 (vllm#49343 "[BugFix] eagle draft max position embeddings",
//     underlying issue #48894) — the pure-unit cases exercising
//     SpeculativeConfig._maybe_override_draft_max_position_embeddings directly:
//       test_override_raises_smaller_value    :46-52
//       test_override_keeps_sufficient_value  :55-61
//       test_override_ignores_missing_attribute :64-71
//
// The upstream helper mutates draft_hf_config.max_position_embeddings in place
// and logs when it raises the value; our port
// (SpeculativeConfig::MaybeOverrideDraftMaxPositionEmbeddings) takes the value by
// reference as std::optional<int> — nullopt mirrors upstream's `getattr(...,
// None)` "attribute missing" — and RETURNS true iff it raised the value (the
// caller emits the log line upstream does inside the helper). So the assertion
// `_override_logged(...)` maps here to "the helper returned true".
//
// The two model-integration cases upstream
// (test_eagle_draft_inherits_target_max_model_len,
//  test_independent_draft_model_keeps_its_own_limit) construct real
// ModelConfig(EAGLE3_DRAFT / AR_MODEL) + SpeculativeConfig objects, which fetch
// HF config files and build a full draft ModelConfig — machinery our T0
// SpeculativeConfig subset does not carry (draft-config resolution is deferred to
// the eagle/eagle3 model-loader claim). They are recorded as SKIPPED in
// porting-inventory.md with that tracked reason rather than watered down here;
// the eagle/eagle3 loader claim ports them as e2e/parity gates when it lands.
//
// RED-first note: with the clamp reverted to a no-op (return false; leave the
// value untouched) the "raises_smaller_value" case fails on BOTH the returned
// override flag and the mutated value — the two assertions this fix exists for.
#include <doctest/doctest.h>

#include <optional>

#include "vllm/config/speculative.h"

using vllm::SpeculativeConfig;

TEST_CASE("MaybeOverrideDraftMaxPositionEmbeddings raises a smaller value") {
  // test_override_raises_smaller_value: draft 2048 < target 8192 -> raised, log.
  std::optional<int> draft_mpe = 2048;
  const bool overridden =
      SpeculativeConfig::MaybeOverrideDraftMaxPositionEmbeddings(
          draft_mpe, /*target_max_model_len=*/8192);
  CHECK(draft_mpe.has_value());
  CHECK(*draft_mpe == 8192);
  CHECK(overridden);  // upstream: _override_logged(caplog) is True
}

TEST_CASE("MaybeOverrideDraftMaxPositionEmbeddings keeps a sufficient value") {
  // test_override_keeps_sufficient_value: draft 8192 >= target 8192 -> no change.
  std::optional<int> draft_mpe = 8192;
  const bool overridden =
      SpeculativeConfig::MaybeOverrideDraftMaxPositionEmbeddings(
          draft_mpe, /*target_max_model_len=*/8192);
  CHECK(draft_mpe.has_value());
  CHECK(*draft_mpe == 8192);
  CHECK_FALSE(overridden);  // upstream: not _override_logged(caplog)
}

TEST_CASE("MaybeOverrideDraftMaxPositionEmbeddings ignores a missing value") {
  // test_override_ignores_missing_attribute: getattr(..., None) -> leave alone.
  std::optional<int> draft_mpe = std::nullopt;
  const bool overridden =
      SpeculativeConfig::MaybeOverrideDraftMaxPositionEmbeddings(
          draft_mpe, /*target_max_model_len=*/8192);
  CHECK_FALSE(draft_mpe.has_value());  // stays absent
  CHECK_FALSE(overridden);
}

TEST_CASE("MaybeOverrideDraftMaxPositionEmbeddings only ever raises, never lowers") {
  // Guards the "never lower" invariant (a larger draft context than the target
  // is left intact), mirroring the >= short-circuit that the independent-draft
  // integration case rests on.
  std::optional<int> draft_mpe = 131072;
  const bool overridden =
      SpeculativeConfig::MaybeOverrideDraftMaxPositionEmbeddings(
          draft_mpe, /*target_max_model_len=*/8192);
  CHECK(*draft_mpe == 131072);
  CHECK_FALSE(overridden);
}
