// Model-free, token-EXACT gate for the beam-search algorithm (ROAD-V1-C7
// `SAMPLE-BEAM`), ported from vllm/entrypoints/generate/beam_search/{utils,offline}
// .py @ 555967922. Beam search is DETERMINISTIC, so the scoring + top-k-beam
// selection + EOS handling + length-penalty are gated token-for-token against a
// HAND-COMPUTED beam-search tree over a fixed toy logprob table — the real
// correctness content, needing no model.
//
// The reference tree (beam_width=2, eos=0, length_penalty=1.0, prompt=[7]):
//
//  step 1 — beam [7]:  next {1:-0.1, 2:-0.5, 3:-1.0, 0(eos):-2.0}
//    candidates: [7,1] -0.1, [7,2] -0.5, [7,3] -1.0 ; [7,0] -2.0 -> completed
//    scores (sl=2): [7,1] -0.05, [7,2] -0.25, [7,3] -0.5  -> keep [7,1],[7,2]
//    completed: {[7,0] cum -2.0}
//  step 2 — beams [7,1] (-0.1), [7,2] (-0.5):
//    [7,1] next {1:-0.2, 0(eos):-0.3, 2:-1.5, 3:-2.0}
//    [7,2] next {2:-0.05, 3:-0.4, 0(eos):-1.0, 1:-3.0}
//    non-eos candidates (sl=3): [7,1,1] -0.1, [7,2,2] -0.1833, [7,2,3] -0.3,
//        [7,1,2] -0.5333, [7,1,3] -0.7, [7,2,1] -1.1667  -> keep [7,1,1],[7,2,2]
//    eos -> completed: [7,1,0] cum -0.4, [7,2,0] cum -1.5
//  final: completed + surviving beams, sorted by score desc, top 2:
//    [7,1,1] -0.1, [7,2,2] -0.1833  (beats [7,1,0] -0.2, [7,2,0] -0.75, [7,0] -2.0)
#include "vllm/entrypoints/beam_search.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

#include <string>

#include "vllm/logprobs.h"

using vllm::BeamSearchInstance;
using vllm::BeamSearchSequence;
using vllm::BeamSearchStep;
using vllm::get_beam_search_score;
using vllm::Logprob;
using vllm::LogprobsOnePosition;
using vllm::SortBeamsKey;

namespace {

// Build one position's {token_id -> Logprob} dict in the given insertion order
// (the order beam search iterates, mirroring the sampler's rank order).
LogprobsOnePosition MakeDict(
    const std::vector<std::pair<int32_t, float>>& entries) {
  LogprobsOnePosition pos;
  int rank = 1;
  for (const auto& e : entries) {
    Logprob lp{};
    lp.logprob = e.second;
    lp.rank = rank++;
    lp.decoded_token = std::string();  // fully initialize (silences GCC move warn)
    pos.put(e.first, std::move(lp));
  }
  return pos;
}

// The offline.py:178-189 tail: fold surviving beams into completed, sort by the
// length-penalty score DESCENDING (stable), take the top-beam_width.
std::vector<BeamSearchSequence> FinalTopK(BeamSearchInstance instance,
                                          std::optional<int> eos, double lp,
                                          int beam_width) {
  for (BeamSearchSequence& b : instance.beams) {
    instance.completed.push_back(std::move(b));
  }
  std::stable_sort(instance.completed.begin(), instance.completed.end(),
                   [&](const BeamSearchSequence& a, const BeamSearchSequence& b) {
                     return SortBeamsKey(a, eos, lp) > SortBeamsKey(b, eos, lp);
                   });
  if (static_cast<int>(instance.completed.size()) > beam_width) {
    instance.completed.resize(static_cast<std::size_t>(beam_width));
  }
  return instance.completed;
}

}  // namespace

// ─── 1. get_beam_search_score: the exact scoring formula (utils.py:137) ──────
TEST_CASE("beam_search: get_beam_search_score matches vLLM's formula exactly") {
  const std::optional<int> eos = 0;

  // No EOS: score = cum / seq_len^penalty, seq_len == len(tokens) (incl. prompt).
  CHECK(get_beam_search_score({7, 1, 1}, -0.3, eos, 1.0) ==
        doctest::Approx(-0.3 / 3.0));
  // Last token == EOS: seq_len decremented by one.
  CHECK(get_beam_search_score({7, 1, 0}, -0.4, eos, 1.0) ==
        doctest::Approx(-0.4 / 2.0));
  CHECK(get_beam_search_score({7, 0}, -2.0, eos, 1.0) ==
        doctest::Approx(-2.0 / 1.0));
  // length_penalty != 1.0 raises seq_len to that power.
  CHECK(get_beam_search_score({7, 1, 1}, -0.3, eos, 2.0) ==
        doctest::Approx(-0.3 / 9.0));
  CHECK(get_beam_search_score({7, 1, 0}, -0.4, eos, 2.0) ==
        doctest::Approx(-0.4 / 4.0));
  // eos_token_id == nullopt never decrements seq_len (the no-eos / ignore path).
  CHECK(get_beam_search_score({7, 1, 0}, -0.4, std::nullopt, 1.0) ==
        doctest::Approx(-0.4 / 3.0));
}

// ─── 2. Two full steps token-exact vs the hand-computed tree ─────────────────
TEST_CASE("beam_search: step expand/score/select/EOS matches the reference tree") {
  const std::optional<int> eos = 0;
  const int beam_width = 2;
  const double lp = 1.0;

  BeamSearchInstance instance(/*prompt_tokens=*/{7});
  REQUIRE(instance.beams.size() == 1);
  REQUIRE(instance.beams[0].tokens == std::vector<int32_t>{7});

  // ── step 1 ──
  LogprobsOnePosition d1 =
      MakeDict({{1, -0.1f}, {2, -0.5f}, {3, -1.0f}, {0, -2.0f}});
  const bool stop1 = BeamSearchStep(instance, {&d1}, eos, /*ignore_eos=*/false,
                                    beam_width, lp);
  CHECK_FALSE(stop1);

  // Top-2 non-eos continuations, best score first.
  REQUIRE(instance.beams.size() == 2);
  CHECK(instance.beams[0].tokens == std::vector<int32_t>{7, 1});
  CHECK(instance.beams[0].cum_logprob == doctest::Approx(-0.1));
  CHECK(instance.beams[1].tokens == std::vector<int32_t>{7, 2});
  CHECK(instance.beams[1].cum_logprob == doctest::Approx(-0.5));
  // [7,3] fell outside beam_width; [7,0] retired to completed.
  REQUIRE(instance.completed.size() == 1);
  CHECK(instance.completed[0].tokens == std::vector<int32_t>{7, 0});
  CHECK(instance.completed[0].cum_logprob == doctest::Approx(-2.0));

  // ── step 2 ──
  LogprobsOnePosition d2a =
      MakeDict({{1, -0.2f}, {0, -0.3f}, {2, -1.5f}, {3, -2.0f}});  // beam [7,1]
  LogprobsOnePosition d2b =
      MakeDict({{2, -0.05f}, {3, -0.4f}, {0, -1.0f}, {1, -3.0f}});  // beam [7,2]
  const bool stop2 = BeamSearchStep(instance, {&d2a, &d2b}, eos,
                                    /*ignore_eos=*/false, beam_width, lp);
  CHECK_FALSE(stop2);

  // Top-2 by score: [7,1,1] (-0.1) then [7,2,2] (-0.1833).
  REQUIRE(instance.beams.size() == 2);
  CHECK(instance.beams[0].tokens == std::vector<int32_t>{7, 1, 1});
  CHECK(instance.beams[0].cum_logprob == doctest::Approx(-0.3));
  CHECK(instance.beams[1].tokens == std::vector<int32_t>{7, 2, 2});
  CHECK(instance.beams[1].cum_logprob == doctest::Approx(-0.55));

  // completed now: [7,0] then the two step-2 EOS beams in beam order.
  REQUIRE(instance.completed.size() == 3);
  CHECK(instance.completed[0].tokens == std::vector<int32_t>{7, 0});
  CHECK(instance.completed[1].tokens == std::vector<int32_t>{7, 1, 0});
  CHECK(instance.completed[1].cum_logprob == doctest::Approx(-0.4));
  CHECK(instance.completed[2].tokens == std::vector<int32_t>{7, 2, 0});
  CHECK(instance.completed[2].cum_logprob == doctest::Approx(-1.5));

  // ── final top-k (offline.py:178-189 tail) ──
  std::vector<BeamSearchSequence> best =
      FinalTopK(instance, eos, lp, beam_width);
  REQUIRE(best.size() == 2);
  CHECK(best[0].tokens == std::vector<int32_t>{7, 1, 1});  // score -0.1
  CHECK(best[1].tokens == std::vector<int32_t>{7, 2, 2});  // score -0.1833
  // Descending by score.
  CHECK(SortBeamsKey(best[0], eos, lp) > SortBeamsKey(best[1], eos, lp));
}

// ─── 3. ignore_eos keeps an EOS token as a normal beam ───────────────────────
TEST_CASE("beam_search: ignore_eos does not retire EOS beams") {
  const std::optional<int> eos = 0;
  BeamSearchInstance instance(/*prompt_tokens=*/{7});
  LogprobsOnePosition d = MakeDict({{0, -0.1f}, {1, -0.5f}, {2, -1.0f}, {3, -2.0f}});
  const bool stop =
      BeamSearchStep(instance, {&d}, eos, /*ignore_eos=*/true, /*beam_width=*/2, 1.0);
  CHECK_FALSE(stop);
  // With ignore_eos, the eos token (id 0) competes as a normal beam and, being
  // the highest-logprob token, is kept — nothing retired to completed.
  CHECK(instance.completed.empty());
  REQUIRE(instance.beams.size() == 2);
  CHECK(instance.beams[0].tokens == std::vector<int32_t>{7, 0});
  CHECK(instance.beams[1].tokens == std::vector<int32_t>{7, 1});
}

// ─── 4. A null per-beam dict (completed/aborted sequence) contributes nothing ─
TEST_CASE("beam_search: a null logprob dict yields no continuations for that beam") {
  const std::optional<int> eos = 0;
  BeamSearchInstance instance(/*prompt_tokens=*/{7});
  // Two active beams: expand the first one, then set up a step where beam[0] has a
  // dict and beam[1] is null.
  LogprobsOnePosition d1 = MakeDict({{1, -0.1f}, {2, -0.5f}, {3, -1.0f}, {5, -1.2f}});
  BeamSearchStep(instance, {&d1}, eos, false, /*beam_width=*/2, 1.0);
  REQUIRE(instance.beams.size() == 2);  // [7,1], [7,2]

  LogprobsOnePosition d2 = MakeDict({{4, -0.2f}, {5, -0.4f}, {6, -0.6f}, {8, -0.9f}});
  BeamSearchStep(instance, {&d2, /*null for beam[1]*/ nullptr}, eos, false,
                 /*beam_width=*/2, 1.0);
  // Only beam[0]'s continuations survive; beam[1] (null) produced none.
  REQUIRE(instance.beams.size() == 2);
  for (const BeamSearchSequence& b : instance.beams) {
    REQUIRE(b.tokens.size() == 3);
    CHECK(b.tokens[1] == 1);  // all descend from [7,1]
  }
}

// ─── 5. An exhausted instance (no beams) stops ───────────────────────────────
TEST_CASE("beam_search: step over an empty beam set returns stop") {
  const std::optional<int> eos = 0;
  BeamSearchInstance instance(/*prompt_tokens=*/{7});
  instance.beams.clear();
  CHECK(BeamSearchStep(instance, {}, eos, false, 2, 1.0));
}
