// Ported from: vllm/tests/v1/spec_decode/test_ngram.py @ 555967922 (vLLM 0.26.0.dev0)
//   test_find_longest_matched_ngram_and_propose_tokens -> the matcher cases
//   test_ngram_proposer                                -> the batch-propose cases
//
// SPEC-NGRAM (ROAD-V1-D3). Host-side; runs everywhere (no checkpoint / GPU). The
// numba multithreading of the upstream is a pure perf detail; only the per-row
// proposed tokens are the spec, and those are what this gate checks 1:1.
#include <doctest/doctest.h>

#include <cstdint>
#include <vector>

#include "vllm/v1/spec_decode/ngram_proposer.h"

using vllm::v1::spec_decode::FindLongestMatchedNgramAndProposeTokens;
using vllm::v1::spec_decode::NgramConfig;
using vllm::v1::spec_decode::NgramPropose;

namespace {

std::vector<int32_t> Match(const std::vector<int32_t>& tokens, int min_n,
                           int max_n, int max_model_len, int k) {
  return FindLongestMatchedNgramAndProposeTokens(
      tokens.data(), static_cast<int>(tokens.size()), min_n, max_n, max_model_len,
      k);
}

}  // namespace

TEST_CASE("ngram: _find_longest_matched_ngram_and_propose_tokens (test_ngram.py:16-64)") {
  // No match: the suffix 2-gram [6,5]... no earlier occurrence.
  {
    std::vector<int32_t> t = {1, 2, 3, 4, 1, 2, 3, 5, 6};
    CHECK(Match(t, 2, 2, 1024, 2).empty());
  }

  // Suffix [2,3] matched earlier at index 1; propose the k that followed.
  {
    std::vector<int32_t> t = {1, 2, 3, 4, 1, 2, 3};
    CHECK(Match(t, 2, 2, 1024, 3) == std::vector<int32_t>{4, 1, 2});
    CHECK(Match(t, 2, 2, 1024, 2) == std::vector<int32_t>{4, 1});
    CHECK(Match(t, 1, 1, 1024, 3) == std::vector<int32_t>{4, 1, 2});
    CHECK(Match(t, 1, 1, 1024, 2) == std::vector<int32_t>{4, 1});
  }

  {
    std::vector<int32_t> t = {1, 3, 6, 2, 3, 4, 1, 2, 3};
    CHECK(Match(t, 2, 2, 1024, 3) == std::vector<int32_t>{4, 1, 2});
    // Return on the FIRST (earliest) match: 1-gram [3] first occurs at index 1,
    // followed by [6,2].
    CHECK(Match(t, 1, 1, 1024, 2) == std::vector<int32_t>{6, 2});
  }
}

TEST_CASE("ngram: NgramProposer.propose batch cases (test_ngram.py:67-166)") {
  auto propose = [](int min_n, int max_n, int k, int max_model_len,
                    const std::vector<bool>& has_sampled,
                    const std::vector<int32_t>& ntns,
                    const std::vector<std::vector<int32_t>>& rows) {
    NgramConfig cfg;
    cfg.min_n = min_n;
    cfg.max_n = max_n;
    cfg.max_model_len = max_model_len;
    std::vector<const int32_t*> ptrs;
    ptrs.reserve(rows.size());
    for (const auto& r : rows) ptrs.push_back(r.data());
    return NgramPropose(cfg, k, has_sampled, ntns, ptrs);
  };

  // No match.
  {
    auto r = propose(2, 2, 2, 1024, {true}, {5}, {{1, 2, 3, 4, 5}});
    CHECK(r[0].empty());
  }
  // No match for 4-gram.
  {
    auto r = propose(4, 4, 2, 1024, {true}, {7}, {{1, 2, 3, 4, 1, 2, 3}});
    CHECK(r[0].empty());
  }
  // No 4-gram but a 3-gram match.
  {
    auto r = propose(3, 4, 2, 1024, {true}, {7}, {{1, 2, 3, 4, 1, 2, 3}});
    CHECK(r[0] == std::vector<int32_t>{4, 1});
  }
  // Both 4-gram and 3-gram match -> the 4-gram wins ([1,2], not [5,1]).
  {
    auto r = propose(3, 4, 2, 1024, {true}, {12},
                     {{2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4}});
    CHECK(r[0] == std::vector<int32_t>{1, 2});
  }
  // 2-gram + 3-gram but not 4-gram -> [1,2], not [5,2].
  {
    auto r = propose(2, 4, 2, 1024, {true}, {10},
                     {{3, 4, 5, 2, 3, 4, 1, 2, 3, 4}});
    CHECK(r[0] == std::vector<int32_t>{1, 2});
  }
  // Multiple 3-gram matches -> always the first ([100,1]).
  {
    auto r = propose(3, 3, 2, 1024, {true}, {15},
                     {{1, 2, 3, 100, 1, 2, 3, 200, 1, 2, 3, 300, 1, 2, 3}});
    CHECK(r[0] == std::vector<int32_t>{100, 1});
  }
  // Multibatch: req0 (5 tokens) matches, req1 (3 tokens) no match.
  {
    auto r = propose(2, 2, 2, 1024, {true, true}, {5, 3},
                     {{1, 2, 3, 1, 2}, {4, 5, 6}});
    CHECK(r[0] == std::vector<int32_t>{3, 1});
    CHECK(r[1].empty());
  }
  // Non-contiguous: req1 is in prefill (has_sampled=false) -> empty; req0/req2
  // both propose from their own suffix.
  {
    auto r = propose(2, 2, 2, 20, {true, false, true}, {5, 3, 5},
                     {{1, 2, 3, 1, 2}, {4, 5, 6}, {7, 8, 9, 7, 8}});
    REQUIRE(r.size() == 3u);
    CHECK(r[0] == std::vector<int32_t>{3, 1});
    CHECK(r[1].empty());
    CHECK(r[2] == std::vector<int32_t>{9, 7});
  }
}
