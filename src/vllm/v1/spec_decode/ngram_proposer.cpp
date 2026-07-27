// Ported from: vllm/v1/spec_decode/ngram_proposer.py @ 555967922 (vLLM 0.26.0.dev0)
//   _find_longest_matched_ngram_and_propose_tokens  -> FindLongestMatchedNgramAndProposeTokens
//   NgramProposer.propose / batch_propose           -> NgramPropose
//
// SPEC-NGRAM (ROAD-V1-D3). Draft-free proposer; see the header for the contract.
#include "vllm/v1/spec_decode/ngram_proposer.h"

#include <algorithm>

namespace vllm {
namespace v1 {
namespace spec_decode {

std::vector<int32_t> FindLongestMatchedNgramAndProposeTokens(
    const int32_t* origin_tokens, int total_token, int min_ngram, int max_ngram,
    int max_model_len, int k) {
  // Do not generate draft tokens if the context is shorter than the minimum
  // n-gram (ngram_proposer.py:198-200).
  if (total_token < min_ngram) {
    return {};
  }
  // Do not generate draft tokens beyond the max model length (:202-204).
  k = std::min(k, max_model_len - total_token);
  if (k <= 0) {
    return {};
  }

  // Flip tokens: the goal becomes finding the longest prefix of the reversed
  // sequence (i.e. the suffix of the original) that matches an earlier position
  // (:206-214). tok(j) indexes the reversed array without materializing it:
  //   tokens[j] == origin_tokens[total_token - 1 - j].
  const auto tok = [&](int j) -> int32_t {
    return origin_tokens[total_token - 1 - j];
  };

  // Longest-prefix-suffix table over the first max_ngram reversed prefixes only
  // (capped to bound memory + the ngram length; :216-222).
  std::vector<int32_t> lps(static_cast<size_t>(max_ngram), 0);

  int longest_ngram = 0;
  int position = 0;

  // lps[0] is always 0; start at index 1 (:224-227).
  int prev_lps = 0;
  int i = 1;
  while (i < total_token) {
    if (tok(prev_lps) == tok(i)) {
      // Token match: tokens[:prev_lps+1] is the longest prefix that is a suffix
      // of tokens[:i+1] (:229-235).
      prev_lps += 1;
      // Update position when longest_ngram matches prev_lps so we keep the
      // EARLIEST occurrence in the original tokens (latest in the reversed
      // tokens) (:237-247).
      if (prev_lps >= longest_ngram) {
        longest_ngram = prev_lps;
        position = i;
      }
      if (i < max_ngram) {
        lps[static_cast<size_t>(i)] = prev_lps;
      }
      if (prev_lps == max_ngram) {
        // Cap the match at max_ngram: fall back to lps[max_ngram-1] (:248-253).
        prev_lps = lps[static_cast<size_t>(max_ngram - 1)];
      }
      i += 1;
    } else if (prev_lps != 0) {
      // Token mismatch: try the next-longest prefix (:255-259).
      prev_lps = lps[static_cast<size_t>(prev_lps - 1)];
    } else {
      // Mismatch with no remaining prefix (:261-263).
      i += 1;
    }
  }

  if (longest_ngram < min_ngram) {
    // No valid ngram found (:265-267).
    return {};
  }

  // Flip the position back to the original tokens: the matched ngram starts at
  // total_token-1-position, so drafting begins right after it (:269-276).
  const int start_position = total_token - 1 - position + longest_ngram;
  k = std::min(k, total_token - start_position);
  std::vector<int32_t> out;
  out.reserve(static_cast<size_t>(k));
  for (int j = 0; j < k; ++j) {
    out.push_back(origin_tokens[start_position + j]);
  }
  return out;
}

std::vector<std::vector<int32_t>> NgramPropose(
    const NgramConfig& cfg, int num_speculative_tokens,
    const std::vector<bool>& has_sampled,
    const std::vector<int32_t>& num_tokens_no_spec,
    const std::vector<const int32_t*>& token_rows) {
  const int num_requests = static_cast<int>(has_sampled.size());
  std::vector<std::vector<int32_t>> draft_token_ids(
      static_cast<size_t>(num_requests));

  for (int i = 0; i < num_requests; ++i) {
    // valid_ngram_requests filter (ngram_proposer.py:157-172): skip a request
    // that sampled no token this step (still prefilling / discarded) or that has
    // already reached the max model length.
    if (!has_sampled[static_cast<size_t>(i)]) {
      continue;
    }
    const int num_tokens = num_tokens_no_spec[static_cast<size_t>(i)];
    if (num_tokens >= cfg.max_model_len) {
      continue;
    }
    draft_token_ids[static_cast<size_t>(i)] =
        FindLongestMatchedNgramAndProposeTokens(
            token_rows[static_cast<size_t>(i)], num_tokens, cfg.min_n, cfg.max_n,
            cfg.max_model_len, num_speculative_tokens);
  }
  return draft_token_ids;
}

}  // namespace spec_decode
}  // namespace v1
}  // namespace vllm
