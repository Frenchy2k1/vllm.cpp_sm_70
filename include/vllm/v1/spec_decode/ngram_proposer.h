// Ported from: vllm/v1/spec_decode/ngram_proposer.py @ 555967922 (vLLM 0.26.0.dev0)
//
// SPEC-NGRAM (ROAD-V1-D3): the draft-FREE n-gram speculative proposer. It matches
// the suffix of a sequence's own generated tokens against an earlier n-gram in the
// SAME sequence and proposes the k tokens that followed that earlier occurrence.
// There is NO draft model — the proposal is a pure host-side string match, so it
// reuses the EXACT landed verify/reject/take_draft_token_ids loop (the drafts are
// verified by the target model exactly like MTP/DFlash drafts).
//
// This header exposes the KMP-LPS longest-suffix-ngram matcher
// (_find_longest_matched_ngram_and_propose_tokens) and the per-batch propose
// (NgramProposer.propose / batch_propose). The numba multithreading of the upstream
// is a pure performance detail (single-threaded here; identical output); we mirror
// the ALGORITHM, which is what determines the proposed tokens.
#ifndef VLLM_V1_SPEC_DECODE_NGRAM_PROPOSER_H_
#define VLLM_V1_SPEC_DECODE_NGRAM_PROPOSER_H_

#include <cstdint>
#include <vector>

namespace vllm {
namespace v1 {
namespace spec_decode {

// _find_longest_matched_ngram_and_propose_tokens (ngram_proposer.py:184-267).
// Finds the longest n-gram of length in [min_ngram, max_ngram] that matches the
// suffix of `origin_tokens[:total_token]` at an EARLIER position (the earliest
// such occurrence when several tie), then returns up to `k` tokens that follow it.
// Returns an empty vector when no ngram >= min_ngram matches, when the context is
// shorter than min_ngram, or when the model length cap leaves no room.
//   origin_tokens: pointer to the request's context tokens (length total_token).
//   min_ngram, max_ngram: the prompt_lookup_min/max window (both >= 1, min<=max).
//   max_model_len: the model length cap (bounds k to max_model_len - total_token).
//   k: the desired number of draft tokens (num_speculative_tokens).
std::vector<int32_t> FindLongestMatchedNgramAndProposeTokens(
    const int32_t* origin_tokens, int total_token, int min_ngram, int max_ngram,
    int max_model_len, int k);

// The batch propose (NgramProposer.propose + batch_propose, ngram_proposer.py:
// 128-180). For each of `num_requests` rows: propose drafts only when the row
// sampled a token this step (`has_sampled[i]`) and has not reached max_model_len;
// otherwise its result is empty. Mirrors the upstream `valid_ngram_requests`
// filter + the per-request matcher call. Single-threaded (the upstream numba
// `prange` is a perf detail; the per-row output is identical).
//   token_rows[i]: pointer to request i's context (>= num_tokens_no_spec[i] long).
//   num_tokens_no_spec[i]: request i's context length (tokens without spec).
// `num_speculative_tokens` may be <= cfg-k (upstream asserts <= self.k); we clamp.
struct NgramConfig {
  int min_n = 0;
  int max_n = 0;
  int max_model_len = 0;
};

std::vector<std::vector<int32_t>> NgramPropose(
    const NgramConfig& cfg, int num_speculative_tokens,
    const std::vector<bool>& has_sampled,
    const std::vector<int32_t>& num_tokens_no_spec,
    const std::vector<const int32_t*>& token_rows);

}  // namespace spec_decode
}  // namespace v1
}  // namespace vllm

#endif  // VLLM_V1_SPEC_DECODE_NGRAM_PROPOSER_H_
