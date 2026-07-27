// Ported from: vllm/logprobs.py @ 555967922 (vLLM 0.26.0.dev0)
//
// The public logprobs value types the OpenAI server serializes: `Logprob`
// (one {logprob, rank, decoded_token} entry), `LogprobsOnePosition`
// (dict[int, Logprob] — one position's alternatives), and the per-request
// `SampleLogprobs` / `PromptLogprobs` sequences. Field names + semantics mirror
// upstream 1:1.
//
// DEVIATIONS, recorded:
//   - `LogprobsOnePosition` upstream is a Python `dict[int, Logprob]` whose
//     ITERATION ORDER (insertion order) is load-bearing for the OpenAI
//     serialization (the sampled token is inserted first, then top-k in rank
//     order, and `enumerate(step.items())` slices by that order). We mirror the
//     dict with an explicit `order` vector plus the `token_id -> Logprob` map,
//     and `put()` reproduces Python-dict semantics: a repeated key keeps its
//     FIRST position but takes the LAST written value (dict-comprehension
//     last-wins), exactly as `append_logprobs_for_next_position`'s
//     `{token_id: Logprob(...) for ...}` does.
//   - We port only the NON-flat representation (`list[dict]`); upstream's
//     `FlatLogprobs` GC-optimization is an equivalent alternate container
//     selected by `SamplingParams.flat_logprobs` (which our Sampler never sets),
//     so it is intentionally omitted (add it behind a flag if the flat path is
//     ever wired).
#ifndef VLLM_LOGPROBS_H_
#define VLLM_LOGPROBS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace vllm {

// Logprob (vllm/logprobs.py:12-24): infos for OpenAI-compatible logprobs and
// token ranks. `rank` is the 1-based vocab rank (>=1); `decoded_token` is the
// detokenized string (None when detokenization is disabled).
struct Logprob {
  float logprob = 0.0f;
  std::optional<int> rank;
  std::optional<std::string> decoded_token;
};

// LogprobsOnePosition (vllm/logprobs.py:27 `LogprobsOnePosition =
// dict[int, Logprob]`): one position's {token_id -> Logprob}, iterated in
// insertion order.
struct LogprobsOnePosition {
  // token ids in insertion order (the dict key order).
  std::vector<int32_t> order;
  std::unordered_map<int32_t, Logprob> entries;

  // Python dict semantics: overwrite the value, but keep the FIRST insertion
  // position for a repeated key (matches the sampled-token-in-top-k case).
  void put(int32_t token_id, Logprob lp) {
    auto res = entries.insert_or_assign(token_id, std::move(lp));
    if (res.second) order.push_back(token_id);  // newly inserted key
  }
  bool empty() const { return order.empty(); }
  std::size_t size() const { return order.size(); }
  const Logprob* find(int32_t token_id) const {
    auto it = entries.find(token_id);
    return it == entries.end() ? nullptr : &it->second;
  }
};

// {token_id -> logprob} per generated token (sample) / prompt token (prompt).
// PromptLogprobs' first entry is None (the first prompt token has no logprob).
using SampleLogprobs = std::vector<LogprobsOnePosition>;
using PromptLogprobs = std::vector<std::optional<LogprobsOnePosition>>;

// append_logprobs_for_next_position (vllm/logprobs.py:175-206): build one
// position's Logprob dict from the sampler's row and append it to
// `request_logprobs`. `token_ids`/`logprobs`/`decoded_tokens` are the
// [sampled | top-k] row (length k+1); `rank` is the sampled token's rank; the
// top-k get ranks 1..k. `num_logprobs == -1` means "all" (k := len-1).
inline void AppendLogprobsForNextPosition(
    SampleLogprobs& request_logprobs, const std::vector<int32_t>& token_ids,
    const std::vector<float>& logprobs,
    const std::vector<std::optional<std::string>>& decoded_tokens, int rank,
    int num_logprobs) {
  int k = num_logprobs == -1 ? static_cast<int>(logprobs.size()) - 1 : num_logprobs;
  if (k < 0) k = 0;
  LogprobsOnePosition pos;
  // ranks = chain((rank,), range(1, num_logprobs+1)): sampled first, then top-k.
  const std::size_t n = token_ids.size();
  for (std::size_t i = 0; i < n; ++i) {
    const int this_rank = (i == 0) ? rank : static_cast<int>(i);
    Logprob lp;
    lp.logprob = logprobs[i];
    lp.rank = this_rank;
    if (i < decoded_tokens.size()) lp.decoded_token = decoded_tokens[i];
    pos.put(token_ids[i], std::move(lp));
    if (static_cast<int>(i) >= k) break;  // sampled (i==0) + k top-k entries
  }
  request_logprobs.push_back(std::move(pos));
}

}  // namespace vllm

#endif  // VLLM_LOGPROBS_H_
