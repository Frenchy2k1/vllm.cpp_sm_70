// CUDA-graph batch dispatch: which batches are graph-capturable, and under what
// shape key (SPEC-DSPARK W8, issue #442).
//
// Ported from vLLM @ 555967922:
//   v1/worker/gpu/cudagraph_utils.py:95-105  get_uniform_token_count
//   v1/cudagraph_dispatcher.py:37            uniform_decode_query_len
//   v1/cudagraph_dispatcher.py:143-146       the FULL-graph branch
//
// WHY THIS EXISTS. We mirrored vLLM's MODEL but not its graph DISPATCHER, and
// that single divergence is the measured DSpark parity gap. Upstream's "uniform"
// test is that every request in the batch shares a query_len -- NOT that the
// query_len is 1 -- and its captured decode length is DEFINED as the speculative
// verify shape (`1 + num_speculative_tokens`). So vLLM graphs the T=1+k verify by
// construction. Our gate is `pure_decode == (num_actual_tokens == num_reqs)`
// (qwen3_5_dense.cpp:159, qwen3_5_moe.cpp:128), which only ever matches
// query_len == 1, so our verify runs EAGER every step.
//
// Measured cost of that divergence (.agents/specs/dspark-spec-decode.md §6l/§6m):
// the 35B lane sits at 0.870x of the pinned graphed oracle where acceptance is
// high and the verify therefore runs every step, and 0.981x where acceptance is
// low. Our verify is ~32.0 ms of a 36.6 ms step; the oracle's ENTIRE step is
// ~31.8 ms.
//
// This header is the FIRST slice of the port and is deliberately INERT: pure
// shape arithmetic with no capture, no allocation and no caller yet, so it
// cannot change behavior. Wiring it (descriptor-keyed captures that route the
// spec attention + GDN paths for a 1+k span) is the rest of #442. Flipping a
// predicate ALONE would be a bug: it would send a spec batch through a graph
// captured for query_len == 1.
#ifndef VLLM_V1_WORKER_GPU_CUDAGRAPH_DISPATCH_H_
#define VLLM_V1_WORKER_GPU_CUDAGRAPH_DISPATCH_H_

#include <cstdint>
#include <optional>

namespace vllm {
namespace v1 {

// The query length a captured "decode" graph is built for.
//
// Upstream: `self.uniform_decode_query_len = 1 + num_speculative_tokens`
// (cudagraph_dispatcher.py:37). With speculation OFF this is 1, which is exactly
// today's pure-decode shape, so adopting it is a no-op for the non-spec path --
// that is what makes the eventual wiring safe to land incrementally.
inline int64_t UniformDecodeQueryLen(int64_t num_speculative_tokens) {
  return 1 + (num_speculative_tokens > 0 ? num_speculative_tokens : 0);
}

// The uniform token count of a batch, or nullopt when the batch is not uniform.
//
// Upstream get_uniform_token_count (cudagraph_utils.py:95-105): a batch is
// uniform iff every request carries the same query_len, tested as
// `max_query_len == num_tokens // num_reqs && num_tokens == max_query_len * num_reqs`.
// The second clause is not redundant -- integer division alone accepts a ragged
// batch whose total happens to floor to max_query_len (e.g. 2 reqs, lens 3 and 2,
// max 3, 5 // 2 == 2 != 3 rejects; but lens 4 and 2 with max 3 would floor to 3
// and pass without it).
inline std::optional<int64_t> UniformTokenCount(int64_t num_reqs, int64_t num_tokens,
                                                int64_t max_query_len) {
  if (num_reqs <= 0 || num_tokens <= 0 || max_query_len <= 0) return std::nullopt;
  if (max_query_len == num_tokens / num_reqs && num_tokens == max_query_len * num_reqs) {
    return max_query_len;
  }
  return std::nullopt;
}

// Whether this batch is a uniform DECODE batch for the configured speculation
// width, i.e. the shape a FULL graph is captured for.
//
// Upstream takes the FULL-graph branch when the batch is uniform and the mode
// offers FULL (cudagraph_dispatcher.py:143). `num_speculative_tokens == 0` makes
// this exactly today's pure-decode predicate; k > 0 additionally admits the
// T = num_reqs * (1 + k) speculative VERIFY, which is the batch we currently run
// eager and upstream captures.
inline bool IsUniformDecodeBatch(int64_t num_reqs, int64_t num_tokens,
                                 int64_t max_query_len,
                                 int64_t num_speculative_tokens) {
  const std::optional<int64_t> uniform =
      UniformTokenCount(num_reqs, num_tokens, max_query_len);
  return uniform.has_value() &&
         *uniform == UniformDecodeQueryLen(num_speculative_tokens);
}

// The request count a captured graph of `num_tokens_padded` tokens serves.
//
// Upstream: `num_reqs = min(num_tokens_padded // uniform_decode_query_len,
// max_num_seqs)` with `assert num_tokens_padded % uniform_decode_query_len == 0`
// (cudagraph_dispatcher.py:144-145). Returns nullopt where upstream would trip
// that assert, so a caller cannot silently capture a shape the padding never
// produces.
inline std::optional<int64_t> UniformDecodeNumReqs(int64_t num_tokens_padded,
                                                   int64_t num_speculative_tokens,
                                                   int64_t max_num_seqs) {
  const int64_t q = UniformDecodeQueryLen(num_speculative_tokens);
  if (num_tokens_padded <= 0 || max_num_seqs <= 0) return std::nullopt;
  if (num_tokens_padded % q != 0) return std::nullopt;
  const int64_t reqs = num_tokens_padded / q;
  return reqs < max_num_seqs ? reqs : max_num_seqs;
}

}  // namespace v1
}  // namespace vllm

#endif  // VLLM_V1_WORKER_GPU_CUDAGRAPH_DISPATCH_H_
