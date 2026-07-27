// ENG-SGLANG-BEHAVIOR-FLAG (SW1) — cache-aware LPM admission ordering.
//
// Ports the behavioral intent of SGLang's SchedulePolicy.calc_priority /
// _sort_by_longest_prefix (python/sglang/srt/managers/schedule_policy.py:176,205
// @ v0.5.15 f63458b) onto OUR waiting deque + block-hash APC longest-match. The
// `lpm` policy reorders the waiting queue by descending cached-prefix match so
// cache hits are admitted first. It is OUTPUT-NEUTRAL: it changes only the ORDER
// requests are admitted, never the tokens each computes.
//
// The gate (the RED line):
//   (a) OUTPUT-NEUTRAL — for the SAME admitted set, each request is scheduled to
//       compute the SAME number of new tokens under lpm and fcfs (the cached
//       prefix it reuses is identical regardless of admission order), and the
//       total prefix-cache hits over that set are identical.
//   (b) THE REORDER HAPPENED — a later-arrived high-match request is admitted
//       AHEAD of an earlier-arrived no-match request under lpm; under fcfs the
//       earlier one wins (the RED baseline: no reorder).
//   (c) HITS REALIZED EARLIER — under capacity pressure lpm serves strictly more
//       cache-hit tokens by a given step than fcfs (the throughput lever;
//       measured with the PrefixCacheStats counters).
#include <doctest/doctest.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "vllm/config/scheduler.h"
#include "vllm/sampling_params.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/metrics/stats.h"
#include "vllm/v1/request.h"
#include "vt/dtype.h"

using vllm::SamplingParams;
using vllm::SchedulerConfig;
using vllm::SchedulerPolicy;
using vllm::v1::CachingMetrics;
using vllm::v1::FullAttentionSpec;
using vllm::v1::get_request_block_hasher;
using vllm::v1::init_none_hash;
using vllm::v1::KVCacheConfig;
using vllm::v1::Request;
using vllm::v1::Scheduler;
using vllm::v1::sha256_cbor;
using vt::DType;

namespace {

constexpr int kBlockSize = 16;

std::unique_ptr<Scheduler> CreateScheduler(SchedulerPolicy policy,
                                           bool enable_caching = true,
                                           int max_num_seqs = 8,
                                           int num_blocks = 4096) {
  SchedulerConfig cfg;
  cfg.max_num_seqs = max_num_seqs;
  cfg.max_num_batched_tokens = 8192;
  cfg.enable_chunked_prefill = true;
  cfg.max_model_len = 8192;
  cfg.watermark = 0.0;
  cfg.policy = policy;

  KVCacheConfig kv_cfg;
  kv_cfg.num_blocks = num_blocks;
  kv_cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<FullAttentionSpec>(kBlockSize, /*num_kv_heads=*/1,
                                          /*head_size=*/1, DType::kF32));
  return std::make_unique<Scheduler>(cfg, kv_cfg, kBlockSize, enable_caching);
}

// A prompt of `shared_blocks` common-prefix blocks then `unique_blocks` blocks
// unique to `seed`. shared_blocks == 0 gives an all-unique prompt (no match).
std::vector<int32_t> SharedPrefixPrompt(int shared_blocks, int unique_blocks,
                                        int seed) {
  std::vector<int32_t> prompt;
  for (int b = 0; b < shared_blocks; ++b) {
    for (int i = 0; i < kBlockSize; ++i) {
      prompt.push_back(1000 + b * kBlockSize + i);
    }
  }
  for (int b = 0; b < unique_blocks; ++b) {
    for (int i = 0; i < kBlockSize; ++i) {
      prompt.push_back(500000 + seed * 10000 + b * kBlockSize + i);
    }
  }
  return prompt;
}

Request* AddRequest(Scheduler& sched, const std::string& id,
                    const std::vector<int32_t>& prompt) {
  auto hasher = get_request_block_hasher(kBlockSize, sha256_cbor);
  SamplingParams params;
  params.max_tokens = 16;
  auto req = std::make_unique<Request>(id, prompt, params,
                                       /*arrival_time=*/0.0, hasher);
  Request* raw = req.get();
  sched.add_request(std::move(req));
  return raw;
}

// 4 shared prefix blocks (64 tokens) + 1 unique block (16 tokens) = 80 tokens.
constexpr int kSharedBlocks = 4;
constexpr int kUniqueBlocks = 1;
constexpr int kSharedTokens = kSharedBlocks * kBlockSize;  // 64

}  // namespace

// ---------------------------------------------------------------------------
// (a) + (b): the reorder happens and it is output-neutral.
// A warm request caches a 4-block prefix P. Then TWO requests arrive: "early"
// (no shared prefix, arrives FIRST) and "late" (prefix P + unique tail, arrives
// SECOND). With ample capacity both admit in one step, so we can read the
// admission ORDER directly from scheduled_new_reqs and confirm the per-request
// token work + total hits are identical between fcfs and lpm.
// ---------------------------------------------------------------------------
TEST_CASE("lpm reorders admission by longest prefix, output-neutral vs fcfs") {
  init_none_hash(sha256_cbor);

  auto run = [](SchedulerPolicy policy) {
    auto sched = CreateScheduler(policy, /*enable_caching=*/true,
                                 /*max_num_seqs=*/8);
    // Warm: cache prefix P (kSharedBlocks blocks). This request keeps running.
    AddRequest(*sched, "warm",
               SharedPrefixPrompt(kSharedBlocks, kUniqueBlocks, /*seed=*/1));
    sched->schedule();

    // early arrives first (no shared prefix); late arrives second (shares P).
    AddRequest(*sched, "early", SharedPrefixPrompt(0, kSharedBlocks + kUniqueBlocks,
                                                   /*seed=*/2));
    AddRequest(*sched, "late",
               SharedPrefixPrompt(kSharedBlocks, kUniqueBlocks, /*seed=*/3));
    return sched;
  };

  // -- RED baseline: fcfs does NOT reorder -> early (earlier arrival) admits first.
  auto fcfs = run(SchedulerPolicy::kFCFS);
  auto out_fcfs = fcfs->schedule();
  REQUIRE(out_fcfs.scheduled_new_reqs.size() == 2);
  CHECK(out_fcfs.scheduled_new_reqs[0].req_id == "early");  // no reorder.
  CHECK(out_fcfs.scheduled_new_reqs[1].req_id == "late");

  // -- GREEN: lpm reorders -> late (higher cached-prefix match) admits first.
  auto lpm = run(SchedulerPolicy::kLPM);
  auto out_lpm = lpm->schedule();
  REQUIRE(out_lpm.scheduled_new_reqs.size() == 2);
  CHECK(out_lpm.scheduled_new_reqs[0].req_id == "late");  // reorder happened.
  CHECK(out_lpm.scheduled_new_reqs[1].req_id == "early");

  // -- (a) OUTPUT-NEUTRAL: same per-request token work under both policies.
  // "late" reuses the 64 cached tokens of P (computes 80-64=16 new) and "early"
  // computes its full 80 tokens, IDENTICALLY regardless of admission order.
  CHECK(out_fcfs.num_scheduled_tokens.at("late") ==
        out_lpm.num_scheduled_tokens.at("late"));
  CHECK(out_fcfs.num_scheduled_tokens.at("early") ==
        out_lpm.num_scheduled_tokens.at("early"));
  CHECK(out_lpm.num_scheduled_tokens.at("late") ==
        (kSharedBlocks + kUniqueBlocks) * kBlockSize - kSharedTokens);  // 16
  CHECK(out_lpm.num_scheduled_tokens.at("early") ==
        (kSharedBlocks + kUniqueBlocks) * kBlockSize);  // 80

  // Total prefix-cache hits over the admitted set are identical (only WHICH
  // request is served first differs) -> the reorder is output-neutral.
  CHECK(fcfs->prefix_cache_metrics().aggregated_query_hit() ==
        lpm->prefix_cache_metrics().aggregated_query_hit());
  CHECK(lpm->prefix_cache_metrics().aggregated_query_hit() == kSharedTokens);
}

// ---------------------------------------------------------------------------
// (c): under capacity pressure lpm realizes cache hits EARLIER. With one free
// slot (warm occupies the other), a single step admits exactly one of the pair.
// fcfs admits "early" (0 hits); lpm admits "late" (64 hit tokens). By that step
// lpm has served strictly more cache-hit tokens -> the throughput lever.
// RED-first: the same assertion fails under fcfs (0 hits), passes under lpm.
// ---------------------------------------------------------------------------
TEST_CASE("lpm serves cache hits earlier than fcfs under capacity pressure") {
  init_none_hash(sha256_cbor);

  auto run_one_contended_step = [](SchedulerPolicy policy) -> long {
    // max_num_seqs=2: warm keeps one slot, leaving exactly one for the pair.
    auto sched = CreateScheduler(policy, /*enable_caching=*/true,
                                 /*max_num_seqs=*/2);
    AddRequest(*sched, "warm",
               SharedPrefixPrompt(kSharedBlocks, kUniqueBlocks, /*seed=*/1));
    sched->schedule();  // warm caches P (0 hits: cache was empty).

    AddRequest(*sched, "early", SharedPrefixPrompt(0, kSharedBlocks + kUniqueBlocks,
                                                   /*seed=*/2));
    AddRequest(*sched, "late",
               SharedPrefixPrompt(kSharedBlocks, kUniqueBlocks, /*seed=*/3));
    auto out = sched->schedule();  // one free slot -> admits exactly one.
    CHECK(out.scheduled_new_reqs.size() == 1);
    return sched->prefix_cache_metrics().aggregated_query_hit();
  };

  const long fcfs_hits = run_one_contended_step(SchedulerPolicy::kFCFS);
  const long lpm_hits = run_one_contended_step(SchedulerPolicy::kLPM);

  CHECK(fcfs_hits == 0);              // RED baseline: no-match "early" admitted.
  CHECK(lpm_hits == kSharedTokens);   // 64 hit tokens: "late" admitted first.
  CHECK(lpm_hits > fcfs_hits);        // hits realized earlier under lpm.
}

// ---------------------------------------------------------------------------
// lpm with prefix caching OFF has no cache to match against -> it resolves to
// fcfs admission order (arrival order preserved). Byte-identical to fcfs.
// ---------------------------------------------------------------------------
TEST_CASE("lpm with prefix caching off falls back to fcfs order") {
  init_none_hash(sha256_cbor);
  auto sched = CreateScheduler(SchedulerPolicy::kLPM, /*enable_caching=*/false,
                               /*max_num_seqs=*/8);
  // Even if "late" would share a prefix, with caching off nothing is cached, so
  // admission stays in arrival order.
  AddRequest(*sched, "early", SharedPrefixPrompt(0, kSharedBlocks + kUniqueBlocks,
                                                 /*seed=*/2));
  AddRequest(*sched, "late",
             SharedPrefixPrompt(kSharedBlocks, kUniqueBlocks, /*seed=*/3));
  auto out = sched->schedule();
  REQUIRE(out.scheduled_new_reqs.size() == 2);
  CHECK(out.scheduled_new_reqs[0].req_id == "early");  // arrival order.
  CHECK(out.scheduled_new_reqs[1].req_id == "late");
}
