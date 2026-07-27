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
//
// SW2 (this file's second half) refines the LPM sort with SGLang's in-batch
// prefix-collision de-prioritization (schedule_policy.py:253-301): when two
// waiting requests share an UNCACHED prefix, the second is sorted behind all
// non-colliding requests so the first populates the cache and the second HITS it
// next round instead of redundantly recomputing the prefix. Still output-neutral.
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
                                           int num_blocks = 4096,
                                           int max_num_batched_tokens = 8192,
                                           bool enable_chunked_prefill = true) {
  SchedulerConfig cfg;
  cfg.max_num_seqs = max_num_seqs;
  cfg.max_num_batched_tokens = max_num_batched_tokens;
  cfg.enable_chunked_prefill = enable_chunked_prefill;
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

// Build a prompt from an explicit list of block "kinds": each kind maps to a
// deterministic, unique 16-token block (token = kind*1000 + i). Two prompts that
// share a leading run of kinds therefore share that many leading block hashes
// (block hashes are chained over the prefix) — exactly what SW2's in-batch
// collision check keys on. Distinct kinds never collide.
std::vector<int32_t> BlocksPrompt(const std::vector<int>& block_kinds) {
  std::vector<int32_t> prompt;
  for (int kind : block_kinds) {
    for (int i = 0; i < kBlockSize; ++i) {
      prompt.push_back(kind * 1000 + i);
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

// ===========================================================================
// SW2 — in-batch prefix-collision de-prioritization.
// Ported 1:1 from SGLang SchedulePolicy._compute_prefix_matches
// (python/sglang/srt/managers/schedule_policy.py:253-301 @ f63458b): among the
// LPM-sorted waiting requests, a request whose longest prefix COLLIDES with a
// prefix an earlier waiting request is already about to compute (uncached, in
// flight) is sorted BEHIND all non-colliding requests (the `float("inf")` sort
// key at :311). The collider waits one round, then hits the now-populated cache.
//
// Shared fixture across the SW2 cases:
//   - C  = 2 blocks (32 tokens) — cached by a running "warm" request.
//   - P  = C + 3 more blocks = 5 blocks (80 tokens) — the colliders' shared
//          prefix; blocks 3-5 are UNCACHED (warm never computed them).
//   - collide_a / collide_b = P + 1 unique block (6 blocks / 96 tokens). Their
//          REAL cached match is C = 32 tokens (<= kInBatchCheckThreshold), so
//          both are subject to the in-batch check; they share P (80 tokens >=
//          kInBatchDeprioritizeThreshold), so the SECOND is de-prioritized.
//   - solo = an all-unique prompt (4 blocks / 64 tokens): NO cache match, NO
//          collision.
// Block kinds: C={10,11}; P={10,11,12,13,14}; warm={10,11,90};
//              collide_a={10,11,12,13,14,91}; collide_b={10,11,12,13,14,92};
//              solo={20,21,22,93}.
// ===========================================================================
namespace {
const std::vector<int> kC = {10, 11};
const std::vector<int> kWarm = {10, 11, 90};
const std::vector<int> kCollideA = {10, 11, 12, 13, 14, 91};
const std::vector<int> kCollideB = {10, 11, 12, 13, 14, 92};
const std::vector<int> kSolo = {20, 21, 22, 93};
constexpr int kCollideRealMatch = 2 * kBlockSize;   // 32 (C cached)
constexpr int kCollideTokens = 6 * kBlockSize;      // 96
constexpr int kSoloTokens = 4 * kBlockSize;         // 64
}  // namespace

// ---------------------------------------------------------------------------
// (b) THE DE-PRIORITIZATION HAPPENED + (a) OUTPUT-NEUTRAL. With ample capacity
// all three requests admit in one step, so admission order is read directly.
// Arrival order is [collide_a, collide_b, solo]; that is ALSO the raw-longest-
// prefix (LPM-without-SW2) order, since collide_a/collide_b match 32 > solo's 0
// and the tie keeps arrival — so fcfs is a faithful "without SW2" RED baseline.
//   RED (fcfs / raw-prefix order): the collider (collide_b) admits AHEAD of the
//       non-colliding solo.
//   GREEN (lpm+SW2): collide_b is de-prioritized BEHIND solo.
// Output-neutral: for the same admitted set, every request computes the SAME new
// tokens under fcfs and lpm, and the total prefix-cache hits are identical.
// ---------------------------------------------------------------------------
TEST_CASE("sw2 de-prioritizes in-batch prefix collider, output-neutral") {
  init_none_hash(sha256_cbor);

  auto run = [](SchedulerPolicy policy) {
    auto sched = CreateScheduler(policy, /*enable_caching=*/true,
                                 /*max_num_seqs=*/8);
    AddRequest(*sched, "warm", BlocksPrompt(kWarm));  // caches C.
    sched->schedule();
    AddRequest(*sched, "collide_a", BlocksPrompt(kCollideA));  // arrives 1st
    AddRequest(*sched, "collide_b", BlocksPrompt(kCollideB));  // arrives 2nd
    AddRequest(*sched, "solo", BlocksPrompt(kSolo));           // arrives 3rd
    return sched;
  };

  // -- RED baseline: fcfs (== raw-prefix order) admits the collider before solo.
  auto fcfs = run(SchedulerPolicy::kFCFS);
  auto out_fcfs = fcfs->schedule();
  REQUIRE(out_fcfs.scheduled_new_reqs.size() == 3);
  CHECK(out_fcfs.scheduled_new_reqs[0].req_id == "collide_a");
  CHECK(out_fcfs.scheduled_new_reqs[1].req_id == "collide_b");  // collider ahead
  CHECK(out_fcfs.scheduled_new_reqs[2].req_id == "solo");       // of solo (RED)

  // -- GREEN: lpm+SW2 de-prioritizes collide_b BEHIND the non-colliding solo.
  auto lpm = run(SchedulerPolicy::kLPM);
  auto out_lpm = lpm->schedule();
  REQUIRE(out_lpm.scheduled_new_reqs.size() == 3);
  CHECK(out_lpm.scheduled_new_reqs[0].req_id == "collide_a");  // 1st seer stays
  CHECK(out_lpm.scheduled_new_reqs[1].req_id == "solo");       // solo promoted
  CHECK(out_lpm.scheduled_new_reqs[2].req_id == "collide_b");  // collider last

  // -- (a) OUTPUT-NEUTRAL: same per-request new-token work under both policies.
  // The admitted SET is identical (all three, one step); SW2 only changes the
  // ORDER. Note collide_b computes only 16 new tokens under BOTH policies: our
  // APC caches collide_a's blocks at allocation time (kv_cache_manager.cpp:267),
  // so the SECOND collider hits collide_a's just-cached prefix WITHIN the same
  // step regardless of SW2 -> the shared uncached prefix is computed exactly
  // once either way (see the dedicated finding case below).
  for (const char* id : {"collide_a", "collide_b", "solo"}) {
    CHECK(out_fcfs.num_scheduled_tokens.at(id) ==
          out_lpm.num_scheduled_tokens.at(id));
  }
  CHECK(out_lpm.num_scheduled_tokens.at("collide_a") ==
        kCollideTokens - kCollideRealMatch);  // 96 - 32 (C) = 64
  CHECK(out_lpm.num_scheduled_tokens.at("collide_b") ==
        kCollideTokens - 5 * kBlockSize);  // 96 - 80 (full P, cached by A) = 16
  CHECK(out_lpm.num_scheduled_tokens.at("solo") == kSoloTokens);  // 64

  // Total prefix-cache hits over the admitted set are identical -> the reorder
  // is output-neutral (only WHICH request is served first differs). All three
  // admit in ONE step with no budget rejection, so each is queried exactly once:
  // collide_a 32 (C) + collide_b 80 (C + A's cached P-tail) + solo 0 = 112.
  CHECK(fcfs->prefix_cache_metrics().aggregated_query_hit() ==
        lpm->prefix_cache_metrics().aggregated_query_hit());
  CHECK(lpm->prefix_cache_metrics().aggregated_query_hit() ==
        kCollideRealMatch + 5 * kBlockSize);  // 32 + 80 = 112
}

// ---------------------------------------------------------------------------
// (c) REDUNDANT-PREFILL / HIT-RATE lever — NOT-APPLICABLE in our engine, and
// THIS is the honest finding. SGLang's SW2 exists because its radix cache is
// updated only AFTER the forward pass, so two colliders in the SAME batch both
// MISS and redundantly compute the shared prefix; SW2 defers the second so it
// hits next round. OUR APC caches a request's blocks at ALLOCATION time (during
// scheduling — kv_cache_manager.cpp:267), so the SECOND same-step collider
// already HITS the first's just-cached prefix. The shared uncached prefix is
// therefore computed EXACTLY ONCE even WITHOUT SW2 -> there is no redundant
// prefill for SW2 to remove, and its hit-rate is IDENTICAL with or without it.
//
// Proof (single step, ample capacity, so no budget rejection distorts counts):
// under BOTH fcfs and lpm+SW2 the pair {collide_a, collide_b} computes 64 + 16
// = 80 new tokens total (P computed once by A + B's 16-token unique tail), NOT
// 64 + 64 = 128 (which a post-forward radix like SGLang's would incur). SW2 is
// thus compute- and hit-rate-neutral here; its only effect is admission ORDER.
// ---------------------------------------------------------------------------
TEST_CASE("sw2 redundant-prefill lever is subsumed by within-step APC dedup") {
  init_none_hash(sha256_cbor);

  auto pair_new_tokens = [](SchedulerPolicy policy) -> std::pair<int, int> {
    auto sched = CreateScheduler(policy, /*enable_caching=*/true,
                                 /*max_num_seqs=*/8);
    AddRequest(*sched, "warm", BlocksPrompt(kWarm));  // caches C.
    sched->schedule();
    AddRequest(*sched, "collide_a", BlocksPrompt(kCollideA));
    AddRequest(*sched, "collide_b", BlocksPrompt(kCollideB));
    auto out = sched->schedule();  // ample capacity -> both admit this step.
    REQUIRE(out.scheduled_new_reqs.size() == 2);
    return {out.num_scheduled_tokens.at("collide_a"),
            out.num_scheduled_tokens.at("collide_b")};
  };

  // Even under fcfs (NO SW2) the second collider auto-dedups within the step.
  const auto [fcfs_a, fcfs_b] = pair_new_tokens(SchedulerPolicy::kFCFS);
  CHECK(fcfs_a == kCollideTokens - kCollideRealMatch);  // 64: A computes P once.
  CHECK(fcfs_b == kCollideTokens - 5 * kBlockSize);     // 16: B reuses A's P.
  CHECK(fcfs_a + fcfs_b == 80);  // NOT 128 -> no redundant prefill to remove.

  // SW2 (lpm) yields the IDENTICAL per-request compute: it is hit-rate-neutral.
  const auto [lpm_a, lpm_b] = pair_new_tokens(SchedulerPolicy::kLPM);
  CHECK(lpm_a == fcfs_a);
  CHECK(lpm_b == fcfs_b);
}

// ---------------------------------------------------------------------------
// SW2 is INERT when the shared prefix is already fully cached: the colliders'
// real match then exceeds kInBatchCheckThreshold, so the in-batch check is
// skipped and ordering is pure LPM (raw longest prefix). This is why the SW1
// LPM cases above (whose shared prefix P is cached by "warm") are unchanged.
// Here P (5 blocks) is fully cached, so both colliders match 80 > 32 and NEITHER
// is de-prioritized; they keep their (equal) raw-match order ahead of solo.
// ---------------------------------------------------------------------------
TEST_CASE("sw2 inert when collision prefix already cached (check-threshold gate)") {
  init_none_hash(sha256_cbor);
  auto sched = CreateScheduler(SchedulerPolicy::kLPM, /*enable_caching=*/true,
                               /*max_num_seqs=*/8);
  // Warm caches the FULL P (5 blocks) + a tail, so colliders match all of P.
  AddRequest(*sched, "warm", BlocksPrompt({10, 11, 12, 13, 14, 90}));
  sched->schedule();
  AddRequest(*sched, "collide_a", BlocksPrompt(kCollideA));
  AddRequest(*sched, "collide_b", BlocksPrompt(kCollideB));
  AddRequest(*sched, "solo", BlocksPrompt(kSolo));
  auto out = sched->schedule();
  REQUIRE(out.scheduled_new_reqs.size() == 3);
  // Both colliders match 80 (> check threshold) -> not de-prioritized; pure LPM
  // keeps them (arrival-stable) ahead of the no-match solo.
  CHECK(out.scheduled_new_reqs[0].req_id == "collide_a");
  CHECK(out.scheduled_new_reqs[1].req_id == "collide_b");  // NOT sent to the back
  CHECK(out.scheduled_new_reqs[2].req_id == "solo");
}
