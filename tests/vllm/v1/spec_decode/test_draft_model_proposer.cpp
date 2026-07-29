// Ported from (test-porting protocol .agents/test-porting.md):
//   * vllm/tests/v1/e2e/spec_decode/test_spec_decode.py @ 555967922
//       test_eagle_correctness  :500-561 — the `method`/separate `model` spec
//       config path (:527-532) and THE equivalence assertion (:544-555): the
//       speculative run's output must MATCH the non-speculative reference run.
//       For a separate DRAFT model the method is "draft_model"
//       (vllm/config/speculative.py:684, uses_draft_model :1195, runner
//       vllm/v1/worker/gpu_model_runner.py:604-609).
//   * The greedy accept invariant it rests on is
//       vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py greedy path,
//       already gated by tests/vllm/v1/spec_decode/test_rejection_sampler.cpp.
//
// This unit gate realizes that e2e equivalence DETERMINISTICALLY at the brick
// level: a tiny synthetic target LM and a tiny synthetic draft LM (both as
// next-token-logits oracles). We propose K draft tokens with the net-new
// DraftModelProposeGreedy, verify them with the LANDED RejectionSampler against
// the target's own expanded logits, and assert the accepted token stream equals
// what the target's plain greedy run would have produced (the spec-decode
// correctness invariant — every accepted token is one the non-speculative target
// would emit). We additionally assert the USEFULNESS property that is specific to
// the proposer: when the draft LM agrees with the target, EVERY draft is accepted
// (num_sampled == k+1), and — RED-first — that this full acceptance depends on the
// proposer's autoregressive feed-back (a proposer that fails to feed its own
// drafts back mispredicts positions 2..k and under-accepts).
#include <doctest/doctest.h>

#include <array>
#include <cstdint>
#include <vector>

#include "vllm/v1/spec_decode/draft_model_proposer.h"
#include "vllm/v1/spec_decode/rejection_sampler.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

using vllm::v1::RejectionSampler;
using vllm::v1::RejectionSamplerOutput;
using vllm::v1::spec_decode::DraftLogitsFn;
using vllm::v1::spec_decode::DraftModelProposeBatch;
using vllm::v1::spec_decode::DraftModelProposeGreedy;
using vllm::v1::spec_decode::GreedyArgmax;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

constexpr int kVocab = 32;

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue Q() { return Queue{Cpu(), nullptr}; }

// A one-hot-ish logit row that argmaxes to `token` (sharp, so both the proposer's
// GreedyArgmax and the RejectionSampler's target argmax pick it unambiguously).
std::vector<float> OneHot(int32_t token) {
  std::vector<float> row(static_cast<size_t>(kVocab), 0.0f);
  row[static_cast<size_t>(token)] = 10.0f;
  return row;
}

// The synthetic TARGET language model: a fixed no-fixed-point permutation of the
// vocab keyed on the last context token. Successive greedy tokens are DISTINCT
// (g0 = last+7, g1 = last+14, ...), which is what makes the RED-first
// "repeat-proposer" genuinely under-accept.
DraftLogitsFn TargetOracle() {
  return [](const std::vector<int32_t>& context) {
    const int32_t last = context.back();
    return OneHot(static_cast<int32_t>((last + 7) % kVocab));
  };
}

// A DRAFT model that agrees with the target for the first `agree_steps` predicted
// positions, then diverges (emits a token the target would NOT). agree_steps ==
// large => a "perfect" draft (always agrees).
DraftLogitsFn DraftOracle(int agree_steps, const std::vector<int32_t>& base) {
  const size_t base_len = base.size();
  return [agree_steps, base_len](const std::vector<int32_t>& context) {
    const int32_t last = context.back();
    // How many tokens have been drafted so far == how far past the base context.
    const int step = static_cast<int>(context.size() - base_len);
    if (step < agree_steps) {
      return OneHot(static_cast<int32_t>((last + 7) % kVocab));  // == target
    }
    return OneHot(static_cast<int32_t>((last + 3) % kVocab));  // diverges
  };
}

// The target's plain greedy continuation of `context` for `n` tokens (the
// NON-speculative reference run — the equivalence denominator).
std::vector<int32_t> TargetGreedy(const DraftLogitsFn& target,
                                  std::vector<int32_t> context, int n) {
  std::vector<int32_t> out;
  for (int i = 0; i < n; ++i) {
    const int32_t g = GreedyArgmax(target(context));
    out.push_back(g);
    context.push_back(g);
  }
  return out;
}

// Verify a single request's drafts through the LANDED RejectionSampler: build the
// target's expanded verify logits (1 + k rows: target(context),
// target(context+[d0]), ..., target(context+[d0..d_{k-1}])) and the draft_sampled
// array (row 0 = the previous token, rows 1..k = the drafts), exactly the
// StepInputs shape the sampler consumes.
RejectionSamplerOutput Verify(const DraftLogitsFn& target,
                              const std::vector<int32_t>& context,
                              const std::vector<int32_t>& drafts) {
  const int k = static_cast<int>(drafts.size());
  const int rows = k + 1;
  std::vector<float> logits(static_cast<size_t>(rows) * kVocab, 0.0f);
  std::vector<int32_t> draft_sampled(static_cast<size_t>(rows), 0);
  std::vector<int32_t> cu = {0, rows};

  std::vector<int32_t> running = context;
  for (int j = 0; j < rows; ++j) {
    const std::vector<float> row = target(running);
    for (int v = 0; v < kVocab; ++v) {
      logits[static_cast<size_t>(j) * kVocab + static_cast<size_t>(v)] =
          row[static_cast<size_t>(v)];
    }
    if (j < k) {
      draft_sampled[static_cast<size_t>(j) + 1] = drafts[static_cast<size_t>(j)];
      running.push_back(drafts[static_cast<size_t>(j)]);
    }
  }

  Queue q = Q();
  Tensor t = Tensor::Contiguous(logits.data(), DType::kF32, Cpu(),
                                {static_cast<int64_t>(rows), kVocab});
  RejectionSampler sampler(k);
  return sampler.forward(q, t, draft_sampled, cu);
}

}  // namespace

// ---------------------------------------------------------------------------
// GreedyArgmax: torch.argmax tie-break — the LOWEST index among the maxima.
TEST_CASE("draft_model: GreedyArgmax picks the lowest-index maximum (torch tie-break)") {
  CHECK(GreedyArgmax({0.1f, 0.9f, 0.3f}) == 1);
  CHECK(GreedyArgmax({5.0f, 5.0f, 1.0f}) == 0);  // tie -> lowest index
  CHECK(GreedyArgmax({-1.0f, -2.0f, -0.5f}) == 2);
}

// k==0 proposes nothing (llm_base_proposer.py:610-616); k==1 is the single-step
// early exit (:618-627).
TEST_CASE("draft_model: propose sizes match k (0/1/many) and are autoregressive") {
  const DraftLogitsFn target = TargetOracle();
  const std::vector<int32_t> ctx = {5};  // last token = the target's just-sampled

  CHECK(DraftModelProposeGreedy(target, ctx, 0).empty());

  const std::vector<int32_t> one = DraftModelProposeGreedy(target, ctx, 1);
  CHECK(one.size() == 1);
  CHECK(one[0] == (5 + 7) % kVocab);  // greedy first draft

  const std::vector<int32_t> many = DraftModelProposeGreedy(target, ctx, 4);
  CHECK(many.size() == 4);
  // Fed a target-equivalent oracle, the drafts ARE the target's greedy chain.
  CHECK(many == TargetGreedy(target, ctx, 4));
}

// ---------------------------------------------------------------------------
// THE equivalence invariant (test_spec_decode.py:544-555, ref == spec): the
// accepted token stream MUST equal the target's own greedy run — for every
// draft/target (dis)agreement pattern.
TEST_CASE("draft_model: accepted tokens == target greedy run (spec-decode equivalence)") {
  const DraftLogitsFn target = TargetOracle();
  const int k = 3;

  for (int agree = 0; agree <= k; ++agree) {
    const std::vector<int32_t> ctx = {9};
    const DraftLogitsFn draft = DraftOracle(agree, ctx);
    const std::vector<int32_t> drafts = DraftModelProposeGreedy(draft, ctx, k);

    const RejectionSamplerOutput out = Verify(target, ctx, drafts);
    // The non-speculative reference: the target's plain greedy continuation.
    const std::vector<int32_t> ref = TargetGreedy(target, ctx, k + 1);

    CAPTURE(agree);
    const int ns = out.num_sampled[0];
    REQUIRE(static_cast<int>(out.sampled_token_ids[0].size()) == ns);
    // Every emitted token equals the reference greedy token at its position.
    for (int j = 0; j < ns; ++j) {
      CHECK(out.sampled_token_ids[0][static_cast<size_t>(j)] ==
            ref[static_cast<size_t>(j)]);
    }
    // Accepted length = leading agreements + 1 (the bonus or the replacement).
    CHECK(ns == agree + 1);
    CHECK(out.num_rejected[0] == k - agree);
  }
}

// The USEFULNESS property — a PERFECT draft (agrees everywhere) is fully accepted
// (num_sampled == k+1). This is what makes speculative decoding a speedup, and it
// is specific to the proposer's autoregressive feed-back.
TEST_CASE("draft_model: a target-matching draft is fully accepted (num_sampled == k+1)") {
  const DraftLogitsFn target = TargetOracle();
  const int k = 4;
  const std::vector<int32_t> ctx = {2};
  const DraftLogitsFn draft = DraftOracle(/*agree_steps=*/k + 1, ctx);  // perfect

  const std::vector<int32_t> drafts = DraftModelProposeGreedy(draft, ctx, k);
  const RejectionSamplerOutput out = Verify(target, ctx, drafts);

  CHECK(out.num_sampled[0] == k + 1);
  CHECK(out.num_rejected[0] == 0);
  CHECK(out.sampled_token_ids[0] == TargetGreedy(target, ctx, k + 1));
}

// RED-first witness — full acceptance DEPENDS on the autoregressive feed-back.
// A broken proposer that re-queries the SAME base context every step (never
// appending its own drafts) emits a repeated first token; the target's distinct
// greedy chain then rejects at position 1, so it under-accepts even though the
// draft LM is "perfect". The real DraftModelProposeGreedy accepts all k. If the
// implementation ever dropped the feed-back, the real-vs-broken gap below would
// vanish and this case would fail.
TEST_CASE("draft_model: autoregressive feed-back is load-bearing (RED-first)") {
  const DraftLogitsFn target = TargetOracle();
  const int k = 4;
  const std::vector<int32_t> ctx = {2};
  const DraftLogitsFn draft = DraftOracle(/*agree_steps=*/k + 1, ctx);  // perfect

  // The broken, non-autoregressive proposer: every step re-queries `ctx`.
  std::vector<int32_t> broken;
  for (int i = 0; i < k; ++i) broken.push_back(GreedyArgmax(draft(ctx)));
  // It repeats the first token: [g0, g0, g0, g0].
  CHECK(broken[0] == broken[1]);
  const RejectionSamplerOutput broken_out = Verify(target, ctx, broken);
  CHECK(broken_out.num_sampled[0] == 2);  // g0 accepted, then rejected at pos 1

  // The REAL proposer feeds back and is fully accepted.
  const std::vector<int32_t> real = DraftModelProposeGreedy(draft, ctx, k);
  const RejectionSamplerOutput real_out = Verify(target, ctx, real);
  CHECK(real_out.num_sampled[0] == k + 1);
  CHECK(real_out.num_sampled[0] > broken_out.num_sampled[0]);
}

// The batch propose: one context per request, empty rows skipped (mirrors the
// runner only drafting for rows that sampled a token this step).
TEST_CASE("draft_model: batch propose is per-request, empty context skipped") {
  const DraftLogitsFn target = TargetOracle();
  const std::vector<std::vector<int32_t>> contexts = {{1}, {}, {8}};
  const std::vector<std::vector<int32_t>> got =
      DraftModelProposeBatch(target, contexts, 3);
  REQUIRE(got.size() == 3);
  CHECK(got[0] == TargetGreedy(target, {1}, 3));
  CHECK(got[1].empty());
  CHECK(got[2] == TargetGreedy(target, {8}, 3));
}
