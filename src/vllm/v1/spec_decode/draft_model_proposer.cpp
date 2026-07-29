// Ported from: vllm/v1/spec_decode/draft_model.py:19 +
// vllm/v1/spec_decode/llm_base_proposer.py (propose :502-767, _greedy_sample
// :428-438, set_inputs_first_pass :838-851, the K-1 draft loop :682-761) @
// 555967922 (vLLM 0.26.0.dev0). See draft_model_proposer.h for the exact scope,
// the standalone-LM reduction, and the reused verify half (SPEC-REJECTION).
#include "vllm/v1/spec_decode/draft_model_proposer.h"

namespace vllm {
namespace v1 {
namespace spec_decode {

int32_t GreedyArgmax(const std::vector<float>& logits) {
  // torch.argmax tie-break: FIRST (lowest) index among the maxima. The
  // RejectionSampler's target argmax uses the same rule, so a draft that equals
  // the target argmax is accepted (llm_base_proposer.py:438 `.argmax(dim=-1)`).
  int32_t best = 0;
  float best_v = logits[0];
  for (int32_t i = 1; i < static_cast<int32_t>(logits.size()); ++i) {
    if (logits[static_cast<size_t>(i)] > best_v) {
      best_v = logits[static_cast<size_t>(i)];
      best = i;
    }
  }
  return best;
}

std::vector<int32_t> DraftModelProposeGreedy(const DraftLogitsFn& draft_logits,
                                             const std::vector<int32_t>& context,
                                             int k) {
  // k == 0: the drafter still would run a forward to keep its KV in sync, but no
  // draft tokens are requested (llm_base_proposer.py:610-616). Host brick: no KV.
  if (k <= 0) {
    return {};
  }

  std::vector<int32_t> drafts;
  drafts.reserve(static_cast<size_t>(k));

  // The running context fed to the draft oracle. Step 0 uses the incoming context
  // as-is (it already carries the target's just-sampled next token at its last
  // slot; set_inputs_first_pass :838-851). Each later step appends the previous
  // draft token — the autoregressive feed-back `input_ids = draft_token_ids[-1]`
  // of the K-1 loop (:686, :713-761). k==1 never enters the loop (early exit
  // :618-627); k>1 runs it k-1 more times.
  std::vector<int32_t> running = context;
  for (int step = 0; step < k; ++step) {
    const std::vector<float> logits = draft_logits(running);
    const int32_t draft = GreedyArgmax(logits);
    drafts.push_back(draft);
    if (step + 1 < k) {
      running.push_back(draft);
    }
  }
  return drafts;
}

std::vector<std::vector<int32_t>> DraftModelProposeBatch(
    const DraftLogitsFn& draft_logits,
    const std::vector<std::vector<int32_t>>& contexts, int k) {
  std::vector<std::vector<int32_t>> out;
  out.reserve(contexts.size());
  for (const std::vector<int32_t>& context : contexts) {
    // A row with no running context proposes nothing (the runner only drafts for
    // requests that sampled a token this step; empty == skip).
    if (context.empty()) {
      out.emplace_back();
      continue;
    }
    out.push_back(DraftModelProposeGreedy(draft_logits, context, k));
  }
  return out;
}

}  // namespace spec_decode
}  // namespace v1
}  // namespace vllm
