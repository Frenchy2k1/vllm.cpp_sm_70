// Ported from:
//   * vllm/v1/spec_decode/draft_model.py:19 (DraftModelProposer) @ 555967922
//     (vLLM 0.26.0.dev0) — the classic model-agnostic SEPARATE draft-model
//     proposer: a full smaller model runs K autoregressive steps to propose K
//     draft tokens. It sets pass_hidden_states_to_model=False (draft_model.py:29)
//     and shares NEITHER embeddings NOR lm_head with the target
//     (draft_model.py:108-115) — i.e. it is a plain standalone LM, unlike the
//     hidden-state methods (MTP/EAGLE/DFlash) that tap the target's residual.
//   * vllm/v1/spec_decode/llm_base_proposer.py @ 555967922 — the propose loop the
//     draft model reduces to:
//       - _greedy_sample :428-438  (draft_token = compute_logits(h).argmax(-1))
//       - set_inputs_first_pass :838-851  (shift target ids by one, insert the
//         target's just-sampled next_token at each request's last slot)
//       - propose :502-767  (first forward + sample; then the K-1 autoregressive
//         draft loop :682-761 which feeds the previous draft token back in as the
//         next input_id; k==1 early-exits at :618-627)
//
// Scope (SPEC-DRAFT-MODEL W1, the first CPU-buildable brick): the GREEDY propose
// half — run a draft model K steps autoregressively, argmax each step, feed the
// drafted token back, return the K draft ids. It reuses the ALREADY-LANDED verify
// half unchanged: the drafts are handed to `RejectionSampler` (SPEC-REJECTION),
// which accepts the longest prefix the target agrees with, so every emitted token
// is one the target's own greedy run would have produced (the spec-decode
// equivalence invariant). Only the PROPOSER is net-new here.
//
// The draft model is abstracted as a next-token-logits oracle (DraftLogitsFn):
// given a request's running token context, return the vocab-sized logits for the
// NEXT token. This mirrors `compute_logits(model(input_ids))[last]` reduced to the
// standalone-LM greedy path, without pulling the whole GPU model runner into the
// CPU brick — exactly the way SPEC-NGRAM's matcher is a pure host algorithm that
// then reuses the shared verify/reject loop. The real GPU model forward + paged KV
// + CUDA-graph wiring behind this seam is the named later brick (W3, DGX).
#ifndef VLLM_V1_SPEC_DECODE_DRAFT_MODEL_PROPOSER_H_
#define VLLM_V1_SPEC_DECODE_DRAFT_MODEL_PROPOSER_H_

#include <cstdint>
#include <functional>
#include <vector>

namespace vllm {
namespace v1 {
namespace spec_decode {

// A draft model as a next-token-logits oracle. `context` is the request's running
// token ids; the return is the vocab-sized logit row for the token that follows
// `context`. Mirrors the standalone draft model's
// `compute_logits(model(input_ids))` at the last position (llm_base_proposer.py:
// 438). The context is non-empty on every call (it always carries at least the
// target's just-sampled next token).
using DraftLogitsFn =
    std::function<std::vector<float>(const std::vector<int32_t>& context)>;

// Greedy argmax with the torch.argmax tie-break: the LOWEST index among maxima
// wins (matches the RejectionSampler's target argmax, so accept-iff-equal is
// consistent on both sides). `logits` must be non-empty.
int32_t GreedyArgmax(const std::vector<float>& logits);

// DraftModelProposeGreedy — the k-step greedy autoregressive propose for ONE
// request (draft_model.py:19 + llm_base_proposer.py propose :502-767, reduced to
// the greedy, pass_hidden_states_to_model=False standalone-LM path).
//
//   draft_logits  the draft model's next-token oracle.
//   context       the request's running token ids INCLUDING the target's
//                 just-sampled next token at the last slot (the
//                 set_inputs_first_pass :838-851 insertion). Must be non-empty.
//   k             num_speculative_tokens (>= 0). k==0 returns empty
//                 (llm_base_proposer.py:610-616); k==1 is the early-exit single
//                 forward (:618-627).
//
// Returns the k greedily-drafted token ids. Step 0 drafts argmax(draft(context));
// each subsequent step appends the previous draft to the context and re-queries
// the oracle (the autoregressive feed-back, :682-761 `input_ids = draft[-1]`).
std::vector<int32_t> DraftModelProposeGreedy(const DraftLogitsFn& draft_logits,
                                             const std::vector<int32_t>& context,
                                             int k);

// The batch propose (llm_base_proposer.py propose over the whole InputBatch): one
// context per request, K drafts each, returned row-major [num_reqs][k]. A request
// whose context is empty (e.g. still chunk-prefilling, no sampled token this step)
// proposes nothing — mirrors the runner only drafting for rows that sampled.
std::vector<std::vector<int32_t>> DraftModelProposeBatch(
    const DraftLogitsFn& draft_logits,
    const std::vector<std::vector<int32_t>>& contexts, int k);

}  // namespace spec_decode
}  // namespace v1
}  // namespace vllm

#endif  // VLLM_V1_SPEC_DECODE_DRAFT_MODEL_PROPOSER_H_
