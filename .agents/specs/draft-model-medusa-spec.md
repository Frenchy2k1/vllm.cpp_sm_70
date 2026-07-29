# Generic draft-model + Medusa speculative decoding (`CLAIM-SPEC-DRAFT-MEDUSA`)

Spike + W1 CPU brick for the last spec-decode records-gap named by the
[feature-gap analysis](vllm-feature-gap-analysis.md) (lines 82-83): the classic
model-agnostic **separate draft-model** path
(`vllm/v1/spec_decode/draft_model.py:19`) and **Medusa** multi-head speculation
(`vllm/v1/spec_decode/medusa.py:18`). We already have MTP / DFlash / ngram /
EAGLE3 / DSpark proposers and the shared verify/accept machinery; only the two
new PROPOSERS are net-new. Pinned oracle `/home/mudler/_git/vllm` @ `555967922`
(vLLM 0.26.0.dev0). CPU-only research lane (DGX offline); the GPU e2e + speed
gate is a named later brick.

Creates the counted rows **`SPEC-DRAFT-MODEL`** (ACTIVE, W1 landed) and
**`SPEC-MEDUSA`** (SPIKE, deferred to W2) in
[engine-matrix.md](../engine-matrix.md).

## Scope

- **In (W1):** the generic separate draft-model GREEDY propose brick — run a
  standalone draft LM K autoregressive steps, argmax each step, feed the drafted
  token back, return the K draft ids — plus its unit gate reusing the LANDED
  `RejectionSampler` verify, and the `"draft_model"` `--speculative-config`
  accept. CPU-only, host-side, additive + default-inert.
- **In (W0 spike, both methods):** the full design of the `DraftModelProposer` and
  the `MedusaProposer`, the `SpeculativeConfig` method selection, how both reuse
  the landed acceptance sampler + scheduler seam, exact files, upstream tests,
  gates, W-breakdown.
- **Out:** the Medusa PROPOSER (W2 — needs the target's Medusa heads, a model
  change); the real GPU draft-model forward behind the oracle + the DGX e2e
  greedy equivalence gate + the throughput speed gate (W3, DGX-offline); the
  heterogeneous-vocab `VocabMapping`, dynamic-SD K schedule, and tree verify
  (later bricks). No production runner wiring in W1.

## Upstream chain

The spec-decode STEP is the same propose -> verify -> accept loop for every
method; only the PROPOSER differs. The shared halves are ALREADY landed:

- **Verify / accept** — `RejectionSampler`
  (`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py` greedy path),
  ported to `src/vllm/v1/spec_decode/rejection_sampler.{h,cpp}` (row
  `SPEC-REJECTION`, DONE): given the target's EXPANDED logits (1 + k rows per
  request) and the scheduled draft ids, accept the longest prefix the target's
  argmax agrees with, emit accepted drafts + one bonus/replacement token. This is
  what makes every emitted token one the non-speculative target would produce.
- **Scheduler seam** — `SpeculativeConfig` (`include/vllm/config/speculative.h`)
  already carries `uses_draft_model()` (method=="draft_model", `speculative.py:
  1195`) feeding `NumLookaheadTokens() = k` (`scheduler.py:287-288`), and the
  spec-metadata ABI (`Request::spec_token_ids`, `scheduled_spec_decode_tokens`,
  `update_draft_token_ids`, `InputBatch::num_accepted_tokens`) landed with
  SPEC-MTP I2.

### Proposer 1 — generic draft model (`draft_model.py:19`)

`DraftModelProposer(SpecDecodeBaseProposer)` runs a full SEPARATE smaller model K
autoregressive steps to propose K draft tokens. Distinctives vs the
hidden-state methods:

- `pass_hidden_states_to_model=False` (`draft_model.py:29`) — it is a plain
  standalone LM; it does NOT tap the target's residual stream (unlike
  MTP/EAGLE/DFlash).
- shares NEITHER embeddings NOR lm_head with the target (`draft_model.py:
  108-115`).
- config: method `"draft_model"` (`speculative.py:684` — the DEFAULT when a
  separate `model` is given and no other method auto-detects; `uses_draft_model`
  `:1195`); a separate `draft_model_config` built from the `model` key
  (`:692-701`); TP of draft must equal target TP (`draft_model.py:63-78`);
  optional heterogeneous-vocab `VocabMapping` when the draft tokenizer differs
  (`draft_model.py:34-58`).
- runner: `gpu_model_runner.py:604-609` constructs it; the propose loop is
  `SpecDecodeBaseProposer.propose` (`llm_base_proposer.py:502-767`): shift the
  target ids by one and insert the target's just-sampled `next_token` at each
  request's last slot (`set_inputs_first_pass :838-851`), run the draft forward,
  argmax (`_greedy_sample :428-438`), then the K-1 autoregressive loop
  (`:682-761`) that feeds the previous draft token back as the next input_id.
  k==1 early-exits at `:618-627`.

### Proposer 2 — Medusa (`medusa.py:18`)

`MedusaProposer` is NOT autoregressive: the target model carries N extra Medusa
LM heads, each of which predicts ONE future position from the SAME target hidden
state. `propose` (`medusa.py:40-58`): `blocks = model(target_hidden_states)`;
`logits = model.compute_logits(blocks)` (a list, one per head); `draft_tokens =
stack([logit.argmax(-1) for logit in logits], dim=1)` -> `[batch, num_heads]`. It
consumes the target's hidden tap (like MTP) but emits all N drafts in ONE pass
(like DFlash's parallel drafting), so `num_speculative_tokens == num_heads`.
Config: method `"medusa"` (`speculative.py:822,888-889`), draft is the Medusa
head checkpoint; runner `gpu_model_runner.py:642-645`. Verify/accept is the SAME
`RejectionSampler` (the N head drafts are linear, not a tree, at this pin —
`draft_tokens[:, 0..N-1]`).

## Our baseline

- `SPEC-REJECTION` DONE — the verify/accept loop we reuse UNCHANGED.
- `SPEC-NGRAM` DONE — the precedent for a proposer that is a pure host algorithm
  reusing the shared verify loop; `SPEC-DRAFT-MODEL` follows the same shape (the
  proposer is net-new, the verify is reused).
- `SpeculativeConfig.uses_draft_model()` / `NumLookaheadTokens()` already correct
  for method=="draft_model"; only the JSON parser rejected the method (fixed W1).

## W1 delivered (this change) — generic draft-model greedy propose

The more tractable of the two (a full model forward as the proposer; Medusa needs
the multi-head target tap and is deferred to W2). Files:

- **`include/vllm/v1/spec_decode/draft_model_proposer.h`** — `DraftLogitsFn`
  (the draft model as a next-token-logits oracle: `context -> vocab logits`,
  mirroring `compute_logits(model(input_ids))[last]` reduced to the standalone-LM
  greedy path, without pulling the GPU model runner into the CPU brick, exactly
  as SPEC-NGRAM's matcher is a host algorithm), `GreedyArgmax` (torch.argmax
  tie-break: lowest index), `DraftModelProposeGreedy` (the k-step autoregressive
  propose for one request), `DraftModelProposeBatch` (per-request, empty-context
  rows skipped).
- **`src/vllm/v1/spec_decode/draft_model_proposer.cpp`** — the propose: step 0
  argmax(draft(context)); each later step appends the previous draft and
  re-queries (the `:682-761` feed-back); k==0 empty, k==1 single forward.
- **`tests/vllm/v1/spec_decode/test_draft_model_proposer.cpp`** — realizes the
  upstream e2e equivalence (`tests/v1/e2e/spec_decode/test_spec_decode.py:
  544-555`, ref==spec) DETERMINISTICALLY: a tiny synthetic target LM + draft LM
  as oracles; propose K -> verify with the LANDED `RejectionSampler` against the
  target's expanded logits -> assert the accepted stream equals the target's own
  greedy run (every accepted token is one the non-speculative target emits), for
  every draft/target (dis)agreement pattern. Plus the usefulness property (a
  target-matching draft is fully accepted, num_sampled==k+1) and the **RED-first**
  witness that full acceptance DEPENDS on the autoregressive feed-back (a
  non-feed-back proposer repeats its first token and under-accepts).
- **`src/vllm/config/speculative.cpp`** — `ParseSpeculativeConfigJson` accepts
  `"draft_model"` (requires the `model` key + `num_speculative_tokens`,
  `speculative.py:692-701`).

## Port map

| Upstream | Ours (W1) |
|---|---|
| `vllm/v1/spec_decode/draft_model.py:19` + `llm_base_proposer.py` propose :502-767 (greedy standalone-LM reduction) | `include/vllm/v1/spec_decode/draft_model_proposer.h`; `src/vllm/v1/spec_decode/draft_model_proposer.cpp` (`GreedyArgmax`, `DraftModelProposeGreedy`, `DraftModelProposeBatch`) |
| `vllm/config/speculative.py:684,692-701,1195` (method `draft_model` select) | `src/vllm/config/speculative.cpp` (`ParseSpeculativeConfigJson` `draft_model` accept); `include/vllm/config/speculative.h` `uses_draft_model()`/`NumLookaheadTokens()` (already landed) |
| `rejection_sampler_utils.py` greedy verify/accept | `src/vllm/v1/spec_decode/rejection_sampler.{h,cpp}` (`SPEC-REJECTION`, reused UNCHANGED) |
| `tests/v1/e2e/spec_decode/test_spec_decode.py:500-561` (ref==spec) | `tests/vllm/v1/spec_decode/test_draft_model_proposer.cpp` |
| `vllm/v1/spec_decode/medusa.py:18` (MedusaProposer) | deferred W2 (spike only) |

## Tests to port

| Upstream | Here | State |
|---|---|---|
| `tests/v1/e2e/spec_decode/test_spec_decode.py:500-561` (ref==spec equivalence; `method`/separate `model` config) | `test_draft_model_proposer.cpp` (deterministic unit realization) | LANDED |
| `tests/v1/spec_decode/test_rejection_sampler_utils.py` (greedy accept) | `test_rejection_sampler.cpp` (SPEC-REJECTION, reused) | LANDED (prior) |
| Medusa multi-head argmax propose (`medusa.py:40-58`; no dedicated upstream unit — covered by `test_speculators_correctness.py`) | `test_medusa_proposer.cpp` | SKIPPED / W2 |

## Gates

- **W1 (this change):** unit gate `test_draft_model_proposer` GREEN on CPU;
  equivalence invariant (accepted == target greedy) + full-acceptance +
  RED-first feed-back witness. Clean CPU `-Werror` build. Record checkers rc=0.
- **W2 (Medusa):** the `MedusaProposer` multi-head argmax propose + its unit gate.
- **W3 (DGX, later brick):** the real draft-model forward behind `DraftLogitsFn`
  (paged KV, CUDA-graph) + an e2e greedy gate on a real tiny-draft/target pair
  proving our-draft-ON == vLLM-draft-ON token-exact, plus the throughput speed
  gate (match-or-beat vLLM on every axis). DGX-offline -> deferred, honest
  residual.

## Work breakdown

1. **W0** — this spike (committed with W1).
2. **W1** — the generic draft-model greedy propose brick + unit gate + config
   parser (this change). ACTIVE.
3. **W2** — Medusa proposer (multi-head argmax) + unit gate. SPIKE row.
4. **W3** — DGX e2e greedy equivalence gate + speed gate for both methods
   (draft-model forward + Medusa heads behind the real model runner). Deferred.

## Dependencies

- Reuses `SPEC-REJECTION` (verify/accept) and the `SpeculativeConfig` scheduler
  seam — both landed. No new dependency.
- W3 depends on a real tiny-draft + tiny-target checkpoint pair and the GPU model
  runner spec loop (shared with the MTP/DFlash runner wiring).

## Risks / decisions

- **Additive + default-inert.** `draft_model_proposer.{h,cpp}` compile into
  `vllm::vllm` but are called from NO production path in W1 (no runner
  construction of a `DraftModelProposer`); with no `SpeculativeConfig` the engine
  is byte-identical. The config parser only ADDS an accepted method; existing
  methods are untouched.
- **Do not re-port the verify half.** The equivalence invariant rests entirely on
  the already-gated `RejectionSampler`; the brick only adds the proposer, mirror
  of the SPEC-NGRAM shape.
- **Medusa deferred honestly** as a SPIKE row, not implemented, because its
  multi-head target-tap propose needs the target model's Medusa heads (a model
  change) that a pure host brick cannot stand up meaningfully.
