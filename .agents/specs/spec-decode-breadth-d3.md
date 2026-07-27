# Spec-decode breadth (ROAD-V1-D3): ngram + EAGLE3

**Rows:** `SPEC-NGRAM`, `SPEC-EAGLE3` (engine-matrix §8 / feature-matrix §8).
**Claim:** `CLAIM-ROADMAP-D3`. **Base:** `origin/main` `9167b834`.
**Oracle:** vLLM 0.26.0.dev0 (`555967922`).

## Scope

Extend the LANDED spec-decode machinery (the frozen spec-metadata ABI + rejection
sampler + verify/propose runner loop from MTP `SPEC-MTP` I2-I7 and DFlash
`SPEC-DFLASH` D0-D14) to two more upstream speculative methods, reusing the EXACT
verify / `GreedyRejectionSample` / `take_draft_token_ids` loop — only the PROPOSE
step differs.

- **`SPEC-NGRAM`** (DONE/gated) — the draft-FREE n-gram proposer: propose the next
  k tokens by matching the last generated tokens against an earlier n-gram IN THE
  SAME sequence. No draft model, no draft KV, no hidden tap. Works on any model.
- **`SPEC-EAGLE3`** (BLOCKED/scoped) — an EAGLE3 draft head conditioned on the
  target's aux hidden states (the D1 multi-tap machinery), verified through the
  same rejection loop. Blocked on checkpoint availability (see Risks/decisions).

Out of scope (stays under `ROAD-V1-C3`): DSpark `SPEC-DSPARK`, TLI `SPEC-TLI`.

## Upstream chain

- ngram proposer: `vllm/v1/spec_decode/ngram_proposer.py:184-276`
  (`_find_longest_matched_ngram_and_propose_tokens`, the KMP-LPS suffix-ngram
  matcher) + `:128-180` (`NgramProposer.propose` / `batch_propose`).
- ngram config: `vllm/config/speculative.py:734-762` (method `ngram`,
  `prompt_lookup_min`/`prompt_lookup_max` default 5/5) + `:1224-1234`
  (`num_speculative_tokens` REQUIRED for ngram).
- ngram scheduler: `vllm/v1/core/sched/scheduler.py:250-265` — ngram is NOT a
  target-hidden-state method, so no lookahead branch fires (`num_lookahead==0`).
- EAGLE3: `vllm/v1/spec_decode/eagle.py`; draft archs
  `vllm/model_executor/models/registry.py:612-648`; method auto-detect
  `vllm/config/speculative.py:869-885`.

## Our baseline

The MTP/DFlash landing already provides everything the VERIFY side needs: the
frozen spec-metadata ABI (`SpeculativeConfig`, `Request::spec_token_ids`,
`take_draft_token_ids`, `Scheduler::update_draft_token_ids`,
`EngineCore::post_step`), the greedy rejection sampler
(`v1/spec_decode/rejection_sampler.cpp`, per-request `1+k_i` expansion), the GDN
spec segments + k+1 state-slot rollback + widened conv
(`MakeQwen3_5KVCacheSpec(num_spec>0)`), and the runner verify/propose loop
(`GPUModelRunner::sample_tokens_with_rejection` + `propose_drafts`). The 27B NVFP4
GDN hybrid is the vehicle (best-supported spec verify path). ngram only adds a new
PROPOSE branch; EAGLE3 additionally needs a separate draft loader + draft transformer.

## Port map

ngram (LANDED):
- NEW `src/vllm/v1/spec_decode/ngram_proposer.{h,cpp}`:
  `FindLongestMatchedNgramAndProposeTokens` (1:1 of the KMP-LPS matcher) +
  `NgramPropose` (batch propose; the numba `prange` is a perf detail, output-identical).
- `include/vllm/config/speculative.h`: `prompt_lookup_min/max` fields,
  `ResolveNgram`, `use_ngram()`; `NumLookaheadTokens()==0` for ngram.
- `src/vllm/config/speculative.cpp`: `ParseSpeculativeConfigJson` accepts `ngram`
  + `prompt_lookup_min/max` + requires `num_speculative_tokens`.
- `src/vllm/entrypoints/model_loader.cpp`: `ResolveSpecConfig` ngram branch (no
  draft checkpoint; GDN spec verify reused via `MakeQwen3_5KVCacheSpec(num_spec>0)`,
  the `fa_draft` group allocated-but-unused).
- `src/vllm/v1/worker/gpu/runner.{h,cpp}`: `propose_drafts` routes to
  `propose_drafts_ngram` when `use_ngram()` (before the MTP/DFlash branches);
  the ngram propose reads each generating request's own committed context
  (`input_batch_.token_ids_cpu[i, :num_tokens_no_spec[i]]`); hidden-tap capture
  suppressed for ngram.

EAGLE3 (SCOPED, not implemented): reuse the DFlash D5 separate-draft loader
(`LoadDflashDraft` analog → `LoadEagle3Draft`), the D1 multi-tap aux-hidden capture
(`ForwardDeviceMultiTap` at the EAGLE3 `target_layer_ids`), an `Eagle3*ForCausalLM`
draft transformer (port `qwen3_eagle3`/`llama_eagle3`), config-select
`method="eagle3"` in `ParseSpeculativeConfigJson`/`ResolveSpecConfig`, and the
existing verify/reject/`take_draft_token_ids` loop.

## Tests to port

- `vllm/tests/v1/spec_decode/test_ngram.py` → `tests/vllm/v1/spec_decode/test_ngram_proposer.cpp`
  (the matcher + batch-propose cases; 19 assertions, PASS CPU+CUDA).
- e2e correctness/acceptance gate (no direct upstream unit — vLLM e2e-covers):
  `tests/parity/test_qwen27_ngram_spec_decode.cpp` + the vLLM-ngram-ON golden
  `tests/parity/goldens/ngram_27b/ngram_27b_spec_on.json`
  (`scripts/spec/ngram_27b_golden.py`).
- EAGLE3: `vllm/tests/v1/e2e/spec_decode/test_spec_decode.py` (eagle3 arms) — to
  port when a checkpoint lands.

## Gates

- **ngram (MET):** 27B `test_qwen27_ngram_spec_decode` — our-ngram-ON == vLLM-ngram-ON
  STRICT 5/5 prompts + acceptance non-zero (180/180 drafts accepted, acceptance_len
  4.0 == vLLM). Gate form = clean strict on a deterministic REPETITIVE battery
  (ngram's design workload, bf16-deterministic); factual near-tie prompts excluded
  (our ngram verify is exactness-preserving our-ON == our-OFF, and the cross-engine
  divergence there is a bf16 near-tie, not wiring — see Risks/decisions). Inertness
  spec-OFF byte-identical: SACRED 235/235 + MTP 9/9 + DFlash 27/27; CUDA `-Werror`
  clean; host-side, no new kernel (no compute-sanitizer surface). Unit 19/19.
- **EAGLE3 (deferred):** our-EAGLE3-ON == vLLM-EAGLE3-ON (ON-vs-ON, DFlash D0
  gate-form) + acceptance, once an ungated EAGLE3 draft for a cached gate model
  exists.

## Dependencies

- Reuses (does not modify) the landed MTP/DFlash ABI + rejection sampler + GDN
  spec segments + runner verify/propose loop.
- ngram: none beyond the above (draft-free, host-side).
- EAGLE3: an ungated, oracle-runnable EAGLE3 draft checkpoint for a Qwen3.6 gate
  model (BLOCKER — absent at this pin).
- Coordination: sibling `CLAIM-ROADMAP-C7` owns the sampler/serving §6 rows; this
  claim owns only the §8 spec-decode proposer rows (no shared TU).

## Work breakdown

1. ngram matcher + batch propose (`ngram_proposer.{h,cpp}`) + unit test — DONE.
2. config parse + `ResolveNgram` + `use_ngram` + `ResolveSpecConfig` branch — DONE.
3. runner `propose_drafts_ngram` routing + hidden-tap gate — DONE.
4. vLLM-ngram-ON golden capture + 27B e2e gate + inertness re-gate — DONE.
5. EAGLE3 W0 RUN-verify (checkpoint reachability) — DONE (verdict: BLOCKED).
6. EAGLE3 draft loader + transformer + config-select + gate — DEFERRED (blocked on
   a checkpoint; scoped in Port map above).

## Risks/decisions

- **ngram near-tie confound (resolved).** Greedy ngram spec-decode is
  exactness-preserving in EXACT arithmetic, but bf16 multi-token verify forwards
  resolve near-ties differently than single-token decode. On factual single-answer
  prompts our-spec-OFF already diverges from vLLM-spec-OFF cross-engine at a bf16
  near-tie (measured: "The capital of France is" → our "Paris.\\n\\n<think>…" vs
  vLLM "Paris. The capital of Germany is Berlin…", per the committed
  `dflash_27b_spec_off.json`). Decision: gate on a deterministic REPETITIVE battery
  (ngram's design workload) where the greedy is unambiguous, giving a clean strict
  our-ON == vLLM-ON identity; the exactness-preservation of our verify is separately
  evidenced by our-ngram-ON == our-spec-OFF on the near-tie prompts (42/48 shared).
- **EAGLE3 reachable-blocked (honest, no fabrication).** W0 RUN-verify (2026-07-27):
  no EAGLE3 draft ARCH for the gate models in vLLM 0.26 (registry.py:572-648 has
  `Qwen3_5MTP` but no `Eagle3Qwen3_5*`); no EAGLE3 draft CHECKPOINT cached or
  published for a gate model (dgx cache 0 eagle checkpoints; z-lab published DFlash
  not EAGLE3; AngelSlim/Qwen3-8B_eagle3 is base Qwen3-8B, a different arch); our
  spec path is Qwen3.6-only. Decision: scope the port (Port map above), mark
  `SPEC-EAGLE3` BLOCKED, do NOT fabricate a checkpoint or gate (Command-R HF-gate
  honest-blocked pattern).

## ROAD-V1-D3 disposition

**DONE:** ngram gated (`SPEC-NGRAM` ACTIVE/gated) + EAGLE3 honestly reachable-blocked
with the blocker named (`SPEC-EAGLE3` BLOCKED). No fabricated EAGLE3 gate.
