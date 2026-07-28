# ROAD-V1-C7 sampling controls + logprobs — wiring spec (`SAMPLE-CORE`, `SAMPLE-LOGIT-FILTERS`, `SAMPLE-LOGPROBS`)

*(Live spec, 2026-07-27. Base `origin/main` `489f7771`. Pin vLLM 0.26.0.dev0
`555967922`. Owner claim `CLAIM-ROADMAP-C7`. This spec covers the leaf rows
`SAMPLE-CORE`, `SAMPLE-LOGIT-FILTERS` (advanced to `ACTIVE`) and
`SAMPLE-LOGPROBS` (payload end-to-end now `DONE`, W5, `CLAIM-ROADMAP-C7-LOGPROBS`).
`SAMPLE-PROMPT-LOGPROBS` payload path is DONE; its row stays `ACTIVE` pending the
runner prompt-logits source.)*

## Scope

Wire the full OpenAI/`SamplingParams` sampling-control surface end to end so the
already-implemented device-neutral `Sampler` (a pure function of logits + params)
actually receives every control in production, and gate each transform exactly
against vLLM 0.26 on the CPU reference backend. In scope: temperature, top_p,
top_k, min_p, presence/frequency/repetition penalties, seed, stop / stop_token_ids,
min_tokens, logit_bias, allowed_token_ids, bad_words, the sample-logprobs count,
and the ordered application (`Sampler.forward` steps 1-9). Out of scope (named
residuals, separate rows): the logprobs PAYLOAD serialization end-to-end
(`SAMPLE-LOGPROBS` engine/OpenAI path), `prompt_logprobs`
(`SAMPLE-PROMPT-LOGPROBS`), `n>1` fan-out, torch-Philox exact RNG
(`SAMPLE-PHILOX`), `logprob_token_ids`/`logprobs_mode` (`SAMPLE-LOGPROB-TOKEN-IDS`),
beam search, custom logits-processor plugins.

## Upstream chain

- `vllm/sampling_params.py:264,313,318,321,337,341` — the `min_tokens`,
  `_all_stop_token_ids`, `logit_bias`, `allowed_token_ids`, `bad_words`,
  `_bad_words_token_ids` fields; `:388-413` from_optional logit_bias int-key +
  clamp; `:500` `_all_stop_token_ids.update(stop_token_ids)`; `:508-623`
  `_verify_args`; `:629-698` `update_from_generation_config` /
  `update_from_tokenizer`.
- `vllm/v1/worker/gpu_input_batch.py:269-288,435-471,573-582,686-700,819-830,889-963,1150-1155`
  — per-slot `num_logprobs` / `has_allowed_token_ids` / `allowed_token_ids_mask`
  / `bad_words_token_ids`, add/remove/condense/swap maintenance, and
  `_make_sampling_metadata`.
- `vllm/v1/sample/sampler.py:20,72,243,309,396` — the ordered pipeline + gather.
- `vllm/entrypoints/openai/completion/protocol.py:58,93,108,288-376` and
  `chat_completion/protocol.py:202,282,283,614-697` — request fields +
  `to_sampling_params`.

## Our baseline

The `Sampler` (`src/vllm/v1/sample/sampler.cpp`) already implements ALL transforms
(allowed_token_ids -> bad_words -> min_tokens -> logit_bias -> penalties ->
temperature -> min_p -> top_k/top_p -> sample -> gather_logprobs), and the ops
(`logits_processor/builtin.cpp`, `ops/bad_words.cpp`, `ops/penalties.cpp`) are
CPU-gated. The GAP was pure WIRING: `SamplingParams` deferred logit_bias /
allowed_token_ids / bad_words; `InputBatch::build_sampling_metadata` left min_p /
min_tokens / logit_bias / allowed_token_ids_mask / bad_words / max_num_logprobs at
defaults; the OpenAI protocol did not parse the three filter fields;
`InputProcessor::UpdateFromTokenizer` was a no-op.

## Port map

- `include/vllm/sampling_params.h` + `src/vllm/sampling_params.cpp` — undefer
  `logit_bias`/`allowed_token_ids`/`bad_words`/`bad_words_token_ids`/
  `all_stop_token_ids`; add bad_words + allowed_token_ids validation; seed
  `all_stop_token_ids` in `PostInit`.
- `include/vllm/entrypoints/openai/protocol.h` + `src/.../protocol.cpp` — parse
  the three fields (`ParseLogitFilters`); convert + clamp logit_bias
  (`ApplyLogitFilters`) in both `to_sampling_params`.
- `include/vllm/v1/worker/gpu/input_batch.h` + `src/.../input_batch.cpp` — per-slot
  `min_p_cpu` / `min_tokens` / `logit_bias` / `num_logprobs` /
  `has_allowed_token_ids` / `allowed_token_ids_mask` / `bad_words_token_ids`,
  add/remove/condense/swap maintenance, `build_sampling_metadata` population,
  `max_num_logprobs()`.
- `src/vllm/v1/engine/input_processor.cpp` — `UpdateFromGenerationConfig` adds eos
  to `all_stop_token_ids`; `UpdateFromTokenizer` tokenizes bad_words.

## Tests to port

- `tests/vllm/test_sampling_params.cpp` — bad_words / allowed_token_ids validation
  + `all_stop_token_ids` seeding (mirrors `test_sampling_params_e2e.py`).
- `tests/vllm/v1/worker/test_input_batch.cpp` — controls reach `SamplingMetadata`;
  default request stays empty (inertness); `-1` logprobs dominates; index-keyed
  controls follow swap/condense.
- `tests/vllm/v1/test_input_processor.cpp` — bad_words tokenization +
  `all_stop_token_ids` seeding.
- `tests/vllm/entrypoints/openai/test_protocol.cpp` — logit_bias string-key + clamp,
  allowed_token_ids, bad_words parse; non-integer key + empty allowed_token_ids
  rejection.

## Gates

- Each transform is a pure function of (logits, params); gated exactly (not
  near-tie) on the CPU reference backend via the Sampler + the metadata-wiring
  tests. RED-first: disabling a wiring line (e.g. `md.logit_bias`) fails the gate.
- Inertness: default / greedy path builds byte-identical `SamplingMetadata`
  (all filters empty, `max_num_logprobs` None) -> SACRED greedy gates unaffected.
- CPU `-Werror` clean; the seven record checkers bare RC.

## Dependencies

None beyond the landed sampler + ops + tokenizer. No GPU: the transforms are
device-neutral (the Sampler runs on the CPU Queue for the gate; the CUDA-Queue
execution is dgx-pending exactly like the rest of the sampler, not new work).

## Work breakdown

- W1 `SamplingParams` field + validation surface (DONE).
- W2 OpenAI protocol parse + logit_bias clamp (DONE).
- W3 `InputBatch` per-slot wiring + `build_sampling_metadata` + maintenance (DONE).
- W4 `InputProcessor` bad_words tokenization + eos stop-set (DONE).
- W5 (DONE 2026-07-27, `CLAIM-ROADMAP-C7-LOGPROBS`) `SAMPLE-LOGPROBS` payload:
  ported `vllm/logprobs.py` (`Logprob`/`LogprobsOnePosition`/
  `append_logprobs_for_next_position` -> `include/vllm/logprobs.h`) +
  `vllm/v1/engine/logprobs.py` (`LogprobsProcessor` ->
  `include/vllm/v1/engine/logprobs.h`+`src/.../logprobs.cpp`). Threaded
  `SamplerOutput.logprobs_tensors` -> `ModelRunnerOutput.logprobs` (runner) ->
  `LogprobsTensors::slice_request` -> `EngineCoreOutput.new_logprobs` (scheduler)
  -> `OutputProcessor` LogprobsProcessor -> `CompletionOutput.logprobs` ->
  OpenAI `CompletionLogProbs`/`ChatCompletionLogProbs` serialization
  (`BuildCompletionLogProbs`/`BuildChatLogprobs` in `serving_utils.cpp` +
  `to_json` in `protocol.cpp`). Gated on the CPU reference engine
  (`test_openai_serving`) + a vLLM-0.26 serialization oracle
  (`test_openai_logprobs`, RED-first). The prompt-logprobs OUTPUT path
  (`_update_prompt_logprobs`/`pop_prompt_logprobs` -> `RequestOutput.prompt_logprobs`
  -> serving) is DONE too, but `SAMPLE-PROMPT-LOGPROBS` stays `ACTIVE`: the runner
  does not yet compute prompt-position logits (the tensor SOURCE — a runner/prefill
  addition adjacent to C5). `echo` prompt-prepend + the `num_logprobs==-1`
  full-vocab payload shape remain deferred (finite-K is the gated path).

## Risks/decisions

- DECISION: `num_logprobs` keeps our `-1` ("all") sentinel rather than upstream's
  `vocab_size`, because our `Sampler::forward` branches on `-1` directly; recorded
  as a deviation. `max_num_logprobs()` lets `-1` dominate any finite request.
- DECISION: the `allowed_token_ids` empty-list + logit_bias non-integer-key checks
  are enforced in `Verify()` / `ApplyLogitFilters` (surface as a 400); the
  model-config vocab-range check stays engine-time (deferred), matching upstream's
  split between `_verify_args` and `verify(model_config)`.
- RISK: the index-keyed control maps (min_tokens / logit_bias / bad_words) must
  follow the row through condense/swap; mirrored on `swap_dict_values` /
  pop-reinsert and gated by the swap/condense test.

## `SAMPLE-CUSTOM-PROCESSORS` — custom logits processors (ACTIVE, `CLAIM-C7-CUSTOM-LOGITS`)

Closes the C7 `SAMPLE-CUSTOM-PROCESSORS` inventory row AND the SGLANG-DISTINCT
`SGLANG-SAMPLING-CUSTOM` row (same capability). A host-registered per-request
callback the sampler invokes each decode step, BEFORE sampling, to modify the
request's logits.

- Upstream chain: vLLM `SamplingParams.logits_processors` +
  `vllm/v1/sample/logits_processor/interface.py:60` (`LogitsProcessor.apply`) +
  application at `vllm/v1/sample/sampler.py:399` (the non-argmax-invariant loop,
  AFTER allowed_token_ids/bad_words and the builtin min_tokens/logit_bias, BEFORE
  penalties; custom procs append after builtins in the manager,
  `logits_processor/state.py:152-165`). SGLang
  `python/sglang/srt/sampling/custom_logit_processor.py:24`
  (`CustomLogitProcessor`).
- Port map (deviation, recorded): we do NOT port vLLM's `LogitsProcessors` plugin
  object graph nor SGLang's dill-serialized Python callable. Instead we carry a
  single per-request C-ABI callback (fn ptr + `void* user_data`), exposed as
  `vllm_logits_processor` (`include/vllm.h`, ABI bumped 7→8). Carrier struct
  `vllm::LogitsProcessorCallback` (`include/vllm/logits_processor_callback.h`),
  field on `vllm::SamplingParams.logits_processor` and
  `SamplingMetadata.logits_processors` (req_index -> callback), applied by
  `apply_logits_processors` (`src/vllm/v1/sample/logits_processor/builtin.cpp`)
  wired at `src/vllm/v1/sample/sampler.cpp` right after `apply_logit_bias`. The
  callback receives the generated token-ids so far + a mutable f32 logits row
  view [vocab] and edits in place. On a unified-memory backend (CPU/GB10) the row
  is edited in place; on a discrete backend it stages down/up.
- Contract: fn == nullptr (default) => byte-identical to a build with no
  processor. `InputBatch` tracks the callback per-slot and follows the row through
  remove/condense/swap exactly like logit_bias; `needs_output_token_ids` and the
  metadata rebuild gate now also fire when a processor is registered.
- Gates: `tests/vllm/v1/sample/test_sampler.cpp` (forces-token EXACT + per-request
  + empty-map inert, RED-first: reverting the sampler wiring flips the forced
  token back to the baseline argmax); `test_logits_processors.cpp` (mutate row /
  no-op empty / null-fn skip); `tests/capi/test_capi.cpp` (ABI v8 end-to-end: the
  processor forces the generated token, fires once per decode step, differs from
  the untouched greedy baseline). Default inertness = the pre-existing SAMPLE-*
  gates unchanged.
- RESIDUAL (honest): a single per-request C callback, not a batched multi-processor
  plugin graph; no Python-side registration; on the async scheduler the generated
  output-token view the callback sees is fed back by the scheduler (may lag) — the
  strict per-step token-ids contract is gated deterministically at the sampler
  level, not through the async engine.

## `SAMPLE-N` — parallel sampling / `n>1` fan-out (ACTIVE, `CLAIM-C7-N-SAMPLING`)

Closes the C7 `SAMPLE-N` inventory row: the OpenAI `n` sampling parameter
(multiple output sequences per request), previously `n==1`-only.

- Upstream chain: a request with `n>1` is expanded by the frontend into n CHILD
  requests that SHARE the prompt (and its prefill KV) — `ParentRequest`
  (`vllm/v1/engine/parallel_sampling.py:13`) hands out child ids
  `"{index}_{parent}"` + n==1 child sampling params (seeded children get
  `seed+index`, `:52-81`), the `LLMEngine` fans out
  (`vllm/v1/engine/llm_engine.py:270-293`; `n==1` returns before the fan-out), and
  the `OutputProcessor` aggregates the n child `CompletionOutput`s back into one
  `RequestOutput` via `parent_req.get_outputs`
  (`vllm/v1/engine/output_processor.py:323-331`), popping the parent once its last
  child finishes (`:720-722`). The offline driver sets `output_kind = FINAL_ONLY`
  (`vllm/entrypoints/offline_utils.py:561`), the aggregation mode.
- Our port: `vllm::v1::ParentRequest`
  (`include/vllm/v1/engine/parallel_sampling.h` + `.cpp`) mirrors
  `get_child_info`/`_get_child_sampling_params`/`get_outputs` 1:1;
  `LLMEngine::FanOutParallelSampling` (`src/vllm/v1/engine/llm_engine.cpp`) is the
  n>1 branch of both `add_request` overloads (n==1 falls through untouched);
  `RequestState.parent_req` + the `make_request_output` aggregation branch +
  `parent_requests_` cleanup (`src/vllm/v1/engine/output_processor.cpp`); the
  OpenAI layer already builds one indexed `choice` per `CompletionOutput`
  (`src/vllm/entrypoints/openai/serving_completion.cpp`), and `n` already rode on
  `SamplingParams` (`include/vllm/sampling_params.h:128`) + `to_sampling_params`
  (`src/vllm/entrypoints/openai/protocol.cpp:398,433`).
- Determinism gate design: vLLM FORBIDS `n>1` under greedy sampling
  (`vllm/sampling_params.py:625` `_verify_greedy_sampling`, which we mirror in
  `SamplingParams::VerifyGreedySampling`). So the clean exact n>1 gate uses
  `top_k=1` sampling — a LEGAL n>1 config whose sampling collapses to the argmax,
  i.e. every child is token-identical to the single greedy result.
- Gates: `tests/vllm/v1/test_llm_engine.cpp` ("n>1 fans out into n token-identical
  deterministic outputs" — RED-first: before the fan-out an n>1 request returns
  exactly ONE output (`REQUIRE(1 == 4)`); GREEN after: n outputs, each index 0..n-1
  token-identical to BOTH the top_k=1 and the greedy single-sequence result; the
  other 7 `test_llm_engine` cases + `test_output_processor` prove n==1 inertness);
  `tests/vllm/entrypoints/openai/test_serving.cpp` ("n>1 returns n indexed
  deterministic choices" — n choices, index fields 0..n-1, identical text, usage
  counts all n).
- RESIDUALS (honest, named): `best_of` (`SAMPLE-BEAM` / `SERVE-COMPLETION-LONGTAIL`)
  and beam search stay INVENTORIED; the async-streaming per-child collation through
  `RequestOutputCollector::Merge` is structurally present but the AsyncLLM n>1 wiring
  path is not gated (sync + non-streaming OpenAI paths are); the C-ABI carries no
  `n` field (needs an ABI bump + a multi-output return shape) so C-ABI requests stay
  `n==1`. Prompt tokens are shared; prefill-KV sharing rides on the existing
  block-hash APC (no dedicated n>1 KV-share optimization added).

## `SAMPLE-BEAM` — beam search (ACTIVE, `CLAIM-C7-BEAM`)

Closes the C7 `SAMPLE-BEAM` inventory row: beam search as an OUTER loop over the
engine (it is NOT a core-sampler param), the sibling multi-sequence feature to
`SAMPLE-N` — it REUSES the multi-output return, does not perturb the n>1 / n=1 /
plain-sampling paths.

- Upstream chain: `BeamSearchParams` (`vllm/sampling_params.py:1114`:
  `beam_width`, `max_tokens`, `ignore_eos`, `temperature`, `length_penalty`);
  `BeamSearchSequence` / `BeamSearchOutput` / `BeamSearchInstance` /
  `get_beam_search_score` / `create_sort_beams_key_function`
  (`vllm/entrypoints/generate/beam_search/utils.py:18,102,112,137,156`); the
  driver `BeamSearchOfflineMixin.beam_search` + `_beam_search_step`
  (`vllm/entrypoints/generate/beam_search/offline.py:58,193`). The per-step loop:
  run ONE decode per beam with `SamplingParams(logprobs=2*beam_width, max_tokens=1,
  temperature=temperature)` (`offline.py:118-123`), read each beam's single
  generated position's `logprobs[0]` dict (`offline.py:303`), form a candidate per
  `(token_id, logprob)` with `cum_logprob += logprob` (`offline.py:308-314`), send
  EOS candidates (`token_id == eos && !ignore_eos`) to `completed`
  (`offline.py:316-317`) and the rest to `new_beams`, sort `new_beams` by the
  length-penalty score DESCENDING and keep the top-`beam_width`
  (`offline.py:320-325`); stop at `max_tokens` or when all beams are exhausted
  (`offline.py:219-220`). Final: append the surviving active beams to `completed`,
  sort DESCENDING, return the top-`beam_width` (`offline.py:178-189`).
- Scoring formula (the correctness content, `utils.py:137-153`):
  `get_beam_search_score(tokens, cum_logprob, eos, length_penalty) = cum_logprob /
  (seq_len ** length_penalty)` where `seq_len = len(tokens)` (which INCLUDES the
  prompt, since `BeamSearchSequence.tokens` starts at the prompt ids) minus one
  when `tokens[-1] == eos`. `create_sort_beams_key_function` closes over
  `(eos, length_penalty)` and sorts by this key.
- Our port: `include/vllm/entrypoints/beam_search.h` + `.cpp` (namespace `vllm`).
  The correctness content is a MODEL-FREE core — `get_beam_search_score`,
  `SortBeamsKey`, `BeamSearchStep(instance, per_beam_logprobs, eos, ignore_eos,
  beam_width, length_penalty)` (mirrors `_beam_search_step`'s non-structured path
  1:1) — plus a thin engine driver `BeamSearch(LLMEngine&, prompt_tokens, params,
  eos, tokenizer?)` that sources each step's per-beam next-token dict from
  `LLMEngine::generate(beam.tokens, SamplingParams{logprobs=2*beam_width,
  max_tokens=1, temperature})` and calls `BeamSearchStep`, then does the final
  sort/select. `std::stable_sort` with a strict-`>` comparator reproduces Python's
  `sorted(reverse=True)` EXACTLY (Timsort is stable; `reverse=True` keeps the input
  order of equal-key elements).
- Determinism gate design: beam search under `temperature=0` (the default) is
  greedy per-beam ⇒ fully DETERMINISTIC, so it is gated token-for-token. The
  MANDATORY gate is model-free: a hand-computed beam tree (`beam_width=2`, a fixed
  toy logprob table, two full steps) whose expected beams/tokens/scores/EOS
  routing/length-penalty are computed by hand and asserted against `BeamSearchStep`
  + `get_beam_search_score` EXACTLY — this is the real correctness content and
  needs no model.
- Gates: `tests/vllm/entrypoints/test_beam_search.cpp` (5 cases / 45 assertions):
  (1) `get_beam_search_score` exact — no-eos, EOS seq_len decrement, `length_penalty
  != 1.0`, `eos == nullopt`; (2) two full steps token-EXACT vs the reference tree
  (expand → score → keep top-`beam_width` → EOS→completed) + the final top-k tail;
  (3) `ignore_eos` keeps the EOS token as a normal beam; (4) a null per-beam dict
  (completed/aborted sequence) yields no continuations; (5) an exhausted instance
  stops. **RED-first:** with `BeamSearchStep` stubbed to no-op the toy fails
  `REQUIRE(instance.beams.size() == 2)` (`1 == 2`) and the e2e fails `1 == 5`; the
  real expand/select flips both to GREEN. Plus `tests/vllm/v1/test_llm_engine.cpp`
  ("beam search returns beam_width scored continuations, bw=1 == greedy") — e2e over
  the synthetic CPU Qwen3.6 engine: `beam_width` distinct full-length continuations
  of the prompt ordered by descending score, `beam_width=1` beam search is
  token-identical to plain greedy `generate`, and the wider beam finds `>=`
  cumulative logprob.
- Inertness: the sampling/serving suites stay byte-identical — the feature is a NEW
  additive TU + header with ZERO edits to any existing compiled path (the only test
  edit is a new case + one include in `test_llm_engine.cpp`): `test_sampler` 11/11,
  `test_output_processor` 8/8, `test_scheduler` 36/36, `test_openai_serving` 28/28
  UNCHANGED; the n>1 / n=1 / plain-sampling paths are untouched.
- RESIDUALS (honest, named): the OpenAI-endpoint `use_beam_search` / `best_of`→beam
  mapping and the C-ABI beam params are NOT wired (the `LLM`-API `BeamSearch` driver
  is the entry point); grammar-constrained beam search (the structured-output
  bitmask branch of `_beam_search_step`, `offline.py:222-258,329-455`) and the
  encoder-decoder / LoRA beam variants are out of scope; the multi-prompt batch loop
  (`offline.py:141-173`) is single-prompt here.
  UPDATE 2026-07-28 (`CLAIM-C7-BESTOF-BEAM-API`): the `use_beam_search`→beam and
  `best_of`→fan-out ENDPOINT mappings are now WIRED (see the section below); the
  C-ABI beam params + grammar/enc-dec/LoRA beams remain residual.

## `SAMPLE-BEST-OF` + `use_beam_search` — the OpenAI-endpoint surface (ACTIVE, `CLAIM-C7-BESTOF-BEAM-API`)

Wires the two named `SAMPLE-N`/`SAMPLE-BEAM` endpoint residuals so the C7 sampling
track is usable end-to-end from the HTTP API. Strict default-path inertness: a
request with NEITHER `best_of>n` NOR `use_beam_search` set is byte-identical to
before.

### Honest vLLM-0.26 endpoint-availability finding (`555967922`)
- `use_beam_search` **IS** a real vLLM-0.26 OpenAI-server surface, on BOTH endpoints:
  `vllm/entrypoints/openai/completion/protocol.py:73` (field) + `:260`
  (`to_beam_search_params`); `vllm/entrypoints/openai/completion/serving.py:173-205`
  (`if request.use_beam_search: … self.beam_search(...)`);
  `vllm/entrypoints/openai/chat_completion/protocol.py:249` + `:589`;
  `vllm/entrypoints/openai/chat_completion/serving.py:319-343`. The per-beam
  generator is `vllm/entrypoints/generate/beam_search/online.py:28-220` (yields ONE
  `RequestOutput` whose `outputs` list is the `beam_width` best beams → choices).
- `best_of` **HAS BEEN DROPPED** from vLLM 0.26's live path. `grep -rn best_of vllm/`
  finds it ONLY at `vllm/entrypoints/openai/chat_completion/protocol.py:1048`, a
  vestigial field on `BatchChatCompletionRequest` that is **never consumed** (no
  `SamplingParams.best_of`, no ranking, no `best_of` on `CompletionRequest`/
  `ChatCompletionRequest`). We therefore implement the CLASSIC OpenAI-spec / vLLM-V0
  `best_of` contract (generate `best_of`, return the `n` highest-cumulative-logprob),
  gated on OUR deterministic fan-out — there is no 0.26 `best_of` oracle to gate
  token-exactness against, which is recorded honestly rather than fabricated.

### Port map
- `include/vllm/entrypoints/openai/protocol.h`: `best_of` / `use_beam_search` /
  `length_penalty` fields on `CompletionRequest` + `ChatCompletionRequest` (best_of on
  chat is a recorded deviation — upstream carries it only on Batch); the
  `to_beam_search_params(int)` declarations.
- `src/vllm/entrypoints/openai/protocol.cpp`: `from_json` parses the three fields;
  `ApplyBestOf` (anon ns) maps `best_of`→`sp.n` fan-out count + forces `sp.logprobs=0`
  for cumulative-logprob ranking (AFTER `sp.logprobs` is assigned, so a user count is
  not clobbered) + rejects `best_of<n`; `to_beam_search_params` mirrors
  completion/protocol.py:260 & chat_completion/protocol.py:589 (`beam_width=n`,
  `temperature` None→1.0, `length_penalty`, `ignore_eos`).
- `include/vllm/entrypoints/openai/serving_utils.h` + `.cpp`: `SelectBestOf` — stable
  sort by DESCENDING cumulative logprob, keep top-n, re-index 0..n-1; INERT (returns
  input unchanged, indices preserved) when nothing to trim.
- `src/vllm/entrypoints/openai/serving_completion.cpp` + `serving_chat.cpp`:
  `use_beam_search` routing (tokenize the (chat-rendered) prompt → `BeamSearch(sync
  LLMEngine, …)` → `beam_width` beams as choices; reject streaming beam as upstream
  serving.py:136); `set_beam_search_tokenizer` seam (the driver needs a tokenizer +
  eos); the `best_of` top-n trim guarded on `request.best_of` (default path binds the
  outputs with NO copy/re-rank).

### Gates (CPU, exact, RED-first) — `tests/vllm/entrypoints/openai/test_serving.cpp`
- `SelectBestOf` unit: ranks by cumulative logprob, keeps top-n, re-indexes, stable on
  ties, INERT when return_n≥size.
- `best_of`→`sp.n` + forced ranking logprob (RED-first: before, best_of was an unknown
  ignored field); user logprob count NOT clobbered; `best_of<n` rejected.
- e2e `best_of=4, n=2` (top_k=1 collapse) returns EXACTLY n=2 ranked, re-indexed
  choices; `best_of` unset / `==n` inertness (params identical, no forced logprob).
- `use_beam_search` `to_beam_search_params` round-trip; endpoint beam choices IDENTICAL
  to the direct `BeamSearch` driver call over a fresh identical engine (completion +
  chat); streaming+beam rejected.
- Inertness: the full `test_openai_serving` (37 cases, incl. n>1 / logprobs / streaming
  / stop) stays byte-identical; clean CPU `-Werror` 0-warn.

### RESIDUALS (honest, named)
- Beam search over the PRODUCTION AsyncLLM HTTP server: the merged `BeamSearch` driver
  is `LLMEngine&`-based (sync), the production `examples/server/main.cpp` runs AsyncLLM
  → `set_beam_search_tokenizer` is a no-op there and a beam request raises "requires the
  synchronous engine". Needs an async `BeamSearch` driver (a separate porting task,
  mirroring `online.py`).
- Streaming beam search (upstream also rejects it, serving.py:136).
- C-ABI `best_of`/beam fields (ABI bump), AsyncLLM streaming per-child best_of
  collation, grammar-constrained beams.
