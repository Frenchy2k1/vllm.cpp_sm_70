# `SAMPLE-PROMPT-LOGPROBS` — the runner prompt-logits source

*(Live spec, 2026-08-09. Base `origin/main` `bd6b3936`. Pin vLLM 0.26.0.dev0
`555967922`. Issue [#223](https://github.com/mudler/vllm.cpp/issues/223). Row
`SAMPLE-PROMPT-LOGPROBS` (`.agents/engine-matrix.md:132`, `PARTIAL`). Closes the
named residual of `ROAD-V1-C7` recorded in
[roadmap-v1-completion.md](roadmap-v1-completion.md) §1 and §3.4.)*

## Scope

Port vLLM's `_get_prompt_logprobs_dict` onto our runner so
`ModelRunnerOutput.prompt_logprobs_dict` is actually populated, making
`SamplingParams.prompt_logprobs` a working feature instead of a silent no-op.

**In scope (W1):** the runner-side source — logits at prompt positions, the
`log_softmax` + gather, the chunked-prefill `in_progress` accumulation, the
`-1` → `vocab_size` widening, emit-on-final-chunk, and the request-lifetime
bookkeeping. Gated on the CPU reference backend end-to-end through the engine to
`RequestOutput.prompt_logprobs`.

**Out of scope (named residuals, own follow-ups):** the OpenAI `echo` +
`prompt_logprobs` response serialization in
`serving_completion.cpp` / `serving_chat.cpp` (W2 — the helper
`BuildCompletionLogProbs` already handles the echoed-prompt position shape,
`serving_utils.cpp:122`); `logprobs_mode` variants beyond `raw_logprobs`
(`SAMPLE-LOGPROB-TOKEN-IDS`); prompt logprobs for prompt-**embeds** requests
(upstream skips them too, `gpu_model_runner.py:5635-5637`); any GPU speed axis
(upstream itself declares this path "a rare feature, prioritize simple,
maintainable loop over optimal performance", `:5622-5623`).

## Upstream chain

- `vllm/v1/worker/gpu_model_runner.py:5612-5719` — `_get_prompt_logprobs_dict`,
  the function being ported.
- `vllm/v1/worker/gpu_model_runner.py:3841-3845` — the call site: it runs on
  `hidden_states[:num_scheduled_tokens]` with `scheduler_output.num_scheduled_tokens`,
  after the sampled ids are written back.
- `vllm/v1/worker/gpu_model_runner.py:677-679,1199,1305-1310` — the
  `num_prompt_logprobs` map: seeded at `add_request` from
  `sampling_params.prompt_logprobs` (`vocab_size` when `-1`), popped when the
  request finishes, and deleted when its prefill completes (`:5710-5712`).
- `vllm/v1/sample/sampler.py:309,396` — `compute_logprobs` (log_softmax) and
  `gather_logprobs`, reused verbatim for prompt positions.
- `vllm/v1/engine/logprobs.py:121-206` — the consumer we already ported.

## Our baseline

Everything below the runner is DONE and gated (issue #223 lists the anchors):
`LogprobsProcessor::UpdatePromptLogprobs`/`pop_prompt_logprobs`
(`src/vllm/v1/engine/logprobs.cpp:75,100`), the scheduler slice
(`src/vllm/v1/core/sched/scheduler.cpp:915-938`), and
`RequestOutput.prompt_logprobs` (`src/vllm/v1/engine/output_processor.cpp:224`).
`grep -rn prompt_logprobs src/vllm/v1/worker/` returns nothing: the source is the
whole gap.

## Design

### Where the prompt logits come from

Upstream keeps `hidden_states` and calls `self.model.compute_logits(...)` on the
prompt slice. Our forward applies `lm_head` **inside** the model and already
carries a gather list — `ModelForwardInput::logits_indices` — naming exactly which
rows of the flattened token stream get an `lm_head` row
(`src/vllm/v1/worker/gpu/runner.cpp:1143-1148`). The 1:1 equivalent of "compute
logits for these prompt positions" is therefore **to name those positions in the
gather list**, not to add a second `lm_head` seam.

`StepInputs` gains `prompt_logprob_indices` (the flat token-stream rows) plus the
per-request slicing metadata. `execute_model` hands the forward
`logits_indices ++ prompt_logprob_indices`, so the returned logits are
`[num_logits + num_prompt_rows, vocab]`: rows `[0, num_logits)` are the sampler's
(unchanged, same order), rows `[num_logits, …)` are the prompt rows in request
order.

**Inertness.** When no request asked for prompt logprobs, `prompt_logprob_indices`
is empty and the gather list handed to the forward is the *same vector* it is
today — the production path is byte-identical, with no extra allocation, no extra
row, and no extra kernel. This is the discipline the async-tap fields already
follow (`hidden_tap`/`aux_tap`, `runner.cpp:1168-1186`).

**Recorded deviation.** Naming the rows in the gather list rather than slicing
`hidden_states` changes the `lm_head` GEMM's row count when (and only when) a
request asks for prompt logprobs, so the *sampled* rows of such a step are not
guaranteed bit-identical to the same step without the flag. Upstream has the same
property for the opposite reason (it adds a second `compute_logits` call on the
same weights). It cannot affect any step of any request that did not ask.

### Per-request row selection (1:1 `:5638-5670`)

For request `r` with `num_prompt_logprobs = k`, at `num_computed_tokens = start_idx`
and `num_scheduled_tokens = num_tokens`:

```
start_tok            = start_idx + 1
num_remaining_tokens = num_prompt_tokens - start_tok
num_logits           = num_tokens <= num_remaining_tokens
                         ? num_tokens                    (a chunk; more remain)
                         : num_remaining_tokens          (final chunk -> emit)
```

`num_logits <= 0` produces nothing (the exact-prefill edge, `:5668-5671`). The
rows are `query_start_loc[req_idx] + [0, num_logits)`; the target token for row
`i` is prompt token `start_tok + i` — the *next* token, which is what the logprob
is scored against.

### Accumulation + emit (1:1 `:5646-5652,5698-5712`)

A request's `LogprobsTensors` covers `num_prompt_tokens - 1` positions and
`k + 1` columns, allocated once on first sight and held in
`GPUModelRunner::in_progress_prompt_logprobs_` (our stand-in for upstream's
`request.in_progress_prompt_logprobs_cpu`; we have no per-request state object on
the runner). Each step writes the slice `[start_idx, start_idx + num_logits)`.
The final chunk moves the whole tensor into
`ModelRunnerOutput.prompt_logprobs_dict` and drops both the in-progress entry and
the `num_prompt_logprobs` entry. Aborted/finished requests are dropped in
`update_states`' removal pass (`:1199`).

### Scoring (1:1 `:5688-5697`)

`raw_logprobs = log_softmax(prompt_logits)` via `vt::ComputeLogprobs` — the same
op the sampler uses (`sampler.cpp:284`) — then `GatherLogprobs(raw, n, vocab, k,
tgt_token_ids)`. `GatherLogprobs` is today file-local in `sampler.cpp`; it is
promoted to a declared entry point in `include/vllm/v1/sample/sampler.h` with no
change to its body, mirroring upstream's `self.sampler.gather_logprobs`. Prompt
positions bypass every logits processor, which is why upstream notes
`processed_*` and `raw_*` coincide here (`:5691-5693`).

### `prompt_logprobs == -1`

Widened to `vocab_size` at admission, exactly as upstream (`:1306-1310`). This
differs from the `num_logprobs` sample-side handling, which preserves our `-1`
sentinel for the sampler to consume — that deviation is already recorded in
`input_batch.h`; the prompt path takes upstream's form because `GatherLogprobs`
needs a concrete `k`.

## Risks

1. **Perturbing the production path.** Mitigated by strict inertness (above) and
   by a no-prompt-logprobs regression assertion in the gate.
2. **Chunked prefill off-by-one.** The `start_tok`/`num_remaining_tokens`
   arithmetic is the subtle part; the gate exercises a prompt split across
   chunks, plus the `num_logits <= 0` exact-prefill edge, both RED-first.
3. **Row ordering.** The prompt rows must be appended in the same request order
   the slicing loop assumes. Asserted in the runner and covered by a
   two-concurrent-request case.

## Tests

New `tests/vllm/v1/worker/test_prompt_logprobs.cpp`, CPU reference backend, on the
existing tiny synthetic Qwen3.5 fixture used by `test_serving.cpp`:

1. **Payload shape** — `prompt_logprobs=k` on an N-token prompt yields
   `num_prompt_tokens` positions, position 0 `None`, positions `1..N-1` carrying
   `k+1` entries (or `k` when the target is already in the top-k), with the
   prompt token itself always present. 1:1 `logprobs.py:162-187`.
2. **Values** — every `logprobs[i][tok]` equals an independently computed
   `log_softmax(logits_i)[tok]` from a direct forward in the test, and
   `rank` is the 1-based count of `>=`. RED-first: before the change the list is
   empty.
3. **Chunked prefill** — the same prompt with a chunk size that splits it emits
   the identical tensor as the single-chunk run.
4. **Exact-prefill edge** — a chunk boundary landing exactly at
   `num_prompt_tokens - 1` produces no extra rows and still emits.
5. **Two concurrent requests** with different `k` get their own tensors.
6. **Inertness** — a request without `prompt_logprobs` yields
   `prompt_logprobs == nullopt` and byte-identical sampled tokens to current main.
7. **`-1`** widens to `vocab_size` columns.

Existing gates that must stay green: `test_logprobs`, `test_serving`,
`test_llm_engine`, `test_scheduler`, `test_sampler`, `test_input_batch`.

## Gate

CPU reference backend, `ctest` full suite green, focused suite RED-first
(captured before/after). No GPU axis is claimed by this row: the feature is
correctness-only and upstream ships it as an explicitly unoptimized path.

## Evidence

Captured in the PR body: the RED run, the GREEN run, the full `ctest` summary,
and the reviewer's mutation results.

## Stop conditions

- Return `NEEDS_DECISION` if closing this needs the OpenAI `echo` serialization
  (it does not — that is W2 and its own row entry).
- Return `NEEDS_CONTEXT` if the tiny CPU fixture cannot express a chunked prefill
  (the scheduler's `max_num_batched_tokens` is settable per-harness, so it can).
- Never widen the gather list on a step where no request asked; if that becomes
  unavoidable, stop and re-spec.

## Outcome

*(filled when the row reaches `DONE`)*
