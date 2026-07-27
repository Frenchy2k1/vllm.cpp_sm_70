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
