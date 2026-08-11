# `SAMPLE-LOGPROB-TOKEN-IDS` — `logprob_token_ids` generative scoring

*(Live spec, 2026-08-10. Base `origin/main` `e63d11d3`. Pin vLLM 0.26.0.dev0
`555967922`. Issue [#264](https://github.com/mudler/vllm.cpp/issues/264). Row
`SAMPLE-LOGPROB-TOKEN-IDS` (`.agents/engine-matrix.md:133`), `INVENTORIED` ->
`PARTIAL`.)*

## Scope

`SamplingParams.logprob_token_ids` asks for logprobs at an EXPLICIT set of vocab
ids instead of the top-k. vLLM's generative-scoring endpoint drives it
(`entrypoints/generate/generative_scoring/serving.py:247-255` sends
`max_tokens=1` + `logprob_token_ids=label_token_ids`), and it is the efficient
answer to "score these five labels" — no `logprobs=-1`, no full-vocab sort.

In scope:

- the `SamplingParams` field, its `num_logprobs()` derivation, and the two
  validations that need no model config;
- `InputBatch` per-request plumbing (`add_request` / `remove_request` /
  `make_sampling_metadata`), mirroring the `num_logprobs` map shape;
- `Sampler::gather_specific_token_logprobs`, ported 1:1;
- the raw-logprobs snapshot condition (`sampler.py:86`) and the
  explicit-ids-win precedence rule (`sampler.py:133-136`);
- the two `num_logprobs`-property consumers our port hard-coded to
  `sampling_params.logprobs` (scheduler slice gate, `LogprobsProcessor`).

Out of scope, named as residuals:

- **The OpenAI `logprob_token_ids` request field** on `/v1/completions` and
  `/v1/chat/completions` (`completion/protocol.py:95,356-361,448-468`,
  `chat_completion/protocol.py:270,679-683,748-757`) and the
  `/v1/generative_scoring` endpoint itself. This change lands the library
  surface (`SamplingParams` -> engine -> `CompletionOutput.logprobs`); the HTTP
  request fields and their four cross-field validations are a separate leaf.
- **`logprobs_mode` variants beyond `raw_logprobs`** — the row's other half.
  PR [#258](https://github.com/mudler/vllm.cpp/pull/258) is OPEN for it and
  touches the same snapshot block in `sampler.cpp`; this change deliberately
  does not pre-empt it. That is why the row lands `PARTIAL`, not `ACTIVE`.
- **The vocab-range validation** of the requested ids, which upstream performs
  in `verify(model_config)` (`sampling_params.py:783-793`). Our `Verify()` has
  no model config, exactly as for `allowed_token_ids`, whose vocab-range check
  is already a recorded engine-time deferral. The sampler bounds the ids anyway
  (see Design) so an out-of-range id is a loud error, never a read past the row.
- **Issue [#249](https://github.com/mudler/vllm.cpp/issues/249)** — `k <= vocab`
  is unbounded in `GatherLogprobs`. NOT fixed here; a separate row owns it.

## Upstream chain

Read at the pin, not from memory.

- `vllm/sampling_params.py:31-33` — `MAX_LOGPROB_TOKEN_IDS = 128`, "must match
  the per-request row width allocated by the sampler's `LogprobTokenIdsState`".
- `vllm/sampling_params.py:278-283` — the field: `logprob_token_ids: list[int] |
  None`, "more efficient than `logprobs=-1` when you only need logprobs for a
  small set of tokens".
- `vllm/sampling_params.py:724-729` — the `num_logprobs` property: `logprobs` if
  set, else `len(logprob_token_ids)` if set, else `None`. This is what the rest
  of the engine reads, NOT the raw `logprobs` field.
- `vllm/sampling_params.py:772-801` — `_validate_logprobs`, the
  `logprob_token_ids` half: length `<= MAX_LOGPROB_TOKEN_IDS`; every id in
  `[0, vocab_size)`; and when both are set, `logprobs == len(logprob_token_ids)`.
- `vllm/v1/sample/sampler.py:86` — `if num_logprobs is not None or
  sampling_metadata.logprob_token_ids:` gates the raw snapshot.
- `vllm/v1/sample/sampler.py:111-136` — the gather call, the three-way
  `num_logprobs` branch, and the precedence override: `if
  logprob_token_ids_tensors is not None and num_logprobs is not None:
  logprobs_tensors = logprob_token_ids_tensors`.
- `vllm/v1/sample/sampler.py:151-225` — `gather_specific_token_logprobs`: the
  padded `[batch, max_num_tokens + 1]` id matrix, sampled token in column 0
  (always valid), each request's ids at `1 : n+1`, `gather(-1, …)`, padded
  positions `masked_fill(-inf)`, and `token_ranks =
  batched_count_greater_than(logprobs, sampled_logprobs)` over the FULL vocab.
- `vllm/v1/sample/metadata.py:49` — `logprob_token_ids: dict[int, list[int]] |
  None`, keyed by req_INDEX.
- `vllm/v1/worker/gpu_input_batch.py:273` (`dict[str, list[int]]`, req_ID-keyed),
  `:443-444` (`add_request`), `:574` (`remove_request` pop), `:934-951`
  (`make_sampling_metadata` converts id -> index over the live batch only).
- `vllm/v1/core/sched/scheduler.py:1815-1821` — the slice gate reads
  `request.sampling_params.num_logprobs is not None`, the property.
- `vllm/v1/engine/logprobs.py` / `vllm/logprobs.py:175-206` — the consumer, which
  is already shared and needs no change beyond being handed the property.

Upstream ALSO carries a second, newer implementation of the same feature in the
V2 worker (`vllm/v1/worker/gpu/sample/logprob.py:114-235`, a Triton kernel plus
a `LogprobTokenIdsState` GPU buffer). That is the V2 sampler stack; our sampler
mirrors the V1 `vllm/v1/sample/sampler.py` one, so the V1 path is the port
target. Noted so the next reader does not think an anchor was missed.

## Our baseline

- `include/vllm/v1/sample/metadata.h:107-109` already DECLARES
  `std::optional<std::map<int, std::vector<int32_t>>> logprob_token_ids`, in a
  block titled "STUBS". Nothing writes it, nothing reads it.
- `include/vllm/v1/sample/sampler.h:30-32` lists the gather as DEFERRED.
- `src/vllm/v1/sample/sampler.cpp:275-278` says so in a comment: "logprob_token_ids
  (generative-scoring) is a deferred stub, so the snapshot is driven solely by
  max_num_logprobs."
- `include/vllm/sampling_params.h:20` lists `logprob_token_ids, flat_logprobs,
  num_logprobs()` among the deferred fields; the struct does not carry it.
- `src/vllm/v1/engine/logprobs.cpp:35` reads `sampling_params.logprobs` with the
  trailing comment `// sampling_params.num_logprobs`. The comment names the
  upstream property; the code is the raw field. Identical today, divergent the
  moment `logprob_token_ids` exists.
- `src/vllm/v1/core/sched/scheduler.cpp:920` gates the logprobs slice on
  `sampling_params.logprobs.has_value()` where upstream gates on the property.

So the seam is half-built and two consumers silently assume the two are the same.

## Port map

| Upstream (`555967922`) | Local anchor |
|---|---|
| `sampling_params.py:31` `MAX_LOGPROB_TOKEN_IDS` | `kMaxLogprobTokenIds`, `include/vllm/sampling_params.h` |
| `sampling_params.py:278-283` the field | `SamplingParams::logprob_token_ids`, same header |
| `sampling_params.py:724-729` `num_logprobs` property | `SamplingParams::num_logprobs()`, `src/vllm/sampling_params.cpp` |
| `sampling_params.py:773-782,795-801` length + `logprobs == n` checks | `SamplingParams::Verify()`, same file |
| `gpu_input_batch.py:273,443-444` | `InputBatch::logprob_token_ids` + `add_request`, `src/vllm/v1/worker/gpu/input_batch.cpp` |
| `gpu_input_batch.py:574` | `InputBatch::remove_request`, same file |
| `gpu_input_batch.py:934-951` (id -> index over the live batch) | `InputBatch::make_sampling_metadata`, same file |
| `sampler.py:86` snapshot condition | `Sampler::forward`, `src/vllm/v1/sample/sampler.cpp` |
| `sampler.py:151-225` `gather_specific_token_logprobs` | `GatherSpecificTokenLogprobs`, same file |
| `sampler.py:119-136` branch + precedence | `Sampler::forward`, same file |
| `scheduler.py:1818` property gate | `src/vllm/v1/core/sched/scheduler.cpp` |
| `logprobs.py` consumer fed by the property | `src/vllm/v1/engine/logprobs.cpp` |

## Design

**Keying.** Upstream keys by req_ID in `InputBatch` and by req_INDEX in
`SamplingMetadata`; we mirror both. Our `metadata.h` field is ALREADY index-keyed
(`std::map<int, std::vector<int32_t>>`), which is what the sampler needs, since
the sampler only sees dense rows. Keying `InputBatch`'s copy by req_id is what
makes `condense()` free: id-keyed maps survive reindexing untouched, exactly like
the sibling `num_logprobs` map, and `make_sampling_metadata` re-derives the
indices from `req_id_to_index` each build. Index-keying `InputBatch` instead
would mean a `MoveDictValue` call in `condense` for no gain and one more place to
get wrong.

**Python truthiness.** `if sampling_metadata.logprob_token_ids:` is false for
BOTH `None` and `{}`. The C++ guard is therefore
`has_value() && !->empty()`, and `make_sampling_metadata` sets the optional only
when the id-keyed map is non-empty — mirroring `if self.logprob_token_ids:` at
`gpu_input_batch.py:936`.

**The gather.** One host function beside `GatherLogprobs`, over the same
`[n * vocab]` raw-logprobs snapshot:

- `max_num_tokens = max(len(ids))` over the map; row width is `max_num_tokens+1`.
- Column 0 is the sampled token for EVERY row, valid for every row — including
  rows with no entry in the map, which upstream also emits (their columns
  `1..max` are padding). Padded cells carry token id 0 and logprob `-inf`,
  matching `torch.zeros` + `masked_fill(~valid_mask, -inf)`.
- The rank is `#{j : row[j] >= row[sampled]}` over the FULL vocab, the same
  1-based `batched_count_greater_than` `GatherLogprobs` already uses. Computing
  it over the requested subset would make "rank 1 of 3 labels" meaningless;
  upstream computes it over the vocab and says why at `sampler.py:212-213`.

**Bounds (issue #249's defect class, not its instance).** `logprobs.gather(-1,
ids)` in torch raises on an out-of-range index; the C++ equivalent would read
past the row. Every id is therefore `VT_CHECK`ed into `[0, vocab)` and every map
key into `[0, n)` before it indexes anything — a loud error, mirroring what torch
does, not a silent clamp and not a read past the row. `GatherLogprobs`'
unbounded `k` is a different function and stays untouched here (#249).

**The property.** `num_logprobs()` is added and the two consumers that spell it
`sampling_params.logprobs` today are switched to it. Without that, a
`logprob_token_ids`-only request produces sampler output the scheduler never
slices and the processor would truncate to zero entries — the capability would
not be reachable through the engine at all.

**Snapshot + fast path.** `sampler.py:86`'s `or` becomes
`num_logprobs.has_value() || has_token_ids`, and the ENG-ASYNC-SCHED
device-resident greedy fast path — which is OURS, not upstream's, and skips the
gather entirely — is gated on the same combined predicate. Missing that second
edit would make the feature silently vanish on the async greedy path.

## Tests to port

`tests/v1/sample/test_logprobs.py:413-425` is the only upstream unit test for the
field and covers just the vocab-bounds `verify()` half, which is our named
engine-time deferral. `tests/entrypoints/openai/chat_completion/test_logprob_token_ids.py`
is an HTTP end-to-end over a real model — the endpoint surface is out of scope
here and there is no oracle-free harness for it. So the cases below are WRITTEN
against the upstream algorithm, and recorded as written rather than ported.

1. `tests/vllm/v1/sample/test_sampler.cpp` — one request, explicit ids, greedy:
   the row is `[sampled | id0 | id1]`, logprobs equal the raw logprobs at those
   ids, and the rank is over the full vocab. **RED:** `logprobs_tensors` has no
   value at all (nothing consumes the field).
2. Same file — heterogeneous lengths across a 3-row batch, one row absent from
   the map: width is `max+1`, short rows and the absent row are `-inf` in their
   padding, column 0 is each row's sampled token.
3. Same file — precedence: both `max_num_logprobs = 2` and explicit ids set; the
   returned shape is the explicit-ids one (`sampler.py:133-136`).
4. Same file — an out-of-range id throws instead of reading past the row.
5. `tests/vllm/v1/worker/test_input_batch.cpp` — the ids reach
   `SamplingMetadata` keyed by req_index; `max_num_logprobs` stays unset when
   only `logprob_token_ids` is set; removal drops the entry; the default request
   leaves the optional unset (inertness).
6. `tests/vllm/test_sampling_params.cpp` — `num_logprobs()` derivation (three
   cases) and the two validations.
7. `tests/vllm/v1/test_llm_engine.cpp` — end-to-end over the CPU engine: a
   `logprob_token_ids` request returns, per generated token, exactly the
   requested ids plus the sampled token, and nothing else.

## Gates

CPU reference backend. No GPU, no model download, no kernel.

```sh
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_VULKAN=OFF -DVLLM_CPP_METAL=OFF
cmake --build build-cpu -j 18
./build-cpu/tests/test_sampler
./build-cpu/tests/test_input_batch
./build-cpu/tests/test_sampling_params
./build-cpu/tests/test_llm_engine
ctest --test-dir build-cpu -j 6 --output-on-failure
```

Any `-j` failure is re-run SERIALLY before it is called a regression;
`test_openai_api_server`, `test_openai_conformance`, `test_async_llm` and
`test_engine_core_proc` starve under load on this box.

The transforms are deterministic host functions over the raw snapshot, so the
gate is EXACT, not near-tie. No throughput axis is claimed or owed: the change
is inert (`std::optional` unset) for every request that does not set the field.

## Dependencies

None. Independent of `SAMPLE-PROMPT-LOGPROBS` (#223) and of #249. Textually
adjacent to open PR #258 (`logprobs_mode`), which edits the same snapshot block
in `sampler.cpp`; whichever lands second rebases.

## Work breakdown

Single change. W1 = params + plumbing + gather + the two property consumers +
tests. There is no W2 within this scope; the OpenAI request field and
`logprobs_mode` are separate leaves named under Scope.

## Risks/decisions

1. **Switching two consumers from `logprobs` to `num_logprobs()`.** Behaviour is
   identical for every request that does not set `logprob_token_ids`, which is
   every request that exists today; the inertness cases pin that.
2. **`-inf` padding reaching the payload layer.** It cannot: the processor
   truncates each row at `num_logprobs()` entries, which for a
   `logprob_token_ids` request is exactly that request's own id count, so its
   padding columns are never read. Case 7 proves it end to end rather than by
   argument.
3. **Rank over the full vocab is O(vocab) per row.** Same cost `GatherLogprobs`
   already pays for its rank; the win is avoiding the full SORT, not the scan.
4. **Conflict with PR #258.** Accepted; the overlap is ~5 lines of one condition.

## Evidence

In the PR body: the RED output verbatim, the focused GREEN counts, and the full
`ctest` summary with any serial re-runs spelled out.

## Stop conditions

- If the ids cannot be bounds-checked at the gather without changing
  `GatherLogprobs`, stop — do not widen the change into #249.
- If making the feature reachable end to end turns out to require the OpenAI
  request field after all, stop and re-spec rather than growing the PR.
- Never satisfy a red gate by relaxing the padding or rank assertions.

## Outcome

*(Filled when the row closes. The row lands `PARTIAL`, not `DONE`: its
`logprobs_mode` half is unported and owned by open PR #258, and the OpenAI
request field is a named residual, so there is no `DONE` outcome to record yet.)*
