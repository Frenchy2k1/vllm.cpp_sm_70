# `SAMPLE-LOGPROBS` — `logprobs=-1` widens at admission

*(Live spec, 2026-08-09. Base `origin/main` `58f43f66`. Pin vLLM 0.26.0.dev0
`555967922`. Issue [#231](https://github.com/mudler/vllm.cpp/issues/231). Row
`SAMPLE-LOGPROBS` (`.agents/engine-matrix.md:131`, `DONE`) — a bugfix to a
closed row, not a lifecycle move.)*

## Scope

`logprobs=-1` ("give me every vocab entry") crashes the engine. Mirror upstream's
handling — widen the sentinel to `vocab_size` at admission — so one gathered
shape reaches every consumer and the sampler's raw-vocab branch becomes as
unreachable here as it is upstream.

In scope: `InputBatch::add_request`, `InputBatch::max_num_logprobs`, the comment
on the sampler branch that stays, and the tests. Out of scope: `logprobs_mode`
variants and `logprob_token_ids` generative scoring (`SAMPLE-LOGPROB-TOKEN-IDS`,
`INVENTORIED`); the OpenAI `logprobs` request field, whose valid range is 0..5
and never carries `-1` — this is the library/`SamplingParams` surface.

## Upstream chain

- `vllm/v1/worker/gpu_input_batch.py:434-440` — the widening being ported:
  `self.num_logprobs[req_id] = self.vocab_size if sampling_params.logprobs == -1
  else sampling_params.logprobs`.
- `vllm/v1/sample/sampler.py:120-131` — the three-way branch, including the
  `num_logprobs == -1` raw-vocab arm. Reachable only from a hand-built
  `SamplingMetadata`, because `max_num_logprobs` is fed from the widened map.
- `vllm/sampling_params.py:588-592` — `-1` is a legal value, validated.
- `vllm/v1/engine/logprobs.py:69-119` — the consumer, which reads the gathered
  three-array shape unconditionally.

## Our baseline

The port is faithful at the sampler (`src/vllm/v1/sample/sampler.cpp:343-352`
matches `sampler.py:122-125` arm for arm). The divergence is one layer up:
`src/vllm/v1/worker/gpu/input_batch.cpp:292-297` deliberately PRESERVED the
sentinel, and `max_num_logprobs()` at `:481-497` propagated it, both recorded as
an intentional deviation in `input_batch.h`. That routes live requests into the
branch upstream cannot reach.

`src/vllm/v1/engine/logprobs.cpp:51-73` then indexes `logprob_token_ids` and
`selected_token_ranks`, which that shape leaves empty. Its guard is `width <= 0`;
the raw-vocab shape sets `num_tokens_per_position = vocab`, so the guard passes
and the reads run off the end of two empty vectors.

## Port map

| Upstream (`555967922`) | Local anchor |
|---|---|
| `gpu_input_batch.py:434-440` (widen `-1` → `vocab_size`) | `InputBatch::add_request`, `src/vllm/v1/worker/gpu/input_batch.cpp` |
| `gpu_input_batch.py:1150-1151` (`max(num_logprobs.values())`) | `InputBatch::max_num_logprobs`, same file — plain max once every value is concrete |
| `sampler.py:122-125` (the raw-vocab arm, unreachable on the V1 path) | `src/vllm/v1/sample/sampler.cpp` — kept, and its unreachability + differing shape now stated where a future reader will meet it |

## Design

One line at admission. `num_logprobs[req_id] = *sp.logprobs == -1 ? vocab_size :
*sp.logprobs`, exactly as upstream. `max_num_logprobs()` loses its sentinel
special case and becomes the plain max upstream's `max(...)` already was: a
request asking for "all" now carries the largest possible count and wins that max
on its own.

Nothing downstream changes. `GatherLogprobs` with `k == vocab` produces the
ordinary `[n, vocab+1]` shape; `AppendLogprobsForNextPosition` already handles
`num_logprobs == -1` on the *engine* side by deriving `k` from the row width
(`logprobs.h:82`), so the `LogprobsProcessor` reads it correctly without change.

**Why not teach the consumer the second shape.** It is the other available fix
and it is worse: it keeps a deviation whose only effect is to make our engine
carry two logprob shapes where upstream carries one, and every future consumer
would have to know that. Removing the deviation deletes the class of bug.

## Tests to port

Upstream has no test for this (the value cannot reach the branch there), so these
are written, not ported, and recorded as such.

1. `tests/vllm/v1/test_llm_engine.cpp` — a `logprobs=-1` request through the
   engine returns one entry per generated token, each carrying every vocab id
   exactly once, the sampled token at rank 1, and a row that exponentiates to
   1.0. **RED: SIGSEGV** inside `UpdateSampleLogprobs`.
2. Same file — a finite `logprobs=2` request still returns at most `k+1` entries,
   guarding the ordinary path against a regression in the same edit.
3. `tests/vllm/v1/worker/test_input_batch.cpp` — `-1` is widened at admission
   (the map holds `vocab_size`, never the sentinel), both alongside a finite
   request and alone.

The existing case `C7 wiring: -1 logprobs sentinel dominates max_num_logprobs`
asserted the deviation, so it is REPLACED, not relaxed: its assertion
(`max_num_logprobs == -1`) is exactly the behaviour that crashes, and the
replacement asserts the mirrored value with the reason written beside it.

## Gates

CPU reference backend.

```sh
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_VULKAN=OFF -DVLLM_CPP_METAL=OFF
cmake --build build-cpu -j 18
./build-cpu/tests/test_llm_engine
./build-cpu/tests/test_input_batch
ctest --test-dir build-cpu -j 6 --output-on-failure
```

A failure under `-j` is re-run serially before it is called a regression.

## Dependencies

None. No kernel, no vt op, no ABI, no model file, no GPU. Independent of
`SAMPLE-PROMPT-LOGPROBS` (#223): that row's `-1` path was already correct,
because the runner widens to `vocab_size` for prompt logprobs the way this
change now does for sampled ones.

## Work breakdown

Single change. There is no W2.

## Risks/decisions

1. **Replacing an existing assertion.** Mitigated by stating in the test itself
   why the old one encoded the defect, and by the engine-level RED that shows
   what the old behaviour actually did.
2. **A hand-built `SamplingMetadata` can still reach the raw-vocab branch.** True
   upstream too. The branch stays (mirroring), and the comment now says the shape
   differs so a new consumer branches instead of indexing blindly.
3. **`vocab_size` columns is a large allocation.** `GatherLogprobs` at
   `k == vocab` does a full sort per row. That is what "all logprobs" costs, and
   what upstream costs; no speed claim is made or owed.

## Evidence

In the PR body: the RED crash, the GREEN runs, the full `ctest` summary.

## Stop conditions

- If the `-1` sentinel turns out to be load-bearing anywhere else, stop and
  re-spec rather than widening the fix.
- Never make the consumer's `width <= 0` guard broader to swallow the shape —
  that hides the defect instead of removing it.

## Outcome

*(2026-08-09. Row stays `DONE`; the fix removes a recorded deviation.)*

**What the bug actually was.** Not a bad port. `sampler.cpp:343-352` matches
`sampler.py:122-125` arm for arm, and reading only those two files makes the
crash look like the sampler's fault. The defect was a DELIBERATE choice one layer
up — preserving the `-1` sentinel instead of widening it — written down in
`input_batch.h` as an intentional deviation, with the reasoning "our Sampler
reads it directly". That was true. What it missed is that the branch it routes
into produces a DIFFERENT shape (empty ids and ranks), and upstream can only
afford that branch because its own input batch can never reach it. We adopted the
branch without adopting the widening that makes it dead.

The first version of issue #231 said upstream has no such branch. That was wrong
and is corrected in a comment on the issue rather than silently: the branch
exists at the pin, it is simply unreachable there.

**Measured.** RED: `SIGSEGV` inside `UpdateSampleLogprobs` for the engine case,
and `-1 == 1024` for the admission cases. GREEN: `test_llm_engine` 13/13 (228
assertions), `test_input_batch` 26/26 (190), clean CPU Release build with zero
warnings under `-Werror`, full `ctest` **360/360** (729 s, no flake, no serial
re-run needed).

**Rejected: teaching `UpdateSampleLogprobs` the raw-vocab shape.** It fixes the
crash and keeps the deviation, so our engine would carry two logprob shapes where
upstream carries one, and every future consumer would need to know that. The
widening deletes the class of bug instead of the instance.

**Kept deliberately:** the sampler's `-1` arm. Upstream keeps it, and a caller
that hand-builds `SamplingMetadata` can still reach it, so the comment there now
states both that it is unreachable from the input batch and that its shape
differs — which is the fact whose absence caused this.

**Default.** No flag. `logprobs=-1` was already a validated, legal value; it now
returns what it says.
