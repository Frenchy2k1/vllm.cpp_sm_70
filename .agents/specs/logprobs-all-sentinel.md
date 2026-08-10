# `SAMPLE-LOGPROBS` — `logprobs=-1` widens at admission

*(Live spec, 2026-08-09. Base `origin/main` `58f43f66`. Pin vLLM 0.26.0.dev0
`555967922`. Issue [#231](https://github.com/mudler/vllm.cpp/issues/231). Row
`SAMPLE-LOGPROBS` (`.agents/engine-matrix.md:131`, `DONE`) — a bugfix to a
closed row, not a lifecycle move.)*

*(**Record repair, 2026-08-10**, base `origin/main` `1c1749cb`. The fix landed as
`fd9af7d9` (merged `723d96a8`, PR #236) and the code is correct, but this spec
shipped with the crash site attributed to `LogprobsProcessor::UpdateSampleLogprobs`
and the Scope excluding the HTTP surface on a false premise about a "0..5 range".
Both are corrected below — see "Our baseline", "Scope" and "Outcome". The code the
repair adds is one test; `add_request`, `max_num_logprobs()` and the sampler are
untouched.)*

## Scope

`logprobs=-1` ("give me every vocab entry") crashes the engine. Mirror upstream's
handling — widen the sentinel to `vocab_size` at admission — so one gathered
shape reaches every consumer and the sampler's raw-vocab branch becomes as
unreachable here as it is upstream.

In scope: `InputBatch::add_request`, `InputBatch::max_num_logprobs`, the comment
on the sampler branch that stays, and the tests. Out of scope: `logprobs_mode`
variants and `logprob_token_ids` generative scoring (`SAMPLE-LOGPROB-TOKEN-IDS`,
`INVENTORIED`), and porting upstream's `check_logprobs` request validation or its
`max_logprobs` model cap — that is issue
[#249](https://github.com/mudler/vllm.cpp/issues/249).

**The HTTP surface is NOT excluded, and no range is enforced anywhere in our
tree.** `grep max_logprobs src include` returns nothing;
`src/vllm/entrypoints/openai/protocol.cpp:520` assigns `sp.logprobs = logprobs`
with no check; `src/vllm/sampling_params.cpp:110` rejects only `< 0 && != -1`.
Upstream deliberately admits `-1` on the **chat** surface
(`chat_completion/protocol.py:785-796`: "`top_logprobs` must be a positive value
or -1"), and both ends of that path already exist here —
`ChatCompletionRequest::to_sampling_params` (`protocol.cpp:562`) maps
`top_logprobs` straight into `sp.logprobs`, and `ChatTopLogprobs`
(`serving_utils.cpp:165`) already reads `-1` as "keep every entry". So
`{"logprobs": true, "top_logprobs": -1}` is a real capability this change
unblocks, and it gets an end-to-end case (tests, item 4).

The **completion** surface is the one place we diverge from upstream on the
value: `completion/protocol.py:496-500` rejects `logprobs < 0` with a 400, while
we accept it and `BuildCompletionLogProbs` emits empty `top_logprobs` maps
(`idx > -1` breaks on the first entry). That is #249's job, not this row's.

## Upstream chain

- `vllm/v1/worker/gpu_input_batch.py:434-440` — the widening being ported:
  `self.num_logprobs[req_id] = self.vocab_size if sampling_params.logprobs == -1
  else sampling_params.logprobs`.
- `vllm/v1/sample/sampler.py:120-131` — the three-way branch, including the
  `num_logprobs == -1` raw-vocab arm. Reachable only from a hand-built
  `SamplingMetadata`, because `max_num_logprobs` is fed from the widened map.
- `vllm/sampling_params.py:588-592` — `-1` is a legal value, validated.
- `vllm/v1/outputs.py:41-50` (`LogprobsLists.slice_request`) — the FIRST
  consumer, called from `scheduler.py:1815-1821`. Numpy row-slicing upstream;
  a flat-vector `assign` here.
- `vllm/v1/engine/logprobs.py:69-119` — the SECOND consumer, which reads the
  gathered three-array shape unconditionally.

## Our baseline

The port is faithful at the sampler (`src/vllm/v1/sample/sampler.cpp:343-352`
matches `sampler.py:122-125` arm for arm). The divergence is one layer up:
`src/vllm/v1/worker/gpu/input_batch.cpp:292-297` deliberately PRESERVED the
sentinel, and `max_num_logprobs()` at `:481-497` propagated it, both recorded as
an intentional deviation in `input_batch.h`. That routes live requests into the
branch upstream cannot reach.

**Where it dies: `LogprobsTensors::slice_request`**
(`src/vllm/v1/outputs.cpp:31-37`), called from the scheduler at
`src/vllm/v1/core/sched/scheduler.cpp:920-924` — the first thing to touch the
sampler's output, one layer *before* the engine's `LogprobsProcessor`. The
raw-vocab shape carries `num_tokens_per_position == vocab` while
`logprob_token_ids` and `selected_token_ranks` are EMPTY, so
`assign(begin() + begin*w, begin() + end*w)` builds a `num_positions * vocab`
range out of a null `begin()` and memcpys from address 0.

Upstream's own `slice_request` (`vllm/v1/outputs.py:41-50`) is numpy row-slicing,
which silently yields empty arrays for that shape. Our flat-vector port cannot;
it is a faithful port of a call that upstream never makes on this shape.

`src/vllm/v1/engine/logprobs.cpp:51-77` is a real SECOND consumer with the same
defect — it indexes the same two arrays, and its `width <= 0` guard does not
screen the raw-vocab shape, which sets `num_tokens_per_position = vocab`. It is
simply never reached: with the widening reverted and both consumers instrumented,
the run prints `ENTER slice_request req_idx=0 num_positions=1 ntpp=24 ids=0
lps=24 ranks=0` and then SIGSEGVs; `UpdateSampleLogprobs` is never entered.

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

**Why not teach the consumers the second shape.** It is the other available fix
and it is worse: there are already TWO consumers that would need it
(`slice_request` and `UpdateSampleLogprobs`), and it keeps a deviation whose only
effect is to make our engine carry two logprob shapes where upstream carries one,
so every future consumer would have to know that too. Removing the deviation
deletes the class of bug, and one line at admission fixes both consumers.

## Tests to port

Upstream has no test for this (the value cannot reach the branch there), so these
are written, not ported, and recorded as such.

1. `tests/vllm/v1/test_llm_engine.cpp` — a `logprobs=-1` request through the
   engine returns one entry per generated token, each carrying every vocab id
   exactly once, the sampled token at rank 1, and a row that exponentiates to
   1.0. **RED: SIGSEGV** inside `LogprobsTensors::slice_request`
   (`src/vllm/v1/outputs.cpp:31-37`), reached from `scheduler.cpp:920-924`.
2. Same file — a finite `logprobs=2` request still returns EXACTLY 2 entries per
   position, at ranks 1 and 2 (`[sampled | top-2]` is `k+1 == 3` wide and greedy
   dedups the sampled token against top-1), guarding the ordinary path against a
   regression in the same edit. Asserted exactly, not as a `>= 1 && <= 3` range.
   This case cannot see the SAMPLER-side gather width and does not claim to:
   `AppendLogprobsForNextPosition` truncates to the request's own count, so
   widening every request to `vocab_size` leaves the client payload identical.
   That width is pinned at the input-batch seam instead — item 3, plus the
   pre-existing `tests/vllm/v1/worker/test_input_batch.cpp:641`; an unconditional
   widening in `add_request` turns `:641` and `:687` RED.
3. `tests/vllm/v1/worker/test_input_batch.cpp` — `-1` is widened at admission
   (the map holds `vocab_size`, never the sentinel), both alongside a finite
   request and alone.
4. `tests/vllm/entrypoints/openai/test_serving.cpp` — chat `{"logprobs": true,
   "top_logprobs": -1}` end-to-end: `kVocab` distinct entries per generated
   token, summing to 1.0, and serialized into the JSON body. This path was
   unreachable in practice before the widening even though both of its ends were
   already implemented (see Scope).

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
./build-cpu/tests/test_openai_serving
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
- Never make a consumer tolerant of the raw-vocab shape — not `slice_request`'s
  bounds, not `UpdateSampleLogprobs`'s `width <= 0` guard. That hides the defect
  instead of removing it.

## Outcome

*(2026-08-09, amended 2026-08-10. Row stays `DONE`; the fix removes a recorded
deviation. The fix landed as `fd9af7d9` (merged `723d96a8`, PR #236). The
"Measured" paragraph below originally named `UpdateSampleLogprobs` as the crash
site — that shipped wrong and is corrected here, on branch
`row/SAMPLE-LOGPROBS-RECORD-REPAIR`, with the instrumented proof included.)*

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

**Measured.** RED: `SIGSEGV` inside `LogprobsTensors::slice_request`
(`src/vllm/v1/outputs.cpp:31-37`) for both the engine case and the chat
`top_logprobs=-1` case, and `-1 == 1024` for the admission cases. The crash site
was established by instrumenting BOTH candidate consumers with the widening
reverted: the run prints `ENTER slice_request req_idx=0 num_positions=1 ntpp=24
ids=0 lps=24 ranks=0` and dies; `UpdateSampleLogprobs` — a real second consumer
with the identical defect — is never entered, because the scheduler slices before
the output processor ever runs.

GREEN: `test_llm_engine` 13/13 (231 assertions), `test_input_batch` 26/26 (190),
`test_openai_serving` 42/42 (556), clean CPU Release build with zero warnings
under `-Werror`, full `ctest` **360/360** (729 s, no flake, no serial re-run
needed).

**Rejected: teaching the consumers the raw-vocab shape.** There are two of them —
`slice_request` and `UpdateSampleLogprobs` — which is itself the argument: it
fixes the crash and keeps the deviation, so our engine would carry two logprob
shapes where upstream carries one, and every future consumer would need to know
that. The widening deletes the class of bug instead of the instance, and fixes
both consumers with one line.

**Kept deliberately:** the sampler's `-1` arm. Upstream keeps it, and a caller
that hand-builds `SamplingMetadata` can still reach it, so the comment there now
states both that it is unreachable from the input batch and that its shape
differs — which is the fact whose absence caused this.

**Default.** No flag. `logprobs=-1` was already a validated, legal value; it now
returns what it says.
