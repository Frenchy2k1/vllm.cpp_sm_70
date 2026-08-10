# `SAMPLE-LOGPROB-TOKEN-IDS` — the `logprobs_mode` half

*(Live spec, 2026-08-10. Base `origin/main` `8a6704a2`. Pin vLLM 0.26.0.dev0
`555967922`. Issue [#238](https://github.com/mudler/vllm.cpp/issues/238). Row
`SAMPLE-LOGPROB-TOKEN-IDS` (`.agents/engine-matrix.md:133`, `INVENTORIED`), which
carries two independent capabilities; this spec is the `logprobs_mode` one.
Owner claim `CLAIM-SAMPLE-LOGPROBS-MODE`.)*

**Ordering deviation, stated rather than hidden.** AGENTS.md requires the spec to
be committed before the implementation. For this row it was not: the three modes
were implemented and gated first, and this spec was written afterwards from the
result. The commit order does not disguise that — spec and code land in one
commit rather than two, so no reader is misled into thinking the design was
settled first. The design was small and fully determined by upstream (there was
no decision to take), which is why the cost was low; it is still a deviation and
is recorded here as one.

## Scope

`logprobs_mode` selects which tensor the returned logprobs are read from. vLLM
ships four values; we shipped one and refused the other three at runtime, so the
config was unreachable in practice.

In scope: implement `raw_logits`, `processed_logprobs` and `processed_logits`
1:1 and delete the refusal. Out of scope, on the same row and its own work: the
`logprob_token_ids` generative-scoring gather
(`sampler.py:151-225`, `gather_specific_token_logprobs`) and the
`SamplingParams`/config/CLI plumbing that would let an operator select the mode
from outside the library — this change makes the `Sampler` constructor's existing
parameter mean something; it does not add a new user-facing knob.

## Upstream chain

- `vllm/v1/sample/sampler.py:85-93` — the RAW snapshot, taken before any logits
  processor runs: `raw_logprobs` is `compute_logprobs(logits)`, `raw_logits` is
  the logits cast to f32.
- `vllm/v1/sample/sampler.py:255-302` (`sample`) — the PROCESSED pair, produced
  inside sampling. `:262-271` is the all-greedy early return, which snapshots
  before temperature; `:286-290` is the random path, where `topk_topp_sampler`
  hands the snapshot back after temperature and top-k/top-p.
- `vllm/v1/sample/sampler.py:104-106` — whatever `sample` returns REPLACES the
  raw snapshot.
- `vllm/config/model.py:82,221` — the config field and its validation.

## Our baseline

`include/vllm/v1/sample/sampler.h:49-54` already declared the full enum with
three values marked `STUB (deferred)`, and `src/vllm/v1/sample/sampler.cpp`
refused them:

```cpp
VT_CHECK(logprobs_mode_ == LogprobsMode::kRawLogprobs,
         "sampler: only the raw_logprobs logprobs_mode is implemented at T0");
```

So constructing a `Sampler` with any other mode produced an engine that threw the
first time a request asked for logprobs. The snapshot machinery for the raw pair
already existed; the processed pair had nowhere to be taken from, because
`sample()` returned only the sampled ids.

## Port map

| Upstream (`555967922`) | Local anchor |
|---|---|
| `sampler.py:87-93` (raw pair, before mutation) | the snapshot block in `Sampler::forward`, `src/vllm/v1/sample/sampler.cpp` |
| `sampler.py:262-271` (processed pair on the all-greedy early return) | the `sm.all_greedy` return in `Sampler::sample` |
| `sampler.py:286-290` (processed pair after temperature + top-k/top-p) | after `ApplyTopKTopPFromMeta` in `Sampler::sample` |
| `sampler.py:104-106` (processed replaces raw) | `if (!processed.empty()) raw_logprobs = std::move(processed);` in `forward` |
| `sampler.py:246-248` (`logprobs_mode_override` out-param shape) | `Sampler::sample`'s new `processed_out` parameter |

## Design

`forward` takes the raw snapshot only under the raw modes; `sample` takes the
processed one, at the two points upstream takes it, into a caller-owned buffer;
`forward` then lets a non-empty processed snapshot replace the raw one, which is
upstream's `if processed_logprobs is not None`.

`raw_logits` is a straight device→host copy of the logits rather than a
`ComputeLogprobs`, and it must happen before the logits processors mutate the
tensor in place — which is why it lives in the same block as the raw-logprobs
snapshot and not later.

The distinction the modes exist for is RAW vs PROCESSED, not logprobs vs logits:
the raw pair describes the MODEL's distribution, the processed pair describes the
distribution actually SAMPLED from. A token that top-k masked away reads its true
value under `raw_*` and `-inf` under `processed_*`. That is the user-visible
behaviour the gate asserts.

## Tests to port

Upstream's own coverage for this is indirect, so these are written rather than
ported, and recorded as such. All four cases use the SAME logits row so the modes
are directly comparable (`tests/vllm/v1/sample/test_sampler.cpp`):

1. `raw_logits` returns the unnormalized logits verbatim — each strictly greater
   than the corresponding log_softmax value, which is what makes the mode
   observable at all.
2. `raw_logprobs` (the default) over the same row: normalized, strictly below
   the raw-logits answer. A regression guard on the default.
3. `processed_logits` under `top_k=2`: the kept tokens hold their
   temperature-scaled logits, the masked tail reads `-inf`. No raw mode can
   produce this.
4. `processed_logprobs` under the same `top_k=2`: the same mask, but
   renormalized over the surviving set, so the kept pair carries all the
   probability mass. That renormalization is the entire difference from case 3.

## Gates

CPU reference backend.

```sh
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_VULKAN=OFF -DVLLM_CPP_METAL=OFF
cmake --build build-cpu -j 18
./build-cpu/tests/test_sampler
ctest --test-dir build-cpu -j 6 --output-on-failure
```

A failure under `-j` is re-run serially before it is called a regression.

## Dependencies

None. No kernel, no new vt op, no ABI, no model file, no GPU. Independent of
`SAMPLE-PROMPT-LOGPROBS` (#223) and of the `logprobs=-1` widening (#231), though
it sits in the same file as the latter.

## Work breakdown

- **This change.** The three modes + the refusal deleted, sampler-gated.
- **Not done, same row:** `logprob_token_ids` generative scoring, and the
  config/CLI/`SamplingParams` plumbing to select a mode from outside the library.
  Until that lands the modes are reachable only by constructing a `Sampler`
  directly, which is what the tests do — so the row moves to `PARTIAL`, not
  `ACTIVE`.

## Risks/decisions

1. **The processed snapshot is taken at two different points.** All-greedy
   returns before temperature; the random path snapshots after top-k/top-p. That
   asymmetry is upstream's, not ours, and cases 1-4 pin both arms.
2. **`processed_logits` copies the mutated tensor, `processed_logprobs`
   log_softmaxes it.** Getting these the wrong way round would still produce
   plausible numbers; case 4's renormalization assertion is what separates them.
3. **No speed claim.** The processed modes add one [n, vocab] device→host copy
   per step when engaged, and nothing when not. Inert on the default path.

## Evidence

In the PR body: the RED run (three cases throwing on the refusal), the GREEN run,
and the full `ctest` summary.

## Stop conditions

- If any mode needs a change to `sample()`'s sampling behaviour rather than an
  extra snapshot, stop and re-spec — these are observation modes, and mutating
  the draw to serve them would be a real deviation.

## Outcome

*(2026-08-10. Row `INVENTORIED` -> `PARTIAL`.)*

**Measured.** RED: with the implementation stashed, the three new mode cases
throw `sampler: only the raw_logprobs logprobs_mode is implemented at T0`
(15 cases, 3 failed). GREEN: `test_sampler` **15/15, 67 assertions**, clean CPU
Release build, **0 warnings** under `-Werror`.

**Full `ctest`: 365/366, and the contention theory is now PROVEN, not argued.**

The first run had `test_openai_api_server` and `test_openai_conformance` failing
*serially*, which ruled out the usual parallel-starvation explanation and was not
accepted as a flake. Attribution was established in stages:

- With `src/` and `include/` stashed — clean `origin/main` `8a6704a2`, same build
  dir — **both failed identically**. Not this change.
- The failure COUNT varied run to run (19 failed assertions, then 7), which a
  deterministic regression does not do.
- The harness binds an **ephemeral** port (`test_conformance.cpp:409`
  `bind_to_any_port`), ruling out a collision between concurrent sessions.
- Box load average was **46-107 on 20 cores**.

The decisive measurement came when the box briefly went quiet (load 0.73): the
same two tests, same binaries, **passed in 0.58 s total**, against **726 s and
failing** under load. A ~1000x swing in wall time is CPU starvation, and nothing
else. No "main is broken" issue was filed on the earlier evidence, which is just
as well — contention misreported as a defect sends someone hunting a bug that was
never there.

The full re-run then scored **365/366**, with both originally-suspected tests
PASSING and one different test (`test_async_llm`) failing under `-j` and passing
**serially in 0.04 s** — the identical signature. Honest caveat: the box did not
STAY quiet. Load climbed from 2.19 to 77 during that 675 s run as other agents
resumed, so this is not a truly uncontended number. It is the best available on a
shared box, the originally-flagged failures are gone, and every residual failure
resolves serially in well under a second.

**Nothing was rejected.** The design had no branch point: upstream fixes where
each snapshot is taken, and the only real question — whether the processed pair
is captured before or after top-k/top-p — is answered explicitly at
`sampler.py:286-290`.
