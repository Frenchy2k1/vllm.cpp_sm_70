# Task guide — measuring performance

How to produce a number worth believing. The rules are in
[`AGENTS.md`](../AGENTS.md); this is the method.

## The denominator

vLLM is the bar, quant-matched, in its **production** configuration. Never
benchmark against `--enforce-eager` and call it parity. llama.cpp may appear
only as an explicitly labelled secondary comparison.

Both sides run the pinned oracle on identical model artifacts, prompts, token
counts, batching, concurrency, and sampling. If the two sides differ in any of
those, the ratio means nothing.

Prove the oracle actually *runs* the model before trusting it as a
denominator — constructing a config proves nothing.

## Getting a clean measurement

One GPU job at a time. Take the box lock before any measurement, stop competing
services, and never run two large models at once — unified-memory boxes reboot
rather than swap.

Calibrate the noise band from repeated identical legs *before* interpreting a
delta. Discard cold legs for a named cause, never because they are
inconvenient. Use paired, order-alternated A/B legs and a majority rule; a
single pair is an anecdote.

Prefer an instrument that is immune to page-cache effects (GPU-active time per
step) over wall clock when the host is doing heavy I/O.

Budget the disk before the run. A production RelWithDebInfo CUDA build tree is
about **169 GiB** — the build contract claimed ~3 GiB until 2026-08-10, a 56x
underestimate on the one number that decides whether a grid fits. A full disk
does not fail loudly: it voids the binding through memory-return tolerance while
still emitting plausible ratios. Leave real headroom, and delete the tree once
the evidence directory is captured (evidence is tens of MiB).

Two ratio sets that disagree may be two different HARNESSES rather than a
regression. Compare their absolute numbers before believing either; ratios are
scale-invariant and hide an order-of-magnitude mismatch completely. If the
change between the readings is provably inert (a byte-identical refactor),
suspect the measurement, not the code.

## Reading a profile

**A whole-run kernel ranking is a trap.** It sums prefill and decode, so the
top-percentage kernel is frequently one-time prefill work that no decode step
touches. Use a decode-only window or diff two sequence lengths. A `Max` far
above the `Median` means you are looking at a mixture, not a hot loop.

Profile the entire step, not only the kernels. Several of the largest wins here
were host-side waste, not slow math.

Before accepting a gap as "GPU-bound", trace both implementations with the same
tool on the same workload and compare what actually ran.

## Recording it

Record the exact build and run recipe, revisions, model hashes, environment,
clock and contention state, raw output, and the same-binary A/B. Reproduce on an
idle box before acceptance.

Record every required axis — throughput, latency, memory — as both values and
ratios. An axis below floor is an open gap, not a rounding error.

Never record a ceiling. An apparent same-architecture limit is an unresolved
implementation difference; name the next traceable hypothesis instead.

Accepted and pending results go in [`benchmark-record.md`](benchmark-record.md)
and `docs/BENCHMARKS.md`. Method specific to one lever stays in
[`parity-lever-protocol.md`](parity-lever-protocol.md).
