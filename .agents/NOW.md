# NOW — the one-Read resume surface

<!-- now-updated: 2026-08-04 -->

Read this FIRST, every session. It is a SNAPSHOT, rewritten in place: what is
live, what gate is being chased, what to do next. It is never a log — evidence
lives in the append-only [state.md](state.md), [parity-ledger.md](parity-ledger.md)
and [benchmark-record.md](benchmark-record.md). Budget: 100 lines.
Refresh it in the SAME change as any `state.md` append (`check-now-current.py`).

## Live claims

**In flight (developer-confirmed 2026-08-04): DeepSeek and Laguna speed parity.
Nothing else.** Everything else with landed code sits in `ANCHOR-BACKFILL` or is
awaiting anchor backfill; treat any other `ACTIVE` row as not-yet-triaged, not
as someone's live work.

| Live track | State on record | Next step |
|---|---|---|
| Laguna-XS NVFP4 decode | 88% of vLLM | Remaining overlap window; bandwidth-contention-capped |
| DeepSeek-V4-Flash | 96% of ds4 | Q8_0 projection-GEMV weight-stream floor |

Not live, but open and often mistaken for live: 35B prefill TTFT (0.79x-0.86x),
multimodal speed grids (unmeasured), the vLLM 0.26 re-benchmark (pending since
the pin advance), and both SGLang floor arms (never run).

## Current gate

Token-exact (or the ratified distributional gate where vLLM's own greedy is
non-deterministic) against the pinned oracle, AND >= vLLM on every throughput
axis / <= vLLM on every latency and memory axis, on both gate models, reproduced
2-3x on an idle box. See [gates.md](gates.md) and
[benchmark-protocol.md](benchmark-protocol.md).

Parity pin: vLLM `555967922` (0.26.0.dev0). **Known staleness:** the roadmap
order-0 row is still framed against v0.25.0 and predates the pin advance.

## Next actions

1. **Finish the anchor backfill.** Batch 1 done: 12 engine rows made precise, 10
   more rows left `ACTIVE`. **114 remain `SPIKE`/`ACTIVE`** (backend 36, model
   31, kernel 16, engine 27, quant 4) and cannot move yet: every honest
   destination is in `EVIDENCED_STATES` and requires anchors they lack, and
   inventing anchors would be the dishonesty the record prevents. Root cause:
   the lifecycle has **no zero-cost parking state** for a landed-but-unowned
   row, so rows rot in `ACTIVE`. Pick one: backfill anchors row by row, or add
   an explicit unowned/dormant state with no anchor requirement.
2. **Unblock the record-era rollover.** AGENTS.md § *Periodic live-document
   compaction* already mandates it and the trigger is met (`state.md` 2.5 MB,
   `parity-ledger.md` 2.0 MB, `benchmark-record.md` 1.0 MB), but it is currently
   UNEXECUTABLE: `check-agent-record.py` binds `DONE` rows to exact LINE anchors
   in `parity-ledger.md` (`EVIDENCE_ANCHOR_FILES`, 43 live references), so
   freezing that file invalidates the evidence graph. **Prerequisite:** re-anchor
   `DONE` evidence by ledger ROW ID instead of line number, then roll.
   `state.md` and `benchmark-record.md` carry no line anchors and can roll now.

## Protocol invariants that bite most often

- Every commit carries `FOLLOWING_AGENTS_PROTOCOL` + `Assisted-by:`; never
  `Co-Authored-By` or `Signed-off-by` from AI.
- Three MUST-route seams: fusion catalog, merged-GEMM family, born-on-the-runner
  decode. Not routing is drift, allowlist consciously or fold.
- Mirror vLLM; do not ask the user how a feature should behave.
- `nsys` BOTH sides before any perf claim; whole-run kernel sums mix prefill in.
- Never weaken a checker to make a transition pass; repair the record.
- Chain the push to the gate in ONE command (`gate && git push`); a separate
  push line runs even when the checker before it failed. Never force-push main.
- Verify you are not on a stale base before trusting the tree: this digest was
  first drafted against a branch 34 commits behind main, where the headline
  Laguna and DeepSeek numbers were both wrong.
