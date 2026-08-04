# NOW — the one-Read resume surface

<!-- now-updated: 2026-08-04 -->

Read this FIRST, every session. It is a SNAPSHOT, rewritten in place: what is
live, what gate is being chased, what to do next. It is never a log — evidence
lives in the append-only [state.md](state.md), [parity-ledger.md](parity-ledger.md)
and [benchmark-record.md](benchmark-record.md). Budget: 100 lines.
Refresh it in the SAME change as any `state.md` append (`check-now-current.py`).

## Live claims

**This section is NOT yet ground truth.** The matrices carry 187 rows in
`ACTIVE` and 79 in `SPIKE`; nothing close to that is really in flight, so the
live set cannot be reconstructed from the record. Triaging it is next action 1,
and it needs the developer, not an agent, to say which rows are still live.

| Track | State on record | Next step |
|---|---|---|
| Laguna-XS NVFP4 decode | 88% of vLLM | Remaining overlap window; bandwidth-contention-capped |
| DeepSeek-V4-Flash | 96% of ds4 | Q8_0 projection-GEMV weight-stream floor |
| 35B prefill TTFT | 0.79x-0.86x every concurrency | Portable fusion of norm/quant/act/combine glue |
| Multimodal image/audio/video | Correctness gated, speed unmeasured | Per-modality speed grids |
| vLLM 0.26 re-benchmark | Pending since the pin advance | Re-run the binding grids on the advanced pin |
| SGLang floor arms | Never ran | Both arms of the SGLang comparison |

## Current gate

Token-exact (or the ratified distributional gate where vLLM's own greedy is
non-deterministic) against the pinned oracle, AND >= vLLM on every throughput
axis / <= vLLM on every latency and memory axis, on both gate models, reproduced
2-3x on an idle box. See [gates.md](gates.md) and
[benchmark-protocol.md](benchmark-protocol.md).

Parity pin: vLLM `555967922` (0.26.0.dev0). **Known staleness:** the roadmap
order-0 row is still framed against v0.25.0 and predates the pin advance.

## Next actions

1. **Triage the claim/matrix state.** 187 `ACTIVE` + 79 `SPIKE` rows make
   "claim before editing" unenforceable and hide what is live. Note the
   coupling: `check-agent-record.py` requires every `SPIKE`/`ACTIVE` row's owner
   to appear in `coordination.md`, so archiving claims means moving the matrix
   rows, not just the claim table.
2. **Unblock the record-era rollover.** AGENTS.md § *Periodic live-document
   compaction* already mandates it and the trigger is met (`state.md` 2.5 MB,
   `parity-ledger.md` 2.0 MB, `benchmark-record.md` 1.0 MB), but it is currently
   UNEXECUTABLE: `check-agent-record.py` binds `DONE` rows to exact LINE anchors
   in `parity-ledger.md` (`EVIDENCE_ANCHOR_FILES`, 43 live references), so
   freezing that file invalidates the evidence graph. **Prerequisite:** re-anchor
   `DONE` evidence by ledger ROW ID instead of line number, then roll.
   `state.md` and `benchmark-record.md` carry no line anchors and can roll now.
3. **Compact `roadmap_v1.md`.** 484 lines of 2026-07-18 narrative sit above the
   portfolio table. Cut to a <=30-line current position; table first.
4. **Budget `docs/STATUS.md`.** 292 KB and unbudgeted, the same shape
   `docs/BENCHMARKS.md` was in before its 2026-08-04 conversion. Extend
   `check-public-doc-tables.py` to cover it.
5. **Tier `AGENTS.md`** to <=200 lines (T0 non-negotiables / T1 directive
   summaries + links / T2 index); move rationale bodies into `.agents/` docs.

## Protocol invariants that bite most often

- Every commit carries `FOLLOWING_AGENTS_PROTOCOL` + `Assisted-by:`; never
  `Co-Authored-By` or `Signed-off-by` from AI.
- Three MUST-route seams: fusion catalog, merged-GEMM family, born-on-the-runner
  decode. Not routing is drift, allowlist consciously or fold.
- Mirror vLLM; do not ask the user how a feature should behave.
- `nsys` BOTH sides before any perf claim; whole-run kernel sums mix prefill in.
- Never weaken a checker to make a transition pass; repair the record.
- Verify you are not on a stale base before trusting the tree: this digest was
  first drafted against a branch 34 commits behind main, where the headline
  Laguna and DeepSeek numbers were both wrong.
