# NOW — the one-Read resume surface

<!-- now-updated: 2026-08-04 -->

Read this FIRST, every session. It is a SNAPSHOT, rewritten in place: what is
live, what gate is being chased, what to do next. It is never a log — evidence
lives in the append-only [state.md](state.md), [parity-ledger.md](parity-ledger.md)
and the benchmark record. Budget: 100 lines.

## Live claims

Working head: `origin/main` (the old `laguna-s21-w7-speed-profile` branch is
fully merged/superseded; work from main).

| Claim / track | State | Next command or step |
|---|---|---|
| Laguna NVFP4 decode speed | 87% of vLLM (37.55 vs 43.10); gap LOCALIZED same-tool to the bf16 GEMV invocation (f32-out `cublasLtMatmul` vs vLLM's bf16-out `cublasGemmEx` on the identical `gemvx` kernel; 2/3 o_proj, 204 vs 139 us/call) | Gated `VT_LAGUNA_BF16OUT_GEMV` A/B in flight (o_proj first); if it converts, extend qkv/router/lm_head → parity at bf16 |
| DeepSeek-V4-Flash decode | **PARITY with ds4** (16.28 vs 16.33, 0.997x, same-session clean). HC-expand fusion byte-exact but perf-neutral, held default-OFF | Optional beat-path only: f16 tensor-core DSA/router (near-tie class — needs user gate-class ratification) |
| f32-out GEMV systemic audit | Only laguna (high) + deepseek_v4 bf16 tower (medium) affected; gate models & on-framework dense unaffected (bf16-out by construction, e2e-bench-verified) | Same-tool re-verify deepseek_v4's bf16 tower after the Laguna fix proves out |
| Invocation-parity prevention | CI guard (`check-gemv-invocation-consistency.py`) + AGENTS.md invocation-parity checklist being landed (worktree agent) | Review + merge; CUDA build-verify the `kGemvHeuristicAlgos` constant refactor on dgx |
| MiniMax-H3 lane | Portable path complete; e2e prompt-conditioned video on real weights (Thor). Speed = NVFP4 FP4 device path, sm_121-gated | PR #26 rebase + supports-audit synthesis (workflow ran; integrate) |
| Protocol substrate repair | BENCHMARKS.md converted to scoreboard (landed); STATUS.md budget + record-era roll still open | Items below |

In-flight branches (gated default-OFF, not pushed): `laguna-fp4proj-prod`
(b357a4f6, fp4 opt-in 141-162% of vLLM), `laguna-bf16-gemv` (6a4edee3),
`laguna-legacy-gemv` (91634ca7), `laguna-pipeline-decode` (b7786ff1),
`ds4-hc-expand-fuse` (200c86bd).

## Current gate

Unchanged: token-exact (or the ratified distributional gate) against the pinned
vLLM oracle, AND ≥ vLLM on every throughput axis / ≤ on latency and memory, on
both gate models, reproduced 2–3x on an idle box. See [gates.md](gates.md) and
[benchmark-protocol.md](benchmark-protocol.md). Parity pin: vLLM `555967922`
(0.26.0.dev0).

Method rules hardened this cycle (AGENTS.md): the STRUCTURAL lens (same kernel,
different throughput ⇒ audit the context; scan the REFERENCE's own rationale —
vLLM or ds4/SGLang/llama.cpp — as a default lane; the scan GENERATES hypotheses,
per-shape MEASUREMENT arbitrates; distrust aggregate bytes/time and CROSS-TOOL
comparisons — the Laguna "ceiling" was a cross-tool artifact, twice).

## Next actions

1. **Land the Laguna bf16-out GEMV fix** if the in-flight A/B converts o_proj
   204→139 us in-graph; then stack qkv/router/lm_head and re-measure vs vLLM.
2. **Merge the invocation-parity prevention** (CI guard + AGENTS.md checklist);
   CUDA build-verify the byte-exact `kGemvHeuristicAlgos` refactor on dgx.
3. **Same-tool re-verify deepseek_v4's bf16 resident tower** (the one other
   f32-out caller) once the Laguna fix proves the mechanism.
4. **Restore `local-ai-worker`** on dgx when the GPU campaign ends
   (`docker update --restart=always local-ai-worker && docker start ...`).
5. **Protocol substrate — partly done.** Claim triage DONE (54 rows out of
   `ACTIVE`, 29 claims retired); `docs/STATUS.md` now under a shrink-only
   ratchet; roadmap compacted; `AGENTS.md` tiered. REMAINING: (a) anchor
   backfill, 98 rows still `SPIKE`/`ACTIVE` (backend 36, engine 26, kernel 16,
   model 16, quant 4) because every honest destination needs code/test anchors
   they lack — root cause is that the lifecycle has no zero-cost parking state
   for a landed-but-unowned row; 6 model rows need a DECISION, not an anchor
   (architecture unregistered). (b) The record-era rollover is BLOCKED:
   `check-agent-record.py` binds `DONE` rows to exact LINE anchors in
   `parity-ledger.md` (43 references), so freezing it invalidates the evidence
   graph — re-anchor by ledger ROW ID first. `state.md` and
   `benchmark-record.md` have no line anchors and can roll now.

**Accepted 2026-08-04, not yet enforced:** the operator/helper protocol,
[specs/operator-helper-protocol.md](specs/operator-helper-protocol.md). Roles are
DECLARED (asked when the user has not said) then MATERIALIZED into a lock or a
worktree+PR, and only re-derived from there; operator merges PRs first and drives features
only via sub-agents; helpers work in worktrees on `row/<ROW-ID>` and open a
DRAFT PR at the START, which IS the claim. W1-W5 guards are the work implied.

## Protocol invariants that bite most often

- Every commit carries `FOLLOWING_AGENTS_PROTOCOL` + `Assisted-by:`; never
  `Co-Authored-By` or `Signed-off-by` from AI.
- Three MUST-route seams: fusion catalog, merged-GEMM family, born-on-the-runner
  decode. Not routing is drift — allowlist consciously or fold.
- Mirror vLLM; do not ask the user how a feature should behave.
- `nsys` BOTH sides, SAME tool, before any perf claim; cross-tool per-kernel
  comparisons can NEVER establish invocation parity; whole-run sums mix prefill.
- GPU box discipline: park `local-ai-worker` for the whole campaign, flock
  `$HOME/gpu.lock`, single-load steady-state, never reload per rep, named tmux.
- Never weaken a checker to make a transition pass; repair the record.
