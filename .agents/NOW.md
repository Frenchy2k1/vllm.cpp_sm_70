# NOW — the one-Read resume surface

<!-- now-updated: 2026-08-09 -->

Snapshot, not log. History is git; evidence:
[parity ledger](parity-ledger.md), and benchmarks. Budget: 100 lines / 6,000
characters.

## Live claims

Work: exact-chunks on main `1ce0d662b`; sm_120 measured at `3d2581551`.

| Claim / track | State | Next command or step |
|---|---|---|
| `SPEC-DSPARK` | **WORKS on 35B**: ON==OFF 48/48. ★fixed engine-wide draft-drop | Draft step ~6x a target step |
| State record (#166) | **157 imports = 3,231,342 bytes** at `776c56f1`; 95/95 | Force-update #166; rerun readiness |
| Laguna NVFP4 / DeepSeek-V4 decode | **CLOSED, byte-exact, default-ON**: 1.03x vLLM, 1.144x ds4 | Laguna vLLM K-run |
| 27B NVFP4 @`0893e160` | **0.85x**: FP8 native, tokens MATCH; #213 head packed | #213 grid + RSS remeasure post-#150 |
| f32-out GEMV audit | **CLAIM WRONG**: 35B runs 41 `CastF32`/step (3.1%) | Fold into the 35B lever |
| Invocation-parity | CI guard + checklist landing | build-verify `kGemvHeuristicAlgos` |
| MiniMax-H3 | **PRUNED ckpts RUN (#241): Q8_0 renders, seam 0.9941** | same-binary A/B |
| Kimi-Linear-48B | 122/128 held; e2e NOT ESTABLISHED | tiktoken-only ckpt: no warm server |
| 35B mid-band | **LANDED +1.31% c8**; 2.7% per-launch gap WITHDRAWN (kernels identical) | Glue 13.00% vs 11.37%: 2x SiLU + `CastF32` |
| Qwen3.5-4B sm_120 | Exact chunks ON: 3.072x kernel / +2.272% run; sealed-vLLM tput 1.021x PASS; latency/VRAM OPEN | Spike residual 1.609x conv gap |
| RPi5 A76 CPU | **R5 asm GREEN; llama NOT MET**: 0.461x pf, 0.653x dec | W6: BF16 GEMM |
| MXFP4 parity | c1 1.020, c2-c8 0.962-0.969. **#82 CLOSED: ptxas-lineage REFUTED** | TERMINAL: at parity |
| SERVE-ASYNC-DENSE-MIRROR | **LANDED+VERIFIED** (`f9c969ae`): async mirror, dense Qwen3; SACRED 184/184 | Sibling scope one-liner |
| CPU levers (`QUANT-GGUF-CIQ-GEMM`) | Profile DONE: decode **47% threadpool sync**, prefill **~39% paged attn** | Parakeet encoder; attn dtype hoist |
| `/v1/videos` OpenAI shape | **MERGED** (#71): Sora `model`/`size`/`seconds` + `GET /{id}/content` | `row/SERVE-VIDEOS-REFS`: reference conditioning |
| `ENG-LOAD-DIRECT-UPLOAD` (#150) | **default ON:** weights VIEW the mmap; 27B load **1.54x warm / 1.61x cold** | merged qkv/gate_up + lm_head |
| Vulkan 27B | decode **MET 4.36 vs 4.35**. **LOADMEM: load held the model TWICE, 100.759 -> 53.413 GiB** | Load-phase host build is the new peak |
| `BACKEND-ROCM` | **(b) fix in; #140 gfx1201 hipBLAS + Gemma-4 MoE landed; W0 green** | compile + M2 ([spec](specs/rocm-unified-memory-b.md)) |
| TP spike #287 (PR #143) | **TP-W1 LANDED**: rank-group table + TP handle (6/6); DSR leak FIXED (unblocks #127/#154/#155) | TP-W2 (linears + loader) |
| Release | **ACTIVE; required W1-W11/W13 implemented in #196** | Finish hosted ten-SM proof; rebase/push; run full eight-tuple dry run |
| Surface coverage (`ARCH-ONE-SURFACE`) | ROW 8 + #139; **embeddings live (#137): ABI v15, endpoint, fold 4/4-231** | Real-checkpoint oracle cosine |

In-flight, default-OFF, not pushed: see the row's spec.

## Current gate

Token-exact (or ratified distributional) vs pinned vLLM; ≥ throughput and ≤
latency/memory on every axis, both gate models, reproduced 2–3x idle. See
[verification](verification.md). Pin: vLLM `555967922` (0.26.0.dev0).


## Next actions

0. **35B mid-band: first lever LANDED** (+1.31% c8, +1.38% c4). The fused
   shared gate_up sink still took the MoE-marlin route (20320 launches = 5.4%
   GPU, a config vLLM never launches); `VT_MARLIN_DENSE_PAIR` ON. Block lever
   REFUTED; the 2.7% per-launch gap WITHDRAWN (same geometry, counts AND
   registers; a cross-tool artifact). NEXT: glue, 13.00% vs 11.37%.
1. **27B NVFP4 0.72x -> 0.85x** (FP8 tower native). Next: NVFP4 MLP marlin, 68%
   of roof. Dense-marlin +0.5%; Triton-AOT GDN a WASH.
2. **Spike the Parakeet encoder row** (vLLM: `nano_nemotron_vl.py`; the
   transducer half is NOT in vLLM: separate call).
3. **Qwen3.5-4B sm_120:** GREEN + reprofiled. Spike the residual 1.609x conv
   gap; latency/VRAM and gate models stay open.
2. **Merge the invocation-parity prevention** (CI guard + AGENTS.md checklist);
   CUDA build-verify the byte-exact `kGemvHeuristicAlgos` refactor on dgx.
4. **Restore `local-ai-worker`** on dgx at campaign end (`--restart=always`).
5. **Protocol substrate — partly done.** Triage/audit + `STATUS.md` ratchet +
   `AGENTS.md` tiering DONE. REMAINING: anchor backfill (6 model rows need a
   DECISION); record-era rollover BLOCKED on `DONE` rows bound to
   `parity-ledger.md` LINE anchors (re-anchor by ROW ID).

**Operator/helper protocol** ([spec](workflow.md)): roles are a coordinator
record or worktree+PR; helpers claim `row/<ROW-ID>` with a DRAFT PR. Role gates
ENFORCE `agent-start.py` → claim → preflight. Review FAIL loops to a fresh
implementer until PASS. Queue: 10 rows; backfill 79, 30 anchored.
**Upstream inventory** ([spec](specs/upstream-derived-inventory-2026-08-05.md)):
SM060/061/070 below vLLM's floor = OUT-OF-SCOPE; COMP-*/DISTRIBUTED-* are REAL
unported work; **all 362 archs have rows**; llama.cpp's 11 extra devices IN
SCOPE (`ROAD-V1-D6`).

## Protocol invariants that bite most often

- Every commit carries `FOLLOWING_AGENTS_PROTOCOL` + `Assisted-by:`; never
  `Co-Authored-By` or `Signed-off-by` from AI.
- Three MUST-route seams: fusion, merged-GEMM, born-on-the-runner decode.
  Not routing is drift; allowlist consciously or fold.
- Mirror vLLM; never ask how a feature should behave.
- `nsys` BOTH sides, SAME tool, before any perf claim; cross-tool comparisons
  never establish invocation parity; whole-run sums mix prefill.
- GPU: park `local-ai-worker`, flock `$HOME/gpu.lock`, single-load
  steady-state, never reload per rep, named tmux.
- Never weaken a checker to pass; repair the record.
- Work happens in its own worktree on a task branch; the shared checkout stays
  clean on `main`, never a work surface. Land via `row/*` PR or authorized
  local merge; remove the worktree after.
