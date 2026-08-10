# ROCm decode-graph capture — port the `vt::Backend` capture seam to hipGraph

**Row:** `BACKEND-ROCM` (backend-matrix, `ACTIVE`).
**Claim:** `CLAIM-ROCM-DECODE-GRAPH` (unclaimed at time of writing).
**Issue:** **PENDING — file before implementation starts.** Per AGENTS.md "no
work without an open GitHub issue"; the number then lands here, in the
[roadmap intake table](../roadmap_v1.md), and in the PR body, and the three must
agree. Draft text: §9 W0.
**Base:** `6e4244f0` on `docs/rocm-gfx1200-m2-spec`, so this **stacks on
[PR #273](https://github.com/mudler/vllm.cpp/pull/273)**, where
[rocm-gfx1200-m2-correctness.md](rocm-gfx1200-m2-correctness.md) lands;
`check-agent-record` fails with a dangling link if based on `main`. #273 merges
first. The source tree is `f323907e` (`upstream/main`, 2026-08-10) — #273 is
docs-only, so every `file:line` below holds against either. Re-anchor at
implementation time; `check_links` validates ranges.
**Board:** AMD Radeon RX 9060 XT (`gfx1200`, Navi 44, RDNA4, discrete), ROCm
7.2.3, hipClang/Clang 22.0.0 — the only board with hardware access here. Records
say gfx1200 and must not imply the four #41 boards
(gfx1151/gfx1103/gfx1100/gfx1201) are covered.

---

## 1. Why this, and what it is worth

`vt::Backend`'s graph-capture virtuals are implemented only for CUDA.
`SupportsGraphCapture()` is false on ROCm (`rocm_backend.hip:22-26` says so
explicitly) and `RocmPlatform` does not override `support_static_graph_mode()`,
inheriting false from `interface.h:189`. Decode-graph classes gate on both
(`qwen3.cpp:494-498`), so every ROCm decode step pays full host launch cost
while vLLM on the same board replays captured hipGraphs.

Measured on gfx1200, 2026-08-10, 128in/128out batch 8, ours
(`examples/vllm-bench` on `build-hip`) vs a real vLLM-ROCm oracle at this
project's pin `555967922` in production config (**not** `--enforce-eager`):

| Model | Layers | hidden x inter | Ours | Oracle | Ratio |
|---|---|---|---|---|---|
| Qwen3-0.6B | 28 | 1024 x 3072 | 184.69 tok/s | 552.65 tok/s | 2.99x |
| Qwen3-1.7B | 28 | 2048 x 6144 | 150.50 tok/s | 286.42 tok/s | 1.90x |
| Qwen3-4B | 36 | 2560 x 9728 | 98.49 tok/s | 143.36 tok/s | 1.46x |

Single-stream agrees: our TPOT 13.55 -> 24.09 -> 42.31 ms, oracle 5.21 ->
12.34 -> 27.13 ms/token, ratio 2.60x -> 1.95x -> 1.56x.

**The premise was tested by a scaling experiment before any code, and it
survived.** Launch overhead is fixed per decode step; compute is not. A
launch-dominated gap must shrink as compute per step grows; a
kernel-quality-dominated one must not. Across a 7.9x span of per-step compute
the gap falls monotonically **2.99x -> 1.90x -> 1.46x**.

The 0.6B/1.7B pair is the cleanest control: identical layer count (28), so an
identical launch count per step, with ~4x the compute. Qwen3-4B's 36 layers look
like a confound but are not — launches and compute both scale with layers, so
the layer count cancels and `L/C` depends only on per-layer width
(`1 / (hidden x intermediate)`). All three sit on one curve.

**It also bounds the win.** Fitting `ratio = alpha + beta / (hidden x inter)`
across the three points gives **alpha ~= 1.36x** (size-independent: kernel
quality, inductor fusion, the Triton attention path) and **beta ~= 1.65** in
units where 0.6B's `hidden x inter` = 1 — an overhead contribution of ~1.65x at
0.6B, ~0.41x at 1.7B, ~0.21x at 4B. So roughly half the 0.6B gap is fixed
overhead, and the expected outcome is all three sizes converging on **~1.36x**:
a real win, and **not parity**. A ~1.4x residual would remain.

Treat the fit as provisional. An earlier two-point version gave `alpha ~= 1.54x`;
Qwen3-4B then measured 1.46x, below that asymptote, which a curve cannot do, so
the third point forced the refit. The form is approximate — predicted vs measured
is 3.01/2.99, 1.77/1.90, 1.57/1.46 — `hidden x inter` is a proxy for per-step
compute rather than a measurement of it, and no trace has been taken on either
side. D4 carries this.

**Structural reason, independent of the numbers.** `vt::Backend`'s capture
virtuals (`include/vt/backend.h:181-195`) are documented as a multi-backend seam
("CUDA Graphs / Metal ICB / Vulkan CB"), but CUDA is its only implementation:
Metal (`metal_backend.mm:13`), Vulkan (`vulkan_backend.cpp:16`) and ROCm all
carry the same `stays FALSE` note. A one-implementation abstraction is unproven.
hipGraph is the cheapest available second, since `MTLIndirectCommandBuffer` and
a pre-recorded `VkCommandBuffer` are genuinely different models.

*Provenance.* Stock upstream checkpoints, SHA-256-verified against HF blob
hashes: `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B` (`169ad53e...30ed5` /
`912becff...deff9`), `Qwen/Qwen3-4B` (`328a91d3...51223` / `6cd087b3...21ca5` /
`e4bf4369...f4ca1`). 4B needs `--gpu-memory-utilization 0.85` on the oracle
side (~8.0 GB weights + ~3 GB captured graphs against 15.92 GiB). Oracle =
`vllm bench throughput` / `latency` in the
`vllm-rocm-oracle:555967922-gfx1200` container (recipe in
[rocm-gfx1200-m2-correctness.md](rocm-gfx1200-m2-correctness.md)). Single run per
cell on a board that also drives a display: indicative, and **not** the
2-3x-reproduced-idle standard gate 5 requires. `Qwen/Qwen3-8B` does not fit
(~16.4 GB bf16 against 15.92 GiB), and no quantized path is available — ROCm
registers 23 ops to CUDA's 84, none of them quantized, and a discrete board has
no reference tier, so an unregistered op throws.

## 2. Scope

**In scope.**
1. `src/vt/rocm/rocm_backend.hip` — the six capture virtuals against hipGraph,
   mirroring `cuda_backend.cu:194-286`, replacing the `stays FALSE` note at
   `:22-26`.
2. `src/vllm/platforms/rocm.cpp` — add the `support_static_graph_mode()`
   override (`rocm.cpp:67` is today a comment explaining the inherited false,
   not an override).
3. `tests/vt/test_rocm_backend.cpp` — a RED-first capture/replay case (§6).
4. Records: this spec, the roadmap intake row, `backend-matrix.md`, and
   `docs/ROCM.md` §5's M3 text.

**Out of scope.**
- **New decode-graph model siblings.** Only models that already have one
  benefit; writing more is separate work with its own correctness gate.
- **`VT_BENCH_PROFILE_CONTROL`** (`cuda_backend.cu:230-262`) — CUDA-profiler
  instrumentation, not load-bearing. A `rocprofiler` equivalent is later work;
  the first cut omits it and says so in the code.
- **Any model-level edit.** Every decode-graph class already gates generically
  with no `is_cuda()` anywhere (`qwen3.cpp:494-498`, `qwen3_moe.cpp:385`,
  `deepseek_v2.cpp:896`, `voxtral.cpp:438`, `qwen3_5.cpp:7846,8167`). Flipping
  the two flags suffices; needing a model edit would mean the seam had failed.
- **Metal / Vulkan capture.** Different APIs, different specs.
- **The M3 attention-backend NAME registration** (`rocm.cpp:105` returns `{}`)
  and the stale `rocm.cpp:91` comment claiming `kPagedAttention` is unregistered
  for `kROCM` — it is registered (`rocm_ops.hip:92`) and ran `vt-native` on this
  board. Both real, both separate; a bug found in passing gets its own issue.

## 3. Upstream chain

**No upstream analog.** Graph capture in vLLM is torch's
(`CompilationConfig.cudagraph_mode`); there is no `csrc/` file to port. The
`vt::Backend` capture seam is an additive abstraction, same class as the ROCm
`hipMallocManaged` decision (`rocm-unified-memory-b.md`), and goes in the porting
inventory as additive with this spec as its record.

What is mirrored is upstream's *behaviour*: vLLM on ROCm captures hipGraphs by
default in production config, the configuration our gate measures against.
Observed on this board 2026-08-10 — 51 piecewise + 35 full captures, ~6 s, in the
`vllm bench` log — so "hipGraph capture works on gfx1200/ROCm 7.2.3" is measured,
not inferred.

Internal anchors:
- `include/vt/backend.h:181-195` — the six virtuals and the multi-backend comment.
- `src/vt/backend.cpp:29-34` — base impls: five `VT_CHECK(false, ...)` throws,
  `DestroyGraph` a no-op. An unimplemented backend fails loudly; the model-level
  `enabled` gate keeps it off the path.
- `src/vt/cuda/cuda_backend.cu:178-286` — the implementation to mirror. Its
  capture-contract comment (`:183-193`) is the real specification of what a
  caller must honour.
- `src/vllm/platforms/interface.h:185-189`, `cuda.cpp:58-59`, `rocm.cpp:67`.
- `src/vllm/model_executor/models/qwen3.cpp:489-560` — the consumer:
  per-padded-size `SizeSlot`s with fixed-address persistent buffers, `Refresh()`
  in place, invalidate-and-recapture on a block-table column change.

## 4. Our baseline

**Reused as-is** — most of the change's value is that none of this is written:
the whole `vt::Backend` interface (no signature changes); every decode-graph
class and its capture-contract handling (padded slots, pre-warm, persistent
buffers, column-change invalidation); `test_cuda_backend.cpp:102-153` as the test
shape; the `tests/CMakeLists.txt` pattern that compiles
`test_rocm_backend.cpp` everywhere as a bit-rot guard but links it only under
`VLLM_CPP_HIP`.

**New:** the `hip*` graph calls. Mechanically a near-copy, but never compiled
here.

## 5. Port map

| CUDA (`cuda_backend.cu`) | ROCm (`rocm_backend.hip`) |
|---|---|
| `:194` `SupportsGraphCapture()` | same |
| `:200-203` `BeginCapture` — `cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal)` | `hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal)` |
| `:204-212` `EndCapture` — end → destroy prior `exec_` → instantiate → destroy graph | `hipStreamEndCapture` / `hipGraphExecDestroy` / `hipGraphInstantiate` / `hipGraphDestroy` |
| `:214-216` `Replay` — `cudaGraphLaunch(exec_, s)` | `hipGraphLaunch` |
| `:221-227` `EndCaptureGraph` — opaque-handle variant for the multi-size slot map | same shape; returns `hipGraphExec_t` as `void*` |
| `:229-266` `ReplayGraph(q, graph)` | same, minus `VT_BENCH_PROFILE_CONTROL` (§2) |
| `:284-286` `DestroyGraph` | `hipGraphExecDestroy` |
| `platforms/cuda.cpp:58-59` `support_static_graph_mode()` | new override in `platforms/rocm.cpp` |

Every name is a long-stable HIP runtime API with the same signature and
semantics as its CUDA counterpart — the property that lets upstream compile
`csrc/` for both through hipify.

## 6. Tests — RED first, and the right red

Nothing to port. One new case in `tests/vt/test_rocm_backend.cpp`, mirroring
`test_cuda_backend.cpp:102-153`, HIP-header-free like the rest of that file
(every assertion through public `vt::` seams; needing a HIP header would mean the
seam leaks):

1. `SupportsGraphCapture()` is true.
2. Allocate `src`/`dst` once — fixed pointers, per the capture contract. Load
   `src` with pattern A before capture.
3. `BeginCapture` → one d2d `Copy(dst, src)` → `EndCapture`.
4. `Replay` → `dst` reads back A. Proves the graph ran.
5. **Mutate `src` in place (same address) to B, `Replay` → `dst` must read B.**
   The load-bearing assertion: replay re-executes the captured copy over the
   persistent buffer rather than replaying a snapshot, which is how a decode
   graph picks up each new token's inputs, and the failure `rocm_backend.hip:23-25`
   warns about.
6. Handle variant: `EndCaptureGraph` → `ReplayGraph` → `DestroyGraph`, same
   A-then-B assertion — that is the path decode graphs actually take.

**RED-first.** The reviewer sees it fail for the intended reason first. Two
mutations that must turn it red, in a scratch copy, restored byte-for-byte:
- `ReplayGraph` a no-op → step 4 fails.
- `EndCaptureGraph` snapshots `src` into a temporary and replays from that →
  step 4 passes, **step 5 fails**. If step 5 survives this, the test is not
  testing the thing it exists for.

## 7. Gates

1. **Build.** Clean `-Werror`, 0 warnings, HIP build on gfx1200. Non-HIP CPU
   build still object-compiles the test file and full `ctest` stays green.
2. **`ctest -R 'rocm|cross_device'`** — the existing 3 binaries plus the new
   case. Baseline on this board is 3/3.
3. **Correctness — the gate that can actually bite.** Re-run the Qwen3-0.6B
   real-oracle battery from
   [rocm-gfx1200-m2-correctness.md](rocm-gfx1200-m2-correctness.md) with capture
   ON. Qwen3 has a decode-graph sibling, so capture changes its execution path,
   and its `'The capital of france is'` prompt is a known, measured,
   version-sensitive near-tie: our CPU and ROCm backends already disagree, and
   two real vLLM builds disagree with each other. A captured graph could move it
   again. **Required outcome is an honest report, not a specific token** — re-run
   the same real-oracle comparison that resolved it before and state what it
   does. A flip the oracle also produces is not a regression; a silent one is.
4. **`VT_DECODE_GRAPH_STATS=1`** must show a non-zero replay count, proving
   capture engaged rather than falling through to eager — otherwise gate 5's
   numbers are meaningless.
5. **Performance, measured against a pre-registered prediction.** Re-run §1 on
   all three sizes, same-binary A/B (capture ON vs `VLLM_CPP_CUDAGRAPH=0`), idle
   box, reproduced 2-3x per AGENTS.md. Written down before the work so it cannot
   be retrofitted:

   | Model | Today | Predicted with capture |
   |---|---|---|
   | Qwen3-0.6B | 2.99x | ~1.36x |
   | Qwen3-1.7B | 1.90x | ~1.36x |
   | Qwen3-4B | 1.46x | ~1.36x |

   The load-bearing shape is **convergence**: capture removes the size-dependent
   term, so all three should land together. That is stronger evidence than any
   single number. 4B has the least room to move and is the weakest individual
   signal but the best convergence check. A result materially short of the
   prediction is a finding to record, not one to bury — it would mean the
   fixed-overhead term is smaller than the fit implies, and redirect to D4.
   **Do not describe any outcome as parity**; ~1.36x is the predicted floor for
   this change alone.
6. **`GetReferenceTierHits()` == 0** in any perf measurement — structurally
   impossible on a discrete board, assert anyway.
7. **Records green:** `agent-preflight.sh --staged`, `check-agent-record.py`,
   `check-doc-checkpoint.py`, `check-public-doc-tables.py`,
   `check-device-leakage.py` (this adds no shared-layer device predicate; the
   ratchet must not move).

## 8. Risks and decisions

**D1 — `LtWorkspace` can `hipMalloc` mid-capture; the most likely way this
fails.** `rocm_matmul_hipblaslt.hip:243-255` allocates the hipBLASLt workspace
lazily in the GEMM path, growing on demand (`if (need > cap) { hipFree;
hipMalloc; }`). Allocation inside a capture region is illegal and invalidates
it. CUDA's contract comment (`cuda_backend.cu:186-190`) names exactly this and
notes cuBLASLt's workspace is a one-time per-context alloc there. *Mitigation:*
the decode-graph pre-warm runs the same shapes at the same padded size before
capture, which should grow `cap` to its high-water mark. *If it does not:*
`hipStreamEndCapture` fails loudly rather than corrupting — the acceptable
direction. **W1 verifies this rather than trusting it**; a shape appearing only
at capture time would be a latent, board-specific trap.

**D2 — the Qwen3-0.6B near-tie may move.** Covered by gate 3. Separate because
it is the one outcome that could look like a regression while being nothing of
the kind, and quietly re-baselining a golden is what "never weaken a checker"
exists to stop.

**D3 — gfx12 is new silicon, its kernels still maturing upstream.**
vllm-project/vllm#45916 is open, and the oracle's log on this board printed
"Cannot use ROCm custom paged attention kernel, falling back to Triton". It
captured graphs anyway — but around *its* attention path, not ours. Our native
`rocm_paged_attn.hip` inside a capture region is the unproven combination.
*Mitigation:* W2 captures a real Qwen3 decode step, not just the W1 micro-test;
`hipStreamCaptureModeThreadLocal` turns any illegal op into a loud failure.

**D4 — the attribution is evidenced, not profiled.** §1's scaling result is real
evidence: a monotonic fall across a 7.9x compute span, with the 0.6B/1.7B pair
isolating it at constant layer count. But the alpha estimate has already moved
once under new data (1.54x -> 1.36x), a standing warning against treating the
fit as settled, and no same-tool trace (`rocprof` both sides) exists to isolate
launch overhead from everything else capture changes. Until one does, "capture
recovers ~1.6x at 0.6B" is a calibrated prediction, not a measured attribution:
W4 says "consistent with", not "because of". The residual alpha ~= 1.36x is not
claimed as a floor — it is the next thing to attack.

**D5 — one board, one arch.** gfx1200 only. hipGraph is not arch-specific and
the four #41 boards are likelier-supported RDNA3/CDNA parts, but none has run
this. Records say gfx1200; the other boards stay `PENDING-community` exactly as
the W1 approach-(b) delta does today.

## 9. Work breakdown

- **W0 — file the issue, commit this spec.** No code. *Gate: issue open, number
  agreeing in three places.* Draft:
  > **Title:** ROCm: no decode-graph capture — hipGraph seam unimplemented,
  > costing up to ~3x decode throughput vs vLLM on gfx1200
  > **Body:** `vt::Backend`'s graph-capture virtuals are CUDA-only;
  > `SupportsGraphCapture()` is false on ROCm and `support_static_graph_mode()`
  > inherits false, so every decode step pays full host launch cost. Measured on
  > RX 9060 XT (gfx1200, ROCm 7.2.3) vs a real vLLM-ROCm oracle at pin
  > `555967922` in production config, batch 8, 128in/128out: Qwen3-0.6B 184.69
  > vs 552.65 tok/s (2.99x), Qwen3-1.7B 150.50 vs 286.42 (1.90x), Qwen3-4B 98.49
  > vs 143.36 (1.46x). The gap shrinks monotonically as per-step compute grows
  > while launch count stays fixed — the signature of launch overhead, not
  > kernel quality. vLLM captures 51 piecewise + 35 full hipGraphs on this same
  > board, so hipGraph capture demonstrably works here.
  > Spec: `.agents/specs/rocm-decode-graph.md`.
- **W1 — backend seam + micro-test.** §5 port map, §6 test, RED-first. Verify D1
  with a capture around a GEMM. *Gate: 1, 2, 7.*
- **W2 — flip `support_static_graph_mode()`, run a real model.** Qwen3-0.6B end
  to end with capture engaged. *Gate: 3, 4. Nothing proceeds past a red.*
- **W3 — measure against the prediction.** Gate 5 + 6, all three sizes,
  same-binary A/B, idle box, 2-3x reproduced. Convergence is the result, not any
  single ratio. Record a miss as readily as a hit.
- **W4 — records.** This spec's `## Outcome`, `backend-matrix.md`,
  `docs/ROCM.md` §5 M3, `docs/BENCHMARKS.md` if gate 5 yields an accepted
  measurement, `NOW.md` if the row's lifecycle state moves.

## 10. Stop conditions

Return `NEEDS_DECISION` rather than improvising if:
- Gate 3 shows a Qwen3-0.6B token change the real oracle does **not** reproduce
  — a genuine correctness signal, not a near-tie, and it stops the work.
- D1 needs a real allocator change (pre-warm hook, workspace cap) — that widens
  scope from porting a seam into shared-allocator surgery.
- Gate 5 lands materially short: concretely, if Qwen3-0.6B does not fall below
  **~2.2x** (under half the predicted ~1.63x recovery), W3 stops and D4's
  profiling becomes the next spec rather than iterating blind here. The
  threshold is fixed in advance so it is a stop condition, not a judgement call
  made after seeing the number.

Return `NEEDS_CONTEXT` if the W0 issue cannot be filed (no work without one).
