# ROCm decode-graph capture — port the `vt::Backend` capture seam to hipGraph

**Row:** `BACKEND-ROCM` (backend-matrix, `ACTIVE`).
**Claim:** `CLAIM-ROCM-DECODE-GRAPH` (unclaimed at time of writing).
**Issue:** **PENDING — must be filed before implementation starts.** Per
AGENTS.md "no work without an open GitHub issue"; the number then lands in the
[roadmap intake table](../roadmap_v1.md), in this spec's header, and in the PR
body, and those three must agree. Draft issue text: §9.
**Base:** `6e4244f0` on `docs/rocm-gfx1200-m2-spec` — i.e. **this work stacks on
[PR #273](https://github.com/mudler/vllm.cpp/pull/273)**, which is where
[rocm-gfx1200-m2-correctness.md](rocm-gfx1200-m2-correctness.md) lands. That
dependency is real, not incidental: §1's measurements and gate 3's near-tie
context both come from that investigation, and `check-agent-record` fails with a
dangling link if this spec is based on `main` instead. #273 merges first. The
underlying source tree is `f323907e` (`upstream/main`, 2026-08-10) — #273 is
docs-only and touches nothing under `src/`, `include/` or `tests/`, so every
`file:line` below is equally valid against either. Re-anchor at implementation
time anyway (anchor drift is a standing hazard, and `check_links` validates
ranges).
**Board:** AMD Radeon RX 9060 XT (`gfx1200`, Navi 44, RDNA4, discrete), ROCm
7.2.3, hipClang/Clang 22.0.0. This is the ONLY board with hardware access here.
Findings must say gfx1200 and not imply the four #41 boards
(gfx1151/gfx1103/gfx1100/gfx1201) are covered.

---

## 1. Why this, and what it is worth

Measured on gfx1200, 2026-08-10, 128in/128out, our engine vs a real vLLM-ROCm
oracle at this project's own pin (`555967922`), oracle in production config
(**not** `--enforce-eager`):

| Model | Layers | hidden x inter | Ours (batch 8) | Oracle (batch 8) | Ratio |
|---|---|---|---|---|---|
| Qwen3-0.6B | 28 | 1024 x 3072 | 184.69 tok/s | 552.65 tok/s | 2.99x |
| Qwen3-1.7B | 28 | 2048 x 6144 | 150.50 tok/s | 286.42 tok/s | **1.90x** |
| Gemma-3-1B-it | 26 | 1152 x 6912 | 146.43 tok/s | 438.09 tok/s | 2.99x |

Single-stream agrees: ours TPOT 13.55 -> 24.09 ms across the two Qwen3 sizes,
oracle 5.21 -> 12.34 ms/token, ratio 2.60x -> 1.95x.

**The premise was tested by a controlled experiment BEFORE committing to
implementation, and it survived.** Qwen3-0.6B vs Qwen3-1.7B is a clean control:
**identical layer count (28)**, so an identical number of kernel launches per
decode step, with roughly **4x the compute per step** (2x hidden, 2x
intermediate). Launch overhead is fixed per step; compute is not. The gap fell
**2.99x -> 1.90x**. A gap dominated by kernel quality would not move under that
change; a gap dominated by fixed per-step overhead must. That is the cheapest
available test of this spec's premise, and it is why the spec proceeds.

**It also BOUNDS the expected win, which matters more than the direction.**
Fitting `ratio = alpha * (1 + L/C)` to the two Qwen3 points (alpha = the
size-independent efficiency advantage, L = fixed per-step overhead, C = compute
per step):

- **alpha ~= 1.54x** — persistent, does NOT shrink with size. Kernel quality,
  torch.compile/inductor fusion, the Triton attention path.
- **launch-overhead term ~= 1.45x at 0.6B, ~= 0.36x at 1.7B.**

So roughly half the 0.6B gap is launch overhead, and the realistic outcome of
this work is **0.6B 2.99x -> ~1.54x and 1.7B 1.90x -> ~1.54x**. That is a real,
worthwhile win, and it is **not parity**: a ~1.5x residual from non-launch
causes would remain, and this spec must not be written up as closing the gap.
Gate 5 is calibrated against that number, not against parity.

Two data points fitted with a one-line model is indicative, not rigorous: "4x
compute" is an approximation (attention and MLP scale differently, and KV heads
are identical at 8 in both), alpha is *assumed* size-independent, and no trace
has been taken on either side. D4 carries that.

The oracle's runs captured 51 piecewise + 35 full hipGraphs before their timed
sections. Ours captured none, because `SupportsGraphCapture()` is `false` on
ROCm, so every decode step pays full host-side launch cost.

*Measurement provenance.* Checkpoints are stock upstream, SHA-256-verified
against their HF blob hashes on download: `Qwen/Qwen3-0.6B`,
`Qwen/Qwen3-1.7B` (shards `169ad53e...30ed5` / `912becff...deff9`),
`unsloth/gemma-3-1b-it` (`3d4ef8d7...8516b6`, ungated mirror of the gated
`google/gemma-3-1b-it`). Ours = `examples/vllm-bench` on `build-hip`; oracle =
`vllm bench throughput` / `vllm bench latency` in the
`vllm-rocm-oracle:555967922-gfx1200` container (§8/D3, and the build recipe in
[rocm-gfx1200-m2-correctness.md](rocm-gfx1200-m2-correctness.md)). Single run
per cell on a board that also drives a display — indicative, and NOT the
2-3x-reproduced-idle standard gate 5 requires of the real measurement.
`Qwen/Qwen3-8B` was considered and does not fit: ~16.4 GB of bf16 weights
against 15.92 GiB of VRAM, before any KV cache.

**Second, structural reason.** `vt::Backend`'s capture virtuals
(`include/vt/backend.h:181-195`) are documented as a multi-backend seam
("CUDA Graphs / Metal ICB / Vulkan CB"), but **CUDA is the only implementation
that exists**. Metal (`metal_backend.mm:13`), Vulkan (`vulkan_backend.cpp:16`)
and ROCm (`rocm_backend.hip:22`) all carry an explicit `SupportsGraphCapture()
stays FALSE` comment. A one-implementation abstraction is an unproven
abstraction. ROCm is the cheapest possible second implementation — hipGraph is
a near-exact mirror of the CUDA graph API, where `MTLIndirectCommandBuffer` and
a pre-recorded `VkCommandBuffer` are genuinely different models — so this is
also the cheapest available evidence that the seam generalizes at all.

**What it does NOT do, stated up front.** It does not speed up Gemma-3.
`grep -rl DecodeGraph include/ src/` returns Qwen3 (dense + MoE), DeepSeek-V2/V4,
Voxtral and Laguna — **there is no `Gemma3DecodeGraph` on any backend, CUDA
included**. Gemma gains exactly nothing from this change until someone writes
that sibling (sweep-gemma.md's W7, unclaimed). The measurable vehicle here is
Qwen3-0.6B.

## 2. Scope

**In scope.**
1. `src/vt/rocm/rocm_backend.hip` — implement the six capture virtuals against
   hipGraph, mirroring `src/vt/cuda/cuda_backend.cu:194-286` virtual for
   virtual, and delete the `stays FALSE` scope note at `rocm_backend.hip:22-26`
   (replacing it with what is now true).
2. `src/vllm/platforms/rocm.cpp` — add a `support_static_graph_mode()` override
   returning `true` (today the class does not override it and inherits
   `false` from `interface.h:189`; `rocm.cpp:67` is a comment explaining that
   inheritance, not an override). Replace that comment.
3. `tests/vt/test_rocm_backend.cpp` — a capture/replay case, RED-first (§6).
4. Records: this spec, the roadmap intake row, `backend-matrix.md`'s
   `BACKEND-ROCM` row, and `docs/ROCM.md` §5's M3 text.

**Out of scope, each with a reason.**
- **`Gemma3DecodeGraph`** — does not exist on any backend; a new model-level
  decode-graph sibling is its own change with its own correctness gate. Writing
  it here would bundle two independently-reviewable things (§1).
- **`VT_BENCH_PROFILE_CONTROL`** (`cuda_backend.cu:230-262`, `cudaProfilerStart/
  Stop` arming around a targeted replay) — CUDA-profiler-specific
  instrumentation, not load-bearing for correctness. A `rocprofiler` equivalent
  is a later, separate change. The first cut omits it and says so in the code.
- **Any model-level edit.** Every decode-graph class already gates on
  `platforms::GetPlatform(...).support_static_graph_mode() &&
  b.SupportsGraphCapture()` with no `is_cuda()` anywhere (verified:
  `qwen3.cpp:494-498`, `qwen3_moe.cpp:385`, `deepseek_v2.cpp:896`,
  `voxtral.cpp:438`, `qwen3_5.cpp:7846,8167`). Flipping the two flags is
  sufficient; a model edit would mean the seam had failed.
- **Metal / Vulkan capture.** Different APIs, different specs.
- **The M3 attention-backend NAME registration** (`rocm.cpp:105`
  `get_attn_backend_priority` returns `{}`) and the stale
  `rocm.cpp:91` comment claiming `kPagedAttention` is not registered for
  `kROCM` — it IS (`rocm_ops.hip:92`), and it ran `selected=vt-native` on this
  board. Both are real, both are separate; per AGENTS.md a bug found while
  doing something else gets its own issue, not a silent fix.

## 3. Upstream chain

**No upstream vLLM analog, and that is recorded rather than papered over.**
Graph capture in vLLM is torch's (`CompilationConfig.cudagraph_mode`,
`CUDAGraphMode.FULL_AND_PIECEWISE`); there is no `csrc/` file to port. Our
`vt::Backend` capture seam is an ADDITIVE abstraction, in the same class as the
ROCm `hipMallocManaged` decision (`rocm-unified-memory-b.md` §"Upstream
grounding") — it goes in the porting inventory as additive, with this spec as
its record.

What IS mirrored is upstream's *behaviour*: vLLM on ROCm captures hipGraphs by
default in production config, which is precisely the configuration our gate
measures against. Observed directly on this board 2026-08-10 (51 piecewise + 35
full captures, ~6 s, in the `vllm bench` log) — so "hipGraph capture works on
gfx1200/ROCm 7.2.3" is measured, not inferred.

Internal anchors (base `f323907e`):
- `include/vt/backend.h:181-195` — the six virtuals + the multi-backend comment.
- `src/vt/backend.cpp:29-34` — base impls: five `VT_CHECK(false, ...)` throws,
  `DestroyGraph` a no-op. So an unimplemented backend fails loudly, and the
  model-level `enabled` gate is what keeps it off the path.
- `src/vt/cuda/cuda_backend.cu:178-286` — the implementation to mirror,
  including its capture contract comment (`:183-193`), which is the real
  specification of what a caller must honour.
- `src/vllm/platforms/interface.h:185-189`, `src/vllm/platforms/cuda.cpp:58-59`,
  `src/vllm/platforms/rocm.cpp:67`.
- `src/vllm/model_executor/models/qwen3.cpp:489-560` — `Qwen3DenseDecodeGraph::
  Impl`, the consumer: per-padded-size `SizeSlot`s with fixed-address persistent
  host/device buffers, `Refresh()` in place, invalidate-and-recapture on a
  block-table column change.

## 4. Our baseline

**REUSED as-is** (this is most of the change's value — nothing here is written):
- The whole `vt::Backend` virtual interface. No signature changes.
- Every decode-graph model class and its capture contract handling — padded
  size slots, pre-warm, persistent buffers, column-change invalidation.
- `test_cuda_backend.cpp:102-153` as the test shape to mirror.
- The `tests/CMakeLists.txt` pattern that COMPILES `test_rocm_backend.cpp`
  everywhere as a bit-rot guard but LINKS it only under `VLLM_CPP_HIP`.

**NEW, and why it cannot be reused:** the `hip*` graph calls themselves.
Mechanically a near-copy, but new code that has never been compiled here.

## 5. Port map

| CUDA (`cuda_backend.cu`, base `f323907e`) | ROCm (`rocm_backend.hip`) |
|---|---|
| `:194` `SupportsGraphCapture() { return true; }` | same |
| `:200-203` `BeginCapture` — `cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal)` | `hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal)` |
| `:204-212` `EndCapture` — `cudaStreamEndCapture` → destroy prior `exec_` → `cudaGraphInstantiate` → `cudaGraphDestroy` | `hipStreamEndCapture` / `hipGraphExecDestroy` / `hipGraphInstantiate` / `hipGraphDestroy` |
| `:214-216` `Replay` — `cudaGraphLaunch(exec_, s)` | `hipGraphLaunch` |
| `:221-227` `EndCaptureGraph` — opaque-handle variant for the multi-size slot map | same shape, `hip*`; returns `hipGraphExec_t` as `void*` |
| `:229-266` `ReplayGraph(q, graph)` | same, **minus** the `VT_BENCH_PROFILE_CONTROL` block (§2) |
| `:284-286` `DestroyGraph` — `cudaGraphExecDestroy` | `hipGraphExecDestroy` |
| `platforms/cuda.cpp:58-59` `support_static_graph_mode() { return true; }` | new override in `platforms/rocm.cpp`, same |

Every name above is a documented, long-stable HIP runtime API with the same
signature and semantics as its CUDA counterpart — which is the same property
that lets upstream compile `csrc/` for both through hipify.

## 6. Tests — RED first, and the RED must be the right red

Nothing to port (additive abstraction). One new case in
`tests/vt/test_rocm_backend.cpp`, mirroring `test_cuda_backend.cpp:102-153`,
and **deliberately HIP-header-free** like the rest of that file (every
assertion through public `vt::` seams; if a test needed a HIP header, the seam
would be leaking):

1. `SupportsGraphCapture()` is true.
2. Allocate `src`/`dst` ONCE (fixed pointers — the capture contract). Load `src`
   with pattern A pre-capture.
3. `BeginCapture` → one d2d `Copy(dst, src)` → `EndCapture`.
4. `Replay` → `dst` reads back pattern A. *Proves the graph ran at all.*
5. **Mutate `src` in place (same address) to pattern B, `Replay` again → `dst`
   must read back B.** *This is the load-bearing assertion.* It proves replay
   RE-EXECUTES the captured copy over the persistent buffer rather than
   replaying a snapshot — which is exactly the mechanism by which a decode
   graph picks up each new token's inputs, and exactly the failure
   `rocm_backend.hip:23-25` warns about ("a graph that captures a wrong stream
   is a silent correctness bug").
6. Handle variant: `EndCaptureGraph` → `ReplayGraph` → `DestroyGraph`, same
   A-then-B assertion, since that is the path the decode graphs actually take
   (`Replay`/`EndCapture` are the single-graph legacy pair).

**RED-first discipline.** The reviewer must see this fail for the intended
reason before it passes. Two mutations that MUST turn it red, run in a scratch
copy and restored byte-for-byte:
- Make `ReplayGraph` a no-op → step 4 fails (graph never ran).
- Make `EndCaptureGraph` snapshot `src` into a temporary and replay from that
  instead of the captured pointer → step 4 passes, **step 5 fails**. If step 5
  does not fail under this mutation, the test is not testing the thing it
  exists for.

## 7. Gates

1. **Build.** Clean `-Werror`, 0 warnings, HIP build on gfx1200. Non-HIP CPU
   build still object-compiles the test file (bit-rot guard) and full `ctest`
   stays green.
2. **`ctest -R 'rocm|cross_device'`** — the existing 3 binaries plus the new
   case. Baseline today on this board is 3/3 pass.
3. **Correctness, non-negotiable, and this is the one that can actually bite.**
   Re-run BOTH real-oracle batteries from
   [rocm-gfx1200-m2-correctness.md](rocm-gfx1200-m2-correctness.md) with capture
   ON:
   - **Gemma-3-1B-it** 6-prompt battery — unaffected *by construction* (no
     `Gemma3DecodeGraph` exists, so no capture happens), but re-run anyway;
     "assume unaffected" is not this project's standard.
   - **Qwen3-0.6B** — this one HAS a decode-graph sibling, so capture ON changes
     its execution path. Its `'The capital of france is'` prompt is a *known,
     measured, version-sensitive near-tie* where our CPU and ROCm backends
     already disagree and two real vLLM builds already disagree with each other.
     A captured graph could plausibly move it again. **Required outcome is an
     honest report, not a specific token**: re-run the same real-oracle
     comparison that resolved it before and state what it does. A flip that the
     oracle also produces is not a regression; a silent unreported flip is.
4. **`VT_DECODE_GRAPH_STATS=1`** (already printed by the decode-graph `Impl`
   destructors) must show a non-zero replay count — proving capture engaged
   rather than silently falling through to eager, which would make gate 5's
   numbers meaningless.
5. **Performance, as a MEASUREMENT against a PRE-REGISTERED prediction.** Re-run
   the §1 comparison on **both** Qwen3-0.6B and Qwen3-1.7B, same-binary A/B
   (capture ON vs `VLLM_CPP_CUDAGRAPH=0`), both arms on an idle box, reproduced
   2-3x per AGENTS.md. The prediction from §1's fit, written down before the
   work so it cannot be retrofitted:

   | Model | Today | Predicted with capture |
   |---|---|---|
   | Qwen3-0.6B | 2.99x | ~1.54x |
   | Qwen3-1.7B | 1.90x | ~1.54x |

   The load-bearing shape is that **both sizes should converge on the same
   ratio**, since the size-dependent term is what capture removes. Convergence
   is stronger evidence than either number alone. Record what actually happens.
   A result materially worse than predicted is a finding to record, not a
   failure to bury; it would mean the fixed-overhead term is smaller than the
   fit implies and redirect to D4's profiling. **Do NOT quote Gemma-3 as
   improved** — it has no decode-graph sibling and cannot be (§1, D6). **Do NOT
   describe any outcome as reaching parity**; ~1.54x is the predicted floor for
   this change alone.
6. **`GetReferenceTierHits()` == 0** in any perf measurement (discrete board;
   should be structurally impossible, assert anyway).
7. **Records green:** `scripts/agent-preflight.sh --staged`,
   `check-agent-record.py`, `check-doc-checkpoint.py`, `check-public-doc-tables.py`,
   `check-device-leakage.py` (this change adds no device predicate to the
   shared layer — the ratchet must not move).

## 8. Risks and decisions

**D1 — `LtWorkspace` can `hipMalloc` mid-capture, and that is the single most
likely way this fails.** `src/vt/rocm/rocm_matmul_hipblaslt.hip:243-255`
allocates the hipBLASLt workspace lazily, in the GEMM call path, growing it on
demand (`if (need > cap) { hipFree; hipMalloc; }`). Allocation inside a capture
region is illegal and invalidates the capture. CUDA's contract comment
(`cuda_backend.cu:186-190`) names exactly this ("NO cudaMalloc/cudaFree inside
the region — the scratch pool must be pre-warmed so every allocation is a pool
hit"), and notes cuBLASLt's workspace is a one-time per-context alloc there.
*Mitigation:* the decode-graph pre-warm step already runs the same shapes at the
same padded size before capture, which should grow `cap` to its high-water mark
first. *Failure mode if it does not:* `hipStreamEndCapture` fails loudly — a
thrown error, not silent corruption — which is the acceptable direction. **W1
verifies this explicitly rather than trusting it**, because a shape that only
appears at capture time would be a latent, board-specific trap.

**D2 — the Qwen3-0.6B near-tie may move.** Covered by gate 3. Called out
separately because it is the one outcome that could *look* like a regression
while being nothing of the kind, and the temptation to quietly re-baseline a
golden is exactly what this project's "never weaken a checker" rule exists to
stop.

**D3 — gfx12 is new silicon and its kernels are still maturing upstream.**
vllm-project/vllm#45916 ("Triton split-KV paged decode fallback for gfx12") is
open, and the oracle's own log on this board printed *"Cannot use ROCm custom
paged attention kernel, falling back to Triton implementation."* The oracle
captured graphs successfully anyway — but it captured graphs around *its*
attention path, not ours. Our native `rocm_paged_attn.hip` inside a capture
region is the genuinely unproven combination. *Mitigation:* W2 captures a real
Qwen3 decode step, not just the W1 micro-test; `hipStreamCaptureModeThreadLocal`
turns any illegal op into a loud capture failure.

**D4 — the launch-overhead attribution is now EVIDENCED but still not PROFILED,
and the spec must not launder the one into the other.** The §1 scaling
experiment is real evidence: a controlled 4x compute increase at constant layer
count moved the gap 2.99x -> 1.90x, which a kernel-quality-dominated gap would
not do. What is still missing is a same-tool trace (`rocprof` on both sides) that
*isolates* launch overhead from the other things graph capture changes. Until
that exists, "graph capture recovers ~1.45x" is a calibrated prediction, not a
measured attribution. Per AGENTS.md (trace both sides with the same tool before
a throughput claim; never declare a ceiling), the write-up in W4 says
"consistent with" and not "because of" unless the trace has been run. The
residual alpha ~= 1.54x is explicitly NOT claimed to be a floor — it is the next
thing to attack, and naming it is what keeps the gap open.

**D5 — one board, one arch.** gfx1200 only. hipGraph is not arch-specific and
the four #41 boards are all likelier-supported RDNA3/CDNA parts, but none of
them has run this. Every record says gfx1200, and the other boards stay
`PENDING-community` exactly as the W1 approach-(b) delta does today.

**D6 — Gemma-3-1B does not sit on the Qwen3 scaling curve, and that is an open
question this spec does not answer.** An earlier reading of the matching 2.99x
on Gemma-3-1B and Qwen3-0.6B claimed it proved architecture-independence. The
1.7B point refutes that reasoning: Gemma-3-1B has ~2.5x the MLP compute of
Qwen3-0.6B (1152x6912 vs 1024x3072), so on the fitted curve it should land near
**2.1x**, and it measures **2.99x**. So there IS an architecture-dependent
component, and Gemma is relatively slower on our side than its compute alone
explains — plausibly its unusual attention shape (head_dim 256, 4 query heads,
1 KV head) or the GeGLU/dual-rope path, neither investigated. The original
matching ratio was substantially coincidence. This does not block the work
(Gemma has no decode-graph sibling and is unaffected either way, §1), but it is
a real, recorded, unexplained residual and it must not be quietly dropped
because it is inconvenient. It wants its own investigation after W3.

## 9. Work breakdown

- **W0 — file the issue, commit this spec.** No code. *Gate: issue open, number
  agreeing in three places.* Draft issue text:
  > **Title:** ROCm: no decode-graph capture — hipGraph seam unimplemented,
  > costing ~3x decode throughput vs vLLM on gfx1200
  > **Body:** `vt::Backend`'s graph-capture virtuals are implemented only for
  > CUDA; `SupportsGraphCapture()` is false on ROCm and
  > `support_static_graph_mode()` inherits false, so every decode step pays full
  > host launch cost. Measured on RX 9060 XT (gfx1200, ROCm 7.2.3) against a
  > real vLLM-ROCm oracle at pin `555967922` in production config: ours 146.43
  > vs 438.09 tok/s (Gemma-3-1B-it) and 184.69 vs 552.65 tok/s (Qwen3-0.6B) —
  > the same 2.99x on two unrelated architectures, which points at a
  > backend-wide launch-overhead term rather than kernel quality. vLLM captures
  > 51 piecewise + 35 full hipGraphs on this same board, so hipGraph capture
  > demonstrably works here. Spec: `.agents/specs/rocm-decode-graph.md`.
- **W1 — the backend seam + micro-test.** §5 port map, §6 test, RED-first.
  Explicitly verify D1 (workspace pre-warm) with a capture around a GEMM.
  *Gate: 1, 2, 7.*
- **W2 — flip `support_static_graph_mode()` and run a real model.** Qwen3-0.6B
  end to end with capture engaged. *Gate: 3, 4 — the correctness gates. Nothing
  proceeds past a red here.*
- **W3 — measure against the pre-registered prediction.** Gate 5 + 6, on BOTH
  Qwen3-0.6B and Qwen3-1.7B, same-binary A/B, idle box, 2-3x reproduced. The
  convergence of the two ratios is the load-bearing result, not either number.
  Record what happens, including a miss.
- **W4 — records.** This spec's `## Outcome` (what was measured, what was
  rejected, why the default is what it is), `backend-matrix.md`,
  `docs/ROCM.md` §5 M3, `docs/BENCHMARKS.md` if gate 5 yields an accepted
  measurement, `NOW.md` if the row's lifecycle state moves.

## 10. Stop conditions

Return `NEEDS_DECISION` rather than improvising if:
- Gate 3 shows a Qwen3-0.6B token change that the real oracle does **not**
  reproduce — that is a genuine correctness signal, not a near-tie, and it stops
  the work.
- D1 turns out to need a real allocator change (a pre-warm hook, a workspace
  cap) — that widens scope from "port a seam" into shared-allocator surgery and
  wants its own decision.
- Gate 5 lands materially short of the §1 prediction — concretely, if
  Qwen3-0.6B does not fall below ~2.2x (i.e. under half the predicted 1.45x
  recovery) — W3 stops and D4's profiling becomes the next spec rather than
  iterating blind on this one. The threshold is stated here, in advance, so it
  is a stop condition and not a judgement call made after seeing the number.

Return `NEEDS_CONTEXT` if the issue in W0 cannot be filed (no work without one).
