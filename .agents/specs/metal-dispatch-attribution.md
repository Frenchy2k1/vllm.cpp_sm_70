# Metal dispatch attribution — the Metal backend is SUBMIT bound, not kernel bound

**Rows this spec owns:** `BACKEND-METAL-MLX` (work row `M3c`, batched encoders)
and `BACKEND-GATE-METAL-MLXLM` (the MLX-LM competitor floor). It also supplies
the missing evidence base for both: an execution trace on Apple, which the
project did not previously have.

Measured 2026-07-27 on the M4 (Apple M4, 16 GiB unified, macOS 26.5.2), commit
`41d7f8d7`, Qwen3-1.7B-bf16 p=512 g=128 b=1, every run under the `${GPU_LOCK}`
exclusion.

---

## 0. Executive answer

1. **The binding constraint is the DISPATCH MODEL, not kernel quality.** The
   Metal backend opens one command buffer per op and does `commit` +
   `waitUntilCompleted` (`metal_ops.mm` `Encoder::Finish`). That round trip
   measures **~186 us** on this box, and a 128-token generation pays it
   **50,944 times**.
2. **That alone caps us BELOW the competitor floor.** ~395 dispatches per decode
   token x 186 us = **~73 ms/token of pure round trip = a ~13.6 tok/s ceiling
   with infinitely fast kernels**. MLX-LM measures **27.9 tok/s** on the same box
   and model. **No amount of kernel work can reach the floor while the dispatch
   model stands.** This is the first hard, arithmetic statement the Metal track
   has about its own ceiling.
3. **The recorded lever ranking was wrong and is corrected here.** The M3b
   narrative named "`M3c` (batched encoders) and a simdgroup GEMM" as co-equal
   next levers. They are not: `M3c` is the **unlock**, and the GEMM lever
   **cannot be cashed until `M3c` lands**, because the GEMM's own ceiling sits
   above the dispatch ceiling.
4. **The small elementwise kernels are ~96% pure overhead.** `vt_rms_norm`,
   `vt_silu_and_mul`, `vt_reshape_and_cache` and `vt_rope_from_cache` together
   spend **4.69 s of wall for 0.19 s of GPU work** in a 34.47 s run. They are
   13.6% of total runtime and 0.8% of the GPU's actual work.
5. **No op is silently running on the CPU.** `GetReferenceTierHits()` is **0** in
   both arms, so the S5 portable tier is not involved and cannot be blamed. That
   hypothesis is closed, not left open.

---

## 1. Method, and why it is in-process

AGENTS.md § "TRACE THE EXECUTION, not just the code" requires an execution trace
before any throughput claim: reading dispatch code says what COULD run, not what
DID. On CUDA that instrument is `nsys`. On Apple it is Instruments' Metal System
Trace, **which requires a full Xcode**. The M4 has Command Line Tools only
(`xcrun -f xctrace` -> "not a developer tool"; no `/Applications/Xcode.app`), so
no external profiler can run there at all.

So the trace is collected in-process, by `VT_METAL_PROFILE`
([include/vt/metal_profile.h](../../include/vt/metal_profile.h), implemented in
[metal_ops.mm](../../src/vt/metal/metal_ops.mm)). Per dispatch it separates:

| phase | definition | what it isolates |
|---|---|---|
| `encode_s` | `Encoder` ctor to start of `commit` | HOST work: pipeline lookup, argument binds |
| `wait_s` | wall time inside `commit` + `waitUntilCompleted` | the full submit/sync round trip |
| `gpu_s` | `MTLCommandBuffer.GPUEndTime - GPUStartTime` | the interval the GPU actually executed |

`gpu_s / wait_s` is the decision number. Near 1.0 = genuinely compute bound, so
kernels are the lever. Far below = submit bound, and kernel tuning cannot reach
the difference.

**Both arms come from ONE binary**, the arms selected by
`VT_OP_PROVIDER_DISABLE=mlx`, which is the same-binary A/B the benchmark
protocol demands.

---

## 2. Measured — native MSL arm (fully attributed)

34.47 s duration, 3.71 tok/s aggregate output.

| | value | note |
|---|--:|---|
| dispatches | 50,944 | one command buffer each |
| host encode | 0.213 s | 0.6% of the run: host binding is NOT the problem |
| submit + wait wall | 33.882 s | **98.3% of total runtime** |
| real GPU busy | 22.566 s | **66.6% of that wall** |
| round-trip overhead | **11.316 s** | 32.8% of the entire run |

| kernel | count | encode ms | wait ms | gpu ms | gpu/wait |
|---|--:|--:|--:|--:|--:|
| `vt_matmul` | 21,632 | 70.6 | 26,513.9 | 20,751.2 | 78.3% |
| `vt_rms_norm` | 14,464 | 78.4 | 2,803.3 | 116.4 | **4.2%** |
| `vt_paged_attention` | 3,584 | 8.4 | 2,379.7 | 1,619.1 | 68.0% |
| `vt_reshape_and_cache` | 3,584 | 8.5 | 725.6 | 27.2 | **3.7%** |
| `vt_silu_and_mul` | 3,584 | 34.8 | 718.3 | 20.7 | **2.9%** |
| `vt_rope_from_cache` | 3,584 | 8.5 | 632.7 | 25.1 | **4.0%** |
| `vt_greedy_argmax` | 128 | 1.4 | 35.2 | 4.7 | 13.4% |
| `vt_embedding` | 128 | 1.8 | 27.7 | 0.7 | 2.4% |
| `vt_cast` | 128 | 0.3 | 23.1 | 0.7 | 3.2% |
| `vt_rope_cos_sin_cache` | 128 | 0.4 | 22.5 | 0.7 | 2.9% |

**The round-trip constant, four independent estimates.** A kernel that does
almost no GPU work measures the fixed cost directly, as `(wait - gpu) / count`:
`vt_rms_norm` **186 us**, `vt_silu_and_mul` **195 us**, `vt_reshape_and_cache`
**195 us**, `vt_rope_from_cache` **170 us**. They agree, which is what makes this
a constant of the dispatch model rather than a property of any one kernel.

---

## 3. Measured — MLX provider arm (partially attributed, stated as such)

20.55 s duration, 6.23 tok/s. 36,480 dispatches through our `Encoder`, wait
10.472 s, GPU busy 3.554 s, ratio **33.9%**.

**This arm is NOT fully attributed and must never be quoted as if it were.**
MLX-served GEMMs never enter our `Encoder`, so they are invisible to this
instrument: `vt_matmul` falls from 21,632 to 7,168 (MLX absorbed exactly 14,464,
the `kMatmulBT` weight GEMMs), and **10.08 s of the 20.55 s is unaccounted** —
MLX's own dispatch plus its compute. Closing that needs the provider seam
instrumented too (row `M3c-4` below).

What the arm DOES establish: replacing the GEMM cut runtime 34.47 s -> 20.55 s,
and the *remaining* per-op traffic still costs 6.92 s of non-GPU time on our side
alone. Study §5.3 predicted exactly this as the "sync tax": one command-buffer
commit+wait per delegated op unless we share a command buffer with MLX's stream.
It is now measured, not predicted.

---

## 4. The ceiling arithmetic (the load-bearing claim)

Decode dispatches per token: `(50,944 - ~396 prefill) / 128` = **~395**.

| assumption | per-token round trip | implied ceiling |
|---|--:|--:|
| 186 us (small-kernel constant) | 73.5 ms | **13.6 tok/s** |
| 222 us (measured mean, 11.316 s / 50,944) | 87.7 ms | **11.4 tok/s** |

MLX-LM on the same box and model: **27.9 tok/s** (b=1, re-measured 2026-07-27).

**So the dispatch model alone puts the competitor floor out of reach by ~2x,
before a single kernel is considered.** This is why `M3c` is not one lever among
several; it is the precondition for every other Metal performance lever having
anywhere to land.

---

## 5. What MLX does differently

MLX's lazy graph terminates in an eager per-op encode layer (study §5.1,
`mlx/backend/metal/eval.cpp:32-48`), but the **eval boundary**, not the op
boundary, is where it commits. One `eval()` encodes many ops into a command
buffer and commits once. It therefore pays the ~186 us round trip a handful of
times per token where we pay it ~395 times. Its `steel` simdgroup-matrix GEMM is
a real second advantage, and our MLX-provider arm already cashes part of it, but
the structural difference is the submission granularity.

---

## 6. Structured spike contract

### Scope

One question: **where does the Metal backend's wall-clock time actually go, and
therefore which lever can move it.** In scope: per-dispatch time attribution
(host encode vs submit/wait vs real GPU busy), the resulting ranking of the
Metal performance levers, and the permanent instrument that produces it. Out of
scope: implementing any lever. This spike does not change a kernel, does not
change dispatch behaviour, and claims no speedup. It changes what the next
change should be.

### Upstream chain

vLLM is an orchestration layer and its kernels live downstream, so the chain
that matters here is the Apple one, not the CUDA one. `MTLCommandQueue` ->
`MTLCommandBuffer` -> `MTLComputeCommandEncoder` (Metal framework), with
`GPUStartTime`/`GPUEndTime` as the only runtime-exposed GPU-side timestamps
available without Instruments. The competitor's chain is MLX
(`mlx/backend/metal/eval.cpp:32-48`, study §5.1): a lazy graph whose commit
boundary is `eval()`, over `steel` simdgroup-matrix GEMM kernels. There is no
vLLM-side upstream to mirror for this row: vLLM has no Metal backend, so the
"mirror vLLM" directive has nothing to bind here, and that absence is recorded
deliberately rather than left implicit.

### Our baseline

`metal_ops.mm` `Encoder`: one `MTLCommandBuffer` plus one
`MTLComputeCommandEncoder` per op, `Finish()` doing `endEncoding` + `commit` +
`waitUntilCompleted`, i.e. fully synchronous at op granularity.
`SupportsGraphCapture()` is false (`MTLIndirectCommandBuffer` unimplemented). 18
of 75 ops native, the rest served by the S5 portable CPU tier. Measured
consequence: **50,944 dispatches, 33.882 s of submit+wait, 22.566 s of GPU busy,
~186 us fixed round trip** on Qwen3-1.7B-bf16 p=512 g=128 b=1.

### Port map

Nothing is ported. The one file changed for the instrument is
[metal_ops.mm](../../src/vt/metal/metal_ops.mm) (the `Encoder` records three
timestamps), plus the new public surface
[include/vt/metal_profile.h](../../include/vt/metal_profile.h). The files the
FOLLOW-UP rows will touch, named now so the work breakdown is grounded:
`metal_ops.mm` (`Encoder` lifetime, `M3c-1`/`M3c-2`), `metal_msl.h` +
`metal_ops.mm` (`kFusedChain` coverage for the elementwise chain, `M3c-3`),
`metal_mlx_provider.mm` (provider-side attribution and command-buffer sharing,
`M3c-4`/`M5b`), `metal_msl.h` (`vt_matmul_bt_gemv`, `M3d`; a simdgroup-matrix
GEMM for prefill remains unbuilt).

### Tests to port

**None, and that is a decision rather than an omission.** vLLM has no Metal
backend and no dispatch-attribution facility, so its `tests/` tree contains no
module this could re-express; the test-porting directive has nothing to carry.
What this change does add is our own gate:
[test_metal_backend.cpp](../../tests/vt/test_metal_backend.cpp) "Metal dispatch
profile attributes host encode, wait and GPU busy" — asserts the facility
records nothing while disabled, exactly one row per dispatch while enabled, that
the named kernel row appears, and the `gpu_s <= wait_s` invariant the entire
attribution rests on. The follow-up rows inherit the existing M3b oracle gate
unchanged.

### Gates

- **Correctness, a precondition never traded off.** Every follow-up row re-runs
  the M3b device-appropriate oracle gate (16/16: hard anchor REQUIRE plus the
  <= 0.5 nat near-tie band). Batching or reordering that changes tokens is a bug,
  not a win, and is reverted rather than re-baselined.
- **Performance.** `BACKEND-GATE-METAL-MLXLM` binds ours >= MLX-LM on every axis
  (throughput, req/s, TTFT, TPOT/ITL, peak memory). This spike does NOT move that
  row; it establishes why it cannot pass yet and what must land first.
- **Attribution completeness.** A future perf claim on the MLX arm is not
  admissible until `M3c-4` closes the 10.08 s blind spot (§3).
- **Reproduction is a gate.** §7 recipe, re-run >= 2 times, same-binary A/B,
  whole series under one `${GPU_LOCK}` exclusion.

### Dependencies

- **Hardware:** the M4 is the only Apple box available; it has Command Line
  Tools only, so Instruments/`xctrace` is unavailable and this in-process
  instrument is the sole trace source. No other Apple silicon is claimed.
- **Blocking order (CONFIRMED, then re-scoped):** `M3d` depended on `M3c-1`, and
  with dispatch fixed it did land a 1.89x win. But the SHAPE of `M3d` named here
  was wrong: profiling after `M3c-1` showed the time is in m=1 GEMVs, not in the
  prefill GEMMs a simdgroup-matrix kernel would serve. `M5b` depends on `M3c-4`
  for the evidence to judge it.
- **Quiet-box dependency:** a BINDING number additionally needs
  `com.localai.worker` booted out and the aerial wallpaper disabled, both of
  which need interactive sudo the agent does not have (commands in
  [environment.md](../environment.md)).
- **No new build dependency.** The instrument uses only Metal and the C++
  standard library; it does not require the MLX provider to be built.

### Work breakdown

| row | work | why ranked here |
|---|---|---|
| `M3c-1` | **LANDED 2026-07-27.** Batch dispatches into one command buffer, commit at a flush point | Recovered ~11.0 s of the measured 11.3 s: b=1 1.50x (3.68 -> 5.52 tok/s), commits 50,944 -> 454, GPU busy time UNCHANGED, gpu_busy_frac 66.6% -> 98.4%. `M3d` is now the live lever |
| `M3c-2` | **SUBSUMED BY `M3c-1`, 2026-07-27.** The batched design already waits only at flush points (`Synchronize`/`Copy`/`Memset`/`Free`/`DestroyQueue`), never per op | The spec expected these to be separable; the implementation showed they are not. Recorded rather than left as phantom remaining work. What is still owed is ASYNC flush (commit without `waitUntilCompleted`, double-buffered), which the 98.4% GPU-bound result says is now worth little |
| `M3c-3` | Route the tiny elementwise chain (rms_norm, silu, rope, reshape_and_cache) through the existing `kFusedChain` Tier-1 interpreter on Metal | Cuts dispatch COUNT at source: those four are 25,216 of 50,944 dispatches for 0.8% of GPU work |
| `M3c-4` | Instrument the MLX provider path so the MLX arm is fully attributed | Removes the 10.08 s blind spot; a gate cannot bind on a partially attributed arm |
| `M3d` | **RE-SCOPED AND LANDED 2026-07-27 as a decode GEMV, NOT the simdgroup GEMM this row assumed.** Shape-class profiling after `M3c-1` showed **21,464 of 21,632 matmuls are m=1 decode GEMVs, all BT**; only 168 are prefill GEMMs, so the simdgroup-matrix GEMM would have optimised 0.8% of the dispatches. Implemented instead: one simdgroup per output column, streaming the contiguous BT weight row coalesced, reduced with `simd_sum` | **1.89x** decode and **27x more accurate** (f32 NMSE 2.68e-14 vs the tile kernel's 7.40e-13). Required a Metal golden re-capture with vLLM-oracle re-validation, because the committed golden was captured with the less accurate kernel. The simdgroup-matrix GEMM for the 168 prefill dispatches is UNBUILT and is now a separate, much smaller row |
| `M5b` | Share a command buffer with MLX's stream instead of committing per delegated op | The measured form of study §5.3's sync tax |

### Dead ends (recorded so the next pass does not re-try them)

**Small-m BT GEMV-style kernel for batched decode (m=2..16) — TRIED 2026-07-27,
MEASURED SLOWER, REVERTED.** The obvious follow-up to `M3d` was to extend the
one-simdgroup-per-column GEMV to carry m activation rows, reusing each B row
across all m. It is CORRECT (f32 NMSE 2.42e-14 vs the tile kernel's 6.61e-13,
27x better, and the SACRED gate passed on a re-captured golden) and it is a
REGRESSION at every m > 1, monotonically worse as m grows. Two reps, same-binary
A/B under the GPU lock:

| B | tile | small-m | ratio |
|--:|--:|--:|--:|
| 1 | 5.41 / 5.54 | 10.72 / 10.54 | 1.98x / 1.90x (the `M3d` GEMV, unchanged) |
| 2 | 9.33 / 9.28 | 8.08 / 8.07 | **0.87x** |
| 4 | 13.92 / 13.88 | 10.83 / 10.81 | **0.78x** |
| 8 | 18.32 / 18.18 | 12.91 / 12.92 | **0.71x** |
| 16 | 21.60 / 21.64 | 14.28 | **0.66x** |

**Root cause, and it is structural rather than a tuning miss.** The kernel gives
each simdgroup one output column and therefore re-reads ALL of A from device
memory per simdgroup: with N columns that is `N * m * K` of A traffic, which
GROWS with m. That is exactly the observed shape of the regression. The tile GEMM
stages both the A and B tiles in threadgroup memory shared across a 16x16 output
tile, so its A traffic is 16x lower. At m=1 A is a single row, small enough to
stay cache-resident, which is why the same structure WINS there and loses
everywhere else.

**Do not re-try by tuning this shape** (unroll factor, simdgroups per
threadgroup, accumulator count). The missing property is A REUSE ACROSS COLUMNS,
which needs 2-D blocking. The correct kernel for batched decode is therefore the
same one prefill wants: a **2-D blocked simdgroup-matrix GEMM**, i.e. the
original `M3d` idea, now correctly scoped to m > 1 rather than to the m=1 case
that turned out to dominate the dispatch count.

**Process note worth keeping:** the golden was re-captured (with full oracle
re-validation) BEFORE the speed A/B was run, and then had to be reverted along
with the kernel. Benchmark first, re-capture only once the change is known to be
worth keeping.

### Tuning attempts on the simdgroup GEMM (both measured, both rejected)

The 32x32 / BK=8 shape that shipped was not assumed to be optimal; two obvious
widenings were tried and MEASURED, and both are recorded so they are not retried.
Same-binary runs, GPU lock, b=1 p=512 g=128.

| shape | tok/s | TTFT | verdict |
|---|--:|--:|---|
| **32x32, BK=8 (shipped)** | **13.27** | 2524 ms | baseline |
| 32x32, BK=32 | 13.28 | 2546 ms | **NEUTRAL** — reverted |
| 64x64, BK=16 | 11.48 | 4004 ms | **WORSE (-14%)** — reverted |

**BK=32 was neutral, which refutes the barrier hypothesis.** The reasoning for it
was that BK=8 pays two threadgroup barriers per 8 elements of K, so BK=32 would
amortise the same barriers over 4x the MACs. It changed nothing, so barrier
traffic is not this kernel's limit.

**64x64 was materially worse, and the cause is occupancy.** A 64x64 tile with 128
threads needs 16 `simdgroup_float8x8` accumulators per simdgroup and 24 KB of
threadgroup memory (sa 4 KB + sb 4 KB + sc 16 KB), which leaves room for only one
threadgroup per core and removes the latency hiding the smaller tile enjoyed. A
64x64 tile is not wrong in principle, but it needs MORE THREADS (8 simdgroups,
256 threads) and a smaller output staging buffer, not simply wider blocks over
the same 128 threads. That is the form worth trying next, if any.

**What this means for the remaining gap.** With barriers and tile width both
excluded, the mm kernel's cost is most likely the staging itself: every element
of A and B passes through a branchy scalar `vt_load` and is written to
threadgroup memory as f32, doubling LDS traffic for bf16 operands. A
dtype-specialised staging path (vector loads, no per-element switch) is the next
hypothesis, and it is the SAME hypothesis as the decode GEMV's.

### GEMM roofline: the kernel is 3x off MLX, measured in isolation

**The decisive diagnostic for the parity goal.** End-to-end deltas could not say
whether the mm kernel or its surroundings were at fault, so it was timed in
ISOLATION at the model's real prefill shapes, against MLX's steel GEMM in the
SAME binary through the provider seam (`VT_MM_BENCH=1`, opt-in microbenchmark in
`tests/vt/test_metal_backend.cpp`).

| shape | ours | MLX steel | ratio |
|---|--:|--:|--:|
| qkv 512x2048x2048 | 990 GFLOP/s | 2640 | **2.7x** |
| mlp-up 512x2048x6144 | 1019 | 3325 | **3.3x** |
| mlp-dn 512x6144x2048 | 1002 | 3293 | **3.3x** |

Ours sits at **~1.0 TFLOP/s, ~23% of the M4's ~4.3 TFLOPS fp32 peak**; MLX is at
**~77%**. **The gap is inside our kernel**, not in dispatch, staging branches,
barriers or tile width, all of which were separately measured and excluded.

**A note that reframes the MLX provider.** MLX's raw GEMM is 3x ours, yet
enabling the MLX provider end to end measured NO faster (12.21 s vs 12.17 s at
b=1). Those two facts together locate the provider's cost precisely: study §5.2's
correction says `Matmul::eval_gpu` re-`set_data`s its output from MLX's own
allocator, so every delegated GEMM pays an O(M*N) copy back plus its own
commit+wait. At prefill shapes that output is megabytes per call. **Fixing the
provider's output path may be worth more than rewriting our kernel**, and it is
now the cheaper of the two routes to prefill parity.

**Dead end recorded — bfloat simdgroup matrices are NOT available.** The obvious
fix, and what MLX does, is to keep operands in bf16 through the mma instead of
expanding to f32 in threadgroup memory (half the LDS traffic, faster mma).
Attempted and it does not compile: `simdgroup_load` has no overload for a bfloat
matrix on this toolchain (`no matching function for call to 'simdgroup_load'`,
runtime MSL compile). MSL's simdgroup_matrix supports half and float only.
Staging as `half` would work but is REFUSED: half saturates at 65504 and
activation outliers can exceed it, which trades a correctness hazard for speed.

### OPEN LEAD: 64x64 with 8 simdgroups is 1.8x faster and currently WRONG

The strongest unclosed lead for the parity goal. The failed 4-simdgroup 64x64
attempt concluded it needed MORE THREADS rather than wider blocks; that form was
then built and measured:

| kernel | qkv 512x2048x2048 | mlp-up 512x2048x6144 | mlp-dn 512x6144x2048 |
|---|--:|--:|--:|
| shipped 32x32, 4 simdgroups | 990 GFLOP/s | 1019 | 1002 |
| **64x64, 8 simdgroups (2x4)** | **1712** | **1907** | **1821** |
| MLX steel (target) | 2640 | 3325 | 3293 |

**1.8x, closing roughly half the distance to MLX.** Each simdgroup owns a 32x16
quadrant = 4x2 blocks of 8x8, i.e. 8 accumulators, deliberately not the 16 that
sank the 4-simdgroup version on occupancy. Threadgroup memory 20 KB
(sa 2 + sb 2 + sc 16).

**IT IS NUMERICALLY WRONG AND WAS NOT SHIPPED.** NMSE 0.994 (i.e. garbage) on
f32 64x512x128 — a shape with NO ragged edge in M, N or K, so this is an indexing
or synchronisation defect, not an edge-guard slip. Inspection of the tile
coverage (sg_r = sgitg/4 over 2x32 rows, sg_c = sgitg%4 over 4x16 columns), the
accumulator-to-`sc` mapping, the barrier order and every stride did not locate
it. Recorded rather than guessed at further.

**BISECTED 2026-07-27. Three hypotheses tested and REFUTED; the search space is
now much smaller.**

1. **The simdgroup mapping is EXONERATED.** Both decompositions were built and
   both fail identically: 2x4 (sg_r = sgitg/4 over 2x32 rows, acc[4][2]) and 4x2
   (sg_r = sgitg/2 over 4x16 rows, acc[2][4]). Speed is the same either way
   (1712/1907/1821 vs 1640/1861/1781 GFLOP/s), so the defect is in the STAGING or
   BARRIER structure, not in how simdgroups divide the tile.
2. **The threadgroup-size limit is REFUTED.** The suspicion was that 20 KB of
   threadgroup memory pushes `maxTotalThreadsPerThreadgroup` below the 256
   threads dispatched, which in a Release build is silent UB rather than a fault.
   `DispatchGrid2D` now asserts that limit (a fix worth having regardless, kept
   in the tree) and **it does not fire**.
3. **A host/kernel thread-count mismatch is REFUTED.** `kMmTile = 64` and
   `kMmSimdgroups = 8` are both correctly applied, so 256 threads really are
   dispatched and the kernel's `nthreads` matches.

**THE SHARPEST CLUE, and where to resume: the failure is m-dependent.**
m=2 and m=16 PASS (2.5e-06, 2.7e-06); m=64 and m=512 FAIL (NMSE 0.994, 1.001).
With BM=64, small m means tile rows >= m are zero-padded and masked at write-out,
so **the defect only affects tile rows >= 16**. That is consistent with part of
`sa` never being staged, or being staged after it is read. Instrument the sa
staging for rows 16..63 first: dump `sa` for a single threadgroup, or stage a
known constant and check which rows survive to the output.

**THE DEFECT IS NOW FINGERPRINTED (2026-07-27).** A per-row diagnostic
(`VT_MM_ROWDIAG=1`, committed and skipped by default) feeds A[r][*] = r+1 and
B = 1 so each output row's VALUE names the source row it actually read. At
m=n=k=64 on the 64x64/8-simdgroup kernel:

```
rows  0..15  correct
rows 16..31  ZERO            (never written)
rows 32..47  read A[16..31]  (written, WRONG source row)
rows 48..63  ZERO            (never written)
```

**Read this carefully, because it excludes the obvious explanations.** Pure
thread under-coverage would leave zeros but every written row would carry its
CORRECT source. Here the written rows are correct in count but shifted: the tile
is being filled at HALF the expected row spacing, so 64 rows of source land in
32 rows of destination, and the rest is never touched.

That is a STRIDE disagreement between the staging writes (`sa[r][kk]`, which the
compiler lays out with whatever row pitch it chooses for
`threadgroup float sa[VT_MM_BM][VT_MM_BK]`) and the `simdgroup_load(..., &sa[R][kk],
VT_MM_BK)` calls, which assume the pitch is exactly `VT_MM_BK`. It works at
BM=32 and breaks at BM=64, which is consistent with the compiler padding the
inner dimension for bank-conflict avoidance at the larger size.

**Next step, and it is concrete:** stop passing `VT_MM_BK` as the
`elements_per_row` argument and derive the ACTUAL pitch, or restructure `sa` as a
flat `threadgroup float[]` with an explicitly computed stride so the write and
the read cannot disagree. Verify with `VT_MM_ROWDIAG=1` (expects "NONE"), then
`VT_MM_BENCH=1` for the 1.8x, then the f32 64x512x128 arm.

**Reproduce:** the f32 64x512x128 arm catches the defect in one run;
`VT_MM_ROWDIAG=1` says which rows and which sources; `VT_MM_BENCH=1` gives speed.

### Risks/decisions

- **Batching changes failure semantics (accepted, mitigated).** Today a failed
  command buffer is attributed to exactly one op by construction, and
  `VT_CHECK([cmd_ error] == nil, ...)` names it. Batched, an error covers a span.
  `M3c-1` must keep a debug mode that reverts to one-op-per-buffer.
- **The ~186 us is THIS box** (M4, CLT-only). Not claimed for other Apple
  silicon, and any other machine re-measures before citing it.
- **The MLX arm must not be read as fully attributed** (§3). Recorded here
  because the number is quotable-looking and would be wrong to quote.
- **Decision: the instrument ships permanently rather than as a throwaway
  patch.** There is no external profiler on this platform, so deleting it would
  delete the only evidence base for every future Metal perf claim. Cost when off
  is one relaxed atomic load per dispatch, beside an operation that already
  blocks on the GPU.
- **Decision: no lever was implemented in this change.** Measuring and fixing in
  one commit would have made the attribution unauditable against a moving tree.

## 7. Repro

```sh
# M4, under the GPU lock; one binary, both arms.
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_METAL=ON \
  -DVLLM_CPP_MLX=ON -DMLX_ROOT=$HOME/mlx-venv/lib/python3.9/site-packages/mlx
cmake --build build -j10 --target vllm-bench
M=$HOME/hf-cache/hub/models--mlx-community--Qwen3-1.7B-bf16/snapshots/9cd6692855d3e06772228e9a962b2606359b2d24
# native arm (fully attributed)
/usr/bin/lockf -k /tmp/gpu env VT_METAL_PROFILE=1 VT_OP_PROVIDER_DISABLE=mlx \
  ./build/examples/vllm-bench --model $M --num-prompts 1 --input-len 512 \
  --output-len 128 --concurrency 1
# MLX arm (partially attributed, see section 3)
/usr/bin/lockf -k /tmp/gpu env VT_METAL_PROFILE=1 \
  ./build/examples/vllm-bench --model $M --num-prompts 1 --input-len 512 \
  --output-len 128 --concurrency 1
```

Isolation achieved: the three `actions.runner` LaunchAgents were verified
job-idle, booted out and restored; the whole series held `/usr/bin/lockf -k
/tmp/gpu`. Not achieved: `com.localai.worker` and the aerial wallpaper stayed up
(no passwordless sudo), so these remain **INDICATIVE, not binding**, exactly like
every prior M4 number. The RATIOS this spec turns on (gpu/wait per kernel) are
far more robust to that contention than absolute throughput is, since both terms
are measured inside the same dispatch.
