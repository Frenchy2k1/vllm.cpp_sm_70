# ds4 Q8_0 decode-GEMV — the ds4 SIDE ncu, captured and DIFFed (2026-07-31)

Measurement-only (no production code). Closes the one residual of `ds4-q8-ncu-2026-07-30.md`:
**ds4's ncu side was never captured** ("the agent stalled in prefill"). Captured now under
`sudo ncu` on the GB10 (sm_121a), and DIFFed against our recorded kernel counters. Answers the
raw-vs-raw question: *why does ds4 move the byte-identical 6.15 GiB Q8_0 tower at ~90% of the
240 GB/s roofline while our `QuantDotGemmQ8_0Kernel` moves it at ~67%?*

Base main `0078e85a`. Branch `ds4-q8-raw-mechanism` (records-only, NOT pushed).
Oracle: `~/w8run/ds4/ds4` + `ds4_cuda.cu`, model `DeepSeek-V4-Flash-…-imatrix.gguf` (80.76 GiB).

---

## 0. Method (and the two prior-agent traps, both fixed)

`sudo ncu` HW counters ARE unblocked on the box (`ncu 2025.3.1.0`, passwordless sudo).
ds4 decodes under a CUDA graph, so `--graph-profiling node`. Target the decode kernel by
name; SKIP prefill by launch count. Prompt `"The capital of France is"`, greedy, `-n 30`.
Worker absent (`docker ps` empty). All runs under `flock $HOME/gpu.lock`, ONE process.

Two traps that stalled the 8 prior attempts, both diagnosed and fixed here:

1. **`--launch-skip 20000` skipped EVERYTHING.** The decode kernels
   (`matmul_q8_0_preq_warp8_kernel`, …) have DISTINCT symbol names from the prefill
   variants (`…_batch_`, `…_tok2/4/8_`, `…_kslice_`), so the name filter counts DECODE
   launches only — of which there are ~259/step × 30 ≈ 8k total. `20000` overshot all of
   them → header-only CSV. FIX: `--launch-skip 88` (past 2 decode steps of the plain kernel,
   44/step) / `180` for the pair kernel.
2. **`--replay-mode kernel` OOM-REBOOTED the box.** First real attempt (kernel replay, default
   ctx **32768**) hard-rebooted the GB10: ncu's kernel-replay save/restore buffer over an
   80 GB-resident model in the 119 GB UNIFIED pool tips it OOM (`gb10-unified-memory-oom-reboots-box`).
   FIX: **`--replay-mode application`** (re-runs the app per pass, NO large save/restore buffer)
   + `-c 512` + `--gpu-vram 85` + a live watchdog that kills ncu if `avail_mb < 12000`. Under
   application replay the min avail stayed **26.6 GB** across both captures — box safe.

Captured CSVs: `~/w8run/ncu_out/ds4_plain2.csv`, `ds4_pair2.csv` (8 steady launches each).

---

## 1. THE DIFF (headline deliverable) — ds4 vs ours, byte-identical Q8_0 weights

Our-side numbers are the recorded Brick-13 sudo-ncu (`ds4-q8-ncu-2026-07-30.md`); ds4-side
are this session's medians over the steady decode launches.

| counter | ds4 `preq` (1:1 kernel) | ds4 `pair` (fused ×2) | **OURS** `QuantDotGemmQ8_0Kernel` |
|---|---|---|---|
| **long-scoreboard / issue-active** | **17.2** | 36.3 | **54.4** |
| registers / thread | **56** | 48 | **39** |
| achieved occupancy (`sm__warps_active` %) | **60.3 %** | 52.9 % | **71.9 %** |
| L1 sector hit-rate | 97.4 % | 97.3 % | 96.7 % |
| **L2 sector hit-rate** | **2.6 %** | 6.7 % | **10.7 %** |
| bytes/sector (global ld) | 1.70 | 1.67 | 2.0 |
| grid / block | 4096 / 256 | — / 256 | 256–512 / 256 |
| achieved DRAM BW (prior nsys byte-acct) | **~217 GB/s = 90 %** | — | **~161 GB/s = 67 %** |

`dram__throughput.avg.pct_of_peak_sustained_elapsed` returns **n/a** on GB10 (unified LPDDR5X
exposes different metric names) for BOTH sides — the 90 %/67 % are the prior nsys byte-accounting,
which the counters below now EXPLAIN.

### The monotone signature (three data points, one law)
`regs`  56 → 48 → 39   (ds4_preq → ds4_pair → ours)
`longSB` 17 → 36 → 54   **inversely tracks register count**
`occ`   60 → 53 → 72   **NON-monotone → occupancy is NOT the driver**

Long-scoreboard (the global-memory-latency stall) falls as registers RISE and is INDEPENDENT of
occupancy. That is the textbook signature of **register-resident memory-level parallelism (MLP)**:
the kernel with more registers keeps more independent 34-byte weight-block loads in flight, so the
compute of block *i* hides the DRAM latency of block *i+1…i+k*, keeping LPDDR5X saturated.

---

## 2. What the DIFF REFUTES (evidence-first — two task hypotheses are dead)

- **L2-residency hypothesis — REFUTED.** ds4's L2 hit-rate (**2.6 %**) is LOWER than ours (10.7 %).
  ds4 is NOT keeping weight blocks / scales resident in L2; it reads MORE from DRAM, not less. The
  90 % is not an L2 tiling trick. (Consistent with LPDDR5X: weights are read once per step, no reuse.)
- **Coalescing hypothesis — REFUTED.** ds4's bytes/sector (**1.70**) is slightly WORSE than ours
  (2.0). Both over-fetch the 34-byte misaligned blocks heavily; both have it L1-ABSORBED (97 % L1
  hit). ds4 does NOT use a better thread→byte mapping. Our Brick-4 aligned-repack correctly measured flat.
- **Occupancy — NOT the driver** (ds4 wins at LOWER occupancy, 60 % vs 72 %). Confirms Brick-11's occupancy-negative.

## 3. What the DIFF CONFIRMS — the mechanism, precisely named

**ds4's edge is memory-LATENCY HIDING via register-resident MLP (a deeper per-warp in-flight
load pipeline), reached at LOWER occupancy by spending registers (56 vs 39), NOT via L2 residency,
coalescing, or occupancy.** long-scoreboard 17.2 vs 54.4 = 3.2× better latency hiding on the
byte-identical stream.

### Source grounding (both kernels read, `ds4_cuda.cu:4343` vs `cuda_quant_dot.cu:834`)
The two plain kernels are **algorithmically identical**: warp-per-SINGLE-row, 8 warps/block (256
threads), block-strided `for (b = lane; b < blocks; b += 32)`, `#pragma unroll` inner 8×`__dp4a`,
34-byte blocks {half d; int8 qs[32]}, `warp_sum` reduce, lane-0 write. **NEITHER has
`__launch_bounds__`** (grep: 0 in ds4_cuda.cu; 0 in our file). Yet nvcc allocates ds4 **56** regs
and ours **39** — ds4 gets the deeper load pipeline, ours is register-starved into a shallow one.
Two concrete code deltas that plausibly drive nvcc's choice (candidates to test, not yet proven):
- **Activation representation.** ds4 pre-quantizes the activation into SEPARATE 4-byte-aligned
  `int8 xq[]` + `float xscale[]` arrays → a single **aligned** `int32` load per 4 elems
  (`load_i8x4_i32_aligned`). Ours reads the activation from the INTERLEAVED 34-byte `BlockQ8_0`
  struct via `GetIntB2` = two 16-bit loads (2-byte-aligned). Interleaved addressing raises our
  register/address pressure and constrains load hoisting.
- **Weight load.** ds4 assembles the weight int32 byte-wise (`load_i8x4_i32_unaligned`); ours uses
  `GetIntB2` (2×`uint16`). Different load-instruction selection changes how many block-loads nvcc
  keeps outstanding.

### Why the Brick-13 "ILP" attempt failed while ds4 succeeds (the key reconciliation)
Brick 13 recorded the ILP axis as MEASURED-NEGATIVE and CLOSED the Q8_0 front. But Brick-13's ILP
was **N-output-ROWS-per-warp** (`QuantDotGemmQ8_0MultiRowKernel`, `cuda_quant_dot.cu:872`) — a
kernel that CHANGED the row→warp mapping. Measured (ncu, `~/w8run/ncu_out/ilp_ilp2.csv`): it drove
long-scoreboard to **85–127** (WORSE than the 54 baseline) and dropped occupancy to 42–82 %. **It
moved the wrong axis.** ds4 proves the target (longSB 17) is reachable on the SINGLE-row structure —
via intra-row load-pipeline DEPTH, not multi-row. So the front was closed on the wrong mechanism.

---

## 4. VERDICT

**(A) A portable mechanism EXISTS and is newly-evidence-backed — with two dead hypotheses removed
and a bounded, honestly-modest payoff.**

- **Portable lever (named precisely):** raise our plain Q8_0 GEMV's *intra-row* memory-level
  parallelism to ds4's operating point — a DEEPER software-pipelined weight-block load pipeline
  (manually prefetch the next K blocks' scale+qs into registers before consuming the current
  `__dp4a`), letting/forcing nvcc to ~56 regs at ~60 % occupancy (e.g. `#pragma unroll K` on the
  block loop, and/or `__launch_bounds__(256, 3)` / `-maxrregcount` to lift the register cap),
  optionally adopting ds4's pre-quantized 4-byte-aligned activation for a clean aligned int32 load.
  **Bit-exact** (same integer `__dp4a` order, same f16-scale fold). This is DISTINCT from the failed
  Brick-13 multi-row ILP and is the 7th, best-targeted attempt — the only axis the 6 prior bricks
  (4 align / 8 fusion / 11 sub-warp / 12 launch / 13 multi-row-ILP) never touched.
- **Dead ends (do NOT build):** L2-residency tiling (ds4's L2 hit is LOWER) and coalescing/thread-map
  rework (ds4's bytes/sector is WORSE) — both REFUTED by the counters.

**Brutally-honest bounds (this is why it is not oversold):**
1. **Compiler-sensitive.** nvcc chose 39 vs 56 regs on near-identical code; FORCING the deeper
   pipeline may not linearly reproduce ds4's longSB 17 (it may spill, or plateau at ~30). It is a
   real candidate, not a guaranteed win, on a front where 6 axes already went flat/negative.
2. **Bounded payoff — does NOT overtake ds4 raw.** Full capture of the plain-kernel sub-gap
   (67 %→90 % BW, −10.5 ms/step on the Q8_0 bucket) is worth **~+1.5–2 tok/s** (≈12.1→~13.9 GPU,
   ~11.7→~13.3 wall) per the prior byte-accounting. That closes the Q8_0 SUB-gap but does **NOT**
   reach ds4's raw **16.5 tok/s** — the MoE-expert + glue residual is a separate ~15 ms/step front.
   So **a pure raw-vs-raw OVERALL beat via this one kernel is not on the table**; raw decode stays
   behind ds4 even with a full Q8_0 capture. The `pair` kernel diff (longSB 36 at 48 regs) also
   shows ds4's launch-consolidation (2 weight streams/launch) is a SECOND, independent contributor
   already characterized in `ds4-q8-kernel-profile-2026-07-30.md` §1 — not re-opened here.

**Net:** the raw-vs-raw honest truth is that ds4's Q8_0 GEMV wins by a *real, portable, latency-hiding
mechanism* (register-resident MLP), worth one more targeted bit-exact attempt for ~+1.5–2 tok/s; but
even fully captured it narrows rather than erases the raw decode gap. The user's rejection of the
spec/MTP "trick" stands — this is the legitimate raw lever, correctly identified, with its ceiling
stated plainly. Decision (build the intra-row-MLP kernel as Brick 14 / accept ~13 t/s raw ceiling)
surfaced to the user.

## 5. Reproduce
```
# on dgx.casa, worker absent, flock held; application replay = memory-safe:
~/w8run/ncu_out/run_ds4_ncu2.sh "matmul_q8_0_preq_warp8_kernel" 88 8 plain2   # 1:1 kernel
~/w8run/ncu_out/run_ds4_ncu2.sh "matmul_q8_0_pair_preq_warp8_kernel" 180 8 pair2  # fused ×2
# metrics: lts/l1tex__t_sector_hit_rate.pct, smsp__…long_scoreboard…ratio,
#          sm__warps_active.avg.pct_of_peak_sustained_active,
#          smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio, launch__registers_per_thread
```
