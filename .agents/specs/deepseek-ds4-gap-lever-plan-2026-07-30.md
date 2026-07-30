# DeepSeek-V4 Decode Lever Plan — 11.41 → 16.5 tok/s

**Denominator (Lane A, measured on main `18ed6f03`, GB10, 30-step steady window):** ours 85.35 ms/step GPU-active (11.2 t/s), ds4 54.86 ms (17.2 t/s). Gap = **30.5 ms/step, ds4 1.556×**. Decode is **98.5% GPU-bound** — the gap is kernel bandwidth/occupancy, *not* host overhead. Roofline: ours **42–45%** of 240 GB/s, ds4 **65–70%**.

The single most important honesty check up front: **Lane A's measured group diff overrides Lane C's framing where they conflict.** Lane C nominates MHC (Lever 2) as ~7.7% headroom, but Lane A *measured* the whole MHC group at **9.72 ours vs 9.82 ds4 — a tie (+0.1 ms, we're fractionally ahead)**. MHC is **not real headroom**. Likewise routed-expert MoE (group D): we **win by 2.6 ms** (Brick 6 fusion) — keep, don't touch.

---

## (1) Ranked lever table (measured-% anchored)

| # | Lever | Grounded gain (Lane A) | tok/s effect (cumulative) | Risk | Effort | Dependency |
|---|---|---|---|---|---|---|
| **1** | **Dense Q8_0 activation-quant fusion (preq prologue)** — fold the 646× `QuantizeQ8_0` (4.5 ms) + `QuantizeQ8K` (4.61 ms, group G) round-trip into the GEMM prologue | **~9 ms** of group E+G; removes ~9 GB avoidable DRAM traffic (write-then-reread quant activations). Lane A §5. | 11.2 → **~13.5** | **near-tie** (NMSE 5e-4 gate; core dot bit-exact) | med | none |
| **2** | **Q8_0 sub-warp GEMV tiling + occupancy** (Lane B: LANES∈{32,16,8}, 8 rows/block, `nb`-dispatch) | **~10–11 ms** residual of group E (41.1 ms `QuantDotGemmQ8_0` core). **Speculative** — see §2. | 13.5 → **~15.5** | near-tie (int core bit-exact) | med | pairs w/ #1 (same file) |
| **3** | **Route → warp-topk** (`RouteKernel` single-thread@T=1 → `router_select_warp_topk`, Lane C Lever 1) | **6.5 ms** (group C, measured 25×) | +~1.3 | **bit-exact** (tie-break preserved) | low-med | none |
| **4** | **QKV norm+RoPE+KV single-kernel fusion** (129× `NormRopeRows`@35µs → one 3.9µs kernel, group A) | **4.0 ms** (group A, measured) | +~0.9 | near-tie | med | none |
| **5** | Shared-expert gate/up merge + `moe_combine` fold-into-down-epilogue (Lane C Lever 3) | small (~1 ms) | +~0.2 | bit-exact (merge) / near-tie (fold) | low | bundle w/ #1 |
| **6** | Shared/routed stream overlap (Lane C Lever 5) | marginal (98.5% GPU-bound → little slack) | ~0 | bit-exact | med | cudagraph-capture care |
| — | **MHC mix→GEMM** (Lane C Lever 2) | **NONE — measured tie (F +0.1)** | — | — | — | **DROP** |
| — | **routed-expert MoE** (group D) | **we win +2.6** | — | — | — | **KEEP as-is** |
| — | attention heads8/fp8-KV (Lane C Lever 4) | 0.6 ms @short-ctx | — | near-tie/lossy | med-high | **DEFER** (long-context/footprint only) |

Levers 1–4 sum to the measured **~30 ms** that *is* the gap (E+G+C+A). Levers 5/6 are cleanup; MHC/D/attention are explicitly off the critical path.

---

## (2) Top lever — Q8_0 kernel-structure rewrite (Levers 1+2), + honest probability

Two distinct sub-levers living in the same file (`src/vt/cuda/cuda_quant_dot.cu`). **Split them because their confidence is very different.**

### 1 — preq prologue fusion (HIGH confidence, do first)
- **What:** eliminate the standalone `QuantizeQ8_0` (646 launches, 4.5 ms) and `QuantizeQ8K` (86 launches, 4.61 ms) passes. Quantize the activation *once* inside the GEMM prologue, ds4's `matmul_q8_0_preq_warp8_kernel` pattern (`ds4_cuda.cu:4343`, dispatch `:12254`). ds4 spends **0.42 ms total** on the same quantization (346× tiny `quantize_q8_0_f32`/`f32_to_f16`, ~1.2µs each) vs **our 9.1 ms**.
- **Why it's real headroom regardless of Brick 8:** the 9.1 ms round-trip is **present in the profiled current-main binary** (git-blob-identical to `18ed6f03`, Lane A provenance). Whatever "fusion" Brick 8 refuted, it did **not** remove this measured DRAM round-trip. This is directly the ~9 GB of avoidable traffic that pins us at 43% roofline (Lane A §5).
- **Activation reuse (pair fusion):** port `matmul_q8_0_pair_preq_warp8_kernel` (`ds4_cuda.cu:4485`) — the adjacent MLA projections reading the same hidden (q_a/kv_a, gate/up) quantize once, load the activation once across both weight matrices. We already have this on the K-quant MoE path (`QuantDotGemmGroupedFusedSwiGLUKernel:643`); the Q8_0 path lacks it.

### 2 — sub-warp GEMV tiling (SPECULATIVE, gate before over-investing)
- **Warp mapping:** template on `LANES∈{32,16,8}`; one row per subgroup; `grid = ceil(m·n/(256/LANES))`; block 256 threads. Dispatch `LANES = nb≤16 ? 8 : nb≤48 ? 16 : 32`. **Free step-0:** bump the existing full-warp launch `kWarpsPerBlock 4→8` to match ds4's 8 rows/block (`cuda_quant_dot.cu:944`).
- **Occupancy target:** ds4's own perf note (`ds4_cuda.cu:17073-17085`) measured the full-warp/big-block layout at **~16% occupancy ("grid too small to fill the device")** and the sub-warp restructure gave **~4× more resident blocks**. Our short-K MLA/LoRA projections (K∈{512,1536} → nb∈{16,48}) sit in exactly that regime: at nb=16 a 32-lane warp **wastes 50% of lanes** yet still pays the full 5-step `__shfl_down` reduction.
- **Correctness:** integer `__dp4a` accumulation is order-independent → `sumi` **bit-exact** for any LANES; only the final float scale-sum re-associates → within the existing **NMSE 5e-4** near-tie gate (`test_ops_quant_dot`). ds4 asserts its variants are bit-identical (`:17085`).

### Honest probability it helps — what's DIFFERENT vs Bricks 3/4/8
- **Brick 3 (dp4a core), Brick 4 (aligned int4 load), Brick 8 (fusion axis)** all failed to move this kernel. Lane B confirms the **core is a wall** (ds4's `dot_i8x32_dp4a:4169` = our 8×`__dp4a` core, identical) and we already ported the aligned load (`QuantDotGemmQ8_0AlignedKernel:794`).
- **Sub-lever 1 (fusion) is different from Brick 8** because it is not a numerics-glue fold — it is a **bandwidth-traffic elimination measured as still-present** (9.1 → ~0.4 ms). This is the highest-confidence piece: **~9 ms is near-certain** because it's a direct DRAM round-trip removal, not a FLOP change.
- **Sub-lever 2 (tiling) is the genuinely-new axis** none of Bricks 3/4/8 touched: **occupancy / lane-utilization**, ds4-measured and ds4-fixed. Honest probability: **high that it moves the short-K MLA/LoRA projections** (they're demonstrably in the 16%-occupancy / 50%-idle-lane regime); **low that it moves the big-K GEMMs** (lm_head K=7168, nb=224 — already lane-saturated and block-plentiful; won't budge). So the tiling win is **bounded by how much of the 41.1 ms sits in short-K projections** — realistically **half to two-thirds of the ~10 ms**, i.e. **5–8 ms, not the full 11**.

**Verification-first mandate for the implementation agent:** re-confirm on current main that Brick 8 did *not* already attempt the preq prologue fusion specifically (grep the Brick-8 record). The measured 9.1 ms round-trip says it didn't, but confirm before building.

---

## (3) Honest reachable-ceiling verdict

**Is 16.5 attainable, or is there an irreducible gap?**

- **No irreducible gap is identified.** ds4 hits 17.2 on **pure q8 kernels (fp16-cache = 0.00 GiB)** — apples-to-apples keep-quant-vs-keep-quant (Lane A). Same model, same weights, same GEMV M=1 regime. The 30.5 ms is entirely attributed to specific kernels ds4 fuses/tiles better, all of which have a concrete ds4 file:line precedent.
- **Roofline confirms headroom is physical, not wishful:** we're at 42–45% of 240 GB/s, ds4 at 65–70%. The gap is **~9 GB of avoidable activation-quant traffic** (Lever 1, directly removable) **plus low-occupancy GEMV** (Lever 2). Neither is a FLOP wall — decode is memory-bound, and the bytes we waste are enumerable.
- **Realistic best-case ladder (bit-exact/well-attributed levers only):**
  - Lever 1 (preq fusion) alone: **~13.5 t/s** (near-certain)
  - + Lever 3 (route warp-topk, bit-exact): **~14.5 t/s**
  - + Lever 4 (QKV fusion): **~15.3 t/s**
  - + Lever 2 partial (short-K tiling, 5–8 ms): **~15.8–16.5 t/s**
- **Verdict: 16.5 is reachable but sits at the top of the realistic band.** The bit-exact/high-confidence levers (1+3+4) bank **~15.3 t/s (+3.9 over baseline, ~87% of the way)**. The last ~1.2 t/s to 16.5 rides on Lever 2's occupancy tiling landing on the short-K projections — **high-probability but the one speculative dependency**. Best plausible case if tiling lands fully ≈ **ds4-parity 16.5–17.2**; conservative floor if tiling only partially helps ≈ **15.5–16.0**.
- **Wishful-thinking to reject:** don't budget any gain from MHC (measured tie), routed-MoE (we already win), attention (0.6 ms, short-ctx), or stream overlap (98.5% GPU-bound leaves no slack). Anyone proposing those as tok/s levers is fighting the measured denominator.

---

## (4) Recommended execution sequence + gates

Single worktree (`isolation: worktree`), Lever 1/2 share `cuda_quant_dot.cu`. Every gate: **SACRED near-tie distributional gate + token-exact re-check** on the DS4 golden, plus re-profile the 30-step window with `~/ds4prof/kwin.py`. Safe run recipe (no reboot): **`--gpu-vram 90 DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=28000`**, under `flock $HOME/gpu.lock`, from a **clean page cache** (our mmap leaves ~90 GiB reclaimable that starves the profile → false 11.4 t/s).

1. **Step 0 — free win:** `kWarpsPerBlock 4→8` (`cuda_quant_dot.cu:944`, matching ds4's 8 rows/block). Gate: no correctness drift, measure delta. *(minutes)*
2. **Lever 1 — preq prologue fusion** (highest confidence). First confirm Brick-8 record didn't already try preq. Build the fused-quant prologue + pair variant (`ds4_cuda.cu:4343`, `:4485`). **GATE A:** token-exact/near-tie pass **and** re-profiled `QuantizeQ8_0`+`QuantizeQ8K` share drops from 9.1 ms → <1 ms. Expected **~13.5 t/s**. This is the go/no-go for the whole campaign — if the round-trip doesn't fuse away, stop and re-measure.
3. **Lever 3 — route warp-topk** (bit-exact, independent, cheap). Port `router_select_warp_topk` (`ds4_cuda.cu:10113`). **GATE B:** bit-exact selection (tie-break `av==bv && ai<bi`), route share 6.7→<0.5 ms.
4. **Lever 4 — QKV+norm+RoPE+KV fusion.** **GATE C:** group-A share 4.75→<1 ms, near-tie holds.
5. **Lever 2 — sub-warp tiling** (the speculative bet, sequenced last so the sure wins are banked first). Templated `LANES` kernel + `nb`-dispatch. **GATE D:** measure occupancy (target ds4's ~4× block count on short-K projections) and short-K GEMV time; **if it doesn't move the nb≤48 projections, cut losses** — the bit-exact levers already bank ~15.3 t/s.
6. **Lever 5 cleanups** (shared-expert merge + combine-fold) only if a final push to 16.5 is still short.

**Stop conditions:** after Gate C, if we're at ~15.3 t/s and Lever 2 profiling (Gate D) shows the short-K projections are already saturated, declare **15.3–15.5 t/s the honest reachable point** and record the residual as roofline-bound rather than chasing the last t/s with high-variance tiling work.

**Key evidence:** Lane A denominator `~/ds4prof/{ours,ds4}.sqlite` + `kwin.py`; group diff table (E −19.8, C −6.5, G −4.4, A −4.0, F tie, D +2.6 ours). Lane B: `cuda_quant_dot.cu:756/944` (ours), `ds4_cuda.cu:4343/4485/17073-17085/17119` (ds4 preq/pair/occupancy). Lane C: `RouteKernel` `cuda_deepseek_v4.cu:570` vs `router_select_warp_topk` `ds4_cuda.cu:10113`; MHC tie confirmed by Lane A group F.