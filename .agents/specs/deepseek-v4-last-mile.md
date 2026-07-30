# DeepSeek-V4-Flash — the LAST MILE to ds4's 16.5 tok/s (keep-quant GEMM microarch + fp8 KV)

Status: **BRICK 0 (profile-only) — the measured roofline below. No implementation yet
(the ranked implementable lever comes after the coordinator synthesizes this with the
parallel source research: llama.cpp `mmvq.cu`/`mmq.cu` tiling, vLLM/flashinfer/DeepGEMM
fusion, ds4 `fp8_ds_mla`).**

## 0. Where we are
The device-resident decode is the shipped default at **~7.96 tok/s** (24-tok) on ONE DGX
GB10 vs ds4's **16.5 tok/s** on the SAME 80.7 GB `q2-imatrix` file (`CLAIM-DEEPSEEK-V4-*`).
The glue-kernel tune already collapsed the glue from 44%→21% of GPU time; the step is now
**GEMM-bound (GEMM ~78% · glue ~21% · attn ~1.3%)**. This campaign attacks the GEMMs + KV.

**CORRECTNESS ANCHOR (never regress):** the current default is token-identical / characterized
COHERENT near-tie to host ("…Paris." on P0; open-ended prompts diverge only at genuine near-tie
positions into coherent deterministic text). Every GEMM/KV change stays within that — bit-identical
where the integer-dot order is preserved, characterized near-tie (stated, RED-first) where fp8 /
reassociation reorders. Gates: `test_deepseek_v4_gguf_load` + `test_cuda_deepseek_v4` green.

## 1. The measured roofline (BRICK 0 deliverable)

**Method.** No `ncu` on the DGX (Nsight Compute absent) → the roofline is computed from `nsys`
per-kernel GPU time (`--cuda-graph-trace=node`, real 80.7 GB model, `--max-tokens 50` = 1 prefill
+ 49 decode, steady-state) ÷ the exact weight bytes each kernel moves per step. **GB10 peak DRAM
BW = 240.2 GB/s MEASURED** (a float4 copy microbench, read+write, 2 GB buffers, sm_121a) ≈ 88% of
the ~273 GB/s LPDDR5X theoretical. Byte/elem: Q8_0 = 34 B/32 = **1.0625**, IQ2_XXS = 66 B/256 =
**0.2578**, Q2_K = 84 B/256 = **0.3281**. Config: H=4096, qlr=1024, nh=64, hd=512, ne=256, topk=6,
mi=2048, ng=8, olr=1024, V=129280, 43 layers.

**Per-step weight bytes (T=1, weight read once regardless of M):**
- **QuantDotGemmQ8_0** (MLA wq_a/wq_b/wkv + o-LoRA wo_a/wo_b + shared×3, all layers = 6.04 GB;
  + lm_head 0.56 GB) = **6.60 GB/step** (wq_b + wo_b + lm_head dominate: 33.6M + 33.6M elems/layer + 529M).
- **QuantDotGemmGrouped\<IQ2_XXS\>** (moe_gate_exps + moe_up_exps, 6 selected experts/layer) = **1.12 GB/step**.
- **QuantDotGemmGrouped\<Q2_K\>** (moe_down_exps, 6 experts/layer) = **0.71 GB/step**.
- Total keep-quant weight read = **~8.43 GB/step**.

**THE ROOFLINE TABLE** (`src/vt/cuda/cuda_quant_dot.cu`; nsys totals over the 50-pass run):

| kernel | % of GPU step | weight/step | time/step | **achieved BW** | **% of 240 peak** | int8-MACs/step | MAC-rate | verdict |
|---|---|---|---|---|---|---|---|---|
| `QuantDotGemmQ8_0Kernel` | 35.6% | 6.60 GB | 44.0 ms | **~150 GB/s** | **63%** | 6.21 G | 141 G-MAC/s | **MEMORY-bound** (near roofline; ~1.6× headroom) |
| `QuantDotGemmGroupedKernel<IQ2_XXS>` | 20.2% | 1.12 GB | 25.0 ms | **~45 GB/s** | **19%** | 4.33 G | 173 G-MAC/s | **DEQUANT/LATENCY-bound** (~3-5× headroom) |
| `QuantDotGemmGroupedKernel<Q2_K>` | 10.0% | 0.71 GB | 12.4 ms | **~57 GB/s** | **24%** | 2.17 G | 175 G-MAC/s | **DEQUANT/LATENCY-bound** (~2.5× headroom) |
| `QuantizeQ8KKernel` (act-quant) | 8.1% | tiny ([P,K] act) | 10.0 ms | n/a | — | — | **LAUNCH-bound** (~139 tiny launches/step) |
| `QuantizeQ8_0Kernel` (act-quant) | 3.8% | tiny ([1,K] act) | 4.6 ms | n/a | — | — | **LAUNCH-bound** (~656 tiny launches/step) |

**Compute-bound? NO — none of them.** Every GEMM's int8-MAC rate (141-175 G-MAC/s = **0.14-0.18
TOP/s**) is <1% of GB10's int8 compute peak (CUDA-core dp4a is ~tens of TOPS; the Blackwell int8
tensor cores are hundreds of TOPS). **So a T=1 decode matvec is MEMORY-bound, and TENSOR CORES DO
NOT HELP it** — tensor-core MMQ (`ggml-cuda/mmq.cu`) is a PREFILL / large-M lever; the decode-relevant
reference is the **matvec** path `ggml-cuda/mmvq.cu` (vectorized dequant + dp4a + per-lane ILP to hide
DRAM/codebook latency). This is the key correction to "tiled/tensor-core MMQ."

**Occupancy (no ncu → reasoned):** all three GEMMs launch `dim3(32,4)` = 128 threads/block, ~no
shmem, one warp per output row → theoretical occupancy is high (N/P warps: lm_head N=129280, experts
P·N=6·2048); the grouped kernels being at only 19-24% of BW peak at high theoretical occupancy is the
signature of **LATENCY-bound, not occupancy-bound** — the warp-per-output has too little ILP to hide
the IQ2_XXS codebook (`d_iq2xxs_grid`/`d_ksigns`) + K-quant sub-block unpack + f16-scale dependent chains.

## 2. The levers, RANKED by measured headroom (grounded)

1. **Grouped-MoE dequant efficiency (IQ2_XXS + Q2_K, ~30% of the step, at only ~19-24% of BW peak — the
   5×/2.5× gap, the BIGGEST efficiency lever).** Our `QuantDotGemmGroupedKernel` /
   `QuantDotGemmKernel<W>` (`cuda_quant_dot.cu` `QuantDotGemmKernel`, the `DotIQ2XXS`/`DotQ2K` per-block
   scalar dequant) is correctness-grade: one warp/output, scalar codebook lookups, no dp4a, dependent
   f16-scale chains → LATENCY-bound. PORT TARGET: **llama.cpp `ggml-cuda/mmvq.cu`** (`mul_mat_vec_q`,
   `vec_dot_iq2_xxs_q8_1` / `vec_dot_q2_K_q8_1` in `vecdotq.cuh`) — vectorized dequant, `__dp4a`, multiple
   rows/iters per lane for ILP. Ground the port there + our `DotIQ2XXS`/`DotQ2K` (`cuda_quant_dot.cu`) +
   the CPU oracle `cpu_quant_dot.cpp VecDotIQ2_XXSQ8_K`/`VecDotQ2_KQ8_K`.
   NOTE: llama.cpp's mmvq pairs the weight with a **Q8_1** activation (32-block, with a sum term), while
   our keep-quant oracle uses **Q8_K** (256-block) — porting mmvq's dequant while KEEPING our Q8_K
   activation preserves the integer-dot bit-exactness vs our CPU reference (else it's a new characterized
   near-tie — state which).

2. **Fuse the activation-quant INTO the GEMM (`QuantizeQ8K` + `QuantizeQ8_0` = ~12% of the step,
   LAUNCH-bound: ~795 tiny launches/step + a scratch write→read round-trip).** For T=1 the activation is
   `[1,K]`/`[P,K]` (tiny) — quantizing it in the GEMM kernel's prologue (each block's threads quantize the
   activation tile they need) removes ~795 launches + the Q8_K/Q8_0 scratch traffic (our `FusedChain`
   territory; vLLM/ds4 fuse quant+GEMM). Contained, pure kernel-count/traffic win. Ground: our
   `QuantizeQ8KKernel`/`QuantizeQ8_0Kernel` + `EnsureScratch` (`cuda_quant_dot.cu`).

3. **Q8_0 GEMM coalescing/vectorization (35% of the step, at 63% of BW peak — memory-bound, ~1.6×
   headroom).** `QuantDotGemmQ8_0Kernel` reads 34-byte blocks with per-lane scalar int8 loads; float4/
   `int4`-vectorized block loads + `__dp4a` + better coalescing can push 63%→~85% of peak. Smaller relative
   gap but the biggest single share; do it after (1)+(2). Ground: our `QuantDotGemmQ8_0Kernel` +
   llama.cpp `vec_dot_q8_0_q8_1` (`vecdotq.cuh`).

4. **fp8 KV (`fp8_ds_mla`, per-token E4M3 latent) — NOT a short-context speed lever.** Attention
   (`DecodeAttnGKernel`) is only **~1.3%** of the short-context step. Its value is (a) ds4 apples-to-apples
   parity (ds4 stores fp8 KV), (b) memory footprint, (c) LONG-context scaling — the KV read grows with
   context, so it matters at 256+ tokens, not at 24. Mirror ds4 `fp8_ds_mla` (448 fp8 UE8M0 + 64 bf16
   rope, 576 B stride — already ported as a host ref in `deepseek_v4_compressor.*`); MEASURE its effect at
   a long context, and be honest it is a parity/scaling lever, not the decode-tok/s lever.

## 3. HONEST projected final tok/s
- **Hard roofline:** 8.43 GB/step at the 240 GB/s peak, with dequant free and ALL compute fused into the
  weight read = 35.1 ms/step = **~28.5 tok/s** — the absolute ceiling we will NOT reach.
- **ds4 (16.5 tok/s = 60.6 ms/step)** corresponds to reading the same ~8.43 GB at **~139 GB/s = 58% of the
  BW roofline** with near-complete fusion (ds4 folds the glue + activation-quant into the GEMMs). So ds4 is
  a "58%-of-peak, fully-fused" implementation.
- **Us today (7.96 tok/s = 125.6 ms/step):** the GEMMs alone take ~96 ms (~88 GB/s effective ≈ 37% of peak)
  and the glue+attn ~30 ms (24%). So we are at roughly HALF ds4's BW efficiency, plus un-fused overhead.
- **Realistic target:** closing the grouped-MoE 5× gap (lever 1) + fusing the activation-quant (lever 2) +
  the Q8_0 coalescing (lever 3) + folding the residual glue should reach **~13-16 tok/s**; full ds4 parity
  (16.5) needs near-complete fusion AND peak-ish BW on every GEMM, so treat **~14-16 tok/s as the honest
  target and 16.5 as the stretch**; the ~28 tok/s hard ceiling is out of reach on this quant mix. fp8 KV
  does NOT move the short-context number (it's a parity/long-context lever).

## 4. Bricks
- **Brick 0 — this profile (DONE, profile-only).**
- **Brick 1 — __dp4a vectorized-dequant matvec for the grouped kernels (DONE + DGX-gated, commit `c1f92d24`).**
  Ported llama.cpp `mmvq.cu`/`vecdotq.cuh`'s SIMD dequant into `DotIQ2XXS` (`vecdotq.cuh:920-928` + ds4
  `dev_iq2_dp4a_8`) + `DotQ2K` (`vecdotq.cuh:329-354` + ds4 `dev_dot_q2_16`), keeping our warp-per-output +
  Q8_K activation. **BIT-IDENTICAL** integer core (`__dp4a` = exact int32); the `test_cuda_quant_dot` nmse≤1e-6
  gate CAUGHT a signed-overflow UB in the sign broadcast (`(int)uint8 * 0x01010101` overflows for signs≥128) —
  RED-first worked; fixed to `unsigned`, then **2/2·105601, nmse≤1e-6 zero drift**. Real model resident-default
  TOKEN-IDENTICAL "…Paris.". **SPLIT RESULT (honest):**
  - **Q2_K grouped: 2.35× (median 265→104 µs; 24% → 56% of BW peak — the MAC WAS the bottleneck; now
    memory-bound). 10.0% → 4.4% of the step.**
  - **IQ2_XXS grouped: FLAT (median 269→265 µs, ~17-19% of peak).** GROUNDED root cause: `d_iq2xxs_grid[256]`
    is `__device__ __constant__`, and our warp-per-output has each of the 32 lanes look up a DIFFERENT grid
    index → **divergent constant-memory reads are 32-way serialized per warp** = the bottleneck, NOT the MAC.
    So dp4a (which only vectorized the already-cheap inner MAC) does not help IQ2.
  - Decode **8.01 → 8.51 tok/s (+6%, 4 stable warm runs)** vs ds4 16.5.
- **Brick 1b — IQ2_XXS grid-lookup fix (DONE + DGX-gated, commit `e4d8845b`).** Moved `d_iq2xxs_grid[256]` +
  `d_ksigns_iq2xs[128]` from `__device__ __constant__` to `__device__ GLOBAL` (`cuda_quant_iq_tables.cuh`), so
  the divergent per-lane reads go through L2 (cached, cross-lane parallel) instead of the ~32×-serialized
  constant path. BIT-IDENTICAL (same literals; `test_cuda_quant_dot` **2/2·105601 nmse≤1e-6 ZERO drift**;
  `test_cuda_deepseek_v4` 18/18·34176; `test_deepseek_v4_gguf_load` 12/12·531; real model resident-default
  TOKEN-IDENTICAL "…Paris."). **RESULT: IQ2_XXS grouped 2.45× (nsys median 265→108 µs; 19% → 46% of the
  240 GB/s peak — now memory-bound-ish like Q2_K 56% / Q8_0 63%; its share fell 22.8% → 10.0% of the step).
  Decode 8.51 → 9.58 tok/s (+12.5%, 5 stable warm runs) vs ds4 16.5** (host `=0` 7.63 → 8.47, shares the GEMM).
  The grouped-MoE dequant lever (Bricks 1+1b) is DONE: both grouped kernels now memory-bound (IQ2 46%, Q2_K 56%).
- **Brick 2 — routed gate/up activation preq-reuse / broadcast (DONE + DGX-gated, commit `99d2b282`→amended).**
  The resident-decode routed experts fed `xrep` (topk-identical copies of the shared hidden x) into the gate + up
  grouped GEMMs, re-quantizing an IDENTICAL row per expert per GEMM. Mirrors ds4's preq pattern (quantize the
  shared activation ONCE, broadcast across the P experts). The grouped providers (`cuda_quant_dot.cu`, Q8_K + Q8_0)
  detect a 1-row activation (`act.shape[0]==1 && P>1`) → quantize ONE row, kernels read block-set 0 for all p
  (`bcast`); `ops.cpp` accepts act rows == P or 1; CPU grouped ref broadcast-aware; the resident forward (eager +
  graph) passes x with `act_rows=1`, dropping the xrep buffer + the per-layer topk `AsyncCopyF`. BIT-IDENTICAL
  (identical input → identical block-quant → identical integer dot; `test_cuda_quant_dot` **2/2·105601 nmse≤1e-6**;
  `test_cuda_deepseek_v4` 18/18·34176; `test_deepseek_v4_gguf_load` **13/13·631** incl. a new broadcast==replicated
  BYTE-IDENTICAL + RED-first case; real 80.7 GB model resident-default TOKEN-IDENTICAL "…Paris.", ids byte-equal to
  host `=0`). **RESULT (nsys re-profile, 50 tok): `QuantizeQ8K` 9.8% → 7.3%, `QuantizeQ8_0` 4.5% → 4.7% (bucket
  ~14.3% → ~12.0%). Decode 9.58 → 10.02 tok/s (+4.6%, 6 stable warm runs 9.99–10.04)** (host `=0` 8.47 → 8.50) vs
  ds4 16.5 (now ~61% of ds4; campaign-cumulative host 6.44 → 10.02 = +56%). HONEST: the quant bucket is
  **LAUNCH-bound** (fixed per-launch overhead dominates — reducing per-launch rows 6→1 barely cuts it), so the
  bucket-% fell modestly; the +4.6% comes mostly from eliminating the 6×/layer xrep host-copies. The remaining
  quant lever is LAUNCH-COUNT reduction: fuse the quant INTO the GEMM, or dedup gate+up into ONE quantize.
- **Brick 3 — Q8_0 GEMM vectorized dp4a dot (DONE + DGX-gated, commit `91e51aea`→amended). BIT-EXACT but a NEAR-FLAT
  result — the Q8_0 matvec is at the 34-byte-block MEMORY WALL, not instruction-bound.** `QuantDotGemmQ8_0Kernel`
  (+grouped) read the 32 int8/block as 32 scattered int8 loads + a scalar-MAC loop; Brick 3 reads them as 8 int32
  + 8 `__dp4a` (mirrors llama.cpp `ggml-cuda/vecdotq.cuh:vec_dot_q8_0_q8_1_impl` + `VDR_Q8_0_Q8_1_MMVQ=2`). The
  Q8_0 `qs` sits at offset 2 in the 34-byte block (uint16 d + 32 int8), so — unlike the 4-byte-aligned Q8_K qs the
  Brick-1 IQ2/Q2_K path int-loads directly — a naked int32 load is mis-aligned; `GetIntB2` (bit-exact port of
  `ggml-cuda/common.cuh:get_int_b2`, two uint16 little-endian loads) reconstructs the identical byte pattern → `__dp4a`
  extracts the same signed int8 lanes as `(int)qs[p]`. BIT-IDENTICAL (`test_cuda_quant_dot` q8_0 nmse≤1e-6 **2/2·105601
  ZERO drift**; `test_cuda_deepseek_v4` 18/18·34176; `test_deepseek_v4_gguf_load` 13/13·631; real 80.7 GB model
  resident-default TOKEN-IDENTICAL "…Paris.", ids byte-equal to host `=0`; SASS: exactly 8 IDP inside the Q8_0 float
  kernel). **RESULT (nsys, 50 tok): the Q8_0 kernel moved only 2.182 s → 2.149 s (−1.5%; median 30,112 → 29,472 ns,
  −2.1%) = 45.0% → 44.6% of step ≈ 63% → ~64% of the 240 GB/s peak. Decode 10.02 → 10.07 tok/s (+0.5%, 6 warm runs
  10.05–10.09, non-overlapping vs Brick 2's 9.99–10.04)** (host `=0` 8.50 → 8.56). **THE FINDING:** dp4a + int-reads
  cut instructions but NOT the bottleneck — the kernel is DRAM-bandwidth-bound, and the 34-byte-misaligned block
  precludes true 128-bit coalesced loads (the lane-stride-34 access pattern is unchanged). Reaching 63% → ~85% needs
  an ALIGNED WEIGHT REPACK at load (separate f16 scales + 16-byte-aligned contiguous int8, ggml `mmq` load_tiles /
  ds4 warp8-preq style) enabling float4/int4 loads — numerically NEUTRAL but it trades the in-place unified-memory
  mmap design (a ~Q8_0-tower copy + load-time cost). **BOUNDARY: the bit-exact IN-KERNEL GEMM levers are now
  largely exhausted** (both grouped dequant kernels memory-bound after B1/B1b; the Q8_0 matvec at its block-layout
  wall after B3). Further speed = either the aligned repack (numerics-neutral, design-changing) OR the
  numerics-delicate last mile (dequant-cache, fp8-KV). We stand at **10.07 tok/s ≈ 61% of ds4 16.5** vs the
  ~13–15 projected bit-exact ceiling.
- **Brick 4 — aligned Q8_0 weight repack for coalesced CUDA loads — IMPLEMENTED + DGX-GATED, but a MEASURED
  NEGATIVE: the hypothesis (63%→~85% via aligned int4 loads) is REFUTED. Recommend NOT shipping (default-OFF).**
  (commit `b8a3f691`→amended.) Repacked the Q8_0 tower at load into the coalesced layout — deinterleave each
  tensor into `[all qs (32B/block, 16-byte-aligned) | all scales (uint16)]` so a warp lane reads its 32 int8 via
  two aligned `int4` (128-bit) loads (`QuantDotGemmQ8_0AlignedKernel`); `RepackQ8_0Cuda` copies off the mmap +
  `DropSpanResidency` drops the source pages (CIQ G7 pattern). Opt-in `VT_V4_Q8_0_ALIGN` (default OFF); `wo_a`
  excluded (row-sliced). **BIT-EXACT** (`test_cuda_quant_dot` **3/3·105841** incl. a new aligned==plain
  BYTE-IDENTICAL + RED-first case; `test_cuda_deepseek_v4` 18/18; `test_deepseek_v4_gguf_load` 13/13; real 80.7 GB
  model ALIGNED resident ids BYTE-EQUAL to plain, "…Paris."). **RESULT — NO SPEEDUP: decode 10.07 → 10.03 tok/s
  (FLAT, within noise; aligned 6 warm 10.02–10.06). nsys: the aligned kernel (7 full-tensor Q8_0 GEMMs) 1.671 s +
  the still-plain `wo_a` 0.506 s = 2.177 s vs Brick 3's all-plain 2.149 s — MARGINALLY WORSE (+1.3%).** PEAK RSS:
  aligned **91.05 GiB** vs plain 86.33 GiB = **+4.7 GiB** (safe — 28 GiB headroom under the 119 pool — but not the
  hoped net-flat; the source-page drop is a high-water no-op and the owned 6.1 GiB tower is added). **WHY IT FAILED:**
  the aligned int4 loads did not help (and separating the scales into their own section added a scattered per-block
  uint16 gather). Combined with Brick 3 (dp4a also no help), the Q8_0 matvec is NOT load/ALU/alignment-bound — it is
  LATENCY/OCCUPANCY-bound (1-warp-per-output, small nb/lane, ~33k tiny launches/step, the warp-reduce). **THE
  BIT-EXACT IN-KERNEL GEMM LEVERS ARE EXHAUSTED.** Recommend REVERT-code-keep-record (or merge default-OFF as a
  guarded recorded-negative). Further Q8_0 speed = a kernel-STRUCTURE change (multiple outputs/warp or a
  batched/persistent kernel — risky, uncertain) OR the numerics-delicate last mile (fp8-KV, dequant-cache) — a USER
  CALL. We stand at **10.07 tok/s ≈ 61% of ds4 16.5** vs the ~13–15 projected bit-exact ceiling.
- **Brick 5 — fp8 KV (parity/long-ctx, measured at 256+).** Rollback-able (new path default-safe; the resident default stays correct).
- **Brick 6 — FUSION lever: routed-MoE gate+up+silu into ONE kernel (ds4 `moe_gate_up_mid`) — DONE + DGX-gated, bit-exact, +5.2%.**
  CORRECTS the Brick-4 "bit-exact IN-KERNEL GEMM levers EXHAUSTED" framing: that was about the GEMM *mainloop*;
  the CROSS-kernel FUSION lever (fewer launches + less HBM glue, ds4's actual edge per `best-gemm-path`) was
  untouched and is a NEW bit-exact win. The resident routed MoE ran {gate grouped-GEMM + up grouped-GEMM +
  topk×2 `AsyncCopyF` + topk `ClampedSwiGLU`} as SEPARATE launches, writing the gate/up intermediates (`gr`/`ur`)
  to HBM and reading them back. New `QuantDotGemmGroupedFusedSwiGLUKernel<W>` (`cuda_quant_dot.cu`, ground: ds4
  `moe_gate_up_mid_decode_lut_qwarp32_kernel:17127`): ONE warp per (expert-slot p, mid-row j) computes BOTH the
  gate dot and the up dot vs the SAME broadcast Q8_K x (quantized ONCE), keeps them in registers, writes
  `adown[p*mi+j] = silu(min(gate,limit))·clamp(up,±limit)`. Wired through `MoeDeviceKernels::moe_gate_up_swiglu`
  → eager + graph resident paths (`VT_V4_FUSED_MOE=0` A/B fallback). **BIT-IDENTICAL** — same `DotSuperblock`
  integer core, same 32-lane warp-tree reduce, same `FinalFactor`, same `ClampedSwiGLUKernel` formula (α=1,β=0,
  `Sig=1/(1+e^-x)`); the route weight is NOT folded (stays in `moe_combine`, post-down ⇒ the down-GEMM input bytes
  are unchanged). Gates: `test_cuda_quant_dot` 3/3·105841, `test_cuda_deepseek_v4` 18/18·34176,
  `test_deepseek_v4_gguf_load` 13/13·631 (all UNCHANGED); real 80.7 GB model resident+graph fused vs unfused =
  **byte-IDENTICAL 49-token sequence** ("…11111 16 455 6102 294 8760 344 …" cycle). **RESULT (median of 3 warm
  runs each, --gpu --kv-cache --graph, 50 tok):** decode **10.27 → 10.80 tok/s (+5.2%, non-overlapping bands
  fused 10.79–10.82 / unfused 10.24–10.27)** vs ds4 16.5 (~65% of ds4; closes ~8.5% of the residual gap). **nsys
  (`--cuda-graph-trace=node`, 13 decode steps × 43 layers):** total kernel instances **36,647 → 32,175 (−12.2%)**;
  IQ2 `QuantDotGemmGroupedKernel<0>` (gate+up) 1548→0 folded into `…FusedSwiGLUKernel<0>` 559 (153.6→99.1 ms over
  13 steps); `ClampedSwiGLUKernel` 3913→559 (routed 6/layer collapsed); `QuantizeQ8KKernel` 2322→1763 (−559: x
  quantized once, not per gate+up — the activation-quant fusion, lever-2, came free). PEAK RSS unchanged (86.7 GiB;
  `gr`/`ur`/`gate_up_r` buffers now unused, no new alloc). **HONEST residual:** the graph path already amortizes
  host-launch overhead, so this win is the HBM gate/up round-trip + the redundant-quant, ~5%; the LARGE remaining
  ds4 gap is the still-UN-fused per-step glue — nsys hot glue over 13 steps: `RopeKernel` 119.9 ms (1677 inst),
  `QuantizeQ8KKernel` 106.7 ms, `MhcPreFinishKernel` 98.7 ms (1118), `RouteKernel` 77.4 ms (559) — plus the BW-
  efficiency gap (ds4 at ~58% of the roofline). NEXT LEVER (STEP 1b): fuse Rope into the qk-write, and the
  MhcPre/Route glue, into fewer kernels (bit-exact where elementwise; characterized near-tie where a reduction
  reorders). Commit `4ffcb96b`.
- **Brick 7 — FUSION lever: per-head RMSNorm+RoPE fused + RoPE tail parallelized (ds4 head_rms_norm_rope_tail_kernel
  / dsv4_qkv_rms_norm_rows_kv_rope_kernel) — DONE + DGX-gated, BIT-EXACT, +5.5%.** Attacks the Brick-6 nsys top glue
  (`RopeKernel`). The resident decode ran, per layer, {`rms_norm_rows` ; `rope`} as SEPARATE launches at 3 sites
  (q per-head norm+fwd-rope, kv norm+fwd-rope, and the standalone inverse o-rope), round-tripping the normalized
  q/kv through HBM, and the RoPE itself was ONE-THREAD-PER-ROW (64 threads doing the double cos/sin serially).
  New `NormRopeRowsKernel` (`cuda_deepseek_v4.cu`; ground: ds4 `head_rms_norm_rope_tail_kernel:5873` +
  `dsv4_qkv_rms_norm_rows_kv_rope_kernel:5779`): ONE kernel, block-per-row, does the RMS reduction (identical double
  block-tree reduce to `RmsNormRowsKernel`) then applies RoPE to the tail with the pairs SPLIT ACROSS THREADS.
  **BIT-IDENTICAL** because the RoPE recurrence `theta_extrap *= theta_scale` is a LEFT-FOLD — pair p's angle is
  `pos·theta_scale^p`, so a thread reaching pair p by p sequential mults from `pos` reproduces the recurrence's exact
  double product order; cos/sin stay double; no ds4 `attn_factor`/mscale (we have none). `do_norm=false`+`inverse=true`
  covers the standalone inverse o-rope with the same parallelized bit-exact tail (no norm). Wired through
  `DsaDeviceKernels::norm_rope_rows` into BOTH the eager `ForwardResidentDecodeGguf` AND the captured `V4Graph::Step`
  (`VT_V4_FUSED_ROPE=0` A/B fallback). **NOTE (RED-first caught it):** the FIRST build wired only the eager path and
  measured FLAT (10.85 vs 10.87) because the benchmark config is `VT_V4_DECODE_GRAPH=1` — the nsys showed `RopeKernel`
  UNCHANGED (no `NormRopeRowsKernel`), proving the graph body was a separate un-edited code path; wiring `V4Graph::Step`
  fixed it. **BIT-EXACT gates:** new unit case `norm_rope_rows == split {rms_norm_rows;rope}` BYTE-IDENTICAL over the
  q/kv/o modes + RED-first (`test_cuda_deepseek_v4` **19/19·66949**, was 18); `test_cuda_quant_dot` **3/3·105841**,
  `test_deepseek_v4_gguf_load` **13/13·631** (both UNCHANGED); real 80.7 GB model resident+graph, VT_V4_FUSED_ROPE=1
  vs =0 = **byte-IDENTICAL 49-token sequence** ("…11111 16 455 6102 294 8760 344…"). **RESULT (median of 4 warm runs
  each, --gpu --kv-cache VT_V4_DECODE_GRAPH=1, 50 tok):** decode **10.82 → 11.41 tok/s (+5.5%, non-overlapping bands
  split 10.80–10.85 / fused 11.41–11.44)** vs ds4 16.5 (~69% of ds4, from ~65.5%; closes ~10.4% of the residual gap).
  **nsys (`--cuda-graph-trace=node`, 14 decode steps × 43 layers):** `RopeKernel` 129.1 ms / 1806 inst (8.4% of the
  step) → `NormRopeRowsKernel` 63.9 ms / 1806 inst (4.4%) = **-50.5% RoPE-glue GPU time (2.02×)** from the parallelized
  tail, PLUS the two per-layer `rms_norm_rows` launches (q per-head + kv, ~1204 launches) folded away (`RmsNormRowsKernel`
  1820 → ~616 inst). PEAK RSS **86.68 GiB unchanged** (no new alloc; reuses qact/dn/o). **HONEST residual:** the new
  top glue is `QuantizeQ8KKernel` 110 ms (7.6%), `MhcPreFinishKernel` 106 ms (7.3%), `RouteKernel` 83.6 ms (5.7%) —
  plus the BW-efficiency gap (ds4 at ~58% of the roofline vs our GEMMs). NEXT LEVER: fuse the activation-quant into the
  consuming GEMM prologue (lever-2, LAUNCH-bound `QuantizeQ8K`), and the MhcPre reduction glue (characterized near-tie
  where the Sinkhorn/mix reduction reorders). Commit `4ffcb96b`+`e2ab0690`.
- **Brick 8 — FUSE QuantizeQ8K into the grouped-GEMM prologue (lever-2, launch-bound act-quant) — IMPLEMENTED + DGX-GATED,
  BIT-EXACT, but a MEASURED NEGATIVE: −22% decode. The "launch-bound, fuse it" hypothesis is REFUTED; REVERT-code-keep-record
  (main unaffected, stays 11.41).** Brick 7's nsys named `QuantizeQ8KKernel` (110 ms / 7.6%) the top remaining glue and called
  it LAUNCH-bound. Brick 8 folded that activation-quant INTO the two DeepSeek-V4 decode keep-quant grouped kernels' prologue
  (the fused gate+up+SwiGLU, and the routed down-proj): a new `BlockQuantizeRowQ8K` quantizes ONE activation row per block
  cooperatively into a tiny nsb·292 B shared tile (`QuantDotGemmGroupedFusedSwiGLUQuantKernel` + `QuantDotGemmGroupedQuantKernel`,
  ds4 preq-in-kernel pattern), removing the separate `QuantizeQ8KKernel` launch + its Q8_K HBM round-trip. Guarded by
  `VT_V4_FUSED_QUANT` (eligibility nsb≤128 & bcast|n%4==0), wired via the shared providers so BOTH the eager forward AND the
  captured `V4Graph::Step` inherit it (no Brick-7-style split-path trap). **BIT-EXACT** (shared `QuantizeSuperblockQ8K` ⇒
  byte-identical Q8_K, same integer dot): unit gate `test_cuda_quant_dot` **4/4·106909** (a new grouped fused-quant case,
  per-expert + BROADCAST, at the tight nmse≤1e-6 vs the CPU oracle — all pass), `test_cuda_deepseek_v4` **19/19·66949**,
  `test_deepseek_v4_gguf_load` **13/13·631**; real 80.7 GB model resident+graph `VT_V4_FUSED_QUANT=1` vs `=0` = **byte-IDENTICAL
  50-token sequence** ("…Paris. The capital of France is Paris…"). **RESULT — MEASURED NEGATIVE (4 interleaved warm runs, `--gpu
  --kv-cache VT_V4_DECODE_GRAPH=1`, 50 tok, non-overlapping bands): decode 11.46 → 8.91 tok/s (−22.3%; fused 8.87–8.93 / unfused
  11.45–11.53).** The `=0` baseline 11.46 ≈ Brick 7's 11.41 (fresh A confirmed). **nsys (`--cuda-graph-trace=node`, 20 tok) —
  the mechanism, GROUNDED:** the fused path REMOVES `QuantizeQ8KKernel` (125.0M ns, 2279 inst) but the grouped kernels BALLOON:
  gate/up SwiGLU 177,387 → 437,688 ns/inst (**2.47×**), down Q2_K 105,687 → 521,658 (**4.94×**), down IQ2 104,809 → 366,642
  (**3.50×**) — **+630M ns added vs 125M saved = net −504M ns**. Because there are THOUSANDS of GEMM blocks per grouped launch
  (the down GEMM alone is ~P·n/4 ≈ 6144 blocks/layer, each now re-quantizing ITS activation row behind a `__syncthreads`
  barrier), folding the quant into the prologue trades ONE cheap per-row quant (whose small Q8_K output is L2-cached for every
  block) for thousands of redundant re-quants + a block barrier on the GEMM critical path. The "launch-bound" framing DOES NOT
  hold under the cudagraph: standalone `QuantizeQ8K` is only 54.9 µs/inst (a cheap node), not launch-dominated. **DISPOSITION:
  REFUTED with NO salvage** — per-block re-quant is fundamental to "fold quant into the GEMM prologue" (the only quantize-once
  design IS the standalone kernel we already have). Code REVERTED, record kept; production default unchanged (11.41). **The same
  mechanism refutes fusing `QuantizeQ8_0` (5.3%) too — do NOT retry it.** **CORRECTED NEXT LEVER (from the `=0` nsys):** the
  DOMINANT kernel is now `QuantDotGemmQ8_0Kernel` at **49.3%** (927M ns — the Q8_0 MLA/o-LoRA/shared-expert/lm_head matvecs),
  which Bricks 3+4 established is latency/occupancy-bound (bit-exact in-kernel levers exhausted) → the real remaining lever is a
  kernel-STRUCTURE change (multiple-outputs-per-warp or a batched/persistent Q8_0 matvec so one warp reuses the loaded activation
  across several output rows — cuts the ~33k tiny warp launches/step), NOT a fusion. After that, the reduction glue
  `MhcPreFinishKernel` (7.7%, single-block Sinkhorn/mix) + `RouteKernel` (6.0%) is the characterized-near-tie lever. Branch
  `brick8-fused-quant` (records-only after revert), NOT pushed.
- **Brick 9 — ds4-gap Step 0 + Lever 1 (dense Q8_0 activation-quant preq grid) — IMPLEMENTED + DGX-GATED,
  BIT-EXACT, REAL +4.3% decode; PARTIAL GO for the 11.41→16.5 campaign (plan magnitude REFUTED).** Distinct
  from Brick 8 (VERIFIED before building): Brick 8 fused `QuantizeQ8K` INTO the GROUPED-MoE GEMM prologue
  (per-block cooperative re-quant behind `__syncthreads` × thousands of blocks/launch → −22%, refuted, and its
  disposition said the same refutes fusing `QuantizeQ8_0`). Lever 1 does NOT fuse into any GEMM prologue: it
  quantizes the DENSE Q8_0 activation exactly ONCE into scratch (as before) and the GEMM reads the pre-quantized
  buffer — mirroring ds4's actual `preq` pattern (`matmul_q8_0_preq_warp8_kernel` READS `xq`/`xscale` produced
  by a SEPARATE standalone `quantize_q8_0_f32_kernel`, `ds4_cuda.cu:4228`/`:4343`/dispatch `:12254`; ds4 does
  NOT fuse the quant into the GEMM). **The measured 6× gap was the standalone quant's GRID:** our
  `QuantizeQ8_0Kernel` mapped one THREAD to a whole 32-block → a `[1,K]` decode activation launched only 2 blocks
  (device-starved, 6.98 µs/launch × 646 = 4.51 ms/step); the new `QuantizeQ8_0PreqKernel` uses ds4's one-warp-
  per-32-block `{nb,m}` grid → 1.53 µs/launch × 646 = 0.99 ms/step (ds4's ~1.2 µs). Guarded `VT_V4_Q8_PREQ_QUANT`
  (default-ON, `=0` = legacy for A/B), wired inside `MatmulQ8_0Cuda`+`MatmulQ8_0GroupedCuda` so BOTH the eager
  forward AND the captured `V4Graph::Step` inherit it — nsys confirms `QuantizeQ8_0PreqKernel` appears in the
  graph body (Brick-7 split-path closed). **BIT-IDENTICAL** (amax MAX-reduction associative+exact, same d=amax/127
  + roundf; `test_cuda_quant_dot` 4/4·106081 asserts preq==legacy byte-for-byte, RED-first). **MEASURED on clean
  pristine-main `81074c94` baseline (GB10, 30-step kwin, 4 warm reps, one `flock`, worker down):** baseline
  reproduces the Lane-A denominator (`QuantizeQ8_0` 4.5094 / `QuantizeQ8K` 4.6031 ms / 84.87 ms/step / 11.44 tok/s);
  Lever 1 → dense-Q8_0 quant 4.51→0.99 ms (−3.52, 4.56×), GPU-active 84.87→81.31 ms/step, decode **11.44→11.92
  tok/s (+4.3%, non-overlapping 11.897–11.942 vs 11.414–11.453)**, token-exact golden reproduced;
  `test_cuda_deepseek_v4` 19/19·66949, `test_deepseek_v4_gguf_load` 15/15·931. **Step 0** (`kWarpsPerBlock` 4→8,
  `cuda_quant_dot.cu:944`): bit-identical, MEASURED NEUTRAL (`QuantDotGemmQ8_0Kernel` 63.2 µs/launch unchanged;
  the 41 ms GEMV is memory/latency-bound, not block-count-bound). **GO/NO-GO VERDICT — PARTIAL GO, plan magnitude
  REFUTED:** the addressable dense-Q8_0 round-trip DID fuse away (Gate A(i) bit-exact PASS, Gate A(iii) decode
  improves non-overlapping), but Gate A(ii) (`QuantizeQ8_0`+`QuantizeQ8K` 9.1→<1 ms) is NOT met — the `QuantizeQ8K`
  half (4.60 ms, grouped) is Brick-8-refuted for fusion and Brick-2-deduped, i.e. the plan mis-scoped it into
  Lever 1. Realized **+0.49 tok/s vs projected +2.3** (→13.5), ~1/5 of projection; the 11.41→16.5 ladder must be
  re-based (levers 2/3/4 re-measured before trusting 16.5). Lever 1 SHIPS default-on (bit-exact, real). Branch
  `ds4-lever1-preq-quant` (Step 0 + Lever 1 commits), NOT pushed.

## 5. Grounding (every impl cites upstream, per [[ground-every-impl-in-upstream]])
- Our kernels: `src/vt/cuda/cuda_quant_dot.cu` (`QuantDotGemmQ8_0Kernel`, `QuantDotGemmKernel<W>` +
  `DotIQ2XXS`/`DotQ2K`/`DotQ8_0`... , `QuantDotGemmGroupedKernel`, `QuantizeQ8KKernel`, `QuantizeQ8_0Kernel`).
- The dequant/vec-dot ORACLE (bit-exact target): `src/vt/cpu/cpu_quant_dot.cpp`
  (`VecDotQ8_0Q8_0`, `VecDotIQ2_XXSQ8_K`, `VecDotQ2_KQ8_K`) + `cpu_quant_act.cpp` (`QuantizeRowQ8_0/Q8_K`).
- The perf PORT reference (decode matvec): **llama.cpp `ggml/src/ggml-cuda/mmvq.cu`** (`mul_mat_vec_q`) +
  **`ggml-cuda/vecdotq.cuh`** (`vec_dot_{q8_0,iq2_xxs,q2_K}_q8_1`, `__dp4a`); PREFILL only: `ggml-cuda/mmq.cu`.
- fp8 KV: ds4 `fp8_ds_mla` + our host ref `src/vllm/model_executor/models/deepseek_v4_compressor.*`.
