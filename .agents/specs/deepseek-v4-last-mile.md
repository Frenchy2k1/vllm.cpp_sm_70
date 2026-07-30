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
- **Brick 1b (NEW, from the Brick 1 finding) — IQ2_XXS grid-lookup fix (the bigger grouped kernel, 20-23% of
  the step, grid-serialization-bound).** Move `d_iq2xxs_grid` out of `__constant__` to GLOBAL (L2-cached,
  divergent access parallelized), or mirror mmvq.cu's grid handling. This is where the IQ2 win is.
- **Brick 2 — activation-quant fusion.** **Brick 3 — Q8_0 coalescing.** **Brick 4 — fp8 KV (parity/long-ctx,
  measured at 256+).** Each rollback-able (new path default-safe; the current resident default stays correct).

## 5. Grounding (every impl cites upstream, per [[ground-every-impl-in-upstream]])
- Our kernels: `src/vt/cuda/cuda_quant_dot.cu` (`QuantDotGemmQ8_0Kernel`, `QuantDotGemmKernel<W>` +
  `DotIQ2XXS`/`DotQ2K`/`DotQ8_0`... , `QuantDotGemmGroupedKernel`, `QuantizeQ8KKernel`, `QuantizeQ8_0Kernel`).
- The dequant/vec-dot ORACLE (bit-exact target): `src/vt/cpu/cpu_quant_dot.cpp`
  (`VecDotQ8_0Q8_0`, `VecDotIQ2_XXSQ8_K`, `VecDotQ2_KQ8_K`) + `cpu_quant_act.cpp` (`QuantizeRowQ8_0/Q8_K`).
- The perf PORT reference (decode matvec): **llama.cpp `ggml/src/ggml-cuda/mmvq.cu`** (`mul_mat_vec_q`) +
  **`ggml-cuda/vecdotq.cuh`** (`vec_dot_{q8_0,iq2_xxs,q2_K}_q8_1`, `__dp4a`); PREFILL only: `ggml-cuda/mmq.cu`.
- fp8 KV: ds4 `fp8_ds_mla` + our host ref `src/vllm/model_executor/models/deepseek_v4_compressor.*`.
