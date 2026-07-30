# nvfp4 W4A4 M=1 decode GEMM — STEP-0 measurement: NO REAL HOLE (rank-2 closed)

Date: 2026-07-30 · HW: DGX GB10 (sm_121) · Ref: best-gemm-path-2026-07-30.md lever #2.

## Question (Iron-Law measure-first, before writing any kernel)
best-gemm-path rank-2 claimed the nvfp4 **W4A4 M=1 DECODE** GEMM on our production
27B/35B gate models is **mis-routed** to the tensor-core PREFILL tactic
`Fp4GemmSm120` (`cuda_matmul_nvfp4_cutlass.cu`, `qwen3_5.cpp:MatmulNvfp4CutlassModel`),
structurally wasting tensor cores at M=1 (one MMA tile ≥128 rows), and that a
memory-bound **dp4a fp4 matvec + per-M router** would be faster. The task required
proving the cost is real on hardware BEFORE building the kernel — and STOPPING if
the M=1 path is already at the memory roofline.

## Method
Gated microbench `VT_FP4_M1_BENCH=1` in `tests/vt/test_ops_nvfp4_fp4.cpp` (never
runs in the SACRED gates). Times the **production** path `vt::MatmulNvfp4Cutlass`
(the exact op `MatmulNvfp4CutlassModel` dispatches to, with the real autotuner +
plan cache) at the four real Qwen3.6-27B-NVFP4 dense W4A4 projection shapes
(hidden=5120, inter=17408, 24×hd256 q, 4 kv-heads) across M∈{1,8,32,64,128}, 5
warmup + 200 timed iters via CUDA events. Reports per-call µs, achieved
weight-read GB/s, and % of an empirical 512-MiB device-to-device copy ceiling
measured on the same box (240 GB/s). Autotuner verbose captured the M=1 tactic.

## Measurement (sm_121, DtoD ceiling 240 GB/s)
```
shape                    M   us/call    GB/s  %DtoD
qkv    [8192x5120]       1     41.43     569  237   (17-24MB weights: L2-resident in isolated loop)
qkv    [8192x5120]      32     35.30     668  278
o      [5120x6144]       1     30.78     575  240
o      [5120x6144]      32     18.49     957  399
gate_up[34816x5120]      1    389.15     258  107   (100MB weight: exceeds L2 -> true HBM roofline)
gate_up[34816x5120]     32    400.19     251  104
gate_up[34816x5120]    128    428.03     234   98
down   [5120x17408]      1    176.94     283  118   (50MB weight: HBM-bound)
down   [5120x17408]     32    174.06     288  120
down   [5120x17408]    128    204.86     245  102
```
Autotuner tactic at **M=1 (bucket=1)** — all four select **swap-AB** tiles:
```
N=8192  K=5120  -> id=2 128x32x128/swap/stream-k   (41.9us)
N=5120  K=6144  -> id=4 128x32x256/swap/dp         (30.5us)
N=34816 K=5120  -> id=4 128x32x256/swap/dp        (383.2us)
N=5120  K=17408 -> id=6 128x32x256/swap/stream-k  (173.6us)
```

## Verdict — NO REAL HOLE. dp4a matvec would NOT raise decode tok/s. STOP.
1. **The premise is refuted by measurement.** M=1 does NOT route to a prefill
   tactic. The cutlass autotuner keys on an M bucket; at bucket=1 it already
   selects **swap-AB, smallest-CTA-M (128×32)** tactics — exactly the small-M
   decode path the spec's "corrections" hypothesized. The "per-M router" lever #2
   proposed to build **already exists inside cutlass's autotuner**.
2. **M=1 is at the memory roofline.** For the weight-dominant projections that
   exceed L2 (gate_up 100MB, down 50MB) the per-call time is **flat across M**
   (gate_up: M=1 389µs ≈ M=32 400 ≈ M=128 428; down: M=1 177 ≈ M=32 174) and sits
   **at/above the empirical HBM copy ceiling** (258/283 GB/s vs 240 GB/s probe;
   the probe under-reads true peak). Flat-across-M + at-roofline is the signature
   of a MEMORY-bound kernel: the weight is streamed once at HBM bandwidth and the
   tensor-core tile under-utilization is fully hidden behind HBM latency. Were it
   compute-bound (wasting tensor cores), M=1 would be *faster* than M=128 — it is
   not; it is equal, then rises with real M. Padding M=1→128 costs no extra HBM.
3. **Arithmetic roofline confirms it.** ~12 GB of W4A4 weights streamed per token
   ÷ ~260 GB/s ≈ 46 ms/tok — exactly the sum of the per-projection M=1 times
   ×64 layers. A dp4a matvec reads the identical bytes at the identical bandwidth:
   it cannot beat "read the weight once at HBM peak," which cutlass already does.
4. **The small-shape >100% DtoD is an isolated-loop L2 artifact, not headroom.**
   qkv/o (17–24 MB) stay L2-resident across the 200-iter loop, so M=1 shows a tiny
   fixed ~10µs launch/setup overhead over their L2 minimum. In the real 64-layer
   decode these weights (2.5 GB aggregate/token) stream from HBM like the big ones
   → memory-bound there too. Even eliminating that 10µs entirely saves <3% of the
   ~638µs/layer GEMM budget, and a dp4a matvec carries its own launch overhead.

## Scope of the finding
- **27B dense** Fp4GemmSm120 projections: measured, memory-roofline-bound at M=1.
- **35B (A3B MoE)**: dense attention projections use the identical Fp4GemmSm120
  path → same regime a fortiori; the MoE expert GEMM is a *separate* kernel
  (`MoeGroupedGemmNvfp4KernelNaive`) that already keeps a naive small-P DECODE
  path by design (`cuda_matmul_nvfp4.cu`). No dense-path hole on either model.

## Corrections carried to the spec
`Fp4GemmSm120` is the right vehicle at BOTH prefill AND M=1 decode: cutlass's
per-bucket autotuner selects swap-AB small-M tactics that hit the memory roofline.
The "M=1 mis-route" was an inference; the hardware shows no mis-route and no cost.
Decode wins on these models remain only: fewer bytes/weight (lower-bit) or fewer
glue passes — NOT a different GEMM at M=1.
