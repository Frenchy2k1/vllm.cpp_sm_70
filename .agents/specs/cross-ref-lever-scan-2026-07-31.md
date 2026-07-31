# Cross-reference speed-lever scan — Laguna-NVFP4 + DeepSeek-Q8_0 (2026-07-31)

USER-directed parallel scan of 5 sources (vLLM `/home/mudler/_git/vllm`, SGLang
`/home/mudler/_git/sglang`, llama.cpp `/home/mudler/_git/llama.cpp`, ds4/DwarfStar
`dgx:~/w8run/ds4/ds4_cuda.cu`, and our own tree) for **batch-1 RAW-decode** speed
levers. Two goals, both apples-to-apple same-quant, **NO spec-decode/MTP**:
- **PATH A — Laguna-S-2.1 NVFP4 (W4A4): beat vLLM.** Our GGUF = 7.7 tok/s.
- **PATH B — DeepSeek-V4-Flash Q8_0 GGUF: beat ds4 16.5-17.2.** Our raw ~13.2.

Every candidate below was **adversarially verified against our actual code** (the
scan agents guessed our layout in places — those guesses are corrected here).

## Headline corrections (adversarial verification)
1. **ds4 register-prefetch is a MYTH.** DwarfStar's source has no cp.async / double-buffer /
   software prefetch (ds4 agent grep). Our Brick-14 (`ds4-q8-register-prefetch`,
   measured dead-flat) was rightly abandoned — the theory was wrong, not the kernel.
2. **ds4's 90% mechanism = full-warp-per-row + lane-strided blocks** (`ds4_cuda.cu`
   `matmul_q8_0_preq_warp8_kernel`): 256-thread block = 8 rows, 32 lanes/row stride K, so a
   warp reads 32 consecutive 34-byte Q8_0 blocks = **1088 contiguous bytes = one coalesced
   burst**; `__shfl` reduce, no shared mem, dp4a.
3. **Our production Q8_0 uses SUB-warp for small/medium-K** (`cuda_quant_dot.cu:1183`
   `LaunchQ8_0Subwarp`: nb≤16 → 8 lanes/row, nb≤48 → 16, else 32). At 8 lanes/row a warp-step
   reads only ~272 contiguous bytes vs ds4's 1088. **This is the one real, testable structural
   difference for the MoE-expert medium-K GEMVs.** Task #209 chose sub-warp for occupancy; ds4's
   data argues full-warp-per-row wins. NEEDS a fresh DGX A/B — not a guaranteed win.
4. **vLLM's 18.8 tok/s bar is MARLIN = W4A16** (weights NVFP4, activations bf16). vLLM's real
   GB10 default is **FLASHINFER_CUTLASS (true W4A4)** — faster. The honest Laguna bar is the
   W4A4 number; re-measure with vLLM's real default, not MARLIN.

## PATH A (Laguna beat vLLM) — ranked
| # | Lever | Source | Our status | Type |
|---|---|---|---|---|
| A1 | **NVFP4 tensor-core arm (#230)** — W4A4 forward on the 27B/35B path | (structural) | N1a landed; loader/forward pending | kernel+fusion |
| A2 | **Single-fused SM12x NVFP4 MoE** (dispatch+2 GEMMs+SwiGLU+topk, in-kernel FP4 act-quant, MMA-scales cached once) | vLLM `flashinfer_b12x_moe.py:30`; SGLang `flashinfer_cutedsl_moe.py:22` | absent | kernel+fusion |
| A3 | **Shared-expert aux-stream overlap** (1 bf16 shared expert ∥ routed) | vLLM `shared_experts.py:99`; we did it for 35B (#55) | reuse-able | host-orch |
| A4 | **SWA-512 bounded-window KV** (36 of 48 layers read ≤512 KV) | vLLM `flash_attn.py:443` | ? verify | mem-layout |
| A5 | **Fused qk-norm + dual-RoPE + KV-store** one kernel | SGLang `fused_qk_norm_rope_store.py:60` | absent | fusion |
| A6 | **Fused shared-expert INTO the routed grouped GEMM** (extra top-k slot) | SGLang `topk.py:598` | absent | fusion |
| A7 | **Full-CUDA-graph decode replay** (incl. attention) | SGLang `decode_cuda_graph_runner.py`; vLLM `cudagraph_dispatcher.py:205`; ds4 graph | Laguna absent (DeepSeek has it) | host-orch |
| A8 | **QuantizeQ8K-dedup via `MoeGateUpSwiGLUGrouped`** (GGUF path, W11 #1) | ds4 `moe_gate_up_mid`; our op exists, Laguna unwired | pending | fusion |

## PATH B (DeepSeek beat ds4 raw) — ranked
| # | Lever | Source | Our status | Verdict |
|---|---|---|---|---|
| B1 | **Port ds4 exact `warp8`: 32-lane/row + 8-rows/block for medium-K Q8_0** | ds4 `matmul_q8_0_preq_warp8_kernel` | we ship sub-warp 8/16-lane for nb≤48 | TESTABLE — the one real structural diff; DGX A/B needed, not guaranteed |
| B2 | **PDL (Programmatic Dependent Launch)** — overlap back-to-back tiny GEMV launches | llama.cpp `common.cuh:114`; SGLang `elementwise.py:650` | **ZERO — verified gap** | Blackwell batch-1 launch-latency; both paths |
| B3 | activation-quant-once (`preq`) across a token's GEMMs | ds4 `quantize_q8_0` once; llama.cpp `mmvq.cu:1191` | partial (per-GEMM) | reuse win beyond llama.cpp |
| B4 | fused gate+up+SiLU one kernel | ds4 `moe_gate_up_mid`; llama.cpp `mmvq.cu:526` | have for ds4 grouped | landed |
| — | register-prefetch | (none — myth) | Brick-14 dead | DEBUNKED |
| — | MTP self-spec | — | — | EXCLUDED (spec-decode) |

**Honest Path-B bottom line:** raw-vs-raw beating ds4 16.5 remains hard. The only
non-refuted new candidate is **B1** (ds4's exact warp8 config vs our sub-warp) — a
concrete DGX A/B — plus **B2 (PDL)** as a launch-latency win. If B1 measures flat like
the prior 6 axes, Path B raw is genuinely closed under the no-spec constraint.

## Cross-cutting VERIFIED gap
**PDL (B2/A-wide):** we have none; both llama.cpp and SGLang use `cudaTriggerProgrammatic
LaunchCompletion`/`gdc_launch_dependents` on Blackwell. Batch-1 decode is launch-latency-bound
between the dozens of µs GEMVs; PDL hides those gaps. Highest-confidence portable lever for
BOTH paths (once kernels are the bottleneck).

## Recommended order
1. **A1 NVFP4 Laguna arm** (the apples-to-apple path to beat vLLM; biggest Path-A structural win).
2. **B1 ds4-warp8 A/B** (cheap DGX measurement — settles whether Path B is truly closed).
3. **B2/A-wide PDL** (portable, both paths, verified gap).
4. **A2/A8 fused MoE** (single-fused NVFP4 + QuantizeQ8K-dedup).
