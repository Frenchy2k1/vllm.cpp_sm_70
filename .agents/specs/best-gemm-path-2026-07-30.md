# Best-in-class GEMM path per regime (vLLM + SGLang + ds4 + llama.cpp) — 2026-07-30

Source-grounded research (5-lane parallel workflow `wf_029c4ac9`, all cited `file:line`)
answering the user policy **[[strive-for-best-in-class-not-just-parity]]**: for every
quant regime × phase, what is the best-in-class GEMM path across vLLM/SGLang/ds4/llama.cpp,
what does OUR engine do, the gap, and the lever + vehicle. Governs every future perf lever.

## Foundational finding (governs all decode work)
**T=1 decode is memory-bound in EVERY regime** (fp8-e4m3, nvfp4, int8, bf16/fp16): one
tensor-core MMA tile spans ≥128 rows, so tensor cores are structurally wasted at M=1 —
confirmed by llama.cpp (M=1→MMVQ/mmvf, int8-MMA only above its batch threshold,
`mmq.cu:281`) and ds4 (dp4a at T=1, dequant→cuBLAS only at M>1). We are AT/NEAR the
weight-bytes-read-once roofline via cuBLASLt gemv (bf16/fp8) + dp4a matvec (GGUF k-quant,
1:1 llama.cpp/ds4). **NO tensor-core GEMM raises decode — never recommend one for M=1.**
Decode wins are only: (a) fewer glue passes (FusedChain), (b) fewer bytes/weight
(block-fp8/low-bit), (c) fixing the one measured mis-route (below). **Prefill M>1 is the
only compute-bound place tensor-core work raises the ceiling.**

## best_path_table (regime × phase)
| regime | phase | best-in-class | OURS | lever / via |
|---|---|---|---|---|
| fp8 | decode-M1 | cuBLASLt/nvjet fp8 gemv; DeepSeek block-fp8 = FlashInfer swapAB-DeepGEMM M<32; default act = DYNAMIC per-token | per-tensor STATIC fp8 (cuBLASLt/cutlass), scalar alpha | dynamic-per-token quant (FusedChain kQuantFp8 variant); AOT-vendor DeepGEMM block-fp8 |
| fp8 | prefill | cutlass C3x scaled_mm SM120 + per-token/channel EVT; block-fp8 = DeepGEMM contiguous-grouped | shipping SM120 C3x but scalar-alpha (EVT dropped) | add block-scaled/per-token ScaledEpilogue EVT (cutlass port); AOT-vendor DeepGEMM grouped |
| bf16 | decode-M1 | cuBLASLt gemv (memory-bound) | cuBLASLt TN/NN at parity + FusedChain glue | **already-optimal** (no GEMM lever) |
| bf16 | prefill | cuBLASLt + Inductor EVT epilogue fusion | cuBLASLt at parity; ~20% glue as separate kernels | FusedChain byte-exact folds (modest ~3.5%@c1) |
| fp16 | both | cuBLASLt fp16 (same roofline as bf16) | **NOT SUPPORTED — f16 inputs THROW** (`cuda_matmul.cu:220-227`) | one-branch `CUDA_R_16F` enablement (cuBLASLt) — **rank 1** |
| low-bit | decode-M1 | dp4a matvec (llama.cpp MMVQ / ds4 Q8 dp4a); Marlin small-M | GGUF at parity; **nvfp4 W4A4 M=1 MIS-ROUTED to the prefill cutlass tactic on our 27B/35B** | fp4 dp4a decode matvec + per-M router — **rank 2, real hole on gate models** |
| low-bit | prefill | nvfp4→cutlass Fp4GemmSm120 (LIVE); GPTQ/AWQ→Marlin; k-quant→int8-MMA MMQ | nvfp4 W4A4 at FlashInfer parity (Fp4GemmSm120 LIVE); k-quant = dp4a at all M (tensor cores idle); GPTQ/AWQ stubs | int8-MMA MMQ for k-quant prefill (port llama.cpp mmq.cu); AOT-vendor Marlin w4a16 |

## Ranked levers
1. **fp16 dtype enablement** (cuBLASLt one-branch) — TRIVIAL; unblocks the entire fp16 Llama/Mistral family at bf16 roofline. Breadth.
2. **nvfp4 W4A4 decode dp4a matvec + per-M router** — MEDIUM; fixes the ONE measured perf hole on our PRODUCTION 27B/35B gate models (M=1 uses the tensor-core prefill tactic). SPEED on what we ship.
3. **fast_op kSiluMul{Fp8,Fp4}Quant** (1-launch MoE gate+up+silu epilogue) — LOW; recipes already declared (`recipes.h:131,182`); collapses ds4's 3-4 MoE launches. FusedChain + one kernel.
4. **Block-scaled/per-token ScaledEpilogue EVT on SM120 C3x fp8 + dynamic-per-token quant** — MEDIUM; reaches vLLM's default fp8 accuracy + per-channel/block. cutlass-port + FusedChain.
5. **AOT-vendor DeepGEMM block-fp8 + masked/contiguous grouped fp8 MoE** — HIGH; unblocks DeepSeek-V3/V4-class 128×128 block-fp8 checkpoints. Breadth.
6. **AOT-vendor Marlin w4a16** (GPTQ/AWQ) — HIGH; unblocks the entire GPTQ/AWQ int4 family (stubs today). Breadth.
7. **int8-MMA MMQ for GGUF k-quant PREFILL** — HIGH; k-quant prefill leaves tensor cores idle. Port llama.cpp mmq.cu.
8. **bf16 prefill FusedChain glue folds** — MEDIUM, modest (~3.5%@c1).

## FusedChain verdict (the user's "follow ds4 but with our fusion framework")
**YES for the quant PROLOGUE/EPILOGUE glue (already half-built); NO for the GEMM mainloop.**
FusedChain (`fused_recipe.h:54-67`) owns elementwise/norm/quant/rope with a per-ROW operand
model + quant terminals — it maps cleanly onto ds4 preq (quantize-once, our grouped
keep-quant `bcast` already does this), the MoE silu(gate)·up·routeweight epilogue
(`kSiluMulQuantFp8`/`kSiluMulFp4Quant`, `recipes.h:131,182`), and vLLM's Inductor
rms_quant/act_quant fusion passes (byte-exact, all backends). It has **NO matmul/reduce-over-K
opcode BY DESIGN** — every contraction stays a first-class `vt::` op that a recipe SANDWICHES
via `fast_op` (`fused_recipe.h:134-152`). Correct shape: **[FusedChain quant-prologue → fast_op =
<vendored/ported GEMM OpId> → FusedChain epilogue]**. The vendored/ported GEMM cores
(DeepGEMM block-fp8, Marlin w4a16, int8-MMA MMQ, cutlass per-token EVT, fp4 dp4a decode) are
NOT absorbed into FusedChain — they are the sandwiched op. Two recipe extensions make the
quant-glue match exact: per-token-group + UE8M0 `kQuantFp8` (DeepGEMM layout) and a
dynamic-per-token variant (vLLM default).

## Corrections carried forward
- **`Fp4GemmSm120` is LIVE**, not dormant — it is the production W4A4 GEMM for 27B/35B
  (`cuda_matmul_nvfp4_cutlass.cu:994`, `qwen3_5.cpp:1738-1961`). Right vehicle for W4A4
  PREFILL; WRONG one at M=1 decode (lever #2).
- GPTQ/AWQ "stubs only" is inferred (task #170 W0/W1) — confirm by direct read before lever #6.
- SGLang was not installed on the box; its masked-vs-contiguous grouped-fp8 claims are inferred
  from the shared DeepGEMM/FlashInfer lineage — verify against a real SGLang tree before acting.
- sm_90/sm_100 C3x TUs are build-verify-only, never executed on hardware.

## Honest ceiling
On what we SHIP today (27B/35B nvfp4-W4A4 + per-tensor fp8) we are near the achievable
ceiling EXCEPT (1) the W4A4 decode mis-route (#2) and (2) ~20% prefill glue (#8) — both
modest. The BIG levers (fp16, GPTQ/AWQ Marlin, block-fp8/DeepGEMM, int8-MMA MMQ) are
**breadth/correctness unlocks for model FAMILIES we cannot run fast at all** — they widen
coverage, they do not speed up what already runs.
