# AWQ + GPTQ native quantized compute (`CLAIM-QUANT-AWQ-GPTQ`)

Spike for the two most common community weight formats: **AWQ** (AutoAWQ /
llm-awq, W4A16) and **GPTQ** (AutoGPTQ / GPTQModel, W2-8A16). Rows
[`QUANT-AWQ`](../quantization-matrix.md) / [`QUANT-GPTQ`](../quantization-matrix.md)
were `INVENTORIED` with NO compute path. This spec inventories the whole vLLM +
kernel chain, the exact files to port, the upstream tests, the GB10
kernel-selection logic, and a row-sized W-breakdown. Sources pinned to vLLM
`555967922` (0.26.0.dev0) at `/home/mudler/_git/vllm`.

## Scope

- **In:** AWQ (`quant_method=="awq"`, 4-bit, group-wise, asymmetric) and GPTQ
  (`quant_method=="gptq"`, 2/3/4/8-bit, group-wise, optional act-order) —
  config parse, the packed weight layout (pack axis, bit order, zeros/scales,
  `g_idx`), the CPU reference dequant, and the Marlin repack + GEMM that vLLM
  actually runs on the GPU (incl. GB10 selection). Support ALL vLLM's modes:
  the Marlin fast path (primary on sm_121), the Triton/exllama fallbacks, and
  the AWQ/GPTQ MoE experts.
- **Out (this claim, W1):** only the INT4 CPU unpack+dequant-to-bf16 primitive
  and its unit gate. Loader wiring, the GPU Marlin compute, 8/2/3-bit and MoE
  are later named bricks.
- **DISTINCT lanes (do NOT touch):** MXFP4/NVFP4 packed-fp4 (DeepSeek-V4 /
  Kimi-K3), compressed-tensors WNA16 (`QUANT-CT-WNA16`), `moe_wna16`
  (`QUANT-MOE-WNA16`). AWQ/GPTQ are INTEGER affine schemes; on the GPU they ride
  the SAME Marlin family we already vendor for NVFP4 (`src/vt/cuda/marlin/`,
  `include/vt/cuda/marlin_repack.h`, `src/vt/cuda/cuda_moe_marlin.cu`).

## Upstream chain

What AWQ and GPTQ are: both quantize a linear `W[in=K, out=N]` group-wise and
affine, `w ≈ (q - z) * s` per group of `group_size` input rows. They differ ONLY
in packing axis, bit order and zero convention:

| | AWQ | GPTQ |
|---|---|---|
| `qweight` | int32 `[K, N/8]`, packed along **N** (out) | int32 `[K/8, N]`, packed along **K** (in) |
| bit order in int32 | AWQ reverse `[0,4,1,5,2,6,3,7]` | standard `[0..7]` |
| `qzeros` | int32 `[K/G, N/8]`, N-packed, AWQ order | int32 `[K/G, N/8]`, N-packed, standard order |
| `scales` | `[K/G, N]` fp16/bf16 | `[K/G, N]` fp16/bf16 |
| zero-point | explicit asymmetric (`zero_point`) | `q_zp + zero_offset`, `1` (v1) / `0` (v2); Marlin symmetric = `uint4b8` bias-8 |
| act-order | none | optional `g_idx[K]` (desc_act) row→group perm |
| bits | 4 (8 only via Marlin) | 2, 3, 4, 8 |

Grounding files (pinned vLLM `555967922`):
- Layout: `vllm/model_executor/layers/quantization/utils/quant_utils.py:758-867`
  (`gptq_pack==pack_rows` K-axis; `awq_pack==pack_cols` after `[0,2,4,6,1,3,5,7]`
  interleave; `unpack_cols`). Reverse-AWQ order: `auto_awq.py:73-77`.
- Config: AWQ `auto_awq.py:171-411` (`AutoAWQConfig.from_config` :237-257,
  `TYPE_MAP={4:uint4}` :179, filenames :234, detect :259-283). GPTQ
  `auto_gptq.py:97-303` (`TYPE_MAP={(4,True):uint4b8,(8,True):uint8b128}`
  :101-104, `desc_act`/`sym`/`dynamic`/`checkpoint_format`).
- CPU / reference dequant: AWQ `awq_triton.py:11-105`
  (`iweights=(packed>>shifts)&0xF`, `shifts=reverse_awq_order*4`,
  `(iweights-zeros)*scales`); GPTQ
  `csrc/libtorch_stable/quantization/gptq/qdq_4.cuh` (`dequant_4bit_8_gptq`,
  `w=(q-z)*s`), zero `= stored + zero_offset`, `zero_offset` at
  `.../gptq/q_gemm.cu:201-202,256` (`use_v2_format ? 0 : 1`); 8-bit `qdq_8.cuh`,
  2/3-bit `qdq_2.cuh`/`qdq_3.cuh`.
- GPU Marlin (the fast path): repack
  `csrc/libtorch_stable/quantization/marlin/awq_marlin_repack.cu`,
  `.../gptq_marlin_repack.cu`; GEMM `.../marlin.cu` + `marlin_template.h` +
  `dequant.h`. AWQ is first rewritten to GPTQ-standard layout by
  `_convert_awq_to_standard_format` (`auto_awq.py:93-168`) then handed to the
  shared kernel. MoE: `AutoAWQMoEMethod` (`auto_awq.py:547-842`) +
  `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py` → Marlin MoE.

## Our baseline

- We have NO AWQ/GPTQ path today (`QUANT-AWQ`/`QUANT-GPTQ` were `INVENTORIED`).
- We DO vendor the Marlin kernel family for NVFP4 W4A16
  (`QUANT-NVFP4-CT-W4A16` ACTIVE): `src/vt/cuda/marlin/`,
  `include/vt/cuda/marlin_repack.h`, `src/vt/cuda/cuda_moe_marlin.cu`,
  `include/vllm/model_executor/models/dense_nvfp4_gemm.h`. W4 reuses this.
- CPU dequant-to-bf16 utilities we mirror in structure: `nvfp4_dequant.cpp`,
  `gguf_dequant.cpp` (a pure dequant primitive, unit-gated in isolation before
  a loader consumes it). W1 follows that pattern exactly.

## Port map

| vLLM source | Our target | Brick |
|---|---|---|
| `awq_triton.py:11-105` + `qdq_4.cuh` + `q_gemm.cu:201-202` | `src/vllm/model_executor/model_loader/awq_gptq_dequant.cpp` + `include/.../awq_gptq_dequant.h` (INT4 unpack+dequant to bf16, both formats, `g_idx`, `zero_offset`) | **W1 (this claim)** ✅ |
| `tests/kernels/quantization/test_awq_triton.py` + `qdq_4.cuh`/`q_gemm.cu` | `tests/vllm/test_awq_gptq_dequant.cpp` | **W1** ✅ |
| `auto_awq.py:237-257` / `auto_gptq.py` config + `quant_utils.py` | AWQ/GPTQ config recognizer in the safetensors loader (`R`) | W2 |
| `_convert_awq_to_standard_format` + weight-loader wiring | resident loader → bf16 expand (`M`,`E` CPU) on a dense model | W3 |
| `awq_marlin_repack.cu`, `gptq_marlin_repack.cu`, `marlin.cu` | extend `marlin_repack.h` + `src/vt/cuda/marlin/` for int4/int8 `uint4b8`/`uint8b128` (`C` GPU) — **kernel-matrix row** | W4 |
| `qdq_8.cuh`, `qdq_2/3.cuh` | GPTQ 8/2/3-bit dequant breadth | W5 |
| `AutoAWQMoEMethod`, `int_wna16` MoE | Marlin MoE for AWQ/GPTQ experts (`C` MoE) | W6 |

## GB10 / sm_121 kernel selection

`AutoAWQConfig.get_quant_method` (`auto_awq.py:285-360`) / GPTQ analog build an
`MPLinearLayerConfig` and call `choose_mp_linear_kernel`
(`vllm/model_executor/kernels/linear/__init__.py:689`). The CUDA priority list
(`_POSSIBLE_KERNELS[CUDA]`, :396-406) is `CutlassW4A8, Machete, AllSpark,
Marlin, Conch, Exllama, TritonW4A16, Humming`. For W4A16 int4/int8 group-wise at
AWQ/GPTQ shapes, **Marlin** is the first to report support on sm_121 — the same
resolution we OBSERVED for NVFP4-W4A16 (`Using MarlinNvFp4LinearKernel`). CPU
list (:418-422) `Dynamic4bit, Zentorch, CPUWNA16` — our W1 dequant is the
materialization those consume. **GPU parity target = Marlin repack + GEMM; CPU
target = dequant-to-bf16 (W1) feeding the existing bf16 GEMM.**

## Tests to port

- `tests/kernels/quantization/test_awq_triton.py` (`awq_dequantize_torch`
  reference) → **PORTED (W1)** in `tests/vllm/test_awq_gptq_dequant.cpp`: the
  reverse-AWQ order + `(w-z)*s` reference re-expressed as hand-computed known
  int32 cases (independent arithmetic oracle) plus a randomized double-precision
  layout roundtrip.
- GPTQ `qdq_4.cuh` dequant + `q_gemm.cu` zero_offset → GPTQ hand-computed
  `zero_offset` v1/v2 case + act-order `g_idx` case + randomized roundtrip (W1).
- `tests/quantization/test_auto_awq.py`, `test_auto_gptq.py`, `test_gptq_v2.py`,
  `test_gptq_dynamic.py` — model-load + generate e2e; need a real checkpoint +
  loader/Marlin. Named here; checked in SKIPPED with reason at W3/W4.
- `test_marlin_gemm` / `awq_marlin_repack` correctness → W4 with the kernel.

## Gates

- **W1 (this claim):** unit gate — hand-computed exact bf16 over known packed
  int32 (RED-first) + randomized double-precision roundtrip, both formats,
  `g_idx` and `zero_offset` v1/v2 covered, argument-validation aborts. CPU
  `-Werror` clean. No model, no GPU. (BENCHMARKS: NOT-APPLICABLE — no throughput
  owed by a CPU dequant primitive.)
- W3 CPU `E`: a real small AWQ or GPTQ dense model, greedy token-exact vs the
  pinned vLLM oracle (near-tie-robust where vLLM's own greedy is non-deterministic).
- W4 GPU `C`/`P`: Marlin repack+GEMM bit-exact vs `awq_dequantize`→bf16 matmul;
  throughput vs vLLM on the same workload (benchmark protocol, both gate axes).

## Dependencies

- W4 reuses the vendored Marlin (`src/vt/cuda/marlin/`); the int4/`uint4b8` GEMM
  variant must be added without disturbing the NVFP4 path (own only the new
  template instantiations; coordinate before touching shared `marlin_template.h`).
- W2/W3 depend on the safetensors loader recognizing the AWQ/GPTQ config and the
  fp16/bf16 scale decode (W1 takes decoded f32 scales, keeping the primitive
  pure). W1 has NO dependency — it is self-contained.

## Work breakdown

- **W0** — this spec (records-only). ✅ this commit.
- **W1** — CPU INT4 unpack+dequant to bf16 (AWQ + GPTQ), unit-gated. ✅ this commit.
- W2 — config recognizer + safetensors probe (`R`).
- W3 — resident loader → bf16 expand + CPU dense `E` gate.
- W4 — Marlin repack + GEMM GPU `C` (kernel-matrix row; bump the KERNEL constant
  with dated rationale) + `P`.
- W5 — GPTQ 8/2/3-bit dequant breadth.
- W6 — AWQ/GPTQ MoE (Marlin MoE experts).

## Risks/decisions

- The GPTQ **zero_offset** (v1 `+1` vs v2 `0`) is a silent-correctness landmine:
  W1 makes it an explicit parameter and gates BOTH.
- AWQ's reverse bit order + N-axis packing vs GPTQ's standard order + K-axis
  packing is the other landmine — W1 gates both layouts against an INDEPENDENT
  reference packer; `col7=15` on the AWQ case forces a set MSB so a
  sign-extension unpack bug surfaces (payloads read as `uint32`).
- Scales are fp16/bf16 on disk; W1 takes decoded f32 scales so the brick stays a
  pure unpack+dequant kernel (the fp16 decode belongs to the loader at W2/W3).
- Do NOT declare parity from source alone: W4 must nsys the Marlin path on GB10
  and mirror the actual resolved kernel (per the whole-chain directive).
