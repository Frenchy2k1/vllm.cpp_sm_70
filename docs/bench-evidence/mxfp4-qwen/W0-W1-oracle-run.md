# QUANT-CT-MXFP4 — W0 + W1 empirical (oracle run on GB10)

Date 2026-08-05. Box dgx.casa (GB10, sm_121, cap 12.1). Oracle
`~/venvs/vllm-oracle` -> `vllm-oracle-v0.25.0-stage` (vLLM **0.25.0**,
compressed_tensors 0.17.0). Checkpoint `Yi30/Qwen3-8B-MXFP4` (dense
`Qwen3ForCausalLM`, compressed-tensors `mxfp4-pack-quantized`, group 32, true
W4A4 checkpoint, 5.79 GiB loaded). Run: offline vLLM `LLM` API, greedy
(`temperature=0, seed=0`), `enforce_eager=True`, `gpu_memory_utilization=0.30`,
`max_model_len=2048`, 4 prompts x 48 tokens. Both GPU flock locks held, mem gate
`free -g >= 90` (94-95 GiB free), single load, tmux + done-marker.

## W1 — the kernel that actually runs (parity target), with a runtime correction

**Default config CRASHES on sm_121.** `init_mxfp4_linear_kernel` selects the
FIRST supported kernel = `FlashInferMxFp4LinearKernel` (its `is_supported` =
`has_device_capability(100) AND has_flashinfer_cutedsl()`, both True on GB10):

```
INFO [__init__.py:835] Using FlashInferMxFp4LinearKernel for MXFP4 GEMM
```

but at engine start (KV-cache profiling forward) FlashInfer's runtime capability
check REJECTS the cute-dsl mxf4 backend for sm_121:

```
flashinfer.utils.BackendSupportedError: mm_fp4 does not support backend 'cute-dsl' with capability 121
RuntimeError: Engine core initialization failed.
```

So `is_supported` is optimistic (cap >= 100) but the flashinfer `mm_fp4` cute-dsl
backend only covers datacenter Blackwell (sm_100), NOT consumer GB10 (sm_121).
**The default oracle path is non-functional for this checkpoint on GB10.**

**Working path = Marlin W4A16** via the recorded lever
`VLLM_DISABLED_KERNELS=FlashInferMxFp4LinearKernel`, which drops FlashInfer from
the candidate list so `init_mxfp4_linear_kernel` falls to the next supported:

```
INFO [__init__.py:835] Using MarlinMxFp4LinearKernel for MXFP4 GEMM
```

`MarlinMxFp4LinearKernel.apply_weights` -> `apply_fp4_marlin_linear(...,
weight_global_scale=None)` — **W4A16 weight-only fp4 Marlin GEMM** (the E8M0 group
scale folded, no global). This is the real GB10 parity target and it
**revalidates the row's original Laguna-B2 "Marlin W4A16" hypothesis**; the
source-only trace that concluded "FlashInfer W4A4" was wrong for sm_121 because it
did not account for flashinfer's runtime backend gate. Lesson: trace the
execution, not just the dispatch source.

## W0 — greedy golden (PROOF the oracle RUNS the checkpoint)

`PYEXIT=0`. Coherent, correct greedy output on all 4 prompts (full token ids in
`golden_marlin_w4a16.json`):

- "The capital of France is" -> " Paris. ... Italy is Rome. ... Germany is Berlin. ... Spain is Madrid. ..."
- "Q: What is 2 + 2? A:" -> " 4. Q: What is 3 + 3? A: 6. ... 5 + 5? A: 10"
- "Once upon a time, in a small village," -> coherent story (wise old man, magical tree)
- "def fibonacci(n):" -> correct recursive implementation

This satisfies the hard W0 oracle rule: the pinned oracle BUILDS+RUNS a greedy
golden on a real Qwen MXFP4 checkpoint, not merely constructs the config.

## Consequences for W2-W4

- Parity target on GB10 = **Marlin W4A16 mxf4** (weight-only fp4, bf16 activation,
  group 32, E8M0 scale, no global). Our engine already has Marlin FP4 infra
  (`src/vt/cuda/marlin/...`, NVFP4/AWQ/GPTQ) — W2 extends the FP4 Marlin format
  plumbing for group-32 E8M0 (the Laguna B2 route), NOT a new cute-dsl W4A4 kernel.
- The golden here is the W3 e2e gate reference. vLLM greedy on a dense 8B may be
  bf16-non-deterministic -> distributional gate if strict token-exact does not hold.
- A W4A4 arm is only reachable if/when flashinfer ships an sm_121 mxf4 backend
  (or we implement the cutlass mxf4 mma directly); it is not the GB10 bar today.
- The `VLLM_DISABLED_KERNELS=FlashInferMxFp4LinearKernel` env is REQUIRED for the
  oracle to run this checkpoint on GB10 — record it in the W4 bench recipe.
