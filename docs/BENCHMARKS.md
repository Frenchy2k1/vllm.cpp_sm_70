# Benchmarks

Measured results for vllm.cpp, against the reference engine each workload
actually competes with. Every number here was produced on real hardware, greedy,
with the reference in its own production configuration. Ties are called ties,
and where we are behind the row says so.

This page is a **scoreboard**: one row per subject, kept current. It is not a
changelog. The full attempt record, including refuted hypotheses, profiler
traces, and superseded numbers, lives in
[.agents/benchmark-record.md](../.agents/benchmark-record.md). For what the
project is and how to run it, see the [README](../README.md); for per-capability
lifecycle state, see [docs/STATUS.md](STATUS.md); for what is supported at all,
see [docs/FEATURES.md](FEATURES.md).

## At a glance

| Reference | Workload | Headline | Tokens |
|---|---|---|---|
| **vLLM** | Qwen3.6-27B NVFP4, GB10 | ahead 4.5% at c1, **tie** at c2 to c32 | identical |
| **vLLM** | Qwen3.6-35B-A3B NVFP4, GB10 | ahead at c16/c32, behind at c1 to c8 | identical |
| **vLLM** | DeepSeek-V2-Lite (MLA), GB10 | 0.86x to 0.95x throughput, TTFT wins at c4/c8 | identical |
| **vLLM** | Laguna-XS-2.1 NVFP4, GB10 | 87% of vLLM (37.6 vs 43.1 tok/s), gap localized to the bf16 GEMV invocation | near-tie |
| **llama.cpp** | Qwen3.5-2B GGUF, CPU aarch64 | prefill **1.18x ahead**, decode tie, memory parity | byte-identical |
| **MLX-LM** | Qwen3-0.6B, Apple M4 | 97.6% warm total, prefill ahead | near-tie |
| **DwarfStar** | DeepSeek-V4-Flash GGUF, GB10 | **parity**, 0.997x (16.28 vs 16.33 tok/s) | n/a, GGUF peer |

Reading the ratios: throughput is ours/reference, latency is reference/ours, so
**1.0 or higher is a win** everywhere on this page.

## vLLM, online serving

The binding comparison. vLLM runs its **production graphed config**, never
`--enforce-eager`, because the graphed config is the honest denominator.

| Model | Quant | vLLM pin | Axes passing | Disposition |
|---|---|---|---:|---|
| Qwen3.6-27B | NVFP4 | 0.25.0 | **115/124** | Effective parity-or-better, two-grid totality |
| Qwen3.6-35B-A3B | NVFP4 `modelopt_mixed` | 0.25.0 | 70/124 | c4-c32 win, c1/c2 residual 2-4% |
| DeepSeek-V2-Lite | bf16 MLA | 0.25.0 | 4/25 | Attributed miss, row stays `ACTIVE` |
| Qwen3.5-4B | bf16 direct-load | 0.24.0 | 5/8 | Throughput 0.98x, TTFT and memory win |

### Qwen3.6-27B by concurrency

Medians of three interleaved repetitions, 1,024 in / 128 out, cache off, closed
loop. Output is token-for-token identical to vLLM at every point.

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| **vllm.cpp** tok/s | **86.05** | 159.68 | 292.34 | 508.77 | 801.76 | 1095.01 |
| vLLM tok/s | 82.32 | 158.03 | 290.31 | 505.46 | 789.16 | 1076.25 |
| **Ratio** | **1.045x** | 1.011x | 1.007x | 1.007x | 1.016x | 1.017x |
| Axes passing | 20/20 | 20/20 | 18/20 | 15/20 | 19/20 | 18/20 |

We are nominally ahead at all six, but only c1 means anything. Our run-to-run
noise band is 0.5% and c2 through c32 land between 0.7% and 1.7%, so **treat
those five as ties**, not as wins. The nine axes that fail in both grids are one
tradeoff, not nine problems: our synchronous deterministic forward loses on
low-concurrency *median* decode and TTFT, and wins the corresponding *tail* and
the same metric at higher concurrency (c8 p99 ITL 0.86x, but 1.055x at c16 and
1.078x at c32).

### Qwen3.6-35B-A3B by concurrency

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| **vllm.cpp** tok/s | 491.5 | 767.7 | 1236.4 | 1855.1 | **2489.1** | **3030.5** |
| vLLM tok/s | 601.9 | 904.7 | 1366.8 | 1923.2 | 2464.9 | 2993.0 |
| **Ratio** | 0.817x | 0.849x | 0.905x | 0.965x | **1.010x** | **1.013x** |
| Mean TPOT | 0.813x | 0.852x | 0.929x | **1.022x** | **1.103x** | **1.097x** |

All four memory axes beat vLLM. The open gaps are low-batch MoE decode (the
Marlin grouped GEMM is inefficient at batch 1 and scales to winning by c16) and
prefill TTFT at 0.79x to 0.86x across the board.

### DeepSeek-V2-Lite (MLA)

Medians of 3 reps, 1,024 in / 128 out. The vLLM arm runs `--moe-backend triton`,
which is its best **stable** graphed config on GB10: the auto-selected FlashInfer
CUTLASS MoE backend hard-rebooted the box five times. The substitution does not
flatter us, we lose against it.

| Concurrency | Output tok/s ours / vLLM | Ratio | Median TTFT | Median TPOT |
|---:|---|---:|---:|---:|
| 1 | 33.18 / 38.17 | 0.87x | 0.95x | 0.90x |
| 2 | 52.63 / 55.27 | 0.95x | 0.88x | **1.03x** |
| 4 | 70.36 / 81.51 | 0.86x | **1.04x** | 0.86x |
| 8 | 102.37 / 116.35 | 0.88x | **1.12x** | 0.86x |

Peak memory is the decisive win: **31.38 GiB against vLLM's 68.5 GiB**, with the
caveat that vLLM pre-reserves a fixed fraction up front while we allocate the KV
blocks the workload needs. Real difference in operating footprint, not evidence
of a lower per-token KV cost.

### Laguna-XS-2.1 (NVFP4)

Both arms NVFP4, single request, batch 1, GB10.

| Arm | Decode tok/s | Ratio |
|---|---:|---:|
| vLLM NVFP4, graphed | 43.10 | 1.00x |
| **vllm.cpp NVFP4**, resident decode + CUDA graph | **37.55** | **0.87x** |

The gap is localized, same-tool (nsys graph-node tracing on BOTH engines,
2026-08-04): the entire +3.1 ms/step is the bf16 M=1 projection GEMV bucket,
two thirds of it o_proj (ours ~196-204 us vs vLLM 139 us per call, the
identical cuBLAS `gemvx` kernel). Attention is tied (537 vs 527 us), MoE is
within 0.3 ms, and we win the norm/cast glue by 0.46 ms.

Ruled out by gated same-tool A/Bs: the invocation itself (matching vLLM's
bf16-output `cublasGemmEx` exactly leaves o_proj at ~196 us, a GPU wash and a
host regression), stream contention, clocks, allocator pools, and the host
gap. The open question is why the identical instantiation runs faster per call
for vLLM; a per-shape byte audit (do our GEMV shapes match vLLM's exactly?) is
the next probe. The earlier "overlap window" explanation is superseded.

## Memory

Qwen3.6-27B NVFP4, GB10, whole serving window.

| Axis | vllm.cpp | vLLM | Ratio | Result |
|---|---:|---:|---:|---|
| Peak PSS | 24.88 GiB | 28.18 GiB | 1.133x | **PASS** |
| Peak RSS | 24.88 GiB | 28.56 GiB | 1.148x | **PASS** |
| Peak GPU memory | 40,996 MiB | 70,531 MiB | 1.720x | **PASS** |
| Peak `MemAvailable` drop | 68.35 GiB | 80.66 GiB | 1.180x | **PASS** |

35B steady-serving PSS is 3.53 GiB against vLLM's 13.3 GiB after the routed-expert
host mirror is freed once the device Marlin resident is built.

## llama.cpp, CPU

Same GGUF file both arms, `dgx.casa` GB10 aarch64 (20 cores), idle, 3 reps,
llama.cpp `237ad9b96` built fresh on the same host.

| Axis | vllm.cpp | llama.cpp | Ratio | Result |
|---|---:|---:|---:|---|
| Prefill | **223.8 tok/s** | 177.3 | **1.18x** | **PASS** |
| Decode | 24.7 tok/s | 25.4 | 0.97x | tie |
| Peak memory | 2.83 GiB | 2.80 GiB | 1.01x | **PARITY** |

Decode lands inside llama.cpp's own run-to-run spread, and the memory difference
is 30 MiB on a 2.8 GiB working set. Prefill is the only axis with a real gap and
it goes our way. Output tokens are **byte-identical** to llama.cpp's greedy
decode and to our own CPU reference path. Single-stream only: we have not
measured concurrent serving against llama.cpp's server.

## MLX-LM, Apple M4

Qwen3-0.6B, warm, batch 1, 6 interleaved runs.

| Axis | vllm.cpp | MLX-LM | Ratio |
|---|---:|---:|---:|
| Prefill TTFT | **524.5 ms** | 532.6 ms | **1.015x** |
| Decode | 27.23 tok/s | 27.85 | 0.978x |
| Warm total | 24.37 tok/s | 24.96 | 0.976x |

The 2.4% is a real gap, not noise: our spread was 0.12% and MLX-LM's 0.34%. All
of it sits in decode, 0.81 ms per token. Indicative rather than binding: two
models, 18 of 75 ops native, and the 97.6% needs the optional MLX GEMM provider
shape-gated to prefill (95.9% on the default build).

## DwarfStar, GGUF

DeepSeek-V4-Flash cannot run on vLLM on a single GB10 at all: every
vLLM-loadable checkpoint is 156 GB or larger against a 119 GiB unified pool, so
the only quant that fits is extreme-low-bit GGUF, which vLLM cannot load here.
GGUF was forced by the hardware. A policy-correct vLLM comparison needs 2x GB10
Sparks with TP2 and is owed.

| Engine | Quant | Decode tok/s | Ratio |
|---|---|---:|---:|
| DwarfStar (`ds4`) | IQ2_XXS mixed | 16.33 | 1.00x |
| **vllm.cpp** | same GGUF | **16.28** | **0.997x, parity** |

Parity, measured same-session clean (2026-08-04, single-load steady both arms);
the earlier 15.87/96% and 17.13 figures are superseded. The structure behind
the tie is real and asymmetric: an nsys trace of both engines shows ds4 does
about 1.8x less GPU work per token but is host-launch-bound at 57% GPU-busy
(it graphs only prefill), while we do more GPU work at 97% busy because our
decode runs as one captured CUDA graph. The old "Q8_0 weight-stream floor"
framing is corrected: our int8 GEMV is at per-launch parity with ds4's; ds4's
lighter step comes from routing its small DSA/router/output tensors through f16
tensor cores, an optional beat-path for us (near-tie class), not a deficit.

## Speculative decoding

| Speculator | Model | Result | Status |
|---|---|---|---|
| MTP | Qwen3.6-27B NVFP4 | token-identical to vLLM MTP, **~4% faster at c1**; on-par at c2-c8 | `DONE` |
| DFlash | Qwen3.6-27B NVFP4 | ~2x over spec-off, below vLLM throughput | bf16 acceptance floor open |

## How we measure

**Hardware.** NVIDIA GB10 / DGX Spark (sm_121a) for CUDA, `dgx.casa` aarch64 for
CPU, Apple M4 for Metal. GB10's 119 GiB pool is unified, so host and device
memory compete; end-to-end wall-clock on a cold page cache is unusable there,
and steady-state per-step timing or `nsys` GPU-busy is the anchor.

**Oracle pin.** vLLM 0.26.0.dev0 (`55596792`) plus transformers 5.14.1, built from
source for sm_121a. Speed figures labelled 0.25.0 are the last binding run; the
engine is unchanged by the pin advance and a 0.26 re-benchmark is pending.
Correctness re-validated bit-identical across the advance, zero golden drift.

**Protocol.** Greedy, closed loop, three interleaved repetitions per point, one
`flock` across the whole series, same-binary A/B for every lever, cold legs
discarded. Workload equivalence between arms is audited, not assumed: batch cap,
token budget, context, corpus bytes, KV and SSM dtypes, kernel family, and
graphed decode all match, and the audit is
[recorded](../.agents/specs/benchmark-equivalence-audit-2026-07-15.md). The
2026-08-04 agent-record substrate, triage, compaction, CI-concurrency and
anchor-backfill work, and the operator/helper protocol W0+W1, touched no engine code and moved no number: NOT
APPLICABLE, nothing to reproduce.

**Vocabulary.** *Token-exact* means our output ids equal the reference's, byte
for byte. *Near-tie* means the reference's own greedy decode is not deterministic
at this precision, so the gate is distributional: our output must fall inside the
set the reference produces across K runs. *Tie* means the difference is inside
the measured run-to-run noise band, which is 0.5% on GB10 and 0.12% to 0.34% on
M4. We never publish a partial, contended, or stale-denominator number as
binding, and when a denominator turns out to be wrong we correct every ratio
built on it rather than keeping the flattering one.

## Open gaps

| Track | Status | Next gate |
|---|---|---|
| 35B prefill TTFT | 0.79x to 0.86x at every concurrency | Portable fusion of the norm/quant/act/combine glue |
| 35B low-batch MoE decode | c1 TPOT 0.73x, wins by c16 | Attribute and close the batch-1 grouped GEMM |
| DeepSeek-V2-Lite MLA | Attributed miss, `ACTIVE` | Throughput at every concurrency |
| Laguna-XS NVFP4 | 87% of vLLM, gap localized same-tool to the bf16 GEMV invocation (f32-out vs vLLM's bf16-out `gemvx`) | Gated bf16-output invocation A/B, o_proj first |
| DeepSeek-V4-Flash | **Parity with ds4 (0.997x)** | Optional beat-path: f16 tensor-core DSA/router (near-tie class) |
| DeepSeek-V4-Flash vs vLLM | Infeasible on one Spark | 2x GB10 with TP2 over the NCCL seam |
| DFlash speculative decode | Below vLLM throughput | bf16 acceptance floor ~0.85x |
| Multimodal image, audio, video | Correctness gated, speed unmeasured | Per-modality speed grids |
| Qwen3-dense decode CUDA-graph | Token-exact pass, ~4.3% e2e directional | Steady-state per-step tok/s |
| vLLM 0.26 re-benchmark | Pending | Re-run the binding grids on the advanced pin |
| SGLang floor arms | Never ran | Both arms of the SGLang comparison |
| cuBLAS invocation-parity guard | CI guard landed (CPU); `kGemvHeuristicAlgos` refactor build-verify owed | `nvcc` rebuild + SACRED gate on dgx |

## Reproduce

| Benchmark | Entry point |
|---|---|
| vLLM online grid | `.agents/specs/competitive-benchmarks.md`, evidence under `dgx:~/work/vllm.cpp-online-gate/evidence/` |
| CPU vs llama.cpp | Same GGUF both arms, 3 reps under one `flock $HOME/gpu.lock`; `VT_GGUF_KEEP_F16=0` reproduces the pre-L7 baseline |
| Laguna NVFP4 decode | `flock $HOME/gpu.lock ./build-cuda/examples/laguna-gen --model ~/laguna-xs-nvfp4 --gpu`; `drop_caches` first, create the CUDA context before loading weights |
| DeepSeek-V4-Flash decode | `deepseek-v4-gen --gpu --kv-cache` on `ds4flash.gguf`, captured under tmux |
| Metal vs MLX-LM | Paired A/B harness, interleaved runs, cold legs discarded |

Build flags, environment variables, and the full gate list are in
[BUILD.md](BUILD.md) and [ENVIRONMENT.md](ENVIRONMENT.md).
