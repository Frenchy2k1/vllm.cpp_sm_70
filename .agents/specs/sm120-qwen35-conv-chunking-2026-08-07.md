# sm_120 Qwen3.5 batched prefill-conv chunking — structured spike

**Rows:** `KERNEL-SSM-MAMBA`, feeding `ROAD-V1-C2-LOCAL-BF16`.
**Hardware/workload:** RTX 5070 Ti (`sm_120`), Qwen3.5-4B plain BF16,
128 ShareGPT requests, 128 output tokens, concurrency 32,
`max_num_batched_tokens=2048`, greedy.  **Lifecycle:** implemented, locally
gated and default ON; 27B/35B release gates remain hardware-unavailable.

## Measured selection

The corrected production-frontend comparison at `3f35356e0` completed three
memory and three timed repetitions per direct-ON, pinned-vLLM and direct-OFF
arm under one `/tmp/gpu` lock.  Direct ON and OFF are loader controls; both
reported `AsyncLLM`, async scheduling enabled and maximum concurrent batches 2.

Direct ON versus pinned vLLM means are:

| Axis | ours | vLLM | ours / vLLM |
|---|---:|---:|---:|
| total throughput | 6633.750 tok/s | 6643.593 tok/s | 0.998518x |
| output throughput | 733.540 tok/s | 734.630 tok/s | 0.998516x |
| mean TTFT | 1051.333 ms | 937.584 ms | 1.121322x |
| mean TPOT / ITL | 35.457 ms | 33.906 ms | 1.045721x |
| peak VRAM | 13054 MiB | 12820 MiB | 1.018253x |

The same-tool graph-node traces isolate request processing to 22.370 seconds
locally and 22.269 seconds in vLLM.  Both are saturated (98.54% and 99.34% GPU
busy by interval union), so a host-only polling change is not the first lever.
Sampling covers essentially identical work: 16,442 local versus 16,443 vLLM
rows.  The largest clear non-GEMM kernel-family gap is batched GDN prefill
`causal_conv1d`:

| Trace metric | ours `CausalConv1dFwdRegKernel` | vLLM `_causal_conv1d_fwd_kernel` |
|---|---:|---:|
| launches | 1,728 | 1,893 |
| total GPU time | **720.954 ms** | **145.421 ms** |
| mean launch | 417.219 us | 76.821 us |

The selected primary micro-metric is therefore **steady-interval total GPU time
for batched GDN prefill `causal_conv1d` on the exact c32 workload**.  Baseline
gap: **4.958x**.  This metric is preferred over isolated TPOT because the old
synchronous frontend demonstrated that queueing can improve TTFT while making
TPOT worse without improving device work.

## Whole-chain cause

Pinned vLLM's Triton kernel takes `batch_ptr` and `token_chunk_offset_ptr` and
maps every program to exactly one `(sequence, BLOCK_M token chunk)` before
processing the feature tile
(`${VLLM_SOURCE}/vllm/model_executor/layers/mamba/ops/causal_conv1d.py:15-28,71-79,123-124`).
The scheduler/backend metadata constructs those exact chunk descriptors.

Our register kernel has the same register-window arithmetic, but
`kConvRegChunkMaxSeqs=4`; `LaunchConvFwdReg` sets `gridZ=1` whenever the batch
contains more than four sequences, so each channel block serially walks an
entire request (`src/vt/cuda/cuda_gdn.cu:702-705,720-766,793-819`).  The trace
shows that production c32 path directly: the dominant local shapes are
`grid=(64,28..32,1)` at roughly 410-466 us, while vLLM uses exact token-chunk
programs with feature-grid 32 at roughly 82-95 us for the large waves.

The missing data is already an explicit recorded deviation:
`GDNAttentionMetadata` omits vLLM's `batch_ptr` and
`token_chunk_offset_ptr` because the original sequential C++ kernel had no
consumer (`include/vllm/v1/attention/backends/gdn_attn.h:33-43`).  That
assumption is now refuted for performance on the production batched path.

## Port, not reinvention

Implement vLLM's exact chunk list through the existing GDN metadata and device
input path, then make `CausalConv1dFwdRegKernel` consume one descriptor per
program.  Do not launch a rectangular `num_sequences * ceil(total_tokens/M)`
grid: it over-launches by the batch size and is not the upstream algorithm.
Preserve the current tap-order float accumulation, state read/write semantics,
BF16 I/O, and `VT_CONV_REG=0` rollback.  Tune `BLOCK_M`/feature tile only after
the exact upstream work partition is measured; the first discriminator is the
metadata/dispatch port, not an arbitrary sm_120-only kernel.

No GEMM claim is made from this trace.  The GEMM templates differ between the
engines, so any later GEMM lever must separately satisfy the four-axis
invocation-parity gate (C dtype, compute/scale type, entry point/algo policy and
resolved same-tool template).

## Tests and acceptance

RED-first tests must cover:

1. host metadata for unequal sequence lengths produces every `(sequence,
   token-chunk)` exactly once, no missing or padded work;
2. CUDA chunked versus current register path is byte-identical for output and
   final convolution state over BF16/F32 inputs, initial/fresh state, unequal
   lengths, `T < K-1`, and batches above four sequences;
3. a mutant restoring `gridZ=1` above four sequences fails the structural
   launch/metadata test;
4. cached Qwen3.5-4B correctness remains 3/3 cases and 1672/1672 assertions;
   direct ON/OFF remains 128/128 identical per repetition.

Performance is measured first as a same-binary, same-workload graph-node trace
under the 25 GiB user-systemd cap and one `${GPU_LOCK}`.  The micro-metric must
improve outside run noise and move toward vLLM's 145.421 ms; a default flip also
requires the project's kernel-efficiency bar and byte-exactness.  Then repeat
the full 18-leg comparison.  No end-to-end axis may regress: total/output
throughput must reach or exceed vLLM, all latency axes must move toward or beat
vLLM, and peak VRAM must not exceed the 13054 MiB baseline or vLLM floor.  The
27B and 35B gate-model correctness suites remain required before a shared
default change; this 4B run cannot extrapolate support to them.

## Evidence and rollback

- benchmark aggregate: `/tmp/qwen35-async-3f35356e0/aggregate.json`, SHA-256
  `d006d6ffd6d014fc1861a30c126f603117a1ff80b339172d63fab75f2ae07f1d`
- local trace: `/tmp/qwen35-async-3f35356e0-ours.nsys-rep`, SHA-256
  `182426e961ccaa53896e7cb7e9c4d2ba1cc35ba695ea3a9c3c727195326c8dab`
- vLLM trace: `/tmp/qwen35-async-3f35356e0-vllm.nsys-rep`, SHA-256
  `9d543390741629fc3873ad857706595cf880ee2fcce63af3e4be17e1a343355a`
- full record: [Qwen3.5-4B sm_120 evidence](../../docs/bench-evidence/qwen35-4b-sm120-main-20260807.md)

Rollback keeps the existing `VT_CONV_REG=0` tiled/scalar path and adds a
same-binary switch for the new exact-chunk dispatch until the full gate closes.

## Implementation and measured disposition

The implementation follows the spike exactly: `ComputeCausalConv1dMetadata`
enumerates every `ceil(sequence_length / 8)` work item, the step metadata owns
and uploads the two i32 descriptor arrays once, and every GDN layer reuses them.
The register kernel consumes one descriptor per `grid.y` program. The
same-binary control is `VT_CONV_EXACT_CHUNKS`; it defaults ON and `=0` selects
the prior whole-sequence mapping. `VT_CONV_REG=0` remains the independent
tiled/scalar rollback.

RED-first metadata coverage, affected model fixtures, full CUDA GDN tests and
cached Qwen3.5-4B correctness are green; default and rollback output are byte
identical in three production pairs. Same-binary `nsys` reduces causal-conv
GPU time from 720.047 to 234.607 ms (**3.069x**), leaving a **1.613x** gap to
the sealed pinned-vLLM 145.421 ms trace. Three alternating enclosing pairs
improve total/output throughput **2.152%**, TTFT **2.945%**, TPOT/ITL **1.920%**
and E2E latency **2.118%**, with no VRAM regression. The local default now
measures **1.021246x** the sealed vLLM throughput; TTFT (**1.085812x**) and
TPOT/ITL (**1.024597x**) remain slower, and VRAM remains above the vLLM floor.

The attempted fresh 18-leg oracle reruns are VOID because the oracle's
FlashInfer/Torch/Triton JIT environment became unstable after 13/18 legs. The
accepted evidence is the same-binary trace plus three-pair local A/B, compared
to the sealed same-hardware/same-workload vLLM denominator. See the linked full
record for manifests and exact artifacts. No 4B result is extrapolated to the
unavailable release-gate models.

The clean transplant onto `upstream/main` `f91a5917a` is revalidated: focused
CPU 6/6, CUDA GDN 66/66·4300, cached 4B 3/3·1672, and byte-identical same-binary
token files. Its fresh graph-node profile reproduces causal-conv
720.216507→234.379395 ms (**3.072866x**) and the profiled enclosing run
6587.66→6727.35 tok/s (**+2.1205%**). The old-branch gain therefore survives
the clean transplant; the sealed vLLM conv residual is **1.611730x**.
