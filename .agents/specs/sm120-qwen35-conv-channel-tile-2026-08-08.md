# sm_120 Qwen3.5 causal-conv residual — structured spike

**Rows:** `KERNEL-SSM-MAMBA`, feeding `ROAD-V1-C2-LOCAL-BF16`.
**Hardware/workload:** RTX 5070 Ti (`sm_120`), Qwen3.5-4B plain BF16,
128 ShareGPT requests, 128 output tokens, concurrency 32,
`max_num_batched_tokens=2048`, 1,280 KV blocks, greedy. **Lifecycle:** spike;
implementation and release-model gates are pending.

## Measured selection

A fresh graph-node `nsys` profile at `7f66792f8`, after the exact-chunk and
opt-in post-conv-tile changes, reports:

| Family | ours | pinned vLLM | ours / vLLM | local excess |
|---|---:|---:|---:|---:|
| prefill causal conv | **234.255 ms** / 1,728 calls | **145.532 ms** / 1,897 calls | **1.6096x** | **88.723 ms** |
| fused post-conv | 122.511 ms / 1,728 calls | 108.035 ms / 1,923 calls | 1.1340x | 14.476 ms |

The next metric is therefore **total GPU time of
`CausalConv1dFwdRegKernel` on the exact c32 production workload**, with the
dominant launch shape used as its first micro-metric. Shape-grouping the same
tool's kernel rows divides that total further:

| Exact programs | Calls | Total | Mean | Share of local conv |
|---:|---:|---:|---:|---:|
| 279 | 576 | 86.052 ms | 149.395 us | 36.7% |
| 280 | 336 | 50.137 ms | 149.218 us | 21.4% |
| 156-158 | 216 | 17.647 ms | 81.699 us | 7.5% |
| all remaining shapes | 600 | 80.419 ms | mixed | 34.3% |

The 279-280-program waves alone are 912 launches and **136.189 ms (58.1%)**,
so their mean launch time is the selected micro-metric. This avoids guessing
from a whole-run aggregate when a launch-shape-local change is available.

Artifacts:

- local report `/tmp/qwen35-next-7f66792f.nsys-rep`, SQLite export
  `/tmp/qwen35-next-7f66792f.sqlite`;
- pinned-vLLM report `/tmp/qwen35-async-3f35356e0-vllm.nsys-rep`, SQLite
  export `/tmp/qwen35-async-3f35356e0-vllm.sqlite`.

Nsight Compute is not present in the selected Nix CUDA environment or the host,
so no hardware-counter claim is made. Installing a profiler package is outside
this campaign's permission envelope. Register counts and launch geometry below
come from the same `nsys` CUPTI rows, not an inferred occupancy claim.

## Whole-chain difference and hypotheses

Pinned vLLM launches Triton's `_causal_conv1d_fwd_kernel` with
`BLOCK_M=8`, `BLOCK_N=256`, four warps and two stages
(`${VLLM_SOURCE}/vllm/model_executor/layers/mamba/ops/causal_conv1d.py:16-63,78-79,692-742`).
On the dominant wave that resolves to `grid=(279,32,1)`, block 128, and 32
registers/thread. Its 279-program launches average 88.346 us.

The local exact-descriptor port already matches `BLOCK_M=8` and enumerates the
same sequence/chunk programs, but retains a runtime convolution width and one
channel per thread: `kConvRegN=128`, `grid=(64,279,1)`, block 128, and 43
registers/thread (`src/vt/cuda/cuda_gdn.cu:702-854`). Its 279-program launches
average 149.395 us. Qwen3.5-4B has 8,192 convolution channels and width four,
so 64 versus 32 feature blocks is exactly the `BLOCK_N` difference.

Two independent hypotheses remain and must be measured separately:

1. **Width specialization.** vLLM's `KERNEL_WIDTH` is a compile-time constant;
   local `k` is runtime and reserves arrays through `kConvRegMaxW+1=9` while
   guarding every unrolled tap. Dispatching the production `k=4` instantiation
   can remove dead taps/branches and reduce register or instruction cost without
   changing the grid.
2. **Two channels per thread.** A 128-thread block can process two coalesced
   128-channel stripes, giving a 256-channel feature tile and 32 feature blocks
   like upstream. It halves block-level descriptor/control work, but duplicates
   each thread's channel-local weights and window. Register pressure can refute
   this even when the grid looks better.

These are ordered discriminators, not one combined patch: first compare the
width-four specialization against the runtime-width kernel at the unchanged
128-channel tile; only then add the two-channel instantiation and compare it
against the specialized one. Unsupported widths and dimensions retain the
current kernel.

## Port and rollback

Add compile-time width/channel-count instantiations of the existing register
window, preserving for each channel the exact operation sequence: bias; taps
`j=0..3`; current SiLU; store; window shift; raw-input final-state writeback.
The experiment must not change exact descriptor metadata, scheduler/model
routing, tensor strides, or post-conv dispatch.

Expose an explicit same-binary experiment selector with three arms:

- `0`: current runtime-width, one-channel kernel (sealed baseline);
- `1`: width-four specialization, one channel per thread;
- `2`: width-four specialization, two channels per thread.

Unset retains arm 0 until the evidence supports a default change. Values or
shapes outside the supported experiment fall back to arm 0. Document the
selector in `docs/ENVIRONMENT.md`; factor its parse/launch-contract predicates
into the portable GDN prefill header so CPU-tier tests can kill accidental
default or grid changes.

## Tests and acceptance

RED-first coverage must prove:

1. the selector defaults to arm 0, accepts only the named arms, and invalid
   values roll back;
2. the production `C=8192,K=4` launch contract is respectively 64/64/32 feature
   blocks with block 128, while partial tiles round up safely;
3. CUDA arms 1 and 2 are byte-identical to arm 0 for output and final state over
   BF16/F32 I/O, initial/fresh state, unequal exact chunks, `T<K-1`, partial
   channels and packed row strides;
4. deleting the width specialization or restoring a 128-channel arm-2 tile
   makes a named structural/mutation test fail;
5. the full CUDA GDN suite and cached Qwen3.5-4B correctness remain green, and
   production token files are byte-identical across all measured arms.

First profile the three arms as the same binary under one `${GPU_LOCK}`. An arm
is retained only if it improves the 279-280-program mean outside run noise and
does not worsen total causal-conv time. The winning arm then owes an enclosing
same-workload profile: total/output throughput, TTFT, TPOT/ITL, E2E and peak
VRAM may not regress. Any default flip remains blocked on repeated local A/B
and the hardware-unavailable 27B/35B release-model gates; the 4B vehicle cannot
substitute for them.

No GEMM claim is made. A later GEMM selection separately owes the four-axis,
same-tool invocation-parity proof.
