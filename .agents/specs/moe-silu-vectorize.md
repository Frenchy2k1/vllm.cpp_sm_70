# `PERF-35B-SILU-VECTORIZE` — SiluAndMul is ~9x slower per launch than vLLM's

Issue: [#213](https://github.com/mudler/vllm.cpp/issues/213) (`PERF-27B-LMHEAD-FP4`:
Qwen3.6 NVFP4 baselines, 27B and 35B-A3B, must reach vLLM speed parity).
Owning row: `PERF-27B-LMHEAD-FP4`. Lifecycle: `ACTIVE`.

## Scope

`vt::SiluAndMul` on CUDA (`src/vt/cuda/cuda_ops.cu:557`). Nothing else. The MoE
routing, the Marlin GEMMs and the shared-expert path are all out of scope; this
row changes one elementwise kernel and its launcher.

## The measurement that opened it

Paired profile at c8 on `nvidia/Qwen3.6-35B-A3B-NVFP4`@`491c2f1e` (ours nsys,
vLLM the harness's own torch-profiler tool), normalised per 24-request leg and
per layer-step (387 steps x 40 layers):

| engine | kernel | launches / layer-step | time / leg | per launch |
|---|---|---:|---:|---:|
| ours | `SiluAndMulKernel` | 2.00 | 695 ms (**4.1%** of GPU) | **22.6 us** |
| vLLM | `triton_poi_fused_mul_silu_slice_0` | 1.00 | 38.2 ms | 2.47 us |
| vLLM | `vllm::act_and_mul_kernel<..., __nv_bfloat162, ...>` | 0.98 | 36.9 ms | 2.42 us |

**Launch counts MATCH (2.00 vs 1.98).** The gap is entirely per-launch cost:
~9.2x. In GPU-time share that is 4.1% against 0.50%, i.e. **3.6 percentage
points** — over half the ~6% mid-band deficit that remains after the two levers
already landed (`VT_MARLIN_DENSE_PAIR`, `VT_SHARED_DOWN_BF16`).

Correction recorded with this spec: an earlier note claimed we launch "2x the
SiLU kernels" of vLLM. That was wrong — it matched only ONE of vLLM's two
SiLU-family kernel names. The counts are equal; the cost is not.

## Why ours is slow (hypothesis, to be confirmed by the A/B)

`SiluAndMulKernel` is a flat grid-stride loop over `n = t*d` that recomputes the
row and column with **two integer divisions per element**:

```cpp
const int64_t i = idx / d;
const int64_t j = idx - i * d;
```

and then does three scalar accesses. Integer division is tens of cycles, and the
op is otherwise pure bandwidth. vLLM's `act_and_mul_kernel`
(`csrc/activation_kernels.cu`) instead uses **one block per token row** with
`threadIdx.x` striding over `d` (no division at all) and a **vectorized
`__nv_bfloat162`** element type, i.e. 2 elements per access.

## Design

Mirror vLLM's structure rather than inventing one:

1. Block-per-row launch: `grid = t`, `threads = min(d, 1024)`, `j` from
   `threadIdx.x` with a stride loop over `d`. Removes both divisions.
2. Vectorized path when `d % 2 == 0` and all three pointers are 4-byte aligned:
   load/store `__nv_bfloat162`, compute silu on both halves in f32.
3. Keep the existing scalar kernel as the fallback for odd `d`, unaligned
   buffers, and non-bf16 dtypes, so no shape loses support.

## Numerics

The silu math stays `gate / (1 + expf(-gate))` in f32 and the store still rounds
once, per element, exactly as now. Vectorising changes only how many elements a
thread handles, never the arithmetic or its order (this is elementwise — there
is no reduction to reassociate). **The change is expected to be bit-identical**,
and that is a gate condition below, not an assumption.

## Tests

- RED first: a unit case in the existing `SiluAndMul` op test covering
  `d % 2 == 1` (fallback), an aligned even `d` (vector path), and a `t == 0`
  edge, asserting byte-equality against the current kernel's output.
- Mutation: force the vector path on an odd `d` and prove the test fails.

## Gates

- `test_qwen36_paged_engine` **315/315** and `test_qwen27_paged_engine`
  **235/235**, `Status: SUCCESS`, with **assertion counts unchanged**. Counts are
  part of the gate identity — a changed count is RED even at "0 failed" (that
  trap already produced a false green once on this row).
- Same-binary A/B, 3 reps/arm, order-alternated, c4 and c8, single load per arm.
- Byte-identical warm-server greedy continuation across arms.

## Risks

- Occupancy: `d` for the routed path is `moe_intermediate_size` = 512 and for the
  shared path `Is` = 512, so a block-per-row grid is `t*topk` or `t` blocks —
  small at decode. If that under-fills the GPU the win may not materialise; the
  A/B decides, and a grid-stride over rows is the fallback shape.
- Alignment: NVFP4/bf16 buffers come from `DBuf`; the vector path must check
  alignment at runtime, not assume it.

## Stop conditions

- If the A/B shows < 0.5% at both c4 and c8, record NEGATIVE and stop; do not
  iterate on tiling.
- If bit-identity fails, stop and report rather than reaching for the
  distributional gate — this op has no legitimate reassociation.

## Evidence

`dgx:~/mbprof/kern.csv` (ours), `dgx:~/vlprof2/vllm-profile/` (vLLM),
`/tmp/vsilu.py` + `/tmp/silu.py` normalisation scripts.

## Outcome

**NEGATIVE — the premise was an averaging artifact, and the lever is dropped.**

Implemented the row-blocked kernel exactly as designed (block per token row, no
integer divisions, flat kernel kept as fallback), behind `VT_SILU_ROW`. It is
bit-identical: `test_qwen36_paged_engine` 315/315 and `test_qwen27_paged_engine`
235/235, `Status: SUCCESS`, assertion counts unchanged on BOTH arms, and all
four warm-server greedy probes byte-identical.

Same-binary A/B, 3 reps/arm, order-alternated:

| conc | flat | row | delta |
|---|---|---|---|
| c8 | 195.6, 197.3, 196.6 -> 196.64 | 195.9, 196.3, 196.3 -> 196.29 | **-0.17%** |
| c4 | 141.7, 141.7, 141.8 -> 141.72 | 142.4, 141.8, 142.6 -> 142.45 | +0.52% |

Bands OVERLAP at both points. The stop condition in this spec ("< 0.5% at both
c4 and c8 -> record NEGATIVE and stop; do not iterate on tiling") governs, so
the knob was NOT landed.

**Why the premise was wrong.** The 22.6 us per launch that motivated this row is
a MEAN over a bimodal distribution: `min 1.34 us, med 18.88 us, max 979.78 us,
stddev 47.69 us`. A handful of enormous prefill launches drag the mean up. Our
DECODE-phase SiluAndMul is ~1.3 us -- FASTER per launch than the ~2.45 us vLLM
average it was compared against. There was never a 9.2x gap to close.

This is the prefill/decode mixing trap the record already documents (whole-run
kernel aggregates mix phases; use a decode-only window or a two-length diff, and
watch for Max >> Med). It was not applied when this spec was written: the
comparison used our whole-run mean against vLLM's whole-run mean, over different
prefill/decode mixes. **Before quoting a per-launch ratio, read the DISTRIBUTION
(min/med/max/stddev), not the mean.**

Consequence for the mid-band: the "3.6 percentage points of glue" figure that
this row was scoped around does not survive either -- it was computed from the
same inflated mean. The mid-band's remaining ~5% is still unattributed, and the
next attempt must start from a decode-only window on BOTH engines under ONE
tool, which is the same condition that closed the per-launch marlin question.
