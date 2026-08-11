# PERF-FP8-ALPHA-FOLD — fold the per-column FP8 alpha into the GEMM epilogue

Issue: [#402](https://github.com/mudler/vllm.cpp/issues/402) (§3 "Lever B"); the
same buffer is the subject of [#417](https://github.com/mudler/vllm.cpp/issues/417)
Finding 1 (the f32-vs-bf16 axis, a DIFFERENT lever on the same tensor — see
§Coordination).
Row: `PERF-FP8-ALPHA-FOLD`
Gate model: `nvidia/Qwen3.6-27B-NVFP4` @`0893e1606ff3d5f97a441f405d5fc541a6bdf404`
Also applies to: `nvidia/Qwen3.6-35B-A3B-NVFP4` @`491c2f1e` (same FP8 tower)

## Scope

Remove the standalone per-column alpha pass that follows the merged FP8 GDN
`in_proj_qkvz` GEMM, by applying that alpha **inside the cuBLASLt epilogue**
through `CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO`.

**In scope:** a vector-alpha overload of the `vt::MatmulFp8CublasLt` seam, its
CUDA cuBLASLt implementation and capability-gated fallback, the plan-key field
that keeps vector-alpha plans from aliasing scalar-alpha plans, and the two
model call sites that today hand-roll the epilogue as a second launch
(`MergedFp8QkvzD`, and its structurally identical default-OFF attention sibling
`MergedFp8QkvD`).

**Out of scope:** the GEMM's output dtype (`#417`/`row/PERF-27B-BF16-FP8-OUT`),
the CUTLASS FP8 path (`VT_DENSE_CUBLASLT_FP8=0` keeps today's two-launch form
verbatim), `vt::MulColVecF32` itself (it stays — it is the fallback and the CPU
tier's only implementation), the `folded == true` case (already one launch, and
untouched), any default flip, and any lifecycle move.

## The gap, MEASURED

Both-arms `nsys`, same instrument, oracle graphed with `--language-model-only`,
identity-asserted, workload matched, 27B at T=4096 prefill:

| | value |
|---|---|
| total prefill gap | **+281.94 ms/request** (ours 3805.18 vs pin 3523.25) |
| `marlin::Marlin` | 512 calls on BOTH arms, **at parity within 0.17%** |
| non-GEMM glue | **92.5% of the gap** |
| **`MulColVecF32Kernel`** | **122.99 ms/req over 48 calls = 43.6% of the entire gap** |

Measured **209.5 GB/s = 77% of the device's 273.1 GB/s peak**: the pass is
bandwidth-saturated, so its cost *is* its width — a full f32 read-modify-write of
a `[4096, 16384]` tensor, per GDN layer, to apply two distinct scalars.

**Why this is not the launch-count lever #402 sized as NEUTRAL.** #402 §4 bounds
this class at 1.04% *on the decode step*, where the pass is 48 tiny launches over
`[1, 16384]` and the cost really is launch overhead — and cites two NEUTRAL
priors of that shape (`.agents/specs/glue-fusion-2026-07-19.md`,
`.agents/specs/moe-silu-vectorize.md:106-108`). At T=4096 the same pass is 4096x
wider and DRAM-bound, so it is a different regime and those priors do not
transfer. Acceptance here is a **prefill** A/B, not a decode one; a decode-only
measurement neither confirms nor refutes this lever.

## Why we pay it and vLLM does not

`ResidentFp8Qkvz` (`src/vllm/model_executor/models/qwen3_5.cpp:3270-3321`) sets
`folded = (qkv.alpha == z.alpha)`. On this checkpoint that is **FALSE**: the two
shards carry different `weight_scale`s, so no single GEMM scalar reproduces both
halves and the alpha becomes a per-output-column vector.

vLLM never reaches that state. `ModelOptFp8LinearMethod.process_weights_after_loading`
(`vllm/model_executor/layers/quantization/modelopt.py:519-529` @`555967922`)
calls `requantize_with_max_scale`
(`vllm/model_executor/layers/quantization/utils/w8a8_utils.py:76-107`), which
**dequantizes every shard and re-quantizes it against `max_w_scale`**, leaving
ONE scalar `weight_scale` for the whole merged linear. Upstream buys its
single-scalar epilogue with a lossy requantization of the weights; we keep the
shards bit-exact and pay a separate pass instead.

So this is not "mirror a fusion vLLM has". It is: keep our strictly-more-exact
per-shard scales and stop paying a separate DRAM pass for them, because cuBLASLt
applies a per-row alpha vector in the epilogue for free. (#417 already records
that upstream's requantization is lossy where ours is exact, so switching to
upstream's single scalar would be a token-changing decision, not this row.)

## The mechanism, and why the layout already matches

`cublasLt.h` (CUDA 13, `nvidia/cu13/include/cublasLt.h`):

- `:953` — `CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO = 3`, documented
  "alpha pointer targets an array in device memory, beta is zero".
- `:1284-1287` — `CUBLASLT_MATMUL_DESC_POINTER_MODE`: "When
  `CUBLASLT_POINTER_MODE_DEVICE_VECTOR` is in use, **alpha/beta vector lengths
  must match number of output matrix rows**."
- `:967` — `CUBLASLT_POINTER_MODE_MASK_ALPHA_DEVICE_VECTOR_BETA_ZERO = 8`, the
  bit to test in `CUBLASLT_ALGO_CAP_POINTER_MODE_MASK` (`:2419`).

Our FP8 D is created **column-major** as `(rows = key.n, cols = key.m,
ld = key.n)` — `src/vt/cuda/cuda_matmul.cu:527-529`, keys documented at
`src/vt/cuda/fp8_plan_cache.h:70-73`, derivation in the block comment at
`cuda_matmul.cu:456-464` (`C = D = out : col-major [N,M]`). cuBLASLt's "output
matrix rows" is therefore `N` — **our row-major output's COLUMNS**. The existing
resident `alpha_vec` is `f32 [total_n]`, one entry per output column, already on
device, already contiguous, built once at load
(`qwen3_5.cpp:3295-3305`). It is exactly the vector cuBLASLt wants, in exactly
the layout it wants. Nothing new is allocated and no parallel path is built.

## Design

1. **Seam.** Add a vector-alpha overload
   `vt::MatmulFp8CublasLt(q, out, a_fp8, b_fp8, const Tensor& alpha_vec)`,
   mirroring the tensor-alpha overload the NVFP4 path already took
   (`.agents/specs/nvfp4-device-alpha.md`, `include/vt/ops.h`). Contract:
   `out[m][n] = alpha_vec[n] * sum_k a[m][k] * b[n][k]`, f32 out only,
   `alpha_vec` f32 `[N]` on the queue device. The op is **total**: the two-launch
   form is an internal fallback, never a caller obligation.
2. **Plan key.** `Fp8PlanKey::scale_mode` already exists and is already in `==`
   and the hash (`fp8_plan_cache.h:86-104`) with only the value `0` defined. Add
   `kFp8ScaleModeAlphaDeviceVec = 1` so a vector-alpha plan can never alias the
   scalar-alpha plan for the same shape — the pointer mode changes the
   descriptor AND can change the selected algo, which is precisely what that
   field exists to separate.
3. **Capability gate.** Set the pointer mode on the descriptor *before* the
   heuristic, then verify `CUBLASLT_ALGO_CAP_POINTER_MODE_MASK` on the returned
   algo carries bit 8. If the heuristic returns nothing, or the cap refuses, the
   op falls back to the current two-launch form (scalar GEMM at alpha=1, then
   `vt::MulColVecF32`) — byte-for-byte today's behavior.
4. **Toggle.** `VT_FP8_ALPHA_VEC_EPILOGUE`, **DEFAULT OFF**, ON only for exactly
   `"1"` (the same parse as its two neighbours in this file). OFF selects the
   fallback in the same binary, so the A/B is same-binary and needs no rebuild.
   The operator flips the default on evidence; this row does not.
5. **Diagnostics.** The vector-alpha GEMM logs under `VT_GEMM_ALGO_LOG=1` with
   the distinct epilogue tag `TN-fp8-alphavec`, so the selected `algoId`/`splitK`
   of the new plan is directly comparable with the scalar plan's line.

## Byte-exactness — the argument, and what pins it

Today: `f32 accum -> x 1.0 -> store f32 -> load f32 -> x alpha -> store f32`.
After: `f32 accum -> x alpha -> store f32`.

The scale type is `CUDA_R_32F` (`cuda_matmul.cu:508-509`, `key.scale_type`), the
store dtype is unchanged, and `x 1.0` is exact in IEEE-754. So both forms perform
the **same single f32 multiply on the same f32 accumulator**. The removed
round-trip is a no-op numerically.

**The real risk is not the multiply, it is the ALGORITHM.** The pointer mode is
part of the descriptor the heuristic sees, so cuBLASLt may return a different
algo — including a different split-K — and f32 addition is not associative. This
is the identical shape-conditional risk `#213` measured and documented
(`.agents/specs/perf-27b-gdn-fp8-merged-qkvz.md:106-136`): at the 27B gate shape
(M=1, K=5120, n=16384) cuBLASLt chose `splitK=1` and the merge was bitwise exact,
while a toy shape chose `splitK=4`/`8` where bit-equality is unattainable by
construction. Reuse that method exactly: **assert bitwise at the gate shapes** and
read the selected algo out of `VT_GEMM_ALGO_LOG=1` on both arms.

If the gate shapes are not bitwise, this row **STOPS and returns
NEEDS_DECISION**. Correctness is not traded for throughput.

## Tests

RED first, in this order:

1. **CPU tier** (`tests/vt/test_fp8_plan_cache.cpp`) — the pure plumbing, which
   is where the two silent-correctness failures live:
   - `VT_FP8_ALPHA_VEC_EPILOGUE` is OFF unless the value is exactly `"1"`.
   - `Fp8ScaleModeFor(true) != Fp8ScaleModeFor(false)`, and two keys differing
     only in that field are two distinct map entries — a collision here would
     reuse a scalar-alpha algo for a vector-alpha matmul.
   - `Fp8AlphaVecCapSupported(mask)`: false for `0`, for `HOST|DEVICE|
     DEVICE_VECTOR` (7) and for `BETA_HOST` alone (16); true for `8` and for a
     full mask. This predicate is the ONLY thing standing between an unsupported
     algo and a wrong result, so it is tested independently of any GPU.
   - Mutation proof: setting `kFp8ScaleModeAlphaDeviceVec` back to `0` must turn
     the suite RED.
2. **CUDA tier** (`tests/vt/test_ops_fp8_cutlass.cpp`) — vector-alpha output is
   **byte-identical** to `scalar GEMM at alpha=1 then MulColVecF32`, at the real
   27B (K=5120, N=10240+6144) and 35B (K=2048) GDN shapes, at M=1 and M=3, with
   two distinct shard alphas. Same method and the same standing precondition as
   `#213`'s equivalence case.
3. **Fallback** — with the toggle OFF the op reproduces the two-launch path
   exactly (same test, forced arm).
4. **Model gates unchanged:** `test_qwen27_paged_engine` **235/235** and
   `test_qwen36_paged_engine` **315/315**. A changed assertion COUNT is RED even
   when it prints "passed".

## Gates

- Focused: the tests above; both SACRED engine gates at their exact counts.
- Full: CUDA `ctest -j 1` on `sm_121a` (`-j 4` OOM-reboots the box).
- Correctness: greedy continuation vs the pinned oracle on both gate models, under
  the ratified distributional gate (the oracle's own greedy is undetermined at
  ~8/32 positions on the synthetic corpus).
- Speed: same-binary `VT_FP8_ALPHA_VEC_EPILOGUE=0|1` A/B at **T=4096 prefill**
  (the regime the lever was measured in), 3 reps, medians, order-alternated,
  idle box, band measured first. Confirm with `nsys --cuda-graph-trace=node` that
  `MulColVecF32Kernel` falls 48 -> 0 per request and that no new kernel replaces
  it.
- Invocation parity: `VT_GEMM_ALGO_LOG=1` on both arms; record `algoId`, `tile`,
  `stages`, `splitK` for `TN-fp8` and `TN-fp8-alphavec` at the gate shapes.

## Risks

- **Algo/split-K reselection** — the whole numerical risk; see above. Detected by
  test 2, not assumed away.
- **The 35B shares this tower.** Any change here moves both gate models; both
  SACRED gates run.
- **Graph capture.** Nothing new is allocated per call and `alpha_vec` is a
  load-time resident, so the capture hazard that bit `PERF-27B-LMHEAD-FP4` does
  not apply — but the plan cache is default OFF, so the heuristic still runs
  inside capture exactly as it does today. Unchanged, deliberately.
- **Driver variance.** A driver whose fp8 algos do not advertise bit 8 silently
  gets today's behavior. That is the correct outcome, and it is why the cap is
  checked rather than assumed.

## Coordination

`row/PERF-27B-BF16-FP8-OUT` (#417 Finding 1) narrows the SAME buffer to bf16,
which would halve this pass rather than remove it (−61.5 ms of the 122.99). The
two levers are not additive and must not both be counted: if this row lands, the
pass is gone and #417's saving on *this* tensor collapses to the conv/post-conv
consumers it also names. This row deliberately does NOT touch the output dtype,
so the two can land in either order without conflict.

## Stop conditions

- Not bitwise at the gate shapes -> `NEEDS_DECISION`. Never adjust a golden.
- `alpha_vec` not already device-resident in the right layout -> report the cost,
  do not build a parallel path. (It is; verified above.)
- Do not flip the default, do not widen into the output dtype, do not touch the
  `folded == true` path.

## Now

Spec committed. Implementation next: the vt seam + CUDA pointer-mode arm + CPU
plumbing tests. GPU evidence (tests 2-4 and every gate above) is **owed** and
cannot be produced in this worktree — the implementer host has no `nvcc` and no
CUDA device, so every numerical claim ships UNVERIFIED for the operator to run.

## Outcome

Pending.
