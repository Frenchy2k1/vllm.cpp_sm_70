# PERF-FP8-ALPHA-FOLD — fold the per-column FP8 alpha into the GEMM epilogue

Issue: [#402](https://github.com/mudler/vllm.cpp/issues/402) (§3 "Lever B"); the
same buffer is the subject of [#417](https://github.com/mudler/vllm.cpp/issues/417)
Finding 1 (the f32-vs-bf16 axis, a DIFFERENT lever on the same tensor — see
§Coordination).
Issue: [#339](https://github.com/mudler/vllm.cpp/issues/339) — **the issue
§Attempt 4 actually implements**, and it was linked from nowhere until this
repair. #339 is "every fp8 input projection asks for an f32 output, selecting the
slower nvjet template family (vLLM emits bf16)"; `VT_GDN_FP8_IN_BF16` is exactly
its fix for the merged GDN `in_proj`. Its own evidence (48 f32-out projections at
18.51 ms vs 48 bf16-out at 7.05 ms, `benchmark-record.md:17209-17217`) is the
prior for this lever, and its upstream anchor `modelopt.py:458` is the one
§Attempt 4 cites.
Row: `PERF-27B-LMHEAD-FP4` — the owning roadmap row. `PERF-FP8-ALPHA-FOLD` is
this work's branch name, not a roadmap block; #339 and #402 are both placed under
`PERF-27B-LMHEAD-FP4` in the roadmap issue table.
Gate model: `nvidia/Qwen3.6-27B-NVFP4` @`0893e1606ff3d5f97a441f405d5fc541a6bdf404`
Also applies to: `nvidia/Qwen3.6-35B-A3B-NVFP4` @`491c2f1e` (same FP8 tower)

> **No committed gate executes this path.** The SACRED 27B gate pins
> `models--unsloth--Qwen3.6-27B-NVFP4` @`890bdef7`, a **bf16-tower** checkpoint;
> the fp8 tower named above lives in `nvidia`@`0893e160` and no committed gate
> loads it. See §Outcome, "What the green suite does and does not say".

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

## The mechanism is UNAVAILABLE on this hardware — MEASURED 2026-08-11

The pointer-mode arm is implemented, correct, and **never executes on GB10**.
Operator run of the 27B gate, same binary, `VT_GEMM_ALGO_LOG=1`:

| arm | `TN-fp8-alphavec` plan tags | `TN-fp8` algo lines |
|---|---|---|
| `VT_FP8_ALPHA_VEC_EPILOGUE=0` | 0 | 5 |
| `VT_FP8_ALPHA_VEC_EPILOGUE=1` | **0** | **5 (IDENTICAL)** |

The fallback fired on every call, so **both arms ran byte-identical code**. The
0.9954 / 0.9973 A/B taken from those runs is therefore **VOID, not negative** —
it measured the same code against itself and says nothing about the lever.
cuBLASLt on sm_121a does not hand back a vector-alpha-capable algo for this fp8
shape. The arm stays in the tree, **default OFF**, for hardware or a driver that
does offer the mode; §Risks already anticipated exactly this outcome.

**What the run could NOT tell us, and now can.** A refused plan emits nothing at
all, so "no heuristic for the shape once the pointer mode is on the descriptor"
and "an algo was returned but its `CUBLASLT_ALGO_CAP_POINTER_MODE_MASK` refuses
our mode" were indistinguishable — and they point at different next steps. The
refusal now logs its NAMED cause and the mask actually read
(`cuda_matmul.cu:590-596`, `Fp8PlanRefusalFor` in `fp8_plan_cache.h`, pinned by
`tests/vt/test_fp8_plan_cache.cpp`). Re-run with `VT_GEMM_ALGO_LOG=1` and read
`reason=` / `pointerModeCapMask=` rather than re-deriving an absence.

## The z-slice fallback is REJECTED: it cannot be byte-exact

The obvious fallback — run the GEMM with **scalar** `alpha = qkv.alpha` and scale
only the 6144-column z-slice afterwards, a 2.7x narrower pass worth ~62% of the
122.99 ms — **fails the row's own correctness bar**, and the arithmetic says so
before any GPU does.

`alpha_vec` is genuinely two-valued and the slice is genuinely well-shaped; both
preconditions hold (§Verified below). The defect is the multiply, not the layout.
A scalar GEMM alpha applies to **every** output column, so the z-slice does not
keep a raw accumulator to scale — it must be *corrected* by the ratio
`r = fl(z.alpha / qkv.alpha)`:

```text
today     out = fl(acc * B)                 -- acc*1.0 is exact, then ONE multiply
z-slice   out = fl(fl(acc * A) * r)         -- TWO roundings, and r itself inexact
```

That is double rounding on an already-rounded product, not "the same per-column
scalar applied elsewhere". Measured over 2e6 random f32 accumulators per case at
representative modelopt folded alphas (`input_scale * weight_scale`), **25-36% of
the z-slice's f32 words differ from today's result by 1 ulp** — six of six
random (A,B) pairs, worst case 35.84%.

There is exactly one escape, and it is a property of the checkpoint, not of the
code: **iff `qkv.alpha` is exactly a power of two**, `acc * A` is exact and
`z.alpha / qkv.alpha` is exact (both are pure exponent shifts), so the corrected
suffix reproduces `fl(acc * B)` bit for bit — the same sweep returns **0 / 2e6**
mismatches for `A = 0.0078125`. Nothing in `ResidentFp8Qkvz` guarantees or checks
that, and a modelopt `amax/448` scale is not a power of two in general. Gating
the arm on `std::frexp(qkv.alpha).first == 0.5` would be sound but would leave a
lever that silently does nothing on almost every checkpoint.

Per §Stop conditions ("Not bitwise at the gate shapes -> `NEEDS_DECISION`. Never
adjust a golden"), the z-slice is **not implemented**. It is a ~76 ms/req
throughput win in exchange for a 1-ulp perturbation of ~30% of the GDN
`in_proj_qkvz` output on both gate models — a correctness trade this row is
explicitly forbidden from making unilaterally.

## Verified while rejecting it (both preconditions HOLD)

Recorded so the next attempt does not re-derive them:

1. **`alpha_vec` IS two-valued.** `qwen3_5.cpp:3304-3306` builds it with exactly
   two `std::fill` runs — `qkv.alpha` over `[0, qkv.n)`, `z.alpha` over
   `[qkv.n, total_n)`. Two runs, first one longer (10240 vs 6144 at the 27B gate),
   so the z-slice is correctly the narrower half to correct.
2. **The z-slice is the right shape for a narrowed launch.** In the row-major
   `[M, total_n]` f32 output a column range is contiguous *within* each row and
   strided across rows — 6144 f32 = 24 KB contiguous per row, so coalescing is
   unaffected and the traffic falls with the width, as a bandwidth-bound pass
   requires. It needs **no new kernel**: `MulColVecF32Kernel`
   (`cuda_glue.cu:94-112`) already takes `row_size` and `row_stride` separately,
   so a strided sub-view with `row_size = z.n`, `row_stride = total_n` and the
   data pointer offset by `qkv.n` floats drives the existing kernel unchanged.

The third precondition — that folding a scalar alpha into the GEMM leaves the
**prefix** bit-exact — is UNVERIFIED and is itself shape-conditional: `alpha` is
a runtime argument, not a descriptor field, so it cannot reselect the algo, but
under a split-K reduction scheme whether alpha is applied before or after the
partial reduction decides whether `fl(acc*A)` is even well-defined. #213 measured
`splitK=1` at this gate shape; that would have to be re-confirmed.

## The A-scale OUTER_VEC API is ALSO refused — MEASURED 2026-08-12

The second cuBLASLt mechanism for the same fusion was probed directly on GB10 and
is **refused at heuristic time on every shape and every output dtype tried**.
Unlike attempt 1 this was settled BEFORE any integration, by a standalone probe
(`scripts/probe_fp8_outer_vec_scale.cu`, `scripts/probe_fp8_outer_vec_dtypes.cu`)
whose control arms prove it executed.

Device `NVIDIA GB10` sm_121a, cuBLASLt `130101`, CUDA runtime/driver 13000,
driver `580.159.03`.

Pass 1 — the ten real gate shapes (27B `N=10240+6144, K=5120` and 35B
`N=4096+2048, K=2048`, each at `M ∈ {1, 3, 128, 1024, 4096}`), f32 D:

| descriptor form | result, ALL TEN shapes |
|---|---|
| host scalar alpha (today's shipped form) | `SUCCESS`, algoId 67, **splitK=1**, matmul + sync OK |
| `A_SCALE_POINTER` → device f32 **scalar**, mode `SCALAR_32F` | `SUCCESS`, **identical algo**, matmul + sync OK |
| `A_SCALE_POINTER` → device f32 `[N]`, mode `OUTER_VEC_32F` | heuristic **`CUBLAS_STATUS_INVALID_VALUE`**, 0 algos |
| `A`+`B` both `OUTER_VEC_32F` | heuristic **`CUBLAS_STATUS_NOT_SUPPORTED`**, 0 algos |

Pass 2 — is the refusal conditioned on our f32 output, or on our shapes? Swept
`D ∈ {f32, bf16, f16, e4m3}` × 4 shapes (the two gate shapes, plus a canonical
`4096³` square as a device-level control) × 5 scale configs:

- `OUTER_VEC_32F` accepted in **0 of 48** swept cells (4 shapes × 4 dtypes × the
  3 vector-scale configs) — of which 36 have a GREEN control at the same
  shape/dtype, so the refusals are the mode's, not the descriptor's.
- The no-scale and `SCALAR_32F` controls succeed at **f32, bf16 AND f16** on every
  shape (algoId 67), so the probe, the layouts and the TN descriptor are sound.
- `e4m3` D is `NOT_SUPPORTED` for *every* config including the controls — an
  unrelated limit, not evidence about the scale mode.

**Named cause: `CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F` is not implemented
for e4m3 TN matmuls by cuBLASLt 13.1.1 on GB10/sm_121a.** It is a
device/driver-level absence, NOT shape-conditional and NOT output-dtype-
conditional — the square control and the bf16/f16 columns refuse identically.

Three things this pins that the reasoning had assumed:

1. The A-scale **pointer** mechanism genuinely works here — the `SCALAR_32F` arm
   returns the same algo as the host scalar. Only the vector **MODE** is missing,
   so the earlier reading ("gated by scale/compute-type support, not by
   `POINTER_MODE_CAP_MASK`") was right about *which* gate applies and wrong only
   about the outcome.
2. The refusal lands on **`cublasLtMatmulAlgoGetHeuristic`**, not on
   `cublasLtMatmul` as `cublasLt.h:1420-1430` implies for an unsupported scale
   combination. A probe that only checked the matmul return code would have read
   a false green; one that only checked the heuristic on the shipped form would
   have seen nothing at all.
3. `splitK=1` at every gate shape and every M — confirming the standing
   precondition #213 measured, for whichever mechanism a future driver offers.

**Do not re-derive this.** Re-run the two probes (each builds in seconds with
`nvcc -arch=sm_121a ... -lcublasLt -lcudart` and needs no model, no engine build
and ~0.5 GB of device memory) against a NEW driver or a different GPU. A cell
that flips to `ok#NN` is the signal that this row is live again; pass 1 then
byte-compares the epilogue against the two-launch reference automatically.

### Why it was the right thing to try (kept, so the record explains itself)

The pointer-mode API is refused, but it was **not the only** way cuBLASLt applies
a per-column f32 vector in an fp8 epilogue, and the alternative had not been
tried.

`CUBLASLT_MATMUL_DESC_A_SCALE_POINTER` (17) with `CUBLASLT_MATMUL_DESC_A_SCALE_MODE`
(31) set to `CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F` (3) — `cublasLt.h:923-940`,
"Scaling factors are vectors of `CUDA_R_32F` values ... expected to have M and N
elements respectively, and each (i,j)-th element of product of A and B is
multiplied by i-th element of A scale". With our TN layout (`op(A)` = weight
`[N,K]`, D column-major `[N,M]`), that A-side length "M" **is our N** — so it
consumes the *same* resident `f32 [N]` vector, in the *same* layout, with no
repack and no new allocation, exactly as §The mechanism derived for the alpha
vector.

Why it is a genuinely different shot rather than the same one renamed:

- It is fp8's **per-channel/rowwise scaling** path — the one `torch._scaled_mm`
  uses for rowwise fp8 on Hopper/Blackwell — not a general alpha-vector epilogue,
  so it is far likelier to be implemented for exactly our e4m3 TN shape.
- It is gated by **scale/compute-type support**, surfacing as `CUBLAS_INVALID_VALUE`
  from `cublasLtMatmul`, *not* by `CUBLASLT_ALGO_CAP_POINTER_MODE_MASK` — the cap
  bit that refused us has no authority over it.
- It applies the vector to the product, i.e. one multiply on the f32 accumulator,
  so the byte-exactness argument of §Byte-exactness carries over unchanged, and
  the same bitwise gate-shape test decides it.

That reasoning was sound and the conclusion was wrong: the mode is simply absent
here. Settling it cost one standalone probe and no integration — which is the
transferable lesson, since attempt 1 cost an integration plus a void A/B to learn
strictly less.

## An integration hazard this probe surfaced, for whoever gets a driver that works

`A_SCALE_POINTER` is a **descriptor** attribute, whereas attempt 1's alpha was a
per-call **argument** to `cublasLtMatmul`. Our fp8 plan cache
(`GetOrBuildCachedFp8Plan`) is keyed by shape and returns a shared descriptor, so
caching a plan that carries a baked scale pointer would apply the FIRST GDN
layer's `alpha_vec` to all 48 same-shaped layers — a silent wrong-numbers bug
that no shape-keyed test would catch. Any future integration must either set the
pointer per call on the fetched plan or refuse to cache vector-scale plans;
`Fp8PlanKey` cannot express the difference, because the pointer is not part of
the shape. Recorded here because the refusal means nobody will hit it *yet*.

## Now

Both cuBLASLt mechanisms for this fusion are **measured unavailable on
GB10/sm_121a**, each with a named cause, and the third route is measured
non-exact. The row has no remaining implementable path on this hardware:

| attempt | mechanism | verdict |
|---|---|---|
| 1 | `POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO` | REFUSED — no algo advertises the cap bit; arm implemented, inert, default OFF |
| 2 | scalar GEMM alpha + z-slice correction | REJECTED — double rounding, 25-36% of z-slice words off by 1 ulp; not implemented |
| 3 | `A_SCALE_POINTER` + `OUTER_VEC_32F` | REFUSED — mode unimplemented for e4m3 TN; 0 of 48 cells, all shapes and dtypes |
| 4 | bf16 D — HALVE the pass, don't eliminate it | IMPLEMENTED, default OFF, UNMEASURED. No vendor capability needed; see §Attempt 4 |

Open decisions for the operator, in order of value:

1. Run attempt 4's gates and A/B (§Attempt 4). It is the only lever on this tensor
   that depends on no cuBLASLt capability, it is built and default OFF, and Pass 2
   already proved cuBLASLt serves a bf16 D for these exact fp8 shapes (algoId 67 at
   every shape, `splitK=1`). It is NOT value-neutral, so the SACRED gates decide it.
2. Whether a 1-ulp perturbation of ~30% of this tensor is worth ~76 ms/req
   (attempt 2). This row's default answer is no and it is forbidden from deciding
   unilaterally.
3. Re-run the two probes on any new driver/GPU before assuming attempts 1 and 3
   are permanently refused.

**Not a ceiling.** The pass is 43.6% of the measured 27B prefill deficit and the
arithmetic that would remove it is exact; what is missing is a vendor kernel, on
this driver, and that is a dated fact rather than a limit. The next traceable
hypotheses are (a) #417's bf16 narrowing, which halves the same pass with no
vendor dependency, and (b) fusing the multiply into the *consumer* of this buffer
(the conv/post-conv reader) so the vector costs no separate DRAM pass at all —
untried, and dependent on no cuBLASLt capability whatsoever.

GPU evidence for the capability question is now **paid** (probes above, run on
dgx.casa under the GPU lock). What remains owed is only what a *landed* lever
would need: the SACRED engine gates and a prefill A/B. Nothing here changes
runtime behavior, so neither is triggered.

## Attempt 4 — HALVE the pass instead of eliminating it (IMPLEMENTED, UNMEASURED)

Hypothesis (a) of §Not a ceiling is now built, DEFAULT OFF, behind
`VT_GDN_FP8_IN_BF16`. It depends on **no cuBLASLt capability at all**: it does not
try to remove the pass, only to halve the bytes it moves.

The pass is bandwidth-saturated at 77% of peak, so its cost IS its width. Emitting
the merged fp8 qkvz GEMM's D as **bf16** makes the read-modify-write move 4
bytes/element instead of 8 — an expected ~61 ms/req, ~21.8% of the measured 27B
prefill gap. Pass 2 of the probe already established the premise on this hardware:
cuBLASLt serves a bf16 D at algoId 67 for these exact fp8 shapes, at every gate
shape and every M, with `splitK=1`.

What changed, and why none of it is a parallel path:

- `vt::MulColVecF32` gained a **bf16 store arm** as a dtype axis on the existing
  kernel — the same widening `SigmoidGateBf16Kernel` took for `Tattn` — on CUDA
  *and* CPU, so it stays a portable op rather than a CUDA-only capability. `col`
  stays f32 and the multiply stays f32 on both arms; only the store rounds. The
  f32 arm is byte-identical to the `*=` it replaces.
- `vt::MatmulFp8CublasLtAlphaVec` accepts a bf16 `out`, which is what removed the
  blocker `row/PERF-GDN-BF16-CHAIN` had to decline against.
- `MergedFp8QkvzD` takes a `want` dtype (default f32 = today, byte for byte).

**A bf16 D is REFUSED the epilogue arm, deliberately.** At f32 the epilogue and
fallback arms are one arithmetic, which is what lets
`VT_FP8_ALPHA_VEC_EPILOGUE` be a pure performance A/B. At bf16 they are not: the
fallback rounds twice (store bf16, multiply, store bf16) and the epilogue rounds
once — the *same* double-rounding defect that got attempt 2 rejected, pointing the
other way. Rather than let a performance toggle move tokens, a bf16 D always takes
the two-launch arm. Nothing is lost today (the epilogue arm is measured dead at
every dtype here), and the note is left that the single-rounding epilogue form is
the MORE vLLM-faithful one, so lifting the gate is a deliberate future decision.

**This attempt is NOT byte-preserving and does not claim to be.** Narrowing D
rounds the accumulator to bf16 *before* the alpha multiply instead of after it, so
tokens can move. The direction is toward the oracle — vLLM emits bf16 here
(`modelopt.py:458`) and our f32 is the deviation — but that is an expectation, not
evidence. **The two SACRED engine gates decide it**, at their exact case AND
assertion counts; a lost token is `NEEDS_DECISION`, never a re-cut golden.

Owed before any default flip: both SACRED gates per arm, the same-binary
`VT_GDN_FP8_IN_BF16=0|1` A/B at T=4096 prefill, and an nsys kernel list showing
`MulColVecF32Kernel` unchanged in COUNT (48) but halved in bytes/time, with
`CastBf16Kernel` falling as the `z` cast disappears.

## Outcome

**No ELIMINATION lever landed, and the reason is external; a HALVING lever is now
built and awaiting its gates.** All three routes to folding the per-column FP8
alpha into the GEMM are measured, not argued:

- **Measured:** the standalone probes above — controls green on every shape and
  output dtype, `OUTER_VEC_32F` refused in 0/20 cells; and the earlier
  `VT_GEMM_ALGO_LOG=1` run showing zero `TN-fp8-alphavec` tags. Also measured
  incidentally and useful to a neighbouring row: `splitK=1` at every gate shape,
  and a bf16 D served for these exact fp8 shapes.
- **Rejected and why:** the z-slice fallback, on double-rounding (25-36% of words
  off by 1 ulp over 2e6 accumulators, 6/6 alpha pairs) — a correctness trade this
  row is forbidden to make. Exact only if `qkv.alpha` is a power of two, which no
  checkpoint guarantees.
- **Built, not measured:** attempt 4 (§Attempt 4) halves the pass instead of
  removing it, by emitting a bf16 D. It needs no vendor capability, it is DEFAULT
  OFF, and it is NOT byte-preserving — the SACRED gates decide it. The implementing
  host had no `nvcc` and no GPU, so nothing in it is numerically verified.
- **Why the defaults are set as they are:** `VT_FP8_ALPHA_VEC_EPILOGUE` stays
  DEFAULT OFF because its arm cannot execute on this hardware — the fallback
  fires on every call, so the default is the *only* behavior either way, and the
  arm is retained solely for hardware that offers the mode. `VT_GDN_FP8_IN_BF16`
  stays DEFAULT OFF because it changes values and no gate has run. No default
  flipped; no golden touched; with both toggles unset, runtime behavior is
  byte-identical to before this row.

The measured 122.99 ms/req (43.6% of the 27B prefill gap) is therefore still
open — half of it addressable by a lever that is now built and unmeasured, and the
whole of it explicitly NOT declared a ceiling. §Not a ceiling's remaining untried
hypothesis is fusing the multiply into the buffer's CONSUMER (the conv/post-conv
reader) so the vector costs no separate DRAM pass at all. The row's lasting product
is the elimination of two plausible mechanisms with named causes, a re-runnable
instrument that settles them in seconds on new hardware, one recorded plan-cache
hazard for whoever gets that hardware, and a portable bf16 store width on
`vt::MulColVecF32` that the GDN-chain row was blocked on.

### What the green suite does and does not say

**No committed gate executes this path.** This is the single most important
sentence in the row and it is not softened anywhere: the SACRED 27B gate pins
`models--unsloth--Qwen3.6-27B-NVFP4` @`890bdef7`, which is a **bf16-tower**
checkpoint. The fp8 tower that `VT_GDN_FP8_IN_BF16` modifies is
`nvidia/Qwen3.6-27B-NVFP4` @`0893e160`, and no committed gate loads it.

That is measured, not inferred. Forcing BOTH toggles permanently ON — deleting
the env checks outright, so the arm cannot be skipped — still leaves the full CPU
suite green: **395/395** on the four-lever stack the mutation was run against,
and **392/392** on this reduced stack, which drops the three suites levers 2-4
added. A suite that is identical with the code forced on and forced off is a suite
that never ran the code. So the green result carries no information
about this lever whatsoever, and quoting it as if it did would be the exact
dishonesty this row is trying not to commit.

`row/GATE-27B-FP8-TOWER-GOLDEN` is building the arm that would actually execute
it. Until that lands, every numeric claim in §Attempt 4 stays **UNMEASURED**,
`docs/FEATURES.md` says `UNMEASURED`, and the toggle stays DEFAULT OFF.

Two consequences worth stating plainly, because both were found by review rather
than by a gate:

- **`splitK=1` is now ENFORCED, not merely stated.** It had been a byte-exactness
  precondition of the bf16-D arm since §Byte-exactness, and nothing checked it:
  the only `CUBLASLT_ALGO_CONFIG_SPLITK_NUM` read in the tree lived inside
  `MaybeLogGemmAlgo`, which returns immediately unless `VT_GEMM_ALGO_LOG=1` — so
  on every production run the precondition was unobserved. It is not a cosmetic
  gap: `out_type` is part of `Fp8PlanKey`'s `==` and its hash, so a bf16 D
  deliberately selects a DIFFERENT plan from the f32 D, and a different split-K is
  precisely the freedom that key grants cuBLASLt. Had it been taken, the delta
  would have been a REDUCTION-ORDER change wearing a store-width change's clothes
  — and a bf16 store is very good at hiding those from a token gate. The verdict
  now lives in `Fp8Bf16DSplitKVerdict` (`src/vt/cuda/fp8_plan_cache.h`, pure and
  CPU-tested, four mutations proved red) and `RequireBf16DSplitKOne`
  (`src/vt/cuda/cuda_matmul.cu`) throws on the bf16-D path unless the selected
  plan reports `splitK == 1`. An UNREADABLE splitK refuses too — unknown is
  neither absence nor success — and it is a hard refusal rather than a `<1%`
  tolerance, because a tolerance here would only be measuring how well bf16
  conceals the defect. The f32-D arm never claimed the premise and is not held to
  it.
- **`AlphaVecBf16TakesTwoLaunch` is NOT coverage of what this lever changes.**
  It is a good test of a different thing. It pins that a bf16 D REFUSES the
  cuBLASLt epilogue arm and always takes the two-launch fallback — but *both* of
  its arms apply alpha AFTER the GEMM, so both are `round -> scale`. The
  arithmetic §Attempt 4 actually changes is `round-then-scale` vs
  `scale-then-round`: narrowing D rounds the f32 accumulator to bf16 BEFORE the
  alpha multiply, where the f32-D path rounds after it. Nothing in the tree
  compares those two orders. Do not cite this case as evidence that the narrowing
  is safe; it does not test that, and it was never written to.

### Why this branch ships lever 1 ALONE

The four-lever "maxstack" was measured against lever 1 by itself on the canonical
shape and could not be told apart from it: **0.9755 mean / 0.9707 median for all
four, against 0.9758 mean for bf16-D alone**, with spreads of 1.035 and 1.049 —
overlapping arms, so the grid does not resolve a difference. A decode trace of the
same shape reads **91.72% decode**, and levers 2-4 are all prefill glue. Their
measured end-to-end effect on the shape we gate is ZERO.

They are PARKED with their history and evidence intact, not deleted:

| lever | toggle | branch |
|---|---|---|
| 2 | `VT_GDN_FP8_ALPHA_IN_CONV` (§Attempt 5) | `row/PERF-ALPHA-IN-CONV-PARKED` |
| 3 | `VT_SILU_VEC` | `row/PERF-GLUE-KERNELS-PARKED` |
| 4 | `VT_RMSNORM_PREFILL_GEOMETRY` | `row/PERF-GLUE-KERNELS-PARKED` |

Lever 2 additionally carries an untested seam: 9 kernels across 3 backends and 8
model call sites, whose model-layer wiring has no test at any tier
([#468](https://github.com/mudler/vllm.cpp/issues/468)). Shipping that for a
number the instrument cannot see is not a trade this row makes. The parked
branches each record the same reasoning in their spec's `## Now`, so it is not
re-derived by whoever picks them up.
