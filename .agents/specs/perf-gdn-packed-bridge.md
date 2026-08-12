# GDN packed decode: bridge the dtype PREDICTION to the fp8 GEMM that produces it

**Row:** `PERF-GDN-PACKED-BRIDGE` · **issue:**
[#365](https://github.com/mudler/vllm.cpp/issues/365) · **composes:**
[`perf-27b-gdn-packed-reachable.md`](perf-27b-gdn-packed-reachable.md) `ab0b14a9`
× [`perf-fp8-alpha-fold.md`](perf-fp8-alpha-fold.md) `918d4546` · **status:**
`SPIKE`.

Composed against `918d4546` itself, not `row/PERF-MAXSTACK-27B` `e5407053`
which integrates it: tighter (only the row depended on), and `e5407053` is a
bare `Merge branch ...` commit with NO trailers, which reds `commit-trailers` on
every gate. That is a pre-existing maxstack defect, reported not propagated.

## Issue-table keying (#365 is an umbrella)

`#365` is the multi-lever "27B gap DECOMPOSED" issue: the dense-Marlin `gate_up`
half AND this GDN packed-decode half are both inside it. The issue table is keyed
by issue NUMBER and `scripts/check-agent-record.py` refuses a number listed
twice, so it can hold exactly one row per issue. On `origin/main` that entry
belongs to `PERF-27B-DENSE-MARLIN-GATEUP`, and this row leaves it byte-for-byte
intact rather than clobbering a key it does not own (AGENTS.md, Records).

That means `ab0b14a9` carries a record defect worth reporting rather than
propagating: its roadmap edit REWROTE main's `#365` row in place, retargeting it
from `PERF-27B-DENSE-MARLIN-GATEUP` to `PERF-27B-GDN-PACKED-REACHABLE`. Merging
it forward reintroduces the duplicate and reds `check-agent-record`. The
composition here drops that edit; whoever lands `ab0b14a9` owes the same fix or a
deliberate re-key of the umbrella.

## THIS ROW HAS NO TOKEN GATE ON THE PATH IT MODIFIES

Stated first because it bounds every claim below.

SACRED `test_qwen27_paged_engine` pins `models--unsloth--Qwen3.6-27B-NVFP4` at
revision `890bdef7` (`tests/parity/hf_snapshot.h:31,36`). That is a **bf16-tower**
checkpoint: `LoadGdnDense` only populates `in_proj_qkv_fp8` when the tensor dtype
is `F8_E4M3` (`qwen3_5_dense_weights.cpp:426-429`), and on `890bdef7` it is not.
Every line this row changes is reachable **only** on a native-fp8 GDN tower —
`nvidia/Qwen3.6-27B-NVFP4`@`0893e160`. So a green 235/235 says nothing whatsoever
about this change: the gate never executes the branch.

Consequently:

- **Correctness is NOT claimed by this row.** Not "probably fine", not "byte
  identical by construction" — unclaimed.
- Speed MAY be measured, because a speed A/B does not depend on a golden.
- The token gate is owed by `row/GATE-27B-FP8-TOWER-GOLDEN`, which is capturing
  real goldens for `nvidia@0893e160`. **Both toggles must stay DEFAULT OFF until
  that lands and passes**, and no default flip may be argued from this spec.
- The composed arm rounds the fp8 GEMM's f32 accumulator to bf16 *before* the
  alpha multiply instead of after (`perf-fp8-alpha-fold.md`), and swaps the
  decomposed recurrence for a different kernel. Tokens CAN move. A lost token is
  `NEEDS_DECISION`, never a re-cut golden.

## What is broken

Composing the two rows does nothing. Both toggles ON still leaves
`vt::GdnPackedDecode` deselected, because the composed tree carries **two
independent sources for one dtype**:

| | the merged fp8 `mixed_qkv` dtype | consumed by |
|---|---|---|
| maxstack | `fp8_indt` = `GdnFp8InBf16Enabled() && indt==BF16 && outdt==BF16` -> **BF16** | `MergedFp8QkvzD`, i.e. the GEMM that actually allocates the buffer |
| REACHABLE | `GdnFp8MixedQkvDType()` = `return DType::kF32;` — no env dependency at all | `GdnProjectedMixedQkvDType` -> `GdnPackedDecodeDTypesCompatible` |

`GdnPackedDecodeDTypesCompatible` requires `mixed_qkv == kBF16`, so it returns
false, `ShouldUsePackedGdnDecode` returns false, and the arm is inert. ON == OFF.
Any A/B run in that state measures nothing and would be VOID.

This is fail-safe (the prediction under-reports capability, so nothing aborts and
no wrong kernel runs) but it is exactly the drift `ab0b14a9` introduced the helper
to prevent.

### The guard that was supposed to catch it tested the wrong side

`ab0b14a9` guarded the merged arm with

```cpp
VT_CHECK(GdnFp8MixedQkvDType() == DType::kF32,
         "qwen3_5 merged FP8 GDN qkvz: the merged arm emits F32; "
         "GdnFp8MixedQkvDType must agree");
```

That asserts a property of the PREDICTOR against a literal. It cannot observe
what the GEMM allocates, so it passes unchanged while the merged arm emits bf16.
Resolving the compose conflict to maxstack's side deletes it outright. Either
way the invariant it names — predictor agrees with producer — is unprotected.

## Design

### 1. The bridge

`GdnFp8MixedQkvDType()` stops being a constant and takes maxstack's own
three-term expression, plus the arm selector, because **the toggle is
merged-arm-only**: `fp8_indt` reaches `MergedFp8QkvzD` and nothing else; the
split fp8 arm still hardcodes `DType::kF32` on both of its call paths
(`MatmulFp8CutlassPreQuantD` / `MatmulFp8CutlassD`). A predictor that ignored
the arm would claim BF16 on a checkpoint that takes the split path and be wrong
in the UNSAFE direction (predict BF16, produce F32 -> `vt::GdnPackedDecode`
throws on the uniformity check).

So one new input joins `GdnMixedQkvDTypeInputs`:

```
fp8_merged_arm  // ShouldUseMergedGdnFp8Qkvz(...) held for this layer
```

and the fp8 leg of the prediction becomes
`fp8_merged_arm ? fp8_out_dtype : kF32`.

`GdnMergedFp8QkvzEligibilityFor(d, w, conv_dim, value_dim)` is already in scope
at the eligibility call site, and `ProjectGdnQkvz` selects the arm with the same
predicate over the same inputs, so the two cannot disagree.

### 2. The replacement guard — asserts the real invariant, both directions

The point of the helper is that the PREDICTION equals what the GEMM ALLOCATES.
So the guard compares those two, on both arms, in both directions:

- merged arm: `plan dtype == predictor(fp8_merged_arm=true)`
- split arm: `hardcoded F32 == predictor(fp8_merged_arm=false)`

A one-sided `== kF32` is what failed; an equality between producer and predictor
fails whichever side moves. Placed where the buffer is allocated, so it observes
the value actually used rather than re-deriving it.

### 3. Deliberately unchanged

- Every default. Both toggles stay OFF.
- `ShouldUsePackedGdnDecode`, `GdnPackedDecodeDTypesCompatible`,
  `vt::GdnPackedDecode`'s contract, and every kernel.
- The split arm's dtype. Narrowing it is `perf-fp8-alpha-fold.md`'s to make, not
  this row's.

## Tests (RED first)

CPU tier, `tests/vllm/models/test_qwen27_paged_forward.cpp`. Composed-tree
baseline before any edit: **27 cases / 752 assertions, `Status: SUCCESS`**. A
changed assertion count is RED even when it prints "passed".

RED must fail for the intended reason — that the predictor is a constant:

1. merged arm + toggle ON -> predictor says BF16 (fails at HEAD: says F32).
2. merged arm + toggle OFF -> F32.
3. **split** arm + toggle ON -> F32 (the merged-only property; this is the term
   that keeps the prediction from lying in the unsafe direction).
4. `indt`/`outdt` rollback each independently force F32 even with the toggle ON.
5. end-to-end: `GdnPackedDecodeDTypesCompatible` fed the predictor's output is
   true only in case 1.

## Gates

- `scripts/agent-preflight.sh --staged` and on committed HEAD.
- CPU: `test_qwen27_paged_forward`, case AND assertion counts before/after.
- SELECTION (GPU, `-DVLLM_CPP_TRITON=ON` — without it
  `RecordGdnPackedDecodeTritonLaunch` is compiled out and `triton_launches`
  reads 0, indistinguishable from "did not fire"): after
  `ResetGdnPackedDecodeDebugStats()`,
  `packed_launches == 48 * steps && triton_launches == 48 * steps` is the cubin
  firing; `packed_launches == 48*steps && triton_launches == 0` is the hand
  kernel with the cubin REJECTED; `packed_launches == 0` is the model never
  selecting packed.
- NOT a gate here: any token count. See the first section.

## Expected payoff

`GdnDecodeFusedKernel` 28.08 us/call x48 vs vLLM's packed 19.21 us/call
(1.3109 vs 0.9084 ms/step) — about **+0.425 ms/step** of a +1.81 ms/step decode
deficit, plus `GdnPostConvFastKernel` (+0.131 ms/step, no vLLM counterpart)
which the packed kernel absorbs. Anything materially larger is a measurement
defect, not a windfall.

## Stop conditions

- Predictor and producer disagree anywhere -> the new guard fires; fix the
  bridge, never widen the guard.
- Selection counters show `packed_launches == 0` with both toggles on -> report
  the actual deselecting term; do not widen `ShouldUsePackedGdnDecode`.
- `triton_launches == 0` with `packed_launches > 0` -> report which
  `TryTritonPackedDecode` guard rejected it; do not relax the AOT predicate.
- Any token claim -> refused until `row/GATE-27B-FP8-TOWER-GOLDEN` lands.

## Now

Composed and bridged; CPU tier green. Owed: the GPU selection proof under
`-DVLLM_CPP_TRITON=ON`, then a decode-only A/B, then the token gate from
`row/GATE-27B-FP8-TOWER-GOLDEN` before any default flip.
