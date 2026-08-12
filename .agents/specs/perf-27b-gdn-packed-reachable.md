# PERF-27B-GDN-PACKED-REACHABLE — the vendored FLA packed-decode kernel is unreachable on the NVFP4 27B by construction

Issue: [#365](https://github.com/mudler/vllm.cpp/issues/365)
Row: `PERF-27B-GDN-PACKED-REACHABLE`
Base SHA: `0eb049f7e3fe522a1e8763c59be5bcfbbab53139`
Related: [#362](https://github.com/mudler/vllm.cpp/issues/362) (the first half of
the decode decomposition), `row/PERF-27B-BF16-FP8-OUT` /
[#339](https://github.com/mudler/vllm.cpp/issues/339) (makes the fp8 GDN
projection able to emit bf16 — the *other* half of the unlock, already
implemented there, deliberately NOT reimplemented here).

## 0. Scope

Make the already-vendored upstream GDN packed-decode kernel **eligible** on an
fp8 GDN tower, behind a default-OFF toggle, so the operator can A/B it. The
deliverable is the **eligibility predicate**, not a kernel.

In scope:

- Replace the blanket "the GDN weights are fp8, therefore no packed decode"
  exclusion at `qwen3_5.cpp:4096` with a rule keyed on the **actual dtypes** the
  packed op requires, mirroring upstream.
- A pure, CPU-testable decision surface for that rule.
- A default-OFF env toggle so the relaxation is a same-binary A/B and the
  production default is byte-identical to `main`.

Out of scope, explicitly:

- Making the fp8 GDN GEMM emit bf16. That is `row/PERF-27B-BF16-FP8-OUT`
  (`VT_BF16_GEMM_OUT_FP8` + `VT_FP8_REQUANT_MAX_SCALE`), already implemented and
  measured there. This row is designed to *compose* with it, not duplicate it.
- Regenerating the Triton AOT cubin. That is a maintainer/toolchain task
  (`cmake/TritonAOTKernels.cmake`) and it needs its own row. §6 records exactly
  what a regen would have to change and when it is needed.
- Any measurement. CPU only; no GPU, no benchmark. The operator owns the A/B.

## 1. The measurement that motivates it

Identical batch-1 27B decode workload, both arms traced (#362/#365):

| | ours | vLLM |
|---|---|---|
| kernel | `vt::cuda::GdnDecodeFusedKernel` | `fused_recurrent_gated_delta_rule_packed_decode_kernel` |
| ms/step | **1.3109** | **0.9084** |
| calls/step | 48.000 | 47.812 |

**+0.403 ms/step, 9% of the measured +4.40 ms/step total gap.**

The name on our side is the tell. `GdnDecodeFusedKernel` is the **decomposed**
recurrence — the path taken when packed decode is *not* selected.
`GdnPackedDecodeKernel` / the vendored `gdn_decode_h48` cubin never appear. We
already ship upstream's exact kernel
(`src/vt/cuda/triton_aot_vendored/sm_121a/gdn_decode_h48.{h,cu}`, generated
verbatim from the upstream Triton kernel at the exact 27B geometry H=16, HV=48,
K=128, V=128, BK=128, BV=32) and it is default-ON via
`VT_GDN_PACKED_DECODE_TRITON` — and on this checkpoint it can never fire.

## 2. Blocker A — the model-level exclusion (this row)

`src/vllm/model_executor/models/qwen3_5.cpp:4090-4099` builds the
`dtype_compatible` term of `detail::GdnPackedDecodeEligibility`:

```cpp
indt == DType::kBF16 && outdt == DType::kBF16 &&
    MergedGdnBaOutputDType(true) == DType::kBF16 &&
    // An FP8 GDN tower feeds fp8 projections that vt::GdnPackedDecode
    // rejects ("mixed_qkv/a/b/out must share FP16/BF16/F32 dtype").
    w.in_proj_qkv_fp8.Empty() && w.in_proj_z_fp8.Empty() &&
    (state.ssm_state.dtype == ...)
```

`nvidia/Qwen3.6-27B-NVFP4` is `modelopt_mixed`; its GDN `in_proj_qkv` **is**
`F8_E4M3`. So the 27B is excluded **by construction**, before any dtype is
looked at.

### 2.1 The stated reason is real, but it is a proxy

`vt::GdnPackedDecode` really does require uniformity — `src/vt/ops.cpp:2043`:

```cpp
VT_CHECK(IsFloat(mixed_qkv.dtype) && a.dtype == mixed_qkv.dtype &&
             b.dtype == mixed_qkv.dtype && out.dtype == mixed_qkv.dtype,
         "gdn_packed_decode: mixed_qkv/a/b/out must share FP16/BF16/F32 dtype");
```

and the fp8 GDN in_proj emits **f32** (`qwen3_5.cpp:3408-3409` hardcode
`DType::kF32`; `MergedFp8QkvzD` allocates f32) while `a`/`b` come from
`MergedGdnBaOutputDType(packed=true)` = **bf16** (`:3094-3102`, the bf16
`in_proj_ba` weight, unaffected by the fp8 tower) and `out` is `dcore` at
`outdt` = **bf16**. f32 vs bf16 → the `VT_CHECK` would throw.

So the exclusion is **correct today** and **wrong in general**: it keys on the
*weight* storage format when the thing that actually matters is the *activation*
dtype the projection writes. The moment the fp8 GEMM emits bf16 — exactly what
`row/PERF-27B-BF16-FP8-OUT` does — the uniformity rule is satisfied and the
exclusion is pure loss.

### 2.2 Upstream has no such coupling

FLA's `fused_recurrent_gated_delta_rule` requires only last-dim contiguity per
tensor and casts each operand to f32 on load
(`vllm/model_executor/layers/fla/ops/fused_recurrent.py:256-336` @ `702f4814`,
the revision the cubin was vendored from). Nothing upstream conditions the
packed decode on how `in_proj_qkvz` is quantized; the quantization method is a
`LinearMethodBase` detail that has already produced an activation by the time
`_forward_core_decode_non_spec` (`qwen_gdn_linear_attn.py:1644-1695`) runs.
Mirroring upstream means gating on the produced dtype.

## 3. Blocker B — the AOT bake pins an f32 SSM state (verify, do not fix here)

`TryTritonPackedDecode` (`src/vt/cuda/cuda_gdn.cu:5198-5203`) requires
`mixed_qkv/out/a/b` bf16 **and** `state.dtype == kF32` **and**
`a_log.dtype == kF32`. Those are the baked constexpr dtypes of the vendored
cubin, not a policy choice.

Production GDN state is **bf16** (`qwen3_5_common.cpp:52-54`: `conv_dtype` is
`kBF16` and `ResolveMambaSsmCacheDType` returns `conv_dtype` for
`auto`/absent). The 2026-07-06 ledger entry that made the state bf16 did so to
mirror vLLM's `mamba_cache_dtype=auto` → bf16 and it bought +4.5% throughput, so
bf16 state is not a defect to be reverted.

The task brief asserted the f32-state requirement is liftable at runtime via the
server's `--mamba-ssm-cache-dtype float32`. **That is verified FALSE for our
server and TRUE by two other routes** — see §7.

Blocker B is therefore *out of this row's scope but decisive for whether the
operator's A/B can reach the Triton cubin at all*. Both outcomes are recorded
in §6 so the operator can choose.

## 4. Design

Three pure additions in `vllm::detail`, declared in `qwen3_5_internal.h` and
defined in `qwen3_5.cpp` next to the sibling predicates. All are CPU-testable;
none of them touches a kernel.

### 4.1 `GdnProjectedMixedQkvDType` — predict what the projection will emit

Packed-decode selection happens at `:4083`, **before** `ProjectGdnQkvz` runs at
`:4133`, because the decision changes how BA is projected (`ProjectGdnBA(...,
packed_decode)`). So the eligibility cannot observe `mixed.dtype`; it must
predict it. The helper mirrors `ProjectGdnQkvz`'s branch order exactly:

```
has_bf16_qkvz_owner ? in_dtype           // packed bf16 owner (27B bf16, 4B)
: has_fp8_qkv_owner ? fp8_out_dtype      // native-fp8 owner (modelopt_mixed)
                    : in_dtype;          // split bf16 owner (GGUF/synthetic)
```

`fp8_out_dtype` is sourced from ONE place — a new file-local
`GdnFp8MixedQkvDType()` that `ProjectGdnQkvz` itself also reads in place of its
two hardcoded `DType::kF32` literals. That is what makes the prediction unable
to drift from the projection, and it is the single line
`row/PERF-27B-BF16-FP8-OUT` (or its successor) has to change for the two rows to
compose. On this base SHA it returns `kF32`, unconditionally.

### 4.2 `GdnPackedDecodeDTypesCompatible` — the honest rule

```
mixed_qkv == BF16 && ba_out == mixed_qkv && core_out == mixed_qkv
  && ssm_state ∈ {F32, F16, BF16}
```

This is the op's own uniformity contract (`ops.cpp:2043-2047`) plus the BF16 pin
the model's packed leg already carried, and **nothing about weight storage**.
The SSM state stays an independent dtype, exactly as upstream treats it.

### 4.3 `PackedGdnDecodeFp8TowerEnabled` — the toggle

`VT_GDN_PACKED_DECODE_FP8_TOWER`, **DEFAULT OFF**, ON only when the value's
first character is `'1'` — the house default-OFF convention
(`GdnPackedRegTileFlagIsOn`, `src/vt/cuda/gdn_packed_reg_tile.h`).

Call site term:

```cpp
GdnPackedDecodeDTypesCompatible({mixed_dt, ba_out, outdt, state.ssm_state.dtype})
    && (!gdn_fp8_tower || PackedGdnDecodeFp8TowerEnabled())
```

`gdn_fp8_tower := !w.in_proj_qkv_fp8.Empty() || !w.in_proj_z_fp8.Empty()`.

- **OFF (default, production):** the fp8 clause reproduces the old exclusion
  term exactly → selection is byte-identical to `main` on every checkpoint.
- **ON:** the fp8-weight term disappears and eligibility is decided **purely by
  dtypes**. On this base SHA the fp8 projection still emits f32, so an fp8 tower
  is *still* correctly excluded — by the honest rule instead of by construction.
  Compose with `VT_BF16_GEMM_OUT_FP8` and it becomes eligible.

Why keep a toggle at all when the dtype rule is already safe: it is the
same-binary rollback of the *predicate semantics*, and it means the operator can
attribute a regression to this change rather than to the dtype lever it is
stacked with.

### 4.4 What this deliberately does NOT change

- `ShouldUsePackedGdnDecode` itself. It already takes a single
  `dtype_compatible` bool and never saw the weights; the fp8 term lived at the
  call site. Its signature and every other term are untouched.
- `PackedGdnDecodeEnvSelected` (the 27B gate's env truth table). The new toggle
  can only *permit*, never *deselect*: on a non-fp8 checkpoint the clause is
  `(!false || …)` = true. The existing truth table stays correct and is
  re-asserted unchanged.
- Any default, any kernel, any dtype actually produced at run time.

## 5. Tests (RED first) — decisions, not numbers

The Triton path is CUDA-only, so **the CPU tier pins DECISIONS and cannot pin a
single number**. Every test below asserts which branch the model *would* take.
No test here proves the cubin runs, is correct, or is faster.

`tests/vllm/models/test_qwen27_paged_forward.cpp` (already the home of the
`ShouldUsePackedGdnDecode` and `PackedGdnDecodeEnvSelected` truth tables):

1. `GdnProjectedMixedQkvDType` mirrors `ProjectGdnQkvz`'s branch order — bf16
   owner → `in_dtype`; fp8 owner → `fp8_out_dtype`; split owner → `in_dtype`;
   and the bf16 owner **wins** when both are populated (the order is what makes
   the prediction faithful).
2. `GdnPackedDecodeDTypesCompatible` accepts all-bf16 + f32/f16/bf16 state, and
   rejects each single-field deviation: f32 `mixed_qkv` (today's fp8 tower),
   f32 `ba_out`, f32 `core_out`, an integer state dtype.
3. **The row's whole point:** with a bf16 `mixed_qkv` and an f32 state the path
   is SELECTED, and it is selected *identically* whether the weights are fp8 or
   bf16 — the predicate does not key on fp8 per se. With an f32 `mixed_qkv` it
   is NOT selected, again identically for both weight formats.
4. The toggle: default OFF (`nullptr`, `"0"`, `"2"`, `""` → false), ON only for
   a `'1'`-leading value; and the composed call-site clause reproduces the
   legacy exclusion when OFF.
5. `PackedGdnDecodeEnvSelected`'s existing truth table is unchanged.

RED before: (1)/(2)/(3)/(4) do not compile against `main` (the helpers do not
exist), which is the strongest available RED for a new decision surface; the
*behavioural* RED is (3), which fails on any implementation that keeps a
weight-format term. A changed doctest ASSERTION COUNT is RED even when it prints
"passed" — before/after counts are reported.

## 6. Stop conditions and the honest negative outcome

If the f32-state bake makes the packed path unreachable in every
production-realistic configuration, **that is the finding** and it is recorded
rather than worked around by forcing a config nobody would run.

What an AOT regen would need, if it comes to that: re-bake
`gdn_decode_h48` from `triton_kernels/fused_recurrent_packed_decode.py` with the
`h0`/`ht` pointers typed `bf16` instead of `f32`
(`cmake/TritonAOTKernels.cmake`), then widen the `state.dtype == kF32` guard at
`cuda_gdn.cu:5201` to accept the baked variant. That is a toolchain task with
its own correctness gate and it is NOT in this row.

If relaxing Blocker A had required changing a shared seam's contract —
`vt::GdnPackedDecode`'s dtype rule, or `ShouldUsePackedGdnDecode`'s signature —
this row returns `NEEDS_DECISION` instead of widening it. It does not: the
change is confined to how the call site *computes* an existing bool.

## 7. Blocker B, verified

`--mamba-ssm-cache-dtype float32` is a **vLLM `serve` flag**, not ours. It
appears in this tree only in the vLLM arm of the A/B harness
(`scripts/dgx-online-serving.sh:476`, the `else` branch that runs
`vllm serve`) and in `tools/bench/profile_vllm_online_gate.py:165`, which passes
it into `LLM(...)`. The `ours` arm (`:440-450`) passes `--model/--port/
--num-blocks/--max-num-seqs/--max-num-batched-tokens/--no-enable-prefix-caching/
--served-model-name` and nothing else, and no such flag exists anywhere in
`src/`. **So the brief's stated route is FALSE for our server.**

The underlying claim — the SSM state dtype is controllable at run time without a
code change — is nonetheless **TRUE**, by two other routes:

1. `VT_GDN_STATE_BF16=0` (`qwen3_5_common.cpp:55-63`) forces **both** the conv
   and SSM caches to F32. Process env, no code change, no server flag. This is
   the operator's route.
2. `config.json`'s `mamba_ssm_dtype: "float32"` →
   `detail::ResolveMambaSsmCacheDType` (`qwen3_5.cpp:410-420`) → `kF32`. A
   checkpoint property, not a runtime one, so it is only a route if the
   checkpoint already carries it.

Route 1 is a *diagnostic* rollback: the ledger
(`.agents/specs/benchmark-equivalence-audit-2026-07-15.md:67,94`) records that
f32 SSM stays the mirror-correct default only on CPU and that
`VT_GDN_STATE_BF16` "must never" be used to dress a benchmark. Forcing f32 state
costs the +4.5% the bf16 state bought. **An A/B that turns it on is measuring a
handicapped baseline unless BOTH arms carry it.** That is the operator's call
and it is stated here so the call is made knowingly.

## 8. Gates

CPU tier only, on this worktree. `-DVLLM_CPP_TRITON` is not set and no GPU is
touched, so the cubin is not linked and none of these tests observe it.

- `scripts/agent-preflight.sh --staged`
- Focused: `tests/test_qwen27_paged_forward` — assertion counts reported
  before/after.
- Box load is ~87 at claim time; if the full `ctest` cannot run clean, exactly
  what was and was not run is reported rather than implied.

Public documents owed: `docs/FEATURES.md` only. `check-doc-checkpoint`
classifies any edit under `src/vllm/model_executor/models/` as a
`feature_surface` change, so the fp8-weights row records the new toggle. No
lifecycle state moves, so `STATUS`/`BENCHMARKS`/`NOW` are not owed — and no
measurement exists to put in `BENCHMARKS` anyway.

## 9. Evidence

Recorded in the commit body and the row report: base SHA, spec SHA,
implementation SHA, `file:line` of every edit, RED-before/GREEN-after with
assertion counts, and the exact runtime configuration the operator needs.

## Now

Spec committed; implementation is the eligibility predicate + its CPU decision
tests. Next step after landing: the operator's same-binary A/B with
`VT_GDN_PACKED_DECODE_FP8_TOWER=1` stacked on the fp8-bf16-out lever, on a GPU.
