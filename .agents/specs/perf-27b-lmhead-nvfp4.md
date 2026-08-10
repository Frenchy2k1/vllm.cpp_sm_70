# PERF-27B-LMHEAD-FP4 — keep the ModelOpt NVFP4 `lm_head` packed

Issue: [#213](https://github.com/mudler/vllm.cpp/issues/213)
Row: `PERF-27B-LMHEAD-FP4`
Gate model: `nvidia/Qwen3.6-27B-NVFP4` @`0893e1606ff3d5f97a441f405d5fc541a6bdf404`
Base: `origin/main` @`04069bd7`; the review-findings round is rebased onto
`origin/main` @`723d96a8`.

## Scope

The dense Qwen3.6 loader dequantizes a ModelOpt NVFP4 `lm_head` into a BF16
`[in,out]` operand at load. Keep it packed and run the logits GEMM on the same
Marlin W4A16 family vLLM pins for this head.

**In scope:** the dense `lm_head` weight path only — loader, weight struct,
resident staging, and the two `DenseLmHead` consumers (gather and non-gather),
the eager `ForwardLogits` arm, and the dense MTP sibling.

**Out of scope:** the FP8 tower output dtype (`PERF-27B-FP8-BF16-OUT`), the
gate_up merge (`PERF-27B-DENSE-GATEUP-MERGE`), the embedding table, the MoE
path's head, and anything on the `unsloth` repos.

## The gap, verified against current code

`LoadLmHeadAnyDtype`'s `U8` branch
(`src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:266-306`) runs
`DequantCtNvfp4WeightToF32` into a full f32 array, rounds to BF16, then
`TransposeBf16` into an `OwnedTensor {in_dim, out_dim}`. The comment at
`:205-211` records this as deliberate deferral:

> Keeping the head quantized end-to-end would save ~2.3 GiB but needs an
> `lm_head_fp4`-style field on the dense weights plus a forward branch; that is
> a follow-up, not this fix.

Consequence at decode: the logits GEMM re-reads ~2.543 GB of BF16 every step
where the packed head is ~0.715 GB (`K*N/2` packed + `K*N/16` F8_E4M3 block
scales). Storing the operand transposed to `[K,N]` additionally forces a
row-major NN GEMM, for which no `nvjet_sm121` kernel exists, so cuBLASLt falls
back to the legacy `cutlass_80_tensorop_s16816gemm_bf16_128x64_32x6_nn_align2` —
an Ampere tile on an `sm_121a` part.

## Upstream anchors

- `vllm/model_executor/layers/quantization/modelopt.py:2508-2536` —
  `ModelOptMixedPrecisionConfig.get_quant_method` accepts `ParallelLMHead` and
  returns `ModelOptNvFp4LinearMethod` for `quant_algo == "NVFP4"`.
- `modelopt.py:2491-2496` — `_quantized_layer_prefix_candidates` appends the
  bare `lm_head` key, so the mixed scheme is *designed* to resolve a quantized
  head.
- `modelopt.py:1249,1283-1284` — `ModelOptNvFp4W4A16LinearMethod` pins
  `MarlinNvFp4LinearKernel`, explicitly because the generic priority list would
  otherwise first-pick a W4A4 cutlass kernel on this hardware.
- `modelopt.py:1365` — vLLM **deletes** `input_scale` on the W4A16 path
  (`process_weights_after_loading`); the placeholder is registered at
  `modelopt.py:1358`. Verified against the pinned oracle `555967922`.
- `vllm/model_executor/layers/logits_processor.py:98-133` — `_apply_head` calls
  `lm_head.quant_method.apply` every step; nothing materializes BF16.
- `marlin_utils_fp4.py:157-218,221-306` — `prepare_fp4_layer_for_marlin` /
  `apply_fp4_marlin_linear`, the repack and apply we already vendored.

## Design

1. Add `Nvfp4Weight lm_head_fp4` to the dense weight struct
   (`include/vllm/model_executor/models/qwen3_5_dense.h`).
2. In the loader's `U8` branch, stop dequantizing: route through the existing
   `IsNvfp4Projection` / `LoadNvfp4AnyNaming` path into `lm_head_fp4`, keeping
   the ModelOpt `weight_scale_2`-as-scale convention already handled at
   `:274-288`.

   **Deviation, ratified in review:** the design said to retire the stale
   "lm_head is never quantized" rule in `IsQwen27QuantizedLinear`
   (`qwen3_5_dense_weights.cpp:527` today, not `:506`). It was deliberately NOT
   retired: the function has ZERO production callers — only its own routing test
   in `test_qwen27_dense_forward.cpp` — so changing it would move no behavior
   while invalidating a checked-in expectation. The routing that matters lives in
   `LoadDenseLmHead`.
3. A `DenseLogitsF32D` helper gains the packed branch so the tied and `nk`
   cases keep one code path. All three consumers route through it: the gathered
   and non-gathered paged arms and the eager `ForwardDense` arm.
4. Build the head's resident **pre-capture**, from the registry `prepare` hook
   (`Qwen3_5DenseModel::PrepareLmHeadResident`), mirroring the MoE
   `PrepareMarlinResident`. `BuildMarlinDenseResident` copies a function-local
   host float into the device buffer; built lazily inside a captured region, the
   copy source does not outlive the capture. On a backend with NO fp4 GEMM the
   same hook builds the dequantized bf16 operand instead, so the fallback never
   dequantizes per forward call.
5. `PrepareBf16Resident` needs no head-specific branch: a packed head's bf16
   owner is empty by construction, and the function is only reached under
   `IsPlainBf16Qwen3_5Dense`, which is false whenever the head is packed. **The
   RSS win comes from the LOADER no longer building the f32 + bf16 arrays**, not
   from anything skipped at staging time.
6. The dense MTP ctor (`:6684-6686`) has no `lm_head_fp4_` sibling; add it.

Gate the whole thing behind `VT_LMHEAD_FP4` (default ON once green) so the A/B
is same-binary and there is an in-binary rollback.

**BF16 and FP8 heads are untouched.** Those branches keep their current bytes,
so every recorded `unsloth` benchmark is unaffected.

## Risks

- **Graph hang, not fault.** Marlin's fp32 reduce spins on lock words; a
  workspace that is not zero at allocation hangs forever. Zero at alloc.
- **`IsTrueW4A4()` flip.** This checkpoint ships `lm_head.input_scale`.
  Consuming it would select the W4A4 GEMM that vLLM explicitly refuses
  (`modelopt.py:1365`) AND make `PrepareLmHeadResident` early-return, silently
  skipping the pre-capture build. The rule holds under BOTH spellings:
  `LoadCtNvfp4Raw` consumes `input_global_scale` unconditionally (correct for a
  27B TOWER projection, wrong for an output head), so `LoadDenseLmHead` drops the
  activation globals unless `VT_MODELOPT_W4A4=1`. Assert
  `lm_head_fp4.IsTrueW4A4() == false` for both namings.
- **A compressed-tensors head read as tied.** The model loader's head probe must
  accept `<proj>.weight_packed` as well as `<proj>.weight`
  (`DenseCheckpointHasLmHead`); a bare `.weight` probe reads a CT head as
  `tie_word_embeddings` and computes the logits off the embedding table.
- **Gate blindness.** `test_qwen27_paged_engine` (235/235) runs `unsloth`
  @`890bdef7`, which ships a **BF16** head — that gate cannot see this path. A
  fresh greedy continuation on `nvidia`@`0893e160` against the pinned oracle is
  mandatory, not optional.

## Tests

RED first, `tests/parity/test_qwen27_dense_lmhead_fp4.cpp`:

1. Synthetic `modelopt_mixed` fixture: `U8 lm_head.weight` + `F8_E4M3
   lm_head.weight_scale` + f32 `lm_head.weight_scale_2`. Assert
   `lm_head_fp4.Empty() == false`, the BF16 owner is absent, and the resident
   byte count is `K*N/2 + K*N/16`. Red today: the field does not exist.
2. Numerical: logits from the packed head match a reference dequant-then-GEMM
   within the Marlin W4A16 tolerance already used by the 32B-NVFP4A16 op tests.
3. Assert `IsTrueW4A4() == false` for the loaded head.

Added in the review-findings round, each pinned by a mutation that turns it RED:

4. The same `IsTrueW4A4() == false` assertion under compressed-tensors names
   (`weight_packed` / `weight_global_scale` / `input_global_scale`), and
   `DenseCheckpointHasLmHead` accepting the CT spelling.
5. Both PAGED lm_head call sites — the gathered (prefill/mixed) and the
   non-gathered arm — against the eager reference. The numerical case above runs
   only `ForwardDense`, so reverting either paged arm to the bf16 owner was
   invisible.
6. The fallback dequant is built ONCE (pointer identity across two forwards),
   not per call, and the registry `prepare` hook builds it before any forward.

Port anchor: `marlin_utils_fp4.py` tolerances and shapes as used by the existing
NVFP4A16 op tests.

## Gates

- Focused: the new test, plus `test_qwen27_paged_engine` 235/235 unchanged.
- Full: CUDA `ctest` on `sm_121a`, clean Release build with
  `-DVLLM_CPP_CUTLASS_DIR` and `-DVLLM_CPP_TRITON=ON`.
- Correctness: greedy continuation on `nvidia`@`0893e160`, 32 tokens,
  `ignore_eos`, captured from the same warm process as the throughput numbers,
  compared against the pinned oracle. Token-exact, or a ratified near-tie with
  the oracle's own top-2 margin recorded.
- Speed: `VT_LMHEAD_FP4=0|1` same-binary A/B, one `flock $HOME/gpu.lock`, warm
  server, 3 reps per leg, order-alternated, medians of per-rep medians, at
  c1/c2/c4/c8. Expected effect is far above the 0.5% noise band, so e2e
  throughput resolves it; report `nsys --cuda-graph-trace=node` instance counts
  for the logits kernel on both legs as the invocation-parity evidence.
- Memory: peak host RSS on both legs. Expected delta **1.70 GiB**: the bf16
  head is `2*K*N` = 2,543,206,400 B = 2.368 GiB and the packed head is
  `K*N/2 + K*N/16` = 715,264,000 B = 0.666 GiB, at the real 248320x5120.

## Evidence

`dgx:~/work/vllm.cpp-online-gate/evidence/<sha>/lmhead-fp4/` — raw A/B legs,
both `nsys` reports, the continuation transcripts from both engines, RSS
samples, and the build recipe.

## Stop conditions

- Stop and report `NEEDS_DECISION` if the packed head cannot be made
  token-exact-or-ratified against the oracle.
- Stop if the head's `weight_scale_2` convention does not match the ModelOpt
  reading already used at `:274-288`.
- Do not widen scope into the FP8 tower, the gate_up merge, or the MoE head.

## Outcome

Pending.
