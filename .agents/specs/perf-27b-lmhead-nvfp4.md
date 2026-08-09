# PERF-27B-LMHEAD-FP4 — keep the ModelOpt NVFP4 `lm_head` packed

Issue: [#213](https://github.com/mudler/vllm.cpp/issues/213)
Row: `PERF-27B-LMHEAD-FP4`
Gate model: `nvidia/Qwen3.6-27B-NVFP4` @`0893e1606ff3d5f97a441f405d5fc541a6bdf404`
Base: `origin/main` @`04069bd7`

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
- `modelopt.py:1359-1362` — vLLM **deletes** `input_scale` on the W4A16 path.
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
   `:274-288`. Retire the stale "lm_head is never quantized" rule at `:506`.
3. `DenseLmHead` (`qwen3_5.cpp:1248-1250`) gains the packed branch so the tied
   and `nk` cases keep one code path. Both consumers must route through it:
   `:7024-7026` (gather) and `:7028-7030` (non-gather), plus the eager
   `ForwardLogits` arm at `:6671-6674`.
4. Build the Marlin resident **pre-capture**, mirroring `:6424-6425`. A resident
   built inside capture bakes a stack address and fails on replay.
5. Stop staging the BF16 head owner in `PrepareBf16Resident` (`:6456`), or the
   ~2.3 GiB RSS win does not materialize.
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
  (`modelopt.py:1359-1362`). Keep `VT_MODELOPT_W4A4=0` and assert
  `lm_head_fp4.IsTrueW4A4() == false`.
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
- Memory: peak host RSS on both legs; expect ~2.3 GiB lower with the head packed.

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
