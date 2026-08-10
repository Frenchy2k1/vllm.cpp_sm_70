# PERF-27B-GDN-FP8-QKVZ — merge the GDN FP8 input projections

Issue: [#213](https://github.com/mudler/vllm.cpp/issues/213)
Row: `PERF-27B-GDN-FP8-QKVZ`
Gate model: `nvidia/Qwen3.6-27B-NVFP4` @`0893e1606ff3d5f97a441f405d5fc541a6bdf404`
Also applies to: `nvidia/Qwen3.6-35B-A3B-NVFP4` @`491c2f1e` (same FP8 tower)

## Scope

Extend the existing merged-qkvz GDN input projection to an **FP8 W8A8 owner**, so
the 27B/35B ModelOpt tower issues ONE merged GEMM per layer instead of two.

**In scope:** the FP8 arm of `ProjectGdnQkvz` / `MergedGdnQkvzEnabled`, its loader
owner, and the dispatch predicate that currently excludes FP8.

**Out of scope:** the BF16 leaf (already shipped), `in_proj_b`/`in_proj_a`, the
attention `qkv` merge (`MergedFp8QkvEnabled`, a separate default-OFF row), the
`lm_head` row (`PERF-27B-LMHEAD-FP4`), GGUF and synthetic split owners.

## The gap, MEASURED

Decode-only `nsys` two-length diff (8 vs 136 tokens), `--cuda-graph-trace=node`,
**both arms same tool**, idle box, node-level tracing proven by integral launch
counts. Two capture pairs agree to 0.005%.

Both arms read **identical** FP8 bytes — 7,214,202,880 B = 6.7188 GiB/step across
208 `F8_E4M3` tensors. This is not a traffic difference and not a kernel-quality
difference; our Marlin and cuBLASLt kernels reach the same GiB/s as vLLM's on the
shapes we share. It is **shape**.

| | ours | vLLM | delta |
|---|---|---|---:|
| GDN input projections | 25.433 ms, **96** GEMMs, `in_proj_qkv` at **129.3 GiB/s** | 17.555 ms, **48** merged qkvz at **213.6 GiB/s** | **+7.88 ms/step** |
| GDN `out_proj` + attn `o_proj` | 9.751 ms, 192.3 GiB/s | 10.165 ms, 184.5 GiB/s | −0.41 |
| attn q,k,v | 5.352 ms | 5.159 ms, merged | +0.19 |

Our `in_proj_qkv` (10240×5120) runs at **60% of the 213.6 GiB/s the same tower
reaches on the merged shape**. Issued at that rate it would cost 10.97 ms instead
of 18.12 ms. The whole 27B deficit is 17.3292 ms/step and closes to four decimal
places as `lm_head` 8.6414 + FP8 tower 7.6068 + splitK 0.0532 + other 1.0279.

## Why the capability exists but cannot be reached

`.agents/specs/gdn-merged-input-projections.md` shipped the merged path and
**deliberately excluded FP8**:

- `:54` — "35B native | existing FP8 qkv/z | … Inert in W1/W2; qkv/z have quant
  scales and belong to `KERNEL-GEMM-FP8`."
- `:336` — "**35B qkv/z is FP8.** It stays outside this BF16 leaf."

`MergedGdnQkvzEnabled` additionally requires `GdnInDType() == GdnOutDType()`
(`qwen3_5.cpp:2372-2460`, seam `qwen3_5_internal.h:52-60`), which an FP8 owner
does not satisfy. So the exclusion was a scoping decision, correct at the time,
and this row is its follow-up — not a bug fix.

## Upstream anchor

vLLM merges qkvz for this model and runs it as ONE GEMM per layer. Measured on
the oracle's own trace:

```
17.555 ms/step, 48 launches
cudnn_generated_fort_native_sm120_matMul_pointwise_pointwise_knob_20_32x32x128_0_0
  — GDN in_proj_qkvz, MERGED
```

Its `FlashInferFP8ScaledMMLinearKernel` is *selected* at config time but on
`sm_121` dispatches into cuDNN; vLLM also uses cuBLASLt for the other 64 fp8
projections. Read #252 before assuming a library choice matters here — it does
not; the merge does. The upstream structure to mirror is
`MergedColumnParallelLinear`-style packing of qkv+z into one column-parallel
weight, exactly as the BF16 leaf already mirrors it.

## Design

1. Build a merged FP8 owner at LOAD: qkv and z concatenated along N, mirroring the
   BF16 leaf's owner rather than inventing a second layout.
2. **Scales must be provably compatible.** ModelOpt FP8 here is per-tensor. If
   `in_proj_qkv.input_scale != in_proj_z.input_scale` bitwise, or the weight
   scales cannot be folded without changing arithmetic, the merge MUST NOT fire —
   fall back to the legacy two-GEMM path. Check this at load, once, not per step.
   `Fp8SharedInputScale` (`qwen3_5.cpp:5472-5486`) already expresses this predicate
   for a sibling case; reuse it rather than writing a second one.
3. Relax the dispatch predicate to admit an FP8 owner, keeping every other guard.
   Do NOT relax `GdnInDType()==GdnOutDType()` for the BF16 path.
4. Gate behind `VT_GDN_MERGED_QKVZ_FP8` (default ON once green) so the A/B is
   same-binary and there is an in-binary rollback.
5. `mixed_qkv`/`z` remain last-dim views of the merged output, as in the BF16 leaf.

## Risks

- **Scale folding changes numerics.** The gate is token-exactness against the
  pinned oracle. If folding is not bit-preserving, the row fails — do not trade it.
- **The 35B shares this tower.** A change here moves both gate models; run both.
- **Graph capture.** Build the merged resident pre-capture, as
  `PERF-27B-LMHEAD-FP4` had to; a resident built inside capture bakes an address.
- **Interaction with `PERF-27B-LMHEAD-FP4`.** Independent masses (lm_head is
  Marlin/BF16, this is the FP8 tower), so they add — but measure the combination,
  do not assume 8.1 + 7.9.

## Tests

RED first:
1. Launch-count assertion: GDN fp8 input projections per decode step == 48 (one
   per layer), red at 96.
2. Numerical: merged output byte-identical to the concatenation of the two legacy
   GEMM outputs on the gate shapes.
3. Predicate: with deliberately mismatched `input_scale`s, the merge does NOT fire
   and the legacy path runs.
4. `VT_GDN_MERGED_QKVZ_FP8=0` reproduces the legacy path exactly.

Port the applicable cases from the BF16 leaf's suite rather than writing new ones
where they transfer.

## Gates

- Focused: the tests above; `test_qwen27_paged_engine` 235/235 and
  `test_qwen36_paged_engine` 315/315 unchanged.
- Full: CUDA `ctest -j 1` on `sm_121a` (`-j 4` OOM-reboots the box).
- Correctness: greedy continuation on `nvidia`@`0893e160` vs the pinned oracle.
  Note the oracle's own greedy is undetermined at ~8/32 positions on the synthetic
  corpus (top-2 margin exactly 0.000000), so a text difference alone is NOT a
  failure — obtain the oracle's top-k margins and judge against the ratified
  distributional gate.
- Speed: same-binary `VT_GDN_MERGED_QKVZ_FP8=0|1` A/B, 3 reps, medians,
  order-alternated, idle box, band measured first. Expected ~7.9 ms/step of ~98.9,
  far outside the ±0.03%–1.85% band. Confirm via `nsys --cuda-graph-trace=node`
  that the fp8 input-projection launches fall 96 → 48.

## Evidence

`dgx:~/work/vllm.cpp-online-gate/evidence/<sha>/gdn-fp8-qkvz/` — A/B legs, both
`nsys` reports with launch counts, continuation transcripts, oracle margins.

## Stop conditions

- Stop and report `NEEDS_DECISION` if the scales cannot be folded bit-exactly.
- Do not widen into the attention `qkv` merge or `lm_head`.
- Do not relax the BF16 leaf's predicate.

## Outcome

Pending.
