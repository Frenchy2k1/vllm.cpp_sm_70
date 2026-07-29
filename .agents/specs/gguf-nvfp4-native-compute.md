# Native NVFP4 compute from a GGUF container (`QUANT-GGUF-NVFP4`, column `C`)

Companion to [gguf-nvfp4-notes.md](gguf-nvfp4-notes.md), which owns the `R`/`M`
columns (read + materialize) and explicitly scoped `C` OUT. This spike owns `C`
alone: making an NVFP4 GGUF weight COMPUTE in fp4 instead of being expanded to
bf16 at load.

**Headline finding, measured before any code was written (Sec A below): the two
27B NVFP4 containers are NOT the same model.** The GGUF NVFP4-quantizes 192
tensors the safetensors keeps in BF16, and its activation global scales differ.
So the frequently-cited "the two containers diverge at token index 4" is NOT
(only) a compute-precision artifact and CANNOT be closed by this row. The gate
is therefore weight-level and op-level exactness, not cross-container token
identity. See Sec A.

## Spike record

| Field | Content |
|---|---|
| Scope | `QUANT-GGUF-NVFP4` column `C` ONLY: a residency that keeps ggml type-40 weights in fp4 and dispatches the EXISTING NVFP4 GEMMs on them. IN: the residency decision, the load-time repack into the kernels' operand layout, the sidecar/per-expert scale plumbing into `Nvfp4Weight`, and the Qwen3.6 dense + MoE GGUF loader wiring. OUT: `E`/`P` columns, ANY new fp4 kernel, ANY change to the safetensors NVFP4 path (gated, must stay byte-identical), and the GDN `in_proj_*` family (no fp4 field exists on `GdnLayerWeights` for them and the 27B V-head reorder makes them `kTransformedWeight` anyway) |
| Upstream chain | No vLLM GGUF path exists (deviation §9). The COMPUTE being dispatched is already-ported vLLM: `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py:29-32,95-141` (W4A4 + the `use_a16` mode), `vllm/model_executor/kernels/linear/__init__.py:842,879-892` (kernel selection), `cutlass_scaled_fp4_mm_sm120a`. Our lifts: `vt::MatmulNvfp4` / `kMatmulNvfp4Fp4` / `kMatmulNvfp4Cutlass` (`include/vt/ops.h:129,134-135`, `src/vt/cuda/cuda_matmul_nvfp4*.cu`). The GGUF CONTAINER mirrors the local ggml type-40 fork (`ggml/src/ggml-common.h:211-217`), inventoried in [gguf-nvfp4-notes.md](gguf-nvfp4-notes.md) Sec 1-2 and MEASURED in its Sec 5 |
| Our baseline | `gguf_dequant.cpp:123` decodes type 40 correctly and `qwen3_5_gguf_weights.cpp:330` resolves the `<stem>.scale` sidecar and the per-expert slabs, but the ONLY product is bf16: `RouteGgufTensor` (`gguf_keep_quant.cpp:113`) has no residency that yields fp4, `vt::DType` has no NVFP4 member, and `KeepQuantDType` requires a `vt::cpu::HasQuantDotKernel` block dtype that NVFP4 does not and cannot have. The CONSUMER side is already complete and needs no change: `DenseMlpWeights::{gate,up,down}_proj_fp4` / `FullAttnLayerWeights::{q,k,v,o}_proj_fp4` / `Qwen3_5MoeWeights::expert_*_fp4` exist, and `DenseMlpBlock` (`qwen3_5.cpp:5246`) and the attention block already branch on `!*.Empty()` and on `IsTrueW4A4()` |
| Port map | `gguf_dequant.{h,cpp}`: new `RepackGgufNvfp4Rows` — the pure byte permutation from ggml blocks to the (`weight_packed[N,K/2]` torch-pairwise, `weight_scale[N,K/16]` linear fp8) pair, beside the existing case-40 decoder that owns the same layout facts. `gguf_keep_quant.{h,cpp}`: `GgufResidency::kNvfp4Fp4`, `Nvfp4Fp4DType`, `GgufNvfp4ComputeAvailable()`, `GgufLoadPolicy::nvfp4_fp4` + its `VT_GGUF_NVFP4_FP4` / `VT_GGUF_NVFP4_W4A4` env reads. `qwen3_5_gguf_weights.{h,cpp}`: `OwnGgufNvfp4` (2-D) and `OwnGgufNvfp4Experts` (3-D, one `Nvfp4Weight` per expert slab) building `Nvfp4Weight` from the repack + the `.scale` / `.input_scale` sidecars; `OwnMatmulWeightOrNvfp4` wiring in `LoadAttnGguf`, `LoadQwen3_5DenseFromGguf` and `LoadMoeGguf`. NO change to any `src/vt/` kernel, to `nvfp4_dequant.*`, or to the safetensors loaders |
| Tests to port | Nothing to port: vLLM has no GGUF. The executable spec is again CROSS-FORMAT EQUIVALENCE against the already-gated safetensors container, now at the OPERAND level rather than the value level, using the goldens that already exist (`tests/vllm/gguf_nvfp4_goldens.inc`, which carries both containers' bytes for the same three [4,512] slices). New cases in `tests/vllm/test_gguf_nvfp4.cpp` (repack byte-identity, non-vacuity, guards) and `tests/vllm/test_gguf_keep_quant.cpp` (routing totality for the new residency). Asset-gated real-file sweep + the e2e run stay in the existing `VLLM_NVFP4_GGUF`/`VLLM_NVFP4_ST` and `VLLM_QWEN36_GGUF` cases |
| Gates | (1) **BYTE-IDENTITY**: `RepackGgufNvfp4Rows` output == the safetensors `weight_packed` / `weight_scale` bytes for the same slice, ZERO differing bytes, and the `.scale` sidecar == `float32(1)/float32(weight_global_scale)` bit-for-bit. This makes the fp4 GEMM provably the SAME already-gated kernel on the SAME operand bytes, so no op-level numerics are re-litigated. (2) Non-vacuity: the wrong (torch-pairwise-in, split-half-out) permutation must FAIL gate 1. (3) Routing totality: every tensor still routed exactly once; NVFP4 in a non-verbatim role still expands. (4) The SACRED regression set unmoved: `test_qwen27_paged_engine` 235/235, `test_gguf_nvfp4`, `test_gguf_keep_quant`, `test_nvfp4_dequant`, `test_qwen36_paged_engine`. (5) e2e: the 27B NVFP4 GGUF loads and generates coherently with the fp4 path OBSERVED to have run, at materially lower weight residency. **NOT a gate, and cannot be**: token identity against the safetensors container (Sec A) |
| Dependencies | `vt::MatmulNvfp4` + the cutlass fp4 family (CUDA-only; a CPU build correctly keeps expanding to bf16). `GgufGlobalScales` for the `.scale` sidecar. The 27B/35B assets on dgx.casa |
| Work breakdown | N1 the repack primitive + its byte-identity gate. N2 the residency + policy + routing gate. N3 the dense (27B) loader wiring. N4 the MoE (35B) stacked-expert wiring. N5 the GPU e2e run + residency measurement. N6 the honest record of the cross-container ceiling (Sec A) |
| Risks/decisions | **DECISION: NVFP4 does NOT get a `vt::DType` member.** Every `vt::DType` block encoding is SELF-CONTAINED — a fixed `BlockBytes` record that determines its own values — and is consumed through `HasQuantDotKernel`/`vec_dot`. NVFP4 is not: the value needs a per-tensor (per-EXPERT, on stacked weights) scalar that lives in a different TENSOR, and its GEMM takes three separate operands. Adding `kNVFP4` would make `BlockDTypeFromGgmlTypeId(40)` hand back a dtype with no `BlockToFloat`, no `vec_dot`, and a `RowSizeBytes` that silently lies about what determines the weight. The fp4 residency is therefore a `Nvfp4Weight` (3 buffers + 3 scalars), exactly as the safetensors path already models it. **RISK: a plausible-but-wrong repack.** A nibble permutation preserves the value histogram, so a wrong repack yields finite, sane-looking logits and silently degrades output — the same failure mode the `M` column had. Mitigated the same way: gate on BYTE-IDENTITY against the other container, never a tolerance, and prove the gate rejects the wrong permutation. **RISK: silently NOT taking the fp4 path.** A mis-wired loader that left the bf16 fields populated would pass every correctness gate. Mitigated by asserting the residency decision directly and by requiring the bf16 field to be EMPTY when the fp4 field is filled. **DECISION: reconstructed divisors.** `Nvfp4Weight::weight_global_scale_inv` / `input_global_scale_inv` hold the ON-DISK DIVISORS, which the GGUF does not store — it stores their reciprocals. They are reconstructed as `1.0f/sidecar` and may differ from the safetensors divisor by one ULP; recorded rather than hidden, and irrelevant next to the container deltas in Sec A |

## A. The cross-container ceiling — MEASURED, and it bounds what `C` can claim

The task this row is usually framed by ("the GGUF expands to bf16 while the
safetensors runs true W4A4, and they diverge at token index 4; native fp4 should
close that") assumes the two containers hold the SAME weights. **They do not.**
Measured directly on `~/bench/q36-27b-nvfp4.gguf` and
`~/bench/q36-27b-nvfp4-vllm/model.safetensors`:

| | GGUF | safetensors |
|---|---|---|
| NVFP4 tensors | **496** | **304** |
| `ffn_{gate,up,down}` / `mlp.{gate,up,down}_proj` | 192 NVFP4 | 192 NVFP4 |
| `attn_{q,k,v,output}` / `self_attn.{q,k,v,o}_proj` | 64 NVFP4 | 64 NVFP4 |
| `ssm_out` / `linear_attn.out_proj` | 48 NVFP4 | 48 NVFP4 |
| `attn_qkv` / `linear_attn.in_proj_qkv` | 48 **NVFP4** | 48 **BF16** |
| `attn_gate` / `linear_attn.in_proj_z` | 48 **NVFP4** | 48 **BF16** |
| `ssm_alpha` / `linear_attn.in_proj_a` | 48 **NVFP4** | 48 **BF16** |
| `ssm_beta` / `linear_attn.in_proj_b` | 48 **NVFP4** | 48 **BF16** |

496 - 304 = 192, exactly the GDN `in_proj_{qkv,z,a,b}` family that the
safetensors `recipe.yaml` ignores and the GGUF converter quantized anyway
(already noted qualitatively in [gguf-nvfp4-notes.md](gguf-nvfp4-notes.md) Sec
5.6; here it is quantified). Dequantizing the GGUF's `blk.N.attn_qkv` and
comparing against the safetensors' BF16 `in_proj_qkv` over the leading 64 rows
(q rows, so no V-head reorder is involved):

```
layer 0   max|d| = 1.09e-2   mean relative error = 0.180
layer 1   max|d| = 1.13e-2   mean relative error = 0.181
layer 40  max|d| = 1.25e-2   mean relative error = 0.186
```

That is ordinary 4-bit quantization error, and it is present in the GGUF and
absent from the safetensors. Second delta: the two containers' ACTIVATION global
scales disagree on 4 of 5 sampled projections (`ffn_gate` 808 vs 812, `ffn_up`
344 vs 276, `ffn_down` 5.219 vs 3.547, `attn_output` 724 vs 292; only `attn_q`
matched at 28.75), so even the shared W4A4 projections quantize activations
differently. The WEIGHT global scales, by contrast, are bit-identical on all
five.

**Consequence.** Token-for-token identity between the two containers is
unreachable no matter how the GGUF computes, and any residual divergence after
this row is chargeable to the 192-tensor weight delta and the activation-scale
delta, not to a defect in the fp4 path. The gate this row DOES own is exact and
sharper: the fp4 OPERANDS the GGUF hands the kernels must be byte-identical to
the ones the safetensors hands the SAME kernels, on every projection the two
containers share.

## B. Why this is a repack, not a kernel change

The GGUF block (`d[4]` fp8-e4m3 then `qs[32]`, ggml SPLIT-HALF nibbles) and the
compressed-tensors pair (`weight_packed[N,K/2]` torch-PAIRWISE nibbles,
`weight_scale[N,K/16]` linear fp8) hold the SAME bytes in a different
arrangement. Verified on the real files over 128 rows of five projections
spanning both shapes and both tensor families:

```
blk.0.ffn_gate.weight     K=5120   scales_eq=True packed_eq=True (0 differing bytes) scale2_bitexact=True
blk.7.ffn_up.weight       K=5120   scales_eq=True packed_eq=True (0 differing bytes) scale2_bitexact=True
blk.63.ffn_down.weight    K=17408  scales_eq=True packed_eq=True (0 differing bytes) scale2_bitexact=True
blk.3.attn_q.weight       K=5120   scales_eq=True packed_eq=True (0 differing bytes) scale2_bitexact=True
blk.3.attn_output.weight  K=6144   scales_eq=True packed_eq=True (0 differing bytes) scale2_bitexact=True
```

So the whole of column `C` is: permute at load, fill the `Nvfp4Weight` fields
the forward ALREADY consumes, and let the already-gated kernels run. No kernel
takes a GGUF-shaped operand; nothing in `src/vt/` is touched.
