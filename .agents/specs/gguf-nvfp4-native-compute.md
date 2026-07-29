# Native NVFP4 compute from a GGUF container (`QUANT-GGUF-NVFP4`, column `C`)

Companion to [gguf-nvfp4-notes.md](gguf-nvfp4-notes.md), which owns the `R`/`M`
columns (read + materialize) and explicitly scoped `C` OUT. This spike owns `C`
alone: making an NVFP4 GGUF weight COMPUTE in fp4 instead of being expanded to
bf16 at load.

**Two findings, both measured, and they point in opposite directions — the
honest answer needs both.**

1. The two 27B NVFP4 containers are NOT the same model: the GGUF
   NVFP4-quantizes 192 tensors the safetensors keeps in BF16, and their
   activation global scales differ (Sec A). So cross-container token identity
   cannot be this row's GATE — it is not guaranteed by construction, and a
   prompt where it failed would not indict the fp4 path.
2. And yet **it holds, measured**: once the GGUF computes in fp4, its 24-token
   greedy stream is IDENTICAL to the safetensors container's, where the bf16
   arm of the SAME binary diverges at index 4 (Sec C). So on this prompt the
   recorded divergence really was the compute delta, and closing it closed the
   divergence.

The binding gate is therefore weight-level and op-level exactness (Sec B); the
token result is REPORTED, and is the strongest available end-to-end signal.

**Sec D (2026-07-29) closes the MoE stacked-expert arm** on the real 35B A3B, and
that run also found and fixed a latent router-orientation defect the arm made
reachable. It carries one honest negative of its own: at that case's engine
params the 24-token greedy stream is not run-to-run stable (D.6.1), so the
binding results there are weight-level, not token-level.

## Spike record

| Field | Content |
|---|---|
| Scope | `QUANT-GGUF-NVFP4` column `C` ONLY: a residency that keeps ggml type-40 weights in fp4 and dispatches the EXISTING NVFP4 GEMMs on them. IN: the residency decision, the load-time repack into the kernels' operand layout, the sidecar/per-expert scale plumbing into `Nvfp4Weight`, and the Qwen3.6 dense + MoE GGUF loader wiring. OUT: `E`/`P` columns, ANY new fp4 kernel, ANY change to the safetensors NVFP4 path (gated, must stay byte-identical), and the GDN `in_proj_*` family (no fp4 field exists on `GdnLayerWeights` for them and the 27B V-head reorder makes them `kTransformedWeight` anyway) |
| Upstream chain | No vLLM GGUF path exists (deviation §9). The COMPUTE being dispatched is already-ported vLLM: `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py:29-32,95-141` (W4A4 + the `use_a16` mode), `vllm/model_executor/kernels/linear/__init__.py:842,879-892` (kernel selection), `cutlass_scaled_fp4_mm_sm120a`. Our lifts: `vt::MatmulNvfp4` / `kMatmulNvfp4Fp4` / `kMatmulNvfp4Cutlass` (`include/vt/ops.h:129,134-135`, `src/vt/cuda/cuda_matmul_nvfp4*.cu`). The GGUF CONTAINER mirrors the local ggml type-40 fork (`ggml/src/ggml-common.h:211-217`), inventoried in [gguf-nvfp4-notes.md](gguf-nvfp4-notes.md) Sec 1-2 and MEASURED in its Sec 5 |
| Our baseline | `gguf_dequant.cpp:123` decodes type 40 correctly and `qwen3_5_gguf_weights.cpp:330` resolves the `<stem>.scale` sidecar and the per-expert slabs, but the ONLY product is bf16: `RouteGgufTensor` (`gguf_keep_quant.cpp:113`) has no residency that yields fp4, `vt::DType` has no NVFP4 member, and `KeepQuantDType` requires a `vt::cpu::HasQuantDotKernel` block dtype that NVFP4 does not and cannot have. The CONSUMER side is already complete and needs no change: `DenseMlpWeights::{gate,up,down}_proj_fp4` / `FullAttnLayerWeights::{q,k,v,o}_proj_fp4` / `Qwen3_5MoeWeights::expert_*_fp4` exist, and `DenseMlpBlock` (`qwen3_5.cpp:5246`) and the attention block already branch on `!*.Empty()` and on `IsTrueW4A4()` |
| Port map | `gguf_dequant.{h,cpp}`: new `RepackGgufNvfp4Rows` — the pure byte permutation from ggml blocks to the (`weight_packed[N,K/2]` torch-pairwise, `weight_scale[N,K/16]` linear fp8) pair, beside the existing case-40 decoder that owns the same layout facts. `gguf_keep_quant.{h,cpp}`: `GgufResidency::kNvfp4Fp4`, `Nvfp4Fp4DType`, `GgufNvfp4ComputeAvailable()`, `GgufLoadPolicy::nvfp4_fp4` + its `VT_GGUF_NVFP4_FP4` / `VT_GGUF_NVFP4_W4A4` env reads. `qwen3_5_gguf_weights.{h,cpp}`: `OwnGgufNvfp4` (2-D) and `OwnGgufNvfp4Experts` (3-D, one `Nvfp4Weight` per expert slab) building `Nvfp4Weight` from the repack + the `.scale` / `.input_scale` sidecars; `OwnMatmulWeightOrNvfp4` wiring in `LoadAttnGguf`, `LoadQwen3_5DenseFromGguf` and `LoadMoeGguf`. NO change to any `src/vt/` kernel, to `nvfp4_dequant.*`, or to the safetensors loaders |
| Tests to port | Nothing to port: vLLM has no GGUF. The executable spec is again CROSS-FORMAT EQUIVALENCE against the already-gated safetensors container, now at the OPERAND level rather than the value level, using the goldens that already exist (`tests/vllm/gguf_nvfp4_goldens.inc`, which carries both containers' bytes for the same three [4,512] slices). New cases in `tests/vllm/test_gguf_nvfp4.cpp` (repack byte-identity, non-vacuity, guards) and `tests/vllm/test_gguf_keep_quant.cpp` (routing totality for the new residency). Asset-gated real-file sweep + the e2e run stay in the existing `VLLM_NVFP4_GGUF`/`VLLM_NVFP4_ST` and `VLLM_QWEN36_GGUF` cases |
| Gates | (1) **BYTE-IDENTITY**: `RepackGgufNvfp4Rows` output == the safetensors `weight_packed` / `weight_scale` bytes for the same slice, ZERO differing bytes, and the `.scale` sidecar == `float32(1)/float32(weight_global_scale)` bit-for-bit. This makes the fp4 GEMM provably the SAME already-gated kernel on the SAME operand bytes, so no op-level numerics are re-litigated. (2) Non-vacuity: the wrong (torch-pairwise-in, split-half-out) permutation must FAIL gate 1. (3) Routing totality: every tensor still routed exactly once; NVFP4 in a non-verbatim role still expands. (4) The SACRED regression set unmoved: `test_qwen27_paged_engine` 235/235, `test_gguf_nvfp4`, `test_gguf_keep_quant`, `test_nvfp4_dequant`, `test_qwen36_paged_engine`. (5) e2e: the 27B NVFP4 GGUF loads and generates coherently with the fp4 path OBSERVED to have run, at materially lower weight residency. **NOT a gate**: token identity against the safetensors container (Sec A) — REPORTED instead, and it came out IDENTICAL 24/24 where the bf16 arm diverges at index 4 (Sec C). **ALL FIVE MET** on GB10 sm_121a, Sec C |
| Dependencies | `vt::MatmulNvfp4` + the cutlass fp4 family (CUDA-only; a CPU build correctly keeps expanding to bf16). `GgufGlobalScales` for the `.scale` sidecar. The 27B/35B assets on dgx.casa |
| Work breakdown | N1 the repack primitive + its byte-identity gate. N2 the residency + policy + routing gate. N3 the dense (27B) loader wiring. N4 the MoE (35B) stacked-expert wiring. N5 the GPU e2e run + residency measurement (DONE, Sec C). N6 the honest record of the cross-container weight delta (Sec A) and of the fact that it did NOT prevent token identity (Sec C). **N7 the MoE arm's OWN hardware gate on the real 35B (DONE, Sec D)**: per-expert cross-container byte identity + per-expert scale indexing + the two mutants + the load/e2e A/B - and the latent router-orientation defect that run exposed and this change fixes (Sec D.5) |
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

**Consequence.** Token-for-token identity between the two containers is not
GUARANTEED by construction, so it cannot be this row's gate; any residual
divergence is chargeable to the 192-tensor weight delta and the activation-scale
delta before it is chargeable to the fp4 path. The gate this row DOES own is
exact and sharper: the fp4 OPERANDS the GGUF hands the kernels must be
byte-identical to the ones the safetensors hands the SAME kernels, on every
projection the two containers share.

**Corrected 2026-07-29 by measurement (Sec C):** the prediction drawn from this
section — that the divergence therefore could not close — was WRONG in practice.
It closes completely on the measured prompt. The two deltas above are real, but
the GDN `in_proj` quantization error does not move the greedy argmax here, while
the bf16-vs-fp4 compute difference does. Both halves of the finding are kept
because the first still governs what may be GATED and the second is what was
actually observed.

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

## C. MEASURED on GB10 (2026-07-29) — the divergence closes, and the residency halves

Build proven production-configured three ways before any number was trusted:
`grep -ci "cutlass not found" configure.log` = **0** (the log prints `CUTLASS
found at /home/mudler/cutlass-4.5.0; enabling sm120a NVFP4 cutlass GEMM`,
`Marlin NVFP4 W4A16 MoE GEMM enabled`, `FlashAttention-2 ... ENABLED for arch(es)
[121a]`, 29 `Triton AOT:` lines, `MANIFEST hashes OK`); `cuobjdump -lelf` on both
gate binaries = **41 cubins, all `sm_121a`, zero `sm_75`** (`build.ninja` carries
`arch=compute_121a,code=[compute_121a,sm_121a]`; `CMakeCache.txt` remains the
decoy); SACRED `test_qwen27_paged_engine` **235/235, exit 0**, 30.91 s, 23.67 GiB.
Clean tree: `git archive` of the local commit into `~/work/nvfp4-c/src`.

### C.1 The same-binary A/B, one `flock $HOME/gpu.lock` series, idle box

`test_qwen27_gguf_nvfp4_compute -tc="*generates through the fp4 path*"`, greedy,
24 tokens, prompt "The capital of France is", 2 reps per arm, interleaved
0/1/1/0 so a drift would show:

| `VT_GGUF_NVFP4_FP4` | first 8 generated ids | peak RSS | wall (load+generate) |
|---|---|---|---|
| `0` (bf16 expansion) | `11751 13 271 248068 271 248069 271 4639` | 50.78 / 50.83 GiB | 1:58.50 / 1:36.30 |
| `1` (native fp4) | `11751 13 271 248068 198 8160 579 264` | 25.67 / 25.67 GiB | 0:41.41 / 0:40.14 |

Both arms reproduced their token stream exactly 2 of 2, rc=0 each. The safetensors
container, generated in its own process on the same binary and prompt, produces
`11751 13 271 248068 198 8160 579 264 ...`:

* **bf16 arm vs safetensors: diverges at index 4** (`271` against `198`).
* **fp4 arm vs safetensors: IDENTICAL, divergence index 24 of 24.**

So the divergence index goes **4 -> none**. That is the strongest end-to-end
signal available, and it doubles as the "the fp4 path actually ran" proof: the
switch changes the tokens, so it is not a no-op.

The two engines only co-reside in one process on the fp4 arm. On the bf16 arm the
run is **OOM-killed** while loading the safetensors reference beside the expanded
GGUF, on the 119 GiB unified pool — which is the residency win stated as a
constraint rather than a number.

### C.2 Weight residency, from the routing audit

`test_qwen27_gguf_nvfp4_compute` case 1, both policies in one process: **256 of
851 routed tensors** take the fp4 residency (64 layers x dense MLP gate/up/down,
plus 16 full-attention layers x q/k/v/o), with **0** of them left bf16. Those same
256 projections cost **35 840 MiB** expanded and **10 080 MiB** fp4-resident, a
**3.56x** reduction — exactly 2 bytes/element against 0.5625.

### C.3 Regression, same build, under the same lock

`test_qwen27_paged_engine` **235/235** (SACRED), `test_qwen36_paged_engine`
**2 cases / 315/315** (SACRED), `test_nvfp4_dequant` 4/47, `test_gguf_nvfp4`
**11/11, 2784/2784** with the real assets (its full-file sweep compares **192**
NVFP4 tensors across both containers), `test_gguf_keep_quant` 37/5985, `test_gguf`
30/103, `test_gguf_dequant` 15/480, `test_gguf_qwen36_loader` 6/286 — every one
rc=0. Box idle throughout (3-4 GiB used at series start and end, no CUDA compute
apps); `/` at 92% before and after; no doctest `-s`.

### C.4 What this run did NOT cover

**The MoE (35B) stacked-expert arm was code-complete and UNVERIFIED at this
point.** It is now hardware-gated; see Sec D, which supersedes this paragraph.

Still uncovered: a serving-throughput arm (tokens/s, TTFT, TPOT at concurrency)
against the vLLM oracle, which is why `P` stays `-`.

## D. The MoE stacked-expert arm, MEASURED on the real 35B (2026-07-29)

Asset `~/bench/q36-35b-a3b-nvfp4.gguf` (23.8 GB, `qwen35moe`, 41 blocks, 256
experts, top-8), against its own modelopt safetensors sibling
`~/bench/q36-35b-a3b-nvfp4-vllm/`. Same GB10 sm_121a build discipline as Sec C
(0 `cutlass not found`; `cuobjdump -lelf` **41 cubins, all `sm_121a`** on all
three gate binaries; SACRED `test_qwen27_paged_engine` **235/235** before any
number was trusted); clean tree via `git archive` into `~/work/nvfp4-moe/src`;
one `flock $HOME/gpu.lock` per series; evidence `~/work/nvfp4-moe/ev{,2}/`.

### D.1 What is actually NVFP4 in this file - a different subset from the 27B

| family | 27B GGUF | 35B A3B GGUF |
|---|---|---|
| MoE routed experts `ffn_{gate,up,down}_exps` | n/a | **NVFP4**, 120 3-D `[256,out,in]` |
| MoE shared experts `ffn_*_shexp` | n/a | **NVFP4**, 120 2-D |
| dense MLP `ffn_{gate,up,down}` | NVFP4 | n/a |
| full attention `attn_{q,k,v,output}` | NVFP4 | BF16 |
| GDN `attn_qkv` / `attn_gate` / `ssm_{alpha,beta,out}` | NVFP4 | BF16 |
| MTP layer (`blk.40`, `nextn.*`) | n/a | BF16 (the recipe ignores `mtp*`) |
| `output.weight` (lm_head) | - | NVFP4, but `NoNvfp4()` -> expands by design |

241 NVFP4 tensors: 121 2-D with a `[1]` `.scale`, 120 3-D with a `[256]` one, and
a matching `.input_scale` for every one. So on the 35B this row's subset IS the
MoE arm; the dense/attention arm the 27B exercises is not present here, and vice
versa. The two files together cover the whole of column `C`.

### D.2 Weight level: per-expert byte identity against the safetensors sibling

The 35B safetensors stores experts UNSTACKED (`mlp.experts.<e>.<proj>.weight` U8
`[N,K/2]` + `.weight_scale` fp8 `[N,K/16]` + `.weight_scale_2` f32), which is
exactly the per-expert counterpart a stacked GGUF slab must reproduce. New
asset-gated case in `tests/vllm/test_gguf_nvfp4.cpp`, over every NVFP4 expert
tensor of every layer x experts {0, 1, 2, 7, 128, 254, 255}:

```
compared 840 (tensor, expert) slabs across both containers; 840 per-expert scales bit-checked
[doctest] test cases:    14 |    14 passed | 0 failed | 0 skipped
[doctest] assertions: 12650 | 12650 passed | 0 failed |
```

**ZERO differing bytes** on `weight_packed` and on `weight_scale`, for every one
of the 840 slabs, and the GGUF `<stem>.scale[e]` bit-identical to expert `e`'s
`weight_scale_2` for all 840. The 120 shared-expert 2-D projections were checked
the same way (0 differing bytes, `.scale` bit-identical, and `gate.scale2 ==
up.scale2` on every layer, which is what `SharedGateUpFusedEligible` requires).

**Container note, and it differs from the 27B.** This checkpoint is MODELOPT, whose
`weight_scale_2` is ALREADY the multiply form, so the GGUF sidecar equals it
DIRECTLY. The 27B's compressed-tensors sibling stores the DIVISOR, so there the
same sidecar equals `float32(1)/float32(weight_global_scale)`. The loader is
unaffected (it only ever reads the GGUF side and multiplies), but a cross-container
check has to know which convention it is looking at.

### D.3 Non-vacuity: the two mutants this arm could plausibly have carried

A wrong slab offset or a wrong scale index keeps every byte a legal fp4 operand
and yields finite, plausible logits, so both are constructed and required to be
rejected (`tests/vllm/test_gguf_nvfp4.cpp`, synthetic fixture with per-expert
DISTINCT blocks, scales and input_scales):

* **`scales[0]` for every expert** - rejected: expert `e`'s `scale2` and
  `input_global_scale_inv` differ from expert 0's.
* **expert 0's SLAB for every expert** - rejected: more than a quarter of the
  packed bytes differ.

The real file makes the first mutant discriminable too: layer 0's three expert
stacks carry **117 / 117 / 138 distinct** `.scale` values across 256 experts, and
the loader-level assertion requires more than `E/4` distinct `scale2` values per
stack. The `.input_scale` sidecar is `[256]` but is CONSTANT across experts on
this file, so ITS per-expert indexing is only discriminable in the fixture; that
is recorded rather than claimed.

### D.4 The load: what the fp4 residency actually moves on the 35B

`test_qwen36_gguf_nvfp4_compute` case 1, `rc=0`, 154 008/154 008 assertions,
17.3 s, peak RSS 21.6 GiB:

```
35B fp4 arm: 240 of 733 routed tensors took the fp4 residency (120 stacked-expert);
120 fp4 expert stacks over 40 layers, 0 bf16 stacks;
expert residency 17280 MiB fp4 + 0 MiB bf16; true-W4A4=1
per-expert slab byte-identity: 12 slabs
router_gate: nk=1 shape [256, 2048]  (policy keep_quant=1 expand_nk=1 nvfp4_fp4=1 nvfp4_w4a4=1)
```

Every expert stack is fp4-resident with its bf16 counterpart EMPTY (the invariant
the forward's `!empty()` dispatch keys on), 256 `Nvfp4Weight`s per stack, and the
sampled per-expert operands byte-identical to a direct repack of that expert's own
row range of the file.

### D.5 THE DEFECT this run found - a latent router-orientation bug

**The first 35B NVFP4 GGUF forward THREW.** Not a load failure and not garbage
output: `vt: matmul: inner dims mismatch at src/vt/ops.cpp:103`, from

```
#1  vt::Matmul(...)
#2  vllm::(anonymous namespace)::MoeBlockFusedMarlinCuda(...)
#3  vllm::(anonymous namespace)::MoeBlock(...)
#4  vllm::(anonymous namespace)::RunLayerPaged(...)
```

Cause: the two fp4 fused MoE blocks (`MoeBlockFusedCuda`,
`MoeBlockFusedMarlinCuda`) issued the ROUTER GEMM as a bare `vt::Matmul`,
assuming the SAFETENSORS layout `[K=H, N=E]` (`LoadBf16Transposed`). The GGUF
loader keeps the router gate in the file's own `[N=E, K=H]` with `nk = true`
under `expand_nk`, which is DEFAULT ON wherever the quantized GEMM is registered
- and CUDA registers it since `KERNEL-QUANT-CIQ-GEMM-CUDA` (2026-07-29). So the
activation `[T, 2048]` met a `[256, 2048]` gate. The reference MoE loop had always
branched on the flag (`MatmulBf16`) and the bf16 fast path REFUSES `nk=true`
outright (`MoeBf16FastLayoutOk`); only the fp4 fused blocks assumed.

The bug is PRE-EXISTING and was made REACHABLE by this row: a GGUF load never
produced fp4-resident experts before, so a GGUF-loaded router gate never entered
those blocks. It is invisible to `test_qwen36_paged_engine` (safetensors,
`nk=false`) and to the 27B dense arm (no MoE block at all).

Deterministic once attributed - 3 of 3 reps threw, and 3 of 3 reps with
`VT_GGUF_KEEP_QUANT=0` (which turns `expand_nk` off, restoring the transposed
gate) passed and produced the correct stream. In the first series 1 of 4 fp4 runs
happened to pass, which is what made it look intermittent before the mechanism
was known.

**Fix:** `MoeRouterLogits` (`qwen3_5.cpp`) branches on `router_gate.nk` and issues
`vt::MatmulBT` for the `[N,K]` layout, exactly as `MatmulBf16D` does; both fused
blocks call it. Falling through to the reference loop - the precedent
`MoeBf16FastLayoutOk` sets - is NOT available here, because the block's bf16
expert fields are EMPTY by construction. Provably inert for the safetensors path:
`nk == false` still issues the identical `vt::Matmul`, and the SACRED gates are
unmoved (D.7).

### D.6 End to end, post-fix: same-binary A/B, one flock, idle box

`test_qwen36_gguf_nvfp4_compute -tc="*generates through the fp4 path*"`, greedy,
24 tokens, prompt "The capital of France is":

| `VT_GGUF_NVFP4_FP4` | first 8 generated ids | peak RSS | wall (load+generate) |
|---|---|---|---|
| `1` (native fp4), 2 reps | `11751 11 264 3177 34756 364 1141 8807` | 22.72 / 22.72 GiB | 0:28.76 / 0:29.55 |
| `0` (bf16 expansion) | `11751 11 264 3177 34756 364 1141 8807` | 68.50 GiB | 1:51.91 |

Both arms `rc=0`; the fp4 arm reproduced its stream exactly 2 of 2. Peak RSS
**68.50 -> 22.72 GiB (3.01x)** and load-and-generate **1:51.9 -> 0:28.8 (3.9x)**.

**The tokens do NOT change here, and that is the CORRECT result - unlike the 27B.**
On the 27B the switch flips bf16-expansion against TRUE W4A4 (fp4 activations), so
it moves tokens. On the 35B the routed experts run the fused Marlin **W4A16**
grouped GEMM in both arms (it consumes `scale2` and ignores `alpha`), so the fp4
arm and the bf16 arm are the same arithmetic over the same quantized weights and
token identity is the LOSSLESSNESS signal. "The fp4 path really ran" is therefore
carried by the residency (3.01x RSS, 240/733 tensors routed fp4 with 0 left bf16),
the 3.9x load-and-generate delta, and - unmistakably - by D.5: the bf16 arm never
enters `MoeBlockFusedMarlinCuda` at all, which is why only the fp4 arm ever threw.

Text, both arms: `" Paris, a city renowned for its rich history, culture, and
iconic landmarks. Situated in the north-central part of"`.

### D.6.1 A REAL OPEN ITEM: this case's greedy stream is not run-to-run stable

Established by repetition, and it BOUNDS every token-level statement above.
`VT_GGUF_NVFP4_W4A4=0` (the `use_a16` opt-out) produced the canonical stream in
2 of 3 runs and a different, still coherent one in the third
(`264 3177 3750 364 1141 …`, `" a city known for its art, fashion, and
gastronomy."`). The safetensors reference likewise produced one stream in 3 of 4
runs and another in the fourth. The initial reading - that `W4A4=0` was a
DETERMINISTIC mode difference - is therefore RETRACTED: it was this instability.

Consequences, stated rather than smoothed over:

* the binding results of this arm are the WEIGHT-LEVEL byte identity (D.2/D.3)
  and the residency/routing audit (D.4), none of which depend on a token stream;
* the fp4-vs-bf16 A/B still stands as measured - the fp4 arm reproduced its
  stream 3 of 3 and the bf16 arm matched it - but "identical tokens" is reported
  with this instability attached, not as a token-exactness claim;
* `test_qwen36_paged_engine` IS token-exact (315/315) at ITS engine params, so
  this is a property of this case's `block_size 32 / num_blocks 256 /
  max_num_seqs 1` configuration rather than of the checkpoint or of column `C`.
  Attributing it (a plausible candidate is the Marlin grouped GEMM's cross-SM
  atomic reduction, which is order-nondeterministic in fp) is OWED WORK and is
  NOT claimed here.

**Cross-container, REPORTED and NOT gated.** Against the modelopt safetensors
sibling on the same binary and prompt the divergence index came out **7** in
three runs and **16** in one - unstable for the reason just given, and in any
case not guaranteed by construction: the GGUF stores the GDN and full-attention
families as BF16 where the safetensors keeps native FP8 W8A8, so those layers are
different arithmetic before any expert runs.

**One asymmetry worth recording.** The GGUF carries `<stem>.input_scale`, so with
the default `VT_GGUF_NVFP4_W4A4=1` the loader sets `IsTrueW4A4()` - which our
SAFETENSORS 35B loader (`LoadNvfp4Raw`) deliberately never does. On this model
that only reaches the SHARED expert; the routed-expert grouped GEMM is W4A16
regardless of `alpha`. Both modes generate coherently and neither is gated.

### D.7 Regression, same build, same lock, every one rc=0

`test_qwen27_paged_engine` **235/235** (SACRED), `test_qwen36_paged_engine`
**2 cases / 315/315** (SACRED), `test_nvfp4_dequant` 4/47, `test_gguf_keep_quant`
37/5985, `test_gguf_nvfp4` **14 cases / 12 650 assertions** with the real assets
(the 27B sweep still 192 tensors, the new 35B sweep 840 per-expert slabs),
`test_qwen27_gguf_nvfp4_compute` (the dense arm) 2/2 . 778/778 with its fp4
stream unchanged. Box idle throughout (3-4 GiB used at series start and end, no
CUDA compute apps); `/` at 92% before and after; no doctest `-s`.

Re-confirmed on the EXACT committed tree (idle box, one flock, `cuobjdump -lelf`
41 cubins all `sm_121a` on all three gate binaries): residency 154 008/154 008
rc=0 (0:29.31, 21.6 GiB), e2e fp4 rc=0 with the same 24 tokens (0:31.09, 22.7
GiB), asset sweep 14/12 650 rc=0, `test_qwen27_paged_engine` 235/235,
`test_qwen36_paged_engine` 315/315. An earlier attempt at this same confirmation
was VOIDED and re-run: an unrelated 66 GiB CUDA process took the box to 119/119
GiB and OOM-killed it, which is exactly the contention hazard this asset carries.
