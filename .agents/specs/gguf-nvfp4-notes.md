# NVFP4 GGUF (`QUANT-GGUF-NVFP4`, ggml type 40)

**Current binding result (2026-07-28).** Type 40 is READ and MATERIALIZED. The
container was determined by BYTE-COMPARING the real files, not inferred: Sec 5
is the measured evidence and supersedes any earlier assumption from the fork
sources in Sec 1-4, which remain the correct citation for the type ID and block
GEOMETRY but describe a DIFFERENT, self-contained producer. **The GGUFs we must
actually load carry a per-tensor `<stem>.scale` f32 sidecar TENSOR, so their
blocks alone do NOT determine the weight.** Implementation
[gguf_dequant.cpp:123](../../src/vllm/model_executor/model_loader/gguf_dequant.cpp#L123),
loader plumbing
[qwen3_5_gguf_weights.cpp:330](../../src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp#L330),
gate [test_gguf_nvfp4.cpp](../../tests/vllm/test_gguf_nvfp4.cpp).

## Spike record

| Field | Content |
|---|---|
| Scope | `QUANT-GGUF-NVFP4`: read and dequantize ggml type 40 out of a GGUF, including the per-tensor / per-expert scale sidecar the container keeps outside the blocks. Loader materialization (`M`) only. NOT in scope: a keep-quant residency, a native NVFP4 GGUF GEMM (`C`), or an end-to-end model gate (`E`) |
| Upstream chain | No mainline ggml type 40: it is a fork/toolchain extension. Fork enum + block geometry `ggml/include/ggml.h:429-431`, `ggml/src/ggml-common.h:211-217` (Sec 1-2). The NUMERICS are NVIDIA NVFP4 as vLLM already implements it: `vllm/model_executor/layers/quantization/modelopt.py` (W4A16 dequant) and `compressed_tensors` `nvfp4-pack-quantized`, both mirrored in `nvfp4_dequant.{h,cpp}` |
| Our baseline | `gguf_reader.cpp:271-279` already tabulated (64, 36) traits for type 40; `nvfp4_dequant.cpp` already implemented the block math for the safetensors container. The ONLY gap was the container: `gguf_dequant.cpp` had no case 40 and no way to carry a per-tensor scale |
| Port map | `gguf_dequant.{h,cpp}`: `GgmlTypeNeedsGlobalScale` + 4-argument `DequantGgufRowTo{F32,Bf16}` overloads + the case-40 block decode reusing `kE2M1Lut` / `F8E4M3ToF32`. `qwen3_5_gguf_weights.cpp`: `GgufGlobalScales` sidecar resolution and the `DqSlabs` per-expert slab loop behind the existing `DqBf16` / `DqF32` |
| Tests to port | No upstream ggml test covers type 40 (it is not upstream). The executable spec is CROSS-FORMAT EQUIVALENCE against the already-gated safetensors NVFP4 path (`tests/vllm/test_nvfp4_dequant.cpp`, itself ported from `nvfp4_emulation_utils.dequantize_to_dtype`): `tests/vllm/test_gguf_nvfp4.cpp` with real bytes from both containers, plus the asset-gated full-file sweep |
| Gates | (1) bit-identical to `DequantNvfp4ToBf16` on real Qwen3.6-27B bytes from both containers; (2) the scale-less entry point REFUSES type 40 rather than assuming 1.0; (3) the full-file sweep over every shared NVFP4 projection of the real 18.8 GB GGUF vs the 26.4 GB safetensors |
| Dependencies | `nvfp4_dequant.h` (`kE2M1Lut`, `F8E4M3ToF32`, `kNvfp4GroupSize`); `gguf_reader` traits for type 40; the GDN v-head reorder (`ReorderVCols`) for `ssm_out`, which is orthogonal and already present |
| Work breakdown | W1 container determination by byte comparison (done, Sec 5). W2 dequant + sidecar plumbing + gates (done). W3 OPEN: a real end-to-end NVFP4 GGUF model load / greedy gate, which needs more than dequant (Sec 6). W4 OPEN: keep-quant residency + a native NVFP4 GGUF GEMM |
| Risks/decisions | The dangerous failure is a plausible-but-wrong dequant: a nibble permutation preserves the value histogram and the norms, so it yields finite, sane-looking logits and silently degrades output. DECISION: gate on BIT-EXACT cross-format equality, never a tolerance; and make the missing sidecar a THROW, never a 1.0 default. The wrong-order variant was run deliberately and the gate caught it on 1686/2048, 1935/2048 and 1678/2048 values (Sec 5.4) |

## Prior art: killgate fork sources (mined 2026-07-03)

Mined 2026-07-03 on dgx.casa from mudler's llama.cpp forks:

- `~/llama-phase84-attn-only-source` (primary source citations below)
- `~/llama-phase93-qwen3next-gqa-bcast` (identical `ggml_type` enum — verified,
  same ids 39/40/41 at `ggml/include/ggml.h:429-431`)
- `~/killgate_series/*.patch` (patches 0015/0017/0020/0023/0025 exercise
  `GGML_TYPE_NVFP4` in CUDA MMQ/MoE paths; the type itself is defined in the
  fork source tree, not introduced by a patch)

## 1. Fork type ids (ggml/include/ggml.h)

The fork's `enum ggml_type` matches mainline llama.cpp exactly through id 35,
then appends two fork-specific ids after mainline's MXFP4:

```
GGML_TYPE_MXFP4 = 39,  // MXFP4 (1 block)            ggml.h:429  (same as mainline)
GGML_TYPE_NVFP4 = 40,  // NVFP4 (4 blocks, E4M3 scale) ggml.h:430  (FORK EXTENSION)
GGML_TYPE_Q1_0  = 41,                                 ggml.h:431  (FORK EXTENSION)
GGML_TYPE_COUNT = 42,                                 ggml.h:432
```

File-type (ftype) ids, `gguf-py/gguf/constants.py`: `MOSTLY_NVFP4 = 39`,
`MOSTLY_Q1_0 = 40` ("except 1d tensors").

## 2. NVFP4 block layout (id 40)

Source: `ggml/src/ggml-common.h:211-217`:

```c
#define QK_NVFP4 64
#define QK_NVFP4_SUB 16  // sub-block size for per-group scales
typedef struct {
    uint8_t d[QK_NVFP4/QK_NVFP4_SUB]; // UE4M3 scales (4 bytes, one per 16-element sub-block)
    uint8_t qs[QK_NVFP4/2];           // packed 4-bit E2M1 values (32 bytes)
} block_nvfp4;
```

- **block_elems = 64**, **block_bytes = 36** (4 + 32). Confirmed by the fork's
  `gguf-py/gguf/constants.py:4595`: `GGMLQuantizationType.NVFP4: (64, 4 + 32)`
  and by `ggml/src/ggml.c:741-748` type_traits (`.blck_size = QK_NVFP4`,
  `.type_size = sizeof(block_nvfp4)`).
- **Scales**: one unsigned E4M3 (**UE4M3**, no sign bit semantics — decoded by
  `ggml_ue4m3_to_fp32`, `ggml/src/ggml-impl.h:502`) scale byte per 16-element
  sub-block; 4 sub-blocks per 64-element block. This is the per-16 NVFP4
  micro-block scale layout (not per-32 like MXFP4).
- **Elements**: 4-bit **E2M1** codes, two per byte, decoded through the shared
  `kvalues_mxfp4` LUT (values {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}, stored as
  int8 2x-scaled entries in ggml-common.h).
- **Nibble order** (`dequantize_row_nvfp4`, `ggml/src/ggml-quants.c:531-554`):
  within sub-block `s`, byte `qs[s*8 + j]` (j in 0..7) holds element `j` in the
  low nibble and element `j + 8` in the high nibble; `y = kvalue * d`.
- **No per-tensor scale tensor IN THIS FORK.** Its quantizer
  (`quantize_row_nvfp4_ref`, ggml-quants.c:346) picks `d = ue4m3(amax_sub / 6)`
  per sub-block and stores nothing outside the block, so a file from THIS
  producer is self-contained. **SUPERSEDED as a general claim by Sec 5:** the
  NVFP4 GGUFs we actually have to load come from a different producer and DO
  carry the NVIDIA per-tensor global scale, as a sidecar tensor. A loader must
  therefore never assume self-containment; ours refuses to guess.

### Q1_0 (id 41), for completeness

`ggml-common.h:177-182`: `QK1_0 = 128`, `block_q1_0 = { ggml_half d; uint8_t
qs[QK1_0/8]; }` → **128 elems / 18 bytes** (`constants.py:4596`: `(128, 2+16)`).
Sign-bit-per-element ternary-ish 1-bit format; not observed in any APEX file.

### MXFP4 (id 39, same as mainline)

`ggml-common.h:205-210`: `QK_MXFP4 = 32`, 1 byte E8M0 scale + 16 bytes packed
e2m1 → **32 elems / 17 bytes**. Killgate patches 0015/0017 tune NVFP4 and MXFP4
together in the CUDA MMQ path.

## 3. APEX tensor-type histogram decoded

Observed histogram `{0: 301, 11: 159, 12: 178, 13: 34, 14: 1, 22: 60}` is from
`Qwen3.6-35B-A3B-APEX-I-Mini.gguf` (733 tensors,
`dgx.casa:/home/mudler/work/apex/qwen36_35b/`). Decoded against the FORK's
enum (which is identical to mainline for ids <= 35):

| id | fork enum name | block (elems, bytes) | count | used for |
|----|----------------|----------------------|-------|----------|
| 0  | F32    | (1, 4)     | 301 | norms, `ffn_gate_inp`, ssm_a/conv1d/dt/norm, 1-d tensors |
| 11 | Q3_K   | (256, 110) | 159 | token_embd, GDN attn_gate/ssm_{alpha,beta,out}, some expert ffn_{gate,up}_exps |
| 12 | Q4_K   | (256, 144) | 178 | attn_qkv, shared-expert ffn_*_shexp, attn_output, some ffn_down_exps |
| 13 | Q5_K   | (256, 176) | 34  | higher-precision shexp / ffn_down_exps in early layers |
| 14 | Q6_K   | (256, 210) | 1   | output.weight |
| 22 | IQ2_S  | (256, 82)  | 60  | MoE expert weights ffn_{down,gate,up}_exps.weight in 20 layers |
| 23 | IQ4_XS | (256, 136) | 60 (Quality variants) | MoE expert weights |
| 8  | Q8_0   | (32, 34)   | 120 (Balanced/Quality variants) | — |

**id 22 is IQ2_S, NOT NVFP4.** Verified by reading the fork enum
(`ggml.h:412`) and by dumping the file with the fork's own gguf-py: e.g.
`blk.10.ffn_down_exps.weight` shape ggml-[512, 2048, 256], type IQ2_S,
nbytes = 85_983_232 = 512*2048*256 / 256 * 82 — matches our traits math
(256-elem blocks, 82 bytes: 2 d + 64 qs + 16 qh).

All seven APEX GGUFs in `~/work/apex/qwen36_35b/` were histogrammed:

```
APEX-Balanced / I-Balanced: {0: 301, 8: 120, 13: 81, 14: 231}
APEX-Compact  / I-Compact : {0: 301, 11: 90, 12: 191, 14: 151}
APEX-I-Mini               : {0: 301, 11: 159, 12: 178, 13: 34, 14: 1, 22: 60}
APEX-Quality / I-Quality  : {0: 301, 8: 120, 13: 30, 14: 222, 23: 60}
```

**No APEX file uses NVFP4 (40) or any fork-specific id.** The APEX quant sweep
is pure mainline K-quants + i-quants. NVFP4 GGUFs do exist elsewhere in
mudler's fleet (killgate patches benchmark Qwen3-32B-NVFP4 and
Qwen3.6-35B-A3B NVFP4 dense/MoE), so the reader supports id 40 for those.

## 4. Implications for vllm.cpp

- `GgufFile` traits table (`src/vllm/model_executor/model_loader/gguf_reader.cpp`)
  now carries ids 22, 23 (needed by APEX Mini/Quality) and 39, 40, 41
  (MXFP4 + the two fork extensions). nbytes math is `numel / block_elems *
  block_bytes` with divisibility enforced — matches the fork's
  `ggml_row_size` for all these types (none has padding).
- **M2.2 kernels (NVFP4 dequant/GEMM)**: per-16 UE4M3 sub-block scale, e2m1
  LUT shared with MXFP4, low-nibble = element j / high-nibble = element j+8
  within a sub-block's 8 bytes. `QR_NVFP4 = 2`, `QI_NVFP4 = 8`
  (ggml-common.h:109-110) for the CUDA int-pack view. Killgate patch 0017
  notes dense NVFP4 decode GEMM is weight-read bandwidth-bound on GB10 with
  mmq_y=128 tiles; patch 0015 gates MoE token-tile selection for
  NVFP4/MXFP4 MoE (256 experts, top-8) — relevant when we port MMQ.
- Expert weights are the NVFP4 target in fork models (MoE `*_exps` 3-d
  tensors), same tensor family APEX-Mini puts in IQ2_S — M2.2 should plan
  for quantized 3-d expert tensors with per-expert row strides in blocks.

**M2.2 kernel-author trap (review finding):** the fork's `ggml_ue4m3_to_fp32`
returns value × 0.5 to compensate for the 2×-scaled `kvalues_mxfp4` LUT (see
ggml-impl.h comment; 0x7F decodes to 0). If you write your own UE4M3 decode
from the bias=7 spec while reusing the fork LUT, you will be 2× off.

## 5. MEASURED container evidence (2026-07-28) — this is what we load

Sections 1-4 are the fork PRIOR ART. This section is the DIRECT MEASUREMENT of
the two NVFP4 GGUFs on `dgx.casa` that block `SPEC-MTP-GGUF`'s GPU gate, and it
is what the implementation follows. Method: a self-contained GGUF header walk
(`scripts/gen-gguf-nvfp4-goldens.py` carries the same reader) plus a byte-level
comparison against the compressed-tensors export of the SAME quantization run.

### 5.1 The files

| File | Arch | ftype | Tensor types | NVFP4 tensors | Sidecars |
|---|---|---|---|---|---|
| `~/bench/q36-27b-nvfp4.gguf` (18.8 GB) | `qwen35` | 39 | `{F32: 1345, BF16: 2, NVFP4: 496}` | 496, all 2-D | 496 x `.scale` `[1]` |
| `~/bench/q36-35b-a3b-nvfp4.gguf` | `qwen35moe` | 39 | `{F32: 792, BF16: 202, NVFP4: 241}` | 121 2-D + 120 3-D `[E,out,in]` | 121 x `.scale` `[1]`, 120 x `.scale` `[256]` |

Every NVFP4 weight has a matching `.scale` sidecar (0 missing, checked
exhaustively), plus an `.input_scale` that is the W4A4 ACTIVATION scale and is
unused on our weight-only path. `general.tags` on the 27B names
`compressed-tensors`, and the oracle
`~/bench/q36-27b-nvfp4-vllm/` is the `nvfp4-pack-quantized` export of the same
model (`config.json` `quantization_config.format`, `group_size: 16`,
`scale_dtype: torch.float8_e4m3fn`).

### 5.2 The block layout, established by byte comparison

`block_nvfp4` is 64 elements in 36 bytes, `uint8 d[4]` then `uint8 qs[32]`:

- `d[s]` is an **IEEE fp8-e4m3fn** scale for 16-element sub-block `s`. Not a
  ggml UE4M3: the four bytes of every GGUF block are **byte-identical** to the
  corresponding four `weight_scale` bytes on the safetensors side.
- sub-block `s` owns `qs[s*8 .. s*8+8)`; byte `j` holds element `s*16 + j` in
  the LOW nibble and element `s*16 + j + 8` in the HIGH nibble. This is the ggml
  SPLIT-HALF packing of Sec 2, and it differs from the torch PAIRWISE packing
  (`2i` low, `2i+1` high) that `nvfp4_dequant.cpp` reads.
- The scales come FIRST, the nibbles second.

Verified by unpacking both containers and comparing element-for-element over
the first 64 rows of `blk.0.ffn_gate`, `blk.0.ffn_down`, `blk.7.ffn_up` and
`blk.63.ffn_down` (~4.4 M nibbles): scale bytes identical, nibbles identical
under exactly this permutation, zero mismatches.

### 5.3 The per-tensor scale: MULTIPLY, not divide

`<stem>.scale` is f32 and is **bit-identical to `float32(1) /
float32(weight_global_scale)`** on the safetensors side, i.e. the modelopt
`weight_scale_2` convention `nvfp4_dequant.h` already documents, NOT the
compressed-tensors divisor. Checked bit-for-bit on `blk.0.ffn_gate` (wgs 6624),
`blk.0.ffn_down` (2880), `blk.0.ssm_out` (2160), `blk.7.ffn_up` (11904),
`blk.63.ffn_down` (5344). So the value is

```
out[o, i] = bf16( e2m1_lut[nibble] * ( f8_e4m3(d[s]) * gguf_scale ) )
```

with the group scale formed FIRST, exactly as in `DequantNvfp4ToBf16`, which is
why the two containers agree BIT for BIT rather than approximately.

A stacked expert tensor's sidecar holds **one f32 per expert** (`[256]` on the
35B), in expert order, matching vLLM's `w13_weight_scale_2`. The expert axis is
the slowest, so each expert is a contiguous slab of whole blocks.

### 5.4 Why the gate is bit-exact

A nibble-permutation bug preserves the value histogram inside every group of
16, so the tensor norms are unchanged and the model still produces finite,
plausible logits: exactly the silent-degradation failure this row must not
ship. The wrong (torch pairwise) order was therefore implemented and run
DELIBERATELY before the right one, and the cross-format gate rejected it on
1686/2048, 1935/2048 and 1678/2048 values across the three CI slices.

### 5.5 `ssm_out` also carries the GGUF v-head tiling

`blk.N.ssm_out` is NVFP4 in both containers but is NOT byte-comparable raw: the
converter stores GDN value heads TILED, so GGUF head `r*num_k + k` is HF
grouped head `k*num_v_per_k + r` (verified on layer 0 of the 27B: `num_k` 16,
`num_v_per_k` 3, head dim 128, all 48 heads matched). That is exactly the
mapping `ReorderVCols` in `qwen3_5_gguf_weights.cpp` already applies at load, so
it is not an NVFP4 question. It IS why the cross-format sweep compares the MLP
projections only.

### 5.6 The 27B GGUF quantizes MORE than its safetensors sibling

The checkpoint recipe (`recipe.yaml`) ignores `lm_head`, `mtp.*`, `visual.*` and
the whole GDN `in_proj_{qkv,z,a,b}` family, which stay BF16 in the safetensors.
The GGUF stores `attn_qkv`, `attn_gate`, `ssm_alpha` and `ssm_beta` as NVFP4
anyway. Those tensors have no safetensors counterpart, so the oracle covers
`ffn_{gate,up,down}` and `ssm_out` only; the container itself is proven by
those, since the encoding does not vary per tensor.

**QUANTIFIED 2026-07-29, and it is larger than "those tensors have no
counterpart" suggests:** the GGUF holds **496** NVFP4 tensors against the
safetensors' **304**, and the 192-tensor difference is exactly the GDN
`in_proj_{qkv,z,a,b}` family. Dequantizing the GGUF's `blk.N.attn_qkv` against
the safetensors' BF16 `in_proj_qkv` over the leading (unreordered) 64 rows gives
mean relative errors 0.180 / 0.181 / 0.186 at layers 0 / 1 / 40. The activation
global scales differ too. So the two containers are DIFFERENT MODELS, and the
often-repeated reading that their greedy divergence at token index 4 is a
bf16-vs-fp4 compute artifact is retired: it is a weight difference.
[gguf-nvfp4-native-compute.md](gguf-nvfp4-native-compute.md) Sec A carries the
table.

## 6. What is NOT done

- `C` (native quantized compute): **LANDED 2026-07-29 as `part`, in the
  companion spike [gguf-nvfp4-native-compute.md](gguf-nvfp4-native-compute.md).**
  The statement that stood here — that NVFP4 has no `vt::DType` block encoding
  and no `vec_dot` so `RouteGgufTensor` must expand it — was true and remains
  true, but the CONCLUSION drawn from it was wrong: NVFP4 does not need a vt
  block dtype, because its consumer is not a `vec_dot` but the existing
  `vt::MatmulNvfp4*` family, whose operands the ggml blocks REPACK into byte for
  byte. The dense MLP, full attention and the MoE experts now compute in fp4 on
  CUDA; the GDN `in_proj_*` family and `ssm_out` still expand (V-head reorder),
  as does any CPU build. The MoE STACKED-EXPERT arm - the per-expert `[256]`
  `.scale` handling Sec 5.3 above describes - is HARDWARE-GATED as of 2026-07-29
  on the real 35B A3B file (840 per-expert slabs byte-identical to the modelopt
  safetensors, mutation-proved); see that spike's Sec D.
- `E` (end-to-end): dequant alone does not load the 27B/35B models. The
  remaining gaps are model-level, not container-level, and are tracked on
  `SPEC-MTP-GGUF`.
- The load-time tensor-routing audit does not yet observe the `.scale` /
  `.input_scale` sidecars (they are read directly, not routed), so a totality
  assertion over an NVFP4 file would count them as unrouted.
