# MXFP4 compressed-tensors quant path — spike + W1

**Row:** `QUANT-CT-MXFP4` (`.agents/quantization-matrix.md`). **Claim:**
`CLAIM-QUANT-MXFP4`. **State:** `INVENTORIED` → `ACTIVE` (W0 spec + W1 CPU dequant
brick landed; the GPU W4A4 fp4 GEMM is a named later brick).

**Base:** current `main` HEAD `42c56b51` (isolated worktree, CPU-only, foreground,
NOT pushed). **Pinned oracle:** `${VLLM_SOURCE}` = `/home/mudler/_git/vllm` @
`5559679229bc961848b121ccdeaa8fa5d79bec98` (vLLM 0.26.0.dev0).

**Why now:** shared unblocker. Both **DeepSeek-V4-Flash** (W6 MegaMoE MXFP4
experts) and **Kimi-K3** (its real 2.8T checkpoint is `mxfp4-pack-quantized`)
currently REFUSE MXFP4 in their loaders, citing exactly this scope. This row OWNS
the MXFP4 quant path; it does NOT touch those two model files. A separate AWQ+GPTQ
lane is live; MXFP4 is a DISTINCT scheme and does not touch AWQ/GPTQ files.

---

## Scope

MXFP4 is the OCP Microscaling FP4 weight format, exposed to vLLM through the
compressed-tensors `mxfp4-pack-quantized` scheme (`CompressedTensorsW4A4Mxfp4`):

- **4-bit float weights (E2M1)** packed two-per-uint8 — element `2i` = low nibble,
  `2i+1` = high nibble; bit 3 = sign, bits 0..2 index the E2M1 magnitude LUT
  `{0, .5, 1, 1.5, 2, 3, 4, 6}`. Identical packing to our NVFP4.
- **Per-group E8M0 scales, group_size 32** (NVFP4 is 16). One uint8 E8M0 scale per
  32 consecutive input elements: `weight_scale` is `[out, in/32]`.
- **NO global scale** (NVFP4 carries a per-tensor `weight_global_scale`).
- On-disk tensors: `weight_packed` (U8 `[N, K/2]`), `weight_scale` (U8 E8M0
  `[N, K/32]`). Both LINEAR layout (no swizzle on disk; the kernels swizzle in
  `process_weights_after_loading`).

**Dequant math** (the golden this port mirrors 1:1):
`w[o,i] = e2m1_lut[nibble(o,i)] * 2^(weight_scale[o, i/32] - 127)`.

**Explicit contrast vs our NVFP4** (`nvfp4_dequant.{h,cpp}`,
`nvfp4_emulation.cpp`, rows `QUANT-NVFP4-CT-W4A4` / `QUANT-NVFP4-CT-W4A16`):

| Axis | NVFP4 (ours, landed) | MXFP4 (this row) |
|---|---|---|
| group_size | 16 | **32** |
| block scale codec | fp8-e4m3 byte via `F8E4M3ToF32` | **E8M0 byte, `2^(byte-127)`** (exact pow2) |
| global scale | per-tensor `weight_global_scale` (divisor) | **none** |
| dequant | `lut * f8(scale) * global` | `lut * 2^(scale-127)` |
| E2M1 packing | 2 nibbles/byte, low=even | **identical** (reused) |

Because the E8M0 scale is an exact power of two, the bf16 store only re-homes the
E2M1 exponent and is exact for finite in-range scales — no rounding subtlety like
the NVFP4 `weight_scale_2` bf16 round.

## Upstream chain (`file:line`, pin 555967922)

- **Scheme / config parse:**
  `compressed_tensors/schemes/compressed_tensors_w4a4_mxfp4.py:20-97`
  (`CompressedTensorsW4A4Mxfp4`, `self.group_size = 32`, `create_weights` registers
  `weight_packed` U8 `[N,K/2]` + `weight_scale` U8 `[N,K/32]`, `get_min_capability`
  80). Scheme selection: `compressed_tensors.py` `_get_scheme_from_parts` /
  `_is_fp4a4_nvfp4`-family branch that maps `format == "mxfp4-pack-quantized"` to
  this class.
- **E8M0 scale semantics:** `utils/mxfp8_utils.py:61-65,222`
  (`scale_biased = floor(log2(amax)) + 127`; `descale = exp2(scale - 127)`).
- **Numerical GOLDEN (dequant reference):** `tests/quantization/reference_mxfp4.py:28-117`
  `dq_mxfp4_torch` = `e8m0_to_half(2^(byte-127))` + `upcast_fp4_to_fp16_or_bf16`
  (E2M1 bit unpack) + reshape-to-32 group multiply. This is what W1 ports.
- **GPU GEMM dispatch (the actual thing GB10 runs):**
  `kernels/linear/__init__.py:808-845` `init_mxfp4_linear_kernel` iterates
  `_POSSIBLE_MXFP4_KERNELS[CUDA] = [FlashInferMxFp4LinearKernel,
  MarlinMxFp4LinearKernel, HummingMxFp4LinearKernel]` (`:466-475`).
  - `mxfp4/flashinfer.py:18-76` — **true W4A4** via FlashInfer CUTLASS cute-dsl
    (`flashinfer_mxfp4_quantize` the activation to fp4, `flashinfer_scaled_fp4_mm`
    with `use_nvfp4=False`, `block_size=32`); `is_supported` requires
    `has_device_capability(100)` AND `has_flashinfer_cutedsl()`. **GB10 (sm_121,
    cap 121 ≥ 100) selects this WHEN flashinfer cute-dsl is present**, else falls
    through.
  - `mxfp4/marlin.py:9-52` — **W4A16 weight-only** fallback
    (`prepare_fp4_layer_for_marlin` / `apply_fp4_marlin_linear`,
    `weight_global_scale=None` — the MXFP4 marlin path folds the E8M0 scale, no
    global). Selected on GB10 when cute-dsl is absent.
  - MoE analogue: `compressed_tensors_moe/compressed_tensors_moe_w4a4_mxfp4.py`
    (group 32, the DeepSeek-V4/Kimi-K3 expert path).

## Our baseline (reuse-vs-new, our `file:line`)

- **REUSE (the E2M1 half of the codec is already ours):**
  `include/vllm/model_executor/model_loader/nvfp4_dequant.h:32-38` (`kE2M1Lut`),
  `.../nvfp4_dequant.cpp:33-76` (`DequantNvfp4ToBf16` — the exact row/group/nibble
  loop MXFP4 clones, swapping group 16→32 and the scale codec),
  `.../compressed_tensors/nvfp4_emulation.{h,cpp}` (the W4A4 emulation shape the
  future MXFP4 GEMM emulation mirrors). `vt::F32ToBF16` / `vt::BF16ToF32`
  (`vt/dtype.h`) for the bf16 round.
- **NEW (this row):** the E8M0 scale decode (`2^(byte-127)`, no fp8, no global),
  group_size 32, and the MXFP4 dequant emitters. Landed as
  `model_loader/mxfp4_dequant.{h,cpp}` — additive TUs, SACRED-inert, ZERO edits to
  the NVFP4 path. Later bricks: the compressed-tensors scheme-selection wiring
  (a `CompressedTensorsW4A4Mxfp4`-equivalent method), the GPU W4A4 fp4xfp4 GEMM +
  activation quant (FlashInfer-parity) and the Marlin W4A16 fallback, and the MoE
  expert path both models consume.

## Port map

- `include/vllm/model_executor/model_loader/mxfp4_dequant.h` — `kMxfp4GroupSize`
  (32), `E8M0ToF32`, `DequantMxfp4ToBf16`, `DequantMxfp4ToF32`.
- `src/vllm/model_executor/model_loader/mxfp4_dequant.cpp` — the implementations,
  a single templated row loop shared by the bf16/f32 emitters.
- `CMakeLists.txt` — add the TU to the `vllm_cpp` library sources.
- `tests/vllm/test_mxfp4_dequant.cpp` + `tests/CMakeLists.txt` — the unit gate.
- **Named later bricks (NOT this change):** scheme-selection method + loader probe
  (mirror `schemes/nvfp4.h` `Make*Method`); GPU W4A4 fp4 GEMM + activation quant;
  Marlin W4A16 MXFP4 fallback; the MoE expert dequant/GEMM. The two model loaders
  (DeepSeek-V4, Kimi-K3) then drop their MXFP4 refusal and call this path.

## Tests to port

- `tests/quantization/reference_mxfp4.py:28-117` `dq_mxfp4_torch` → re-expressed as
  `RefDqMxfp4` (double precision) + literal hand cases inside
  `tests/vllm/test_mxfp4_dequant.cpp`. **PORTED (W1).**
- `tests/quantization/test_compressed_tensors.py:937-962`
  `test_compressed_tensors_mxfp4` (loads `nm-testing/TinyLlama-1.1B-Chat-v1.0-MXFP4`,
  asserts `scheme.group_size == 32` + greedy generate) → the e2e model gate;
  **SKIPPED-with-reason** until the scheme-selection + GPU GEMM bricks land and a
  fitting MXFP4 checkpoint is available on-box (tracked here, not yet a local test).

## Gates

- **W1 (landed):** `test_mxfp4_dequant` — E8M0 decode known-byte cases; hand-computed
  32-group dequant (bf16 + f32); the E8M0-vs-fp8 and group-32-vs-16 RED traps;
  multi-row/multi-group offset arithmetic; randomized rel-error vs the
  double-precision golden with bf16==f32 exactness. CPU `-Werror` 0-warn.
- **Later:** GPU W4A4 fp4 GEMM unit gate vs this CPU dequant (mirror NVFP4's
  emulation gate); the `test_compressed_tensors_mxfp4` e2e once a checkpoint runs.

## Dependencies

- **Shared with `CLAIM-DEEPSEEK-V4-*` and `CLAIM-KIMI-K3-SCOPE`:** MXFP4 must not be
  implemented twice. This row is the single owner; those model rows consume it and
  keep their refusal until the wiring brick lands.
- **DISTINCT from the AWQ+GPTQ lane** (`QUANT-AWQ` / `QUANT-GPTQ`): no shared files.
- GPU bricks depend on FlashInfer cute-dsl availability on GB10 (else Marlin W4A16).
- `nvfp4_dequant.h` (`kE2M1Lut`, `F8E4M3ToF32`) — reused, not modified.

## Work breakdown

- **W0 — spec (this file) + row records.** DONE.
- **W1 — CPU MXFP4 weight unpack + E8M0 dequant to bf16/f32 + unit gate.** DONE.
- **W2 — scheme-selection method + loader probe** (mirror `schemes/nvfp4.h`),
  materialize-to-bf16 wired for a linear projection. NAMED, not started.
- **W3 — GPU W4A4 fp4xfp4 GEMM + activation quant** (FlashInfer-parity) and the
  **Marlin W4A16 MXFP4 fallback** (GB10 dispatch mirror). NAMED.
- **W4 — MoE MXFP4 expert path** (group-32) both models consume. NAMED.
- **W5 — e2e model gate** (`test_compressed_tensors_mxfp4` equivalent) once a
  fitting checkpoint runs; then the model rows drop their MXFP4 refusal.

## Risks / decisions

- **CPU dequant is the truth, not the throughput path** (DECISION) — W1 is the
  CPU-reference both GPU GEMM bricks validate against, exactly like NVFP4's
  emulation. Not a perf claim.
- **E8M0 NaN edge (byte 0xFF)** returned as NaN per the OCP spec, not silently
  `2^128`; QAT scales are always finite so this never fires in practice (RECORDED).
- **GB10 GPU path is capability-gated** — true W4A4 only when FlashInfer cute-dsl is
  present; else Marlin W4A16 weight-only. Mirror vLLM's selection exactly at W3.
- **No on-box e2e yet** — the two owning checkpoints are huge (Kimi-K3 2.8T does not
  fit one GB10; DeepSeek-V4-Flash NVFP4/fp8 need multi-Spark). The dequant brick is
  gateable at unit scale today; the e2e stays derive-and-ship until a fitting MXFP4
  vehicle runs (RECORDED, mirrors both model specs).
