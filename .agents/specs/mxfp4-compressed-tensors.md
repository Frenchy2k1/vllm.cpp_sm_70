# MXFP4 compressed-tensors quant path — spike + W1

**Row:** `QUANT-CT-MXFP4` (`.agents/quantization-matrix.md`). **Claim:**
`CLAIM-QUANT-MXFP4`. **State:** `INVENTORIED` → `ACTIVE` (W0 spec + W1 CPU dequant
brick landed; the GPU W4A4 fp4 GEMM is a named later brick).

**Base:** current `main` HEAD `42c56b51` (isolated worktree, CPU-only, foreground,
NOT pushed). **Pinned oracle:** `${VLLM_SOURCE}` = `/home/mudler/_git/vllm` @
`5559679229bc961848b121ccdeaa8fa5d79bec98` (vLLM 0.26.0.dev0). **Runnable oracle
on-box** = `~/venvs/vllm-oracle -> vllm-oracle-v0.25.0-stage` (vLLM **0.25.0**,
compressed_tensors 0.17.0). The 0.25.0 tree is the one W0-W5 gate against; its
mxfp4 dispatch is byte-for-byte the same shape as the 0.26 pin (verified below).

> **W0-W5 UPDATE (2026-08-05, USER-priority "full MXFP4 at vLLM parity,
> benchmarked on a Qwen model", branch `row/QUANT-CT-MXFP4`).** DeepSeek/Kimi are
> NOT the vehicle (won't fit / not the target); **Qwen dense is**. This block
> re-scopes W2-W5 around a real on-box Qwen MXFP4 checkpoint and pins the parity
> target from the RUNNING oracle. See "## W0-W5 (Qwen vehicle) — 2026-08-05".

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
  vehicle runs (RECORDED, mirrors both model specs). **SUPERSEDED for the e2e vehicle
  by the Qwen dense path below (2026-08-05): a small dense Qwen3-8B MXFP4 checkpoint
  fits GB10 and the oracle registers its arch, so the e2e gate is now reachable
  independently of DeepSeek/Kimi.**

---

## W0-W5 (Qwen vehicle) — 2026-08-05

USER re-scope: **full MXFP4 support at vLLM parity, benchmarked on a Qwen model.**
Qwen dense is the vehicle (DeepSeek/Kimi explicitly out — fit/target). This section
pins the parity target from the RUNNING 0.25.0 oracle and lays the W2-W5 contract.
Empirical (RUN/BUILD/BENCH) steps are GPU-gated and may lag the spec (box shared;
locks + disk contended); the design + oracle-support proofs below are not.

### W0 — checkpoint gateability

**Vehicle: `Yi30/Qwen3-8B-MXFP4`** (HF). The clean dense W0 vehicle.
- `config.json`: `architectures=["Qwen3ForCausalLM"]` (dense — the best-supported
  Qwen family, NOT the new hybrid `Qwen3_5ForConditionalGeneration`),
  `quant_method="compressed-tensors"`, `format="mxfp4-pack-quantized"`,
  `group_size=32`, `ignore=["lm_head"]`. **`input_activations` is SET**
  (`dynamic=true`, `num_bits=4`, `type=float`, `group_size=32`) → this is a **true
  W4A4** checkpoint (weights AND activations MXFP4), which selects the W4A4 GEMM on
  GB10 (see W1). Weights: 4-bit float, group 32, symmetric.
- Size: 2 shards, **6.18 GB** safetensors total; complete tokenizer +
  `generation_config` + chat template. Fits GB10 trivially (~6 GiB weights).
- **Oracle-support PROVED at the import/registry layer (0.25.0, on dgx):**
  `ModelRegistry.get_supported_archs()` contains `Qwen3ForCausalLM`; and
  `from ...schemes.compressed_tensors_w4a4_mxfp4 import CompressedTensorsW4A4Mxfp4`
  imports clean. **Oracle-RUN proof (greedy golden) is GPU-gated — QUEUED.**
- Alternatives surveyed: `Yi30/Qwen3-8B-MXFP4-LLMC` (same, produced by
  llm-compressor — the mechanical-repro arm), `Yi30/Qwen3-8B-MXFP4-FP8KV[-FP8Attn]`
  (adds fp8 KV/attn — extra axes, avoid for the clean gate);
  `olka-fi/Qwen3.5-27B-MXFP4` (genuine `mxfp4-pack-quantized` but **weight-only**,
  `input_activations=null`, arch `Qwen3_5ForConditionalGeneration` — the new hybrid;
  a valid W4A16 secondary vehicle but a harder arch); `OsaurusAI/Qwen3.6-27B-MXFP4`
  = MLX mode (`{group_size,bits,mode:mxfp4}`), NOT compressed-tensors — rejected.
- **`llm-compressor` is NOT installed** in the oracle venv (self-quantize is a
  fallback only if no checkpoint runs; not needed — a runnable checkpoint exists).

### W1 — the kernel the oracle ACTUALLY runs (parity target)

Traced in the **0.25.0 site-packages** (the runnable oracle), `file:line`:
- `compressed_tensors/schemes/compressed_tensors_w4a4_mxfp4.py`
  `CompressedTensorsW4A4Mxfp4.__init__` → `self.kernel =
  init_mxfp4_linear_kernel()` (`model_executor/kernels/linear/__init__.py:804`).
- `_POSSIBLE_MXFP4_KERNELS[CUDA] = [FlashInferMxFp4LinearKernel,
  MarlinMxFp4LinearKernel, HummingMxFp4LinearKernel]` (`__init__.py:462-466`).
  `init_mxfp4_linear_kernel` returns the **first** whose `is_supported()` is True.
- `FlashInferMxFp4LinearKernel.is_supported` (`mxfp4/flashinfer.py:22-28`):
  `current_platform.has_device_capability(100) and has_flashinfer_cutedsl()`.
  **On GB10 both are True** (device cap `(12,1)`; `has_flashinfer_cutedsl()`
  returns **True** on-box — verified via import). So **the FlashInfer W4A4 kernel
  is selected FIRST; Marlin is never reached.**
- **Parity target = true W4A4 fp4xfp4 GEMM via FlashInfer CUTLASS cute-dsl**
  (`mxfp4/flashinfer.py:apply_weights`): `flashinfer_mxfp4_quantize(x)` quantizes
  the activation to mxf4, then `flashinfer_scaled_fp4_mm(x_fp4, weight, x_scale,
  weight_scale, backend="cute-dsl", block_size=32, use_nvfp4=False)`. Weight scale
  is swizzled + N padded to mult-of-128 in `process_weights_after_loading`
  (`swizzle_mxfp4_scales`).
- **This overrides the row's earlier "Marlin W4A16 fallback" hypothesis for THIS
  box.** Marlin `MxFp4` (`mxfp4/marlin.py`, `weight_global_scale=None`) is the
  **non-Blackwell / cute-dsl-absent** fallback only. Per mirror policy the W4A4
  path is our target on GB10; the W4A16 Marlin path stays the documented fallback
  for sm_80..sm_89 and cute-dsl-absent boxes.
- **Runtime confirmation (QUEUED, GPU-gated):** the oracle logs
  `"Using FlashInferMxFp4LinearKernel for MXFP4 GEMM"` (`__init__.py` `logger.info_once`)
  — grep it in the W0 run; and same-tool nsys to name the cute-dsl GEMM kernel.
- Env overrides that would change the pick (record for the A/B): `--linear-backend`
  != auto (filters the kernel list) and `VLLM_DISABLED_KERNELS` (can disable
  FlashInfer to force Marlin — the exact lever to A/B the two arms on one box).

### W1 EMPIRICAL RESULT (2026-08-05, RUNTIME — supersedes the source-only pick above)

Ran the oracle on `Yi30/Qwen3-8B-MXFP4` on GB10 (evidence:
`docs/bench-evidence/mxfp4-qwen/`). The source trace said FlashInfer; the RUNTIME
says FlashInfer is **selected but CRASHES on sm_121**:
- `init_mxfp4_linear_kernel` logs `Using FlashInferMxFp4LinearKernel for MXFP4 GEMM`
  (is_supported passes: cap 121 >= 100, cute-dsl present), THEN engine start dies with
  `flashinfer.utils.BackendSupportedError: mm_fp4 does not support backend 'cute-dsl'
  with capability 121`. FlashInfer's cute-dsl mxf4 backend covers sm_100 (datacenter
  Blackwell) but NOT sm_121 (GB10). **The default oracle config is non-functional
  for this checkpoint on GB10.**
- The WORKING path = `VLLM_DISABLED_KERNELS=FlashInferMxFp4LinearKernel` -> the next
  supported kernel = `MarlinMxFp4LinearKernel` (W4A16 weight-only fp4 Marlin,
  `apply_fp4_marlin_linear(weight_global_scale=None)`). Greedy golden PYEXIT=0,
  coherent+correct (Paris/Rome/Berlin, 2+2=4, fibonacci) — W0 satisfied.
- **CORRECTION: the GB10 parity target is Marlin W4A16 mxf4, NOT FlashInfer W4A4.**
  This REVALIDATES the row's original Laguna-B2 Marlin W4A16 hypothesis. The
  source-only W1 conclusion above was wrong for sm_121 because it did not model
  flashinfer's RUNTIME backend gate. `is_supported` != actually-runs. Trace the
  execution, not just the dispatch source.

### W2 — native keep-quant compute route (design)

> **REVISED per the W1 empirical result:** the GB10 target is **Marlin W4A16
> mxf4** (weight-only fp4, bf16 activation), NOT the cute-dsl W4A4 GEMM. Route
> through our EXISTING Marlin FP4 infra (`src/vt/cuda/marlin/...`,
> `cuda_marlin_repack.cu`, the NVFP4/AWQ/GPTQ Marlin path) exactly as the Laguna
> B2 route did for NVFP4: extend the FP4 Marlin format plumbing for group-32 E8M0
> (no global scale). The cutlass-W4A4 extension below stays a FUTURE arm, only
> reachable once a flashinfer sm_121 mxf4 backend exists or we write the cutlass
> mxf4 mma directly; it is not today's GB10 bar.

Target = W4A4 mxf4xf4 (GB10) with the Marlin W4A16 mxf4 fallback documented.
Route through the SAME families vLLM uses, mirroring the landed NVFP4 lane:

1. **Reuse the NVFP4 cutlass fp4 tensor-core GEMM** (`src/vt/cuda/
   cuda_matmul_nvfp4_cutlass.cu` + `nvfp4_cutlass_tactics*`). mxf4 differs from
   nvf4 only in the block-scale FORMAT: group **32** (not 16), **E8M0** scale bytes
   (not fp8-e4m3), and **no global scale**. CUTLASS block-scaled fp4 supports both
   (`ScaleVectorSize`/`SFVecSize` 16 vs 32, UE8M0 vs UE4M3 SF dtype) — this is
   exactly flashinfer's `use_nvfp4=False, block_size=32`. mxf4 warp
   `mma.sync ...mxf4nvf4` is consumer-Blackwell (sm_121) available (see
   `no-fa2`/Thor state notes: mxf4 tensor cores are consumer-Blackwell-only, which
   GB10 IS). New: a `vt::MatmulMxfp4Fp4` op (or an `mxf4` mode flag on the nvfp4
   op) + a mxf4 activation-quant emitter (per-token per-32-group amax → E8M0 scale
   → E2M1 pack), mirroring vLLM `flashinfer_mxfp4_quantize`.
2. **Scheme-selection method mirroring `schemes/nvfp4.h`**: add `schemes/mxfp4.h`
   with `Mxfp4W4A4LinearMethod` (+ a `Mxfp4W4A16` arm for the Marlin/weight-only
   fallback) and a `MakeLinearMethod` factory chosen ONCE from the checkpoint (the
   loader probes `weight_packed` + the `mxfp4-pack-quantized` format string), NOT a
   per-call tensor-name probe. Honors the three MUST-route seams
   (LinearMethod/QuantizationConfig policy split, `vt::` op registry gate, shared
   decode runner). Weight staging: MXFP4 packed bytes are a DEVICE-GEMM operand →
   route through `ResidentWeight` (the keep-quant-device-slice rule: a raw host-byte
   view is all-zeros on GB10).
3. **Loader probe**: recognize `format=="mxfp4-pack-quantized"` and the
   `input_activations` presence to distinguish W4A4 (both) vs weight-only W4A16, and
   populate an `Mxfp4Weight` (packed `[N,K/2]` U8 + E8M0 `weight_scale` `[N,K/32]`
   U8, no global). Reuse the `Nvfp4Weight`/`OwnedTensor` residency plumbing.
4. **CPU reference (the gate truth, extends W1 dequant):** add
   `compressed_tensors/mxfp4_emulation.{h,cpp}` mirroring `nvfp4_emulation.*` but
   SIMPLER (no global scales): `RefScaledMxfp4Quant` (activation → mxf4: per-token
   per-32-group amax, E8M0 block scale `2^(floor(log2(amax/6))+127)` clamped, E2M1
   cast+pack), and `RunMxfp4Emulation` (dequant weight via existing
   `DequantMxfp4ToF32` + activation round-trip + f32 matmul). This is the
   software-emulation arm the GB10 fp4xf4 GEMM validates against, exactly as
   `EmulationNvFp4LinearKernel` is for NVFP4.

### W2 code surface (our side, for the continuation) — Marlin W4A16 mxf4

The dense `Qwen3ForCausalLM` (the Yi30 vehicle's arch) IS already in our engine, so
W2 is a quant-lane extension, not a new model:
- **Model + loader:** `src/vllm/model_executor/models/qwen3_weights.cpp`
  `LoadQwen3ForCausalLMWeights` (dense Qwen3 text gate) + `qwen3_dense.cpp`.
- **Scheme detection:** `src/vllm/entrypoints/model_loader.cpp` (compressed-tensors /
  `uses_nvfp4_w4a4()` seam, `:750`) — add the `format=="mxfp4-pack-quantized"` probe
  alongside the nvfp4-pack detection.
- **Weight struct:** mirror `Nvfp4Weight` (`dense_nvfp4_gemm.h`) as an `Mxfp4Weight`
  (packed `[N,K/2]` U8 + E8M0 `weight_scale` `[N,K/32]` U8, NO global). ResidentWeight
  staging (keep-quant-device-slice rule).
- **GEMM:** mirror `MatmulNvfp4W4A16D` (`dense_nvfp4_gemm.h:426`) +
  `MarlinW4A16Enabled` (`:83`) as `MatmulMxfp4W4A16D`, extending the Marlin FP4 repack
  (`src/vt/cuda/cuda_marlin_repack.cu`, `marlin_repack.h`) to consume E8M0 group-32
  scales with no global (vLLM `apply_fp4_marlin_linear(weight_global_scale=None)`).
- **Selection method:** `schemes/mxfp4.h` mirroring `schemes/nvfp4.h`
  (`MakeLinearMethod`), chosen once from the checkpoint.
- **Build:** git-archive this branch to a fresh DGX dir + CUDA build (~1-2 GiB tree,
  fits the 31 GiB free; NOT ~21 GiB — that was a vLLM-oracle-source-tree figure).

### W3 — correctness gates

- **Unit (CPU, buildable off-GPU):** extend `tests/vllm/test_mxfp4_dequant.cpp` (or a
  sibling `test_mxfp4_emulation.cpp`) with the activation-quant + emulated-W4A4
  round-trip vs a double-precision reference; the E8M0-vs-fp8, group-32-vs-16, and
  no-global RED traps carried over. Mirror NVFP4's emulation gate.
- **GPU unit (RED-first):** the `vt::MatmulMxfp4Fp4` GEMM vs the CPU emulation
  reference (near-exact), INCLUDING the **M=1 decode mis-route RED trap** (the
  recorded nvfp4 M=1 class — a new fp4 GEMM path can silently mis-route at batch 1).
- **e2e SACRED greedy golden** vs the oracle on `Yi30/Qwen3-8B-MXFP4`. vLLM's own
  greedy on a dense small model may be bf16-non-deterministic → use the ratified
  DISTRIBUTIONAL gate (ours in vLLM's K-run set) if strict token-exact does not
  hold; verify a bigger dense model strict where feasible.

### W4 — benchmark (Qwen, the binding bar)

`tools/bench/online_gate.py` ours-vs-oracle on `Yi30/Qwen3-8B-MXFP4`, SAME
checkpoint both arms, c1..c8 (>=), 3 reps, single load per arm, idle box, GPU lock
held. Match-or-beat is the bar; record honestly. Note the A/B lever:
`VLLM_DISABLED_KERNELS=FlashInferMxFp4LinearKernel` forces the oracle onto Marlin
W4A16 for a second reference point (W4A4 vs W4A16 on one box).

### Empirical status (2026-08-05)

- **W0 DONE** — checkpoint downloaded (6.18 GB) and the oracle RAN a greedy golden
  (PYEXIT=0, coherent+correct). Golden + evidence in `docs/bench-evidence/mxfp4-qwen/`.
- **W1 DONE** — runtime-traced: FlashInfer W4A4 selected-but-crashes on sm_121;
  Marlin W4A16 is the working GB10 path (see the W1 EMPIRICAL RESULT block).
- **W2 IMPLEMENTED + BUILDS + RUNS; e2e correctness RED (commit `7068dca6`)** — the
  native Marlin-W4A16-mxf4 keep-quant path landed (kernel-gen MXFP4 config +
  regenerated group_blocks=2 instances/selector; `MarlinProcessExpertScalesMxfp4`;
  `MoeMarlinArgs.{group_size,mxfp4}` + launcher branch; `Nvfp4Weight.{group_size,
  is_mxfp4}` + `dense_nvfp4_gemm.h` branch + `MatmulMxfp4W4A16D`;
  `dense_weight_loaders.h` MXFP4 loaders; `qwen3_weights.cpp` detect+load). Clean
  `-Werror` CUDA build on GB10; loads Yi30/Qwen3-8B-MXFP4; dispatches the native
  group_blocks=2 Marlin kernel; runs. **NOT token-exact vs the golden** — a
  DETERMINISTIC, UNIFORM (prefill+decode, NOT graph-related: identical with
  `VLLM_CPP_CUDAGRAPH=0`) numerics error: robust tokens survive (" Paris", "there"
  match) but the rest degenerates. LOCALIZATION: the MXFP4 scale permute is PROVEN
  byte-exact vs vLLM's `mxfp4_marlin_process_scales` (128x256 CPU check, 1024/1024),
  and the fp4 dequant (`dequant_skip_flop=false` bias path) is a faithful lift, so
  the residual is inside the group_blocks=2 Marlin GEMM interaction — a path our
  prior NVFP4-only usage (group_blocks=1) never exercised.
- **W3 e2e — COMPUTE PATH GREEN (async-off token-exact 3/4); async-default degeneration
  ROOT-CAUSED as a PRE-EXISTING non-MXFP4 bug.** With `VT_ASYNC_SCHED=0` our engine's
  native Marlin mxf4 e2e is TOKEN-EXACT vs the golden on 3 of 4 prompts (p1/p2/p4);
  the 4th (open-ended story) diverges after the identical first token — bf16 /
  implementation non-determinism (near-tie/distributional regime). The DEFAULT
  (async on) degenerated, but NOT from MXFP4: the async executor overlaps the prior
  step's output-copy with the forward, and classic dense `Qwen3ForCausalLM` lacks the
  async device-mirror fix (the #31 class wired only for the gate models) — a
  quant-independent, pre-existing classic-dense-Qwen3 async bug (SEPARATE row).
  Evidence: `docs/bench-evidence/mxfp4-qwen/W3-e2e-result.md`. So the MXFP4
  keep-quant COMPUTE is correct (3/4 e2e token-exact + all unit gates below).
- **W3 unit gate — GREEN (`test_ops_moe_grouped.cpp`, commit `8469e333`).** The MXFP4
  Marlin GEMM (`MoeGroupedGemmNvfp4Marlin`, mxfp4 args) vs the INDEPENDENT CPU dequant
  reference (`DequantMxfp4ToF32` + f32 matmul) at K=256,N=128: **max_rel 3.8e-3** at
  M=1 AND M=8 — pure bf16 rounding, NOT a systematic error. **PROVES the MXFP4
  keep-quant compute (repack, E8M0 scale processing, group_blocks=2 dispatch,
  launcher) is correct.** So the e2e degeneration (`7068dca6`) is **NOT the GEMM**.
- **e2e residual localization (as of `e28130ee`) — every MXFP4 component verified
  correct; the bug is NOT in the compute or the loader byte interpretation:**
  - scale permute byte-exact vs vLLM at ALL model shapes (N=6144/12288, K=12288);
  - GEMM unit gate 0.38% at M=1/M=8 (N=128);
  - loader dequant of the REAL layer-0 q_proj = sane+correct weights (min -0.5, max
    0.5, mean|.| 0.02; scales 2^-8..2^-7), shapes match, U8-scale discriminator works;
  - model dispatch (`IsNvfp4()` = `!qkv_proj_fp4.Empty()`) routes MXFP4 correctly;
  - dense `Qwen3ForCausalLM` + NVFP4-W4A16 are known token-exact e2e (model-matrix).
  Remaining suspects, in order: (1) **the group_blocks=2 kernel at LARGE N/K** — the
  unit gate only ran N=128; the extended shapes ({4096,4096},{4096,12288},{12288,4096})
  are committed RUN-PENDING (`e28130ee`); (2) a model-integration subtlety (per-layer
  activation diff vs the oracle to find the first divergent op). NEXT: run the extended
  unit gate; if RED at a model shape → large-shape kernel fix; if GREEN → per-layer
  activation dump vs oracle.
- **W4 NOT REACHED** — bench (`online_gate.py` c1..c8x3, oracle arm MUST set
  `VLLM_DISABLED_KERNELS=FlashInferMxFp4LinearKernel`) is owed once the e2e is green.
- Build recipe (reproducible): git-archive branch to `~/work/mxfp4-w2`, `cmake -B
  build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVLLM_CPP_CUDA=ON
  -DVLLM_CPP_CUDA_ARCHITECTURES=121a -DVLLM_CPP_MARLIN=ON -DVLLM_CPP_TRITON=OFF
  -DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0`, then `ninja -C build vllm-cli` (build
  SPECIFIC targets — bare `ninja` builds 100+ tests that whole-archive libvllm.a =
  28 GiB, blows the disk floor).
