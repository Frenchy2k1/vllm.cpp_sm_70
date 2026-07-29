# fp8 KV cache (`cache_dtype=fp8*`) — spike + W1 (`KV-FP8`, `QUANT-KV-FP8`)

Rows: `KV-FP8` (engine-matrix, KV cache and memory) and `QUANT-KV-FP8`
(quantization-matrix). HIGH-priority feature gap #5
([vllm-feature-gap-analysis.md](vllm-feature-gap-analysis.md)): the standard
memory/throughput lever that halves the KV footprint by storing K/V as fp8 with
a per-tensor dequant scale. Claim `CLAIM-KV-FP8`.

Pinned oracle vLLM `555967922` (0.26.0.dev0) at `/home/mudler/_git/vllm`; every
anchor below is `file:line` in that tree. REUSE the landed fp8-e4m3 codecs (no
re-port).

## Scope

- **In (this spike + W1):** the vLLM fp8-KV surface (the `cache_dtype`
  config, `BaseKVCacheMethod` k/v scale handling, the fp8 store in
  `reshape_and_cache*`, the fp8 dequant on the attention read, the calibrated
  vs checkpoint scale, and the halved-block memory accounting); and the first
  CPU-buildable brick: an fp8-e4m3 K/V **store** (`vt::ReshapeAndCacheFp8`) +
  the fp8 **read** dequant in CPU paged attention + the `cache_dtype` config
  parse (`vllm::v1::ParseCacheDType`), all unit-gated RED-first.
- **Out (named later bricks):** the CUDA fp8-KV store kernel and the CUDA
  fp8-KV paged-attention read (the GPU is the memory-halving e2e), fp8_e5m2 CPU
  compute, per-attention-head scales, the full engine-runner integration
  (half-sized KV blocks in the real runner + checkpoint `k_scale`/`v_scale`
  threading + `--kv-cache-dtype`/`--calculate-kv-scales` CLI), and the vendor
  KV dtypes (`fp8_inc`, `fp8_ds_mla` — `QUANT-KV-FP8-VENDOR`) and turboquant /
  nvfp4 / per-token-head KV (`KV-NVFP4-TURBO`).

## Upstream chain

The vLLM fp8-KV path is NOT delegated to a dependency (unlike the GEMMs) — the
store/read kernels are vLLM's own csrc:

- **Config.** `CacheDType` Literal (`vllm/config/cache.py:19-36`) —
  `auto`/`fp8`/`fp8_e4m3`/`fp8_e5m2` (+ `fp8_inc`/`fp8_ds_mla`/`turboquant_*`/
  `int*_per_token_head`/`nvfp4`); `cache_dtype: CacheDType = "auto"`
  (`:76`, "if auto, use model dtype; fp8 == fp8_e4m3"); `calculate_kv_scales`
  (`:99-104`, dynamic on-the-fly k/v scale when the checkpoint has none).
  `is_quantized_kv_cache` (`vllm/utils/torch_utils.py`) = `cache_dtype != "auto"`.
- **Scale handling.** `BaseKVCacheMethod` (`kv_cache.py:42`) adds `_k_scale`/
  `_v_scale`/`_q_scale`/`_prob_scale` to the Attention layer;
  `process_weights_after_loading` (`:108-191`) loads per-tensor scales from the
  checkpoint (`k_scale`/`v_scale`), enforces **per-tensor only** ("Only support
  per-tensor scaling factor for fp8 KV cache", `:145-148`), defaults to 1.0 when
  absent (with the uncalibrated warning `:169-174`), and duplicates a single
  `kv_scale` to both. `KVCacheScaleParameter` (`:17-38`) is the scalar `()`/`(1,)`
  loader.
- **Store.** `reshape_and_cache_flash` (`csrc/libtorch_stable/cache_kernels.cu:
  746`) → `reshape_and_cache_flash_kernel` (`:314-401`); the fp8 branch is
  `CopyWithScaleOp` (`:241-252`), `dst = fp8::scaled_convert<cache_t, scalar_t,
  kv_dt>(src, scale)`. The scale convention is documented at
  `csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh:296-300`:
  **`FP8 = Quantize(HP / scale)`; `Dequant(FP8) * scale = HP`.** `k_scale`/
  `v_scale` are `[1]` (per-tensor) or `[num_heads]` (per-head, `kv_scale_stride`,
  `:365-401`). Store dtype `cache_t = uint8_t`; the fp8 *interpretation* is the
  `Fp8KVCacheDataType` template param (`csrc/attention/dtype_fp8.cuh:9-13`).
- **Read (dequant).** The fp8 attention read multiplies back by the scale:
  `scaled_vec_conversion<float, uint8_t>` (`quant_utils.cuh:302-308`) =
  `half_to_float(fp8_to_half(byte)) * scale`. Consumed by the FA/flashinfer fp8
  paths and the reference `test_cache.py`'s `convert_fp8`.
- **Memory accounting.** An fp8 KV element is 1 byte vs bf16's 2 → the KV block
  is half-sized; `AttentionSpec.page_size_bytes` (`kv_cache_interface.py:380-398`)
  derives from the storage dtype, so `num_gpu_blocks` (profiled) roughly doubles.

## Our baseline

The vt runtime already has: `vt::ReshapeAndCache` + the CPU paged-attention read
(`src/vt/cpu/cpu_cache.cpp`, `src/vt/cpu/cpu_paged_attn.cpp`) over the NHD
unbind-slice cache; the fp8-e4m3 codecs, landed twice and bit-identical —
`vllm::F8E4M3ToF32`/`F32ToF8E4M3`
(`src/vllm/model_executor/model_loader/nvfp4_dequant.cpp:11`,
`src/vllm/model_executor/layers/quantization/compressed_tensors/nvfp4_emulation.cpp:14`)
and the vt-layer in-file copies `Fp8ToF32`/`F32ToFp8` (`src/vt/cpu/cpu_ops.cpp:
418-460`, written because "vt does not depend on vllm"); and the storage-dtype
resolver seam `ResolveKvCacheDType` (`include/vllm/v1/kv_cache_dtype.h`). W1
adds the canonical vt-layer fp8-KV codec home (`include/vt/fp8_kv.h`, reusing —
not re-porting — that e4m3 math, and the future consolidation target for the two
existing copies), the fp8 store op, the read dequant, and the config parse.

## Port map

W1 (this change; CPU-only, `-Werror`):
- `include/vt/fp8_kv.h` (NEW) — `Fp8KVCacheDataType` enum (mirror
  `dtype_fp8.cuh:9-13`) + `F8E4M3ToF32`/`F32ToF8E4M3`/`StoreKvFp8E4M3`/
  `LoadKvFp8E4M3` (bit-match the landed codecs; the store/load scale convention
  from `quant_utils.cuh:296-308`).
- `include/vt/ops.h` — `OpId::kReshapeAndCacheFp8`, `ReshapeAndCacheFp8Fn`,
  `vt::ReshapeAndCacheFp8` decl, and the additive `PagedAttentionArgs`
  `kv_cache_dtype`/`k_scale`/`v_scale` fields (default kAuto/1.0 → every existing
  caller byte-identical).
- `src/vt/ops.cpp` — the `ReshapeAndCacheFp8` wrapper (validate + dispatch) and
  the `PagedAttention` fp8 dtype guard (kI8 cache + scales when != kAuto).
- `src/vt/cpu/cpu_cache.cpp` — `ReshapeAndCacheFp8Kernel` (store =
  `Quantize(hp / scale)`; mirror `reshape_and_cache_flash_kernel` fp8 branch).
- `src/vt/cpu/cpu_paged_attn.cpp` — the read-side dequant
  (`Dequant(fp8) * k_scale|v_scale`) when `args.kv_cache_dtype != kAuto`.
- `include/vllm/v1/kv_cache_dtype.h` — `ParseCacheDType` + `IsQuantizedKvCache`
  (mirror `CacheDType` + `is_quantized_kv_cache`).
- `tests/vt/test_ops_fp8_kv_cache.cpp` (NEW) + its `tests/CMakeLists.txt` line.

Later bricks: the CUDA fp8 store + fp8 paged-attention read (GPU memory-halving
e2e); the runner/spec integration (half-sized blocks + checkpoint scale
threading + CLI); fp8_e5m2 CPU compute; per-head scales.

## Tests to port

- `vllm/tests/kernels/attention/test_cache.py::test_reshape_and_cache` (the
  `kv_cache_dtype == "fp8"` branch, `:97-165`) → the store→dequant round-trip
  within `atol=0.001, rtol=0.1` — LANDED as `test_ops_fp8_kv_cache.cpp`
  (round-trip + fp8-vs-bf16 baseline + the paged-attention e2e). The `"fp8"`
  parametrization also covers e5m2 in `KV_CACHE_DTYPE`; the e5m2 half is
  SKIPPED-with-reason (refused as a later brick) by the `refuses e5m2` case.
- `vllm/tests/kernels/attention/test_cache.py::test_reshape_and_cache_flash`
  (the flash-layout fp8 + per-head/nvfp4 cases, `:180-260`) — the per-head and
  nvfp4 scale-type arms are later bricks (per-tensor is W1); named here.

## Gates

- **Correctness (W1, CPU):** `test_ops_fp8_kv_cache` — 8 cases / 511 assertions
  GREEN: store→read within the e4m3 band, fp8 tracks the bf16 KV baseline
  (NMSE < 1%), paged-attention over the fp8 cache matches the bf16-cache output
  within 5%, `ParseCacheDType` mirrors the CacheDType surface. RED-first proven:
  a wrong store direction (`hp * scale`) fails 3 cases / 480 assertions; a wrong
  read `v_scale` diverges > 0.05 from the baseline; an auto (no-dequant) read of
  an fp8 cache is refused. No sibling regressions (reshape 12/12, paged 14/14).
- **Later:** the CUDA fp8 store + read parity vs this CPU reference; the real
  memory-halving e2e (KV blocks ~2× on a gate model at token parity) is the
  binding gate and is DGX-blocked (docs/BENCHMARKS PENDING).

## Dependencies

None to land W1 (reuses the vt-layer fp8-e4m3 math + the existing NHD paged
cache). Downstream: the CUDA kernels need the GB10 lane; the memory-halving e2e
needs the runner/spec integration + a gate model; per-head scales and the
vendor/turbo/nvfp4 KV dtypes are separate rows.

## Work breakdown

| Brick | Content | State |
|---|---|---|
| W0 | this spike | DONE (this commit) |
| W1 | CPU fp8-e4m3 store + read dequant + config parse + unit gate | DONE (this commit) |
| W2 | CUDA fp8-e4m3 store + fp8 paged-attention read (parity vs W1) | later |
| W3 | runner/spec integration: half-sized KV blocks + checkpoint k/v_scale threading + `--kv-cache-dtype`/`--calculate-kv-scales` | later |
| W4 | memory-halving e2e on a gate model (the binding gate, DGX) | later |
| W5 | fp8_e5m2 CPU+CUDA compute; per-attention-head scales | later |

## Risks/decisions

- **Storage as `DType::kI8`, interpretation as a separate enum.** vLLM's kernel
  carries `cache_t = uint8_t` + the `Fp8KVCacheDataType` template param; we mirror
  that exactly (the dtype.h "the byte never guesses its semantic type" rule,
  `include/vt/dtype.h:20-32`), which avoids adding fp8 to the `DType` enum (and
  its `-Wswitch` blast radius across every backend) while staying faithful.
- **Additive op, not a signature change.** `ReshapeAndCacheFp8` is a NEW op, not
  a widened `ReshapeAndCacheFn` — changing the shared alias would break the
  CUDA/Metal registrations' `static_cast` (touching Metal, forbidden). The read
  side rides additive default-inert `PagedAttentionArgs` fields, so every float
  caller is byte-identical.
- **Codec reuse, not a third copy.** `include/vt/fp8_kv.h` is the canonical
  vt-layer fp8-KV codec home; it is bit-identical to the landed
  `vllm::F32ToF8E4M3`/`F8E4M3ToF32` and to the `cpu_ops.cpp` in-file
  `F32ToFp8`/`Fp8ToF32`, which a later cleanup should consolidate onto it.
- **e5m2 refused, not silently mis-stored.** The config parses `fp8_e5m2` (full
  mirror of the surface) but the CPU compute refuses it with a named-later-brick
  reason — shipping an fp8-e5m2 codec unvalidated against the (offline) oracle
  would violate RED-first.
- **Honest residual.** W1 is a correctness brick; the real *memory/throughput*
  win (the point of the feature) is the GPU store/read + the halved-block runner
  integration, both DGX-blocked and named W2-W4.
