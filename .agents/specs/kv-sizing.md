# KV-cache auto-sizing — `ROAD-V1-MEM` M0 spike

Row: `ROAD-V1-MEM` (issue [#83](https://github.com/mudler/vllm.cpp/issues/83),
user-directed 2026-08-06). Status: **M1 + M2 LANDED (absolute-bytes knob +
group-aware divisor, CPU); M3 profile run dgx-gated.** Parity pin: vLLM
`555967922` (0.26.0.dev0).

## The gap (we are BEHIND vLLM on this axis)

Today the KV pool is a **raw block count the user types by hand**:
`EngineParams::num_blocks = 256` (`include/vllm/entrypoints/model_loader.h:60`,
beside `block_size = 32` `:59`), exposed verbatim as `--num-blocks N`
(`examples/server/main.cpp`), carried on the C ABI as
`vllm_model_params.num_blocks` at the same 256 default (`src/capi/vllm_c.cpp`),
and landing as `BlockPool(num_gpu_blocks, ...)` which asserts `> 0` and otherwise
TRUSTS it (`include/vllm/v1/core/block_pool.h`, `src/vllm/v1/core/block_pool.cpp`).
So a user must convert "40 GB free, 32k context, concurrency 8" into a block
count themselves, and a wrong guess either wastes VRAM or OOMs mid-run — strictly
worse ergonomics than vLLM, which auto-sizes from a single fraction and errors
pre-flight when the model does not fit.

## What vLLM does (the design to mirror)

Three knobs, all `vllm/config/cache.py`, in override precedence:

1. **`gpu_memory_utilization`** (`cache.py:68`, default **0.92**) — a FRACTION of
   the device's free memory that vLLM is allowed to consume in total (weights +
   activations + KV). The default path.
2. **`kv_cache_memory_bytes`** (`cache.py:182`) — an ABSOLUTE KV-pool size. When
   set it **ignores `gpu_memory_utilization`** (`cache.py:189`): fine-grained
   control for users who know their non-KV footprint.
3. **`num_gpu_blocks_override`** (`cache.py:87`) — a final escape hatch that
   overrides the profiled block count outright.

The auto-size computation lives in the worker's `determine_available_memory`
(`vllm/v1/worker/gpu_worker.py:497-599`):

```
requested_memory      = init_free_memory * gpu_memory_utilization     # the budget
profile_run()  →  non_kv_cache_memory   = weights + peak activation headroom
cudagraph_estimate    = profile_cudagraph_memory()   # only if graphs captured
available_kv_bytes    = requested_memory - non_kv_cache_memory - cudagraph_estimate
num_gpu_blocks        = available_kv_bytes / bytes_per_block          # from kv_cache_config
```

Key properties to carry over:
- The **profile run** is a real forward at the max shape (a synthetic
  batch/seqlen) whose peak allocator watermark measures the non-KV cost — vLLM
  does not estimate weights + activations analytically, it MEASURES them.
- `available_kv_bytes <= 0` is a **pre-flight error** ("model + activations do
  not fit in the requested budget"), not a runtime OOM.
- `bytes_per_block = block_size * num_layers * 2(KV) * num_kv_heads * head_dim *
  dtype_size`, summed group-aware across heterogeneous KV groups (our runner
  already carries per-layer KV geometry: `kv_cache_groups`, the Gemma-4 / Kimi
  het-KV work) — the divisor is NOT uniform for MLA / GDN / sliding-window archs.

## Our design (additive, mirror-faithful, default flips to auto)

W-plan (all CPU/design except the profile run, which is GPU-gated):

**LANDED (M1 + M2, CPU, 2026-08-08):** `EngineParams` gained
`gpu_memory_utilization` (0.92) + `kv_cache_memory_bytes`; `num_blocks` became
the override (default 0 = auto). The C ABI mirrors both at **v16**
(`vllm_model_params.gpu_memory_utilization` / `.kv_cache_memory_bytes`, appended,
zero-value preserves behaviour). `--gpu-memory-utilization` / `--kv-cache-memory`
on `examples/server` + `examples/cli`. `ResolveNumBlocks`
(`model_loader.cpp`) applies the precedence `num_blocks > kv_cache_memory_bytes >
256`; the absolute-bytes branch divides by `KVBytesPerBlock(kv_cfg)`
(`kv_cache_interface.cpp`, M2), a group-aware sum over attention specs
(GDN/Mamba excluded). Gates: `test_kv_cache_interface` KVBytesPerBlock 5/5 (dense
/ MLA / hybrid / het-KV per-layer / divisor), `test_capi` v16 round-trip, full
CPU suites green. **M3 (the util profile run) stays dgx-gated** — until it lands
the util branch falls back to 256, so the default path is byte-identical.

- **M1 — the knobs on our surfaces.** Add to `EngineParams` (and mirror on the C
  ABI `vllm_model_params`, at the next ABI bump — the field is appended,
  zero-value = "auto"): `gpu_memory_utilization` (double, default 0.92),
  `kv_cache_memory_bytes` (int64, 0 = unset), keep `num_blocks` as the
  `num_gpu_blocks_override` escape hatch (0 = auto). `--gpu-memory-utilization`
  and `--kv-cache-memory` on `examples/server` + `examples/cli`; `--num-blocks`
  becomes the override, documented as such. Precedence exactly vLLM's:
  `num_blocks>0` wins, else `kv_cache_memory_bytes>0`, else the util path.
- **M2 — `bytes_per_block` from the runner's KV geometry.** A pure function over
  the already-known per-group `(num_kv_heads, head_dim, dtype, block_size)` —
  group-aware sum, unit-tested against hand-derived bytes for dense (Qwen3),
  MLA (DeepSeek-V4 latent), and het-KV (Gemma-4 per-layer, Kimi KDA+MLA). CPU,
  no device.
- **M3 — the profile run.** A single max-shape forward on the target device that
  reads the peak resident bytes (weights + activation watermark) via the backend
  (`vt::Backend` memory-info seam; on CUDA the allocator high-water, on the
  unified GB10 the host+device unified reading — see the caveat below), then
  `available_kv = free*util - non_kv`, `num_blocks = available_kv /
  bytes_per_block`; pre-flight `VT_CHECK(available_kv > 0)` with the vLLM-shaped
  message. GPU-gated (needs a real load); the M2 divisor + M1 precedence are
  CPU-gate-able first with a synthetic non-KV number.
- **M4 — gate.** Token-exact vs the SAME run with an explicit `--num-blocks`
  equal to the auto-computed count (auto-size changes only the block COUNT, never
  the numerics), plus a pre-flight-error test (util too small → loud error, not
  OOM), plus a memory-headroom assertion (resident stays under the requested
  fraction). Default flips to auto (`num_blocks` unset → profiled), so this is a
  `parity-enablers-ship-as-defaults` flip once the profile run is dgx-verified.

## GB10 caveat (do not skip)

The GB10 119 GiB pool is **UNIFIED** — `gpu_memory_utilization` there reserves
HOST RAM, and a `0.85`-class value has hard-rebooted the box (memory
`gb10-unified-memory-oom-reboots-box`). So: (a) the profile run's free-memory
reading must come from the unified pool, not a device-only counter; (b) the
default 0.92 that is safe on a discrete GPU is NOT automatically safe on the
unified board — the pre-flight budget must leave host headroom, and the M4 gate
must run at a conservative util with `free -g` monitoring, never alongside a big
oracle. The discrete-ROCm / CUDA path uses the device free-memory reading
unchanged.

## Not in this spike

No code. The `bytes_per_block` formula, the profile-run seam, and the ABI field
are DESIGNED here and grounded in the pin; M1-M4 are the implementation rows this
spike unblocks (the row moves `INVENTORIED → READY`). The absolute
`kv_cache_memory_bytes` swap-space / CPU-offload knobs vLLM also carries
(`swap_space`) are OUT of M0 scope — KV auto-sizing on the GPU pool is the
user-facing wart #83 names; offload is a later row.
