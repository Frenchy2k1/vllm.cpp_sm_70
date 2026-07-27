# ENG-ASYNC-SCHED W4 — discrete-CUDA device-resident sampled tokens

Row: `ENG-ASYNC-SCHED` W4 (engine matrix). Prerequisite W3 is landed and
default-ON. This spike scopes the remaining leaf: making the async-scheduling
overlap real on a **discrete** CUDA GPU, where W3 currently degrades to a
synchronizing host path.

## The defect this closes

W3 landed the device combine/scatter kernels
(`src/vt/cuda/cuda_combine_tokens.cu`) and wired them into the runner, but both
call sites are gated on `is_integrated_gpu()`:

- `src/vllm/v1/worker/gpu/runner.cpp:821` — device combine, else host combine.
- `src/vllm/v1/worker/gpu/runner.cpp:1763` — device scatter, else host scatter
  preceded by `vt::GetBackend(dev.type).Synchronize(queue_)`.

The gate is correct as written: the kernels take the runner's **host**
`std::vector` buffers directly, which is only legal where the platform reports
pageable device access (GB10's UMA). A discrete GPU answers `false` and takes
the host branch, so:

1. the host must know step N's sampled token before it can build step N+1's
   `input_ids`, which forces a full-device `Synchronize` inside
   `sample_tokens_async`; and
2. the depth-2 `step_with_batch_queue` loop (`src/vllm/v1/engine/core.cpp:115`)
   therefore cannot overlap anything — the scheduler is async, the runner is not.

Measured on the local RTX 5070 Ti (`is_integrated_gpu()` == false), 2026-07-25,
attribution-complete node-mode nsys: 497 `cudaStreamSynchronize` calls following
the 256-byte sampled-ID D2H, **20.975 s total, 42.20 ms/call**, against a binding
mean TPOT of 43.72 ms. The matched vLLM trace instead waits on sampled output via
`cudaEventSynchronize` (1,342 calls) AFTER launching the next batch. Evidence:
`docs/bench-evidence/qwen35-4b-main-repair-20260725.md`.

## What upstream does (the mirror obligation)

Read from `${VLLM_SOURCE}` (installed release tree, vLLM 0.24.0 — cited as
installed-release source, NOT as the parity pin):

- `vllm/v1/worker/gpu/states.py:64` —
  `self.last_sampled_tokens = torch.zeros(max_num_reqs, 1, dtype=torch.int64,
  device=device)`. It is a **GPU tensor**, unconditionally, on every platform.
  There is no integrated/discrete branch upstream at all.
- `vllm/v1/worker/gpu/input_batch.py:296-360` — the combine kernel loads
  `last_sampled_tokens_ptr + req_state_idx` on device and stores into the device
  `input_ids`.
- `vllm/v1/worker/gpu/input_batch.py:449-473` — the post_update kernel stores the
  freshly sampled id back into `last_sampled_tokens_ptr + req_state_idx` on
  device.
- `vllm/v1/worker/gpu/states.py:132` `remove_request` — **upstream never
  condenses**. A finished request's slot goes onto `free_indices` and is reused;
  the req_state index is stable for the request's lifetime, which is exactly why
  the GPU tensor never has to be permuted, and why `idx_mapping` (batch row ->
  req_state) exists.

So the discrete path is not a new design; it is the upstream design, which our
W3 leaf implemented only for the UMA case.

## The one real complication: our batch condenses

`InputBatch::condense()` (`src/vllm/v1/worker/gpu/input_batch.cpp:554`) moves the
last live row into a freed slot (`last_sampled_tokens[empty] =
last_sampled_tokens[last]`, line 641), and `swap_states` (line 737) swaps two
rows. Upstream's free-index pool has neither. With a device-resident buffer the
host no longer holds the values it would need to perform those moves, and reading
them back would reintroduce the synchronize this row exists to delete.

Resolution: keep the moves on the HOST as bookkeeping, but record them and replay
them ON DEVICE in stream order. `InputBatch` gains an ordered pending-op log
(seed on `add_request`, move on `condense`, swap on `swap_states`); the runner
drains it each step and applies it with one small kernel BEFORE the combine. The
ops are host-known (indices, and a host-known value for the seed), so no device
read is needed, and stream ordering makes the replay exact rather than racy.
Rejected alternative: put `last_sampled_tokens` in pinned mapped host memory so
the existing kernels work unchanged — the host `condense` read of a value the
device wrote is then an unsynchronized read, which is precisely the
removal/condensation hazard, and it would be latent rather than loud.

## Work breakdown

- **W4a** Runner-owned persistent device buffers sized to the batch bound:
  `last_sampled` [max_num_reqs], `prefill_len` [max_num_reqs], `query_start_loc`
  [max_num_reqs+1], `seq_lens` [max_num_reqs], `input_ids`
  [max_num_batched_tokens], plus a pinned host staging buffer so the per-step H2D
  is a real async copy rather than a pageable staging copy.
- **W4b** `InputBatch` pending-op log + the device replay kernel
  (`LaunchApplyLastSampledOps`), with RED-first unit coverage of seed/move/swap
  ordering.
- **W4c** `ModelForwardInput::device_token_ids` (default `nullptr`, so every
  other model and the whole non-async path is byte-identical) honored by the
  Qwen3.5 dense/MoE `EmbedInto`, so the forward embeds the DEVICE ids the combine
  patched instead of a host vector.
- **W4d** Flip both `is_integrated_gpu()` gates to select
  UMA-in-place / discrete-device-buffer rather than device / host, and delete the
  `Synchronize` on the discrete branch.
- **W4e** Remove the other per-step barrier: `EmbeddingKernelCuda`
  (`src/vt/cuda/cuda_ops.cu:638-669`) does a `cudaMalloc` + `cudaMemcpyAsync` +
  **`cudaStreamSynchronize`** + `cudaFree` for its out-of-range flag on EVERY
  call (531 calls, 12.6 ms total in the same trace). That is negligible while the
  engine is serialized and becomes a hard barrier the moment it is not, so W4
  is not measurable until it is gone. Fix: a persistent per-device flag plus a
  deferred check (read the previous call's flag through a completed event), which
  keeps the loud failure at the cost of at most one step of latency. **W4e must
  land before the W4 A/B, or the A/B measures the wrong thing.**

## Gates

- Correctness: `test_qwen35_plain_weights --no-skip` 3/3 unchanged; the CPU-tier
  combine/scatter/input-batch suites unchanged; a new RED-first unit test for the
  pending-op replay.
- Identity: token-for-token identity across `{default, VT_ASYNC_RUNNER=1,
  VT_ASYNC_RUNNER=0}` on the 128-request benchmark corpus, including a run whose
  requests finish at staggered lengths so `condense` actually fires.
- Speed: same-binary A/B under one `flock /tmp/gpu`, >=3 repetitions, against the
  matched vLLM arm on the identical corpus. The target is the TPOT axis; TTFT
  must be watched (upstream pays a TTFT premium for async scheduling).
- No 4B result implies anything about the 27B/35B release gates.

## Status

`SPIKE` — scoped here, not implemented. The 2026-07-27 post-pull revalidation
must land first, because it re-measures the denominator this row is aimed at.
