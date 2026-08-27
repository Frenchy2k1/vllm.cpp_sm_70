# W5-fp8w — tp>1 per-rank device-resident FP8 W8A16 tail MLP

Row: `BACKEND-DISTRIBUTED-TP-CLEAN`
Issue: [#2058](https://github.com/mudler/vllm.cpp/issues/2058)
Owner: the tp>1 tail-MLP wall (layers 56–63 of the 27B NVFP4 checkpoint,
`fp8w=1`). Non-narrative; every claim cites a file:line and is either verified
or PENDING the next gate.

## Scope

Make the `tp>1` `DenseMlpBlock` fp8w arm device-resident per rank, mirroring the
already-landed M-B3 fp4 fix (`vt_cuda_mlp_shard_runT_fp4`). Today the arm
(`src/vllm/model_executor/models/qwen3_5.cpp` `DenseMlpBlock`, fp8w branch)
D2H's the resident per-channel fp8 weights (`Fp8PerChannelWeight`: raw e4m3fn
`[n,k]` bytes + `f32 [n]` per-output-column scale), row-parallel host-decodes
them to f32, and uploads f32 gate/up/down slices before the host `runT` +
one AllReduceSum over `[O]`. Measured **~1600–1780 ms/layer** pre-fix (lazy-alloc
fix already shed ~340 ms/layer of shared-vector zeroing on this arm), the
dominant remaining TP>1 step cost once the attention NCCL-churn + fp4 lazy-alloc
fixes landed. Weights already load and run at tp=1; this is **routing, not
format**.

## Already true (verified, do not re-derive)

- M-B3 fp4 device GEMM: `runT_fp4` + `MlpGuActFp4`/`MlpDownFp4` + selfcheck
  (`src/vt/cuda/nccl_communicator.cu`), token-exact on the real 27B serve.
- Attention NCCL-churn fix (retained `static CudaCommGroup`, ~520→3 ms/layer)
  and the fp4 lazy host-vector fix (empty-at-decl, ~340 ms/layer shed) — both
  verified: `test_nccl_group` 20/20 (197,529 assertions) and TP2 serve
  token-exact (`' Paris.\nThe capital of Germany is Berlin.'`).
- fp8w host decode matches `out[n,k] = F8E4M3ToF32(w[n,k]) * scale[n]`, no bf16
  rounding (`qwen3_5.cpp` fp8w arm); `DevF8E4M3ToF32` already in the .cu
  (`:1005`).

## Scope (this row)

1. `MlpGuActFp8w` / `MlpDownFp8w` kernels (mirror `MlpGuActFp4`/`MlpDownFp4`
   scheduling + reduction order, but decode by the per-output-column f32 scale
   only — no group scale, no scale2, **no bf16 rounding**).
2. `vt_cuda_mlp_shard_runT_fp8w` wrapper: process-lifetime group + per-rank
   stream/buffer reuse, per-rank resident fp8 slice GEMM, one AllReduceSum
   over `[O]`, reduced result on rank 0 — byte structure of `runT_fp4`.
3. Per-rank fp8w residency build in `DenseMlpBlock` (thread-per-lane, one-time
   lazy, mirrors the fp4 residency) + fast-path dispatch `fp8w_device`.
4. Model-free selfcheck `vt_cuda_mlp_shard_runT_fp8w_selfcheck` + wiring into
   `tests/vt/test_nccl_group.cpp`.

## Gate (green / measured; else re-open, not tolerance)

- G1 model-free: 2-GPU selfcheck — synthetic fp8 e4m3fn bytes + per-column
  scales (both signs, normal/subnormal exponents), device kernel vs
  host-decode + `runT` reference, within 1e-3.
- G-suite: full `test_nccl_group` 20/20 (197,529 assertions) on GPUs 2,3.
- G5: TP2 serve on 2×V100 (snapshot
  `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108`) byte-identical to the certified
  tp2==tp1 baseline `' Paris.\nThe capital of Germany is Berlin.'`; the fp8w
  tail layers must execute on the device path.

## Risks

- Dtype width: keep the fp8→f32 decode; never silently widen/narrow. A collapse
  to a numerically-better dequant is invisible to the token gate — assert the
  shape/bytes-moved contract, not only tokens.
- Multiplication order must match the host `runT` reference (accumulator +
  tree), not only the value.
- A built-kernel failure faults loudly (return `rc`), never a silent host
  fallback (mirror `runT_fp4`).

## Evidence contract

- Selfcheck red → green (device kernel vs `runT` reference).
- Record build/run recipe, revisions, model snapshot, raw output token stream.

## Git integration

Single pull request (spec + implementation); spec committed before
implementation. Owner: `BACKEND-DISTRIBUTED-TP-CLEAN`.

## Owed

- None downstream. The fp4 body is M-B3 (done). Block-fp8 #1189 and tp1
  fp8-output-dtype #339 are separate, out of scope.