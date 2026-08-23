# TP rollout (step 2/2) — per-rank sharded dense forward

Status: DESIGN (not yet implemented). Primes the continuation.
Depends on (all shipped + green): per-device affinity, in-process NCCL
collectives, `vt::CudaCommGroup` + `vllm::{TensorParallel,TpShard,TpAllReduceSum}`
seam, loader-slice placement, runner-forward primitive, and the retained
engine bridge (`vt_cuda_tp_acquire`/`vt_cuda_tp_release`, 5/5 tests on 4xV100).

## Why it can't be a bolt-on
`Qwen3_5DenseModel::ForwardDevice/Forward` (include/.../qwen3_5_dense.h, impl in
src/vllm/model_executor/models/qwen3_5_dense.cpp + qwen3_5.cpp) is static and
computes the FULL [T,H]->lm_head on ONE queue/device, with the weights whole
(fp4-resident or bf16). Reducing the full logits would quadruple-add. Correct TP
needs per-rank shard + per-boundary reduce.

## Design (touchpoints)

1. **Thread the `tp` param.** Add `const vllm::TensorParallel* tp = nullptr` to
   `Qwen3_5DenseModel::Forward` / `ForwardDevice` / `ForwardDeviceTap`
   (default null => engine unchanged => gates byte-identical).
   Callers that end-to-end wire it: the `LoadedEngine` decode path + the
   in-tree tests (RunQwen27Logits etc.) pass null today.

2. **Per-layer column/row shards.** In `DenseForwardLayers` (the SwiGLU MLP + the
   per-layer attention): when `tp && tp->tp_size()>1`,
   - gate/up: host stage each rank's `TpShard(intermediate)` columns; GEMM on
     rank i's device (brick-GEMM fp16 W-path, already sm70-verified).
   - down/o_proj: row-parallel partial per rank over its input shard, then
     `TpAllReduceSum(&tp, q, out)` on the boundary (the runner-forward
     primitive, proven).
   Keep the M=1/tiny decode path exact — no reduce for world==1 (the existing
   `tp_size()==1` early-return in TpAllReduceSum already guarantees this).

3. **Head**: `lm_head` [vocab] column-shard per rank -> reduce via AllGather
   into the full logits (or AllReduceSum when the shards are disjoint-summed);
   logits buffer on rank 0 is authoritative.

4. **Runner attach**: at `LoadedEngine` construct (or registry `create`),
   `tg = vt_cuda_tp_acquire(VLLM_TP || device_count)`; retain for the model
   lifetime; pass `&TensorParallel{ comm = tp->Rank(r) }` per ranked thread.
   Default VLLM_TP=1 => acquire returns null => the tp1 engine stays
   byte-identical.

## Verification gate (must not regress)
- Run the ENTIRE current suite after the change (5/5 nccl, 7/7 backend,
  5/5 sm70-fa2, 315/315 paged, 241/241 op) — all must stay green with tp1.
- New gate: on the 4xV100, tp=4 == tp=1 tokens for a short real continuation
   (the runner-forward equality, at model level). Mark UNREACHED until that
   gate lands — do not claim tp>1 until it passes.

## Sequence
1. Add `tp` params + null-default (regression-neutral, commits first).
2. Per-rank sharding in DenseForwardLayers (commit per layer-family, gate each).
3. Head + runner attach.
4. tp==tp1 token-equal gate.

## Not in scope here
- Multi-node (ncclCommInitAll is single-process/local); GQA KV-split deferral;
  MoE (A3B) expert TP (separate); tensor/pipeline hybrid.