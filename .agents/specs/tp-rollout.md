# TP rollout (step 2→3) — per-rank sharded dense forward + stretch
 (paged GQA-exact KV shard, MoE expert TP, Qwen3.8 gate)

Status: step 2 DONE + green (per-rank dense MLP shard, KV shard, lm_head shard,
runner attach, 4-GPU dense tp==tp1 8-token gate reached). Step 3 "stretch"
designed below (GQA-exact paged KV, MoE expert TP, Qwen3.8 loader/capture gate)
and being implemented.
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
1. Add `tp` params + null-default (regression-neutral, commits first). DONE (`2779cf38`).
2. Per-rank sharding in DenseForwardLayers (commit per layer-family, gate each).
3. Head + runner attach.
4. tp==tp1 token-equal gate.

## Phase after Qwen3.5 lands (all Forwards threading)
Once the Qwen3.5 dense (and MoE) TP is token-equal, thread the SAME `tp` through
the other hand-rolled arch `::Forward` classes. This is MECHANICAL plumbing, not
per-class reimplementation (the TpShard/TpAllReduceSum exact suiteet lives in
`tensor_parallel.h` + the shared loaders):
- ~58 `::Forward` implementation files carry `const vllm::TensorParallel* tp = nullptr`
  (default) and pass it into their layer paths that already use the shared
  GEMM/attention ops — the sharding math is already folded in by those ops.
- Verify per arch: tp1 null-default stays byte-identical; tp==tp1 gate per arch
  as it (opt-in) enables TP.

## Not in scope here
- Multi-node (ncclCommInitAll is single-process/local); tensor/pipeline hybrid.

## Step 3 (stretch) — GQA-exact paged KV shard + MoE expert TP + Qwen3.8 gate

Scope: (1) per-rank **paged** KV shard in the `RunDenseLayerPaged` decode path,
GQA-clean; (2) **MoE (A3B) expert TP**, weight-parallel (vLLM default); (3) a
**Qwen3.8-27B-NVFP4** loader/capture gate; then a full no-regression sweep.

### KNOWN DISCREPANCY first (why the shard must be GQA-exact, not gather)

The landed dense tp>1 primitive (`vt_cuda_attn_kv_shard_run`: `AttnKvHostRef`
`nccl_communicator.cu:824`, `AttnKvShardPartial` `:854`) runs `for kh in
[0,Hkv)` summing `exp(q·k)` and `exp(q·k)·v` over ALL kv-heads, then
`out = num/den`. The tp1 `AttentionKernel` (cuda_ops.cu:1468) is standard GQA
(`g = h/(hq/hk)`, one kv-head per q-head). These are different math. Measured on
a 27B-shaped layer (Hq=24/Hkv=4/Dh=256, random normed q/k/v):
`max|gqa − gather| = 4.86`, `frac |Δ| > 1e-3 = 0.998`. The dense 8-token
tp==tp1 gate passes only on greedy-argmax robustness — **weak baseline, not
byte-exact**. Both the gather kernel and its selfcheck reference are internally
consistent (they pass each other) but wrong relative to GQA.

**Adopted rule:** any new attention shard uses the GQA-exact decomposition
below. The all-kv-heads gather loop is NOT copied into the paged path.

### GQA-exact KV shard design (dense fix + new paged primitive)

For each `(t, hq, d0)`: `g = hq / qpk` with `qpk = Hq/Hkv`.
- If `g ∈ [hk0, hk0+per)` (this rank's kv slice): compute the **full
  single-kv-head softmax over positions** `s ≤ key_end[t]` (paged:
  `block_table`-indexed K/V reads; dense: contiguous `[S,Hkv,D]`). No sum over
  kh.
- Else: contribute 0.
- **AllReduceSum** across ranks. Because each q-head's softmax is computed on
  exactly one rank, the reduce gathers a complete output with **no global
  num/den divide**. Token-exact to unsharded `vt::PagedAttention` /
  `vt::Attention` by construction. Full output on every rank → `o_proj` stays a
  full GEMM.

Paged shape (NHD `[num_blocks, block_size, num_kv_heads, head_dim]`, K and V
interleaved per block) is staged to host → `[S,Hkv,D]` per rank. Correctness
first; perf explicitly out of scope for the token-gate milestone.

Hkv divisibility: world must divide Hkv or replicate kv-heads / zero-fill.
27B Hkv=4 → world 2 (per=2); 35B Hkv=2 → world 2 (per=1).

### MLP shard: batched, not per-token (design decision, 2026-08-19)

The landed `DenseMlpBlock` tp>1 path called `vt_cuda_mlp_shard_run` **per
token**: one fresh `CudaCommGroup` + a full O×I×H host double reference per
token per layer. At 27B (O=H=5120, I=17408) that is ~1.7e14 double-FMA per
token per layer — hours per forward. This was the paged-gate "hang"
(localised by markers: the GQA attention branch runs in ~50ms; the stall was
always the MLP).

Design: **one batched call per layer** — `vt_cuda_mlp_shard_run_b(T,O,H,I,
verify, x, gate, up, down, out)` + the `DenseMlpRankPartialBatch` kernel (one
thread per (t,o); rank r owns the intermediate slice `[i0, i0+I/W)`; the
group AllReduceSum over `[T*O]` yields the full result on every rank). One
NCCL group per layer, one kernel launch per rank. `verify=0` (production) is
compute-only: the engine-level tp==tp1 token comparison is the correctness
gate and per-element host-reference parity at production shape is both
redundant and unrunnable. `verify=1` computes a double-precision host
reference (parallelised over (t,o) rows) and enforces the same 5e-4 relative
tolerance — used only by the selfcheck at fresh + 27B-shaped dimensions.
The fp4-resident weights are host-decoded **once per layer** (not per token);
the tp>1 batched path supports the fp4 representation and throws loudly for
bf16/split-Matmul-B resident weights (the synthetic-CPU lane never reaches
tp>1, so nothing regresses).

### MoE (A3B) expert TP — weight-parallel

vLLM's MoE TP default is weight-parallel: every rank holds all `E` experts;
per-expert `gate`/`up` are column-sharded in the intermediate, `down` is
row-sharded; AllReduceSum combines. tp1 keeps the fused WMMA path. Thread `tp`
through the MoE `ForwardDense` (`qwen3_5.cpp:8198` takes no `tp` today); the
paged path currently throws on tp>1 (`8733-8737`).

### Qwen3.8-27B-NVFP4 gate

Loader + smoke/deterministic capture, **no oracle golden**; config identical to
Qwen3.6-27B → port of the 27B capture path. `unsloth/Qwen3.8-27B-NVFP4` snapshot
present in the box HF cache (13 shards, 21.98 GiB).

### Verification for this phase
- `test_nccl_group` (world=2 on GPUs 2,3) incl. the new model-free paged
  GQA-exact selfcheck (synthetic paged cache, host GQA reference) → green.
- 27B paged tp==tp1 token gate (no-Triton path) → green.
- 35B A3B paged tp==tp1 (world=2, Hkv=2) → green.
- Full no-regression sweep: fa2 5/5, backend 7/7, nccl, dense + 35B paged
  engine, dense tp==tp1.
- Mutation proof: delete the paged production call site in a scratch copy →
  gate must go red; restore byte-for-byte.

## Owed
- **Fix the dense tp>1 attention from gather to GQA-exact.** Real latent
  correctness bug (see KNOWN DISCREPANCY). Cannot file a GitHub issue from this
  box (no `gh`, no tokens) → recorded in `.agents/issue-index.md`; highest
  existing issue is #1152.
- Merge `origin/main` (63 ahead / 61 behind, 3 conflicts) + repair the branch's
  record debt (11 empty-body `docs:` commits, 4 oldest sm70 commits missing
  trailers, 8 doc-checkpoints) before landing — deferred to landing time; the
  branch is local-only (never pushed), so the history rewrite is safe.