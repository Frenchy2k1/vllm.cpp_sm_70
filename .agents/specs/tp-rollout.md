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
row-sharded; AllReduceSum combines. tp1 keeps the fused path byte-identical.

**Wire chain (corrected).** The MoE paged forward takes **no `tp` today and
would silently run tp=1** — `ForwardLayers` (`qwen3_5.cpp:8019`), `ForwardBody`
(`:8182`) and `Forward` (`:8197`) carry no `tp` parameter; `MoeBlock`
(`:6943`) / `RunLayerPaged` (`:7454`) take none either. The
"tp>1 per-rank sharding not yet landed" throws at `:9656-9708` are the DENSE
legacy `Qwen3_5DenseModel::ForwardDevice*` guards, **not** the MoE path — the
earlier draft of this section mis-attributed them. Mirror the dense chain
(`DenseForwardBody:9092` → `DenseForwardLayers:8954` → `RunDenseLayerPaged`
→ `DenseMlpBlock:7183` hook): thread `tp` through `Forward`/`ForwardBody`/
`ForwardLayers` → `RunLayerPaged` → `MoeBlock`, and put the tp>1 shard inside
`MoeBlock` (`:6943`) exactly where `DenseMlpBlock` has its hook.
`RunLayerPaged` ALSO passes `tp` to `FullAttnBlockPaged` (the 10 full-attn
layers take the same per-rank GQA KV-head shard the 27B dense gate already
exercises; GDN layers and the MoE router/shared-expert/combine replicate on
every rank), mirroring `RunDenseLayerPaged`'s `FullAttnBlockPaged(tp)` +
`DenseMlpBlock(tp)` split.

**Host-bytes prerequisite.** The production 35B MoE path is
`MoeBlockFusedMarlinCuda` (`MarlinMoeEnabled()` default ON); its resident build
`BuildMoeMarlinResident` (`:6413`) **releases the host fp4 bytes**
(`w.expert_*_fp4[].packed.ReleaseHost()`, `:6609-6614`) when
`VT_MOE_HOST_FREE` is ON (default). The tp>1 shard host-decodes from those
bytes (`vt_host_decode_nvfp4_f32`), so the gate must run with
`VT_MOE_HOST_FREE=0` (a documented same-binary A/B toggle: it retains the host
copies, device compute unchanged). tp1 keeps the Marlin path regardless.

**Cost model (measured, 2026-08-21).** The prior "5.4 h infeasible" verdict is
superseded, and the *single-thread* "0.805 s/tensor conservative" figure is
retired by measurement: `vt_host_decode_nvfp4_f32` parallelizes the row strip
9× (1.14 s single-thread → **0.126 s per full `[E,I,H]` tensor at 48 threads**;
row-strip decode is memory-bandwidth bound and does parallelize across the
E·I rows). Per MoE layer (gate `[E·I,H]` + up `[E·I,H]` + down `[E·H,I]`) ≈
**0.378 s**, ×40 layers ×9 forwards ≈ **136 s** decode (single-thread worst
case ≈ 465 s — still in budget). Decode re-runs every forward (a 40×3.14 GB ≈
125 GB host f32 cache is rejected, matching the dense 27B's rejected 68 GB
cache).

**H2D is NOT a lever here (measured, not calibrated).** Each rank's f32 shard
is the FULL expert set (the kernel strides by the full I and reads its slice):
`[E·I,H]×2 + [E·H,I]` = **3.00 GB/rank**. Measured H2D at that size is
**7.5 GB/s (pageable ≈ pinned, both ≈ 0.43 s)** — the 27B's "~430 MB/s" was a
mis-calibration, not a real pageable rate. Per-rank H2D ≈ 0.43 s/layer,
comparable to the 0.38 s decode; ×40×9 ×2 ranks ≈ 326 s. The naive per-call
pageable H2D design therefore costs ≈ 136 s decode + 326 s H2D + a small
per-rank f32 GEMM ≈ **~500 s**, well inside the 3600 s clamp. **No
device-residency lever is required** — the earlier "GO conditional on a
device-resident per-rank shard" was premised on the 430 MB/s figure; at the
measured 7.5 GB/s the simple per-layer decode + per-layer H2D design fits.
The end-to-end wall is still validated by the real 35B paged gate.

**Design (selfcheck-first).** Generalize the proven batched dense shard
(`vt_cuda_mlp_shard_run_b` `:766`, `DenseMlpRankPartialBatch` kernel `:650`)
to per-pair expert indexing:
- `MoeMlpRankPartialBatch(x[P,H], exp[P] i32, gate[E,I,H], up[E,I,H],
  down[E,H,I] f32, out[P,H], per_i, i0, I, E)` — one thread per (p,o); rank r
  owns the intermediate slice `[i0, i0+per)` of **every** expert; `e=exp[p]`
  selects the per-expert weight base; partial `[P,H]`; the group AllReduceSum
  over `[P*H]` yields the full expert_out on every rank.
- `vt_cuda_moe_shard_run_b(P,H,I,E,verify,x,exp,gate,up,down,out)` — spawns W
  rank threads (`NcclDevScope`), each H2D's the per-layer full expert-set f32
  buffer to its device (7.5 GB/s, measured), launches its intermediate slice,
  AllReduceSum. `verify=0` (production) is compute-only; `verify=1` computes a
  double-precision host reference (parallel over (p,o)) and enforces the 5e-4
  relative tolerance — selfcheck only (the reference is O(P·H·I·H) and
  unrunnable at production shape, same as the dense `verify`).
- Model hook in `MoeBlock` (`:6943`): when tp>1, host-decode the full expert
  set (requires `VT_MOE_HOST_FREE=0`), H2D the per-layer full f32 buffer to
  every rank, produce `expert_out [P,H]` via the shard, then the **UNCHANGED**
  router (`MoeRouterLogits`+`MoeRouterTopK`) and `MoeCombine` (shared +
  weighted). Replaces only the grouped-GEMM launches
  (`MoeBlockFusedCuda:6382-6389` / the Marlin pair); router and combine run
  on-device as in the fused path, replicated on every rank.
- **Model-free selfcheck first** (`vt_cuda_moe_shard_selfcheck`, wired into
  `test_nccl_group`): random `[P,H]` x + `[P]` exp + random expert weights at a
  representative MoE shape (E=16, I=128, H=96, P=32, world=2 — enough pairs
  over enough experts to prove per-pair indexing + intermediate slicing +
  AllReduce without the 3.14 GB production H2D), `verify=1` parity vs the
  double reference. Red-first: the selfcheck is red until the kernel is
  correct, green after. The real-shape correctness is then the 35B paged gate.

**Acceptance (this phase):** 35B A3B paged tp==tp1 token gate (world=2,
Hkv=2, `CUDA_VISIBLE_DEVICES=2,3`, `VT_MOE_HOST_FREE=0`), same-binary A/B,
device compute unchanged; then the full no-regression sweep (fa2, backend,
nccl, 27B dense tp==tp1, 35B paged engine) + the reachability mutation
(delete the `MoeBlock` tp>1 call site in a scratch copy → gate red).

**Outcome (measured 2026-08-21, GPUs 2,3, V100 32 GB, world=2, Hkv=2 →
per-rank 1 KV head).** The 35B A3B paged tp==tp1 token gate is **GREEN**:
`2-GPU == tp1` (9 tokens `6511 314 9564 369 19241 13 198 760 6511`), 1 passed
| 0 failed, **wall 3205.89 s (~53 min)**. The tp>1 expert shard (f32
host-decode weights → per-pair expert GEMM → NCCL AllReduceSum over `[P,H]`,
the intermediate `I` sliced per rank) reproduces the tp1 Marlin arm
TOKEN-FOR-TOKEN over all 40 MoE layers, so the **replicate-vs-shard fork
resolves in favor of SHARD** — no fallback to a replicated MoE is needed, and
the MoE expert shard is a first-class TP path, not a perf-only approximation.
The first (prefill) argmax `6511` matches the 27B dense gate's first token;
the divergent decode tail (`198 760 6511` vs the 27B's `271 248068 271`) is
expected — a different model family, not a divergence signal (the gate is
tp==tp1 within the 35B, not 35B==27B). Note: the decode runs are
prefill+8 at `VT_MOE_HOST_FREE=0`; the wall is decode-host-decode dominated
(~0.378 s/layer MoE decode × 40 × 9 forwards × 2 arms), not GPU-bound.

**MoE reachability mutation (measured 2026-08-21, GPUs 2,3).** The tp>1
expert-shard call site in `MoeBlock` (`qwen3_5.cpp`) was mutated in the
canonical working tree, rebuilt, re-gated, and restored byte-for-byte (git
verified the mutation line absent after restore):
- **Value-corrupting variant goes RED**: `ob[0] += 1000.0f` after the
  `vt_cuda_moe_shard_run_b` call (corrupting the reduced `expert_out[P,H]`
  row 0 before the shared-expert + `MoeCombine` consume it) flips the tp>1
  arm — `REQUIRE(t1 == tw)` fails
  ({6511,314,9564,369,19241,13,198,760,6511} vs {6511,1305,1305,1305,1305,
  1305,1305,1305,1305}), EXIT=1, 1 failed / 0 passed. The first (prefill)
  argmax still matches; the corrupted pair-0 expert contribution diverges
  decode from step 2 as the paged state accumulates — proving the shard
  output is load-bearing (the gate measures the capability, not a class).
- **Restore is GREEN**: rebuilt clean, gate re-ran to the known 9 tokens
  (the bg_1 green on the byte-identical pre-mutation source).


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

### Performance levers (token-exact; landed 2026-08-19)

The paged tp>1 27B gate ran ~86 min. Per-layer (GDN/MLP) cost split into two
**single-threaded, weight-derived** host loops plus a GPU span:
fp4→f32 decode (~4.7 s, `vt_host_decode_nvfp4_f32`) + masked `gu`/`dmask`
weight-buffer build + pageable H2D (~3.5 s, inside `vt_cuda_mlp_shard_run_b`)
 GEMM+AllReduce (~1 s). The decode and build loops are each **token-exact to
parallelize**: every output element is written exactly once by deterministic
per-element math (LUT + sign + scale, bf16-RNE round / masked copy) with no
cross-element accumulation, so any row partition is bit-identical.

Landed levers (scheduling/caching only — the f32 values feeding the same
kernel are unchanged, so the paged tp==tp1 token gate must still match):
- **A: parallel fp4 host decode** in `vt_host_decode_nvfp4_f32` — row-partition
  the N rows across `min(ncpu, N)` host threads (`std::thread`, same idiom as
  the verify reference). 1 thread → N threads, ~N× on the 4.7 s decode.
- **B: hoisted + parallel `gu`/`dmask` build** in `vt_cuda_mlp_shard_run_b` —
  the full merged `[2I,H]` gate/up and `[O,I]` down are built ONCE on the main
  thread, in parallel over I rows, before the rank loop, instead of serially
  inside each rank worker. The full mask equals the union of the per-rank
  slices (rank r reads its `[i0, i0+per)` rows, strides by the full I), so the
  per-rank result is unchanged. Each rank worker now only H2D-copies (reading
  the shared host buffers is safe: read-only) + launches + reduces.

Rejected / deferred (recorded, not implemented):
- **Decode-once host cache**: a full f32 cache is ~68 GB host (64 layers ×
  ~1.07 GB) — memory-prohibitive on this 125 GB shared box; an LRU < 64 layers
  has ~0 hit-rate against the sequential 64-layer sweep. Parallel-per-layer
  (A/B) gives the same ~N× win with no 68 GB.
- **Persistent `CudaCommGroup`**: saves only the ~16 ms `ncclCommInitAll`
  per call (~2 s over the whole run) for a signature thread-through across the
  primitives. Poor ROI; the GEMM (HBM-bound on weight re-reads, ~1 s/layer
  floor at 27B T=1) dominates the remainder.

Verification for the perf change (measured 2026-08-20, GPUs 2,3):
- `test_nccl_group` (host-decode known-nibble selfcheck + fresh + 27B-shaped
  batched f32 parity, world=2) green: 16/16 cases, 2092/2092 assertions, ~23 s.
- 27B paged tp==tp1 token gate still matches the known tokens
  (6511 314 9564 369 19241 13 271 248068 271): **wall 2935 s (49 min) vs the
  5139 s (85.7 min) marker-span baseline** (both `[pagedtp]`/wall on the same
  2x16 GB V100 pair, same binary family) — the ~8.3 s GDN/MLP host span is
  now overlapped, the residual floor is the HBM-bound f32 GEMM weight
  re-reads. Non-regression: dense tp==tp1 gate green (112 s), `test_nccl_group`
  green; only `nccl_communicator.cu` (levers A+B) and these records changed.

### Mutation proof (measured 2026-08-20, GPUs 2,3)

The paged tp>1 attention call site (`FullAttnBlockPaged` →
`vt_cuda_attn_gqa_shard_run`, qwen3_5.cpp) was mutated in the canonical
working tree, rebuilt, re-gated, and the tree restored byte-for-byte (git
blob-verified before and after):
- **Guard-disabled variant is VACUOUS** — reading the fall-through
  (`vt::PagedAttention` over the per-rank full paged cache) shows it computes
  correct attention under tp>1 when the cache is replicated per rank, so that
  mutation would stay green and prove nothing. Not run; recorded as the
  reason it was skipped.
- **Value-corrupting variant goes RED**: `oh[0] += 1000.0f` after the shard
  call flips the tp>1 arm — `REQUIRE(t1 == tw)` fails
  ({6511,314,9564,369,19241,13,271,248068,271} vs {6511,138104,...}), EXIT=1,
  1 failed / 0 passed. The first (prefill) token still matches; corruption
  diverges decode from token 2, as the paged cache accumulates.
- **Restore is GREEN**: rebuilt clean, paged gate re-ran and matched the known
  9 tokens (EXIT=0).

Full no-regression sweep after the restore (GPUs 2,3): `test_sm70_fa2_decode`
5/5, `test_cuda_backend` 7/7, `test_nccl_group` 16/16, paged tp==tp1 green.


## Owed
- **Fix the dense tp>1 attention from gather to GQA-exact.** Real latent
  correctness bug (see KNOWN DISCREPANCY). Cannot file a GitHub issue from this
  box (no `gh`, no tokens) → recorded in `.agents/issue-index.md`; highest
  existing issue is #1152.
- Merge `origin/main` (63 ahead / 61 behind, 3 conflicts) + repair the branch's
  record debt (11 empty-body `docs:` commits, 4 oldest sm70 commits missing
  trailers, 8 doc-checkpoints) before landing — deferred to landing time; the
  branch is local-only (never pushed), so the history rewrite is safe.