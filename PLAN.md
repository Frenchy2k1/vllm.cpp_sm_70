# PLAN — sm70 multi-device TP stretch (paged-KV shard + MoE expert TP + Qwen3.8 gate)

Working tracker for the deferred "stretch" of the TP rollout on
`row/tp-3-stretch`. The authoritative *design* lives in
`.agents/specs/tp-rollout.md`; the *campaign* lives in `handoff.md`. This file
tracks only *this* increment's progress and decisions. Keep it current.

## Position

Dense 27B tp>1 decode is **wired and token-gated** (8 tokens == tp1, `40bad253`).
This increment adds: (1) per-rank **paged** KV in the `RunDenseLayerPaged`
decode path, (2) **MoE (A3B) expert TP**, (3) a **Qwen3.8-27B-NVFP4**
loader/capture gate, then a full no-regression sweep.

## Hard constraints (do not break)

- **GPU:** only **GPU 2 and 3** are usable for experiments/tests (0 and 1 are
  owned by other tasks). Steer every GPU run with `CUDA_VISIBLE_DEVICES=2,3`
  → world = **2**. 27B `Hkv=4` → `per=2`; 35B `Hkv=2` → `per=1`. Both divide.
- **tp1 byte-identity is load-bearing.** tp null / tp_size==1 must leave the
  original resident path untouched. tp>1 is either sharded-correct or throws.
- **Build tree:** `build-sm70` (CUDA arch 70, TRITON=OFF). NCCL group tests
  need `-DVLLM_CPP_NCCL=ON`. 27B/35B paged-engine golden gates need
  TRITON=ON + cutlass (not this build) → use the op_parity paged tp==tp1 or the
  35B engine gate instead.
- **tp>1 primitives must save/restore the caller's CUDA device**
  (`SaveRestoreLaunchCap` pattern, `nccl_communicator.cu:440`).
- **Do not edit tests/verification assets to pass a gate.**

## Merge state (branch vs origin/main)

Branch is **63 ahead / 61 behind** `origin/main`, diverged at `dd8a3b0e`
(origin/main is *not* an ancestor of HEAD). origin/main carries the
**T0 ONE-SURFACE ABI refactor**. `git merge-tree` reports **3 conflicts**:
`.agents/parity-ledger.md`, `docs/STATUS.md`,
`src/vllm/model_executor/models/qwen3_5.cpp`. Main's qwen3_5.cpp change
(374+/50-) does **not** touch this increment's edit regions
(`FullAttnBlockPaged` / `MoeBlock` / `RunDenseLayerPaged` / `Forward` threading),
so the conflicts are confined to the record/doc lines. Merge is deferred to
landing (see debt).

## THE load-bearing math finding (fixes a handoff mis-statement)

tp1 dense attention and tp>1 dense attention are **not** the same computation:

- **tp1** `AttentionKernel` (`cuda_ops.cu:1468`): `g = h/(hq/hk)` — **GQA**, each
  q-head attends to exactly one kv-head.
- **tp>1** `AttnKvShardPartial`/`AttnKvHostRef` (`nccl_communicator.cu:824,854`):
  `for kh in [0,Hkv)` summing `exp(q·k)` and `exp(q·k)·v` over **all** kv-heads,
  then `out = num/den`. This is a **gather over all kv-heads**, not GQA.

Experiment: `max|gqa − gather| = 4.86`, `99.8%` elements differ >1e-3. The
dense 8-token gate passes **only** because greedy argmax is robust to the
difference. **It is a weak baseline, not byte-exact tp==tp1.** Both the kernel
and its self-check reference use gather — they agree with each other but are
wrong relative to true GQA.

**Consequence:** the paged shard must **not** copy the all-kv-heads loop.
Correct design = **GQA-exact decomposition** (see below). Whether to also fix
the dense primitive is a follow-up product decision (recorded as debt), outside
this increment unless re-scoped.

## Design: GQA-exact KV shard (paged + dense)

For each `(t, hq, d0)`: `g = hq / qpk` with `qpk = Hq/Hkv`.
- If `g` is in this rank's kv-head slice `[hk0, hk0+per)`: compute the **full
  single-kv-head softmax over positions** `s ≤ key_end[t]` (paged:
  `block_table`-indexed K/V reads; dense: contiguous `[S,Hkv,D]`). No sum over
  kh.
- Else: contribute 0.
- **AllReduceSum** across ranks: each q-head's softmax is computed on exactly
  one rank, so the reduce gathers a complete output with **no global num/den
  divide**. Token-exact to unsharded `vt::PagedAttention`/`vt::Attention` by
  construction. `o_proj` stays a full GEMM (full output on every rank).

Reuse the proven in-process pattern: tp>1 forward runs on the caller device;
shard primitive internally spawns per-rank threads on local GPUs (own
`vt::CudaCommGroup`), AllReduce, returns the full result.

## Phase plan

- [x] **0. Environment + gates classified.** GPUs 2/3 free; env cmake/ninja/
  PyYAML located (in conda env, not bare PATH); 20 gate failures split into
  stale-base (cleared by merge) vs real record debt. **DONE.**
- [ ] **1. Decide merge strategy.** (a) merge origin/main now (resolve 3
  conflicts) — protocol-faithful, isolates true remaining debt; (b) defer merge
  to landing, do additive TP work on current branch. **DEFERRED to landing.**
- [x] **2. Spec.** `.agents/specs/tp-rollout.md` extended with the GQA-exact paged
  shard design + batched MLP shard + MoE expert-TP + Qwen3.8 gate. **DONE.**
- [x] **3. Attention shard primitive.** `vt_cuda_attn_gqa_shard_run` (GQA-exact
  kernel + double-precision host reference, `NcclDevScope` per rank) in
  `nccl_communicator.cu`. Model-free selfcheck in `tests/vt/test_nccl_group.cpp`
  (causal + full-window, Hkv=4). 16/16 test cases, 2092/2092 assertions. **DONE.**
- [x] **4. Wire paged TP.** `FullAttnBlockPaged` tp>1 GQA branch (host-gather
  paged KV → `[S,Hkv,D]` f32 → gqa shard; `o_proj` full GEMM); `tp` threaded
  `Forward` → `DenseForwardBody` → `DenseForwardLayers` → `RunDenseLayerPaged`.
  tp1 byte-identical (path untouched). **DONE.** See MLP fix below.
- [x] **4b. MLP shard fix (was the paged-gate blocker).** `DenseMlpBlock` tp>1
  used a **per-token** `vt_cuda_mlp_shard_run` — one fresh `CudaCommGroup` +
  full O×I×H host double reference **per token per layer**. At 27B (O=H=5120,
  I=17408) that is ~1.7e14 double-FMA/token/layer → hours, i.e. the "hang".
  Fixed with a **batched** `vt_cuda_mlp_shard_run_b(T,O,H,I,verify,...)`:
  `DenseMlpRankPartialBatch` kernel (one thread per (t,o), rank owns an
  I/W slice), ONE group, ONE `[T*O]` AllReduce; `verify=0` in production
  (compute-only — the engine's token-vs-tp1 comparison is the real check),
  `verify=1` host-parity for the selfcheck (parallelised over (t,o) rows).
  fp4-resident weights host-decoded once per layer. Non-fp4 tp>1 throws loudly.
  Selfchecks: fresh shape (verify=1) + 27B production shape (T=9, verify=0,
  spot-checked vs host ref). **DONE.**
- [x] **4c. Paged tp==tp1 token gate GREEN.** `test_op_parity -tc="qwen27 paged
  logits tp==tp1*"`: 1 prefill (T=9) + 8 decodes; tp1 vs 2-GPU arms match all
  9 tokens (`6511 314 9564 369 19241 13 271 248068 271`), SUCCESS, EXIT=0
  (2026-08-19, ~2h run). TEMP markers removed after; no-regression sweep
  (nccl 16/16, fa2 5/5, backend 7/7, dense tp==tp1 8 tokens) green on the
  cleaned tree.
- [ ] **5. MoE expert TP.** `MoeBlock` weight-parallel branch (vLLM default);
  35B tp==tp1 gate at world=2 (Hkv=2 → per=1).
- [ ] **6. Qwen3.8-27B-NVFP4 gate.** Loader + smoke/deterministic capture.
- [ ] **7. Mutation proof.** Delete the paged production call site in a scratch
  copy → gate must go red; restore byte-for-byte.
- [ ] **8. No-regression sweep + records.** nccl/fa2/backend/op_parity gates;
  update `docs/STATUS.md`, `handoff.md`, spec Now/Outcome.
- [ ] **9. Record-debt repair (landing).** Rebase 63-commit campaign: 11
  empty-body `docs:` commits, 4 oldest sm70 commits missing trailers, 8
  doc-checkpoint omissions. Branch is local-only (never pushed) → safe rewrite.

## Decisions logged this segment

- **MLP tp>1 = batched, not per-token.** The per-token `vt_cuda_mlp_shard_run`
  (fresh group + full host reference per token) is unrunnable at production
  shape; the paged gate's objective is attention-shard token parity, and the
  batched `vt_cuda_mlp_shard_run_b` gives a *stronger* baseline (true TP-sharded
  MLP vs the per-token one that never completed). Perf of the shard itself
  (CPU host-decode + f32 reference kernel) is out of scope — only token parity
  vs the tp1 arm matters.
- **Paged tp>1 GQA branch is NOT the hang.** Markers proved layer-0 (GDN,
  replicated) attention completes in ~1ms and the stall is in the MLP; the
  attention branch first runs at layer 15, in ~35ms.
- **The dense tp==tp1 gate (40bad253, 129s) is a weak baseline**: it routes the
  MLP through the tp1 full GEMM (never the shard), so it proves only the dense
  *attention* gather path — which is itself the latent gather-vs-GQA
  discrepancy recorded above. The paged gate is the strong one.

## Verification gates (exact)

```
# NCCL primitives (world=2 on GPUs 2/3)
CUDA_VISIBLE_DEVICES=2,3 ./build-sm70/tests/test_nccl_group
# sm70 attention no-regression
CUDA_VISIBLE_DEVICES=2,3 ./build-sm70/tests/test_sm70_fa2_decode
CUDA_VISIBLE_DEVICES=2,3 ./build-sm70/tests/test_cuda_backend
# dense tp==tp1 token gate (WEAK baseline — see debt)
CUDA_VISIBLE_DEVICES=2,3 ./build-sm70/tests/test_op_parity -tc="qwen27 dense logits tp==tp1*"
# paged tp==tp1 token gate (THIS increment; ~2h)
CUDA_VISIBLE_DEVICES=2,3 ./build-sm70/tests/test_op_parity -tc="qwen27 paged logits tp==tp1*"
```

## Debt register

- **Dense tp>1 gather is a real (latent) discrepancy** vs tp1 GQA. File an
  issue + spec `## Owed` on the fix. **Cannot file a GitHub issue from this
  box** (no `gh` CLI, no tokens) → record in `.agents/issue-index.md` as owed.
  Highest existing issue: #1152.
- **Merge origin/main + record-debt repair** deferred to landing (phase 9).
  Branch never pushed; 20 gate failures are pre-existing campaign debt, not
  introduced by this increment.