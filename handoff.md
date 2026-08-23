# Handoff — vllm.cpp 1.2.2-lane: sm70 (Volta) support + Multi-GPU TP

Scope: bring `mudler/vllm.cpp` (the C++ vLLM kernel port, "the impl lane") to
build + serve the two target models on 4×V100 (sm_70, 16 GB, no fp8/fp16-mma
tensor cores; the fp4-only/tc-free class), with the 1.2.2 Python-lane oracle as
ground truth.

Primary goal: parity with the pinned vLLM — token-level dense (27B) + engine
(35B) — on the real path (fp16 GEMMs on fp16 input/weight activations), fp4
resident in VRAM. **Milestone achieved + held**: 27B dense 241/241 + 35B engine
315/315 (traces 1/1) token-for-token. See `.agents/parity-ledger.md` +
`.agents/STATUS.md`.

## Repository / branch

- Path (canonical, this session): `/home/nvidia/Dev/vllm.cpp.qwen`, branch
  `row/tp-3-stretch` (= `sm70-support` tip `2a1f79b8`). (Historical ports of
  this handoff name `/shared/local/Programing/vllm.cpp` and `~/vllm.cpp-w0`.)
- Upstream: `origin` = `https://github.com/mudler/vllm.cpp`
- Branch base: branched at `6db04e7`; a full **merge of `origin/main` (269 commits)
  landed as `796e6e65`** (doc-only conflicts: our `docs/STATUS.md` kept,
  upstream `tests/scripts/test_main_baseline.py` adopted). Status: 53 ahead / 0
  behind origin/main. All sm70 + multi-device work is additive on top of that.
- Commit history (newest first, all committed — see `git log`):
  `66f48f80` DenseMtpBlock tp>1 per-rank sharded dense-MLP · `2f8f7da0` bf16
  dense-MLP shard assembly + model-free test · `07b35339` host NVFP4 decode ·
  `1cf78163` engine-scale mlp_shard_run · `c3d0bf22` reusable
  vt_cuda_mlp_shard_run · `37ca9261` tp into DenseMtpBlock/RunDenseLayerPaged ·
  `c3d0bf22` reusable `vt_cuda_mlp_shard_run` · `37ca9261` tp into
  `DenseMlpBlock`/`RunDenseLayerPaged` · `dc4921a3` tp into `ForwardDense` ·
  `2de9d60b` dense-MLP body shard · `4cf28063` tp>1 throw-guard ·
  `19e90c72`+`6a66a006` tp-rollout spec · `2779cf38` tp step-1 null-default ·
  `faff54af`/`0a51d94d` retained engine bridge · `796e6e65` merge ·
  `8d303584` runner-forward · `fc58ae0b` loader slice · `392166c8`
  `CudaCommGroup`+TP seam · `b4901998` in-process collectives.

## Box + environment (shared 4×V100 box)

- Host: `nvidia@192.168.10.20` (pw `paradise`), 4×V100 16 GB sm70 (MF152 V100
  SXM2), 32-core Threadripper PRO 3975WX, Linux. (Roadrunner box is at
  192.168.10.21.)
- Conda env: `~/miniconda3/envs/1cat-vllm-sm70` (python 3.12.3, cmake 4.1.2,
  ninja, torch 2.13 cu128, **nvidia-nccl-cu12 2.30.7** (system ninja +
  `nccl.h` for the port).
- Build tree on box: `~/vllm.cpp-w0` with build dir `build-sm70`.
- CMake configure (nccl on, ninja, arch):
  ```
  cmake -DVLLM_CPP_NCCL=ON \
    -DNCCL_INCLUDE_DIR=$ENV/lib/python3.12/site-packages/nvidia/nccl/include \
    -DNCCL_LIBRARY=$ENV/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so \
    -G Ninja -DCMAKE_MAKE_PROGRAM=$ENV/bin/ninja build-sm70
  cmake --build build-sm70 --target <t> -j36
  ```
- Parity oracle (also running on the box): 1.2.2-snapshot `vllm` (docker
  `oracle-ada:555967922`, see `/shared/local/Programing/oracle-ada/Dockerfile`),
  `vllm serve` single-GPU with `--trust-remote-code`, dense bf16 or NVFP4.
- Model snapshots in box HF cache: `unsloth/Qwen3.6-27B-NVFP4` (~23 GB, rev
  `8900bdef7`), `nvidia/Qwen3.6-35B-A3B-NVFP4` (17 shards), `unsloth/...-27B-GGUF`
  (UD-Q4_K_XL / UD-IQ2_XXS), `deepseek-ai/DeepSeek-V3.2`, `usernet/SWE-Gym-Codeserver`
  (188 MB). **`unsloth/Qwen3.8-27B-NVFP4`** (22 GB) downloaded 2026-08-18 and
  complete — loader/capture gate still to build (see Next Steps).

## What works (verified on the box)

### sm70 attention + serving (the parity lane)
- `test_sm70_fa2_decode` 5/5 (Real FA2 H2N decode == pytorch, + graph-capture
  safety = flush).
- `test_qwen35_paged_engine` 315/315 + `test_qwen27_paged_engine` 241/241 (35B
  NVFP4 / 27B dense) — token-for-token vs golden, compiled off the merged
  upstream `796e6e65`.
- `test_cuda_backend` 7/7 (per-device allocation/launch affinity — a CUDA op
  on `Device{kCUDA,i}` executes on GPU `i`).

### Multi-device TP substrate (the current feature, all committed + green)
Ladder, each a real primitive verified on the 4×V100:
1. Per-device CUDA backend registration + per-op `cudaSetDevice` affinity.
2. In-process NCCL collectives (`ncclCommInitAll`, per-rank streaming via
   `vt::CudaCommGroup`, threaded in-thread-stream blocking).
3. `vt::CudaCommGroup` + the W2 `vllm::{TensorParallel, TpShard, TpAllReduceSum}`
   seam over real NCCL (row/column parallel GEMM in vllm.cpp).
4. `vt_cuda_loader_slice_selfcheck` (per-rank shard placement + reconstruct).
5. `vt_cuda_sharded_forward_selfcheck` (device-sharded GEMM + group reduce ==
   single-GPU forward — the "runner forward" primitive).
6. `vt_cuda_dense_mlp_shard_selfcheck` (fused `down(silu(gate x)·up x)` column
   gate/up + down row-reduce == unsharded MLP).
7. **`vt_cuda_mlp_shard_run(O,H,I,…)`** — the reusable, arbitrary-shape dense-MLP
   shard over the group; engine-scale verified (`I=1024, H=1024, O=1024`) with
   a RELATIVE parity tol (5e-4·max(1,|ref|)).
8. **Host NVFP4 decode `vt_host_decode_nvfp4_f32`** — bit-exact mirror of the
   in-kernel `F8E4M3` + `DecodeFp4Byte` (E2M1 LUT + bf16-RNE product),
   known-nibble verified.
9. `vt_cuda_tp_acquire(tp)` / `vt_cuda_tp_release` — the **retained engine
   bridge**: the runner obtains an owned `CudaCommGroup` for tp>1 (null for
   tp1 → byte-identical).
10. `tp` (null-default) threaded through `Qwen3_5DenseModel::{Forward,
    ForwardDevice, ForwardDeviceTap, ForwardDeviceMultiTap, ForwardDense}` and
    into `RunDenseLayerPaged` → `DenseMlpBlock`; **`tp>1` throws** (fail-loudly)
    until the per-rank shard is hooked into the resident fp4/bf16 path.

`test_nccl_group` runs all of this: 10/10, 29 assertions (bf16 assembly
test now committed + 10/10 green — see below).

## Landed since handoff (all committed, box build green)

- `2f8f7da0` **bf16 dense-MLP shard assembly** (`vt_cuda_mlp_shard_run_bf16`, the
  engine-facing bf16 gate - `[2I,H]` merged gate_up + `[O,I]` down → f32, runs
  `vt_cuda_mlp_shard_run`), + a model-free synthetic test. `test_nccl_group`
  10/10, 29 assertions.
- `66f48f80` **DenseMtpBlock tp>1 branch** — real per-rank sharded dense-MLP:
  host-decode the resident fp4 (or bf16-raw / split Matmul-B) gate/up/down into
  the shard's `[out,in]` f32 layout, then run the group-sharded dense-MLP per
  token over the in-process NCCL group, writing the reduced `[T,H]` bf16 back.
  tp1 (null) never enters → resident tensor-core path byte-identical; non-NCCL
builds fault loudly. Box regression: nccl 10/10, fa2 5/5, backend 7/7
  (58), qwen36 paged-engine 35B case token-equality 149/149 green.
  (The 27B/35B paged-engine gates require `VLLM_CPP_TRITON=ON` + cutlass; this
  box build has TRITON=OFF, so those gates measure the build — unchanged.)
- `c643c311` **KV/GQA-shard attention primitive** (`vt_cuda_attn_kv_shard_run`):
  KV-column split across ranks; softmax num/den both additive over kv, the group
  AllReduceSum reduces both → out = num/den. `test_nccl_group` 11/11 incl. a
  model-free KV-split (T=3,Hq=4,Hkv=8) == single-GPU softmax test.
- `87ee78ce` **lm_head column-shard + AllGather primitive**
  (`vt_cuda_lm_head_shard_run`): per-rank [T,per] partial logits, group AllGather
  → full [T,vocab]. `test_nccl_group` 12/12, 479 assertions.

- `9e246680` **sequence-aware `vt_cuda_attn_kv_shard_run`** — causal
  `[S,Hkv,D]` window. `test_nccl_group` 12/12, 479 assertions.
- `40bad253` **per-rank dense attn wired + REAL tp==tp1 token gate** — the
  multi-GPU token gate the spec marked UNREACHED; `test_op_parity
  -tc="qwen27 dense logits tp==tp1*"` on the real 4×V100 27B NVFP4 gives
  **4-GPU dense == tp1 (8 tokens identical)**, SUCCESS.

## TP step 2 (end-to-end tp>1 decode) — WIRING DONE, ALL GATES GREEN

Authoritative spec: `.agents/specs/tp-rollout.md`. All three row/col-parallel
PRIMITIVE bricks plus the forward/runner wiring are committed + green on the
box. The multi-GPU token gate the spec once marked UNREACHED is now reached.

1. **`DenseMtpBlock` tp>1 branch** — DONE (`66f48f80`): host-decode fp4/bf16
   gate/up/down → `vt_cuda_mlp_shard_run`/`_bf16` per token → reduced `[T,H]`.
   tp1 keeps the resident fp4 tensor-core path (tp null → branch never entered).
2. **Attention + KV shard** — DONE (`c643c311` + `9e246680`: KV-split with a
   causal `[S,Hkv,D]` window); wrapped into the unpaged FullAttnBlock by
   `40bad253` (q full on every rank, per-rank KV-slice partial softmax,
   AllReduceSum → complete output per rank, so o_proj stays a full GEMM).
   **INTEGRITY NOTE (2026-08-19):** this primitive currently computes a
   *gather-over-all-kv-heads* — it runs `for (int kh = 0; kh < Hkv; ++kh)` in
   both `AttnKvHostRef` (nccl_communicator.cu:824) and `AttnKvShardPartial`
   (:854) — which is NOT the GQA the tp1 `AttentionKernel` uses
   (`g = h/(hq/hk)`, cuda_ops.cu:1468). The dense tp==tp1 gate passes only
   because greedy argmax is robust to the difference: **it is a WEAK baseline,
   not proof of byte-exact parity.** Fixing the dense primitive to GQA-exact is
   a tracked follow-up; the NEW paged shard (below) is GQA-exact by design.
3. **Head (lm_head) column-shard + AllGather** — primitive DONE (`87d78ce`);
   with the layer output full on every rank, the token head needs no shard.
4. **Runner attach** — the real token-equal gate drives `ForwardDense` at tp==1
   vs tp==world over all-local GPUs; the gate proves tp==tp1.
5. **Token-equal gate** — **REACHED (weak)**: 4-GPU dense == tp1 (8 tokens
   identical), SUCCESS. **Caveat (2026-08-19):** the tp>1 attention primitive
   uses gather-over-all-kv-heads, which ≠ tp1 GQA; the gate passes on greedy
   argmax robustness only. See the KNOWN-DISCREPANCY section below.

Remaining stretch (NOT in the dense-token gate): per-rank paged-KV in the
`RunDenseLayerPaged` decode path, and MoE (A3B) expert TP — deferred in spec.

After Qwen3.5 dense (and MoE) reach token-equal, the spec's "all Forwards
threading" phase: add the null-default `tp` to the ~58 other hand-rolled arch
`::Forward` classes (mechanical; the sharding math stays in `tensor_parallel.h`).

## TP step 3 — paged tp>1 performance levers (token-exact)

After the paged tp==tp1 token gate went green (2-GPU == tp1, 9 tokens, ~86 min
wall), the perf work targeted the measured ~99% cost: the MLP shard's
per-layer fp4→f32 host decode (4.7 s) + masked `gu`/`dmask` weight-buffer
build/pageable H2D (3.5 s). Both are **single-threaded host loops, token-exact
to parallelize** (each element written once by deterministic per-element math,
no cross-element accumulation). Landed in `nccl_communicator.cu`:
- **A:** `vt_host_decode_nvfp4_f32` row-partitions the N rows across
  `min(ncpu, N)` host threads.
- **B:** `vt_cuda_mlp_shard_run_b` builds the full merged `[2I,H]` + `[O,I]`
  weight buffers ONCE on the main thread, in parallel over I rows, before the
  rank loop (was: serially inside each rank worker); workers now only H2D +
  launch + AllReduce.
Rejected (recorded in spec + PLAN.md): decode-once host cache (~68 GB host,
prohibitive; LRU < 64 layers ~0 hit-rate) and persistent `CudaCommGroup`
(~16 ms/call ≈ 2 s total; the HBM-bound f32 GEMM weight re-reads are the
remaining floor).
**Measured (2026-08-20, GPUs 2,3, same binary family):** `test_nccl_group`
green (16/16 cases, 2092 assertions, ~23 s); the 27B paged tp==tp1 token gate
still matches the known tokens (6511 314 9564 369 19241 13 271 248068 271) and
wall drops from the **5139 s (85.7 min) marker-span baseline to 2935 s (49
min)**; dense tp==tp1 gate green (112 s) — no regression. The token-exact
levers are confirmed: the f32 values feeding the kernel are unchanged, only
their host-thread scheduling is.


## Working agreements / do-not-break

- **tp1 is the load-bearing path** and must stay byte-identical at every TP
  commit (guard: tp null / tp_size==1 → original resident path; tp>1 either
  sharded-correct or throws — never silently tp1 math).
  **On `tp>1` attention specifically (2026-08-19): "sharded-correct" now means
  GQA-exact — each q-head attends to exactly one kv-head `g = hq/qpk` — not the
  all-kv-heads gather the current dense primitive does. Do not copy the gather
  loop into any new attention path.**
- `run_pipeline_gate.py` is always fresh-fetched (reviewer parity, never
  marked pass from git); `--required-token-exact` makes a no-match fail.
- `test_sm70_fa2_decode` and the 35B/27B engine are the sm70 no-regression
  gates; keep them green when touching attention/GEMM.
- No flashy test names; the suite must stay fast + safe.

## KNOWN DISCREPANCY — dense tp>1 attention is gather, not GQA (2026-08-19)

**Do not treat the dense tp==tp1 gate (8 tokens) as byte-exact proof.** The
tp>1 attention primitive (`vt_cuda_attn_kv_shard_run`, `AttnKvHostRef` +
`AttnKvShardPartial` in `nccl_communicator.cu:824,854`) sums over **all** kv
heads:
```
for kh in [0,Hkv): e = exp( (q·k[s,kh])*scale ); den+=e; num+=e*v[s,kh,d0]
out = num/den
```
The tp1 `AttentionKernel` (cuda_ops.cu:1468) is standard GQA:
```
g = h / (hq/hk);  // one kv head per q head; softmax over positions only
```
Experiment (27B-shaped layer Hq=24/Hkv=4/Dh=256, random normed q/k/v):
`max|gqa − gather| = 4.86`, `frac |Δ| > 1e-3 = 0.998`. **Different math.** The
gather kernel and its on-host self-check reference agree with each other but
are both wrong relative to true GQA. This is a REAL, LATENT correctness gap in
the dense tp>1 path; it is masked only by greedy-argmax robustness.

**Correct design (adopted for the paged shard, and the fix target for dense):**
per `(t,hq,d0)`, `g = hq/qpk` (qpk = Hq/Hkv). If `g` is in this rank's kv slice
`[hk0, hk0+per)`, compute the full single-kv-head softmax over positions
(no sum over kh); else contribute 0. AllReduceSum across ranks → because each
q-head's softmax is computed on exactly one rank, the reduce gathers a
complete output with **no global num/den divide**, and the result is
token-exact to unsharded `vt::Attention` by construction.

**Owed:** file the dense-gather bug (issue-index row + spec `## Owed`) and fix
the dense primitive. Blocked on the repo (no `gh`, no tokens) → record in
`.agents/issue-index.md`; highest existing issue is #1152.

## Current session (2026-08-19) box state

- Canonical checkout: `/home/nvidia/Dev/vllm.cpp.qwen`, branch
  `row/tp-3-stretch` (= `sm70-support` tip `2a1f79b8`). `~/vllm.cpp-w0` is an
  old dirty checkout whose committed tree is byte-identical (only
  STATUS.md/parity-ledger differ).
- Build tree: `build-sm70` (CUDA arch 70, TRITON=OFF, `-DVLLM_CPP_NCCL=ON`).
  cmake/ninja/PyYAML live in `~/miniconda3/envs/1cat-vllm-sm70` — export
  `PATH=$ENV/bin:$PATH` before build/gate runs.
- **GPU constraint (user, 2026-08-19): only GPU 2 and 3 are usable** for
  experiments/tests (0 and 1 are owned by other tasks). Steer every GPU run
  with `CUDA_VISIBLE_DEVICES=2,3` → world=2. 27B `Hkv=4`→`per=2`; 35B `Hkv=2`→
  world=2 (`per=1`).
- **Merge state:** branch is 63 ahead / 61 behind `origin/main` (`6792dc43`),
  diverged at `dd8a3b0e`; `origin/main` is NOT an ancestor of HEAD. Merge
  (deferred to landing) has 3 conflicts — `parity-ledger.md`, `docs/STATUS.md`,
  `qwen3_5.cpp`; main's qwen3_5.cpp change does **not** touch this increment's
  edit regions (FullAttnBlockPaged/MoeBlock/RunDenseLayerPaged/Forward).
- **Gate debt:** 20 failures, all pre-existing campaign debt; stale-base
  (cleared by merge) vs real record debt (11 empty-body `docs:` commits, 4
  oldest sm70 commits missing trailers, 8 doc-checkpoint). Branch is local-only
  (never pushed) → history rewrite is safe.

## Paged tp==tp1 gate — localized, fixed, running (2026-08-19)

The paged tp==tp1 token gate (`test_op_parity -tc="qwen27 paged logits
tp==tp1*"`, the real 27B, world=2 on GPUs 2/3) initially **hung** in the tp>1
arm's prefill `Forward`. Markers proved the GQA attention branch is *not* the
cause: layer 0 (GDN, replicated) completes in ~1ms and the first full-attn GQA
branch runs at layer 15 in ~50ms. The stall is in **`DenseMlpBlock`'s tp>1
path**, which ran `vt_cuda_mlp_shard_run` **per token** — one fresh
`CudaCommGroup` + a full O×I×H host double reference **per token per layer**.
At 27B (O=H=5120, I=17408) that is ~1.7e14 double-FMA/token/layer → hours;
it looked like a hang but is just catastrophically slow.

**Fix (this segment):** a **batched** `vt_cuda_mlp_shard_run_b(T,O,H,I,verify,
x,gate,up,down,out)` in `nccl_communicator.cu` + a `DenseMlpRankPartialBatch`
kernel (one thread per (t,o); rank owns an I/W intermediate slice). The whole
[T,H] batch runs in ONE group / ONE `[T,O]` AllReduce; fp4 weights are
host-decoded once per layer. `verify=0` (production) is compute-only — the
engine's token-vs-tp1 comparison is the real correctness check; `verify=1`
runs the (parallelised) host-parity selfcheck. `DenseMlpBlock` tp>1 now calls
it once (was a per-token loop). Non-fp4 tp>1 throws loudly. Selfchecks added:
fresh shape (verify=1) + 27B production shape (T=9, H=5120, I=17408, verify=0,
spot-checked vs host ref). `test_nccl_group` **16/16, 2092/2092**.

**Gate status: GREEN (2026-08-19).** ~2h run completed EXIT=0:
`paged tp==tp1 token gate: 2-GPU == tp1 (9 tokens: 6511 314 9564 369 19241
13 271 248068 271)` — 2 assertions passed. Gate size `REQUIRE` corrected to
`1+kDecodes` (=9 tokens). All TEMP diagnostic markers
(`[pgmlp]`/`[pglyr]`/`[pgpre]`/`[pgtp]`/`[gqa]`/`[mlp]`, `PgMarkMs`,
`PagedTpMark`) removed afterward; no-regression sweep on the cleaned tree:
nccl 16/16 (2092), fa2 5/5, backend 7/7 (42), dense tp==tp1 (8 tokens) — all
SUCCESS.

**Edit-tool gotcha (cost repeated clobbering this session):** a bare
`PUT <N>:` followed by a `+` body **replaces** line N (behaves like
`PUT N.=N:`), it does not insert. To insert without deleting the anchor use
`PUT <N:` (before) or `PUT >N:` (after). Every marker insertion must be
re-read to confirm it did not eat the anchor's following lines.

## Quick commands (resume)

```
# box build + run the TP substrate suite
ssh nvidia@192.168.10.20 'ENV=~/miniconda3/envs/1cat-vllm-sm70; cd ~/vllm.cpp-w0 && \
  env PATH="$ENV/bin:$PATH" $ENV/bin/cmake --build build-sm70 --target test_nccl_group -j36 && \
  ./build-sm70/tests/test_nccl_group'
# local repo
cd /shared/local/Programing/vllm.cpp && git status && git log --oneline -20
```

## Open questions for the next owner
- Is in-process (`ncclCommInitAll`, one process world=N) the intended multi-GPU
  topology vs. vLLM's multi-process unique-id broadcast? (Current primitive
  path is in-process; the transport is also ported for multi-process ranks.)
- GQA KV-split for the decode's page layout — **ANSWERED 2026-08-19**: use the
  GQA-exact decomposition above (`g = hq/qpk`, per-rank single-kv-head softmax,
  AllReduceSum, no num/den). Stage the paged `[S,Hkv,D]` KV to host → call a
  paged GQA-exact shard; correctness first, perf out of scope for the token
  milestone. Hkv divisibility: world divides Hkv (27B Hkv=4 @ world=2 → per=2;
  35B Hkv=2 @ world=2 per=1); for W>Hkv use kv-head replication / zero-fill.
- MoE (A3B) expert TP: weight-parallel is the vLLM default — every rank holds
  all experts; per-expert gate/up column-shard, down row-shard, AllReduceSum
  combine; `tp` must thread through the MoE `ForwardDense` (currently takes no
  `tp`), paged path currently throws on tp>1 (`qwen3_5.cpp:8733-8737`).
- Whether to also back-fix the dense tp>1 attention to GQA-exact (see KNOWN
  DISCREPANCY). Tracked as owed; filed once `gh`/a token is available.