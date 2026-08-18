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

- Path: `/shared/local/Programing/vllm.cpp` (branch `sm70-support`)
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
2. **Attention + KV/GQA shard** — DONE (`c643c311` + `9e246680`: KV-split with
   a causal `[S,Hkv,D]` window); wrapped into the unpaged FullAttnBlock by
   `40bad253` (q full on every rank, KV-head-split AllReduceSum → complete
   output per rank, so o_proj stays a full GEMM).
3. **Head (lm_head) column-shard + AllGather** — primitive DONE (`87d78ce`);
   with the layer output full on every rank, the token head needs no shard.
4. **Runner attach** — the real token-equal gate drives `ForwardDense` at tp==1
   vs tp==world over all-local GPUs; the gate proves tp==tp1.
5. **Token-equal gate** — **REACHED**: 4-GPU dense == tp1 (8 tokens identical),
   SUCCESS — the gate the handoff previously called UNREACHED-until-wiring.

Remaining stretch (NOT in the dense-token gate): per-rank paged-KV in the
`RunDenseLayerPaged` decode path, and MoE (A3B) expert TP — deferred in spec.

After Qwen3.5 dense (and MoE) reach token-equal, the spec's "all Forwards
threading" phase: add the null-default `tp` to the ~58 other hand-rolled arch
`::Forward` classes (mechanical; the sharding math stays in `tensor_parallel.h`).

## Working agreements / do-not-break

- **tp1 is the load-bearing path** and must stay byte-identical at every TP
  commit (guard: tp null / tp_size==1 → original resident path; tp>1 either
  sharded-correct or throws — never silently tp1 math).
- `run_pipeline_gate.py` is always fresh-fetched (reviewer parity, never
  marked pass from git); `--required-token-exact` makes a no-match fail.
- `test_sm70_fa2_decode` and the 35B/27B engine are the sm70 no-regression
  gates; keep them green when touching attention/GEMM.
- No flashy test names; the suite must stay fast + safe.

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
- GQA KV-split for the decode's page layout (deferred until step 2 is running).
- MoE (A3B) expert TP (separate, after the dense tp>1 path is token-equal).