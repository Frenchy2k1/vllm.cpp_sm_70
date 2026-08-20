# PLAN — full sm70 (Volta) support in vllm.cpp

Lane: `sm70-support` at `/home/nvidia/Dev/vllm.cpp.DS4F`, box `192.168.10.20`
(`h100`, 4×V100-DGXS-32GB, SM70 7.0), oracle conda `~/miniconda3/envs/1cat-vllm-sm70`
(vLLM 1.2.2). Current state: **65 ahead / 0 behind `origin/main`** at `be90d513`;
the dense TP-2 token gate (27B, tp==tp1 GREEN) and the MoE TP step-1/thread
commits landed; a large MoE-TP step-2 block is **staged but uncommitted**.

This plan is the next 20 steps to carry the sm70 lane to completion. Each step
names its gate, its verification, and its regression. tp1 (null `tp`) is the
load-bearing path and must stay byte-identical through every step.

## Running conventions (every step)

- **Dense TP is the completed prerequisite, not a plan step.** Dense
  tensor-parallel already landed and is token-gated: `TpShard`/`TpAllReduceSum`
  + the KV/GQA-shard attention and lm_head AllGather primitives, wired
  end-to-end through `Forward`/`ForwardDevice`/`ForwardDeviceTap`/
  `ForwardDeviceMultiTap`/`ForwardDense` (`qwen3_5_dense.h:271-326`, all
  `tp=nullptr` default), and the real gate at
  `tests/parity/test_op_parity.cpp:2430` — "qwen27 dense logits tp==tp1",
  4-GPU dense == tp1 (8 tokens identical) SUCCESS on this box. Commits
  `2de9d60b` … `40bad253` (the "tp==tp1 GREEN" commit). The MoE work below
  builds strictly on this ground; there is no dense-TP gap to fill first.

- Edit only the active lane (`/home/nvidia/Dev/vllm.cpp.DS4F`). Siblings
  `vllm.cpp.qwen` / `~/vllm.cpp-w0` are stale — do not edit.
- No-regression gates (must stay green via `tp1`/null): `test_nccl_group`,
  `test_sm70_fa2_decode`, `test_cuda_backend`, `test_qwen35_paged_engine`,
  `test_qwen27_paged_engine`.
- Build/run:
  ```
  ENV=~/miniconda3/envs/1cat-vllm-sm70; cd /home/nvidia/Dev/vllm.cpp.DS4F &&
  env PATH="$ENV/bin:$PATH" $ENV/bin/cmake --build build-sm70 --target <t> -j36
  ./build-sm70/tests/<t>
  ```
- GPU lease: this box is not a fleet device (`rc` absent); take the file mutex
  `${GPU_LOCK:-$HOME/gpu.lock}` (free on 2026-08-19). Never `ssh` to a box for
  GPU work. Do not push/merge without recorded authority (none recorded → stay
  local).

---

## Phase A — Land the staged MoE-TP step-2 block (steps 1–5)

### Step 1 — Commit the staged MoE-TP step-2 block
Change: the 7 staged files (`nccl_communicator.cu` online-softmax + `Hkv<W`
zero-partials; `qwen3_5.{h,cpp}` MoeBlock tp>1 host-decode + per-layer f32 cache
+ `MoeTpHostResident`; the three headers' `tp` params; the two test files).
Gate: `git stash` shows tp1 regression is nil (the null-tp early-return never
enters). Verify: leave out the preflight; commit with `FOLLOWING_AGENTS_PROTOCOL`
+ trailers (no `Co-Authored-By` / `Signed-off-by` from the agent). The brand
must be `row/<ID>` via an open issue, or `sm70-support` (lane default).
Regression: after commit, `git status` clean; prior HEAD ancestry preserved.

### Step 2 — Build the staged block (compile gate)
Gate: `cmake --build build-sm70` compiles -Werror-clean on `test_nccl_group`
and `test_op_parity`.

### Step 3 — Green the staged model-free tests
Gate: `./test_nccl_group` 12/x including the two new cases (Hkv<W kv-shard,
35B `H=2048,I=512,E=256,top=8` MoE geometry), with `VT_TP_DIAG` both default-off
(no output) and `VT_TP_DIAG=1` (nan-loud diagnostics stable).
Regression: no new failure vs the prior 12/12 run.

### Step 4 — Run the sm70 no-regression gates
Gate: `test_sm170_fa2_decode` 5/5, `test_cuda_backend` 7/7. tp1 path untouched.

### Step 5 — Run the `qwen36 moe tp==tp1` token gate on the box
Gate: `test_op_parity -tc="qwen36 moe tp==tp1*"` on the real 35B NVFP4
(fp4-resident) gives 4-GPU MoE == tp1, `>=2` tokens identical (target: the 8
used in the dense gate). This is the MoE analog of the 27B dense gate (step of
the spec's "REACHED" gate).

---

## Phase B — MoE paged-KV decode path (steps 6–8)

### Step 6 — Per-rank paged-KV in `RunDenseLayerPaged`
Change: thread `tp` into the paged decode attention path so KV is
partitioned per rank (the handoff names this "not in the dense/MoE token
gates"). Gate: model-free paged attention == single-GPU reference per rank.
Regression: tp1 paged path byte-identical (null-tp early-return).

### Step 7 — KV/GQA shard over paged blocks
Change: extend `vt_cuda_attn_kv_shard_run` to take paged KV blocks/blocks
index. Gate: paged-kV-shard == unpaged shard reference (same head split).
Regression: `test_sm170_fa2_decode` still green.

### Step 8 — MoE token gate with paged-KV decode
Gate: rerun `qwen36 moe tp==tp1` with the paged decode enabled; token-identical.

---

## Phase C — the "all Forwards" threading wave (steps 9–13)

### Step 9 — Define the wave targets
Change: enumerate the ~58 hand-rolled `::Forward` classes from the model
registry and grep which already carry `tp` (many do not). Verify: a list in the
capture. No gate yet (documentation).

**Enumeration DONE (2026-08-19, scout):** 24 hand-rolled forward families
total; **2 already carry `const vllm::TensorParallel* tp`** (Qwen3_5Model,
Qwen3_5DenseModel — the dense+MoE sm70 lane). **22 are tp-absent** and each
already ends `..., const std::vector<int32_t>& logits_indices = {}`, so appending
`, const vllm::TensorParallel* tp = nullptr)` is the uniform pattern:
commandr, deepseek_v2, deepseek_v4, gemma{1,2,3,4}, laguna, minicpm, minicpm3,
muse_glimmer, olmo2, opt, phi, phi3, qwen3 (dense), qwen3_moe, stablelm, glm4,
kimi_k3, kimi_linear. The llama/mistral/internlm2 families carry NO own forward
(reuse the shared `dense_attn` seam) — TP must thread into `dense_attn_block.h`,
not per-family.

### Step 9 — Define the wave targets
Change: enumerate the ~58 hand-rolled `::Forward` classes from the model
registry and grep which already carry `tp` (many do not). Verify: a list in the
capture. No gate yet (documentation).

### Step 10 — Add the null-default `tp` param central
Change: add `const vllm::TensorParallel* tp = nullptr` to each arch `::Forward`
and pass it into the shared GEMM/attention ops (sharding math stays in
`tensor_parallel.h`). Gate: tp null / tp_size==1 → existing path byte-identical.
Verify: whole diff is mechanical (no semantics).

### Step 11 — per-arch tp1 regression for the first wave
Gate: run each touched arch's existing native gate (token-exact or near-tie)
with tp1 — must stay green.

### Step 12 — optional per-arch tp>1 token-equal gate
Gate: for each arch that enables TP (opt-in `VLLM_TP`), tp==tp1 token-equal on
the box; where the checkpoint does not fit, record the gate as
`UNREACHED-with-reason`, never as pass.

### Step 13 — Roll and record per-arch
For each threaded arch, update `.agents/parity-ledger.md` (a row) and the matrix.
No per-row doc churn elsewhere.

---

## Phase D — MoE expert paging + residual correctness (steps 14–16)

### Step 14 — MoE expert paging in decode
Change: expert weights page in/out per layer where the box does not hold all
E experts (fp4-resident, but verify the bound for 35B on one V100-32GB).
Gate: decoder occupancy correctness; softmax/router numeric equality vs
non-paged.

### Step 15 — Expert-shard bf16 parity A/B
Gate: `VT_SM70_MOE_WMMA` off/on both token-exact on the 35B paged-engine.
Regression: the ladder `test_nccl_group` still 12/x.

### Step 16 — Full model token gate (both dense + MoE on the box)
Gate: `test_qwen27_paged_engine` 241/241 at-tp1 AND the new `test_qwen36
paged_engine` moe-tp path both green with tp into the 27B dense forward still
null.

---

## Phase 5 — beyond tensor parallel (steps 17–20)

### Step 17 — Reconcile the parity-ledger graph
Gate: mirror the corrected handoff state (24B dense, 35B MoE, the MoE-tp token
gate, the paged decode) as ledger rows with verdicts and dates. Rename each
row's `## Now` per `AGENTS.md` §public-documents only on their lifecycle change.

### Step 18 — Regression: the full sm70 suite
Run the complete no-regression set in one sitting on the idle box under the
file mutex: 12/x nccl, 5/5 fa2, 7/7 backend, 27B/35B paged gates. Any fail →
step to next (the pytest supplementary regression set runs tp1).

### Step 19 — Final `run_pipeline_gate` parity pass
Fresh-fetch `run_pipeline_gate.py` (never trust a git-cached copy) with
`--required-token-exact` against the pinned oracle 1.2.2; record the result in
the ledger.

### Step 20 — Author the row	/ push + merge review
Prepare the reviewed `row/<ID>` PR (or authorized local merge) body = the landed
commit message ending with the trailer block. Never force-push; the human
submitter owns the merge. Only with recorded authority.

---

## Verifications / regressions checklist (applies above)
- Permanent sm70 feature → run the explicit gate + the full no-reg set.
- Every change keeps the `tp1` null early-return: tp null / size==1 → original
  path; tp>1 → either sharded-correct or throws, never silent tp1 math.
- After each turnover of tests, remove any scratch (`VT_TP_DIAG` stays env-gated
  at default off; do not ship a diagnostic as behavior).

## Notes / open items
- In-process `ncclCommInitAll` is single-process/local; multi-node is out of
  scope here.
- The staged MoE block carries `VT_TP_DIAG` — decide keep (env-gated) vs strip
  before promoting step 1.
- Author positional- `parity-ledger` "Most meaningful" line does not exist in
  the records (verified by search); no invented edit was made.