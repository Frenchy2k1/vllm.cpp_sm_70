# Milestone-0 Status — Parity pin (sm70-support)

Date: 2026-08-12
Branch: `sm70-support`
Silicon: 4× **Tesla V100-DGXS-32GB, compute_cap 7.0** (`192.168.10.20`, Ubuntu 24.04.4) — sm70 premise confirmed directly via `nvidia-smi --query-gpu=name,compute_cap`.

## The two-oracle contract (plan §2.5)

| Lane | Reference | Model / route | Grade |
|---|---|---|---|
| same-silicon | 1Cat-vLLM 1.2.2 (V100) | `Qwen3.6-27B-AWQ`, TP4 | SEMANTIC / near-tie (AWQ INT4 ≠ NVFP4) |
| semantic | pinned vLLM 0.26.0 `55596792…` (Ada) | `Qwen2.5-3B-Instruct` (`oracle-ada`) | SEMANTIC, never STRICT |
| impl token-exact | this repo | `nvidia/Qwen3.6-35B-A3B-NVFP4` @ `491c2f1e…` | same-model engine gate |
| impl token-exact | this repo | `unsloth/Qwen3.6-27B-NVFP4` @ `890bdef7…` | dense forward gate |

Cross-route caveat, fixed: the 27B-AWQ oracle and the NVFP4 fast paths are
**different quant routes** — parity is graded at the semantic / near-tie level
(token-string greedy + per-position argmax), **not** bit-exact logits.

## Measured results (all on the V100, this session)

1. **35B engine gate — GREEN.** `test_qwen36_paged_engine`, full paged
   LLMEngine stack (`FromModelDir` → prefill → KV → decoder → Sampler) greedily
   decodes the pinned 35B-A3B-NVFP4 and reproduces the golden **token-for-token**:
   `315/315` assertions pass, `Status: SUCCESS!`.
2. **27B dense forward gate — prefill-exact, tok6 near-tie.** `test_op_parity
   -tc="qwen27 dense logits*"` on `unsloth/Qwen3.6-27B-NVFP4`:
   - per-position prefill `argmax_match = 9/9` (dense `ForwardDense` == oracle);
   - greedy `6/16` then the **tok6 near-tie** — emulation side (`271` "\\n\\n")
     vs the golden's native side (`198` "\\n"). The production W4A4 GEMM
     (`VT_CUTLASS_NVFP4`, sm_90+) that would select the native side is absent by
     design in the sm_70 build; on sm_70 our hand W4A4 kernel is the legitimate
     answer. Recorded as a near-tie, not a defect.
3. **27B-AWQ oracle capture — deterministic.** 8-prompt × 3-run greedy battery
   (T=0, `enable_thinking:false`) from the live 1Cat server:
   `tests/parity/goldens/qwen36_27b_awq_oracle/{trace,meta,prompt_ids}.*`
   committed `39d89189`, `24/24` rows identical across runs (no near-ties in
   battery). Comparator: `.agents/compare-oracle-impl.py` (smoke: oracle-vs-itself
   = 8 exact / 0 diverged).
4. **Decoder binaries** built+verified on the box: `test_qwen36_paged_engine`
   (57 MB), `test_qwen2_paged_engine` (57 MB), `test_op_parity`.
5. **sm70 expert GEMM (brick H) — GREEN.** The A3B-35B expert decode now
   runs a Volta **fp16-WMMA** grouped kernel (`MoeGroupedGemmNvfp4WmmaF16`,
   fp32 acc, bf16→fp16 saturation-clamped) instead of the CUDA-core naive fill.
   The 35B engine gate with this path default stays **token-exact 315/315**
   (real 35B activations, incl. 6-concurrent × 16-token greedy). Marlin was
   sm_75+ and bf16-WMMA sm_80+ (Volta has no bf16 MMA), so Volta tensor cores
   are now engaged on the expert path too.
6. **Wider-head paged attention — GREEN.** Decode/prefill head_dim widened to
   {64,128,192,256}; V100 dev-vs-ref 2.00e-5 (192) / 1.74e-5 (256); suite 5/5? no —
   `test_sm70_fa2_decode` 5/5 SUCCESS. Commit `59c0d05a`.
7. **y-dg-sync (graph-capture safety) — GREEN.** sm70 decode proven replayed
   inside a CUDA graph bit-identically to eager (dev 0.000e+00); gates out any
   host-sync leak into the captured steady-state decode. Commit `cd5b2782`.

## Multi-device / TP status (2026-08-18)

The sm_70 TP rollout is COMPLETE through the end-to-end dense token gate:

- The engine remains a **single-worker, tp_size==1** runner by default; the whole
  multi-device path is ADDITIVE (a null-default `tp` threads every forward, so the
  single-GPU engine is byte-identical). The distributed path runs when a caller
  passes a `TensorParallel` with tp_size>1.
- **NCCL TP transport (`VLLM_CPP_NCCL=ON`)** compiled into the sm70 build
  (VT_NCCL=1) with the conda torch's bundled `nvidia/nccl`.
- **Per-device affinity DONE**: `CudaDeviceScope` on every device-touching
  `CudaBackend` method — a `Device{kCUDA,i}` backend operates on GPU i
  (`test_cuda_backend` 7/7).
- **In-process collectives DONE**: `ncclCommInitAll` in one process,
  `vt::CudaCommGroup` + `vllm::{TensorParallel,TpShard,TpAllReduceSum}` seam,
  loader slice, runner forward; `test_nccl_group` 10/10.

### Multi-device — TP step-2 DONE (2026-08-18): per-rank dense decode + token gate
Three row/col-parallel PRIMITIVES + wiring, all committed + green:
1. **DenseMtpBlock tp>1** (`66f48f80`): host-decode fp4/bf16 gate/up/down →
   `vt_cuda_mlp_shard_run`/`_bf16` per token → reduced `[T,H]`.
2. **KV/GQA-shard attention** (`c643c311`+`9e246680`): softmax num/den both
   additive over kv — rank owns a kv-head slice, AllReduceSum reduces both,
   out = num/den; causal `[S,Hkv,D]` window.
3. **lm_head column-shard + AllGather** (`87ee78ce`): vocab-column-parallel,
   per-rank `[T,per]` partial logits, AllGather → `[T,vocab]`.
4. **Wiring + device-boundary fix** (`40bad253`): FullAttn/DenseMtpBlock thread
   tp; KV-head-split attention outputs the COMPLETE softmax per rank so o_proj
   stays a full GEMM. Fixed `ncclCommInitAll`/`Destroy` re-binding the calling
   thread's CUDA device (invalid-resource-handle) — both shard entries +
   `CudaCommGroup::Create` now save/restore the caller device.
- **REAL token-equal gate REACHED**: `test_op_parity -tc="qwen27 dense logits
  tp==tp1*"` on the actual 27B NVFP4 gives **4-GPU dense == tp1
  (8 tokens identical), SUCCESS** (the gate the spec marked UNREACHED).

Regression: `test_nccl_group` 12/12 (479), fa2 5/5, backend 7/7, 35B engine
token-equality 149/149 — tp1 path byte-identical.
Deferred: per-rank **paged-KV** in the `RunDenseLayerPaged` decode path, MoE
(A3B) expert TP.

## Known near-tie / caveats

- 27B tok6 tie (`198` vs `271`): native side requires the production CUTLASS
  build; sm_70 hand kernel yields the emulation side. Do not "fix" by special-casing.
- Quant routes differ by design: AWQ(INT4) oracle vs NVFP4 impl — compare
  semantically, never logit-exact.
- Decoder dequant/upload is a slow single-thread host-side path (perf item, not
  correctness).

## Next

1. ~~**Multi-device TP on sm70 (in progress).**~~ **DONE** — TP step-2 dense
   decode wired + real tp==tp1 token gate reached (2026-08-18). Volta-gen
   networks fine; kernels already per-shard. Remaining stretch: per-rank
   paged-KV in `RunDenseLayerPaged`, MoE (A3B) expert TP (both deferred).
2. Ada semantic oracle: `oracle-ada` at `55596792…` (Dockerfile updated for the
   24.04 PEP-668 base AND the cu128 torch pin — driver-capped, no driver change)
   when an Ada host is available.