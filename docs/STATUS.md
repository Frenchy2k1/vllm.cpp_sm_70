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

## Multi-device / TP status (2026-08-12)

- vllm.cpp engine is a **single-worker, tp_size==1** runner: `qwen3_*` model
  class treats tp_size as a no-op ("tp_size==1 ⇒ whole tensors + all-reduce
  identity"). TP in the fork that ran -tp4 on the V100 was **upstream 1Cat-vLLM**,
  not this engine.
- NCCL TP transport exists (`vt/cuda/nccl_communicator.cu`, PORTED 1:1) but is
  build-gated `VLLM_CPP_NCCL` (our build is **OFF**) and is infrastructure only,
  not wired into a multi-worker runner.
- **Next phase: multi-device TP on sm70.** Volta is NOT the blocker (NCCL +
  per-shard sm70 kernels are fine); the work is a real build (enable
  `VLLM_CPP_NCCL`, NCCL lib on the box, runner slice/all-reduce wiring).

### Multi-device — Phase 1 DONE (2026-08-12)
- **NCCL transport compiled into the sm70 build.** `VLLM_CPP_NCCL=ON`, NCCL from
  the conda torch's bundled `nvidia/nccl` (`libnccl.so` symlink + header).
  `nccl_communicator.cu` compiles to the REAL `vt::Communicator` (VT_NCCL=1),
  links into `libvllm.a`; `test_sm70_fa2_decode` still 5/5 SUCCESS (single
  device unchanged).
- Phase 2 (next): per-op `cudaSetDevice` affinity so a device-i backend
  allocates/launches on GPU i (the code flagged CREAMSKILL gap in
  `cuda_backend.cu`); Phase 3: runner slice + AllReduce/AllGather wiring for
  tp>1 across the 4×V100.

## Known near-tie / caveats

- 27B tok6 tie (`198` vs `271`): native side requires the production CUTLASS
  build; sm_70 hand kernel yields the emulation side. Do not "fix" by special-casing.
- Quant routes differ by design: AWQ(INT4) oracle vs NVFP4 impl — compare
  semantically, never logit-exact.
- Decoder dequant/upload is a slow single-thread host-side path (perf item, not
  correctness).

## Next

1. **Multi-device TP on sm70 (in progress).** Enable `VLLM_CPP_NCCL` in the
   build, bring NCCL onto the box, wire the runner slice + all-reduce/all-gather
   so -tp>1 works across the 4×V100. Volta-gen networks fine; the kernels are
   already per-shard.
2. Ada semantic oracle: `oracle-ada` at `55596792…` (Dockerfile updated for the
   24.04 PEP-668 base AND the cu128 torch pin — driver-capped, no driver change)
   when an Ada host is available.