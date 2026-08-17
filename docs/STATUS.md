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

## Known near-tie / caveats

- 27B tok6 tie (`198` vs `271`): native side requires the production CUTLASS
  build; sm_70 hand kernel yields the emulation side. Do not "fix" by special-casing.
- Quant routes differ by design: AWQ(INT4) oracle vs NVFP4 impl — compare
  semantically, never logit-exact.
- Decoder dequant/upload is a slow single-thread host-side path (perf item, not
  correctness).

## Next

1. GEMM-family op-consumer hookup out of the paged decode head (Phase-4).
2. Wider-head `fused_mma_forward_paged` superset; y-dg-sync coverage.
3. Ada semantic oracle (`oracle-ada`, `55596792…`) when an Ada host is available.