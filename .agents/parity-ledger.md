# Parity ledger — sm70-support (Milestone-0 pin)

Rows record a measured oracle-vs-impl comparison and its verdict. A row is
"live" until a later run supersedes it. Update in place, don't append noise.

## 1. 35B engine token-exact · GREEN
- impl: `test_qwen36_paged_engine` (sm_70 build, arch 70) on V100
- model: `nvidia/Qwen3.6-35B-A3B-NVFP4` @ `491c2f1ea524c639598bf8fa787a93fed5a6fbce`
- oracle: repo pinned golden (greedy_ids token-for-token)
- result: **315/315, Status SUCCESS** (2026-08-12, V100)
- verdict: same-model correct; marks the decoder as the load-bearing impl reference.

## Row | 27B dense forward · prefill-exact / tok6 near-tie

| impl | `test_op_parity -tc="qwen27 dense logits*"` (sm_70) |
| model | `unsloth/Qwen3.6-27B-NVFP4` @ `890bdef7a42feba6d83b6e17a03315c694112f2a` |
| result | prefill `argmax_match = 9/9`; greedy `6/16` (divergence @ tok6); top-1000 logit gap `1.296` (non-boundary) |
| analysis | tok6 near-tie: golden native `198`, emulation `271`. Emulation side is produced because the sm_70 build lacks `VT_CUTLASS_NVFP4` (sm_90+) — that guard is intentional (engine gate throws without it). sm_70 hand W4A4 kernel output is the legitimate answer on this silicon. |
| verdict | acceptable at semantic/near-tie; native side obtain 1 only on an sm_90+ production build. NOT to be closed by a special-case. |

## sm70 fp16-WMMA expert GEMM (brick H) · GREEN

| impl | `MoeGroupedGemmNvfp4WmmaF16` (fp16 mma.sync m16n16k16, fp32 acc) + decode launcher, `VT_SM70_MOE_WMMA` (default ON) |
| route | A3B-35B expert decode previously ran the CUDA-core naive fill (Marlin sm_75+, bf16-WMMA sm_80+; Volta has no bf16 MMA). Brick makes experts ride Volta fp16 tensor cores; bf16->fp16 saturates at ±65504 (fp16 5-bit exponent, prevents inf near-tie poison) |
| acceptance | `test_qwen36_paged_engine` (35B-A3B-NVFP4, real activations, incl 6 concurrent x 16 tokens) with brick default ON: **315/315 token-exact, Status SUCCESS** (2026-08-12, V100) |
| A/B | `VT_SM70_MOE_WMMA=0` returns the naive fill (the pre-brick baseline) |

## 27B-AWQ oracle capture · deterministic

| lane | 1Cat-vLLM 1.2.2 on V100 (`192.168.10.20:8000`, model `qwen3.6-27b-awq-mtp`) |
| battery | 8 prompts × 3 runs, greedy T=0, `enable_thinking:false`, max_tokens 32 |
| artifact | `tests/parity/goldens/qwen36_27b_awq_oracle/{trace,meta,prompt_ids}.json*` (commit `39d81e89`) |
| determinism | 24/24 rows identical across runs (no near-ties in battery) |
| caveat | **quant-route**: AWQ(INT4) vs our NVFP4 fast paths — grade semantically (greedy token-string + per-position argmax), never bit-exact. |
| comparator | `.agents/compare-oracle-impl.py` — smoke: oracle-vs-itself = 8 exact / 0 diverged |

## Box fact (evidence)

```
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
Tesla V100-DGXS-32GB, 7.0, 32768 MiB   (x4)
```
sm70 premise: CONFIRMED indirectly (`h100` hostname is a red herring).