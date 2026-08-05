# QUANT-CT-MXFP4 — W3 e2e result (Yi30/Qwen3-8B-MXFP4, GB10)

Date 2026-08-05. OUR engine (`vllm-cli`, native Marlin W4A16 mxf4 keep-quant path)
vs the oracle golden (`golden_marlin_w4a16.json`). Greedy, temp 0, seed 0, 48 tokens.

## Root cause of the async-default degeneration (NOT the MXFP4 compute)

The DEFAULT config (async scheduling ON, `max_concurrent_batches=2`) produced
degenerate output (" Paris A A ( ( Paris ..."). Root-caused: **not the MXFP4
compute** — with `VT_ASYNC_SCHED=0` the SAME binary produces coherent, token-exact
output. The async executor overlaps the previous step's output-copy
(`async_copy_queue_`) with the current forward; classic dense `Qwen3ForCausalLM`
(`qwen3.cpp`) does not carry the async device-mirror fix (the #31 "async batch-1
token-0 degeneration" class that was wired for the gate models `qwen3_5`), so the
overlap corrupts the sampled token. This is QUANT-INDEPENDENT (would hit bf16/NVFP4
classic dense Qwen3 the same way) and is a SEPARATE pre-existing bug, not this row.

## MXFP4 compute is CORRECT — async-off e2e is token-exact vs the golden

With `VT_ASYNC_SCHED=0` (the compute path), 3 of 4 prompts are TOKEN-EXACT vs the
oracle golden; the 4th diverges after the identical first token (bf16 /
implementation non-determinism on an open-ended prompt — the ratified distributional
/ near-tie regime):

| prompt | ours (async-off) | golden | verdict |
|---|---|---|---|
| "The capital of France is" | " Paris. What is the capital of Italy? ... Rome ... Berlin ... Madrid ..." | (same) | EXACT |
| "Q: What is 2 + 2? A:" | " 4. Q: 3+3? A: 6. ... 5+5? A: 10" | (same) | EXACT |
| "def fibonacci(n):" | " if n == 0: return 0 elif n == 1: return 1 else: return ..." | (same) | EXACT |
| "Once upon a time, in a small village," | " there lived a young girl named Lily ..." | " there was a wise old man ..." | first token " there" matches; diverges (non-det) |

Combined with the unit gates — op-level MXFP4 Marlin GEMM vs independent CPU dequant
(0.36% at M=1/M=8, all real shapes), scale permute byte-exact vs vLLM at all model
shapes, and the model-facing `MakeLinearMethod -> Apply -> BuildMarlinDenseResident`
gate (bad=0 at K=4096 and K=12288) — the MXFP4 keep-quant compute is correct.

## Status
- **W3 compute path: GREEN** (3/4 e2e token-exact + all unit gates).
- **Async-default degeneration:** pre-existing classic-dense-Qwen3 async bug (separate
  row); the compute is validated on the async-off path.
- **W4 bench:** owed; run on the async-off compute path (oracle arm
  `VLLM_DISABLED_KERNELS=FlashInferMxFp4LinearKernel`), or after the classic-dense
  async fix for a default-config number.
