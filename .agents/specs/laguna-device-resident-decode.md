# Laguna-NVFP4 device-resident decode → vLLM 18.8 (byte-exact) — executable plan

**Goal (USER 2026-08-01):** "we can't be lower than vLLM. IF vLLM does 18.8, technically
we should be able as well." Laguna Marlin default = ~10 tok/s; residual 1.9× to vLLM-NVFP4
18.8 is **host-orchestration** (MEASURED), not kernels (the Marlin MoE kernel IS vLLM's
own). AGENTS.md §396: same arch/model/GPU → the number is reachable; the "ceiling" is a
specific closable diff. **Byte-exact** — changes only WHERE buffers live, not the math
(directly NOT a correctness dodge; near-tie is OFF the table per the user).

## Measured root cause
`LagunaForwardGgufCached` (laguna.cpp:1278) is HOST-ORCHESTRATED: every `LqGemm` /
`LqGemmNvfp4Fp4` / `LagunaMoeResidentMarlin` takes a host `std::vector<float>` in and
returns one out (upload→GEMM→download), and the glue (`RmsNorm`/`ApplyRope`/`GateUpSilu`/
attention 712-745/residual adds) runs on host vectors. So each of 48 layers round-trips
the activation host↔device many times. nsys (whole-process) confirmed host API dominates;
the GEMV-path record measured ~22,115 `cudaStreamSynchronize`/token. A **decode-isolated**
number (60-tok − 20-tok run ÷ 40) is being captured to `lag60_diff.txt` to size the split
(malloc→scratch-pool / memcpy→residency / sync→graph).

## The build (mirror DeepSeek `ForwardResidentDecodeGguf`, laguna.cpp analog)
Write `LagunaForwardResidentDecode(weights, q, tok, pos, kv, be)` for T=1, gated
`VT_LAGUNA_RESIDENT_DECODE` (default-OFF until byte-exact-proven, then ON), guarded by a
`CanRunResidentDecode`-style check (T==1, CUDA queue, nvfp4 present). Keep ONE device
activation buffer `x[H]` live across all 48 layers; download only the final `[V]` logits;
ONE step-boundary drain.

**Per-layer op → device kernel (reuse existing; all in cuda_deepseek_v4.cu / dense glue):**
1. input RMSNorm → `RmsNormRowsKernel` (bf16/f32, existing)
2. q/k/v proj → Marlin/`MatmulBT` device path writing into device buffers (NO download)
3. RoPE → `NormRopeRowsKernel` / a Laguna RoPE device kernel (standard GQA rope; DeepSeek's
   is MLA-split — may need a plain-rope variant)
4. attention (decode, GQA over the on-device KV cache) → `DecodeAttnKernel` analog; KV cache
   stays device-resident (append this token's K/V on device)
5. o_proj → device `MatmulBT`, accumulate residual on device (`AddKernel`)
6. post-attn RMSNorm → `RmsNormRowsKernel`
7. MoE → `LagunaMoeResidentMarlin` ALREADY device-resident, but make it take/return a
   DEVICE buffer (drop the host `std::vector` in/out at 463); router topk on device
   (`RouterGate`/`RouteWarp` from DeepSeek) OR keep the tiny topk on host (cheap, few floats)
8. residual add on device
Final: RMSNorm + lm_head `MatmulBT` → download `[V]` → argmax (host or `on-GPU argmax`).

## Gate (reproduction is a gate, byte-exact)
- Golden ids UNCHANGED vs the current Marlin default (`2,785,9626,377,15360,395` → same
  continuation `22345 83 290 350 674 330 5541 966 340 9626 377 15360 81 …`, first 13 match
  the vLLM-Marlin golden). Byte-identical ⇒ the residency change is proven correct.
- Same-binary A/B `VT_LAGUNA_RESIDENT_DECODE=0/1`; kernel-duration / steady-state per-step
  (cold legs discarded; wall-clock useless on GB10 per the reload-swing note).
- Then: decode CUDA-graph capture of the resident step (vLLM's mechanism) for the last mile
  to 18.8; and move the 238s load-repack fully to model-load (LagunaBuildMarlinResidents).

## Risks
- The attention/RoPE device kernels are DeepSeek-MLA-shaped; Laguna is standard GQA → may
  need plain variants (small). KV-cache-on-device append is the fiddliest piece.
- Byte-exactness of the on-device glue vs the host reference (RMSNorm/rope/silu order) — the
  RMSNorm-saga hazard; gate RED-first. If a device-glue op is a near-tie not byte-exact,
  that op stays host (residency still removes most round-trips) — do NOT relax the gate.

Same "Brick D" class as DeepSeek device-resident decode (task #228). This is the byte-exact
path to vLLM parity; the compute is already at parity via the shared Marlin kernel.
