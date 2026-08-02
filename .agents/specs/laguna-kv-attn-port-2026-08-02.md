# Laguna decode KV + attention → shared-framework port (W7 KV-attn) — 2026-08-02

`CLAIM-LAGUNA-KV-ATTN-BF16` — DGX-gated on `~/laguna-xs-nvfp4` (GB10), binary
`~/laguna-n4-build/build-cuda/examples/laguna-gen`, token-ids
`2,785,9626,377,15360,395`, `VT_LAGUNA_RESIDENT_DECODE=1 VT_LAGUNA_DECODE_GRAPH=1`
(the ~35 tok/s decode-graph path this task targets).

## Goal
Port Laguna's private, off-framework decode KV + attention toward vLLM parity. Laguna
keeps a PRIVATE per-layer **f32** KV (`LagunaKvCache::k_dev/v_dev`,
`LagunaGraph::cache_k/cache_v`) + a bespoke f32 GQA flash decode kernel family
(`DecodeAttnGqa{,G,Split,SplitG}Kernel`, `src/vt/cuda/cuda_laguna.cu`) — NOT the shared
`dense_attn::AttnBlock` / `vt::PagedAttention` / FA2 split-KV that 27B/35B use.

## Scope taken: the bf16-KV slice (NOT the full AttnBlock port)
The full `AttnBlock` port was SCOPED OUT (OWED) as too large + SACRED-risky for one
gated pass, because it requires, additively:
1. Extending the shared FA2 **pure-decode** path (`LaunchDecodeFA2Bf16`,
   `cuda_flash_attn_fa2.cu:705`) — which THROWS on `window_size.has_value()` — to
   support **per-layer sliding window** (Laguna is 12 global + 36 sliding-512) under
   the `seqlenq_ngroups_swapped` group-swap, plus a **softplus head-gate epilogue** hook
   after attention (Laguna's per-head `softplus(g_proj(x))` out-gate). Neither exists.
2. Growing `AttnBlock` (which hardcodes Qwen conventions) to carry Laguna's per-head
   QK-norm + YaRN/sliding dual-RoPE + `g_proj` — or a Laguna-specific block reusing the
   shared FA2 kernel. Both touch the SACRED 27B/35B `dense_attn_block.h` used by the
   sacred gates.

Given disk at 99% (incremental-only) + SACRED risk, this pass did the **tractable
slice** the task offered: convert Laguna's private KV to **bf16** (vLLM's own paged-KV
dtype), halving the DRAM bytes the memory-bound decode attention streams, entirely
inside Laguna's own kernels (SACRED path byte-untouched).

## What was implemented (tested diff, then REVERTED — see Result)
4 files, +122/−60, no new kernels:
- `laguna.h`: `k_dev/v_dev` `std::vector<float>` → `std::vector<uint16_t>` (bf16).
- `laguna_device.h`: `decode_attn_gqa{,_g}` cache K/V and `append_kv_row` dest → `void*`
  (bf16); the new-token row `knew/vnew` stays f32.
- `cuda_laguna.cu`: `DecodeAttnGqa{,G,Split,SplitG}Kernel` read the cache as
  `__nv_bfloat16*` and dequant → f32 in the shared-mem staging (dot/softmax stay
  f32-accum, = vLLM's bf16-in/f32-accum flash); the graph kernels round-trip the f32
  new row through bf16 (`__bfloat162float(__float2bfloat16(x))`) so the row attended
  this step == the bf16 the append writes for next step; `AppendKvRowKernel` casts
  f32→bf16 (`__float2bfloat16`, RNE) as it writes.
- `laguna.cpp`: eager + graph migration cast the host f32 prefill KV → bf16 via
  `vt::CastBf16` (RNE); eager append replaces the raw f32 `Backend::Copy` with a
  `vt::CastBf16` into the bf16 slot; a `DrainQueue` seals each one-time migration.

Build: incremental CUDA build on GB10 sm_121a, **7 TUs rebuilt, 0 warnings, BUILD_0**
(so the type change compiles clean across every header consumer).

## Gate result — WASH on speed + BREAKS the near-tie prefix (both runs, deterministic)

Prefix (task golden first-20 = `22345 83 268 33586 81 855 397 874 367 6376 815 340
9626 377 15360 83 1729 756 1205 565`):

| path | generated ids (first ~16) | prefix match | decode/step (best of 2) |
| --- | --- | --- | --- |
| **base f32 KV** | `22345 83 268 33586 81 855 397 874 367 6376 815 340 9626 377 15360 83 …` | **YES (≥20)** | 3.84s / 127 = **33.1 tok/s** |
| **bf16 KV** | `22345 83 22345 395 6885 365 340 12243 48055 929 377 340 6028 83 268 33586 …` | **NO — diverges at token 2** (`22345` vs `268`) | 3.93s / 127 = **32.3 tok/s** |

- **SPEED: WASH.** bf16 decode 3.93s vs base 3.84s (within run-to-run noise: base
  spread 3.84–4.03, bf16 3.93–4.07). It did **NOT** move the residual — at the XS gate's
  short context (~130 tok) the decode-attention KV read is a small fraction of decode
  time; `lm_head_gemv` (streams the ~600 MB tower) + Marlin MoE dominate (matches the
  prior nsys note "Marlin MoE only 8.8% at M=1"). Halving a small share = noise.
- **CORRECTNESS: near-tie FLIP.** bf16 output is coherent (it even re-emits the golden
  prefix from position ~14) but the argmax at **decode step 2** flips 268→22345, so it
  no longer shares vLLM's first-20 prefix. Note vLLM ITSELF uses bf16 KV yet emits 268:
  our DEVICE regime is not bit-identical to vLLM, so it sits in a near-tie where any
  perturbation (even one that moves us TOWARD vLLM's dtype) can flip a token. The f32 KV
  happened to align to vLLM's first-20; bf16 KV does not.

## Verdict + what landed
Two strikes (no speed gain + strict-prefix regression) → **NOT landed as a behavioral
change.** The default path stays f32 KV (base golden prefix + ~33 tok/s preserved;
re-verified on DGX after restore). The tested bf16-KV diff is recorded here (above) so a
future **long-context** re-measure (where the KV read grows to dominate attention, and
where the near-tie gate methodology could be re-based) can re-apply it directly.

## OWED (the real levers, unchanged by this)
1. **Full `AttnBlock` port** — the framework win (bf16 paged KV from the spec dtype,
   shared FA2 split-KV decode, in-graph slot_mapping KV write, drops Laguna's
   runner-routing + merged-gemm allowlist). Needs the shared FA2 decode grown for
   per-layer sliding-window + a post-attn gate hook, additively (§Scope). Large; SACRED.
2. **The XS decode residual is GEMV/MoE, not attention** — this wash CONFIRMS it. The
   ~17% gap to vLLM 42.46 lives in `lm_head_gemv` + Marlin MoE at M=1, not KV traffic.
