# SPIKE / SPEC — Gemma-4 MoE ROCm FP8 + SharedK-WMMA (#317)

**Status:** ACTIVE lab path (dual R9700 gfx1201) · PR tip `feat/gemma4-rocm-fp8-split`  
**Claim class:** ROCm serve correctness + decode/prefill knobs (not CUDA SACRED parity)  
**Related PRs:** #234 sampler · #227 KV fail-fast · #316 SSE · #328 bare toolcall · **this**

## Problem
Gemma-4-26B MoE FP8 on consumer RDNA4 needs: resident dual-GPU experts, peer mix, SharedK-WMMA prefill, decode multi-CTA KV splits, and portable `vt::` seams so `models/` never names HIP (device-leakage).

## In scope (this PR)
| Area | Files (representative) | Gate |
|------|------------------------|------|
| Portable fused_ops / backend hooks | `include/vt/fused_ops.h`, `src/vt/fused_ops.cpp`, `backend.h` | CPU link + device-leakage |
| FP8 ExpertGeGLU / channel GEMV | `rocm_fp8_channel_gemv.hip`, `rocm_gemma4_experts.hip` | Lab A/B + quality (Paris/arith) |
| MoE host policy | `gemma4_moe.cpp` / `.h` | Lab load + decode |
| Prefill SharedK-WMMA | `rocm_paged_attn.hip` (prefill path) | Prefill eng ~2k class @11–12k |
| Decode KV splits / slide | `rocm_paged_attn.hip` | `VT_ATTN_DECODE_*` recipe |
| Env surface | `docs/ENVIRONMENT.md` `VT_GEMMA4_*` / `VT_ATTN_*` | check-env-doc |

## Out of scope
- SSE keepalives (#316), ROCm sampler (#234), bare tool parser (#328)
- CUDA SACRED gemma4-E4B text golden (dgx) — unchanged by ROCm HIP paths when `HasRocm` off
- Full MoE token-exact vs vLLM on RDNA4 (no AMD CI runners)

## Env recipe (lab binding)
See `docs/ENVIRONMENT.md` and lab `gemma4-fp8-recipe.env`:
`VT_GEMMA4_FP8_HW_CVT`, `VT_ATTN_DECODE_KV_SPLITS`, `VT_ATTN_DECODE_SLIDE_*`, resident experts, prefill GEMM_M / PEER_ACT / SharedK-WMMA.

## Automated coverage in-tree
1. **Device-leakage** — models/ must not call `vt::rocm::*` (CI `check-device-leakage`).
2. **CPU unit** — `test_gemma4_rocm_fp8_seams` (this PR): fused_ops symbols resolve; env knobs parse inert defaults on CPU.
3. **Existing** — `test_gemma4_paged_engine` / registry e2e remain CUDA/dgx optional skips.
4. **Lab (not CI)** — exclusive decode_depth_curve + Paris/READY on dual R9700; evidence in contributor lab notes, not forced into STATUS ratchet growth.

## Residuals (named, not silent)
- HIP guards polish in dense `gemma4.cpp` paths
- Broader parity matrix rows when AMD CI exists
- Further decode micro-opts stay lab-recipe until KEEP+quality

## Merge criteria (maintainer)
- Spec present (this file)
- Device-leakage OK; env documented
- CPU seam test green
- No SSE/sampler/toolcall scope creep
- Sanitize ambient reds treated as main baseline unless unique product fail

## Scope note (2026-08-11 rebuild on main)

This tip is **ROCm FP8 MoE + SharedK-WMMA + fused_ops + expert LRU/prewarm**.

**Explicitly deferred** (localai-bot hold on prior tip):
- `ForwardGemma4Layers` extract / layer-loop restructure in `gemma4.cpp`
- `Gemma4DecodeGraph` / pure-decode hipGraph driver
- Any unguarded CUDA-path forward refactor requiring GB10 token-exact golden

Rationale: those changes touch the gate model forward on all backends. ROCm kernels
and env-gated MoE paths do not. Revisit as a **separate** PR with CUDA golden.

`VT_GEMMA4_MLP_MOE_PARALLEL` is documented but **not** wired in this tip (was only
in the deferred layer-loop path).
