# QUANT-CT-MXFP4-FLASH-PTXAS — arbitrate the ptxas lineage behind vLLM's faster flash decode SASS

<!-- spec-status: CLOSED -->
Row: `QUANT-CT-MXFP4-FLASH-PTXAS` (helper, `row/QUANT-CT-MXFP4-FLASH-PTXAS`).
Base: `origin/main` `362a3c99`. Vehicle: `Yi30/Qwen3-8B-MXFP4` (dense
`Qwen3ForCausalLM`, W4A16 Marlin keep-quant); oracle arm
`VLLM_DISABLED_KERNELS=FlashInferMxFp4LinearKernel`, vLLM 0.25.0-stage. GB10 sm_121a.

## Why this row exists

#75 closed the codegen lens to a MEASURED verdict: same `flash_fwd_splitkv` source
(vendored from vLLM's pinned `2c839c33`), same grid (1×3×64), same 8.33% occupancy
(both smem-limited to 1 CTA/SM by 81.92 KB), same L2 ~1%, and — with
`compute_80 + -use_fast_math` (Build C) — matched vLLM's register (241) AND
instruction (17,008 vs 17,020) counts EXACTLY. Yet Build C is ~167 us/call, vLLM
is ~157 us/call. #75 attributed the residual to "vLLM's wheel `ptxas`
SASS-scheduling quality (older CUDA 12.x lineage)". This row arbitrates that
attribution to a MEASURED yes/no: get the wheel's exact ptxas lineage, re-assemble
our flash PTX with it, and A/B the kernel.

## W1 — arbitrate the ptxas lineage
(a) Identify the wheel's toolkit lineage: `cuobjdump` the fa2 `.so`.
(b) Obtain that ptxas (venv `nvidia-cuda-nvcc-cu12` wheels or a scratch pip download).
(c) Compile our flash decode TU to PTX (compute_80 + fast-math per #75 Build C),
assemble with the candidate ptxas lineages → cubin, load via `cuModule` in a
microbench with the c8 decode params, and A/B (CUDA-event medians + ncu).
THE ARBITER: does a different-lineage-ptxas cubin hit ~156-157 us? NO ⇒ refutation,
close flash as measured-irreducible-for-us, STOP.

## W2 — vendor (only on a YES)
Mirror the GDN Triton-AOT precedent: commit the cubin + exact regen recipe under the
vendored-kernels tree; a load path routing the sm_121a decode flash launch through
the vendored cubin, gated `VT_FA2_VENDORED_CUBIN`, default per parity-enablers ONLY
IF the full battery is green (near-tie razor for the fast-math numerics; #44 smoke;
SACRED 0.6B/4B + 32B strict; async; memcheck; eager+graphed; both GQA ratios).
hd256 (27B/35B) cross-model projection measured, not flipped, as a follow-up.

## W3 — binding + verdict
c1..c8 ×3 production defaults vs 1.020/0.962/0.966/0.969. THE PARITY VERDICT:
≥1.0 every axis ⇒ MXFP4 parity DONE; short ⇒ honest terminal map with the fresh
decomposition (flash-after-arbitration + glue ~18% + marlin/host remainder).

## Gates
Byte-exact razor: #44 MXFP4-8B smoke 3/3 + coherent. If fast-math shifts the
reduction order: SACRED 0.6B/4B distributional + 32B strict, async, memcheck,
eager+graphed. Box safety: BOTH flock locks, free -g ≥ 90, worker STOPPED, tmux +
done-markers, sequential arms, single-load steady-state, disk floor 15G.

## CLOSED — verdict (2026-08-06, THE ARBITER = NO)

The ptxas-lineage hypothesis is REFUTED three ways. **(a) No old lineage exists:**
`cuobjdump` of vLLM 0.25.0's `_vllm_fa2_C.abi3.so` = 52 `sm_80` cubins + 52
`.target sm_80 .version 9.0` PTX (ISA 9.0 = **CUDA 13.0**, our own major), NO
sm_12x cubin. On GB10 the sm_80 SASS cannot run, so vLLM's flash SASS is
**driver-JIT'd from compute_80 PTX by the box CUDA-13.0 driver** — the SAME
assembler #75's Build C (`code=compute_80`) already used. There is no separate
"wheel ptxas" that baked a fast sm_121a cubin. **(b) Impossible by construction:**
the only sub-13 ptxas on the box (12.8) tops out at sm_120a and cannot target
sm_121a or read PTX 9.0 — an old-CUDA-12.x sm_121a cubin cannot be built at all.
**(c) Measured tie:** a same-params cuModule A/B (harness `bench_flash_ptxas.cu`,
c8 GQA-swap decode, grid (1,3,64), dyn-smem 81 920 B) times the byte-identical
decode kernel from our compute_80+fast-math PTX and from vLLM's PTX #30, each via
driver-JIT / ptxas 13.0 / ptxas 13.2 (all REG=241). Module/native ratios ∈
[0.969, 1.013] — a ±1.3% tie; the lone 0.969 (vLLM-PTX+ptxas13.0) is contradicted
by both sister vLLM arms (1.007, 1.013) and our-PTX+ptxas13.0 (1.004), i.e. box
drift, not a lever. **Corrected mechanism:** the flash kernel codegen is at PARITY
across every reachable toolchain AND vs vLLM's own PTX (~144–151 us in isolation);
the +10 us/call the engine shows (ours 167 / vLLM 157, #75) is ENGINE CONTEXT
(neighbour-kernel L2/orchestration, per #69), not the kernel's SASS. This RETIRES
#75's "ptxas SASS-quality" attribution. **W2/W3 none owed:** no vendor (nothing
beat the driver JIT we already use); binding UNCHANGED c1 1.020 / c2 0.962 / c4
0.966 / c8 0.969; MXFP4 terminal below-floor at c2–c8 with the corrected map
(flash codegen at parity; residual = context/glue/marlin-host). hd256 (27B/35B)
projection: structurally identical (sm_80-only PTX driver-JIT'd), NO hd256 vendor
owed. No functional code shipped; CMakeLists NOTE + benchmark-record (#82) capture
the closed levers. No byte-exact razor / SACRED battery owed (nothing shipped).
