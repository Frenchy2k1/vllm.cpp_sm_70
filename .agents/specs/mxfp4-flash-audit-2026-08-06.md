# QUANT-CT-MXFP4-FLASH-AUDIT — the flash decode gap vs vLLM (audit → MEASURED verdict)

<!-- spec -->

Row: `QUANT-CT-MXFP4-FLASH-AUDIT` (helper, worktree `row/QUANT-CT-MXFP4-FLASH-AUDIT`,
draft PR #69). Base: `origin/main` `4ce9fb74`. Vehicle: `Yi30/Qwen3-8B-MXFP4` (dense
`Qwen3ForCausalLM`, compressed-tensors W4A16 Marlin keep-quant). Oracle: vLLM **0.25.0**,
arm `VLLM_DISABLED_KERNELS=FlashInferMxFp4LinearKernel`.

## Target
The #67 refutation re-attributed the c2-c8 MXFP4 residual FLASH-dominant and OWED a
STRUCTURAL-lens audit of why the IDENTICAL-grid `flash_fwd_splitkv` decode kernel runs slower
on our side.

## W1 — DONE. The flash term is REAL, and on CURRENT main SMALLER than #57.
`analyze_decode2.py` c8 decode-window (both `nsys --cuda-graph-trace=node`, modal marlin=144 /
gridZ=64): ours 168.8 vs vLLM 156.3 us/call = **+12.5 us/call (+450 us/step)**, not #57's +807.
Current-main's leaner marlin/glue drop flash 178.1→168.8 — a hint the residual is L2/context.

## W2 — DONE. The mechanism is occupancy/L2, NOT `-use_fast_math`.
- **Lens 1 (cuobjdump, HYPOTHESIS):** vLLM v0.25.0 pins vllm-flash-attn @ `2c839c33` (exact
  commit we vendored) → source byte-identical; kernel-version REFUTED. vLLM's flash-attn is
  built `--use_fast_math`; ours was not (ours 5448 instrs/REG246 → 4832/REG255 with it, =
  vLLM's 4880/REG255, HMMA/LDSM/LDGSTS identical). SUGGESTED the scalar bloat was the gap.
- **Lens 2 (MEASURED, the arbiter):** controlled SAME-BUILD nsys A/B (dense-direct default,
  flash source identical, only the flag differs): `-use_fast_math` makes flash **+21 us/call
  SLOWER (168.8→189.8)**. Memory-latency-bound kernel; the higher reg count (246→255) LOWERS
  occupancy, which dominates the instruction reduction. cuobjdump was necessary but NOT
  sufficient (STRUCTURAL-lens "MEASURED not inferred").
- **ncu (OURS):** occupancy **8.3%** (register-limited, 4/12 warps/SM), L2 hit **53%**, ~38%
  short-scoreboard (LDSM→MMA smem) + ~37% CTA-barrier stalls = occupancy-starved latency
  exposure. vLLM at 255 regs is still 156.3 → its edge is L2/scheduling, not occupancy. The
  ours-vs-vLLM ncu diff was LOST to a shared-box OOM-reboot (twice, 3-way contention); OWED.

## W3 — VERDICT
`-use_fast_math` REJECTED + REVERTED (measured +21 us/call regression; a non-byte-exact change
with a negative speed effect); a CMakeLists NOTE records why so it is not re-tried. No
functional code ships. Token-exact footnote: fast-math passed #44 smoke 3/3 + coherent (a
faithful mirror of vLLM's build, just not a speed win). The +450 us/step flash residual is
occupancy/L2 — a default-flip's SACRED battery + c1..c8 binding are NOT owed (no win to flip).
NEXT lever (counter-pointed): lift flash occupancy above 8.3% (register pressure /
`__launch_bounds__`) or cut the barrier/smem stalls; and the vLLM-side ncu on an idle box.
