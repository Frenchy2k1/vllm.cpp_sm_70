# QUANT-CT-MXFP4-FLASH-OCCUPANCY — the owed ours-vs-vLLM flash decode ncu diff + the occupancy/L2/codegen lever

<!-- spec-status: CLOSED -->
Row: `QUANT-CT-MXFP4-FLASH-OCCUPANCY` (helper, `row/QUANT-CT-MXFP4-FLASH-OCCUPANCY`).
Base: `origin/main` `f7a1e322`. Vehicle: `Yi30/Qwen3-8B-MXFP4` (dense
`Qwen3ForCausalLM`, W4A16 Marlin keep-quant); oracle arm
`VLLM_DISABLED_KERNELS=FlashInferMxFp4LinearKernel`. GB10 sm_121a.

## Why this row exists

`#68/#69` (FLASH-AUDIT) closed the compile lens: our flash source is byte-identical
to vLLM's pinned `2c839c33`; `-use_fast_math` was TRIED and REJECTED (measured
+21 us/call regression — the 246→255 reg bump cuts occupancy on a latency-bound
kernel). Fresh same-tool nsys: ours 168.8 vs vLLM 156.3 us/call at c8 (+12.5,
+450 us/step). `#69` characterized OURS (occ 8.3%, L2 53%, ~38% smem-scoreboard +
~37% CTA-barrier stalls) but the vLLM-side ncu was LOST to a box OOM-reboot — the
precise ours-vs-vLLM ncu diff is OWED. This row runs that diff on an idle box and
takes whichever lever it names.

## W0 free finding (from existing c8 nsys CSVs, no GPU) — the premise is REFRAMED

`gpu_trace_c8_dflt` (ours) and `vllm_offline_trace_c8` (vLLM), flash_fwd_splitkv
non-combine, c8 dominant decode:

| engine | grid (GrdX×Y×Z) | blk | Reg/Trd | mean us/call | combine |
|---|---|---|---|---|---|
| OURS  | 1×3×64 | 128 | **216** | **174.2** | 64×1×1 reg46 3.7us |
| vLLM  | 1×3×64 | 128 | **241** | **155.8** | 64×1×1 reg64 3.9us |

The grid is **IDENTICAL** (num_splits=3, gridZ=64=batch×kv_heads — same GQA-pack,
same split heuristic), and **ours uses FEWER registers (216 < 241)** → ours has
MORE occupancy headroom, not less. The `#69` "8.3% occupancy, register-limited"
number was the batch-1 short-context num_splits=1 kernel (a DIFFERENT regime); the
prior ncu pair was context-MISMATCHED (ours ~5-token cli prompt vs vLLM lens=1024).
So at the real c8 decode kernel, occupancy is NOT the vLLM advantage. The residual
is codegen (our native sm_121a SASS vs vLLM's sm_80-PTX driver-JIT SASS) or L2.

## Work items

- **W1** — matched-workload ncu diff on BOTH engines at c8 (grid 1×3×64):
  OURS `vllm-bench --input-len 1024 --output-len 128 --concurrency 8`; vLLM
  `vllm_offline_decode.py NPROMPTS=8`. Full section set (LaunchStats, Occupancy,
  SOL, MemoryWorkload/L2/DRAM, WarpStateStats, SchedulerStats). Table the diff;
  disambiguate codegen (SM throughput / instruction stats differ) vs L2 (memory
  section differs).
- **W2** — the lever the diff names, mirror-first. Leading candidate: compile our
  flash TUs for `80-virtual` (compute_80 PTX → driver JIT to sm_121), MIRRORING
  vLLM's exact build path, if the diff shows codegen. A build-flag change like
  fast-math — runtime arbitrates; keep the `#69` lesson (measure, never assume).
  Byte-exact-first; near-tie razor + full battery if the reduction order shifts.
- **W3** — binding c1..c8 ×3 production defaults vs `1.020/0.962/0.966/0.969`;
  THE PARITY VERDICT (≥1.0 every axis ⇒ MXFP4 DONE; short ⇒ honest residual map).

## Gates

Byte-exact razor: #44 MXFP4-8B smoke 3/3 token-exact + coherent. If reduction
order shifts: SACRED 0.6B/4B distributional + 32B strict, async, memcheck, eager
+ graphed. Box safety: BOTH flock locks, free -g ≥ 90, worker STOPPED, tmux +
done-markers, sequential arms, single-load steady-state.

## CLOSED — verdict (2026-08-06)

W1 DONE (matched-c8 ncu, both engines, grid 1x3x64): **vLLM runs at the SAME
8.33% occupancy** (both smem-limited to 1 CTA/SM by the byte-identical 81.92 KB
smem), L2 ~1% both, identical stall structure. The ONLY measured diff was +13%
executed instructions — the CODEGEN hypothesis. W2 tested every mirror-first
lever and REFUTED all of them: arch-mirror (compute_80 PTX driver-JIT) is
neutral; and building vLLM's EXACT recipe (compute_80 + `-use_fast_math`)
reproduces vLLM's SASS profile EXACTLY (241 reg, 17,008 instr vs vLLM 241/17,020)
and is STILL ~167 us, ~10 us slower than vLLM's ~157 us. So matching vLLM's arch,
fast-math, register AND instruction count does NOT close the gap: the kernel is
not instruction-bound, and the residual is vLLM's wheel-`ptxas` SASS-scheduling
quality, un-reachable from our nvcc 13.0. W3 VERDICT: MXFP4 stays BELOW-FLOOR
(binding unchanged 1.020/0.962/0.966/0.969), NO lever exists on our stack, NO
default flip owed. No functional code shipped; a CMakeLists NOTE + benchmark
record (#75) capture the closed levers so they are not re-tried. NO byte-exact
razor / SACRED battery owed (nothing shipped to gate).
