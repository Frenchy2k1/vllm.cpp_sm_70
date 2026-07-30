# Keep-quant fused ops → SHARED infrastructure + declarative merged-GEMM descriptor (2026-07-30)

Base: `e2ab0690`. Branch `keepquant-shared-ops-2026-07-30` (NOT pushed).
User directive: "both in parallel" + "also build the declarative descriptor" — the
extensibility-first answer to "do other archs gain from what we do?" Today the fused
keep-quant MoE kernels were siloed to DeepSeek; this makes them a first-class shared op
plus a declarative descriptor so any keep-quant MoE arch inherits them.

## What shipped

### 1. PROMOTED shared op `OpId::kMoeGateUpSwiGLUGrouped`
The DeepSeek-V4 Brick-6 fused routed-MoE **gate+up+SwiGLU** kernel — previously an
arch-private function pointer on the `MoeDeviceKernels::moe_gate_up_swiglu` seam
(`deepseek_v4_device.h`) — is now a first-class `vt::` op, MIRRORING the
`kMatmulBTQuantGrouped` pattern exactly:
- enum `OpId::kMoeGateUpSwiGLUGrouped` appended before `kCount` (no id renumber) — `include/vt/ops.h`
- typedef `MoeGateUpSwiGLUGroupedFn` + host shim `vt::MoeGateUpSwiGLUGrouped(q,out,act,gate_w,up_w,expert_ids,limit)` with full validation — `include/vt/ops.h` / `src/vt/ops.cpp`
- **CUDA** realization: the EXISTING `vt::cuda::MoeGateUpSwiGLUGroupedCuda` (the very
  `QuantDotGemmGroupedFusedSwiGLUKernel` DeepSeek's wrapper already calls) registered
  under the new OpId by an append-only `FusedMoeSharedRegistrar` — **BIT-IDENTICAL move**,
  a shared-op wrapper, not a rewrite — `src/vt/cuda/cuda_quant_dot.cu`
- **CPU** composite golden `MoeGateUpSwiGLUGroupedKernel`: two `MatmulBTQuantGroupedKernel`
  (gate,up) + the elementwise clamped-SwiGLU — `src/vt/cpu/cpu_quant_gemm.cpp`

Semantics (per output `(p,j)`, `e = expert_ids[p]`, `xq` = act row quantized to Q8_K once):
```
gate = min(FinalFactor·(gate_w[e,j]·xq),  limit)
up   = clamp(FinalFactor·(up_w[e,j]·xq), -limit, limit)
out[p,j] = gate · sigmoid(gate) · up          (α=1, β=0)
```
`limit = +inf` reduces to plain `silu(gate)·up` — a standard SwiGLU MLP with no clamp,
so a dense/MoE MLP of ANY keep-quant model is a 1-expert (or E-expert) instance.

CUDA support: the K-quant / IQ family (`Q2_K…Q6_K`, `IQ2_XXS`, `IQ3_XXS`) — the
DeepSeek-V4 IQ2_XXS/Q2_K towers. `Q8_0` keeps its dedicated grouped path. The CPU
provider composes `MatmulBTQuantGrouped`, so it accepts every block-quant dtype
(including the synthetic-gguf `Q8_0` the unit test builds).

### 2. DECLARATIVE DESCRIPTOR `vt::MergedGemmGroup` — `include/vt/merged_gemm.h` + `src/vt/merged_gemm.cpp`
The contraction-tier analog of `FusedRecipe` (which is contraction-FREE by design —
FusedChain owns norm/quant/act/rope GLUE and SANDWICHES a first-class GEMM via `fast_op`).
`MergedGemmGroup` names the mirror case: **N GEMMs that SHARE operand A**, each with its
own weight operand, collapsed by a fused elementwise **epilogue** into ONE launch.
```
struct MergedGemmGroup { int arity; MergedEpilogue epilogue; int fast_op; const char* name; };
enum class MergedEpilogue { kNone, kSiluMulClamp };
```
Grounded in the best-gemm-path "[FusedChain prologue → fast_op = <GEMM OpId> → FusedChain
epilogue]" shape (`.agents/specs/best-gemm-path-2026-07-30.md`): a group is the middle box
when the middle is several merged GEMMs plus a byte-exact epilogue.

Selection is by **(weight dtype, merge-arity, epilogue)**; a backend that has not
registered the fast op inherits the Tier-0 composite automatically — the same additivity
the FusedChain catalog has. Adding a merged-GEMM shape = ONE constexpr descriptor
(+ optionally one structural kernel), no model-site edits.

**Realized instance (end-to-end):** `kKeepQuantGateUpSwiGLU = {arity=2,
epilogue=kSiluMulClamp, fast_op=kMoeGateUpSwiGLUGrouped, "keepquant_gate_up_swiglu"}`.
Dispatch `vt::MergedGemm(...)`:
- fast_op registered for the device → ONE fused launch (the promoted shared op);
- else (or `force_composite=true`) → the **Tier-0 COMPOSITE**: `arity`× `vt::MatmulBTQuantGrouped`
  into f32 temps + the epilogue, BYTE-EXACT to the standalone-op sequence — the CPU
  reference tier, mirroring `fused_recipe.h`'s Tier-0/fast_op tiering.
The `force_composite` A/B is exactly what the unit test uses to prove composite == fused
byte-for-byte.

### 3. Unit gate (RED-first) — `tests/vllm/models/test_deepseek_v4_gguf_load.cpp`
Two new CPU cases over synthetic keep-quant gate/up towers:
- `MoeGateUpSwiGLUGrouped: fused shared op == 2× grouped GEMM + clamped-SwiGLU` — the
  fused op is BYTE-IDENTICAL to the manual composite; RED-first: a different `limit` changes output.
- `MergedGemm descriptor: fast op == Tier-0 composite (byte-identical)` — `force_composite`
  false vs true vs the fully-manual standalone composite, all byte-identical; asserts the
  descriptor names the promoted shared op as its `fast_op`.
CPU gate GREEN: **15/15 cases · 931 assertions** (13 pre-existing + 2 new; 300 assertions
in the 2 new cases), clean RelWithDebInfo `-Werror`.

## Reuse — "others benefit automatically"
The promoted op + descriptor make the tuned single-launch fused MoE kernel a SHARED
primitive: any keep-quant MoE arch calls `vt::MergedGemm(kKeepQuantGateUpSwiGLU, …)` (or
`vt::MoeGateUpSwiGLUGrouped` directly) and inherits the fast kernel for its dtype, with the
byte-exact Tier-0 composite as the portable fallback for any backend lacking the fast op —
no per-arch kernel work.

**The inheritable kernel win is MEASURED.** The promoted kernel IS the Brick-6
`moe_gate_up_mid` fusion, whose decode win was measured on the real 80.7 GB DeepSeek-V4 on
the DGX GB10: **10.27 → 10.80 tok/s (+5.2%, BIT-EXACT, `6795717f` MERGED)**, kernel
instances −12.2%. Promoting it to a shared op means every keep-quant MoE arch inherits that
class of win by calling ONE op instead of {2× grouped GEMM + clamped-SwiGLU}.

### Honest residual — the second-model e2e benchmark is BLOCKED, not skipped
A fresh before/after decode-tok/s benchmark on a DIFFERENT keep-quant MoE model is NOT
runnable in this environment, for reasons verified in source/on-box:
- **No second keep-quant grouped-MoE model exists in-tree.** DeepSeek-V2/V4 are the only
  users of the keep-quant grouped path; qwen3_5 / Qwen3-Coder MoE are fp4 / bf16
  (`MoeGroupedGemmNvfp4` / `MoeGroupedGemmBf16`), not block-quant, so the fused keep-quant
  kernel does not apply without a new bf16/fp4 structural-kernel tier (future work — the
  descriptor is ready for it: add one `MergedGemmGroup` + one kernel).
- **No K-quant MoE checkpoint on the DGX** (only diffusiongemma/nemotron/opt-125m; root
  disk 96% full) and **no host K-quant encoder** (`from_float` is populated only for the
  Q8_0/Q8_K ACTIVATION encodings, not the Q2_K…Q6_K weight encodings), so a synthetic
  K-quant tower for an on-GPU A/B cannot be built either. Downloading a multi-GB MoE
  checkpoint onto a 96%-full root risks the documented OOM-reboot / disk-fill hazards.
So the reuse win is proven at the KERNEL level (the shared op + the measured Brick-6 +5.2%)
and the framework level (descriptor byte-exact across tiers), with the full second-model
e2e decode-tok/s benchmark a NAMED residual pending a runnable keep-quant MoE checkpoint.

## Concurrency
Front B (Brick 8) owns `deepseek_v4.cpp` + the QuantizeQ8K fusion in `cuda_quant_dot.cu` /
`cuda_deepseek_v4.cu`. This change owns `op_ids`/`ops.h`/`ops.cpp`/`merged_gemm.{h,cpp}` +
the CPU provider, and adds ONLY an append-only registrar + shared-op wrapper in
`cuda_quant_dot.cu` (no refactor of B's kernels). `deepseek_v4.cpp` is UNTOUCHED (it stays
on its working private-seam path; reuse is proven via the shared op + descriptor, not by
retrofitting the DeepSeek call sites).

## Gates
- CPU: `test_deepseek_v4_gguf_load` 15/15 · 931 (new shared-op + descriptor cases GREEN, RED-first).
- DGX GB10 (RelWithDebInfo, CUTLASS 4.5.0): DeepSeek regressions stay GREEN + bit-exact —
  `test_cuda_deepseek_v4`, `test_cuda_quant_dot`, `test_deepseek_v4_gguf_load` (the promotion
  is a bit-identical move of the same kernel; see BENCHMARKS.md for the run).
- No new env flag (default CPU/CUDA providers).
