# Consistency Fold-Plan: Cross-Cluster Merged-GEMM Descriptor Rollout

Synthesis of 4 cluster inventories (qwen / deepseek+moe / gemma-and-dense / MM-towers), ~60 candidate sites. Feeds the descriptor-generalization front **wq6sodui3**.

---

## (1) Cross-arch pattern census

Counts fold *sites* (a model may own several), bucketed by current state.

| Pattern | Total sites | Un-fused (needs fold) | Hand-fused / siloed (needs promotion to shared) | Already-shared (exemplar / done) |
|---|---|---|---|---|
| **QKV-proj** | ~20 | 13 — Gemma×4, qwen3_dense(bf16 default), qwen3_coder, dflash, dflash_gguf, deepseek_v2 ×2 (MLA A-proj), whisper, gemma4_vision, gemma4_audio(host) | 4 — 27B (ResidentNvfp4Qkv), 35B (ResidentQkvFp8), qwen3_vl_vision (weight fused, compute split) | 3 — OLMo-2, Granite, StableLM (merged MatmulBT+QkvSplit) |
| **Gated MLP (SwiGLU/GeGLU)** | ~15 | 9 — Gemma×4, OLMo-2, Granite, StableLM, deepseek_v2 dense+ref-loop, deepseek_v4 shared-expert | 2 — gemma4_vision (bespoke [2I,H] DevW), qwen3_dflash (hand-called merged) | 4 — qwen3_dense (MlpGateUpMethodBase), 27B (fp4), qwen3_5_gguf, deepseek_v2 (merged GEMM, unfused epilogue) |
| **MoE gate-up** | ~7 | 4 — qwen3_coder(ref loop), qwen3_5_gguf(keep-quant loop), deepseek_v2, gemma4 PLE | 3 — ds4 (reference hand-fusion), 35B routed (cutlass 2-GEMM MIXED), 35B shared-expert(partial) | — |
| **Norm-rope / add-norm glue** | ~13 | 8 — Gemma×4, OLMo-2, deepseek_v2 MLA, whisper (LayerNorm), gemma4_vision ×2 | 2 — ds4 NormRopeRows, ds4 (feeds MLA) | 3 — qwen3_dense/27B/35B (kAttnQkNormRope[Gate], kFusedAddRmsNormStd) |
| **Non-gated MLP / bias-epilogue / other** | ~10 | most — whisper fc+bias, qwen3_vl fc+bias, gemma4_audio FFN/lconv(host), voxtral projector, dflash tap-combine, 35B GDN in_proj | — | — |

**Headline:** the merged-GEMM *weight layouts already exist* almost everywhere (loader packs `fused_qkv_a_proj`, `kv_a_proj_with_mqa`, `gate_up_proj`, resident `qkv_w`, stacked experts). The gap is **launch/epilogue granularity and seam routing**, not weight marshaling. Two clusters already hold the *reference tuned kernels* (nvfp4/fp8 exemplars in qwen3_5; keep-quant grouped + `moe_gate_up_swiglu` + `NormRopeRows` in deepseek_v4) — so most high-value folds inherit a tuned kernel for **zero new kernel code**.

---

## (2) RANKED fold order (grouped by shared-op target)

Ranking rule: **(a)** inherits an already-tuned kernel for free · **(b)** bit-exact ≫ reordering near-tie · **(c)** breadth of reuse. Bit-exact wide-reuse free-kernel folds first; near-tie/consistency-only and new-recipe folds last.

### TIER A — bit-exact, drop-in, inherits tuned kernel, wide reuse (do first)

**A1. bf16 SwiGLU MLP → `layers::UnquantizedMlpGateUpMethod` seam** *(shared-op: MlpGateUpMethodBase)* — ✅ **DONE 2026-07-30** (branch `worktree-wf_9054a036-47d-1`, NOT pushed)
Drop-in, byte-for-byte identical op sequence, inherits the nvfp4 `GateUpFusedMarlinD` arm for free the moment a quantized checkpoint ships. **All 5 sites folded; no loader concat needed (every arch already packs a merged `[2I,H] gate_up_proj`).**
- ✅ OLMo-2 — `olmo2.cpp` `Olmo2MlpBlock` → direct `UnquantizedMlpGateUpMethod` arm. **DGX SACRED token-exact gate RAN + PASSED** (`test_olmo2_paged_engine` 16/16 · 92 assertions; anchor no-op REQUIRE vs pre-fold `our_ids.npy` PASSED = byte-exact no-op on GPU).
- ✅ Granite — `granite.cpp` `GraniteMlpBlock` → full `layers::MakeMlpGateUpMethod` factory (weights ARE `Qwen3DenseMlpWeights`, so nvfp4-ready today). **Build-verified on production CUDA stack + link/load-verify (golden NOT committed → no vs-vLLM gate this pass); covered by CPU composite golden.**
- ✅ StableLM — `stablelm.cpp` `StablelmMlpBlock` → direct `UnquantizedMlpGateUpMethod` arm. **Build-verified on production CUDA stack + link/load-verify (golden NOT committed → no vs-vLLM gate this pass); covered by CPU composite golden.**
- ✅ qwen3_dflash — `qwen3_dflash.cpp` (was hand-merged, off-seam) → direct arm. **Build-verified + CPU composite golden.**
- ✅ deepseek_v2 dense/shared-expert epilogue — `deepseek_v2.cpp` `DenseMlp` (ONE fold covers BOTH the dense-layer MLP call site AND the shared-expert epilogue call site). **Build-verified + CPU composite golden.**

Exemplar copied: `qwen3.cpp:91-96` (already on the seam). **REUSE PROOF landed:** the CPU composite-golden RED-first unit case (`test_linear_method`, GREEN on the DGX production CUDA stack) shows the fused seam is BYTE-IDENTICAL to the standalone `{MatmulBT;SiluAndMul}` sequence; OLMo-2's SACRED gate proves that same shared method is token-exact end-to-end on the real GPU forward. Shared method TU (`linear.h`/`nvfp4.h`) UNTOUCHED → 27B/35B/qwen3_dense stay strict without re-gate. No new env flag. **HONEST:** 1 arch empirically token-exact-gated (OLMo-2); Granite/StableLM goldens were never committed (`granite_greedy_2b`/`stablelm_greedy_1_6b` do not exist) so those get build-verify + link/load this pass — capturing their oracle goldens is available follow-up.

**A2. MLA A-proj merge (3→1) → single `vt::MatmulBT` over `fused_qkv_a_proj`** *(shared-op: merged-QKV, MLA instance)* — ✅ **DONE 2026-07-31** (branch off `f2e463d5`, NOT pushed; landed TOGETHER with A5, they share the MLA path).
Bit-exact (identical GEMM math, merged N). The `kv_c`(nope)+`k_pe`(rope) A-projections now collapse to ONE `vt::MatmulBT` over the merged `[L+R, H]` weight rows: no-q-lora issues one GEMM over the already-merged `kv_a_proj_with_mqa`; q-lora slices `fused_qkv_a_proj` into `[0:ql]` (q_c, own GEMM → contiguous for its `q_a_layernorm`) + `[ql:ql+L+R]` (merged kv, one GEMM). **The contiguous-output-slice constraint was resolved WITHOUT touching `vt::RmsNorm`** (the documented deferral reason): the merged kv output's latent slice is strided, so the downstream norm is done by the NEW fused op A5 which reads the strided `[L+R]` row directly and internally splits — the plain-`RmsNorm`-needs-contiguity blocker never fires. Behind `VT_MLA_FUSED_NORM_ROPE` (default-ON; `=0` restores the byte-exact 3/2-GEMM split path). Net per-layer: q-lora 3→2 A-proj GEMMs, no-q-lora 2→1, plus the norm+k_pe-rope launch fold (A5).
- deepseek_v2 q-lora branch — `mla_attention.cpp` (was `:314-316`) ✅
- deepseek_v2 no-q-lora (kv_c+k_pe) — `mla_attention.cpp` (was `:332-335`) ✅
- **Free carry:** kimi_k3 inherits this once its forward lands (`kimi_k3.cpp:20-30`).

**A3. keep-quant grouped MoE fold → `kMatmulBTQuantGrouped`** *(shared-op: grouped keep-quant, Front-A / task #200-201)* — ✅ **DONE 2026-07-31** (Laguna `96b4652f` + qwen3_5 `ae201441`/`b4f5610a`).
Both keep-quant models now route their routed-expert MoE through the shared `vt::MatmulBTQuantGrouped` descriptor: **Laguna W9** (experts pre-stacked → grouped, byte-exact same-binary A/B, 1.38× decode) and **qwen3_5 GGUF W2+W3a+W3b** (stacked-tower loader + `ResidentWeight`-staged slice forward + grouped MoE; DGX byte-exact — grouped(=1) vs per-expert(=0) APEX-Compact continuations BYTE-IDENTICAL + strict golden pass; device-staging bug root-caused en route, see [[keepquant-device-slice-needs-residentweight]]). "Fold structurally, use consistently" realized on both keep-quant models.
"The biggest single fold in the cluster." Bit-exact (identical per-block vec_dot + f32 accum), inherits the ds4-tuned grouped keep-quant vec_dot for free; removes E host round-trips + E×2 tiny MMQ launches → ~3 grouped launches.
- qwen3_5_gguf — the per-expert host-gather loop (was `qwen3_5.cpp:4256-4266`) → ds4 `cuda_quant_dot.cu`/`cpu_quant_gemm.cpp`
- Free carry: qwen3_5_gguf dense MLP routes through the same descriptor.

**★ SCOPING VERIFIED 2026-07-31 (source-grounded, current main `64ce8eb1`) — premise HOLDS, but NOT a drop-in; it is a real loader+block fold, not a launch swap.** Traced the actual current path (line numbers had drifted): the qwen3_5 GGUF keep-quant MoE is the **host-gather reference loop** in the GGUF `MoeBlock` (`qwen3_5.cpp` ~`:5129-5170`) — it downloads router weights to host, builds per-expert `(token,slot)` lists on host, and calls **`ExpertMlp`** (`:4266`) per activated expert. `ExpertMlp` = `MatmulF32(gate)` + `MatmulF32(up)` + host SwiGLU + `MatmulBf16(down)`. **CRUX RESOLVED — the compute IS already keep-quant:** `MatmulF32`/`MatmulBf16` (`:841`) call `ResidentWeight(d,w)` which **does NOT dequant** — it wraps `w.bytes` at the keep-quant `w.dtype` (carrying the `repacked` i8mm marker, `:793-805`) and runs `vt::MatmulBT` (the block-quant vec_dot). So the per-expert loop already runs the SAME per-block vec_dot + f32 accum as ds4's `kMatmulBTQuantGrouped` ⇒ the grouped fold IS bit-exact by the ds4-established vec_dot identity (NO dequant-vs-keepquant numerics change — the hazard I checked for does not fire). The overhead being removed is real: per activated expert = 3 device GEMM launches + **2 full host round-trips** (`dout.Download` in `MatmulF32`) + host token memcpy gather.
**REMAINING WORK (3 parts, all CPU-buildable; DGX SACRED gate OWED):**
1. **Loader/materialization stacking** — experts are `std::vector<OwnedTensor> expert_gate/up/down` (per-expert, `:591`). `kMatmulBTQuantGrouped` needs a stacked `[E*N,K]` block-quant weight + an `eid` expert-index. All experts share quant-type+shape ⇒ stacking = byte-concatenation of the per-expert block buffers (mirror how the fp4/bf16 fast paths build their resident pointer arrays at `:4986-4988`). Fold-plan gap #5.
2. **Keep-quant grouped MoE block** — a `MoeBlock…KeepQuantGroupedCuda` analog of the existing `MoeBlockFusedCuda` (fp4) / `MoeBlockBf16Cuda` (bf16), consuming the shared descriptor exactly like ds4's `MoeFusedResident` (`deepseek_v4.cpp:531,560,1266`): build pair→token `row_map`, one grouped gate + one grouped up (or the fused `kMoeGateUpSwiGLUGrouped`) + device SwiGLU + one grouped down; **route-weight stays in `moe_combine`** (invariant). Default-ON flag with byte-exact same-binary fallback to the `ExpertMlp` loop.
3. **RED-first byte-exact A/B unit test** — grouped == per-expert `ExpertMlp` loop, raw memcmp (RED-first: a wrong stack-stride / scale fails).
This is the ds4 keep-quant grouped MoE block given to qwen3_5 GGUF (which today has fp4 + bf16 fused paths but NOT a keep-quant grouped one — precisely the A3 gap). Sizable (loader stack + new block + test); sequenced after the Laguna speed line. GPU cap (200/200 agents) reached ⇒ executed hands-on when the GPU flock frees.

**A4. bf16 grouped MoE gate-up + fused SwiGLU → NEW shared `vt` op `kMoeGroupedGemmBf16GateUpSilu`** *(task #201)* — ✅ **DONE 2026-07-31** (branch `fold/a4-bf16-grouped-moe` off `d05fff2d`, NOT pushed).
ds4 already proved the dot core + SwiGLU epilogue bit-identical to separate-GEMM. **OPEN-QUESTION RESOLVED (§3 gap 3): qwen3_coder ALREADY routes to task #90's fast bf16 grouped-MoE GEMM — NOT the `qwen3_5.cpp:4256` reference loop.** Coder's MoE flows `qwen3_moe.cpp` `RunMoeBlock` → `qwen3_5.cpp` `MoeBlock` → **`MoeBlockBf16Cuda`** (guarded on `kMoeGroupedGemmBf16` registered + `MoeBf16FastEnabled()` default-ON + `MoeBf16FastLayoutOk`), which runs `{vt::MoeGroupedGemmBf16(gate f32); vt::MoeGroupedGemmBf16(up f32); vt::MoeSiluMul}`. So A4 for coder is PURELY a launch-reduction/consistency fold (fewer launches), NOT a perf-gap fix. **HONEST DEVIATION on the shared-op target:** the bf16 grouped path marshals experts as per-expert weight-POINTER ARRAYS `[E] i64` + a pair→token `row_map` (NOT the contiguous `[E*N,K]` + broadcast-act convention the keep-quant `kMoeGateUpSwiGLUGrouped` signature takes), so the bf16 arm is realized as a SIBLING op `kMoeGroupedGemmBf16GateUpSilu` (bf16 twin of the keep-quant op; documented in `merged_gemm.h`) rather than literally the same OpId. BIT-IDENTICAL to `{2× vt::MoeGroupedGemmBf16 (f32) + vt::MoeSiluMul}` in every regime by construction: decode/non-WMMA reuses the exact `MoeGroupedGemmBf16NaiveSplitK` partials + a fused reduce+SwiGLU (5→3 launches, drops the two f32 [P,I] reduce round-trips); prefill/WMMA reuses `LaunchGroupedBf16` twice + the identical silu-mul (byte-for-byte the old three ops). Default-ON behind `VT_MOE_BF16_FUSED_GATEUP` (`=0` = composite everywhere, same-binary rollback). Route-weight stays in `moe_combine` post-down (invariant preserved). **No loader change needed** (both archs already pack the per-expert bf16 gate/up towers as resident device pointers). GATED (DGX GB10 sm_121a, RelWithDebInfo + cutlass-4.5.0 + triton): A/B unit `test_ops_moe_grouped_bf16_gate_up_silu` PASS (fused == composite byte-identical, RED-first memcmp, naive/split-K/WMMA regimes); `test_deepseek_v2_paged_engine` SACRED **8/8 UNCHANGED** (223 assertions, near-tie prompts/gaps IDENTICAL to the A2/A5 baseline ⇒ bit-exact); `test_qwen3coder_paged_engine` vLLM-0.25.0 golden **6/6 PASS** (138 assertions); `test_ops_moe_grouped_bf16` regression PASS.
- deepseek_v2 — `deepseek_v2.cpp:405-408` (Brick-6 shape) ✅
- qwen3_coder — `qwen3_5.cpp` `MoeBlockBf16Cuda` (reached via `qwen3_moe.cpp` `RunMoeBlock`; ALREADY on task #90's fast GEMM) ✅
- **Free carry:** kimi_k3 inherits it once its bf16 grouped-MoE forward lands.
- ds4 shared-expert reuse — `deepseek_v4.cpp:1413-1415`,`:1627-1629` (KEEP-QUANT, already on `kMoeGateUpSwiGLUGrouped`; NOT a bf16 site — out of A4 scope).
- Reference promoted: ds4 `moe_gate_up_swiglu` STRUCTURE. **Constraint met: route-weight stays in `moe_combine` post-down (not folded).**

**A5. MLA norm-rope → shared `kFusedNormRope`** *(shared-op: FusedChain norm-rope, MLA instance)* — ✅ **DONE 2026-07-31** (branch off `f2e463d5`, NOT pushed).
Bit-exact. **HONEST DEVIATION from the literal "promote ds4 NormRopeRows" plan:** ds4's `NormRopeRowsKernel` is structurally a DIFFERENT op — it RMS-reduces the FULL `[nope+rope]` head vector (per-head qk-norm) with the weight over all of it, and recomputes cos/sin ANALYTICALLY (base/freq/YaRN). DeepSeek-V2's decoupled MLA is not that: `kv_a_layernorm` normalizes the LATENT `[0,L)` ONLY (RMS domain = L, not L+R), the decoupled `k_pe` is roped UNNORMED/UNWEIGHTED, and the rope reads a PRECOMPUTED cos/sin CACHE. Blindly reusing NormRopeRows would be a NON-bit-exact fold (wrong RMS domain + analytic-vs-cache cos/sin). So `kFusedNormRope` is a NEW shared vt op (CPU+CUDA, `ops.h`/`ops.cpp`/`cpu_ops.cpp`/`cuda_ops.cu`), grounded in ds4's row-parallel norm+rope STRUCTURE but adapted to V2's decoupled semantics: one launch reads the merged `[T, L+R]` kv row, RMS-norms `[0,L)` with `kv_a_layernorm` (mirroring `RmsNormRowKernel` exactly), and rotates `[L, L+R)` from the cos/sin CACHE (mirroring `RopeFromCacheKernel` exactly, GPT-J/neox selectable). BIT-IDENTICAL to `{vt::RmsNorm(latent); vt::RopeFromCache(k_pe)}` — RED-first proven (`tests/vt/test_ops_fused_norm_rope.cpp`, f32+bf16, both rope styles; a full-width-RMS or normed-rope bug fails the byte-check). The query-side `q_pe` rope stays a distinct standalone `RopeFromCache` binding (per-head-independent → bit-identical whether roped with k_pe or alone); the ds4 per-head/inverse-o-RoPE forms are UNCHANGED. Decoupled asymmetry (latent normed, rope part not) preserved by construction.
- deepseek_v2 MLA — `mla_attention.cpp` (was `:347-367`) → new `vt::FusedNormRope` (`kFusedNormRope`) ✅
- **Free carry:** kimi_k3 inherits it once its MLA forward lands.

**GATED (A2+A5 together, DGX GB10 sm_121a, RelWithDebInfo + cutlass-4.5.0 + triton, mandatory-flag-proven):** `test_deepseek_v2_paged_engine` SACRED **8/8 UNCHANGED** (92/128 strict, 3/8 near-tie, max gap 0.25 nats @ prompt[3] tok=9 — IDENTICAL to pre-fold; committed `our_ids.npy` anchor matched; goldens md5 BEFORE==AFTER) ⇒ end-to-end BIT-EXACT. `test_ops_fused_norm_rope` 2/2 (CPU RED-first + CUDA), `test_deepseek_v2_load` 4/4, `test_deepseek_v2_forward` 11/11. Shared-op inertness: `test_qwen27_paged_engine` **235/235 (16/16)** (near-tie razor canary), `test_qwen3_paged_engine` 0.6B 16/16 + 4B 16/16. 35B not run (structurally inert: shared RmsNorm/RopeFromCache/GEMM untouched; op additions are additive). All 4 record checkers rc=0.

### TIER B — bit-exact glue folds onto existing FusedChain catalog (Gemma sweep)

**B1. Gemma attn-preamble → `kAttnQkNormRope` / `kAttnQkNormRopeGate`** *(shared-op: FusedChain, already in catalog)* — ⚠️ **CHARACTERIZED NOT-BIT-EXACT-FOLDABLE onto the EXISTING catalog (2026-07-30); DEFERRED, not silently swapped.**
Composite == the 3 standalone ops → bit-exact **provided the cos/sin cache path is used** (no `RopeNeox`→`RopeFromCache` swap in the same step, which would trip the Qwen3 1-ULP FMA near-tie). The B1 hazard FIRED on inspection:
- Gemma-1/Gemma-2 — **N/A** (no q/k-norm; the preamble recipe needs the norm weights, and both use `RopeNeox` on a raw q/k).
- Gemma-3 (`gemma3.cpp:145-152`) — qk-norm is **GemmaRMSNorm (1+w, `gemma=true`)** AND uses **`RopeNeox`** (not the cos/sin cache). `kAttnQkNormRope` is `gemma=false` + `RopeFromCache` (TWO mismatches: norm variant + a forbidden RoPE-impl swap); `kAttnQkNormRopeGate` carries an attention **output gate** Gemma-3 does not have. Neither existing recipe is bit-exact → would need a NEW `gemma=true` non-gated `RopeNeox` recipe. Left STANDALONE.
- Gemma-4 (`gemma4.cpp:224-258`) — the preamble is **heterogeneous/conditional**: q-norm always, k-norm + v-norm only when `!is_kv_shared`, RoPE is `RopeFromCache` on full layers but **`RopeNeox` on sliding layers**, and shared-KV layers rope only q (k discarded). The fixed q+k-norm+rope macro does not fit (the weight-less v-norm sits outside it, sliding layers use `RopeNeox`, shared-KV layers skip k). Only the full+non-shared subset structurally matches; a partial conditional fold is possible but needs the fused op proven byte-identical to Gemma-4's exact `RopeFromCache(prop_cache, positions)` first. Left STANDALONE this pass.
→ **B1 delivers no bit-exact fold on the current catalog; recorded honestly (fold-plan Iron Law: characterize the near-tie, do not silently swap). Follow-up: a `gemma=true` non-gated `RopeNeox` recipe (Gemma-3) + the Gemma-4 conditional handling.**

**B2. Gemma residual add+norm → `kFusedAddRmsNorm` (GEMMA 1+w, `recipes.h:38`)** *(shared-op: FusedChain)* — ✅ **DONE 2026-07-30** (branch off `18ed6f03`, NOT pushed).
Bit-exact (Tier-0 composite dispatches to the same standalone RmsNorm). **Folded only the residual-carrying norms; sandwich post-norms left standalone.** Adopted behind `FusedChainAdoptEnabled()` (existing default-ON flag, byte-exact `else` fallback), mirroring qwen3.
- Gemma-1 — `gemma.cpp` input + post_attention + final (all 3 residual-carrying gemma norms). ✅
- Gemma-2 — `gemma2.cpp` input + pre_feedforward + final; post_attention/post_feedforward (`:245,:255`) are sandwich post-norms with NO residual add → **marked NOT-fusable, left standalone** (folding them would be incorrect). ✅
- Gemma-3 — `gemma3.cpp` input + pre_feedforward + final (extension beyond the plan's enumeration; same clean pattern); post_attention/post_feedforward NOT-fusable, standalone. ✅
- Gemma-4 — **N/A**: its decoder norms are standalone plain (`gemma=false`) with the residual added SEPARATELY (PLE/YOCO structure `norm(hidden)` before `hidden = attn + r`), not the add-then-norm pattern; no fused-add-norm site to fold.

### TIER C — bit-exact but requires a NEW shared-op sibling (build once, many consumers)

**C1. GeGLU MLP method — build the GeluAndMul sibling of `UnquantizedMlpGateUpMethod`** *(shared-op: MlpGateUpMethodBase, GeGLU arm — NEW)* — ✅ **DONE 2026-07-30** (4 text Gemma sites; branch off `18ed6f03`, NOT pushed).
Same [2I,H] merged operand + `GeluAndMul(tanh)` instead of `SiluAndMul`. Bit-exact. `vt::GeluAndMul` ALREADY existed (registered CUDA+CPU op) → **NO new vt op needed**; added only `layers::UnquantizedMlpGateUpGeluMethod` (sibling of `UnquantizedMlpGateUpMethod` under `MlpGateUpMethodBase`, `linear.h`). All 4 text Gemma MLPs folded onto it (each already packs a merged `[2I,H] gate_up_proj` → no loader concat). CPU composite-golden RED-first unit case added (`test_linear_method`: the GeGLU seam == standalone `{MatmulBT; GeluAndMul}` BYTE-IDENTICAL; RED-first proven — a `SiluAndMul` reference fails the byte-check).
- Gemma-1 `gemma.cpp` · Gemma-2 `gemma2.cpp` · Gemma-3 `gemma3.cpp` · Gemma-4 `gemma4.cpp` — all folded. ✅
- gemma4_vision `gemma4_vision.cpp` (bespoke [2I,H] DevW + clamp epilogue) — **NOT in this pass** (vision tower; needs the clamp-epilogue hook), deferred to the MM-tower tier.
- **nvfp4 GeGLU arm**: the seam/method-base now hosts GeGLU, but the fused nvfp4 arm (`GateUpFusedMarlinD`) still hardcodes `SiluAndMul` — a GeGLU nvfp4 arm is a clean follow-up (Gemma is bf16-only today). Honest: "inherits nvfp4 for free" is the SEAM, realized when a GeGLU nvfp4 arm + a quantized Gemma checkpoint land.

**C2. Vision-tower QKV merge where the weight is ALREADY resident-fused** *(shared-op: merged-QKV + bias epilogue)* — ✅ **DONE 2026-07-31** (2 vision towers; branch `fold/c2-vision-qkv-bias` off `d21c442d`, NOT pushed).
Zero weight marshaling where the weight is pre-fused; bit-exact (contiguous output split), the NEW piece is the **fused merged-[3d] BIAS epilogue** (the text bf16 merged-QKV D1 has NO bias — AttnBlock VT_CHECKs `qkv_bias.Empty`). Both folds route through the NEW shared helper `models::FusedMergedQkvBiasSplit` (`src/vllm/model_executor/models/merged_qkv_fold.h`): ONE `vt::MatmulBT` over the merged `[3d,H]` weight + an OPTIONAL merged-bias `vt::Add` + a contiguous `vt::QkvSplit`. A/B unit `tests/vt/test_ops_qkv_merged_bias.cpp` (CPU, RED-first): merged `MatmulBT`+bias-`Add`+`QkvSplit` == three separate `{MatmulBT + bias}` BYTE-IDENTICAL (a wrong bias-slice fails the check) — 2/2·14 PASS.
- ✅ qwen3_vl_vision — `qwen3_vl_vision.cpp` (resident `qkv_w [3H,H]` + `[3H]` bias at `:337-338`) → ONE MatmulBT + merged **[3H] BIAS epilogue** + `QkvSplit` (removed the `SubRows`/`SubVec` 3× LinearBias). The bias-carrying vision path is the C2 NEW piece. **GATED e2e on DGX** (loads the HF checkpoint + runs the folded C++ tower): `test_qwen3vl_e2e` image→text token-exact.
- ✅ gemma4_vision — `gemma4_vision.cpp` (q/k/v were SEPARATE `[H,H]` resident weights, NO bias) → loader now packs a merged `qkv_proj [3H,H]` (mirrors the existing `gate_up` concat; a load-time marshal, NOT pre-fused like qwen3_vl) + ONE MatmulBT (no bias) + `QkvSplit`; the per-slice OUTPUT clamp epilogue stays on the split qb/kb/vb. Bit-exact-by-construction + CPU A/B + CUDA build-verified.
- whisper — `whisper_audio.cpp:256-258` (k-bias zero-padded) — OUT of this pass (audio tower, separate file scope); deferred with the E-tier tower epilogues.

**C3. nvfp4 MoE fused-w13 on the cutlass path (35B)** *(shared-op: grouped nvfp4, fused-w13)*
Extend the non-Marlin `MoeGroupedGemmNvfp4` path to the already-tuned fused-w13 operand layout so both routes share one merged grouped operand. Guarded bit-exact (`gate.scale2==up.scale2`, falls back to split — same guard vLLM uses). -1 grouped GEMM.
- 35B — `qwen3_5.cpp:4539-4544` (MIXED today) vs `:4585-4609` (fused-w13 target)

### TIER D — reordering near-ties: consistency-only, gated, do AFTER the bit-exact tiers

**D1. bf16 merged-QKV default flip → shared `MatmulBT`+`QkvSplit`** *(shared-op: merged-QKV, bf16 instance)* — ✅ **DONE 2026-07-31** (branch `fold/d1-bf16-merged-qkv` off `08eec2a8`, NOT pushed).
**Measured NEUTRAL on Qwen3-4B** (decode compute-bound) — a pure launch-count/consistency fold, not a speed win. The six bf16 consumers now default to ONE `vt::MatmulBT` over the merged `[qdim+2kdim,H]` owner + a contiguous `vt::QkvSplit` (OLMo-2/Granite/StableLM exemplar). The shared gate `Qwen3QkvMergeEnabled()` (alias `MergedQkvEnabled()`, env `VT_QWEN3_QKV_MERGE`) FLIPPED default-ON; `=0` restores the byte-identical 3-shard A/B in the same binary. **No loader concat needed** — every consumer already packs the merged `qkv_proj` owner. RoPE handling UNCHANGED (no RopeNeox→RopeFromCache swap — the second-1-ULP-flip hazard avoided). The merge is bit-exact GEMM math (wider N; per-output-row reduction identical — `tests/vt/test_ops_qkv_merge.cpp` CPU byte-identity, RED-first, 14/14); the ONLY byte effect is the downstream FA2 1-ULP on the 0.6B genuine bf16 near-tie, so the whole family batched behind ONE 0.6B near-tie golden regen.
- ✅ qwen3_dense `dense_attn_block.h` (`Qwen3QkvMergeEnabled` default flipped ON) · ✅ qwen3_coder `qwen3_moe.cpp` (inherits via the shared dense `AttnBlock`) · ✅ qwen3_dflash `qwen3_dflash.cpp` · ✅ Gemma-1/2/3/4 (`gemma.cpp`, `gemma2.cpp`, `gemma3.cpp`, `gemma4.cpp`)
- **NOT in D1 scope:** dflash_gguf `qwen3_dflash_gguf.cpp` is the KEEP-QUANT merged instance (routes via `kMatmulBTQuantGrouped`, S2), a separate follow-up — not the bf16 path this batch flips.
- Reference: OLMo-2 `olmo2.cpp:98-103` shows the target form (already merged).
- **GATED (DGX GB10 sm_121a, RelWithDebInfo + cutlass-4.5.0 + triton):** A/B unit `test_ops_qkv_merge` 2/2·14 (merged == 3-shard byte-identical, RED-first wrong-split fails); **golden regen shifted the 0.6B ANCHOR at exactly 2 divergence points** (prompt[5] tok11, prompt[10] tok10, then greedy cascades) — regenerated via `qwen3-neartie-gap.py` teacher-forcing the vLLM oracle, ALL our tokens within the near-tie band (max gap **0.125 nats** ≪ 0.5); `our_ids.npy` md5 `c849abce`→`13b59dd6`, `neartie_gap_mnats.npy` `f716ecce`→`b79989c5`. **Qwen3-4B our_ids.npy md5 UNCHANGED (0 token diffs — measured-neutral) + qwen3coder UNCHANGED (0 diffs).** Re-gate clean: `test_qwen3_paged_engine` 0.6B **16/16** (11 strict + 5 near-tie, max 0.125) + 4B **16/16** (11 strict + 5 near-tie, max 0.25) · `test_qwen3coder_paged_engine` **6/6** · `test_gemma2_forward` SACRED **48/48** (global+sliding) · `test_gemma4_paged_engine` STRICT **32/32** · shared-header canary `test_qwen27_paged_engine` **235/235** UNCHANGED (27B resident nvfp4 QKV — this bf16 path untouched).

**D2. GDN in_proj merge (qkv+z along N)** *(shared-op: merged-QKV, GDN instance)*
Near-tie, distributionally sensitive; GDN layers are the majority of the 35B hybrid so it compounds — but verify token-exact gate. `qwen3_5.cpp:2952-2958`.

**D3. OLMo-2 full-width qk-norm+rope** — ✅ **DONE (code + CPU RED-first bit-exact gate; DGX SACRED OWED) 2026-07-31** (branch `fold/d3-olmo2-fullwidth-qknorm` off `d21c442d`, NOT pushed).
OLMo-2's FULL-WIDTH q/k RMSNorm (`_apply_qk_norm`, olmo2.py:113-117,160-172: RMSNorm over the WHOLE q-dim/k-dim, all heads folded into one variance statistic) + standard NeoX RoPE preamble now folds onto the SHARED FusedChain catalog. **The shape-param generalization:** `FStep` gains a `norm_full_width` structural flag (`fused_recipe.h`, additive, default false = per-head Dh → every existing recipe byte-identical), and a new sibling recipe `kAttnQkNormRopeFullWidth` (`recipes.h`) sets it on its two RMSNorm steps — the full-width variant of Qwen3-dense's `kAttnQkNormRope`, same operand table/step wiring, only the bound norm SHAPE differs (operand 0/2 = `[T,qdim]`/`[T,kdim]` with a full-width `[qdim]`/`[kdim]` weight, vs the per-head `[.,Dh]`/`[Dh]`). Because the Tier-0 composite's `RmsNorm` reduces over the bound row's last dim, the composite realizes the full-width norm BYTE-EXACTLY with NO new primitive — it dispatches the EXACT standalone `RmsNorm(q,[qdim]) + RmsNorm(k,[kdim]) + RopeFromCache` sequence OLMo-2 hand-called before this fold. `olmo2.cpp` `Olmo2AttnBlock` (was `:110-115,131-139`) routes through `vt::FusedChain(kAttnQkNormRopeFullWidth, ...)` when RoPE runs from a cache (YaRN full-attn / default-ON bf16 cos-sin cache); the in-place `RopeNeox` fallback (`VT_QWEN3_ROPE_CACHE=0`) keeps the standalone sequence (mirrors the qwen3.cpp guard — the recipe's kRope is RopeFromCache, so routing RopeNeox through it would swap RoPE impls). Realization: **composite-only** (`fast_op=kNoFastOp`) — the per-head `kAttnQkNormRope` bespoke fast kernel (Metal) assumes a Dh reduction, so the full-width variant does not claim it; a full-width fast kernel is a clean follow-up perf step (§S4). The generalization is ADDITIVE: `kAttnQkNormRope` (per-head, fast_op intact) is untouched → qwen3/27B byte-identical.
- **GATED (CPU RED-first, this box):** `tests/vt/test_ops_fused_chain.cpp` new case `kAttnQkNormRopeFullWidth composite == full-width RmsNorm(q)+RmsNorm(k)+RopeFromCache` — byte-exact (raw f32 memcmp) across MHA + GQA + head_dim 64/128 shapes, RED-first discrimination (the full-width result is proven DISTINCT from a per-head Dh norm, so a wrong-domain realization fails). The FULL `test_ops_fused_chain` suite **10/10 · 379 assertions** GREEN under the new `FStep` ABI (the per-head `kAttnQkNormRope` regression + all 10 recipes intact → the shape-param is additive). `olmo2.cpp` + the vt runtime rebuild `-Werror` clean.
- **OWED (DGX GB10 sm_121a, RelWithDebInfo + cutlass-4.5.0 + triton):** the OLMo-2 SACRED token-exact gate `test_olmo2_paged_engine` (`olmo2_greedy_1b`, 16/16) + shared-catalog canaries `test_qwen3_paged_engine` 0.6B/4B + `test_qwen27_paged_engine` 235/235. NOT run in this pass — executed on a CPU-only box (`VLLM_CPP_CUDA=OFF`, no GPU/cutlass, DGX unreachable). Bit-exact BY CONSTRUCTION on CUDA (fast_op=kNoFastOp → the identical device-agnostic composite that the CPU byte-check pins; OLMo-2's committed default rope path is cache-based → the fold fires, composite == the pre-fold standalone sequence), but the empirical DGX SACRED run is still owed before D3 is considered fully gated (a pass ≠ the code ran).

### TIER E — MM-tower epilogues needing NEW recipes (towers furthest from parity; lowest ROI/site)

New shared-op/recipe cost, tower-local reuse. Land after A–C.
- **Bias+activation GEMM epilogue** (`kMatmulBT`+bias+`GeluErf/GeluTanh`): whisper fc1 `whisper_audio.cpp:323-335`, qwen3_vl fc1 `qwen3_vl_vision.cpp:559-561`, voxtral projector `voxtral.cpp:675-677` (+ move to ResidentWeight).
- **Rank-1 bias epilogue on every Linear**: whisper `:137-140` (~256 `vt::Add` launches/forward), qwen3_vl `:92-95`.
- **`kFusedAddLayerNorm`** (mean-subtract+bias sibling of `kFusedAddRmsNormStd`): whisper `:315+321`,`:336+346`.
- **Post-norm sandwich `kRmsNorm+Add`**: gemma4_vision `:384-385`,`:398-399`.
- **gemma4_audio** — needs a **device port first** (`:263-265` qkv, `:241-243`/`:356-364` FFN/lconv `kGluSigmoid`); host-f32 today, far from parity.
- gemma4_vision per-head norm+rope (`:355-357,360-367`) — near-tie (extend `kAttnQkNormRope` to 2-axis multidim rope + weightless v-norm).

---

## (3) "Use it consistently" end state

Every arch's MLP / QKV / MoE / attn-glue resolves through **one descriptor family**, quant arms selected by checkpoint, glue via the FusedChain catalog. Required shared-op / descriptor instances:

**S1. `MlpGateUpMethodBase` (gate-up MLP method)** — arms:
- `UnquantizedMlpGateUpMethod` **SwiGLU** *(exists)* ← OLMo-2, Granite, StableLM, dflash, deepseek_v2 dense/shared
- **GeGLU** sibling *(NEW, C1)* ← Gemma×4, gemma4_vision
- `Nvfp4W4A16MlpGateUpMethod` / `GateUpFusedMarlinD` *(exists)* — inherited free by all above on quantized checkpoints
- fused `SiluAndMul`/`GeluAndMul` epilogue *(bf16 fold)* ← deepseek_v2 epilogue

**S2. Merged-QKV descriptor** — instances:
- bf16 `MatmulBT`+`QkvSplit` *(exists, D1)* ← dense, coder, dflash, Gemma×4 (+ OLMo/Granite already there)
- bf16 **+ bias epilogue** *(NEW)* ← StableLM (`stablelm.cpp:88-93`), whisper, qwen3_vl_vision, gemma4_vision
- nvfp4 `ResidentNvfp4Qkv` *(exists — 27B exemplar)* · fp8 `ResidentQkvFp8` *(exists — 35B exemplar)*
- keep-quant merged (via `kMatmulBTQuantGrouped`) ← dflash_gguf
- **MLA A-proj** instance with contiguous-output slice *(A2)* ← deepseek_v2, kimi_k3
- **GDN in_proj** instance *(D2)* ← 35B

**S3. Grouped-MoE gate-up + SwiGLU (`kMoeGateUpSwiGLU`, promote ds4)** — arms:
- bf16 grouped *(A4, verify task #90)* ← coder, deepseek_v2, kimi
- keep-quant grouped `kMatmulBTQuantGrouped` *(A3)* ← qwen3_5_gguf, ds4
- nvfp4 fused-w13 (cutlass + Marlin unified) *(C3)* ← 35B routed + shared
- Invariant across all arms: route-weight stays in `moe_combine`.

**S4. FusedChain glue catalog** — `kAttnQkNormRope` / `kAttnQkNormRopeGate` / `kFusedAddRmsNorm(Std)` *(exist)* + `kFusedNormRope` (MLA, A5, ✅) + `kAttnQkNormRopeFullWidth` (full-width qk-norm variant, D3, ✅ composite-only); **NEW remaining:** `kFusedAddLayerNorm` (whisper), post-norm `kRmsNorm+Add` sandwich (gemma4_vision), bias+gelu epilogue + `kGluSigmoid` (towers).

At the end state, adding a new dense/MoE arch is *born fused*: pick the quant arm, bind the FusedChain recipes, no per-model GEMM+act copy. kimi_k3 is the proof — it inherits A2/A4/A5 with zero Kimi-specific work.

---

## (4) Gaps / risks

1. **Descriptor coverage wq6sodui3 MUST hit** to serve real patterns (not just the bf16 happy path): **(i)** bias epilogue on the merged-QKV descriptor — required by StableLM *and* every vision/audio tower, currently `AttnBlock` `VT_CHECK`s `qkv_bias.Empty`; **(ii)** the GeGLU arm of the MLP method — 5 Gemma-family consumers blocked without it; **(iii)** contiguous-output slicing on the MLA A-proj merge (the exact reason the un-fused form was chosen, `mla_attention.cpp:285-294`); **(iv)** all four MoE quant arms sharing one grouped descriptor with the route-weight boundary preserved. A descriptor that only does bf16 unbiased dense QKV+MLP would strand StableLM, the towers, MLA, and MoE.

2. **Near-tie golden churn (D-tier).** bf16 merged-QKV flips one 0.6B near-tie token and is *measured neutral* on 4B — batch all D1 consumers behind a single golden regen; do NOT interleave with the RopeNeox→RopeFromCache swap (compounds a second 1-ULP flip). 27B/35B stay strict 16/16 exact and must be re-gated after any shared-descriptor touch.

3. **Task-#90 unknown (A4).** Whether qwen3_coder already routes to a landed fast bf16 grouped-MoE GEMM or still hits the `qwen3_5.cpp:4256` reference loop is unverified — resolve before estimating the coder win; the same descriptor must not regress it.

4. **Gemma-2 post-norm hazard (B2).** `gemma2.cpp:245,255` are sublayer-output norms with no residual add — folding them onto add+rmsnorm is an *incorrect* fold. Recipe binding must mark them NOT-fusable.

5. **Loader dependencies.** A4 needs a stacked `[E,2I,H]` gate_up expert tensor; full 3-way MLA merge on V2-Lite needs `q_proj` stacked into `kv_a`. These are loader changes, not just launch merges.

6. **gemma4_audio is pre-port (E).** Pure host-f32 loops; the "fold" is a device port + merge in one — governed by the audio near-tie gate, not comparable ROI to the text folds. Sequence it last.

7. **Guard equality for quant folds (C3, A4-nvfp4).** Fused-w13 is only bit-exact when `gate.scale2==up.scale2`; the split fallback must stay wired (no regression on mismatch) — mirror vLLM's guard.

8. **Already-done exemplars — do not "re-fold".** qwen3_dense MLP/norm, 27B (all), 35B QKV+norm, OLMo-2/Granite QKV, qwen3_5_gguf swiglu are shipped defaults and token-exact-gated; they are the *reference operands* the folds target, not work items.

**Suggested execution order:** A1 → A2 → A3 → A4 (after #90 check) → A5 → B1/B2 → C1 → C2 → C3 → D1 (single golden regen) → D2/D3 → E. Tiers A–B are all bit-exact and free-kernel; that is where the immediate consistency + speed lives.