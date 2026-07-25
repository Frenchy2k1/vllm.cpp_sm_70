# Fusion-consistency audit — is the portable fusion framework used everywhere?

*Lifecycle: `AUDIT` (read-only static analysis). Base: `origin/main` `39943fc`.
Claim `CLAIM-FUSION-CONSISTENCY-AUDIT`. Grounds: the `KERNEL-FUSION-FRAMEWORK`
row (`kernel-matrix.md`), `specs/portable-fusion-framework.md`,
`specs/glue-fusion-2026-07-19.md`, `recipes.h` @ HEAD.*

## Question

"Are we consistent with our fusion framework and use it everywhere?" — i.e. does
every FUSED / FUSABLE op-chain in a model forward route through the declared
`vt::FusedChain(recipe)` catalog, or do sites hand-fuse / run unfused, bypassing it?

## VERDICT — MOSTLY. The framework is proven and the MVP models are fully
migrated; the post-W2 dense-model sweep adopted it INCONSISTENTLY.

- The catalog + dispatch is complete and proven end-to-end: W0-W4 all landed
  (declare-once recipe → realize-per-backend; all 4 backends — CPU/CUDA/Metal/
  Vulkan — register `kFusedChain`; whole-catalog backend-additivity test green).
- The **MVP gate model (qwen3_5 family: base/dense/moe/mtp) is FULLY migrated** —
  every bespoke fused op routes through `FusedChain(recipe)` behind
  `FusedChainAdoptEnabled()` (default ON) with a same-binary hand-call rollback.
- The **W3 additive-model pattern (`kFusedAddRmsNorm{,Std}`) was adopted by
  qwen3, qwen3_moe, deepseek_v2** — those route their add+residual+RMSNorm glue
  through the catalog.
- **DRIFT: gemma, gemma2, gemma3, glm4, phi3 hand-call the residual
  `vt::RmsNorm(...,&res)` add+RMSNorm chain and NEVER touch the catalog** — even
  though the recipe for both variants already exists. These 5 dense models
  (added during the recent-dense sweep) skipped the one-line adoption. This is
  the one real, ranked-by-value drift class.
- Deliberately-not-fused (correct, not drift): olmo2 (post-norm, residual add is
  separate from the norm — no add+RMSNorm chain), granite (residual-multiplier
  arch, norm reads residual but does not fuse the add), opt (LayerNorm, not
  RMSNorm). These never trip the detector.
- Known-deferred (recorded in spike §10 W2, not accidental): the qwen3_5 GDN
  glue (`kGdnPostConv`/`kGdnGBeta`/`kGdnConvSplit`) and `kMoeCombineGate` remain
  bespoke — no recipe declared yet; W2's remainder.

## Consistency table (each fused/fusable site → classification → fix)

| Site / op-chain | Where (file) | Classification | Fix |
|---|---|---|---|
| add+res+gemma-RMSNorm (input/post/final) | `qwen3_5.cpp:5384` etc. | CATALOG `kFusedAddRmsNorm` | — |
| add+res+std-RMSNorm | `qwen3.cpp:105,115,171`; `qwen3_moe.cpp:74,84,169`; `deepseek_v2.cpp:486,500,586` | CATALOG `kFusedAddRmsNormStd` | — |
| RMSNorm+static-fp8 quant | `qwen3_5.cpp:5133` | CATALOG `kRmsNormQuantFp8` | — |
| gated-RMSNorm+fp8 (GDN out_proj) | `qwen3_5.cpp:3041,3466,3874` | CATALOG `kRmsNormGatedQuantFp8` | — |
| silu·up→NVFP4 (MoE down_proj) | `qwen3_5.cpp:5307` | CATALOG `kSiluMulFp4Quant` | — |
| attn·sigmoid(gate)→NVFP4 (o_proj) | `qwen3_5.cpp:2025` | CATALOG `kSigmoidGateFp4Quant` | — |
| qk-norm+rope+gate (attn preamble) | `qwen3_5.cpp:3945,4094` | CATALOG `kAttnQkNormRopeGate` | — |
| qk-norm+rope (f32 path, non-gated) | `dense_attn_block.h:412` | CATALOG `kAttnQkNormRope` (f32 A/B only) | — (bf16 default path stays hand-called; see below) |
| **add+res+gemma-RMSNorm** | **`gemma2.cpp:235,249,306`; `gemma.cpp`; `gemma3.cpp`** | **HAND-FUSION DRIFT** | route `FusedChain(kFusedAddRmsNorm)` behind `FusedChainAdoptEnabled()` |
| **add+res+std-RMSNorm** | **`glm4.cpp:169,182,243`; `phi3.cpp:137,143,192`** | **HAND-FUSION DRIFT** | route `FusedChain(kFusedAddRmsNormStd)` |
| qk-norm+rope, bf16 default path | `dense_attn_block.h:441-447` | HAND-FUSION (partly deliberate) | rope-cache byte-identity + near-tie razor keeps bf16 path hand-called; low value, leave with reason |
| GDN post-conv / g·beta / conv-split | `qwen3_5.cpp:2990,2998,2999,...` | DEFERRED (no recipe; spike §10 W2) | declare GDN glue recipes if/when a 2nd backend needs them |
| MoE combine+gate | `qwen3_5.cpp:4465,4832` | DEFERRED (no recipe; spike §10 W2) | declare `kMoeCombineGate` recipe (deferred) |
| post-norm RMSNorm (no residual add) | `olmo2.cpp:206,215`; `granite.cpp:165,175,237` | DELIBERATELY-NOT-FUSED | leave — no add+RMSNorm chain exists |
| LayerNorm (+residual) | `opt.cpp:179` | DELIBERATELY-NOT-FUSED | leave — LayerNorm, not RMSNorm; no recipe class |

## Framework coverage vs vLLM's fusion set (are recipes complete?)

Portable pattern-match passes (spike §1b) that are on our Qwen gate hot path and
their catalog status:

| vLLM pass | Recipe | Status |
|---|---|---|
| `RMSNormQuantFusionPass` static-fp8+res | `kRmsNormQuantFp8` | declared + adopted |
| `RMSNormQuantFusionPass` (plain add+rms, no quant) | `kFusedAddRmsNorm{,Std}` | declared + adopted (qwen3/moe/deepseek), drift (gemma/glm4/phi3) |
| `ActivationQuantFusionPass` NVFP4 | `kSiluMulFp4Quant` | declared + adopted |
| `ActivationQuantFusionPass` static-fp8 | `kSiluMulQuantFp8` | declared (W3 proof), not wired to a model |
| `QKNormRoPEFusionPass` | `kAttnQkNormRopeGate` / `kAttnQkNormRope` | declared + adopted |
| sigmoid-gate→fp4 (Inductor glue) | `kSigmoidGateFp4Quant` | declared + adopted |
| gated-RMSNorm+fp8 (fla GDN) | `kRmsNormGatedQuantFp8` | declared + adopted |
| `ActivationQuantFusionPass` fp8-block | — | not declared (per-block quant variant; low priority) |
| `AttnQuantFusionPass` (epilogue-into-attn) | — | class B — realized per attn backend, not a standalone recipe (correct) |
| `RopeKVCacheFusionPass` | — | not declared (decode rope+kv-cache fold; future) |
| MLA passes (`MLAAttnQuant`, `MLARoPEKVCacheCat`) | — | inventory-only (DeepSeek MLA, not Qwen gate) |
| SP / AllReduce / AsyncTP passes | — | multi-GPU, HW-blocked, inventory-only |

Coverage of the on-hot-path portable set is essentially complete; the gaps are
either lower-priority quant variants, epilogue-into-attention (correctly not a
standalone chain), or HW-blocked multi-GPU passes.

## Fix plan (ordered by value; follow-on increments, NOT this pass)

1. **`FUSION-DENSE-MIGRATE` — route gemma/gemma2/gemma3/glm4/phi3 add+RMSNorm
   through the catalog** (5 models, ~3 sites each, recipe already exists). One-line
   adoption per site behind `FusedChainAdoptEnabled()` + rollback else-branch,
   mirroring qwen3.cpp. Byte-exact by construction (Tier-0 composite dispatches to
   the same `vt::RmsNorm(residual)` primitive); re-gate token-exact per model.
   Remove each stem from `scripts/fusion-consistency-allowlist.txt` as it lands.
   *Perf-neutral (structural plumbing, not a lever) — same as W0/W2.*
2. **(low value) dense_attn_block bf16 qk-norm-rope** — leave hand-called
   (rope-cache byte-identity + near-tie razor); documented, not worth forcing.
3. **(deferred) GDN glue + MoE-combine recipes** — declare only if/when a second
   backend needs them (spike §10 W2 remainder); no correctness or additivity gap
   today (single CUDA realization).

## Enforcement — CI check (IMPLEMENTED, green)

`scripts/check-fusion-consistency.py` (+ mutation test
`tests/scripts/test_check_fusion_consistency.py`, wired into the `agent-record`
CI job) — analogous to `check-env-doc.py`. It flags any model TU that hand-calls
the residual `vt::RmsNorm(...,&res)` add+RMSNorm chain yet NEVER references
`vt::FusedChain`, unless the stem is on `scripts/fusion-consistency-allowlist.txt`
(known-drift / deliberately-deferred, with a reason). The allowlist currently
carries the 5 drift models so the gate is GREEN at HEAD while:

- a NEW model that hand-fuses add+RMSNorm without the catalog must be a conscious,
  reviewable allowlist entry (it cannot land silently), and
- migrating a drift model = removing its allowlist line, which is the gate closing.

Honest limitation: the check is a coarse per-FILE floor ("has this model engaged
the catalog at all for its add+RMSNorm glue"), not a per-SITE proof — it catches
a wholly-unmigrated model (the actual drift shape here) and gates new bypasses;
it does not prove every individual site inside an already-adopted file is routed.
A per-site AST check was judged not worth the false-positive surface (the
same-binary rollback `else`-branch hand-calls legitimately coexist with adoption).

## Honest coverage statement

EXHAUSTIVE over the add+residual+RMSNorm chain across ALL 12 landed model
forwards (grep-verified per file). The bespoke quant/activation/GDN fused ops were
inventoried against `recipes.h` and the qwen3_5 call sites specifically
(the only model carrying them). The vLLM-vs-catalog coverage table is grounded in
spike §1b (the enumerated finite pass set), not a fresh re-grep of upstream.
Static analysis only — no build, no GPU, no token-exact re-run (correctness of
the existing adopted paths is already gated by the W0-W4 tests + the 235/235 +
315/315 engine gates; this audit changed no forward).
