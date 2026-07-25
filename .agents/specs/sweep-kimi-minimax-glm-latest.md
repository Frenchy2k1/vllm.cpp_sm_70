# SPIKE: frontier sweep — Kimi K3 / MiniMax (latest, ~M2.7) / GLM (latest)

**Scoping assessment + per-model mechanical-port disposition** for three
user-named frontier families. READ-ONLY design: no build, no download, no gate.
HF metadata (API + `raw/config.json`) fetched 2026-07-25; nothing downloaded.

**Base:** `origin/main` `39943fc`. **Oracle pin:** `/home/mudler/_git/vllm` @
`e24d1b24`; installed oracle = **vLLM 0.25.0** on `dgx.casa` (verified: it
constructs all target arch classes — `kimi_linear`, `minimax_m2`, `minimax_m3`,
`glm4_moe`, `glm4_moe_lite`, `deepseek_v2`/`GlmMoeDsa`, and the
`kimi_gdn_linear_attn` kernel all present in site-packages).
**Claim:** `CLAIM-SWEEP-FRONTIER-KMG`.
**GB10 budget:** 119 GiB UNIFIED (host+device shared). **dgx free disk:** 82 GiB
(the binding breadth constraint; `df -h /` 2026-07-25 = 98% used).

**User decision folded in (via coordinator, 2026-07-25):** these are to be
MECHANICALLY PORTED NOW — the additive architecture files written composing our
LANDED shared ops — NOT left as scope-and-defer spikes. An HW-blocked e2e SACRED
gate is NOT a reason to skip the port; it is a reason to SUBSTITUTE the
honesty-pass gates (config/registry resolution → loader weight-map coverage on
ONE shard → unit parity of the genuinely-new ops at real dims → clean `-Werror`
build) for the e2e token gate. A ported+unit/loader/config-gated model with an
HW-blocked e2e (measured N GiB over the 119 GiB pool) is a REAL deliverable and
is recorded as exactly that — NEVER as a claimed SACRED pass.

**Coordination — this spike does NOT edit ANY model-matrix row; all three
families' rows are already owned by other ACTIVE claims.** Per the detailed
model-matrix owner column: `MODEL-TEXT-kimi-linear-kimi-linear-for-causal-lm`,
`MODEL-TEXT-minimax-m2-mini-max-m2-for-causal-lm` and `MODEL-TEXT-minimax-m3-mini-max-m3-sparse-for-causal-lm` are owned by
`CLAIM-MLA-DEEPSEEK`; `MODEL-TEXT-glm4-moe-glm4-moe-for-causal-lm` (and the whole GLM/DSA/V4 set) by
`CLAIM-GLM-DSA-LATEST-DEEPSEEK`. This spike OWNS ONLY its own file (and the record
surfaces). Its corrections and mechanical-port plans for every row are supplied as
CROSS-REFERENCED input to those two claims — the exact precedent of
`glm-dsa-latest-deepseek.md` §0.1 vs the MLA campaign — enumerated in §5 for the
owners to reconcile. It does NOT touch any `MODEL-MM-*` row or
`multimodal-track.md` (owned by the concurrent multimodal spike, base `39943fc`).
It is recorded as a NOTE (not a `SPIKE`/`ACTIVE` matrix row) in `coordination.md`.

---

## 0. Headline — the user's named versions vs the pin (the gating fact)

| User asked | In the pin? | Newest REGISTERED arch of that family | Verdict |
|---|---|---|---|
| **Kimi K3** | **NO — no `KimiK3*` class anywhere.** | Text: `KimiLinearForCausalLM` (Kimi-Linear, hybrid KDA+MLA+MoE). Big MoE (K2) resolves via `DeepseekV3ForCausalLM` (no dedicated Kimi text entry). K2.5 = `KimiK25ForConditionalGeneration` (multimodal). | K3 is a FUTURE/unreleased arch not in vLLM 0.25.0 → cannot oracle-gate. Substitute the FITTING Kimi we CAN gate: **Kimi-Linear-48B** (§1). |
| **MiniMax ~M2.7** | No `M2.7` arch; "M2.7" is a point-release on the **M2** arch. | `MiniMaxM3SparseForCausalLM` (**M3**, newer than M2 — block-sparse "MSA" indexer). `MiniMaxM2ForCausalLM` (M2, full softmax MoE) is the one a ~M2.x checkpoint loads. | M2 = HW-blocked; M3 = HW+DEP+MM-blocked (§2). |
| **GLM latest** | Yes. | `GlmMoeDsaForCausalLM` (**GLM-5**, DSA) is newest; `Glm4MoeForCausalLM` (**GLM-4.5/4.6**) below it. We ALREADY have `Glm4ForCausalLM` ✅ + `Glm4MoeLiteForCausalLM` ✅ (GLM-4.7-Flash). | "latest GLM" beyond ours = GLM-4.5/4.6 (`Glm4Moe`, HW-blocked bf16 / fp8 marginal-fit) + GLM-5 (`GlmMoeDsa`, DEP+HW-blocked) — already scoped by the GLM/DSA spike (§3). |

**Every GLM `Glm*` registry entry at the pin** (`registry.py`): `:82-83`
ChatGLM; `:112` `GlmForCausalLM`; `:113` `Glm4ForCausalLM` ✅; `:114`
`Glm4MoeForCausalLM` (GLM-4.5/4.6); `:115` `Glm4MoeLiteForCausalLM` ✅
(GLM-4.7-Flash); `:116` `GlmMoeDsaForCausalLM`→`deepseek_v2` (GLM-5/DSA); `:218`
Glm-as-embedding; `:397-401` multimodal (glm4v/glm4_1v/glm_ocr/glmasr); `:620-622`
MTP. **What "latest GLM" adds beyond our two landed rows:** `Glm4Moe` (Qwen3-MoE
attention + DeepSeek-V2 sigmoid grouped router; ZERO new kernels) and `GlmMoeDsa`
(DeepSeek-V3.2 verbatim; DSA sparse indexer).

---

## 1. Kimi — smallest FITTING variant is `Kimi-Linear-48B-A3B` (the standout)

**Latest registered Kimi text arch:** `KimiLinearForCausalLM` (`kimi_linear.py`,
`registry.py:139`). Big MoE (Kimi-K2) has NO dedicated Kimi text class — it loads
as `DeepseekV3ForCausalLM`. Kimi-K3 absent (§0).

**Checkpoints + GB10 fit** (HF API, 2026-07-25):

| Checkpoint | Arch | Params | On disk | GB10 (119 GiB) |
|---|---|---|---|---|
| `moonshotai/Kimi-Linear-48B-A3B-Instruct` | `KimiLinearForCausalLM` | 49.1B (bf16) | **91.5 GiB** | **FITS (0.77× pool).** Disk: 91.5 > 82 GiB free → needs ~10 GiB reclaim first |
| `moonshotai/Kimi-K2-Instruct-0905` | `DeepseekV3ForCausalLM` | 1.03T (fp8-native) | 958.5 GiB | **HW-BLOCKED (8.05× over).** Also DeepSeek-V3 MLA stack |

**Real config (`Kimi-Linear-48B-A3B-Instruct/config.json`):** hidden 2304, 27
layers, vocab 163840, `num_attention_heads=num_key_value_heads=32` head_dim 72
(full-attn layers are MHA); MLA `kv_lora_rank=512`, `q_lora_rank=null`,
`qk_rope=64`, `qk_nope=128`, `v_head=128`, `mla_use_nope=true`; MoE 256 experts
top-8, `moe_intermediate_size=1024`, `first_k_dense_replace=1`,
`moe_router_activation_func=sigmoid`, `routed_scaling_factor=2.446`;
**hybrid** `linear_attn_config`: 20 **KDA** (Kimi delta-net) layers + 7 full-attn
**MLA** layers (`full_attn_layers=[4,8,12,16,20,24,27]`), KDA head_dim 128, 32
heads, `short_conv_kernel_size=4`.

**Oracle:** vLLM 0.25.0 CONSTRUCTS AND (because it fits) can SERVE it on GB10 →
**this is the ONLY one of the three frontier models that gets a REAL e2e SACRED
gate.** The KDA path is FLA-style Triton (same family as our landed GDN spec
path), so it runs on sm_121.

**Reuse-vs-new (per layer):**

| Layer / op | Reuses LANDED | Genuinely NEW |
|---|---|---|
| MLA attn (7 layers) — `KimiMLAAttention` over `MLAModules` | DeepSeek-V2 MLA block (W6), `DeepseekV2Weights` loader (W7), `TRITON_MLA` decode + `FLASH_ATTN` prefill. `q_lora_rank=null` → the DeepSeek-V2-Lite (no-q-lora) MLA branch | `mla_use_nope=true` config branch (drop the nope-rope split) — a small MLA variant, unit-gated |
| MoE (all layers, `first_k_dense_replace=1`) | `vt::MoeGroupedGemmBf16` grouped MoE + shared expert (`RunMoeBlock`); the extended sigmoid + `routed_scaling_factor` router (the shared `noaux_tc`-family extension the GLM/MLA campaigns landed) | none |
| **KDA linear-attn (20 layers)** — `KimiGatedDeltaNetAttention(GatedDeltaNetAttention)` | our landed **GDN base** machinery (chunk-scan state, causal-conv, spec split-merge) | **★ THE ONE NEW KERNEL:** Kimi Delta Attention fine-grained gate — `fused_kda_gate`, `FusedRMSNormGated`, `chunk_kda_with_fused_gate`, per-token `dt_bias`, and **separate q/k/v `conv1d`** (kernel 4) rather than the fused GDN conv. This is a GDN *variant*, not a new family — but it is real new-kernel work with its own unit gate |
| embed / rope / norms | standard NeoX rope (full-attn layers), `vt::RmsNorm` | none |

**Mechanical-port increment (additive files + honesty-pass gate):**
1. `kimi_linear_registry.cpp` (one `REGISTER_VLLM_MODEL`) + config parse of
   `linear_attn_config`/`mla_use_nope` → **GATE-A: config/registry resolution
   from the real `config.json`, no GPU** (kda/full-attn layer schedule resolves;
   MLA dims resolve).
2. `kimi_linear.{h,cpp}` + `kimi_linear_weights.cpp` composing the MLA block +
   `RunMoeBlock` + a new `KdaLinearAttnBlock` → **GATE-B: loader weight-map
   coverage on ONE downloaded shard, zero unmapped / zero missing, correct shapes
   for KDA q/k/v conv + dt_bias + MLA fused_qkv_a_proj.**
3. The KDA gated-delta kernel (`vt::KimiDeltaGate` / extend the GDN op) →
   **GATE-C: UNIT PARITY of the KDA gate + FusedRMSNormGated at real dims
   (32 heads, head_dim 128, conv 4) vs a CPU reference or a dumped vLLM tensor**;
   `mla_use_nope` MLA branch unit-gated the same way.
4. Clean `-Werror` build → **GATE-D.**
5. **Because it FITS: run the REAL e2e SACRED token gate** vs the vLLM 0.25.0
   oracle after ~10 GiB dgx-disk reclaim (near-tie methodology; a 49B MoE is above
   the small-dense near-tie regime → STRICT expected). This is the payoff that
   makes Kimi-Linear worth doing FIRST despite the one new kernel.

**Disposition: IMPLEMENTABLE-ADDITIVE-e2e** (fits + oracle-serves), gated by a
real SACRED token gate; ONE new kernel (KDA gate) carries its own unit gate.
Precondition: reclaim ~10 GiB dgx disk (own build trees, user-approved per
[[extensibility-first-additive-hw-models]]).

---

## 2. MiniMax — M2 is the loadable arch; both M2 and M3 are HW-blocked

**Latest registered MiniMax text arch:** `MiniMaxM3SparseForCausalLM`
(`vllm.models.minimax_m3`, `registry.py:154`) — NEWER than
`MiniMaxM2ForCausalLM` (`:153`). Older lightning-attention MiniMax
(`MiniMaxText01`, `MiniMaxM1`) are `_PREVIOUSLY_SUPPORTED` since 0.23.0
(`registry.py:721-724`). **There is no "M2.7" arch** — a ~M2.x checkpoint loads
as `MiniMaxM2ForCausalLM`.

**Checkpoints + GB10 fit** (HF API, 2026-07-25):

| Checkpoint | Arch | Params | On disk | GB10 (119 GiB) |
|---|---|---|---|---|
| `MiniMaxAI/MiniMax-M2` | `MiniMaxM2ForCausalLM` | 228.7B (**fp8-native**, `float8_e4m3fn` block 128×128) | **214.3 GiB** | **HW-BLOCKED (1.80× over).** No smaller/lite variant exists |
| `MiniMaxAI/MiniMax-M3` | `MiniMaxM3SparseForCausalLM` | 427B (bf16) | 795.5 GiB | **HW-BLOCKED (6.68× over) + DEP-BLOCKED + MULTIMODAL** (`image-text-to-text`; block-sparse MSA needs `fmha_sm100`, GB10 is sm_121) |
| `MiniMaxAI/MiniMax-M1-40k` / `MiniMax-Text-01` | `_PREVIOUSLY_SUPPORTED` (0.23.0) | 456B | — | not oracle-constructible at 0.25.0 |

**★ MATRIX CORRECTION (fact, handed to the owning claim).** The
`MODEL-TEXT-minimax-m2` row (owned by `CLAIM-MLA-DEEPSEEK`) says "~230B /
**~428 GiB bf16**, ~4× over". The checkpoint is **fp8-native, 214.3 GiB, 1.80×
over** — there is no bf16 MiniMax-M2 checkpoint. The verdict (HW-blocked) stands,
but at ~1.8× not ~4×. This spike does NOT edit that row; the correction is handed
to its owner (§5).

**Real config (`MiniMax-M2/config.json`):** hidden 3072, 62 layers, vocab 200064,
GQA `48 heads / 8 kv` head_dim 128, `rotary_dim=64` (partial rope 0.5),
`use_qk_norm=true`, `sliding_window=null` (full attention), MoE 256 experts top-8,
`scoring_func=sigmoid`, `use_routing_bias=true`.

**Reuse-vs-new — the CHEAPEST port of the three, ZERO new kernels:**

| Layer / op | Reuses LANDED | Genuinely NEW |
|---|---|---|
| Attention (full softmax GQA) | shared dense GQA + paged attn; **partial rope** (landed via GLM/Phi-3); **QK-norm** (landed via Qwen3/OLMo) | none. **NOTE: M2 has NO lightning/linear attention** — MiniMax dropped it after M1. So the "MiniMax lightning-attention" new-kernel concern does NOT apply to M2 (it applies only to the `_PREVIOUSLY_SUPPORTED` M1/Text-01) |
| MoE 256/top-8 | `vt::MoeGroupedGemmBf16` + shared expert + the extended sigmoid + `e_score_correction_bias`(=`use_routing_bias`) grouped router | `GateLinear` wiring detail only (no new kernel) |
| fp8 weights | the landed FP8 path (NVFP4/FP8 quant) — but this ckpt is **block-128×128 fp8** (DeepSeek-style block scales); confirm our block-fp8 GEMM covers this exact layout at loader time | block-fp8 SCALE layout parse (loader-only, if not already covered) |

**Mechanical-port increment (honesty-pass — e2e is HW-blocked):**
1. `minimax_m2_registry.cpp` + config parse → **GATE-A: config/registry
   resolution from the real `config.json`** (GQA/partial-rope/QK-norm/MoE +
   `use_routing_bias` resolve).
2. `minimax_m2.{h,cpp}` + `minimax_m2_weights.cpp` composing dense-GQA attn (with
   partial rope + QK-norm) + `RunMoeBlock` + the sigmoid/routing-bias router →
   **GATE-B: loader weight-map coverage on ONE downloaded fp8 shard** (zero
   unmapped/missing; block-128×128 scale tensors mapped; correct shapes).
3. **GATE-C: UNIT PARITY at real M2 dims** of the router (256 experts, top-8,
   sigmoid + routing-bias) and the partial-rope+QK-norm attention block vs CPU
   reference / dumped vLLM tensor. (No genuinely-new compute kernel → no
   new-kernel unit gate beyond these composed ops.)
4. Clean `-Werror` build → **GATE-D.**
5. **e2e token gate: HW-BLOCKED — 214.3 GiB is 1.80× the 119 GiB pool.** Recorded
   as "architecture ported + unit/loader/config-gated; e2e HW-blocked (95.3 GiB
   over pool)". NEVER a claimed SACRED pass.

**Disposition: HONESTY-PASS-BLOCKED (HW).** Cheapest mechanical port (0 new
kernels), highest pure code reuse, but no runnable e2e gate on GB10. **M3:**
DEP+HW+MM-blocked → registry/config-resolution only (its block-sparse MSA indexer
is a new kernel AND needs `fmha_sm100`; it is also a multimodal row owned
elsewhere — not ported here).

---

## 3. GLM (latest) — already scoped; `Glm4Moe` fp8 is the fitting-variant flag

The GLM family is owned end-to-end by `CLAIM-GLM-DSA-LATEST-DEEPSEEK`
([`glm-dsa-latest-deepseek.md`](glm-dsa-latest-deepseek.md)). We ALREADY landed
`Glm4ForCausalLM` ✅ (GLM-4-9B-0414) and `Glm4MoeLiteForCausalLM` ✅
(GLM-4.7-Flash). "Latest GLM" beyond those:

| Checkpoint | Arch | Params | On disk | GB10 (119 GiB) |
|---|---|---|---|---|
| `zai-org/GLM-4.5-Air` | `Glm4MoeForCausalLM` | 110.5B | 205.8 GiB bf16 | HW-BLOCKED (1.73× over) |
| `zai-org/GLM-4.5-Air-FP8` | `Glm4MoeForCausalLM` | 110.5B | **104.8 GiB fp8** | **★ HW-MARGINAL — FITS (0.88× pool).** Would be our FIRST GLM-MoE e2e if fp8-checkpoint loading for this arch lands |
| `zai-org/GLM-5` | `GlmMoeDsaForCausalLM` | 753.9B | 1404 GiB bf16 | HW-BLOCKED (11.8× over) + DEP-BLOCKED (DSA sm12x sparse, non-functional flashinfer XQA path — GLM spec §0.2) |

**Reuse-vs-new for `Glm4Moe` (mechanical port, ZERO new kernels):** Qwen3-MoE
attention (GQA + optional QK-norm + partial NeoX rope — all landed) + a
near-verbatim `DeepseekV2MoE` sigmoid grouped router (landed) + `RunMoeBlock`.
The GLM spec §0.4 already establishes this is a pure composition of landed ops.

**★ FITTING-VARIANT FLAG (jumps the queue if fp8-loading lands):**
`GLM-4.5-Air-FP8` at 104.8 GiB FITS GB10 (marginal). Since `Glm4Moe` needs ZERO
new compute kernels, the ONLY thing between us and our **first GLM-MoE e2e SACRED
gate** is fp8-checkpoint loading for this arch (a quantization-matrix row, not a
model-matrix row). This is the highest-value fitting-variant opportunity in the
whole GLM family and is supplied as input to the GLM claim (§5).

**Disposition (deferred to the GLM claim):** `Glm4Moe` = mechanical port,
HONESTY-PASS at bf16 (205.8 GiB) TODAY, e2e-gateable via the fp8 variant if
fp8-loading lands. `GlmMoeDsa`/GLM-5 = DEP-BLOCKED (DSA) + HW-BLOCKED. Not
re-owned here.

---

## 4. Ranking

**By pure mechanical portability (least-new-op first):**
`Glm4Moe` (0 new kernels) ≈ `MiniMax-M2` (0 new kernels) < `Kimi-Linear` (1 new
kernel: the KDA gate).

**Recommended sweep order (gateability-weighted — the user's ambition is a REAL
token-exact gate + speed, so a model that gets a real e2e gate outranks a cheaper
honesty-pass):**

1. **Kimi-Linear-48B — DO FIRST.** The ONLY one of the three that gets a REAL
   e2e SACRED gate on GB10 (fits 91.5 < 119 GiB, oracle serves). High reuse
   (MLA + sigmoid-router + bf16 grouped-MoE + GDN base). Pays for its one new
   KDA kernel with a real correctness gate. Precondition: ~10 GiB disk reclaim.
   **New-kernel flag: KDA gated-delta (fused_kda_gate + FusedRMSNormGated +
   per-q/k/v conv) — the one place this "mechanical" sweep becomes real
   new-kernel work + its own unit gate.**
2. **MiniMax-M2 — cheapest port, honesty-pass only.** 0 new kernels (GQA +
   partial rope + QK-norm + sigmoid/routing-bias MoE + block-fp8 loading), but
   HW-blocked (214.3 GiB, 1.80× over) → no e2e gate. Do after Kimi because it
   cannot earn a real gate.
3. **GLM `Glm4Moe` (GLM-4.5-Air) — coordinate with the GLM claim.** 0 new
   kernels, honesty-pass at bf16. **fp8 variant (104.8 GiB) is the jump-the-queue
   opportunity**: it would become our first GLM-MoE e2e the moment fp8-checkpoint
   loading for the arch lands.

**Registry/config-resolution-only (hard-blocked, no port):** Kimi-K2
(958.5 GiB / DeepSeek-V3 arch), Kimi-K3 (absent from pin), MiniMax-M3
(795.5 GiB + sm100 sparse + multimodal), GLM-5/`GlmMoeDsa` (1404 GiB + DSA
DEP-blocked).

---

## 5. Cross-referenced corrections handed to the owning claims

Mirroring `glm-dsa-latest-deepseek.md` §0.1. This spike does not edit these rows;
the owners reconcile.

- **To `CLAIM-MLA-DEEPSEEK` (owns `MODEL-TEXT-kimi-linear-kimi-linear-for-causal-lm`):** Kimi-Linear-48B
  **FITS GB10 (91.5 GiB, 0.77× pool)** and is the only frontier model here with a
  real e2e gate. The row currently reads "MLA half unlocked; full model not
  gated". Upgrade: it is IMPLEMENTABLE-ADDITIVE-e2e with ONE new kernel (KDA
  gate); the MLA half is landed, the delta is the KDA gated-delta-net + the
  hybrid layer schedule + `mla_use_nope`. Its claim's own HW-verdict prose
  ("MiniMax-M2 ~428 GiB") should also adopt the corrected 214.3 GiB fp8 figure.
- **To `CLAIM-GLM-DSA-LATEST-DEEPSEEK` (owns `MODEL-TEXT-glm4-moe-glm4-moe-for-causal-lm`):** the
  `Glm4Moe` mechanical port is 0-new-kernel; the **GLM-4.5-Air-FP8 (104.8 GiB)
  fitting variant** would yield the first GLM-MoE e2e SACRED gate if fp8-checkpoint
  loading for the arch lands. Recommend flagging this fp8-loading dependency as
  the unblock for that row (already consistent with GLM spec §0.5).

---

## 6. Structured contract

**Scope.** Design-only scoping + per-model mechanical-port disposition for the
three named frontier families, grounded in the pinned registry, HF metadata, and
the dgx oracle/cache. OWNS ONLY `.agents/specs/sweep-kimi-minimax-glm-latest.md`
(and the record surfaces). Recorded as a coordination NOTE.

**Out of scope (with reason):** implementation of anything (spike). Editing ANY
model-matrix row — Kimi-Linear / MiniMax-M2 / MiniMax-M3 (owned by
`CLAIM-MLA-DEEPSEEK`) and Glm4Moe (owned by `CLAIM-GLM-DSA-LATEST-DEEPSEEK`) are
cross-referenced instead. MiniMax-M3, Kimi-K2/K2.5, GLM-5, GLM multimodal —
HW/DEP/MM-blocked, registry/config-resolution only. Any `MODEL-MM-*` row /
`multimodal-track.md` (concurrent spike).

**Upstream chain.** `vllm/model_executor/models/registry.py:113-116,139,153-156`;
`kimi_linear.py:63,103,179,287,381` + `layers/mamba/gdn/kimi_gdn_linear_attn.py`;
`transformers_utils/configs/kimi_linear.py:37-144`; `minimax_m2.py:72,139,432`;
`transformers_utils/configs/minimax_m3.py`; `vllm/models/minimax_m3/nvidia/`
(indexer_msa, sparse_attention_msa, mtp); `glm4_moe.py`; `deepseek_v2.py`
(`GlmMoeDsa`). Real configs fetched 2026-07-25 (`raw/config.json`, no download).

**Port map / gates / disposition:** §1 (Kimi, e2e), §2 (MiniMax-M2, honesty-pass),
§3 (GLM, deferred to GLM claim). The honesty-pass gate set (A config/registry,
B loader weight-map on one shard, C unit parity of new ops at real dims, D clean
`-Werror` build) SUBSTITUTES for the e2e SACRED token gate wherever HW-blocked.

**Tests to port** (inventory; nothing ported by this spike):
`tests/models/registry.py` `_HfExamplesInfo` for KimiLinear/MiniMaxM2/Glm4Moe
(config/registry resolution, no GPU); `tests/models/test_initialization.py`
(construct-only); Kimi KDA-gate + MiniMax router unit cases at real dims (built by
us — no upstream text-correctness fixtures exist for these); MiniMax-M3 /
GlmMoeDsa sparse tests SKIPPED (DEP-blocked).
