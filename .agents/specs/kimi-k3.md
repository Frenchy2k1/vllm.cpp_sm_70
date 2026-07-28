# Kimi K3 — W0 SCOPE spike (records-only, DERIVE-AND-SHIP)

**Claim:** `CLAIM-KIMI-K3-SCOPE`. **Row:**
`MODEL-MM-kimi-k3-kimi-k3-for-conditional-generation` (`KimiK3ForConditionalGeneration`,
NEW row, INVENTORIED→SPIKE here).
**State:** SPIKE — CPU-only, records-only. NO build, NO GPU, NO download.

**Base:** current `main` HEAD `df18ca918160a8a8741382a07c7ee8a7323efd00`.
**Pinned oracle:** `${VLLM_SOURCE}` = `/home/mudler/_git/vllm` @
`5559679229bc961848b121ccdeaa8fa5d79bec98` (vLLM 0.26.0.dev0).

**Signal (honest, up front):** Kimi K3 is **DERIVE-AND-SHIP**, NOT hardware-verified.
The published checkpoint is **2.8T params / ~1.56 TB on disk (MXFP4)** and does **NOT fit** the
DGX Spark's 119 GiB unified pool — off by **~12×** (§3). There is **no on-box oracle golden**,
exactly like the beyond-vLLM CUDA-arch bricks. BUT unlike a pure CUDA brick, K3's core
primitives are gateable against a REAL oracle at small scale: **K3's text backbone IS
`KimiLinearForCausalLM`** (config-declared), which is registered + implemented in the pinned
vLLM, and the fitting **Kimi-Linear-48B-A3B (~89–91 GiB)** proxy runs on GB10 (§4). So the
derive-and-ship signal is: (a) REAL unit/small-shape gates of KDA + MLA + MoE against the
Kimi-Linear-48B oracle, plus (b) build-verify + structural review of the K3 scale-up
(896-expert MoE, MXFP4, AttnRes, MoonViT-V2 vision) vs vLLM's `kimi_linear.py`/`kimi_k25.py` —
DERIVED, testing-welcome, no board ran the 2.8T model.

---

## 0. Scope (headline verdict)

**Kimi K3 is NOT in the pinned vLLM oracle.** `ModelRegistry` (`registry.py`) at
`555967922` registers `KimiLinearForCausalLM` (`:140`), `KimiVLForConditionalGeneration`
(`:459`), `KimiK25ForConditionalGeneration` (`:460`), `MoonshotKimiaForCausalLM` (`:461`) — but
**no `KimiK3ForConditionalGeneration` and no `kimi_k3` config/model file**. K3 shipped
2026-07-27, after the pin. **Scoped from the HF `config.json` + the closest registered arch
(`KimiLinearForCausalLM`, its literal text backbone).**

**Arch (from `moonshotai/Kimi-K3` config.json — DERIVED from an HF fetch, not byte-verified):**
a multimodal MoE — `architectures: ["KimiK3ForConditionalGeneration"]`, `model_type: "kimi_k3"`,
wrapping a `text_config` whose own `architectures: ["KimiLinearForCausalLM"]` + a
`vision_config` (MoonViT-V2). The text backbone is a **massively scaled-up Kimi-Linear hybrid**:
- **H=7168, L=93, 96 heads, vocab 163840.**
- **Hybrid attention:** 69 **KDA** (Kimi Delta Attention — gated-delta-net linear attention)
  layers + 24 **full-attention MLA** layers ("Gated Mixture-of-Latents"), selected per-layer by
  `linear_attn_config.kda_layers` / `full_attn_layers`.
- **MLA geometry:** `kv_lora_rank=512`, `q_lora_rank=1536`, `qk_nope_head_dim=128`,
  `qk_rope_head_dim=64` — the **standard DeepSeek-V2/V3 MLA geometry we have already landed**.
- **MoE:** **896 routed experts, top-16, 2 shared experts**, `moe_intermediate_size=3072`,
  DeepSeek-style gate with `e_score_correction_bias` (`noaux_tc`-family).
- **KDA config:** `head_dim=128`, `num_heads=96`, `short_conv_kernel_size=4`,
  `use_full_rank_gate=true`, `gate_lower_bound=-5.0`.
- **Quant:** `mxfp4-pack-quantized` via **compressed-tensors** (num_bits 4, group_size 32,
  symmetric, float/e8m0 scales); the tech report adds **MXFP8 activations** (QAT). **MXFP4 is a
  NEW quant path for us** — we have NVFP4 (group 16) but not MXFP4 (group 32, e8m0).
- **Vision:** MoonViT-V2 (~401M, patch 14, 27 layers) — a NEW tower for us (closest reuse: our
  Qwen3-VL tower).
- **Report-only / UNCONFIRMED novelties (flag honestly):** **AttnRes ("Attention Residuals")** —
  described in the K3 tech report as a hyper-connection-like residual, but it is **NOT present in
  the fetched config.json and NOT in the pinned `kimi_linear.py`** (whose decoder uses plain fused
  add+RMSNorm residuals, `kimi_linear.py:355-378`). Treat AttnRes as an UNVERIFIED K3 delta until
  the real `modeling_kimi_k3.py` is read. Likewise MXFP8-activations and the exact 401M ViT size
  are report-derived.

**Reuse verdict:** **HEAVY reuse.** Every core text primitive already exists in our tree — GDN
linear attention (KDA's parent family), the exact DeepSeek MLA geometry, the DeepSeek-style
MoE router + grouped GEMM, and the Kimi-K2 tokenizer/tool parser. The K3 text backbone is,
structurally, our **Qwen3.6-35B GDN-hybrid-MoE twin** (`Qwen3_5MoeForConditionalGeneration`,
already DONE text-path) with KDA instead of Qwen-GDN and DeepSeek-MLA full layers. NET-NEW = the
**KDA decay/output kernel deltas** (already scoped on the Kimi-Linear row), **MXFP4**, **AttnRes
(if real)**, and the **MoonViT-V2 tower**.

**Effort verdict:** the *primitives* are mostly landed; the campaign is (1) the KDA kernel
delta (a known, separately-scoped kernel brick), (2) MXFP4 compressed-tensors, (3) the
scaled-up loader/config (896 experts, 93 hybrid layers), (4) the MoonViT-V2 vision tower, and
(5) AttnRes if the real modeling file confirms it. The decisive early brick is **W1: gate the
shared KDA+MLA+MoE primitives on the FITTING Kimi-Linear-48B proxy** — that is a REAL oracle
gate; the 2.8T K3 itself is derive-and-ship on top of it.

---

## 1. Upstream chain — Kimi K3 primitive map (`file:line`)

K3 itself has no vLLM file (post-pin). Anchors are its literal backbone class
`KimiLinearForCausalLM` and the Kimi-K2.5 vision wrapper, both under
`/home/mudler/_git/vllm/vllm/`.

### 1.1 Registry + config (K3 ABSENT; backbone PRESENT)
- `model_executor/models/registry.py:140` `KimiLinearForCausalLM` → `kimi_linear`.
- `registry.py:459-461` `KimiVLForConditionalGeneration` / `KimiK25ForConditionalGeneration` /
  `MoonshotKimiaForCausalLM`. **No `KimiK3*`.**
- Kimi-K2/K2.5's *dense* text backbone is `DeepseekV2ForCausalLM` with a `DeepseekV3Config`
  (`configs/kimi_k25.py:10,83,101`; `kimi_k25.py:375`) — that is the **old, DeepSeek-lineage**
  Kimi. **K3 is different**: its `text_config.architectures` is `["KimiLinearForCausalLM"]`, the
  **hybrid KDA/MLA/MoE lineage**, not `DeepseekV3ForCausalLM`. (This CORRECTS the 2026-07-25
  frontier-sweep note that said "big Kimi MoE loads as `DeepseekV3ForCausalLM`" — true for K2,
  NOT for the newly-released K3.)
- Config fields resolved off `configs/kimi_linear.py`: `is_mla` (`:119`), `is_moe` (`:130`),
  `is_kda_layer(layer_idx)` (`:144`, `(layer_idx+1) in kda_layers`), `is_linear_attn` (`:134`).
  K3's `linear_attn_config` carries the `kda_layers`/`full_attn_layers` split (69/24 of 93).

### 1.2 Text hybrid decoder — `models/kimi_linear.py`
- `KimiDecoderLayer` (`:288`): per-layer dispatch —
  `if config.is_kda_layer(layer_idx): self.self_attn = KimiGatedDeltaNetAttention` (`:304`)
  `else: KimiMLAAttention` (`:311`). MoE vs dense MLP by `first_k_dense_replace`/`moe_layer_freq`
  (`:328-347`). **Residual = plain fused add+RMSNorm** (`input_layernorm`/`post_attention_layernorm`,
  `:352-378`) — **NO AttnRes here** (K3's AttnRes, if real, is a delta).
- `KimiMLAAttention` (`:180`): builds `MLAModules` (`:250`) + `MultiHeadLatentAttentionWrapper`
  (`:264`); NoPE variant (`kimi_linear.py:213` asserts `use_nope`). Self-documented "Main
  reference: DeepseekV2 vllm Implementation". Geometry = K3's `qk_nope=128`/`qk_rope=64`/
  `kv_lora=512`/`q_lora=1536`.
- `KimiMoE` (`:104`): `FusedMoE` (`:153`) with `shared_experts` (`:141-148`), `top_k` (`:156`),
  `gate` + `e_score_correction_bias` (`:138`). DeepSeek-style `noaux_tc` router.
- `KimiLinearForCausalLM` (`:557`): the CausalLM; KDA state via `MambaState*Calculator.kda_*`
  (`:605-633`), `IsHybrid`/`HasInnerState`/`MixtureOfExperts` interfaces (`:52`).

### 1.3 KDA — `models/layers/mamba/gdn/kimi_gdn_linear_attn.py` (444 LoC)
- `KimiGatedDeltaNetAttention(GatedDeltaNetAttention)` (`:85`) — **subclasses the GDN attention**;
  reuses GDN state/conv/chunked-delta machinery, overrides the gate + output.
- Three separate short convs `q_conv1d`/`k_conv1d`/`v_conv1d` (`:171-190`, `conv_size=4`), each
  a `ColumnParallelLinear` reshaped to conv weights (`:196-198`).
- Per-head decay: `A_log` (`:200`, `[1,1,H,1]` f32) + `dt_bias` (`:157`); the KDA-specific
  **per-channel `[H,D]` decay** comes from a low-rank `f_a_proj`/`f_b_proj` bottleneck (matrix
  note; upstream kernel `kda.py:1019,1126`), which plain GDN does NOT have.
- Gate/output kernels: `fused_kda_gate`, `chunk_kda_with_fused_gate`, `fused_recurrent_kda`
  (`:19-22`); output `FusedRMSNormGated(head_dim, activation="sigmoid")` (`:219`) — the gated-
  linear-attention output norm plain GDN lacks. `use_full_rank_gate`/`gate_lower_bound=-5.0`.

### 1.4 Vision — MoonViT-V2 (scoped from the K2.5 tower `kimi_k25_vit.py`)
- Closest upstream reference: `models/kimi_k25_vit.py` (`MoonViT3dPretrainedModel`,
  `KimiK25MultiModalProjector`, `vision_tower_forward`) + `kimi_k25.py:290`
  `KimiK25ForConditionalGeneration` (language_model re-entry `:370-375`, media placeholder
  merge). K3's `vision_config` (patch 14, 27 layers, MoonViT-V2) is a V2 evolution; the real
  tower forward is in the post-pin `modeling_kimi_k3.py` (NOT available) — scope from K2.5 +
  our Qwen3-VL tower and RE-verify when the pin advances.

### Kimi-K3-vs-Kimi-Linear deltas (what CHANGED vs the pinned backbone)
| Axis | Kimi-Linear (pinned `kimi_linear.py`) | Kimi K3 |
|---|---|---|
| Scale | 48B-A3B (small H/L, ~256 experts) | **H=7168, L=93, 896 experts, top-16, 2 shared** |
| Attention mix | KDA + MLA hybrid | **same hybrid**, 69 KDA + 24 MLA (config `kda_layers`) |
| MLA geometry | NoPE MLA (`use_nope`) | `qk_nope=128`/`qk_rope=64`/`kv_lora=512`/`q_lora=1536` |
| Residual | plain fused add+RMSNorm | **AttnRes (report-only, UNCONFIRMED)** |
| Quant | bf16 / fp8 | **MXFP4 weights (group 32, e8m0) + MXFP8 acts (QAT)** |
| Multimodal | text-only | **KimiK3ForConditionalGeneration + MoonViT-V2 vision** |

---

## 2. Our baseline — reuse-vs-new map (our `file:line`)

Anchors under `/home/mudler/_git/vllm.cpp/`.

### REUSE (landed primitives K3's text path builds on — the bulk of the model)
- **GDN linear attention (KDA's parent family)** — `src/vt/cuda/cuda_gdn.cu`,
  `src/vt/cuda/gdn_packed_decode_triton.h`, `src/vt/cuda/gdn_prefill_conv.h`,
  `src/vt/cuda/gdn_packed_reg_tile.h`, `src/vllm/v1/attention/backends/gdn_attn.cpp` +
  `include/vllm/v1/attention/backends/gdn_attn.h`, AOT kernels
  `src/vt/cuda/triton_aot_vendored/sm_121a/gdn_*`. Mature + gated in production (Qwen3.6-27B/35B
  GDN-hybrid). **REUSE:** the conv-state/cache layout, `GDNAttentionMetadata`, conv update,
  chunked-delta recurrence, WY solve. **DOES NOT COVER:** KDA's per-channel `[H,D]` low-rank
  decay (`f_a_proj`/`f_b_proj`), the sigmoid-gated output norm, and the 3 separate q/k/v short
  convs — the NET-NEW KDA kernel delta (already scoped on the Kimi-Linear row).
- **DeepSeek MLA (exact K3 geometry)** — `src/vllm/model_executor/models/deepseek_v2.cpp`,
  `include/vllm/model_executor/models/mla_attention.h`,
  `src/vllm/model_executor/layers/attention/mla_attention.cpp`, `src/vt/cuda/cuda_mla_attn.cu`,
  `src/vt/cuda/cuda_mla_prefill.cu`. The MLA campaign (W1-W6) landed the block + load-time
  `kv_b_proj→W_UK/W_UV` absorption, split-KV decode, chunked-context prefill, the decoupled
  `is_neox_style=False` RoPE + YaRN cache, and BOTH `q_lora` branches — gated at DeepSeek-V3's
  real 512/1536/128/64 dims (the SAME as K3). **REUSE ~wholesale** for the 24 full-attn layers.
- **DeepSeek-style MoE** — `src/vt/cuda/cuda_moe.cu`, `src/vt/cuda/cuda_moe_marlin.cu`,
  `src/vt/cuda/marlin/libtorch_stable/moe/`, `src/vllm/model_executor/models/qwen3_moe.cpp`,
  `src/vllm/model_executor/models/qwen3_5_moe.cpp`, the DeepSeek-V2 MoE block in
  `deepseek_v2.cpp`, `noaux_tc` router unit-gated `tests/vt/test_ops_moe_router_grouped.cpp`.
  **REUSE:** the grouped GEMM + shared-expert + `e_score_correction_bias` router; SCALE to 896
  experts / top-16 / 2 shared. The GLM-4.7-Flash row already exercises `noaux_tc` sigmoid e2e.
- **GDN-hybrid MoE MODEL (structural twin)** — `src/vllm/model_executor/models/qwen3_5_moe.cpp`
  (`Qwen3_5MoeForConditionalGeneration`, text-path DONE 315/315). Same shape as K3's text
  backbone: linear-attn + full-attn hybrid + MoE + `ForConditionalGeneration` mm wrapper.
  **REUSE the whole hybrid-decoder + loader skeleton**; swap Qwen-GDN→KDA, add MLA full-attn
  layers, scale MoE.
- **Kimi tokenizer + tool parser** — `src/vllm/parser/kimi_k2.cpp` +
  `include/vllm/parser/kimi_k2.h`, `src/vllm/entrypoints/openai/tool_parsers/kimi_k2.cpp` +
  header. Kimi-K2 tokenizer/tool-call already landed; K3 shares the Kimi tokenizer family.
- **CUDA-graph decode / paged engine / hybrid KV** — the runner already runs GDN-hybrid + MoE
  (Qwen3.6). K3 text path needs no new engine scaffolding.

### NEW (genuinely net-new)
1. **KDA kernel delta** — per-channel `[H,D]` low-rank decay (`f_a_proj`/`f_b_proj`), the
   sigmoid-gated output norm (`FusedRMSNormGated`), the 3 separate q/k/v short convs, and the
   `fused_kda_gate`/`chunk_kda_with_fused_gate`/`fused_recurrent_kda` variants. Adjacent to our
   GDN; **already scoped as a kernel campaign on the Kimi-Linear row** — do NOT duplicate.
2. **MXFP4 (compressed-tensors, group 32, e8m0)** — we have NVFP4 (`src/vt/cuda/cuda_matmul_nvfp4.cu`,
   `.../compressed_tensors/nvfp4_emulation.cpp`) at group 16 but **not MXFP4** (group 32, e8m0/
   UE8M0 block scales, `mxfp4-pack-quantized`). The `compressed_tensors/` dir has no MXFP4 scheme.
   NEW quant path (the DeepSeek-V4 MegaMoE MXFP4-experts scope is adjacent — coordinate).
3. **AttnRes (Attention Residuals)** — UNCONFIRMED; if the real modeling file has it, a residual-
   topology delta (hyper-connection-like, cf. DeepSeek-V4 MHC). Build a reference first.
4. **MoonViT-V2 vision tower** — a new ViT + projector + media-merge; closest reuse is our
   Qwen3-VL tower and the K2.5 `kimi_k25_vit.py` reference.
5. **896-expert / 93-hybrid-layer loader + config** — the scaled `KimiLinearForCausalLM` config
   parse + weight map (fused q/kv-a projections, per-layer kda/full split, w13/w2 for 896
   experts, MXFP4 packing).

---

## 3. HW-fit — Kimi K3 does NOT fit ONE GB10 (119 GiB unified)

**The math (2.8T params, MXFP4):**
- MXFP4 ≈ 4 bits/weight + e8m0 block scale (8 bits / 32 elems = 0.25 bits/weight) ≈ 4.25
  bits/param ≈ **0.53 B/param**. `2.8e12 × 0.53 ≈ 1.49 TB` of weights; with bf16 embeddings/
  norms/router/vision the published repo is **~1.56 TB** (as reported on HF).
- GB10 unified pool = 119 GiB ≈ 127.8 GB ≈ 1452 GiB… wait: 119 GiB. `1.56 TB ≈ 1452 GiB`.
  **1452 GiB / 119 GiB ≈ 12.2×** over the pool (weights alone ~1.35 TiB ≈ **11.4×**). **DOES
  NOT FIT — off by ~12×**, before any KV/activation. (104B *activated* params are irrelevant to
  storage — all 2.8T weights must be resident.)
- To hold it you need **~13 GB10 Sparks** (13×119 = 1547 GiB) or a big multi-GPU box:
  8×H200-141GB (1128 GB) is still short of the ~1.56 TB checkpoint; ~14–16×H100-80GB or a
  16×H200 / NVL72-class node is the realistic single-node home.

**Smaller variant that WOULD fit?** No small **K3** checkpoint is published (2.8T only). BUT the
**same text arch family** has **Kimi-Linear-48B-A3B ≈ 89–91 GiB** (bf16), which **FITS GB10**
(HW-MARGINAL per the existing Kimi-Linear row — pushes dgx root fs >97%, needs ~10 GiB reclaim).
That is the **gateable proxy** for K3's shared KDA+MLA+MoE primitives (§4). K3-specific deltas
(896-expert scale, MXFP4, AttnRes, MoonViT-V2) stay derive-and-ship on top.

| Vehicle | Size | Fits 119 GiB? | Gateable? | Note |
|---|---|---|---|---|
| `moonshotai/Kimi-K3` (MXFP4, 2.8T) | ~1.56 TB | **NO (~12×)** | no on-box oracle | derive-and-ship target; also NOT in the pinned oracle |
| **`moonshotai/Kimi-Linear-48B-A3B`** (bf16) | ~89–91 GiB | **YES** (HW-marginal) | **YES** — registered `KimiLinearForCausalLM` in the pin | **the proxy for the shared primitives** |

---

## 4. Derive-and-ship plan — the honest signal + how a HW-rich user verifies

K3 won't fit → **no on-box oracle golden**, exactly like the beyond-vLLM CUDA-arch bricks. The
signal is two-tier:

- **(a) REAL primitive gates (on the FITTING proxy).** Gate KDA, DeepSeek-MLA (already done at
  V3 dims), and the `noaux_tc` MoE router against **Kimi-Linear-48B-A3B** running on the pinned
  vLLM oracle on GB10 — a genuine RUN-verified greedy golden (strict or the ratified near-tie
  band per `[[near-tie-distributional-gate]]`). This proves the *shared* K3 primitives, not the
  2.8T assembly. Serialize via the GPU flock (`[[sharing-a-gpu-with-flock]]`), stop
  `local-ai-worker`, keep `gpu_memory_utilization` LOW (unified pool, `[[gb10-unified-memory-oom-reboots-box]]`),
  reclaim ~10 GiB disk first.
- **(b) DERIVED build-verify + structural review (for the K3 scale-up).** For the parts the proxy
  can't cover — 896-expert MoE, MXFP4, AttnRes, MoonViT-V2 — the signal is: our engine BUILDS the
  K3 config/loader/forward, unit-tests each new op vs an independent reference, and a structural
  1:1 review vs vLLM's `kimi_linear.py`/`kimi_k25.py` (and, when the pin advances,
  `modeling_kimi_k3.py`). **DERIVED, testing-welcome, no board ran the 2.8T model.**

**How a user WITH the hardware verifies K3 end-to-end:** on a ~13-Spark cluster / multi-Spark /
16×H200-class box, load the 1.56 TB MXFP4 checkpoint and run a greedy golden. **Caveat:** the
pinned oracle (`555967922`, 0.26.0.dev0) does **NOT** implement `KimiK3ForConditionalGeneration`
— so even a HW-rich user cannot oracle-gate K3 against THIS pin; they must (i) bump the vLLM pin
to a release that adds `kimi_k3`, then token-exact vs it, or (ii) fall back to our-engine
self-consistency + the proxy-gated primitives. The Kimi-Linear-48B proxy IS gateable against the
current pin today.

---

## 5. Bring-up W-plan (row-sized bricks; each cites the vLLM file it ports FROM)

- **W0 (this):** scope — arch map, reuse-vs-new, HW-fit, derive-and-ship plan, records. DONE.
- **W1 — PROXY primitive gate.** Gate KDA + MLA + `noaux_tc` MoE on **Kimi-Linear-48B-A3B** vs
  the pinned oracle on GB10 (§4a). Ports FROM `kimi_linear.py` + `kimi_gdn_linear_attn.py`. The
  decisive REAL-signal brick. (Shares the KDA kernel work already scoped on the Kimi-Linear row.)
- **W2 — K3 registry stub + config parse (no forward).** `KimiK3ForConditionalGeneration` +
  `kimi_k3` config: `text_config`(=KimiLinear: H 7168, L 93, `kda_layers`/`full_attn_layers`,
  896/16/2 MoE, `kv_lora`/`q_lora`/`qk_nope`/`qk_rope`, `linear_attn_config`) + `vision_config`.
  REFUSE-by-name until the forward lands. Ports FROM `configs/kimi_linear.py` + `configs/kimi_k25.py`.
- **W3 — MXFP4 compressed-tensors quant path.** `mxfp4-pack-quantized` (num_bits 4, group 32,
  e8m0 scales) load + compute-in-quant, adjacent to our NVFP4. Ports FROM the compressed-tensors
  MXFP4 scheme + `cuda_matmul_nvfp4.cu` structure. Unit-gate the dequant/GEMM vs a reference.
- **W4 — KDA kernel delta.** Per-channel `[H,D]` low-rank decay (`f_a_proj`/`f_b_proj`), sigmoid-
  gated output norm, 3 q/k/v short convs. Ports FROM `kimi_gdn_linear_attn.py` + `kda.py`
  (`:1019,1126`). REUSE our GDN state/conv/WY machinery. Unit-gate vs an independent recurrence.
- **W5 — scaled hybrid loader + text forward.** Compose the 93-layer KDA/MLA hybrid + 896-expert
  MoE loader (fused q/kv-a, per-layer split, MXFP4 packing). Ports FROM `kimi_linear.py:557-646`
  (`load_weights`). REUSE the Qwen3.6-35B GDN-hybrid-MoE forward skeleton (`qwen3_5_moe.cpp`).
- **W6 — AttnRes (IF the real modeling file confirms it).** Residual-topology delta; build a
  double-precision reference first (cf. DeepSeek-V4 MHC). RED-first. Ports FROM `modeling_kimi_k3.py`
  (pending pin advance). If absent, DROP this brick.
- **W7 — MoonViT-V2 vision tower + projector + merge.** Ports FROM `kimi_k25_vit.py` +
  `kimi_k25.py:290-464`; REUSE our Qwen3-VL tower patterns. Text path stays byte-identical
  (gate mm off).
- **W8 — DERIVE-AND-SHIP build-verify.** Clean CUDA build + the proxy-gated primitives + the K3
  unit gates green; structural 1:1 review recorded. Row stays SPIKE (no on-box e2e). Ship as
  DERIVED/testing-welcome, mirroring the beyond-vLLM CUDA bricks.

---

## 6. Dependencies / risks / decisions
- **DERIVE-AND-SHIP** — K3 (2.8T/1.56 TB) can NOT be e2e-gated on GB10; the honest signal is the
  proxy gate (Kimi-Linear-48B) + build-verify. (DECISION.)
- **Pinned oracle lacks `kimi_k3`** — even HW-rich users can't oracle-gate K3 against `555967922`;
  needs a pin advance. The proxy IS gateable at the current pin. (RISK, recorded.)
- **Config-derived, not source-derived** — the arch map is from an HF `config.json` FETCH + the
  backbone class; **AttnRes, MXFP8-activations, and the exact ViT size are report-only/UNCONFIRMED**
  until `modeling_kimi_k3.py` is read. Never invent K3 details beyond this. (RISK.)
- **Shared with `CLAIM-MLA-DEEPSEEK` (Kimi-Linear row) + the DeepSeek-V4 MXFP4 scope** — the KDA
  kernel and MXFP4 must not be implemented twice; coordinate before W3/W4. (DEP.)
- This CORRECTS the 2026-07-25 frontier-sweep "Kimi K3 ABSENT / loads as DeepseekV3ForCausalLM"
  note: K3 is now a REAL released arch whose backbone is `KimiLinearForCausalLM`, not DeepseekV3.
