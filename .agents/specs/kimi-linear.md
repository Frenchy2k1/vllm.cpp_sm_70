# Kimi-Linear-48B-A3B — W0 SPIKE (full spike contract, e2e-gateable)

**Claim:** `CLAIM-KIMI-LINEAR-W0`. **Row:**
`MODEL-TEXT-kimi-linear-kimi-linear-for-causal-lm` (`KimiLinearForCausalLM`,
`SPIKE`→`READY` here). **State:** SPIKE spec — CPU-only, records-only. NO build,
NO GPU, NO download this brick (two GPU jobs are queued ahead; the W0 GPU golden
capture is a SEPARATE later step whose ready-to-run recipe is §8).

**Base:** worktree HEAD `10dd23eeee3c14a14a3e7f3fd5ee77ace21d8d77` (off
`origin/main`). **Pinned oracle:** `${VLLM_SOURCE}` = `/home/mudler/_git/vllm` @
`5559679229bc961848b121ccdeaa8fa5d79bec98` (vLLM 0.26.0.dev0).

**Signal (honest, up front):** unlike its 2.8T sibling Kimi-K3 (DERIVE-AND-SHIP,
~12× over one GB10 — see [kimi-k3.md](kimi-k3.md)), **Kimi-Linear-48B-A3B FITS
one GB10** (48.9B bf16 ≈ 91.5 GiB on disk, 0.77× the 119 GiB unified pool) AND
the pinned vLLM oracle **constructs and can serve it** — so this is the ONE Kimi
text model that earns a **REAL e2e SACRED token gate**. Its three heaviest
primitives are already landed and gated in our tree: **DeepSeek-MLA** (campaign
W1-W6), the **DeepSeek-style sigmoid/`noaux_tc` grouped MoE**, and the **GDN
linear-attention family** (Qwen3.6 27B/35B, production). The genuinely-new work
is narrow: the **KDA (Kimi Delta Attention) device kernel** — whose four
net-new-vs-GDN numerics are ALREADY ported as portable host references +
unit-gated (task #173, `CLAIM-KDA-KERNEL`, `kimi_kda.{h,cpp}`, `test_kimi_kda`
14/14·36) — plus the hybrid **layer-schedule wiring**, the **NoPE MLA** config
branch, and the **loader name-map**.

---

## 0. Scope (headline verdict)

**Kimi-Linear IS in the pinned oracle.** `ModelRegistry` registers
`KimiLinearForCausalLM` (`registry.py:140` → `kimi_linear`,
`KimiLinearForCausalLM`), backed by `models/kimi_linear.py`,
`layers/mamba/gdn/kimi_gdn_linear_attn.py`, `layers/mla.py`
(`MLAModules`/`MultiHeadLatentAttentionWrapper`), `layers/fused_moe`, and
`transformers_utils/configs/kimi_linear.py`. The 0.25.0 oracle on dgx was
verified (frontier sweep 2026-07-25) to construct the class and its
`kimi_gdn_linear_attn` kernel path.

**Arch (from `moonshotai/Kimi-Linear-48B-A3B-Instruct/config.json`, fetched
2026-08-05 — authoritative, supersedes the 2026-07-25 sweep figures):** a
**hybrid KDA/MLA + DeepSeek-style-MoE decoder**, dense-embedding, 27 layers:
- **H=2304, L=27, 32 attention heads, vocab 163840, `intermediate_size=9216`,
  `rms_norm_eps=1e-5`, bf16, `tie_word_embeddings=false`.**
- **Hybrid attention, per-layer:** **20 KDA** (Kimi-Delta gated linear attention)
  layers + **7 full-attention MLA** layers. Schedule (config `linear_attn_config`,
  **1-indexed**): `full_attn_layers=[4,8,12,16,20,24,27]`;
  `kda_layers=[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23,25,26]`. Selector
  `is_kda_layer(layer_idx) := (layer_idx+1) in kda_layers`
  (`configs/kimi_linear.py:144-148`); everything not-KDA is MLA
  (`kimi_linear.py:304-326`).
- **MLA geometry (NoPE):** `kv_lora_rank=512`, **`q_lora_rank=null`** (the
  no-q-lora branch), `qk_nope_head_dim=128`, `qk_rope_head_dim=64`,
  `v_head_dim=128`, **`mla_use_nope=true`** — `KimiMLAAttention` hard-asserts
  `use_nope is True` and `q_lora_rank is None` (`kimi_linear.py:214-215`) and
  passes **`rotary_emb=None`** into `MLAModules` (`:253`). So the full-attn layers
  are **position-encoding-free MLA**: the nope/rope dim split still exists
  structurally (kv_a projects to `kv_lora_rank+qk_rope_head_dim=576`) but **no
  RoPE rotation is applied anywhere in the model**.
- **MoE (DeepSeek-style):** **256 routed experts, top-8, 1 shared expert**,
  `moe_intermediate_size=1024`, `first_k_dense_replace=1`, `moe_layer_freq=1`,
  `moe_router_activation_func=sigmoid`, `routed_scaling_factor=2.446`,
  `moe_renormalize=true`, `use_grouped_topk=true` but **`num_expert_group=1`/
  `topk_group=1`** (grouping is trivial — same shape as DeepSeek-V2-Lite, but
  WITH `e_score_correction_bias` + sigmoid + routed-scaling, which V2-Lite lacks).
  Router gate carries `e_score_correction_bias` (`kimi_linear.py:138,166`,
  the `noaux_tc` family). Layer 0 (0-indexed) is a **dense** `KimiMLP`
  (`intermediate_size=9216`) because `first_k_dense_replace=1`; layers 1..26 are
  MoE.
- **KDA config (`linear_attn_config`):** `head_dim=128`, `num_heads=32`,
  `short_conv_kernel_size=4` → projection size `128*32=4096`; conv over q/k/v.
- **MTP:** **`num_nextn_predict_layers=0`** — the shipped Instruct checkpoint has
  **NO MTP draft head**. The arch supports it (loader skips spec layers via
  `get_spec_layer_idx_from_weight_name`, `kimi_linear.py:486-488`; state shape
  carries `num_spec`, `:616-627`), but it is not part of this checkpoint's e2e
  path. See §5 "MTP (deferred)".

**Corrections folded (mirroring the kimi-k3.md precedent):** the 2026-07-25
frontier sweep said Kimi-Linear full-attn layers are "MHA (head_dim 72)". **They
are MLA, not MHA** — `kimi_linear.py:311` builds `KimiMLAAttention` for
non-KDA layers. The top-level `head_dim=72` (=2304/32) is not used by the MLA
layers (they use the MLA dims above). On-disk size is **~91.5 GiB / 48.9B**, not
the sweep's "91.5 GiB" for a differently-stated param count — the FIT verdict
(0.77× pool) stands.

**Reuse verdict: HEAVY.** MLA (7 layers), the sigmoid/`noaux_tc` grouped MoE (all
layers), and the GDN state/conv/chunk machinery (KDA's parent) are landed. The
model is structurally our **Qwen3.6-35B GDN-hybrid-MoE twin**
(`Qwen3_5MoeForConditionalGeneration`, text-path DONE 315/315) with **KDA in
place of Qwen-GDN** and **MLA (not GQA) as the full-attention layer**. NET-NEW =
the KDA device kernel (host-ref-oracled already), the NoPE-MLA branch, the
hybrid layer schedule + het-KV wiring, and the loader name-map.

---

## 1. Upstream chain — Kimi-Linear primitive map (`file:line` @ 555967922)

All under `/home/mudler/_git/vllm/vllm/`.

### 1.1 Registry + config
- `model_executor/models/registry.py:140` — `KimiLinearForCausalLM` →
  `("kimi_linear","KimiLinearForCausalLM")`.
- `transformers_utils/configs/kimi_linear.py` — `KimiLinearConfig`; fields
  `:34-108`; `is_mla` `:118-127`, `is_moe` `:129-131`, `is_linear_attn`
  `:133-142`, `is_kda_layer(layer_idx)` `:144-148` (`(layer_idx+1) in kda_layers`).

### 1.2 Text hybrid decoder — `models/kimi_linear.py`
- `KimiDecoderLayer` (`:288`): per-layer dispatch — `is_kda_layer` ⇒
  `KimiGatedDeltaNetAttention` (`:304-309`), else `KimiMLAAttention` (`:311-326`).
  MoE-vs-dense by `first_k_dense_replace`/`moe_layer_freq` (`:328-347`). Residual =
  **plain fused add + RMSNorm** (`input_layernorm`/`post_attention_layernorm`,
  `:348-378`) — no AttnRes, no hyper-connections.
- `KimiMLAAttention` (`:180-285`): the NoPE, no-q-lora MLA. Builds `MLAModules`
  (`:250-263`, `rotary_emb=None`, `q_a_layernorm=None`, `q_b_proj=None`,
  `is_sparse=False`) + `MultiHeadLatentAttentionWrapper` (`:264-277`).
  Self-documented "Main reference: DeepseekV2 vllm Implementation".
- `KimiMoE` (`:104-177`): `ReplicatedLinear` gate (`:130-136`) +
  `e_score_correction_bias` (`:138`), optional shared expert `KimiMLP`
  (`:140-149`), `FusedMoE` (`:153-168`) with `scoring_func=sigmoid`,
  `use_grouped_topk`, `num_expert_group`/`topk_group`, `routed_scaling_factor`.
- `KimiMLP` (`:64-101`): `MergedColumnParallelLinear` gate_up + `RowParallelLinear`
  down + `SiluAndMul`.
- `KimiLinearModel` (`:381-554`): embed → layers → final RMSNorm; `load_weights`
  (`:460-554`) — `.gate_up_proj` stacking (`:461-465`) +
  `fused_moe_make_expert_params_mapping` (`w1`/`w2`/`w3`, `:469-475`) + KDA
  bias/kv-scale handling (`:534-552`).
- `KimiLinearForCausalLM` (`:557-646`): `HasInnerState`/`SupportsPP`/
  `MixtureOfExperts`/`IsHybrid`; KDA state via `MambaState*Calculator.kda_*`
  (`:600-633`).

### 1.3 KDA — `models/layers/mamba/gdn/kimi_gdn_linear_attn.py` (445 LoC)
- `KimiGatedDeltaNetAttention(GatedDeltaNetAttention)` (`:85`, subclasses the GDN
  base `gdn/base.py:22`) — REUSES GDN state/conv/`GDNAttentionMetadata`/chunked
  recurrence, OVERRIDES the gate + output. Modules `:120-226`: q/k/v `q_proj`/
  `k_proj`/`v_proj`; low-rank decay `f_a_proj`(H→head_dim)+`f_b_proj`(head_dim→H·D)
  (`:142-156`); `dt_bias` (`:157-161`); `b_proj`(H→num_heads, β) (`:163-169`);
  three separate `q/k/v_conv1d` (`:171-198`, `conv_size=4`, fp32); per-head
  `A_log` (`:200-203`); gate low-rank `g_a_proj`+`g_b_proj` (`:205-218`); output
  `o_norm=FusedRMSNormGated(head_dim,"sigmoid")` (`:219`) + `o_proj` (`:220-226`).
- forward (`:233-268`): decode path uses `fused_kda_gate` then
  `fused_recurrent_kda`; prefill path uses `chunk_kda_with_fused_gate`
  (`:394-441`). Output = `o_norm(core_attn_out, g2)` (`:266`).

### 1.4 KDA kernels — `third_party/flash_linear_attention/ops/kda.py` (1647 LoC)
The delta-rule recurrence is SHARED with GDN; the KDA-specific pieces are the gate
and the gated output:
- **REUSED from GDN** (imports `:19-25`): `chunk_gated_delta_rule_fwd_h`
  (chunk hidden-state scan), **`fused_recurrent_gated_delta_rule_fwd_kernel`**
  (THE GDN decode kernel — `fused_recurrent_kda` at `:32-146` is a thin wrapper
  that only substitutes the KDA gate `g`), `solve_tril`, `l2norm_fwd`,
  `chunk_local_cumsum`.
- **KDA-NEW device kernels:** `FusedRMSNormGated` (`:436-518`) sigmoid-gated RMS
  output norm; `kda_gate_fwd_kernel`/`fused_kda_gate` (`:1541-1646`) the
  per-channel decay gate `g = -exp(A_log[h])·softplus(g1+dt_bias)`;
  `kda_gate_cumsum_fwd_kernel`/`fused_kda_gate_chunk_cumsum` (`:1182-1303`) the
  prefill chunk-local cumsum·RCP_LN2 variant; and the chunked-prefill trio
  `chunk_kda_scaled_dot_kkt_fwd` (`:717-815`, per-channel-gated K·Kᵀ),
  `recompute_w_u_fwd` (`:960-1017`), `chunk_gla_fwd_o_gk` (`:1126-1181`, GLA-style
  output with per-channel gk decay) driven by `chunk_kda`/`chunk_kda_with_fused_gate`
  (`:1458-1540`).

### 1.5 MLA / MoE / state shared machinery
- `layers/mla.py` — `MLAModules`, `MultiHeadLatentAttentionWrapper`.
- `layers/fused_moe/*` — `FusedMoE`, `fused_moe_make_expert_params_mapping`.
- `layers/mamba/mamba_utils.py` — `MambaStateDtypeCalculator.kda_state_dtype`
  (`:130-137`, returns `(cache_dtype||model_dtype, float32)`) and
  `MambaStateShapeCalculator.kda_state_shape` (`:270-294`): conv_state
  `(divide(proj+2·proj_k, tp), conv_k-1)` = `(12288, 3)` here, recurrent_state
  `(num_heads/tp, head_dim, head_dim)` = `(32,128,128)`.
- `v1/attention/backends/gdn_attn.py` — `GDNAttentionMetadata` (KDA reuses it).

---

## 2. Our baseline — reuse-vs-new map (our `file:line` @ HEAD `10dd23ee`)

Under `/home/mudler/_git/vllm.cpp/`.

### REUSE (landed + gated — the bulk of the model)
- **DeepSeek MLA (exact Kimi geometry, minus RoPE)** —
  `include/vllm/model_executor/models/mla_attention.h` +
  `src/vllm/model_executor/layers/attention/mla_attention.cpp` (campaign W6:
  projections with both q-lora branches, `kv_b_proj→W_UK/W_UV` load-time
  absorption, mscale²-scaled softmax, prefill-MHA/decode-MQA dispatch),
  `src/vllm/model_executor/models/deepseek_v2.cpp`, `src/vt/cuda/cuda_mla_attn.cu`
  (`TRITON_MLA` split-KV decode, W4), `src/vt/cuda/cuda_mla_prefill.cu`
  (`FLASH_ATTN` prefill + chunked context, W5). Kimi's `q_lora_rank=null` +
  `kv_lora=512`/`qk_nope=128`/`qk_rope=64`/`v=128` are the DeepSeek-V2-Lite MLA
  dims we already gate e2e (DeepSeek-V2-Lite 8/8; GLM-4.7-Flash 8/8). **Delta:
  the NoPE branch** — pass `rotary_emb=None`/skip the decoupled RoPE (a
  SIMPLIFICATION; §NEW-2).
- **DeepSeek-style sigmoid/`noaux_tc` grouped MoE + shared expert** —
  `src/vllm/model_executor/models/deepseek_v2.cpp` `RunMoeBlock`
  (`:348` `vt::MoeRouterTopKArgs`, `:358-359` `e_score_correction_bias` bias
  tensor, `:463` shared-expert), `src/vt/cuda/cuda_moe.cu` grouped GEMM +
  `src/vt/cuda/cuda_moe_marlin.cu`; the `noaux_tc` router is unit-gated at real
  V3 dims in `tests/vt/test_ops_moe_router_grouped.cpp` (sigmoid, top-8,
  routed-scaling, WITH bias). Kimi's `num_expert_group=1`/`topk_group=1` is the
  trivial-group case; sigmoid + `e_score_correction_bias` + `routed_scaling=2.446`
  is EXACTLY the landed path (GLM-4.7-Flash already gates it e2e). Scale to
  256 experts / top-8 / 1 shared.
- **GDN family (KDA's parent — state/conv/recurrence)** —
  `src/vt/cuda/cuda_gdn.cu`, `src/vt/cuda/gdn_packed_decode_triton.h`,
  `src/vt/cuda/gdn_prefill_conv.h`, `src/vt/cuda/gdn_packed_reg_tile.h`,
  `src/vllm/v1/attention/backends/gdn_attn.cpp` +
  `include/vllm/v1/attention/backends/gdn_attn.h`, AOT cubins
  `src/vt/cuda/triton_aot_vendored/sm_121a/gdn_*`. REUSE: conv-state/cache layout,
  `GDNAttentionMetadata`, conv update, chunked-delta hidden-state scan, WY solve,
  and the fused-recurrent decode kernel (which `fused_recurrent_kda` wraps 1:1).
- **KDA host references (task #173 — the four net-new-vs-GDN numerics, LANDED)** —
  `include/vllm/model_executor/models/kimi_kda.h` +
  `src/vllm/model_executor/models/kimi_kda.cpp` (`vllm::kimi_kda`:
  `KdaLowRankDecay`, `KdaDecayGate`, `KdaDecayGateChunkCumsum`, `FusedRMSNormGated`,
  `KdaShortConv`, `L2NormRows`), unit-gated `tests/vllm/models/test_kimi_kda.cpp`
  (14/14·36, clean CPU `-Wall -Werror -Wextra`) vs hand-derived literals +
  double-precision references. **These ARE the oracle** for the KDA device
  kernel (§NEW-1). Ported 1:1 with `file:line` on both sides
  (`kimi_kda.h:40-61`).
- **GDN-hybrid-MoE MODEL skeleton (structural twin)** —
  `src/vllm/model_executor/models/qwen3_5_moe.cpp` (`Qwen3_5MoeLoadedModel`,
  text-path DONE 315/315): the per-layer hybrid dispatch + loader + het-KV +
  born-on-the-runner decode. REUSE the whole skeleton; swap Qwen-GDN→KDA and
  standard-GQA-full-attn→MLA.
- **Kimi tokenizer + tool parser** — `src/vllm/parser/kimi_k2.cpp` +
  `src/vllm/entrypoints/openai/tool_parsers/kimi_k2.cpp` (Kimi-K2 family; shares
  the tokenizer). Config-descent + text-backbone name-map already exist in
  `src/vllm/model_executor/models/kimi_k3_weights.cpp`
  (`EnumerateKimiK3TextBackboneTensors`, grounded 1:1 in `kimi_linear.py:104-378,
  460-554` + `kimi_gdn_linear_attn.py:102-226`) — the Kimi-Linear loader is that
  enumeration at 48B dims, un-nested from the K3 multimodal wrapper.
- **Three MUST-route seams (mandatory — plan routing, do NOT hand-roll):**
  `include/vt/fused_recipe.h` + `include/vt/recipes.h` (`vt::FusedChain` — norm/
  quant/act/combine glue), `include/vt/merged_gemm.h`
  (`layers::MlpGateUpMethodBase`/`vt::MergedGemmGroup` — the fused gate_up + expert
  GEMMs), and the shared decode runner (`src/vllm/v1/worker/gpu/runner.cpp`
  `ModelRegistry::Forward` + `dense_attn` `AttnBlock` + on-GPU sampling). KDA
  layers route through the GDN backend (born-on-runner, as Qwen3.6 does); MLA
  layers route through the `mla_attention` block (born-on-runner, as DeepSeek
  does); MoE routes through `RunMoeBlock`/`MergedGemmGroup`; norms/quant fold via
  `FusedChain`. The born-on-runner CI guard
  (`scripts/check-runner-routing-consistency.py`) must stay green — the model's
  decode routes through `ModelRegistry::Forward`, never a hand-rolled loop.

### NEW (genuinely net-new)
1. **KDA device kernel.** Compose the REUSED GDN chunk-hidden-scan + fused-recurrent
   decode kernels with the KDA-NEW gate/KKT/GLA-output (`kda.py` §1.4). **The
   decode path is nearly free** — `fused_recurrent_kda` = GDN's fused-recurrent
   kernel fed a precomputed per-channel gate; the only new decode kernel is
   `fused_kda_gate`. **The prefill path** needs the KDA-specific
   `chunk_kda_scaled_dot_kkt` + `recompute_w_u` + `chunk_gla_fwd_o_gk` +
   `fused_kda_gate_chunk_cumsum`. **Port assessment:** the four net-new numerics
   (per-channel low-rank decay, the decay gate + chunk-cumsum, `FusedRMSNormGated`,
   the 3 q/k/v short convs + q/k L2-norm) are ALREADY host-ref'd and unit-gated
   (task #173); the device kernel is a structure-port of those refs + a reuse of
   the GDN chunk/recurrent kernels, gated vs the `kimi_kda` host oracle at real
   dims (32 heads, head_dim 128, conv 4). RED-first, `compute-sanitizer` clean,
   additive to `cuda_gdn.cu` (GDN gate byte-identical by construction).
2. **NoPE MLA branch.** `mla_attention` currently always builds the decoupled
   DeepSeek RoPE cache; Kimi passes `rotary_emb=None` (`kimi_linear.py:253`) and
   asserts `use_nope` (`:214`). Add a config flag that keeps the nope/rope dim
   split but **skips the RoPE rotation** (positions unused in attention). Small,
   unit-gated vs our own MLA with RoPE disabled + the vLLM `KimiMLAAttention`
   forward.
3. **Hybrid layer schedule + het-KV wiring.** Per-layer `is_kda_layer` dispatch
   (20 KDA + 7 MLA), `first_k_dense_replace=1` dense-layer-0, and the
   **heterogeneous per-layer KV**: KDA layers hold GDN (conv+recurrent) state,
   MLA layers hold the 576-wide compressed latent (`num_kv_heads=1`). Both cache
   types are landed individually (Qwen3.6 GDN-hybrid; DeepSeek MLA latent);
   combining GDN-state + MLA-latent in one hybrid-KV-group model is the new wiring
   (reuse the hybrid-KV-group machinery, add MLA as a group type). Plain fused
   add+RMSNorm residual (no AttnRes).
4. **Loader name-map + config parse.** `ParseKimiLinearParams` (nested
   `linear_attn_config`, `is_kda_layer` schedule, `mla_use_nope`, MoE scalars) +
   the 27-layer KDA/MLA + 256-expert weight map (fused gate_up for dense/shared
   MLP; expert `w1/w2/w3`; KDA `q/k/v/f_a/f_b/g_a/g_b_proj`, `b_proj`, `q/k/v_conv1d`,
   `A_log`, `dt_bias`, `o_norm`, `o_proj`; MLA `kv_a_proj_with_mqa`/`q_proj`/
   `kv_a_layernorm`/`kv_b_proj`/`o_proj`; router gate + `e_score_correction_bias`).
   Reuse `EnumerateKimiK3TextBackboneTensors` (K3 scaffold) at 48B dims.

---

## 3. Quant / HW-fit — Kimi-Linear-48B FITS one GB10 (119 GiB unified)

| Vehicle | Size | Fits 119 GiB? | Gateable? | Note |
|---|---|---|---|---|
| `moonshotai/Kimi-Linear-48B-A3B-Instruct` (bf16) | **~91.5 GiB / 48.9B** | **YES (0.77× pool)** | **YES** — registered in the pin, oracle serves | the e2e vehicle; disk needs ~10 GiB reclaim first (dgx root ~98% used) |
| fp8 / NVFP4 quant | ~46–52 GiB | yes (comfortable) | if a community quant appears | none published at spec time; bf16 is the gate |

**OOM discipline (the GB10 unified pool is host+device shared —
[[gb10-unified-memory-oom-reboots-box]]):** weights ~91.5 GiB leave ~27 GiB for
KV + activations + the CUDA context. vLLM `gpu_memory_utilization` reserves HOST
RAM on this box, so **keep it LOW** — set the util ceiling from
`(weights + a few GiB) / 119` and START conservative (~0.55–0.65, NOT 0.85 which
hard-rebooted the box 3×). `sudo drop_caches` before any wall-clock; park
`local-ai-worker` (`docker stop`); `flock $HOME/gpu.lock`; single-load
steady-state (never reload per rep — GB10 reload swings make e2e wall-clock
useless, use ncu/steady-state); named tmux + done-marker (ssh drops eat ~⅓ runs).
Reclaim ~10 GiB dgx disk (own build trees) before the 91.5 GiB download.

**No MXFP4 here.** MXFP4 (group-32/e8m0) is Kimi-K3's path (§kimi-k3.md W3), NOT
Kimi-Linear's — the 48B checkpoint is plain bf16.

---

## 4. Correctness gates

**W0 oracle-RUNS rule (the gate that lets this row be READY):** prove the pinned
vLLM oracle (`555967922`) BUILDS + RUNS a greedy golden for Kimi-Linear-48B-A3B on
GB10 — not just that `AutoConfig` constructs. The sweep confirmed construction and
the model FITS ⇒ it SERVES. §8 is the ready-to-run capture recipe.

**Strict vs near-tie selection:** a 48.9B MoE is well ABOVE the small-dense
bf16-nondeterminism regime, so **STRICT token-exact is expected** (like 27B/35B).
At capture time, confirm the oracle's greedy is K-run deterministic; if the bf16
MoE shows run-to-run drift, fall back to the ratified **distributional near-tie
gate** (ours ∈ the oracle's K-run set) per [[near-tie-distributional-gate]].

**Token-exact battery (mirror 27B/35B):** greedy goldens over the standard prompt
set on GB10 vs the oracle, plus the per-primitive unit gates:
- KDA host refs — `test_kimi_kda` 14/14·36 (LANDED).
- KDA device kernel — vs the `kimi_kda` host oracle at real dims (W3),
  `compute-sanitizer` clean, run-to-run bit-reproducible.
- NoPE MLA — vs our MLA with RoPE disabled + vLLM `KimiMLAAttention` forward, at
  Kimi dims (32 heads, qk_nope128/qk_rope64/v128/kv_lora512, q_lora=null).
- Sigmoid/`noaux_tc` router — `test_ops_moe_router_grouped` at 256e/top-8/
  scaling 2.446 with bias (the machinery is landed; add the Kimi shape).
- Loader — weight-map coverage on ONE downloaded shard: zero unmapped, zero
  missing, correct shapes for KDA conv/f_a/f_b/dt_bias/A_log + MLA fused proj +
  256-expert w1/w2/w3.

**Tests to port from vLLM `tests/`:** `tests/models/registry.py` `_HfExamplesInfo`
for `KimiLinearForCausalLM` (config/registry resolution, no GPU);
`tests/models/test_initialization.py` (construct-only). No upstream
text-correctness fixture exists for Kimi-Linear (the FLA `kda` ops ship no vLLM
text-golden) → the KDA/NoPE-MLA/router unit cases are ours (built at real dims),
and the e2e oracle golden is the correctness truth.

---

## 5. W0–W7 work breakdown (row-sized bricks; each cites what it ports FROM)

| W | Brick | Ports FROM | Blocked-by | Venue |
|---|---|---|---|---|
| **W0** | THIS spike — arch inventory, reuse/new map, HW-fit, gates, records | — | — | CPU. **DONE** |
| **W1** | Registry + `ParseKimiLinearParams` (nested `linear_attn_config`, `is_kda_layer` schedule, `mla_use_nope`, MoE scalars). GATE: config/registry resolution from the real `config.json`; KDA/MLA schedule + MLA dims resolve | `registry.py:140`, `configs/kimi_linear.py:34-148`; mirror `kimi_k3_registry.cpp`/`ParseKimiK3Params` + `deepseek_v2` registry | — | CPU |
| **W2** | Loader name-map `kimi_linear_weights.cpp`. GATE: weight-map coverage on ONE shard (zero unmapped/missing, correct shapes) | `kimi_linear.py:460-554` + `fused_moe_make_expert_params_mapping`; reuse `EnumerateKimiK3TextBackboneTensors` | W1 | CPU (1 shard) → GPU (full load) |
| **W3** | KDA device kernel (gate + chunk-KKT + GLA-output; decode = gate + reused GDN recurrent). GATE: vs `kimi_kda` host oracle at real dims, sanitizer-clean | `kimi_gdn_linear_attn.py:233-441` + `kda.py:717-1646`; REUSE our GDN chunk/recurrent kernels + `kimi_kda.{h,cpp}` oracle | — (host refs landed) | GPU (unit-gateable on CPU host-ref first) |
| **W4** | Route the 7 full-attn layers through `mla_attention` with `rotary_emb=None` (**NoPE branch**). GATE: NoPE unit vs RoPE-off MLA + vLLM forward | `kimi_linear.py:180-285`; REUSE `mla_attention.{h,cpp}` + `cuda_mla_*.cu` | W1 | GPU / CPU unit |
| **W5** | MoE assembly — all-but-layer-0 through `RunMoeBlock` (256e/top-8/1-shared/sigmoid/`noaux_tc`/scaling 2.446); layer-0 dense `KimiMLP`. GATE: router unit at Kimi shape | `kimi_linear.py:104-177`; REUSE `deepseek_v2.cpp RunMoeBlock` + `cuda_moe*.cu` | W2 | GPU / CPU unit |
| **W6** | Forward assembly + het-KV wiring — per-layer KDA/MLA dispatch + dense/MoE MLP + plain add+RMSNorm residual; GDN-state + MLA-latent hybrid KV; route through `ModelRegistry::Forward` (born-on-runner) | `kimi_linear.py:353-458`; REUSE `qwen3_5_moe.cpp` skeleton + hybrid-KV groups | W3,W4,W5 | GPU |
| **W7** | Engine gate — e2e SACRED token golden vs oracle on GB10 (STRICT expected), THEN speed (`nsys` both sides, vLLM graphed denominator, match/beat every axis) | §8 recipe | W6 + GPU slot + ~10 GiB disk reclaim | GPU |

**MTP (deferred, OPTIONAL).** `num_nextn_predict_layers=0` in the 48B-Instruct
checkpoint ⇒ no MTP head, not part of the e2e path. If a future Kimi checkpoint
ships `num_nextn_predict_layers>0`, the draft head reuses `qwen3_5_mtp` /
`deepseek_v4_mtp` (`specs/deepseek-v4-mtp.md`, `specs/mtp-spec-decode.md`) with the
loader's `get_spec_layer_idx_from_weight_name` skip already handled upstream. Not
scoped as a W-brick here.

---

## 6. Matrix + claim + records

- **Model matrix** `MODEL-TEXT-kimi-linear-kimi-linear-for-causal-lm`:
  `SPIKE`→`READY`, Spike link → `specs/kimi-linear.md`; checklist entry mark
  `📋`→`🚧`; rollup `SPIKE 7→6`, add `READY 1` (Total 327 unchanged; MODEL row
  count unchanged — an in-place state advance, not a new row).
- **Claim** `CLAIM-KIMI-LINEAR-W0` added to `coordination.md` (owns ONLY:
  `specs/kimi-linear.md`, the Kimi-Linear matrix row + checklist rollup, this
  claim row, the roadmap breadth note, `docs/STATUS.md`/`docs/BENCHMARKS.md`/
  `docs/FEATURES.md` one-liners, `.agents/NOW.md` live-claim row, and a `state.md`
  entry). Co-owner with `CLAIM-MLA-DEEPSEEK` (MLA half) and `CLAIM-KDA-KERNEL`
  (KDA host refs) — this claim adds the **dedicated full spike** and READY
  transition; it touches NO model/kernel source (records-only).
- Record gates run green: `check-agent-record.py`, `check-model-checklist.py`,
  `check-doc-checkpoint` (matrix edit triggers STATUS/BENCHMARKS + FEATURES —
  satisfied), `check-now-current` (NOW.md refreshed in the same change as the
  state append).

---

## 7. Structured contract

**Scope.** The full W0 spike contract for `KimiLinearForCausalLM`
(Kimi-Linear-48B-A3B-Instruct), grounded in the pinned vLLM source + the
authoritative `config.json`, so implementation (W1) can start the moment this
lands. OWNS ONLY the records surfaces in §6.

**Out of scope (with reason):** implementation of anything (spike); Kimi-K3
(separate DERIVE-AND-SHIP row, [kimi-k3.md](kimi-k3.md)); MXFP4 (K3-only); MTP
(absent from this checkpoint); any `MODEL-MM-*` / multimodal row.

**Dependencies / risks / decisions:**
- **e2e-GATEABLE (DECISION).** Kimi-Linear-48B FITS one GB10 and the oracle
  serves it → a REAL SACRED token gate, unlike its 2.8T K3 sibling. STRICT
  expected; near-tie fallback ratified.
- **Disk precondition (RISK).** 91.5 GiB checkpoint vs ~98%-full dgx root →
  reclaim ~10 GiB (own build trees) before download.
- **Shared claims (DEP).** KDA device kernel work is shared with
  `CLAIM-KDA-KERNEL` (host refs, don't re-port) and K3 W4; the MLA reuse is
  `CLAIM-MLA-DEEPSEEK`. Coordinate before W3/W4 so nothing is implemented twice.
- **NoPE MLA (RISK, small).** Our `mla_attention` always builds the RoPE cache;
  the `rotary_emb=None` branch is a simplification but must be explicitly gated —
  do not silently apply RoPE to a NoPE model.
- **Config-authoritative.** Dims are from the HF `config.json` fetch 2026-08-05,
  not the 2026-07-25 sweep (which mislabeled the full-attn layers "MHA" — they are
  MLA). Re-verify at load time against the downloaded `config.json`.

---

## 8. W0 GPU golden-capture recipe (ready to run on the next free GPU slot)

Run on dgx.casa (sm_121, GB10) once a GPU slot frees; SEPARATE from this
CPU-only spike. Serialize + protect the box:

1. **Reclaim + park.** Free ~10 GiB dgx disk (prune old `source-*`/build trees,
   `go clean -cache`); `docker stop local-ai-worker` (restore
   `--restart=always`+start after); `flock $HOME/gpu.lock`.
2. **Oracle build/run check.** Confirm the pinned oracle (`~/venvs/vllm-oracle`
   → 0.25.0-stage or the 555967922 build) constructs AND serves
   `moonshotai/Kimi-Linear-48B-A3B-Instruct` — this is the W0 oracle-RUNS proof,
   not just `AutoConfig`.
3. **Greedy golden capture (context-first, low util).** `sudo drop_caches`;
   create the CUDA context BEFORE loading weights; `gpu_memory_utilization`
   ~0.55–0.65 (unified pool — HOST RAM; do NOT use 0.85). Force the triton MoE
   path. Named tmux + done-marker. Capture greedy token IDs for the standard
   prompt set (mirror the 27B/35B battery), K≥3 runs to establish
   determinism → STRICT vs near-tie band.
4. **Optionally** capture a **reduced-layer** golden first via
   `--hf-overrides '{"num_hidden_layers": 4}'` (2 KDA + 2 MLA, keeps one dense +
   one MoE layer) as a fast smoke oracle before the full 27-layer run — cheaper
   and OOM-safe while wiring W1–W6.
5. **Record** the golden md5 + the determinism verdict into the Kimi-Linear row
   and `benchmark-record.md`; that unblocks W7's e2e gate. Speed comes after
   token-exact (nsys BOTH sides, vLLM graphed denominator, match/beat every axis).

---

## 9. W6 DEVICE FORWARD SEAM LANDED (2026-08-05, `CLAIM-KIMI-LINEAR-W6`)

The born-on-the-runner `KimiLinearModel::ForwardDevice` (the DEFAULT `gather_logits`
production/runner path) replaces the refuse-by-name stub. It composes the
`[rows,vocab]` logits via the landed CPU reference (`KimiLinearModel::Forward` ->
HostForwardSeq, the whole 27-layer hybrid, honoring `logits_indices` gather) and
hands them back **DEVICE-RESIDENT** — a pooled `DBuf` wrapped verbatim like
`deepseek_v2.cpp:633 WrapDeviceLogits`, so `ForwardLogits.on_device()==true` on CPU
and CUDA alike and the runner's on-GPU sampler consumes them with NO host logit
download on the default path (the third MUST-route seam). `check-runner-routing-
consistency` reclassifies Kimi-Linear device-resident (refuse-skipped stubs 2->1,
NO allowlist) and `check-fusion-consistency` stays green. Gate
`tests/vllm/models/test_kimi_linear_forward.cpp` **7/7·300** — case (e) asserts the
device path returns `on_device()` logits byte-exact to the host reference, greedy
tokens identical, `logits_indices` gather -> one device row. Row `SPIKE -> ACTIVE`.

**GPU-verify-pending W7 device-COMPUTE residual (the reuse-wiring plan, authored as
comments in `kimi_linear.cpp`, NOT gateable CPU-only):** a `DBuf`-resident bf16
forward that routes each layer through the reused device blocks —
- **KDA (20 layers):** q/k/v/f_a/f_b/g_a/g_b `vt::MatmulBT`, the 3 short convs
  `vt::CausalConv1dFwd/Update`, q/k `vt::L2Norm`, the reused GDN decode/prefill
  `vt::GdnDecode`/`GdnPrefill`/`GdnPackedDecode`, `vt::RmsNormGated` output norm; the
  KDA decay gate `g = -exp(A_log)*softplus(f_b(f_a(x))+dt_bias)` has NO device
  exp/softplus op -> **host-fallback** via `vllm::kimi_kda` {KdaLowRankDecay,
  KdaDecayGate} then upload (a genuinely-missing KDA piece, W7-speed residual).
- **NoPE-MLA (7 layers):** `mla::ForwardMlaAttentionBlock` with NoPE `MlaBlockDims`
  (rotary_emb=None, q_lora=null, scaling qk_head_dim**-0.5).
- **MoE (26 layers):** `vt::MoeRouterTopK` (kSigmoid, top-8, group 1/1, scaling
  2.446, `e_score_correction_bias`) + grouped GEMM (CPU fallback: per-expert
  `Matmul`+`MoeSiluMul`) + shared expert + `vt::MoeCombine`; dense layer-0 SwiGLU.
- **Fusion:** add+RMSNorm via `vt::FusedChain(kFusedAddRmsNormStd)`; het-KV = the MLA
  latent-576 group + the KDA GDN MambaSpec group advanced in place per step.
This lands the runner SEAM + the full PLAN so only the GPU verify (the compute vs the
pinned oracle, the GDN Triton-AOT cubins, bf16 numerics) + the W0/W7 e2e SACRED
golden remain. Every op named is CPU+CUDA-registered except the bf16 grouped-MoE GEMM.

---

## 10. W7 DEVICE COMPUTE LANDED, CPU-gated (2026-08-05, `CLAIM-KIMI-LINEAR-W7`)

The §9 plan is now IMPLEMENTED in the additive TU
`src/vllm/model_executor/models/kimi_linear_device.cpp`.
`KimiLinearModel::ForwardDeviceCompute` composes the whole 27-layer hybrid over
POOLED f32 `DBuf`s through the SHARED `vt::` device ops (mirroring `deepseek_v2.cpp`'s
device forward). **On device** (genuine `vt::` dispatch): `vt::Embedding`; the
residual add+RMSNorm glue via `vt::FusedChain(kFusedAddRmsNormStd)`; every projection
via `vt::MatmulBT` (host weights are torch `[out,in]`=`[N,K]`); the 3 KDA short convs
via `vt::CausalConv1dFwd` (silu, fresh zero conv-state); q/k `vt::L2Norm`; the KDA
output `vt::RmsNormGated` (sigmoid); the MoE router `vt::MoeRouterTopK` (sigmoid
`noaux_tc`, group 1/1, `e_score_correction_bias`, `routed_scaling=2.446`); the
per-expert / dense / shared SwiGLU via `vt::MoeSiluMul`; the weighted `vt::MoeCombine`;
the `lm_head` `vt::MatmulBT`; device-resident logits via the pooled-`DBuf` carrier.

**Two documented HOST-FALLBACK islands (the W7-speed residuals — no portable device
op yet):** (1) the **KDA per-k-channel gated-delta RECURRENCE + its decay gate**
`g=-exp(A_log)*softplus(f_b(f_a(x))+dt_bias)` + `beta=sigmoid(b_proj)` — `vt::GdnDecode`
/`GdnPrefill` carry only a per-HEAD scalar decay `g[T,Hv]` (ops.h), so they CANNOT
express KDA's per-channel `g[T,H,D]`; computed on host from the device-resident
q/k/v/g1/beta via the landed `kimi_kda` refs + the reference recurrence, then
uploaded. (2) the **NoPE-MLA attention CORE** (causal softmax) — the device path is
`mla::ForwardMlaAttentionBlock` over the runner's paged het-KV + load-time W_UK/W_UV
absorption (the born-on-runner residual), so this seam keeps every MLA projection +
`kv_a_layernorm` + `kv_b` ON DEVICE and computes only the softmax core on host
(identical materialized-MHA math, NoPE so no RoPE).

**Why a CPU match is a REAL gate:** the CPU backend runs the SAME `vt::` dispatch (a
pooled `DBuf` is a device buffer on CPU; weights alias the host f32 bytes exactly as
`ResidentWeight`'s CPU branch), and activations are f32 (the reference holds f32), so
the device compute matching the W2 host f32 reference proves the residual-stream /
vt-op / MoE-routing / device-logits WIRING the GPU will run — only the GPU numerics
stay pending. Gate `tests/vllm/models/test_kimi_linear_forward.cpp` **12/12·614**:
(f) KDA layer device==ref (rtol 3e-3), (g) NoPE-MLA (rtol 1e-4), (h) MoE + dense
(rtol 2e-4/1e-4 — the device router selects the SAME experts, lowest-index tie),
(i) the whole `ForwardDeviceCompute` == ref logits (rtol 5e-3) AND
greedy-token-identical AND device-resident, (j) `ForwardDevice` reaches the device
compute under the opt-in flag. Runner routing: `ForwardDevice` routes to
`ForwardDeviceCompute` under `VT_KIMI_DEVICE_COMPUTE=1` (default OFF keeps the
CPU-verified W6 host-ref compose as production until GPU-verified).

**HONEST — NOT DONE for GPU (box down):** bf16 activations (vLLM parity), the GDN
Triton-AOT decode cubins, the paged het-KV, the grouped-MoE slabs, and the e2e SACRED
greedy golden vs the pinned oracle (§8) stay a NAMED pending. Box-return: CUDA build
(`-DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0`), run
`test_kimi_linear_forward` on a CUDA queue (token-identical to the W2 ref within a
bf16 envelope), then the §8 e2e golden, then speed (`nsys` both sides, replacing the
two host-fallback islands with a device KDA per-channel recurrence + exp/softplus gate
op + the paged `mla::ForwardMlaAttentionBlock` + the grouped-MoE slabs). Row STAYS
`ACTIVE`.

---

## 11. W7 GPU-VERIFY LANDED — device compute runs on GB10; e2e still disk-blocked (2026-08-06, `CLAIM-KIMI-LINEAR-W7` GPU-verify, branch `row/MODEL-KIMI-LINEAR-GPU`)

The §10 device compute is now GPU-VERIFIED on dgx.casa (GB10, sm_121a). A clean
from-`origin/main` CUDA build (`-DVLLM_CPP_CUDA=ON -DVLLM_CPP_TRITON=ON
-DVLLM_CPP_CUDA_ARCHITECTURES=121a -DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0`,
nvcc 13.0.88, Release, `-Werror` clean, built in `/dev/shm` because the root disk is
100% full) resolved the full PRODUCTION stack — the configure log prints `CUTLASS
found … enabling sm120a NVFP4 cutlass GEMM`, `FlashAttention-2 prefill/decode: ENABLED
[121a]`, and the vendored `Triton AOT … sm_121a (MANIFEST hashes OK)` GDN kernels; `nm`
on the test binary confirms all 14 GDN AOT stable symbols
(`gdn_{chunko,chunko_bf16,decode,deltah,kkt,tril,wu}_h{32,48}_default`) linked in.
`tests/vllm/models/test_kimi_linear_forward.cpp` runs **12/12·614 GREEN on the GPU**,
BOTH arms: `VT_KIMI_DEVICE_COMPUTE=1` (the device-compute path, the gate) AND the
default host-ref W6 compose. So the f32 `vt::` device dispatch (embed / `FusedChain`
add+RMSNorm / `MatmulBT` projections / `CausalConv1dFwd` / `L2Norm` / `RmsNormGated` /
`MoeRouterTopK` / `MoeSiluMul` / `MoeCombine` / lm_head) matches the W2 host f32
reference on real Blackwell hardware within the same tolerances the CPU gate used — no
numerics divergence, no DeepSeek-class trap (norms bf16→ReadF32, async-inputs,
keep-quant slices, capture) triggered. The two documented host-fallback islands (KDA
per-channel recurrence+gate, NoPE-MLA softmax core) still run on host by design.

Oracle gateability RE-CONFIRMED (cheap, no weight load): the live `~/venvs/vllm-oracle`
(→ `vllm-oracle-v0.25.0-stage`, vLLM 0.25.0) registers `KimiLinearForCausalLM`
(`registry.py`) with `models/kimi_linear.py` + `transformers_utils/configs/kimi_linear.py`
present — it CAN construct/serve the model (matches the 2026-07-25 sweep and §0).

**STILL disk-BLOCKED (honest, recorded — not hidden):** the §8 e2e SACRED greedy
golden vs the oracle needs the 91.5 GiB bf16 checkpoint, which is ABSENT (not in
`~/models`, not in the HF hub cache) and cannot be fetched — dgx root is 100% full with
only **34 GiB free** (need ~91.5 GiB; the ~60 GiB reclaim would mean deleting other
agents' ACTIVE laguna/ds4/bench campaign trees, which this row will not do
unilaterally, and there are no stale `source-*` grid trees to prune). No smaller
published quant exists (§3). So Steps 3-5 of the §8 box-return recipe are recorded
disk-BLOCKED; the oracle golden + bf16-activation parity + speed stay the NAMED
residual. What GPU-verify PROVES: the f32 device-compute WIRING (vt-op dispatch, MoE
routing, device-resident logits, GDN cubin linkage) runs on GB10. What it does NOT:
bf16 numerics vs the oracle, and the e2e token gate. Row STAYS `ACTIVE`.

---

## 12. §8 SACRED ORACLE GOLDEN CAPTURED (STRICT); full our-engine e2e f32-loader-blocked (2026-08-06, branch `row/MODEL-KIMI-LINEAR-E2E`)

Disk was unblocked upstream (129 GiB free) and the mission resumed on a fresh
branch off `origin/main` `c1a7b452` (the §11 GPU-verify records merged as #37).

**Step 3 (download) DONE.** `moonshotai/Kimi-Linear-48B-A3B-Instruct` pulled into the
HF cache (snapshot `e1df551a…`), 20 safetensors shards, exact size 98,245,528,576 B =
**91.5 GiB bf16 / ~49.1B params**; `config.json` matches this spec byte-for-byte
(27 layers, H=2304, vocab 163840, 256 experts, `num_nextn_predict_layers=0`, the KDA
`kda_layers`/`full_attn_layers` schedule). df landed at 37 GiB free (> the 15 GiB
floor). tiktoken tokenizer via remote code (`tokenization_kimi.py`, `tiktoken.model`).

**Step 4 (§8 oracle golden) DONE — STRICT.** Captured on GB10 with the live
`~/venvs/vllm-oracle` (→ 0.25.0-stage), oracle ALONE under both flock locks, after
`drop_caches` with 116 GiB available. Recipe (mirrors `deepseek-v2-oracle-capture.py`):
`moe_backend=triton` (MANDATORY — the FlashInfer CUTLASS MoE workspace OOM-reboots this
box), `enforce_eager`, `max_num_batched_tokens=512`, `max_num_seqs=1`,
`max_model_len=2048`, per-prompt batch=1, K=3, T=16, the 8-prompt battery.
**MEMORY (the crux): vLLM footprint ≈ util × 119 GiB, so a 91.5 GiB model needs
util ≈ 0.82 (NOT the ≤0.60 that cannot even hold the weights — 0.60×119=71<91.5).**
`gpu_memory_utilization=0.82` → ~97.6 GiB footprint, **min available memory 15 GiB
throughout, no reboot**. Verdict: **ALL 8 prompts DETERMINISTIC over K=3 → STRICT
token-exact gate** (as §4 predicted for a 48.9B MoE). All completions coherent (Paris/
Rome/Berlin; `def fibonacci`; "Au … Latin word aurum"; Shakespeare's Hamlet). Golden
committed at `tests/parity/goldens/kimi_linear_greedy/` (`greedy_ids.npy` [8,16],
`greedy_dist.npy` [8,16,3], `p{0-7}_prompt.i32`); recipe at
`scripts/kimi-linear-oracle-capture.py`. NOTE: the §8-step-4 reduced-layer smoke
(`hf_overrides num_hidden_layers=4`) is INCOMPATIBLE with the Kimi hybrid schedule in
vLLM 0.25.0 (loader `KeyError: layers.4.block_sparse_moe.…`); the full run needs no
override, so the smoke was skipped.

**Step 5 (our-engine e2e) BLOCKED on OUR f32 loader — NOT disk, NOT oracle.**
`kimi_linear_weights.cpp::MaterializeHost` decodes EVERY weight tensor to row-major
`std::vector<float>` (f32) for the W2 CPU reference — all 27 layers + 256 experts, so
~49.1B params × 4 B ≈ **183 GiB resident, well over the 119 GiB unified pool** (plus the
mmap'd bf16 shards during load). Loading the full model in our engine would OOM-reboot
the box, so it was NOT attempted (safety). This is exactly the §10/§11 "bf16 activations
(vLLM parity)" residual: the **bf16-resident loader + forward** must land before the
full-model our-engine e2e token gate can run. The device compute is already GPU-verified
(§11, `test_kimi_linear_forward` 12/12·614), and the STRICT golden is now committed and
ready to gate against the moment the bf16 path exists.

**Step 6.** Row STAYS `ACTIVE` (the e2e token gate did not run). `VT_KIMI_DEVICE_COMPUTE`
default STAYS OFF (a parity-enabler flip needs the e2e gate green). No tok/s (no
full-model our-engine load was safe). Residuals now precisely: (a) the bf16-resident
loader/forward (unblocks the full our-engine e2e vs this golden), (b) the paged het-KV +
GDN Triton-AOT decode + grouped-MoE device slabs, (c) speed. Box left clean.

---

## 13. bf16-RESIDENT loader/forward — POOL MATH + design (2026-08-06, branch `row/MODEL-KIMI-LINEAR-BF16`)

The final phase-3 brick: replace the f32 `MaterializeHost` (183 GiB) with a bf16-resident
loader/forward so the full-model e2e fits the 119 GiB unified pool and can gate against the
committed STRICT golden (§12).

### POOL MATH (done BEFORE building, per the design constraint)
- Weights bf16: exact 98,245,528,576 B = **91.5 GiB** (§12 index).
- Device-resident staging: cudaMalloc'd `OwnedTensor::d_dev` (native GPU memory, no ATS
  penalty — the [[gb10-weight-residency-ats-penalty]] lever), **91.5 GiB** in the unified
  pool. Per-tensor stage-then-`ReleaseHost` keeps only ONE tensor's host bytes live at a
  time (biggest = embed/lm_head [163840,2304] bf16 = 0.72 GiB), so the LOAD peak is
  ~91.5 GiB device + <1 GiB host + the mmap read window (`ReleaseSourcePages` per tensor).
- Activations f32 (the residual stream stays f32 — norms/convs/L2Norm/RmsNormGated/
  host-islands unchanged): the golden decodes T ≤ prompt(≤20)+16 ≈ 36 tokens; the largest
  buffers are the last-row logits [1,V=163840] f32 = 0.64 MB, KDA [T,4096], router [T,256],
  expert_out [T,8,2304] — all < ~0.3 GiB total at T≈36. The per-GEMM bf16 activation cast
  scratch [T,K] is tiny.
- Norm/scale vectors kept host f32 (ReadF32-on-demand from bf16, the [[gate-comparing...]]
  bf16→ReadF32 trap): input/post/final norms + kv_a_layernorm + o_norm + a_log + dt_bias +
  gate + e_score_correction_bias, ~27 layers × few KB = < 0.1 GiB.
- CUDA context (reserved FIRST, before load, per the GB10 recipe `examples/laguna_gen/
  main.cpp:185-194`): ~1-2 GiB.
- **STEADY total ≈ 91.5 + ~0.3 + ~0.1 + ~2 ≈ 94 GiB → ~25 GiB headroom. CLOSES.** The
  e2e harness recomputes context (no paged KV) so there is no KV cache to budget; a future
  engine path adds only the small MLA latent-576 (7 layers) + GDN state groups.

### DESIGN (grounded in the recorded winning patterns)
1. **bf16-resident storage** — a new `KimiLinearResidentWeights` of `OwnedTensor`s
   (`qwen3_5_weights.h:40`, reusing its `d_dev` cache), one per checkpoint tensor, mirroring
   `laguna_weights.cpp` / `gemma_weights.cpp`. Loader path uses `dense_loaders::LoadBf16Direct`
   (`dense_weight_loaders.h:63`) per tensor + `ReleaseSourcePages` (`safetensors_reader.cpp:317`),
   NEVER `MaterializeHost`. The f32 `KimiLinearHostWeights` + `MaterializeHost` STAY for the
   SMALL-config unit gate (tiny shapes) — the full-model path must never materialize f32.
2. **Resident GEMM** — a `KimiResidentBf16W(q,w,dev)` helper mirroring `laguna.cpp:125-139`
   (cudaMalloc+one-H2D to `d_dev`, byte-exact; CPU aliases host bytes) + a `GemmBf16(d,out_f32,
   act_f32,w,{N,K})` mirroring `laguna.cpp:1939-1946`: `vt::CastBf16(act_f32→bf16 scratch)` then
   `vt::MatmulBT(out_f32, act_bf16, w_bf16-resident)` — the (bf16,bf16)->f32 combo the CUDA
   MatmulBT SUPPORTS (`cuda_matmul.cu:3`; the elementwise f32-act×bf16-weight it LACKS,
   `cuda_deepseek_v4.cu:1821`). This makes the GEMM numerics vLLM-bf16 (best token-exact
   chance vs the bf16 oracle golden), keeps the residual stream f32.
3. **Device forward** — bf16 variants of `DeviceForwardBody`/`KdaLayerDevice`/`MlaLayerDevice`/
   `MoeBlockDevice`/`DenseMlpDevice` (`kimi_linear_device.cpp`) that take the `OwnedTensor`
   weights and call `GemmBf16` in place of `WF32`+`MatmulBT` at each of the ~20 GEMM sites; the
   two host-fallback islands (KDA recurrence, NoPE-MLA softmax) are UNCHANGED (f32 activations,
   small). Norms via `ReadF32`/`ResidentWeightF32` (`dense_attn_block.h:202`).
4. **Runner** — `ForwardDevice` drops the `host.materialized` precondition when the resident
   weights are present; the registry `LoadKimiLinearForCausalLM` uses the bf16 loader for the
   full model. GB10 load recipe (context-first + shard-release) in a Kimi gen driver / the e2e
   harness, mirroring `examples/laguna_gen/main.cpp:185-237`.
5. **e2e vehicle** — a greedy-decode harness (mirroring the CPU `KimiLinearGreedyDecode`) that
   loads the bf16-resident checkpoint, decodes the §12 8-prompt battery × 16 tokens through the
   bf16 `ForwardDeviceCompute`, and compares token-exact to `tests/parity/goldens/
   kimi_linear_greedy/greedy_ids.npy`. NOTE: the `VT_KIMI_DEVICE_COMPUTE=0` arm (the pure f32
   host `Forward`) CANNOT run the full model (183 GiB f32) — only the `=1` bf16 device path
   fits, so the full-model gate is the `=1` arm; the `=0` arm stays the tiny-config reference.

### Gates
CPU: `test_kimi_linear_forward` stays 12/12·614 (f32 tiny path untouched) + a tiny-config bf16
device gate (bf16 forward == f32 reference within a bf16 tolerance). dgx CUDA build (usual
flags, nm GDN). Full-model e2e vs the STRICT golden (`=1` arm), free -g ≥ 90 before load,
memory-monitored; if the pool math does not close in practice, STOP and record.

### IMPLEMENTATION LANDED — loader/forward + CPU bf16 gate (2026-08-06, `row/MODEL-KIMI-LINEAR-BF16`)
The §13 design is now implemented (`file:line`):
- **bf16 loader** `LoadKimiLinearResidentBf16Weights` + `StageKimiResidentBf16` + `BuildKimiResidentFromHost`
  (`kimi_linear_weights.cpp`): every large matmul weight (embed/lm_head/all *_proj/all MLP
  gate/up/down/router gate) via `dense_loaders::LoadBf16Direct` -> `OwnedTensor`, then (on a CUDA
  queue) staged to `d_dev` (cudaMalloc + one H2D) and `ReleaseHost`'d immediately after its copy so
  the LOAD peak holds only one tensor's host bytes; the tiny norm/scale/conv/bias vectors decoded
  to host f32 (`ReadFloatVec`, dtype-agnostic — checkpoint has only `dt_bias`/`A_log` in F32, the
  rest BF16). NEVER `MaterializeHost` (183 GiB f32). Real checkpoint = 20 shards, 91.5 GiB bf16,
  tie_word_embeddings=false, 20 KDA + 7 NoPE-MLA layers, 256 experts.
- **Resident weights** `KimiLinearResidentWeights` (`kimi_linear.h`) mirroring `laguna_weights.cpp`
  over `OwnedTensor::d_dev`; the router gate is bf16-resident too (matches vLLM's bf16 router — a
  f32 gate could flip near-tie top-8 expert selections; the primary token-divergence risk).
- **Device forward** bf16 variants `DeviceForwardBodyBf16` / `Kda|Mla|MoeBlock|DenseMlp|SwiGlu
  DeviceBf16` (`kimi_linear_device.cpp`): the Laguna cast-GEMM `GemmBf16` (CastBf16 act -> bf16 scratch,
  MatmulBT (bf16,bf16)->f32 vs `ResidentBf16W` — vLLM's projection numerics) at each of the ~20 GEMM
  sites; f32 residual stream; the two host-fallback islands (`KdaRecurrenceIsland`, `MlaSoftmaxIsland`)
  EXTRACTED and shared with the f32 path (byte-identical), small vectors via `WF32`.
- **Runner** `ForwardDevice` drops the `host.materialized` precondition on the resident path and
  routes to `ForwardDeviceCompute`, which dispatches bf16 (`weights.resident.resident`) vs f32.
- **e2e harness** `examples/kimi_linear_gen/main.cpp`: context-first GB10 load recipe + shard-release,
  greedy-decode the 8-prompt golden battery x 16 tokens through the bf16 `ForwardDeviceCompute`,
  token-compare to `greedy_ids.npy`.

**CPU gate GREEN:** `test_kimi_linear_forward` **13/13·656** — the original 12/12·614 (f32 tiny path
UNTOUCHED, incl. the extracted islands) + NEW case (k) `BuildKimiResidentFromHost` bf16-resident
`ForwardDeviceCompute` == the independent f32 host reference within a bf16 envelope (rtol 6e-2 + atol
scaled to |logit|).

### FULL-MODEL GB10 e2e RAN — NEAR-TIE 106/128 (6/8 prompts token-exact); pool math CLOSES (2026-08-06, `row/MODEL-KIMI-LINEAR-BF16`)
The full 48.9B model now RUNS end-to-end on one GB10 through the bf16-resident path (the primary
mission goal — the f32-loader block is CLEARED). dgx CUDA build (`-DVLLM_CPP_CUDA=ON
-DVLLM_CPP_TRITON=ON -DVLLM_CPP_CUDA_ARCHITECTURES=121a -DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0`,
nvcc 13.0.88, Release, `-Werror` clean, built in `/dev/shm`); `nm` = all 14 GDN AOT stable symbols
linked; `test_kimi_linear_forward` **13/13·656** in the CUDA-built binary. `kimi-linear-gen --gpu`
loads the checkpoint (context-first + shard-release) and greedy-decodes the §12 8-prompt battery x16
through the bf16 `ForwardDeviceCompute`, token-compared to `greedy_ids.npy`.

**MEMORY (pool math CLOSES in practice):** load 117.6s; **host RSS PEAK 1.7 GiB** (stage-then-
ReleaseHost keeps ONE tensor's host bytes live, exactly the §13 design); **device peak 98.5 GiB**
(100896 MiB, ~91.5 GiB weights + activations/context/page-cache); **min available 21.6 GiB**
throughout — ABOVE the 15 GiB floor and matching the predicted ~25 GiB headroom. NO OOM, NO reboot.
A load-only smoke first proved the ~20k per-tensor cudaMalloc staging closes safely before the run.

**TOKEN GATE: NEAR-TIE, not STRICT — 106/128 (82.8%).** Prompts 0,1,3,4,5,6 are **16/16 token-exact**;
prompt 2 diverges from token 0 (got 261 vs golden 276); prompt 7 matches 6 then diverges at a comma
boundary (pos 6: got 387 vs golden 11) and cascades (greedy is path-dependent). The golden is
DETERMINISTIC (K=3 identical) at both divergence points, so this is a genuine numerics NEAR-TIE, not
a wiring bug — 96 consecutive token-exact tokens across 6 prompts prove the loader / resident-GEMM /
MoE-routing / residual-stream WIRING is correct.

**DIVERGENCE ROOT CAUSE (honest):** the correctness-vehicle forward is NOT bit-identical to vLLM's
bf16 stream. (1) The residual stream is kept f32 (the §13 design, so the two host-fallback islands
can consume it); vLLM ROUNDS the residual to bf16 after every add + before every RMSNorm, so vLLM's
norm variance is computed over bf16 values, ours over f32 (higher precision). (2) The KDA recurrence
+ NoPE-MLA softmax islands run in f64 on host; vLLM runs the GDN Triton-AOT decode + FA2 MLA in
bf16/f32 on device. So our forward is MORE accurate than vLLM per-op, which lands a DIFFERENT top-1
where vLLM's deterministic bf16 top-1 has a small margin (punctuation / word boundaries; longer
prompts accumulate more delta — p2 is the longest at 14 prompt tokens and flips immediately).

**PATH TO STRICT (the named W7-speed residuals, unchanged):** replace the two host-fallback islands
with the device GDN per-channel-decay recurrence + exp/softplus gate op and the paged
`mla::ForwardMlaAttentionBlock`, then carry a bf16 residual stream end-to-end (matching vLLM's
rounding). That closes the numerics gap AND removes the host round-trips (the current 1.59 tok/s is
the O(n^2) full-recompute + host-island rate, not an optimized decode).

**tok/s:** 1.59 tok/s steady (0.630 s/step over 127 steps) — the correctness-vehicle rate (full
recompute per step + host islands), a NAMED speed residual. **DEFAULT:** `VT_KIMI_DEVICE_COMPUTE`
STAYS OFF (parity-enablers: flip ON only with the token gate green at the flipped default; a near-tie
is not token-exact). Row STAYS `ACTIVE`.

---

## 14. W7-speed STRICT lever MEASURED — bf16 regime recovers 106→120/128, plateaus; device islands remain the residual (2026-08-07, `row/KIMI-LINEAR-STRICT-SPEED`)

The recorded path to STRICT ("device islands + bf16 residual stream = the same work as
speed") was implemented as three env-gated numeric knobs in `kimi_linear_device.cpp`
(default OFF → the f32 vehicle is byte-identical, CPU gate `test_kimi_linear_forward`
**13/13·656** in the CUDA binary) and MEASURED on GB10 (the full 48.9B model, the §12
128-token gate vs the STRICT deterministic golden). Clean-from-`origin/main` CUDA build
(`-DVLLM_CPP_CUDA=ON -DVLLM_CPP_TRITON=ON -DVLLM_CPP_CUDA_ARCHITECTURES=121a -DVLLM_CPP_
CUTLASS_DIR=…cutlass-4.5.0`, nvcc 13.0.88, Release, 14 GDN AOT symbols linked). Memory
safe throughout every reload (min-avail ≥ 115 GiB, both flock locks, reclaim-wait, no
reboot).

### Knobs (`kimi_linear_device.cpp`)
- `VT_KIMI_BF16_RESIDUAL` — carry the residual stream in bf16 like vLLM's
  `fused_add_rms_norm` (residual stored bf16, block outputs bf16, RMSNorm variance over
  the f32 pre-store sum). Implemented as in-place `CastBf16`→`CastF32` rounds at exactly
  vLLM's rounding points (embed out, each block out, residual after each add) keeping f32
  STORAGE so the islands still read f32 — byte-matching vLLM's fused-add order.
- `VT_KIMI_BF16_ISLANDS` — round the host-fallback island INPUTS (KDA q/k/v/g1/beta,
  NoPE-MLA q/kv/kpe) to bf16 (RNE) before the recurrence/softmax.
- `VT_KIMI_ISLAND_F32ACC` — f32 (not f64) accumulation in the islands. **MEASURED NEGATIVE**,
  kept as a documented-negative A/B knob.

### Measurement (token match /128 vs the deterministic golden; STRICT required)
| Config | env | /128 | verdict |
|---|---|---|---|
| control | (none) | **106** | reproduces §13 baseline exactly |
| residual only | `BF16_RESIDUAL` | 106 | net-zero — SHUFFLES flips (fixes p2, BREAKS p3 into a `163586×` repeat loop) |
| islands only | `BF16_ISLANDS` | 106 | fixes p2 but destabilizes p3 (repeat loop) — net-zero |
| **residual + islands** | `BF16_RESIDUAL BF16_ISLANDS` | **120** | **BEST** — p0–p6 all 16/16 exact; only p7 flips |
| + island-output bf16 | `…BF16_ISLANDS(out)` | 90 | **REGRESSION** (reverted) |
| + f32 accumulation | `…ISLAND_F32ACC` | 91–106 | **NEGATIVE** (reverted from the ISLANDS path) |

### Flip ledger / razor verdict (the deterministic golden arbitrates every flip)
- The two levers INTERACT: island bf16-input rounding fixes p2 but destabilizes p3 into a
  degenerate repeat (`163586×`); the bf16 residual stream then RE-stabilizes p3 (kills the
  repeat). Together they make **p0–p6 all 16/16 token-exact** (was 6/8 → 7/8 fully exact).
- The SOLE remaining divergence at 120/128 is **p7 position 8**: the golden deterministically
  emits `18705`, our island emits `58084` (a genuine near-tie), then the greedy path cascades
  (8/16 on p7). A single near-tie flip across the whole 8-prompt battery.
- Further precision-"matching" (rounding the island OUTPUT to bf16 → 90; f32 accumulation →
  91–106) is a COIN-FLIP that regresses, because it is not vLLM's ACTUAL GDN-Triton / FA2
  kernel arithmetic — it just perturbs which near-ties flip. Host-precision-matching PLATEAUS
  at 120/128.

### Verdict + default
**NO arm reaches STRICT** (best 120/128 is a DIVERGENCE; the golden is K=3 deterministic so
STRICT — not the distributional gate — is required). Per parity-enablers, `VT_KIMI_DEVICE_
COMPUTE` and all three knobs STAY **OFF** (a near-tie is not token-exact). Row STAYS `ACTIVE`.

### The named residual — the device islands (why host-matching cannot close p7)
The one principled path to STRICT is routing the two islands through vLLM's ACTUAL device
kernels, but it is NOT a drop-in (the mission's own assessment, now proven by measurement):
- **KDA:** vLLM's decay is **per-k-channel** `g[T,H,D]` (`kimi_gdn_linear_attn.py` +
  `third_party/flash_linear_attention/ops/kda.py`), but our `vt::GdnDecode`/`GdnPrefill`
  carry only a **per-HEAD scalar** decay `g/beta[T,Hv]` (`include/vt/ops.h:1797,1846`). So the
  vendored GDN Triton-AOT cubins CANNOT express KDA — a NEW per-channel-decay GDN kernel
  (`g[T,H,D]` + the `-exp(A_log)*softplus(f_b(f_a(x))+dt_bias)` gate) is required.
- **NoPE-MLA:** needs the paged `mla::ForwardMlaAttentionBlock` (FA2) over the runner's het-KV
  (the born-on-runner residual), not a host softmax.
These are ALSO the speed levers — the correctness vehicle re-computes the WHOLE sequence every
decode step (O(n²)) with a host Download/upload per KDA/MLA layer per step. That is why the
measured tok/s is invariant to the numeric knobs.

### Speed (HW-forced-indirect — vLLM cannot serve this model on one GB10 at bf16)
Steady tok/s over 127 steps (single-load, medians, cold first-leg discarded): control **1.31**,
islands **1.30**, resIsl **1.30** (first-step ~0.62 s, steady ~0.77 s/step). The extra bf16
rounding casts cost ~0.15 s/step vs the §13 1.59 baseline; ALL configs are the same O(n²)
full-recompute + host-island rate. **Honest bar:** the §12 oracle golden capture itself needed
`gpu_memory_utilization=0.82` (~97.6 GiB) with only 15 GiB min-avail for a SINGLE-seq eager
run — vLLM cannot SERVE Kimi-Linear-48B at bf16 on ONE GB10 with any KV headroom, so a direct
`vllm bench throughput` arm is HW-infeasible; the comparison is recorded as HW-forced-indirect
(our absolute 1.30 tok/s + the per-step GPU-active anchor). No isolated host-tail lever (grouped
MoE seam, on-GPU sampling) moves the needle without the device-island + paged-incremental-decode
rewrite, which is the SAME W7-speed residual. Scoped as the named follow-up, not forced.

---

## 15. PER-CHANNEL-DECAY KDA DEVICE KERNEL LANDED (2026-08-07, `row/KIMI-KDA-DEVICE-KERNEL`)

The §14-named residual — "our `vt::GdnDecode`/`GdnPrefill` carry only a per-HEAD scalar decay
`g[T,Hv]`, so a NEW per-channel-decay GDN kernel (`g[T,H,D]`) is required" — is now IMPLEMENTED
as the additive device op **`vt::KdaGatedDeltaRule`**, the genuinely-net-new-vs-GDN primitive.

**Grounding (file:line, BOTH sides @ pin 555967922).** KDA's decode path REUSES the exact GDN
recurrence kernel — `fused_recurrent_kda` (`third_party/flash_linear_attention/ops/kda.py:109-146`)
calls `fused_recurrent_gated_delta_rule_fwd_kernel` with `IS_KDA=True`
(`ops/fused_recurrent.py:88-175`). The SOLE net-new numeric is the decay application: plain GDN
does `b_h *= exp(b_g)` (per-HEAD scalar, `fused_recurrent.py:132-134`), KDA does
`b_h *= exp(b_gk[None, :])` (per-K-CHANNEL, `:136-137`) — `g` is `[T,Hv,Dk]`, one log-decay per K
channel of the value head's `[Dv,Dk]` state, broadcast across the Dv rows. Everything else
(decay → predict → beta → rank-1 update → read-out, all `tl.float32` on bf16 loads) is byte-for-byte
GDN's recurrence. The op is thus GdnPrefill's per-channel twin; the shared GDN kernels are UNTOUCHED
(Qwen3.6 27B/35B gate byte-identical — `test_ops_gdn` 58/58·1825 unchanged).

**Implementation (ours, additive).** OpId `kKdaGatedDeltaRule` + `KdaGatedDeltaRuleFn`
(`include/vt/ops.h`); wrapper + per-channel-g validation (`src/vt/ops.cpp`); CPU
`KdaHeadTokenStep`/`KdaGatedDeltaRuleKernel` (`src/vt/cpu/cpu_ops.cpp`, GdnHeadTokenStep with a
per-`ki` decay vector); CUDA `KdaScanKernel` + `KdaGatedDeltaRuleKernelCuda` (`src/vt/cuda/cuda_gdn.cu`,
GdnScanKernel staging the per-K decay in shared memory, +1 dk-array). Both dual-registered CPU+CUDA.

**Unit gate (RED-first, `tests/vt/test_ops_kda_recurrence.cpp`) — 3/3·6 CPU-green:**
(1) EQUIVALENCE — with `g` broadcast from a per-head scalar, the per-channel op reduces
**BIT-IDENTICALLY** to the landed+gated `vt::GdnPrefill` (out & state exact-float-equal), tying the
net-new op to a proven reference with zero new numerics; (2) PER-CHANNEL — distinct per-channel
decay vs the from-first-principles f64 island reference (`KdaRecurrenceIsland` math) at documented
f32 tolerance (atol 1e-4, rtol 3e-3); (3) VALIDATION — rejects per-head `g` and unset scale; plus a
CPU↔CUDA parity case (GPU-pending). GDN untouched, `test_kimi_kda` 14/14, `test_kimi_linear_forward`
13/13·656 unchanged.

**Wiring (opt-in, default OFF).** `KdaRecurrenceIsland` (`kimi_linear_device.cpp`) gains a
`VT_KIMI_DEVICE_KDA` branch: q_n/k_n/v (already device-resident) feed `vt::KdaGatedDeltaRule` with a
fresh zero state + qsl=[0,T]; only the elementwise decay gate (`KdaDecayGate`) + beta = sigmoid(b)
stay host (numerically stable; the numerically-sensitive object is the RECURRENCE). Requires
`VT_KIMI_DEVICE_COMPUTE=1`. CPU whole-forward gate passes with the flag ON (`test_kimi_linear_forward`
13/13·656, f32 device recurrence within the forward's rtol 5e-3 vs the f64 ref) — the WIRING is
correct. Default OFF (parity-enabler) keeps the f64 host path as production.

**Why this is the STRICT path (spec §14 razor).** §14 proved host-precision-matching PLATEAUS at
120/128 because the f64 island is MORE precise than vLLM and coin-flips near-ties (f32-accumulation
knob regressed 120→91-106). This op runs vLLM's ACTUAL f32-on-bf16 recurrence arithmetic on device,
not a host approximation — the principled STRICT lever AND the speed lever (it is the per-step device
recurrence the paged-incremental-decode rewrite needs).

**GPU-VERIFIED + FULL-MODEL GATE MEASURED on GB10 (2026-08-07, sm_121a, clean Release CUDA build,
Triton-AOT vendored, cutlass-4.5.0).** Kernel GPU-verify: `test_ops_kda_recurrence` **4/4·8 GREEN**
on the CUDA binary (the CPU↔CUDA parity case confirms `KdaScanKernel` == the CPU kernel on Blackwell);
GDN untouched `test_ops_gdn` 66/66·4242; `test_kimi_kda` 14/14; 23 KDA symbols linked. Full 48.9B
128-token gate vs the §12 STRICT golden, single-load per config, memory-safe throughout (host RSS peak
1.7 GiB, min-avail 21 GiB, freed cleanly between configs, NO reboot):

| Config | env | /128 | tok/s | verdict |
|---|---|---|---|---|
| control (f64 host recurrence) | `DEVICE_COMPUTE=1` | 106 | 1.35 | reproduces §13/§14 baseline |
| **device-KDA** | `DEVICE_COMPUTE=1 DEVICE_KDA=1` | **122** | **4.24** | **NEW BEST on BOTH axes** |
| device-KDA + bf16 knobs | `…DEVICE_KDA=1 BF16_RESIDUAL BF16_ISLANDS` | 90 | 4.19 | REGRESSION (reverted) |

**RESULT (the §14 thesis CONFIRMED).** The device recurrence — vLLM's ACTUAL f32-on-bf16 arithmetic —
moves **106→122/128** (prompts 0-6 all 16/16; only p7 diverges at pos-6, `387` vs golden `11`, a comma
near-tie) AND is **3.1× FASTER (1.35→4.24 tok/s)**. It beats BOTH the control (106) AND §14's
host-precision best (120, which needed both bf16 knobs). It FIXES the p2 divergence the f64 host path
had — because it runs the right arithmetic, not a coin-flip. The §14 bf16 knobs are now SUPERSEDED and
COUNTERPRODUCTIVE (device-KDA + bf16 REGRESSES 122→90, reintroducing p3's `163586×` repeat loop) — they
were tuned to compensate for the f64 host island's over-precision; on the already-correct device
arithmetic they perturb the wrong way. The speed win is because the device recurrence kills the host
Download/f64-recompute/upload round-trip and runs the O(T²) recurrence in parallel on the GPU.

**Default + parity-enabler.** `VT_KIMI_DEVICE_KDA` STAYS OFF (122/128 is still a DIVERGENCE, not STRICT;
parity-enablers flip only with the token gate green). But the result reframes the residual: it is now a
SINGLE near-tie (p7 pos-6) and the recorded next brick is the clear path to STRICT + more speed.

**NAMED residuals to STRICT (the p7 near-tie).** vLLM processes the PROMPT with the CHUNKED prefill
kernel (`chunk_kda`), we still run the RECURRENT form over the whole sequence; and the 7 NoPE-MLA layers
still use a host f64 softmax island. Closing p7 needs (c) the KDA chunked-prefill kernel family +
(d) paged `mla::ForwardMlaAttentionBlock` for the NoPE-MLA layers + (e) paged-incremental decode
(persistent KDA state + MLA-KV) to kill the remaining O(n²) recompute (more speed still). Options for
(c) mirror-first: regen a Triton-AOT cubin from FLA's KDA kernels for sm_121a (`scripts/regen-triton-
aot.sh`), or a native `chunk_kda` port. Row STAYS `ACTIVE`.

---

## 16. DEVICE NoPE-MLA attention lever MEASURED-NEGATIVE; STRICT still owed the ACTUAL FA2/chunk_kda kernels (2026-08-07, `row/KIMI-STRICT-CLOSE`, #107)

The §15 residual (d) — "the 7 NoPE-MLA layers still use a host f64 softmax island … closing p7
needs paged `mla::ForwardMlaAttentionBlock`" — was attempted in its device-COMPUTE form (the §15
device-KDA pattern applied to the MLA half) and MEASURED-NEGATIVE on GB10. The one-brick STRICT-close
did NOT land; the honest verdict re-confirms §14's razor.

**Implementation (`kimi_linear_device.cpp`, additive, default OFF).** New knob `VT_KIMI_DEVICE_MLA` +
helper `MlaAttnCoreDevice`: the NoPE causal softmax over per-head `[k_nope|k_pe(shared)]`/`v` runs
through the shared device op `vt::Attention` (f32 online max-subtracted softmax — vLLM's FA2
accumulation regime) instead of the f64 host `MlaSoftmaxIsland`. `vt::Attention` carries a single
head-dim for q/k/v while MLA is asymmetric (`qk = qk_nope+qk_rope = 192`, `v = 128`), so the value is
PADDED to `qk` with zeros — the weighted sum over the zero tail is 0, so `out[:, :, :v]` is byte-exact
to the unpadded math (softmax weights depend only on `q·k`). q views `dq` directly as `[T,nah,192]`;
key is built per `(t,h)` as `[k_nope | k_pe(broadcast)]`. Wired into both the f32 and bf16
`MlaSoftmaxIsland` paths. MLA dims VERIFIED from the real 48.9B `config.json` (not the K3 numbers):
`nah=32, qk_nope=128, qk_rope=64, v_head_dim=128, kv_lora=512, q_lora=None`; 7 full-attn/MLA layers
(`full_attn_layers=[4,8,12,16,20,24,27]`), 20 KDA.

**Unit gate (RED-first, CPU) GREEN.** `test_kimi_linear_forward` **14/14·825** (was 13/13·656) —
NEW case (g2) `KimiMlaAttnCoreDevice` (pad-V + `vt::Attention`) == a from-first-principles f64
causal-softmax reference at the Kimi MLA geometry (rtol 3e-3). RED-first verified: a perturbed scale
fails 108 assertions. Env-gated whole-forward runs green (`VT_KIMI_DEVICE_MLA=1` alone and with
`VT_KIMI_DEVICE_KDA=1`, 14/14·825). Same on the GB10 CUDA binary (210 GDN + 23 KDA syms linked).

**Full 48.9B GB10 gate — MEASURED NEGATIVE (single-load per config, `flock $HOME/gpu.lock`, min-avail
21 GiB, no reboot; the golden is the §12 STRICT `greedy_ids.npy`).**

| Config | env (all `VT_KIMI_DEVICE_COMPUTE=1`) | /128 | tok/s | verdict |
|---|---|---|---|---|
| control (device-KDA) | `DEVICE_KDA=1` | 122 | 4.24 | reproduces §15 EXACTLY (p0-p6 16/16, p7 10/16) |
| **+ device-MLA** | `DEVICE_KDA=1 DEVICE_MLA=1` | **109** | **3.89** | **REGRESSION both axes** |

**Why negative (the §14 razor, re-proven).** device-KDA WORKS (106→122) because the recurrence is the
SAME algorithm as vLLM's decode kernel, just f32-on-bf16 — it matches. But vLLM's MLA prefill uses
**FA2** (a specific flash tiling/reduction order); `vt::Attention`'s plain f32 online-softmax is the
right MATH but a DIFFERENT reduction ORDER, so — exactly like §14's host-precision-matching plateau —
it COIN-FLIPS near-ties: it BREAKS p3 16/16→3/16 (into the same `163586×` degenerate repeat the §14
bf16 knobs caused) while p7 stays diverged at 10/16. And it is SLOWER (4.24→3.89): the per-`(t,h)`
key/value build copies + the 192-dim pad-V waste add overhead to the O(n²) recompute path. An
approximation of vLLM's kernel is not enough — only the ACTUAL kernel matches.

**Verdict + default.** `VT_KIMI_DEVICE_MLA` STAYS **OFF**, kept as a documented-MEASURED-NEGATIVE A/B
knob (parity-lever precedent: §14's `ISLAND_F32ACC`/output-bf16). device-KDA (122/128, 4.24 tok/s)
remains the best config, itself default OFF (122 ≠ STRICT). Row STAYS `ACTIVE`.

**STRICT residual, sharpened by this measurement.** p7 (and now the coin-flip class generally) needs
vLLM's ACTUAL kernels, NOT a device approximation: (c) the **chunk_kda** prefill kernel family
(`chunk_kda_scaled_dot_kkt` + `recompute_w_u` + `chunk_gla_fwd_o_gk` + `fused_kda_gate_chunk_cumsum`,
FLA `ops/kda.py`) — mirror-first via a Triton-AOT regen for sm_121a (`scripts/regen-triton-aot.sh` +
new `triton_kernels/*.py`), the spec's named prime suspect; (d) the paged
`mla::ForwardMlaAttentionBlock` (FA2) for the 7 NoPE-MLA layers — NOT the `vt::Attention` approximation
tried here; (e) **paged-incremental decode** — coupled with (d) because it needs a decode/paged
attention op (`query_len ≠ key_len`), which `vt::Attention` cannot express; it kills the O(n²)
full-recompute (the current 4.24 tok/s is the recompute rate). Each is a substantial multi-kernel
brick, not a one-shot; recorded as the named follow-on.

---

## 17. chunk_kda PREFILL AOT PORT — kernel set + pinned-config record + regen recipe (Phase-1 spike, 2026-08-07, `row/KIMI-CHUNK-KDA-AOT`)

The §15/§16 STRICT residual (c) — "vLLM processes the PROMPT with the CHUNKED `chunk_kda`
kernel, we still run the RECURRENT form; a different reduction order coin-flips the p7
near-tie" — is here scoped, grounded, and DE-RISKED to the point of mechanical execution.
This section is the **AOT regen recipe + pinned-config record** the mission asks for.
The authored Triton harness bodies are STAGED in
[`.agents/specs/kda-chunk-aot/`](kda-chunk-aot/) (CI-safe: the drift check globs
`triton_kernels/*.py` non-recursively, so a staged sibling directory does not gate).
**Phase-2** moves them into `triton_kernels/`, adds the declarations below to
`cmake/TritonAOTKernels.cmake`, regenerates the sm_121a cubins
(`scripts/regen-triton-aot.sh`), wires the `vt::KdaChunkPrefill` op, and runs the gates.

### 17.1 The EXACT forward-only kernel set (`chunk_kda_with_fused_gate` → `_fwd`, file:line @ 555967922)
The prefill driver is `kimi_gdn_linear_attn.py:141` → `chunk_kda_with_fused_gate`
(`kda.py:1492`) → `chunk_kda_with_fused_gate_fwd` (`:1416`) →
`_chunk_kda_fwd_with_cumulative_g` (`:1306`). The kernel chain, in launch order:

| # | Step | FLA kernel(s) `kda.py:line` | New/Reuse |
|---|---|---|---|
| 1 | fused decay-gate + chunk-local cumsum·RCP_LN2 | `kda_gate_cumsum_fwd_kernel` `:1182-1254` | **NEW** |
| 2 | per-channel-gated K·Kᵀ + q·kᵀ (A, Aqk), inter | `chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter` `:521-618` | **NEW** |
| 3 | …same, intra | `chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra` `:627-715` | **NEW** |
| 4 | invert the strictly-lower-tri A (WY solve) | `solve_tril` / `merge_16x16_to_64x64_inverse_kernel` | **REUSE `gdn_tril_h32`** |
| 5 | recompute W, U (+ kg) per-K-channel | `recompute_w_u_fwd_kernel` `:817-957` | **NEW** (KDA per-channel, ≠ GDN `wy_fast.py`) |
| 6 | chunked hidden-state scan (h, v_new, final) | `chunk_gated_delta_rule_fwd_h` (`chunk_delta_h.py`, imported `:19`) | **REUSE `chunk_delta_h.py`, NEW pin** |
| 7 | GLA-style output with per-K gk decay | `chunk_gla_fwd_kernel_o` `:1019-1123` | **NEW** |

So **5 genuinely-new Triton kernels** (steps 1,2,3,5,7) + **1 new PIN of an existing .py**
(step 6: `chunk_delta_h.py` recompiled with `USE_GK=1, USE_EXP2=1, USE_G=0`, ≠ the GDN
`gdn_deltah` pin `USE_G=1, USE_GK=0, USE_EXP2=0`) + **1 pure reuse** (step 4:
`gdn_tril_h32`, byte-identical signature). The decode path is UNCHANGED — it stays the
#104 recurrent `vt::KdaGatedDeltaRule` (mirroring vLLM's own prefill=chunk / decode=recurrent
split). Backward kernels are NOT owed (forward-only inference).

### 17.2 Pinned-config record (Kimi KDA shapes: H=32, Hg=32, K=V=128, BT=64, BC=16, NC=4)
Autotune metaparams cannot be expressed in AOT, so each is PINNED. `num_warps`/`num_stages`
are **correctness-invariant** (they change tiling/pipelining, not the numeric result), pinned
mirroring the GDN precedent; the shape pins (BK/BV/BD/BC/NC) follow FLA's driver-fixed values
and heuristic lists. Dtypes MIRROR FLA's exact buffer choices (bf16 activations/intermediates;
fp32 for gk-cumulative / A / Aqk / recurrent-state) — **Phase-2 confirms each against the
`vt::KdaChunkPrefill` buffer contract before regen**.

| base | staged .py | kernel | BK | BV | BD | warps | stages | grid |
|---|---|---|---|---|---|---|---|---|
| `kda_gate_cumsum` | kda_gate_cumsum.py | `kda_gate_cumsum_fwd_kernel` | — | — | 64 | 4 | 2 | `2,NT,32` |
| `kda_kkt_inter` | chunk_kda_kkt.py | `…intra_sub_inter` | 64 | — | — | 4 | 3 | `NT,16,32` |
| `kda_kkt_intra` | chunk_kda_kkt.py | `…intra_sub_intra` | 128 | — | — | 4 | 2 | `NT,4,32` |
| `kda_wu` | recompute_w_u_kda.py | `recompute_w_u_fwd_kernel` | 64 | 64 | — | 4 | 3 | `NT,32,1` |
| `kda_deltah_h32` | chunk_delta_h.py (reuse) | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | — | 64 | — | 4 | 3 | `2,NH,1` |
| `kda_gla_o` | chunk_gla_o.py | `chunk_gla_fwd_kernel_o` | 64 | 64 | — | 4 | 3 | `NT,2,32`† |

†`kda_gla_o` grid is `(cdiv(V,BV), NT, H)` = `(2, NT, 32)`; expressed as `2,NT,32` with `NT`
the trailing carrier. **Scalar-constant pins baked as literals** (Triton AOT mis-packs fp32
scalars — see `chunk_o.py` note 3): `scale = K**-0.5` (kkt ×2, gla_o); softplus `beta=1.0`,
`threshold=20.0`, `cumsum_scale=RCP_LN2` (gate_cumsum); `DOT_PRECISION="ieee"` (wu).

### 17.3 The regen recipe — `cmake/TritonAOTKernels.cmake` declarations to ADD (Phase-2)
Signatures use the vendored `*dtype:align` / scalar / constexpr form. `NT`/`NH` are the
trailing grid carriers. Insert inside `vllm_triton_aot_declare_all()` after the GDN WY block:
```
# KDA chunk-prefill family (Kimi-Linear; H=32). Mirrors the GDN WY pins.
_vllm_triton_aot_declare(kda_gate_cumsum kda_gate_cumsum.py kda_gate_cumsum_fwd_kernel 4 2
  "2,NT,32"
  "*bf16:16, *fp32:16, *fp32:16, *fp32:16, *i32:16, *i32:16, i32, i32, 32, 128, 64, 64, 1, 1")
_vllm_triton_aot_declare(kda_kkt_inter chunk_kda_kkt.py chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter 4 3
  "NT,16,32"
  "*bf16:16, *bf16:16, *fp32:16, *bf16:16, *fp32:16, *fp32:16, *i32:16, *i32:16, i32, i32, 32, 128, 64, 16, 64, 4, 1")
_vllm_triton_aot_declare(kda_kkt_intra chunk_kda_kkt.py chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra 4 2
  "NT,4,32"
  "*bf16:16, *bf16:16, *fp32:16, *bf16:16, *fp32:16, *fp32:16, *i32:16, *i32:16, i32, i32, 32, 128, 64, 16, 128, 1")
_vllm_triton_aot_declare(kda_wu recompute_w_u_kda.py recompute_w_u_fwd_kernel 4 3
  "NT,32,1"
  "*bf16:16, *bf16:16, *bf16:16, *bf16:16, *bf16:16, *bf16:16, *bf16:16, *bf16:16, *bf16:16, *fp32:16, *i32:16, *i32:16, i32, i32, 32, 128, 128, 64, 64, 64, 0, 1, 1")
_vllm_triton_aot_declare(kda_deltah_h32 chunk_delta_h.py chunk_gated_delta_rule_fwd_kernel_h_blockdim64 4 3
  "2,NH,1"
  "*bf16:16, *bf16:16, *bf16:16, *bf16:16, *fp32, *fp32:16, *bf16:16, *fp32:16, *fp32:16, *i32:16, *i32:16, i32, i32, 32, 32, 128, 128, 64, 64, 0, 1, 1, 1, 1, 1, 1")
_vllm_triton_aot_declare(kda_gla_o chunk_gla_o.py chunk_gla_fwd_kernel_o 4 3
  "2,NT,32"
  "*bf16:16, *bf16:16, *fp32:16, *bf16:16, *bf16:16, *fp32:16, *i32:16, *i32:16, i32, i32, 32, 128, 128, 64, 64, 64, 1")
```
`kda_deltah_h32` note: the arg order is `k,v,w,v_new,g,gk,h,h0,ht,cu_seqlens,chunk_offsets,
T,NH,H,Hg,K,V,BT,BV,USE_G,USE_GK,USE_INITIAL_STATE,STORE_FINAL_STATE,SAVE_NEW_VALUE,
IS_VARLEN,USE_EXP2` — vs the GDN `gdn_deltah` the alignment marker MOVES from `g`(now dead,
`*fp32`) to `gk`(now used, `*fp32:16`), `Hg` flips 16→32 (KDA has no delta-rule GQA), and the
flag triple flips to `USE_G=0,USE_GK=1,…,USE_EXP2=1`. Steps 4 (`gdn_tril_h32`) and 6
(`chunk_delta_h.py`) need NO new .py file — reuse the vendored source.

### 17.4 The `vt::KdaChunkPrefill` op design (Phase-2 wiring, mirrors `cuda_gdn.cu` GdnPrefill)
A new additive op routing PROMPT-length KDA (`query_len == key_len`, the prefill step) through
the six cubins; decode (`query_len==1`) stays `vt::KdaGatedDeltaRule` (#104). Orchestration
(exactly `_chunk_kda_fwd_with_cumulative_g`): allocate per-`(T,H,·)` scratch —
`g_cum[T,H,128] f32`, `A/Aqk[T,H,64] f32`, `A_inv[T,H,64] bf16`, `w[T,H,128] bf16`,
`u[T,H,128] bf16`, `kg[T,H,128] bf16`, `h[NT,H,128,128] bf16`, `v_new[T,H,128] bf16`, then
launch (1) gate_cumsum → g_cum; (2)+(3) kkt → A,Aqk; (4) `gdn_tril_h32_default` → A_inv;
(5) wu(A_inv,gk=g_cum) → w,u,kg; (6) `kda_deltah_h32_default`(k=kg,w,u,gk=g_cum,h0=zeros,
ht=state) → h,v_new,final_state; (7) gla_o(q,v_new,g=g_cum,A=Aqk,h) → out. `cu_seqlens=[0,T]`,
`chunk_indices`/`chunk_offsets` from `prepare_chunk_indices`. Loader/launcher per the vendored
`std::call_once(load_gdn_*)` + `*_default(stream, …)` pattern (`cuda_gdn.cu:4594-4740`).
Dispatch guard: fire only when q/k/v are the pinned Kimi KDA geometry (H=32,K=V=128) and
`VLLM_CPP_TRITON`; else fall back to the recurrent island. The prefill/decode SPLIT mirrors
vLLM's `kimi_gdn_linear_attn.py:233-268` (decode `fused_recurrent_kda`, prefill
`chunk_kda_with_fused_gate`).

### 17.5 Gate plan (Phase-2, RED-first)
1. **Unit** (`tests/vt/test_ops_kda_chunk_prefill.cpp`): (a) the chunk op == the recurrent
   `vt::KdaGatedDeltaRule` (#104) to the documented chunked-vs-recurrent reduction-order delta
   (NOT bit-exact — different order; assert the p-th token argmax stable, or an rtol band); (b)
   vs the #173 host refs (`kimi_kda.cpp` `KdaDecayGateChunkCumsum` etc.) per intermediate; (c)
   if feasible, an **FLA-python golden** captured on the oracle venv at Kimi shapes as the direct
   oracle (the chunked result should match FLA's own `chunk_kda` output). RED-first: perturb a
   pin/scale, see it fail.
2. **Full 48.9B GB10 correctness gate** (re-park worker FIRST): 128 vs the §12 STRICT golden with
   `VT_KIMI_DEVICE_KDA=1` **+ chunk-prefill ON**. Target **STRICT** (close the p7 prefill-order
   near-tie).
3. **Speed ladder — ours-vs-vLLM at MATCHED config (USER 2026-08-07: the success bar is MEET
   vLLM SPEED; the §14/§16/#107 "HW-forced-indirect" framing is SUPERSEDED).** vLLM demonstrably
   RUNS Kimi-Linear-48B on ONE GB10 — the §12 golden capture used it at `gpu_memory_utilization
   =0.82`, single-seq, eager. So measure both arms at that EXACT recipe on the SAME prompts:
   (a) OUR arm — steady decode tok/s + prefill TTFT with `DEVICE_KDA=1` + chunk-prefill; (b) the
   **vLLM arm at the §12 launch config** (single-seq, eager, util 0.82) — steady decode tok/s +
   prefill TTFT. Report the ladder as a MEASURED ours/vLLM ratio, not an indirect statement.
   **OOM-REBOOT PROTOCOL (util 0.82 with 91.5 GiB weights is the tightest vLLM config ever run on
   this box — treat as reboot-risk):** run the vLLM arm SEQUENTIAL after our runs; `local-ai-worker`
   PARKED; `sudo drop_caches` before wall-clock; **PRE-WARM FlashInfer's autotune in a throwaway
   start at TINY util FIRST** (cold autotune at high util is a recorded OOM-reboot trigger); memory
   monitor MANDATORY; ONE attempt — if it OOMs, record the attempt honestly and do NOT retry at
   higher risk. `flock` both GPU locks; single-load steady-state.
4. On STRICT **and** ≥ vLLM speed at matched config: default flips per parity-enablers with proofs;
   model-matrix row moves. Below vLLM on any axis = an open gap (a MEASURED distance-to-bar), not done.

### 17.6 Status
**Phase-1 (this spike) DONE**: kernel set enumerated + classified; 5 harness bodies authored
(verbatim FLA ports, AOT-adapted) + staged; pinned-config record + regen recipe committed; the
`vt::KdaChunkPrefill` op + gate plan specified. **NOT YET**: regen (Phase-2 — coupled to the
op's confirmed buffer dtypes), the C++ op, and the numeric gates. Row STAYS `ACTIVE`. No STRICT
verdict is claimed here.

---

## Structured contract (machine-readable — mirrors deepseek-v4-flash.md)

## Scope
Bring up `KimiLinearForCausalLM` (Kimi-Linear-48B-A3B, `MODEL-TEXT-kimi-linear-kimi-
linear-for-causal-lm`) — a hybrid **20 KDA + 7 NoPE-MLA** decoder with a DeepSeek-
style **256-expert sigmoid `noaux_tc` MoE** (top-8, 1 shared, `first_k_dense_replace
=1`). It FITS one GB10 (91.5 GiB / 0.77x pool) and the pinned oracle serves it, so it
earns a REAL e2e SACRED token gate. In scope through W6: registry+config (W1), loader
name-map (W2), the CPU reference forward (W2-W6), and the born-on-the-runner
device-resident-logits SEAM (W6). Out of scope (named residuals): the DBuf-resident
device COMPUTE (§9, GPU-verify-pending W7), the e2e SACRED golden (W7, §8 recipe),
speed, MXFP4 (K3-only), and MTP (`num_nextn_predict_layers=0` in this checkpoint).

## Upstream chain
Pinned vLLM `555967922` (0.26.0.dev0): `registry.py:140` -> `kimi_linear`,
`KimiLinearForCausalLM`; `transformers_utils/configs/kimi_linear.py` (config +
`is_kda_layer`); `models/kimi_linear.py` (`KimiDecoderLayer`, `KimiMLAAttention`
NoPE, `KimiMoE` sigmoid-`noaux_tc`, load_weights); `models/layers/mamba/gdn/
kimi_gdn_linear_attn.py` (KDA modules + forward); `third_party/flash_linear_
attention/ops/kda.py` (the KDA gate + gated output, delta-rule shared with GDN);
`layers/mla.py`, `layers/fused_moe/*`, `layers/mamba/mamba_utils.py` (kda_state
shape/dtype). Full `file:line` map in §1.

## Our baseline
HEAVY reuse of landed+gated primitives (§2): DeepSeek MLA (`mla_attention.{h,cpp}`,
`cuda_mla_*.cu`), the sigmoid/`noaux_tc` grouped MoE + shared expert
(`deepseek_v2.cpp RunMoeBlock`, `cuda_moe*.cu`, `MoeRouterTopK`), the GDN
state/conv/recurrence family (KDA's parent: `cuda_gdn.cu`, `gdn_attn.*`, AOT
cubins), the KDA host references (`kimi_kda.{h,cpp}`, `test_kimi_kda` 14/14), and the
Qwen3.6-35B GDN-hybrid-MoE model skeleton (`qwen3_5_moe.cpp`). The born-on-the-runner
DBuf/FusedChain/MergedGemm/mla::ForwardMlaAttentionBlock seams are all dual-
registered CPU+CUDA (only the bf16 grouped-MoE GEMM is CUDA-only).

## Port map
Ours (@ HEAD): `include/vllm/model_executor/models/kimi_linear.h`;
`src/vllm/model_executor/models/kimi_linear_registry.cpp` (REGISTER + het-KV spec),
`kimi_linear_weights.cpp` (`ParseKimiLinearParams`/`EnumerateKimiLinearTensors`/
loader + host materialization), `kimi_linear_forward.cpp` (the CPU reference forward:
`KimiKdaLayerForward`/`KimiNoPEMlaLayerForward`/`KimiMoeRoute`/`KimiMoeBlockForward`/
`KimiDenseMlpForward` + `Forward`), `kimi_linear.cpp` (`ForwardDevice` device-
resident SEAM + the W7 device-compute reuse-wiring plan). NET-NEW = the KDA device
kernel (host-ref-oracled), the NoPE-MLA branch, the hybrid schedule + het-KV wiring,
the loader name-map.

## Tests to port
`tests/vllm/models/test_kimi_linear_scaffold.cpp` (9/9, registry+config+name-map),
`test_kimi_linear_forward.cpp` (7/7·300: per-op KDA/NoPE-MLA/router gates + the whole
2-layer forward + greedy decode + the W6 `ForwardDevice` device-resident gate),
`test_kimi_kda.cpp` (14/14, the four net-new-vs-GDN numerics). From vLLM `tests/`:
`registry.py` `_HfExamplesInfo` + `test_initialization.py` (construct-only — no GPU);
no upstream text-correctness fixture exists, so the e2e oracle golden (§8) is the
correctness truth.

## Gates
Unit (CPU, landed): the three test files above, all green on a clean CPU build;
`check-runner-routing-consistency`/`check-fusion-consistency`/`check-model-checklist`/
`check-agent-record` rc=0. Pending (GPU): the e2e SACRED greedy golden vs the pinned
oracle on GB10 (STRICT expected for a 48.9B MoE; near-tie fallback ratified), then
the device-compute gate (device==host within bf16 near-tie) and speed (nsys both
sides, vLLM graphed denominator, match/beat every axis).

## Dependencies
W1 (registry/config) -> W2 (loader) -> W2-W6 (CPU reference) -> W6 (device SEAM,
landed). W7 device compute depends on the reused GDN/MLA/MoE device blocks (landed)
+ a free GB10 slot + ~10 GiB dgx disk reclaim. Shared claims: `CLAIM-MLA-DEEPSEEK`
(MLA half), `CLAIM-KDA-KERNEL` (KDA host refs) — coordinate before the W7 device
KDA/MLA kernels so nothing is implemented twice.

## Work breakdown
**DONE:** W0 spike -> W1 registry/config/loader -> W2-W6 CPU reference forward -> W6
device-resident-logits SEAM -> **W7 DBuf-resident device COMPUTE, CPU-gated (§10 —
the whole hybrid over pooled DBufs via the shared `vt::` ops; two documented
host-fallback islands: the KDA per-k-channel recurrence+gate and the NoPE-MLA softmax
core; `test_kimi_linear_forward` 12/12·614 device==W2 ref + greedy-identical; runner
opt-in `VT_KIMI_DEVICE_COMPUTE=1`).** **Residual (keeps the row out of DONE, all
GPU-verify / box-down):** (a) the GPU numerics of the device compute — bf16
activations (vLLM parity), the GDN Triton-AOT cubins, the paged het-KV, the
grouped-MoE slabs — token-exact vs the pinned oracle on a CUDA queue; (b) the two
host-fallback islands -> device ops (a KDA per-channel-decay recurrence + an
exp/softplus KDA-gate op; the paged `mla::ForwardMlaAttentionBlock` over W_UK/W_UV
absorption) (W7-speed); (c) the W0/W7 e2e SACRED golden on GB10 (§8 recipe);
(d) speed. Full W0-W7 table in §5; W7 detail in §10.

## Risks/decisions
- **Decision (correctness-first, box down):** W6 lands the runner SEAM + the full
  device-compute PLAN but NOT the device compute, because CPU cannot gate the GDN
  Triton-AOT cubins or the bf16 numerics against the oracle — authoring it now would
  present as "done" without evidence. Mirrors the DeepSeek-V4 device-forward cadence.
- **Decision:** `ForwardDevice` returns device-resident logits via a pooled `DBuf`
  (`on_device()` on CPU+CUDA), so the CPU gate exercises the exact born-on-the-runner
  contract the GPU will; no runner-routing allowlist entry is needed.
- **Risk (small):** the NoPE-MLA branch must skip RoPE explicitly (do not silently
  apply the decoupled DeepSeek RoPE to a NoPE model) — gated by the `mla_use_nope`
  assert in `ParseKimiLinearParams` + the NoPE unit reference.
- **Risk:** GB10 unified-memory OOM on the 91.5 GiB load — keep `gpu_memory_util`
  ~0.55-0.65 (§3), never 0.85.
