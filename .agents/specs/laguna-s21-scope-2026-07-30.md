# Laguna-S-2.1 (`LagunaForCausalLM` / `laguna`) — W0 bring-up scope

**Status:** W0 SCOPE (read-only research; no model download, no GPU, no code).
**Date:** 2026-07-30. **Branch:** `spike/laguna-s21-w0-scope` off `137c739f`.
**Vehicle:** `unsloth/Laguna-S-2.1-GGUF` UD-Q4_K_XL (~73.4 GB, 3 shards).
**Upstream family:** Poolside Laguna (S-2.1 = 118B total / ~8B active MoE, agentic-coding, native interleaved reasoning, 1M ctx).

> **HEADLINE (the extensibility payoff): Laguna is ~85–90 % reuse.** Every
> component it needs already exists in our tree from a prior model: the Q4_K
> keep-quant decode (DeepSeek-V4 GGUF path — the ANTICIPATED "biggest new
> kernel" is **already landed**), the dual-regime per-layer RoPE
> (plain-sliding + YaRN-full-attn, from **OLMo-3**), the interleaved
> global/sliding-window attention (from **Gemma-2/Gemma-3**), and the sigmoid
> `noaux_tc` + `e_score_correction_bias` + shared-expert + `routed_scaling`
> MoE router (from **DeepSeek-V2/GLM-4**). The genuinely NEW work is three
> small ops, not a kernel: a per-head **softplus attention output gate**, an
> **ungrouped** variant of the noaux router, and the Laguna GGUF name-map.

---

## 0. TL;DR verdicts

| Axis | Verdict |
|---|---|
| vLLM support | **YES — native `laguna.py` in vLLM registry.** Not proprietary-only. `LagunaForCausalLM` / `model_type=laguna`. MIRROR-vLLM applies: we port `vllm/model_executor/models/laguna.py`. |
| Oracle-gateability (our pin `555967922` / 0.26.0.dev0) | **GATEABLE (pending a light W1 run-check).** `laguna.py` landed early July (DFlash-drafter commit dated **2026-07-03**), well before our pin (advanced 2026-07-26), so the model file IS in the pinned tree. Config uses a NESTED `rope_parameters` dict per layer-type — `laguna.py` handles it explicitly (unlike the OLMo-3 `KeyError:'rope_theta'` abort), so it should CONSTRUCT+RUN. |
| Oracle for OUR Q4_K GGUF | **Dual-oracle (ds4/antirez pattern).** vLLM does NOT run this GGUF (no laguna-GGUF path); it runs `poolside/Laguna-S-2.1-NVFP4` or `-FP8` (fits GB10 119 GiB; **BF16 235 GiB does NOT fit**). For a token-exact greedy gate on the SAME Q4_K bytes, the natural reference is **llama.cpp (Poolside fork, branch `laguna`)** — the toolchain that PRODUCED this GGUF. |
| Q4_K decode (feared "big new kernel") | **ALREADY DONE.** `DotQ4K` / `DotSuperblock<kQ4_K>` in `cuda_quant_dot.cu` + CPU `VecDotQ4_K`; Q5_K/Q6_K/Q8_0 too. UD dynamic mix is entirely covered. |
| Biggest genuinely-NEW piece | Per-head **softplus attention output gate** (`g_proj: hidden→num_heads`, softplus in fp32, broadcast over head_dim). Small op; `Softplusf` already exists (gemma4_audio). |
| Riskiest bit | **Per-layer VARIABLE query-head count** (global layers 48 heads, sliding layers **72** heads — `num_attention_heads_per_layer`), on top of the dual RoPE + variable window. Runner already does per-layer heterogeneous KV head_dim (Gemma-4 G1b); this extends it to per-layer Q-head count. |

---

## 1. EXACT config + arch metadata

Source: `poolside/Laguna-S-2.1/config.json` (fetched 2026-07-30) +
`generation_config.json` + vLLM `laguna.py` + Poolside model card.

```json
{
  "architectures": ["LagunaForCausalLM"],
  "model_type": "laguna",
  "auto_map": {"AutoConfig":"configuration_laguna.LagunaConfig",
               "AutoModelForCausalLM":"modeling_laguna.LagunaForCausalLM"},
  "vocab_size": 100352,
  "hidden_size": 3072,
  "intermediate_size": 12288,                // dense MLP (layer 0 only)
  "num_hidden_layers": 48,
  "num_attention_heads": 48,                 // BASE (global layers)
  "num_key_value_heads": 8,                  // GQA
  "head_dim": 128,
  "max_position_embeddings": 1048576,        // 1M
  "rms_norm_eps": 1e-06,                      // RMSNorm throughout
  "attention_bias": false,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",

  "num_experts": 256,
  "num_experts_per_tok": 10,                  // top-10
  "moe_intermediate_size": 1024,
  "shared_expert_intermediate_size": 1024,    // 1 shared expert
  "norm_topk_prob": true,                     // renormalize
  "moe_routed_scaling_factor": 2.5,
  "moe_router_logit_softcapping": 0.0,        // no softcap
  "moe_apply_router_weight_on_input": false,
  "router_aux_loss_coef": 0.0,
  "decoder_sparse_step": 1,
  "mlp_only_layers": [0],                     // layer 0 = dense MLP; 1..47 = MoE

  "sliding_window": 512,
  "gating": "per-head",                       // attention OUTPUT gate
  "gating_types": [ per_head x48 ],
  "num_attention_heads_per_layer": [48,72,72,72, 48,72,72,72, ...],  // 1:3 pattern
  "layer_types": ["full_attention","sliding_attention"x3, ...x12 blocks],
  "mlp_layer_types": ["dense","sparse"x47],

  "rope_parameters": {
    "full_attention":     {"rope_type":"yarn","rope_theta":500000.0,"factor":128.0,
                           "original_max_position_embeddings":8192,
                           "beta_slow":1.0,"beta_fast":32.0,
                           "attention_factor":1.4852030263919618,
                           "partial_rotary_factor":0.5},   // rotary_dim = 64
    "sliding_attention":  {"rope_type":"default","rope_theta":10000.0,
                           "partial_rotary_factor":1.0}     // rotary_dim = 128
  },
  "bos_token_id": 2, "eos_token_id": [2,24], "pad_token_id": 9
}
```

`generation_config.json`: `temperature 1.0, top_p 1.0, top_k 20, do_sample true`,
`reasoning_parser "poolside_v1"`, `tool_call_parser "poolside_v1"`,
`enable_thinking true`, and a **DFlash** speculative config
(`method dflash`, `poolside/Laguna-S-2.1-DFlash`, `num_speculative_tokens 15`).
(For a greedy gate we override to temperature 0 / do_sample off, spec-off.)

### Derived structural facts
- **48 layers**: layer 0 dense MLP; layers 1–47 MoE (`decoder_sparse_step 1`).
- **Attention pattern 1:3 global:sliding** → **12 global** (layers 0,4,8,…,44)
  + **36 sliding-window-512**. Interleaved (Gemma-2/3 style), but with the extra
  twist below.
- **Per-layer variable Q heads**: global = **48** heads (Q dim 6144, GQA group 6);
  sliding = **72** heads (Q dim 9216, GQA group 9). KV heads 8, head_dim 128, both.
- **Dual RoPE by layer-type**: full-attn = **YaRN** (θ 500000, factor 128, partial
  rotary 0.5 → rotate 64 of 128 dims, attention_factor mscale 1.4852); sliding =
  **plain** RoPE (θ 10000, full 128-dim rotary). This is exactly OLMo-3's schema.
- **Per-head softplus attention output gate** (`gating:"per-head"`, all layers):
  `gate = softplus(g_proj(x).float()).type_as(attn); attn = (attn.view(H,d) * gate[...,None]).flatten()`.
- **MoE router = sigmoid `noaux_tc`**: `scoring_func="sigmoid"`, `use_grouped_topk=False`,
  `e_score_correction_bias` (aux-loss-free), `renormalize=norm_topk_prob(true)`,
  `routed_scaling_factor 2.5` applied to output, top-10/256, **+1 shared expert**
  run in parallel. **NOTE:** the marketing card says "softplus router"; the vLLM
  IMPLEMENTATION uses SIGMOID scoring — softplus lives only in the ATTENTION gate.
  We mirror vLLM.

---

## 2. vLLM support + oracle-gateability (MIRROR-vLLM)

**(a) In the registry?** **YES.** `vllm/model_executor/models/laguna.py` exists and
`LagunaForCausalLM` is registered (Poolside Laguna family: XS.2 / S-2.1 / M.1, with
`recipes.vllm.ai/poolside/Laguna-S-2.1` and a `docs.vllm.ai/.../models/laguna/` API
page). Classes: `LagunaForCausalLM`, `LagunaModel`, `LagunaDecoderLayer`,
`LagunaAttention` (softplus output gate + per-layer sliding window via `layer_types`),
`LagunaMoE` (sigmoid `FusedMoE` + optional shared expert), `LagunaMLP` (dense,
`mlp_only_layers`). This is NOT a "proprietary, vLLM-unsupported" case — the earlier
premise is REVERSED.

**(b) Can our PINNED oracle RUN it?** **Expected YES; W1 confirms.**
- `laguna.py` predates our pin: the "Add Laguna XS.2.1 DFlash drafter support" commit
  is dated **2026-07-03**; base laguna support is older; our pin `555967922`
  (0.26.0.dev0, transformers 5.14.1) was advanced **2026-07-26**. So the file is IN
  the pinned tree.
- The OLMo-3 gate failure was a NESTED `rope_parameters` `KeyError:'rope_theta'` in an
  older transformers. Laguna's `laguna.py` explicitly walks the nested dict
  (`base_rope = top_rope.get(layer_type) or top_rope.get("full_attention")`), and our
  pin ships transformers 5.14.1 (newer than the 5.13.1 that blocked OLMo-3), so the
  nested-rope hazard is handled. **Residual gateability risk (W1):** `auto_map` points
  at remote code (`configuration_laguna`/`modeling_laguna`); native `laguna.py` should
  win, but a `VLLM_TRUST_REMOTE_CODE=1` may be needed for registry inspection. Verify
  with a LIGHT `AutoConfig`+registry check (no weights), per the config-constructs !=
  model-runs rule, then a small NVFP4/FP8 greedy run on GB10.

**(c) Oracle for a token-exact gate on OUR Q4_K GGUF — dual-oracle (ds4 pattern):**
- vLLM does not have a laguna-GGUF loader path (GGUF + this custom MoE arch is not
  wired), so vLLM cannot gate the Q4_K bytes directly. vLLM runs the
  **`poolside/Laguna-S-2.1-NVFP4`** (or `-FP8` / `-INT4`) checkpoint — these fit the
  GB10 119 GiB unified pool; **BF16 (235 GiB, 5 shards) does NOT fit** a single Spark.
- The clean **token-exact** reference for our keep-quant path is **llama.cpp, Poolside
  fork, branch `laguna`** (carries full Laguna incl. DFlash; produced this unsloth
  GGUF). Same bytes → strict greedy gate is meaningful, exactly like ds4/antirez was
  for DeepSeek-V4.
- Plan: **behavior/structural mirror = vLLM `laguna.py`** (+ a coherence golden on
  NVFP4/FP8); **token-exact same-quant gate = llama.cpp on the identical UD-Q4_K
  GGUF.** Where the quants differ (NVFP4 vs Q4_K), use the near-tie distributional
  gate; the strict bar is against llama.cpp on the shared quant.

---

## 3. llama.cpp arch reference (structural authority for the GGUF forward)

- **Poolside fork of llama.cpp, branch `laguna`** — full Laguna support incl. DFlash
  speculative decode; this is the reference that WROTE the GGUF. It implements the
  `laguna` arch: `LLM_ARCH_LAGUNA` (or equivalent) in `llama-arch.cpp` (name-map +
  KV keys), the graph builder in `llama-model.cpp` (`build_laguna`), and the Q4_K
  vec_dot is the standard `ggml-quants.c` / `vecdotq.cuh` `vec_dot_q4_K_q8_1`.
- **Use it for:** (i) the exact GGUF **tensor name-map** (`blk.N.attn_*`,
  `blk.N.ffn_gate_exps/up_exps/down_exps`, the shared-expert tensors, the attention
  **gate** tensor name, the per-layer q/k/v shapes with variable head count), (ii) the
  GGUF **metadata keys** (`laguna.attention.sliding_window`, the per-layer rope scaling
  keys, `laguna.expert_count`/`expert_used_count`/`expert_shared_count`, the
  attention-gate flag), (iii) confirmation of the **softplus gate** + **sigmoid router**
  placement, (iv) the mainline llama.cpp note that Laguna is "**BF16 and Q4_K_M only**"
  — i.e. Q4_K is the sanctioned quant, which is exactly the one we already decode.
- (Mainline llama.cpp may not yet carry `laguna`; the Poolside fork is the source of
  truth. Cross-check the community `RayCodes_Laguna_2.1` test suite for setup parity.)

---

## 4. REUSE MAP vs our codebase (component-by-component)

Legend: **R** = reuse as-is / with config · **r** = reuse + small extension ·
**N** = new (but small).

| Component | Verdict | Reuse source (file) | Delta |
|---|---|---|---|
| GGUF keep-quant load + name-map | **r** | DeepSeek-V4 GGUF path: `gguf_keep_quant.cpp`, `gguf_reader.cpp`, `qwen3_5_gguf_weights.cpp`, `kMatmulBTQuantGrouped` (`cuda_quant_dot.cu`) | NEW: Laguna tensor **name-map** (from llama.cpp fork). ~90 % reuse. |
| **Q4_K decode (feared big kernel)** | **R** | `cuda_quant_dot.cu` `DotQ4K`/`DotSuperblock<kQ4_K>` + CPU `VecDotQ4_K` (`cpu_quant_dot.cpp`, `cpu_quant_blocks.h BlockQ4_K`) — landed for ds4 keep-quant, on CPU **and** CUDA (`KERNEL-QUANT-CIQ-GEMM-CUDA`) | **ZERO new.** Q4_K + Q5_K/Q6_K/Q8_0 all present; UD-Q4_K_XL mix fully covered (see §5). |
| Interleaved global + sliding-window(512) attn | **R** | `gemma2.cpp`/`gemma3.cpp` `is_sliding` per-layer from `layer_types` + `sliding_window`; runner local-mask kernel (`Long-context RoPE + sliding-window` STATUS row, GB10-gated) | Config-drive `layer_types`; window 512. ~95 % reuse. |
| Dual per-layer RoPE (YaRN full-attn / plain sliding) | **R** | **OLMo-3** `olmo2_weights.cpp BuildOlmo3YarnCache` (get_rope yarn) + gemma3 dual-rope-theta selection; YaRN(beta_fast/slow/mscale) in the shared scaled-RoPE (`Long-context RoPE` row) | Build TWO caches (YaRN-64dim full, plain-128dim sliding), select by `layer_types`. ~90 % reuse. |
| Partial rotary (full-attn rotary_dim 64) | **R** | `phi_weights.cpp`/`granite.cpp` partial-rope `rotary_dim = int(head_dim*factor)`; GLM-4 partial interleaved rope | Combine partial(0.5) + YaRN on the full-attn cache (both pieces exist). |
| GQA attention (8 KV, variable Q heads) | **r** | shared dense attention + FA2 paged/decode; runner heterogeneous per-layer KV head_dim (**Gemma-4 G1b**, task #148) | NEW: extend runner to per-layer variable **Q-head count** (48 global / 72 sliding). Small metadata wiring. |
| MoE: sigmoid `noaux_tc` + `e_score_correction_bias` + shared expert + `routed_scaling` | **r** | **DeepSeek-V2** `deepseek_v2_weights.cpp` (`noaux_tc`, `e_score_correction_bias`, sigmoid) + **GLM-4-MoE-lite** `noaux_tc` grouped router; shared `MoeGateUpSwiGLU`/`MergedGemmGroup` grouped-expert GEMM; DeepSeek-V4 shared-expert parallel path | NEW: **ungrouped** variant (`use_grouped_topk=False` — DROP the group step, simpler than ds2's grouped path); `apply_routed_scale_to_output=true`. ~90 % reuse. |
| Per-head **softplus attention OUTPUT gate** | **N (small)** | `Softplusf` (gemma4_audio.cpp:20); DeepSeek-V4 per-head q-gating is the closest cousin (per-head scalar broadcast) | NEW op: `g_proj: hidden→num_heads`, `softplus` fp32, broadcast × attn over head_dim. ~30-line host op + a fused-epilogue later. This is the ONE genuinely-Laguna primitive. |
| RMSNorm / SwiGLU dense MLP (layer 0) / embed / lm_head (untied) | **R** | shared `vt::RmsNorm`, `MlpGateUpMethodBase` SwiGLU seam, `FusedChain(kFusedAddRmsNorm)` | Config-drive only. |
| Reasoning + tool parser `poolside_v1` | **N (small)** | reasoning-parser + tool-parser frameworks (LANDED, tasks #171/#100) | NEW `poolside_v1` parser config (interleaved thinking). Serving-layer, not forward. |
| DFlash speculative decode | **R (optional)** | **SPEC-DFLASH** already implemented e2e (D0–D12, `dflash-correctness-done` memory) for Qwen | Laguna ships a DFlash drafter (`poolside/Laguna-S-2.1-DFlash`, 15 tokens). Reuse the existing DFlash machinery for a later speed lane; NOT needed for the greedy correctness gate. |

**Quantified reuse: ~85–90 % reuse, ~10–15 % genuinely new — and the "new" is
small host ops (softplus attn gate, ungrouped-router flag, name-map,
variable-Q-head wiring, poolside_v1 parser), NOT a new compute kernel.** The
DeepSeek-V4-MoE + Gemma-sliding-attn + OLMo-3-dual-rope + our-GGUF-Q4_K "dream"
composition holds, and is BETTER than scoped: the feared Q4_K kernel is already
in-tree.

---

## 5. Quant gap — Q4_K decode: **NONE (already landed)**

The task scoped Q4_K as "the biggest new kernel." It is **already implemented**
for the DeepSeek-V4 GGUF keep-quant path, on BOTH CPU and CUDA:
- CPU: `cpu_quant_blocks.h` `BlockQ4_K` (144-byte super-block, 256 elems, 6-bit
  scales/mins) + `cpu_quant_dot.cpp` `VecDotQ4_K` (ported 1:1 from llama.cpp
  `ggml-quants.c` `vec_dot_q4_K_q8_K`).
- CUDA: `cuda_quant_dot.cu` `DotQ4K(BlockQ4_K*, BlockQ8_K*)` +
  `DotSuperblock<WType::kQ4_K>` in the `DotSuperblock<W>` template family
  (MMVQ-style dequant-in-kernel, `KERNEL-QUANT-CIQ-GEMM-CUDA`, GB10-gated for ds4).
- STATUS.md already advertises `Q4_K/Q5_K/Q6_K` GGUF support.

**UD-Q4_K_XL tensor-type mix (unsloth dynamic quant)** — to be CONFIRMED per-tensor
from the GGUF `general.*`/tensor headers at W2 (light metadata read, no full
download). Expected UD "XL" pattern (verify against the fork):
- Router `gate` / attention `gate` (small) → Q6_K or Q8_0 (accuracy-critical).
- `token_embd` / `output` (lm_head) → Q6_K/Q8_0.
- attention q/k/v/o → Q4_K (some Q6_K on the global layers).
- MoE `ffn_*_exps` (the bulk, 256 experts) → Q4_K, with `ffn_down_exps` often a
  notch higher (Q5_K/Q6_K) — the standard UD "leave the down-proj richer" rule.
- Every one of {Q4_K, Q5_K, Q6_K, Q8_0} is **already decoded** → no new kernel for
  ANY tensor in the mix. Only the per-tensor **name→type** map is new (W2, from the
  GGUF header + llama.cpp fork), which the ds4 name-map checker
  (`check-dsv4-gguf-namemap.py`) pattern already models.

---

## 6. Bring-up W-plan + risks + roadmap row

### W-plan
- **W0 (this):** scope spec + roadmap row + STATUS pointer. DONE.
- **W1 — oracle decision (GB10, light):** (i) LIGHT `AutoConfig`+registry inspect on
  the pin (no weights) to confirm `laguna.py` constructs the nested-rope config
  (config-constructs check); (ii) small greedy golden on `poolside/Laguna-S-2.1-NVFP4`
  (or `-FP8`) — proves the pinned oracle RUNS laguna (the gateability rule);
  (iii) stand up the llama.cpp Poolside-fork on the same UD-Q4_K GGUF for the
  token-exact reference. Deliver: golden token streams from both oracles + the
  gate methodology (strict vs llama.cpp same-quant; near-tie vs vLLM NVFP4).
- **W2 — GGUF loader + name-map + quant mix:** read the UD-Q4_K_XL headers (metadata
  only), build the Laguna name-map (from the fork), wire the keep-quant loader
  (reuse ds4 `gguf_keep_quant` + grouped-expert operands). No new decode kernel.
  Extend `check-dsv4-gguf-namemap.py`-style coverage to laguna.
- **W3 — forward (compose reuse + 3 new ops):** registry stub + `LagunaModel::Forward`
  composing: shared dense attention with (a) per-layer variable Q-head + GQA,
  (b) interleaved sliding-window(512) mask (gemma), (c) dual per-layer RoPE
  (OLMo-3 YaRN-full / plain-sliding), (d) **NEW** per-head softplus output gate;
  dense MLP at layer 0; MoE (DeepSeek-V2 sigmoid `noaux_tc` **ungrouped** + shared
  expert + `routed_scaling 2.5`) at layers 1–47; untied lm_head.
- **W4 — correctness gate (greedy):** strict token-exact vs **llama.cpp** on the
  identical Q4_K GGUF; near-tie distributional vs **vLLM NVFP4/FP8**. Bigger sibling
  cross-check if a deterministic vehicle exists. SACRED gate methodology per
  `near-tie-distributional-gate`.
- **W5 — speed:** every-axis vs the chosen oracle on GB10 (keep-quant decode reuses the
  ds4 CUDA MMVQ path); optional DFlash speculative lane (reuse SPEC-DFLASH) as a
  throughput lever.

### Hardest / riskiest bits (flagged)
1. **Per-layer variable Q-head count (48 vs 72)** — the runner does per-layer KV
   head_dim (Gemma-4) but not per-layer Q-head COUNT + variable GQA group (6 vs 9).
   Attention metadata + q_proj shape per layer. **Medium risk, mechanical.**
2. **Dual RoPE × partial-rotary × YaRN mscale on ONLY the full-attn 64 dims** — three
   reused pieces that must COMPOSE correctly (partial 0.5 + YaRN attention_factor
   1.4852). Numerics-delicate; gate the cos/sin cache against the fork bit-for-bit.
   **Medium risk.**
3. **Per-head softplus attention output gate** — the one new op; small but must match
   vLLM's `softplus(gate.float()).type_as` fp32 semantics + per-head broadcast
   exactly. **Low risk.**
4. **Ungrouped sigmoid-noaux router** — ds2/glm4 are GROUPED; must peel to the
   ungrouped path (simpler) and apply `routed_scaling` to OUTPUT + renorm. Gate the
   top-10 selection + tie-break against vLLM. **Low-medium risk (tie-break razor,
   as in Qwen3-dense).**
5. **ORACLE-BLOCKED risk (residual):** if the pin's transformers cannot construct
   `LagunaConfig` without remote code AND `VLLM_TRUST_REMOTE_CODE` is disallowed, the
   vLLM NVFP4 golden is blocked → fall back to **llama.cpp-only** gating (still a valid
   token-exact same-quant oracle, ds4 precedent). Config-constructs check at W1
   de-risks this before any GPU time.
6. **Memory:** UD-Q4_K_XL ~73.4 GiB fits the 119 GiB GB10 unified pool with room for
   KV; keep `gpu_memory_utilization` low (the unified-pool OOM-reboot hazard), never
   co-run a big vLLM NVFP4 oracle beside our engine.

### roadmap_v1 row (proposed, added to `.agents/roadmap_v1.md`)
`LAGUNA-S21` — `SPIKE` (W0 scoped 2026-07-30). `LagunaForCausalLM`/`laguna`, 118B/8B
MoE. vLLM-NATIVE (`laguna.py`, in pin → gateable). ~85–90 % reuse
(ds4-MoE + gemma-sliding + olmo3-dual-rope + our-Q4_K-GGUF); NEW = per-head softplus
attn gate + ungrouped-sigmoid-noaux router + laguna name-map + variable-Q-head
runner + `poolside_v1` parser. Dual-oracle: vLLM-NVFP4 behavior + llama.cpp-Q4_K
token-exact. Q4_K = ZERO new (ds4 keep-quant covers it).

---

## Sources
- `poolside/Laguna-S-2.1` config.json + generation_config.json (HF, 2026-07-30).
- `unsloth/Laguna-S-2.1-GGUF` model card + file tree (UD-Q4_K_XL 3 shards).
- vLLM `vllm/model_executor/models/laguna.py` (main) + `recipes.vllm.ai/poolside/*`
  + `docs.vllm.ai/.../models/laguna/`; file-history DFlash commit dated 2026-07-03.
- llama.cpp Poolside fork branch `laguna` (arch + Q4_K vec_dot reference).
- Our tree: `cuda_quant_dot.cu`, `cpu_quant_blocks.h`, `deepseek_v2_weights.cpp`,
  `deepseek_v4_moe.cpp`, `gemma2.cpp`/`gemma3.cpp`, `olmo2_weights.cpp`,
  `phi_weights.cpp`, `gemma4_audio.cpp`, `docs/STATUS.md` (Q4_K + sliding-window rows).
```
