# DeepSeek-V4-Flash — W0 SCOPE spike (records-only)

**Claim:** `CLAIM-DEEPSEEK-V4-SCOPE`. **Rows:**
`MODEL-TEXT-deepseek-v4-deepseek-v4-for-causal-lm` (`DeepseekV4ForCausalLM`, owned by
`CLAIM-GLM-DSA-LATEST-DEEPSEEK` — this spike CROSS-REFERENCES + CORRECTS its HW-fit),
`MODEL-SPEC-deepseek-v4-deep-seek-v4-mtp` (`DeepSeekV4MTPModel`, promoted INVENTORIED→SPIKE here).
**State:** SPIKE — CPU-only, records-only. NO build, NO GPU, NO download. The oracle-RUN gate
(W1) is a NAMED next brick, not run here (DGX busy).

**Base:** current `main` HEAD `c497668dd8a0f10123690754fbfa0009037b11b3`.
**Pinned oracle:** `${VLLM_SOURCE}` = `/home/mudler/_git/vllm` @
`5559679229bc961848b121ccdeaa8fa5d79bec98` (vLLM 0.26.0.dev0). V4 is present and NEW there:
it is NOT `vllm/model_executor/models/deepseek_v4.py` — it is a whole platform-split package
`vllm/models/deepseek_v4/{common,nvidia,amd,xpu}/` (~5.7k LoC across ~38 files).

**User driver:** "DeepSeek V4 Flash should be runnable on the DGX Spark." (Note: the "DSpark" in
the checkpoint name `DeepSeek-V4-Flash-DSpark` is the DSpark *speculator*, NOT "DGX Spark" — the
bundle ships the DSpark semi-autoregressive draft head.)

---

## 0. Scope (headline verdict)

DeepSeek-V4-Flash is **~167B, 256-routed-expert MoE, H=4096, L=43** and is **architecturally a
NEW model, not a V3 increment**. It composes:
- **DeepSeek Sparse Attention (DSA / "Lightning Indexer")** MLA — the V3.2 sparse-attention
  family (top-k token selection + compressor + fp8 paged KV + sliding-window), which we have
  NEVER implemented (our MLA is DENSE V2/V3);
- **Manifold / Markov Hyper-Connections (MHC)** — the residual stream is expanded to `hc_mult`
  parallel streams `[T, hc_mult, H]` with Sinkhorn-normalized mixing, implemented ONLY as
  TileLang kernels upstream with ZERO eager reference — a genuinely novel topology;
- **MegaMoE** — a DeepGEMM fused NVFP4 MoE that requires **SM100 (device major==10)** and so does
  NOT run on GB10 (sm_121/major 12); GB10 must use the fallback `FusedMoE` grouped-GEMM path;
- **sqrtsoftplus routing + hash-routed layers + clamped SwiGLU + attention sinks + grouped output
  LoRA** — several new MoE/attention primitives;
- **MTP** and **DSpark** speculative heads reusing the V4 decoder + MHC.

**HW-fit verdict for ONE GB10 (119 GiB unified):** the native `DeepSeek-V4-Flash` fp8-linear +
fp4/fp8-expert checkpoints are ~148–167 GiB and DO NOT fit. The **`nvidia/DeepSeek-V4-Flash-NVFP4`
(W4-everything, ~83 GiB)** DOES fit with context headroom AND matches our sm_12x NVFP4 fast path —
it is the recommended vehicle, IF disk is freed to ~90 GiB (currently ~68 GiB free). See §4.

**Effort verdict:** this is the largest single-model campaign in the matrix — it needs the DSA
indexer/compressor stack (V3.2 territory we deferred as DEP-blocked), the MHC topology (net-new,
no eager ref), and the sqrtsoftplus/hash/clamp MoE deltas, on top of a fresh 512-wide MLA
geometry. It is a MULTI-brick campaign, not a mechanical port. The decisive gate is W1 (prove the
pinned oracle RUNS the NVFP4 checkpoint within 119 GiB on GB10 + a greedy golden).

---

## 1. Upstream chain — V4 arch primitive-by-primitive (`file:line`)

All paths under `/home/mudler/_git/vllm/vllm/`.

### 1.1 Registry + config
- `model_executor/models/registry.py:95` `DeepseekV4ForCausalLM` → `vllm.models.deepseek_v4`.
- `registry.py:608` `DSparkDraftModel` → `DSparkDeepseekV4ForCausalLM`;
  `registry.py:630` `DeepSeekV4MTPModel` → `DeepSeekV4MTP`.
- `transformers_utils/configs/deepseek_v4.py:8` `DeepseekV4Config(model_type="deepseek_v4")`
  — a THIN config (only rope fields defaulted; everything else read off hf_config at runtime).
- `models/deepseek_v4/quant_config.py:29` `DeepseekV4FP8Config` (`get_name()="deepseek_v4_fp8"`)
  — linear/attention layers are ALWAYS FP8 block-quant; expert weights vary:
  `expert_dtype="fp4"` ⇒ MXFP4 experts (group 32, UE8M0 scales) OR NVFP4
  (`moe_quant_algo="NVFP4"`, group 16, `quant_config.py:178-186`); `expert_dtype="fp8"` ⇒ FP8
  block experts (Flash-Base). `is_scale_e8m0` keyed on expert_dtype (`:80`).

### 1.2 MLA attention — `models/deepseek_v4/attention.py`
`DeepseekV4Attention(nn.Module, AttentionLayerBase, ABC)` (`attention.py:122`). This is a
DEDICATED V4 attention (NOT the generic `MLAAttention`; it runs its own forward, `sparse_mla.py:65`
`get_impl_cls` raises).
- **MLA geometry:** `head_dim=512` = **448 NoPE + 64 RoPE** (`sparse_mla.py:76`), `q_lora_rank`,
  and a **grouped OUTPUT LoRA** `o_lora_rank` / `o_groups` — `wo_a` (`attention.py:243`,
  `is_bmm=True`, `bmm_batch_size=n_local_groups`) + `wo_b` (`:253`). V3 has NO output LoRA — this
  is new. Projections: fused `wq_a|wkv` (`fused_wqa_wkv`, `:224`, ReplicatedLinear disable_tp),
  `q_norm`+`kv_norm` RMSNorm, `wq_b` up-proj (`:233`).
- **Attention sinks:** `attn_sink` per-head `[padded_heads]` init `-inf` (`:219`).
- **Sliding-window KV:** `DeepseekV4SWACache` (`:315`), `window_size=config.sliding_window`.
- **fp8_ds_mla KV cache:** UE8M0 block-scaled fp8 packed as uint8, 576B-aligned paged
  (`_resolve_dsv4_kv_cache_dtype`, `:89`; `get_kv_cache_spec` `:626`). Also supports plain bf16 /
  per-tensor fp8 rows (FlashInfer). Fused CUDA insert ops:
  `torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` (`:575`) and bf16/fp8 variants.
- **DSA "Lightning Indexer"** `DeepseekV4Indexer` (`:689`): only on layers with `compress_ratio==4`
  (`:274`). Selects `index_topk` tokens per query via `index_n_heads=64`, `index_head_dim=128`,
  a `wq_b`/`weights_proj`, its own `DeepseekCompressor`, an MXFP4-or-fp8 `DeepseekV4IndexerCache`
  (`:648`), and `SparseAttnIndexer` (`sparse_attn_indexer.py`). `_fill_short_context_topk_indices`
  triton kernel (`:70`) selects all candidates when seq ≤ topk.
- **Compressor** `DeepseekCompressor` (`compressor.py`, ~469 LoC): the CSA/HCA compressor with a
  stateful fp32 recurrent `CompressorStateCache`, paged via its own `CompressorBackend`;
  per-layer `compress_ratios[layer_id]` (4 for indexed C4A layers, else 128), `compress_rope_theta`.
- **Multi-stream execution:** `attn_gemm_parallel_execute` (`:393`) + `attention_impl`
  (`@eager_break_during_capture`, `:453`) fan the compressor / indexer / q-up GEMMs across 3
  aux CUDA streams (TRT-LLM PR #14142 Level-1 overlap). This is a PERF structure, not numerics.
- **Backends (platform subclasses):** `DeepseekV4FlashMLAAttention` (`nvidia/flashmla.py`),
  `DeepseekV4FlashInferMLAAttention` + **`DeepseekV4FlashInferSM120Attention`**
  (`nvidia/flashinfer_sparse.py`). `_select_dsv4_attn_cls` (`nvidia/model.py:760`): **SM12x
  defaults to the FlashInfer sparse DSV4 SM120 class**; the backend
  `DeepseekV4FlashInferMLASparseBackend.supports_compute_capability` returns True for
  **`major in [10, 12]`** (`flashinfer_sparse.py:100`) ⇒ **the V4 sparse-MLA attention path IS
  supported on GB10 sm_121** (unlike V3.2, whose sm_120 sparse path was DEP-blocked in the prior
  MLA campaign — V4 ships a dedicated DSV4 sparse backend). RoPE: `build_deepseek_v4_rope`
  (`common/rope.py:9`) — deepseek_yarn (or llama-scaling), `mscale=0`/`mscale_all_dim=0`
  (YaRN mscale DISABLED), dual theta (`compress_rope_theta` for compressed layers), `rope_dim=64`,
  `is_neox_style=False`, `is_deepseek_v4=True`.

### 1.3 Manifold/Markov Hyper-Connections (MHC) — `models/deepseek_v4/nvidia/model.py`
The residual stream is `[T, hc_mult, H]` (`make_empty_intermediate_tensors:1066`,
`extract... hc_mult` streams). Each `DeepseekV4DecoderLayer` (`:794`) holds `hc_attn_fn`/`hc_ffn_fn`
mixing matrices (`(2+hc_mult)*hc_mult × hc_mult*H`), `hc_*_base`, `hc_*_scale`, and runs
**TileLang** kernels (`model_executor/kernels/mhc/tilelang`): `mhc_pre_tilelang` /
`mhc_pre_broadcast_tilelang` / `mhc_fused_post_pre_tilelang` / `mhc_post_tilelang` and the head
collapse `hc_head_fused_kernel_tilelang` (`:1137`). These **swallow the attn_norm/ffn_norm
RMSNorms** and apply **`hc_sinkhorn_iters` Sinkhorn normalization** (prior scope: 20 iters) with
`hc_eps`, `hc_post_alpha=2.0`. **There is NO eager reference and NO numerical test upstream** —
this is the single hardest correctness item to gate.

### 1.4 DeepSeek-V4 MoE — `models/deepseek_v4/nvidia/model.py:512` `DeepseekV4MoE`
- **256 routed experts** + optional **shared experts** (`DeepseekV4MLP`, `:82`, clamped SwiGLU
  `SiluAndMulWithClamp(swiglu_limit)`).
- **Router:** `GateLinear` (`:554`) with **`scoring_func="sqrtsoftplus"`** (NEW — not sigmoid/
  softmax), `e_score_correction_bias` (`noaux_tc` topk_method, `:579`), `fused_topk_bias`
  (`:704`), `routed_scaling_factor`, `norm_topk_prob` renormalize.
- **HASH-routed layers:** the first `num_hash_layers` layers replace the learned gate with a
  `tid2eid` token-id→expert hash table lookup (`:564-578`, `:696`) — NEW.
- **MegaMoE path** `DeepseekV4MegaMoEExperts` (`:161`): DeepGEMM `fp8_fp4_mega_moe`, uint8-packed
  NVFP4 w13/w2 + UE8M0 per-32 block scales, EP-only, `sqrtsoftplus`-only, fp4-only. **Requires
  `torch.cuda.get_device_capability()[0]==10` (SM100)** (`_check_runtime_supported:307-315`) ⇒
  **NOT usable on GB10** (major 12). Selected only when `kernel_config.moe_backend ==
  "deep_gemm_mega_moe"`; the DEFAULT path is `_init_fused_moe_experts` (`:647`) = the generic
  `FusedMoE` grouped GEMM (NVFP4/MXFP4/FP8 via the standard MoE methods) — **this is the GB10 path
  and the one that reuses our infrastructure.**

### 1.5 MTP + DSpark speculators
- `nvidia/mtp.py:72` `DeepSeekV4MultiTokenPredictorLayer` — separate `e_proj`/`h_proj` (fp8, NOT
  V3's fused `eh_proj`), `enorm`/`hnorm`, an `hc_head` collapse in `compute_logits`, a full
  `DeepseekV4DecoderLayer` `mtp_block` (so MTP inherits the whole MHC + DSA + MoE stack),
  `SharedHead`. V4-specific weight remap in `load_weights`.
- `nvidia/dspark.py:268` `DSparkDeepseekV4ForCausalLM` — semi-autoregressive block drafter:
  `hc_mult`-stream draft layers, `DSparkMarkovHead`, non-causal attention via the sparse indexer,
  `precompute_and_store_context_kv`. Reuses `DeepseekV4DecoderLayer`. (Left INVENTORIED this spike.)

### V4-vs-V3 deltas (what CHANGED)
| Axis | V3 / V3.2 (our `deepseek_v2.py` family) | V4-Flash |
|---|---|---|
| Residual | plain residual + RMSNorm | **MHC** `[T, hc_mult, H]` Sinkhorn streams (TileLang, no eager ref) |
| MLA latent | 512 kv_lora + 64 rope = 576 | **448 NoPE + 64 RoPE = 512** head_dim, **grouped output LoRA** (`o_groups`, bmm `wo_a`/`wo_b`), **attention sinks**, **SWA** |
| Attention | dense MLA (V3) / DSA indexer (V3.2, DEP-blocked on sm_120) | **DSA indexer + compressor**, but with a **dedicated DSV4 sparse backend that SUPPORTS sm_12x** (`major∈{10,12}`) |
| MoE router | sigmoid `noaux_tc` grouped-topk | **`sqrtsoftplus`** + `e_score_correction_bias` + **hash-routed** first N layers |
| MoE experts | bf16/fp8 grouped GEMM | NVFP4/MXFP4/FP8; **MegaMoE (SM100-only)** fast path + FusedMoE fallback |
| MLP | SwiGLU | **clamped** SwiGLU (`swiglu_limit`) |
| RoPE | YaRN (mscale on) | deepseek_yarn **mscale DISABLED**, **dual theta** (compress_rope_theta) |
| KV cache | bf16 / fp8 | **fp8_ds_mla** UE8M0 uint8 576B paged (+ bf16/fp8 fallbacks) |
| MTP | fused `eh_proj` | separate `e_proj`/`h_proj` (fp8) + `hc_head` + full V4 decoder block |

---

## 2. Our baseline — reuse-vs-new map (our `file:line`)

Anchors under `/home/mudler/_git/vllm.cpp/` (repo `src/`, `include/`).

### REUSE (landed primitives V4 can build on)
- **DENSE MLA stack** — `src/vllm/model_executor/models/deepseek_v2.cpp` (1062 LoC) +
  `include/vllm/model_executor/models/mla_attention.h` +
  `src/vllm/model_executor/layers/attention/mla_attention.cpp`: `mla::ForwardMlaAttentionBlock`,
  load-time `kv_b_proj→W_UK/W_UV` absorption, `vt::MlaDecodeAttention` (split-KV, `VT_MLA_SPLIT_FILL`
  occupancy fill), `vt::MlaPrefillAttention` + `mla_chunked_context.h`, `vt::ConcatAndCacheMla`,
  the `is_neox_style=False` decoupled RoPE + YaRN cos/sin cache, the decode-first batch reorder.
  Gated 8/8 on DeepSeek-V2-Lite. **REUSE for:** the q_lora branch, the two RMSNorms, the RoPE
  machinery, the paged MLA cache write, the decode/prefill dispatch. **DOES NOT COVER:** 512-wide
  (448+64) geometry, grouped output LoRA, attention sinks, SWA, the DSA indexer/compressor, or
  fp8_ds_mla KV — all NEW.
- **noaux_tc grouped router** — `tests/vt/test_ops_moe_router_grouped.cpp` unit-gated at V3 dims
  (sigmoid + `e_score_correction_bias` + group mask + routed_scaling). **REUSE the shape**, but V4
  needs a NEW `sqrtsoftplus` scoring func + hash-routed `tid2eid` lookup.
- **NVFP4 grouped GEMM (sm_12x)** — `src/vt/cuda/cuda_matmul_nvfp4_sm100.cu`,
  `cuda_nvfp4_tactics_*.cu`, `nvfp4_persistent_cache.h`, `cuda_scaled_mm_c3x_sm100.cu`,
  `cuda_arch_tactics.cu`. This is the W4 fast path the NVFP4 checkpoint's linear/expert GEMMs map
  onto — the reason NVFP4-everything is the recommended vehicle. **REUSE for:** the FusedMoE-fallback
  expert GEMMs + the fp8/NVFP4 linear projections.
- **MoE grouped GEMM (bf16)** + shared experts — the DeepSeek-V2 MoE block (ungated shared
  experts, grouped topk) landed in `deepseek_v2.cpp`. **REUSE the block structure**; swap router +
  clamp + hash.
- **MTP spec-decode engine loop** — `Qwen3_5MTP`/`Qwen3_5MoeMTP` DONE (k=1 verify/reject/
  `take_draft_token_ids`, `--speculative-config`). **REUSE the engine/acceptance machinery** for
  `DeepSeekV4MTP`; only the draft FORWARD (V4 decoder + MHC + e_proj/h_proj + hc_head) is new.
- **Tokenizer** — DeepSeek `SplitPattern::kDeepSeek` pre-tokenizer landed + parity-gated
  (`test_tokenizer_parity_deepseek.cpp`). Prior scope: V4's tokenizer is plain
  `PreTrainedTokenizerFast` fast-BPE (LOW risk); only the ~700-line chat template needs porting,
  upstream ships golden fixtures.

### NEW (genuinely net-new, no reuse)
1. **MHC hyper-connection topology** — `[T, hc_mult, H]` streams + Sinkhorn mixing + hc_head
   collapse. No eager ref upstream; must be reconstructed from the TileLang kernels + gated against
   a from-scratch double-precision reference. **The hardest correctness item.**
2. **DSA "Lightning Indexer"** — top-k token selection (`index_n_heads=64`/`index_head_dim=128`),
   `SparseAttnIndexer`, MXFP4/fp8 indexer KV cache. Net-new (V3.2 sparse was never landed).
3. **Compressor** (`DeepseekCompressor`) — stateful fp32 recurrent state cache + its own paged
   backend, per-layer compress_ratio 4/128, `compress_rope_theta`. Net-new.
4. **512-wide MLA + grouped output LoRA + attention sinks + SWA** — a new MLA geometry variant of
   our block (448+64 split, `wo_a`/`wo_b` bmm, per-head sink, sliding window).
5. **fp8_ds_mla UE8M0 KV cache** (576B paged) + the 3 fused qnorm/rope/quant/insert CUDA ops.
6. **sqrtsoftplus routing + hash-routed layers + clamped SwiGLU**.
7. **NVFP4 MoE via the FusedMoE fallback at 256 experts** (MegaMoE is SM100-only ⇒ do NOT port it
   for GB10; the fallback grouped-GEMM path is the target).
8. **Multi-stream attention overlap** — a PERF structure (aux streams); correctness-neutral,
   deferred to a speed brick.

---

## 3. Quant + HW-fit decision for ONE GB10 (119 GiB unified)

Ungated checkpoints and sizes (from the task's grounded facts + `quant_config.py` semantics):

| Vehicle | Weights | Fits 119 GiB? | Runs on GB10? | Notes |
|---|---|---|---|---|
| `deepseek-ai/DeepSeek-V4-Flash-DSpark` (fp8 linear + fp4 experts, + DSpark head) | ~166.9 GiB | **NO** | attn yes / MoE fallback | Too big even before KV/activations |
| `deepseek-ai/DeepSeek-V4-Flash` base | ~159.6 GiB | **NO** | — | fp8/fp4 native |
| **`nvidia/DeepSeek-V4-Flash-NVFP4` (W4 everything, group 16)** | **~83 GiB** | **YES** (≈36 GiB headroom) | **YES** — sm_12x sparse-MLA backend + our NVFP4 fast path | **RECOMMENDED PRIMARY VEHICLE** |
| `unsloth/DeepSeek-V4-Flash-GGUF` Q4/Q3/Q2 | Q4 ≈ 85 GiB / Q3 ≈ 68 GiB / Q2 ≈ 55 GiB | Q3/Q2 yes | **NO engine path** | vLLM's V4 has no GGUF loader for this arch; ours has no NVFP4-from-GGUF for V4 ⇒ not an oracle-gateable vehicle |

**Memory math (NVFP4-everything):** ~167B params × 0.5 B/param (W4) ≈ **83 GiB** weights. GB10
unified pool is 119 GiB, so ~36 GiB remain for the fp8_ds_mla KV (small: MLA stores ONE 512-wide
latent per token at fp8 ≈ 576 B/token/layer × 43 layers ≈ 24 KiB/token; a 4k context ≈ ~0.1 GiB)
plus activations, the DSA indexer/compressor caches, and CUDA-graph pools. **NVFP4 fits with a
modest context.** CAVEAT (per `[[gb10-unified-memory-oom-reboots-box]]`): the pool is UNIFIED, so
the oracle's `gpu_memory_utilization` reserves HOST RAM — keep it LOW (≤0.5) or the box hard-reboots;
never run a big oracle alongside ctest.

**Disk action needed (BLOCKER for W1):** the NVFP4 checkpoint is ~83 GiB; DGX free disk is ~68 GiB
now. **Free ~90 GiB before download** (prune old `source-*` per-SHA grid trees + `git clean -cache`
per `[[grid-per-sha-trees-fill-disk]]`), OR fall back to a **GGUF Q3 (~68 GiB)** — but GGUF has NO
oracle/engine path for V4, so it is NOT gateable and the NVFP4 vehicle is strongly preferred.

**Recommendation:** primary vehicle = **`nvidia/DeepSeek-V4-Flash-NVFP4`** (W4, ~83 GiB, fits,
runs on GB10 via the sm_12x sparse-MLA backend + FusedMoE-fallback grouped GEMM, matches our NVFP4
fast path). Free DGX disk to ~90 GiB first. fp8 native = multi-Spark / offload only (out of scope).

---

## 4. Oracle-gateability plan — W1 (the DECISIVE next brick; NOT run here)

Construct ≠ run. W1 must PROVE the pinned oracle LOADS+RUNS+GENERATES the chosen quant within
119 GiB on GB10, and capture a greedy golden. Precise recipe:
- **Checkpoint:** `nvidia/DeepSeek-V4-Flash-NVFP4` (W4, ~83 GiB). **Disk:** free ≥90 GiB first.
- **Oracle:** `${VLLM_SOURCE}` @ `555967922` (0.26.0.dev0), the from-source sm_121a build.
- **Command shape:** greedy (`temperature=0`), `--max-model-len 2048`,
  `--gpu-memory-utilization 0.45` (UNIFIED pool ⇒ keep LOW), `--max-num-seqs 1`, force the
  **NON-MegaMoE MoE backend** (do NOT pass `--kernel-config moe_backend=deep_gemm_mega_moe`; it
  aborts on GB10 with `NotImplementedError: DeepGEMM MegaMoE requires SM100`), let
  `_select_dsv4_attn_cls` default to `DeepseekV4FlashInferSM120Attention`.
- **Expected memory:** ~83 GiB weights + a few GiB KV/activation at 2k ctx ⇒ well under 119 GiB at
  util 0.45. If it OOM-reboots, drop util to 0.40 and max-model-len to 1024.
- **Flock/disk:** serialize the run through the GPU flock mutex (`[[sharing-a-gpu-with-flock]]`);
  `docker stop local-ai-worker` to free the GPU; do NOT run alongside ctest.
- **Gate form:** capture K=5 greedy runs; if vLLM is self-deterministic ⇒ STRICT token-exact bar,
  else the ratified near-tie-distributional bar (`[[near-tie-distributional-gate]]`). Commit the
  greedy golden ids + (if near-tie) the teacher-forced nats.
- **RUN-VERIFIED, not construct-verified** (`[[oracle-gateability-model-runs-not-config-constructs]]`):
  W1 passes only when the oracle EMITS tokens, not when `AutoConfig` builds the config. DEP-risk to
  watch: the DSV4 sparse backend needs a flashinfer build with the DSV4 sparse-MLA symbols
  (`flashinfer_trtllm_batch_decode_sparse_mla_dsv4`) — if absent on this box, W1 is DEP-blocked and
  the row stays SPIKE with that recorded (mirror the V3.2 disposition, but NOTE V4 ships its OWN
  backend so the sm_120 XQA-dense gap that blocked V3.2 may not apply — this is the exact thing W1
  measures).

---

## 5. Bring-up W-plan (row-sized bricks; each cites the vLLM file it ports FROM)

- **W0 (this):** scope — arch map, reuse-vs-new, quant/HW-fit, oracle plan, records. DONE.
- **W1 — ORACLE-RUN GATE + greedy golden.** Prove `nvidia/DeepSeek-V4-Flash-NVFP4` runs on GB10
  within 119 GiB (§4) + capture the golden. **The decisive brick.** If DEP-blocked → record + stop.
- **W2 — registry stub + config parse + loader map (no forward).** `DeepseekV4ForCausalLM` config
  (hc_mult, hc_sinkhorn_iters, index_topk, compress_ratios, num_hash_layers, o_lora_rank/o_groups,
  swiglu_limit, sliding_window, scoring_func, expert_dtype), the NVFP4 weight map + the
  `fused_wqa_wkv`/`compressor.fused_wkv_wgate` stacked mappings + the 256-expert w13/w2 mapping.
  Ports FROM `nvidia/model.py:1150-1350` (`load_weights`, `_make_deepseek_v4_weights_mapper`).
  Gate: loader accounts for 100% of tensors on a single-layer SLICE. Row stays SPIKE.
- **W3 — 512-wide MLA block (dense first, no DSA).** New MLA geometry: 448 NoPE + 64 RoPE,
  grouped output LoRA (`wo_a` bmm + `wo_b`), attention sinks, SWA, the dual-theta mscale-disabled
  RoPE, fp8_ds_mla KV insert. REUSE our `mla_attention` block; ADD the geometry/sink/SWA/output-LoRA
  seams. Ports FROM `attention.py:178-621` + `common/rope.py`. Unit-gate vs a CPU reference.
- **W4 — DSA indexer + compressor.** The Lightning Indexer (top-k select, `index_n_heads=64`),
  `DeepseekCompressor` state cache + its paged backend, `_fill_short_context_topk_indices`,
  compress_ratio 4/128. Ports FROM `attention.py:689-857` + `compressor.py` +
  `sparse_attn_indexer.py`. Unit-gate the top-k selection + compressed KV vs an independent oracle.
- **W5 — MHC hyper-connections.** `[T, hc_mult, H]` streams, Sinkhorn mixing (`hc_sinkhorn_iters`),
  the fused pre/post/head kernels. Ports FROM `model_executor/kernels/mhc/tilelang` (reconstruct;
  NO eager ref) + `nvidia/model.py:868-957,1080-1148`. Gate vs a from-scratch double-precision
  reference; RED-first (perturb Sinkhorn iters ⇒ divergence). **Highest-risk brick.**
- **W6 — DeepSeek-V4 MoE.** sqrtsoftplus router + `e_score_correction_bias` + hash-routed layers
  (`tid2eid`) + clamped SwiGLU + shared experts, over the **FusedMoE-fallback** NVFP4 grouped GEMM
  (NOT MegaMoE). Ports FROM `nvidia/model.py:512-757` + `fused_topk_bias_router.py`. REUSE our
  NVFP4 grouped GEMM + DeepSeek-V2 MoE block; ADD scoring/hash/clamp.
- **W7 — model forward + registry (loads+runs).** Compose W3-W6 into `DeepseekV4Model.forward`
  (`nvidia/model.py:1080-1148`). Prefill argmax vs the W1 golden's first token.
- **W8 — SACRED gate (strict / near-tie).** Full paged-engine greedy vs the W1 golden. Row
  SPIKE→ACTIVE on pass (never DONE without speed).
- **W9 — speed.** Multi-stream attention overlap (`attn_gemm_parallel_execute`), NVFP4 tactics,
  CUDA-graph decode; every-axis vs graphed vLLM. Row→DONE only at vLLM-speed parity.
- **(later) `DeepSeekV4MTP`** — reuse the MTP spec-decode engine loop; new draft forward
  (`nvidia/mtp.py`). **DSpark** left INVENTORIED (a separate campaign).

---

## 6. Dependencies
- W1 blocked on: free DGX disk (~90 GiB) + a flashinfer build carrying the DSV4 sparse-MLA symbols
  (else DEP-blocked). GPU flock + `local-ai-worker` stopped.
- W5 (MHC) blocked on: reconstructing the TileLang kernels with no eager ref (build an independent
  double-precision reference first).
- Shared with `CLAIM-GLM-DSA-LATEST-DEEPSEEK` / `CLAIM-MLA-DEEPSEEK`: the `sqrtsoftplus` extension
  and the MLA-geometry generalization must not be implemented twice — coordinate before W3/W6.

## 7. Risks / decisions
- **MHC has no eager reference and no upstream numerical test** — the single biggest correctness
  risk; must build our own oracle. (RECORDED.)
- **MegaMoE is SM100-only** — do NOT port it for GB10; the FusedMoE fallback is the target
  (`nvidia/model.py:309`). (DECISION.)
- **NVFP4-everything (~83 GiB) is the only fitting, gateable vehicle**; GGUF fits (Q3) but has no
  oracle/engine path for V4. (DECISION.)
- **UNIFIED memory** ⇒ keep oracle `gpu_memory_utilization` ≤0.45 or the box hard-reboots. (RISK.)
- **DSA sm_12x support is asserted in code but must be RUN-verified at W1** (construct ≠ run); V4
  ships its own DSV4 sparse backend, so the V3.2 sm_120 XQA-dense block may not apply — W1 decides.
- This is a MULTI-brick campaign, the largest single-model effort in the matrix; W1 is the gate
  that decides whether it is reachable at all on GB10.

---

## 8. W1/W2 IMPLEMENTATION LANDED (2026-07-28, `CLAIM-DEEPSEEK-V4-IMPL`)

**Base:** current `main` HEAD `df18ca918160a8a8741382a07c7ee8a7323efd00` (`git rev-parse HEAD`).
Row `MODEL-TEXT-deepseek-v4-deepseek-v4-for-causal-lm` TRANSFERRED here from the stale/no-worktree
`CLAIM-GLM-DSA-LATEST-DEEPSEEK` (user-directed pickup). CPU-side scaffolding only — no GPU, no
download; foreground; NOT pushed.

### 8.1 Additive TUs (SACRED-inert — zero edits to existing model forwards)
- `include/vllm/model_executor/models/deepseek_v4.h` — `DeepseekV4Params` + `ParseDeepseekV4Params`
  + weights struct + `DeepseekV4Model` (stub forward) + `MakeDeepseekV4KVCache` (stub).
- `src/vllm/model_executor/models/deepseek_v4_weights.cpp` — config parse + the checkpoint
  name-map + the W2 accounting pass.
- `src/vllm/model_executor/models/deepseek_v4.cpp` — forward SKELETON: both entrypoints
  `VT_CHECK(false, "W3-W8 pending")` (loud, never a silent wrong answer) + the reuse-wiring plan.
- `src/vllm/model_executor/models/deepseek_v4_registry.cpp` — `REGISTER_VLLM_MODEL(deepseek_v4,
  "DeepseekV4ForCausalLM", ...)`, one REGISTER line, ZERO shared-array edit (mirrors
  `deepseek_v2_registry.cpp` / `gemma4_registry.cpp`).
- `tests/vllm/models/test_deepseek_v4_scaffold.cpp` — 4/4 cases, 40/40 assertions.

### 8.2 Loader VERIFIED vs the REAL checkpoint header (the concrete W2 gate)
Evidence: `nvidia/DeepSeek-V4-Flash-NVFP4` `model.safetensors.index.json` (135,235 tensors, 46
shards) + shard-2 safetensors header, both fetched by HTTP range (NO ~156 GiB download). Confirmed
name map + dtypes/shapes (checkpoint uses a FLAT `layers.N.` prefix):
- **512-wide MLA (FP8-block E4M3 + E8M0 block scale):** `attn.wq_a` [1024,4096], `attn.wq_b`
  [32768,1024] (= 64 heads × **512 head_dim**), `attn.wkv` [512,4096], **grouped OUTPUT-LoRA**
  `attn.wo_a` [8192,4096] (= o_groups 8 × o_lora_rank 1024) / `attn.wo_b` [4096,8192],
  `attn.q_norm` [1024] / `attn.kv_norm` [512] BF16, `attn.attn_sink` [64] F32 (per-head sink).
- **DSA:** `attn.compressor.{ape,norm,wgate,wkv}` on 41 layers (compress_ratio≠0);
  `attn.indexer.{compressor.*, weights_proj, wq_b}` on 21 layers (compress_ratio==4).
- **MHC:** `hc_attn_fn` [24,16384] = (2+hc_mult)·hc_mult × hc_mult·H, `hc_{attn,ffn}_{base,fn,scale}`
  F32 per layer + model-level `hc_head_{base,fn,scale}`.
- **MoE:** `ffn.gate.weight` [256,4096]; **3 hash layers** (0,1,2) carry `ffn.gate.tid2eid` and NO
  bias, the other 40 carry `ffn.gate.bias` (noaux_tc); `ffn.shared_experts.w{1,2,3}` FP8-block;
  **256 NVFP4 routed experts** `ffn.experts.E.w{1,2,3}.{weight[U8], weight_scale[E4M3,group16],
  weight_scale_2[F32], input_scale[F32]}`. `mtp.*` (num_nextn_predict_layers=1) SKIPPED (mirrors
  vLLM `AutoWeightsLoader(skip_substrs=["mtp."])`, nvidia/model.py:1474).

The loader's accounting pass (`deepseek_v4_weights.cpp`) enumerates this exact schema from the
parsed params (array-driven per-layer branching for hash/compressor/indexer) and `VT_CHECK`s every
expected tensor is present — the W2 "loader accounts for 100% of tensors" gate, encoded.

### 8.3 HW-FIT REVERSAL (corrects §0 / §3 — the spike's central premise was wrong)
The index `total_size` = **168,266,793,544 B = 156.7 GiB**. The spike's "~83 GiB (W4-everything)
FITS" was a bad estimate: only the 256 routed experts are NVFP4/W4; the 512-wide MLA linears and
the shared experts are FP8 block (`exclude_modules: *.attn.*, *.ffn.shared_experts.*, head,
mtp.*`), and NVFP4 carries a double weight-scale + input-scale. So this "NVFP4" checkpoint is ~the
same size as the native fp4 (148.7 GiB) and **does NOT fit ONE GB10's 119 GiB unified pool.**
⇒ **W1 (single-GB10 oracle run) is MEMORY-INFEASIBLE**, not merely disk-contended. Reaching a
runnable oracle gate needs: multi-node tensor-parallel (the cluster — Thor/Orin/DGX), CPU/unified
offload, or a more aggressive quant (GGUF Q2 ~55 GiB — but V4 has no oracle/engine GGUF path).
The disk-free recipe in §4 is necessary-but-insufficient; the binding constraint is memory.

### 8.4 W1 disposition + residuals
- **W1 — DEFERRED (memory-infeasible on one GB10, box also contended).** If/when a multi-GB10 TP
  or offload path is stood up: free disk ≥170 GiB, fetch `nvidia/DeepSeek-V4-Flash-NVFP4`, `flock`,
  do NOT pass `deep_gemm_mega_moe` (SM100-only), greedy golden. DEP-risk unchanged: flashinfer must
  carry the DSV4 sparse-MLA symbols.
- **W3-W8** unchanged from §5 (512-wide MLA dense-first → DSA indexer/compressor → MHC →
  sqrtsoftplus/hash MoE → forward compose → strict gate). W2b = materialize the FP8-block + NVFP4
  towers (reuse `cuda_matmul_nvfp4_sm100` + the fp8 block loaders) into the accounted layout.

---

## HW-FIT CORRECTION (2026-07-28, user-directed): single-Spark IS viable via ~2-bit GGUF

The W0/W1 "does NOT fit one GB10" verdict was specific to the **NVFP4 (156.7 GiB)**
and **fp8 (167 GiB)** builds. **REFUTED for GGUF:** `unsloth/DeepSeek-V4-Flash-GGUF`
(ungated) ships Unsloth-Dynamic i-quants that FIT the 119 GiB GB10 (measured file
sizes, HF `files_metadata`):

| variant | size | fits 119 GiB |
|---|---|---|
| `UD-IQ1_S` / `UD-IQ1_M` | 82.5 / 86.9 GB | ✅ |
| **`UD-IQ2_XXS`** (user-cited) / `UD-IQ2_M` | **90.9 GB** | ✅ (~28 GiB KV/act headroom) |
| `UD-Q2_K_XL` | 96.8 GB | ✅ (tighter) |
| `UD-IQ3_XXS` | 103 GB | ~borderline |
| `UD-IQ3_S` .. `UD-Q8_K_XL` | 117–162 GB | ❌ |

**Two real vehicles now:** (a) **single Spark** via `UD-IQ2_XXS` GGUF (~91 GB);
(b) **2× Sparks over the interconnect** via NVFP4 (156.7 GiB) / fp8 (167 GiB) — the
`scale-out-distributed.md` multi-Spark path.

**GGUF run-path caveat (honest):** the pinned vLLM oracle almost certainly cannot
load DeepSeek-V4 (new DSA/MHC arch) from GGUF, so the **correctness reference for
the GGUF vehicle is llama.cpp-on-box** (Unsloth publishes these GGUFs FOR llama.cpp),
mirroring the beyond-vLLM CUDA-breadth bricks. This adds to the W3-W8 forward: a
**DeepSeek-V4 GGUF loader** + **IQ2_XXS / i-quant dequant** (we have the C4 GGUF
K-quant loaders — F32/F16/Q4_0/Q8_0/Q3_K/Q4_K/Q5_K/Q6_K — but NOT the IQ i-quants
`IQ1_S`/`IQ2_XXS`/`IQ2_M`, which need their codebook dequant ported). So the
single-Spark GGUF vehicle is: W3-W8 V4 forward + a V4-GGUF loader + IQ2_XXS dequant,
gated vs llama.cpp-on-card (derive-and-ship correctness reference, no vLLM oracle).

## GGUF benchmark loadability — source-level spike (2026-07-28, `CLAIM-DSV4-GGUF-SPIKE`)

Question: can we benchmark **our engine vs vLLM on the SAME `unsloth/DeepSeek-V4-Flash-GGUF UD-IQ2_XXS`** (apples-to-apples)? Answered read-only from source on the DGX (`dgx.casa`, host `promaxgb10-4ad8`): pinned vLLM `/home/mudler/vllm-src` (0.26.0.dev0) + installed oracle `~/venvs/vllm-oracle` (dist-info 0.25.0) + `github.com/vllm-project/vllm-gguf-plugin@main` + our tree (base `e0b233df`).

**Q1 — Does vLLM's GGUF path dequant the i-quants (incl. IQ2_XXS)? YES — but only via the out-of-tree plugin.**
In vLLM 0.26 **GGUF migrated OUT-OF-TREE** to `vllm-gguf-plugin` (`vllm-src/docs/features/quantization/gguf.md:9-14`: "GGUF support has migrated to OOT vllm-gguf-plugin… install before serving"). Verified the pin has **no in-tree GGUF at all**: no `gguf.py` under `vllm/model_executor/layers/quantization/`, zero `GGMLQuantizationType`/`import gguf` refs in the `vllm` package, and **no gguf CUDA kernels** in `vllm-src/csrc` nor any `ggml_dequant` symbol in the installed oracle's `*.abi3.so`. The plugin **does** dequant i-quants: explicit per-type Triton kernels `vllm_gguf_plugin/triton/dequantize/iq_quant/{iq2_xxs,iq1_s,iq1_m,iq2_s,iq2_xs,iq3_s,iq3_xxs,iq4_nl,iq4_xs}.py` **plus** CUDA `csrc/gguf/{dequantize.cuh,vecdotq.cuh,mmvq.cuh,mmq.cuh}` (llama.cpp port). So **IQ2_XXS dequant exists in vLLM's GGUF path** (via the plugin), alongside k-quant `k_quant/{q2_k..q6_k}.py`.

**Q2 — Does `DeepseekV4ForCausalLM` support GGUF in vLLM? NO / unproven.**
`DeepseekV4ForCausalLM` (`vllm-src/vllm/models/deepseek_v4/nvidia/model.py:1333`) is `class DeepseekV4ForCausalLM(nn.Module, SupportsPP, DeepseekV4MixtureOfExperts)` — it implements **only `SupportsPP`**, **not `SupportsQuant`**, and defines **no class-level `packed_modules_mapping`** (grep: none in the whole `deepseek_v4/` package; zero `gguf`/`GGUF` refs). vLLM's GGUF loading keys off the model's `packed_modules_mapping` (the plugin's `quantization/config.py:76,82` maps skip/fuse via `self.packed_modules_mapping`); the older `DeepseekV2ForCausalLM` DOES define one (`deepseek_v2.py:1763`) but **V4 does not**. DeepSeek is **absent** from the plugin's tested-model table (Qwen2.5/3, Phi-3.5, GPT-2, StableLM, Gemma-3, OLMoE, Z-Image, FLUX). Verdict: DeepSeek-V4 GGUF in vLLM is **unwired and unvalidated** (the fused-QKV-a / MoE-expert tensor names have no mapping) — most likely broken, definitely not a supported gate path.

**Q3 — The `gguf` dep + scratch-venv path.** Confirmed **missing**: `~/venvs/vllm-oracle` has **no `gguf` module** and **no `vllm-gguf-plugin`** installed → today's oracle cannot load ANY GGUF. To test, install into a **scratch venv (never mutate the oracle)**: `uv pip install vllm-gguf-plugin` — its `pyproject.toml` pins `dependencies = ["gguf>=0.17.0", "vllm", "torch>=2.9"]`. `gguf>=0.17.0` reads i-quants fine (the plugin's own Triton dequant covers them).

**Q4 — OUR engine coverage + the gaps.** Two independent blockers:
- **DeepSeek-V4 has NO GGUF path at all in our engine.** `deepseek_v4_registry.cpp:61-64` hard-throws `"DeepseekV4ForCausalLM does not support GGUF weights"` unless `source.kind == kSafetensors` (our `deepseek_v2_registry.cpp:67-69` rejects GGUF identically). So even a k-quant GGUF cannot load DeepSeek-V4/V2 in our engine today.
- **Our GGUF dequant lacks BOTH IQ2_XXS and Q2_K.** `gguf_dequant.cpp:85-160` switch covers only types F32(0)/F16(1)/BF16(30)/Q4_0(2)/Q8_0(8)/Q3_K(11)/Q4_K(12)/Q5_K(13)/Q6_K(14)/NVFP4(40); `vt/dtype.cpp:95-97` block list = `{Q4_0,Q8_0,Q3_K,Q4_K,Q5_K,Q6_K,Q8_K}`. **Q2_K (10)** and IQ2_S(22)/IQ4_XS(23) have reader *sizing* traits only (`gguf_reader.cpp:210,230,236`) — **no dequant**; **IQ2_XXS(16)/IQ1_S(19)/IQ2_M** are **absent entirely** (no trait, no dequant). Port source for IQ2_XXS: llama.cpp `ggml/src/ggml-quants.c` `dequantize_row_iq2_xxs` + the `iq2xxs_grid` codebook (256-entry Q-grid) and `ksigns`/`kmask` sign tables in `ggml-common.h` — a codebook dequant, ~1 block struct + ~40-line row loop + the static grid table; modest but net-new (no reuse from the k-quant path).

### VERDICT — NO, the same-quant IQ2_XXS GGUF benchmark is NOT viable today (blocked on BOTH engines)
- **Our side:** DeepSeek-V4 rejects GGUF outright (registry) **and** we have no IQ2_XXS (nor Q2_K) dequant.
- **vLLM side:** GGUF is out-of-tree (plugin+`gguf` lib both uninstalled) **and** `DeepseekV4ForCausalLM` has no `packed_modules_mapping`/GGUF wiring — DeepSeek-V4 GGUF is unproven/likely-broken even after installing the plugin.
- **The `UD-Q2_K_XL` (96.8 GB) k-quant fallback does NOT rescue it:** it still hits the same DeepSeek-V4 GGUF-rejection on our side, we also lack Q2_K dequant, and vLLM still lacks the V4 gguf wiring. Same-quant DeepSeek-V4 GGUF is blocked at the *model-arch* layer, not just the quant-type layer.
- **Ops:** DGX `/` is at **99% (≈53 GiB free)**; the 90.9 GB IQ2_XXS won't fit — the HF cache holds only `config.json` (28 KB, no shards). Freeing to ~100 GiB is a hard prerequisite regardless.

### FALLBACK (grounded)
1. **DeepSeek-V4 apples-to-apples = the NVFP4 vehicle, NOT GGUF.** Both engines have a real NVFP4 path (`nvidia/DeepSeek-V4-Flash-NVFP4` is cached; our dequant handles ggml/NVFP4 type 40 and safetensors NVFP4; vLLM via modelopt). NVFP4 needs 2× Sparks (156.7 GiB) — the `scale-out-distributed.md` path. This is same-*quant* cross-engine but is the fp4 vehicle, not the ~2-bit GGUF one.
2. **GGUF vehicle reference stays llama.cpp-on-card** (Unsloth publishes these FOR llama.cpp; it natively dequants IQ2_XXS). vLLM would run NVFP4/fp8 — **cross-quant, not apples-to-apples** — so a three-way GGUF-vs-GGUF chart is only ours-vs-llama.cpp.
3. **For a genuine same-GGUF-quant CROSS-ENGINE number, pick a model both already load** (not DeepSeek-V4): a k-quant (e.g. `Q4_K_M`/`Q6_K`) on a **Qwen3 / dense arch** — our `qwen3_5_gguf_weights.cpp` path + the plugin's tested Qwen3 support overlap on k-quants we both dequant. That is the only apples-to-apples GGUF cross-engine bench available without net-new work on both sides.

**To make DeepSeek-V4 GGUF IQ2_XXS truly apples-to-apples would require, on BOTH engines:** (ours) lift the V4 registry GGUF rejection + wire a V4-GGUF loader + port IQ2_XXS (and/or Q2_K) dequant; (vLLM) add `packed_modules_mapping`/GGUF compat to `DeepseekV4ForCausalLM` in the plugin (upstream work) — neither exists today.

---

## W3 — 512-wide MLA output seams + DSA Lightning-Indexer SELECTION (2026-07-28, `CLAIM-DEEPSEEK-V4-W3`)

**Base:** current `main` HEAD `308c312a` (`git rev-parse HEAD`). Isolated worktree, CPU-only,
foreground, NOT pushed. User-directed: "implement anyway" — the full-model gate is multi-Spark-
blocked (156.7 GiB, does not fit one GB10; the forward also needs MHC + the sqrtsoftplus/hash MoE,
neither ported), so W3 lands the FORWARD CODE for the genuinely-NEW primitives + UNIT-gates each.

### W3.1 What landed (additive, SACRED-inert)
New TUs `include/vllm/model_executor/models/deepseek_v4_dsa.h` +
`src/vllm/model_executor/models/deepseek_v4_dsa.cpp` — portable HOST (CPU) reference
implementations of the two things W3 owns, each ported 1:1 with `file:line` on both sides:

- **(A) DSA "Lightning Indexer" sparse SELECTION** — the genuinely-new primitive (our MLA is dense
  V2/V3; V3.2 sparse was never landed):
  - `DsaIndexerWeightFold` <- `sparse_attn_indexer.py:203-207` (`weight * softmax_scale *
    head_scale`, `softmax_scale = index_head_dim**-0.5` `attention.py:735`, `head_scale =
    index_n_heads**-0.5` `attention.py:843`; the fp8 per-token `q_scale` is 1 in the fp32 reference,
    documented not silently dropped).
  - `DsaIndexerLogits` <- `v1/attention/ops/triton_fp8_mqa_logits.py:120-156` — the weighted MQA
    logit `logit[t,s] = Σ_h folded_w[t,h] · ReLU( dot(q[t,h,:], k[s,:]) )` over the causal
    candidate window, out-of-window keys `-inf`. **The per-head `ReLU` is the load-bearing
    nuance** (`:129`): it is what makes the indexer a learned sparse SELECTOR rather than a plain
    attention score — a unit case pins it (a negative q·k that WOULD flip the top-k without the
    clip).
  - `DsaTopkSelect` <- `sparse_attn_indexer.py:488-497` (`top_k_per_row`) + the short-context
    all-candidate select `attention.py:70-86,:813-831`: `#candidates ≤ topk` ⇒ every candidate in
    ascending key order then `-1` padding; else the `index_topk=512` largest-logit keys (ties →
    smaller key index, stable).

- **(B) 512-wide MLA OUTPUT seams V2/V3 do NOT have:**
  - `SoftmaxWithSink` <- per-head attention sinks `flashinfer_sparse.py:777,:896` +
    `attention.py:219-222` (the `-inf` init = no effect): an extra per-head sink logit in the
    softmax DENOMINATOR that carries no value, so the returned probs sum to `< 1` by the sink's
    share; numerically stable (max-subtraction).
  - `GroupedOutputLora` <- `nvidia/ops/o_proj.py:58-73`: reshape the per-head attention output into
    `o_groups=8` groups of `heads_per_group*head_dim`, per-group `wo_a` matmul (the bmm einsum
    `"bhr,hdr->bhd"`), concat to `n_groups*o_lora_rank`, then `wo_b` to hidden. (The inverse-RoPE +
    fp8 quant that precede the einsum on GPU reuse our decoupled-RoPE machinery with a negated
    angle — a W7 seam, omitted so the gate isolates the grouped-LoRA linear algebra.)

### W3.2 Gate (unit, honest) — hand-case + structural review, NOT a dumped-oracle rel-L2
The arch is a fixed-config 167B and CANNOT be constructed at a tiny synthetic shape (nor run — it
does not fit one GB10), so per the brief the gate is the MATH against HAND-DERIVED small cases with
literal expected numbers verified from the vLLM source, PLUS a from-first-principles
double-precision reference on randomized shapes. `tests/vllm/models/test_deepseek_v4_dsa.cpp`:
**13/13 cases · 38 assertions GREEN**, clean CPU `-Wall -Werror -Wextra` 0-warn (new lib TU + test).
Literal cases: weight fold `= D^-0.5·H^-0.5`; MQA logit `Σ w·ReLU(q·k)` (the ReLU clip proven load-
bearing); causal-window `-inf`; short-context all-select; full top-k largest-logit; tie→smaller-
index; windowed top-k offset; sink removes probability mass (`Σp = 2/3`); sink stable at large
logits; grouped-LoRA `[1,6]`. Randomized: indexer logits + grouped output-LoRA vs independent
double-precision loops, rel-L2 `< 1e-6`.

### W3.3 SACRED inertness
Additive ONLY. Zero edits to any existing forward — in particular the shared DeepSeek-V2 MLA
(`src/vllm/model_executor/layers/attention/mla_attention.{h,cpp}`, `src/vt/cuda/cuda_mla_attn.cu`)
is UNTOUCHED. Per the brief, extending the SHARED MLA block risks V2, so V4's 512-wide geometry +
sinks + output-LoRA land as a V4-specific path and the shared-mla extraction is a NAMED W7 follow-
on. Proof: `test_deepseek_v4_scaffold` still 4/4·40; the only non-additive edits are the CMake
source/test wiring, the `KERNEL-ATTN-DSA-SPARSE-INDEX` kernel-matrix row (+ its checker count bump
37→38), and the record surfaces.

### W3.4 SGLang benchmark-reference finding (user asked "maybe sglang?")
The same-quant vLLM benchmark is out (V4 has no in-tree vLLM-GGUF path + NVFP4 needs 2 Sparks — see
the GGUF-loadability section). **SGLang `v0.5.15`** (`/home/mudler/_git/sglang`, `git describe` =
`v0.5.15`) **DOES register and implement `DeepseekV4ForCausalLM`** — `EntryClass =
[DeepseekV4ForCausalLM]` in `python/sglang/srt/models/deepseek_v4.py` (2856 LoC) plus
`deepseek_v4_nextn.py`, carrying the FULL V4 stack: the DSA indexer (`C4Indexer`) + `Compressor`
(`python/sglang/srt/layers/attention/dsv4/{indexer,compressor}.py`), MHC (`hc_split_sinkhorn` /
`mhc_fused_post_pre` in `python/sglang/srt/layers/mhc`), grouped `o_lora`, per-head `attn_sink`, the
`compress_ratio ∈ {0,4,128}` per-layer topology, dual-theta RoPE. So SGLang is a **viable second
reference**, and its DSA/compressor kernels are an independent implementation to cross-check our
math against. HOW it would be gated:
- **(a) tiny-shape primitive DUMP oracle** — IF `C4Indexer` / `Compressor` construct at small
  synthetic dims (they take explicit `head_dim`/`n_heads`/`compress_ratio` ctor args, unlike the
  fixed-config full model), dump their reference tensors and gate our `DsaIndexerLogits` /
  `DsaTopkSelect` rel-L2 vs SGLang's — a STRONGER gate than the hand case. This is the recommended
  W4/W7 follow-on (needs an SGLang venv + a GPU or CPU-constructible path — not run this pass).
- **(b) multi-node / 2-Spark benchmark reference** — SGLang faces the SAME single-GB10 memory
  infeasibility (NVFP4 156.7 GiB / fp8 167 GiB), so an END-TO-END SGLang benchmark is only a
  cross-engine number on ≥2 Sparks (the `scale-out-distributed.md` path), same as vLLM.

### W3.5 Landed-vs-residual
- **Landed (W3):** DSA Lightning-Indexer sparse selection math + 512-wide MLA attention-sink softmax
  + grouped output-LoRA — host references + unit gate; `KERNEL-ATTN-DSA-SPARSE-INDEX` (`SPIKE`).
- **Residual (kept as precise stubs/TODOs):** MHC hyper-connections (W5, no eager ref — hardest),
  the sqrtsoftplus + hash-routed MoE over the FusedMoE fallback (W6), the DSA compressor state
  cache + paged backend + the fp8_ds_mla KV insert (W4 device side), the device kernels for the W3
  primitives + folding them into `DeepseekV4Model::Forward` and the shared-mla extraction (W7), and
  the full strict/near-tie engine gate (W8) — all multi-Spark-blocked on one GB10.

---

## W4 — DSA COMPRESSOR forward + fp8_ds_mla KV-cache state (2026-07-29, `CLAIM-DEEPSEEK-V4-W4`)

**Base:** current `main` HEAD `4d1be010` (`git rev-parse HEAD`). Isolated worktree, CPU-only,
foreground, NOT pushed. Continues the W3 host-reference lane: the full-model gate is multi-Spark-
blocked (156.7 GiB, does not fit one GB10; the forward also needs MHC + the sqrtsoftplus/hash MoE),
so W4 lands the FORWARD CODE for the second half of the DSA sparse-attention stack + UNIT-gates it.

### W4.1 What landed (additive, SACRED-inert)
New TUs `include/vllm/model_executor/models/deepseek_v4_compressor.h` +
`src/vllm/model_executor/models/deepseek_v4_compressor.cpp` — portable HOST (CPU) references for the
two things W4 owns, each ported 1:1 with `file:line` on both sides (vLLM primary, SGLang v0.5.15
cross-reference):

- **(A) the DSA COMPRESSOR forward** — where W3's Lightning-Indexer SELECTS keys, the compressor
  POOLS + normalizes them into the compressed latent the MLA reads:
  - `CompressorSaveScoreApe` <- `common/ops/save_partial_states.py:92-101` — the fused save-time APE
    add `score_state[t,d] = score[t,d] + ape[positions[t] % compress_ratio, d]` (the kv half is
    stored verbatim; only the score half gets the position-embedding add, which is why it is fused
    into the state write).
  - `CompressorPoolNorm` <- `common/ops/fused_compress_quant_cache.py:198-218` (cross-checked vs
    SGLang `dsv4/fused_compress_triton.py` `_fused_ape_pool_norm_rope_kernel:57-95`) — the softmax-
    weighted window POOL: `softmax(score, dim=0)` INDEPENDENTLY PER head-dim column, weighted sum of
    the window's kv → compressed latent, then RMSNorm (fp32). **The per-column softmax is the load-
    bearing nuance** — each head-dim channel pools the `(1+overlap)·compress_ratio` window with its
    OWN weights, not one shared attention weight per row; a unit case pins it (a column whose score
    strongly favors one window row diverges from a uniform column in a way only per-column softmax
    produces, and the ratio survives RMSNorm).

- **(B) the fp8_ds_mla KV-CACHE STATE layout** — how the compressed latent is written to / read from
  the paged cache across steps:
  - `MakeFp8DsMlaLayout` <- `compressor.py:307-309` (`_quant_block=64`, `_token_stride=nope+rope*2=
    576`, `_scale_dim = nope//64 + 1 = 8`); geometry cross-checked vs SGLang `dsv4/dequant_k_cache.py
    :12-18` (`DIM_NOPE=448`, `TILE_SIZE=64`): per token 448 fp8 NoPE + 64 bf16 RoPE = 576 contiguous
    bytes, then a padded 7+1 UE8M0 scale region.
  - `Fp8DsMlaEncodeToken` <- `fused_compress_quant_cache.py:238-297` store side — per 64-wide NoPE
    block: bf16-round (the kernel casts fp32→bf16→fp32), per-block `absmax` (clamped ≥1e-4), UE8M0
    power-of-two scale `exponent = ceil(log2(absmax/448))`, `inv_scale = 2^-exponent`, `clamp(q·
    inv_scale, ±448)` → e4m3; the stored scale byte `= clamp(exponent + 127, 0, 255)`. RoPE part →
    bf16 verbatim (the forward RoPE that precedes it on GPU reuses our decoupled-RoPE machinery — a
    W3/W7 seam — so this reference round-trips the bytes and leaves the rotation to W7).
  - `Fp8DsMlaDecodeToken` <- the paged-KV READ, `SGLang dsv4/dequant_k_cache.py:122-136`:
    `nope[d] = e4m3→f32(byte[d]) · 2^(scale_byte[block]-127)`, `rope[j] = bf16→f32`. Reuses the
    landed host helpers `F32ToF8E4M3`/`F8E4M3ToF32` (nvfp4 path) + `vt::F32ToBF16`/`BF16ToF32`.

### W4.2 Gate (unit, honest) — hand-case + structural review, NOT a dumped-oracle rel-L2
Same honest bar as W3 (the fixed-config 167B arch cannot be constructed at a tiny shape, nor run —
it does not fit one GB10). `tests/vllm/models/test_deepseek_v4_compressor.cpp`: **12/12 cases · 164
assertions GREEN** on a CPU Debug full-library build, 0-warn on the new TUs. Hand-derived literal
cases: APE modulo-wrap (`pos 3, cr 2 → ape row 1`); pool = `softmax(score,dim=0)·kv` then RMSNorm;
window masking excludes out-of-range rows; per-column softmax load-bearing (column ratio survives
RMSNorm); V4 layout `448/64/576/7+1`; all-ones NoPE block → UE8M0 byte `119` + exact round-trip;
value-3 block → byte `120`; bf16 rope verbatim. Randomized double-precision references: pool+norm
rel-L2 < 1e-6; an INDEPENDENT recompute of every UE8M0 scale byte; encode→decode round-trip < 0.05
(honest fp8-granularity bound, not 1e-6 — e4m3 carries 3 mantissa bits). **RED-first PROVEN:**
perturbing the scale bias `+127→+126` fails 4 cases / 135 assertions; revert restores 12/12·164.
CAVEAT: the CPU gate uses `-DCMAKE_BUILD_TYPE=Debug` because a pre-existing GCC-13 `-O2`
`-Werror=array-bounds` false positive in `voxtral.cpp` (unrelated to these TUs) breaks the `-O2`
full-library build; the new TUs themselves are `-Wall -Werror -Wextra`-clean.

### W4.3 SACRED inertness
Additive ONLY. Zero edits to any existing forward — in particular the shared DeepSeek-V2 MLA
(`mla_attention.{h,cpp}`, `cuda_mla_attn.cu`) has an EMPTY diff. Proof: `test_deepseek_v4_dsa` still
13/13·38, `test_deepseek_v4_scaffold` 4/4·40. The only non-additive edits are the two CMake source/
test wiring lines, the new `KERNEL-ATTN-DSA-COMPRESSOR` kernel-matrix row (+ its checker count bump
39→40), and the record surfaces.

### W4.4 Landed-vs-residual
- **Landed (W4):** the DSA compressor pool+norm + save-time APE + the fp8_ds_mla KV-state read/write
  layout — host references + unit gate; `KERNEL-ATTN-DSA-COMPRESSOR` (`SPIKE`).
- **Residual:** the compressor STATE-CACHE gather addressing (the `head_offset` overlap window
  indexing into the paged state, the two-stage cr≥128 split, `CompressorMetadata`/`CompressorBackend`
  paging) is a W7 device-integration concern — W4's host reference feeds the ASSEMBLED window so the
  gate isolates the pool/quant compute (`fused_compress_quant_cache.py:169-196` address arithmetic is
  named, not ported). Still residual: MHC (W5), sqrtsoftplus/hash MoE (W6), the fused device kernel
  (`_fused_kv_compress_norm_rope_insert_sparse_attn`) + `DeepseekV4Model::Forward` integration +
  shared-mla extraction (W7), the strict/near-tie engine gate (W8) — all multi-Spark-blocked.
