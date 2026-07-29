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

---

## W5 — Manifold/Markov Hyper-Connections (MHC) forward + Sinkhorn (2026-07-29, `CLAIM-DEEPSEEK-V4-W5`)

**Base:** current `main` HEAD `d0bc0f41` (`git rev-parse HEAD`). Isolated worktree `/home/mudler/_git/vllm.cpp-w5-mhc`,
CPU-only, foreground, NOT pushed. Continues the W3/W4 host-reference lane: the full-model gate is
multi-Spark-blocked (156.7 GiB, does not fit one GB10; the forward still needs the sqrtsoftplus/hash
MoE (W6) + device assembly (W7)), so W5 lands the FORWARD CODE for the MHC residual topology — the
single hardest V4 brick per the W0 scope — + UNIT-gates it.

### W5.0 EAGER-REFERENCE FINDING — corrects the W0/model-matrix "no eager reference" premise
The W0 scope and the model-matrix row assert MHC has "ZERO numerical tests and no eager reference
upstream" (the reason it was flagged the hardest correctness item). **That premise is CORRECTED
here.** The pinned vLLM SHIPS an eager PyTorch reference for the mHC kernels:
- `vllm/model_executor/kernels/mhc/torch.py` — `mhc_pre_torch` (`:6-91`) + `mhc_post_torch` (`:94-106`);
- `vllm/model_executor/kernels/mhc/triton.py` — `hc_head_reduce_triton_kernel` (`:108-140`), the
  head collapse eager math (`rmsnorm_nw` + linear + sigmoid + weighted reduce).

These are the canonical numerics the TileLang / Triton / CUDA / AITER kernels all match. Moreover
**four independent upstream implementations agree byte-for-byte on the Sinkhorn**: `torch.py:75-82`,
`tilelang_kernels.py:126-153` (`_sinkhorn_fwd`), `tilelang.py` (`mhc_pre_big_fuse_with_norm`), and
SGLang `v0.5.15` `python/sglang/srt/layers/mhc.py:110-126` (`hc_split_sinkhorn_kernel`). So the
numerics are UNAMBIGUOUS. Per the brief we still ALSO derive an INDEPENDENT double-precision Sinkhorn
from first principles (alternating row/column normalization toward a doubly-stochastic matrix) and
prove the port agrees — the eager ref pins the exact eps/axis conventions, the derived ref proves the
port is not a transcription of a bug.

### W5.1 What landed (additive, SACRED-inert)
New TUs `include/vllm/model_executor/models/deepseek_v4_mhc.h` +
`src/vllm/model_executor/models/deepseek_v4_mhc.cpp` — portable HOST (CPU) references for the four
MHC pieces, each ported 1:1 with `file:line` on BOTH sides (vLLM primary, SGLang v0.5.15 cross-ref):

- **`MhcSinkhorn`** <- `torch.py:75-82`. The `hc_sinkhorn_iters`(=20) Sinkhorn of the hc_mult×hc_mult
  mixing matrix: seed `M = softmax(logits, dim=-1) + eps` (row softmax over k, +eps), col-norm
  `M /= (Σ_j M + eps)`, then `(iters-1)× [row-norm M/=(Σ_k M+eps), col-norm M/=(Σ_j M+eps)]`. The `+eps`
  is on the softmax-seed NUMERATOR and every normalization DENOMINATOR (exactly as all four upstream
  impls). With eps=0 it converges to a doubly-stochastic matrix; the AXIS ALTERNATION + ITERATION
  COUNT are load-bearing at non-converged counts.
- **`MhcPre`** <- `torch.py:56-91` (`mhc_pre_torch`) + the fold from `tilelang.py`
  `mhc_pre_big_fuse_with_norm`. Flatten the streams, `mixes = residual_flat @ fn.T`, folded weight-free
  RMSNorm `mixes *= rsqrt(sqrsum/(hc·H)+rms_eps)`, then `pre=σ(mixes[:hc]·s0+b)+hc_pre_eps`,
  `post=σ(mixes[hc:2hc]·s1+b)·hc_post_mult(2.0)`, `comb=Sinkhorn(mixes[2hc:]·s2+b)`,
  `layer_input=Σ_j pre[j]·residual[j]`, and (optional) the FOLDED attn/ffn RMSNorm
  `layer_input *= rsqrt(mean(layer_input²)+norm_eps)·norm_weight`. Constants
  `hc_post_alpha=2.0`, `hc_pre_eps=hc_sinkhorn_eps=hc_eps` @ `nvidia/model.py:818-821,:886-894`.
- **`MhcPost`** <- `torch.py:94-106` (`mhc_post_torch`). `new[j,h] = Σ_i comb[i,j]·residual[i,h] +
  post[j]·x[h]` — the einsum `"ij,ih->jh"` (comb mix, SUMS over the first index i) + the post gate add.
- **`HcHeadCollapse`** <- `triton.py:108-140` + `tilelang.py:720-748` (`hc_head_fused_kernel_tilelang`).
  The final collapse of the hc_mult streams to one hidden: weight-free RMSNorm(rms_eps) → `hc_head_fn`
  [hc,hc·H] projection → `pre=σ(mixes·hc_head_scale+hc_head_base)+hc_eps` → `out[h]=Σ_m pre[m]·x[m,h]`.
  `hc_head_scale` is the SCALAR [1] `nvidia/model.py:1038`; the model's final RMSNorm(weight) that
  follows is a separate standard norm, not folded here.

### W5.2 Gate (unit, honest) — DERIVED-eager-reference + hand-case + structural review
Same honest bar as W3/W4 (the fixed-config 167B arch cannot be constructed at a tiny shape, nor run —
it does not fit one GB10; and — the brief's premise — upstream ships no GOLDEN numerical test even
though it ships an eager reference). `tests/vllm/models/test_deepseek_v4_mhc.cpp`: **14/14 cases · 125
assertions GREEN** on a CPU Debug full-library build, `-Wall -Werror -Wextra` 0-warn on the new TUs.
Hand-derived literal cases: all-zero Sinkhorn → uniform doubly-stochastic 1/hc; symmetric-2×2 fixed
point `[[.75,.25],[.25,.75]]`; iteration-count load-bearing (iters=1 col-sums=1 but row-sums≠1 vs
iters=20 both=1); MhcPre fn=0 → gate midpoints (pre=0.5, post=1.0, comb=0.5, layer_input=Σ0.5·res);
the RMSNorm fold `[1,3]→[1,3]/√5`; MhcPost identity-comb + post-add (`25/37`); the mix sums over the
first comb index (uniform comb → stream mean); hc_head fn=0 → stream mean. From-first-principles
DOUBLE-PRECISION references: Sinkhorn / MhcPre / MhcPost / HcHead f32==f64 rel-L2 < 1e-5..1e-4;
Sinkhorn doubly-stochastic convergence (row+col sums=1 within 1e-4 at eps=0). **RED-first PROVEN BOTH
levers:** perturbing the Sinkhorn iteration count (`iters-1→iters-2`) fails 1 case / 9 assertions AND
swapping a normalization axis (col-norm sum over the wrong index) fails 2 cases / 12 assertions —
caught by a DEDICATED SMALL-iteration-count (2/3/5) f32==f64 gate, because at 20 iters the Sinkhorn
has CONVERGED and ±1 iteration is within any tolerance (an honesty fix over a naive iters=20-only
gate, which the first perturbation attempt silently PASSED — recorded so the lesson survives). Revert
restores 14/14·125. CAVEAT: the CPU gate uses `-DCMAKE_BUILD_TYPE=Debug` because the pre-existing
GCC-13 `-O2` `-Werror=array-bounds` false positive in `voxtral.cpp` (unrelated to these TUs) breaks
the `-O2` full-library build; the new TUs themselves are `-Wall -Werror -Wextra`-clean.

### W5.3 OPEN QUESTIONS / derivation assumptions (stated explicitly)
- **bf16 storage rounding between steps is NOT modeled.** The model rounds `residual` and
  `layer_input` to bf16 across attn/ffn boundaries (the mix/Sinkhorn compute stays fp32); these host
  references stay in f32/f64 and leave bf16 storage rounding to the W7 device brick, exactly as W3/W4
  left their RoPE/fp8 seams. The Sinkhorn/pre/post/head MATH is what W5 pins.
- **Every resolved constant is grounded, none guessed.** `hc_post_alpha=2.0` (`model.py:821`),
  `hc_pre_eps=hc_sinkhorn_eps=hc_eps` (both passed as `self.hc_eps`, `:886-894`), `hc_sinkhorn_iters`
  from config (=20 in the W0 scope), `rms_norm_eps` (the mix + head RMSNorm eps), the attn/ffn
  `norm_weight`/`norm_eps` fold (`mhc_pre_big_fuse_with_norm`), `hc_head_scale` scalar (`:1038`).
- **First-layer stream EXPAND** ([T,H] embedding → [T,hc,H] manifold via `mhc_pre_broadcast_tilelang`
  `fn_broadcast`) is a shape/broadcast op folded into the first decoder layer; it is a W7
  forward-assembly seam, not a distinct numeric primitive, so it is NAMED here, not ported in W5.

### W5.4 SACRED inertness
Additive ONLY. Zero edits to any existing forward — MHC is a V4-only topology; the shared DeepSeek-V2
MLA (`mla_attention.{h,cpp}`, `cuda_mla_attn.cu`) and the W3/W4 DSA/compressor TUs are UNTOUCHED
(empty diff). Proof: `test_deepseek_v4_compressor` still 12/12·164, `test_deepseek_v4_dsa` 13/13·38,
`test_deepseek_v4_scaffold` 4/4·40. The only non-additive edits are the two CMake source/test wiring
lines, the new `KERNEL-MHC-SINKHORN` kernel-matrix row (+ its checker count bump 40→41), and the
record surfaces.

### W5.5 Landed-vs-residual
- **Landed (W5):** the MHC hyper-connection expand/mix (`[T, hc_mult, H]` manifold) + the 20-iteration
  Sinkhorn + the folded attn/ffn/head RMSNorms — host references + unit gate; `KERNEL-MHC-SINKHORN`
  (`SPIKE`).
- **Residual:** the device kernels (`mhc_pre`/`mhc_post`/`mhc_fused_post_pre`/`hc_head_fused` +
  the bf16 residual storage) + folding MHC into `DeepseekV4Model::Forward` with the first-layer
  stream expand + the per-layer attn/ffn interleave (W7); the sqrtsoftplus/hash MoE (W6); the
  strict/near-tie engine gate (W8) — all multi-Spark-blocked on one GB10.

---

## W6 — the sqrtsoftplus + hash-routed MoE (2026-07-29, `CLAIM-DEEPSEEK-V4-W6`)

**Base:** current `main` HEAD `5b843be5` (`git rev-parse HEAD`). Isolated worktree
`/home/mudler/_git/vllm.cpp-w6-moe` (branch `deepseek-v4-w6-moe`), CPU-only, foreground, NOT pushed.
Continues the W3/W4/W5 host-reference lane: the full-model gate is multi-Spark-blocked (156.7 GiB, does
not fit one GB10; the forward still needs device assembly (W7)), so W6 lands the FORWARD CODE for the
three genuinely-new MoE primitives + UNIT-gates them. Per the brief, the shared DeepSeek grouped-GEMM /
256-expert / shared-expert / NVFP4 machinery is REUSED, not re-ported — only the NEW scoring + hash-route
+ clamp are net-new, so only those three land here.

### W6.1 What landed (additive, SACRED-inert)
New TUs `include/vllm/model_executor/models/deepseek_v4_moe.h` +
`src/vllm/model_executor/models/deepseek_v4_moe.cpp` — portable HOST (CPU) references for the three
things W6 owns, each ported 1:1 with `file:line` on BOTH sides (vLLM primary, SGLang v0.5.15 cross-ref):

- **`SqrtSoftplus`** <- `fused_topk_bias_router.py:88` (`torch.sqrt(F.softplus(gating_output.float()))`).
  The V4 router score `sqrt(softplus(x))`, `softplus(x)=log(1+exp(x))` (numerically stable
  `max(x,0)+log1p(exp(-|x|))`), distinct from V2/V3's sigmoid/softmax `noaux_tc`. The sqrt∘softplus
  COMPOSITION is the load-bearing nuance.
- **`SqrtSoftplusRouteTopk`** <- `fused_topk_bias_router.py:75-118` (`_topk_softplus_sqrt_torch`, the
  pure-PyTorch fallback = the eager reference the CUDA `topk_hash_softplus_sqrt` kernel matches) + the
  hash branch `:100-106`; hash-table wiring `nvidia/model.py:562-578,:686,:696-717`; FusedMoE-fallback
  router `nvidia/model.py:647-691`. Score every expert; add `e_score_correction_bias` for **SELECTION
  ONLY**; either top-k by the biased score OR — for the first `num_hash_layers` HASH layers — the
  `tid2eid` [vocab, topk] token-id→expert lookup that **BYPASSES top-k**; **GATHER the weights from the
  UNBIASED scores** (using biased scores as weights flattens the distribution, `:90-96`); renormalize
  `/max(Σ,1e-20)`; ×`routed_scaling_factor`. The bias-affects-selection-but-NOT-weights split + the hash
  bypass are the two load-bearing nuances. Cross-checked SGLang `hash_topk.py:137-180` (`_forward_torch`,
  identical tid2eid gather from unbiased scores) + `topk.py:1013-1014` (the score).
- **`ClampedSwiGLU`** <- `activation.py:197-201` (`SiluAndMulWithClamp.forward_native`), used by
  DeepseekV4MLP `nvidia/model.py:126-133`. `gate=clamp(·,max=limit)` (MAX only), `up=clamp(·,-limit,
  +limit)` (BOTH sides), `out=gate·sigmoid(α·gate)·(up+β)`; DeepseekV4MLP passes α=1/β=0. The ASYMMETRIC
  clamp (gate max-only vs up both-sided) is the load-bearing nuance.

**NOT re-ported (REUSED):** the DeepSeek grouped-GEMM expert forward, the 256-expert w13/w2 layout, the
shared-expert block, and the NVFP4/FP8 expert GEMMs are the existing DeepSeek-V2 MoE + NVFP4 machinery.
**NOT ported (SM100-only, not the GB10 target):** MegaMoE (`DeepseekV4MegaMoEExperts`, requires
major==10, `nvidia/model.py:307-315`) — W6 mirrors the FusedMoE-fallback router GB10 runs.

### W6.2 Gate (unit, honest) — hand-case + structural review, NOT a dumped-oracle rel-L2
Same honest bar as W3/W4/W5 (the fixed-config 167B arch cannot be constructed at a tiny shape, nor run —
it does not fit one GB10). `tests/vllm/models/test_deepseek_v4_moe.cpp`: **12/12 cases · 716 assertions
GREEN** on a CPU Debug full-library build; the new TUs are `-Wall -Werror -Wextra`-clean (verified by an
explicit strict compile). Hand-derived literal cases: sqrt∘softplus composition (`softplus(x)=4 ⇒
score=2`, distinct from softplus-alone=4 and raw-logit-sqrt); the bias flips the selection but the
returned weight stays the UNBIASED 1.0 not the biased 3.0 (and without the bias the unbiased argmax is
chosen — proving the bias changed the selection); renormalize divides by the unbiased weight sum;
routed_scaling_factor multiplies; the hash `tid2eid` route picks {3,1} where top-k would pick {2,0};
the asymmetric clamp keeps gate=-5 while clamping up to -2; gate max-clamp / up upper-clamp boundaries;
alpha/beta. From-first-principles double-precision references: router f32==f64 rel-L2 < 1e-5 + EXACT ids
across randomized shapes; SqrtSoftplus f64 + monotonicity; ClampedSwiGLU rel-L2 < 1e-6. **RED-first
PROVEN all three levers:** dropping the sqrt fails 8 cases / 493 assertions; gathering the weights from
the BIASED scores fails 2 cases / 181 assertions; symmetric-clamping the gate fails 2 cases / 6
assertions; revert restores 12/12·716. Each lever is genuinely load-bearing (learning from W5: the
composition case distinguishes sqrt∘softplus from softplus-alone AND from raw-logit-sqrt; the
bias-selection case forces the weight to differ from the biased score; the asymmetry case forces gate
and up to be clamped differently). CAVEAT: the CPU gate uses `-DCMAKE_BUILD_TYPE=Debug` because the
pre-existing GCC-13 `-O2` `-Werror=array-bounds` false positive in `voxtral.cpp` (unrelated) breaks the
`-O2` full-library build; the new TUs themselves are `-Wall -Werror -Wextra`-clean.

### W6.3 OPEN QUESTIONS / derivation assumptions
- **None unresolved — every constant grounded, none guessed.** The score, the bias-for-selection split,
  the hash bypass, the renormalize `1e-20` floor, `routed_scaling_factor`, and the clamp α=1/β=0 defaults
  are all read directly from the cited `file:line`.
- **The device kernel's `is_padding` guard** (zeroes padded rows, `topk_hash_softplus_sqrt`) is a
  device-batching concern — this host reference computes every real row and leaves padding to W7, exactly
  as W3/W4/W5 left their device seams.

### W6.4 SACRED inertness
Additive ONLY. Zero edits to any existing forward — in particular the shared DeepSeek-V2 MoE router /
grouped GEMM / shared experts have an EMPTY diff, and the W3/W4/W5 DSA/compressor/MHC TUs are untouched.
Proof: `test_deepseek_v4_mhc` still 14/14·125, `test_deepseek_v4_compressor` 12/12·164,
`test_deepseek_v4_dsa` 13/13·38, `test_deepseek_v4_scaffold` 4/4·40. The only non-additive edits are the
two CMake source/test wiring lines, the new `KERNEL-MOE-SQRTSOFTPLUS-HASH` kernel-matrix row (+ its
checker count bump 41→42), and the record surfaces.

### W6.5 Landed-vs-residual
- **Landed (W6):** the sqrtsoftplus router score + the noaux_tc bias-for-selection top-k + the `tid2eid`
  hash route + the clamped SwiGLU expert activation — host references + unit gate;
  `KERNEL-MOE-SQRTSOFTPLUS-HASH` (`SPIKE`).
- **Residual:** the device kernels (the router + clamp are the only NEW device kernels; the expert
  grouped-GEMM REUSES the existing NVFP4/FP8 path) + folding the MoE into `DeepseekV4Model::Forward`
  composing W3-W6 (W7 — the brick that finally makes V4 runnable); the strict/near-tie engine gate (W8)
  — all multi-Spark-blocked on one GB10.

---

## W7 — the `DeepseekV4Model::Forward` ASSEMBLY (2026-07-29, `CLAIM-DEEPSEEK-V4-W7`)

**Base:** current `main` HEAD `a856383c` (`git rev-parse HEAD`). Isolated worktree
`/home/mudler/_git/vllm.cpp-w7-forward` (branch `deepseek-v4-w7-forward`), CPU-only, foreground, NOT
pushed. W7 replaces the `VT_CHECK(false, "W3-W8 pending")` stub with a REAL forward that COMPOSES the
four landed host-reference primitive stacks (W3 DSA indexer + 512-wide MLA output seams, W4
compressor + fp8_ds_mla KV, W5 MHC + Sinkhorn, W6 sqrtsoftplus/hash MoE) into an end-to-end logits
producer on the portable CPU path at a SMALL synthetic config. This is the brick that finally makes
DeepSeek-V4 **structurally runnable**.

### W7.1 What landed (owns `deepseek_v4.{h,cpp}` via `CLAIM-DEEPSEEK-V4-W7`)
- `include/vllm/model_executor/models/deepseek_v4.h` — the W7 host-float weight tower
  (`DeepseekV4LayerHostWeights` / `DeepseekV4HostWeights`, added to `DeepseekV4Weights` as
  `host` + `has_host_weights`), the `V4Miswire` RED-first enum, the `V4ForwardTrace` structural-facts
  struct, and the `DeepseekV4ForwardHost` declaration.
- `src/vllm/model_executor/models/deepseek_v4.cpp` — the REAL `DeepseekV4Model::Forward` (runs the
  host composition when `has_host_weights`, else a loud `VT_CHECK` naming the W2b materialization
  residual) + `DeepseekV4ForwardHost` (the composition) + `ForwardDevice` still a loud
  `VT_CHECK(false)` naming the W7-device residual.
- The interleave, grounded 1:1 (`nvidia/model.py:1080-1148` `DeepseekV4Model.forward` + `:866-957`
  `DeepseekV4DecoderLayer.forward`): embed → per layer [first-layer MHC-pre stream EXPAND
  `[T,H]→[T,hc,H]` via the broadcast residual, else fused MhcPost(prev-ffn) + MhcPre(attn)] → 512-wide
  MLA (q wq_a→q_norm→wq_b + kv wkv→kv_norm, RoPE, DSA indexer→topk→compressor→fp8_ds_mla KV, sink
  softmax, grouped o-LoRA) → fused MhcPost(attn) + MhcPre(ffn) → MoE (sqrtsoftplus/hash router +
  shared+routed clamped-SwiGLU) → final MhcPost(last-ffn) → hc_head collapse → norm → lm_head. The
  hash-vs-gated split (`num_hash_layers`), the per-layer `compress_ratio∈{0,4,128}` indexer/compressor
  topology, and the fp8_ds_mla KV round-trip are all exercised.

### W7.2 Gate — STRUCTURAL / composition (tiny shape), NOT a real-checkpoint token gate
`tests/vllm/models/test_deepseek_v4_forward.cpp`: **6/6 cases · 26 assertions GREEN** on a CPU Debug
full-library build (`-Wall -Werror -Wextra`-clean). A tiny V4 (H=8, 4 layers, `num_hash_layers=2`,
hc_mult=4, 2 heads × head_dim 6 = 4 NoPE + 2 RoPE, 4 routed + 1 shared expert, `index_topk=3`,
`compress_ratios={0,4,2,4}`) is built directly (bypassing the head_dim==512 parse validation) and the
assembled forward runs. Asserts: (a) the forward PRODUCES finite logits end-to-end, shape `[T,vocab]`,
DETERMINISTIC across runs; (b) the MHC stream is `[T,hc,H]` (`residual_stream_elems == T*4*H`,
hc_mult==4); (c) the hash layers route by `tid2eid` and the gated layers by learned top-k
(`layer_hash_routed == {1,1,0,0}`); (d) the DSA sparse path SELECTS (indexer on layers 1,3, `index_topk`
= 3 keys for the last query) and the compressor POOLS (layers 1,2,3); (e) `logits_indices` gathers the
requested rows. **RED-first PROVEN three levers**, each changes the output vs the faithful interleave:
route hash layers as gated (ignore `tid2eid`, `layer_hash_routed → {0,0,0,0}`), skip the final MhcPost
fold before the head collapse, and drop the per-head attention sink (plain softmax). SACRED-inert: only
`deepseek_v4.{h,cpp}` + the new test + the CMake test wiring changed; the shared MLA/MoE, the W3-W6
primitive TUs, README/Metal untouched — the four prior V4 tests still 4/40 + 13/38 + 12/164 + 14/125 +
12/716. CAVEAT: CPU Debug (the pre-existing GCC-13 `-O2` `-Werror=array-bounds` voxtral.cpp false
positive forces Debug; the new TU is strict-clean).

### W7.3 Honest 3-state + documented tiny-vs-167B divergences
The CPU forward assembly at tiny shape = **DERIVED + BUILD-VERIFIED (structural)**. This does NOT claim
V4 "runs" a real model — it claims the forward ASSEMBLES + is structurally gated at tiny shape. Where
the tiny forward must diverge from the fixed-config 167B (documented in `deepseek_v4.cpp`, not silent):
(i) the compressor pools a fixed W=2 window, not the real `(1+overlap)*compress_ratio` window (the
state-cache gather addressing is a W7-device concern); (ii) the MLA value is the full decoded latent
(the W_UK/W_UV absorption geometry is the shared-mla-extraction W7 follow-on); (iii) a single
`rope_theta` is used (the compressed layers' dual `compress_rope_theta` is a device-RoPE seam); (iv)
`quant_block == nope_head_dim` (one block) at tiny width. Each reuses the SAME landed primitive math the
device kernels will call.

### W7.4 Named residuals (what still blocks a real single-Spark IQ2_XXS-GGUF run)
- **W7-device:** the CUDA kernels (MHC Sinkhorn, DSA indexer/compressor, sqrtsoftplus router, clamped
  SwiGLU; the expert GEMM REUSES the existing NVFP4/FP8 grouped-GEMM) + `ForwardDevice`.
- **W2b:** materialize the real-checkpoint FP8-block MLA linears + NVFP4 grouped experts into the host
  tower (or the device towers); `Forward` currently `VT_CHECK`s the host tower present.
- **W8:** the full paged-engine strict/near-tie gate vs a real golden — multi-Spark-blocked (156.7 GiB
  NVFP4 / 167 GiB fp8 do not fit ONE GB10).
- **The single-Spark IQ2_XXS-GGUF vehicle** additionally needs the GGUF `blk.N.*` name-map (W2, now
  only needs the checkpoint downloaded to the DGX to read its full 1328-tensor manifest — see the
  GGUF-loadability section) on top of W7-device + W8; the IQ2_XXS/Q2_K dequant already landed
  (`CLAIM-DSV4-GGUF-LOADER`).

---

## W7-device — the CUDA kernels for the 4 new op families (2026-07-29, `CLAIM-DEEPSEEK-V4-W7-DEVICE`)

**Base:** `main` HEAD `33016f34`. Isolated worktree `/home/mudler/_git/vllm.cpp-w7-device` (branch
`deepseek-v4-w7-device`), CPU `-Werror` build-verify + **DGX GB10 CUDA gate under `flock /tmp/gpu`**
(worker stopped for the run then restored), foreground, NOT pushed. W7-device is the last
engineering brick before an actual DeepSeek-V4 run: it lands the CUDA kernels for the four
genuinely-new V4 op families, each unit-gated on the GB10 against the LANDED host reference (the
oracle), and wires `DeepseekV4Model::ForwardDevice` to compose them.

### W7-device.1 What landed (kernel-matrix row `KERNEL-DSV4-W7-DEVICE`, `SPIKE`)
New TU `src/vt/cuda/cuda_deepseek_v4.cu` — CUDA kernels + host-vector launchers + OpProvider
registration for the four families, each a 1:1 device port of the host reference:
- **MHC** (`deepseek_v4_mhc.{h,cpp}`): Sinkhorn (row-softmax+eps seed → col-norm → (iters-1)×
  [row,col]), MhcPre (folded RMSNorm projection → pre/post/comb gates → stream collapse → optional
  attn/ffn RMSNorm fold), MhcPost (comb mix + post gate), HcHeadCollapse.
- **DSA** (`deepseek_v4_dsa.{h,cpp}`): indexer weight-fold, weighted-MQA ReLU logits over the causal
  window (`-inf` out-of-window), causal top-k selection (bit-exact ids), per-head attention-sink
  softmax, grouped output-LoRA (`wo_a` bmm → `wo_b`).
- **Compressor** (`deepseek_v4_compressor.{h,cpp}`): softmax-window pool + RMSNorm, save-time APE add,
  **fp8_ds_mla** KV encode (per-64 UE8M0 power-of-two block scale + e4m3 nope, bf16 rope) / decode.
- **MoE** (`deepseek_v4_moe.{h,cpp}`): sqrtsoftplus score, noaux_tc bias-for-selection top-k OR
  `tid2eid` hash bypass (bit-exact ids, weights from UNBIASED scores), clamped SwiGLU.

**Seam:** each family registers ONE OpProvider under a dedicated OpId (`vt/ops.h`:
`kDeepseekV4{Mhc,Dsa,Compressor,Moe}`) whose `fn` points at a family kernels-struct
(`include/vllm/model_executor/models/deepseek_v4_device.h`); `deepseek_v4_device.cpp` resolves them
via `GetOp`. **NOT re-ported:** the 512-wide MLA attention + expert grouped-GEMM REUSE the existing
NVFP4/FP8 kernels (`cuda_mla_attn.cu`, `cuda_moe*.cu`) — only the four NEW glue families need device
kernels.

**ForwardDevice wiring:** `deepseek_v4.cpp` refactors the W7 composition into ONE
`ForwardComposeImpl` driven by a `V4Backend` policy; `DeepseekV4ForwardHost` binds the host refs (the
oracle, unchanged — `test_deepseek_v4_forward` still 6/6·26), `DeepseekV4Model::ForwardDevice` binds
the CUDA kernels through the seam. The small linear projections stay host in both modes (the real
device path REUSES the existing GEMM/MLA/MoE-grouped kernels — a documented seam).

### W7-device.2 Gate — DGX GB10 (sm_121a) RUNTIME-VERIFIED
`tests/vllm/models/test_cuda_deepseek_v4.cpp` **11/11 cases · 153 assertions GREEN** on the GB10:
each device kernel vs its host-ref oracle at small shape — BIT-EXACT ids (DSA causal top-k,
sqrtsoftplus/hash router selection), `-inf` mask exact, near-tie rel-L2 < 1e-4 for the fp reductions
(Sinkhorn, pool/softmax, sqrtsoftplus — device `expf`/`sqrtf`/`rsqrt` vs host), fp8_ds_mla
encode→decode within the e4m3 granularity bound + bf16 rope bit-exact; PLUS the **ForwardDevice
composition gate** (device forward == host forward, rel-L2 < 2e-3 over the 4-family tiny-config
interleave). **compute-sanitizer memcheck 0 errors.** **RED-first PROVEN:** dropping the sqrt in the
device sqrtsoftplus fails 3 cases / 6 assertions (sqrtsoftplus + router weights + ForwardDevice);
revert restores 11/11·153. Build: CUDA `-Werror` clean; CPU `-Werror` clean. The pre-existing GCC-13
`-O2` `-Werror=array-bounds`/`-Wstringop-overflow` FALSE POSITIVE in `voxtral.cpp` (project #155) was
neutralized MINIMALLY + LOCALLY with a scoped `#pragma GCC diagnostic` around the in-bounds
`BuildPaddedDecodeAttn` copies (advances #155).

### W7-device.3 Honest 3-state + residuals
The CUDA kernels are **RUNTIME-VERIFIED on GB10 at small shape** (real GPU, real host-ref oracle).
This does NOT claim V4 "runs" a real model. Named residuals: **W2b** — materialize the real-checkpoint
FP8-block MLA linears + NVFP4 grouped experts into the device towers (`Forward` VT_CHECKs the host
tower present); **W8** — the full paged-engine strict/near-tie gate vs a real golden (multi-Spark:
156.7 GiB NVFP4 / 167 GiB fp8 do not fit ONE GB10); the **single-Spark IQ2_XXS-GGUF vehicle**
additionally needs the GGUF `blk.N.*` name-map (W2, checkpoint-download-blocked on the 1328-tensor
manifest).

---

## W8 — single-Spark keep-quant memory enabler + GGUF name-map (2026-07-29, `CLAIM-DEEPSEEK-V4-W8`)

**Base:** `main` HEAD `4d618f59`. Isolated worktree `/home/mudler/_git/vllm.cpp-w8-run` (branch
`deepseek-v4-w8-run`), CPU-only Debug gate; DGX SSH used ONLY read-only (host/disk/mem check) + HF
HTTP-range header fetch. Foreground, NOT pushed. W8 was scoped as "the actual single-Spark run +
benchmark"; this lane lands the two GATING prerequisites of that run and reports the run itself as
the honest residual (it is blocked on unimplemented W2b code, not just memory/download).

### W8.1 The keep-quant memory enabler (the crux) — LANDED + GATED
The stated crux: DeepSeek-V4-Flash is 158 B params; the `UD-IQ2_XXS` GGUF is ~91 GiB **because the
weights stay ~2-3-bit**. Dequant-to-bf16 would need ~316 GiB and OOM-reboot the DGX (unified pool).
So the routed experts MUST keep-quant. Our engine did keep-quant for the six k-quant block types via
`kMatmulBTQuant` but IQ2_XXS/Q2_K were DEQUANT-ONLY (`HasQuantDotKernel` false → expand-to-bf16).

W8 adds the keep-quant `vec_dot` for the codebook encodings, each a 1:1 port of the ggml reference
(`llama.cpp @ 237ad9b96 ggml/src/ggml-cpu/quants.c`):
- `VecDotIQ2_XXSQ8_K` <- `quants.c:855` `ggml_vec_dot_iq2_xxs_q8_K_generic`;
- `VecDotIQ3_XXSQ8_K` <- `quants.c:999` `ggml_vec_dot_iq3_xxs_q8_K_generic`;
- `VecDotQ2_KQ8_K`    <- `quants.c:514` `ggml_vec_dot_q2_K_q8_K_generic`.
Plus: the shared codebook tables (`cpu_quant_iq_tables.h`: `iq2xxs_grid`/`iq3xxs_grid`/`ksigns`/`kmask`,
moved out of the dequant anon-namespace so the vec_dots share ONE definition), `DequantIQ3_XXS`
(`ggml-quants.c:2503`, the reference for the gate + the expand fallback), the `kIQ3_XXS` dtype +
geometry (id 18, 98 B block) + three block structs (`BlockQ2_K`/`BlockIQ2_XXS`/`BlockIQ3_XXS`), three
Q8_K traits rows (making `HasQuantDotKernel` TRUE for all three), the two exhaustive `DType` switches
in `ops.cpp`, and the ggml id-18 sizing trait in `gguf_reader.cpp`.

**HONEST CORRECTION of the brief ("IQ2_XXS + Q2_K"):** reading the REAL `UD-IQ2_XXS` manifest, the
routed experts are **IQ2_XXS** (`ffn_gate_exps`/`ffn_up_exps`) + **IQ3_XXS** (`ffn_down_exps`) — Q2_K
is the sibling `UD-Q2_K_XL` vehicle, NOT in the IQ2_XXS build. IQ3_XXS is therefore an EQUAL gating
prerequisite (without it `ffn_down_exps` alone OOMs). All three landed so both vehicles keep-quant.

**HONEST finding on "the CUDA path":** there is NO CUDA keep-quant `vec_dot` for ANY k-quant —
`kMatmulBTQuant` is registered on `kCPU` alone (verified: no CUDA vec_dot/mmvq/mmq in `src/vt/cuda`).
Keep-quant is a CPU-tier feature; on GB10 it runs on the 20 ARM cores against the unified pool, so
IQ2_XXS/IQ3_XXS/Q2_K match the six existing k-quants exactly. Extending "the CUDA path" is N/A.

**Gate (`tests/vt/test_ops_quant_dot.cpp`, the same machinery as the six k-quants):** the three types
added to `kWeightCases`. **19 cases / 130444 assertions GREEN** (CPU Debug, new TUs `-Werror` clean):
`vec_dot` vs an INDEPENDENT f64 dequant-then-dot (≤1e-5·L1 — a tight, cancellation-robust bound),
`MatmulBTQuant` NMSE ≤5e-4 vs dequant-f32, bit-exact across thread counts 1/2/4, ragged-K rejection.
**RED-first PROVEN:** perturbing the IQ2_XXS `0.125` fold (→`0.130`) fails 2 cases / 18 assertions,
revert restores 19/130444. This is REUSABLE beyond V4 (any IQ2_XXS/IQ3_XXS/Q2_K GGUF now keep-quant).

### W8.2 The GGUF `blk.N.*` name-map + FULL coverage — LANDED + GATED
The real 1328-tensor manifest was read from the shard GGUF HEADERS via HF HTTP-range (shard-1 full 5.25 MB
+ 4 MB of shards 2/3) — **no 91 GB download**. Confirmed `general.architecture=deepseek4`,
`split.tensors.count=1328`, and the full `deepseek4.*` config-KV (block_count=43, hash_layer_count=3,
expert_count=256, key/value_length=512, q_lora_rank=1024, output_group_count=8, sinkhorn_iterations=20,
indexer head=64/key=128/top_k=512, compress_rope_freq_base=160000). Topology (verified in the manifest):
43 layers, hash layers {0,1,2}, DSA indexer on the 21 `compress_ratio==4` layers ({2}∪even{4..42}),
compressor on the 41 `compress_ratio!=0` layers; 4 per-layer archetypes (2×24 + 1×34 + 20×28 + 20×34
tensors) + 6 top-level = 1328.

`scripts/check-dsv4-gguf-namemap.py` encodes the 41-slot `blk.N.*`→V4 name map (each slot tagged with
its `GgufTensorRole` so keep-quant residency is derivable) + the topology, GENERATES the full expected
tensor set, and asserts **EXACT set-equality** against the committed real manifest
(`scripts/dsv4_gguf_manifest_names.txt`): **1328/1328 mapped, 0 unmapped, 0 leftover, rc=0**. The V4
registry GGUF reject is replaced with a PRECISE message (keep-quant + name-map landed; W2b pending) —
an honest state advance, not a fake-accept path (W2b is unimplemented, so a full lift would misbehave).

### W8.3 The RUN — HONEST RESIDUAL (did NOT execute; NOT faked)
The run did not execute and was not simulated. It is blocked on the unimplemented **W2b**: materialize
the GGUF keep-quant blocks (+ the F32 MHC/DSA/norm tensors) into the `DeepseekV4` weight towers via the
name map — a genuine engineering brick, not just memory/download. (The DGX was also contended: 87 GiB
of the 119 GiB unified pool already in use by the LocalAI containers.) 3-state: the keep-quant kernels
+ name-map are **DERIVED + BUILD-VERIFIED + UNIT-GATED**; the run is **NOT DONE**. The model-matrix
DeepSeek-V4 row therefore stays `SPIKE`.

**Resume command (exact):** land W2b (GGUF→tower materialization through `check-dsv4-gguf-namemap.py`'s
map + the landed keep-quant `vt::DType`s) → on DGX: free disk ≥100 GiB, `flock $HOME/gpu.lock`,
`docker stop local-ai-worker` (restore `--restart=always`+start after), gpu_mem LOW → download
`unsloth/DeepSeek-V4-Flash-GGUF/UD-IQ2_XXS` (3 shards, ~91 GB) → keep-quant load into `DeepseekV4Model`
→ `ForwardDevice` greedy gen → gate = our-engine self-consistency (run-to-run deterministic greedy) +
coherent output; benchmark TPOT/throughput/peak-mem, llama.cpp-on-card as the competitor floor if it
loads the same GGUF, else ours-only honestly.

### W8.4 Landed-vs-residual
- **Landed (W8):** keep-quant `vec_dot` for IQ2_XXS/IQ3_XXS/Q2_K (`KERNEL-QUANT-CIQ-IQUANT`, the memory
  enabler, gated + RED-first) + the `blk.N.*`→V4 name-map with EXACT 1328-tensor coverage + the precise
  registry message.
- **Residual:** W2b (GGUF→tower materialization) → then the RUN (download + GB10 greedy gen +
  self-consistency/coherence gate + benchmark). W2b is the single remaining engineering brick before a
  real single-Spark DeepSeek-V4 generation.

### W8.5 W8-final — entrypoint wiring LANDED + GATED; the RUN re-scoped to a CODE blocker (2026-07-29, `CLAIM-DEEPSEEK-V4-W8`, base `376e186b`)
W2b (GGUF→tower materialization) landed at `376e186b`, so this lane took the two remaining W8-final
pieces: the **entrypoint wiring** (the "one small code piece W2b named") and the **real run**.

**Entrypoint wiring — LANDED + GATED.** A `deepseek4` arm now exists in the top-level GGUF dispatch:
- `DeepseekV4HfConfigFromGguf(gguf)` (`deepseek_v4_weights.cpp`, decl in `deepseek_v4.h`) resolves the
  params via the existing `DeepseekV4ParamsFromGguf`, sets `architectures = {"DeepseekV4ForCausalLM"}`
  (mapping llama.cpp's `general.architecture=deepseek4` onto the registered vLLM model class, exactly as
  `HfConfigFromGguf` maps the `qwen35*` keys), and republishes the geometry into the typed fields +
  `config.raw` so the registry parse hook `ParseDeepseekV4Config` validates the same geometry.
- `LoadedEngine::FromModelDir` now calls `HfConfigFromGgufDispatch(gguf)` (anon-ns helper) which peeks
  `general.architecture` and routes `deepseek4`→the V4 builder, everything else→`HfConfigFromGguf`.
  Downstream (`ModelRegistry::Resolve` → tokenizer → `Load`→`LoadDeepseekV4FromGguf`) is unchanged.
- **Gate:** `test_deepseek_v4_gguf_load` **6/6 · 168** (CPU Release, full-library `-Werror`-clean build):
  new case asserts `architectures[0]=="DeepseekV4ForCausalLM"`, `model_type=="deepseek4"`, the `raw`
  scalars the parse hook reads (`hc_mult`, `n_routed_experts`, `scoring_func=sqrtsoftplus`,
  `expert_dtype=fp4`, `compress_ratios`), and `ModelRegistry::Resolve(c)`→the V4 factory
  (`is_dense_model=false`). The qwen GGUF path is byte-neutral (`test_model_registry` 24/24 green).

**The RUN — did NOT execute; re-scoped to a CODE blocker (NOT download/box), provable from source.**
The real single-Spark load was NOT attempted because it would OOM-reboot the box: the forward
(`ForwardComposeImpl`, both `Forward` and `ForwardDevice`) composes off the FULLY-DEQUANTIZED f32
`weights.host` tower, and `LoadDeepseekV4FromGguf` builds that host tower **unconditionally** — every
routed-expert tensor via `HostVec`→`DqRowF32`→f32, retained per layer in `hw.layers[l].exp_w1/w3/w2`.
For the real 43-layer/256-expert/mi=2048/H=4096 model that is 3×256×2048×4096×4 B ≈ **~24 GiB per
layer**, ≈ **~1.0 TiB** across 43 layers — it exceeds the 119 GiB unified pool at ~layer 5. The
keep-quant `weights.gguf` tower (~91 GiB, the W2b memory enabler) IS built but is **never read** by the
forward, so the enabler does not help the run. **Named residual (W2c):** rewire the forward to consume
the keep-quant blocks directly (via the CPU CIQ `kMatmulBTQuant` GEMM already landed) and gate off the
host-f32 dequant, so resident stays ~91 GiB. Only then is a single-Spark greedy gen + self-consistency
/coherence gate + benchmark feasible. NO tokens generated (not faked); benchmark disposition stays
PENDING; DGX left exactly as found (LocalAI worker untouched, no 91 GB download). Row stays `SPIKE`.

---

## W2b — GGUF keep-quant TOWER materialization (2026-07-29, `CLAIM-DEEPSEEK-V4-W2B`)

**Base:** `main` HEAD `341dfbb9` (the W8 keep-quant + name-map commit). Isolated worktree
`/home/mudler/_git/vllm.cpp-w2b-tower` (branch `deepseek-v4-w2b-gguf-tower`), CPU-only Debug,
foreground, NOT pushed. W2b is the LAST CODE brick before the real single-Spark run (W8-final): it
wires the landed GGUF `blk.N.*` name-map (`scripts/check-dsv4-gguf-namemap.py`, EXACT 1328/1328) +
the landed keep-quant `vec_dot` (IQ2_XXS/IQ3_XXS/Q2_K, CIQ) into the `DeepseekV4` weight towers, so
the model can actually LOAD from the `UD-IQ2_XXS` GGUF.

### W2b.1 What landed (loader-only, SACRED-inert)
- `include/vllm/model_executor/models/deepseek_v4.h` — the `DeepseekV4GgufWeights` /
  `DeepseekV4GgufLayerWeights` OwnedTensor tower (named slots mirroring the name-map), added to
  `DeepseekV4Weights` as `gguf` + `has_gguf_weights`; declarations for `DeepseekV4ParamsFromGguf`
  and `LoadDeepseekV4FromGguf`.
- `src/vllm/model_executor/models/deepseek_v4_weights.cpp` — the materialization (structurally
  mirroring the Qwen3.6 GGUF path `qwen3_5_gguf_weights.cpp`): `DeepseekV4ParamsFromGguf` resolves
  the `deepseek4.*` KV (block_count, hash_layer_count, expert_count, key_length=head_dim, q_lora_rank,
  output_group_count, sinkhorn_iterations, indexer head/key/top_k, compress_ratios, hyper_connection
  count, …); `LoadDeepseekV4FromGguf` routes EVERY GGUF tensor through `GgufLoadPolicy::Route` with
  its name-map role — **MW/SEW** (512-wide MLA linears wq_a/wq_b/wkv/wo_a/wo_b, router gate,
  shared + the 256 routed experts, lm_head) KEEP their ~2-3-bit blocks COMPRESSED via
  `OwnGgufQuantBlocks` (the ~91 GiB-vs-~316 GiB OOM memory enabler); **V/ET/HASH** (norms, MHC
  hc_*, DSA compressor/indexer, attention sinks, embed, the `tid2eid` hash table, `exp_probs_b` bias)
  dequant to f32 — plus the dequant BRIDGE that fills the tiny-config CPU composition `host` tower
  so a loaded model FORWARDs.
- `src/vllm/model_executor/models/deepseek_v4_registry.cpp` — the GGUF reject is LIFTED:
  `source.kind == kGguf` → `LoadDeepseekV4FromGguf(*source.gguf, config)`.
- `tests/vllm/models/test_deepseek_v4_gguf_load.cpp` — the structural gate (+ CMake wiring).

### W2b.2 The ACCOUNTING contract (the name-map coverage, encoded in C++)
The loader mirrors the Python name-map (`check-dsv4-gguf-namemap.py` PER_LAYER / HASH / GATED /
COMPRESSOR / INDEXER / TOP_LEVEL tables + topology) and TRACKS every consumed tensor name: a missing
required tensor throws (unmapped), and after materialization every file tensor MUST be in the consumed
set (a leftover the map does not cover throws). `accounted_tensors` == the file tensor count.
The real 1328-tensor manifest coverage stays gated separately by `check-dsv4-gguf-namemap.py` (rc=0);
this C++ gate proves the ROUTING + keep-quant residency + accounting + load→forward LOGIC at tiny shape.

### W2b.3 Gate — STRUCTURAL (tiny synthetic GGUF), NOT a real-checkpoint token gate
`tests/vllm/models/test_deepseek_v4_gguf_load.cpp`: **5/5 cases · 149 assertions GREEN** on a CPU
Debug full-library build (loader TUs `-Werror`-clean). A tiny synthetic `deepseek4` GGUF (built with
`gguf_builder.h`, the REAL `blk.N.*` naming convention + a REAL keep-quant ggml type **Q8_0** at tiny
dims: H=32, 4 layers, hash{0,1}, `compress_ratios={0,4,2,4}` so indexer{1,3}/compressor{1,2,3},
hc_mult=2, 4 routed + 1 shared expert) is materialized and asserted:
- (a) **accounting** — `accounted_tensors == 126 == g.Tensors().size()` (none unmapped, none
  leftover); per-layer hash/compressor/indexer topology mapped; the tower slots are non-empty;
- (b) **keep-quant residency** — the 256-expert down proj stays `vt::DType::kQ8_0` (block dtype,
  `nk=true`, shape `[E*out, in]`), byte size = the compressed 34 B/32-elem image `<` the dequant-f32
  image; an MLA linear + lm_head stay Q8_0; a norm stays F32;
- (c) **load → forward** — `DeepseekV4ForwardHost(w.host, …)` produces finite, deterministic logits
  end-to-end over the assembled W7 composition (proves load→forward works).
**RED-first PROVEN:** (i) a MISSING required tensor throws (unaccounted route); (ii) a LEFTOVER
tensor the map does not cover throws; (iii) an expand-instead-of-keep policy leaves the SAME weight
UNCOMPRESSED (bf16) — proving keep-quant is what compresses it. Q8_0 is used because it is a real
keep-quant type with an easy meaningful synthesis (the IQ2_XXS/IQ3_XXS/Q2_K keep-quant vec_dot is
gated by `test_ops_quant_dot`; the tower WIRING here is quant-type-agnostic — it routes by role via
`HasQuantDotKernel`, the identical path the i-quants take).

### W2b.4 Honest 3-state + the W8-final residual
The tiny-synthetic load→forward = **DERIVED + BUILD-VERIFIED (structural)**. It does NOT claim V4
"runs" a real model. The real **91 GB `UD-IQ2_XXS` load + generate stays W8-FINAL** (operational,
NOT this lane): download `unsloth/DeepSeek-V4-Flash-GGUF/UD-IQ2_XXS` (3 shards, ~91 GB) to the DGX
(free disk ≥100 GiB) → `flock $HOME/gpu.lock` + `docker stop local-ai-worker` (restore after) →
keep-quant load into `DeepseekV4Model` via `LoadDeepseekV4FromGguf` → `ForwardDevice` greedy gen →
gate = our-engine self-consistency (run-to-run deterministic greedy) + coherent output → benchmark
TPOT/throughput/peak-mem (llama.cpp-on-card as the competitor floor if it loads the same GGUF, else
ours-only honestly). **Resume command:** `LoadedEngine::FromModelDir` also needs a `deepseek4` arm in
the top-level `HfConfigFromGguf` dispatch (today qwen-only) to route the GGUF to this loader from the
CLI/server — a small entrypoint wiring folded into the W8-final run.
