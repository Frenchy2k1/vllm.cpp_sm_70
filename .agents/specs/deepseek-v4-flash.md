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
