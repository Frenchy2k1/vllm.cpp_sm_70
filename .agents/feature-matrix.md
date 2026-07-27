# Feature matrix — cross-cutting feature parity

**This document is the broad parity coverage view for cross-cutting engine, KV,
scale-out, sampling, serving, spec-decode, long-context, LoRA and loading.** The
stable-ID execution rows and exact code/test anchors live in
[engine-matrix.md](engine-matrix.md). The comprehensive model,
quantization, kernel and platform inventories live in
[model-matrix.md](model-matrix.md),
[quantization-matrix.md](quantization-matrix.md),
[kernel-matrix.md](kernel-matrix.md), and
[backend-matrix.md](backend-matrix.md). The ordered portfolio is
[roadmap_v1.md](roadmap_v1.md).

**Convention (delegation unit):** a planned-but-missing spec is not a spike.
Rows are being migrated to the stable-ID/evidence contract in
[coordination.md](coordination.md); a legacy implemented row without exact code,
test and real-spec anchors is `ANCHOR-BACKFILL`, not protocol-complete `DONE`.

**Lifecycle:** `INVENTORIED -> SPIKE -> READY -> ACTIVE -> GATING -> DONE`, with
`PARTIAL`, `BLOCKED`, and `ANCHOR-BACKFILL` as explicit non-done states. Legacy
emoji rows retain runtime meaning during the evidence migration but cannot be
used as new claims. Tier (T0–T3) remains per porting-inventory.

**Mirror policy:** vLLM parity is the FLOOR for every row (mirror-vllm-always);
the surpass track (roadmap_v1.md "Protocol evolution") builds beyond it, never
instead of it.

---

## 1. Engine core & scheduling

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| Continuous batching / unified scheduler (token-budget, no prefill/decode split) | `v1/core/sched/scheduler.py` | `ANCHOR-BACKFILL` T0 | proven text-only running-first/FCFS/token-budget slice; broad scheduler parity open | `planned: specs/unified-scheduler.md` |
| Chunked prefill (on by default) | `config/scheduler.py` | `ANCHOR-BACKFILL` T0 | basic token-budget chunking; partial-prefill concurrency/MM modes open | `planned: specs/chunked-prefill.md` |
| Prefix caching (APC) | `v1/core/kv_cache_utils.py`, `v1/core/kv_cache_coordinator.py` | `DONE` T0 (dense APC path) | W0 no-prefix coordination + default resolution + CLI override; W1 hit-rate stats; W2 block-hash `extra_keys` (mm/LoRA/`cache_salt`) no-false-share RED-proven; **W3 DONE 2026-07-27 — the FIRST-EVER cache-ON model gate** (`test_qwen3_apc_e2e.cpp` on Qwen3-4B dense, dgx GB10): APC-ON==APC-OFF token-exact 5/6 (1 diff = vLLM-confirmed 0.125-nat near-tie), == vLLM-APC-ON teacher-forced (OFF gap 0.0 / ON gap ≤0.125 nats), hits 2240/2777 (0.807), TTFT 70.1→39.9 ms = 1.76×; NO engine code changed (gate-only). Named non-blocking tails (own rows/future): W4 events (`KV-EVENTS`), W5 partial-block, W6 Mamba-`align` (`KV-MAMBA-ALIGN`), W7 reset endpoint, W8/W9 beyond-vLLM save-restore; every-axis cache-on grid = perf follow-on | [prefix-prompt-caching-parity.md](specs/prefix-prompt-caching-parity.md) |
| Preemption (FCFS tail pop, recompute) | `v1/core/sched/scheduler.py` | `ANCHOR-BACKFILL` T0 | FCFS recompute slice | `planned: specs/preemption.md` |
| CUDA graphs (decode capture/replay, no torch) | `config/compilation.py::cudagraph_mode` | `PARTIAL` T0 | Qwen-specific capture; W3-G `ae9e8ff` now passes FA2 cold/capture/replay/capacity/two-queue tests, strict zero-leak memcheck and paired node traces. Trace shows 224 graph main+combine calls, no capture allocation/free/sync, zero graph D2H and three eager fixed-scratch allocations. Generic modes and broader direct evidence remain open | [FA2 decode spike](specs/fa2-gqa-split-kv-decode.md); `planned: specs/cuda-graphs.md` |
| FA2 paged GQA pure-decode split-KV | `v1/attention/backends/flash_attn.py`; dependency `flash_api.cpp` | `ACTIVE` T0 | Exact 27B ratio-6 BF16/D256 adapter, swap strides, split heuristic, queue-owned graph-stable scratch, model/dispatcher toggle and ported tests pass immutable sm_121a operator/memcheck/model/trace gates at `ae9e8ff`. Default/fallback switch **240 main+combine / 0 old** versus **0 combine / 240 old**. The completed c2/c16 A/B is **1.017668×/1.006548×** mean total throughput but strict-fails **35/40 timing + 5/8 memory**, so no speed credit/exact grid follows. Ratio-8 and other modes intentionally fall back | [FA2 decode spike](specs/fa2-gqa-split-kv-decode.md) |
| GDN packed pure decode | `envs.py::VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE`, `fla/ops/fused_recurrent.py` | `ACTIVE` T0 | `KERNEL-GDN-PACKED-DECODE` mirrors vLLM's default pure non-spec branch across FP16/BF16/F32. Clean `f344dec` closes correctness and `7ff713e` + `24cea4f` close structure. Clean `d82d282` passed model gates/all c2 legs, then failed incomplete at c16 packed r1 with 96/96 HTTP 500 responses and no marker. Partial legs earn no speed credit | [gdn-packed-decode.md](specs/gdn-packed-decode.md) |
| Opt-in batch-invariant execution (`VLLM_BATCH_INVARIANT=1`) | `envs.py`, `model_executor/layers/batch_invariant.py`, `csrc/.../nvfp4_scaled_mm_sm120_kernels.cu` | ☐ T1 | v0.25 production default is off; the opt-in mode changes matmul/norm/attention/collective and NVFP4 dispatch to make one request invariant to neighboring batch rows. W3-C3R executes the default-off contrast for ours and vLLM but does not implement the opt-in feature | `planned: specs/batch-invariant-execution.md` |
| **Async/overlap scheduling** | `v1/core/sched/async_scheduler.py` | ☐ T1 (**promoted**), spike accepted, full stack LANDED CPU-gated | Full W3 stack landed (host machinery + runner input/output halves + enable-flip), default OFF. `f086b64` DGX proof: 5/5 correctness gates PASS; c16 A/B W3-on meanTPOT **−5.4 ms/step (win)** but meanTTFT **+36% (+730 ms)**, throughput neutral (−0.3%). **Admission-delay hypothesis REFUTED** (`test_async_admission_timing.cpp`): depth-2 schedules a new prefill the SAME step as sync/vLLM (`UniProcExecutor` is synchronous too, `uniproc_executor.py:91-106`); the +730 ms is the closed-loop Little's-law consequence of neutral throughput + faster decode (`127×5.4≈686`). Real gate = a depth-2 **throughput** lever (GPU/nsys), not CPU admission. Stays default-OFF. `ENG-ASYNC-SCHED` remains `READY`, uncredited | [async-serving.md](specs/async-serving.md) |
| Priority scheduling (`policy="priority"`) | `v1/core/sched/request_queue.py` | 🚧 T1 `GATING` | W4 queue, priority preemption and request/config plumbing implemented; CPU suites green, GB10 G1 token-exact A/B pending | [async-serving.md](specs/async-serving.md) |
| Partial-prefill concurrency (`max_num_partial_prefills`, long-prefill limits) | `config/scheduler.py` | ☐ T1 | | `planned: specs/partial-prefill-concurrency.md` |
| `step_with_batch_queue` (pipelined batch queue) | `v1/engine/core.py` | ☐ T1 | | `planned: specs/batch-queue-step.md` |
| `scheduler_reserve_full_isl`, pluggable `scheduler_cls`, `stream_interval` | `config/scheduler.py` | `PARTIAL` T1 | fields/internal handling exist; scheduler registry and public stream-interval wiring are absent | `planned: specs/scheduler-knobs.md` |
| Cascade attention (shared-prefix batches) | `config/model.py::disable_cascade_attn` | ☐ T2 | | `planned: specs/cascade-attention.md` |
| DBO / ubatch overlap | `config/parallel.py::enable_dbo` | ☐ T2 | | `planned: specs/dbo-ubatch.md` |

## 2. KV cache & memory

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| Paged KV: BlockPool (free list, refcount, LRU evict) | `v1/core/block_pool.py` | `ANCHOR-BACKFILL` T0 | core free-list/refcount/cache/LRU slice; connector/partial primitives separate | `planned: specs/block-pool.md` |
| KVCacheManager (allocate_slots, watermark, admission) | `v1/core/kv_cache_manager.py` | `ANCHOR-BACKFILL` T0 | named allocation/admission/lookahead slice | `planned: specs/kv-cache-manager.md` |
| Device-resident attention KV + indexed GDN state I/O | `v1/worker/gpu/attn_utils.py`, `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` | `ACTIVE` T0 | W0 stable allocations/fallback and W1 indexed BF16↔F32 gather/scatter remain correctness/safety green with separate 1.021239×/1.006246× component A/Bs and structural copy reduction. Clean `b5c6e4f` re-ranks the exact residual: W2 FP4 improves every concurrency and wins c16/c32 throughput, but FP4 tactic-selection parity, c1-c8, TPOT/ITL and host-memory axes remain ahead of device-residency W2. Inherited pools still fail zero-leak | [device-resident-kv-gdn-state.md](specs/device-resident-kv-gdn-state.md) |
| Hybrid KV coordinator (full-attn + GDN/mamba state groups) | `v1/core/kv_cache_coordinator.py` | `PARTIAL` T0 | cross-group MIN-intersection prefix hit; align/retention modes absent | `planned: specs/hybrid-kv-coordinator.md` |
| Mamba/GDN prefix-cache retention (`mamba_cache_mode=align`, vllm#45845) | `v1/core/` | ☐ T1 (**promoted into competitor gate**) | vLLM hybrid defaults off but explicit cache-on selects align for Qwen3.5/3.6; local 1:1 stub exists, full retention/runtime path is required for matched vLLM/SGLang shared-prefix gating | `planned: specs/mamba-align-retention.md` |
| SlidingWindowSpec + ChunkedLocalAttentionSpec | `v1/kv_cache_interface.py` | `PARTIAL` T1 | Both execution leaves are implemented: W1 sliding-window and W3 chunked-local sizing, registry/grouping, manager prefix/recycling policy, admission and hybrid-disabled conversion pass their ported CPU/property/sanitizer gates (G1/G2). The compute-locality consumers are now GPU-gated (2026-07-27 `CLAIM-ROADMAP-C5`, dgx GB10: Gemma-2/Gemma-3 sliding-window model gates 48/48; `test_chunked_local_attention` 5/5). The KV memory-OPTIMIZATION path (optimized-manager held-block cap vs the full-allocation fallback the current model gates use) still needs a model-level hybrid-manager memory gate (G8) — kept `PARTIAL` honestly | [sliding-local-yarn-long-context.md](specs/sliding-local-yarn-long-context.md) |
| fp8 KV cache (`cache_dtype=fp8*`) | `layers/quantization/kv_cache.py` | ☐ T1 | | `planned: specs/fp8-kv-cache.md` |
| nvfp4 / per-token-head / turboquant KV | `config/cache.py` | ☐ T2 | | `planned: specs/nvfp4-kv-cache.md` |
| KV offload (CPU tiering, LRU/ARC) | `v1/kv_offload/` | ☐ T2 | | `planned: specs/kv-offload.md` |
| External KV-cache provider ABI + LMCache (MP service and in-process connectors) | `config/kv_transfer.py`, `distributed/kv_transfer/kv_connector/v1/{base,lmcache_connector,lmcache_mp_connector}.py` | ☐ T2 | explicit roadmap outcome `KV-EXTERNAL-CACHE`: mirror `kv_producer`/`kv_consumer`/`kv_both`, scheduler/worker metadata, async layer load/store, dynamic external connector modules, failure policy, metrics and cache-lifecycle ownership; gate the official LMCache shared-prefix quickstart plus Qwen3.6 hybrid behavior | `planned: specs/external-kv-cache-lmcache.md` |
| KV connector breadth (NIXL/Mooncake/MultiConnector, **PD disaggregation**) | `distributed/kv_transfer/` | ☐ T2 | builds on the external-provider base seam; remains a separate breadth/scale-out leaf | `planned: specs/kv-connectors-disagg.md` |
| KV events (block create/evict publish) | `config/kv_events.py` | ☐ T2 | | `planned: specs/kv-events.md` |
| MLAAttentionSpec (latent KV) | `v1/kv_cache_interface.py` | ☐ T2 | with DeepSeek family | `planned: specs/mla-kv-spec.md` |
| Sizing: `gpu_memory_utilization`, block overrides | `config/cache.py` | `PARTIAL` T0 | watermark/fixed loader inputs exist; public utilization/cache-byte/override policy absent | `planned: specs/kv-sizing.md` |
| Weight CPU offload (`cpu_offload_gb` UVA per-parameter + layer-group `PrefetchOffloader`) | `config/offload.py`; `model_executor/offloader/` | ☐ T2 | v1-supported at the pin; blanket/name-targeted, NOT router-aware; mirror floor for expert streaming (engine row `ENG-WEIGHT-OFFLOAD`) | `planned: specs/weight-offload-uva.md` |
| Expert streaming from disk (routed-MoE experts paged NVMe→GPU on router output, budgeted resident cache) | absent in-pin (surpass-track); design ref antirez/ds4 | ☐ T2 | corrected engine row `ENG-EXPERT-STREAM` READY: bank-only loader, fixed contiguous Marlin slots, logical→slot remap, explicit router D2H, chunked prefill; W0 trace/baseline first | [expert-streaming.md](specs/expert-streaming.md) |

## 3. Parallelism & scale-out

No distributed runtime in-tree yet (verified: no NCCL/`tensor_parallel` in `src/`).
Upstream's Ray/multiproc executor is replaced by the in-process seam (§9.2) —
multi-process/multi-GPU re-enters through that seam.

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| Tensor parallel (TP) | `distributed/`, `config/parallel.py` | 🚧 T2 (spec written, task #50) | weight-stacking semantics already ported (single-GPU shape); ⚠ impl needs a 2-GPU box (GB10 is single-GPU) | [specs/tensor-parallelism.md](specs/tensor-parallelism.md) |
| Pipeline parallel (PP) | same | ☐ T2 | | `planned: specs/pipeline-parallel.md` |
| Expert parallel (EP) + EPLB | `v1/worker/gpu/eplb_utils.py` | ☐ T2 | | `planned: specs/expert-parallel.md` |
| Data parallel (DP) | `config/parallel.py` | ☐ T2 | | `planned: specs/data-parallel.md` |
| MoE sequence parallelism without DP | `config/parallel.py::use_sequence_parallel_moe`, `distributed/parallel_state.py` | ☐ T2 | v0.25.0 adds the non-DP path and reports 1.9–5.0% E2E throughput; inventory row `PAR-SEQUENCE-MOE` | `planned: specs/sequence-parallel-moe.md` |
| Multi-node (Ray / multiproc executor) | `v1/executor/` | ☐ T3 | re-enters via §9.2 seam; not as-is | `planned: specs/multi-node.md` |

## 4. Model families

The current pin contains 353 unique static architecture IDs; the audited
v0.25.0 target adds three, for 356 after pin advancement, plus a dynamic
Transformers-compatibility path. The complete alias-preserving inventory is
[model-matrix.md](model-matrix.md); this summary is only the roadmap roll-up.

| ID | Category / mechanism | Pinned upstream | State | Grounded summary | Detailed evidence / spike |
|---|---|---|---|---|---|
| `MODEL-GATE-QWEN35` | Qwen3.5/3.6 dense + MoE wrappers | `models/registry.py:556-560` | `PARTIAL` | 27B text submodel STRICT + 27B image+video e2e STRICT 32/32 (M3-b/M3d: ViT/merger present); 35B text submodel STRICT, its ViT/merger still pending | model matrix + existing Qwen specs |
| `MODEL-FACTORY` | architecture -> model factory + reject unknown | `models/registry.py:998-1404` | `PARTIAL` | central ordered type-erased registry now covers both implemented Qwen IDs; live loading resolves the full `architectures` list and rejects unknown/previous/OOT IDs with pinned messages instead of using `num_experts`; execution row `MODEL-FACTORY-registry` is `GATING` on the two-model GPU no-regression campaign | [model-factory-registry.md](specs/model-factory-registry.md) |
| `MODEL-TEXT` | 130 text-generation IDs | `models/registry.py:71-208` | `PARTIAL` | ~19/130 text archs supported+gated (Llama/Qwen3/Qwen3-MoE/Mistral/OPT/DeepSeek-V2/GLM-4/GLM-4.7-Flash/Granite/StableLM/InternLM2/InternLM3/Phi-1-2/Phi-3-4/MiniCPM/MiniCPM3/OLMo-2/Gemma-1-2-3/Yi); OLMo-3 oracle-blocked, Command-R gate-blocked | model matrix |
| `MODEL-POOLING` | embedding, late-interaction, reward, token/sequence classification | `models/registry.py:210-329` | `INVENTORIED` | five distinct work fronts; no pooling runtime | model matrix |
| `MODEL-MM` | 115 v0.25.0-target multimodal IDs | `models/registry.py:331-583` | `PARTIAL` | image+video e2e (Qwen3-VL-4B + 27B `Qwen3_5ForConditionalGeneration` STRICT 32/32) + audio e2e (Voxtral-Mini-3B 14/14) + Whisper encoder 203/203 LANDED; processors + encoder-cache present; speed: tower lever #1 + decode lever #2 (on-GPU greedy argmax + no embed round-trip, bit-exact, 2026-07-27 `CLAIM-MULTIMODAL-SPEED-DECODE`) CLOSED; lever #3 (batched/graphed mm serving) FIRST BRICK landed 2026-07-27 (`CLAIM-MULTIMODAL-SPEED-GRAPH`) — 27B image+video decode now routes through the production `Qwen3_5DenseDecodeGraph` (graph-capturable, token-exact 32/32 held, NEUTRAL at the 27B bandwidth floor); W1 landed 2026-07-27 (`CLAIM-MM-SPEED-GRAPH-W1`) — new `VoxtralDecodeGraph` graph-captures the Voxtral audio decode (bit-exact 14/14 held; A/B 60.94 vs 61.71 ms/tok, non-overlapping, but NARROWS the gap 1.52×→1.49×, does NOT close it — residual is per-step compute, not launch overhead); decode-kernel efficiency ATTRIBUTED + VALIDATED ceiling 2026-07-27 (`CLAIM-MM-SPEED-DECODE-KERN`, §11) — nsys shows the whole ~20 ms/tok residual is the naive scalar `PagedAttentionKernel` decode attention (723 µs × 30 = 21.7 ms/step); the 1:1 vLLM lever (FA2 `flash_attn_varlen` decode) is already in-binary, gated off only because the driver's single KV block (444) isn't ÷16; `block_size÷16` → TPOT 59.4→38.2 ms/tok (−21.2, ~36%) = 0.94× vLLM 40.8 ms (BEATS parity) and the FA2 sequence is a VALID vLLM greedy branch (teacher-force PASS, gap 0.0), but it flips the committed near-tie golden → blocked byte-exact (bf16 near-tie / golden-pinning ceiling; RECORDS-ONLY, 14/14 held, win reachable via block_size÷16 + golden regen); batched c2+ + `image_url`/`audio_url` serving ingestion are the remaining W-plan; 35B ViT and broader mm breadth pending | model matrix |
| `MODEL-SPEC` | 46 v0.25.0-target speculative-draft IDs | `models/registry.py:585-638` | `ACTIVE` | `Qwen3_5MTP` (27B) MTP k=1 `DONE` 2026-07-26; `Qwen3_5MoeMTP` (35B) MoE-MTP `DONE` 2026-07-26 (`383f03db`); `DFlashDraftModel` `DONE` 2026-07-27 (`04dff573`, speed gate met 1.003x; the pin advance to 0.26.0.dev0 resolved vllm#40898); the remaining 43 IDs INVENTORIED | model matrix + [spec-decode specs](specs/mtp-spec-decode.md) |
| `MODEL-TRANSFORMERS` | 14 static aliases + dynamic compatible classes | `models/registry.py:635-680,1096-1164` | `INVENTORIED` | C++ policy/factory not spiked | model matrix |

## 5. Quantization

The canonical leaf inventory is
[quantization-matrix.md](quantization-matrix.md). It separately tracks format
recognition, materialization/repack, native quantized compute, real-model gates
and reference-engine performance.

| ID | Block | State | Grounded summary | Detailed evidence / spike |
|---|---|---|---|---|
| `QUANT-CUDA-GATES` | NVFP4 W4A16, NVFP4 W4A4, gate-specific FP8 W8A8 | `DONE` | support/correctness stays closed; performance remains `ACTIVE` at the current binding `9ecd9d0` **114/124**. FP4 tactics match, and clean `f344dec` closes W1D2/G2 with 27B **235/235** in default+rollback arms plus 35B/GGUF inertness. Component evidence remains open; no quantization speed credit follows | quant matrix §2 + [coverage spike](specs/quantization-coverage.md) |
| `QUANT-GGUF` | llama.cpp encodings and output presets | `PARTIAL` | F32/Q4_0/Q8_0/Q3_K/Q4_K/Q5_K/Q6_K materialize; F16 was corrected to reader-only; CPU threadpool/chunked dispatch is correctness-gated but its B4 speed/RSS checkpoint is pending; no direct compute-in-quant or llama.cpp speed parity | quant matrix §1 |
| `QUANT-VLLM-BREADTH` | generic FP8/MX, AWQ/GPTQ, CT integer, vendor methods, KV | `PARTIAL` | gate-specific implementations exist; generic dispatch/modes remain inventoried | quant matrix §§2-3 |
| `QUANT-MLX` | affine Q2-8, MXFP4/MXFP8/NVFP4, QQ, mixed recipes/imports | `INVENTORIED` | required for Apple backend; no MLX runtime yet | quant matrix §4 |

## 6. Sampling & generation controls

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| Core sampler pipeline (temperature, top-k/p, min-p, penalties, seed, n, stop, min/max_tokens, output_kind) | `v1/sample/sampler.py` | `ACTIVE` T0 | ordered pipeline + every control now WIRED params->metadata->sampler (ROAD-V1-C7: min_p/min_tokens/logit_bias/logprobs-count reach SamplingMetadata; CPU-gated). `n>1` still parsed-not-executed; random/logprob paths synchronize to host | [specs/sampling-controls-c7.md](specs/sampling-controls-c7.md) |
| torch-Philox bit-exact random parity | `v1/sample/ops/` | ☐ T1 | current RNG gumbel-max distribution-correct only | `planned: specs/philox-rng-parity.md` |
| logprobs payload end-to-end | `v1/sample/`, `v1/engine/logprobs.py`, serving | `DONE` T1 | full path WIRED + CPU-gated (ROAD-V1-C7 W5): LogprobsProcessor -> `CompletionOutput.logprobs` -> OpenAI `CompletionLogProbs`/`ChatCompletionLogProbs` serialization; gated vs a vLLM-0.26 oracle (RED-first) + e2e through the CPU engine | [specs/sampling-controls-c7.md](specs/sampling-controls-c7.md) |
| prompt_logprobs | `v1/engine/logprobs.py`, serving | `ACTIVE` T1 | payload plumbing + serialization DONE (LogprobsProcessor prompt path + `RequestOutput.prompt_logprobs`); the runner prompt-position-logits SOURCE (lm_head over prompt tokens) is pending — a runner/prefill addition | `planned: specs/prompt-logprobs.md` |
| logit_bias / allowed_token_ids / bad_words | `sampling_params.py`, `v1/sample/` | `ACTIVE` T1 | WIRED end-to-end (ROAD-V1-C7): SamplingParams fields+validation, OpenAI parse+logit_bias clamp, InputBatch per-slot + build_sampling_metadata + condense/swap, InputProcessor bad_words tokenization; CPU-gated (RED-first) | [specs/sampling-controls-c7.md](specs/sampling-controls-c7.md) |
| best_of / echo / suffix / user fields | `protocol.py` | `PARTIAL` T1 | echo parses but behavior is deferred; remaining fields absent | `planned: specs/completions-longtail-fields.md` |
| Beam search wrapper | `beam_search.py` | ☐ T1 | | `planned: specs/beam-search.md` |
| Reasoning parsers (+ reasoning-gated grammar) | `reasoning/` | ☐ T1 | | `planned: specs/reasoning-parsers.md` |
| Thinking budget | `v1/sample/` | ☐ T1 | stub marked | `planned: specs/thinking-budget.md` |
| Repetition detection | `v1/sample/` | ☐ T1 | | `planned: specs/repetition-detection.md` |
| Custom logits processors (plugin point) | `v1/sample/logits_processor/` | ☐ T2 | → C ABI callback registry (§9.4) | `planned: specs/custom-logits-processors.md` |

## 7. Structured outputs & tool calling

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| Structured outputs: `json`(schema)/`json_object`/`regex`/`choice`/`grammar` + `response_format` | `v1/structured_output/` | `PARTIAL` T0 | seam + native backend work for a bounded JSON-schema subset; PRODUCTION-WIRED 2026-07-23 (`CLAIM-CAPI-STRUCTURED-V2`): `LoadedEngine` owns the manager (native factory) threaded into Scheduler/EngineCore/AsyncLLM (`src/vllm/entrypoints/model_loader.cpp`), and the C ABI exposes the constraints (ABI v2 `structured_*`, `include/vllm.h` + `src/capi/vllm_c.cpp`; tests `tests/capi/test_capi.cpp` structured cases); upstream backend matrix is unported | `planned: specs/structured-outputs.md` |
| xgrammar C++ core as 2nd backend | `structured_output/backend_xgrammar.py` | ☐ T1 | same proven seam; closes whitespace/key-order/exotic-schema parity | `planned: specs/xgrammar-backend.md` |
| STRUCTURAL_TAG (full) | `structured_output/`, `tool_parsers/structural_tag_registry.py` | `PARTIAL` T1 | tool-choice subset works; full response-format/tag surface absent | `planned: specs/structural-tag.md` |
| guidance / outlines backends | `structured_output/` | ☐ T2 | | `planned: specs/guidance-outlines-backends.md` |
| Tool calling: `tools`/`tool_choice` auto+required+named, streaming deltas, Hermes + Qwen3 parsers | `tool_parsers/`, `entrypoints/openai/chat_completion/` | `PARTIAL` T0 | Hermes core works; local Qwen3 parser is a Hermes alias, not upstream Qwen3Engine parity | `planned: specs/tool-calling.md` |
| Unified Streaming Parser Engine | `parser/engine/` | `ACTIVE` T1 | CORE engine LANDED + CPU-GATED 2026-07-27 (`TOOLS-STREAMING-PARSER`, `CLAIM-ROADMAP-C8-PARSER`): the declarative `StreamingParserEngine` (token-ID scanner + prefix-buffering incremental lexer + transition state machine + JSON-arg brace hold-back + drop-info) with qwen3/seed_oss/kimi_k2 configs and the unified name->config registry, EXACT-gated event-for-event vs vLLM 0.26 (8 scenarios, 586/586, RED-first). ASSEMBLY layer LANDED + CPU-GATED 2026-07-27 (`CLAIM-ROADMAP-C8-ASSEMBLY`): the `ParserEngine` (SemanticEvent -> streaming `DeltaMessage` + one-shot `ExtractedToolCallInformation`) with qwen3/seed_oss/kimi_k2 assembled parsers + `parser_manager` dispatch, field-for-field vs vLLM 0.26 (`test_parser_engine_assembly` 9 scenarios, 1652/1652, RED-first 32 asserts). SERVING-SSE dispatch swap LANDED + CPU-GATED 2026-07-27 (`CLAIM-ROADMAP-C8-SERVING`): the OpenAI chat streaming path's `--tool-call-parser` seam routes engine-backed names (qwen3/seed_oss/kimi_k2) through `parser_manager get_parser_engine` and drives `parse_delta`/`parse`, EXACT chunk-for-chunk vs vLLM 0.26 `chat_completion_stream_generator` (`test_openai_serving_chat_stream` 9 scenarios, 210/210, RED-first 6 CHECKs); OFF by default, legacy seam byte-identical. CONFIG FAMILIES LANDED + CPU-GATED 2026-07-27 (`CLAIM-ROADMAP-C8-CONFIGS`): 5 more engine-backed families ported (minimax_m2, glm47_moe, deepseek_v4, deepseek_v32, nemotron_v3) as additive `ParserEngineConfig` builders + regex arg-converters + `Glm47MoeParser` name-`.strip()`, field-for-field vs vLLM 0.26 (`test_parser_engine_assembly` 19 scenarios, 3510/3510, RED-first 2 asserts glm47 name-strip). CONFIG FAMILIES C8-2 LANDED + CPU-GATED 2026-07-27 (`CLAIM-ROADMAP-C8-CONFIGS-2`): the last 2 deferred families gemma4 + inkling ported via 4 additive default-inert assembly-core virtual seams (`preprocess_feed`, virtual `events_to_delta`/`single_pass_parse`/`reset`/`extract_reasoning`, `args_wrapper_keys`) + `gemma4_config`/`inkling_config` (custom key:value + JSON-span arg carvers) + `Gemma4Parser` (channel-injection + `thought\n`-strip) / `InklingParser` (args-key unwrap + trailing-text flush); field-for-field vs vLLM 0.26 (`test_parser_engine_assembly` now 26 scenarios, 4526/4526, adds a non-streaming parse() gate, RED-first for all 4 new seams), the 586/586 + 210/210 gates byte-identical (seams inert). Residual: JSON-schema arg coercion + SERVE-RESPONSE-METRICS + live-engine metric wiring + chat-form /tokenize (C8 stays PARTIAL) | [specs/streaming-parser-engine.md](specs/streaming-parser-engine.md), [specs/parser-assembly-c8.md](specs/parser-assembly-c8.md) |
| Tool parser breadth (Qwen-Coder XML, Mistral, pythonic, …) | `tool_parsers/` | ☐ T1 | abstract parser seam in place | `planned: specs/tool-parser-breadth.md` |

## 8. Speculative decoding

Scoping done → [specs/spec-decode-scoping-2026-07-10.md](specs/spec-decode-scoping-2026-07-10.md)
(B5). Route after speed parity: MTP k=1 on 27B → GDN spec path → DFlash →
DSpark. Both gate checkpoints SHIP
MTP heads. The optional safetensors head loader/standalone forward now exists;
normal target loading still leaves `mtp.*` unloaded until speculative decoding
is configured, exactly as upstream loads its draft model on demand.

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| MTP (Qwen3.6 MTP heads, k=1) | `v1/worker/gpu/spec_decode/`, `models/qwen3_5_mtp.py` | ✅ **DONE (`SPEC-MTP`, 2026-07-26)** | k=1 MTP spec-decode is COMPLETE + gated on the 27B GDN hybrid: three-way token-exact at c1 (our-ON == vLLM `--speculative-config mtp` == our-OFF, acceptance 16/16), c1 above vLLM every-axis, c2-c8 on-par-or-above, mixed-batch concurrency bit-exact, server/CLI/C-ABI `--speculative-config`; spec-OFF byte-identical. c>1 token bar is the ratified near-tie+SPEED form (bf16-batch-nondeterminism). 35B `Qwen3_5MoeMTP` e2e (M-mtp-2) + DFlash tracked separately | [specs/mtp-spec-decode.md](specs/mtp-spec-decode.md) |
| Rejection sampler | `v1/worker/gpu/spec_decode/rejection_sampler.py` | ✅ landed (`SPEC-REJECTION` ACTIVE, I3) | greedy accept rule + per-request logits expansion, CUDA==CPU bit-exact; consumed by MTP DONE | [specs/mtp-spec-decode.md](specs/mtp-spec-decode.md) (2.4) |
| GDN spec segments (metadata + slot-snapshot rollback) | `v1/attention/backends/gdn_attn.py`, `fla/ops/fused_sigmoid_gating.py` | ✅ landed (`SPEC-GDN-SEGMENTS` ACTIVE, I4/I5a/I7) | metadata split + reclassification, `T>1`/IS_SPEC recurrence, conv rollback, k+1 slots, mixed split/merge; bit-exact; consumed by MTP DONE | [specs/mtp-spec-decode.md](specs/mtp-spec-decode.md) (3) |
| DFlash (block-diffusion drafter) | in-pin + published drafts for our models | 🚧 **spec written** (after MTP) | DGX-Spark community container exists; GDN slot memory at k=15 flagged | [specs/dflash-spec-decode.md](specs/dflash-spec-decode.md) |
| DSpark (semi-autoregressive block drafter) | `v1/worker/gpu/spec_decode/dspark/`, `models/{qwen3_dspark,deepseek_v4/nvidia/dspark}.py` | ☐ T1 (**user-promoted**) | DeepSeek-V4 and Qwen3 draft layouts, reduced-vocab mapping, Markov sampling and full-CUDA-graph behavior inventoried as `SPEC-DSPARK`; dedicated spike follows parity/MTP | `planned: specs/dspark-spec-decode.md` |
| TLI heterogeneous-vocabulary spec decode | `v1/spec_decode/vocab_mapping.py`, `config/speculative.py` | ☐ T1 | target↔draft ID mapping and shared-token constrained logits; current upstream validation is greedy draft only; inventory row `SPEC-TLI` | `planned: specs/tli-spec-decode.md` |
| ngram (draft-free proposer) | `v1/spec_decode/ngram_proposer.py` | ✅ **DONE (`SPEC-NGRAM`, 2026-07-27)** | Draft-FREE suffix-ngram matcher (KMP-LPS, 1:1 port) wired as a third method reusing the MTP/DFlash verify/reject/`take_draft_token_ids` loop. 27B gate: 5/5 STRICT our-ngram-ON == vLLM-ngram-ON, 180/180 drafts accepted; unit 19/19; spec-OFF byte-identical (SACRED 235/235 + MTP 9/9 + DFlash 27/27); no new kernel | [specs/spec-decode-breadth-d3.md](specs/spec-decode-breadth-d3.md) |
| EAGLE3 | `v1/spec_decode/eagle.py` | 🚫 **SCOPED — reachable-blocked (`SPEC-EAGLE3`)** | Port designed (reuses DFlash D5 separate-draft loader + D1 aux multi-tap). BLOCKED: no ungated oracle-runnable EAGLE3 draft arch/checkpoint for a Qwen3.6 gate model at pin `555967922` (registry has no `Eagle3Qwen3_5*`; z-lab published DFlash not EAGLE3). No fabricated gate | [specs/spec-decode-breadth-d3.md](specs/spec-decode-breadth-d3.md) |

## 9. Serving surface (OpenAI API, endpoints, CLI, library)

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| `/v1/chat/completions` + `/v1/completions` (SSE streaming) | `entrypoints/openai/` | `ANCHOR-BACKFILL` T0 | basic transport/framing plus W2 live incremental AsyncLLM delivery and disconnect abort are CPU/TSan-proven; full protocol and GB10 online gates remain in leaf rows | `planned: specs/chat-completions-endpoints.md` |
| `/v1/models`, `/health`, `/version` | same | `PARTIAL` T0 | routes work; `/health` always returns 200 instead of checking engine health | `planned: specs/models-health-version.md` |
| **`/metrics` Prometheus, names 1:1** (`vllm:*`) | `v1/metrics/` | `ACTIVE` T0-core | **LANDED + CPU-GATED 2026-07-27 (`SERVE-METRICS`, `CLAIM-ROADMAP-C8`):** self-contained Prometheus registry + text-0.0.4 exposition + the always-on vLLM metric catalog (`PrometheusStatLogger`, names/help/type/buckets/`{model_name,engine}` labels 1:1) + `GET /metrics`; gated by the vLLM scrape spec `EXPECTED_METRICS_V1` (substring, RED-first). Residual: live-engine per-step wiring + config-gated families | [specs/prometheus-metrics.md](specs/prometheus-metrics.md) |
| Per-request timing metrics in chat/completion response bodies | `entrypoints/generate/base/serving.py`, OpenAI protocols/serving | ☐ T1 | v0.25.0 opt-in surface for TTFT/prefill/decode timing, streaming/non-streaming and invalid multi-output suppression; inventory row `SERVE-RESPONSE-METRICS` | `planned: specs/per-request-response-metrics.md` |
| `stream_options` / `include_usage` | `engine/protocol.py`, completion/chat `protocol.py` + `serving.py`, `serve/utils/api_utils.py` | 🚧 **GATING** T1 | native-ID final/continuous completion+chat frames, non-stream validation and force mode are implemented; CPU 105/105, focused ASan+UBSan 3/3 and TSan 1/1 pass. Fresh merged-SHA 27B→35B online evidence remains before `DONE` | [stream-options.md](specs/stream-options.md) |
| `/tokenize`, `/detokenize`, `/ready`, `/ping`, `/server_info`, `/reset_prefix_cache` | various routers | `ACTIVE` T1 | **LANDED + CPU-GATED 2026-07-27 (`SERVE-UTILITY-ENDPOINTS`, `CLAIM-ROADMAP-C8`):** `/tokenize` (prompt form) + `/detokenize` + `/ping` + `/server_info` + `/reset_prefix_cache` (injected callback), all additive/opt-in on the ApiServer, schema-matched to vLLM 0.26. Residual: chat-form `/tokenize`, `/tokenizer_info`, `/ready`, full server_info config dump | [specs/utility-endpoints.md](specs/utility-endpoints.md) |
| Chat templating (Jinja subset, minja-style) | `renderers/hf.py`, `entrypoints/chat_utils.py` | `ANCHOR-BACKFILL` T0 | bounded Qwen3.6 minja/Jinja subset; generic template parity open | `planned: specs/chat-templating.md` |
| AsyncLLM-equivalent streaming engine API | `v1/engine/async_llm.py` | 🚧 **T0 `GATING`** | W2 live SSE/concurrent abort behavior remains CPU/TSan-green. Fixed `max_num_seqs + 4` capacity, the legacy toggle and 32-client/control reserve are implemented. Exact fixed/legacy c32 GPU means are 1097.031/1097.290 tok/s (0.999764×), 8/20 fixed axes, all 1,152 requests/six lifecycles and no sampled stall; the fresh fixed ladder is healthy at c32. Broader every-axis parity remains before `DONE` | [async-serving.md](specs/async-serving.md) |
| C API library (llama.cpp-style, 17-symbol ABI) | — (our packaging) | `ANCHOR-BACKFILL` T0 | dlopen/FFI proof now includes six additive nonblocking request submit/cancel/wait/done/error/free symbols over AsyncLLM; CPU/TSan green, not a claim of LocalAI integration | `planned: specs/c-api-library.md` |
| Rich C++ API (`LLM`/`AsyncLLM` mirror) | `entrypoints/llm.py` | ☐ T1 | | `planned: specs/cpp-api.md` |
| CLI: `serve` + `bench {latency,throughput,serve}` | `entrypoints/cli/` | `PARTIAL` T0 | separate server/bench binaries and one in-process benchmark; server now exposes `max_num_seqs`/`max_num_batched_tokens` for reproducible operating points, but no matching command family | `planned: specs/cli-serve-bench.md` |
| CLI: `chat`, `complete` | `entrypoints/cli/` | ☐ T1 | examples/cli covers basic complete | `planned: specs/cli-chat-complete.md` |
| `/v1/embeddings`, `/pooling`, `/score`, `/rerank` | pooling routers | ☐ T2 | with pooling models (§4) | `planned: specs/pooling-endpoints.md` |
| `/v1/responses`, `/v1/messages` (Anthropic-style), audio | responses/messages routers | ☐ T2 | | `planned: specs/responses-messages-endpoints.md` |
| Sleep/pause/resume, profiling, RL weight-update endpoints | various | ☐ T2–T3 | | `planned: specs/admin-endpoints.md` |
| OTLP tracing | `config/observability.py` | ☐ T2 | | `planned: specs/otlp-tracing.md` |

## 10. LoRA & adapters

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| LoRA runtime (punica-style batched apply) | `lora/`, `v1/worker/lora_model_runner_mixin.py` | ☐ T2 | no runtime in-tree (verified: field placeholders only) | `planned: specs/lora-runtime.md` |
| LoRA dynamic load/unload endpoints | `entrypoints/openai/` | ☐ T2 | after runtime | `planned: specs/lora-endpoints.md` |

## 11. Long context & attention breadth

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| YaRN rope scaling | `layers/rotary_embedding/` | `ACTIVE` T1 | W5 implements typed modern/legacy config, memoized f32/bf16 cache/factory, plain YaRN and its `mrope_section` branch, plus NeoX/GPT-J and 1-D/3-axis supplied-cache CPU/CUDA code. Six pinned-source oracle fixtures and CPU/sanitizer gates pass; the shared scaled-RoPE CUDA apply+cache path is now GPU-GATED (2026-07-27 `CLAIM-ROADMAP-C5`, dgx GB10: `test_ops_rope_cache` 6/6, `test_rotary_embedding` 14/14). Feature-positive model YaRN e2e is REACHABLE-BLOCKED (no cached oracle-runnable YaRN consumer — Nomic/gpt-oss absent, Qwen3-4B is default-rope). Audited Qwen3.6 gates use **default interleaved MRoPE, not YaRN**, so they are regression-only; every-axis SPEED pending | [sliding-local-yarn-long-context.md](specs/sliding-local-yarn-long-context.md) |
| llama3 / longrope / dynamic-NTK scaling | same | `ACTIVE` T1–T2 | all three leaves are implemented AND GPU-GATED feature-positive vs the vLLM 0.26.0.dev0 oracle (2026-07-27 `CLAIM-ROADMAP-C5`, dgx GB10 sm_121a, clean CUDA build of `489f7771`): **W6 Llama 3** — Llama-3.2-1B `test_llama_paged_engine` 16/16 (band-rescale applied at all positions); **W7 Phi-3 LongRoPE** — Phi-4-mini `test_phi3_paged_engine` 16/16 (long cache selected globally, RED-first proven); **W8 dynamic-NTK** — InternLM2 `test_internlm2_paged_engine` 16/16 (identity at gate-battery lengths, mirrors vLLM). Both RoPE recaptures BIT-IDENTICAL to committed goldens on the 0.26 oracle (zero drift). Nine pinned-source f32/bf16 fixtures + CPU/sanitizers still green. Dynamic-NTK nontrivial-scaling e2e (>trained-length prompt) + every-axis SPEED pending | [sliding-local-yarn-long-context.md](specs/sliding-local-yarn-long-context.md) |
| Sliding-window attention backend | `v1/attention/backends/` | `ACTIVE` T1 | W2 implements the shared semantic window, model-config normalization, CPU mask, separately specialized full/local portable-CUDA kernels, vendored FA2 local dispatch, and upstream-derived tests. GPU-GATED 2026-07-27 (`CLAIM-ROADMAP-C5`, dgx GB10 sm_121a): the local-mask CUDA kernel is POSITIVELY gated at operator level (`test_ops_paged_attn` WMMA sliding-window max_abs_err 2.10e-6 + `test_attention_window` 3/3) and the Gemma-2/Gemma-3 sliding-window model gates pass 48/48 CUDA greedy vs the oracle. Window is inert at the short gate battery (ctx < W); long-context positive-mask (prompt > W) model e2e + every-axis performance/memory closure pending | [sliding-local-yarn-long-context.md](specs/sliding-local-yarn-long-context.md) |
| Chunked-local attention wrapper/backend | `model_executor/layers/attention/chunked_local_attention.py` | `GATING` T1 | W4 cached wrapper, six exact virtual-batch shapes, reusable block-table update plan, cudagraph rejection, spec emission and ordinary-backend delegation pass CPU/reference/sanitizer gates (and `test_chunked_local_attention` 5/5 on the clean GB10 CUDA build, `CLAIM-ROADMAP-C5`); model-level e2e is REACHABLE-BLOCKED — no `Llama4ForCausalLM` model row exists (the representative consumer) — so oracle/GPU-trace/every-axis closure stay blocked on that vehicle | [sliding-local-yarn-long-context.md](specs/sliding-local-yarn-long-context.md) |
| MLA backends (latent KV, MQA decode) | `v1/attention/backends/mla/` | ☐ T2 | | `planned: specs/mla-backends.md` |
| Mamba1/Mamba2/short-conv/linear backends | `v1/attention/backends/` | ☐ T2 | | `planned: specs/mamba-backends.md` |
| Encoder / cross-attention | backends | ☐ T2 | | `planned: specs/encoder-cross-attention.md` |

## 12. Platforms & hardware

The exhaustive platform and 13-target CUDA inventory is
[backend-matrix.md](backend-matrix.md). Build-source availability is not runtime
support; every target needs build, correctness, trace, performance and memory
evidence.

| ID | Platform block | State | Grounded summary | Detailed table / spike |
|---|---|---|---|---|
| `BACKEND-CUDA-SM121` | GB10/sm121a | `PARTIAL` | gate workload built, traced, token/perf gated; full component-family coverage is open | [backend row](backend-matrix.md#cuda-target-rows) |
| `BACKEND-CUDA-OTHER` | vLLM sm70/75/80/86/87/89/90/100/101/103/110/120 targets | `ACTIVE` | 9 CUDA arches build-supported (sm80/86/87/89/90a/100a/103a/110/120a, single-arch portable-kernels-only, `-Werror` clean, SASS emitted); sm70/75/101 not build-supported; no non-121a target is runtime-validated | [backend matrix](backend-matrix.md), [CUDA inventory](specs/cuda-architecture-inventory.md) |
| `BACKEND-CPU` | production CPU | `PARTIAL` | persistent threadpool + chunked GEMM/row dispatch is 1/3/20-thread bit-identical and TSAN-clean; idle-host performance/RSS gate and compute-in-quant remain open | [backend matrix](backend-matrix.md) |
| `BACKEND-ROCM` | ROCm | `INVENTORIED` | source/dispatch spike required; no "one flag" support claim | [backend matrix](backend-matrix.md) |
| `BACKEND-MLX` | Apple Metal through MLX | `ACTIVE` | Metal/MLX skeleton ACTIVE: two models (OPT-125m, Qwen3-0.6B) run e2e + pass correctness, native-MSL GEMM, batched command buffers 1.50x (compute-bound at 98%+); optional MLX GEMM provider | [backend matrix](backend-matrix.md) |
| `BACKEND-VULKAN` | Vulkan | `INVENTORIED` | runtime absent | [backend matrix](backend-matrix.md) |
| `BACKEND-XPU` | Intel XPU | `INVENTORIED` | loyal upstream-platform port, runtime absent | [backend matrix](backend-matrix.md) |
| `BACKEND-ANE` | encoder/pooling accelerator | `INVENTORIED` | specialized CoreML route only | [backend matrix](backend-matrix.md) |

## 13. Loading, tokenizer, config

| Feature | Upstream | Status | Notes | Spec |
|---|---|---|---|---|
| safetensors container + model-specific loading | `model_loader/default_loader.py`, `models/utils.py` | `PARTIAL` T0 | Qwen-specific resolvers work. Windowed release removes source-copy coexistence; the additive `LOAD-SAFETENSORS-DIRECT-DENSE` leaf implements layer-bounded same-queue device staging and host-mirror release for ordinary plain-BF16 Qwen3.5 on discrete CUDA. The final 18-leg 4B checkpoint is ON/OFF/vLLM total **5769.99/5660.70/5849.80 tok/s** (stable vLLM confirmation), direct ON=OFF 128/128 outputs in every pair, and peak PSS **2.406/8.592/7.662 GiB**. H32 AOT, plain-BF16 graphs and ratio-4 FA2 are active and trace-proven; ON is **0.9864x** vLLM total/output throughput, while TPOT/ITL **43.72 vs 38.55 ms** keeps the row speed-pending. Generic mapping, quantized/MoE direct streaming, sanitizer availability and external 27B/35B regressions remain open | [plain-BF16 direct-load spike](specs/qwen35-plain-bf16-direct-load.md); [2026-07-25 evidence](../docs/bench-evidence/qwen35-4b-main-repair-20260725.md) |
| GGUF reader + Qwen MoE transform mapping + embedded BPE | no pinned-vLLM implementation; llama.cpp-compatible deviation | `PARTIAL` T0 | 35B real-file parity works; dense and format/compute breadth remain open | `planned: specs/gguf-loader.md` |
| HF `tokenizer.json` byte-level BPE + incremental detok | `tokenizers/registry.py`, `tokenizers/hf.py` | `ANCHOR-BACKFILL` T0 | bounded Qwen byte-BPE slice; deviations need leaf accounting | `planned: specs/hf-tokenizer.md` |
| SentencePiece tokenizer | same | ☐ T1 | needed for Llama-2-era / Gemma checkpoints | `planned: specs/sentencepiece.md` |
| Config surface: dataclass-for-dataclass structs, `vllm serve`-compatible flags | `vllm/config/` | `PARTIAL` T0/T1 | gate-model structs and a small CLI subset incl. explicit scheduler sequence/token budgets only | `planned: specs/config-surface.md` |
| Sharded-state / tensorizer / runai / BnB loaders | `model_loader/` | ☐ T3 | | `planned: specs/loader-longtail.md` |

---

## Reading the matrix into work

1. Follow the ordered portfolio in roadmap_v1.md and select an unblocked
   `INVENTORIED` row or row block.
2. For coordinated roadmap work, claim its stable ID in coordination.md. Write
   the real `specs/<slug>.md` spike in all cases. A plain `planned: specs/...`
   cell is deliberately not a link and does not make the row `READY`.
3. Make the spike reviewable, change the row to `READY`, then implement. Use an
   isolated worktree for parallel claims and the `${GPU_LOCK}` policy for GPU
   work when selected by `developer-preferences.md`; a benchmark is always
   uncontended.
4. `DONE` requires exact local code, ported-test/evidence and real-spec anchors,
   plus the roadmap, README, inventory, ledger and state updates in the same
   change. Legacy emoji rows are migrated by the anchor-backfill workstream.

Currently open T0 debt (pre-existing, tracked): `/metrics` core set (§9),
`SERVE-GATE-ONLINE`, and `SERVE-E2E-NIGHTLY`. GGUF real-file parity A2 passed;
its remaining format/model breadth stays tracked in §5/§13.
