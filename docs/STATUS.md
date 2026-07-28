# vllm.cpp status

This is the per-capability status ledger. [README.md](../README.md) is the
user-facing landing page and carries only headline state; every capability's
current lifecycle stage, its active gaps, and its next gate live here.

Capabilities are labelled: *correctness-complete* (token-exact vs the vLLM
oracle), *speed-pending* (correct, throughput work in progress), *build-only*
(compiles for a target with no runtime proof here), and *hardware-blocked*
(cannot run on the hardware available).

The forensic chronology and the raw evidence live in the append-only
[`.agents/state.md`](../.agents/state.md), the
[parity ledger](../.agents/parity-ledger.md), the area matrices, and
[docs/BENCHMARKS.md](BENCHMARKS.md). This file keeps ONE binding current-state
line per capability, not a run-by-run log.

## Parity pin

The reference numbers here are measured against the pinned vLLM oracle on GB10,
greedy, same workload, same tokens. Where a claim cannot be measured on the
hardware here, it is stated as such rather than implied. **The parity pin
advanced 2026-07-26 to vLLM 0.26.0.dev0 (`55596792`) + transformers 5.14.1**
(from 0.25.0); correctness was re-validated bit-identical on the new oracle
(zero golden drift; see [.agents/specs/pin-advance.md](../.agents/specs/pin-advance.md)),
so the token-exact claims hold against the new pin. Historical speed figures
citing "vLLM 0.25.0" are the last binding measurement against the prior oracle
(our engine is unchanged by the advance); a re-benchmark against 0.26 is pending.

## Capability status

vllm.cpp implements an intentionally focused subset of vLLM, held to
token-for-token correctness against the pinned oracle.

| Capability | State | Notes |
|---|---|---|
| Qwen3.6-27B (NVFP4) text generation | Correctness-complete, at/above vLLM speed | Token-exact greedy on GB10; beats vLLM 0.25.0 total throughput at every concurrency (1.007-1.045x), effective parity 115/124 axes |
| Qwen3.6-35B-A3B (NVFP4, GDN MoE) | Correctness-complete, decode at-parity, prefill speed-pending | Token-exact greedy; decode at or beyond vLLM, remaining gap is prefill TTFT |
| Qwen3 / Qwen2 dense (BF16) | Correctness-complete, speed-pending | Near-tie-robust token-exact vs vLLM (Qwen3-0.6B, Qwen3-4B); c1 effective parity, c8 decode residual |
| Qwen3.5-4B plain BF16 direct loading on discrete CUDA | Correctness-complete, speed-pending | Direct ON and OFF are token-identical; direct loading cuts peak/stable host PSS by 72.0%/91.2%. The pre-transplant H32 AOT, decode-graph and ratio-4 FA2 result reached 0.9864x vLLM 0.25.0; the current-main development branch is pending revalidation |
| Qwen3-Coder-30B-A3B MoE (BF16) | Correctness-complete, speed-pending | Near-tie-robust token-exact 6/6; 11 of 16 binding grid cells at or above vLLM |
| Llama-3.x dense (BF16) | Correctness-complete, speed-pending | Near-tie-robust token-exact 16/16 (Llama-3.2-1B); llama3 RoPE scaling |
| Mistral dense (BF16) | Correctness-complete, speed-pending | Paged-engine token-exact 16/16 (Mistral-7B-v0.3) |
| OPT (learned pos-emb, cross-family) | Correctness-complete, speed-pending | Strict token-exact 6/6 (OPT-125m); additivity canary across model families |
| DeepSeek-V2 MLA | Correctness-complete, speed-pending | Token-exact 8/8 (DeepSeek-V2-Lite); 0.86-0.95x vLLM output rate, TTFT faster at c4/c8 |
| GLM-4 dense (sandwich norms, partial rope) | Correctness-complete, speed-pending | Token-exact 16/16 (GLM-4-9B-0414); first GLM-family model; partial interleaved RoPE + Gemma2 sandwich norms + biased qkv |
| GLM-4.7-Flash (MLA + GLM MoE) | Correctness-complete, speed-pending | Token-exact 8/8 (GLM-4.7-Flash, 31.2B); reuses the DeepSeek-V2 MLA stack; first e2e coverage of the q_lora query branch + noaux_tc sigmoid router with routed-scaling |
| Gemma-3 dense (GeGLU, dual rope, sandwich norms) | Correctness-complete, speed-pending | STRICT token-exact 48/48 greedy (gemma-3-1b-it); first Gemma-family model; GeGLU (gelu_pytorch_tanh) + dual per-layer RoPE theta + Gemma-RMSNorm sandwich norms + sqrt(hidden) embed-scale + query_pre_attn_scalar scaling |
| Gemma-2 dense (attn + final logit soft-cap) | Correctness-complete, speed-pending | Near-tie-band 48/48 (gemma-2-2b-it): 44/48 strict on vLLM's greedy + 4/48 at 0.0-nat ties in vLLM's own logits; proves the attention + final logit soft-cap primitives (attn_logit_softcapping 50 + final 30); the inverse of Gemma-3 (both soft-caps, no QK-norm) |
| Gemma-1 dense (the original Gemma) | Correctness-complete, speed-pending | STRICT token-exact 48/48 greedy (gemma-2b); two fused norms/layer, head_dim scale, GeGLU + sqrt(hidden) embed-scale, tied lm_head; no soft-cap/QK-norm/sliding |
| OLMo-2 dense (pure post-norm, full-width QK-norm) | Correctness-complete, speed-pending | Token-exact 16/16 (OLMo-2-0425-1B); first OLMo-family model; ZERO new compute kernel (pure post-norm `norm_after` + full-width QK-norm reuse existing ops); real ByteLevel-tokenizer gate (no BOS) |
| Phi-3 / Phi-4 dense (Llama subclass, LongRoPE) | Correctness-complete, speed-pending | Token-exact (near-tie-robust) 16/16 (Phi-4-mini-instruct: 7 strict + 9 near-tie, 0 divergent) by the ratified root-divergence gate; bigger-dense STRICT anchor phi-4 (14B) 14/16 fully token-exact + 2 exact-tie (0.0 nats); pre-fused qkv/gate_up loader + LongRoPE cache bit-identical to vLLM's |
| Granite-3 dense (Llama + 4 scalar multipliers) | Correctness-complete, speed-pending | Token-exact 16/16 (granite-3.3-2b-instruct); first IBM Granite model; ZERO new compute kernel; embedding/residual/attention/logits multipliers (attention scale 1/64, not 1/sqrt(head_dim)) threaded over the shared dense path |
| StableLM dense (LayerNorm, partial rope, qkv bias) | Correctness-complete, speed-pending | Token-exact 16/16 (stablelm-2-1_6b): 14/16 strict on vLLM's greedy + 2/16 at bf16 near-ties (max gap 0.438 nats), 0 forward-divergent; first Stability AI model; ZERO new compute kernel (nn.LayerNorm with weight+bias + partial NeoX rope 16/64 + merged qkv bias, all reuse existing ops) |
| OLMo-3 dense (dual rope, interleaved sliding window) | Implemented, oracle-blocked | Loads + runs in our engine (dual rope: plain sliding + YaRN full-attn, per-layer sliding window); no SACRED gate: vLLM 0.25.0 oracle cannot run OLMo-3-1025-7B (`KeyError: 'rope_theta'`; transformers 5.13.1 nests `rope_parameters` per layer-type, no flat `rope_theta`; run-verified W0 2026-07-26) |
| InternLM2 dense (fused-`wqkv` interleaved split) | Correctness-complete, speed-pending | Token-exact 16/16 (internlm2-chat-1_8b): 12/16 strict + 4/16 bf16 near-tie (max gap 0.0 nats), 0 divergent; first InternLM model; ZERO new compute kernel (reuses the Llama dense forward; the only delta is a loader-side de-interleave of the fused `wqkv`, which packs q/k/v interleaved by KV-group) |
| Command-R / Cohere dense (`CohereForCausalLM`) | Implemented, gate-blocked | ZERO-new-kernel port grounded in vLLM `commandr.py`: weight-only Cohere LayerNorm + GPT-J full-width RoPE + PARALLEL residual + `logit_scale` + tied embeddings, all reuse; compiles, links, self-registers. No SACRED gate yet (real checkpoints HF-gated, ungated ones tiny-random, GPU box disk-full); oracle run-verified at W0. See docs/BENCHMARKS.md |
| Phi-1 / Phi-2 dense (`PhiForCausalLM`, parallel residual) | Correctness-complete, speed-pending | Token-exact 16/16 (microsoft/phi-2): 9/16 strict + 7/16 bf16 near-ties (max gap 0.25 nats), 0 forward-divergent; the OLDER Microsoft Phi arch, DISTINCT from Phi-3/Phi-4; ZERO new compute kernel (GPT-J parallel residual, LayerNorm-with-bias, biased qkv/dense, partial NeoX rope 32/80, non-gated NewGELU MLP reusing `vt::GeluTanh`, untied biased lm_head); F16 dtype-aware loader |
| MiniCPM dense (`MiniCPMForCausalLM`, three scalars) | Correctness-complete, speed-pending | Token-exact 16/16 (openbmb/MiniCPM-2B-sft-bf16): 10/16 strict + 6/16 bf16 near-ties (max gap 0.0 nats), 0 forward-divergent; first OpenBMB MiniCPM model; ZERO new compute kernel (the Llama/Granite dense forward plus three scalars: scale_emb, scale_depth/sqrt(layers) residual, dim_model_base logit scaling), tied lm_head; `.bin`-only weights converted to safetensors via trusted torch |
| MiniCPM3 (`MiniCPM3ForCausalLM`, MLA + three scalars) | Correctness-complete, speed-pending | Token-exact 16/16 (openbmb/MiniCPM3-4B): 13/16 strict + 3/16 bf16 near-ties (max gap 0.0 nats), 0 divergent; the first MLA-attention MiniCPM; ZERO new compute kernel (the MiniCPM three scalars with attention swapped GQA to DeepSeek-style MLA, reusing the landed DeepSeek-V2 MLA block; DeepSeek-V2 re-gated 8/8 unchanged); tied embeddings, `.bin`-only weights converted via trusted torch |
| Yi (Llama architecture) | Correctness-complete, speed-pending | Token-exact 16/16 (01-ai/Yi-Coder-1.5B-Chat): 13/16 strict + 3/16 bf16 near-ties (max gap 0.125 nats), 0 divergent; modern Yi adopted the Llama architecture (`architectures: ["LlamaForCausalLM"]`), so it runs on the existing Llama path with zero code changes; confirms a distinct non-Llama-branded checkpoint (64000 vocab, RoPE theta 1e7) loads and decodes correctly |
| InternLM3 (`InternLM3ForCausalLM`, Llama alias) | Correctness-complete, speed-pending | Token-exact 16/16 (internlm3-8b-instruct): 14/16 strict + 2/16 bf16 near-ties (max gap 0.0 nats), 0 divergent; a plain Llama architecture in vLLM 0.25.0 (registered as a one-line Llama alias; dynamic-NTK RoPE factor 6.0, GQA kv=2, untied lm_head), not InternLM2 + sliding window; zero forward or loader delta |
| Long-context RoPE + sliding-window attention | Correctness feature-positive on GB10, speed-pending | Shared scaled-RoPE (YaRN, Llama-3, Phi-3/4 LongRoPE, dynamic-NTK) + sliding-window attention, GPU-gated vs the vLLM 0.26 oracle: LongRoPE (Phi-4-mini) + llama3 (Llama-3.2-1B) + dynamic-NTK (InternLM2) 16/16, sliding-window (Gemma-2/Gemma-3) 48/48, operator local-mask kernel positive. YaRN and chunked-local model e2e are reachable-blocked (no cached vehicle); speed pending |
| Safetensors loading | Supported | Both gate models plus every registered dense/MoE family |
| GGUF loading (F32/F16/Q4_0/Q8_0/Q3_K/Q4_K/Q5_K/Q6_K) | Supported; compute-in-quant on CPU | Weights in six block encodings stay compressed from file to matmul on CPU (no BF16 expansion) |
| CPU backend vs llama.cpp | At or ahead on every axis (GGUF) | Prefill 1.18x ahead, decode at parity, peak memory 1.01x, byte-identical greedy tokens. Single-stream only; no concurrent-serving comparison has been measured |
| Paged KV cache + prefix caching | Supported | Block-paged full attention, hybrid full-attention + GDN state groups, automatic prefix caching (APC) on by default for dense models (cache-ON gated end to end: token-identical output, cache hits, faster TTFT) |
| KV offload to CPU / disk | Built, opt-in, off by default; the disk connector is engine-refused | CPU and disk tiers with identity-checked blocks, selected by `--kv-transfer-config` (or programmatically) over one abstract KVConnector ABI. Worker-side KV store/load is implemented for the LMCache connector only; the CPU/disk connector is scheduler-side only, so the engine now REFUSES it at construction (a loud error, not silently wrong output). Guide: [docs/KV-OFFLOAD.md](KV-OFFLOAD.md) |
| LMCache client (`lm://` remote KV) | Built, opt-in, off by default; a working, verified external KV cache | Pure-C++ `lm://` client wired as an `LMCacheConnector`, no `lmcache` in-process; keys agree byte-for-byte with a real vLLM+LMCache peer, mismatched blocks refused. Proven in a real OPT-125m loop vs a live `lmcache.v1.server`: connector-ON tokens are BIT-IDENTICAL to the connector-OFF cold run (both after an in-process restart and from a cold second process). See docs/BENCHMARKS.md |
| KV-cache events (for external routers) | Built, off by default; generation and payload gated, live ZMQ transport deferred | Block store/remove/clear events (`BlockStored`/`BlockRemoved`/`AllBlocksCleared`) emitted at the prefix-cache sites when enabled, with a `msgpack` payload byte-identical to vLLM's `msgspec` encoding. Behind a publisher seam faithful to `--kv-events-config`; the live ZMQ transport is not wired yet. Off by default, so the prefix-cache path is byte-identical. |
| Sampling | Supported | Greedy, temperature, top-k/p, min-p, presence/frequency/repetition penalties, seed, stop/stop_token_ids, min_tokens, logit_bias, allowed_token_ids, bad_words, in vLLM's exact order. Custom logits processors are supported through a C-ABI callback (`vllm_logits_processor`, ABI v8): a per-request host callback invoked each decode step to modify the logits before sampling; absent by default (byte-identical). Sample logprobs are emitted end-to-end for `/v1/completions` and `/v1/chat/completions` (`logprobs`/`top_logprobs`). Parallel sampling (`n>1`) is supported: a request fans out into n sequences sharing the prompt, returned as n indexed `choices` (`n==1` byte-identical). Beam search is supported through the `BeamSearch` driver — an outer engine loop that scores beams by cumulative logprob with a length penalty and returns the top `beam_width` sequences (deterministic, token-exact vs vLLM's algorithm); it is wired on the OpenAI `use_beam_search` request field for `/v1/completions` and `/v1/chat/completions` over BOTH the synchronous engine AND the production AsyncLLM HTTP server (via an async `BeamSearchAsync` driver that reuses the same scoring and returns beams token-identical to the sync driver), so beam search runs on the real server (the C-ABI beam params and streaming beam are not exposed yet; per-beam concurrent stepping is a named residual — beams are stepped sequentially, byte-identical to the sync driver). `best_of` is supported on both endpoints (generate `best_of` sequences, return the `n` highest-cumulative-logprob ones; `best_of==n` is the default no-op) — vLLM 0.26 itself has dropped `best_of` from its live path, so this follows the classic OpenAI contract. Prompt logprobs and `echo` are not emitted yet |
| Structured output | Supported (subset), engine-enforced | JSON schema, JSON object, regex, choice, GBNF grammar. Constrained decoding runs in the production engine (native grammar backend, per-step logits bitmask) and is reachable from OpenAI `response_format` and the C ABI (ABI v2 `structured_*` fields) |
| Tool-call parsing | 36 parser families / 40 accepted names, streaming | Every vLLM tool parser at the pin except the three Rust/Harmony-backed ones: pure-text parsers ported 1:1, the six engine-backed families reimplemented from their wire formats, all held to the upstream test suites. Selection via `--tool-call-parser` (server), `tool_parser` (C ABI), or template auto-detection; native-syntax forced tool_choice where expressible. Tables: docs/BENCHMARKS.md |
| Reasoning parsing | 7 parsers, streaming | think_auto (auto-detect default: content unless markers appear), deepseek_r1, mistral ([THINK]), minimax_m2 (+append_think), step3, olmo3 - reasoning split engine-side BEFORE tool parsing, streamed as `reasoning` deltas in the chat chunks |
| Unified streaming parser engine | Core, assembly, serving-SSE dispatch landed, gated; all 10 engine-backed families ported (family parity closed); JSON-schema tool-arg type coercion landed | The vLLM 0.26 declarative `parser/engine/` (shared state machine plus all 10 configs: qwen3, seed_oss, kimi_k2, minimax_m2, glm47_moe, deepseek_v4/v32, nemotron_v3, gemma4, inkling) and assembly layer, gated field-for-field vs vLLM 0.26. An engine-backed `--tool-call-parser` name drives the live chat SSE chunks, off by default. When a request's tools declare typed parameters, the assembled tool-call arguments are coerced to the declared JSON types (int/number/bool/string/array/null) 1:1 with vLLM `_fix_arg_types`, in both streaming and one-shot; no schema means the arguments pass through as strings unchanged. Details: .agents/specs/parser-assembly-c8.md |
| OpenAI server | Supported (subset) | `/v1/completions`, `/v1/chat/completions`, streaming SSE, `/v1/models`, `/health`, `/version`, `/ping`, `/metrics` (Prometheus `vllm:*` names), `/tokenize` (both the raw-`prompt` and the chat-`messages` forms, the latter rendered through the same model chat template as chat-completions), `/detokenize`, `/tokenizer_info` (opt-in via `--enable-tokenizer-info-endpoint`, mirrors vLLM's `enable_tokenizer_info_endpoint`; surfaces the tokenizer-config fields the BPE tokenizer can genuinely back), `/server_info`, `/reset_prefix_cache`, `/abort_requests` (abort in-flight requests by id; dev-mode via `--enable-server-dev-mode`, mirrors vLLM's `VLLM_SERVER_DEV_MODE`). The production `vllm-server` binary now serves `/tokenize`, `/detokenize` by default and `/tokenizer_info` / `/abort_requests` behind those flags, wired to the live engine (`CLAIM-C8-SERVE-PROD-WIRING`); `/metrics` and `/reset_prefix_cache` currently have no live backing on the async serving path (handlers exist; production wiring pending an engine stat-logger / reset RPC accessor) |
| Tokenizers | Supported | Byte-level BPE (Qwen/Llama-3/OPT/GPT-2/DeepSeek/OLMo-2) and SentencePiece BPE (Mistral/Gemma), plus GGUF vocab; added-token `lstrip`/`rstrip` whitespace semantics (e.g. Phi-4-mini's special tokens); byte-exact vs the vLLM oracle |
| Multimodal: image to text | Correctness-complete, speed-pending; not in the OpenAI server yet | Strict token-exact 32/32 vs vLLM 0.25.0 on Qwen3-VL-4B and Qwen3.6-27B (`Qwen3_5ForConditionalGeneration`); C++ image processor + vision tower + MRoPE/DeepStack backbone with on-GPU greedy sampling; the 27B decode step is now graph-capturable (routes through the production captured decode, token-exact held); batched serving speed pending |
| Multimodal: video to text | Correctness-complete, speed-pending; not in the OpenAI server yet | End-to-end on Qwen3-VL-4B (near-tie-robust) and Qwen3.6-27B (strict 32/32); reuses the image tower and temporal MRoPE plus video preprocessing (frame sampling, temporal grid) |
| Multimodal: audio to text | Decode beats vLLM; encoder TTFT measured and improved but not yet at parity; serving pending; not in the OpenAI server yet | End-to-end on Voxtral-Mini-3B (near-tie-robust, decoder token-exact 48/48) plus a Whisper-class encoder tower. Decode is graph-captured and runs vLLM's FA2 varlen split-KV kernel (the KV block-size is rounded up to a multiple of 16 to meet its precondition): audio TPOT 39.5 ms/token BEATS vLLM 0.25.0's 40.8 ms (0.97x, non-overlapping bands). The Whisper encoder attention now uses the warp-scoped online-softmax kernel (the same one that beat the vision tower) instead of the naive per-key block-reduction kernel: the encoder forward drops from 8.87 s to 1.89 s (4.7x), token-identical output (16/16, STRICT golden unchanged). But encoder TTFT (~1.89 s) is still far above vLLM's 43 ms: the warp kernel is O(t-squared) and memory-bound on redundant K/V reads at the 1500-frame context, so closing it needs a flash-tiled non-causal head-dim-64 attention plus resident (one-time) encoder weights. Correctness held under the ratified distributional near-tie gate (teacher-force PASS, 0 divergent, strict prefix exact vs vLLM; STRICT golden unchanged; 14/14). Remaining: encoder-attention flash-tiling + resident weights, and there is no batched c2+ or `audio_url` serving ingestion. See docs/BENCHMARKS.md |

| SGLang parity (competitor floor + oracle) | Elevated to a full parity target; scoped, oracle stand-up pending | SGLang v0.5.15 (`f63458b`) whole runtime surface inventoried (44 rows: 23 FUSED into our vLLM-derived engine, 8 SGLANG-DISTINCT opt-ins, 5 inventoried, 8 out-of-scope) and SGLang stood up as a correctness cross-check + a binding perf floor on dgx GB10 via the digest-pinned arm64 cu130 image (no from-source build needed). SGLang is a competitor, not the mirror source (vLLM stays behavior truth). The binding perf gate blocker is our-side `SERVE-ASYNC-LLM`, not SGLang runnability. Inventory + gate methodology: `.agents/sglang-matrix.md`, `.agents/specs/sglang-parity-oracle.md` |

## Speculative decoding

Speculative decoding is available on the Qwen3.5/3.6 checkpoints via
`--speculative-config`. **MTP (k=1)** is end-to-end token-exact vs vLLM on both
gate models (the 27B GDN hybrid `Qwen3_5MTP` and the 35B MoE `Qwen3_5MoeMTP`):
three-way identical at concurrency 1 (our spec-ON == our spec-OFF == vLLM
`--speculative-config mtp` greedy) and faster than vLLM there, on par or above
at higher concurrency (mixed-batch), with the draft head alive and acceptance
matched to vLLM.

**Block-diffusion DFlash** (the z-lab Qwen3.6-27B draft over the 27B NVFP4
target) is correctness-complete and at or above vLLM throughput on GB10 (final
concurrency-1 A/B our-on 29.32 tok/s vs vLLM-on 29.24, non-overlapping bands,
1.003x); it is recorded DONE across the engine, model and kernel matrices, built
D0 through D14 on the vLLM 0.26.0.dev0 stack (which resolves vllm#40898), and it
remains gated behind a spike while its user-facing serving surface is finalized.

**ngram** (method `ngram`, draft-FREE) proposes the next tokens by matching the
sequence's own suffix n-gram, so it needs no draft model and works on any model;
on the 27B it is token-exact vs vLLM's own `--speculative-config ngram` on
repetitive workloads (5/5 prompts, every draft accepted).

The correctness form and the full D0-D14 measured chronology live in
[docs/BENCHMARKS.md](BENCHMARKS.md),
[docs/SPECULATIVE-DECODING.md](SPECULATIVE-DECODING.md) and
[.agents/specs/dflash-spec-decode.md](../.agents/specs/dflash-spec-decode.md).

## Not supported yet

LoRA, multi-GPU, and the full tool-calling template surface. Multimodal
(image/video/audio) is correctness-complete but not yet wired into the OpenAI
server.

## Model family notes

### Gemma

**Gemma 3** (`Gemma3ForCausalLM`, `google/gemma-3-1b-it`) is the **first landed
member** (correctness-complete, speed-pending): STRICT token-exact 48/48 greedy
vs the vLLM 0.25.0 oracle. It reuses infrastructure already in the tree
(gemma-RMSNorm `(1+w)`, the GLM sandwich norms, the shared dense-attention path,
sliding-window attention, tied embeddings) plus one genuinely-new compute
kernel, GeGLU (`gelu_pytorch_tanh`), a bf16 embedding-scale multiply, dual
per-layer RoPE theta, and query_pre_attn_scalar scaling. **Gemma 2**
(`Gemma2ForCausalLM`, `gemma-2-2b-it`) and **Gemma 1** (`GemmaForCausalLM`,
`gemma-2b`) have since landed too, both correctness-complete and speed-pending
at 48/48 greedy vs the same oracle
([spike](../.agents/specs/sweep-gemma.md)): Gemma 2 proves the attention + final
logit soft-cap primitives, Gemma 1 the original two-fused-norms block.

Only **Gemma 4**, the newest registered variant, remains unimplemented: it needs
a large new primitive stack (per-layer embeddings, YOCO KV-sharing, a Gemma-4
MoE) and has only multimodal-wrapped checkpoints, so it is recorded as blocked
rather than supported. Its multimodal path (image + video + **audio**, the only
audio-capable model in the pin) has been assessed
([spec](../.agents/specs/gemma4-multimodal.md)) and was **oracle-blocked at gate
time**: the Gemma-4 vision and audio towers load through Transformers
`AutoModel`, and the gate-time 0.25.0 oracle's transformers (5.13.1) had no
`gemma4` module, so no gate was constructible then. The current 0.26.0.dev0 pin
carries transformers 5.14.1, which ships `gemma4`, so Gemma-4 multimodal is now
reachable on the pin (implementation pending). Audio, the genuinely-new
modality, is staged first on the smallest oracle-runnable audio model (Whisper,
then Voxtral-Mini-3B on the already-landed Mistral backbone).

### OLMo

The **OLMo-2 family** (`Olmo2ForCausalLM` / `Olmo3ForCausalLM`) is the **first
landed OLMo member** (correctness-complete, speed-pending): token-exact 16/16
greedy vs the vLLM 0.25.0 oracle on `allenai/OLMo-2-0425-1B` (STRICT 13/16 +
near-tie-band 3/16, max gap 0.094 nats, 0 forward-divergent). It is the cleanest
dense bring-up yet, needing no new compute kernel: its two distinctive traits
both reuse existing infrastructure. The reordered post-norm placement
(`norm_after`) is a subset of the GLM/Gemma sandwich norms (the same standalone
output-norm plus a plain residual add, without the pre-norms), and its QK-norm
is a full-width RMSNorm reusing the existing norm op at a new shape. Its GPT-NeoX
ByteLevel tokenizer (which prepends no BOS) makes it a real tokenizer-inclusive
gate. `Olmo3ForCausalLM` rides the same class (the 0.25.0 oracle constructs it);
the Olmo-3 interleaved sliding-window path has since landed and runs, but is
oracle-blocked for a gate (see the capability table above).

### Frontier and hardware-blocked families

Larger DeepSeek / GLM / MiniMax / Gemma-4 variants are recorded as
**hardware-blocked** (they do not fit 119 GiB of unified memory on this box) or
**spiked-only**, per the [model matrix](../.agents/model-matrix.md). The
frontier families Kimi / MiniMax / GLM-latest are scoped for mechanical porting
in [a dedicated spike](../.agents/specs/sweep-kimi-minimax-glm-latest.md):
Kimi-Linear-48B is the one that fits GB10 (91.5 GiB) and is e2e-gateable, while
MiniMax-M2 (214.3 GiB fp8), Kimi-K2, MiniMax-M3 and GLM-5 remain hardware- or
dependency-blocked (honesty-pass only). The matrix opens with an
architecture-support checklist (a per-architecture status roll-up covering every
engaged model) that a CI checker keeps in lockstep with the detailed rows.

### Recent dense batch

From a **next-tier batch of recent dense text families**
([spike](../.agents/specs/sweep-recent-dense-batch.md)), the top three are now
implemented (correctness-complete or gating): **Granite-3**
(`GraniteForCausalLM`, `granite-3.3-2b-instruct`) is correctness-complete,
token-exact 16/16 vs the vLLM 0.25.0 oracle (Llama plus four scalar multipliers,
no new compute kernel); **Phi-3 / Phi-4** (`Phi3ForCausalLM`,
`Phi-4-mini-instruct`) is implemented and runs but is **not** recorded as a clean
pass: it scores 15/16, and although its LongRoPE cos/sin cache is bit-identical
to the oracle's, two positions on one prompt fall outside the near-tie band as a
cascade after an exact-tie divergence, so closing it needs a cascade-aware gate
or a strict check on the larger phi-4 (14B); **OLMo-3** (`Olmo3ForCausalLM`,
dual rope plus interleaved sliding window) is implemented and runs in our engine
but has no SACRED gate because the gate-time vLLM 0.25.0 oracle cannot run the
`OLMo-3-1025-7B` checkpoint (its transformers predates OLMo-3's per-layer-type
rope schema).

**Command-R / Cohere** (`CohereForCausalLM`, parallel-residual with a weight-only
LayerNorm and a logit scale) is now **implemented** (a faithful zero-new-kernel
port that compiles, links and self-registers) but has **no SACRED gate** yet:
every real small `CohereForCausalLM` checkpoint is HF-gated with no token on the
GPU box, the only ungated checkpoints are tiny-random (head_dim 8/2, outside the
validated attention path), and the GPU box is disk-full; the oracle was
run-verified at W0 (arch confirmed `CohereForCausalLM`, not
`Cohere2ForCausalLM`).

**Phi-1 / Phi-2** (`PhiForCausalLM`, `microsoft/phi-2`) is now
**correctness-complete** (token-exact 16/16 vs the vLLM 0.25.0 oracle: 9 strict +
7 bf16 near-ties, max gap 0.25 nats, 0 forward-divergent), the OLDER Microsoft
Phi architecture, DISTINCT from Phi-3/Phi-4: a GPT-J parallel-residual decoder
(one nn.LayerNorm with bias feeds both attention and MLP), biased q/k/v/dense
projections, partial NeoX RoPE, a non-gated NewGELU MLP and an untied biased
lm_head, all reusing landed ops (its `gelu_new` maps to the landed
`vt::GeluTanh`, so no new compute kernel), with an F16-checkpoint dtype-aware
loader mirroring vLLM's f16->bf16 cast.

**MiniCPM** (`MiniCPMForCausalLM`, `openbmb/MiniCPM-2B-sft-bf16`) is now
**correctness-complete** (token-exact 16/16 vs the vLLM 0.25.0 oracle: 10 strict
+ 6 bf16 near-ties, max gap 0.0 nats, 0 forward-divergent): the landed
Llama/Granite dense forward plus three scalars (a scale_emb embedding scale, a
scale_depth/sqrt(layers) scaled residual add on each sublayer, and a
dim_model_base logit scaling), tied embeddings, no new compute kernel. Its
checkpoint ships only as a pickle `.bin` with no safetensors and no native
transformers config, so the vehicle was prepared with a trusted torch
`.bin`->safetensors conversion of the official weights (both the oracle and our
engine read the identical bf16), and the gate feeds the oracle's exact prompt ids
(its SentencePiece tokenizer is a follow-up).

**MiniCPM3** (`MiniCPM3ForCausalLM`, `openbmb/MiniCPM3-4B`) is now
**correctness-complete** (token-exact 16/16 vs the vLLM 0.25.0 oracle: 13 strict
+ 3 bf16 near-ties, max gap 0.0 nats, 0 divergent): the first MLA-attention
MiniCPM, and the model that closes the non-trivial part of this tier. It is the
MiniCPM three-scalar skeleton with its attention swapped from GQA to
DeepSeek-style MLA, reusing the landed DeepSeek-V2 MLA block (weight absorption,
absorbed-MQA decode, materialized-MHA prefill) with only three faithful deltas: a
neox-style rope flag threaded through a new default-false field on the shared MLA
dims (so DeepSeek-V2 stays byte-identical, re-gated 8/8), a LongRoPE cos/sin
cache instead of YaRN, and a reuse-only zero-pad of the FlashAttention-2 MLA
prefill from head_dim 96 up to the already-compiled 128. No new compute kernel;
its `.bin`-only weights were converted via trusted torch like MiniCPM.

The remaining recent-dense families are the trivial tail only: **Yi**
(`YiForCausalLM`, a Llama alias) and **InternLM3** (`InternLM3ForCausalLM`,
InternLM2 plus a sliding window).

## Performance detail

**Local Qwen3.5-4B plain BF16 direct loader, speed-pending:** the measured
pre-transplant repair on an RTX 5070 Ti reached 5769.99 total tok/s versus vLLM
0.25.0 at 5849.80 tok/s (0.9864x) on the identical 128-request workload. Direct
loading reduced peak PSS from 8.59 to 2.41 GiB and stable PSS from 8.59 to 0.76
GiB. Mean TPOT/ITL was 43.72 ms versus vLLM's 38.55 ms. Profiling attributes the
residual to the discrete-CUDA sampled-token D2H path synchronizing the main
stream instead of retaining vLLM's event-overlapped device mapping. The repair
has since been transplanted onto current `main`; that development branch is
pending the same-series revalidation, so these numbers do not yet bind to it.
This local 4B diagnostic does not establish 27B/35B support. Exact evidence and
reproduction:
[Qwen3.5-4B main repair](bench-evidence/qwen35-4b-main-repair-20260725.md).

There is no front-page race clip yet; when one is produced it will follow the
LocalAI house style (side-by-side, identical output, honest measured ratios).

## Backend detail

**CUDA architectures.** The runtime-gated production arch is GB10 `sm_121a`
(every gate model, every benchmark). A build-supported cross-family fan-out
(`sm_80/86/87/89`, `sm_90a`, `sm_100a/103a`, `sm_110`) compiles single-arch,
portable-kernels-only (all fp8/fp4/CUTLASS/Marlin/FA2 fast paths resolve EMPTY).
**As of 2026-07-27, `sm_110` is RUNTIME-VERIFIED (portable bf16 path) on real
silicon — the FIRST non-GB10 runtime proof.** vllm.cpp was built natively for
`sm_110` on an NVIDIA Jetson Thor board (aarch64, JetPack R38, nvcc 13.0,
on-box `compute_cap=11.0`; cutlass absent and not needed), Release `-Werror`
0 warnings, 16 TUs of real `sm_110` SASS. It then ran the Llama-3.2-1B
paged-engine greedy gate and was **STRICT token-exact 12/16 prompts (192/192
tokens) vs the committed vLLM oracle golden** (every vLLM-deterministic prompt),
**15/16 bit-identical to the GB10 `sm_121a` anchor**; the remaining 4/16 are the
ratified bf16 near-tie prompts (committed teacher-forced gap 0.000 nats). Scope
is precise: RUNTIME-VERIFIED covers only the portable bf16 path that actually ran;
the fp8/fp4/CUTLASS fast paths on `sm_110` remain DERIVED/NOT-YET (a cutlass-backed
kernel campaign). The other fan-out boards remain build-supported only (no board
here). Repro and evidence: [docs/BENCHMARKS.md](BENCHMARKS.md),
[.agents/backend-matrix.md](../.agents/backend-matrix.md) `BACKEND-CUDA-SM110`.

**Metal (Apple Silicon), indicative, not binding.** Two models run end to end
and pass correctness (OPT-125m, Qwen3-0.6B); 18 of 75 ops are native, the rest
fall back to CPU on unified memory. Kernel work including mma prefill attention
(4.3x), a vectorised decode V accumulation (+1.66%) and vectorised prefill
attention staging (+0.50%) puts warm b=1 throughput at about 96.3% of MLX-LM. Bisecting the GEMM puts it at 97% of MLX's own
(mma issue rate 3.91 TFLOP/s), so the residual is NOT GEMM-led: it is spread
across prefill attention, small prefill kernels and decode, none dominant. The
decode share is 93% weight streaming at 83% of the part's memory peak, so the
remaining items sit against roofs rather than against defects: GEMV at 83% of
memory peak, GEMM at 97% of MLX's, and a re-test of GEMV memory-level
parallelism under the paired harness confirmed the earlier exclusion at -1.03%.
The decode residual is non-GEMV overhead (2.7 ms/token against MLX's implied
2.0), so the next lever is a fused qk-norm-RoPE attention
preamble. The dense recipe (kAttnQkNormRope) is composite-only on every backend
(fast_op = kNoFastOp), so this meant writing the kernel rather than porting one.
That kernel now exists and is validated on Metal (worst element error 1.4e-06 vs
the composite); the default bf16 path is now routed through it
(gated on the rope cache), worth a measured +0.35% and taking warm b=1 throughput
to about 96.0-96.4% of MLX-LM, with decode at 98.0%. The largest remaining lead
is prefill attention, which at 547 GFLOP/s is still 5.2x slower per FLOP than
this device's own GEMM; its online softmax now runs one simdgroup per query
row rather than one thread, worth -16% on the kernel and +0.19% end to end. Its
remaining 4.4x FLOP-rate gap to this device's own GEMM has no open structural
idea left: BK deepening is blocked by a bf16-P precision constraint and eliding
the identity softmax rescale measured no gain. The V-accumulation win came from BISECTING the attention
kernel once a paired ABBA harness with cooldown made 0.2% differences
measurable: the V loop moved 2 bytes per lane per load where the score loop
moved 8, giving 29 GB/s against 64 on identical traffic. That also explains why
five earlier decode hypotheses (fusion, split-K, KV layout, threadgroup width,
barrier depth) were each refuted — the limiter was bytes per instruction in one
loop, not parallelism or layout. An optional MLX GEMM provider is available via
`-DVLLM_CPP_MLX=ON` and currently measures net slower than our own kernels. Full
per-lever chronology: [docs/BENCHMARKS.md](BENCHMARKS.md) and
[.agents/specs/metal-dispatch-attribution.md](../.agents/specs/metal-dispatch-attribution.md).

**CUDA architectures.** The production target is GB10/`sm_121a` (runtime-gated,
both gate models token-exact + at/above vLLM speed). The arch-additivity
framework makes adding a CUDA architecture a data edit; `sm_120a`, `sm_90a`,
Ampere/Ada `sm_80/86/87/89`, datacenter Blackwell `sm_100a/103a` and `sm_110` are
**BUILD-supported, portable-kernels-only** (they compile `-Werror`-clean and emit
real per-arch SASS on dgx, but no such board runs here, so none is runtime
support). The **Ampere major-8 fast-path bring-up is now SPIKED** as a
derive-and-ship campaign
([.agents/specs/cuda-arch-ampere-fastpath.md](../.agents/specs/cuda-arch-ampere-fastpath.md)):
the fast-path kernel bodies vLLM builds for Ampere (FA2, Marlin int4 W4A16 +
fp8-input, AllSpark W8A16, CUTLASS scaled-mm C2x) will be ported 1:1, build-verified,
and shipped **labeled** DERIVED+BUILD-VERIFIED (testing-welcome) — a build is never
a runtime claim. AGX Orin (`sm_87`) is the one reachable non-GB10 board and the
sole target that reaches RUNTIME-VERIFIED, gating the first non-GB10 runtime proof
token-exact vs the vLLM 0.25.0 oracle. **FA2 Ampere (WA-1) has LANDED
DERIVED+BUILD-VERIFIED (testing-welcome) — the `ROAD-V1-D1-CUDA` first brick:** the
`fa2` FEATURE-TABLE cell was widened to `8.0,8.6,8.7,8.9,12.0a,12.1a` (the vendored
kernels are already the `_sm80.cu` FlashAttention-2 bodies, `__CUDA_ARCH__>=800`)
and the FA2 build gate decoupled from the sm_12x NVFP4 flag onto arch-independent
CUTLASS-header availability; single-arch `87` and `80` dgx builds are `-Werror`
0-warn with `cuobjdump`-proven real `sm_87`/`sm_80` FA2 cubins (`86`/`89` inherit
same-major), and `sm_121a` is unchanged (still resolves `121a`, NVFP4 GEMM still
ON) — verified by a re-run gate. NO Ampere board ran it here. AllSpark and
scaled-mm C2x are new bodies; Marlin needs its int4/fp8 instantiations. The whole CUDA-arch
fast-path + beyond-vLLM breadth effort is tracked on the roadmap as
`ROAD-V1-D1-CUDA` (derive-and-ship + a public tested/untested signal matrix), with
AGX Orin (sm_87) and NVIDIA Thor (Blackwell) as the reachable runtime-gate boards.

**Datacenter CUDA arches (Hopper `sm_90a`, Blackwell `sm_100/103/110`) — build-only, fast-path SPIKED.** These arches compile today as **portable-kernels-only** (`build-only`; every wgmma/tcgen05 fast-path FEATURE-TABLE cell resolves EMPTY, no board here) — see [.agents/backend-matrix.md](../.agents/backend-matrix.md). The FRAMEWORK (arch-additivity seams) is done; the FAST-PATH kernel bodies are now scoped for **derive-and-ship** (the llama.cpp model) in [.agents/specs/cuda-arch-datacenter-fastpath.md](../.agents/specs/cuda-arch-datacenter-fastpath.md). The CUTLASS C3x / NVFP4-tcgen05 / grouped-MoE / MLA kernels are CUTLASS template instantiations (the wgmma/tcgen05 MMA lives in cutlass 4.5.0, selected by `ArchTag`), so they are 1:1-portable and BUILD-VERIFIABLE here via a single-arch `90a`/`100a` cross-compile + `cuobjdump` SASS proof, with NO Hopper/B200 board. Such kernels will ship LABELED `DERIVED+BUILD-VERIFIED (testing-welcome)` — a faithful port with a compile+SASS proof, HONESTLY untested (a green link is not execution evidence); cloud-GPU token-exact + every-axis runs upgrade the label to runtime-verified. DeepGEMM (runtime JIT/autotune) is the one path that is `not-yet-buildable / needs-real-port` and ships no fake.

**CUDA arch breadth beyond vLLM (Pascal / Volta / Turing), scoping only.** These
are older NVIDIA arches vLLM DROPS but llama.cpp still supports, so this is a
"support more than vLLM" lane with llama.cpp as both the kernel source and the
competitor floor. Our portable attention uses bf16 tensor-core WMMA, which does
not exist before Ampere, so these arches need a new fp16/non-tensor-core kernel
body (ported 1:1 from llama.cpp's `fattn-tile`/`fattn-vec`). State: a committed
scoping spike ([.agents/specs/cuda-arch-breadth-fp16.md](../.agents/specs/cuda-arch-breadth-fp16.md)),
no code. The lane is derive-and-ship: **Turing (`sm_75`, e.g. the cloud-common
T4) is buildable on our current CUDA 13 toolkit** once the bf16-WMMA path is
compile-guarded and the fp16 body is ported, and would ship labeled
"derived from llama.cpp, build-verified, not hardware-tested here, community
testing welcome". **Volta (`sm_70`, V100) and Pascal (`sm_60`/`sm_61`, P100/P40)
are not-yet-buildable** because CUDA 13 dropped their code generation and no
CUDA 12.x toolkit is provisioned here. There is no vLLM oracle on these cards
(vLLM will not run there), so a real correctness test uses llama.cpp on the same
card plus a newer-card/CPU cross-check; nothing is runtime-verified yet.

## Serving and API notes

- **Automatic prefix caching (APC)** is on by default for dense models (hybrid /
  GDN and attention-free default off, mirroring vLLM), and it now has an
  end-to-end cache-ON gate on `Qwen/Qwen3-4B` (a shared common prefix reused
  across requests). A cache hit never changes the output: cache-ON and cache-OFF
  produce token-identical greedy decodes (differing only where the bf16 model
  itself is a genuine near-tie, exactly as vLLM's own greedy does), and both
  match vLLM's own greedy under teacher-forcing. The prefix cache measurably hits
  (0.807 hit rate on that workload) and pays off: on a cached request the time to
  first token drops about 1.76x versus the cold first request (70 ms to 40 ms on
  the GB10), because the shared prefix's attention is not recomputed. Hit-rate
  statistics are counted per vLLM's own counters. Block-hash extra keys
  (multimodal hash, LoRA adapter name, `cache_salt`) are folded into the block
  hash exactly as vLLM does, so requests differing only in those never share a
  cache block (proven no-false-share). The `cache_salt` OpenAI request field is
  not surfaced yet. A Prometheus `/metrics` endpoint is available (opt-in): it
  exposes vLLM's `vllm:*` metric names, help text, types, buckets and
  `{model_name, engine}` label schema in text format 0.0.4, gated against vLLM's
  own scrape spec. It now serves live per-step values, not just the schema: each
  engine step folds the scheduler snapshot (running/waiting counts, KV-cache
  usage, prefix-cache queries and hits) and the iteration stats (prompt and
  generation token counts, request-success counts, and the time-to-first-token,
  inter-token-latency and end-to-end latency histograms) into the registry,
  matching vLLM's own mapping. The scheduler also records per-request
  QUEUED/SCHEDULED/PREEMPTED engine-core events (1:1 with vLLM, behind the same
  stats gate), which the frontend folds into the per-request queue, prefill and
  inference timing histograms and the preemption counter, so those series now
  carry real durations rather than staying at zero. A behavioural CPU gate drives
  the reference engine for several steps and checks the values track the run. The
  remaining work is the async production-serving path wiring, the chat/completion
  response-body timing surface, and the config-gated metric families (speculative
  decoding, KV connector, multimodal cache, LoRA).
  matching vLLM's own mapping. A behavioural CPU gate drives the reference engine
  for several steps and checks the values track the run. The remaining work is
  the async production-serving path wiring and the config-gated metric families
  (speculative decoding, KV connector, multimodal cache, LoRA).
- **SGLang RadixAttention behavior parity** — scoped 2026-07-27, W1+W2 flags now
  IMPLEMENTED (CPU-gated). SGLang's radix-tree prefix cache is functionally
  equivalent to our block-hash APC (both do automatic longest-prefix KV sharing
  with LRU eviction; the only delta is token/page vs block sharing granularity,
  which is bounded and never changes the output), so RadixAttention is fused into
  APC and `--enable-radix-attention` / `--disable-radix-attention` are aliases for
  the existing prefix-cache toggle (also reachable via the C-ABI tri-state
  `vllm_model_params.enable_prefix_caching`, ABI v7), not a second cache. SGLang's
  overlap scheduler is likewise already our async scheduler. The one
  genuinely-distinct SGLang behavior — cache-aware admission that reorders the
  waiting queue by longest prefix match — is implemented behind
  `--schedule-policy=lpm` (`SchedulerPolicy::kLPM`, output-neutral: it changes only
  which request is admitted first, never any request's tokens; default stays
  `fcfs`). SGLang's in-batch prefix-collision de-prioritization (SW2) is now also
  implemented inside the `lpm` path (a later request that would collide on the
  same not-yet-cached prefix as an earlier waiting request is sorted behind the
  non-colliding requests), still output-neutral. Note: this de-prioritization
  changes only admission order here — our prefix cache already caches a request's
  blocks at allocation time, so a second same-step request sharing an uncached
  prefix already reuses the first's blocks (the redundant-prefill this avoids in
  SGLang, whose cache updates only after the forward pass, does not occur for us).
  Jump-forward constrained decoding (SW3) now has its SAFE subset implemented: an
  opt-in driver (env `VT_ENABLE_JUMP_FORWARD`, default off) emits a grammar-forced
  token WITHOUT a model step ONLY where the grammar leaves exactly one valid token
  at a non-accepting state — provably byte-identical to normal per-token
  constrained decode (the constrained sampler has a single finite-logit token
  there), so no re-tokenization is needed. The general case (a forced byte run
  with several possible tokenizations, which SGLang handles by re-tokenizing and
  rolling back the boundary token) is deliberately not jumped, and wiring the
  driver into the live decode loop (jumped tokens need their KV recomputed) is a
  named residual — so the flag stays off by default. The radix eviction-policy
  knob (SW4) stays a named/deferred opt-in. See
  `.agents/specs/sglang-radixattention.md`.
- **KV persistence to disk / CPU offload** is built (CPU and disk tiers,
  identity-checked blocks, a size-budgeted disk tier) and wired opt-in into the
  scheduler through an abstract `KVConnector` ABI selected by a
  `KVTransferConfig` (a disk-offload connector and the LMCache `lm://` client,
  over the same seam), off by default. The LMCache `lm://` client interoperates
  with a real vLLM+LMCache peer (its cache keys are byte-for-byte identical to
  LMCache's own `ChunkedTokenDatabase`, chunk 256, vLLM's `sha256_cbor`, and a KV
  prefix a real peer wrote is found and loaded back byte-identically over the
  wire), and it is now proven END TO END inside a real generation loop: with the
  connector ON against a live `lmcache.v1.server`, an OPT-125m run loads a
  previously stored prefix's KV and shortcuts prefill, generating tokens
  BIT-IDENTICAL to the connector-OFF cold run (verified both after an in-process
  restart and from a genuinely cold second process). The worker-side KV
  store/load is implemented **for the LMCache connector only**: the CPU/disk
  `OffloadingConnector` implements the scheduler half alone (it shortcuts prefill
  for matched blocks, but nothing moves their bytes into the KV pages), so the
  engine now REFUSES to wire it at construction, on every device, naming the
  connector and the device, rather than serving output computed over KV that was
  never written. Both connectors are selectable from the server with
  `--kv-transfer-config '<json>'`, mirroring vLLM's own flag and JSON shape. The
  disk connector's worker half remains unimplemented and is the named remaining
  step. Usage guide: [docs/KV-OFFLOAD.md](KV-OFFLOAD.md).
- **KV-cache events** (the stream external KV routers and prefix-cache-aware load
  balancers subscribe to) are built and off by default. When enabled, the
  prefix-cache block pool emits `BlockStored`, `BlockRemoved`, and
  `AllBlocksCleared` events at the store, evict, and reset sites, and they
  serialize to a `msgpack` payload that is byte-for-byte identical to vLLM's
  `msgspec` encoding (checked against captures for both the default
  int-truncated block-hash form and the raw-bytes form). Event generation and the
  wire payload go through an abstract publisher seam that mirrors vLLM's
  `--kv-events-config`. The live ZMQ socket transport (the actual publish to an
  external router, with its replay buffer and per-rank ports) is not wired yet,
  so today the events can be generated and encoded but not yet streamed over a
  socket; the seam is built so a real ZMQ publisher drops in without touching the
  emission sites or the encoder. Because it is off by default, the prefix-cache
  path is byte-identical to before.
- **Tool calling** covers the dialect set in the capability table above,
  including the Qwen3-Coder XML parser (`qwen3_coder` / `qwen3_xml` / `mimo`).
  Selection over HTTP is `--tool-call-parser` and `--reasoning-parser`, mirroring
  vLLM's own flag names, alongside the C ABI's
  `vllm_model_params.tool_parser` / `.reasoning_parser`. Both flags keep the
  server's previous behaviour as their default (`hermes`, reasoning off), take
  `auto` to opt into the same chat-template detection the C ABI uses, take `none`
  to disable, and abort startup listing every registered name if given an unknown
  one.
- `/health` reports process liveness rather than a full engine-health probe.
- **Speculative decoding** ships user-facing via `--speculative-config` (OpenAI
  server, example CLI, and C ABI v6), unset by default and byte-identical to the
  non-speculative engine when unset. **MTP (k=1)** on the Qwen3.5/3.6 gate
  checkpoints (those shipping an `mtp.*` head) is token-exact to both our own
  spec-off output and vLLM's own MTP greedy at concurrency 1, and about 1.04x
  faster than vLLM's spec-on decode; the concurrent (multi-request) path shares a
  scheduler step with ordinary prefills over a bit-exact GDN split-merge. Honest
  caveat: above concurrency 1 the 27B greedy output is not bit-stable across
  batch shapes, so exact spec-on/spec-off token agreement is a concurrency-1
  property. **Block-diffusion DFlash** (method `dflash`, the z-lab Qwen3.6-27B
  draft over the 27B NVFP4 target, wired through the same `--speculative-config`
  with an external draft `model` path) is correctness-complete (ratified
  near-tie) and at or above vLLM throughput at concurrency 1 (29.32 vs 29.24
  tok/s, 1.003x), built D0 through D14 on the vLLM 0.26.0.dev0 stack (which
  resolves vllm#40898) and recorded DONE across the engine, model and kernel
  matrices. **ngram** (method `ngram`) is the draft-free suffix-match proposer,
  reusing the same verify/reject loop with no draft model; it is token-exact vs
  vLLM's own ngram on repetitive workloads (27B, 5/5 prompts, every draft
  accepted) and off by default (byte-identical). Measured A/Bs and the D0-D14
  chronology: [docs/BENCHMARKS.md](BENCHMARKS.md); usage guide:
  [docs/SPECULATIVE-DECODING.md](SPECULATIVE-DECODING.md).
- **LoRA and multi-GPU** are not supported yet. **Multimodal** is
  correctness-complete on a single-sequence eager path but not yet wired into the
  OpenAI server, and it is the top roadmap priority. Image-to-text is strict
  token-exact 32/32 vs the vLLM 0.25.0 oracle on both Qwen3-VL-4B and our own
  gate model Qwen3.6-27B (`Qwen3_5ForConditionalGeneration`); video-to-text works
  end-to-end on both (Qwen3-VL-4B near-tie-robust, Qwen3.6-27B strict 32/32); and
  audio-to-text works end-to-end on Voxtral-Mini-3B (near-tie-robust, decoder
  proven token-exact 48/48), the first audio understanding in the tree. The
  pipeline is a pure C++ port end-to-end: the Qwen3-VL image/video processors and
  the Whisper-class audio processor are bit/byte-identical to the oracle, the
  Qwen3-VL vision tower and the Whisper encoder tower are proven faithful within
  the bf16 envelope, and the MRoPE/DeepStack text-backbone contracts are
  unit-exact. Concurrency-1 speed is measured against vLLM 0.25.0's graphed
  config: on Qwen3.6-27B image the decode is at parity and the vision tower,
  after a warp-scoped online-softmax attention kernel plus a resident-weight
  load, is 148 ms/image (0.59x of vLLM's eager encode, faster) with the strict
  image/video gates still 32/32. On Voxtral audio the decode is now graph-captured
  and measured at 60.9 ms per token (vs 61.7 ms eager, a small real
  non-overlapping win), against vLLM's graphed 40.8 ms, so the audio gap is about
  1.49x: the removable per-step launch overhead turned out to be only about 1.25%
  of the per-token time, so the residual is per-step compute (kernel) efficiency,
  not launch overhead. That residual is now fully attributed: nsys shows all of it
  is the naive scalar paged-attention decode kernel (723 us per call over 30
  layers, about 120x the KV memory floor). vLLM runs decode attention on
  flash-attention varlen, which our tree already ships and enables by default for
  this head shape; Voxtral only missed it because the decode driver allocates one
  KV block of size 444, not a multiple of 16, which disqualifies the fast path.
  Rounding that to a multiple of 16 routes decode through the flash kernel and
  reaches 38.2 ms per token, a 36% win that beats vLLM's 40.8 ms, and the
  resulting sequence teacher-forces vLLM with zero divergences (a valid greedy
  branch). It is a bf16 near-tie ceiling: it changes which side of an exact tie is
  taken, so it no longer reproduces the committed near-tie golden byte-for-byte
  and is not adopted here; the byte-exact scalar path is kept at 14/14 with the
  golden unchanged, and the win is one line plus a golden regeneration away.
  On-GPU greedy sampling is in place on both mm decode loops. The open work is
  batched/graphed mm serving, adopting the audio decode win, and higher-concurrency
  throughput. Design and status:
  [.agents/specs/multimodal-track.md](../.agents/specs/multimodal-track.md),
  [.agents/specs/audio-track.md](../.agents/specs/audio-track.md), and
  [.agents/specs/multimodal-speed.md](../.agents/specs/multimodal-speed.md).
- **Environment variables.** A handful of behavior-changing knobs (CPU thread
  count, prefix-cache hash seed, LMCache host/port, the default-on async and CUDA
  fast paths and their rollbacks, the portable-reference bisect switch, GGUF
  loading behavior, the Vulkan device selector) are documented in
  [docs/ENVIRONMENT.md](ENVIRONMENT.md). A CI check
  (`scripts/check-env-doc.py`) keeps that reference honest: any new production env
  var must be documented or explicitly classified kernel-internal.

## Verification and parity

Every model is gated token-for-token against the vLLM 0.25.0 oracle (the
gate-time pin, re-validated bit-identical on the current 0.26.0.dev0 pin) on the
same workload, and every change that could affect correctness or performance is
compared apples-to-apples against vLLM with both numbers and the ratio recorded.
Behavioral CPU tests run under CTest; CUDA correctness, sanitizer, trace, and
performance evidence is recorded per feature rather than inferred from source.
The protocol is in [`.agents/gates.md`](../.agents/gates.md) and
[`.agents/benchmark-protocol.md`](../.agents/benchmark-protocol.md). A CI check
(`scripts/check-fusion-consistency.py`) additionally keeps model forwards routing
their fusable add+RMSNorm glue through the portable fusion catalog rather than
hand-fusing it.

The CPU CTest suite is green (0 real regressions). A 2026-07-27 hygiene pass
triaged the previously-red tests as stale assertions or `-j` contention, not
regressions: the model-registry count assertion tracks the 24 registered
architectures, the GGUF loader rejection case uses a genuinely-unregistered arch,
the DFlash proposer test supplies the now-required `"model"` key, and the two
HTTP-server tests run under CTest `RUN_SERIAL` so a CPU-hog concurrent test can no
longer starve their accept thread.
