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
| GGUF loading (F32/F16/BF16/Q4_0/Q8_0/Q3_K/Q4_K/Q5_K/Q6_K/NVFP4) | Supported; compute-in-quant on CPU for the six block encodings; NVFP4 is materialize-only | Weights in six block encodings stay compressed from file to matmul on CPU (no BF16 expansion). NVFP4 (ggml type 40) now DEQUANTIZES, including the per-tensor (per-expert) `<stem>.scale` sidecar the container keeps outside the blocks; gated BIT-EXACT against the compressed-tensors NVFP4 path on real Qwen3.6-27B bytes from both containers. It expands to bf16 (no NVFP4 GGUF GEMM), and no NVFP4 GGUF model has been run end to end |
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
| Multimodal: image to text | Correctness-complete; vision-forward speed beats vLLM; OpenAI-server content-part parse + processor routing + engine mm-request plumbing landed (CPU); end-to-end serving pending | Strict token-exact 32/32 vs vLLM 0.25.0 on Qwen3-VL-4B and Qwen3.6-27B (`Qwen3_5ForConditionalGeneration`); C++ image processor + vision tower + MRoPE/DeepStack backbone with on-GPU greedy sampling; the 27B decode step is now graph-capturable (routes through the production captured decode, token-exact held). The vision tower now defaults to the flash-tiled non-causal attention (`vt::AttentionDenseFlash`, byte-identical to the previous warp kernel — token-identical, goldens unchanged): the per-image tower forward is ~142 ms vs vLLM 0.25.0's ~250 ms eager encode = 0.57x (faster). Attribution-first profiling found the tower attention is serial-latency-bound (not K/V-bandwidth-bound), so flash tiling is only a 1.04x same-binary A/B over the warp kernel here — the tower already beats vLLM; batched serving speed still pending. OpenAI-server multimodal wiring — CPU bricks 1+2 landed (`CLAIM-MM-SERVING-W1`/`W2`, .agents/specs/mm-serving.md): the chat request parses the OpenAI multimodal content-part array (`image_url`/`input_audio`/`audio_url`), decodes the base64/`data:` payloads, routes them through the existing single-sequence processors to a placeholder-expanded prompt + mm-feature handles, AND the engine now carries them — additive `LLMEngine`/`AsyncLLM` `add_request(MultiModalInputs)`/`generate` overloads (via `InputProcessor::process_inputs_mm`, mm_features onto `EngineCoreRequest`/`Request`), chat-template placeholder-string helpers, and a serving_chat `MultiModalChatFn` seam (default unset ⇒ text path byte-identical). The W3 `MultiModalChatFn` seam BODY now lands (`CLAIM-MM-SERVING-E2E`): `MakeQwen3VLImageChatFn` (chat_mm.{h,cpp}) turns an image chat request into the placeholder-EXPANDED engine input — marker-inject at the mm part position, render the chat template, tokenize (the single `<|image_pad|>` maps to one image_token_id via `EncodeWithSpecialTokens`), then `RouteImageRgb` EXPANDS to 196 image tokens + mm_features — and is wired into `examples/server/main.cpp` (guarded on `preprocessor_config.json`; text-only models leave the seam unset ⇒ byte-identical). Unit-gated `test_chat_mm` 8/8 (the W3 seam-body test drives the real tokenizer + chat template → 196 image tokens; RED line = the text-only path renders 0) + `test_input_processor` 10/10 + `test_openai_serving` (the production seam is invoked on an image request + routed to the engine mm generate overload; text-only never touches it; streaming+mm rejected), CPU-only, no model weights; bare-string chat requests byte-identical (proven). **Residual — the closing GPU `MM-SERVE-E2E` forward is ARCHITECTURALLY BLOCKED (not box contention):** the engine model runner has no multimodal forward — `ModelForwardInput` (`model_registry.h`) has no vision/mm-embed field, `runner.cpp` never reads `Request.mm_features`, Qwen3-VL is not engine-registered, and the M2c `Qwen3VLGenerateGreedy` is a standalone driver outside `ModelRegistry::Forward`. Closing it = fold that forward INTO the registered engine forward (add a vision-embed field to `ModelForwardInput`; runner runs the tower via the encoder cache + merge + MRoPE/DeepStack; register the arch), then a dgx token-exact gate vs the M2c golden — exact recipe in `.agents/specs/mm-serving.md`. **README FLAG:** the README's "not yet wired into the OpenAI server" line is now partly stale (the serving seam IS wired CPU-side; only the GPU forward is residual) — the concurrent session owns README and should update it once the GPU e2e lands. See docs/BENCHMARKS.md |
| Multimodal: video to text | Correctness-complete, speed-pending; OpenAI-server content-part parse + engine mm-request plumbing landed (CPU), end-to-end serving pending | End-to-end on Qwen3-VL-4B (near-tie-robust) and Qwen3.6-27B (strict 32/32); reuses the image tower and temporal MRoPE plus video preprocessing (frame sampling, temporal grid) |
| Multimodal: audio to text | Decode beats vLLM; encoder TTFT measured and improved but not yet at parity; OpenAI-server content-part parse + audio processor routing + engine mm-request plumbing landed (CPU), end-to-end serving pending | End-to-end on Voxtral-Mini-3B (near-tie-robust, decoder token-exact 48/48) plus a Whisper-class encoder tower. Decode is graph-captured and runs vLLM's FA2 varlen split-KV kernel (the KV block-size is rounded up to a multiple of 16 to meet its precondition): audio TPOT 39.5 ms/token BEATS vLLM 0.25.0's 40.8 ms (0.97x, non-overlapping bands). The Whisper encoder attention now uses a flash-tiled non-causal head-dim-64 kernel (`vt::AttentionDenseFlash`): a block of query-warps shares each streamed K/V tile out of shared memory (FlashAttention K/V tiling), with the per-warp online-softmax math copied verbatim from the warp kernel so the output is bit-identical (token-identical). Same-binary A/B: the encoder self-attention drops from 35.11 to 19.29 ms/layer (1.82x, non-overlapping) and the encoder forward from ~1.83 s to ~1.37 s (1.33x), token-identical output (16/16, STRICT golden unchanged, sanitizer 0). The encoder weights are now device-resident: each of the 487 encoder weight tensors is converted to bf16 and uploaded to the GPU once (mirroring the decoder's residency), reused across forwards instead of being re-marshalled every call; same-binary A/B (`VT_WHISPER_ENC_REMARSHAL`) drops the encoder forward a further ~1.37 s to ~0.73 s (1.89x, non-overlapping), removing ~648 ms of per-call host weight marshalling (nsys: 974 fewer Host-to-Device copies, ~2.5 GB less traffic), byte-identical (16/16, STRICT golden unchanged, sanitizer 0). But encoder TTFT (~0.73 s) is still far above vLLM's 43 ms (~17x, was ~32x): the residual is now GPU-compute-bound (the scalar warp-per-query attention plus the conv GEMMs), so closing it needs a tensor-core MMA head-dim-64 non-causal flash attention. Correctness held under the ratified distributional near-tie gate (teacher-force PASS, 0 divergent, strict prefix exact vs vLLM; STRICT golden unchanged; 16/16). Remaining: a tensor-core MMA encoder-attention kernel and dropping the conv host round-trip (a device im2col kernel), and there is no batched c2+ or `audio_url` serving ingestion. See docs/BENCHMARKS.md |

| SGLang parity (competitor floor + oracle) | Oracle STOOD UP + first floor MEASURED (cache-neutral, 27B, c8/c16): throughput/TTFT WIN, TPOT/ITL open GAP | SGLang v0.5.15 (`f63458b`) whole runtime surface inventoried (44 rows: 23 FUSED into our vLLM-derived engine, 8 SGLANG-DISTINCT opt-ins, 5 inventoried, 8 out-of-scope). The `v0.5.15-cu130` arm64 image (`@sha256:d0a667e`) PULLED and RAN the 27B-NVFP4 gate model on GB10 sm_121a with no from-source build. First reproduced SGLang-vs-ours cache-neutral comparison (27B, 3 reps, idle box, one flock, engines sequential): **ours beats the SGLang floor on total/output throughput + req/s (2.21×@c16, 1.44×@c8) and TTFT (6–12× lower), but SGLang wins per-token latency (TPOT/ITL 1.18–1.49× below ours) — a reproduced OPEN GAP.** SGLang is a competitor, not the mirror source (vLLM stays behavior truth). Residuals: 35B, c1/c2/c4 low-conc sweep, shared-prefix cache-ON arm, token-exact cross-check. Numbers + repro: `docs/BENCHMARKS.md`, `.agents/sglang-matrix.md`, `.agents/specs/sglang-parity-oracle.md`. **UPDATE 2026-07-28 (`CLAIM-DECODE-LATENCY-EXPLORE`, measurement only, no source changed): the TPOT/ITL gap is CONFIRMED batch-composition, NOT a decode-kernel deficiency** — on our engine ITL(decode-batch=1)=101.75 ms is already ≤ SGLang's op-point 104–105 ms and rises monotonically with batch (→158.5 ms @ B16); nsys shows every hot decode kernel sub-linear in batch (per-token cost ↓~10×); SGLang's effective decode concurrency is ~4 (not 16) due to its 33 s admission queue, so its low ITL is simply the ITL of a small batch. Our throughput win IS the ITL cost — same lever; knob `max_num_seqs`/`max_num_batched_tokens` already exists, latency-oriented point `max_num_seqs≈8` = ITL −21% at 1.38× SGLang throughput; default stays throughput-oriented. Full data: `.agents/specs/decode-latency-lever.md` |

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

LoRA, multi-GPU, and the full tool-calling template surface. **Scale-out /
distributed execution is scoped but unbuilt** (spike, 2026-07-28): the engine is
single-GPU today (verified — no NCCL / tensor-parallel / process-group code in
`src/`). A single-dimension design is on record covering all three legs —
multi-GPU tensor+pipeline parallel, multiple DGX Sparks over the ConnectX-7
200GbE RoCE/RDMA cable (the path that lets DeepSeek-V4-Flash fp8 ~167 GiB run
across 2×119 GiB Sparks), and MLX multi-node over Thunderbolt — all expressed
ONCE against one `vt::` collective / process-group abstraction with
backend-specific transports (NCCL / RDMA / MLX-ring), mirroring vLLM's
`device_communicators`. `world_size==1` stays byte-identical. **W1 landed
(2026-07-28) the collective ABSTRACTION leg**: `vt::Communicator`
(`include/vt/communicator.h` + `src/vt/communicator.cpp`) with
AllReduce/AllGather/Send/Recv, proven by a CPU in-process multi-rank gate
(`tests/vt/test_communicator.cpp`, a real cross-rank sum, no GPU) and a
byte-identical `world_size==1` no-op. The multi-GPU (TP/PP), multi-Spark, and
MLX transports remain unbuilt — those legs are HW-blocked (no ≥2-GPU box, no
2-Spark cable, no 2-Mac cluster here). Full scope + seam map:
[.agents/specs/scale-out-distributed.md](../.agents/specs/scale-out-distributed.md).
Every parallelism MODE vLLM has (tensor / pipeline / data / expert / sequence /
context parallel) is now enumerated and grounded in upstream source, mapped onto
that one abstraction and priority-ranked, in
[.agents/specs/parallelism-modes.md](../.agents/specs/parallelism-modes.md)
(2026-07-28) — with the honest note that vLLM's "sequence parallel" is a
tensor-parallel compilation pass, not a separate parallel axis.
Multimodal
(image/video/audio) is correctness-complete; its OpenAI-server wiring has landed
all three CPU bricks (content-part parse + processor routing; the engine
mm-request plumbing — `add_request(MultiModalInputs)` on both engines, default-inert
to the text path; and the W3 `MultiModalChatFn` seam BODY that renders an image
chat request into the placeholder-EXPANDED engine input, wired into the production
server). It is not yet servable end-to-end: the closing GPU gate `MM-SERVE-E2E` is
architecturally blocked on the engine model runner having no multimodal forward
(the vision tower + merge lives in the standalone M2c driver, not in
`ModelRegistry::Forward`) — folding that forward into the registered engine path is
the named residual (`.agents/specs/mm-serving.md`).

**Open, not root-caused (observed 2026-07-28):** the C-ABI custom logits
processor case (`tests/capi/test_capi.cpp:410`, ABI v8) SIGSEGVs in a CUDA build
on the GB10 box, while the same suite is 232/232 green on a CPU build. Confirmed
present on pristine `main` `ee3d5960` with no local changes, so it is not a
regression from any in-flight work, but it does mean the "supported" claim for
custom logits processors in the capability table above is verified on CPU only.
One build directory, not yet bisected.

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

**Gemma 4**, the newest registered variant, now has its **text path
correctness-complete and gated** (G1b, 2026-07-28): the
`Gemma4ForConditionalGeneration` language_model stack of `unsloth/gemma-4-E4B-it`
loads through our engine and greedily emits the **exact 32 golden token ids —
STRICT 32/32 token-exact vs the vLLM 0.25.0 golden** (gate
`tests/parity/test_gemma4_paged_engine.cpp`, dgx CUDA). It brings up the large new
primitive stack (per-layer embeddings, YOCO KV-sharing, plain RMSNorm, proportional
partial-RoPE, heterogeneous 256/512 head dims, GeGLU, a per-layer scalar; the
Gemma-4 MoE / k_eq_v / double-wide MLP are off for the E4B checkpoint and stay the
larger-variant follow-on). The G1-named blocker is resolved: **the runner now
allocates a per-layer KV head dimension** (`KVCacheConfig::per_layer_attn_specs`),
so Gemma-4's per-layer 256/512 heads each get a correctly-strided cache — and it is
byte-neutral for every uniform-KV model (the field is empty ⇒ the previous single
uniform allocation; the full CPU runner/KV suite plus the OLMo-2 SACRED GPU gate
16/16 are unchanged). Three additive loader gaps were fixed en route to the
first-ever Gemma-4 forward: nested per-layer `rope_parameters` (config loader), the
Gemma `Replace(" "->"U+2581")` metaspace normalizer (tokenizer), and reading
Gemma-4's per-arch scalars from `raw["text_config"]` rather than the full config
(the real 256/512 head-dim source). Speed vs vLLM is pending. Its
multimodal path (image + video + **audio**, the only
audio-capable model in the pin) has been assessed
([spec](../.agents/specs/gemma4-multimodal.md)) and is now **oracle-gateable —
run-verified** (W0, 2026-07-28): the pinned 0.25.0 oracle (transformers 5.13.1)
loads, runs, and greedily generates the ungated `unsloth/gemma-4-E4B-it`
(`Gemma4ForConditionalGeneration`, 15.99 GB) on GB10, K=5 deterministic, and the
32-token greedy golden is captured
(`tests/parity/goldens/gemma4_e4b_text/`). The earlier "oracle-blocked at gate
time" concern (transformers lacking `gemma4`) is refuted — the module is present
and the model runs. **The IMAGE modality oracle is now captured too** (G2,
2026-07-28): the same E4B model on the pinned 0.25.0 oracle greedily describes a
fixed image (K=5 deterministic ⇒ STRICT gate form, 18 tokens → a coherent
gradient-image caption, 256 soft tokens), committed with four staged vision-tower
reference tensors for per-stage unit-gating
(`tests/parity/goldens/gemma4_e4b_image/`). Grounding the tower corrected an
earlier assumption: the Gemma-4 vision tower is a **custom NaFlex SigLIP2 with
multidimensional vision-RoPE, q/k/v RMSNorm, Gemma-2 sandwich norms, a learned 2-D
position embedding, and a √hidden average-pool-by-position pooler** — it does *not*
drop-in reuse the Qwen3-VL ViT (the block GEMMs/attention are reusable; the
patch-embed, RoPE, norms, pooler and the Gemma-4 NaFlex image processor are new).
The C++ NaFlex SigLIP2 vision tower is now IMPLEMENTED (additive
`gemma4_vision.{h,cpp}`) and PASSES its four per-stage gates vs the
transformers-eager references on the dgx CUDA build (patch-embed rel-L2 2.15e-3,
encoder 3.14e-2, pooled 1.36e-2, projected 1.85e-2; 220/220; compute-sanitizer 0) —
the tower-in-isolation milestone (mirrors Qwen3-VL M2a before its M2c e2e); grounding
it also revealed the vision `Gemma4ClippableLinear` layers carry FINITE trained QAT
activation clamps (not the no-ops the port-map assumed), now implemented. So Gemma-4
multimodal image is blocked only on the remaining engine wiring: the C++ Gemma-4
NaFlex image processor + the mm merge-plumbing (register SupportsMultiModal, the
masked-scatter merge of the 256 projected soft tokens at the image placeholder rows,
the tower→merge→decode fork) for the image→text e2e. **The USM-Conformer AUDIO
tower is now IMPLEMENTED too** (G3, 2026-07-28): the additive
`gemma4_audio.{h,cpp}` forward (`Gemma4AudioModel` + the audio projector) PASSES all
seven per-stage gates f32-exact vs the transformers-eager reference (host f32,
1256/1256: subsample rel-L2 5.4e-7, position-embeddings 8.9e-8, block0 4.2e-7,
block_mid 3.8e-7, block_last 4.4e-6, output_proj 5.9e-6, projected 6.3e-6), ported
1:1 from `modeling_gemma4.py` — a 2×Conv2d subsample, a relative positional
encoding, and 12 Conformer layers (half-step feed-forwards, a chunked-local
attention with Transformer-XL relative-position bias + a tanh soft-cap +
per-dim-scale softplus, and a GLU + depthwise-causal-conv light-conv module), then
`output_proj` and the audio embedder; the FINITE QAT clamps and the exact sliding
window (`dist ∈ [0,12)` — a RED-first-caught off-by-one) are grounded in the source.
So Gemma-4 now has all three modalities tower-proven (text STRICT 32/32, vision and
audio per-stage), and the remaining audio work is the same engine wiring as image —
the Gemma-4 audio feature extractor (mel frontend) + the mm merge-plumbing — plus
a device-resident bf16 forward for speed. The Gemma-4 MoE / k_eq_v / double-MLP
backbone stays the larger-variant follow-on. Audio as a standalone modality is also
proven end-to-end on the smallest oracle-runnable audio models (Whisper encoder,
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
**spiked-only**, per the [model matrix](../.agents/model-matrix.md).
**DeepSeek-V4-Flash** is scoped with W1/W2 CPU scaffolding landed (not yet
runnable) in [deepseek-v4-flash.md](../.agents/specs/deepseek-v4-flash.md): a
~167B/256-expert NEW architecture (DeepSeek Sparse Attention MLA + Manifold
Hyper-Connections + NVFP4/MegaMoE + sqrtsoftplus/hash MoE). As of 2026-07-28 the
additive registry stub + config parse + checkpoint loader name-map are landed and
VERIFIED against the real `nvidia/DeepSeek-V4-Flash-NVFP4` safetensors header
(HTTP-range, no download); the forward is an honest not-yet-implemented stub.
**HW-fit correction:** that NVFP4 checkpoint is **156.7 GiB**, NOT the ~83 GiB the
scoping spike estimated (only the 256 routed experts are W4; the MLA and shared
linears are FP8 plus NVFP4 double-scale overhead), so it does **not** fit ONE
GB10's 119 GiB unified pool. **Single-Spark IS viable via a ~2-bit GGUF (user
correction 2026-07-28):** `unsloth/DeepSeek-V4-Flash-GGUF` `UD-IQ2_XXS` = 90.9 GB
(3 shards, ungated) FITS one GB10 with ~28 GiB headroom (also UD-IQ1_S/IQ1_M/IQ2_M/
Q2_K_XL). So two vehicles: single-Spark ~2-bit GGUF, or 2x-Spark NVFP4/fp8 over the
interconnect. The GGUF vehicle's correctness reference is llama.cpp-on-box (the
pinned vLLM cannot load V4-from-GGUF) and needs a V4-GGUF loader + IQ i-quant dequant
(IQ1_S/IQ2_XXS — not in our C4 K-quant set). The NVFP4/fp8 vehicle stays multi-node.
A source-level spike (2026-07-28) confirmed a **same-quant IQ2_XXS GGUF benchmark of
our engine vs vLLM is NOT viable today**, blocked on both sides: vLLM 0.26 moved GGUF
out-of-tree to the uninstalled `vllm-gguf-plugin` (which *does* dequant IQ2_XXS) but
`DeepseekV4ForCausalLM` has no `packed_modules_mapping`/GGUF wiring; and our engine
hard-rejects GGUF for DeepSeek-V4/V2 and lacks IQ2_XXS/Q2_K dequant — so even the
`UD-Q2_K_XL` k-quant fallback does not rescue it. Apples-to-apples for DeepSeek-V4 is
the NVFP4 vehicle; a true same-GGUF cross-engine number is only available on a
Qwen3/dense k-quant both engines already load. **W3 attention primitives landed
(2026-07-28):** the genuinely-new-vs-V2/V3 math is ported as portable host references
and unit-gated — the DSA "Lightning Indexer" sparse top-k SELECTION (a weighted
multi-query logit with a load-bearing per-head ReLU, then a causal top-512 token
select), plus the two 512-wide-MLA output pieces V2/V3 lack: per-head attention-sink
softmax and grouped output-LoRA. `test_deepseek_v4_dsa` passes 13/13 (hand-derived
literal cases plus double-precision references, clean CPU `-Wall -Werror -Wextra`);
the gate is honest hand-case + structural review, not a dumped-oracle comparison
(the fixed-config 167B model cannot be built at a tiny shape). It is additive and
byte-neutral for DeepSeek-V2. SGLang v0.5.15 registers and implements
`DeepseekV4ForCausalLM` (the full DSA/MHC stack), so it is a viable second reference,
subject to the same single-GB10 memory limit. The remaining bricks — Manifold
Hyper-Connections, the sqrtsoftplus/hash MoE, the device kernels, and the full model
gate — stay multi-Spark-blocked.
The
frontier families Kimi / MiniMax / GLM-latest are scoped for mechanical porting
in [a dedicated spike](../.agents/specs/sweep-kimi-minimax-glm-latest.md):
Kimi-Linear-48B is the one that fits GB10 (91.5 GiB) and is e2e-gateable, while
MiniMax-M2 (214.3 GiB fp8), Kimi-K2, MiniMax-M3 and GLM-5 remain hardware- or
dependency-blocked (honesty-pass only). **Kimi K3** (released 2026-07-27, after the
pin) is scoped **derive-and-ship** in [kimi-k3.md](../.agents/specs/kimi-k3.md): a
2.8T MoE whose text backbone is the Kimi-Linear KDA+MLA+MoE hybrid massively scaled
(H=7168, 93 layers = 69 KDA + 24 MLA, 896 experts, MXFP4 + MoonViT-V2 vision). It
reuses heavily (our GDN, DeepSeek MLA, DeepSeek MoE, the Qwen3.6-35B GDN-hybrid
skeleton); net-new = the KDA kernel delta, MXFP4 (group-32/e8m0), AttnRes
(report-only, unconfirmed) and the MoonViT-V2 tower. It **does not fit** one GB10
(~1.56 TB MXFP4, ~12× the 119 GiB pool) and is not in the pinned oracle, so there is
no on-box golden — like the beyond-vLLM CUDA bricks; the real signal is a primitive
gate on the fitting Kimi-Linear-48B proxy plus build-verify, with full K3
verification left to a multi-Spark / 16×H200-class box. The W2/W5 CPU scaffolding
is now **build-only** (2026-07-28): an additive registry stub, nested
text/vision/quant config descent, the 93-layer KDA/MLA + 896-expert MoE
text-backbone structural name-map (grounded in `kimi_linear.py`), a REFUSE-by-name
forward and an MXFP4-refuse loader, green on a CPU build with a 6/6 scaffold gate;
MXFP4, the KDA kernel delta and the MoonViT-V2 tower stay not-yet-buildable
(deferred to the shared DeepSeek-V4 MXFP4 row, the Kimi-Linear KDA row, and W7).
The matrix opens with an
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

**GDN Triton-AOT cubins are now vendored per-arch (2026-07-28).** The vendored
Triton-AOT GDN fast-path cubins (the measured codegen-win packed decode plus the
delta_h/chunk_o FLA kernels for the Qwen3.6 GDN-hybrid models) previously existed
for GB10 `sm_121a` ONLY. Because a cubin loads only on the SM it was compiled for
and the cross-family arch builds ship `-DVLLM_CPP_TRITON=OFF`, GDN decode on the
other arches ran the slower spilling hand kernel. The full GDN AOT set is now
regenerated and vendored for `sm_80/86/89/90a/100a` (`cuobjdump` shows real
per-target SASS), so a `-DVLLM_CPP_TRITON=ON` single-arch build on those arches
selects the non-spilling path too — **DERIVED+BUILD-VERIFIED (testing-welcome):
no non-`sm_121` board runs a GDN model here, so this is not a runtime
GDN-decode-parity claim on any arch.** `sm_121a` is byte-unchanged (SACRED gate
intact). Evidence: [.agents/specs/triton-aot-per-arch.md](../.agents/specs/triton-aot-per-arch.md).

**As of 2026-07-28, `sm_87` is also RUNTIME-VERIFIED (portable bf16 SYNC path) on
real silicon — the SECOND non-GB10 runtime proof.** vllm.cpp was built
portable-only for `sm_87` on an NVIDIA Jetson AGX Orin (Tegra R36.4.3 / JetPack 6,
aarch64; the integrated GPU self-reports `sm_87`, unified memory). Because the
Orin driver advertises only CUDA 12.6, the CUDA-13 container is refused by the
NVIDIA container runtime, so the build used the JetPack-6 `l4t-jetpack:r36.4.0`
image (nvcc 12.6) with g++-13 (the tree needs GCC ≥ 13). All fp8/fp4/CUTLASS/FA2
fast paths resolve EMPTY on `sm_87` (Ampere: bf16 + int8, no fp8/fp4; cutlass not
present). It ran the Llama-3.2-1B paged-engine greedy gate and was **13/16 prompts
STRICT token-exact vs the committed vLLM 0.25.0 oracle golden, 16/16 under the
near-tie distributional gate, 0 forward-divergent** (exceeding Thor's 12/16), plus
`test_cuda_backend`/`test_cuda_ops` (461 assertions of real on-device kernel
execution). One honest `sm_87` bug surfaced: the DEFAULT asynchronous runner path
crashes on the first forward with an illegal memory access, so RUNTIME-VERIFIED is
scoped to the portable bf16 **synchronous** path (`VT_ASYNC_RUNNER=0`) — the async
runner on `sm_87` is a tracked unblock item. Repro and evidence:
[docs/BENCHMARKS.md](BENCHMARKS.md),
[.agents/backend-matrix.md](../.agents/backend-matrix.md) `BACKEND-CUDA-SM087`.

As of 2026-07-28, the datacenter-Blackwell `sm_100a` fan-out gained its first
FAST-PATH body: the **NVFP4 tcgen05 block-scaled GEMM is BUILD-VERIFIED**
(`DERIVED+BUILD-VERIFIED, testing-welcome`). A faithful 1:1 type-port of vLLM's
`Fp4GemmSm100` (`ArchTag=Sm100` + `KernelScheduleAuto` — CUTLASS 4.5.0 selects the
5th-gen tcgen05 collective) compiles single-arch `100a` on nvcc 13.0 `-Werror`-equiv
0 warnings and `cuobjdump` shows a real `sm_100a` cubin; it is gated by its own
`cutlass-nvfp4-sm100` feature cell (enabled only for `100a`, so the GB10 `sm_121a`
gate build is byte-unchanged). The native consumer `mma.sync kind::mxf4nvf4` fp4
path stays `sm_12x`-only (it does not port to sm_100's tcgen05). **No B200/sm_100
board ran it** — a green compile + SASS is not execution evidence, not runtime
support, and not vLLM-competitive; the other sm_100 fast paths (CUTLASS C3x FP8,
MoE, MXFP4, MLA) remain scoped.

As of 2026-07-28, the Hopper `sm_90a` fan-out gained its first FAST-PATH body: the
**CUTLASS C3x FP8 (W8A8) scaled-mm GEMM is BUILD-VERIFIED**
(`DERIVED+BUILD-VERIFIED, testing-welcome`). A faithful 1:1 type-port of vLLM's
`cutlass_3x_gemm_sm90_fp8` (`ArchTag=Sm90` + `KernelTmaWarpSpecialized*FP8FastAccum`
— CUTLASS 4.5.0 emits the 4th-gen wgmma/TMA warp-specialized collective) compiles
single-arch `90a` on nvcc 13.0 `-Werror` 0 warnings and `cuobjdump` shows a real
`sm_90a` cubin (ptxas even names `wgmma.mma_async` in the emitted kernels); it is
gated by its own `scaledmm-c3x-sm90` feature cell (enabled only for `90a`, so the
GB10 `sm_121a` gate build is byte-unchanged). The consumer `sm_12x` FP8 scaled-mm
body (`ArchTag=Sm120`) is untouched. **No H100/H200/sm_90 board ran it** — a green
compile + SASS is not execution evidence, not runtime support, and not
vLLM-competitive; the sm90 int8/blockwise C3x legs and the other Hopper fast paths
(FA3, Machete, CUTLASS MoE) remain scoped.

As of 2026-07-28, the datacenter-Blackwell `sm_100a` fan-out gained its second
FAST-PATH body (after the DC1 NVFP4 tcgen05 GEMM): the **CUTLASS C3x FP8 (W8A8)
scaled-mm tcgen05 GEMM is BUILD-VERIFIED** (`DERIVED+BUILD-VERIFIED,
testing-welcome`) — the intersection of DC1's tcgen05 arch and the Hopper DC2 C3x
fp8 kernel. A faithful 1:1 type-port of vLLM's `cutlass_3x_gemm_sm100_fp8`
(`ArchTag=Sm100` + `KernelScheduleAuto`/`EpilogueScheduleAuto` — CUTLASS 4.5.0
selects the 5th-gen **tcgen05** warp-specialized collective, a 2SM
`ClusterShape<_2,_2,_1>` default) compiles single-arch `100a` on nvcc 13.0
`-Werror all-warnings` 0 warnings and `cuobjdump` shows a real `sm_100a` cubin (the
emitted kernels are `Sm100TmaUmmaWarpSpecialized` naming `SM100_MMA_F8F6F4`,
`SM100_TMA_2SM_LOAD` and `SM100_TMEM_LOAD`; SASS carries `LDTM`/`tmem`
tensor-memory ops); it is gated by its own `scaledmm-c3x-sm100` feature cell
(enabled only for `100a`, so the GB10 `sm_121a` gate build is byte-unchanged). The
consumer `sm_12x` FP8 body (`ArchTag=Sm120`) and the DC1 sm100 NVFP4 leg are
untouched. **No B200/sm_100 board ran it** — a green compile + SASS is not
execution evidence, not runtime support, and not vLLM-competitive; the sm100
int8/blockwise C3x legs and the other datacenter fast paths (CUTLASS MoE Sm100,
MXFP4, CUTLASS MLA) remain scoped.

A separate **beyond-vLLM breadth lane** targets the older NVIDIA arches vLLM
DROPS but llama.cpp still runs (Pascal/Volta/Turing). Its first brick landed
2026-07-28: **Turing `sm_75` is BUILD-VERIFIED.** The bf16-WMMA prefill kernels
(bf16 tensor-core fragments are Ampere+ only) are now guarded
`#if __CUDA_ARCH__ >= 800`, so a single-arch `75` build compiles `-Werror` 0
warnings on nvcc 13.0 and `cuobjdump` shows real `sm_75` cubins — the TU falls
back to the existing scalar CUDA-core attention path (the llama.cpp fp16 fast
body is a later brick). On `sm_80`+ the guard is a no-op (preprocessor identity),
so the GB10 gate build is byte-identical. This is
**DERIVED+BUILD-VERIFIED (testing-welcome), NOT runtime** — no Turing board ran
it; there is no vLLM oracle on Turing (vLLM dropped it), so correctness is
referenced against llama.cpp-on-card plus a portable cross-check. Volta/Pascal
need a CUDA `<13` toolkit (nvcc 13 rejects `sm_70`/`sm_60`/`sm_61`). See
[.agents/backend-matrix.md](../.agents/backend-matrix.md) `BACKEND-CUDA-SM075`.

**Metal (Apple Silicon), indicative, not binding.** Two models run end to end
and pass correctness (OPT-125m, Qwen3-0.6B); 18 of 75 ops are native, the rest
fall back to CPU on unified memory. Kernel work including mma prefill attention
(4.3x), a vectorised decode V accumulation (+1.66%) and vectorised prefill
attention staging (+0.50%) puts warm b=1 throughput at about 95.9% of MLX-LM by default, and 97.6% with the optional MLX GEMM provider
shape-gated to prefill (`-DVLLM_CPP_MLX=ON`), which is a numerics-deviating
configuration. Both figures are against an interleaved 6-run MLX-LM baseline;
an earlier 99.1% claim used a 2-run baseline containing an outlier. With MLX
gated, prefill is 1.5% AHEAD of MLX-LM and the entire remaining deficit is
decode. The GEMV runs at 83% of memory peak; decode attention runs 2.6x slower
per byte than the GEMV because at b=1 it gets 8x fewer threads in flight, and
three separate attempts to exploit that (flash-decoding, GQA grouping, wider
threadgroups) have each measured worse. Bisecting the GEMM puts it at 97% of MLX's own
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
the identity softmax rescale measured no gain. Prefill's
small-kernel bucket is ~36 ms against MLX-LM's implied ~25, so improving our own
architecture there caps out near 97% rather than parity. A spike comparing MLX's
lazy-graph execution model against ours found our matmul chain already +1.5%
ahead of MLX's own ceiling, so adopting MLX's execution model would not help
decode; the residual is small-op fusion, which is ours to do. The V-accumulation win came from BISECTING the attention
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

**CUDA arch breadth beyond vLLM (Pascal / Volta / Turing).** These
are older NVIDIA arches vLLM DROPS but llama.cpp still supports, so this is a
"support more than vLLM" lane with llama.cpp as both the kernel source and the
competitor floor. Our portable attention uses bf16 tensor-core WMMA, which does
not exist before Ampere, so these arches need the bf16-WMMA path compile-guarded
(and, for a fast path, a new fp16/non-tensor-core kernel body ported 1:1 from
llama.cpp's `fattn-tile`/`fattn-vec`). The lane is derive-and-ship, and its first
brick has LANDED: **Turing (`sm_75`, e.g. the cloud-common T4) is BUILD-VERIFIED**
(2026-07-28, `CLAIM-CUDA-TURING-SM75`). The bf16-WMMA prefill kernels are now
guarded `#if __CUDA_ARCH__ >= 800`, so a single-arch `75` build compiles `-Werror`
0-warn on our current CUDA 13 toolkit and `cuobjdump` shows real `sm_75` cubins —
the TU falls back to the existing scalar CUDA-core attention path (the fp16 fast
body is a separate later brick). On `sm_80`+ the guard is a no-op (byte-identical
GB10 build). It ships labeled "derived, build-verified, not hardware-tested here,
community testing welcome" — no Turing board ran it; a green compile + SASS is not
execution evidence. **Volta (`sm_70`, V100) and Pascal (`sm_60`/`sm_61`, P100/P40)
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
- **SGLang-compatible knobs are now first-class, documented toggles** (2026-07-28,
  `CLAIM-SGLANG-ABI-DOCS`, reconciled to ABI v10). LPM scheduling and jump-forward
  are exposed on the C++ library API AND the C ABI (not env-var/internal-only),
  joining the already-exposed prefix-caching/RadixAttention and
  custom-logits-processor knobs. LPM is selected through the ABI v9 **string** field
  `vllm_model_params.scheduling_policy` (`"fcfs"`/`"priority"`/`"lpm"`); jump-forward
  through the new ABI **v10** tri-state field `vllm_model_params.enable_jump_forward`
  (`0`=default/`1`=on/`2`=off). The server has `--scheduling-policy lpm` and
  `--enable-jump-forward` / `--disable-jump-forward`. All default to today's
  behavior — an all-zero config is byte-identical to before ABI v10, and
  `vllm_abi_version()` returns 10. `VT_ENABLE_JUMP_FORWARD` is retained as an env
  override (wins when set). The user-facing enablement guide for all four behaviors
  is `docs/SGLANG-COMPAT.md` (engineering record: `.agents/specs/sglang-enablement.md`).
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
- **Engine configuration through the C ABI** reached parity with the bundled
  server's flags at **ABI v9** (2026-07-28). `vllm_model_params` now carries
  `max_num_batched_tokens` (chunked-prefill token budget), `scheduling_policy`
  (`fcfs`/`priority`/`lpm`), and `kv_transfer_config` (the external KV connector /
  LMCache JSON, connector name validated against `KVConnectorFactory` at load),
  and honours `tokenizer_config_path`, which had been declared since v1 and never
  read. Every addition is inert at its default, so a v8 caller that zero-fills the
  struct growth gets the byte-identical pre-v9 engine. Malformed
  `speculative_config` / `kv_transfer_config` documents report
  `VLLM_ERR_INVALID_ARGUMENT` rather than `VLLM_ERR_MODEL_LOAD` (the contract
  `vllm.h` has documented since v6). Driver: an embedder (the LocalAI vllm-cpp
  backend) could not expose LMCache or the prefill budget in a model config.
- **DFlash speculative decoding from a GGUF DRAFT WORKS end to end on the GB10
  release target** (`SPEC-DFLASH-GGUF`, `PARTIAL`,
  [spike](../.agents/specs/gguf-dflash-draft.md)). A DFlash draft packaged as a
  single `dflash`-arch GGUF now loads AND generates against a safetensors target
  (axis A): config from the `dflash.*` KVs, weights through a resolver that
  reuses the existing safetensors `LoadQwen3DFlash` body, a `.gguf` branch in the
  draft-path resolution, and the target-shared embedding and lm_head still taken
  from the target, as the GGUF contract intends.
  **Binding result:** on the Qwen3.6-27B NVFP4 safetensors target, the published
  `Q4_K_M` GGUF draft and the bf16 z-lab safetensors draft produce
  token-for-token identical greedy continuations with identical accepted and
  proposed draft counts (20/80 and 42/96 on two prompts, k=16, concurrency 1).
  The 4-bit draft costs nothing in acceptance on this target, which the spike had
  flagged as a real risk.
  Three conventions this had to undo, all invisible to shape and name checks:
  `dflash.target_layers` is stored `+1`-offset; the draft's RMSNorm weights are
  RAW (its converter class does not inherit the Qwen3Next `+1` shift, so unlike
  the trunk and the MTP head they must NOT be un-shifted, which points the
  opposite way); and `vocab_size`, which the GGUF contract deliberately omits,
  must be back-filled from the target or the draft's shared embedding table is
  empty and the first proposal throws. Only the last survived the load-level
  tests, and only generating found it.
  `PARTIAL`: axis B (GGUF target as well, needing the `SharedHeadSource` retype)
  is untouched, and no speed number is owed or measured yet.
- **MTP speculative decoding from a GGUF target WORKS, on CPU and on the GB10
  release target** (`SPEC-MTP-GGUF`, `DONE`,
  [spike](../.agents/specs/gguf-mtp-spec-decode.md)). The head loads from a
  head-carrying GGUF - `HfConfigFromGguf` republishes the depth it already read
  from `<arch>.nextn_predict_layers`, and `LoadQwen3_5MTPFromGguf` reads the
  `nextn` block with the trunk loader's own helpers, so it inherits the GGUF
  (w+1) norm storage, the quantization routing and the torch [N, K] shape order.
  A GGUF exported without the head is refused, naming that as the reason.
  **Both correctness gates pass: spec-ON output is token-identical to spec-OFF**
  on CPU against a real llama.cpp-converted Qwen3.5-2B, and on the GPU against
  the 35B A3B NVFP4 GGUF with 13 drafts proposed / 11 accepted (dgx GB10 under
  `flock`, sm_121a, 2/2 cases 10/10 assertions, 90.2 GiB peak RSS, 8m01s;
  re-confirmed 3/3 on the committed source). Acceptance parity is measured
  against the safetensors sibling of the same quantization run: 12 proposed / 11
  accepted there vs 13 / 11 here. Cross-format token agreement is NOT
  APPLICABLE - it needs an F16/F32 GGUF sibling and every head-carrying export
  on hand is quantized. Throughput is the one open item and is deliberately
  PENDING, see [docs/BENCHMARKS.md](BENCHMARKS.md). Closing commit `edf91449`.
  Two things had held this row open and both are now settled by measurement.
  The build was believed to be `CMAKE_CUDA_ARCHITECTURES=75` on an sm_121 GB10:
  that came from `CMakeCache.txt`, which records the CUDA compiler's own probe
  default and is shadowed by the normal variable the project sets, so it never
  described the compiled code. The real flags and every emitted cubin are
  `sm_121a`. And the GPU arm's 24 tokens differed from the CPU arm's: that is a
  **near-tie, not a defect**, and CPU-vs-GPU token equality was never the bar
  (spec-ON == spec-OFF within one device is). At the one position where the two
  arms fork they still share a bit-identical prefix, and each device's pick is
  the other's rank-2 candidate at a margin of 0.074 nats (GPU) and 0.065 nats
  (CPU) - roughly 7x inside the ratified 0.5-nat near-tie band, and smaller than
  the 0.057-0.082 nats by which the two devices disagree about those same
  tokens' logprobs. Rounding decides it; everything after is one coin flip
  cascading, which is why 24 tokens look unrelated for a 0.07-nat cause.
- **The 35B A3B NVFP4 SAFETENSORS target has EXACT logit ties, and its
  speculative token identity does not hold there** (open, surfaced by
  `SPEC-MTP-GGUF`'s gate, owned by `SPEC-MTP`). Pointed at the safetensors
  sibling of the same quantization run, the same concurrency-1 gate shows
  spec-ON inserting one token relative to spec-OFF, and two runs of spec-OFF
  itself disagreeing. Both divergences land exactly on positions where the top
  candidates carry BIT-IDENTICAL logprobs: that arm has three such exact ties in
  24 tokens, because keeping NVFP4 packed puts its logits on a 1/16 grid where
  distinct tokens collide. The GGUF arm, which expands to bf16, has ZERO exact
  ties over the same 24 tokens (minimum margin 0.048 nats) and reproduces its
  sequence every run - so the GGUF result stands and this is a quantized-GEMM
  determinism question, not a speculative-decoding logic one. Not root-caused
  beyond the tie measurement (one prompt, two runs).
- **Speculative decoding on CPU corrupted the target's own state, and is FIXED**
  (`CPU-SPEC-DIVERGENCE`). `qwen3_5.cpp:3616` sized the GDN state gather/scatter
  row by `(Kw-1)` while the speculative persistent row is `(Kw-1)+num_spec`, so
  the contiguous row helpers mis-strode the cache slot and every channel past the
  first, corrupting post-prefill recurrent state; speculation therefore changed
  the target's own greedy output even when no draft was accepted. Fixed with a
  stride-aware per-channel state copy used only when the two widths differ, so
  every non-speculative path stays byte-identical. CPU-only in effect - the
  fp16/bf16 cache arm routes through the `GdnStateGather`/`Scatter` ops, so CUDA
  was never exposed and no GPU result is affected or retracted. It survived
  because there was no CPU spec-decode gate anywhere in the tree; there is one
  now.
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
  correctness-complete on a single-sequence eager path and the top roadmap
  priority; the OpenAI-server wiring has advanced through its CPU bricks —
  brick 1 (`CLAIM-MM-SERVING-W1`) parses the OpenAI multimodal content-part array
  and routes the decoded media through the existing processors, and brick 2
  (`CLAIM-MM-SERVING-W2`) carries the result into the engine (additive
  `add_request(MultiModalInputs)`/`generate` overloads on both engines via
  `InputProcessor::process_inputs_mm`, chat-template placeholder-string helpers,
  and a default-inert serving_chat multimodal seam), but the closing end-to-end
  GPU gate (`MM-SERVE-E2E` — the model tokenizer/processor seam body plus the mm
  forward) is still pending, so it is not yet servable end-to-end. Image-to-text is strict
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

### Feature-gap map vs pinned vLLM 0.26 (2026-07-28)

A whole-surface sweep of pinned vLLM `555967922` (0.26.0.dev0) against our
matrices ranked what vllm.cpp is MISSING: 8 HIGH, ~19 MED, ~16 LOW gaps, each
grounded in vLLM `file:line`. The material HIGH-priority misses (common,
single-box, user-facing) are LoRA / multi-LoRA runtime, the
pooling/embedding/classify/rerank task class, AWQ + GPTQ native quantized
compute, the xgrammar structured-output backend, fp8 KV cache, and reasoning
parsers. Three medium gaps have no tracked row yet (generic draft-model + Medusa
speculative decode, the offline Batch API, and the plugin system). vLLM has
removed prompt adapters, so that is not a gap. The full prioritized list lives
in [`.agents/specs/vllm-feature-gap-analysis.md`](../.agents/specs/vllm-feature-gap-analysis.md);
this is analysis only, no capability state changed.

The CPU CTest suite is green (0 real regressions). A 2026-07-27 hygiene pass
triaged the previously-red tests as stale assertions or `-j` contention, not
regressions: the model-registry count assertion tracks the 24 registered
architectures, the GGUF loader rejection case uses a genuinely-unregistered arch,
the DFlash proposer test supplies the now-required `"model"` key, and the two
HTTP-server tests run under CTest `RUN_SERIAL` so a CPU-hog concurrent test can no
longer starve their accept thread.
