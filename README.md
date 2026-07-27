# vllm.cpp

**Brought to you by the [LocalAI](https://github.com/mudler/LocalAI) team**, the folks behind LocalAI, the open-source AI engine that runs any model (LLMs, vision, voice, image, video) on any hardware, no GPU required.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](LICENSE)
[![LocalAI](https://img.shields.io/badge/LocalAI-Run_Locally-orange)](https://github.com/mudler/LocalAI)

vllm.cpp is a from-scratch C++20 port of [vLLM](https://github.com/vllm-project/vllm) that runs large language models with no Python, PyTorch, or ggml at inference time. It mirrors vLLM's V1 / Model Runner V2 architecture one-to-one (scheduler, paged KV cache, sampler, engine step loop) on top of its own portable tensor runtime (`vt::`), and ships llama.cpp-style as a library, a stable C API, an example CLI, and an OpenAI-compatible server. It loads Hugging Face **safetensors** and **GGUF** checkpoints, and runs on CUDA, CPU, Metal, and Vulkan from one source tree.

> **Pre-release, under heavy development.** The engine is correctness-complete and speed-competitive for a specific set of models on one GPU (NVIDIA GB10 / DGX Spark, sm_121a), plus a CPU path that matches or beats llama.cpp on GGUF. Read the honest per-feature state in [Features](#features) and [Supported models](#supported-models) below, and the full evidence in [docs/BENCHMARKS.md](docs/BENCHMARKS.md). Capabilities are labelled: *correctness-complete* (token-exact vs the vLLM oracle), *speed-pending* (correct, throughput work in progress), *build-only* (compiles for a target with no runtime proof here), and *hardware-blocked* (cannot run on the hardware available).

The reference numbers below are measured against the pinned vLLM oracle on GB10, greedy, same workload, same tokens. Where a claim cannot be measured on the hardware here, it is stated as such rather than implied. **The parity pin advanced 2026-07-26 to vLLM 0.26.0.dev0 (`55596792`) + transformers 5.14.1** (from 0.25.0); correctness was re-validated bit-identical on the new oracle (zero golden drift; see [.agents/specs/pin-advance.md](.agents/specs/pin-advance.md)), so the token-exact claims hold against the new pin. Historical speed figures below citing "vLLM 0.25.0" are the last binding measurement against the prior oracle (our engine is unchanged by the advance); a re-benchmark against 0.26 is pending.

## Features

vllm.cpp implements an intentionally focused subset of vLLM, held to token-for-token correctness against the pinned oracle. One binding current-state line per capability follows; the detailed chronology and evidence live in [docs/BENCHMARKS.md](docs/BENCHMARKS.md) and the [`.agents/`](.agents/) record.

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
| CPU backend vs llama.cpp | At or ahead on every axis (GGUF) | Prefill 1.18x ahead, decode at parity, peak memory 1.01x, byte-identical greedy tokens |
| Paged KV cache + prefix caching | Supported | Block-paged full attention, hybrid full-attention + GDN state groups, automatic prefix caching (APC) on by default for dense models |
| KV offload to CPU / disk | Built, opt-in, off by default; the disk connector is engine-refused | CPU and disk tiers with identity-checked blocks, selected by `--kv-transfer-config` (or programmatically) over one abstract KVConnector ABI. Worker-side KV store/load is implemented for the LMCache connector only; the CPU/disk connector is scheduler-side only, so the engine now REFUSES it at construction (a loud error, not silently wrong output). Guide: [docs/KV-OFFLOAD.md](docs/KV-OFFLOAD.md) |
| LMCache client (`lm://` remote KV) | Built, opt-in, off by default; a working, verified external KV cache | Pure-C++ `lm://` client wired as an `LMCacheConnector`, no `lmcache` in-process; keys agree byte-for-byte with a real vLLM+LMCache peer, mismatched blocks refused. Proven in a real OPT-125m loop vs a live `lmcache.v1.server`: connector-ON tokens are BIT-IDENTICAL to the connector-OFF cold run (both after an in-process restart and from a cold second process). See docs/BENCHMARKS.md |
| Sampling | Supported | Greedy, temperature, top-k/p, min-p, presence/frequency/repetition penalties, seed, stop/stop_token_ids, min_tokens, logit_bias, allowed_token_ids, bad_words, in vLLM's exact order. Sample logprobs are emitted end-to-end for `/v1/completions` and `/v1/chat/completions` (`logprobs`/`top_logprobs`). Prompt logprobs and `echo` are not emitted yet |
| Structured output | Supported (subset), engine-enforced | JSON schema, JSON object, regex, choice, GBNF grammar. Constrained decoding runs in the production engine (native grammar backend, per-step logits bitmask) and is reachable from OpenAI `response_format` and the C ABI (ABI v2 `structured_*` fields) |
| Tool-call parsing | 36 parser families / 40 accepted names, streaming | Every vLLM tool parser at the pin except the three Rust/Harmony-backed ones: pure-text parsers ported 1:1, the six engine-backed families reimplemented from their wire formats, all held to the upstream test suites. Selection via `--tool-call-parser` (server), `tool_parser` (C ABI), or template auto-detection; native-syntax forced tool_choice where expressible. Tables: docs/BENCHMARKS.md |
| Reasoning parsing | 7 parsers, streaming | think_auto (auto-detect default: content unless markers appear), deepseek_r1, mistral ([THINK]), minimax_m2 (+append_think), step3, olmo3 - reasoning split engine-side BEFORE tool parsing, streamed as `reasoning` deltas in the chat chunks |
| OpenAI server | Supported (subset) | `/v1/completions`, `/v1/chat/completions`, streaming SSE, `/v1/models`, `/health`, `/version`, `/ping`, `/metrics` (Prometheus `vllm:*` names), `/tokenize`, `/detokenize`, `/server_info`, `/reset_prefix_cache` |
| Tokenizers | Supported | Byte-level BPE (Qwen/Llama-3/OPT/GPT-2/DeepSeek/OLMo-2) and SentencePiece BPE (Mistral/Gemma), plus GGUF vocab; added-token `lstrip`/`rstrip` whitespace semantics (e.g. Phi-4-mini's special tokens); byte-exact vs the vLLM oracle |
| Multimodal: image to text | Correctness-complete, speed-pending; not in the OpenAI server yet | Strict token-exact 32/32 vs vLLM 0.25.0 on Qwen3-VL-4B and Qwen3.6-27B (`Qwen3_5ForConditionalGeneration`); C++ image processor + vision tower + MRoPE/DeepStack backbone, single-sequence eager driver. See [Serving and API notes](#serving-and-api-notes) |
| Multimodal: video to text | Correctness-complete, speed-pending; not in the OpenAI server yet | End-to-end on Qwen3-VL-4B (near-tie-robust) and Qwen3.6-27B (strict 32/32); reuses the image tower and temporal MRoPE plus video preprocessing (frame sampling, temporal grid) |
| Multimodal: audio to text | Correctness-complete, speed-pending; not in the OpenAI server yet | End-to-end on Voxtral-Mini-3B (near-tie-robust, decoder proven token-exact 48/48); Whisper-class encoder tower (203/203 faithful) plus adapter into the landed Mistral decoder; first audio understanding in the tree |

Speculative decoding is available on the Qwen3.5/3.6 checkpoints via `--speculative-config`. **MTP (k=1)** is end-to-end token-exact vs vLLM on both gate models (the 27B GDN hybrid `Qwen3_5MTP` and the 35B MoE `Qwen3_5MoeMTP`): three-way identical at concurrency 1 (our spec-ON == our spec-OFF == vLLM `--speculative-config mtp` greedy) and faster than vLLM there, on par or above at higher concurrency (mixed-batch), with the draft head alive and acceptance matched to vLLM. **Block-diffusion DFlash** (the z-lab Qwen3.6-27B draft over the 27B NVFP4 target) is correctness-complete and at or above vLLM throughput on GB10 (final concurrency-1 A/B our-on 29.32 tok/s vs vLLM-on 29.24, non-overlapping bands, 1.003x); it is recorded DONE across the engine, model and kernel matrices, built D0 through D14 on the vLLM 0.26.0.dev0 stack (which resolves vllm#40898), and it remains gated behind a spike while its user-facing serving surface is finalized. **ngram** (method `ngram`, draft-FREE) proposes the next tokens by matching the sequence's own suffix n-gram, so it needs no draft model and works on any model; on the 27B it is token-exact vs vLLM's own `--speculative-config ngram` on repetitive workloads (5/5 prompts, every draft accepted). The correctness form and the full D0-D14 measured chronology live in [docs/BENCHMARKS.md](docs/BENCHMARKS.md), [docs/SPECULATIVE-DECODING.md](docs/SPECULATIVE-DECODING.md) and [.agents/specs/dflash-spec-decode.md](.agents/specs/dflash-spec-decode.md). Not yet supported: LoRA, multi-GPU, and the full tool-calling template surface; multimodal (image/video/audio) is correctness-complete but not yet wired into the OpenAI server (see below). See [Serving and API notes](#serving-and-api-notes).

## Supported models

Every model below passes a token-for-token correctness gate against the vLLM 0.25.0 oracle on GB10 (the gate-time pin; correctness was re-validated bit-identical on the current 0.26.0.dev0 pin, see the note above). Where vLLM's own greedy is deterministic the bar is strict token-exact; where vLLM is self-inconsistent at bf16 near-ties, the bar is a near-tie-robust check (our token within half a nat of vLLM's own teacher-forced argmax on our prefix). "Speed" is a separate bar (match or beat vLLM on every axis), tracked in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

| Architecture | Example checkpoints | Safetensors | GGUF | Correctness | Speed |
|---|---|:---:|:---:|---|---|
| Qwen3.5/3.6 hybrid (GDN + MoE) | Qwen3.6-35B-A3B, Qwen3.6-27B | yes | 35B only | Token-exact | 27B at/above vLLM; 35B prefill-pending |
| Qwen3 / Qwen2 dense | Qwen3-4B, Qwen3-0.6B, Qwen3-32B | yes | dense qwen35 | Token-exact (near-tie-robust) | Speed-pending |
| Qwen3-MoE | Qwen3-Coder-30B-A3B | yes | - | Token-exact (near-tie-robust) | Speed-pending |
| Llama-3.x dense | Llama-3.2-1B | yes | - | Token-exact (near-tie-robust) | Speed-pending |
| Mistral dense | Mistral-7B-v0.3 | yes | - | Token-exact | Speed-pending |
| OPT | OPT-125m | yes | - | Strict token-exact | Speed-pending |
| DeepSeek-V2 (MLA) | DeepSeek-V2-Lite | yes | - | Token-exact | Speed-pending |
| GLM-4 dense | GLM-4-9B-0414 | yes | - | Token-exact | Speed-pending |
| GLM-4.7-Flash (MLA MoE) | GLM-4.7-Flash | yes | - | Token-exact (near-tie-robust) | Speed-pending |
| Gemma-3 dense | google/gemma-3-1b-it | yes | - | Strict token-exact (48/48) | Speed-pending |
| Gemma-2 dense | gemma-2-2b-it | yes | - | Token-exact (near-tie-robust, 48/48) | Speed-pending |
| Gemma-1 dense | gemma-2b | yes | - | Strict token-exact (48/48) | Speed-pending |
| OLMo-2 dense | OLMo-2-0425-1B | yes | - | Token-exact (near-tie-robust) | Speed-pending |
| Granite-3 dense | granite-3.3-2b-instruct | yes | - | Token-exact (16/16) | Speed-pending |
| Phi-3 / Phi-4 dense | Phi-4-mini-instruct, phi-4 (14B) | yes | - | Token-exact (near-tie 16/16; 14B strict 14/16) | Speed-pending |
| Phi-1 / Phi-2 dense (`PhiForCausalLM`) | microsoft/phi-2 | yes | - | Token-exact (16/16; 9 strict + 7 near-tie) | Speed-pending |
| MiniCPM dense (`MiniCPMForCausalLM`) | openbmb/MiniCPM-2B-sft-bf16 | yes | - | Token-exact (16/16; 10 strict + 6 near-tie) | Speed-pending |
| MiniCPM3 (`MiniCPM3ForCausalLM`, MLA) | openbmb/MiniCPM3-4B | yes | - | Token-exact (16/16; 13 strict + 3 near-tie) | Speed-pending |
| OLMo-3 dense | OLMo-3-1025-7B | yes | - | Implemented, oracle-blocked | Speed-pending |
| Qwen3-VL (image + video) | Qwen3-VL-4B-Instruct | yes | - | Strict token-exact 32/32 (image); video near-tie-robust | Speed-pending |
| Qwen3.6-27B vision (image + video) | Qwen3.6-27B (`Qwen3_5ForConditionalGeneration`) | yes | - | Strict token-exact 32/32 (image + video) | Speed-pending |
| Voxtral (audio) | Voxtral-Mini-3B-2507 | yes | - | Near-tie-robust (decoder 48/48 exact) | Speed-pending |

Compressed-tensors NVFP4A16 (W4A16) dense weights also load and compute natively (RedHatAI/Qwen3-32B-NVFP4A16), correctness-complete and speed-pending.

The **Gemma family**: **Gemma 3** (`Gemma3ForCausalLM`, `google/gemma-3-1b-it`) is the **first landed member** (correctness-complete, speed-pending): STRICT token-exact 48/48 greedy vs the vLLM 0.25.0 oracle. It reuses infrastructure already in the tree (gemma-RMSNorm `(1+w)`, the GLM sandwich norms, the shared dense-attention path, sliding-window attention, tied embeddings) plus one genuinely-new compute kernel, GeGLU (`gelu_pytorch_tanh`), a bf16 embedding-scale multiply, dual per-layer RoPE theta, and query_pre_attn_scalar scaling. **Gemma 2** (`Gemma2ForCausalLM`, `gemma-2-2b-it`) and **Gemma 1** (`GemmaForCausalLM`, `gemma-2b`) have since landed too, both correctness-complete and speed-pending at 48/48 greedy vs the same oracle ([spike](.agents/specs/sweep-gemma.md)): Gemma 2 proves the attention + final logit soft-cap primitives, Gemma 1 the original two-fused-norms block. Only **Gemma 4**, the newest registered variant, remains unimplemented: it needs a large new primitive stack (per-layer embeddings, YOCO KV-sharing, a Gemma-4 MoE) and has only multimodal-wrapped checkpoints, so it is recorded as blocked rather than supported. Its multimodal path (image + video + **audio**, the only audio-capable model in the pin) has been assessed ([spec](.agents/specs/gemma4-multimodal.md)) and was **oracle-blocked at gate time**: the Gemma-4 vision and audio towers load through Transformers `AutoModel`, and the gate-time 0.25.0 oracle's transformers (5.13.1) had no `gemma4` module, so no gate was constructible then. The current 0.26.0.dev0 pin carries transformers 5.14.1, which ships `gemma4`, so Gemma-4 multimodal is now reachable on the pin (implementation pending). Audio, the genuinely-new modality, is staged first on the smallest oracle-runnable audio model (Whisper, then Voxtral-Mini-3B on the already-landed Mistral backbone). The **OLMo-2 family** (`Olmo2ForCausalLM` / `Olmo3ForCausalLM`) is the **first landed OLMo member** (correctness-complete, speed-pending): token-exact 16/16 greedy vs the vLLM 0.25.0 oracle on `allenai/OLMo-2-0425-1B` (STRICT 13/16 + near-tie-band 3/16, max gap 0.094 nats, 0 forward-divergent). It is the cleanest dense bring-up yet, needing no new compute kernel: its two distinctive traits both reuse existing infrastructure. The reordered post-norm placement (`norm_after`) is a subset of the GLM/Gemma sandwich norms (the same standalone output-norm plus a plain residual add, without the pre-norms), and its QK-norm is a full-width RMSNorm reusing the existing norm op at a new shape. Its GPT-NeoX ByteLevel tokenizer (which prepends no BOS) makes it a real tokenizer-inclusive gate. `Olmo3ForCausalLM` rides the same class (the 0.25.0 oracle constructs it); the Olmo-3 interleaved sliding-window path has since landed and runs, but is oracle-blocked for a gate (see the table above). Larger DeepSeek / GLM / MiniMax / Gemma-4 variants are recorded as **hardware-blocked** (they do not fit 119 GiB of unified memory on this box) or **spiked-only**, per the [model matrix](.agents/model-matrix.md). The frontier families Kimi / MiniMax / GLM-latest are scoped for mechanical porting in [a dedicated spike](.agents/specs/sweep-kimi-minimax-glm-latest.md): Kimi-Linear-48B is the one that fits GB10 (91.5 GiB) and is e2e-gateable, while MiniMax-M2 (214.3 GiB fp8), Kimi-K2, MiniMax-M3 and GLM-5 remain hardware- or dependency-blocked (honesty-pass only). The matrix opens with an architecture-support checklist (a per-architecture status roll-up covering every engaged model) that a CI checker keeps in lockstep with the detailed rows.

From a **next-tier batch of recent dense text families** ([spike](.agents/specs/sweep-recent-dense-batch.md)), the top three are now implemented (correctness-complete or gating): **Granite-3** (`GraniteForCausalLM`, `granite-3.3-2b-instruct`) is correctness-complete, token-exact 16/16 vs the vLLM 0.25.0 oracle (Llama plus four scalar multipliers, no new compute kernel); **Phi-3 / Phi-4** (`Phi3ForCausalLM`, `Phi-4-mini-instruct`) is implemented and runs but is **not** recorded as a clean pass: it scores 15/16, and although its LongRoPE cos/sin cache is bit-identical to the oracle's, two positions on one prompt fall outside the near-tie band as a cascade after an exact-tie divergence, so closing it needs a cascade-aware gate or a strict check on the larger phi-4 (14B); **OLMo-3** (`Olmo3ForCausalLM`, dual rope plus interleaved sliding window) is implemented and runs in our engine but has no SACRED gate because the gate-time vLLM 0.25.0 oracle cannot run the `OLMo-3-1025-7B` checkpoint (its transformers predates OLMo-3's per-layer-type rope schema). **Command-R / Cohere** (`CohereForCausalLM`, parallel-residual with a weight-only LayerNorm and a logit scale) is now **implemented** (a faithful zero-new-kernel port that compiles, links and self-registers) but has **no SACRED gate** yet: every real small `CohereForCausalLM` checkpoint is HF-gated with no token on the GPU box, the only ungated checkpoints are tiny-random (head_dim 8/2, outside the validated attention path), and the GPU box is disk-full; the oracle was run-verified at W0 (arch confirmed `CohereForCausalLM`, not `Cohere2ForCausalLM`). **Phi-1 / Phi-2** (`PhiForCausalLM`, `microsoft/phi-2`) is now **correctness-complete** (token-exact 16/16 vs the vLLM 0.25.0 oracle: 9 strict + 7 bf16 near-ties, max gap 0.25 nats, 0 forward-divergent), the OLDER Microsoft Phi architecture, DISTINCT from Phi-3/Phi-4: a GPT-J parallel-residual decoder (one nn.LayerNorm with bias feeds both attention and MLP), biased q/k/v/dense projections, partial NeoX RoPE, a non-gated NewGELU MLP and an untied biased lm_head, all reusing landed ops (its `gelu_new` maps to the landed `vt::GeluTanh`, so no new compute kernel), with an F16-checkpoint dtype-aware loader mirroring vLLM's f16->bf16 cast. **MiniCPM** (`MiniCPMForCausalLM`, `openbmb/MiniCPM-2B-sft-bf16`) is now **correctness-complete** (token-exact 16/16 vs the vLLM 0.25.0 oracle: 10 strict + 6 bf16 near-ties, max gap 0.0 nats, 0 forward-divergent): the landed Llama/Granite dense forward plus three scalars (a scale_emb embedding scale, a scale_depth/sqrt(layers) scaled residual add on each sublayer, and a dim_model_base logit scaling), tied embeddings, no new compute kernel. Its checkpoint ships only as a pickle `.bin` with no safetensors and no native transformers config, so the vehicle was prepared with a trusted torch `.bin`->safetensors conversion of the official weights (both the oracle and our engine read the identical bf16), and the gate feeds the oracle's exact prompt ids (its SentencePiece tokenizer is a follow-up). **MiniCPM3** (`MiniCPM3ForCausalLM`, `openbmb/MiniCPM3-4B`) is now **correctness-complete** (token-exact 16/16 vs the vLLM 0.25.0 oracle: 13 strict + 3 bf16 near-ties, max gap 0.0 nats, 0 forward-divergent): the first MLA-attention MiniCPM, and the model that closes the non-trivial part of this tier. It is the MiniCPM three-scalar skeleton with its attention swapped from GQA to DeepSeek-style MLA, reusing the landed DeepSeek-V2 MLA block (weight absorption, absorbed-MQA decode, materialized-MHA prefill) with only three faithful deltas: a neox-style rope flag threaded through a new default-false field on the shared MLA dims (so DeepSeek-V2 stays byte-identical, re-gated 8/8), a LongRoPE cos/sin cache instead of YaRN, and a reuse-only zero-pad of the FlashAttention-2 MLA prefill from head_dim 96 up to the already-compiled 128. No new compute kernel; its `.bin`-only weights were converted via trusted torch like MiniCPM. The remaining recent-dense families are the trivial tail only: **Yi** (`YiForCausalLM`, a Llama alias) and **InternLM3** (`InternLM3ForCausalLM`, InternLM2 plus a sliding window).

## Performance

Numbers are measured on NVIDIA GB10 (DGX Spark, sm_121a) against the vLLM 0.25.0 oracle (the gate-time pin, see the pin note above), greedy, closed loop, input 1024 tokens to output 128, three interleaved repetitions, ratios direction-normalized so 1.0 or higher passes. The full per-axis grids, memory tables, and exact reproduction recipes are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

**Qwen3.6-27B (NVFP4), total throughput vs vLLM 0.25.0:**

| Concurrency | Ours (tok/s) | vLLM (tok/s) | Ratio |
|---:|---:|---:|---:|
| 1 | 86.05 | 82.32 | 1.045x |
| 2 | 159.68 | 158.03 | 1.011x |
| 4 | 292.34 | 290.31 | 1.007x |
| 8 | 508.77 | 505.46 | 1.007x |
| 16 | 801.76 | 789.16 | 1.016x |
| 32 | 1095.01 | 1076.25 | 1.017x |

We beat vLLM on total throughput at every concurrency; effective parity is 115/124 per-axis metrics (two-grid totality), with the residuals being noise-band coin-flips or a favorable determinism tradeoff described in the benchmark record. Peak host memory also passes (24.88 GiB vs vLLM 28.18 GiB). The 35B decode path is at or beyond vLLM everywhere; its remaining gap is prefill time-to-first-token, tracked as active work.

**Local Qwen3.5-4B plain BF16 direct loader, speed-pending:** the measured
pre-transplant repair on an RTX 5070 Ti reached 5769.99 total tok/s versus vLLM
0.25.0 at 5849.80 tok/s (0.9864x) on the identical 128-request workload. Direct
loading reduced peak PSS from 8.59 to 2.41 GiB and stable PSS from 8.59 to 0.76
GiB. Mean TPOT/ITL was 43.72 ms versus vLLM's 38.55 ms. Profiling attributes
the residual to the discrete-CUDA sampled-token D2H path synchronizing the main
stream instead of retaining vLLM's event-overlapped device mapping. The repair
has since been transplanted onto current `main`; that development branch is
pending the same-series revalidation, so these numbers do not yet bind to it.
This local 4B diagnostic does not establish 27B/35B support. Exact evidence and
reproduction:
[Qwen3.5-4B main repair](docs/bench-evidence/qwen35-4b-main-repair-20260725.md).

**CPU vs llama.cpp (GGUF, same file, single binary):** prefill 223.8 tokens/s vs llama.cpp 177.3 (1.18x ahead), decode at parity (24.7 vs 25.4 tokens/s), peak memory 2.83 GiB vs 2.80 (1.01x), and the output tokens are byte-identical to llama.cpp's greedy decode.

There is no front-page race clip yet; when one is produced it will follow the LocalAI house style (side-by-side, identical output, honest measured ratios). Until then the numbers above and in [docs/BENCHMARKS.md](docs/BENCHMARKS.md) are the reference.

## Build

vllm.cpp uses CMake (>= 3.24) and a C++20 compiler (gcc 13/14 and clang are exercised; the tree builds -Werror-clean on gcc 14.2). The core has no ML dependencies; the OpenAI server uses a vendored header-only HTTP transport (cpp-httplib).

```sh
# CPU build (the correctness / CI reference). The server is ON by default.
cmake -S . -B build
cmake --build build -j
ctest --test-dir build
```

```sh
# NVIDIA GB10 build with the vendored fast GDN path.
# Triton-AOT cubins are vendored: Python/Triton is only needed to regenerate
# them (VLLM_CPP_TRITON_REGEN), never to build or run them.
cmake -S . -B build-cuda \
  -DVLLM_CPP_CUDA=ON \
  -DVLLM_CPP_TRITON=ON
cmake --build build-cuda -j
```

The example binaries land under `build/examples/`: `vllm-cli`, `server`, `vllm-bench`, and `tokenize`.

### CMake options

Read from [`CMakeLists.txt`](CMakeLists.txt). Defaults shown are the shipped defaults.

| Option | Default | Purpose |
|---|---|---|
| `VLLM_CPP_CUDA` | `AUTO` | Build the CUDA backend: `ON`, `OFF`, or `AUTO` (on when a CUDA toolchain is found) |
| `VLLM_CPP_CUDA_ARCHITECTURES` | `121a` | Target CUDA arch(s): `121a` (GB10), `120a`/`120a;121a` (consumer Blackwell), and cross-family portable-only build-supported targets `90a`, `80`/`86`/`87`/`89`, `100a`/`103a`, `110` (accelerated paths off, not runtime-proven here). The `a` suffix is required for the native fp4 MMA |
| `VLLM_CPP_METAL` | `AUTO` | Build the Metal backend: `ON`, `OFF`, or `AUTO` (on for an Apple host with an ObjC++ compiler) |
| `VLLM_CPP_VULKAN` | `AUTO` (= `OFF`) | Build the Vulkan backend. Opt-in with `-DVLLM_CPP_VULKAN=ON`; headers are vendored and SPIR-V is committed, so no graphics toolchain is needed |
| `VLLM_CPP_MLX` | `OFF` | Build the optional MLX GEMM provider for Metal (needs `-DMLX_ROOT=<mlx install>`) |
| `MLX_ROOT` | (empty) | Root of an MLX install (`include/` + `lib/`) for `VLLM_CPP_MLX` |
| `VLLM_CPP_SERVER` | `ON` | Build the OpenAI HTTP server (needs `third_party/httplib/httplib.h`; disables itself with a warning if absent) |
| `VLLM_CPP_TRITON` | `OFF` | Consume the vendored per-arch Triton-AOT GDN cubins (CUDA only; no Python needed) |
| `VLLM_CPP_TRITON_REGEN` | `OFF` | Maintainer knob: regenerate the AOT cubins with Python + Triton |
| `VLLM_CPP_CUTLASS_DIR` | `third_party/cutlass` | CUTLASS source root (>= 4.5.0) for the sm120a NVFP4 GEMM |
| `VLLM_CPP_CUTLASS_FETCH` | `OFF` | FetchContent CUTLASS 4.5.0 if not found locally |
| `VLLM_CPP_MARLIN` | `ON` | Build the vendored Marlin NVFP4 W4A16 MoE GEMM (sm_12xa) |
| `VLLM_CPP_BUILD_TESTS` | `ON` | Compile and register ctest targets |
| `VLLM_CPP_BUILD_EXAMPLES` | `ON` | Build the example CLI, server, and bench binaries |
| `VLLM_CPP_BENCH_PROFILE_CONTROL` | `OFF` | Trace-only profiler replay control (never for production timing builds) |

Only GB10 / sm_121a is a runtime-gated CUDA target today. Consumer Blackwell (`120a`) plus the cross-family portable-only targets (`90a` Hopper, `80`/`86`/`87`/`89` Ampere/Ada, `100a`/`103a` datacenter Blackwell, `110`) are build-supported (they compile and emit real machine code) but unproven here (no such board), and non-Apple / non-NVIDIA backends run a subset of operations. See [Acceleration](#acceleration) and the [backend matrix](.agents/backend-matrix.md).

## Running inference (CLI)

`vllm-cli` runs a one-shot completion through the C ABI. Source: [`examples/cli/main.cpp`](examples/cli/main.cpp).

```sh
build/examples/vllm-cli \
  --model /path/to/Qwen3.6-27B \
  --prompt "The capital of France is" \
  --max-tokens 64
```

| Flag | Default | Meaning |
|---|---|---|
| `--model <dir>` | (required) | Model directory (config.json + tokenizer.json + safetensors) |
| `--prompt "<text>"` | (required) | Prompt text |
| `--tokenizer-config <path>` | (none) | Override `tokenizer_config.json` |
| `--max-tokens N` | `16` | Max tokens to generate |
| `--temperature T` | `0.0` | Sampling temperature (`<= 0` means greedy) |
| `--top-p P` | `1.0` | Nucleus cutoff |
| `--top-k K` | `0` | Top-k (`0` means all) |
| `--seed S` | (unset) | RNG seed (enables seeded sampling) |
| `--stream` | off | Stream token deltas to stdout |
| `--speculative-config '<json>'` | (unset) | Speculative decoding, same JSON as vLLM's `--speculative-config` (e.g. `'{"method":"mtp","num_speculative_tokens":1}'`). Unset means no speculation. See [docs/SPECULATIVE-DECODING.md](docs/SPECULATIVE-DECODING.md) |
| `-h`, `--help` | | Print usage and exit |

A throughput/latency harness, `vllm-bench` ([`examples/bench/main.cpp`](examples/bench/main.cpp)), takes `--model`, `--dataset-path`, `--num-prompts`, `--input-len`, `--output-len`, `--concurrency`, `--max-num-batched-tokens`, and `--num-blocks`. A tokenizer smoke tool, `tokenize` ([`examples/tokenize/main.cpp`](examples/tokenize/main.cpp)), takes `<tokenizer.json | model.gguf> <corpus.txt>`.

## OpenAI-compatible server

`server` is a small HTTP server speaking the OpenAI API, so any OpenAI client works by pointing its `base_url` at it. Source: [`examples/server/main.cpp`](examples/server/main.cpp) and [`src/vllm/entrypoints/openai/`](src/vllm/entrypoints/openai/).

```sh
build/examples/server \
  --model /path/to/Qwen3.6-35B-A3B \
  --port 8000 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192
```

```sh
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen3.6-35B-A3B", "prompt": "The capital of France is", "max_tokens": 64}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
print(client.completions.create(model="Qwen3.6-35B-A3B",
                                prompt="The capital of France is",
                                max_tokens=64).choices[0].text)
```

Endpoints (registered in [`src/vllm/entrypoints/openai/api_server.cpp`](src/vllm/entrypoints/openai/api_server.cpp)):

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/completions` | Text completion (JSON or `text/event-stream`) |
| POST | `/v1/chat/completions` | Chat completion (JSON or streaming SSE) |
| GET | `/v1/models` | List the served model |
| GET | `/health` | Process liveness (200) |
| GET, POST | `/ping` | Liveness probe (200, mirrors `/health`) |
| GET | `/version` | Engine version |
| GET | `/metrics` | Prometheus metrics (`vllm:*` names, text format 0.0.4). Registered when a metrics logger is attached |
| POST | `/tokenize` | Tokenize a `prompt` to token ids (optional `token_strs`). Registered when a tokenizer is attached |
| POST | `/detokenize` | Detokenize token ids back to text |
| GET | `/server_info` | Server info (`vllm_config`, `vllm_env`, `system_env`) |
| POST | `/reset_prefix_cache` | Reset the prefix cache; returns `{"success": bool}` |

Server flags:

| Flag | Default | Meaning |
|---|---|---|
| `--model <dir>` | (required) | Model directory (safetensors or `.gguf`) |
| `--host H` | `0.0.0.0` | Bind host |
| `--port P` | `8000` | Bind port |
| `--served-model-name N` | model dir basename | Model id in `/v1/models` and responses |
| `--tokenizer-config F` | `<dir>/tokenizer_config.json` | Chat template / tokenizer config |
| `--block-size N` | `32` | KV block size |
| `--num-blocks N` | `256` | KV blocks |
| `--max-model-len N` | `0` (config default) | Max sequence length |
| `--max-num-seqs N` | `8` | Max concurrent sequences (also sizes the HTTP worker pool) |
| `--max-num-batched-tokens N` | `0` (per-arch default) | Per-step token budget |
| `--enable-prefix-caching` / `--no-enable-prefix-caching` | model default | Override automatic prefix caching |
| `--scheduling-policy fcfs\|priority` | `fcfs` | Scheduler policy |
| `--enable-force-include-usage` | off | Force the usage block in responses |
| `--tool-call-parser <name>` | `hermes` | Tool-call dialect (40 accepted names over 36 parser families). `auto` detects it from the chat template, `none` disables tool parsing. An unknown name aborts startup listing every registered name |
| `--reasoning-parser <name>` | `none` | Reasoning parser (`think_auto`, `deepseek_r1`, `mistral`, `minimax_m2`, `minimax_m2_append_think`, `step3`, `olmo3`). `auto` detects from the chat template, `none` disables. Unknown names abort startup |
| `--kv-transfer-config '<json>'` | (unset) | External KV connector, same JSON as vLLM's own flag. Unset means no connector. See [docs/KV-OFFLOAD.md](docs/KV-OFFLOAD.md) |
| `--speculative-config '<json>'` | (unset) | Speculative decoding, same JSON as vLLM's own flag, e.g. `'{"method":"mtp","num_speculative_tokens":1}'`. Unset means no speculation (byte-identical). Methods `mtp`, `dflash` (external draft `model`), and `ngram` (draft-free suffix-match). Malformed/unsupported aborts startup. Guide: [docs/SPECULATIVE-DECODING.md](docs/SPECULATIVE-DECODING.md) |
| `-h`, `--help` | | Print usage and exit |

For a production deployment, use [LocalAI](https://localai.io), which can embed engines like this behind a model gallery, multi-model serving, the full OpenAI API surface, auth, and metrics.

## Consuming it as a library (C API and C++)

Link `libvllm` (static or shared) and include [`include/vllm.h`](include/vllm.h). It exposes a flat, exception-free, llama.cpp-style C ABI (`VLLM_ABI_VERSION 6`, 19 exported symbols) suitable for `dlopen` / FFI / LocalAI integration.

```c
#include "vllm.h"

vllm_model_params mp = vllm_model_params_default();
mp.model_path = "/path/to/model";

vllm_engine *engine = NULL;
if (vllm_engine_load(&mp, &engine) != VLLM_OK) {
    fprintf(stderr, "%s\n", vllm_last_error());
    return 1;
}

vllm_sampling_params sp = vllm_sampling_params_default();
sp.max_tokens = 64;               // sp.temperature = 0.0 means greedy

vllm_completion out;
if (vllm_complete(engine, "The capital of France is", &sp, &out) == VLLM_OK) {
    printf("%s\n", out.text);
    vllm_completion_free(&out);
}
vllm_engine_free(engine);
```

The ABI covers lifecycle (`vllm_engine_load` / `vllm_engine_free`), blocking and streaming completion (`vllm_complete`, `vllm_complete_stream`), non-blocking concurrent requests (`vllm_request_submit` / `_cancel` / `_wait` / `_done` / `_error` / `_free`), memory helpers, and diagnostics (`vllm_last_error`, `vllm_version`, `vllm_abi_version`). Structured output (ABI v2): set at most one of `structured_json` (JSON-Schema string), `structured_regex`, `structured_choice`/`n_structured_choice`, `structured_grammar` (GBNF), or `structured_json_object` on `vllm_sampling_params` and generation is grammar-constrained per step on every completion entry point. Chat (ABI v3): `vllm_chat` / `vllm_chat_stream` take one OpenAI chat-completions request JSON (messages, tools, tool_choice, sampling) and run the SAME engine-side pipeline as the bundled server: the model's chat template (tokenizer_config.json, or the GGUF `tokenizer.chat_template` metadata) renders the prompt, tool_choice lowers to the structural-tag decode constraint (auto is LAZY: the engine decides when a tool call engages), and tool-call output is parsed engine-side into structured `tool_calls` deltas; the stream callback receives one `chat.completion.chunk` JSON per delta. Multi-turn tool conversations round-trip: assistant-history tool_calls and the tool reply's tool_call_id/name reach the template context. Tool-parser selection (ABI v4): `vllm_model_params.tool_parser` names the tool-call dialect explicitly, and when unset the engine AUTO-DETECTS it from the chat template via an ordered marker table (the llama.cpp `common/chat.cpp` detection idea); an unknown explicit name fails the first chat call with `VLLM_ERR_INVALID_ARGUMENT`. Reasoning selection (ABI v5): `vllm_model_params.reasoning_parser` picks the chain-of-thought splitter the same way (auto-detect: `[THINK]` selects mistral, `<think>` selects think_auto - markerless answers stay pure content, marked ones split R1-identically; nothing detected leaves reasoning parsing off; "none" force-disables). Speculative decoding (ABI v6): `vllm_model_params.speculative_config` takes the same JSON object vLLM's `--speculative-config` does (e.g. `'{"method":"mtp","num_speculative_tokens":1}'`); NULL or empty disables speculation and is byte-identical to the non-speculative engine, and a malformed or unsupported document fails `vllm_engine_load`. Chat templates render through the vendored google/minja engine (the same renderer llama.cpp ships), so full-surface Jinja templates (namespace, macros, filters - e.g. the complete Qwen3.5 template) render faithfully; template resolution still PROBES a render once, and a genuinely broken template degrades - with a stderr witness, never silently per request - to a Hermes-aware plain prompt that keeps structural-tag tool engagement working.

For C++ consumers, the higher-level surface lives under [`include/vllm/`](include/vllm/): `LoadedEngine::FromModelDir(...)` ([`entrypoints/model_loader.h`](include/vllm/entrypoints/model_loader.h)) hands back the synchronous `LLMEngine` ([`v1/engine/llm_engine.h`](include/vllm/v1/engine/llm_engine.h)) or the async `AsyncLLM` ([`v1/engine/async_llm.h`](include/vllm/v1/engine/async_llm.h)) the server itself uses, plus `SamplingParams` and `RequestOutput`. The underlying portable tensor runtime is `vt::` ([`include/vt/`](include/vt/): `tensor.h`, `dtype.h`, `ops.h`, `backend.h`, and friends), which carries no ggml or PyTorch dependency.

## Quantization

| Format | State |
|---|---|
| NVFP4 W4A4 / W4A16 | Both gate-model paths run on GB10, token-exact. FP4 tactics match vLLM; Marlin NVFP4 W4A16 grouped-MoE is the 35B expert path |
| compressed-tensors NVFP4A16 (W4A16), dense | Correctness-complete via the Marlin weight-only path (the same kernel vLLM forces for `use_a16` on sm_121); speed not yet measured |
| GGUF F32 / F16 / Q4_0 / Q8_0 / Q3_K / Q4_K / Q5_K / Q6_K | Supported. On CPU, the six block encodings compute directly on the compressed blocks (no BF16 expansion), byte-identical to the reference path (`VT_GGUF_KEEP_QUANT=0` disables it). GPU builds still expand GGUF weights |
| FP8 (W8A8) | The 35B ModelOpt static per-tensor projection slice is implemented; generic FP8 modes and FP8 KV remain open |
| MXFP4 / MXFP8 | Planned |

## Acceleration

| Backend | Hardware | State |
|---|---|---|
| CPU | x86-64 and arm64 | Correctness / CI reference; at or ahead of llama.cpp on every GGUF axis (prefill 1.18x, decode parity, memory parity), with an Arm i8mm quant-GEMM tier (runtime-detected via auxv on Linux, sysctl on Apple Silicon) |
| CUDA | GB10 / DGX Spark, sm_121a | Gate-model correctness passes; 27B at/above vLLM throughput, 35B prefill-pending. The only runtime-gated CUDA target |
| CUDA | Consumer Blackwell, sm_120a | Build-supported (compiles, emits real sm_120a code, all fast paths resolve) but not runtime-proven here (no such card) |
| CUDA | Hopper, sm_90a | Build-supported, portable-kernels-only (accelerated paths disabled, no Hopper kernel bodies); not runtime-proven here |
| CUDA | Ampere/Ada (sm_80/86/87/89), datacenter Blackwell (sm_100a/103a), sm_110 | Build-supported, portable-kernels-only (accelerated paths disabled, no fast-path bodies for these families); sm_80/sm_100a/sm_110 compiled `-Werror` clean on dgx as per-major representatives, siblings share the bodies; not runtime-proven here (no such boards). sm_70/sm_75 are not build-supported (no bf16 tensor cores) |
| Metal | Apple Silicon | Two models run end to end and pass correctness (OPT-125m, Qwen3-0.6B); 18 of 75 ops native, rest fall back to CPU on unified memory. Kernel work (batched command buffers, simdgroup GEMM, vectorised loads, coalesced attention) puts b=1 at 19.1 tok/s vs MLX-LM's 27.9. Every Metal token matches vLLM's own argmax (256/256). Indicative ([BENCHMARKS](docs/BENCHMARKS.md)) |
| Vulkan | Portable GPU | Skeleton: 8 ops plus the fusion catalogue run and cross-check against CPU and CUDA. No model runs yet; off unless `-DVLLM_CPP_VULKAN=ON` |
| Intel XPU | Intel GPUs | Spiked, hardware-blocked |
| ROCm / ANE | AMD GPUs / Apple Neural Engine | Post-parity roadmap |

## Serving and API notes

- **Automatic prefix caching (APC)** is implemented and on by default for dense models (hybrid / GDN and attention-free default off, mirroring vLLM). Hit-rate statistics are counted per vLLM's own counters. Block-hash extra keys (multimodal hash, LoRA adapter name, `cache_salt`) are now folded into the block hash exactly as vLLM does, so requests differing only in those never share a cache block (proven no-false-share). The `cache_salt` OpenAI request field and the `/metrics` endpoint are not surfaced yet.
- **Automatic prefix caching (APC)** is implemented and on by default for dense models (hybrid / GDN and attention-free default off, mirroring vLLM). Hit-rate statistics are counted per vLLM's own counters. Some block-hash extra keys (multimodal / LoRA / `cache_salt`) are stubbed. A Prometheus `/metrics` endpoint is available (opt-in): it exposes vLLM's `vllm:*` metric names, help text, types, buckets and `{model_name, engine}` label schema in text format 0.0.4, gated against vLLM's own scrape spec; wiring the live per-step engine values into it (and the config-gated metric families) is the remaining work.
- **KV persistence to disk / CPU offload** is built (CPU and disk tiers, identity-checked blocks, a size-budgeted disk tier) and wired opt-in into the scheduler through an abstract `KVConnector` ABI selected by a `KVTransferConfig` (a disk-offload connector and the LMCache `lm://` client, over the same seam), off by default. The LMCache `lm://` client interoperates with a real vLLM+LMCache peer (its cache keys are byte-for-byte identical to LMCache's own `ChunkedTokenDatabase`, chunk 256, vLLM's `sha256_cbor`, and a KV prefix a real peer wrote is found and loaded back byte-identically over the wire), and it is now proven END TO END inside a real generation loop: with the connector ON against a live `lmcache.v1.server`, an OPT-125m run loads a previously stored prefix's KV and shortcuts prefill, generating tokens BIT-IDENTICAL to the connector-OFF cold run (verified both after an in-process restart and from a genuinely cold second process). The worker-side KV store/load is implemented **for the LMCache connector only**: the CPU/disk `OffloadingConnector` implements the scheduler half alone (it shortcuts prefill for matched blocks, but nothing moves their bytes into the KV pages), so the engine now REFUSES to wire it at construction, on every device, naming the connector and the device, rather than serving output computed over KV that was never written. Both connectors are selectable from the server with `--kv-transfer-config '<json>'`, mirroring vLLM's own flag and JSON shape. The disk connector's worker half remains unimplemented and is the named remaining step. Usage guide: [docs/KV-OFFLOAD.md](docs/KV-OFFLOAD.md).
- **Tool calling** covers the dialect set in the Features table above, including the Qwen3-Coder XML parser (`qwen3_coder` / `qwen3_xml` / `mimo`). Selection over HTTP is `--tool-call-parser` and `--reasoning-parser`, mirroring vLLM's own flag names, alongside the C ABI's `vllm_model_params.tool_parser` / `.reasoning_parser`. Both flags keep the server's previous behaviour as their default (`hermes`, reasoning off), take `auto` to opt into the same chat-template detection the C ABI uses, take `none` to disable, and abort startup listing every registered name if given an unknown one.
- `/health` reports process liveness rather than a full engine-health probe.
- **Speculative decoding** ships user-facing via `--speculative-config` (OpenAI server, example CLI, and C ABI v6), unset by default and byte-identical to the non-speculative engine when unset. **MTP (k=1)** on the Qwen3.5/3.6 gate checkpoints (those shipping an `mtp.*` head) is token-exact to both our own spec-off output and vLLM's own MTP greedy at concurrency 1, and about 1.04x faster than vLLM's spec-on decode; the concurrent (multi-request) path shares a scheduler step with ordinary prefills over a bit-exact GDN split-merge. Honest caveat: above concurrency 1 the 27B greedy output is not bit-stable across batch shapes, so exact spec-on/spec-off token agreement is a concurrency-1 property. **Block-diffusion DFlash** (method `dflash`, the z-lab Qwen3.6-27B draft over the 27B NVFP4 target, wired through the same `--speculative-config` with an external draft `model` path) is correctness-complete (ratified near-tie) and at or above vLLM throughput at concurrency 1 (29.32 vs 29.24 tok/s, 1.003x), built D0 through D14 on the vLLM 0.26.0.dev0 stack (which resolves vllm#40898) and recorded DONE across the engine, model and kernel matrices. **ngram** (method `ngram`) is the draft-free suffix-match proposer, reusing the same verify/reject loop with no draft model; it is token-exact vs vLLM's own ngram on repetitive workloads (27B, 5/5 prompts, every draft accepted) and off by default (byte-identical). Measured A/Bs and the D0-D14 chronology: [docs/BENCHMARKS.md](docs/BENCHMARKS.md); usage guide: [docs/SPECULATIVE-DECODING.md](docs/SPECULATIVE-DECODING.md).
- **LoRA and multi-GPU** are not supported yet. **Multimodal** is correctness-complete on a single-sequence eager path but not yet wired into the OpenAI server, and it is the top roadmap priority. Image-to-text is strict token-exact 32/32 vs the vLLM 0.25.0 oracle on both Qwen3-VL-4B and our own gate model Qwen3.6-27B (`Qwen3_5ForConditionalGeneration`); video-to-text works end-to-end on both (Qwen3-VL-4B near-tie-robust, Qwen3.6-27B strict 32/32); and audio-to-text works end-to-end on Voxtral-Mini-3B (near-tie-robust, decoder proven token-exact 48/48), the first audio understanding in the tree. The pipeline is a pure C++ port end-to-end: the Qwen3-VL image/video processors and the Whisper-class audio processor are bit/byte-identical to the oracle, the Qwen3-VL vision tower and the Whisper encoder tower are proven faithful within the bf16 envelope, and the MRoPE/DeepStack text-backbone contracts are unit-exact. Concurrency-1 speed is measured against vLLM 0.25.0's graphed config: on Qwen3.6-27B image the decode is at parity and the vision tower, after a warp-scoped online-softmax attention kernel plus a resident-weight load, is 148 ms/image (0.59x of vLLM's eager encode, faster) with the strict image/video gates still 32/32; the our-side audio timing is not yet measured. The mm paths are single-sequence eager drivers, so batched/graphed mm serving and higher-concurrency throughput are the open work. Design and status: [.agents/specs/multimodal-track.md](.agents/specs/multimodal-track.md), [.agents/specs/audio-track.md](.agents/specs/audio-track.md), and [.agents/specs/multimodal-speed.md](.agents/specs/multimodal-speed.md).
- **Environment variables.** A handful of behavior-changing knobs (CPU thread count, prefix-cache hash seed, LMCache host/port, the default-on async and CUDA fast paths and their rollbacks, the portable-reference bisect switch, GGUF loading behavior, the Vulkan device selector) are documented in [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md). A CI check (`scripts/check-env-doc.py`) keeps that reference honest: any new production env var must be documented or explicitly classified kernel-internal.

## Verification and parity

Every model is gated token-for-token against the vLLM 0.25.0 oracle (the gate-time pin, re-validated bit-identical on the current 0.26.0.dev0 pin) on the same workload, and every change that could affect correctness or performance is compared apples-to-apples against vLLM with both numbers and the ratio recorded. Behavioral CPU tests run under CTest; CUDA correctness, sanitizer, trace, and performance evidence is recorded per feature rather than inferred from source. The protocol is in [`.agents/gates.md`](.agents/gates.md) and [`.agents/benchmark-protocol.md`](.agents/benchmark-protocol.md). A CI check (`scripts/check-fusion-consistency.py`) additionally keeps model forwards routing their fusable add+RMSNorm glue through the portable fusion catalog rather than hand-fusing it.

## Why vllm.cpp

vLLM is an excellent serving framework, but running it drags in a heavy Python / PyTorch / CUDA stack. vllm.cpp is a from-scratch C++20 port focused on inference:

- **No Python at inference.** A single `libvllm` behind a flat C ABI (`include/vllm.h`), easy to embed from C, C++, Go, or Rust, or to `dlopen` from LocalAI.
- **One source tree, many backends.** CUDA, CPU, Metal, and Vulkan from the same code, loading safetensors and GGUF.
- **Faithful to vLLM.** The V1 / Model Runner V2 architecture is mirrored one-to-one so upstream vLLM changes port mechanically, and correctness is held token-for-token against the pinned oracle.
- **Honest numbers.** Every capability is labelled correctness-complete, speed-pending, build-only, or hardware-blocked, and the evidence is in the record.

## Project record

The canonical project record lives under [`.agents/`](.agents/), indexed by [AGENTS.md](AGENTS.md). This README is a human-readable current-state snapshot, not a chronological log: detailed status and evidence live in [`.agents/state.md`](.agents/state.md), the [parity ledger](.agents/parity-ledger.md), the [area matrices](.agents/model-matrix.md), and [docs/BENCHMARKS.md](docs/BENCHMARKS.md). The portfolio-completion plan (what is done, reachable, or hardware/external blocked) is tracked in [`.agents/specs/roadmap-v1-completion.md`](.agents/specs/roadmap-v1-completion.md).

## Citation

If you use vllm.cpp, please cite this repository and the upstream vLLM project:

```bibtex
@software{vllm_cpp,
  title  = {vllm.cpp: a C++ inference engine porting vLLM},
  author = {Di Giacinto, Ettore},
  url    = {https://github.com/mudler/vllm.cpp},
  year   = {2026}
}
```

vLLM is by the vLLM project ([vllm-project/vllm](https://github.com/vllm-project/vllm)). Model weights are governed by their own licenses, so check each model card.

## Author

Ettore Di Giacinto ([@mudler](https://github.com/mudler)).

## License

vllm.cpp is released under the [Apache License, Version 2.0](LICENSE). See [NOTICE](NOTICE) for third-party attributions. The model weights keep their own licenses.

---

Built by the [LocalAI](https://github.com/mudler/LocalAI) team. If you want to run LLMs (and vision, voice, image, and video models) locally on any hardware with an OpenAI-compatible API, [give LocalAI a star](https://github.com/mudler/LocalAI).
