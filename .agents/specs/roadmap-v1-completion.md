# roadmap_v1 completion punch-list — the plan of record for finishing the portfolio

*(records-only planning spec, 2026-07-27. Base `origin/main` `76d2b8ea`. READ-ONLY
analysis; no `src/`/`tests/` touched, SACRED byte-identical. This spec classifies
EVERY `ROAD-V1-*` portfolio row against the true matrix/ledger/state state, names
the HW/external-blocked bounding set, and gives the ranked reachable execution
order. It is a plan, not a support claim: nothing here advances any row's
lifecycle state.)*

## Method + hardware envelope

Every classification below is grounded in the six area matrices
([engine](../engine-matrix.md), [model](../model-matrix.md),
[kernel](../kernel-matrix.md), [backend](../backend-matrix.md),
[quantization](../quantization-matrix.md), [feature](../feature-matrix.md)),
the [roadmap portfolio table](../roadmap_v1.md), and the append-only
[state](../state.md)/[ledger](../parity-ledger.md). "DONE" requires merged code +
a passing gate; where a row ships a capability but the every-axis **speed** gate
or a **breadth** tail is open, the row is REACHABLE-INCOMPLETE, not DONE.

**Reachable HW** = one GB10/DGX-Spark GPU (sm_121a, 119 GiB unified; the vLLM
0.26.0.dev0 oracle box) + a CPU dev box + an Apple M4 Mac mini (Metal/MLX, 16 GiB,
small models only). **NOT available** (⇒ HW-BLOCKED, honesty-pass/build-only
ceiling): any AMD/ROCm, Intel/XPU, discrete-Vulkan, ANE, or multi-GPU board, and
any model that does not fit the 119 GiB unified pool.

## 1. Per-row classification (18 portfolio rows)

Legend: **DONE** (merged+gated) / **RI** (reachable-incomplete on GB10/CPU/M4) /
**HW** (hardware-blocked) / **EXT** (external-blocked: pin/dep/HF-token). Rows that
carry more than one class list the dominant one first.

| Row | Class | Delivered + gated (anchor) | Remaining to close the row |
|---|---|---|---|
| `ROAD-V1-A` perf/SGLang floor | **RI** | 27B **effective parity-or-better ratified** (two-grid totality 115/124: 110 pass-in-both + 5 coin-flip; residuals are the net-positive determinism tradeoff). `ENG-ASYNC-SCHED` DONE. | 35B every-axis closure (first binding 70/124, c4–c32 already win, c1/c2 residual — `BACKEND-GATE-CUDA-VLLM` PARTIAL); then the SGLang floor arms (`BACKEND-GATE-CUDA-SGLANG` BLOCKED on `SERVE-ASYNC-LLM` prod-ON; `-PREFIX` READY). |
| `ROAD-V1-MM` multimodal | **RI** (+EXT sub) | **Image + video STRICT token-exact 32/32** on Qwen3.6-27B and Qwen3-VL-4B; **audio e2e** on Voxtral-Mini-3B (near-tie-robust; decoder 48/48). Correctness is the user's #1 priority and it LANDED. | Every-axis **SPEED** gate (batched/graphed mm serving c2+, audio our-side timing) on all mm rows — none is DONE (`MODEL-MM-*` all PARTIAL/ACTIVE, speed-pending). Gemma-4 mm/audio = EXT (below). Qwen3.6-35B mm needs a vision-inclusive checkpoint download + M2/M3 tower attach. |
| `ROAD-V1-C1` extensibility | **DONE** (cornerstone) | Drop-in kernel ABI W0, Platform seam, model self-registration, and the **portable op-fusion framework ORDER-1 milestone** (W0–W4 merged+gated, `KERNEL-FUSION-FRAMEWORK`); consistency-audit CI check landed. `BACKEND-ABI-VT`/`BACKEND-CUDA-ARCH-ADDITIVITY` seams gated on sm_121a. | Row stays SPIKE-open only for **non-blocking** tail: Tier-1 fusion perf interpreter (composite-only → single-launch), `FUSION-DENSE-MIGRATE` (route 5 drift models), a real Metal/Vulkan catalog realization (M4-reachable / HW-blocked), and migrating a production kernel family onto the common adapter. Correctness cornerstone is closed. |
| `ROAD-V1-C2` model families | **RI** (+HW/EXT sub) | First additive model (Qwen3 dense) + a broad **text sweep correctness-complete + SACRED-gated**: Qwen3/Qwen3Moe/Coder, Llama/Yi/InternLM3, Mistral, GLM-4-9B/GLM-4.7-Flash, Gemma-1/2/3, OPT, DeepSeek-V2-Lite (MLA), OLMo-2, Phi-3/4, Phi-1/2, Granite-3, StableLM, InternLM2, MiniCPM, MiniCPM3 (MLA). 20 ACTIVE model rows. | **SPEED close** on every one (all 20 are "correctness-complete, speed pending"). MoE/SSM breadth (Qwen3-Next, Falcon, Falcon-H1, GraniteMoe*, Cohere2Moe, PhiMoE, Mamba/Jamba/Zamba2/NemotronH) = RI (INVENTORIED/SPIKE). Frontier: Kimi-Linear-48B fits (RI, +KDA kernel); DeepSeek-V3/GLM-5/MiniMax-M2/M3/Kimi-K2 = HW (>119 GiB); Command-R = EXT (HF token). |
| `ROAD-V1-C2-LOCAL-BF16` | **RI** (S) | Local Qwen3.5-4B plain-BF16 diagnostic rebased onto current additive seams; CPU/CUDA + direct ON/OFF token-equivalence green; H32 AOT / plain-BF16 graphs / ratio-4 FA2 landed + trace-proven. | Port device-resident sampled-token mapping to discrete CUDA (remove the measured main-stream wait) and rerun the exact 4B series. Small. |
| `ROAD-V1-C3` spec-decode | **DONE** (core) | **MTP k=1 DONE + gated on BOTH gate models** (`SPEC-MTP`, c1 token-exact + above vLLM, c2–c8 on-par-or-above); **DFlash DONE + speed gate MET** (`SPEC-DFLASH` D14, our-ON ≥ vLLM-ON). | Named tail only: DSpark (`SPEC-DSPARK`) + heterogeneous-vocabulary TLI (`SPEC-TLI`) unspiked — overlaps `ROAD-V1-D3`. Core spec-decode is gate-closed. |
| `ROAD-V1-C4` quantization | **RI** | **3 schemes DONE**: NVFP4-MO-W4A16, NVFP4-CT-W4A4, FP8-MO-STATIC (all R/M/C/E/P). GGUF compute-in-quant live + default-ON; CPU prefill 1.26× ahead of llama.cpp + RSS 1.01×. | GGUF **decode 10× bar** (`QUANT-GGUF-CPU-THREADPOOL` W4: 8.05× < 10× at M=1) — the single GGUF speed blocker. NVFP4-CT-W4A16 perf gate. FP8-generic dispatch (static/dyn × tensor/channel/token/block). Breadth: AWQ/GPTQ/Marlin-wiring, i-quants, MXFP4/MX, bitsandbytes, KV-quant — all INVENTORIED. |
| `ROAD-V1-C5` sliding/YaRN | **RI** | Joint spike accepted; all W1–W8 leaves implemented and CPU/oracle/sanitizer green. **CUDA GPU CLOSURE 2026-07-27 (`CLAIM-ROADMAP-C5`, dgx GB10 sm_121a, clean build of `489f7771`, oracle vLLM 0.26.0.dev0):** shared scaled-RoPE + local-mask CUDA path compiles `-Werror`-clean + RUNS on GB10; feature-positive correctness gates PASS — SWA Gemma-2/Gemma-3 48/48, LongRoPE Phi-4-mini 16/16 (RED-first), llama3 Llama-3.2-1B 16/16, dynamic-NTK InternLM2 16/16; both RoPE 0.26-oracle recaptures BIT-IDENTICAL to goldens. `ATTN-SLIDING-WINDOW`/`ATTN-ROPE-{LLAMA3,LONGROPE,DYNAMIC-NTK}`/`ATTN-YARN` → `ACTIVE`; `ATTN-CHUNKED-LOCAL` + `KV-*-SPEC` honest. | **Honest residual (vehicle-blocked, not skipped):** YaRN model e2e (no cached Nomic/gpt-oss consumer) + chunked-local model e2e (no Llama4 row) REACHABLE-BLOCKED; long-context positive-mask (prompt > W) SWA e2e + KV-memory G8; every-axis **SPEED** tail (all leaves correctness-complete, speed-pending). |
| `ROAD-V1-C6` async/priority serving | **RI** | `ENG-ASYNC-SCHED` **DONE** (`6ea7856`, default-ON, DGX token-neutral). W1/W2/W4 landed. | `SERVE-ASYNC-LLM` (GATING → prod-ON, blocks the SGLang floor + `ROAD-V1-A`), `ENG-PRIORITY-SCHED` + `ENG-CORE-BUSY-LOOP` GPU gates (GATING, held behind SERVE-GATE-ONLINE). |
| `ROAD-V1-C7` sampling/logprobs | **RI** | **W1-W4 LANDED + CPU-GATED 2026-07-27 (`CLAIM-ROADMAP-C7`, NOT pushed):** `SAMPLE-CORE` + `SAMPLE-LOGIT-FILTERS` -> `ACTIVE` — the full sampling-control surface (temperature/top_p/top_k/min_p/penalties/seed/stop/min_tokens/logit_bias/allowed_token_ids/bad_words/logprobs-count) WIRED end-to-end params->protocol->InputProcessor->InputBatch->SamplingMetadata->Sampler + gated exactly on the CPU reference backend (RED-first; default/greedy byte-identical). | **RESIDUAL:** W5 `SAMPLE-LOGPROBS` payload end-to-end (LogprobsProcessor + OpenAI `CompletionLogProbs` serialization, PARTIAL — count wired, sampler produces tensors) + `SAMPLE-PROMPT-LOGPROBS` need the engine-output plumbing + a running-engine gate; then `n>1`, philox-RNG, logprobs_mode, beam-search, custom logits processors (INVENTORIED). |
| `ROAD-V1-C8` tokenize/parse/metrics | **RI** | Tokenizer engines have code (`LOAD-HF-BPE` ANCHOR-BACKFILL, `LOAD-SENTENCEPIECE` ACTIVE 6/6). | `SERVE-METRICS` (`/metrics` Prometheus — the **oldest open T0 debt**), `TOOLS-STREAMING-PARSER`, `SERVE-RESPONSE-METRICS`, `SERVE-UTILITY-ENDPOINTS` (`/tokenize`,`/detokenize`,reset) — all INVENTORIED; tool-parser breadth (§7). |
| `ROAD-V1-C9` recurring sync | **RI** (recurring) | Pin **advanced to `555967922` / vLLM 0.26.0.dev0 + transformers 5.14.1** (`CLAIM-PIN-ADVANCE-W5`); re-gate 296/299 GREEN, zero golden drift. This unblocked OLMo-3 W5, DFlash, Gemma-4 module. | Refresh exact performance denominators + target goldens/tests on 0.26; then the ongoing mechanical cycle. Never terminally "done" (recurring). |
| `ROAD-V1-D1` backend fan-out | **RI + HW** | CUDA-arch **seams landed + gated** on sm_121a (`BACKEND-CUDA-ARCH-ADDITIVITY` ACTIVE: feature table, capability probe, runtime SM-dispatch registry). **CPU REACHABLE** (macOS M4 `-Werror`-clean, 108k assertions). **Metal/MLX REACHABLE + running** (`BACKEND-METAL-MLX` ACTIVE: 18 MSL kernels, 2 models token-exact/near-tie on Apple GPU, M3c-1/M3d landed). | **REACHABLE:** Metal/MLX perf close (still ~2.96× behind MLX-LM b=1; batched-decode tile kernel; binding needs M4 quiet/sudo), CPU B4 speed/RSS gates. **HW-BLOCKED (build-only ceiling):** ROCm (INVENTORIED, no AMD board), Intel XPU (SPIKE, no board + upstream loyalty target incomplete), discrete-Vulkan (ACTIVE skeleton + hermetic SPIR-V, but only llvmpipe/GB10-unified here), ANE (encoder/pooling niche), and every cross-family CUDA kernel body / gencode beyond sm_120/121. |
| `ROAD-V1-D2` multi-GPU/TP | **HW** | `PAR-TP` READY (spike accepted). | Runtime is **HW-BLOCKED: needs a 2-GPU box** (GB10 is single-GPU). Reachable build-only: TP mock/ABI Phase-0. PP/EP/DP/sequence-MoE/multinode all INVENTORIED and also multi-GPU-gated. |
| `ROAD-V1-D3` spec-decode breadth | **DONE** (2026-07-27) | Reused the landed spec machinery (frozen ABI + rejection sampler + runner loop from MTP/DFlash). | `SPEC-NGRAM` **DONE/ACTIVE** — draft-FREE n-gram proposer (1:1 port of `ngram_proposer.py`), 27B gate 5/5 STRICT our-ngram-ON == vLLM-ngram-ON + 180/180 accepted, spec-OFF byte-identical (SACRED 235 + MTP 9 + DFlash 27), no new kernel. `SPEC-EAGLE3` **BLOCKED (honest)** — no ungated oracle-runnable EAGLE3 draft arch/checkpoint for a Qwen3.6 gate model at pin `555967922` (registry has no `Eagle3Qwen3_5*`; z-lab published DFlash not EAGLE3); port scoped. See [spec-decode-breadth-d3.md](spec-decode-breadth-d3.md). |
| `ROAD-V1-D4` KV-disk/LMCache | **RI** (+HW sub) | `KV-OFFLOAD` ACTIVE (CPU+disk tiers W1–W5, 32/48 restart-prefill saved); `KV-EXTERNAL-CACHE` ACTIVE (LMCache `lm://` client W1–W5, bit-identical vs cold prefill on OPT-125m); `KV-CONNECTORS` ACTIVE (ABI + `--kv-transfer-config` CLI). | `KV-EVENTS` (SPIKE), W6 LMCache go/no-go study, W7 named per-sequence save/restore (the beyond-parity item), and a **binding every-axis LMCache grid on a LARGER model** (125M is noise-dominated). **HW-BLOCKED sub:** NIXL/Mooncake/HF3FS/MoRI-IO + P/D disaggregation (need RDMA/multi-node). |
| `ROAD-V1-D4-APC` prefix caching | **RI** (headline) | Deep port already default-ON for dense: `KV-PREFIX-CACHE` PARTIAL (chain hashing, block pool, all coordinators, hybrid intersection, stats, hit-rate 0.75); `KV-BLOCK-POOL`/`KV-HYBRID-COORD` PARTIAL; `ENG-CASCADE-ATTN` verified-not-owed. **W2 `extra_keys` DONE 2026-07-27** (mm/LoRA/`cache_salt` block-hash, 1:1 port; RED-first no-false-share proven `n1 48->0`; CPU-gated on dgx) — **unblocks MM + LoRA cache consumers.** | **W3: the first-ever cache-ON model gate** (token-exact ON/OFF/oracle + every-axis grid). **RI residual:** the two synthetic CPU harnesses are Qwen3_5-hybrid (APC inert); only OPT-125m dense on dgx (noise-dominated). Needs a real dense checkpoint on GPU or a pure-full-attention synthetic harness for full-engine token-identity + TTFT + oracle. Then W4 events, W5 partial-block, W6 Mamba `align` (`KV-MAMBA-ALIGN` SPIKE), W7 `reset_prefix_cache`. |
| `ROAD-V1-D5` LoRA/offload/experts | **RI** | `ENG-EXPERT-STREAM` READY (surpass-track spike accepted; absent in pinned vLLM). | `ENG-EXPERT-STREAM` W0–W2 (bank/reader/cache-policy), `ENG-WEIGHT-OFFLOAD` (INVENTORIED, UVA `cpu_offload_gb`), `LORA-RUNTIME` + `LORA-ENDPOINTS` (INVENTORIED, Punica-style batched apply + dynamic load/unload). LoRA is the user headline; large. |

**Counts (18 portfolio rows):** **DONE = 3** (`C1` cornerstone, `C3` core spec-decode, `D3` spec-decode breadth) · **REACHABLE-INCOMPLETE = 14** (`A, MM, C2, C2a, C4, C5, C6, C7, C8, C9, D1, D4, D4-APC, D5` — note `MM/C2/D1/D4` also carry HW/EXT sub-items) · **HW-BLOCKED (primary) = 1** (`D2`). No row is *purely* external-blocked; EXT applies to sub-items (Gemma-4, Command-R, DeepSeek-V3.2/GLM-5 DSA) inside `MM`/`C2`.

## 2. The blocked set (what bounds "complete roadmap_v1")

These cannot be gated on the current HW; the additive/build-only portion is
landable (honesty-pass ceiling) but the runtime/perf gate is not.

**HW-BLOCKED (need a board we don't have):**
- **Multi-GPU / TP-PP-EP-DP-seqMoE runtime** (`ROAD-V1-D2`, `PAR-*`) — needs ≥2 GPUs.
- **ROCm** (AMD), **Intel XPU** (Intel GPU), **discrete Vulkan** (only llvmpipe/GB10-unified here), **ANE** (encoder/pooling niche) — `ROAD-V1-D1` backend runtimes.
- **CUDA arch fan-out beyond sm_120/121** — only GB10 sm_121 is testable; sm_80/90/100 kernel bodies + gencode are build-only, and every cross-family kernel body under `kernel-matrix` is HW-gated.
- **RDMA/multi-node KV connectors** — NIXL, Mooncake, HF3FS, MoRI-IO, and prefill/decode disaggregation (`ROAD-V1-D4` sub).
- **Frontier models over the 119 GiB pool** — DeepSeek-V3/V3.2 (~642 GiB), GLM-5 (~1404 GiB), MiniMax-M2 (~214 GiB), MiniMax-M3 (~795 GiB), Kimi-K2 (~958 GiB). (Kimi-Linear-48B at 91.5 GiB and GLM-4.5-Air-FP8 at 104.8 GiB DO fit → reachable.)

**EXTERNAL-BLOCKED (need a pin/dep advance or an HF token):**
- **Gemma-4 mm + audio** — the transformers-module block is now cleared by the 0.26 pin, but public checkpoints (≥12B incl. the encoder-free unified 12B) are **HF-gated (no dgx token)**, and the USM-Conformer audio tower is still owed. Reopens with a token.
- **Command-R / Cohere** — implemented (zero-new-kernel) but every real checkpoint is HF-gated (no dgx token) + dgx disk-full; only tiny-random vehicles exist → no SACRED gate.
- **DeepSeek-V3.2 / GLM-5 DSA** — DEP-blocked on the DSA sparse-attention indexer dependency (on top of the memory HW block).

**Now-UNBLOCKED by the 0.26 pin advance (moved OUT of blocked → reachable):**
OLMo-3 W5 sliding-window SACRED gate, and DFlash (already DONE) — both were
oracle-blocked on transformers < the pin.

## 3. Ranked REACHABLE punch-list (execution order)

Ranked by (value × unblocks-others ÷ size). Each row: remaining W-plan skeleton →
gate → size (S/M/L) → vehicle model. `[H]` = user-directed headline.

1. **`ROAD-V1-MM` speed close** `[H]` (top user priority; correctness already landed).
   W: batched/graphed mm serving path (c2+), device-resident mm-token routing, audio
   our-side timing lever. **Gate:** every-axis (tput/TTFT/TPOT/mem) ≥ vLLM at the mm
   operating point, correctness held (image/video 32/32, audio near-tie). **Size M.**
   **Vehicle:** Qwen3.6-27B image+video, Voxtral-Mini-3B audio.
2. **`ROAD-V1-D4-APC` W3 cache-ON gate + W2 extra_keys** `[H]` ("same featureset and
   better"; W2 unblocks MM + LoRA cache consumers). **W2 DONE 2026-07-27**
   (`generate_block_hash_extra_keys` mm/LoRA/`cache_salt` 1:1; RED-first no-false-share
   proven; CPU-gated). W3 cache-ON dense gate is **RI**: synthetic CPU harnesses are
   hybrid (APC inert), only OPT-125m dense on dgx (noise-dominated) — needs a real
   dense checkpoint on GPU / a pure-full-attention synthetic harness + vLLM-APC-ON
   oracle. `skip_reading_prefix_cache` still MISSING. **Gate:** token-exact ON/OFF/oracle
   + every-axis cache-on grid vs vLLM. **Size S→M.** **Vehicle:** a dense model on DGX.
3. **`ROAD-V1-C5` sliding/YaRN GPU closure** (CPU already green; unblocks long-context
   + sliding-window model families). W: compile/run the shared scaled-RoPE + local-mask
   CUDA path; pass feature-positive model/oracle/trace gates. **Gate:** model e2e +
   every-axis on a scaled-RoPE consumer + both Qwen3.6 regressions. **Size M.**
   **Vehicle:** a YaRN/LongRoPE model + Gemma-2 sliding-window.
4. **`ROAD-V1-C7` sampling/logprobs API wiring** `[H]` (unblocks C8 + API parity).
   W: finish `SAMPLE-LOGPROBS`/`SAMPLE-LOGIT-FILTERS` payloads; add `prompt_logprobs`,
   logprobs_mode. **Gate:** end-to-end parity vs vLLM payloads on the server. **Size M.**
   **Vehicle:** any gated dense model via the OpenAI server.
5. **`ROAD-V1-A` 35B closure + SGLang floor** `[H]` (27B already ratified). W: close the
   35B c1/c2 residual (merged-projection fp8 glue + aux-stream slices); flip
   `SERVE-ASYNC-LLM` prod-ON to unblock the SGLang arm. **Gate:** 35B every-axis ≥ vLLM;
   SGLang preflight equivalence then the binding SGLang + prefix arms. **Size M→L.**
   **Vehicle:** Qwen3.6-35B-A3B-NVFP4.
6. **`ROAD-V1-D3` ngram + EAGLE3 — DONE 2026-07-27.** `SPEC-NGRAM` (draft-free
   proposer, no model) LANDED + gated: 27B 5/5 STRICT our-ngram-ON == vLLM-ngram-ON
   + 180/180 accepted, spec-OFF byte-identical, no new kernel. `SPEC-EAGLE3` honest
   reachable-BLOCKED (no ungated oracle-runnable EAGLE3 draft arch/checkpoint for a
   Qwen3.6 gate model at pin `555967922`; port scoped). See
   [spec-decode-breadth-d3.md](spec-decode-breadth-d3.md).
7. **`ROAD-V1-C8` /metrics + streaming parser + utility endpoints** `[H]` (oldest T0
   debt). W: `/metrics` Prometheus core → `TOOLS-STREAMING-PARSER` →
   `SERVE-RESPONSE-METRICS` → `/tokenize`,`/detokenize`,reset. **Gate:** metric-name +
   streamed-delta parity vs vLLM. **Size M→L.** **Vehicle:** the OpenAI server.
8. **`ROAD-V1-C4` GGUF decode 10× + FP8-generic** `[H]` (GGUF DONE is one specific
   lever away). W: `QUANT-GGUF-CPU-THREADPOOL` W4 decode-thread scaling (8.05×→≥10×) +
   x86 SIMD G5/G8; then FP8-generic dispatch; then AWQ/GPTQ/Marlin wiring. **Gate:**
   ≥10× decode A/B + e2e correctness per scheme. **Size S (decode bar) → L (breadth).**
   **Vehicle:** Qwen3.5-2B Q8 (CPU); FP8 checkpoints on DGX.
9. **`ROAD-V1-C6` SERVE-ASYNC-LLM + priority prod gates** (shared with A#5). W: land the
   depth-2 throughput lever, flip `runner_supports_async` prod-ON, close priority/busy-
   loop GPU gates. **Gate:** every-axis no-regression + priority-vs-FCFS token-exact.
   **Size M.** **Vehicle:** 27B online serving.
10. **`ROAD-V1-C2` text-sweep speed close + MoE/SSM breadth** (20 families correctness-
    complete). W: per-family speed lever to vLLM parity; then Qwen3-Next / Falcon-H1 /
    GraniteMoe / Mamba-hybrid campaigns; Kimi-Linear-48B (+KDA kernel, disk reclaim).
    **Gate:** token-exact (near-tie-robust) AND ≥ vLLM every axis per model. **Size L
    (ongoing).** **Vehicle:** each family's smallest gateable checkpoint.
11. **`ROAD-V1-D4` KV-disk/LMCache finish** (mostly landed). W: `KV-EVENTS` payload +
    emission sites; W6 LMCache go/no-go; W7 named save/restore; larger-model LMCache
    grid. **Gate:** every-axis LMCache grid vs vLLM `--kv-transfer-config` on a large
    model. **Size S→M.** **Vehicle:** a larger dense model + a live LMCache server.
12. **`ROAD-V1-D5` expert-streaming → LoRA** `[H]` (LoRA is headline; large). W:
    `ENG-EXPERT-STREAM` W0–W2; then `LORA-RUNTIME` (Punica batched apply) + `LORA-
    ENDPOINTS` + `ENG-WEIGHT-OFFLOAD`. **Gate:** token-exact LoRA apply vs vLLM + memory
    gate. **Size L.** **Vehicle:** a base model + a public LoRA adapter.
13. **`ROAD-V1-D1` Metal/MLX + CPU perf close** (the only reachable accelerators). W:
    Metal batched-decode tile kernel + binding harness (M4 quiet/sudo); CPU B4 speed/RSS.
    **Gate:** ≥ MLX-LM (Metal) / ≥ llama.cpp (CPU) binding A/B. **Size M.** **Vehicle:**
    Qwen3-1.7B/4B on M4; Qwen3.5-2B Q8 on CPU.
14. **`ROAD-V1-C2-LOCAL-BF16` device-resident sampled-token rerun** (small). **Size S.**
15. **`ROAD-V1-C1` fusion perf interpreter + FUSION-DENSE-MIGRATE** (cornerstone done;
    perf tail, ≤3.5%/step ceiling). **Size M.**
16. **`ROAD-V1-C9` 0.26 denominators/goldens refresh** (recurring). **Size S, ongoing.**
17. **`ROAD-V1-C3` DSpark + TLI** (core spec-decode done; overlaps D3). **Size M.**

## 4. Bottom line

- **DONE (merged+gated): 2** portfolio rows — `C1` (extensibility + fusion ORDER-1
  cornerstone) and `C3` (MTP + DFlash spec-decode, both speed-gated). At the
  *sub-milestone* level far more is delivered (MM image/video/audio **correctness**,
  first additive model + a 20-family text sweep **correctness**, D4 KV-disk+LMCache
  **landed**, D1 CUDA-arch **seams**), but those rows retain an open SPEED or breadth
  gate and so are not row-DONE.
- **REACHABLE-INCOMPLETE: 14** rows — executable now on GB10 + CPU + M4. The single
  dominant cross-cutting gate is **every-axis SPEED**: it is the entire remaining work
  for MM and the 20 correctness-complete model families, and a major part of A, C4, C5,
  C6, D1, D4. The reachable feature-parity gaps (C7 sampling, C8 metrics/parser,
  D4-APC prefix, D5 LoRA) are ordinary ports with existing harnesses. (D3 spec-breadth
  CLOSED 2026-07-27: ngram gated + EAGLE3 honestly blocked.)
- **HW/EXTERNAL-BLOCKED: bounds completeness.** "Complete roadmap_v1" is **NOT fully
  reachable on the current hardware.** It is bounded by, and only by, the blocked set in
  §2: `ROAD-V1-D2` multi-GPU; the `ROAD-V1-D1` non-CUDA accelerators (ROCm, Intel XPU,
  discrete Vulkan, ANE) and CUDA arch fan-out beyond sm_121; RDMA/multi-node KV
  connectors + PD-disaggregation; the >119 GiB frontier models; and the HF-token /
  DSA-dep models (Gemma-4 checkpoints, Command-R, DeepSeek-V3.2/GLM-5). Every one of
  those has a landable additive/build-only portion (honesty-pass ceiling) but no runtime
  gate without the missing board/token/dep. **Everything else — the whole C-track
  feature portfolio, all speed closure, the reachable model zoo, Metal/CPU — is
  reachable and is the punch-list in §3.**
