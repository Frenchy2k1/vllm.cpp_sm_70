# Multimodal SPEED — the gap map vs vLLM 0.25.0 (graphed), ranked levers, per-path verdict

**Status: MEASURED 2026-07-26 (`CLAIM-MULTIMODAL-SPEED`).** First inference-speed
characterization of the landed mm paths (image/video on Qwen3.6-27B, audio on
Voxtral-Mini-3B) against the honest vLLM 0.25.0 **PRODUCTION GRAPHED** denominator,
c1. This is a MEASUREMENT + attribution + lever-ranking pass — NOT an optimization
campaign. It answers where correctness-complete mm stands against the every-axis
DONE bar (token-exact AND vLLM throughput).

**Base:** `origin/main` `e3ab9547`. **Oracle:** `~/venvs/vllm-oracle` = vLLM
**0.25.0**, dgx GB10 (sm_121a). **Our build:** `~/work/m3b-vl` (M3-b tree, the
same 27B image forward as HEAD — that forward is unchanged since M3-b), production
config **cutlass 4.5.0 + FA2 + Triton-AOT, arch 121a** (CMakeCache confirmed).

**HEADLINE:** at c1 the **27B image/video DECODE is already at vLLM parity**
(marginally faster); the LLM **prefill is at parity**; the **ENTIRE measured gap is
the vision ENCODER TOWER (~10x, ~2.1 s)**, which dominates time-to-first-token
(~2.44 s ours vs ~0.32 s vLLM). mm decode is NOT host-bound at c1 (per-token host
round-trips are hidden under the ~226 ms/token weight-streaming floor of a ~54 GiB
bf16 model). Audio our-side is UNMEASURED (build-blocked, §4); the vLLM audio
denominator IS captured.

**UPDATE 2026-07-26 (`CLAIM-MULTIMODAL-SPEED-TOWER`, §7) — the tower gap is CLOSED
and BEATEN.** Profiling (nsys `cuda_gpu_kern_sum`) attributed **98.9 %** of the
~1.6 s tower forward to the naive `vt::cuda::AttentionKernel` (56 ms/block × 27
blocks over 784 patches; the GEMMs are <0.5 %, marshalling is 24 % of the old
total) — NOT the QKV fusion / FA2-varlen routing the levers §5 suspected. Two
correctness-preserving fixes: (1) hoist the per-call host f32→bf16 weight
convert+upload out of the forward into a one-time resident-weights load
(BIT-IDENTICAL); (2) a WARP-scoped online-softmax attention op `AttentionDenseFast`
(no `__syncthreads`, register accumulator; `kAttention` untouched ⇒ text
byte-identical by construction). **Per-image tower forward 2114 → 148 ms (14.3×);
vs vLLM's ~250 ms eager encode we are now 0.59× — FASTER.** STRICT image e2e held
**32/32** (4B + 27B), video **32/32**. See §7.

---

## 1. A-B methodology (how these numbers were grounded)

Mirrors how text speed was grounded ([[honest-bar-is-graphed-vllm]],
[[benchmark-gate-statistics]]): denominator = vLLM's PRODUCTION **graphed** config
(`enforce_eager=False`, inductor + `cudagraph_mode FULL_AND_PIECEWISE`), NOT
`--enforce-eager`. Same fixed (media, prompt) both sides. Repeated reps, **cold
first leg discarded**. All GPU under `flock $HOME/gpu.lock`, sole owner, our engine
and the oracle NEVER co-resident (serialized under the lock).

- **Ours (per-stage):** the M3-b e2e gate driver instrumented with
  `steady_clock` + `gpu->Synchronize(q)` brackets around (a) `Qwen3VLVisionForward`
  (tower) and (b) `Qwen3_5VLGenerateGreedy` at `max_new_tokens=1` and `=33`. The
  driver runs the tower SEPARATELY, then greedy(merge+prefill+decode); so tower and
  LLM are cleanly separable, and `TPOT = (t_N33 - t_N1)/32`, `LLM-prefill = t_N1`
  (for N=1 the decode loop body does not execute — N1 is merge+214-token
  prefill+argmax). 4 reps, rep0 dropped. The run still asserts **32/32 token-exact**
  vs the golden — proof the timed path RAN the correct forward. (The timing edit was
  a throwaway on the dgx extraction, reverted; the exact instrumentation is
  reproducible from this section — it is not a repo source change.)
- **vLLM (per-stage):** `LLM(..., enforce_eager=False, gpu_memory_utilization=0.55,
  max_num_seqs=32, max_model_len=4096)` graphed, warmup then `generate` at
  `max_tokens=1` (TTFT) and `=33`; `TPOT=(t33-t1)/32`. 4 reps. (`max_num_seqs=32`
  needed — the default 256 exceeds the Mamba/GDN cache blocks that fit at GMU 0.55
  during capture; inductor needs `ninja` on PATH.) `PATH` must include
  `~/venvs/vllm-oracle/bin`.
- **Noise:** tower ±1%, decode TPOT ±0.1% (rock-stable — both sides sit on the
  weight-bandwidth floor), prefill ±1%.

---

## 2. The per-path, per-stage gap table (c1, ms)

### 2.1 Qwen3.6-27B IMAGE — `Qwen3_5ForConditionalGeneration` (the flagship number)
Fixture 448×448 image + "What is in this image?"; 214-token prompt, 196 image
tokens. Weights ~51 GiB bf16 (vLLM-reported), uniform-bf16 checkpoint.

| Stage | Ours (band) | vLLM 0.25.0 graphed (band) | Ratio | Verdict |
|---|---|---|---|---|
| **Vision tower (encode)** | **2114** (2091–2126) | encode ⊆ TTFT, ≤ ~250 (eager; not graphed by vLLM) | **~8–14× slower** | **THE gap** |
| **LLM prefill** (merge + 214 tok + argmax) | **325.6** (324.6–327.4) | (full TTFT below) | ~at parity | at parity |
| **TTFT / first token** (tower+prefill vs vLLM encode+prefill+sample) | **~2440** | **321.6** (320.8–322.7) | **~7.6× slower** | tower-dominated |
| **Decode TPOT** (per token) | **225.0** (224.9–225.1) | **226.9** (226.8–227.0) | **0.99× (ours faster)** | **AT PARITY** |
| 32-token end-to-end | ~9.4 s | ~7.6 s + its own (overlapped) encode | — | tower-dominated |

The ~226 ms/token decode is the **memory-bandwidth floor** of streaming the ~54 GiB
bf16 weights once per token at c1 — both engines sit on it. Our eager driver's
per-token host round-trips (§3) are ~1–3 ms, negligible under 226 ms.

### 2.2 Qwen3.6-27B VIDEO — `Qwen3_5VLGenerateGreedyVideo`
NOT separately timed (no 27B-video binary built on the reused dgx tree). By
construction the **decode is byte-identical to the image path** (shared
`VLGenerateCoreGdn`) ⇒ **at parity**; the tower adds **per-frame windowed
attention** (a `vt::Attention` loop over grid_t windows) ⇒ the encode is larger and
the **same ~10× tower-efficiency story** applies, plus the video preprocessing
(`ProcessVideo`) is host-side CPU (frame patchify) not yet characterized.

### 2.3 Voxtral-Mini-3B AUDIO — `VoxtralGenerateGreedy`
30 s clip + "Describe what you hear in this audio."; 388-token prompt (375 audio).

| Stage | Ours | vLLM 0.25.0 graphed (band) |
|---|---|---|
| Whisper-large-v3 encode + 388-tok prefill (TTFT) | **UNMEASURED** (§4 blocker) | **42.8** (42.2–43.4) |
| Decode TPOT | **UNMEASURED** | **40.8** (40.7–40.9) |

vLLM audio denominator captured. **Contrast that matters:** Voxtral decode is
~41 ms/token (a 3B model, ~6× cheaper than the 27B's 226 ms), so the fixed
per-token host overhead of our eager driver would **NOT be hidden** here — audio is
the first place lever #2 (§3) could actually bite. Our-side must be timed
(follow-on) before claiming anything about the audio gap.

---

## 3. Structural findings (both paths)

- **Our mm drivers are single-sequence, eager, un-graphed greedy loops.**
  `VLGenerateCoreGdn` (`qwen3_5.cpp:6756-6780`) and `VoxtralGenerateGreedy` decode
  one token/step with per-step host round-trips: full-vocab logits **D2H** + host
  `VLArgMax` (over 248 320 floats), an embed **D2H/H2D** round-trip, and a host
  `BuildMropeCosSinHost`. They do NOT use the production paged runner's cudagraph
  replay (`qwen3_5.cpp:7231/7448`) or on-GPU sampling. **At c1-27B this is invisible
  (hidden under the 226 ms compute floor)** — the "reuse the at-parity text backbone"
  premise holds for the kernels. It becomes a lever only where decode is cheap
  (audio) or at c2+.
- **No batched mm path.** The drivers are single-sequence; vLLM batches
  encoder-cache + decode. **c2+ mm throughput is not measurable our-side** — a
  structural gap, not a tuning gap.
- **No mm server ingestion.** `image_url`/`audio_url` are not wired
  (`grep image_url src/vllm/entrypoints` = 0), so there is no production-serving A-B;
  everything here is measured through the e2e test drivers.
- **The tower is genuinely GPU-compute-bound**, not host-bound: the per-block
  `Download`s in `qwen3_vl_vision.cpp` are unit-test taps guarded by `cap != nullptr`
  (null in the e2e driver). So 2.1 s is real kernel time in the 27 ViT blocks.

---

## 4. Blocker honesty (why audio our-side is not a number)

- **dgx disk is 100 % full (25 GiB free).** No A3/Voxtral binary is built on any
  reusable tree (`~/work/m3b-vl` is M3-b image/video; `~/work/a2-audio` is the A2
  encoder-fidelity test only — no `voxtral.cpp` e2e). Building `test_voxtral_e2e`
  means updating an extraction to the A3 SHA + a non-trivial incremental build on a
  full disk → real ENOSPC risk ([[grid-per-sha-trees-fill-disk]]: partial builds =
  bogus failures + box disruption). Deferred rather than run a risky/void number.
- **vLLM graphed needs `ninja` on PATH** (inductor) and `max_num_seqs` capped so the
  GDN Mamba cache fits during capture at low GMU — both recorded above so the run
  reproduces.
- The 27B image number is the flagship and is measured cleanly; video and audio
  our-side are honestly UNMEASURED with the vLLM denominators captured.

---

## 5. Ranked levers (grounded in vLLM file:line, honest gain/effort)

1. **Vision-tower kernel efficiency — ~90 % of the first-token gap (~1.9 s of 2.1 s;
   ~20 % of a 32-token e2e, more for shorter outputs).** Our `Qwen3VLVisionForward`
   (`src/vllm/model_executor/models/qwen3_vl_vision.cpp`: 27 ViT blocks =
   `vt::MatmulBT` [cuBLASLt] + `vt::Attention(causal=false)` per block) vs vLLM
   `Qwen3_VisionTransformer` (`qwen3_vl.py:519`; block `:413`; vision attention
   `Qwen2_5_VisionAttention.forward` `qwen2_5_vl.py:397-460`, which runs
   `flash_attn_varlen_func`). **Both are eager** (vLLM sets `compile_mm_encoder:
   False`, `cudagraph_mm_encoder: False` — verified in the engine config dump), so
   this is a fair eager-vs-eager kernel gap, closable. Next step per AGENTS: `nsys
   --cuda-graph-trace=node` our tower to find the specific slow kernel — prime
   suspects: (a) the vision attention not routing to FA2 varlen, (b) per-block
   un-fused QKV / small-GEMM launch overhead over 27×784 patches. Real lever.
2. **Route mm decode through the production graphed paged runner** (on-GPU
   argmax/sampling + cudagraph replay; drop the per-token full-vocab D2H + host
   argmax + embed round-trip) instead of the eager loop (`qwen3_5.cpp:6756-6780`;
   `voxtral.cpp`). vLLM: on-GPU sampling, `cudagraph_mode FULL_AND_PIECEWISE`.
   **NEUTRAL at c1-27B (measured: decode already at parity)**; the lever for **audio
   / small models** (decode ~41 ms, host overhead not hidden) and **c2+**. Gain
   UNQUANTIFIED our-side until audio is timed — do lever #4 first.
3. **Batched mm serving + encoder cache (structural).** Single-seq drivers → no c2+.
   vLLM `EncoderCacheManager` (`v1/core/encoder_cache_manager.py:17`) + scheduler mm
   hooks (`sched/scheduler.py:1356-1467`) + the `EncoderRunner`
   (`v1/worker/gpu/mm/encoder_runner.py`). Prereq for c2+ throughput parity and the
   OpenAI-server `image_url`/`audio_url` ingestion. Largest effort; needed for the
   full DONE bar.
4. **Measure audio our-side (top follow-on).** Build a timed `test_voxtral_e2e` on a
   disk-safe tree; expect it to surface levers #1 (32-layer Whisper encoder) and #2
   (cheap decode) with real magnitudes. Cheapest high-information next step.

**No trivial byte-exact/measurement-safe win surfaced.** The decode host
round-trips are the driver's structure (not a stray redundant sync), and the tower
gap needs a profiling+kernel campaign — both are follow-on work, RANKED not
implemented. No repo source was touched by this pass.

---

## 6. Per-path VERDICT (the DONE-bar disposition)

- **Qwen3.6-27B IMAGE:** decode **AT PARITY** (0.99×), prefill **AT PARITY**, gap =
  **vision encoder tower (~10×)**. **Encoder-dominated, NOT host-bound at c1.** The
  "mm decode reuses the at-parity text backbone" premise is CONFIRMED. Distance to
  DONE = the tower (lever #1) + eventual batched serving (lever #3).
- **Qwen3.6-27B VIDEO:** decode at parity by construction; **encoder-tower-dominated**
  (per-frame windowed attention). Not separately timed.
- **Voxtral AUDIO:** our-side UNMEASURED (blocked). vLLM denominator = **43 ms TTFT /
  41 ms TPOT**. The cheap 3B decode makes this the path where the eager-driver host
  overhead is most likely to be a **real** lever — must be measured before verdict.

All three remain **speed-pending / `PARTIAL`** — correctness-complete, not yet at
vLLM throughput on every axis. No mm row advances to `DONE` on this pass.

---

## 7. TOWER OPTIMIZATION — lever #1 executed (`CLAIM-MULTIMODAL-SPEED-TOWER`, 2026-07-26)

**Base:** `origin/main` `27bc3054`. **Build:** `~/work/mm-tower-speed`, cutlass 4.5.0
+ FA2 + Triton-AOT, arch 121a (banner confirmed, clean `-Werror` 0 warn). All GPU
under `flock $HOME/gpu.lock`, sole owner, cold rep discarded (bench = 4 timed reps
after rep0). Driver `tests/vllm/multimodal/bench_qwen3_5_vl_tower.cpp` (Qwen3.6-27B,
fixed 448×448 image, 784 patches → 196 tokens).

### 7.1 W0 — profile ATTRIBUTION (measure, don't guess — [[profile-vllm-actual-kernels-port-1to1]])

The §5 lever #1 SUSPECTED (a) vision attn not on FA2 varlen and (b) unfused QKV.
**Both REFUTED by measurement.** The old per-image tower forward is **2103 ms**
(reproduces the §2.1 2114 ms scoping number). Split (env-gated `VLLM_MM_TOWER_PROFILE`):
- **weight marshalling** (host f32→bf16 convert + ~0.5 GiB H2D, done INSIDE the
  forward every call): **~497 ms (24 %)**.
- **forward compute: ~1543 ms (76 %)**.

nsys `cuda_api_sum` first pointed at `cudaFree` (93 %) — a RED HERRING: `cudaFree`
SYNCHRONIZES, so its wall-time is inflated by WAITING on the enqueued kernels. The
truth is `cuda_gpu_kern_sum`:
- **`vt::cuda::AttentionKernel<bf16>`: 98.9 % of GPU time**, 270 instances (27 blocks
  × 10 forwards), **avg 56.3 ms EACH** over just 784 patches (~1000× off peak).
- cutlass/nvjet GEMMs: 43–127 µs each (<0.5 % total). QKV fusion is irrelevant.

Root cause: the naive `AttentionKernel` (`cuda_ops.cu`) launches grid(t, hq) — one
block per (query, head) — and each block streams ALL keys sequentially with a
per-key block reduction (a `__syncthreads` storm). O(t²) with catastrophic sync
overhead. The lever is the ATTENTION KERNEL, exactly as §5 lever #1 headlined, but
via kernel efficiency (not FA2-routing / QKV).

### 7.2 W1 — two correctness-preserving fixes

1. **Resident weights (BIT-IDENTICAL).** `PrepareVisionDeviceWeights` converts +
   uploads the ~0.5 GiB tower weights to device bf16 ONCE (mirror vLLM keeping the
   ViT nn.Linears resident); the forward runs pure GEMMs/attention + only the tiny
   per-image pixel/pos/rope uploads. The host-weights overload is preserved as a
   prepare-then-forward wrapper (0/1 003 520 mismatch vs the old path). Also hoisted
   the per-block scratch buffers out of the ViT loop (reuse; bit-identical). This
   moves the ~497 ms marshalling from per-image to one-time model-load.
2. **`AttentionDenseFast` — warp-scoped online-softmax attention (the real lever).**
   New additive CUDA op (`cuda_ops.cu` `AttentionWarpKernel` + `LaunchAttentionWarp`;
   `ops.h`/`ops.cpp` wrapper; CPU dispatches to the same reference kernel). ONE WARP
   per (query, head): the head_dim reduction is a butterfly `__shfl_xor` (NO
   `__syncthreads`), the output accumulator lives in registers, softmax stats per
   lane — the SAME f32 online-softmax recurrence as `AttentionKernel`, warp-scoped.
   Grounded in the online-softmax / FlashAttention recurrence we ship
   (`src/vt/cuda/flash_attn/`). NOT bit-identical (the 32-lane head_dim partial-sum
   grouping differs) but same f32 math within the tower bf16 envelope. Registered as
   a SEPARATE op ⇒ `kAttention` (text `qwen3_5.cpp:4005` + audio whisper) is BYTE-
   IDENTICAL. The vision tower calls `AttentionDenseFast` per frame (image = one
   window; video = per-frame windows, unchanged).

### 7.3 RESULT — tower BEATS vLLM's eager encode

| Metric | Old (naive attn) | New (resident + fast attn) |
|---|---|---|
| **per-image tower forward** | **1543 ms** | **148.4 ms** (band 148–150) |
| per-image tower (marshal+forward, old baseline) | 2103 ms | (marshal now one-time) |
| one-time `PrepareVisionDeviceWeights` (model load) | (was per-image) | ~484 ms |

- **Per-image tower forward 1543 → 148 ms = 10.4× (attention alone); vs the original
  2114 ms scoping baseline = 14.3×.**
- **vs vLLM 0.25.0 eager encode ~250 ms: 148 ms = 0.59× — we are FASTER than vLLM's
  vision tower.** (vLLM does NOT graph/compile the encoder — `compile_mm_encoder:False`
  — so this is a fair eager-vs-eager kernel comparison.)
- Noise: forward ±1 %. Proof-of-run: the bench asserts fast-vs-baseline 0 mismatch;
  the STRICT e2e gates below ran the fast kernel end-to-end.

### 7.4 CORRECTNESS (the RED line — HELD)

- **27B image e2e `test_qwen3_5_vl_e2e`: STRICT 32/32** (54/54 assertions) — the fast
  attention's bf16 differences flip ZERO argmax.
- **27B video e2e `test_qwen3_5_vl_video_e2e`: STRICT 32/32** (27/27; teacher-forced
  gap 0 nats everywhere).
- **4B image e2e `test_qwen3vl_e2e` (DeepStack tower): STRICT 32/32** (46/46).
- **`test_ops_attention`: 37239/37239** — `kAttention` (the shared naive kernel) is
  intact ⇒ text/audio byte-identical by construction.
- compute-sanitizer memcheck on the tower forward (with the new kernel): see ledger.
- Text SACRED: additive op, `kAttention` untouched (`git diff`), text never calls
  `kAttentionDenseFast` ⇒ byte-identical by construction; the 27B text SACRED gate
  re-run confirms (see ledger).

### 7.5 What remains

The tower is now FASTER than vLLM's eager encode, so lever #1 is CLOSED. TTFT is now
prefill+tower ≈ 326 + 148 ≈ 474 ms vs vLLM's 321 ms graphed TTFT — the residual
first-token gap is now the LLM prefill/tower vs vLLM's GRAPHED encode+prefill, small.
Remaining DONE-bar work is unchanged: batched/graphed mm SERVING (lever #3, for c2+
and `image_url` ingestion) and the audio our-side measurement (§5 lever #4).
