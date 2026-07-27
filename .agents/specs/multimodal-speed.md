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

**UPDATE 2026-07-27 (`CLAIM-MM-SPEED-DECODE-KERN`, §11) — the audio-decode ~20 ms residual
ATTRIBUTED + a VALIDATED bf16-near-tie ceiling.** nsys of the graphed Voxtral decode: the
whole residual is ONE kernel — the naive scalar `PagedAttentionKernel` decode attention
(723 µs/call × 30 layers = 21.7 ms/step, ~120× the KV floor), NOT the GEMMs/glue. The 1:1 vLLM
lever (FA2 varlen decode, `flash_attn_varlen_func`) is ALREADY in the binary, gated off only
because the driver's single KV block (444) isn't ÷16. Enabling it: TPOT **59.4→38.2 ms/tok
(−21.2, ~36%) = 0.94× vLLM's 40.8 ms — CLOSES AND BEATS the gap**, and teacher-forcing vLLM on
the FA2 sequence PASSES (0 divergences, gap 0.0 — a valid greedy branch). BUT it flips the
committed near-tie golden's exact-tie branch (`repro` 48→18), so it is blocked byte-exact.
RECORDS-ONLY this pass (14/14 held, golden unchanged); win is one-line + golden-regen. See §11.

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
  `VLGenerateCoreGdn` (decode loop `qwen3_5.cpp:6871-6895`; the `6756-6780` range
  the earlier draft cited is the per-layer KV/GDN-state ALLOC loop, not the decode
  loop) and `VoxtralGenerateGreedy` (`voxtral.cpp:425-442`) decode
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
   argmax + embed round-trip) instead of the eager loop (decode loop
   `qwen3_5.cpp:6871-6895`; `voxtral.cpp:425-442`). vLLM: on-GPU sampling,
   `cudagraph_mode FULL_AND_PIECEWISE`.
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

---

## 8. DECODE SAMPLING — lever #2 executed (`CLAIM-MULTIMODAL-SPEED-DECODE`, 2026-07-27)

**Base:** `origin/main` `f2facf3c`. **Build:** `dgx.casa:~/work/mm-argmax-speed`,
cutlass 4.5.0 + FA2 + Triton-AOT, arch 121a (FA2 banner confirmed, clean configure).
All GPU under `flock $HOME/gpu.lock`, sole owner, cold rep0 discarded.

**What §5 lever #2 asked for, closed:** the mm eager greedy decode loops
`VLGenerateCoreGdn` (`qwen3_5.cpp`, the shared 27B image+video core) and
`VoxtralGenerateGreedy` (`voxtral.cpp`) now (a) run the greedy pick ON the GPU via
`vt::GreedyArgmax` (device vocab reduction, download only the winning int64 id —
not the full `[1,vocab]` f32 logits) and (b) embed the fed decode token ON DEVICE
and hand it straight to the forward, eliminating the redundant embed D2H→H2D
round-trip. The host `VLArgMax`/`ArgMax` scans are REMOVED — the device argmax is
the ONLY greedy path (proof-of-run: the gates below cannot pass unless it ran).

### 8.1 Grounding (ours + vLLM, file:line)
- Ours: `vt::GreedyArgmax` (`src/vt/ops.cpp:2574`; CUDA two-pass lowest-index-tie
  reduction `src/vt/cuda/cuda_sample.cu:83-215`) — the SAME device sampler the
  production paged runner uses (`src/vllm/v1/sample/sampler.cpp:315-318`).
- vLLM: greedy sampler path `vllm/v1/sample/sampler.py` (`sample`→`greedy_sample`,
  `torch.argmax(logits, dim=-1)`, lowest-index tie). Our `GreedyArgmax` mirrors the
  lowest-index tie-break exactly, so the winning token is byte-for-byte the token
  the removed host scan produced.

### 8.2 CORRECTNESS (the RED line — HELD, bit-exact)
The change is bit-identical BY CONSTRUCTION: the removed embed round-trip was a
LOSSLESS bf16 D2H→H2D (same bits), and the device argmax reduces the SAME f32
logits with the SAME lowest-index tie-break as the host scan (argmax winner is
comparison-reduction, order-independent). Verified on a clean dgx build of
`f2facf3c`, goldens md5-identical before+after:
- **27B image `test_qwen3_5_vl_e2e`: STRICT 32/32** (54/54 assertions).
- **27B video `test_qwen3_5_vl_video_e2e`: STRICT 32/32** (27/27, teacher-forced gap 0 nats).
- **4B image `test_qwen3vl_e2e` (unchanged code): STRICT 32/32** (46/46) — no collateral.
- **Voxtral audio `test_voxtral_e2e`: PASS 14/14** — reproduces the committed
  near-tie sequence **48/48** exactly, strict prefix 33/48. Both the device path
  AND the (throwaway A/B) host path produce the identical 48 tokens.

### 8.3 RESULT — decode TPOT A/B (same-binary, throwaway env toggle `VT_MM_HOST_ARGMAX`)
Same-binary A/B (dgx-only throwaway instrumentation, NOT committed): `DEV_NEW` =
shipped device path, `HOST_OLD` = restored full-vocab D2H + host argmax + embed
round-trip. 5 reps audio / 4 reps 27B, rep0 dropped.

| Vehicle | DEV_NEW TPOT (band) | HOST_OLD TPOT (band) | Δ | vs vLLM 0.25.0 graphed | Verdict |
|---|---|---|---|---|---|
| **Voxtral audio (3B)** | **61.85 ms** (61.73–61.94) | 62.08 ms (61.90–62.23) | **−0.25 ms/tok (~0.4%)** | vLLM 40.8 ms → **1.52× slower** | small REAL win; gap is NOT this lever |
| **Qwen3.6-27B image** | **223.0 ms** (221.7–225.2) | 224.0 ms (221.7–227.7) | ~0 (within ±1.5% noise) | vLLM 226.9 ms → **at parity** | **NEUTRAL** (bandwidth floor) |
| Qwen3.6-27B video | = image by construction (shared `VLGenerateCoreGdn`) | — | ~0 | at parity | NEUTRAL |

### 8.4 Honest disposition — lever #2 is CLOSED, the win is small; the audio gap is lever #3
The §5 #2 / §2.3 hypothesis was that audio (cheap ~41 ms decode) is where the
per-token host round-trips would BITE. **Measurement REFINES this:** our audio
decode is **~62 ms/token eager** (not 41 ms), and the host round-trips removed are
only **~0.25 ms of it** — a consistent ~0.4% win, not a large one. Even at 3B the
eager per-step forward dominates, so the host overhead was a thin slice. At 27B the
lever is NEUTRAL (decode sits on the ~222 ms weight-streaming floor; host round-trips
hidden — confirms §2.1). **The real audio-decode gap vs vLLM (1.52×) is eager
per-step launch overhead** — i.e. graphed/batched decode (lever #3), for which
on-GPU sampling is a PREREQUISITE this lever now supplies, but which it does not by
itself close. mm rows stay **speed-pending / `PARTIAL`**: correctness-complete,
decode at-parity on 27B, still 1.52× on audio (lever #3). No mm row advances to DONE.

---

## 9. DECODE GRAPH — lever #3 FIRST BRICK executed (`CLAIM-MULTIMODAL-SPEED-GRAPH`, 2026-07-27)

**Base:** `origin/main` `bd3e15ed`. **Build:** `dgx.casa:~/work/mm-decode-graph`,
cutlass 4.5.0 + FA2 sm_121a + Triton-AOT, arch 121a (FA2 ENABLED banner + Triton
vendored MANIFEST-OK confirmed at configure; clean `-Werror`, 0 warn). All GPU under
`flock /tmp/gpu`, sole owner (`nvidia-smi` idle), cold rep0 discarded.

Lever #3 is "batched/graphed multimodal serving" — the dominant remaining mm speed
residual (§8: the audio 1.52× is eager per-step LAUNCH overhead, closed only by
graphed/batched mm decode). It is a large multi-step effort; this is the SCOPE + the
first reachable, correctness-held brick.

### 9.1 SCOPE — the current mm serving/decode path, grounded (file:line)

- **mm SERVING ingestion is UNWIRED, and the engine cannot consume mm data yet.**
  `from_json(ChatMessage)` parses ONLY bare-string `content`
  (`src/vllm/entrypoints/openai/protocol.cpp:298-300`, comment "multimodal
  content-part arrays deferred"); `grep image_url|audio_url src/vllm/entrypoints` = 0.
  No `multi_modal_data` field reaches `LLMEngine`/`serving_chat` (grep in
  `entrypoints/`+`engine/` = 0). So `image_url`/`audio_url` ingestion is not a
  one-file parse brick — it needs the parse layer AND an engine mm-request path AND
  processor invocation (a multi-W chain; W-plan below). The lever-#2 agent's note is
  CONFIRMED: single-sequence, `image_url`/`audio_url` unwired.
- **mm DECODE was single-sequence + eager + un-graphed.** The 27B image+video share
  `VLGenerateCoreGdn` (`qwen3_5.cpp:6724`); Voxtral audio is `VoxtralGenerateGreedy`
  (`voxtral.cpp:375`). Both ran an eager per-step forward (`DenseForwardLayers` /
  `ForwardLastLogits`) rebuilding host metadata each step — NOT the production paged
  runner's captured decode.
- **The graphed decode step ALREADY EXISTS for the 27B-dense family**, unused by the
  mm path: `Qwen3_5DenseDecodeGraph` (`include/vllm/model_executor/models/qwen3_5_dense.h:314`;
  impl `qwen3_5.cpp:7428`) — the production cold→warm→replay captured decode the text
  runner uses, `BuildPaddedDecode` (`qwen3_5.cpp:7153`) padding a real batch B up to a
  captured size S (`PadToCaptureSize`, `decode_graph_sizes.h:47`). At **S==B it is a
  bit-identical rebuild of the eager inputs** (`qwen3_5.cpp:7151`). Voxtral's Llama/
  Mistral text stack has **NO** decode-graph class (only Qwen3.5/MoE/DeepSeek do).
- **vLLM port target (HARD-rule grounding):** vLLM graphs decode over mm requests via
  the generic decode cudagraph dispatcher (`vllm/compilation/` + `gpu_model_runner`
  `_dummy_run`/capture), with the mm ENCODER kept eager (`compile_mm_encoder:False`)
  and the EncoderCacheManager (`vllm/v1/core/encoder_cache_manager.py:17`) + scheduler
  mm hooks (`sched/scheduler.py:1356-1467`) admitting batched mm requests into the same
  graphed decode step. `Qwen3_5DenseDecodeGraph` is our 1:1 of that per-size captured
  decode; this brick makes the mm decode USE it.

### 9.2 FIRST BRICK — route the 27B mm dense decode through `Qwen3_5DenseDecodeGraph`

`VLGenerateCoreGdn`'s decode loop (shared by 27B image + video) now runs each
pure-decode step through `Qwen3_5DenseDecodeGraph::Step` (a single instance built per
generate, `max_num_reqs=1`), instead of the eager `DenseForwardLayers(...,&mrope_dec)`.
Single-seq ⇒ B=1, `PadToCaptureSize(1,1)=1` ⇒ **S==B==1**, the bit-identical-rebuild
case. The embed runs on device inside `Step`; the returned `[1,vocab]` logits stay on
device and feed `vt::GreedyArgmax` directly (no full-vocab D2H). The eager mrope path
is preserved behind `VT_MM_DECODE_EAGER=1` (default = graph; parity-enabler-as-default
policy). One file touched: `src/vllm/model_executor/models/qwen3_5.cpp`.

**ROPE EQUIVALENCE (why it is token-exact).** During decode every position is a text
token with the MRoPE 3-axis positions equal on all axes (`pos3_dec={p,p,p}`), so
MRoPE degenerates to 1-D RoPE at `p`. `Step` applies 1-D device RoPE from `positions`,
so passing `positions={p}` (`p=abs_idx+delta`, the MRoPE-adjusted decode position)
reproduces the eager mrope rope angle; the KV physical slot stays `abs_idx`. This was
a HYPOTHESIS the token-exact gate ARBITRATES — and it PASSES STRICT (below): the
host-MRoPE-cache vs device-1-D-rope difference flips zero argmax.

### 9.3 CORRECTNESS (the RED line — HELD, token-exact, proven-to-run)
Clean dgx build of `bd3e15ed` + this brick; golden md5 unchanged
(`gen_tokens_i32.bin` = `3bc5f231…`, before==after). Proof-of-run: `VT_DECODE_GRAPH_STATS=1`
printed `captured … padded size S=1 (real B=1)` + **30 replays** on each gate — the
graphed path DID execute.
- **27B image `test_qwen3_5_vl_e2e`: STRICT 32/32** (54/54 assertions; ours == golden
  `760,1156,6587,728,310,10229,1092,369,…`).
- **27B video `test_qwen3_5_vl_video_e2e`: STRICT 32/32** (27/27; teacher-forced gap 0
  nats everywhere — the graphed decode lands the strict target, not just near-tie).

### 9.4 RESULT — decode TPOT A/B (same-binary throwaway toggle `VT_MM_DECODE_EAGER`)
Same-binary A/B (dgx-only throwaway instrumentation, NOT committed): 4 reps/mode in
ONE model load, rep0 dropped. `tpot31 = gen32_wall/31` (includes amortized prefill;
the A/B DELTA is definition-independent).

| Vehicle | GRAPH (band) | EAGER (band) | Δ | vs vLLM 0.25.0 graphed | Verdict |
|---|---|---|---|---|---|
| **Qwen3.6-27B image (32 tok)** | **232.5 ms/tok** (231.8–233.9) | 233.4 ms/tok (233.35–233.5) | **−0.9 ms/tok (~0.4%, graphed faster)** | decode at parity (§2.1) | **NEUTRAL** (bandwidth floor) |
| Qwen3.6-27B video | = image by construction (shared `VLGenerateCoreGdn`) | — | ~0 | at parity | NEUTRAL |

**Honest disposition.** As §8 predicted, graphing 27B mm decode is NEUTRAL — the
per-step host launch overhead the graph removes (~1 ms/tok) is hidden under the
~222 ms weight-streaming floor. The brick's value is STRUCTURAL: it closes the §3
"un-graphed eager loop" gap for the 27B image+video path (the decode step is now
graph-capturable, reusing the production replay), a prerequisite for batched (c2+) mm
decode and the mechanism the audio 1.52× needs. The measured launch-overhead win
lands where decode is CHEAP — the Voxtral audio path — which has NO decode-graph class
yet (W-plan W1). mm rows stay **speed-pending / `PARTIAL`**. No mm row advances to DONE.

### 9.5 Remaining lever-#3 W-plan (W-step → gate → size)
- **W1 — Voxtral (Llama/Mistral) decode-graph class — DONE 2026-07-27
  (`CLAIM-MM-SPEED-GRAPH-W1`, §9.6). Real non-overlapping win, but it NARROWS (does
  NOT close) the 1.52× gap.** Built `VoxtralDecodeGraph` and routed
  `VoxtralGenerateGreedy`'s decode through it. See §9.6.
- **W2 — batched multi-seq mm decode (S>1 / c2+).** Drive `VLGenerateCoreGdn` (and W1's
  audio) with B>1 requests through the SAME graph at padded S∈{2,4,8,…}; needs multi-slot
  GDN/KV caches + the per-request mm merge at prefill. GATE: 2-request batched decode
  token-identical to two single-seq runs + a c2 throughput A/B. Size: **L**.
- **W3 — mm SERVING ingestion (`image_url`/`audio_url`).** Parse content-part arrays in
  `protocol.cpp` → mm feature extraction (the landed `src/vllm/multimodal/*` processors)
  → an engine mm-request path (`multi_modal_data` on the request, the EncoderCacheManager
  seam) → `serving_chat`. GATE: a served mm chat completion whose tokens == the e2e
  driver's. Size: **L** (parse + engine request path + processor wiring; the true
  prerequisite for a production-serving c2+ A/B).

---

## 10. W1 — VoxtralDecodeGraph (`CLAIM-MM-SPEED-GRAPH-W1`, 2026-07-27)

**Base:** `origin/main` `e2b18fc8` (the §9 FIRST-BRICK HEAD). **Build:**
`dgx.casa:~/work/mm-voxtral-graph`, cutlass 4.5.0 + FA2 sm_121a + Triton-AOT, arch 121a
(FA2 ENABLED banner + Triton vendored MANIFEST-OK confirmed at configure; clean build).
All GPU under `flock /tmp/gpu`, sole owner (`nvidia-smi` idle), cold rep0 dropped.

**What §9.5 W1 asked for, closed:** Voxtral's Mistral/Llama text stack was the ONLY mm
text stack with **no decode-graph class** (Qwen3.5-dense/MoE/DeepSeek all had one). Built
`VoxtralDecodeGraph` (`voxtral.{h,cpp}`) — the Voxtral-text sibling of `Qwen3MoeDecodeGraph`
(Qwen3-Coder, the closest precedent: pure full-attention over the SAME
`dense_attn::AttnBlock` + `vt::PagedAttention` stack Voxtral uses, no GDN) — with the SAME
cold→warm→replay state machine, padded-batch capture set (`decode_graph_sizes.h`) and
persistent fixed-address host inputs + persistent embed/logits buffers. Routed
`VoxtralGenerateGreedy`'s pure-decode loop through `VoxtralDecodeGraph::Step`; the eager
path stays behind `VT_MM_DECODE_EAGER=1` (default = graph). One src file + its header.

### 10.1 Grounding (ours + vLLM, file:line)
- Ours: template `Qwen3MoeDecodeGraph` (`qwen3_moe.cpp:332`); the captured region is the
  EXACT `ForwardLastLogits` op sequence (`voxtral.cpp`) the eager decode already ran; the
  embed is kept OUTSIDE the capture (`VoxtralEmbedInto`, mirror `EmbedInto`).
- vLLM: the generic decode cudagraph dispatch — `gpu_model_runner.py::GPUModelRunner`
  (`_dummy_run` warm-up then `capture_model`) + `compilation/cuda_graph.py`
  (pad-to-nearest-captured-size) @ pin `555967922`.
- Capture-safety with growing seq_len: the paged full-attention decode (hd-128, GQA 32/8)
  is the SAME path the already-gated Qwen3-Coder decode graph captures — host `max_seq_len`
  only sizes the split grid; per-request geometry is read from the DEVICE `seq_lens`
  (`cuda_flash_attn_fa2.cu:23-31`), so a captured graph stays correct as the sequence grows.

### 10.2 CORRECTNESS (the RED line — HELD, token-exact, proven-to-run)
Clean dgx build; golden md5 UNCHANGED before+after (`voxtral_golden.json`
`8ab87b7e…`, `voxtral_neartie.json` `3d199c2d…`). Proof-of-run: `VT_DECODE_GRAPH_STATS=1`
printed `captured Voxtral text decode graph … S=1 (real B=1)` + **46 replays** on the gate
— the graphed path DID execute. **`test_voxtral_e2e`: PASS 14/14** — reproduces the
committed near-tie seq **48/48** exactly, **STRICT prefix 33/48**, near-tie result PASS,
worst gap 0.0 nats. Held across all 12 A/B runs (both modes, every run 14/14). The S==B==1
bit-identical-rebuild premise is arbitrated by the token-exact gate and PASSES.

### 10.3 RESULT — decode TPOT A/B (same-binary throwaway toggle `VT_MM_DECODE_EAGER`)
Same-binary A/B (dgx-only throwaway steady-clock instrumentation around the decode loop,
NOT committed): steady-state TPOT excludes the 2 cold+warm decode steps; 6 reps/mode in
separate loads under ONE `flock`, rep0 dropped.

| Vehicle | GRAPH steady (band) | EAGER steady (band) | Δ | vs vLLM 0.25.0 graphed 40.8 ms | Verdict |
|---|---|---|---|---|---|
| **Voxtral-Mini-3B audio (48 tok)** | **60.94 ms/tok** (60.79–61.07) | **61.71 ms/tok** (61.57–61.88) | **−0.77 ms/tok (~1.25%, graphed faster), NON-OVERLAPPING** | **1.52× → 1.49×** | small REAL win; gap NARROWS, does NOT close |

### 10.4 Honest disposition — real non-overlapping win, but the §9.5 hypothesis is REFINED
§9.5 hypothesized W1 is "the 1.52× gap-closer" because the 3B decode is not
bandwidth-floored, so removing the eager per-step LAUNCH overhead would win big.
**Measurement REFINES this:** graphing the decode IS a real, statistically-clean win
(−0.77 ms/tok, ~1.25%, non-overlapping bands) — so there genuinely WAS ~0.77 ms/tok of
removable per-step launch overhead — **but it is a small slice, and it does NOT close the
gap** (1.52×→1.49× vs vLLM's 40.8 ms). The residual ~20 ms/tok is therefore **NOT launch
overhead** (the graph removes essentially all of it): it is per-step **compute / kernel
efficiency** — our eager-C++ decode does more GPU work per step than vLLM's torch.compile-
fused + graphed decode. Closing the audio gap needs a decode-kernel-efficiency pass
(nsys our graphed step vs vLLM's, port the divergent kernels 1:1) and/or batched c2+ (W2),
not more graphing. **STRUCTURAL value delivered:** Voxtral's Mistral/Llama stack now HAS a
decode-graph class (the last mm text stack without one) — a prerequisite for batched
multi-seq mm decode (W2). mm rows stay **speed-pending / `PARTIAL`**: correctness-complete,
audio decode now graphed with a small real win, still ~1.49× vs vLLM. No mm row advances
to DONE.

---

## 11. DECODE-KERNEL EFFICIENCY — the ~20 ms residual ATTRIBUTED + a VALIDATED bf16-near-tie ceiling (`CLAIM-MM-SPEED-DECODE-KERN`, 2026-07-27)

**Base:** `origin/main` `bbcaedd0` (the §10 W1 HEAD). **Build:**
`dgx.casa:~/work/mm-audio-kern`, cutlass 4.5.0 + FA2 sm_121a + Triton-AOT, arch 121a (FA2
ENABLED banner CONFIRMED at configure; clean `-Werror` link). All GPU under `flock /tmp/gpu`,
sole owner (`nvidia-smi` idle, `local-ai-worker` absent), cold rep0 dropped. Oracle for
teacher-forcing = `~/venvs/vllm-oracle-v0.25.0-stage` (vLLM **0.25.0** + mistral_common
1.11.5, the golden-capture stack; the symlinked 0.26 `vllm-oracle-next` NOT used).

§10 left the audio decode residual as "per-step COMPUTE/kernel efficiency, not launch
overhead — nsys our graphed step vs vLLM's, port the divergent kernels 1:1." This section
does exactly that. **Result: the residual is ONE kernel (the decode attention), the 1:1 vLLM
port EXISTS and BEATS vLLM, but it is blocked byte-exact by the committed golden — a
fully-characterized, teacher-force-VALIDATED bf16 near-tie ceiling.**

### 11.1 W0 — ATTRIBUTION (measure, don't guess — [[profile-vllm-actual-kernels-port-1to1]])

nsys `cuda_gpu_kern_sum` with `--cuda-graph-trace=node` over the full e2e (32-layer encoder
+ 388-tok prefill + 47 graphed decode steps; decode kernels isolated by instance count
1410 = 30 text layers × 47 steps). The graphed **decode** step breaks down (per step, ×30
layers):

| Decode kernel | Instances | Avg | Per-step (×30) | Note |
|---|---|---|---|---|
| **`vt::cuda::PagedAttentionKernel` (scalar decode attn)** | **1410** | **723 µs** | **21.7 ms** | **THE residual** — ~120× the KV memory floor (~6 µs for 8 kv-heads × ~430 keys × 128 d bf16) |
| `internal::gemvx::kernel` (cuBLAS bf16 GEMV: qkv/o/gate_up/down) | 7050+1410 | 36–204 µs | ~6 ms | near BW floor — cuBLAS, == vLLM's `F.linear` GEMV; NOT the gap |
| `cutlass ...s16816gemm` (lm_head [1,3072]×[3072,131072]) | 48 | 3.38 ms | (per token) | ~BW floor (805 MB read / ~273 GB/s ≈ 2.95 ms); NOT the gap |
| RmsNorm / SiluAndMul / RopeFromCache / ReshapeAndCache glue | 1440–2928 | 2–4 µs | <0.3 ms | negligible; the `vt::FusedChain` Add+RMSNorm already folds the norm glue |

So the decode gap is NOT the GEMMs (cuBLAS `gemvx`, the same family as vLLM's decode
`F.linear`, near-BW-floor), NOT RMSNorm/RoPE/activation glue, and NOT launch overhead
(§10 graphed that away). It is the **decode ATTENTION kernel**: the naive scalar
`PagedAttentionKernel` at **723 µs/call** — one CTA per (query, head) streaming all keys with
a per-key block `__syncthreads` reduction (the exact O(t²)-sync anti-pattern §7 fixed for the
vision tower), catastrophically underutilized at batch=1. vLLM runs decode attention on
`flash_attn_varlen_func` (the flash-attn split-KV decode) — fast, fully SM-filling.

### 11.2 W1 — the 1:1 vLLM lever is ALREADY IN THE BINARY, gated off by a block_size quirk

Voxtral text decode is head_dim 128, GQA 32q/8kv, bf16, causal, no window — which matches the
`fa2_decode_qwen3` dispatch (`cuda_paged_attn.cu:2620`) **exactly**. That path runs
`LaunchDecodeVarlenFA2Bf16` — our vendored 1:1 of vLLM's `flash_attn_varlen_func` split-KV
decode (`flash_fwd_splitkv_kernel`), DEFAULT-ON, and already "bit-matches vLLM's decode OUTPUT
teacher-forced gap 0.0000" for Qwen3-dense (`qwen3-decode-strict-bitmatch.md`). It is the SAME
production decode kernel the 27B (`fa2_decode_r6`) and 35B (`fa2_decode_r8`) paths use.

**Why Voxtral missed it:** `fa2_decode_qwen3` requires `block_size % 16 == 0`, but
`VoxtralGenerateGreedy` allocates ONE big KV block of `block_size = T0 + max_new_tokens + 8 =
444` (for the 388-tok clip), and 444 % 16 = 12 ≠ 0 → decode fell through to the scalar
`PagedAttentionKernel`. (Prefill's `fa2_prefill_qwen3` has no such check, so prefill already
ran FA2 — 30 `flash_fwd_splitkv` instances @ 54 µs.) Rounding the single block up to a multiple
of 16 (`((T0+max_new+8+15)/16)*16`, seq still fits one block, slot == abs_idx unchanged) routes
decode through FA2. nsys of the FA2 arm confirms the swap: **`flash_fwd_splitkv` 1410 @ 18.5 µs
+ combine 1410 @ 3.1 µs = 0.65 ms/step (39× faster attention); zero `PagedAttentionKernel` left
in decode.**

### 11.3 RESULT — same-binary decode TPOT A/B (throwaway `block_size÷16` + `VT_FA2_DECODE_QWEN3` toggle)

Steady-state TPOT (throwaway `steady_clock` around the decode loop, excludes the 2 cold+warm
capture steps, 4 reps/mode, rep0 dropped; instrumentation NOT committed):

| Vehicle | byte-exact NAIVE (ships) | FA2 varlen decode | Δ | vs vLLM 0.25.0 graphed 40.8 ms | `repro` |
|---|---|---|---|---|---|
| **Voxtral-Mini-3B audio (48 tok)** | **59.4 ms/tok** (59.25–59.53) | **38.2 ms/tok** (38.01–38.40) | **−21.2 ms/tok (~36%), NON-OVERLAPPING** | naive 1.46× → **FA2 0.94× — BEATS vLLM** | 48/48 vs **18/48** |

The −21.2 ms A/B delta matches the nsys per-step attribution (21.7 ms `PagedAttentionKernel`)
to within noise: the residual IS the decode-attention kernel, nothing else.

### 11.4 CORRECTNESS — the RED line, and WHY the win cannot ship byte-exact

The gate's binding assertion is `repro == 48`: our 48 tokens must byte-match the committed
near-tie golden `voxtral_neartie.json::our_tokens` (captured on the scalar kernel). FA2 uses a
different f32 reduction order → different bf16 rounding → it flips the golden's SOLE greedy
branch (pos 33 is a **4-way EXACT tie, gap 0.000** — a 1-ULP logit change decides it), so
`repro` drops 48→18 and `strict_prefix` 33→18 → **`repro==48` FAILS**. Per the RED line
("golden md5 unchanged; a single token flip = wrong; fix, don't ratify") this is NOT shippable.

**Is FA2 wrong, or just a different valid branch?** DEFINITIVELY the latter: teacher-forcing
vLLM 0.25.0 on the FA2 sequence (`scripts/mm/a3_voxtral_neartie_gate.py`, `enforce_eager`,
GMU 0.30, under flock) reports **0 divergent positions, worst gap 0.0000 nats, RESULT PASS** —
every one of the 48 FA2 tokens IS vLLM's teacher-forced argmax. The scalar-kernel golden and
the FA2 sequence are BOTH valid vLLM greedy branches (each teacher-forces to gap 0.0); they
differ only in which side of the pos-33 exact tie they take, after which the divergent context
yields different-but-equally-valid continuations. The `strict_prefix` 33→18 is that branch
point moving earlier, NOT a correctness regression.

Byte-exact shipped path re-verified on a clean rebuild of `bbcaedd0`: **`test_voxtral_e2e`
14/14** (strict prefix 33/48, near-tie seq 48/48, worst gap 0.0), goldens md5 UNCHANGED
(`voxtral_golden.json 8ab87b7e…`, `voxtral_neartie.json 3d199c2d…`, before == after).

### 11.5 VERDICT — a fully-characterized, VALIDATED bf16 near-tie ceiling (RECORDS-ONLY)

The audio decode gap is fully attributed and the 1:1 vLLM lever is validated: **routing Voxtral
decode through FA2 varlen (a one-line `block_size÷16`) takes TPOT 59.4→38.2 ms/tok = 0.94× vLLM
— it CLOSES AND BEATS the 1.49× gap — and the resulting sequence is a vLLM-valid greedy branch
(teacher-force PASS, gap 0.0).** But it changes the bf16 near-tie resolution, so it does not
reproduce the committed near-tie golden, which is pinned to the scalar kernel's exact bf16
rounding. This is the [[near-tie-distributional-gate]] / [[dflash-correctness-done-speed-bf16-blocked]]
family: the speed requires a reduction-order change the token-exact gate forbids, and there is
NO byte-exact faster decode-attention kernel (every faster kernel — FA2 or the `PagedAttention
DecodeOpt/Gqa` warp-shuffle kernels — changes the f32 reduction order). The IRREDUCIBLE-under-
byte-exact-gate portion is the full **−21.2 ms/tok** (the win is entirely gated by the golden).

Per the RED line this pass ships **RECORDS-ONLY** — no code change; the byte-exact scalar path
stays default (14/14, golden unchanged, ~1.46–1.49× vLLM). **Reachable follow-on (recommended,
needs user OK on the golden-change policy):** regenerate `voxtral_neartie.json::our_tokens` from
the FA2 sequence + re-run `a3_voxtral_neartie_gate.py` (already PROVEN PASS, gap 0.0), then land
`block_size÷16` — that claims a validated **~36% audio-decode win that BEATS vLLM** and closes
the LAST mm speed gap. mm rows stay **speed-pending / `PARTIAL`**. No mm row advances to DONE.
