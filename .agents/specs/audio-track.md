# AUDIO track — the genuinely-new AUDIO modality (A0–A3)

**Status: A0 + A1 + A2 + A3 LANDED (2026-07-25; A0/A1 `CLAIM-AUDIO-PIPELINE`, A2
`CLAIM-AUDIO-ENCODER`, A3 `CLAIM-AUDIO-E2E`). The AUDIO leg of the #1
Audio/Video/Image priority is CORRECTNESS-COMPLETE end-to-end (audio→text on
Voxtral-Mini-3B). Only A2-follow (USM-Conformer, Gemma-4 family) + A3 speed
remain.** This is
the modality-first, oracle-runnable
path that stands up audio on the smallest native-tower vehicle
(`openai/whisper-small`) before carrying it to Voxtral-Mini-3B (A3) and Gemma-4
(G3). It does NOT depend on Gemma-4 (which is oracle-blocked, see
[gemma4-multimodal.md](gemma4-multimodal.md) §0.0). Audio is the genuinely-new
modality of the user's #1 Audio/Video/Image priority.

**Base:** `origin/main` `435ba70`. **Oracle pin:** `/home/mudler/_git/vllm` @
`e24d1b24`; dgx oracle `~/venvs/vllm-oracle` = vLLM **0.25.0** + transformers
**5.13.1**. **Claim:** `CLAIM-AUDIO-PIPELINE`. **Precedent mirrored:** the landed
image/video mm INPUT pipeline (`ENG-MM-INPUT-PIPELINE`, M0/M1,
[multimodal-track.md](multimodal-track.md)) — audio is another mm modality on the
same modality-agnostic spine (`MultiModalKwargs`/`MultiModalFeatureSpec`/
`MultiModalHasher`/`EncoderCacheManager`).

Vehicle chain (from [gemma4-multimodal.md](gemma4-multimodal.md) §0.4): **Whisper**
(pipeline + first tower, native `WhisperEncoder`, oracle-certain, fits trivially) →
**Voxtral-Mini-3B** (e2e audio-merge on our LANDED Mistral backbone) → Gemma-4 G3
(USM Conformer, family delta, proven on Granite-Speech-2b). **Encoder-family
caveat (honest):** Whisper/Voxtral use a Whisper-class encoder (2×Conv + vanilla
transformer); Gemma-4/gemma3n/Granite use a USM Conformer — so these vehicles
de-risk the audio pipeline + merge pattern fully, but NOT the Gemma-4 Conformer
tower (A2 delta, proven separately).

---

## 0. What LANDED (A0 + A1)

### A0 — ground + vehicle + the audio-processor oracle reference

Vehicle `openai/whisper-small` confirmed oracle-constructible on dgx (transformers
5.13.1 has `WhisperFeatureExtractor` + `WhisperConfig`; feature extraction is CPU,
no GPU/flock). Config (fetched from HF): `n_fft=400`, `hop_length=160`,
`n_mels=80` (`feature_size`), `sampling_rate=16000`, `chunk_length=30` (⇒
`n_samples=480000`, `nb_max_frames=3000`), `dither=0.0`,
`max_source_positions=1500`.

Captured the **vLLM/HF audio-processor reference** for a fixed, deterministic
synthetic clip (`scripts/mm/a0_audio_ref.py`, run in `~/venvs/vllm-oracle`):
- a 3.0 s multi-tone + seeded-noise waveform written as a canonical PCM16 mono
  16 kHz WAV (committed; the C++ side decodes `int16/32768.0` to the identical
  float32 array — zero decode ambiguity, mirroring how M0 picked a 448×448 image so
  smart_resize is identity);
- the WhisperFeatureExtractor log-mel `input_features` `[80, 3000]` — the REAL
  **torch STFT path** (`_torch_extract_fbank_features`, the one that runs when torch
  is installed, hence what the oracle produced);
- the mel filterbank `[201, 80]` (a deterministic config constant, dumped as a
  golden so the C++ STFT float ops are the only parity variable);
- the audio placeholder-token expansion (`[0]` → `[0]*1500`,
  num_audio_tokens = `max_source_positions` = encoder output length =
  3000 frames / 2 conv-stride; `whisper.py:656,740-753`);
- the mm-hash `MultiModalHasher.hash_kwargs(model_id, audio=<f32 waveform>)`.

Committed fixtures `tests/vllm/multimodal/fixtures/whisper_audio/`:
`audio_tone_16k_mono.wav` (sha `b8bdafe4…`), `audio_waveform_f32.bin` (`ea1f2cbd…`),
`input_features_f32.bin` (`980ef280…`), `mel_filters_f32.bin` (`d18b48f3…`),
`manifest.json`. mm-hash golden `2d0c7e4c2d1f3735…`.

### A1 — the C++ audio input pipeline + the inert engine seam

New, additive TU `src/vllm/multimodal/audio_processor.{h,cpp}` (+ `AudioKwargs` in
`include/vllm/multimodal/inputs.h`, + `MultiModalHasher::HashAudioF32` in
`hasher.{h,cpp}`), mirroring the HF `WhisperFeatureExtractor` + `whisper.py`
(`file:line` in the TU headers):
- **WAV decode** (`DecodeWavPcm16Mono`): canonical PCM16 mono RIFF → float32
  (`int16/32768.0`). Byte-identical to the oracle's decoded waveform.
- **Resample**: identity guard at `cfg.sampling_rate` (16 kHz fixture); a genuine
  windowed-sinc resample (librosa-style) is DEFERRED, mirroring the image
  SmartResize/bicubic identity-only deferral.
- **Log-mel `input_features`** (`ProcessWaveform`): pad/truncate to 480000 →
  torch.stft-equivalent (reflect-pad `n_fft/2`, periodic Hann, hop 160, drop the
  last frame) via a direct DFT over the 201 needed bins → `abs(stft)^2` → `mel_filters.T @
  magnitudes` → `log10(clamp 1e-10)` → `x=max(x, x.max()-8)` → `x=(x+4)/4` →
  `[80, 3000]`.
- **Placeholder expansion** (`ExpandAudioPlaceholders`): `[0]` → `[0]*1500`,
  mirroring `ExpandImagePlaceholders`; fills `[offset,length]` spans.
- **mm-hash** (`HashAudioF32`): extends `MultiModalHasher` for a float32 1-D
  ndarray item (dtype `"<f4"`, shape `(N,)`), the audio analogue of the image
  `"|u1"`/`(H,W,3)` byte stream (`hasher.py:110-127`).
- **Engine seam**: `MultiModalFeatureSpec` already carries `modality` (+`mm_hash`
  +`offset`+`length`, modality-agnostic on `Request`/`EngineCoreRequest`) and the
  `EncoderCacheManager`/LMCache `extra_keys` slot from M1; A1 adds a default-null
  `audio_data` pointer to `AudioKwargs`. With no audio input every field is empty
  and the text/image/video paths are byte-identical.

**OpenAI-server audio ingestion is NOT wired here** — the in-tree server does not
yet ingest image/video (`grep image_url src/vllm/entrypoints` = 0), so there is no
in-tree precedent to mirror for `audio_url`/`input_audio`; the seam is the
modality-agnostic `MultiModalFeatureSpec` on the request, and server ingestion is
owed alongside the image/video server wiring (not this claim).

### The A1 gate (PASS)

`tests/vllm/multimodal/test_audio_processor.cpp` vs the A0 oracle fixture
(RED-first):
- **log-mel `input_features` rel-L2 = 1.96e-7** vs the oracle (measured; a TIGHT
  2e-4 band stated — see tolerance justification). Near-bit-exact.
- WAV decode byte-identical (0 mismatches); placeholder expansion `[0]*1500`
  byte-identical; **mm-hash byte-identical** (`2d0c7e4c…`).
- **RED-first (each blows the band):** perturb the largest mel weight →
  rel-L2 2.6e-3; wrong hop (161) → 0.70; skip the `(x+4)/4` normalization → 9.27.
- **77/77 assertions pass; clean CPU `-Werror` (0 warnings).**

**Tolerance justification (rel-L2 2e-4 band, measured 1.96e-7).** The oracle log-mel
is `torch.stft` (an FFT); ours is a direct DFT over the same window / hop /
reflect-pad. The ONLY numeric difference is float summation order (FFT butterfly vs
DFT accumulate), a well-below-bf16 effect — transformers itself only claims its
torch-vs-numpy STFT agree to 1e-5, and we sit two orders tighter. Bit-exact is
infeasible (different FFT algorithm) but the measured 1.96e-7 is near-bit-exact; the
band has ~1000× headroom over the measurement and every real bug (filterbank / hop /
window / normalization) is >1e-2 (asserted). The ids + mm-hash stay bit/byte-exact.

**Inertness (proven).** `git diff --stat` = additive: new
`src/vllm/multimodal/audio_processor.{cpp,h}` + test + fixtures + script, plus a
new hasher method (shared `hasher.cpp` refactor extracts a `FinalizeHex` helper —
byte-identical behavior) and a default-null `audio_data` field on the shared
`inputs.h` `MultiModalFeatureSpec`. No shared forward / kernel / runner / registry
edit. Because `hasher.cpp`/`inputs.h` are shared TUs, the fast CPU mm + seam gates
were re-run and are byte-identical: image processor 23/23 (proves the hasher
refactor is inert — image mm-hash unchanged), video processor 41/41, request 71/71,
encoder-cache 32/32, text backbone 85/85. No CUDA kernel added ⇒ CPU-only, no
compute-sanitizer needed; `check-device-leakage` unchanged (32 == baseline).

---

## 0b. What LANDED (A2 — the Whisper-class encoder tower, faithful in isolation)

Row `ENG-MM-AUDIO-ENCODER` (engine-matrix, `ACTIVE`, owner `CLAIM-AUDIO-ENCODER`).


New additive TU `include/vllm/model_executor/models/whisper_audio.{h,cpp}` +
`src/…/whisper_audio.cpp`, mirroring transformers `WhisperEncoder`
(`modeling_whisper.py` @ 5.13.1: `WhisperEncoder.forward:641-721`,
`WhisperEncoderLayer.forward:400-430`, `WhisperAttention.forward:298-368`,
`sinusoids:54`; cross-checked vs the faithful vLLM port `whisper.py:458,353,322,
473-476`). Consumes the A1 log-mel `input_features` `[80,3000]`, produces encoder
hidden states `[1500,768]`:
- **conv frontend as im2col + `vt::MatmulBT` (no new CUDA kernel):** Conv1d(80→768,
  k3,pad1,stride1) + GELU-erf → Conv1d(768→768,k3,pad1,**stride2**, halving
  3000→1500) + GELU-erf; the stride-2 matmul emits the already-transposed `[1500,
  768]`. Whisper conv is a FULL cross-channel conv, NOT the depthwise
  `vt::CausalConv1d` (Mamba/GDN), so it is expressed as im2col + the existing GEMM.
- **fixed sinusoidal `embed_positions.weight [1500,768]`** added — dumped as a
  GOLDEN CONSTANT (`enc_embed_positions_f32.bin`, like A1's mel filterbank) so the
  encoder-block math is the parity variable.
- **12 pre-norm encoder blocks:** `self_attn_layer_norm` → q(bias)/k(**NO bias**)/
  v(bias) → full **bidirectional** `vt::Attention(causal=false)` (scale=head_dim⁻⁰·⁵,
  q pre-scaling folded into the call as transformers does) → out_proj → residual →
  `final_layer_norm` → MLP(fc1 768→3072, GELU-erf, fc2 3072→768) → residual.
- **final `layer_norm`** → `[1500,768]`. All GEMMs bf16, norm/softmax accumulate
  f32 (production dtype, matches vLLM's bf16 encoder + the M2a methodology).
- **weight loader:** the C++ unit test loads `encoder.*` (conv/attn/mlp/LN) from a
  dumped f32 dir (`scripts/mm/a2_audio_encoder_weight_dump.py`, uncommitted ~130
  MiB) rounded fp32→bf16 on upload; whisper-small ships fp32, cast to bf16
  (identical rounding both sides).

**Delta from the M2a vision tower:** NO patch-merger, NO DeepStack, NO RoPE (fully
bidirectional, a fixed additive sinusoid instead of positional rotation); a CONV
frontend rather than a patchify matmul; GELU-erf everywhere (vision used tanh-GELU
in the MLP); LN names `self_attn_layer_norm`/`final_layer_norm`. Same Buf/UpBf16/
LinearBias scaffold and unit-gate methodology.

### The A2 gate (PASS) — encoder-tower fidelity vs the dumped oracle

`tests/vllm/multimodal/test_whisper_audio.cpp` runs the C++ tower on the committed
A1 `input_features` golden and compares 4 per-stage taps vs the bf16 transformers
5.13.1 `WhisperEncoder` reference (`scripts/mm/a2_audio_encoder_ref.py`, dumped on
CPU), **on the GPU under `flock` on a cutlass-ON build** (banner CONFIRMED:
cutlass-nvfp4/fp8 + FA2 + Triton-AOT ENABLED for sm_121a), the sibling 27B NOT
co-resident:
- **post_conv rel-L2 = 0.00473** (maxabs 0.039) — the conv frontend, TIGHT;
- **post_pos rel-L2 = 0.00280** (after + sinusoid);
- **block0 rel-L2 = 0.00660** — one full encoder block, TIGHT;
- **final_ln (the `[1500,768]` encoder output) rel-L2 = 0.03049** (maxabs 6.44).
- **203/203 assertions pass; clean CUDA + CPU `-Werror` (0 warnings).**

**Tolerance (measured bf16-depth envelope, bands post_conv/post_pos <8e-3, block0
<1.5e-2, final <5e-2 = measured × ~1.6–2.3).** The conv frontend + first block match
the transformers bf16 reference TIGHTLY (<0.7%), proving the im2col+MatmulBT conv,
the sinusoid add, and the pre-norm block are correct; the final output diverges only
by SMOOTH bf16 accumulation across the 12 layers between two independent bf16 kernel
stacks (our `vt` ops vs transformers' bf16 conv/linear/SDPA), the same ~0.28%/layer
the M2a vision tower measured (there ~0.25%/layer). No discontinuity ⇒ numerical, not
a logic error. Bit-exact is infeasible (different GEMM/attn kernels); token-exact e2e
is the ultimate bar at A3.

**RED-first (the gate bites).** Each structural bug drives a stage FAR past its band
(GPU under `flock`, revert-experiment):
- **wrong conv-stride** (conv2 stride 1 instead of 2 → wrong down-sample): post_conv
  **0.34** (72× the 4.7e-3 band), FAILURE — the conv frontend / stride-2 halving is
  gate-discriminated (and it also sets num_audio_tokens=1500 = A1's placeholder count);
- **missing sinusoidal pos** (skip the `embed_positions` add): post_pos **0.86**,
  block0 **1.53**, final **1.08**, FAILURE;
- **skipped final `layer_norm`** (return pre-LN hidden): final **4.22** (140× the
  5e-2 band), FAILURE.
- Honest non-discriminators (recorded, not loosened): **GELU-tanh vs GELU-erf** stays
  IN the envelope (final 0.0343 vs 0.0305) — the tanh approximation agrees with erf
  BELOW the bf16-depth envelope, so this axis is not gate-discriminated (we use the
  faithful erf regardless, per the reference `nn.functional.gelu`); a **single
  conv1-weight ×3** perturbation is aggregate-insensitive (one of 184 320 conv1
  weights across 1500×768 outputs) — the STRIDE RED is the conv-path discriminator.

**Inertness (proven).** `git diff --stat` vs `adcac8e` = 7 ADDITIVE entries (new
`whisper_audio.{h,cpp}` + test + 2 dump scripts, +1 `CMakeLists.txt` source line, +10
`tests/CMakeLists.txt` lines) plus the committed reference fixtures. NO shared forward
/ kernel / runner / registry / other-model TU edited ⇒ the SACRED text gates + the
landed image/video/audio-pipeline gates are byte-identical BY CONSTRUCTION. `check-
device-leakage` unchanged. No CUDA kernel added (im2col + existing GEMM) ⇒ no
compute-sanitizer run needed.

**USM-Conformer delta (A2-follow, NOT built here).** Gemma-4/gemma3n/Granite use a
USM Conformer audio tower, NOT this Whisper-class encoder: Conv2d stride-2 semicausal
SUBSAMPLING (not 2×Conv1d), Conformer blocks (conv-module + macaron FFN),
RELATIVE-position attention, and attention softcap. That is a genuinely-different
tower — to be proven separately on **Granite-Speech-2b** as an A2-follow sub-item
(new kernels get compute-sanitizer 0). A1/A2 de-risk the Whisper-class pipeline +
tower fully, NOT the Conformer.

---

## 0c. What LANDED (A3 — the FIRST e2e AUDIO→TEXT understanding)

Row `ENG-MM-AUDIO-E2E` (engine-matrix, `ACTIVE`, owner `CLAIM-AUDIO-E2E`) + the
`MODEL-MM-voxtral-voxtral-for-conditional-generation` model row advanced
`INVENTORIED`→`ACTIVE`.

**Vehicle + oracle (verdict: downloadable + oracle-runnable).**
`mistralai/Voxtral-Mini-3B-2507` is NOT HF-gated (`model_info().gated == False`);
downloaded the `load_format=mistral` consolidated.safetensors (8.8 GiB, 761 tensors)
onto dgx. The pinned oracle vLLM 0.25.0 (+ mistral_common 1.11.5 + `soundfile`)
CONSTRUCTS and RUNS the audio→text path (`config_format=mistral`,
`load_format=mistral`, `tokenizer_mode=mistral`). Golden captured for a deterministic
30 s clip + "Describe what you hear" prompt (`scripts/mm/a3_voxtral_oracle_capture.py`):
vLLM greedy is **K=5 self-DETERMINISTIC** ⇒ gate form = STRICT (the near-tie-robust
fallback then applies only for our different-kernel encoder). Golden = 48 tokens, a
real description ("The audio begins with a series of sustained, low-pitched hums or
drones…"). Fixtures committed (WAV, log-mel/mel/sinusoid goldens, prompt+golden
tokens, near-tie result), content-hashed.

**The forward (`voxtral.{h,cpp}`, additive).** Encoder = the A2
`WhisperAudioEncoderForward` instantiated at Voxtral's Whisper-large-v3 encoder
config (d_model 1280 / 32 layers / 20 heads / head_dim 64 / ffn 5120 / 128 mel bins /
1500 src-pos — the A2 tower is fully config-driven, reused verbatim; k_proj no-bias
as A2; sinusoid computed via `transformers.sinusoids`, dumped golden). Projector =
`AudioLanguageAdapter` (downsample-concat reshape `[1500,1280]`→`[375,5120]` factor 4,
then `w_in`→`nn.GELU()`(erf)→`w_out`, NO bias — voxtral.py:382-412,660-668). Merge =
`Qwen3VLMergeMultimodal` (modality-agnostic masked-scatter) into the 375 audio-token
(id 24) rows. Decode = `VoxtralGenerateGreedy` over the LANDED shared dense forward
(`dense_attn::AttnBlock`, qk-norm-optional/1-D NeoX rope/GQA/paged FA2) with the ONLY
forks being inputs_embeds start + untied lm_head. Weight loader reads the mistral
consolidated names (`mm_whisper_embeddings.*` encoder/projector/embed, `layers.N.*`
Mistral, `output`=lm_head).

**The decisive bug (RED evidence).** The mistral-CONSOLIDATED q/k weights are stored
in the Meta-interleaved rope layout; vLLM applies a Meta→HF-NeoX row PERMUTE on the
mistral load path (verified bit-exact: `permute(wq)==vLLM q_proj`,
`scripts/mm/a3_voxtral_wcheck.py`). Loading them raw gave a WRONG decoder — text-only
1/22, e2e 0/48. Adding `PermuteQKBf16` (q 32 heads, k 8 heads; v/o raw) fixed it:
text-only 22/22, first audio token exact.

### The A3 gate (PASS 14/14) — e2e audio→text vs vLLM 0.25.0

`tests/vllm/multimodal/test_voxtral_e2e.cpp`, GPU under `flock` on a cutlass-ON build
(banner CONFIRMED), sibling 27B NOT co-resident, `VLLM_VOXTRAL_SAFETENSORS`→the
consolidated checkpoint:
- **log-mel** (A1 processor at Voxtral 128-mel config) rel-L2 **7.7e-7** vs oracle;
  **375** audio-embed rows (1500/4).
- **STRICT prefix 33/48** exact vs the vLLM greedy golden (the pipeline reproduces
  vLLM greedy token-for-token up to the first bf16 near-tie).
- **Decoder proven token-exact:** feeding vLLM's EXACT audio embeddings into our
  decoder → **48/48**. So the ONLY residual is our audio ENCODER's bf16-kernel
  divergence (encoder rel-L2 8.7% = the A2 ~0.28%/layer envelope over 32 layers;
  vLLM's encoder is bf16 too — bit-exact across different GEMM/attn kernels is
  infeasible, A2 §tolerance).
- **Binding gate = the ratified near-tie-robust gate** (exactly as M3c/M3d):
  teacher-force vLLM on OUR 48-token sequence — **worst gap 0.0 nats, 0 over-band
  failures**; the SOLE greedy branch point (pos 33) is a **4-way EXACT bf16 tie** at
  -2.069 nats (our token, `19246`, `7759`, and the golden `3401` all at -2.0685), and
  every one of our 48 tokens is vLLM's teacher-forced argmax. `voxtral_neartie.json`
  committed; `scripts/mm/a3_voxtral_neartie_gate.py`.

**Inertness (proven).** `git diff --stat` = 2 modified lines only (`CMakeLists.txt`
+1 source line, `tests/CMakeLists.txt` +test wiring) + new additive files. NO shared
forward / kernel / runner / registry / other-model TU edited ⇒ byte-identical by
construction; re-ran **Mistral text 541/541**, **A1 77/77**, **A2 203/203**. Clean
CUDA `-Werror` 0 warn. NO new CUDA kernel (reuses A2 im2col+GEMM + the merge scatter)
⇒ compute-sanitizer not required; `check-device-leakage` unchanged. `benchmark_binding
=false`, speed pending (a target is DONE only at token-exact AND vLLM throughput).

---

## 1. REMAINING (A2-follow + A3 speed)

- **A2-follow — USM Conformer tower** (Granite-Speech-2b), the Gemma-4-family audio
  tower delta (Conv2d stride-2 semicausal subsample + Conformer conv-module +
  relative-position attention + softcap). New kernels get compute-sanitizer 0. GPU
  under `flock`. (Scoped above; not built.)
- **A3 speed** — the audio→text throughput grid vs vLLM 0.25.0 (every-axis), the
  remaining bar before A3 is `DONE` (correctness is complete).

---

## 2. Structured contract

### Scope
IN: A0 (vehicle + oracle references + fixtures + capture script) and A1 (the C++
audio INPUT pipeline: WAV decode + identity resample + log-mel + placeholder
expansion + mm-hash + the inert engine seam), gated by feature-parity + inertness.
Owns `ENG-MM-AUDIO-PIPELINE` (engine-matrix, `ACTIVE`), the new
`src/vllm/multimodal/audio_processor.{h,cpp}`, the `AudioKwargs`+`HashAudioF32`
additions, `tests/vllm/multimodal/test_audio_processor.cpp`, the fixtures, and
`scripts/mm/a0_audio_ref.py`.

OUT (each with a reason): **the audio ENCODER tower** (A2) and **the e2e
audio→text** (A3) — not built; A1 delivers the processed audio features + the seam
only. **Genuine (non-identity) resample** — deferred (windowed-sinc), fixture is
16 kHz. **OpenAI-server audio_url ingestion** — no in-tree image/video server
precedent to mirror; owed with the image/video server wiring. **Gemma-4 audio (G3)**
— oracle-blocked ([gemma4-multimodal.md](gemma4-multimodal.md) §0.0); reuses this
pipeline when it clears. **The Qwen3.6 mm rows + `multimodal-track.md` + the Gemma-4
`MODEL-MM-*` rows** — owned by the concurrent Qwen3.6-video / Gemma-4 agents; not
touched. **Whisper/Voxtral MODEL-MM rows** stay `INVENTORIED` (this is the pipeline,
not a full model port).

### Upstream chain
transformers `feature_extraction_whisper.py` (`_torch_extract_fbank_features`,
`__init__` mel params), `audio_utils.mel_filter_bank` (slaney/slaney);
`vllm/model_executor/models/whisper.py:{103,458,469-476,656,738,740-753}`
(WhisperEncoder conv stride, `get_num_audio_tokens`, `_get_prompt_updates`);
`vllm/multimodal/hasher.py:{50,108-127}` + `processing/inputs.py::get_mm_hashes`.
Landed mm spine anchors ([gemma4-multimodal.md](gemma4-multimodal.md) §0.2).
**Anchor-drift warning:** re-anchor every `file:line` at implementation time.

### Our baseline
REUSE (landed): the modality-agnostic mm spine — `MultiModalKwargs`/
`MultiModalFeatureSpec` (`include/vllm/multimodal/inputs.h`), `MultiModalHasher`
(`src/vllm/multimodal/hasher.cpp`), `EncoderCacheManager`
(`src/vllm/v1/core/encoder_cache_manager.cpp`), the LMCache `extra_keys` slot, and
the `ExpandImagePlaceholders` expansion pattern. NEW (this claim): the whole audio
INPUT pipeline `src/vllm/multimodal/audio_processor.{h,cpp}` (WAV decode, log-mel
STFT, mel projection, placeholder expansion), the `AudioKwargs` feature-tensor kind,
and `MultiModalHasher::HashAudioF32`. NEW (owed, A2/A3): the audio encoder tower
(Whisper-class then USM Conformer) and the projector-merge decode. No
`ENG-MM-AUDIO-PIPELINE` sub-part is or becomes `DONE` until token-exact AND vLLM
throughput; A1 is `ACTIVE` (feature-parity + inertness proven, speed pending).

### Port map
| Piece | Upstream (from) | Ours |
|---|---|---|
| WAV decode → f32 | canonical PCM16 RIFF (`int16/32768.0`) | `audio_processor.cpp::DecodeWavPcm16Mono` |
| log-mel `input_features` | transformers `feature_extraction_whisper.py::_torch_extract_fbank_features` (torch.stft, reflect-pad `n_fft/2`, periodic Hann, hop 160, drop last frame, `abs(stft)^2`, `mel_filters.T@mag`, `log10(clamp 1e-10)`, `max(x,x.max()-8)`, `(x+4)/4`) | `audio_processor.cpp::WhisperAudioProcessor::ProcessWaveform` (direct DFT over 201 bins) |
| mel filterbank `[201,80]` | `audio_utils.mel_filter_bank` (slaney/slaney) | dumped golden constant, loaded (construction deferred) |
| placeholder expansion `[0]`→`[0]*1500` | `whisper.py:656` (`get_num_audio_tokens`=`max_source_positions`), `:740-753` (`_get_prompt_updates`) | `audio_processor.cpp::ExpandAudioPlaceholders` |
| mm-hash (float32 ndarray) | `hasher.py:108-127` + `processing/inputs.py::get_mm_hashes` | `hasher.cpp::MultiModalHasher::HashAudioF32` |
| engine seam | M1 `MultiModalFeatureSpec` (`modality`), `EncoderCacheManager`, `extra_keys` | REUSE + default-null `AudioKwargs audio_data` |

### Tests to port
| Upstream test | Tier | Ours |
|---|---|---|
| whisper feature-extractor / processor parity | T-unit | log-mel `input_features` rel-L2 + placeholder ids + mm-hash (A1, LANDED) |
| `tests/models/multimodal/generation/test_*whisper*` / voxtral | T-e2e | audio→text token-exact (A2 tower, A3 e2e) — owed |

### Gates
1. **Inertness (SACRED).** No audio input ⇒ text+image+video byte-identical.
   PROVEN (additive; shared-TU gates re-run byte-identical).
2. **Audio feature parity (A1).** log-mel rel-L2 within the stated band + ids +
   mm-hash bit/byte-exact, RED-first. PASS (rel-L2 1.96e-7, 77/77).
3. **Audio tower fidelity (A2)** — PASS. Encoder-tower per-stage rel-L2 within the
   measured bf16-depth envelope vs the dumped oracle (post_conv 4.7e-3, block0
   6.6e-3, encoder-output 3.0e-2), RED-first (stride/pos/final-LN blow it), GPU
   under flock on a cutlass-ON build. (USM Conformer A2-follow owed.)
4. **Audio e2e (A3, SACRED)** — PASS. Full C++ pipeline (log-mel→A2 encoder at
   Voxtral config→downsample4→AudioLanguageAdapter→merge→Mistral greedy) vs the vLLM
   0.25.0 golden. Gate form BY MEASUREMENT: vLLM greedy K=5 DETERMINISTIC ⇒ STRICT
   is the bar; STRICT prefix 33/48 exact, then (bit-exact infeasible — encoder uses
   different bf16 GEMM/attn kernels than vLLM's cuBLASLt+FLASH_ATTN) the ratified
   near-tie-robust gate PASSES (teacher-forced worst gap 0.0 nats, sole branch a
   4-way bf16 tie). Decoder proven token-exact (ref-audio 48/48). RED-first: the
   mistral q/k rope-permute bug drove text-only 1/22 & e2e 0/48 → fixed. GPU under
   flock, cutlass-ON.
5. **Build/records.** Clean CPU `-Werror` 0 warn; `check-device-leakage`
   unchanged; the record checkers green. (No CUDA kernel added.)
6. **SPEED** — PENDING (`benchmark_binding=false`); a target is `DONE` only at
   token-exact AND vLLM throughput.

### Dependencies
No hard upward dependency for A0/A1 (the mm spine is landed; feature extraction is
CPU, oracle in `~/venvs/vllm-oracle` on dgx). A3 depends on the LANDED Mistral
backbone (Voxtral decoder) + A2 (the encoder tower). Checkpoint deps (A0 fetched
whisper-small config only; A2/A3 NOT performed): `whisper-small` ~0.5 GiB,
`Voxtral-Mini-3B` ~9.4 GiB, Granite-Speech-2b ~5 GiB — stage sequentially (dgx disk
tight). Downward dep introduced: this audio pipeline is reusable by the whole audio
family incl. Gemma-4 (G3, once its oracle block clears).

### Work breakdown
| Step | State | Gate |
|---|---|---|
| A0 — vehicle + oracle reference (whisper-small; log-mel/mel/placeholder/mm-hash goldens + capture script) | **DONE** | references produced, content-hashed |
| A1 — C++ audio INPUT pipeline + inert seam (decode/resample/log-mel/expansion/mm-hash) | **DONE** | feature-parity 77/77 (rel-L2 1.96e-7, ids+hash bit/byte-exact) + inertness |
| A2 — audio encoder TOWER (Whisper-class, `whisper-small`) | **DONE** | tower fidelity gate PASS 203/203 (post_conv 4.7e-3, block0 6.6e-3, encoder-output 3.0e-2, bf16 envelope; RED-first stride/pos/final-LN blow it; GPU under flock, cutlass-ON) |
| A2-follow — USM Conformer tower (Granite-Speech-2b) | OWED | Conformer tower rel-L2 within bf16 envelope vs oracle dump, RED-first; compute-sanitizer 0 on new kernels (Conv2d subsample, conv-module, rel-pos) |
| A3 — e2e audio→text on Voxtral-Mini-3B (encoder + projector-merge into LANDED Mistral) | **DONE** | e2e gate PASS 14/14: STRICT prefix 33/48 vs vLLM greedy + ratified near-tie-robust gate PASS (worst gap 0.0 nats); decoder token-exact (ref-audio 48/48); inert (Mistral 541/541 + A1 77/77 + A2 203/203). Speed pending |

### Risks/decisions
- **D1 — log-mel is a silent-corruption hazard.** Gated against the oracle
  bit/near-exact, never eyeballed; RED-first proves the gate bites (D3 of the spike).
- **D2 — torch vs numpy STFT.** The oracle uses the torch path (torch installed);
  our DFT targets THAT (measured 1.96e-7), not the numpy path. Recorded so a future
  oracle-env change (torch absent → numpy path) re-captures the reference.
- **D3 — mel filterbank dumped as a golden constant.** Construction-from-config
  (slaney hertz↔mel) is a deferred port; the dumped matrix isolates the STFT as the
  parity variable (mirrors the image path loading config constants).
- **D4 — Whisper-class ≠ Conformer.** A1 de-risks the pipeline + merge pattern, NOT
  the Gemma-4 USM Conformer tower (A2 delta, proven on Granite-Speech-2b).
