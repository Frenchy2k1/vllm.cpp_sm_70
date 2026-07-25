# AUDIO track — the genuinely-new AUDIO modality (A0–A3)

**Status: A0 + A1 LANDED (2026-07-25, `CLAIM-AUDIO-PIPELINE`). A2 (encoder tower)
+ A3 (e2e audio→text) REMAINING.** This is the modality-first, oracle-runnable
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

## 1. REMAINING (A2 + A3)

- **A2 — AUDIO encoder TOWER.** Whisper-class encoder first (2×Conv1d + transformer;
  reuse `vt::` GEMM/attn/LayerNorm/GELU), proven faithful in isolation vs a dumped
  oracle reference (mirror M2a, bf16-envelope tol). Then the USM Conformer delta
  (Conv2d stride-2 semicausal subsample + conv-module + relative-position attention
  + softcap), proven on Granite-Speech-2b — the Gemma-4-family tower. New kernels
  (Conv2d subsample, Conformer conv-module, relative-position bias) get
  compute-sanitizer 0. GPU under `flock`.
- **A3 — e2e AUDIO→text gate.** `Voxtral-Mini-3B`: native Whisper-class encoder
  (A2) + projector (RMSNorm+Linear, `vt::` ops) + masked-scatter merge (REUSE the
  `VLGenerateCore`/`Qwen3VLMergeMultimodal` pattern) into the LANDED Mistral decoder
  → forked greedy decode. Gate: audio→text token-exact vs vLLM 0.25.0, gate form BY
  MEASUREMENT (K=5 self-determinism → STRICT else near-tie). GPU under `flock`,
  memory-careful (~9.4 GiB, never OOM-reboot GB10).

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
3. **Audio tower fidelity (A2)** — owed.
4. **Audio e2e (A3, SACRED)** — owed.
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
| A2 — audio encoder TOWER (Whisper-class → USM Conformer on Granite-Speech-2b) | OWED | tower rel-L2 within bf16 envelope vs oracle dump, RED-first; compute-sanitizer 0 on new kernels |
| A3 — e2e audio→text on Voxtral-Mini-3B (encoder + projector-merge into LANDED Mistral) | OWED | audio→text token-exact vs vLLM 0.25.0, gate form by measurement |

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
