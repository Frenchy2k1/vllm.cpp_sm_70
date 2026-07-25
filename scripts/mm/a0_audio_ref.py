#!/usr/bin/env python3
"""A0 oracle reference capture for the AUDIO input pipeline (audio-track A0/A1).

Drives the REAL HF/vLLM audio processor for the pipeline vehicle
`openai/whisper-small` (244M, native vLLM WhisperEncoder, oracle-constructible on
the pinned oracle = vLLM 0.25.0 + transformers 5.13.1) and dumps the GOLDEN
fixtures that the C++ A1 audio processor must reproduce:

  - a DETERMINISTIC synthetic audio clip written as a canonical PCM16 mono 16 kHz
    WAV (committed; zero decode ambiguity — the C++ side decodes int16/32768.0 to
    the identical float32 array the oracle sees)
  - the WhisperFeatureExtractor log-mel `input_features` [n_mels=80, n_frames=3000]
    (the REAL torch STFT path — torch is installed, so the extractor uses
    `_torch_extract_fbank_features`, which is what actually runs)
  - the mel filterbank matrix [201, 80] (a deterministic config constant, dumped as
    a golden so the C++ side loads the identical projection and the STFT float ops
    are the only parity variable)
  - the audio placeholder-token expansion: Whisper replaces the single encoder
    placeholder `[0]` with `[0] * num_audio_tokens`, num_audio_tokens =
    config.max_source_positions = 1500 (the encoder output length: 3000 mel frames
    -> conv1(stride1) -> conv2(stride2) = /2 = 1500), per
    vllm/model_executor/models/whisper.py:656,740-753 @ e24d1b24
  - the mm-hash: MultiModalHasher.hash_kwargs(model_id, audio=<f32 waveform>) — the
    exact byte stream vLLM's get_mm_hashes produces for a pre-resampled mono f32
    audio item (vllm/multimodal/processing/inputs.py get_mm_hashes)
  - a manifest with the full processor config, shapes/dtypes, and content sha256s

Feature extraction is CPU: NO GPU / NO flock needed. Run in the oracle venv:
  ~/venvs/vllm-oracle/bin/python scripts/mm/a0_audio_ref.py
Fixtures land in ~/audio_fixture, then are copied into
tests/vllm/multimodal/fixtures/whisper_audio/.
"""
import hashlib
import json
import os
import struct
import wave

import numpy as np

from transformers import WhisperConfig, WhisperFeatureExtractor
from vllm.multimodal.hasher import MultiModalHasher

MODEL = "openai/whisper-small"
OUT = os.path.expanduser("~/audio_fixture")
os.makedirs(OUT, exist_ok=True)

SR = 16000
DURATION_S = 3.0
N_SAMPLES = int(round(SR * DURATION_S))  # 48000


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------- deterministic synthetic clip
# A fixed multi-tone + seeded low-amplitude noise waveform: non-trivial spectral
# content across the mel bands (so a wrong filterbank / hop / normalization is
# caught) yet fully deterministic. float64 -> quantized to int16 -> canonical WAV.
t = np.arange(N_SAMPLES, dtype=np.float64) / SR
sig = (
    0.50 * np.sin(2.0 * np.pi * 220.0 * t)
    + 0.30 * np.sin(2.0 * np.pi * 440.0 * t)
    + 0.15 * np.sin(2.0 * np.pi * 1760.0 * t)
    + 0.05 * np.sin(2.0 * np.pi * 3520.0 * t)
)
rng = np.random.RandomState(0xA0D10)
sig = sig + 0.02 * rng.standard_normal(N_SAMPLES)
sig = sig / np.max(np.abs(sig)) * 0.8  # peak 0.8, avoid int16 clip
pcm16 = np.round(sig * 32768.0).astype(np.int32)
pcm16 = np.clip(pcm16, -32768, 32767).astype(np.int16)

wav_path = os.path.join(OUT, "audio_tone_16k_mono.wav")
with wave.open(wav_path, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)  # int16
    w.setframerate(SR)
    w.writeframes(pcm16.tobytes())

# Decode exactly as the C++ WAV reader will: int16 -> float32 / 32768.0. This is
# the model-input waveform (fixture is already 16 kHz mono => resample is identity).
with wave.open(wav_path, "rb") as r:
    assert r.getnchannels() == 1 and r.getsampwidth() == 2 and r.getframerate() == SR
    n = r.getnframes()
    raw = r.readframes(n)
decoded_i16 = np.frombuffer(raw, dtype="<i2")
assert np.array_equal(decoded_i16, pcm16), "WAV round-trip must be exact"
audio_f32 = (decoded_i16.astype(np.float32)) / 32768.0
assert audio_f32.dtype == np.float32 and audio_f32.flags.c_contiguous
audio_f32.tofile(os.path.join(OUT, "audio_waveform_f32.bin"))

# ---------------------------------------------------------------- feature extractor
fe = WhisperFeatureExtractor.from_pretrained(MODEL)
cfg = WhisperConfig.from_pretrained(MODEL)
num_audio_tokens = int(cfg.max_source_positions)  # 1500

feat = fe(audio_f32, sampling_rate=SR, return_tensors="np")
input_features = np.asarray(feat["input_features"], dtype=np.float32)  # [1,80,3000]
assert input_features.ndim == 3 and input_features.shape[0] == 1
input_features = np.ascontiguousarray(input_features[0])  # [80,3000]
input_features.tofile(os.path.join(OUT, "input_features_f32.bin"))

mel_filters = np.ascontiguousarray(np.asarray(fe.mel_filters, dtype=np.float32))  # [201,80]
mel_filters.tofile(os.path.join(OUT, "mel_filters_f32.bin"))

# ------------------------------------------------------- placeholder expansion (ids)
# Whisper: PromptReplacement(target=[0], replacement=[0]*num_audio_tokens)
# (whisper.py:740-753). The encoder placeholder id is 0. Pre-expansion encoder
# prompt = [0]; expanded = [0]*num_audio_tokens.
AUDIO_PLACEHOLDER_ID = 0
pre_expansion_ids = [AUDIO_PLACEHOLDER_ID]
expanded_ids = [AUDIO_PLACEHOLDER_ID] * num_audio_tokens

# ------------------------------------------------------------------------- mm-hash
mm_hash = MultiModalHasher.hash_kwargs(model_id=MODEL, audio=audio_f32)

manifest = {
    "model_id": MODEL,
    "vehicle": "openai/whisper-small (native WhisperEncoder; oracle-constructible)",
    "audio": {
        "wav_file": "audio_tone_16k_mono.wav",
        "waveform_file": "audio_waveform_f32.bin",
        "n_samples": int(N_SAMPLES),
        "sampling_rate": SR,
        "channels": 1,
        "duration_s": DURATION_S,
        "dtype": "float32",
        "decode_contract": "int16_le / 32768.0 (canonical PCM16 mono WAV)",
        "waveform_sha256": sha256_hex(audio_f32.tobytes()),
        "wav_sha256": sha256_hex(open(wav_path, "rb").read()),
    },
    "input_features": {
        "shape": list(input_features.shape),  # [80, 3000]
        "dtype": "float32",
        "note": "WhisperFeatureExtractor torch STFT log-mel (the path that runs "
        "when torch is installed); padded/truncated to chunk_length*sr=480000 "
        "samples => 3000 frames",
        "sha256": sha256_hex(input_features.tobytes()),
        "file": "input_features_f32.bin",
    },
    "mel_filters": {
        "shape": list(mel_filters.shape),  # [201, 80]
        "dtype": "float32",
        "note": "mel_filter_bank(num_frequency_bins=201, num_mel_filters=80, "
        "min=0, max=8000, sr=16000, norm=slaney, mel_scale=slaney); dumped as a "
        "golden config constant (the C++ side loads it; STFT is the parity var)",
        "sha256": sha256_hex(mel_filters.tobytes()),
        "file": "mel_filters_f32.bin",
    },
    "feature_contract": {
        "n_fft": int(fe.n_fft),
        "hop_length": int(fe.hop_length),
        "n_mels": int(fe.feature_size),
        "sampling_rate": int(fe.sampling_rate),
        "chunk_length_s": int(fe.chunk_length),
        "n_samples_padded": int(fe.n_samples),  # 480000
        "nb_max_frames": int(fe.nb_max_frames),  # 3000
        "dither": float(fe.dither),  # 0.0
        "window": "hann (torch.hann_window(400), periodic)",
        "stft": "torch.stft(n_fft=400, hop=160, center=True, pad=reflect, "
        "return_complex=True); magnitudes=|stft[...,:-1]|**2 (drop last frame)",
        "log_mel": "log10(clamp(mel_spec, 1e-10)); "
        "x=max(x, x.max()-8.0); x=(x+4.0)/4.0",
        "max_source_positions": num_audio_tokens,
    },
    "placeholder": {
        "audio_placeholder_id": AUDIO_PLACEHOLDER_ID,
        "num_audio_tokens": num_audio_tokens,
        "arithmetic": "num_audio_tokens = max_source_positions = 1500 = "
        "n_frames(3000) / total_conv_stride(2)",
        "pre_expansion_token_ids": pre_expansion_ids,
        "expanded_len": len(expanded_ids),
        "upstream": "vllm/model_executor/models/whisper.py:656,740-753 @ e24d1b24",
    },
    "mm_hash": mm_hash,
    "mm_hash_contract": "MultiModalHasher.hash_kwargs(model_id=<utf8>, "
    "audio=<f32 c-contiguous 1-D ndarray>); sorted kwargs => 'audio' < 'model_id'",
}

with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print("=== A0 AUDIO ORACLE CAPTURE OK ===")
print("model_id            :", MODEL)
print("wav sha256          :", manifest["audio"]["wav_sha256"])
print("waveform sha256     :", manifest["audio"]["waveform_sha256"])
print("input_features      :", input_features.shape, input_features.dtype,
      "sha256", manifest["input_features"]["sha256"])
print("mel_filters         :", mel_filters.shape, "sha256",
      manifest["mel_filters"]["sha256"])
print("num_audio_tokens    :", num_audio_tokens)
print("mm_hash             :", mm_hash)
print("input_features[0,:5]:", input_features[0, :5].tolist())
print("input_features[:,0][:5]:", input_features[:5, 0].tolist())
print("OUT dir             :", OUT)
