#!/usr/bin/env python3
# P4 (MODEL-AUDIO-PARAKEET-ENCODER) — dump the Parakeet log-mel front-end ORACLE
# fixture for tests/vllm/multimodal/test_parakeet_audio_processor.cpp.
#
# Two upstreams, both dumped, because they disagree by a documented rounding:
#   transformers 5.3.0 audio_utils.mel_filter_bank:453 (float64, slaney/slaney)
#     — the EXACT function vLLM's ParakeetExtractor._get_mel_filters
#       (vllm/model_executor/models/parakeet.py:154-168) calls;
#   librosa.filters.mel (float32) — what HF's own ParakeetFeatureExtractor
#     (transformers/models/parakeet/feature_extraction_parakeet.py:94-97) uses,
#     with a comment at :83-93 saying the ONLY difference is float64 vs float32.
# Our C++ constructs the transformers one in double, so the first is the tight
# gate for the construction and the second bounds the disagreement.
#
# Also dumps the full `ParakeetFeatureExtractor.__call__` (:127-282) output on a
# fixed waveform: preemphasis -> torch.stft -> power -> mel -> log -> per-bin
# normalisation.
#
#   python3 scripts/mm/p4_parakeet_extractor_ref.py \
#       --out tests/vllm/multimodal/fixtures/parakeet_audio
import argparse
import json
import math
import os

import numpy as np
import torch

import librosa
from transformers.audio_utils import mel_filter_bank
from transformers.models.parakeet.feature_extraction_parakeet import (
    ParakeetFeatureExtractor,
)

SR = 16000
NUM_SAMPLES = 8000  # 0.5 s -> 1 + 8000/160 = 51 STFT frames


def waveform() -> np.ndarray:
    """A deterministic, non-degenerate mono signal: three tones plus a seeded
    noise floor, so every mel band carries energy and no band is exactly zero."""
    t = np.arange(NUM_SAMPLES, dtype=np.float64) / SR
    x = (
        0.5 * np.sin(2 * math.pi * 220.0 * t)
        + 0.25 * np.sin(2 * math.pi * 1750.0 * t)
        + 0.15 * np.sin(2 * math.pi * 5200.0 * t)
    )
    x += 0.02 * np.random.default_rng(20260806).standard_normal(NUM_SAMPLES)
    return (x / np.abs(x).max() * 0.9).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)
    torch.set_grad_enabled(False)

    fe = ParakeetFeatureExtractor()
    n_mels, n_fft = fe.feature_size, fe.n_fft

    # [num_freq, num_mels] -> transposed to the [num_mels, num_freq] orientation
    # both upstreams multiply in (parakeet.py:168).
    tf_bank = mel_filter_bank(
        num_frequency_bins=n_fft // 2 + 1,
        num_mel_filters=n_mels,
        min_frequency=0.0,
        max_frequency=SR / 2,
        sampling_rate=SR,
        norm="slaney",
        mel_scale="slaney",
    ).T.astype(np.float32)
    librosa_bank = librosa.filters.mel(
        sr=SR, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=SR / 2, norm="slaney"
    ).astype(np.float32)

    wav = waveform()
    feats = fe(wav, sampling_rate=SR, return_tensors="pt")
    input_features = feats["input_features"][0].to(torch.float32).numpy()
    attention_mask = feats["attention_mask"][0].numpy().astype(np.int32)

    wav.tofile(os.path.join(out, "waveform_f32.bin"))
    tf_bank.tofile(os.path.join(out, "mel_filters_transformers_f32.bin"))
    librosa_bank.tofile(os.path.join(out, "mel_filters_librosa_f32.bin"))
    input_features.tofile(os.path.join(out, "input_features_f32.bin"))
    attention_mask.tofile(os.path.join(out, "attention_mask_i32.bin"))

    manifest = {
        "provenance": {
            "transformers": __import__("transformers").__version__,
            "torch": torch.__version__,
            "librosa": librosa.__version__,
            "feature_extractor": (
                "transformers/models/parakeet/feature_extraction_parakeet.py"
                " ParakeetFeatureExtractor:36"
            ),
            "mel_filter_bank": "transformers/audio_utils.py mel_filter_bank:453",
            "vllm_extractor": (
                "vllm/model_executor/models/parakeet.py ParakeetExtractor:138"
            ),
        },
        "config": {
            "feature_size": n_mels,
            "sampling_rate": fe.sampling_rate,
            "hop_length": fe.hop_length,
            "n_fft": n_fft,
            "win_length": fe.win_length,
            "preemphasis": fe.preemphasis,
            "padding_value": fe.padding_value,
        },
        "num_samples": int(wav.shape[0]),
        "shapes": {
            "mel_filters": list(tf_bank.shape),
            "input_features": list(input_features.shape),
            "attention_mask": list(attention_mask.shape),
        },
        "valid_frames": int(attention_mask.sum()),
        "max_abs_bank_delta": float(np.abs(tf_bank - librosa_bank).max()),
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"wrote extractor fixture to {out}: {manifest['shapes']}", flush=True)


if __name__ == "__main__":
    main()
