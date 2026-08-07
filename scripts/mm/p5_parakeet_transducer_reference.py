#!/usr/bin/env python3
# P5 (MODEL-AUDIO-PARAKEET-TRANSDUCER): run the HuggingFace PRETRAINED Parakeet
# transducer end to end and record what it produced, so the C++ port has a real
# transcript to be token-exact against.
#
# Upstream reference: transformers `main`
#   transformers/models/parakeet/modeling_parakeet.py
#     ParakeetRNNTDecoder:831, ParakeetRNNTJointNetwork:879, ParakeetForRNNT:922,
#     ParakeetTDTJointNetwork:1035, ParakeetForTDT:1052
#   transformers/models/parakeet/generation_parakeet.py
#     ParakeetRNNTDecoderCache:23, ParakeetRNNTGenerationMixin:125,
#     ParakeetTDTGenerationMixin:271
#
# Unlike the tiny-random oracle dump (p5_parakeet_transducer_oracle_dump.py) this
# uses REAL weights and REAL speech, so its output IS a transcript claim. It
# writes a JSON record with the emitted sequence, the per-step durations, the
# decoded text, and the extractor's `input_features`, so the C++ side can gate
# both the features and the ids.
#
#   python3 scripts/mm/p5_parakeet_transducer_reference.py \
#       --model nvidia/parakeet-rnnt-0.6b \
#       --audio /path/to/clip.wav \
#       --out /tmp/rnnt_ref
import argparse
import json
import os
import wave

import numpy as np
import torch

import transformers
from transformers import AutoConfig, AutoProcessor


def read_wav_16k_mono(path: str) -> np.ndarray:
    """16-bit PCM mono reader: the same refusal-not-resample contract the C++
    example uses (feature_extraction_parakeet.py:195-201 refuses a rate mismatch)."""
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1, f"need mono, got {w.getnchannels()} channels"
        assert w.getsampwidth() == 2, f"need 16-bit, got {8 * w.getsampwidth()}-bit"
        assert w.getframerate() == 16000, f"need 16 kHz, got {w.getframerate()}"
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return (pcm.astype(np.float32) / 32768.0).copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    cfg = AutoConfig.from_pretrained(args.model)
    arch = cfg.architectures[0]
    model_cls = getattr(transformers, arch)
    model = model_cls.from_pretrained(args.model, dtype=torch.float32).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    audio = read_wav_16k_mono(args.audio)
    inputs = processor(audio)
    features = inputs["input_features"].to(torch.float32)
    attn = inputs.get("attention_mask")

    out = model.generate(**inputs)
    sequences = out.sequences[0].tolist()
    durations = out.durations[0].tolist() if out.durations is not None else None

    # processing_parakeet.py:128-130: group_tokens is FALSE for a transducer
    # (only CTC collapses runs); the blank is dropped as the tokenizer's pad.
    text = processor.batch_decode(out.sequences)[0]

    blank = cfg.blank_token_id
    emitted = [int(t) for t in sequences if int(t) != blank]

    features.numpy().astype("<f4").tofile(os.path.join(args.out, "input_features.bin"))
    record = {
        "provenance": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "model": args.model,
            "architecture": arch,
            "audio": os.path.abspath(args.audio),
            "attn_implementation": model.encoder.config._attn_implementation,
        },
        "config": {
            "vocab_size": cfg.vocab_size,
            "blank_token_id": cfg.blank_token_id,
            "pad_token_id": cfg.pad_token_id,
            "decoder_hidden_size": cfg.decoder_hidden_size,
            "num_decoder_layers": cfg.num_decoder_layers,
            "hidden_act": cfg.hidden_act,
            "max_symbols_per_step": cfg.max_symbols_per_step,
            "durations": list(getattr(cfg, "durations", []) or []),
            "num_mel_bins": cfg.encoder_config.num_mel_bins,
        },
        "generation_config": {
            "decoder_start_token_id": model.generation_config.decoder_start_token_id,
            "eos_token_id": model.generation_config.eos_token_id,
            "suppress_tokens": model.generation_config.suppress_tokens,
        },
        "shapes": {"input_features": list(features.shape)},
        "valid_frames": int(attn[0].sum()) if attn is not None else int(features.shape[1]),
        "sequences": [int(t) for t in sequences],
        "durations": [int(d) for d in durations] if durations is not None else None,
        "emitted_tokens": emitted,
        "text": text,
    }
    with open(os.path.join(args.out, "reference.json"), "w") as f:
        json.dump(record, f, indent=1)

    print(json.dumps({k: v for k, v in record.items() if k != "sequences"}, indent=1))
    print("SEQUENCES:", record["sequences"])
    print("TEXT:", repr(text))


if __name__ == "__main__":
    main()
