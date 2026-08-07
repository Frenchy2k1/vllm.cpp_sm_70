#!/usr/bin/env python3
"""Deterministic tiny Parakeet checkpoints + clip for the ONE-SURFACE fold gate.

Row `ARCH-ONE-SURFACE` (Parakeet ASR fold): the refactor that moves the
`examples/parakeet_transcribe` pipeline into the library must reproduce the
EXACT ids + transcript the PRE-refactor example produced. The real
`nvidia/parakeet-*` checkpoints were deleted after PR #89 and AGENTS.md's safe
defaults forbid re-downloading multi-GB assets, so the gate runs on the
smallest checkpoint that exercises every stage: this generator emits

  tests/vllm/models/fixtures/parakeet_e2e/
    audio.wav            deterministic 16 kHz mono PCM16 clip (0.25 s)
    ctc/                 HF-format ParakeetForCTC   (config+safetensors+tokenizer)
    rnnt/                HF-format ParakeetForRNNT  (config+safetensors+tokenizer
                                                     +generation_config)
    golden_ctc.txt       stdout of the PRE-refactor example on ctc/  + audio.wav
    golden_rnnt.txt      stdout of the PRE-refactor example on rnnt/ + audio.wav

The goldens are NOT written by this script: they were captured by running the
pre-refactor `parakeet-transcribe` binary (main @ f98e1e48) on these exact
fixtures, and tests/vllm/models/test_parakeet_transcription_fold.cpp gates the
post-refactor library path byte-identical against them. Regenerating the
CHECKPOINTS is safe (every byte is a pure function of the seeds below);
regenerating the GOLDENS requires rebuilding the pre-refactor example.

Weight filler (shared contract with nothing — the committed bytes are the
fixture): w[i] = (((seed*31 + i) % 17) - 8) / 16, exactly representable in f32.
BatchNorm running_var uses a strictly positive variant. The tokenizer is the
published Parakeet shape: BPE model, `Metaspace` pre_tokenizer with
`split: true` + `prepend_scheme: "always"`, and a `Metaspace` DECODER — the
exact file `vllm::Tokenizer` used to refuse (the reason the example carried a
private DecodeIds).
"""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/vllm/models/fixtures/parakeet_e2e"

# ── tiny geometry (shared by both heads) ─────────────────────────────────────
HID = 16          # encoder hidden_size
LAYERS = 1        # encoder layers
HEADS = 2         # attention heads (head_dim 8)
INTER = 32        # feed-forward intermediate
MELS = 8          # num_mel_bins
SUB_C = 4         # subsampling_conv_channels
SUB_F = 4         # subsampling_factor (2 conv stages, stride 2)
KCONV = 3         # conformer depthwise kernel
KSUB = 3          # subsampling conv kernel
VOCAB = 8         # vocab incl. blank
BLANK = 7         # == pad_token_id (CTC blank) == blank_token_id (RNN-T)
DEC_D = 8         # transducer decoder_hidden_size
DEC_L = 1         # transducer LSTM layers


def fill(n: int, seed: int) -> bytes:
    """f32 little-endian: (((seed*31 + i) % 17) - 8) / 16 — exact in float32."""
    return b"".join(
        struct.pack("<f", (((seed * 31 + i) % 17) - 8) / 16.0) for i in range(n)
    )


def fill_pos(n: int, seed: int) -> bytes:
    """Strictly positive variant for BatchNorm running_var: (((...)%17)+9)/16."""
    return b"".join(
        struct.pack("<f", (((seed * 31 + i) % 17) + 9) / 16.0) for i in range(n)
    )


def write_safetensors(path: Path, tensors: list[tuple[str, list[int], bytes]]) -> None:
    header: dict[str, dict] = {}
    data = b""
    for name, shape, raw in tensors:
        start = len(data)
        data += raw
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [start, len(data)],
        }
    blob = json.dumps(header, sort_keys=True).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + data)


def encoder_tensors(seed0: int) -> list[tuple[str, list[int], bytes]]:
    s = seed0
    t: list[tuple[str, list[int], bytes]] = []

    def add(name: str, shape: list[int], pos: bool = False) -> None:
        nonlocal s
        n = math.prod(shape)
        t.append((name, shape, (fill_pos if pos else fill)(n, s)))
        s += 1

    add("encoder.subsampling.layers.0.weight", [SUB_C, 1, KSUB, KSUB])
    add("encoder.subsampling.layers.0.bias", [SUB_C])
    add("encoder.subsampling.layers.2.weight", [SUB_C, 1, KSUB, KSUB])
    add("encoder.subsampling.layers.2.bias", [SUB_C])
    add("encoder.subsampling.layers.3.weight", [SUB_C, SUB_C, 1, 1])
    add("encoder.subsampling.layers.3.bias", [SUB_C])
    out_freq = MELS // (2 * 2)
    add("encoder.subsampling.linear.weight", [HID, SUB_C * out_freq])
    add("encoder.subsampling.linear.bias", [HID])
    for layer in range(LAYERS):
        base = f"encoder.layers.{layer}."
        for ff in ("feed_forward1.", "feed_forward2."):
            add(base + ff + "linear1.weight", [INTER, HID])
            add(base + ff + "linear1.bias", [INTER])
            add(base + ff + "linear2.weight", [HID, INTER])
            add(base + ff + "linear2.bias", [HID])
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            add(base + f"self_attn.{proj}.weight", [HID, HID])
            add(base + f"self_attn.{proj}.bias", [HID])
        add(base + "self_attn.relative_k_proj.weight", [HID, HID])
        add(base + "self_attn.bias_u", [HEADS, HID // HEADS])
        add(base + "self_attn.bias_v", [HEADS, HID // HEADS])
        add(base + "conv.pointwise_conv1.weight", [2 * HID, HID, 1])
        add(base + "conv.pointwise_conv1.bias", [2 * HID])
        add(base + "conv.depthwise_conv.weight", [HID, 1, KCONV])
        add(base + "conv.depthwise_conv.bias", [HID])
        add(base + "conv.norm.weight", [HID])
        add(base + "conv.norm.bias", [HID])
        add(base + "conv.norm.running_mean", [HID])
        add(base + "conv.norm.running_var", [HID], pos=True)
        add(base + "conv.pointwise_conv2.weight", [HID, HID, 1])
        add(base + "conv.pointwise_conv2.bias", [HID])
        for norm in (
            "norm_feed_forward1",
            "norm_self_att",
            "norm_conv",
            "norm_feed_forward2",
            "norm_out",
        ):
            add(base + norm + ".weight", [HID])
            add(base + norm + ".bias", [HID])
    return t


def encoder_config() -> dict:
    return {
        "hidden_size": HID,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "intermediate_size": INTER,
        "num_mel_bins": MELS,
        "subsampling_conv_channels": SUB_C,
        "subsampling_factor": SUB_F,
        "subsampling_conv_kernel_size": KSUB,
        "subsampling_conv_stride": 2,
        "conv_kernel_size": KCONV,
        "attention_bias": True,
        "convolution_bias": True,
        "scale_input": True,
        "hidden_act": "silu",
    }


def tokenizer_json() -> dict:
    # The published Parakeet tokenizer shape: BPE vocab + Metaspace(split=true)
    # pre_tokenizer + Metaspace DECODER (`{"type": "Metaspace", "replacement":
    # "▁", "prepend_scheme": "always", "split": true}` on every published
    # checkpoint — see examples/parakeet_transcribe pre-refactor DecodeIds).
    vocab = {
        "▁the": 0,
        "▁cat": 1,
        "▁sat": 2,
        "at": 3,
        "he": 4,
        "s": 5,
        "▁on": 6,
    }
    return {
        "version": "1.0",
        "pre_tokenizer": {
            "type": "Metaspace",
            "replacement": "▁",
            "prepend_scheme": "always",
            "split": True,
        },
        "decoder": {
            "type": "Metaspace",
            "replacement": "▁",
            "prepend_scheme": "always",
            "split": True,
        },
        "model": {
            "type": "BPE",
            "unk_token": None,
            "vocab": vocab,
            "merges": [],
        },
        "added_tokens": [
            {"id": BLANK, "content": "<blank>", "special": True},
        ],
    }


def write_wav(path: Path) -> None:
    # Four strongly different 62.5 ms segments (tone / loud high tone / chirp /
    # deterministic "noise"), so the per-frame features — and therefore the
    # greedy ids — VARY across the clip. A stationary tone made every frame
    # argmax to the same id, which was a degenerate golden: it never exercised
    # the Metaspace first-piece vs later-piece decode distinction.
    n = 4000  # 0.25 s @ 16 kHz
    samples = bytearray()
    for i in range(n):
        seg = i // 1000
        if seg == 0:
            v = 0.45 * math.sin(2 * math.pi * 440.0 * i / 16000.0)
        elif seg == 1:
            v = 0.9 * math.sin(2 * math.pi * 3100.0 * i / 16000.0)
        elif seg == 2:
            f = 200.0 + (i - 2000) * 3.0  # 200 -> 3200 Hz chirp
            v = 0.6 * math.sin(2 * math.pi * f * i / 16000.0)
        else:
            v = ((((i * 2654435761) >> 7) & 0xFFFF) / 32768.0 - 1.0) * 0.5
        samples += struct.pack("<h", int(round(v * 20000)))
    data = bytes(samples)
    hdr = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    fmt = b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
    path.write_bytes(hdr + fmt + b"data" + struct.pack("<I", len(data)) + data)


def emit_ctc(dir_: Path) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    cfg = {
        "architectures": ["ParakeetForCTC"],
        "model_type": "parakeet_ctc",
        "vocab_size": VOCAB,
        "pad_token_id": BLANK,
        "encoder_config": encoder_config(),
    }
    (dir_ / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    (dir_ / "tokenizer.json").write_text(
        json.dumps(tokenizer_json(), ensure_ascii=False, indent=1) + "\n"
    )
    # Head seeds chosen (probed over the fixed encoder + clip) so the greedy
    # ids VARY across frames: 907/500 yields ids `3 4 3` -> "atheat", a
    # three-piece transcript rather than a degenerate single token.
    tensors = encoder_tensors(seed0=1)
    tensors.append(("ctc_head.weight", [VOCAB, HID, 1], fill(VOCAB * HID, 907)))
    tensors.append(("ctc_head.bias", [VOCAB], fill(VOCAB, 500)))
    write_safetensors(dir_ / "model.safetensors", tensors)


def emit_rnnt(dir_: Path) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    cfg = {
        "architectures": ["ParakeetForRNNT"],
        "model_type": "parakeet_rnnt",
        "vocab_size": VOCAB,
        "blank_token_id": BLANK,
        "pad_token_id": 2,
        "decoder_hidden_size": DEC_D,
        "num_decoder_layers": DEC_L,
        "max_symbols_per_step": 3,
        "hidden_act": "relu",
        "encoder_config": encoder_config(),
    }
    (dir_ / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    (dir_ / "generation_config.json").write_text(
        json.dumps({"decoder_start_token_id": BLANK}, indent=1) + "\n"
    )
    (dir_ / "tokenizer.json").write_text(
        json.dumps(tokenizer_json(), ensure_ascii=False, indent=1) + "\n"
    )
    tensors = encoder_tensors(seed0=101)
    tensors.append(
        ("encoder_projector.weight", [DEC_D, HID], fill(DEC_D * HID, 801))
    )
    tensors.append(("encoder_projector.bias", [DEC_D], fill(DEC_D, 802)))
    tensors.append(
        ("decoder.embedding.weight", [VOCAB, DEC_D], fill(VOCAB * DEC_D, 803))
    )
    for layer in range(DEC_L):
        s = 810 + 4 * layer
        tensors.append(
            (f"decoder.lstm.weight_ih_l{layer}", [4 * DEC_D, DEC_D],
             fill(4 * DEC_D * DEC_D, s))
        )
        tensors.append(
            (f"decoder.lstm.weight_hh_l{layer}", [4 * DEC_D, DEC_D],
             fill(4 * DEC_D * DEC_D, s + 1))
        )
        tensors.append(
            (f"decoder.lstm.bias_ih_l{layer}", [4 * DEC_D], fill(4 * DEC_D, s + 2))
        )
        tensors.append(
            (f"decoder.lstm.bias_hh_l{layer}", [4 * DEC_D], fill(4 * DEC_D, s + 3))
        )
    tensors.append(
        ("decoder.decoder_projector.weight", [DEC_D, DEC_D],
         fill(DEC_D * DEC_D, 830))
    )
    tensors.append(("decoder.decoder_projector.bias", [DEC_D], fill(DEC_D, 831)))
    tensors.append(("joint.head.weight", [VOCAB, DEC_D], fill(VOCAB * DEC_D, 840)))
    tensors.append(("joint.head.bias", [VOCAB], fill(VOCAB, 841)))
    write_safetensors(dir_ / "model.safetensors", tensors)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    write_wav(OUT / "audio.wav")
    emit_ctc(OUT / "ctc")
    emit_rnnt(OUT / "rnnt")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
