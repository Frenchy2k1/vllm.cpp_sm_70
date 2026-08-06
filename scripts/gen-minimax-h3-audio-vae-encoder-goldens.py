#!/usr/bin/env python3
"""Emit tests/vllm/models/minimax_h3_audio_vae_encoder_goldens.inc.

The companion to scripts/gen-minimax-h3-audio-vae-goldens.py, for the ANALYSIS
half of the same VAE. The decoder generator gates BigVGAN; this one gates the
DAC-lineage ENCODER plus the two stages the reference-audio path runs after it:

    preprocess (right-pad to a multiple of hop_length)
      -> Encoder            (dac_audio_vae.py:90-117)
      -> pre_block          (dac_attn_proj.py:69-88, when attn_proj is set)
      -> mean_proj          (dac_audio_vae.py:157)

which is exactly the sequence vLLM-Omni runs to turn a reference waveform into
conditioning rows (vae.py:317-325). It takes the distribution MEAN (`mean_proj`)
and never `logs_proj`, so the same reference always yields the same rows.

Like every other H3 generator this executes the CHECKPOINT'S OWN remote-code
modules as the oracle; the remote code is not vendored here (it ships under the
MiniMax H3 Community License with the checkpoint). Point --h3-vae-source at a
local copy of the checkpoint's `audio_vae/` directory; only the .py files are
needed, not the 605 MB model.safetensors:

    python3 scripts/gen-minimax-h3-audio-vae-encoder-goldens.py \\
        --h3-vae-source ~/h3/FL2VA/audio_vae \\
        --out tests/vllm/models/minimax_h3_audio_vae_encoder_goldens.inc

Weights come from the shared FNV-1a + splitmix64 stream keyed by state_dict
NAME, identical to gen-minimax-h3-goldens.py :: h3_rand and the C++ H3Rand, so
the golden is reproducible from source alone and no weight byte is checked in.

Shipped geometry (metadata.json + config.yaml, and confirmed by the real
1087-tensor manifest): encoder_dim 64, encoder_rates [2,4,4,5,5] (hop 800),
latent_dim 2048, vae_latent_channels 32, attn_proj true with 8 heads. The
reduced configuration below keeps every structural feature — the strided
downsampling stack, the 3 dilated residual units per block, Snake1d, the causal
attention projection with its head-mean + adaptive-average-pool narrowing, and
the GeGLU MLP — and only shrinks the magnitudes.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn

_MASK64 = (1 << 64) - 1


def fnv1a64(name: str) -> int:
    h = 0xCBF29CE484222325
    for byte in name.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & _MASK64
    return h


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def h3_rand(name: str, count: int) -> np.ndarray:
    """Identical to gen-minimax-h3-goldens.py :: h3_rand and the C++ H3Rand."""
    seed = fnv1a64(name)
    out = np.empty(count, dtype=np.float64)
    for i in range(count):
        u = splitmix64((seed + i) & _MASK64)
        out[i] = ((u >> 11) * (2.0**-53)) * 2.0 - 1.0
    return out


def load_remote_package(source: Path) -> types.ModuleType:
    """Import the checkpoint's audio_vae bundle without vendoring it."""
    source = source.expanduser().resolve()
    for required in ("dac_audio_vae.py", "dac_attn_proj.py", "dac_bigvgan.py"):
        if not (source / required).is_file():
            raise SystemExit(
                f"missing {required} in {source}; point --h3-vae-source at the checkpoint's audio_vae/"
            )
    pkg = types.ModuleType("h3_audio_vae")
    pkg.__path__ = [str(source)]
    sys.modules["h3_audio_vae"] = pkg
    spec = importlib.util.spec_from_file_location(
        "h3_audio_vae.dac_audio_vae", source / "dac_audio_vae.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Reduced dimensions. The SHIPPED encoder is 5 strided blocks 64 -> 2048 over a
# hop of 800; this keeps the structure and shrinks the numbers.
ENCODER_DIM = 8
ENCODER_RATES = [2, 2]
LATENT_DIM = 16
VAE_LATENT_CHANNELS = 4
ATTN_HEADS = 2
# Deliberately NOT a multiple of hop_length (4): preprocess must right-pad it,
# and a port that skipped the padding would produce one frame fewer.
INPUT_SAMPLES = 25


class RefAudioEncoder(nn.Module):
    """The encode half of DacAudioVAE, assembled exactly as DacAudioVAE does.

    Submodule NAMES match the checkpoint (`encoder.block.*`, `pre_block.*`,
    `mean_proj.*`), so the emitted golden is keyed by the same state_dict names
    the C++ loader maps onto. DacAudioVAE itself cannot be instantiated at these
    dimensions because its __init__ always builds a full BigVGAN and rejects any
    sample_rate other than 16k/32k (dac_audio_vae.py:162-193); the encode path it
    would build is reproduced here from the same lines.
    """

    def __init__(self, vae_mod) -> None:
        super().__init__()
        # dac_audio_vae.py:148-149
        self.hop_length = int(np.prod(ENCODER_RATES))
        self.encoder = vae_mod.Encoder(ENCODER_DIM, ENCODER_RATES, LATENT_DIM)
        # dac_audio_vae.py:151-158
        if LATENT_DIM % VAE_LATENT_CHANNELS == 0:
            attn_proj_dim = VAE_LATENT_CHANNELS
        else:
            attn_proj_dim = 2 ** int(np.ceil(np.log2(VAE_LATENT_CHANNELS)))
        self.attn_proj_dim = attn_proj_dim
        self.mean_proj = nn.Conv1d(attn_proj_dim, VAE_LATENT_CHANNELS, 1)
        # dac_audio_vae.py:195-196
        self.pre_block = vae_mod.AttnProjection(LATENT_DIM, attn_proj_dim, num_heads=ATTN_HEADS)

    def preprocess(self, audio_data):
        # dac_audio_vae.py:201-209
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        return nn.functional.pad(audio_data, (0, right_pad))

    def forward(self, waveform):
        # vae.py:317-325 — the reference-audio encode, verbatim.
        audio = self.preprocess(waveform.unsqueeze(1))
        latent = self.encoder(audio)
        latent = self.pre_block(latent.transpose(1, 2)).transpose(1, 2)
        return self.mean_proj(latent)


def fill_from_stream(model: nn.Module) -> None:
    """Set every parameter from the deterministic stream, by NAME."""
    state = model.state_dict()
    for name, tensor in state.items():
        count = tensor.numel()
        if name.endswith("original0"):  # weight-norm magnitude g
            values = h3_rand(name, count) * 0.03 + 0.15
        elif name.endswith("original1"):  # weight-norm direction v
            values = h3_rand(name, count) * 0.08
        elif name.endswith(".alpha"):
            # Snake1d divides by (alpha + 1e-9), and upstream initializes alpha to
            # ONES (dac_audio_vae.py:37). Centering on 1 keeps the reciprocal
            # well-conditioned; centering on 0 would make the golden a numerical
            # accident rather than a test.
            values = h3_rand(name, count) * 0.2 + 1.0
        elif name.endswith("zero_k_bias"):
            continue  # a registered BUFFER that must stay zero (dac_attn_proj.py:37)
        elif name.endswith("bias"):  # covers .bias, q_bias and v_bias
            values = h3_rand(name, count) * 0.05
        elif "norm" in name and tensor.dim() == 1 and name.endswith(".weight"):
            values = h3_rand(name, count) * 0.1 + 1.0  # LayerNorm gain, centered on 1
        else:
            values = h3_rand(name, count) * 0.1
        state[name] = torch.from_numpy(values.astype(np.float32)).reshape(tensor.shape)
    model.load_state_dict(state, strict=True)


def emit_f32(out, name: str, values) -> None:
    flat = np.asarray(values, dtype=np.float32).reshape(-1).tolist()
    out.write(f"inline constexpr float {name}[] = {{\n")
    for i in range(0, len(flat), 6):
        chunk = []
        for v in flat[i : i + 6]:
            text = f"{v:.9g}"
            if "." not in text and "e" not in text and "E" not in text:
                text += ".0"
            chunk.append(text + "f")
        out.write("    " + ", ".join(chunk) + ",\n")
    out.write("};\n\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h3-vae-source", required=True, type=Path,
                        help="the checkpoint's FL2VA/audio_vae directory (.py files only)")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    vae_mod = load_remote_package(args.h3_vae_source)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = RefAudioEncoder(vae_mod).eval()
    fill_from_stream(model)

    # One MONO waveform row: the reference path encodes each channel separately
    # (it repeats a mono file up to the model's 2 channels), so one row is the
    # unit under test.
    waveform = torch.from_numpy(
        h3_rand("audiovae.encoder.input", INPUT_SAMPLES).astype(np.float32)
    ).reshape(1, INPUT_SAMPLES)

    # The three staged tensors, so a C++ mismatch can be localized to a stage
    # instead of being a single opaque end-to-end number.
    padded = model.preprocess(waveform.unsqueeze(1))
    enc = model.encoder(padded)                                   # [1, LATENT_DIM, frames]
    proj = model.pre_block(enc.transpose(1, 2)).transpose(1, 2)   # [1, attn_proj_dim, frames]
    latent = model.mean_proj(proj)                                # [1, VAE_LATENT_CHANNELS, frames]
    frames = int(enc.shape[-1])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        out.write(
            "// GENERATED by scripts/gen-minimax-h3-audio-vae-encoder-goldens.py — DO NOT EDIT BY HAND.\n"
            "//\n"
            "// MiniMax-H3 AUDIO VAE **encoder** goldens, produced by executing the CHECKPOINT'S OWN\n"
            "// remote-code modules (FL2VA/audio_vae/dac_audio_vae.py + dac_attn_proj.py) at reduced\n"
            "// dimensions, in the exact sequence vLLM-Omni's reference-audio path runs\n"
            "// (vae.py:317-325): preprocess -> Encoder -> pre_block -> mean_proj. The remote code is\n"
            "// not vendored here; see the generator's docstring. Weights come from the shared H3Rand\n"
            "// stream, so no weight byte is checked in.\n"
            "#pragma once\n\n#include <cstdint>\n\nnamespace vllm_test {\n\n"
        )
        out.write(f"inline constexpr int64_t kH3AudioEncDim = {ENCODER_DIM};\n")
        out.write(f"inline constexpr int64_t kH3AudioEncLatentDim = {LATENT_DIM};\n")
        out.write(
            f"inline constexpr int64_t kH3AudioEncVaeChannels = {VAE_LATENT_CHANNELS};\n")
        out.write(f"inline constexpr int64_t kH3AudioEncHeads = {ATTN_HEADS};\n")
        out.write(f"inline constexpr int64_t kH3AudioEncSamples = {INPUT_SAMPLES};\n")
        out.write(f"inline constexpr int64_t kH3AudioEncFrames = {frames};\n")
        out.write("inline constexpr int64_t kH3AudioEncRates[] = {"
                  + ", ".join(str(r) for r in ENCODER_RATES) + "};\n\n")
        emit_f32(out, "kH3AudioEncInputGolden", waveform.reshape(-1))
        emit_f32(out, "kH3AudioEncBlockGolden", enc.reshape(-1))
        emit_f32(out, "kH3AudioEncAttnProjGolden", proj.reshape(-1))
        emit_f32(out, "kH3AudioEncMeanGolden", latent.reshape(-1))
        out.write("}  // namespace vllm_test\n")
    print(f"wrote {args.out} (frames {frames})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
