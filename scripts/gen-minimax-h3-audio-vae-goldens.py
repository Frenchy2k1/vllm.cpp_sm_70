#!/usr/bin/env python3
"""Emit tests/vllm/models/minimax_h3_audio_vae_goldens.inc.

The MiniMax-H3 audio VAE is **checkpoint REMOTE CODE**: its implementation ships
inside the HF repo (`FL2VA/audio_vae/*.py`) and vLLM-Omni only adapts it under
`trust_remote_code` (vae.py:41-53). A pure-C++ engine cannot do that, so the VAE
must be REIMPLEMENTED — and this generator is what makes that reimplementation
verifiable: it imports the checkpoint's OWN Python modules and runs them at
reduced dimensions as the oracle, exactly like gen-minimax-h3-goldens.py does for
the DiT.

The remote code is NOT vendored into this repository (it ships under the MiniMax
H3 Community License with the checkpoint). Point --h3-vae-source at a local copy
of the checkpoint's `audio_vae/` directory; only the .py files are needed, not the
605 MB model.safetensors:

    python3 scripts/gen-minimax-h3-audio-vae-goldens.py \\
        --h3-vae-source ~/h3/FL2VA/audio_vae \\
        --out tests/vllm/models/minimax_h3_audio_vae_goldens.inc

Architecture (from the checkpoint's config.yaml + metadata.json):
  decode(z) = dec_in_proj(Conv1d 32 -> 2048, k=1) -> BigVGAN
  BigVGAN: conv_pre(2048 -> 1024, k=7) -> 7 x [ConvTranspose1d upsample +
           3 AMPBlock1 residual blocks averaged] -> anti-aliased SnakeBeta ->
           conv_post(-> 1 channel, k=7, no bias) -> clamp[-1, 1]
  upsample_rates [5,5,2,2,2,2,2], kernels [9,9,4,4,4,4,4], resblock kernels
  [3,7,11] with dilations [1,3,5], snakebeta activations in LOG scale.
Every conv is weight-normalized, so the checkpoint stores (g, v) pairs
(`parametrizations.weight.original0/1`) and the loader must materialize
w = g * v / ||v|| with the norm taken over every dim except dim 0.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

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
    """Import the checkpoint's audio_vae bundle without vendoring it.

    The bundle uses relative imports, so it is registered as a real package
    whose __path__ is the source directory.
    """
    source = source.expanduser().resolve()
    for required in ("dac_bigvgan.py", "dac_activations.py", "dac_alias_free_act.py"):
        if not (source / required).is_file():
            raise SystemExit(f"missing {required} in {source}; point --h3-vae-source at the checkpoint's audio_vae/")
    pkg = types.ModuleType("h3_audio_vae")
    pkg.__path__ = [str(source)]
    sys.modules["h3_audio_vae"] = pkg
    spec = importlib.util.spec_from_file_location(
        "h3_audio_vae.dac_bigvgan", source / "dac_bigvgan.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Reduced dimensions. The SHIPPED decoder is 7 upsample stages at
# initial_channel 1024 over 2048 mels; this keeps every structural feature
# (multiple stages, 3 resblock kernels x 3 dilations, the anti-aliased snakebeta,
# the final clamp) and only shrinks the magnitudes.
REDUCED = dict(
    resblock="1",
    num_mels=16,
    upsample_rates=[2, 2],
    upsample_kernel_sizes=[4, 4],
    upsample_initial_channel=16,
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    use_tanh_at_final=False,
    use_bias_at_final=False,
    activation="snakebeta",
    snake_logscale=True,
)
INPUT_FRAMES = 8


def fill_from_stream(model: torch.nn.Module) -> None:
    """Set every parameter/buffer from the deterministic stream, by NAME.

    Scales differ per role so the test exercises realistic magnitudes: weight-norm
    magnitudes (`original0`) stay near 1, directions and biases stay small, and
    the snakebeta log-scale alpha/beta stay near 0 (i.e. exp() near 1).
    """
    state = model.state_dict()
    for name, tensor in state.items():
        count = tensor.numel()
        if name.endswith("original0"):  # weight-norm magnitude g
            # g IS the per-output-channel norm of the materialized weight, so it
            # sets the gain of every conv. Kept well below 1 so the residual stack
            # does not saturate the decoder's final clamp[-1, 1] (a saturated
            # golden would hide errors rather than catch them).
            values = h3_rand(name, count) * 0.03 + 0.15
        elif name.endswith("original1"):  # weight-norm direction v
            # Deliberately small: the residual stack otherwise saturates the final
            # clamp[-1, 1] on most samples, and a saturated golden would HIDE
            # errors instead of catching them.
            values = h3_rand(name, count) * 0.08
        elif name.endswith(".alpha") or name.endswith(".beta"):
            values = h3_rand(name, count) * 0.2
        elif name.endswith(".bias"):
            values = h3_rand(name, count) * 0.05
        elif name.endswith("filter"):
            continue  # the kaiser-sinc filters are COMPUTED, never loaded
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

    bigvgan = load_remote_package(args.h3_vae_source)
    torch.set_grad_enabled(False)

    h = bigvgan.AttrDict(**REDUCED)
    model = bigvgan.BigVGAN(h).eval()
    fill_from_stream(model)

    x = torch.from_numpy(
        h3_rand("audiovae.input", REDUCED["num_mels"] * INPUT_FRAMES).astype(np.float32)
    ).reshape(1, REDUCED["num_mels"], INPUT_FRAMES)
    y = model(x)

    # The kaiser-sinc filter the anti-aliased activation uses (up ratio 2,
    # kernel 12). Emitted so the C++ side can prove its filter construction
    # matches before blaming the rest of the decoder for a mismatch.
    from importlib import import_module
    filt_mod = import_module("h3_audio_vae.dac_alias_free_filter")
    up_filter = filt_mod.kaiser_sinc_filter1d(cutoff=0.5 / 2, half_width=0.6 / 2, kernel_size=12)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        out.write(
            "// GENERATED by scripts/gen-minimax-h3-audio-vae-goldens.py — DO NOT EDIT BY HAND.\n"
            "//\n"
            "// MiniMax-H3 AUDIO VAE decoder goldens, produced by executing the CHECKPOINT'S OWN\n"
            "// remote-code modules (FL2VA/audio_vae/*.py) at reduced dimensions. The remote code\n"
            "// is not vendored here; see the generator's docstring. Weights come from the shared\n"
            "// H3Rand stream, so no weight byte is checked in.\n"
            "#pragma once\n\n#include <cstdint>\n\nnamespace vllm_test {\n\n"
        )
        out.write(f"inline constexpr int64_t kH3AudioVaeNumMels = {REDUCED['num_mels']};\n")
        out.write(f"inline constexpr int64_t kH3AudioVaeFrames = {INPUT_FRAMES};\n")
        out.write(
            f"inline constexpr int64_t kH3AudioVaeInitialChannel = {REDUCED['upsample_initial_channel']};\n")
        out.write(f"inline constexpr int64_t kH3AudioVaeOutSamples = {int(y.shape[-1])};\n\n")
        emit_f32(out, "kH3AudioVaeUpFilterGolden", up_filter.reshape(-1))
        emit_f32(out, "kH3AudioVaeWaveformGolden", y.reshape(-1))
        out.write("}  // namespace vllm_test\n")
    print(f"wrote {args.out} (out samples {int(y.shape[-1])})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
