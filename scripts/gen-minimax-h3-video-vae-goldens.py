#!/usr/bin/env python3
"""Emit tests/vllm/models/minimax_h3_video_vae_goldens.inc.

Like the audio VAE, MiniMax-H3's VIDEO VAE is checkpoint REMOTE CODE
(`FL2VA/video_vae/*.py`, loaded under `trust_remote_code`), so it must be
REIMPLEMENTED rather than adapted. Its real 560-tensor manifest shows the DECODER
-- the half generation needs -- is a 36-block TRANSFORMER, so this generator runs
the checkpoint's OWN `TransformerBlock` at reduced dimensions as the oracle.

The remote code is NOT vendored here (MiniMax H3 Community License). Point
--h3-vae-source at a local copy of the checkpoint's `video_vae/` directory; only
the .py files are needed, not the 10 GB model.safetensors:

    python3 scripts/gen-minimax-h3-video-vae-goldens.py \
        --h3-vae-source ~/h3/FL2VA/video_vae \
        --out tests/vllm/models/minimax_h3_video_vae_goldens.inc

The bundle imports a handful of diffusers symbols (a logger, two mixin bases, two
no-op decorators); those are STUBBED here rather than pulling the whole diffusers
dependency in just to run an oracle.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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
    """Identical to the other H3 generators and the C++ H3Rand."""
    seed = fnv1a64(name)
    out = np.empty(count, dtype=np.float64)
    for i in range(count):
        u = splitmix64((seed + i) & _MASK64)
        out[i] = ((u >> 11) * (2.0**-53)) * 2.0 - 1.0
    return out


def _mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def install_diffusers_stub() -> None:
    if "diffusers" in sys.modules:
        return
    d = _mod("diffusers")
    logging.get_logger = logging.getLogger
    utils = _mod("diffusers.utils", logging=logging, get_logger=logging.getLogger)
    d.utils = utils
    utils.torch_utils = _mod("diffusers.utils.torch_utils", maybe_allow_in_graph=lambda cls: cls)

    class ConfigMixin:
        config_name = "config.json"

    def register_to_config(init):
        """Faithful-enough stand-in: capture the ctor kwargs onto `self.config`.

        The bundle reads `self.config.patch_size` / `patch_size_t` in the decoder
        forward, so a no-op decorator is NOT sufficient.
        """
        import functools
        import inspect

        signature = inspect.signature(init)

        @functools.wraps(init)
        def wrapper(self, *args, **kwargs):
            bound = signature.bind_partial(self, *args, **kwargs)
            bound.apply_defaults()
            captured = {k: v for k, v in bound.arguments.items() if k != "self"}
            captured.pop("kwargs", None)
            object.__setattr__(self, "_h3_config", types.SimpleNamespace(**captured))
            return init(self, *args, **kwargs)

        return wrapper

    ConfigMixin.config = property(lambda self: self._h3_config)
    _mod("diffusers.configuration_utils", ConfigMixin=ConfigMixin,
         register_to_config=register_to_config)

    class ModelMixin(nn.Module):
        pass

    models = _mod("diffusers.models")
    models.modeling_utils = _mod("diffusers.models.modeling_utils", ModelMixin=ModelMixin)
    d.models = models


def load_bundle(src: Path, module: str):
    install_diffusers_stub()
    if "h3vv" not in sys.modules:
        pkg = types.ModuleType("h3vv")
        pkg.__path__ = [str(src)]
        sys.modules["h3vv"] = pkg
    name = "h3vv." + module
    spec = importlib.util.spec_from_file_location(name, src / f"{module}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# Reduced dimensions. The SHIPPED decoder is 36 blocks at 36 tensors each; the
# block is the repeated unit, so one block at small width exercises the same math.
HEADS = 2
DIM_HEAD = 8
DIM = HEADS * DIM_HEAD
SEQ = 6

# Reduced ViT3DDecoder. rope_apply_dim = int(dim_head * 0.75) must stay divisible
# by 2 * n_dim = 6 (RotaryEmbeddingND's constraint): 8 * 0.75 = 6. OK.
DEC_LAYERS = 2
DEC_IN_CH = 4
DEC_OUT_CH = 3
DEC_PATCH = 2
DEC_PATCH_T = 1
DEC_REGISTER_TOKENS = 4
DEC_T, DEC_H, DEC_W = 2, 2, 3


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


def emit_tiling(out) -> None:
    """Section: the video VAE's spatial TILING plan and overlap blend.

    Restated from the checkpoint's klvae.py:192-250 (`split_tiles`, `blend`) --
    importing klvae directly would drag in the whole VAE bundle. The shipped
    config is tile_size 256 / tile_overlap_min 64 / vae_ratio 16 (= prod of
    space_down [2,2,2,2,1,1], the "f16" in f16t4).
    """
    import math as _math

    vae_ratio = 16

    def split_tiles(input_len, tile_size, tile_overlap_min):
        if tile_size >= input_len:
            return [0], [input_len], []
        n = _math.ceil(input_len / tile_size)
        while True:
            overlaps = [tile_overlap_min] * (n - 1)
            remaining = tile_size * n - sum(overlaps) - input_len
            if remaining < 0:
                n += 1
            else:
                break
        remaining_units = remaining // vae_ratio
        for i in range(remaining_units):
            overlaps[i % (n - 1)] += vae_ratio
        starts = [0]
        for i in range(n - 1):
            starts.append(starts[-1] + tile_size - overlaps[i])
        return starts, [tile_size] * n, overlaps

    cases = [(1024, 256, 64), (768, 256, 64), (256, 256, 64), (100, 256, 64),
             (1344, 256, 64), (513, 128, 32)]
    starts_flat, lens_flat, ovl_flat, counts = [], [], [], []
    for input_len, tile, overlap in cases:
        st, ln, ov = split_tiles(input_len, tile, overlap)
        counts.append(len(st))
        starts_flat.extend(st)
        lens_flat.extend(ln)
        ovl_flat.extend(ov + [0] * (len(st) - 1 - len(ov)))

    # blend: linear cross-fade over `blend_extent` along one axis.
    extent = 4
    a = h3_rand("tiling.a", 12)
    b = h3_rand("tiling.b", 12)
    blended = []
    for i in range(extent):
        wa = 1.0 - i / extent
        wb = i / extent
        blended.append(a[len(a) - extent + i] * wa + b[i] * wb)
    blended.extend(b[extent:])

    out.write(f"inline constexpr int64_t kH3TileVaeRatio = {vae_ratio};\n")
    out.write(f"inline constexpr int64_t kH3TileCases = {len(cases)};\n")
    out.write(f"inline constexpr int64_t kH3TileBlendExtent = {extent};\n")
    out.write(f"inline constexpr int64_t kH3TileBlendLen = {len(a)};\n\n")
    emit_i64(out, "kH3TileInputs", [v for c in cases for v in c])
    emit_i64(out, "kH3TileCounts", counts)
    emit_i64(out, "kH3TileStarts", starts_flat)
    emit_i64(out, "kH3TileLens", lens_flat)
    emit_i64(out, "kH3TileOverlaps", ovl_flat)
    emit_f32(out, "kH3TileBlendGolden", np.asarray(blended, dtype=np.float32))


def emit_i64(out, name, values):
    flat = [int(v) for v in np.asarray(values).reshape(-1).tolist()]
    out.write(f"inline constexpr int64_t {name}[] = {{\n")
    for i in range(0, len(flat), 12):
        out.write("    " + ", ".join(str(v) for v in flat[i : i + 12]) + ",\n")
    out.write("};\n\n")


def emit_resnet3d(out, src) -> None:
    """Section: the VAE encoder's ResnetBlock3D (the repeated unit of the 3D CNN).

    GroupNorm(32, eps 1e-6) -> SiLU -> causal Conv3d(3x3x3) -> GroupNorm -> SiLU ->
    Conv3d -> plus a 1x1x1 `nin_shortcut` when the channel count changes. The
    convolution is CAUSAL in time (all temporal padding on the LEFT) and uses the
    checkpoint's `reflect` spatial padding.
    """
    cnn = load_bundle(src, "vae_cnn")
    # GroupNorm uses 32 groups, so channel counts must be multiples of 32.
    in_ch, out_ch = 32, 64
    block = cnn.ResnetBlock3D(in_ch, out_ch, padding_mode="reflect", causal=True).eval()
    state = block.state_dict()
    for name, tensor in state.items():
        scale, offset = (0.1, 0.0)
        if ".weight" in name and tensor.dim() == 1:
            scale, offset = 0.1, 1.0     # group-norm gain
        elif name.endswith(".bias"):
            scale = 0.05
        state[name] = torch.from_numpy(
            (h3_rand("resnet3d." + name, tensor.numel()) * scale + offset).astype(np.float32)
        ).reshape(tensor.shape)
    block.load_state_dict(state, strict=True)

    t, hh, ww = 3, 4, 4  # T>1 so the CAUSAL temporal padding path is taken
    x = torch.from_numpy(
        h3_rand("resnet3d.input", in_ch * t * hh * ww).astype(np.float32)
    ).reshape(1, in_ch, t, hh, ww)
    y = block(x)

    out.write(f"inline constexpr int64_t kH3Res3dInCh = {in_ch};\n")
    out.write(f"inline constexpr int64_t kH3Res3dOutCh = {out_ch};\n")
    out.write(f"inline constexpr int64_t kH3Res3dT = {t};\n")
    out.write(f"inline constexpr int64_t kH3Res3dH = {hh};\n")
    out.write(f"inline constexpr int64_t kH3Res3dW = {ww};\n")
    out.write(f"inline constexpr int64_t kH3Res3dGroups = 32;\n\n")
    emit_f32(out, "kH3Res3dGolden", y.reshape(-1))


def emit_downsample3d(out, src) -> None:
    """Section: Downsample3D — the encoder's strided, asymmetrically-padded conv."""
    cnn = load_bundle(src, "vae_cnn")
    in_ch, out_ch = 32, 32
    t, hh, ww = 3, 4, 4
    block = cnn.Downsample3D(in_ch, out_ch, time_stride=2, space_stride=2,
                             padding_mode="reflect", causal=True).eval()
    state = block.state_dict()
    for name, tensor in state.items():
        scale = 0.05 if name.endswith(".bias") else 0.1
        state[name] = torch.from_numpy(
            (h3_rand("down3d." + name, tensor.numel()) * scale).astype(np.float32)
        ).reshape(tensor.shape)
    block.load_state_dict(state, strict=True)
    x = torch.from_numpy(
        h3_rand("down3d.input", in_ch * t * hh * ww).astype(np.float32)
    ).reshape(1, in_ch, t, hh, ww)
    y = block(x)
    out.write(f"inline constexpr int64_t kH3Down3dInCh = {in_ch};\n")
    out.write(f"inline constexpr int64_t kH3Down3dOutCh = {out_ch};\n")
    out.write(f"inline constexpr int64_t kH3Down3dT = {t};\n")
    out.write(f"inline constexpr int64_t kH3Down3dH = {hh};\n")
    out.write(f"inline constexpr int64_t kH3Down3dW = {ww};\n")
    out.write(f"inline constexpr int64_t kH3Down3dOutT = {int(y.shape[2])};\n")
    out.write(f"inline constexpr int64_t kH3Down3dOutH = {int(y.shape[3])};\n")
    out.write(f"inline constexpr int64_t kH3Down3dOutW = {int(y.shape[4])};\n\n")
    emit_f32(out, "kH3Down3dGolden", y.reshape(-1))


def emit_encoder_fcn3d(out, src) -> None:
    """Section: the whole EncoderFCN3D level loop.

    conv_in -> per level [num_res_blocks x ResnetBlock3D, then an optional
    Downsample3D or a 1x1x1 channel-matching conv] -> GroupNorm -> SiLU -> conv_out.
    The real config is ch 128, ch_mult [1,2,2,4,4,8], space_down [2,2,2,2,1,1],
    time_down [1,2,2,1,1,1], num_res_blocks 2, embed_dim 24; only the sizes shrink.
    """
    cnn = load_bundle(src, "vae_cnn")
    ch = 32
    ch_mult = [1, 2]
    space_down = [2, 1]
    time_down = [2, 1]
    num_res_blocks = 1
    in_channels, z_channels = 3, 4
    t, hh, ww = 3, 4, 4

    enc = cnn.EncoderFCN3D(
        ch=ch, ch_mult=ch_mult, space_down=space_down, time_down=time_down,
        num_res_blocks=num_res_blocks, in_channels=in_channels, z_channels=z_channels,
        double_z=False, padding_mode="reflect", causal=True,
    ).eval()
    state = enc.state_dict()
    for name, tensor in state.items():
        scale, offset = (0.1, 0.0)
        # NOTE "norm" without a leading dot: `norm_out.weight` is a group-norm gain
        # too, and an earlier `".norm" in name` test silently missed it.
        if "norm" in name and name.endswith(".weight") and tensor.dim() == 1:
            scale, offset = 0.1, 1.0
        elif name.endswith(".bias"):
            scale = 0.05
        state[name] = torch.from_numpy(
            (h3_rand("encfcn." + name, tensor.numel()) * scale + offset).astype(np.float32)
        ).reshape(tensor.shape)
    enc.load_state_dict(state, strict=True)

    x = torch.from_numpy(
        h3_rand("encfcn.input", in_channels * t * hh * ww).astype(np.float32)
    ).reshape(1, in_channels, t, hh, ww)
    y = enc(x)

    out.write(f"inline constexpr int64_t kH3EncFcnCh = {ch};\n")
    out.write(f"inline constexpr int64_t kH3EncFcnLevels = {len(ch_mult)};\n")
    out.write(f"inline constexpr int64_t kH3EncFcnResBlocks = {num_res_blocks};\n")
    out.write(f"inline constexpr int64_t kH3EncFcnInCh = {in_channels};\n")
    out.write(f"inline constexpr int64_t kH3EncFcnZCh = {z_channels};\n")
    out.write(f"inline constexpr int64_t kH3EncFcnT = {t};\n")
    out.write(f"inline constexpr int64_t kH3EncFcnH = {hh};\n")
    out.write(f"inline constexpr int64_t kH3EncFcnW = {ww};\n")
    out.write(f"inline constexpr int64_t kH3EncFcnOutT = {int(y.shape[2])};\n")
    out.write(f"inline constexpr int64_t kH3EncFcnOutH = {int(y.shape[3])};\n")
    out.write(f"inline constexpr int64_t kH3EncFcnOutW = {int(y.shape[4])};\n\n")
    emit_i64(out, "kH3EncFcnChMult", ch_mult)
    emit_i64(out, "kH3EncFcnSpaceDown", space_down)
    emit_i64(out, "kH3EncFcnTimeDown", time_down)
    emit_f32(out, "kH3EncFcnGolden", y.reshape(-1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h3-vae-source", required=True, type=Path,
                        help="the checkpoint's FL2VA/video_vae directory (.py files only)")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    src = args.h3_vae_source.expanduser().resolve()
    if not (src / "base_module.py").is_file():
        raise SystemExit(f"missing base_module.py in {src}")
    bm = load_bundle(src, "base_module")
    torch.set_grad_enabled(False)

    # The shipped decoder blocks: RMSNorm (affine), RMS qk-norm WITHOUT affine,
    # gated SiLU feed-forward, and two learned residual scales -- exactly the
    # parameter set the real manifest shows.
    block = bm.TransformerBlock(
        heads=HEADS, dim_head=DIM_HEAD, embed_dim=DIM,
        norm_type="rms_norm", qk_norm_type="rms_norm", qk_norm_affine=False,
        ffn_activation_fn="silu", ffn_use_gated=True, use_scale=True, bias=True,
    ).eval()

    state = block.state_dict()
    for name, tensor in state.items():
        scale, offset = (0.1, 0.0)
        if name.endswith("norm1.weight") or name.endswith("norm2.weight"):
            scale, offset = 0.1, 1.0
        elif name in ("scale1", "scale2"):
            scale, offset = 0.3, 0.0
        elif name.endswith(".bias"):
            scale = 0.05
        values = h3_rand("videovae." + name, tensor.numel()) * scale + offset
        state[name] = torch.from_numpy(values.astype(np.float32)).reshape(tensor.shape)
    block.load_state_dict(state, strict=True)

    x = torch.from_numpy(h3_rand("videovae.input", SEQ * DIM).astype(np.float32)).reshape(1, SEQ, DIM)
    y = block(x)
    inner = block.ff.w1.weight.shape[0] // 2

    # --- the WHOLE ViT3DDecoder (surround + block stack), reduced ---
    # Real hyperparameters (FL2VA/video_vae/source/config.json :: vit_decoder_kwargs):
    # heads 32, dim_head 64, num_layers 36, rms_norm affine, qk rms_norm WITHOUT
    # affine, gated SiLU, rope_theta 100.0, rope_dim_ratio 0.75. Only the sizes shrink.
    vv = load_bundle(src, "vae_vit")
    dec = vv.ViT3DDecoder(
        patch_size=DEC_PATCH, patch_size_t=DEC_PATCH_T, t_causal=False,
        in_channels=DEC_IN_CH, out_channels=DEC_OUT_CH, num_layers=DEC_LAYERS,
        heads=HEADS, dim_head=DIM_HEAD, norm_type="rms_norm", norm_affine=True,
        qk_norm_type="rms_norm", qk_norm_affine=False, ffn_activation_fn="silu",
        ffn_use_gated=True, rope_theta=100.0, rope_dim_ratio=0.75, bias=True, eps=1e-5,
        num_register_tokens=DEC_REGISTER_TOKENS,
    ).eval()
    dstate = dec.state_dict()
    for name, tensor in dstate.items():
        scale, offset = (0.1, 0.0)
        if name.endswith("norm1.weight") or name.endswith("norm2.weight") or name.endswith("norm_out.weight"):
            scale, offset = 0.1, 1.0
        elif name.endswith("scale1") or name.endswith("scale2"):
            scale, offset = 0.3, 0.0
        elif name.endswith(".bias"):
            scale = 0.05
        elif name in ("mask_token",):
            scale = 0.0  # unused at inference (mask_prob 0)
        dstate[name] = torch.from_numpy(
            (h3_rand("videovae.dec." + name, tensor.numel()) * scale + offset).astype(np.float32)
        ).reshape(tensor.shape)
    dec.load_state_dict(dstate, strict=True)

    latent = torch.from_numpy(
        h3_rand("videovae.dec.input", DEC_IN_CH * DEC_T * DEC_H * DEC_W).astype(np.float32)
    ).reshape(1, DEC_IN_CH, DEC_T, DEC_H, DEC_W)
    frames = dec(latent)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        out.write(
            "// GENERATED by scripts/gen-minimax-h3-video-vae-goldens.py — DO NOT EDIT BY HAND.\n"
            "//\n"
            "// MiniMax-H3 VIDEO VAE decoder TransformerBlock goldens, produced by executing the\n"
            "// CHECKPOINT'S OWN remote-code module (FL2VA/video_vae/base_module.py) at reduced\n"
            "// dimensions. The shipped decoder is 36 of these blocks. The remote code is not\n"
            "// vendored here; see the generator's docstring. Weights come from the shared H3Rand\n"
            "// stream, so no weight byte is checked in.\n"
            "#pragma once\n\n#include <cstdint>\n\nnamespace vllm_test {\n\n"
        )
        out.write(f"inline constexpr int64_t kH3VideoVaeBlockHeads = {HEADS};\n")
        out.write(f"inline constexpr int64_t kH3VideoVaeBlockDimHead = {DIM_HEAD};\n")
        out.write(f"inline constexpr int64_t kH3VideoVaeBlockDim = {DIM};\n")
        out.write(f"inline constexpr int64_t kH3VideoVaeBlockSeq = {SEQ};\n")
        out.write(f"inline constexpr int64_t kH3VideoVaeBlockFfInner = {int(inner)};\n\n")
        emit_f32(out, "kH3VideoVaeBlockGolden", y.reshape(-1))

        out.write(f"inline constexpr int64_t kH3VideoVaeDecLayers = {DEC_LAYERS};\n")
        out.write(f"inline constexpr int64_t kH3VideoVaeDecInCh = {DEC_IN_CH};\n")
        out.write(f"inline constexpr int64_t kH3VideoVaeDecOutCh = {DEC_OUT_CH};\n")
        out.write(f"inline constexpr int64_t kH3VideoVaeDecPatch = {DEC_PATCH};\n")
        out.write(f"inline constexpr int64_t kH3VideoVaeDecPatchT = {DEC_PATCH_T};\n")
        out.write(f"inline constexpr int64_t kH3VideoVaeDecRegisterTokens = {DEC_REGISTER_TOKENS};\n")
        out.write(f"inline constexpr int64_t kH3VideoVaeDecT = {DEC_T};\n")
        out.write(f"inline constexpr int64_t kH3VideoVaeDecH = {DEC_H};\n")
        out.write(f"inline constexpr int64_t kH3VideoVaeDecW = {DEC_W};\n")
        out.write(f"inline constexpr int64_t kH3VideoVaeDecFrameT = {int(frames.shape[2])};\n")
        out.write(f"inline constexpr int64_t kH3VideoVaeDecFrameH = {int(frames.shape[3])};\n")
        out.write(f"inline constexpr int64_t kH3VideoVaeDecFrameW = {int(frames.shape[4])};\n\n")
        emit_f32(out, "kH3VideoVaeDecoderGolden", frames.reshape(-1))
        emit_tiling(out)
        emit_resnet3d(out, src)
        emit_downsample3d(out, src)
        emit_encoder_fcn3d(out, src)
        out.write("}  // namespace vllm_test\n")
    print(f"wrote {args.out} (ff inner {int(inner)})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
