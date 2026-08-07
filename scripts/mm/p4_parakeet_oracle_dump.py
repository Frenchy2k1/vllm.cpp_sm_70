#!/usr/bin/env python3
# P4 (MODEL-AUDIO-PARAKEET-ENCODER) — dump a HuggingFace `ParakeetForCTC` ORACLE
# fixture for the C++ encoder + CTC gate.
#
# Upstream oracle: transformers 5.3.0
#   transformers/models/parakeet/modeling_parakeet.py
#     ParakeetEncoderRelPositionalEncoding:51, ParakeetEncoderFeedForward:101,
#     ParakeetEncoderConvolutionModule:116, ParakeetEncoderAttention:259,
#     ParakeetEncoderSubsamplingConv2D:357, ParakeetEncoderBlock:426,
#     ParakeetEncoder:549, ParakeetForCTC:675
#   transformers/models/parakeet/tokenization_parakeet.py
#     ParakeetTokenizer._decode:28-49 (the CTC group-then-drop-blank collapse)
# This is the same module vLLM itself runs: vllm/model_executor/models/
# parakeet.py:37,62 imports and instantiates `transformers.ParakeetEncoder`.
#
# The dump is a SMALL, seeded, randomly-initialised model (no pretrained
# checkpoint is downloaded — see the honesty note in
# tests/vllm/models/test_parakeet_ctc_engine.cpp). Random weights exercise the
# module MATH exactly as pretrained ones would; they just make the decoded text
# meaningless, so the gate is on LOGITS + TOKEN IDS, never on a transcript.
#
# Everything is float32 and little-endian; shapes live in manifest.json.
#
#   python3 scripts/mm/p4_parakeet_oracle_dump.py \
#       --out tests/vllm/models/fixtures/parakeet
import argparse
import json
import os

import numpy as np
import torch

from transformers import ParakeetCTCConfig, ParakeetEncoderConfig, ParakeetForCTC

SEED = 20260806


def tiny_config() -> ParakeetCTCConfig:
    """A deliberately small config that still exercises EVERY structural axis:
    two subsampling stages (so the depthwise+pointwise stage runs), a multi-head
    attention with head_dim > 1, an odd conformer kernel, and >1 encoder block."""
    enc = ParakeetEncoderConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        hidden_act="silu",
        attention_bias=True,
        convolution_bias=True,
        conv_kernel_size=5,
        subsampling_factor=4,  # -> 2 subsampling stages
        subsampling_conv_channels=8,
        num_mel_bins=16,
        subsampling_conv_kernel_size=3,
        subsampling_conv_stride=2,
        dropout=0.0,
        dropout_positions=0.0,
        layerdrop=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        max_position_embeddings=512,
        scale_input=True,
    )
    # configuration_parakeet.py:209-212 only accepts a dict (or None) here.
    return ParakeetCTCConfig(vocab_size=24, pad_token_id=23, encoder_config=enc.to_dict())


def ctc_collapse(ids: list[int], blank: int) -> list[int]:
    """tokenization_parakeet.py:38-42 — group consecutive, then drop the blank."""
    grouped: list[int] = []
    for t in ids:
        if not grouped or grouped[-1] != t:
            grouped.append(t)
    return [t for t in grouped if t != blank]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)

    torch.manual_seed(SEED)
    torch.set_grad_enabled(False)

    cfg = tiny_config()
    model = ParakeetForCTC(cfg).to(torch.float32).eval()

    # HF's default init leaves EVERY nn.Linear/nn.Conv bias at 0, every
    # LayerNorm/BatchNorm at weight=1 bias=0, and BatchNorm running_mean=0
    # running_var=1 — i.e. exactly the values at which a DROPPED bias term, a
    # dropped norm affine or a dropped batch-norm statistic is invisible. Perturb
    # all of them so the fixture actually constrains those terms.
    gen = torch.Generator().manual_seed(SEED + 1)

    def rand(shape, scale):
        return torch.randn(shape, generator=gen) * scale

    for module in model.modules():
        if isinstance(module, (torch.nn.LayerNorm, torch.nn.BatchNorm1d)):
            module.weight.copy_(1.0 + rand(module.weight.shape, 0.2))
            module.bias.copy_(rand(module.bias.shape, 0.2))
            continue
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
            # HF inits every weight at std=initializer_range (0.02), which makes
            # the whole network near-degenerate: attention is ~uniform and the
            # CTC argmax is a constant. Re-init at torch's own fan-in scale so
            # the reference activations and the greedy path are non-trivial.
            fan_in = int(module.weight.numel() // module.weight.shape[0])
            module.weight.copy_(rand(module.weight.shape, fan_in**-0.5))
        if getattr(module, "bias", None) is not None and isinstance(
            module.bias, torch.nn.Parameter
        ):
            module.bias.copy_(rand(module.bias.shape, 0.2))
    for name, buf in model.named_buffers():
        if name.endswith("running_mean"):
            buf.copy_(rand(buf.shape, 0.3))
        elif name.endswith("running_var"):
            buf.copy_(torch.rand(buf.shape, generator=gen) * 0.8 + 0.4)

    enc_cfg = cfg.encoder_config
    batch, n_frames = 2, 40
    feats = torch.randn(
        (batch, n_frames, enc_cfg.num_mel_bins),
        generator=torch.Generator().manual_seed(SEED + 2),
    ).to(torch.float32)
    # Row 0 is full length, row 1 is PADDED — so the subsampling mask, the
    # attention key mask and the convolution-module mask are all exercised.
    lengths = [n_frames, 27]
    mask = torch.zeros((batch, n_frames), dtype=torch.long)
    for i, n in enumerate(lengths):
        mask[i, :n] = 1
    feats = feats * mask.unsqueeze(-1)

    stages: dict[str, torch.Tensor] = {}
    enc = model.encoder
    enc.subsampling.register_forward_hook(
        lambda m, i, o: stages.__setitem__("subsampling", o.detach().clone())
    )
    enc.encode_positions.register_forward_hook(
        lambda m, i, o: stages.__setitem__("pos_embed", o.detach().clone())
    )
    for li, layer in enumerate(enc.layers):
        layer.register_forward_hook(
            lambda m, i, o, li=li: stages.__setitem__(
                f"block{li}", (o[0] if isinstance(o, tuple) else o).detach().clone()
            )
        )
    # Layer-0 sub-stages, so a mismatch localises to a module rather than to
    # "the encoder". Cheap: [B, T', hidden] each.
    enc.layers[0].feed_forward1.register_forward_hook(
        lambda m, i, o: stages.__setitem__("l0_ff1", o.detach().clone())
    )
    enc.layers[0].self_attn.register_forward_hook(
        lambda m, i, o: stages.__setitem__("l0_attn", o[0].detach().clone())
    )
    enc.layers[0].conv.register_forward_hook(
        lambda m, i, o: stages.__setitem__("l0_conv", o.detach().clone())
    )

    outputs = model(input_features=feats, attention_mask=mask, return_dict=True)
    logits = outputs.logits  # [B, T', V]
    hidden = enc(input_features=feats, attention_mask=mask).last_hidden_state

    out_mask = model._get_output_attention_mask(mask, target_length=logits.shape[1])
    sequences = logits.argmax(dim=-1)
    sequences = sequences.masked_fill(~out_mask, cfg.pad_token_id)

    # ---- weights -------------------------------------------------------------
    wdir = os.path.join(out, "weights")
    os.makedirs(wdir, exist_ok=True)
    n_w = 0
    for name, p in model.state_dict().items():
        if name.endswith("num_batches_tracked"):
            continue
        p.to(torch.float32).contiguous().numpy().tofile(
            os.path.join(wdir, name.replace("/", "_") + ".bin")
        )
        n_w += 1

    # ---- inputs / references -------------------------------------------------
    def save(name: str, t: torch.Tensor) -> list[int]:
        a = t.detach().to(torch.float32).contiguous().numpy()
        a.tofile(os.path.join(out, name + ".bin"))
        return list(a.shape)

    shapes = {
        "input_features": save("input_features", feats),
        "subsampling_out": save("subsampling_out", stages["subsampling"]),
        "pos_embed": save("pos_embed", stages["pos_embed"]),
        "l0_ff1": save("l0_ff1", stages["l0_ff1"]),
        "l0_attn": save("l0_attn", stages["l0_attn"]),
        "l0_conv": save("l0_conv", stages["l0_conv"]),
        "block0_out": save("block0_out", stages["block0"]),
        "block1_out": save("block1_out", stages["block1"]),
        "last_hidden_state": save("last_hidden_state", hidden),
        "logits": save("logits", logits),
    }
    np.asarray(sequences.numpy(), dtype=np.int32).tofile(
        os.path.join(out, "greedy_ids.bin")
    )
    shapes["greedy_ids"] = list(sequences.shape)

    collapsed = [
        ctc_collapse([int(t) for t in row], cfg.pad_token_id)
        for row in sequences.tolist()
    ]

    manifest = {
        "provenance": {
            "transformers": __import__("transformers").__version__,
            "torch": torch.__version__,
            "module": "transformers/models/parakeet/modeling_parakeet.py",
            "classes": {
                "ParakeetEncoderRelPositionalEncoding": 51,
                "ParakeetEncoderFeedForward": 101,
                "ParakeetEncoderConvolutionModule": 116,
                "ParakeetEncoderAttention": 259,
                "ParakeetEncoderSubsamplingConv2D": 357,
                "ParakeetEncoderBlock": 426,
                "ParakeetEncoder": 549,
                "ParakeetForCTC": 675,
            },
            "seed": SEED,
            "pretrained": False,
            # T0 "trace the execution, not just the code": ParakeetEncoderAttention
            # picks its path at RUNTIME (modeling_parakeet.py:306-308). This is
            # the path that ACTUALLY ran for this dump. `sdpa` returns exactly
            # ZERO for a fully-masked query row where `eager` would return NaN,
            # which is the behaviour vt::AttentionRelPos documents and mirrors.
            "attn_implementation": model.config._attn_implementation,
        },
        "config": {
            "hidden_size": enc_cfg.hidden_size,
            "num_hidden_layers": enc_cfg.num_hidden_layers,
            "num_attention_heads": enc_cfg.num_attention_heads,
            "num_key_value_heads": enc_cfg.num_key_value_heads,
            "intermediate_size": enc_cfg.intermediate_size,
            "hidden_act": enc_cfg.hidden_act,
            "attention_bias": enc_cfg.attention_bias,
            "convolution_bias": enc_cfg.convolution_bias,
            "conv_kernel_size": enc_cfg.conv_kernel_size,
            "subsampling_factor": enc_cfg.subsampling_factor,
            "subsampling_conv_channels": enc_cfg.subsampling_conv_channels,
            "num_mel_bins": enc_cfg.num_mel_bins,
            "subsampling_conv_kernel_size": enc_cfg.subsampling_conv_kernel_size,
            "subsampling_conv_stride": enc_cfg.subsampling_conv_stride,
            "max_position_embeddings": enc_cfg.max_position_embeddings,
            "scale_input": enc_cfg.scale_input,
            "vocab_size": cfg.vocab_size,
            "pad_token_id": cfg.pad_token_id,
        },
        "batch": batch,
        "input_lengths": lengths,
        "shapes": shapes,
        "collapsed_ids": collapsed,
        "num_weight_tensors": n_w,
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"wrote {n_w} weight tensors + refs to {out}", flush=True)


if __name__ == "__main__":
    main()
