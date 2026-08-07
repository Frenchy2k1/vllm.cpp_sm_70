#!/usr/bin/env python3
# P5 (MODEL-AUDIO-PARAKEET-TRANSDUCER): dump a HuggingFace `ParakeetForRNNT` /
# `ParakeetForTDT` ORACLE fixture for the C++ transducer gate.
#
# Upstream oracle: transformers `main`
#   transformers/models/parakeet/modeling_parakeet.py
#     ParakeetRNNTDecoder:831       (LSTM prediction network, forward :846-876)
#     ParakeetRNNTJointNetwork:879  (forward :888-894)
#     ParakeetForRNNT:922           (get_audio_features :938-950, forward :954-1032)
#     ParakeetTDTJointNetwork:1035  (the duration head, :1042-1044)
#     ParakeetForTDT:1052           (forward :1063-1142)
#   transformers/models/parakeet/generation_parakeet.py
#     ParakeetRNNTDecoderCache:23        (lazy_initialization :31-58, update :60-82)
#     ParakeetRNNTGenerationMixin:125    (_update_model_kwargs_for_generation :141-163,
#                                         _prepare_model_inputs :192-228,
#                                         prepare_inputs_for_generation :233-248,
#                                         generate :250-268)
#     ParakeetTDTGenerationMixin:271     (_update_model_kwargs_for_generation :280-299)
#   transformers/models/parakeet/configuration_parakeet.py
#     ParakeetRNNTConfig:136, ParakeetTDTConfig:188
#
# **The correction this row rests on.** An earlier revision of
# .agents/specs/parakeet-conformer-encoder.md recorded that the transducer had
# "NO upstream in either vLLM or HF transformers". That was true of the LOCALLY
# INSTALLED transformers 5.3.0, which ships only `ParakeetForCTC`, and it is
# FALSE of current upstream: `main` implements the whole transducer stack at the
# lines cited above. So this is mirror work with citable file:line, not a product
# deviation. The spike is corrected in the same change that adds this script.
#
# Like the P4 dump this is a SMALL, SEEDED, RANDOMLY-INITIALISED model: random
# weights exercise the module MATH (every gate of the LSTM, the joint, the
# blank-skip cache path, the frame advance) exactly as pretrained ones would,
# they just make the decoded text meaningless. So the gate is on the DISCRETE
# decode (emitted sequence, per-step durations) plus the continuous decoder and
# joint tensors, and claims nothing about a transcript. The pretrained transcript
# claim is p5_parakeet_transducer_reference.py's job, on real checkpoints.
#
# TRACED, not read (T0): the greedy loop's per-step decoder input SHAPE is
# recorded by a forward hook rather than inferred from `generate()`'s source,
# because `ParakeetRNNTDecoderCache` is not a `past_key_values` cache and the
# base `prepare_inputs_for_generation` slicing rules are not obvious. The
# manifest records what actually ran; the C++ gate asserts it.
#
#   python3 scripts/mm/p5_parakeet_transducer_oracle_dump.py \
#       --out tests/vllm/models/fixtures/parakeet_transducer
import argparse
import json
import os

import numpy as np
import torch

import transformers
from transformers import (
    ParakeetEncoderConfig,
    ParakeetForRNNT,
    ParakeetForTDT,
    ParakeetRNNTConfig,
    ParakeetTDTConfig,
)

SEED = 20260807


def tiny_encoder_config() -> ParakeetEncoderConfig:
    """Same tiny encoder the P4 CTC dump uses, so the transducer fixture isolates
    the NEW modules: the encoder path is already gated by P4."""
    return ParakeetEncoderConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        hidden_act="silu",
        attention_bias=True,
        convolution_bias=True,
        conv_kernel_size=5,
        subsampling_factor=4,
        subsampling_conv_channels=8,
        num_mel_bins=16,
        subsampling_conv_kernel_size=3,
        subsampling_conv_stride=2,
        dropout=0.0,
        dropout_positions=0.0,
        layerdrop=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        max_position_embeddings=5000,
        scale_input=True,
    )


def build(kind: str, init_std: float, blank_delta: float, dur0_delta: float):
    """A deliberately small transducer that still exercises every structural axis:
    >1 LSTM layer (so the stack carries per-layer state), a decoder width that
    differs from the encoder width (so `encoder_projector` cannot be an identity),
    and for TDT a duration set with a 0 in it (the case the blank guard at
    generation_parakeet.py:292 exists for)."""
    torch.manual_seed(SEED)
    enc = tiny_encoder_config()
    common = dict(
        encoder_config=enc,
        vocab_size=17,
        decoder_hidden_size=24,
        num_decoder_layers=2,
        hidden_act="relu",
        max_symbols_per_step=3,
        blank_token_id=16,
        pad_token_id=0,
    )
    if kind == "rnnt":
        model = ParakeetForRNNT(ParakeetRNNTConfig(**common))
    else:
        model = ParakeetForTDT(ParakeetTDTConfig(durations=[0, 1, 2, 3], **common))
    model.eval()

    # SHAPING THE FIXTURE, and why it is necessary. `ParakeetPreTrainedModel.
    # _init_weights` draws from N(0, initializer_range) with initializer_range
    # 0.02, so a config-constructed transducer has near-zero activations: the
    # joint's `relu(encoder + decoder)` (:893) collapses to almost nothing and
    # every logit is essentially `head.bias`, which is initialised to zero. The
    # greedy argmax is then decided by float noise, and the decode degenerates to
    # "the same token at every step": which walks ONE branch of the loop and
    # gates almost nothing.
    #
    # So the transducer sub-modules are re-drawn at a usable scale, and two bias
    # offsets steer the decode. Both are chosen by the search in main() and
    # recorded in the manifest, and the resulting branch mix is asserted by the
    # C++ gate, so a future regeneration cannot silently weaken the fixture:
    #   blank_delta: buys a MIX of blank and non-blank emissions, so the cache's
    #     blank-skip fast path (gen:851-855: on a blank the LSTM does NOT run and
    #     the state does NOT advance) and the ordinary LSTM update BOTH fire, and
    #     for RNN-T so does the max_symbols_per_step forced advance (gen:152-157).
    #   dur0_delta: TDT only: raises the duration-0 column so the decode emits
    #     several symbols at ONE encoder frame, and so the "blank with duration 0
    #     is forced to 1" guard (gen:291-292) has something to guard.
    generator = torch.Generator().manual_seed(SEED + 41)
    with torch.no_grad():
        for module in (model.decoder, model.joint, model.encoder_projector):
            for param in module.parameters():
                param.copy_(torch.randn(param.shape, generator=generator) * init_std)
        model.joint.head.bias[common["blank_token_id"]] += blank_delta
        if kind == "tdt":
            model.joint.head.bias[model.config.vocab_size] += dur0_delta
    # A config-constructed model has no decoder_start_token_id; every published
    # checkpoint sets it to the BLANK (nvidia/parakeet-rnnt-0.6b
    # generation_config.json: decoder_start_token_id 1024 == blank_token_id;
    # nvidia/parakeet-tdt-0.6b-v3: 8192 == blank_token_id), which is also what
    # NeMo's greedy transducer feeds at u=0. Mirror that here.
    model.generation_config.decoder_start_token_id = common["blank_token_id"]
    if kind == "tdt":
        # The published TDT generation_config suppresses the DURATION columns so
        # `generate()`'s argmax over the full joint row can only select a real
        # token (nvidia/parakeet-tdt-0.6b-v3 generation_config.json:
        # suppress_tokens [8193..8197] == vocab_size .. vocab_size+len(durations)-1).
        # Without it the greedy search could "emit" a duration index and then
        # index the embedding out of range.
        model.generation_config.suppress_tokens = list(
            range(model.config.vocab_size, model.config.vocab_size + len(model.config.durations))
        )
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    manifest = {
        "provenance": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "pretrained": False,
            "seed": SEED,
        },
        "models": {},
    }

    frames = 61

    def features_for(model):
        torch.manual_seed(SEED + 1)
        n = model.config.encoder_config.num_mel_bins
        return (
            torch.randn(1, frames, n, dtype=torch.float32),
            torch.ones(1, frames, dtype=torch.long),
        )

    def branch_coverage(kind, model):
        """Score a candidate by how much of the greedy loop it actually walks.
        REQUIRED: both blank and non-blank emissions, and for RNN-T at least one
        max_symbols_per_step forced advance (a non-blank step that still moves
        the frame pointer). PREFERRED, in order: more distinct per-step advances,
        then a non-blank step with advance 0 (several symbols at one frame), then
        a balanced blank/non-blank split."""
        features, mask = features_for(model)
        out = model.generate(input_features=features, attention_mask=mask)
        seq = out.sequences[0].tolist()[1:]
        dur = out.durations[0].tolist()[1:]
        blank = model.config.blank_token_id
        n_blank = sum(1 for t in seq if t == blank)
        n_tok = len(seq) - n_blank
        forced = sum(1 for t, d in zip(seq, dur) if t != blank and d > 0)
        held = sum(1 for t, d in zip(seq, dur) if t != blank and d == 0)
        advances = sorted(set(dur))
        ok = n_blank > 0 and n_tok > 0 and (kind == "tdt" or forced > 0)
        score = (len(advances), 1 if held else 0, min(n_blank, n_tok))
        stats = {
            "blank_emissions": n_blank,
            "token_emissions": n_tok,
            "frame_advancing_tokens": forced,
            "tokens_held_at_frame": held,
            "distinct_advances": advances,
        }
        return ok, score, stats

    for kind in ("rnnt", "tdt"):
        best = None
        # A small grid around the region a wider sweep found usable: the search
        # is here so the choice is reproducible and checkable, not so it explores.
        for init_std in (0.2, 0.25):
            for blank_delta in (1.5, 2.0, 2.5):
                for dur0_delta in (0.5, 1.0, 1.5) if kind == "tdt" else (0.0,):
                    candidate = build(kind, init_std, blank_delta, dur0_delta)
                    ok, score, stats = branch_coverage(kind, candidate)
                    if ok and (best is None or score > best[0]):
                        best = (score, (init_std, blank_delta, dur0_delta), candidate, stats)
        if best is None:
            raise SystemExit(f"{kind}: no shaping produced a decode that walks every branch")
        _, shaping, model, chosen_stats = best

        cfg = model.config
        enc = cfg.encoder_config
        features, attention_mask = features_for(model)

        # TRACE: record the decoder's actual per-step input shape and id.
        steps = []

        def hook(_module, inputs, _output, steps=steps):
            ids = inputs[0]
            steps.append(
                {
                    "input_ids_shape": list(ids.shape),
                    "input_ids": ids.reshape(-1).tolist(),
                }
            )

        handle = model.decoder.register_forward_hook(hook)
        out = model.generate(input_features=features, attention_mask=attention_mask)
        handle.remove()

        # The projected encoder output the greedy loop indexes
        # (ParakeetForRNNT.get_audio_features:938-950 -> pooler_output).
        audio = model.get_audio_features(input_features=features, attention_mask=attention_mask)
        pooled = audio.pooler_output[0]  # [T, decoder_hidden_size]

        # Reference decoder outputs and joint logits for a FIXED token walk, so a
        # C++ failure localises to a module instead of only to the final ids.
        walk = [cfg.blank_token_id, 1, 1, 2, cfg.blank_token_id, 3]
        cache_cls = type(model.decoder)  # only for the docstring's sake
        del cache_cls
        from transformers.models.parakeet.generation_parakeet import ParakeetRNNTDecoderCache

        cache = ParakeetRNNTDecoderCache(cfg)
        dec_steps, joint_steps = [], []
        for i, tok in enumerate(walk):
            dec = model.decoder(torch.tensor([[tok]], dtype=torch.long), cache=cache)
            dec_steps.append(dec[0, -1].clone())
            frame = pooled[min(i, pooled.shape[0] - 1)]
            joint_steps.append(
                model.joint(
                    decoder_hidden_states=dec[0, -1][None, None, None, :],
                    encoder_hidden_states=frame[None, None, None, :],
                ).reshape(-1).clone()
            )

        prefix = kind + "_"
        features.numpy().astype("<f4").tofile(os.path.join(args.out, prefix + "input_features.bin"))
        pooled.numpy().astype("<f4").tofile(os.path.join(args.out, prefix + "encoder_projected.bin"))
        torch.stack(dec_steps).numpy().astype("<f4").tofile(
            os.path.join(args.out, prefix + "decoder_steps.bin")
        )
        torch.stack(joint_steps).numpy().astype("<f4").tofile(
            os.path.join(args.out, prefix + "joint_steps.bin")
        )

        state = {k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}
        wdir = os.path.join(args.out, prefix + "weights")
        os.makedirs(wdir, exist_ok=True)
        for name, tensor in state.items():
            tensor.detach().cpu().numpy().astype("<f4").tofile(os.path.join(wdir, name + ".bin"))
        # The encoder half is P4's fixture shape; dump it under the same key map.
        edir = os.path.join(args.out, prefix + "encoder_weights")
        os.makedirs(edir, exist_ok=True)
        for name, tensor in model.state_dict().items():
            if name.startswith("encoder."):
                tensor.detach().cpu().numpy().astype("<f4").tofile(
                    os.path.join(edir, name + ".bin")
                )

        manifest["models"][kind] = {
            "architecture": type(model).__name__,
            "config": {
                "vocab_size": cfg.vocab_size,
                "decoder_hidden_size": cfg.decoder_hidden_size,
                "num_decoder_layers": cfg.num_decoder_layers,
                "hidden_act": cfg.hidden_act,
                "max_symbols_per_step": cfg.max_symbols_per_step,
                "blank_token_id": cfg.blank_token_id,
                "pad_token_id": cfg.pad_token_id,
                "durations": list(getattr(cfg, "durations", []) or []),
            },
            "encoder_config": {
                "hidden_size": enc.hidden_size,
                "num_hidden_layers": enc.num_hidden_layers,
                "num_attention_heads": enc.num_attention_heads,
                "num_key_value_heads": enc.num_key_value_heads,
                "intermediate_size": enc.intermediate_size,
                "hidden_act": enc.hidden_act,
                "attention_bias": enc.attention_bias,
                "convolution_bias": enc.convolution_bias,
                "conv_kernel_size": enc.conv_kernel_size,
                "subsampling_factor": enc.subsampling_factor,
                "subsampling_conv_channels": enc.subsampling_conv_channels,
                "num_mel_bins": enc.num_mel_bins,
                "subsampling_conv_kernel_size": enc.subsampling_conv_kernel_size,
                "subsampling_conv_stride": enc.subsampling_conv_stride,
                "max_position_embeddings": enc.max_position_embeddings,
                "scale_input": enc.scale_input,
            },
            "generation_config": {
                "decoder_start_token_id": model.generation_config.decoder_start_token_id,
                "eos_token_id": model.generation_config.eos_token_id,
                "suppress_tokens": model.generation_config.suppress_tokens,
            },
            "attn_implementation": model.encoder.config._attn_implementation,
            "shaping": {
                "init_std": shaping[0],
                "blank_delta": shaping[1],
                "dur0_delta": shaping[2],
            },
            # The proof that this fixture actually walks every branch of the
            # greedy loop. The C++ gate asserts these, so a regeneration that
            # degenerated to a single branch would FAIL rather than pass quietly.
            "branch_coverage": chosen_stats,
            "frames": frames,
            "valid_frames": frames,
            "encoder_frames": int(pooled.shape[0]),
            "walk": walk,
            "decoder_trace": steps,
            "sequences": [int(t) for t in out.sequences[0].tolist()],
            "step_durations": [int(d) for d in out.durations[0].tolist()],
        }

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    for kind, m in manifest["models"].items():
        print(kind, m["architecture"], "seq", m["sequences"], "dur", m["step_durations"])
        print("  decoder input shapes:", {tuple(s["input_ids_shape"]) for s in m["decoder_trace"]})


if __name__ == "__main__":
    main()
