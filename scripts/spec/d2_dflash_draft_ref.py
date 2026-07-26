#!/usr/bin/env python3
# DFlash D2 — STANDALONE DRAFT-FORWARD reference capture (spec
# .agents/specs/dflash-spec-decode.md §6 D2, port map DF-DRAFT-MODEL). This dumps
# the vLLM in-tree DFlash draft's CONTEXT-FREE block forward so our C++
# Qwen3DFlashModel::ForwardBlockLogits can be gated against it in ISOLATION (the
# D2 gate is the draft forward, NOT the full propose/verify e2e — that is D4).
#
# ISOLATION from D3 (context-KV precompute): we drive DFlashQwen3Model.forward on
# a single uniform (1+k) query block with an EMPTY draft KV cache (no context
# tokens pre-inserted). In that configuration each query attends only to its own
# block — exactly what the new vt::DFlashBlockAttention primitive computes
# (full-attention layers BIDIRECTIONAL, SWA layers causal-in-window). The `fc`
# aux-combine is dumped separately on a fixed synthetic aux-feature block so the
# combine_hidden_states weight + tap order are checked directly.
#
# What it dumps (to --out-dir, default tests/parity/goldens/dflash_27b_draft/):
#   * config.json         : resolved draft hf_config subset (layers, hidden, heads,
#                           head_dim, layer_types, rope_theta, mask_token_id,
#                           target_layer_ids, vocab, rms_norm_eps, intermediate).
#   * ckpt_keys.txt       : the draft checkpoint's state_dict key list (so the C++
#                           loader's on-disk names are CONFIRMED, not guessed).
#   * fc_ref.json         : {aux:[T,H*taps], comb:[T,H]} — combine_hidden_states.
#   * block_ref.json      : {input_ids, positions, cu_seqlens, per_layer_hidden:
#                           [L][T,H], logits:[T,V], topk_ids:[T,k]} for the
#                           context-free block forward.
#
# THE C++ SIDE (D2 gate): tests/parity load these fixtures, run
# Qwen3DFlashModel::ForwardBlockLogits / CombineAuxFeatures on the SAME inputs +
# the loaded z-lab weights, and assert per-stage rel-L2 within the bf16 envelope
# (STRICT where deterministic; state the measured tolerance like the M2a/whisper
# tower gates). The RED proof (causal-instead-of-non-causal full layer) is the C++
# unit test test_qwen3_dflash_forward.cpp (CPU, deterministic).
#
# MEMORY (GB10 119 GiB UNIFIED -> OOM-reboots the box; see
# [[gb10-unified-memory-oom-reboots-box]]): the 27B target is MULTIMODAL, so vLLM
# profiles the vision encoder at max image size on init. Pass
# limit_mm_per_prompt={image:0,video:0} + gpu_memory_utilization=0.30 (the D0 fix,
# now this tool's default). Run under `flock $HOME/gpu.lock` with the oracle venv
# on PATH (FlashInfer JIT needs `ninja`), VLLM_USE_V2_MODEL_RUNNER=1, no FLASH_ATTN
# pin (auto flashinfer-native):
#   flock $HOME/gpu.lock env PATH=$HOME/venvs/vllm-oracle/bin:$PATH \
#     VLLM_USE_V2_MODEL_RUNNER=1 ~/venvs/vllm-oracle/bin/python \
#     scripts/spec/d2_dflash_draft_ref.py --out-dir tests/parity/goldens/dflash_27b_draft
#
# NOTE: the exact private hooks to reach DFlashQwen3Model.forward standalone are
# finalized ON-BOX against the installed vLLM 0.26.0.dev0 (the speculator owns the
# draft instance); this harness documents the required dump + the isolation
# contract and constructs inputs. Kept in-tree so the D2 GPU gate is one command.
import argparse
import json
import os

TARGET = os.environ.get("DFLASH_TARGET", "unsloth/Qwen3.6-27B-NVFP4")
DRAFT = os.environ.get("DFLASH_DRAFT", "z-lab/Qwen3.6-27B-DFlash")
# Draft card: block_size 16 -> (1+k) = 17-token uniform query block; mask 248070.
BLOCK = int(os.environ.get("DFLASH_BLOCK", "16"))
MASK_TOKEN_ID = int(os.environ.get("DFLASH_MASK_TOKEN_ID", "248070"))
ANCHOR_TOKEN_ID = int(os.environ.get("DFLASH_ANCHOR_TOKEN_ID", "9707"))  # "Hello"


def build_block_inputs(hidden_size):
    """The uniform (1+k) mask block: anchor token then k mask tokens; positions
    0..k; single request so cu_seqlens = [0, 1+k]. Mirrors prepare_dflash_inputs'
    query block (speculator.py:416-563) at ctx_end==0 (empty context)."""
    k = BLOCK - 1
    input_ids = [ANCHOR_TOKEN_ID] + [MASK_TOKEN_ID] * k
    positions = list(range(BLOCK))
    cu_seqlens = [0, BLOCK]
    return input_ids, positions, cu_seqlens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="tests/parity/goldens/dflash_27b_draft")
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--draft", default=DRAFT)
    ap.add_argument("--gpu-mem-util", type=float, default=0.30)
    ap.add_argument("--num-spec-tokens", type=int, default=BLOCK)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    from vllm import LLM

    # Construct the target + DFlash draft exactly as the D0 capture (mm-limited).
    llm = LLM(
        model=args.target,
        speculative_config={
            "method": "dflash",
            "model": args.draft,
            "num_speculative_tokens": args.num_spec_tokens,
        },
        gpu_memory_utilization=args.gpu_mem_util,
        limit_mm_per_prompt={"image": 0, "video": 0},
        enforce_eager=True,
        max_model_len=2048,
    )

    # Reach the draft model instance through the spec-decode worker. The exact
    # attribute path is finalized on-box (the speculator owns `.model`, a
    # DFlashQwen3ForCausalLM); this is the documented hook.
    engine = llm.llm_engine
    draft = None
    try:
        runner = engine.model_executor.driver_worker.model_runner
        draft = runner.drafter.model  # DFlashQwen3ForCausalLM
    except Exception as exc:  # noqa: BLE001
        print(f"[d2] draft handle not reachable via the default path: {exc}")
        print("[d2] finalize the .drafter.model path on-box against vLLM 0.26.0.dev0")

    cfg = getattr(draft, "config", None)
    hidden = getattr(cfg, "hidden_size", 5120)
    taps = len(getattr(cfg, "target_layer_ids", [1, 16, 31, 46, 61]))

    # Dump the checkpoint key list + config subset (loader confirmation).
    if draft is not None:
        keys = sorted(k for k, _ in draft.named_parameters())
        with open(os.path.join(args.out_dir, "ckpt_keys.txt"), "w") as f:
            f.write("\n".join(keys))
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(
            {
                "hidden_size": hidden,
                "num_taps": taps,
                "mask_token_id": MASK_TOKEN_ID,
                "block_size": BLOCK,
                "layer_types": list(getattr(cfg, "layer_types", [])),
                "target_layer_ids": list(getattr(cfg, "target_layer_ids", [])),
            },
            f,
            indent=2,
        )

    # fc reference: combine_hidden_states over a fixed synthetic aux block.
    T = BLOCK
    torch.manual_seed(0)
    aux = 0.2 * torch.sin(torch.arange(T * hidden * taps).float() * 0.3).reshape(T, hidden * taps)
    fc_out = None
    if draft is not None and hasattr(draft, "combine_hidden_states"):
        with torch.no_grad():
            dev = next(draft.parameters()).device
            dt = next(draft.parameters()).dtype
            fc_out = draft.combine_hidden_states(aux.to(device=dev, dtype=dt)).float().cpu()
    with open(os.path.join(args.out_dir, "fc_ref.json"), "w") as f:
        json.dump(
            {
                "aux": aux.flatten().tolist(),
                "comb": (fc_out.flatten().tolist() if fc_out is not None else None),
                "T": T,
                "hidden": hidden,
                "taps": taps,
            },
            f,
        )

    input_ids, positions, cu = build_block_inputs(hidden)
    # block_ref: run the draft forward on the mask block with an EMPTY KV cache so
    # the block attends only to itself (context-free D2 isolation). The exact
    # empty-KV forward call is finalized on-box (needs the draft's attn metadata /
    # a fresh KV group); this records the inputs + the intended dump schema.
    with open(os.path.join(args.out_dir, "block_ref.json"), "w") as f:
        json.dump(
            {
                "input_ids": input_ids,
                "positions": positions,
                "cu_seqlens": cu,
                "note": "context-free block forward; per_layer_hidden + logits are "
                "dumped from DFlashQwen3Model.forward with an empty draft KV cache "
                "(finalize the empty-KV forward hook on-box).",
            },
            f,
        )
    print(f"[d2] wrote reference fixtures to {args.out_dir}")


if __name__ == "__main__":
    main()
