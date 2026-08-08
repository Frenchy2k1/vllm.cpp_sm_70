#!/usr/bin/env python3
"""Generate the committed tiny `LlamaModel` EMBEDDING fixture (ARCH-ONE-SURFACE
ROW 6, the #121 committed-fixture precedent).

Writes tests/vllm/models/fixtures/llama_embed_e2e/:
  config.json        — architectures ["LlamaModel"] (the bare *Model arch the
                       upstream _EMBEDDING_MODELS maps onto the Llama backbone,
                       vllm/model_executor/models/registry.py:230)
  tokenizer.json     — minimal Metaspace BPE vocab (tok::Tokenizer-parseable)
  model.safetensors  — deterministic tiny bf16 backbone in the BARE `*Model`
                       name layout (embed_tokens.weight / layers.N... /
                       norm.weight — NO "model." prefix, NO lm_head), the
                       layout adapters.py:178-181 maps with candidate_prefixes

Deterministic: fixed seed, no torch. Re-running reproduces byte-identical
files, so the committed fixture is reviewable.

Usage: python3 scripts/mm/llama_embed_fixture_gen.py
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "tests/vllm/models/fixtures/llama_embed_e2e"

HIDDEN = 64
LAYERS = 2
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16
INTERMEDIATE = 128
VOCAB = 32


def f32_to_bf16_bits(x: float) -> int:
    """Round-to-nearest-even f32 -> bf16, matching vt::F32ToBF16."""
    (bits,) = struct.unpack("<I", struct.pack("<f", x))
    lsb = (bits >> 16) & 1
    rounding = 0x7FFF + lsb
    return ((bits + rounding) >> 16) & 0xFFFF


class Rng:
    """Deterministic xorshift32 in [-scale, scale) — no numpy dependency."""

    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF or 1

    def next_u32(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x
        return x

    def uniform(self, scale: float) -> float:
        return (self.next_u32() / 2**32 * 2.0 - 1.0) * scale


def bf16_tensor(shape: list[int], seed: int, scale: float = 0.08) -> bytes:
    rng = Rng(seed)
    numel = 1
    for s in shape:
        numel *= s
    return b"".join(
        struct.pack("<H", f32_to_bf16_bits(rng.uniform(scale)))
        for _ in range(numel)
    )


def write_safetensors(path: Path, tensors: dict[str, tuple[list[int], bytes]]) -> None:
    header: dict[str, dict] = {}
    offset = 0
    for name, (shape, data) in tensors.items():
        header[name] = {
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
    hdr = json.dumps(header, sort_keys=True).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        for _, (_, data) in tensors.items():
            f.write(data)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    config = {
        "architectures": ["LlamaModel"],
        "model_type": "llama",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "intermediate_size": INTERMEDIATE,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "vocab_size": VOCAB,
        "max_position_embeddings": 128,
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": False,  # embedding conversion: NO lm_head at all
        "attention_bias": False,
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=1) + "\n")

    # Minimal Metaspace BPE the tok::Tokenizer parses; ids stay < VOCAB.
    tokenizer = {
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
            # Character-level vocab (BPE with empty merges tokenizes each
            # Metaspace pre-token into characters): every lowercase test
            # string encodes to in-vocab ids, so EncodeWithSpecialTokens never
            # fails and ids stay < vocab_size.
            "vocab": {"▁": 0} | {chr(c): i + 1 for i, c in enumerate(range(ord("a"), ord("z") + 1))},
            "merges": [],
        },
        "added_tokens": [],
    }
    (OUT / "tokenizer.json").write_text(json.dumps(tokenizer, indent=1) + "\n")

    qdim = HEADS * HEAD_DIM
    kdim = KV_HEADS * HEAD_DIM
    tensors: dict[str, tuple[list[int], bytes]] = {}
    seed = 1
    # BARE `*Model` layout: no "model." prefix, no lm_head (adapters.py:135-181).
    tensors["embed_tokens.weight"] = ([VOCAB, HIDDEN], bf16_tensor([VOCAB, HIDDEN], seed))
    seed += 1
    tensors["norm.weight"] = ([HIDDEN], bf16_tensor([HIDDEN], seed, 0.5))
    seed += 1
    for layer in range(LAYERS):
        base = f"layers.{layer}."
        tensors[base + "input_layernorm.weight"] = (
            [HIDDEN], bf16_tensor([HIDDEN], seed, 0.5))
        seed += 1
        tensors[base + "post_attention_layernorm.weight"] = (
            [HIDDEN], bf16_tensor([HIDDEN], seed, 0.5))
        seed += 1
        tensors[base + "self_attn.q_proj.weight"] = (
            [qdim, HIDDEN], bf16_tensor([qdim, HIDDEN], seed))
        seed += 1
        tensors[base + "self_attn.k_proj.weight"] = (
            [kdim, HIDDEN], bf16_tensor([kdim, HIDDEN], seed))
        seed += 1
        tensors[base + "self_attn.v_proj.weight"] = (
            [kdim, HIDDEN], bf16_tensor([kdim, HIDDEN], seed))
        seed += 1
        tensors[base + "self_attn.o_proj.weight"] = (
            [HIDDEN, qdim], bf16_tensor([HIDDEN, qdim], seed))
        seed += 1
        tensors[base + "mlp.gate_proj.weight"] = (
            [INTERMEDIATE, HIDDEN], bf16_tensor([INTERMEDIATE, HIDDEN], seed))
        seed += 1
        tensors[base + "mlp.up_proj.weight"] = (
            [INTERMEDIATE, HIDDEN], bf16_tensor([INTERMEDIATE, HIDDEN], seed))
        seed += 1
        tensors[base + "mlp.down_proj.weight"] = (
            [HIDDEN, INTERMEDIATE], bf16_tensor([HIDDEN, INTERMEDIATE], seed))
        seed += 1

    write_safetensors(OUT / "model.safetensors", tensors)
    total = sum(len(d) for _, d in tensors.values())
    print(f"wrote {OUT} (weights {total} bytes, {len(tensors)} tensors)")


if __name__ == "__main__":
    main()
