#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone torch reference for the Muse Glimmer TEXT tower, on REAL weights.

WHY THIS EXISTS (read this before trusting any number it prints).

There is no runnable Muse Glimmer reference on a stock box:

  * released ``transformers`` does not register ``model_type: muse_glimmer``
    (the checkpoint declares ``transformers_version 5.15.0.dev0``; 5.3.0 raises
    ``ValueError: ... does not recognize this architecture``), and the
    checkpoint ships NO remote-code modelling file, so ``trust_remote_code``
    has nothing to trust;
  * our parity pin ``555967922`` contains no ``muse_glimmer`` at all, and the
    only upstream implementation — vLLM PR #51655 head ``075d645af`` — cannot
    be executed without a compiled ``vllm._C`` and a GPU.

So this file TRANSCRIBES the upstream forward into plain ``torch``, with no
``transformers`` and no ``vllm`` import, and runs it on the real checkpoint's
own safetensors. Every block cites the upstream ``file:line`` it was written
from, at vllm#51655 head ``075d645af``,
``vllm/model_executor/models/muse_glimmer.py``:

    MuseGlimmerRMSNorm            :520-552
    _muse_glimmer_use_qk_norm     :456-462
    _muse_glimmer_use_attn_output_gate :464-470
    _muse_glimmer_query_prescale  :472-517
    MuseGlimmerMLP.forward        :1076-1080
    MuseGlimmerAttention.__init__ :1083-1179   (iRoPE mask, scaling, window)
    MuseGlimmerAttention.forward  :1181-1210
    MuseGlimmerDecoderLayer.fwd   :1249-1269
    MuseGlimmerModel.forward      :1301-1345
    compute_logits                :1614-1622
plus ``vllm/transformers_utils/configs/muse_glimmer.py:20-126`` for the config
defaults and ``:1389-1425`` for the checkpoint weight-name conventions.

WHAT IT IS AND IS NOT. It is a SECOND implementation of the same published
spec, written from the python by a different route than our C++. It is NOT the
HF reference executed, because the HF reference does not exist on this machine.
A match therefore establishes that two independent transcriptions of #51655
agree on real weights — it does NOT establish agreement with Meta's own
runtime. Say exactly that and nothing stronger.

MEMORY. The 30B text tower is ~55.7 GB in bf16 and the box has ~73 GB free, so
this script NEVER holds the whole model: it streams ONE decoder layer at a time
out of the safetensors and frees it before reading the next. Peak resident is
embed_tokens + lm_head + one layer, ~7 GB at full depth.

Two modes:
  --layers K   run only the first K decoder layers (a REDUCED model built from
               REAL tensors). With --emit-weights it also writes a derived,
               self-contained safetensors + config.json that our C++ loader can
               consume, so both sides run the identical real bytes.
  --layers 0   run the full 52-layer tower (streams ~55.7 GB off disk).

Outputs, into --out:
  ref_logits.f32   float32 [T, V], row-major, the post-softcap logits
  ref.json         geometry, token ids, per-position argmax + top-k, checksums
  config.json      (with --emit-weights) reduced text-only config
  model.safetensors(with --emit-weights) the derived REAL-tensor checkpoint
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time

import torch

_ST_DTYPE = {
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float16: "F16",
}

# Where the forward runs. Set once from --device; the weight STREAM is unchanged
# either way, so a cuda run and a cpu run are the same arithmetic on different
# hardware and must agree on the argmax.
_DEVICE = torch.device("cpu")


# ───────────────────────────── config ─────────────────────────────


def default_no_rope_layers(num_layers: int) -> list[int]:
    """configs/muse_glimmer.py:20-26 — NoPE every 4th layer, counted BACKWARD."""
    stride = 4
    return [0 if (num_layers - 1 - i) % stride == 0 else 1 for i in range(num_layers)]


def query_prescale(qk_scale_factor, scale_query_by, head_dim: int) -> float:
    """models/muse_glimmer.py:472-517 — disambiguate the two config schemas.

    An explicit ``scale_query_by`` wins. Otherwise the native ``params.json``
    ships the RAW ``qk_scale_factor`` (~43.784 at head_dim 128) and folds
    ``1/sqrt(head_dim)`` itself, while the modular HF ``text_config`` ships it
    ALREADY folded (~3.87). Upstream picks by magnitude against
    ``sqrt(head_dim)``. Getting it wrong scales every query by ~11.3x.
    """
    if scale_query_by is not None:
        return float(scale_query_by)
    if qk_scale_factor is None:
        return 1.0
    sqrt_hd = math.sqrt(head_dim)
    qk = float(qk_scale_factor)
    return qk / sqrt_hd if qk >= sqrt_hd else qk


def flag_default_true(value) -> bool:
    """:456-470 — the modular schema OMITS the flag, so it reads as None. Muse
    Glimmer ALWAYS applies QK-norm and the output gate; only an explicit False
    disables them."""
    return True if value is None else bool(value)


class TextConfig:
    def __init__(self, raw: dict):
        t = raw.get("text_config", raw)
        self.raw_text = t
        self.vocab_size = int(t["vocab_size"])
        self.hidden_size = int(t["hidden_size"])
        self.intermediate_size = int(t["intermediate_size"])
        self.num_hidden_layers = int(t["num_hidden_layers"])
        self.num_attention_heads = int(t["num_attention_heads"])
        self.num_key_value_heads = int(t["num_key_value_heads"])
        self.head_dim = int(t["head_dim"])
        self.rms_norm_eps = float(t.get("rms_norm_eps", 1e-5))
        self.post_norm_eps = float(t.get("post_norm_eps", 1e-8))
        self.sliding_window = t.get("sliding_window", 2048)
        self.output_multiplier = float(t.get("output_multiplier", 0.19611613513818404))
        self.final_logit_softcapping = t.get("final_logit_softcapping", 20.0)
        self.normalize_tok_embeddings = bool(t.get("normalize_tok_embeddings", True))
        self.use_qk_norm = flag_default_true(t.get("use_qk_norm"))
        self.use_attn_output_gate = flag_default_true(t.get("use_attn_output_gate"))
        self.scale_query_by = query_prescale(
            t.get("qk_scale_factor"), t.get("scale_query_by"), self.head_dim
        )
        rope = t.get("rope_parameters") or {}
        self.rope_theta = float(rope.get("rope_theta", t.get("rope_theta", 500000.0)))
        assert t.get("hidden_activation", "silu") == "silu"

        # iRoPE. The RELEASED config ships neither `no_rope_layers` nor a bare
        # theta: it encodes the split TWICE, as `layer_rope_theta[i] == 0` and as
        # `layer_types[i] == "full_attention"`. Upstream's config class only
        # knows `no_rope_layers` and falls back to the backward-counted default;
        # we read the checkpoint's own encoding and CROSS-CHECK all three so a
        # disagreement is loud rather than silent.
        L = self.num_hidden_layers
        explicit = t.get("no_rope_layers")
        from_theta = None
        from_types = None
        theta_list = t.get("layer_rope_theta")
        if theta_list is not None and len(theta_list) == L:
            from_theta = [0 if float(x) == 0.0 else 1 for x in theta_list]
        types = t.get("layer_types")
        if types is not None and len(types) == L:
            from_types = [0 if ty == "full_attention" else 1 for ty in types]
        candidates = [c for c in (explicit, from_theta, from_types) if c is not None]
        for c in candidates[1:]:
            assert c == candidates[0], "muse glimmer iRoPE encodings disagree"
        self.no_rope_layers = candidates[0] if candidates else default_no_rope_layers(L)
        assert len(self.no_rope_layers) == L
        self.default_mask_agrees = self.no_rope_layers == default_no_rope_layers(L)


# ───────────────────────────── the forward ─────────────────────────────


def rms_norm(x: torch.Tensor, weight, offset: float, eps: float) -> torch.Tensor:
    """MuseGlimmerRMSNorm, :520-552. fp32 compute; the weight is applied as
    ``(w + offset)``; ``weight is None`` is the WEIGHTLESS form used by
    ``embed_norm`` (:1286) and the per-head ``qk_norm`` (:1121)."""
    f = x.float()
    out = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        out = out * (weight.float() + offset)
    return out


def rope_neox(x: torch.Tensor, positions: torch.Tensor, base: float) -> torch.Tensor:
    """get_rope(head_dim, ..., is_neox_style=True) as wired at :1163-1174.

    NeoX half-split: the rotary pair is ``(x[i], x[i + d/2])``. x is [T, Hn, D].
    """
    d = x.shape[-1]
    half = d // 2
    inv = base ** (
        -torch.arange(0, half, dtype=torch.float64, device=x.device) * 2.0 / d
    )
    ang = positions.to(torch.float64).unsqueeze(-1) * inv.unsqueeze(0)  # [T, half]
    cos = ang.cos().float().unsqueeze(1)
    sin = ang.sin().float().unsqueeze(1)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def attention(q, k, v, positions, scale, window):
    """Dense causal attention over [T, H*, D] fp32, with an optional sliding
    window. GQA fan-out mirrors vLLM's Attention wrapper (:1167-1179); the
    softmax scale is head_dim**-0.5 and there is NO attention logit softcap
    (``logits_soft_cap=None``, :1174)."""
    T, Hq, D = q.shape
    Hkv = k.shape[1]
    rep = Hq // Hkv
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    # [Hq, T, D]
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # [Hq, T, T]
    idx = torch.arange(T, device=q.device)
    mask = idx.unsqueeze(1) < idx.unsqueeze(0)  # j > i -> masked (causal)
    if window is not None and window > 0:
        mask = mask | ((idx.unsqueeze(1) - idx.unsqueeze(0)) >= window)
    scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v).transpose(0, 1).contiguous()  # [T, Hq, D]


def decoder_layer(x, w, cfg, layer_idx, positions):
    """MuseGlimmerDecoderLayer.forward :1249-1269 + MuseGlimmerAttention.forward
    :1181-1210. Sandwich norms with a baked +1 offset on all four; the two PRE
    norms take ``rms_norm_eps`` and the two POST norms ``post_norm_eps``
    (:1236-1247)."""
    H = cfg.hidden_size
    Hq, Hkv, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    use_rope = cfg.no_rope_layers[layer_idx] == 1

    residual = x
    h = rms_norm(x, w["input_layernorm"], 1.0, cfg.rms_norm_eps)

    q = h @ w["q_proj"].T
    k = h @ w["k_proj"].T
    v = h @ w["v_proj"].T
    q = q.view(-1, Hq, D)
    k = k.view(-1, Hkv, D)
    v = v.view(-1, Hkv, D)
    if cfg.use_qk_norm:
        # WEIGHTLESS RMSNorm over head_dim, fp32, applied BEFORE RoPE; then the
        # query pre-scale (:1189-1196). The softmax scale stays head_dim**-0.5.
        q = rms_norm(q, None, 0.0, cfg.rms_norm_eps) * cfg.scale_query_by
        k = rms_norm(k, None, 0.0, cfg.rms_norm_eps)
    if use_rope:
        q = rope_neox(q, positions, cfg.rope_theta)
        k = rope_neox(k, positions, cfg.rope_theta)
    # iRoPE: RoPE layers are SLIDING-window, NoPE layers are FULL (:1114-1116,
    # :1167-1168).
    window = cfg.sliding_window if use_rope else None
    a = attention(q, k, v, positions, D**-0.5, window).reshape(-1, Hq * D)
    if cfg.use_attn_output_gate:
        # The gate reads the layer input hidden states — i.e. the OUTPUT of
        # input_layernorm, since that is what the layer passes to self_attn
        # (:1203-1206 with :1256-1258).
        a = torch.sigmoid(h @ w["output_gate_proj"].T) * a
    o = a @ w["o_proj"].T
    o = rms_norm(o, w["post_attention_layernorm"], 1.0, cfg.post_norm_eps)
    x = residual + o

    residual = x
    h2 = rms_norm(x, w["pre_feedforward_layernorm"], 1.0, cfg.rms_norm_eps)
    g = h2 @ w["gate_proj"].T
    u = h2 @ w["up_proj"].T
    m = (torch.nn.functional.silu(g) * u) @ w["down_proj"].T
    m = rms_norm(m, w["post_feedforward_layernorm"], 1.0, cfg.post_norm_eps)
    return residual + m


# ───────────────────────── streaming weight access ─────────────────────────


class Shards:
    """safetensors reader that opens each shard once and slices tensors on
    demand, so only the tensors we ask for are ever resident."""

    def __init__(self, ckpt: str):
        from safetensors import safe_open

        index = os.path.join(ckpt, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as f:
                self.map = json.load(f)["weight_map"]
        else:
            self.map = {}
        self._open = {}
        self._safe_open = safe_open
        self.ckpt = ckpt
        if not self.map:
            path = os.path.join(ckpt, "model.safetensors")
            h = safe_open(path, framework="pt")
            self._open[path] = h
            for name in h.keys():
                self.map[name] = "model.safetensors"

    def handle(self, shard: str):
        path = os.path.join(self.ckpt, shard)
        if path not in self._open:
            self._open[path] = self._safe_open(path, framework="pt")
        return self._open[path]

    def has(self, name: str) -> bool:
        return name in self.map

    def get(self, name: str) -> torch.Tensor:
        return self.handle(self.map[name]).get_tensor(name)


LAYER_KEYS = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.gate_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

# Our short name  <-  the canonical checkpoint suffix. The attention OUTPUT gate
# ships as `self_attn.gate_proj` and collides by suffix with the MLP's
# `mlp.gate_proj`; upstream renames it to `output_gate_proj` FIRST
# (:1389-1397). Keeping the full `self_attn.`/`mlp.` prefix here makes the
# collision structurally impossible.
SHORT = {
    "input_layernorm": "input_layernorm",
    "post_attention_layernorm": "post_attention_layernorm",
    "pre_feedforward_layernorm": "pre_feedforward_layernorm",
    "post_feedforward_layernorm": "post_feedforward_layernorm",
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "output_gate_proj": "self_attn.gate_proj",
    "gate_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}


def layer_prefix(shards: Shards, idx: int) -> str:
    """The canonical export prefixes the language model with
    ``model.language_model.``; the legacy guac export uses ``model.``
    (:1362-1381)."""
    for pre in ("model.language_model.layers.", "model.layers."):
        if shards.has(f"{pre}{idx}.input_layernorm.weight"):
            return f"{pre}{idx}."
    raise KeyError(f"no decoder layer {idx} under either checkpoint convention")


def root_name(shards: Shards, *candidates: str) -> str:
    for c in candidates:
        if shards.has(c):
            return c
    raise KeyError(f"none of {candidates} present")


# ───────────────────────── derived checkpoint writer ─────────────────────────


def write_safetensors_streaming(path: str, entries):
    """Write a safetensors file without ever holding more than one tensor.

    ``entries`` is a sequence of ``(name, torch.Tensor)`` producers; we make two
    passes over it (once for the header offsets, once for the bytes), so it must
    be a callable returning a fresh generator.
    """
    header = {}
    offset = 0
    sizes = []
    for name, shape, nbytes, dtype in entries(meta_only=True):
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
        sizes.append((name, nbytes))
    blob = json.dumps(header, separators=(",", ":")).encode()
    pad = (-len(blob)) % 8
    blob += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        written = 0
        for name, tensor in entries(meta_only=False):
            b = tensor.contiguous().view(torch.uint8).numpy().tobytes()
            assert len(b) == dict(sizes)[name], f"{name} size drift"
            f.write(b)
            written += len(b)
            del tensor, b
    return offset


# ───────────────────────────────── main ─────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--layers",
        type=int,
        default=4,
        help="run only the first K decoder layers; 0 = the full tower",
    )
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--bos", action="store_true", help="prepend bos_token_id")
    ap.add_argument(
        "--emit-weights",
        action="store_true",
        help="also write a derived REAL-tensor safetensors + config our C++ "
        "loader can consume (only meaningful with --layers K < depth)",
    )
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument(
        "--device",
        default="cpu",
        help="torch device for the forward ('cpu' or 'cuda'). Weights are still "
        "streamed one layer at a time, so a GPU run holds the same working set; "
        "the RESULT must not depend on this flag and cross-checking that is the "
        "point of exposing it.",
    )
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    global _DEVICE
    _DEVICE = torch.device(args.device)

    with open(os.path.join(args.ckpt, "config.json")) as f:
        raw = json.load(f)
    cfg = TextConfig(raw)
    depth = cfg.num_hidden_layers
    k = depth if args.layers in (0, depth) else args.layers
    assert 1 <= k <= depth, f"--layers must be in [1,{depth}]"
    cfg.num_hidden_layers = k
    cfg.no_rope_layers = cfg.no_rope_layers[:k]

    shards = Shards(args.ckpt)
    os.makedirs(args.out, exist_ok=True)

    # ── tokenize with the checkpoint's OWN tokenizer ──
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(args.ckpt, "tokenizer.json"))
    ids = tok.encode(args.prompt, add_special_tokens=False).ids
    if args.bos:
        ids = [int(raw.get("text_config", raw).get("bos_token_id", 200000))] + ids
    ids = ids[: args.max_tokens]
    T = len(ids)
    positions = torch.arange(T, device=_DEVICE)
    print(f"[ref] prompt={args.prompt!r} -> {T} tokens {ids}", flush=True)
    print(
        f"[ref] depth={k}/{depth} hidden={cfg.hidden_size} vocab={cfg.vocab_size} "
        f"scale_query_by={cfg.scale_query_by} rope_theta={cfg.rope_theta} "
        f"irope={cfg.no_rope_layers}",
        flush=True,
    )

    emb_name = root_name(
        shards, "model.language_model.embed_tokens.weight", "model.embed_tokens.weight"
    )
    norm_name = root_name(
        shards, "model.language_model.norm.weight", "model.norm.weight"
    )

    t0 = time.time()
    emb_rows = shards.get(emb_name)[torch.tensor(ids)].to(_DEVICE, torch.float32)
    # MuseGlimmerModel.embed_input_ids :1301-1302 — a WEIGHTLESS RMSNorm, NOT
    # Gemma's sqrt(hidden_size) multiplier (:1285-1286).
    x = rms_norm(emb_rows, None, 0.0, cfg.rms_norm_eps) if cfg.normalize_tok_embeddings else emb_rows
    del emb_rows

    for l in range(k):
        pre = layer_prefix(shards, l)
        w = {}
        for short, suffix in SHORT.items():
            w[short] = shards.get(f"{pre}{suffix}.weight").to(_DEVICE, torch.float32)
        x = decoder_layer(x, w, cfg, l, positions)
        del w
        print(
            f"[ref] layer {l:>3} rope={cfg.no_rope_layers[l]} "
            f"|x|={x.abs().max().item():.4f} t={time.time() - t0:.1f}s",
            flush=True,
        )

    # Final norm carries NO +1 offset, unlike all four sandwich norms (:1296).
    x = rms_norm(x, shards.get(norm_name).to(_DEVICE, torch.float32), 0.0,
                 cfg.rms_norm_eps)
    lm_head = shards.get("lm_head.weight")
    logits = (x @ lm_head.to(_DEVICE, torch.float32).T) * cfg.output_multiplier
    del lm_head
    cap = cfg.final_logit_softcapping
    if cap:
        logits = float(cap) * torch.tanh(logits / float(cap))
    print(f"[ref] forward done in {time.time() - t0:.1f}s", flush=True)

    logits = logits.detach().to("cpu", torch.float32).contiguous()
    lp = os.path.join(args.out, "ref_logits.f32")
    with open(lp, "wb") as f:
        f.write(logits.numpy().tobytes())

    topk = torch.topk(logits, k=8, dim=-1)
    argmax = topk.indices[:, 0].tolist()
    ref = {
        "source": "scripts/mm/muse_glimmer_text_ref.py (torch transcription of "
        "vllm#51655 head 075d645af muse_glimmer.py)",
        "checkpoint": os.path.abspath(args.ckpt),
        "torch_version": torch.__version__,
        "device": str(_DEVICE),
        "prompt": args.prompt,
        "token_ids": ids,
        "positions": positions.cpu().tolist(),
        "num_hidden_layers": k,
        "full_depth": depth,
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "scale_query_by": cfg.scale_query_by,
        "no_rope_layers": cfg.no_rope_layers,
        "default_irope_mask_agrees": cfg.default_mask_agrees,
        "logits_file": "ref_logits.f32",
        "logits_shape": [T, cfg.vocab_size],
        "argmax": argmax,
        "argmax_text": [tok.decode([i]) for i in argmax],
        "topk_ids": topk.indices.tolist(),
        "topk_values": topk.values.tolist(),
        "logit_absmax": float(logits.abs().max()),
        "logit_mean": float(logits.mean()),
        "finite": bool(torch.isfinite(logits).all()),
    }
    with open(os.path.join(args.out, "ref.json"), "w") as f:
        json.dump(ref, f, indent=2)
    print("[ref] argmax ids  :", argmax, flush=True)
    print("[ref] argmax text :", ref["argmax_text"], flush=True)
    print("[ref] next token  :", repr(ref["argmax_text"][-1]), flush=True)

    if args.emit_weights:
        emit_reduced(args, shards, cfg, raw, k, emb_name, norm_name)
    return 0


def emit_reduced(args, shards, cfg, raw, k, emb_name, norm_name):
    """Write a self-contained REDUCED checkpoint made of REAL tensors, in the
    canonical ``model.language_model.*`` naming, plus a text-only config. Our
    C++ loader consumes this directly, so both sides run identical bytes."""
    out_st = os.path.join(args.out, "model.safetensors")

    plan = [(emb_name, "model.language_model.embed_tokens.weight")]
    for l in range(k):
        pre = layer_prefix(shards, l)
        for suffix in LAYER_KEYS:
            plan.append(
                (f"{pre}{suffix}.weight", f"model.language_model.layers.{l}.{suffix}.weight")
            )
    plan.append((norm_name, "model.language_model.norm.weight"))
    plan.append(("lm_head.weight", "lm_head.weight"))

    def entries(meta_only):
        for src, dst in plan:
            t = shards.get(src)
            if meta_only:
                yield dst, tuple(t.shape), t.numel() * t.element_size(), _ST_DTYPE[t.dtype]
            else:
                yield dst, t
            del t

    total = write_safetensors_streaming(out_st, entries)
    print(f"[ref] wrote {out_st} ({total / 1e9:.2f} GB, {len(plan)} tensors)", flush=True)

    text = dict(raw["text_config"])
    text["num_hidden_layers"] = k
    for key in ("layer_types", "layer_rope_theta"):
        if key in text:
            text[key] = text[key][:k]
    text["no_rope_layers"] = cfg.no_rope_layers
    cfgj = {
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "model_type": "muse_glimmer",
        "dtype": "bfloat16",
        "text_config": text,
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfgj, f, indent=2)
    print(f"[ref] wrote {os.path.join(args.out, 'config.json')}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
