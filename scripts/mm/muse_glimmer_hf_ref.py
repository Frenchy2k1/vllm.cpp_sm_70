"""Run the REAL HF Muse Glimmer reference on the released checkpoint.

This is the genuine reference implementation from huggingface/transformers
branch `exportable-muse` (transformers 5.16.0.dev0), not our transcription.
Agreement here is a materially stronger claim than agreement with a port we
wrote ourselves from the same upstream source.

Emits the same bundle shape scripts/mm/muse_glimmer_text_ref.py does, so the
existing C++ comparison gate can consume it unchanged.
"""

import argparse, json, os, sys
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--out", required=True)
p.add_argument("--prompt", default="The capital of France is")
p.add_argument("--device", default="cuda")
p.add_argument("--layers", type=int, default=0,
               help="truncate the text tower to N layers (0 = full depth). Bounded memory "
                    "so a shared box is not OOMed by a 60 GB load.")
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)

from transformers import AutoConfig, AutoTokenizer
# muse_glimmer registers under the multimodal AutoModel classes, not
# AutoModelForCausalLM; use the concrete class directly.
from transformers.models.muse_glimmer import MuseGlimmerForConditionalGeneration

cfg = AutoConfig.from_pretrained(args.ckpt)
if args.layers:
    # Truncating the config makes from_pretrained materialize only these layers,
    # so peak memory is ~layers/52 of the full tower. The remaining weights are
    # simply not loaded.
    cfg.text_config.num_hidden_layers = args.layers
    for k in ("layer_types", "no_rope_layers", "layer_rope_theta"):
        v = getattr(cfg.text_config, k, None)
        if isinstance(v, list) and len(v) > args.layers:
            setattr(cfg.text_config, k, v[: args.layers])
tok = AutoTokenizer.from_pretrained(args.ckpt)
print(f"[hf] config {type(cfg).__name__} layers={cfg.text_config.num_hidden_layers}", flush=True)

ids = tok(args.prompt, add_special_tokens=False)["input_ids"]
print(f"[hf] prompt={args.prompt!r} -> {len(ids)} tokens {ids}", flush=True)

print("[hf] loading weights (this reads ~60 GB)...", flush=True)
# No device_map: that path needs `accelerate`, and installing it into this venv
# would pull deps we deliberately kept out. Load on CPU, then move -- at reduced
# depth the tower is small enough that this is cheap and bounded.
model = MuseGlimmerForConditionalGeneration.from_pretrained(
    args.ckpt, config=cfg, dtype=torch.bfloat16
)
model = model.to(args.device)
model.eval()
print(f"[hf] loaded {type(model).__name__}", flush=True)

with torch.no_grad():
    out = model(input_ids=torch.tensor([ids], device=model.device))
logits = out.logits[0].float().cpu()          # [T, V]

argmax = logits.argmax(-1).tolist()
top2 = torch.topk(logits[-1], 2)
meta = {
    "source": "REAL HF reference: huggingface/transformers @ exportable-muse (5.16.0.dev0)",
    "checkpoint": args.ckpt,
    "prompt": args.prompt,
    "token_ids": ids,
    "positions": list(range(len(ids))),
    "num_hidden_layers": int(cfg.text_config.num_hidden_layers),
    "full_depth": 52,
    "vocab_size": int(logits.shape[-1]),
    "hidden_size": int(cfg.text_config.hidden_size),
    "logits_file": "ref_logits.f32",
    "logits_shape": list(logits.shape),
    "argmax": argmax,
    "argmax_text": [tok.decode([i]) for i in argmax],
    "last_position_top2_ids": top2.indices.tolist(),
    "last_position_top2_values": top2.values.tolist(),
    "logit_absmax": float(logits.abs().max()),
}
logits.numpy().astype(np.float32).tofile(os.path.join(args.out, "ref_logits.f32"))
with open(os.path.join(args.out, "ref.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("[hf] argmax     :", argmax, flush=True)
print("[hf] argmax text:", meta["argmax_text"], flush=True)
print("[hf] top2       :", meta["last_position_top2_ids"], meta["last_position_top2_values"], flush=True)
print("[hf] |logit|max :", meta["logit_absmax"], flush=True)
print("[hf] wrote", args.out, flush=True)
