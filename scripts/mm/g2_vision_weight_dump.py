#!/usr/bin/env python3
# G2-impl — dump the Gemma-4 E4B `model.vision_tower.*` + `model.embed_vision.*`
# weights to a directory of f32 .bin files keyed by (prefix-stripped) tensor name,
# for the C++ NaFlex SigLIP2 vision-tower per-stage unit gate. NOT committed
# (~330 MiB f32); the test reads it via VLLM_GEMMA4_VISION_WEIGHTS.
#
# Also dumps the Gemma4ClippableLinear clip buffers (input_min/max,
# output_min/max) as 1-float .bin files (E4B's vision_config has
# use_clipped_linears=True with FINITE trained bounds, so the C++ tower applies
# clamp(linear(clamp(x,in),out)) on the 7 attention/MLP linears per block).
import glob
import os

import numpy as np
import torch
from safetensors import safe_open

CKPT = os.environ["CKPT_DIR"]
OUT = os.environ["WEIGHT_OUT"]
os.makedirs(OUT, exist_ok=True)

VT = "model.vision_tower."
EM = "model.embed_vision."

CLIP = ("input_min", "input_max", "output_min", "output_max")
n = 0
n_clip = 0
for st in sorted(glob.glob(os.path.join(CKPT, "*.safetensors"))):
    with safe_open(st, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith(VT):
                rel = k[len(VT):]
            elif k.startswith(EM):
                rel = "embed_vision." + k[len(EM):]
            else:
                continue
            base = rel.rsplit(".", 1)[-1]
            t = f.get_tensor(k)
            if base in CLIP:
                # bf16-round the bound to match torch's bf16 clamp buffer.
                t.to(torch.bfloat16).to(torch.float32).contiguous().numpy().astype(
                    "float32").tofile(os.path.join(OUT, rel + ".bin"))
                n_clip += 1
                continue
            t.to(torch.float32).contiguous().numpy().tofile(os.path.join(OUT, rel + ".bin"))
            n += 1

print(f"dumped {n} weight tensors + {n_clip} clip scalars to {OUT}", flush=True)
