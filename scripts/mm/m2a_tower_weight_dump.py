#!/usr/bin/env python3
# M2a — dump the Qwen3-VL-4B `model.visual.*` weights to a directory of f32 .bin
# files keyed by tower-relative tensor name, for the C++ vision-tower unit gate.
# NOT committed (≈1.2 GiB f32); the test reads it via VLLM_QWEN3VL_WEIGHTS.
import glob
import os

import numpy as np
import torch
from safetensors import safe_open

CKPT = os.environ["CKPT_DIR"]
OUT = os.environ["WEIGHT_OUT"]
os.makedirs(OUT, exist_ok=True)

n = 0
for st in sorted(glob.glob(os.path.join(CKPT, "*.safetensors"))):
    with safe_open(st, framework="pt", device="cpu") as f:
        for k in f.keys():
            if ".visual." not in k:
                continue
            rel = k.split("visual.", 1)[1]  # tower-relative
            t = f.get_tensor(k).to(torch.float32).contiguous().numpy()
            t.tofile(os.path.join(OUT, rel + ".bin"))
            n += 1
print(f"dumped {n} visual weight tensors to {OUT}", flush=True)
