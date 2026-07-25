#!/usr/bin/env python3
# M2a — dump vLLM Qwen3_VisionTransformer reference activations for the fixed M1
# fixture image, to anchor the C++ vision-tower unit gates (RED-first).
#
# Grounds: vllm/model_executor/models/qwen3_vl.py @ e24d1b24
#   Qwen3_VisionTransformer (:519), forward (:800), prepare_encoder_metadata (:704).
# Runs the tower STANDALONE (no LLM) in the production model dtype (bfloat16),
# loading only `model.visual.*` weights from the Qwen3-VL-4B checkpoint.
#
# Outputs (all little-endian f32, widened from bf16 where noted) + a manifest with
# sha256 + shape, into the fixtures dir.
import glob
import hashlib
import json
import os
import sys

import numpy as np
import torch

CKPT = os.environ["CKPT_DIR"]
FIX = os.environ["FIX_DIR"]
OUT = os.environ["OUT_DIR"]
os.makedirs(OUT, exist_ok=True)

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.bfloat16)

# ---- minimal single-process distributed env (vLLM layers require it) ----
from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.config import VllmConfig, ModelConfig, set_current_vllm_config

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29591")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer

cfg = json.load(open(os.path.join(CKPT, "config.json")))
vcfg = Qwen3VLVisionConfig(**cfg["vision_config"])
print("vision_config:", vcfg, flush=True)

mcfg = ModelConfig(model=CKPT, tokenizer=CKPT, trust_remote_code=True,
                   dtype="bfloat16", enforce_eager=True)
vllm_config = VllmConfig(model_config=mcfg)
device = "cuda"

_ctx = set_current_vllm_config(vllm_config)
_ctx.__enter__()
init_distributed_environment(
    world_size=1, rank=0, distributed_init_method="env://", local_rank=0, backend="gloo"
)
initialize_model_parallel(tensor_model_parallel_size=1)

tower = Qwen3_VisionTransformer(vcfg, norm_eps=1e-6, quant_config=None, prefix="visual")
tower = tower.to(device=device, dtype=torch.bfloat16).eval()

# ---- load visual.* weights ----
from safetensors import safe_open

visual_weights = {}
for st in sorted(glob.glob(os.path.join(CKPT, "*.safetensors"))):
    with safe_open(st, framework="pt", device="cpu") as f:
        for k in f.keys():
            if ".visual." in k:
                # strip leading "model.visual." -> tower-relative name
                rel = k.split("visual.", 1)[1]
                visual_weights[rel] = f.get_tensor(k)
print("loaded visual tensors:", len(visual_weights), flush=True)
missing = tower.load_weights([(k, v) for k, v in visual_weights.items()])
print("load_weights returned (loaded set size):", len(missing), flush=True)

# ---- inputs from the fixture ----
pv = np.fromfile(os.path.join(FIX, "pixel_values_f32.bin"), dtype=np.float32).reshape(784, 1536)
grid_thw = np.fromfile(os.path.join(FIX, "image_grid_thw_i64.bin"), dtype=np.int64).reshape(3).tolist()
print("grid_thw:", grid_thw, "pixel_values:", pv.shape, flush=True)
x = torch.from_numpy(pv).to(device=device, dtype=torch.bfloat16)
grid_list = [grid_thw]

captures = {}

def save(name, t):
    a = t.detach().to(torch.float32).cpu().contiguous().numpy()
    a.tofile(os.path.join(OUT, name + ".bin"))
    captures[name] = {"shape": list(a.shape), "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
    print(f"  dumped {name} shape={list(a.shape)}", flush=True)

# ---- encoder metadata (pos_embeds, rope cos/sin) ----
meta = tower.prepare_encoder_metadata(grid_list, device=torch.device(device))
save("pos_embeds", meta["pos_embeds"])                # [784,1024]
save("rotary_cos", meta["rotary_pos_emb_cos"])        # [784,64]
save("rotary_sin", meta["rotary_pos_emb_sin"])        # [784,64]
cu = meta["cu_seqlens"].detach().to(torch.int32).cpu().numpy()
cu.tofile(os.path.join(OUT, "cu_seqlens_i32.bin"))
captures["cu_seqlens"] = {"shape": list(cu.shape), "values": cu.tolist()}
print("  cu_seqlens:", cu.tolist(), flush=True)

# ---- hooks for intermediates ----
h = []
h.append(tower.patch_embed.register_forward_hook(lambda m, i, o: save("patch_embed_out", o)))
h.append(tower.blocks[0].register_forward_hook(lambda m, i, o: save("block0_out", o)))
h.append(tower.merger.register_forward_hook(lambda m, i, o: save("merger_out", o)))
for j in range(len(tower.deepstack_merger_list)):
    h.append(tower.deepstack_merger_list[j].register_forward_hook(
        (lambda idx: (lambda m, i, o: save(f"deepstack_{idx}_out", o)))(j)))

# also capture (patch_embed + pos_embed) as the true block input
out = tower.forward(x, grid_list)
save("tower_out", out)  # [196, 10240]
for hh in h:
    hh.remove()

json.dump(captures, open(os.path.join(OUT, "ref_manifest.json"), "w"), indent=2)
print("DONE. manifest:", os.path.join(OUT, "ref_manifest.json"), flush=True)
