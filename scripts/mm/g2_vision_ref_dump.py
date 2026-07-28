#!/usr/bin/env python3
# Gemma-4 G2 — staged vision-tower reference dump (M2a playbook).
#
# Loads ONLY the vision tower + projector weights from the E4B checkpoint,
# runs the transformers-eager Gemma4VisionModel + Gemma4MultimodalEmbedder on
# the EXACT processor outputs captured by g2_gemma4_image_oracle_capture.py, and
# dumps the four stage tensors for future C++ per-stage unit-gating:
#   (1) patch_embedder out  [1, P, 768]
#   (2) encoder last_hidden  [1, P, 768]
#   (3) pooled+stripped      [T, 768]   (Gemma4VisionModel.forward output)
#   (4) projected features   [T, 2560]  (embed_vision output == merge input)
#
# vLLM's gemma4_mm runs this SAME tower in eager mode (docstring), so these are
# the faithful stage references for the vLLM image golden. CPU is sufficient
# (small tower); no GPU flock needed.
#
# Env: CKPT_DIR, REF_DIR (the vision_refs dir with proc_*.npy), OUT_DIR.
import hashlib
import json
import os

import numpy as np
import torch


def _dump(d, name, t):
    a = t.detach().float().cpu().numpy()
    np.save(os.path.join(d, f"{name}.npy"), a)
    return {"shape": list(a.shape), "dtype": str(a.dtype),
            "sha256": hashlib.sha256(a.tobytes()).hexdigest()}


def main():
    CKPT = os.environ["CKPT_DIR"]
    REF = os.environ["REF_DIR"]
    OUT = os.environ["OUT_DIR"]
    os.makedirs(OUT, exist_ok=True)

    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4MultimodalEmbedder,
        Gemma4VisionModel,
    )

    cfg = AutoConfig.from_pretrained(CKPT, trust_remote_code=True)
    vcfg = cfg.vision_config
    tcfg = cfg.text_config
    dtype = torch.bfloat16

    vt = Gemma4VisionModel(vcfg).to(dtype).eval()
    embed = Gemma4MultimodalEmbedder(vcfg, tcfg).to(dtype).eval()

    # Load matching weights (checkpoint prefixes: model.vision_tower / model.embed_vision).
    st = os.path.join(CKPT, "model.safetensors")
    vt_sd, em_sd = {}, {}
    with safe_open(st, framework="pt") as f:
        for key in f.keys():
            if key.startswith("model.vision_tower."):
                vt_sd[key[len("model.vision_tower."):]] = f.get_tensor(key).to(dtype)
            elif key.startswith("model.embed_vision."):
                em_sd[key[len("model.embed_vision."):]] = f.get_tensor(key).to(dtype)
    miss_v, unexp_v = vt.load_state_dict(vt_sd, strict=False)
    miss_e, unexp_e = embed.load_state_dict(em_sd, strict=False)
    print("vt loaded:", len(vt_sd), "missing:", list(miss_v)[:6], "unexpected:", list(unexp_v)[:6], flush=True)
    print("embed loaded:", len(em_sd), "missing:", list(miss_e), "unexpected:", list(unexp_e), flush=True)

    pv = torch.from_numpy(np.load(os.path.join(REF, "proc_pixel_values.npy")))
    pid = torch.from_numpy(np.load(os.path.join(REF, "proc_image_position_ids.npy")))
    print("pixel_values", tuple(pv.shape), pv.dtype, "position_ids", tuple(pid.shape), pid.dtype, flush=True)

    captured = {}

    def _grab(nm):
        def hook(_m, _i, out):
            t = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
            captured[nm] = t
        return hook

    vt.patch_embedder.register_forward_hook(_grab("patch_embedder"))
    vt.encoder.register_forward_hook(_grab("encoder_last_hidden"))

    meta = {}
    with torch.no_grad():
        vis = vt(pixel_values=pv, pixel_position_ids=pid)          # pooled+stripped
        proj = embed(inputs_embeds=vis.last_hidden_state)           # projected
    meta["patch_embedder"] = _dump(OUT, "ref_patch_embedder", captured["patch_embedder"])
    meta["encoder_last_hidden"] = _dump(OUT, "ref_encoder_last_hidden", captured["encoder_last_hidden"])
    meta["pooled_stripped"] = _dump(OUT, "ref_pooled_stripped", vis.last_hidden_state)
    meta["projected_merge_input"] = _dump(OUT, "ref_projected", proj)

    with open(os.path.join(OUT, "vision_ref_manifest.json"), "w") as f:
        json.dump({"note": "transformers-eager Gemma4VisionModel+embed_vision on the "
                           "vLLM golden's processor outputs; vLLM runs this tower in eager "
                           "→ faithful stage refs for C++ unit-gating",
                   "dtype": "bfloat16", "stages": meta}, f, indent=2)
    for k, v in meta.items():
        print(f"REF {k} shape={v['shape']} sha={v['sha256'][:16]}", flush=True)
    print("WROTE vision_ref_manifest.json", flush=True)


if __name__ == "__main__":
    main()
