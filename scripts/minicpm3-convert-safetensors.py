#!/usr/bin/env python3
# MiniCPM3 W0 VEHICLE PREP (reproducible, gate-neutral steps on the HF snapshot),
# a straight generalization of scripts/minicpm-convert-safetensors.py:
#
# 1. WEIGHTS: convert the OFFICIAL openbmb/MiniCPM3-4B checkpoint from
#    pytorch_model.bin (pickle) to model.safetensors, IN PLACE in its HF snapshot
#    dir. Python-side conversion using the TRUSTED official pickle on the oracle
#    box — NOT a pickle deserializer in our C++ engine (our loader stays
#    safetensors-only). The resulting model.safetensors carries the SAME weights,
#    so BOTH the vLLM 0.25.0 oracle golden AND our paged engine read identical
#    bf16. MiniCPM3-4B ties embeddings (no lm_head.weight), so no shared-storage
#    hazard.
#
# 2. CONFIG: MiniCPM3-4B config.json ALREADY carries "model_type": "minicpm3", so
#    this is a no-op (kept for parity with the MiniCPM script / community mirrors).
#    Arch dispatch is by architectures=["MiniCPM3ForCausalLM"] regardless.
#
# 3. TOKENIZER: re-express any SentencePiece whitespace normalizer into the form
#    our tokenizer parser accepts (empty normalizer Sequence + Metaspace
#    pre_tokenizer), IF the tokenizer.json uses one. The gate is ID-based (feeds
#    the oracle's exact prompt ids via the TokensPrompt engine path and compares
#    ids), so tokenization is unused and this only lets the engine construct +
#    detokenize output. Applied defensively; a no-op if the tokenizer already has
#    an empty normalizer + a pre_tokenizer.
#
# Run on dgx: ~/venvs/vllm-oracle/bin/python scripts/minicpm3-convert-safetensors.py
import glob
import json
import os

import torch
from safetensors.torch import save_file

REPO_DIR = os.environ.get("MINICPM3_REPO_DIR", "models--openbmb--MiniCPM3-4B")


def find_snapshot():
    base = os.path.join(os.path.expanduser("~"), ".cache/huggingface/hub",
                        REPO_DIR, "snapshots")
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        if os.path.exists(os.path.join(d, "config.json")):
            return d
    raise SystemExit(f"no snapshot with config.json under {base}")


def main():
    snap = find_snapshot()
    binf = os.path.join(snap, "pytorch_model.bin")
    outf = os.path.join(snap, "model.safetensors")
    print(f"snapshot: {snap}")
    if os.path.exists(outf):
        print(f"model.safetensors already present ({os.path.getsize(outf)} bytes); "
              f"re-converting")
    sd = torch.load(binf, map_location="cpu", weights_only=True)
    # Drop non-parameter rotary-cache buffers (vLLM skips rotary_emb.inv_freq /
    # cos_cached / sin_cached; our loader never references them). Keep the rest.
    clean = {}
    dropped = []
    for k, v in sd.items():
        if "rotary_emb.inv_freq" in k or "rotary_emb.cos_cached" in k or \
           "rotary_emb.sin_cached" in k:
            dropped.append(k)
            continue
        clean[k] = v.contiguous()
    print(f"tensors: {len(clean)} kept, {len(dropped)} rotary-cache dropped "
          f"({dropped[:3]}{'...' if len(dropped) > 3 else ''})")
    has_lmhead = any(k == "lm_head.weight" for k in clean)
    print(f"lm_head.weight present: {has_lmhead} (False => tied embeddings)")
    dtypes = {str(v.dtype) for v in clean.values()}
    print(f"dtypes present: {dtypes}")
    save_file(clean, outf, metadata={"format": "pt"})
    print(f"wrote {outf} ({os.path.getsize(outf)} bytes)")

    # 2. CONFIG: ensure model_type present (MiniCPM3-4B already has it).
    cfgp = os.path.join(snap, "config.json")
    cfg = json.load(open(cfgp))
    if "model_type" not in cfg:
        cfg["model_type"] = "minicpm3"
        json.dump(cfg, open(cfgp, "w"), indent=4)
        print("config.json: added model_type=minicpm3")
    else:
        print(f"config.json: model_type already {cfg['model_type']!r}")

    # 3. TOKENIZER: faithfully re-express a Prepend/Replace normalizer as a
    #    Metaspace pre_tokenizer, only when present (gate is ID-based).
    tokp = os.path.join(snap, "tokenizer.json")
    if os.path.exists(tokp):
        tok = json.load(open(tokp))
        norm = tok.get("normalizer")
        needs = isinstance(norm, dict) and norm.get("type") == "Sequence" and \
            norm.get("normalizers")
        if needs or tok.get("pre_tokenizer") is None:
            tok["normalizer"] = {"type": "Sequence", "normalizers": []}
            tok["pre_tokenizer"] = {"type": "Metaspace", "replacement": "▁",
                                    "prepend_scheme": "first", "split": False}
            json.dump(tok, open(tokp, "w"), ensure_ascii=False)
            print("tokenizer.json: normalizer -> empty Sequence; "
                  "pre_tokenizer -> Metaspace")
        else:
            print("tokenizer.json: normalizer/pre_tokenizer already accepted; "
                  "no change")


if __name__ == "__main__":
    main()
