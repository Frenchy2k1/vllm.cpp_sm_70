#!/usr/bin/env python3
# MiniCPM W0 VEHICLE PREP (three reproducible, gate-neutral steps on the HF snapshot):
#
# 1. WEIGHTS: convert the OFFICIAL openbmb/MiniCPM-2B-sft-bf16 checkpoint from
#    pytorch_model.bin (pickle) to model.safetensors, IN PLACE in its HF snapshot
#    dir. This is a Python-side conversion using the TRUSTED official pickle on the
#    oracle box — NOT a pickle deserializer in our C++ engine (our loader stays
#    safetensors-only). The resulting model.safetensors carries the SAME weights, so
#    BOTH the vLLM 0.25.0 oracle golden AND our paged engine read identical bf16.
#    MiniCPM-2B ties embeddings (no lm_head.weight), so no shared-storage hazard.
#
# 2. CONFIG: add "model_type": "minicpm" to config.json. The raw openbmb config omits
#    it (it relies on auto_map/trust_remote_code); our HfConfig loader requires the
#    field. Arch dispatch is by architectures=["MiniCPMForCausalLM"] regardless; the
#    community safetensors mirrors add the same field. (Follow-up: accept an
#    auto_map-only config without model_type.)
#
# 3. TOKENIZER: re-express MiniCPM's SentencePiece whitespace handling into the form
#    our tokenizer parser accepts. The raw tokenizer.json encodes it as a normalizer
#    Sequence [Prepend "▁", Replace " "->"▁"] with a null pre_tokenizer; our parser
#    rejects a non-empty normalizer Sequence and requires a pre_tokenizer. This is
#    exactly a SentencePiece Metaspace pre_tokenizer, so we neutralize the normalizer
#    to the accepted empty Sequence and set pre_tokenizer=Metaspace (a FAITHFUL
#    re-expression). The gate is ID-based (feeds the oracle's exact prompt ids via the
#    TokensPrompt engine path and compares ids), so tokenization is unused and this
#    only lets the engine construct + detokenize output. (Follow-up: implement the
#    normalizer Prepend/Replace natively so the string path needs no prep.)
#
# Run on dgx: ~/venvs/vllm-oracle/bin/python scripts/minicpm-convert-safetensors.py
import glob
import json
import os
import sys

import torch
from safetensors.torch import save_file

REPO_DIR = "models--openbmb--MiniCPM-2B-sft-bf16"


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
    # Drop any non-parameter buffers ColossalAI-style rotary caches (vLLM skips
    # rotary_emb.inv_freq / cos_cached / sin_cached; our loader never references
    # them). Keep everything else exactly.
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

    # 2. CONFIG: add model_type (our loader requires it; arch dispatch is by
    #    architectures=["MiniCPMForCausalLM"]).
    cfgp = os.path.join(snap, "config.json")
    cfg = json.load(open(cfgp))
    if "model_type" not in cfg:
        cfg["model_type"] = "minicpm"
        json.dump(cfg, open(cfgp, "w"), indent=4)
        print("config.json: added model_type=minicpm")
    else:
        print(f"config.json: model_type already {cfg['model_type']!r}")

    # 3. TOKENIZER: faithfully re-express the Prepend/Replace normalizer as a
    #    Metaspace pre_tokenizer (gate is ID-based; tokenization unused).
    tokp = os.path.join(snap, "tokenizer.json")
    if os.path.exists(tokp):
        tok = json.load(open(tokp))
        tok["normalizer"] = {"type": "Sequence", "normalizers": []}
        tok["pre_tokenizer"] = {"type": "Metaspace", "replacement": "▁",
                                "prepend_scheme": "first", "split": False}
        json.dump(tok, open(tokp, "w"), ensure_ascii=False)
        print("tokenizer.json: normalizer -> empty Sequence; pre_tokenizer -> Metaspace")


if __name__ == "__main__":
    main()
