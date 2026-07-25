#!/usr/bin/env python3
# M2c — capture the vLLM 0.25.0 greedy image->text GOLDEN for the fixed
# (image, prompt) on Qwen3-VL-4B AND measure K-run self-determinism (the gate-form
# decision per [[near-tie-distributional-gate]]: deterministic -> STRICT token-exact;
# non-deterministic near-tie -> distributional gate). enforce_eager, greedy.
#
# The image is the committed M0 fixture (raw 448x448x3 uint8) so the whole pipeline
# (our C++ processor + tower + backbone) is gated on the IDENTICAL pixels.
# Env: CKPT_DIR, FIX_DIR (committed qwen3vl fixture), OUT_DIR, K (default 5), N (max_tokens).
import hashlib
import json
import os

import numpy as np
import torch
from PIL import Image

CKPT = os.environ["CKPT_DIR"]
FIX = os.environ["FIX_DIR"]
OUT = os.environ["OUT_DIR"]
K = int(os.environ.get("K", "5"))
N = int(os.environ.get("N", "32"))
os.makedirs(OUT, exist_ok=True)

# fixed fixture image (exact bytes our C++ pipeline consumes)
raw = np.fromfile(os.path.join(FIX, "image_rgb_uint8_448x448x3.bin"), dtype=np.uint8).reshape(448, 448, 3)
img = Image.fromarray(raw, mode="RGB")

from vllm import LLM, SamplingParams

llm = LLM(model=CKPT, tokenizer=CKPT, trust_remote_code=True, dtype="bfloat16",
          enforce_eager=True, gpu_memory_utilization=0.30, limit_mm_per_prompt={"image": 1},
          max_model_len=4096)

# Same textual content as the M0 fixture, via the chat template (the real serving path).
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "What is in this image?"},
]}]
tok = llm.get_tokenizer()
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("PROMPT:\n", prompt, flush=True)

sp = SamplingParams(temperature=0.0, max_tokens=N)
runs = []
for i in range(K):
    out = llm.generate([{"prompt": prompt, "multi_modal_data": {"image": img}}], sp)
    ids = list(out[0].outputs[0].token_ids)
    txt = out[0].outputs[0].text
    runs.append(ids)
    print(f"run {i}: {len(ids)} toks  ids[:16]={ids[:16]}", flush=True)
    print(f"        text={txt!r}", flush=True)

# determinism analysis
ref = runs[0]
identical = all(r == ref for r in runs)
# first divergence position across the K runs
div = None
for pos in range(max(len(r) for r in runs)):
    vals = {tuple([r[pos]]) if pos < len(r) else None for r in runs}
    if len(vals) > 1:
        div = pos
        break
print(f"\nK={K} DETERMINISTIC={identical}  first_divergence_pos={div}", flush=True)
gate = "STRICT" if identical else "NEAR-TIE (distributional)"
print(f"GATE FORM (by measurement) = {gate}", flush=True)

np.array(ref, dtype=np.int32).tofile(os.path.join(OUT, "gen_tokens_i32.bin"))
manifest = {
    "model_id": "Qwen/Qwen3-VL-4B-Instruct", "K": K, "max_tokens": N,
    "deterministic": identical, "first_divergence_pos": div,
    "gate_form": gate, "prompt": prompt,
    "ref_token_ids": ref, "ref_text": runs[0] and None,
    "gen_tokens_sha256": hashlib.sha256(np.array(ref, dtype=np.int32).tobytes()).hexdigest(),
    "all_runs": runs,
}
with open(os.path.join(OUT, "gen_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print("WROTE gen_tokens_i32.bin + gen_manifest.json", flush=True)
