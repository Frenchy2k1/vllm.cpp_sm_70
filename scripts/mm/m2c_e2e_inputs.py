#!/usr/bin/env python3
# M2c — dump the EXACT vLLM 0.25.0 model input token ids (prompt_token_ids, with
# the single <|image_pad|> expanded to 196 image tokens) for the committed fixture
# (image, prompt), so the C++ e2e gate drives its forked forward on the identical
# model input as the golden. The tokenizer text path is out of M2c's numeric
# scope; this isolates the gate to the tower+backbone+merge forward.
#
# Writes tests/vllm/multimodal/fixtures/qwen3vl_text/input_ids_i32.bin (int32).
# Does NOT touch gen_tokens_i32.bin / gen_manifest.json (the committed golden).
# Env: CKPT_DIR, FIX_DIR (committed qwen3vl image fixture), OUT_DIR (qwen3vl_text).
import json
import os

import numpy as np
from PIL import Image

CKPT = os.environ["CKPT_DIR"]
FIX = os.environ["FIX_DIR"]
OUT = os.environ["OUT_DIR"]
os.makedirs(OUT, exist_ok=True)

raw = np.fromfile(os.path.join(FIX, "image_rgb_uint8_448x448x3.bin"),
                  dtype=np.uint8).reshape(448, 448, 3)
img = Image.fromarray(raw, mode="RGB")

from vllm import LLM, SamplingParams

llm = LLM(model=CKPT, tokenizer=CKPT, trust_remote_code=True, dtype="bfloat16",
          enforce_eager=True, gpu_memory_utilization=0.30,
          limit_mm_per_prompt={"image": 1}, max_model_len=4096)

messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "What is in this image?"},
]}]
tok = llm.get_tokenizer()
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

sp = SamplingParams(temperature=0.0, max_tokens=1)
out = llm.generate([{"prompt": prompt, "multi_modal_data": {"image": img}}], sp)
ids = list(out[0].prompt_token_ids)
n_img = sum(1 for t in ids if t == 151655)
print(f"prompt_token_ids len={len(ids)}  image_tokens={n_img}", flush=True)
print(f"first image token at offset {ids.index(151655)}", flush=True)
assert n_img == 196, f"expected 196 image tokens, got {n_img}"

np.array(ids, dtype=np.int32).tofile(os.path.join(OUT, "input_ids_i32.bin"))
with open(os.path.join(OUT, "input_ids_manifest.json"), "w") as f:
    json.dump({"model_id": "Qwen/Qwen3-VL-4B-Instruct", "len": len(ids),
               "image_tokens": n_img, "image_offset": ids.index(151655),
               "prompt": prompt}, f, indent=2)
print("WROTE input_ids_i32.bin", flush=True)
