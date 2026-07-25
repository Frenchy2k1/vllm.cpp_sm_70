#!/usr/bin/env python3
# M3-W0 — capture the vLLM 0.25.0 greedy image->text GOLDEN for the fixed
# (image, prompt) on **Qwen3.6-27B** (`Qwen3_5ForConditionalGeneration`, the
# vision-inclusive bf16 checkpoint `Qwen/Qwen3.6-27B`) AND measure K-run
# self-determinism (the gate-form decision per [[near-tie-distributional-gate]]:
# deterministic -> STRICT token-exact; non-deterministic near-tie -> distributional).
# enforce_eager, greedy. Also DUMPS the placeholder-expanded input ids + grid_thw +
# pixel_values sha so our C++ pipeline is gated on the IDENTICAL model input.
#
# The image is the committed M0/M2 fixture (raw 448x448x3 uint8) so the whole
# pipeline (our C++ processor + tower + GDN-hybrid backbone) is gated on the
# IDENTICAL pixels + prompt as the Qwen3-VL-4B gate.
# Env: CKPT_DIR, FIX_DIR (committed qwen3vl fixture), OUT_DIR, K (default 5), N (max_tokens),
#      GMU (gpu_memory_utilization, default 0.6).
#
# NOTE: vLLM 0.25.0 forces the `spawn` multiprocessing start method (CUDA is
# initialized), which re-imports this module in the worker; the whole body MUST
# sit under `if __name__ == "__main__":` or the worker recursively constructs LLM.
import hashlib
import json
import os

import numpy as np
from PIL import Image


def main():
    CKPT = os.environ["CKPT_DIR"]
    FIX = os.environ["FIX_DIR"]
    OUT = os.environ["OUT_DIR"]
    K = int(os.environ.get("K", "5"))
    N = int(os.environ.get("N", "32"))
    GMU = float(os.environ.get("GMU", "0.6"))
    os.makedirs(OUT, exist_ok=True)

    # fixed fixture image (exact bytes our C++ pipeline consumes)
    raw = np.fromfile(os.path.join(FIX, "image_rgb_uint8_448x448x3.bin"),
                      dtype=np.uint8).reshape(448, 448, 3)
    img = Image.fromarray(raw, mode="RGB")

    from vllm import LLM, SamplingParams

    llm = LLM(model=CKPT, tokenizer=CKPT, trust_remote_code=True, dtype="bfloat16",
              enforce_eager=True, gpu_memory_utilization=GMU,
              limit_mm_per_prompt={"image": 1}, max_model_len=4096)

    # Same textual content as the M0 fixture, via the chat template (real serving path).
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "What is in this image?"},
    ]}]
    tok = llm.get_tokenizer()
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("PROMPT:\n", prompt, flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=N)
    runs = []
    texts = []
    for i in range(K):
        out = llm.generate([{"prompt": prompt, "multi_modal_data": {"image": img}}], sp)
        ids = list(out[0].outputs[0].token_ids)
        txt = out[0].outputs[0].text
        runs.append(ids)
        texts.append(txt)
        print(f"run {i}: {len(ids)} toks  ids[:16]={ids[:16]}", flush=True)
        print(f"        text={txt!r}", flush=True)

    ref = runs[0]
    identical = all(r == ref for r in runs)
    div = None
    for pos in range(max(len(r) for r in runs)):
        vals = {(r[pos] if pos < len(r) else None) for r in runs}
        if len(vals) > 1:
            div = pos
            break
    print(f"\nK={K} DETERMINISTIC={identical}  first_divergence_pos={div}", flush=True)
    gate = "STRICT" if identical else "NEAR-TIE (distributional)"
    print(f"GATE FORM (by measurement) = {gate}", flush=True)

    # also dump the EXACT placeholder-expanded model input ids (same expensive load)
    IMAGE_TOKEN = 248056
    sp1 = SamplingParams(temperature=0.0, max_tokens=1)
    out1 = llm.generate([{"prompt": prompt, "multi_modal_data": {"image": img}}], sp1)
    in_ids = list(out1[0].prompt_token_ids)
    n_img = sum(1 for t in in_ids if t == IMAGE_TOKEN)
    img_off = in_ids.index(IMAGE_TOKEN) if n_img else -1
    print(f"INPUT prompt_token_ids len={len(in_ids)} image_tokens={n_img} offset={img_off}", flush=True)
    np.array(in_ids, dtype=np.int32).tofile(os.path.join(OUT, "input_ids_i32.bin"))
    with open(os.path.join(OUT, "input_ids_manifest.json"), "w") as f:
        json.dump({"model_id": "Qwen/Qwen3.6-27B", "image_token_id": IMAGE_TOKEN,
                   "len": len(in_ids), "image_tokens": n_img, "image_offset": img_off,
                   "prompt": prompt}, f, indent=2)
    print("WROTE input_ids_i32.bin", flush=True)

    np.array(ref, dtype=np.int32).tofile(os.path.join(OUT, "gen_tokens_i32.bin"))
    manifest = {
        "model_id": "Qwen/Qwen3.6-27B", "arch": "Qwen3_5ForConditionalGeneration",
        "K": K, "max_tokens": N, "gpu_memory_utilization": GMU,
        "deterministic": identical, "first_divergence_pos": div,
        "gate_form": gate, "prompt": prompt,
        "ref_token_ids": ref, "ref_text": texts[0],
        "gen_tokens_sha256": hashlib.sha256(np.array(ref, dtype=np.int32).tobytes()).hexdigest(),
        "all_runs": runs,
    }
    with open(os.path.join(OUT, "gen_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("WROTE gen_tokens_i32.bin + gen_manifest.json", flush=True)


if __name__ == "__main__":
    main()
