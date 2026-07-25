#!/usr/bin/env python3
"""M3c golden-only: reuse the already-saved video fixture (from
m3c_video_oracle_capture.py) and capture just the greedy generation golden +
K-run determinism + the chat-templated model input ids. Smaller profiling
footprint (max_model_len bounds the dummy-video frame count) to avoid the 2048
dummy-frame eager-ViT stall. Env: OUT_DIR, K (5), N (32), GMU (0.5), MML (2048)."""
import hashlib
import json
import os

import numpy as np


def main():
    MODEL = "Qwen/Qwen3-VL-4B-Instruct"
    OUT = os.environ.get("OUT_DIR", os.path.expanduser("~/mm_video_fixture"))
    K = int(os.environ.get("K", "5"))
    N = int(os.environ.get("N", "32"))
    GMU = float(os.environ.get("GMU", "0.5"))
    MML = int(os.environ.get("MML", "2048"))

    man = json.load(open(os.path.join(OUT, "manifest.json")))
    v = man["video"]
    T, Hh, Ww = v["shape"][0], v["shape"][1], v["shape"][2]
    video = np.fromfile(os.path.join(OUT, v["raw_file"]), dtype=np.uint8).reshape(T, Hh, Ww, 3)
    metadata = v["metadata"]
    video_token_id = man["config"]["video_token_id"]

    from vllm import LLM, SamplingParams
    print("constructing LLM...", flush=True)
    llm = LLM(model=MODEL, tokenizer=MODEL, trust_remote_code=True, dtype="bfloat16",
              enforce_eager=True, gpu_memory_utilization=GMU,
              limit_mm_per_prompt={"image": 0, "video": 1}, max_model_len=MML,
              max_num_seqs=1)
    print("LLM ready", flush=True)
    tok = llm.get_tokenizer()
    messages = [{"role": "user", "content": [
        {"type": "video"},
        {"type": "text", "text": "What is happening in this video?"},
    ]}]
    chat_prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    sp = SamplingParams(temperature=0.0, max_tokens=N)
    req = {"prompt": chat_prompt, "multi_modal_data": {"video": (video, metadata)}}
    runs, texts = [], []
    for i in range(K):
        o = llm.generate([req], sp)
        ids = list(o[0].outputs[0].token_ids)
        runs.append(ids); texts.append(o[0].outputs[0].text)
        print(f"run {i}: {len(ids)} toks ids[:16]={ids[:16]}", flush=True)
        print(f"        text={texts[-1]!r}", flush=True)

    ref = runs[0]
    identical = all(r == ref for r in runs)
    div = None
    for pos in range(max(len(r) for r in runs)):
        if len({(r[pos] if pos < len(r) else None) for r in runs}) > 1:
            div = pos; break
    gate = "STRICT" if identical else "NEAR-TIE (distributional)"
    print(f"\nK={K} DETERMINISTIC={identical} first_div={div} GATE={gate}", flush=True)

    sp1 = SamplingParams(temperature=0.0, max_tokens=1)
    o1 = llm.generate([req], sp1)
    gen_in = list(o1[0].prompt_token_ids)
    np.array(gen_in, dtype=np.int32).tofile(os.path.join(OUT, "gen_input_ids_i32.bin"))
    n_vid = sum(1 for t in gen_in if t == video_token_id)
    off = gen_in.index(video_token_id) if n_vid else -1
    print(f"GEN input ids len={len(gen_in)} video_tokens={n_vid} first_off={off}", flush=True)

    np.array(ref, dtype=np.int32).tofile(os.path.join(OUT, "gen_tokens_i32.bin"))
    json.dump({"model_id": MODEL, "arch": "Qwen3VLForConditionalGeneration",
               "K": K, "max_tokens": N, "gpu_memory_utilization": GMU, "max_model_len": MML,
               "deterministic": identical, "first_divergence_pos": div, "gate_form": gate,
               "chat_prompt": chat_prompt, "ref_token_ids": ref, "ref_text": texts[0],
               "gen_tokens_sha256": hashlib.sha256(np.array(ref, dtype=np.int32).tobytes()).hexdigest(),
               "gen_input_ids_len": len(gen_in), "gen_video_tokens": n_vid,
               "gen_video_first_offset": off, "all_runs": runs},
              open(os.path.join(OUT, "gen_manifest.json"), "w"), indent=2)
    print("WROTE gen_tokens_i32.bin + gen_input_ids_i32.bin + gen_manifest.json", flush=True)


if __name__ == "__main__":
    main()
