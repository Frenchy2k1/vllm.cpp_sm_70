#!/usr/bin/env python3
"""M3d — capture the vLLM 0.25.0 VIDEO oracle for a FIXED (video, prompt) on
Qwen/Qwen3.6-27B (Qwen3_5ForConditionalGeneration, the GDN-hybrid gate model).

This is the 27B sibling of scripts/mm/m3c_video_oracle_capture.py. It reuses the
IDENTICAL fixed synthetic clip (same RNG seed / NF / HW / FPS) so the C++ video
processor consumes byte-identical pixels; only the model, the video/image token
ids (read from hf_config), and the golden tokens change. The 27B tower has EMPTY
deepstack, so there is no multiscale golden here — only the merger stream matters.

Two captures over the same clip:
  1. processor dump (cheap): pixel_values_videos (bf16-as-f32 production golden),
     video_grid_thw, the placeholder-expanded prompt ids, timestamps, video-token
     count/offset — self-contained fixture for the C++ e2e gate.
  2. generation golden (loads the 54 GiB bf16 27B): greedy enforce_eager K-run
     self-determinism -> gate form (STRICT vs near-tie) + golden token ids/text.

Env: OUT_DIR (default ~/mm_video_fixture_27b), K (default 5), N (max_tokens 32),
     GMU (gpu_memory_utilization, default 0.6 — proven safe for the 27B image
     oracle; the pool is UNIFIED so keep it LOW and run ALONE), NF (8), HW (128),
     FPS (2.0). Body under __main__ (vLLM spawn).
"""
import hashlib
import json
import os

import numpy as np


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    MODEL = os.environ.get("MODEL", "Qwen/Qwen3.6-27B")
    OUT = os.environ.get("OUT_DIR", os.path.expanduser("~/mm_video_fixture_27b"))
    K = int(os.environ.get("K", "5"))
    N = int(os.environ.get("N", "32"))
    GMU = float(os.environ.get("GMU", "0.6"))
    NF = int(os.environ.get("NF", "8"))
    HW = int(os.environ.get("HW", "128"))
    FPS = float(os.environ.get("FPS", "2.0"))
    os.makedirs(OUT, exist_ok=True)

    import torch

    # ---- deterministic synthetic video fixture (raw uint8 T,H,W,3) ----
    # SAME seed/params as m3c_video_oracle_capture.py -> byte-identical clip.
    rng = np.random.RandomState(20260725)
    video = rng.randint(0, 256, size=(NF, HW, HW, 3), dtype=np.uint8)
    assert video.flags.c_contiguous
    raw_name = f"video_rgb_uint8_{NF}x{HW}x{HW}x3.bin"
    video.tofile(os.path.join(OUT, raw_name))

    metadata = {
        "fps": FPS,
        "duration": NF / FPS,
        "total_num_frames": NF,
        "frames_indices": list(range(NF)),
        "video_backend": "opencv",
        "do_sample_frames": False,
    }

    # ---- (1) processor dump ----
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY

    model_config = ModelConfig(model=MODEL, limit_mm_per_prompt={"image": 0, "video": 1})
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    info = processor.info
    hf_config = info.get_hf_config()

    video_token_id = int(hf_config.video_token_id)
    vision_start_token_id = int(hf_config.vision_start_token_id)
    vision_end_token_id = int(hf_config.vision_end_token_id)
    image_token_id = int(hf_config.image_token_id)

    prompt = "<|vision_start|><|video_pad|><|vision_end|>What is happening in this video?"
    mm_data = {"video": [(video, metadata)]}
    out = processor(
        prompt,
        mm_items=info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs={},
    )

    def get(o, k):
        try:
            return o[k]
        except Exception:
            return getattr(o, k)

    prompt_ids = list(get(out, "prompt_token_ids"))
    mm_kwargs = get(out, "mm_kwargs")
    mm_hashes = get(out, "mm_hashes")

    item = mm_kwargs["video"][0]
    pixel_values_videos = item["pixel_values_videos"].data
    video_grid_thw = item["video_grid_thw"].data
    timestamps = None
    for tk in ("timestamps", "second_per_grid_ts"):
        if tk in item:
            try:
                timestamps = item[tk].data
            except Exception:
                timestamps = item[tk]
            break

    pv = pixel_values_videos.cpu().to(torch.float32).contiguous().numpy()
    gt = video_grid_thw.cpu().to(torch.int64).contiguous().numpy()
    pv.tofile(os.path.join(OUT, "pixel_values_videos_f32.bin"))
    gt.tofile(os.path.join(OUT, "video_grid_thw_i64.bin"))

    np.array(prompt_ids, dtype=np.int32).tofile(os.path.join(OUT, "input_ids_i32.bin"))
    n_video_tokens = int(prompt_ids.count(video_token_id))
    video_offset = prompt_ids.index(video_token_id) if n_video_tokens else -1
    ts_list = [float(x) for x in (list(timestamps) if timestamps is not None else [])]

    hfp = info.get_hf_processor()
    vproc = hfp.video_processor
    merge_size = int(vproc.merge_size)
    manifest = {
        "model_id": MODEL,
        "arch": "Qwen3_5ForConditionalGeneration",
        "video": {
            "shape": list(video.shape), "dtype": "uint8", "layout": "THWC_RGB",
            "sha256": sha256_hex(video.tobytes()), "raw_file": raw_name,
            "metadata": metadata,
        },
        "pixel_values_videos": {
            "shape": list(pv.shape), "dtype": "bfloat16_as_float32",
            "sha256": sha256_hex(pv.tobytes()), "file": "pixel_values_videos_f32.bin",
        },
        "video_grid_thw": {
            "shape": list(gt.shape), "dtype": "int64", "values": gt.tolist(),
            "file": "video_grid_thw_i64.bin",
        },
        "timestamps": ts_list,
        "prompt": prompt,
        "expanded_prompt_token_ids": prompt_ids,
        "expanded_len": len(prompt_ids),
        "n_video_tokens": n_video_tokens,
        "video_first_offset": video_offset,
        "mm_hash": mm_hashes["video"][0],
        "config": {
            "video_token_id": video_token_id,
            "image_token_id": image_token_id,
            "vision_start_token_id": vision_start_token_id,
            "vision_end_token_id": vision_end_token_id,
            "patch_size": int(vproc.patch_size),
            "temporal_patch_size": int(vproc.temporal_patch_size),
            "merge_size": merge_size,
            "image_mean": list(vproc.image_mean),
            "image_std": list(vproc.image_std),
        },
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("=== M3d 27B VIDEO PROCESSOR DUMP OK ===", flush=True)
    print("video sha256      :", manifest["video"]["sha256"], flush=True)
    print("pixel_values_videos:", pv.shape, "sha", manifest["pixel_values_videos"]["sha256"], flush=True)
    print("video_grid_thw    :", gt.tolist(), flush=True)
    print("timestamps        :", ts_list, flush=True)
    print("video_token_id    :", video_token_id, "vision_start", vision_start_token_id,
          "vision_end", vision_end_token_id, flush=True)
    print("expanded len      :", len(prompt_ids), "n_video_tokens", n_video_tokens,
          "first_offset", video_offset, flush=True)
    print("mm_hash           :", manifest["mm_hash"], flush=True)

    # ---- (2) generation golden ----
    from vllm import LLM, SamplingParams

    llm = LLM(model=MODEL, tokenizer=MODEL, trust_remote_code=True, dtype="bfloat16",
              enforce_eager=True, gpu_memory_utilization=GMU,
              limit_mm_per_prompt={"image": 0, "video": 1}, max_model_len=4096)
    tok = llm.get_tokenizer()
    messages = [{"role": "user", "content": [
        {"type": "video"},
        {"type": "text", "text": "What is happening in this video?"},
    ]}]
    chat_prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("CHAT PROMPT:\n", chat_prompt, flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=N)
    runs, texts = [], []
    req = {"prompt": chat_prompt, "multi_modal_data": {"video": (video, metadata)}}
    for i in range(K):
        o = llm.generate([req], sp)
        ids = list(o[0].outputs[0].token_ids)
        runs.append(ids)
        texts.append(o[0].outputs[0].text)
        print(f"run {i}: {len(ids)} toks ids[:16]={ids[:16]}", flush=True)
        print(f"        text={texts[-1]!r}", flush=True)

    ref = runs[0]
    identical = all(r == ref for r in runs)
    div = None
    for pos in range(max(len(r) for r in runs)):
        vals = {(r[pos] if pos < len(r) else None) for r in runs}
        if len(vals) > 1:
            div = pos
            break
    gate = "STRICT" if identical else "NEAR-TIE (distributional)"
    print(f"\nK={K} DETERMINISTIC={identical} first_divergence={div} GATE={gate}", flush=True)

    sp1 = SamplingParams(temperature=0.0, max_tokens=1)
    o1 = llm.generate([req], sp1)
    gen_in_ids = list(o1[0].prompt_token_ids)
    np.array(gen_in_ids, dtype=np.int32).tofile(os.path.join(OUT, "gen_input_ids_i32.bin"))
    n_vid_gen = sum(1 for t in gen_in_ids if t == video_token_id)
    vid_off_gen = gen_in_ids.index(video_token_id) if n_vid_gen else -1
    print(f"GEN input ids len={len(gen_in_ids)} video_tokens={n_vid_gen} first_off={vid_off_gen}", flush=True)

    np.array(ref, dtype=np.int32).tofile(os.path.join(OUT, "gen_tokens_i32.bin"))
    gmanifest = {
        "model_id": MODEL, "arch": "Qwen3_5ForConditionalGeneration",
        "K": K, "max_tokens": N, "gpu_memory_utilization": GMU,
        "deterministic": identical, "first_divergence_pos": div, "gate_form": gate,
        "chat_prompt": chat_prompt, "ref_token_ids": ref, "ref_text": texts[0],
        "gen_tokens_sha256": sha256_hex(np.array(ref, dtype=np.int32).tobytes()),
        "gen_input_ids_len": len(gen_in_ids), "gen_video_tokens": n_vid_gen,
        "gen_video_first_offset": vid_off_gen, "all_runs": runs,
    }
    with open(os.path.join(OUT, "gen_manifest.json"), "w") as f:
        json.dump(gmanifest, f, indent=2)
    print("WROTE gen_tokens_i32.bin + gen_input_ids_i32.bin + gen_manifest.json", flush=True)


if __name__ == "__main__":
    main()
