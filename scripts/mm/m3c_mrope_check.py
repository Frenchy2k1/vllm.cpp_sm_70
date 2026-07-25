#!/usr/bin/env python3
"""Localize M3c: compare vLLM's video get_rope_index positions to our C++
Qwen3VLGetRopeIndexVideo algorithm (replicated here) on the EXACT gen_input_ids +
video_grid_thw of the e2e fixture. If they match, the video temporal MRoPE is
exact and the e2e divergence is a downstream bf16 near-tie, not a position bug."""
import json
import os

import numpy as np


def our_get_rope_index_video(ids, grid_thw, sms, vs_id, vid_id, ve_id):
    T = len(ids)
    grid_t, h, w = grid_thw
    llm_h, llm_w = h // sms, w // sms
    expected = llm_h * llm_w
    rows = [[], [], []]
    st = 0
    last_max = -1
    any_ = False

    def find_from(tok, frm, upto):
        for i in range(frm, upto):
            if ids[i] == tok:
                return i
        return -1

    def append_text(st_idx, text_len):
        nonlocal last_max
        for i in range(text_len):
            p = st_idx + i
            rows[0].append(p); rows[1].append(p); rows[2].append(p)
        if text_len > 0:
            last_max = max(last_max, st_idx + text_len - 1)

    search = 0
    for f in range(grid_t):
        vs = find_from(vs_id, search, T)
        ve = find_from(ve_id, vs, T)
        vid = find_from(vid_id, vs, ve)
        actual = (ve - vid) if vid >= 0 else 0
        video_offset = vid if vid >= 0 else vs + 1
        if actual == 0:
            search = ve + 1; continue
        text_len = video_offset - st
        st_idx = (last_max + 1) if any_ else 0
        append_text(st_idx, text_len)
        bias = text_len + st_idx
        for j in range(expected):
            h_idx, w_idx = j // llm_w, j % llm_w
            rows[0].append(bias); rows[1].append(bias + h_idx); rows[2].append(bias + w_idx)
        last_max = max(last_max, bias + max(llm_h - 1, llm_w - 1))
        st = video_offset + actual
        search = ve + 1
        any_ = True
    if st < T:
        st_idx = (last_max + 1) if any_ else 0
        append_text(st_idx, T - st)
    arr = np.array(rows, dtype=np.int64)
    delta = int(arr.max()) + 1 - T
    return arr, delta


def main():
    OUT = os.environ.get("OUT_DIR", os.path.expanduser("~/mm_video_fixture"))
    ids = np.fromfile(os.path.join(OUT, "gen_input_ids_i32.bin"), dtype=np.int32).tolist()
    man = json.load(open(os.path.join(OUT, "manifest.json")))
    grid = man["video_grid_thw"]["values"]
    c = man["config"]
    vs_id, vid_id, ve_id = c["vision_start_token_id"], c["video_token_id"], c["vision_end_token_id"]
    sms = c["merge_size"]

    ours, our_delta = our_get_rope_index_video(ids, grid, sms, vs_id, vid_id, ve_id)
    print("OURS delta:", our_delta, "shape", ours.shape)

    # vLLM reference
    from transformers import AutoConfig
    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
    from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
    import torch

    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    # locate the video placeholder block start (first vision_start)
    off = ids.index(vs_id)
    length = 0
    # block spans until the last vision_end of the video (grid_t frames)
    pos = off
    for _ in range(grid[0]):
        pos = ids.index(ve_id, ids.index(vs_id, pos)) + 1
    length = pos - off
    feat = MultiModalFeatureSpec(
        data={"video_grid_thw": type("X", (), {"data": torch.tensor(grid)})()},
        modality="video", identifier="v0",
        mm_position=PlaceholderRange(offset=off, length=length))
    ref, ref_delta = Qwen3VLForConditionalGeneration._get_mrope_input_positions(
        input_tokens=ids, mm_features=[feat], config=cfg)
    ref = ref.numpy()
    print("VLLM delta:", ref_delta, "shape", ref.shape)

    same = np.array_equal(ours, ref)
    print("POSITIONS EQUAL:", same, "DELTA EQUAL:", our_delta == ref_delta)
    if not same:
        d = np.where(ours != ref)
        print("first mismatch cols:", sorted(set(d[1].tolist()))[:10])
        for col in sorted(set(d[1].tolist()))[:6]:
            print(f"  col {col}: ours={ours[:,col].tolist()} vllm={ref[:,col].tolist()} id={ids[col]}")
    else:
        print("row0[:30]:", ours[0, :30].tolist())


if __name__ == "__main__":
    main()
