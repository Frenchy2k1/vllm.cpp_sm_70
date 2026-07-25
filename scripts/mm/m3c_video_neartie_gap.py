#!/usr/bin/env python3
# M3c VIDEO near-tie-robust CORRECTNESS golden (see [[near-tie-distributional-gate]];
# mirrors scripts/qwen3-neartie-gap.py for the multimodal video e2e).
#
# The video e2e's sole divergence vs vLLM's greedy golden is a bf16 near-tie in the
# SHARED decode driven by the M2a tower's bf16 envelope (rel-L2 0.072). The honest
# "mirror vLLM" bar: is OUR token, given OUR exact prefix, one vLLM's OWN logits
# cannot confidently separate from vLLM's argmax? This TEACHER-FORCES vLLM 0.25.0 on
# OUR engine's exact generated VIDEO sequence (our_ids_i32.bin, dumped by the gate
# with VT_DUMP_IDS=1) WITH the identical video mm input, and records, per position,
# the gap in MILLI-nats between vLLM's argmax logprob and OUR token's logprob. A tiny
# gap (<= 500 mnats = 0.5 nats) is a bf16 near-tie (structurally correct); a large
# gap (or our token outside vLLM's top-K) is a REAL forward divergence the gate fails.
#
# Emits into the fixture dir (default tests/vllm/multimodal/fixtures/qwen3vl_video):
#   neartie_gap_mnats_i32.bin  [T] i32  vLLM teacher-forced gap per OUR token
#                                       (0 = our token IS vLLM's argmax; 99_999_000 =
#                                       our token outside vLLM top-K => gate fails).
# Reuses the committed video fixture (raw video + gen_manifest chat prompt); the
# UN-expanded prompt (single <|video_pad|>) is re-expanded to 64 by vLLM when the
# mm video is attached, matching the e2e's gen_input_ids.
#
# Run on dgx with the oracle venv:
#   PATH=$HOME/venvs/vllm-oracle/bin:$PATH VLLM_USE_V1=1 \
#     ~/venvs/vllm-oracle/bin/python scripts/mm/m3c_video_neartie_gap.py \
#       --fixture-dir tests/vllm/multimodal/fixtures/qwen3vl_video
import argparse
import json
import os

import numpy as np

OUTSIDE_TOPK_MNATS = 99_999_000  # our token not even in vLLM top-K => real bug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--gpu-mem-util", type=float, default=0.40)
    args = ap.parse_args()

    FIX = args.fixture_dir
    gman = json.load(open(os.path.join(FIX, "gen_manifest.json")))
    man = json.load(open(os.path.join(FIX, "manifest.json")))
    golden = list(gman["ref_token_ids"])
    our = np.fromfile(os.path.join(FIX, "our_ids_i32.bin"), dtype="<i4").tolist()
    T = len(our)
    assert T == len(golden), f"our({T}) != golden({len(golden)})"

    vshape = man["video"]["shape"]
    NF, HW = int(vshape[0]), int(vshape[1])
    raw = man["video"]["raw_file"]
    video = np.fromfile(os.path.join(FIX, raw), dtype=np.uint8).reshape(NF, HW, HW, 3)
    metadata = man["video"]["metadata"]

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, tokenizer=args.model, trust_remote_code=True,
              dtype="bfloat16", enforce_eager=True, gpu_memory_utilization=args.gpu_mem_util,
              limit_mm_per_prompt={"image": 0, "video": 1}, max_model_len=4096)
    tok = llm.get_tokenizer()

    # UN-expanded prompt (single <|video_pad|>) + OUR tokens; vLLM re-expands the
    # video_pad->64 when mm_data is attached (== the e2e's gen_input_ids prefix).
    unexp = tok.encode(gman["chat_prompt"], add_special_tokens=False)
    full = list(unexp) + list(our)
    req = {"prompt_token_ids": full, "multi_modal_data": {"video": (video, metadata)}}
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=args.topk)
    out = llm.generate([req], sp)[0]
    plp = out.prompt_logprobs
    expanded = list(out.prompt_token_ids)
    P = len(expanded) - T
    assert expanded[P:] == list(our), "teacher-forced tail != our_ids (prompt reconstruction)"
    print(f"expanded prompt len={len(expanded)} base={P} (our tokens @ [{P}:{P+T}])", flush=True)

    gap_mnats = np.zeros((T,), dtype="<i4")
    max_gap, worst, n_div = 0.0, None, 0
    print(f"=== teacher-forced near-tie gap: {args.model} VIDEO (OUR prefix) ===", flush=True)
    for j in range(T):
        d = plp[P + j] or {}
        m = {int(t): float(v.logprob) for t, v in d.items()}
        arg = max(m, key=m.get) if m else -1
        arg_lp = m[arg] if m else 0.0
        ot = int(our[j])
        if ot in m:
            gp = max(0.0, arg_lp - m[ot])
            gap_mnats[j] = int(round(gp * 1000.0))
            if gp > max_gap:
                max_gap, worst = gp, (j, gp)
        else:
            gap_mnats[j] = OUTSIDE_TOPK_MNATS
            print(f"  tok{j:2d}: OUR TOKEN {ot} OUTSIDE vLLM top-{args.topk} (REAL divergence)",
                  flush=True)
        if ot != int(golden[j]):
            n_div += 1
            print(f"  tok{j:2d}: our={ot} vLLM_greedy={int(golden[j])} vLLM_argmax={arg} "
                  f"gap={gap_mnats[j] / 1000.0:.4f} nats", flush=True)

    gap_mnats.tofile(os.path.join(FIX, "neartie_gap_mnats_i32.bin"))
    print(f"=== {n_div} token-divergent positions vs vLLM greedy; "
          f"max near-tie gap {max_gap:.4f} nats (worst tok {worst}) ===", flush=True)
    print(f"wrote {FIX}/neartie_gap_mnats_i32.bin  [{T}] i32", flush=True)
    verdict = "NEAR-TIE-ROBUST PASS" if int(gap_mnats.max()) <= 500 else "FORWARD DIVERGENCE"
    print(f"GATE PREVIEW: max gap {int(gap_mnats.max())} mnats => {verdict}", flush=True)


if __name__ == "__main__":
    main()
