#!/usr/bin/env python3
# Teacher-forced near-tie gap golden using HuggingFace transformers (CPU/GPU).
# Same outputs as scripts/qwen3-neartie-gap.py but does not require a CUDA vLLM
# oracle venv — useful on aarch64 Blackhole hosts that only have torch+cpu.
#
# Emits into --golden-dir (default names; override with --ids-name / --gap-name):
#   our_ids*.npy           [N,T] i32
#   neartie_gap_mnats*.npy [N,T] i32  (milli-nats; 99_999_000 = outside top-K)
#
# Example (Tenstorrent bootstrap dump):
#   python scripts/qwen3-neartie-gap-transformers.py \
#     --model Qwen/Qwen3-0.6B \
#     --golden-dir tests/parity/goldens/qwen3_greedy_0_6b \
#     --our-ids our_ids_tenstorrent.i32 \
#     --ids-name our_ids_tenstorrent.npy \
#     --gap-name neartie_gap_mnats_tenstorrent.npy
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "The capital of France is",
    "Once upon a time,",
    "In the beginning God created",
    "The quick brown fox jumps over",
    "def fibonacci(n):",
    "Water boils at a temperature of",
    "The theory of relativity was developed by",
    "To be or not to be, that is",
    "The largest planet in our solar system is",
    "Machine learning is a subfield of",
    "The mitochondria is the powerhouse of",
    "Roses are red, violets are",
    "The first president of the United States was",
    "E equals m c",
    "A journey of a thousand miles begins with",
    "The chemical symbol for gold is",
]
OUTSIDE_TOPK_MNATS = 99_999_000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--golden-dir", required=True)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument(
        "--our-ids",
        default="our_ids.i32",
        help="raw int32 dump from VT_DUMP_IDS (N*T little-endian i32)",
    )
    ap.add_argument("--ids-name", default="our_ids.npy")
    ap.add_argument("--gap-name", default="neartie_gap_mnats.npy")
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
    )
    args = ap.parse_args()

    gdir = args.golden_dir
    N, T = len(PROMPTS), args.max_tokens
    our_path = os.path.join(gdir, args.our_ids)
    if not os.path.isfile(our_path):
        print(f"missing {our_path}", file=sys.stderr)
        return 1
    our = np.fromfile(our_path, dtype="<i4").reshape(N, T)
    greedy_path = os.path.join(gdir, "greedy_ids.npy")
    greedy = np.load(greedy_path) if os.path.isfile(greedy_path) else None

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    device = torch.device("cpu")
    print(f"loading {args.model} dtype={args.dtype} on {device}...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    )
    model.eval()
    model.to(device)

    gap_mnats = np.zeros((N, T), dtype="<i4")
    max_gap = 0.0
    worst = None
    n_div = 0
    print(f"=== teacher-forced near-tie gap (transformers): {args.model} ===")
    with torch.inference_mode():
        for i in range(N):
            pfile = os.path.join(gdir, f"p{i}_prompt.i32")
            if os.path.isfile(pfile):
                prompt_ids = np.fromfile(pfile, dtype="<i4").tolist()
            else:
                prompt_ids = tok.encode(PROMPTS[i], add_special_tokens=True)
            our_ids = [int(x) for x in our[i]]
            full = list(prompt_ids) + our_ids
            # One forward over the full prefix+continuation; use next-token
            # logits at each position j from the prefix length onward.
            ids = torch.tensor([full], device=device, dtype=torch.long)
            logits = model(ids).logits[0]  # [seq, vocab]
            P = len(prompt_ids)
            for j in range(T):
                # Logits at position P+j-1 predict token at P+j (our_ids[j]).
                pos = P + j - 1
                if pos < 0:
                    gap_mnats[i, j] = OUTSIDE_TOPK_MNATS
                    continue
                row = logits[pos].float()
                logp = torch.log_softmax(row, dim=-1)
                our_tid = our_ids[j]
                topv, topi = torch.topk(logp, k=min(args.topk, logp.numel()))
                top_set = {int(t) for t in topi.tolist()}
                arg_tid = int(topi[0].item())
                arg_lp = float(topv[0].item())
                if our_tid in top_set:
                    our_lp = float(logp[our_tid].item())
                    gap = max(0.0, arg_lp - our_lp)
                    gap_mnats[i, j] = int(round(gap * 1000.0))
                    if gap > max_gap:
                        max_gap, worst = gap, (i, j, gap)
                else:
                    gap_mnats[i, j] = OUTSIDE_TOPK_MNATS
                    print(
                        f"  p{i:2d} tok{j:2d}: OUR TOKEN {our_tid} OUTSIDE top-{args.topk}"
                    )
                if greedy is not None and our_ids[j] != int(greedy[i, j]):
                    n_div += 1
                    print(
                        f"  p{i:2d} tok{j:2d}: our={our_tid} vLLM_greedy={int(greedy[i, j])}"
                        f" tf_argmax={arg_tid} gap={gap_mnats[i, j] / 1000.0:.4f} nats"
                    )
            print(f"  prompt {i}/{N - 1} done")

    ids_out = os.path.join(gdir, args.ids_name)
    gap_out = os.path.join(gdir, args.gap_name)
    np.save(ids_out, our)
    np.save(gap_out, gap_mnats)
    print(
        f"=== {n_div} token-divergent vs greedy_ids; max gap {max_gap:.4f} nats "
        f"(worst {worst}) ==="
    )
    print(f"wrote {ids_out} + {gap_out} {gap_mnats.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
