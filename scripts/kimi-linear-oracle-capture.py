#!/usr/bin/env python3
# Kimi-Linear-48B-A3B (KimiLinearForCausalLM) — vLLM 0.25.0 oracle greedy golden,
# the §8 SACRED capture. Mirrors scripts/deepseek-v2-oracle-capture.py (MLA+MoE
# analog) 1:1 on recipe: per-prompt batch=1 generation, K greedy runs to establish
# vLLM self-determinism (STRICT vs distributional gate), moe_backend=triton
# (MANDATORY on GB10 — the FlashInfer CUTLASS MoE workspace OOM-reboots the box),
# enforce_eager, tiny profiling batch.
#
# CAPTURED 2026-08-06 on dgx (GB10, vLLM 0.25.0-stage, util=0.82, moe=triton,
# enforce_eager, max_num_batched_tokens=512, max_num_seqs=1, K=3, T=16, full 27
# layers): min available memory 15 GiB throughout (no reboot). Verdict:
# ALL 8 PROMPTS DETERMINISTIC over K=3 -> STRICT token-exact gate (as spec §4
# expected for a 48.9B MoE). Golden committed under
# tests/parity/goldens/kimi_linear_greedy/. NOTE --num-layers truncation is
# INCOMPATIBLE with the Kimi hybrid schedule in vLLM 0.25.0 (loader KeyError on
# layers.4.block_sparse_moe.* after hf_overrides); the full run needs no override.
#
# MEMORY (GB10 119 GiB UNIFIED pool, reboots on overshoot): vLLM footprint ~=
# gpu_memory_utilization * 119. Weights ~91 GiB => util MUST be ~0.80-0.85 just to
# hold weights+KV (a lower util errors GRACEFULLY after weight-load, no reboot).
# Default 0.82 => ~97.6 GiB footprint, ~21 GiB headroom (the "~20 GiB slack"),
# with the profiling spike bounded by triton MoE + max_num_batched_tokens=512.
# Run oracle ALONE, drop_caches first, free -g >= 90 before load.
#
# --num-layers N truncates to N hidden layers (hf_overrides) for a memory-SAFE
# reduced-layer smoke golden (§8 step 4) before the full 27-layer run.
import argparse
import os
import sys

import numpy as np

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "In a shocking finding, scientists discovered a herd of unicorns living in",
    "Q: What is 17 * 23?\nA:",
    "The three laws of robotics are",
    "Once upon a time, in a land far away,",
    "The chemical symbol for gold is",
    "To be or not to be, that is",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=os.environ.get(
        "KIMI_MODEL", "moonshotai/Kimi-Linear-48B-A3B-Instruct"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--runs", type=int, default=int(os.environ.get("KIMI_RUNS", "3")))
    ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("KIMI_MAXTOK", "16")))
    ap.add_argument("--gpu-mem-util", type=float,
                    default=float(os.environ.get("KIMI_GPU_UTIL", "0.82")))
    ap.add_argument("--max-model-len", type=int,
                    default=int(os.environ.get("KIMI_MAX_LEN", "2048")))
    ap.add_argument("--max-num-batched-tokens", type=int,
                    default=int(os.environ.get("KIMI_MAX_BATCHED", "512")))
    ap.add_argument("--max-num-seqs", type=int,
                    default=int(os.environ.get("KIMI_MAX_SEQS", "1")))
    ap.add_argument("--moe-backend", default=os.environ.get("KIMI_MOE_BACKEND", "triton"))
    ap.add_argument("--num-layers", type=int,
                    default=int(os.environ.get("KIMI_NUM_LAYERS", "0")),
                    help="0 = full (27); N>0 truncates via hf_overrides for a safe smoke")
    return ap.parse_args()


def generate_per_prompt(llm, sp):
    return {p: llm.generate([p], sp)[0] for p in PROMPTS}


def main():
    args = parse_args()
    from vllm import LLM, SamplingParams

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    N, T, K = len(PROMPTS), args.max_tokens, max(1, args.runs)

    llm_kwargs = dict(model=args.model, dtype="bfloat16", enforce_eager=True,
                      trust_remote_code=True,
                      gpu_memory_utilization=args.gpu_mem_util,
                      max_model_len=args.max_model_len,
                      max_num_batched_tokens=args.max_num_batched_tokens,
                      max_num_seqs=args.max_num_seqs)
    if args.moe_backend:
        llm_kwargs["moe_backend"] = args.moe_backend
    if args.num_layers and args.num_layers > 0:
        llm_kwargs["hf_overrides"] = {"num_hidden_layers": args.num_layers}
    print(f"model={args.model} util={args.gpu_mem_util} moe={args.moe_backend} "
          f"num_layers={args.num_layers or 'full'} max_batched={args.max_num_batched_tokens} "
          f"N={N} T={T} K={K}", flush=True)
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=T)

    dist = np.full((N, T, K), -1, dtype="<i4")
    run0 = np.zeros((N, T), dtype="<i4")
    for k in range(K):
        by_prompt = generate_per_prompt(llm, sp)
        for i, p in enumerate(PROMPTS):
            o = by_prompt[p]
            ids = list(o.outputs[0].token_ids)
            for j in range(min(T, len(ids))):
                dist[i, j, k] = ids[j]
            if k == 0:
                for j in range(min(T, len(ids))):
                    run0[i, j] = ids[j]
                np.array(o.prompt_token_ids, dtype="<i4").tofile(
                    os.path.join(out_dir, f"p{i}_prompt.i32"))
                print(f"prompt[{i}] {p!r}", flush=True)
                print(f"    prompt_ids={list(o.prompt_token_ids)}", flush=True)
                print(f"    -> {ids}  ({o.outputs[0].text!r})", flush=True)

    np.save(os.path.join(out_dir, "greedy_ids.npy"), run0)
    np.save(os.path.join(out_dir, "greedy_dist.npy"), dist)

    print(f"\n=== vLLM SELF-DETERMINISM: N={N} T={T} K={K} ===", flush=True)
    deterministic = True
    for i in range(N):
        seqs = {tuple(int(x) for x in dist[i, :, k]) for k in range(K)}
        multi = [j for j in range(T) if len({int(dist[i, j, k]) for k in range(K)}) > 1]
        if len(seqs) > 1:
            deterministic = False
            print(f"  prompt[{i}] NON-DET: {len(seqs)} seqs; near-tie pos {multi}", flush=True)
        else:
            print(f"  prompt[{i}] deterministic ({K} runs)", flush=True)
    verdict = "ALL DETERMINISTIC -> STRICT token-exact gate" if deterministic \
        else "NON-DETERMINISTIC -> distributional near-tie gate"
    print(f"=== {verdict} ===", flush=True)
    print(f"wrote {out_dir}/greedy_ids.npy {run0.shape}", flush=True)


if __name__ == "__main__":
    main()
