# Cross-engine A/B, ORACLE side, in vLLM's PRODUCTION GRAPHED config.
#
# Every earlier oracle number in this campaign used enforce_eager=True, which the
# house rule forbids as a denominator ("the honest denominator is vLLM's
# production configuration, never --enforce-eager"). This drops it, so cudagraph
# capture is on, which is what a user actually gets.
import json
import os
import sys
import time

import vllm
from vllm import LLM, SamplingParams

# ASSERT THE ORACLE'S IDENTITY (issue #375). The `vllm-oracle` symlink points at
# the preserved v0.25.0 ROLLBACK, not the pin, so a run that merely "works" can
# be measuring the wrong reference -- a deterministic wrong oracle is
# indistinguishable from a right one. .agents/upstream-sync.md pins commit
# 555967922 with FlashInfer 0.6.15.post1 / Torch 2.13.0 / transformers 5.14.1.
PIN_COMMIT = "555967922"
PIN_FLASHINFER = "0.6.15.post1"


def assert_oracle_identity():
    import flashinfer
    problems = []
    if PIN_COMMIT not in vllm.__version__:
        problems.append(f"vllm {vllm.__version__} does not carry pin {PIN_COMMIT}")
    if flashinfer.__version__ != PIN_FLASHINFER:
        problems.append(
            f"flashinfer {flashinfer.__version__} != pinned {PIN_FLASHINFER}")
    if problems:
        raise SystemExit("ORACLE IDENTITY MISMATCH, aborting:\n  " +
                         "\n  ".join(problems))
    print(f"oracle identity OK: vllm={vllm.__version__} "
          f"flashinfer={flashinfer.__version__}")

LANES = {
    "35b": ("nvidia/Qwen3.6-35B-A3B-NVFP4",
            "RedHatAI/Qwen3.6-35B-A3B-speculator.dspark", 8),
    "27b": ("unsloth/Qwen3.6-27B-NVFP4", "satgeze/Qwen3.6-27B-DSpark", 15),
}
PROMPTS = ["The capital of France is", "def fibonacci(n):\n    "]
MAX_TOKENS = 128


def main():
    assert_oracle_identity()
    lane, arm = sys.argv[1], sys.argv[2]
    target, draft, k = LANES[lane]
    greedy = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    kwargs = dict(
        model=target,
        max_model_len=2048,
        max_num_seqs=2,          # matches our arm; also bounds the state budget
        gpu_memory_utilization=0.55,
        enable_prefix_caching=False,
        disable_log_stats=False,
        # NO enforce_eager: this is the graphed production config.
    )
    if lane == "27b":
        kwargs["revision"] = "890bdef7a42feba6d83b6e17a03315c694112f2a"
    if arm == "on":
        kwargs["speculative_config"] = {
            "method": "dspark", "model": draft, "num_speculative_tokens": k,
        }

    llm = LLM(**kwargs)
    llm.generate([PROMPTS[0]], greedy)          # warm-up, discarded

    rows, timing = [], []
    for p in PROMPTS:
        t0 = time.perf_counter()
        res = llm.generate([p], greedy)
        dt = time.perf_counter() - t0
        o = res[0].outputs[0]
        rows.append({"prompt": p, "text": o.text, "token_ids": list(o.token_ids)})
        timing.append({"tokens": len(o.token_ids), "secs": dt,
                       "tok_s": len(o.token_ids) / dt})

    metrics = {}
    try:
        for m in llm.get_metrics():
            n = getattr(m, "name", "")
            if "spec_decode" in n:
                metrics[n] = getattr(m, "value", getattr(m, "sum", None))
    except Exception as e:
        metrics = {"error": repr(e)}

    out = {"engine": "vllm-oracle-PINNED-GRAPHED", "vllm_version": vllm.__version__,
           "lane": lane, "arm": arm, "target": target, "draft": draft, "k": k,
           "graphed": True, "rows": rows, "timing": timing, "metrics": metrics,
           "median_tok_s": sorted(t["tok_s"] for t in timing)[len(timing) // 2]}
    path = os.path.expanduser(f"~/work/dspark-w6/pinned_{lane}_{arm}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"WROTE {path} median_tok_s={out['median_tok_s']:.3f}")


if __name__ == "__main__":
    main()
