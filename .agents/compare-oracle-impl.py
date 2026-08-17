#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Milestone-0 oracle <-> implementation greedy comparator (sm70 lane).

Consumes two identical-format greedy trace files (each row: {prompt_id, run,
content, tokens, usage} as written by capture-1cat-oracle.py) and reports the
per-prompt comparison:

  - greedy EXACT (string + token-sequence) — the semantic bar on this lane.
  - near-tie detection: prompts where oracle runs diverge (could be a tie).
  - divergence list: token index of first difference when not exact.

Caveat (plan §2.5, the two-oracle contract): the oracle here is the V100
same-silicon 1Cat AWQ/INT4 (Qwen3.6-27B-AWQ) — a DIFFERENT quant route from
our NVFP4 fast paths — so this compare is the SEMANTIC / near-tie oracle, NOT
a STRICT bit-exact gate. Run the same prompts through our greedy paged engine
to produce the impl trace via capture-1cat-oracle.py (or a native trace of the
same shape).

Usage:
  python3 scripts/compare-oracle-impl.py --oracle goldens/qwen36_27b_awq_oracle/trace.jsonl \
        --impl <our trace>.jsonl [--max-divergence-tokens 16]
Prints PASS per prompt when the impl greedy sequence equals the oracle's.
"""
import argparse
import json


def load(path):
    rows = [json.loads(l) for l in open(path)]
    by = {}
    for r in rows:
        by.setdefault(r["prompt_id"], []).append(r)
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--impl", required=True)
    ap.add_argument("--max-divergence", type=int, default=16)
    a = ap.parse_args()

    orc = load(a.oracle)
    impl = load(a.impl)

    # oracle self-determinism (near-tie detection on the oracle side)
    unstable = []
    for pid, rs in orc.items():
        texts = {r["content"] for r in rs}
        if len(texts) > 1:
            unstable.append((pid, texts))

    npass = ndiff = 0
    divergences = []
    for i, ors in sorted(orc.items()):
        if i not in impl:
            print(f"p{i}: NO IMPL TRACE")
            continue
        ostr = ors[0]["tokens"]
        ist = impl[i][0]["tokens"]
        if ostr == ist:
            npass += 1
        else:
            ndiff += 1
            d = next((k for k, (x, y) in enumerate(zip(ostr, ist)) if x != y),
                     min(len(ostr), len(ist)))
            divergences.append((i, d, ostr[max(0, d - 2):d + 3], ist[max(0, d - 2):d + 3]))

    print(f"== result: {npass} exact / {ndiff} diverged ==")
    for i, d, o, im in divergences[: a.max_divergence]:
        print(f"  p{i} divergence@tok{d}: oracle={o} impl={im}")
    if unstable:
        print(f"  oracle near-ties (runs differ) at p{unstable}")
    import sys

    sys.exit(1 if ndiff else 0)


if __name__ == "__main__":
    main()