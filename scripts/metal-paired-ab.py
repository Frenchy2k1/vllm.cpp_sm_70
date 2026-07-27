#!/usr/bin/env python3
# Paired A/B for the Metal host, where the naive alternating loop lies.
#
# Measured on the M4 (docs/BENCHMARKS.md, "MEASUREMENT FLOOR"): five IDENTICAL
# runs of one binary spread 10% (23.61 down to 21.31) as the machine heats, and
# the SAME build reads 23.57 standalone, 23.43 running first in a pair and 20.76
# running second. A plain alternating loop therefore attributes drift to whichever
# binary runs second, which is how a tied change first measured 6% "worse".
#
# Two corrections:
#   ABBA blocks       — order-balanced, so linear drift cancels WITHIN a block
#                       rather than accumulating against one arm.
#   cooldown          — idle seconds between runs so each starts from a
#                       comparable thermal state.
# Reports the per-block paired delta plus a sign test, because the paired deltas
# are the only quantity here that drift does not contaminate.
import argparse, json, re, statistics, subprocess, sys, time

TPUT = re.compile(r"Output token throughput \(tok/s\):\s*([0-9.]+)")
DEC = re.compile(r"Mean per-stream decode rate \(tok/s\):\s*([0-9.]+)")

def run(binary, model, args, metric):
    out = subprocess.run([binary, "--model", model] + args, capture_output=True, text=True).stdout
    m = (TPUT if metric == "throughput" else DEC).search(out)
    if not m:
        sys.exit(f"no {metric} in output of {binary}")
    return float(m.group(1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline binary")
    ap.add_argument("--b", required=True, help="candidate binary")
    ap.add_argument("--model", required=True)
    ap.add_argument("--blocks", type=int, default=6, help="ABBA blocks (4 runs each)")
    ap.add_argument("--cooldown", type=float, default=25.0)
    ap.add_argument("--metric", choices=["throughput", "decode"], default="throughput")
    ap.add_argument("--bench-args", default="--num-prompts 8 --input-len 512 --output-len 128 --concurrency 1")
    args = ap.parse_args()
    bargs = args.bench_args.split()

    deltas, rows = [], []
    for blk in range(args.blocks):
        vals = {}
        for who in ("a", "b", "b", "a"):          # ABBA
            binary = args.a if who == "a" else args.b
            time.sleep(args.cooldown)
            v = run(binary, args.model, bargs, args.metric)
            vals.setdefault(who, []).append(v)
            print(f"  block {blk} {who}: {v:.2f}", flush=True)
        a = statistics.mean(vals["a"])
        b = statistics.mean(vals["b"])
        deltas.append(b - a)
        rows.append({"block": blk, "a": a, "b": b, "delta": b - a})
        print(f"block {blk}: A {a:.2f}  B {b:.2f}  delta {b - a:+.2f}", flush=True)

    med = statistics.median(deltas)
    pos = sum(1 for d in deltas if d > 0)
    base = statistics.mean([r["a"] for r in rows])
    print(json.dumps({"median_delta": med, "median_pct": 100 * med / base,
                      "blocks_b_faster": pos, "blocks": len(deltas),
                      "per_block": rows}, indent=2))
    # Sign test: with n blocks, all-one-way is p = 2^-(n-1) two-sided.
    if pos == len(deltas) or pos == 0:
        print(f"CONSISTENT direction in all {len(deltas)} blocks (sign test p = {2**-(len(deltas)-1):.3f})")
    else:
        print(f"INCONCLUSIVE: B faster in {pos}/{len(deltas)} blocks")

main()
