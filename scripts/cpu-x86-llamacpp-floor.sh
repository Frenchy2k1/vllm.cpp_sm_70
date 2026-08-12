#!/usr/bin/env bash
# Binding x86_64 A/B series for BACKEND-GATE-CPU-LLAMACPP (#433).
# Evidence: docs/bench-evidence/cpu-x86-llamacpp-20260811.md
# Run from the repo root of a CPU-only Release build (build-cpu/).
#
# Strictly interleaved -- ours rep N, then llama rep N -- so neither engine can
# own a quiet window the other did not also get.
#
# This box is shared and another session's parallel build drove the load average
# from 3.8 to 82 mid-series once already, so every leg is gated: it will not
# start until the one-minute load average is below QUIET_LOAD with no compiler
# processes running, and if a foreign compiler appears while the leg is running
# the leg is DISCARDED and retried rather than averaged in.
set -u
M=/home/mudler/.config/dante-desktop/models/Qwen3.5-2B-UD-Q8_K_XL.gguf
LB=/home/mudler/llamacpp-build-cpu-bench/bin/llama-bench
OB=./build-cpu/examples/vllm-bench
OUT=evi
T=20
REPS=${REPS:-3}
QUIET_LOAD=${QUIET_LOAD:-3}
# Exact process-NAME match (-x, no -f). Matching full command lines instead
# matches this script's own gate and the pgrep call itself, which deadlocks the
# wait on its own reflection -- that cost a whole series.
BUILDERS='cc1plus|cc1|nvcc|ninja|collect2|as|ld|gcc|g\+\+|cmake'

load1() { cut -d' ' -f1 /proc/loadavg; }
loadall() { cut -d' ' -f1-3 /proc/loadavg; }
# pgrep -c already prints 0 on no match and exits 1, so a `|| echo 0` fallback
# emits "0\n0" and every numeric test that consumes it fails.
builders() { local n; n=$(pgrep -c -x "$BUILDERS" 2>/dev/null); echo "${n:-0}"; }

wait_quiet() {
  local waited=0
  while :; do
    local l b
    l=$(load1); b=$(builders)
    if [ "${l%%.*}" -lt "$QUIET_LOAD" ] && [ "$b" -eq 0 ]; then return 0; fi
    sleep 20; waited=$((waited + 20))
    if [ $((waited % 300)) -eq 0 ]; then
      echo "waiting for quiet: ${waited}s load=$l builders=$b"
    fi
  done
}

run_leg() {  # engine rep -> 0 accepted, 1 discard
  local eng=$1 rep=$2 rc
  wait_quiet
  echo "$eng rep=$rep START load=$(loadall) builders=$(builders)"
  if [ "$eng" = ours ]; then
    /usr/bin/time -v taskset -c 0-19 env VLLM_CPP_CPU_THREADS=$T \
      "$OB" --model "$M" --num-prompts 1 --input-len 128 --output-len 32 \
      --concurrency 1 --seed 0 --temperature 0 \
      > "$OUT/ours-bench-$rep.txt" 2> "$OUT/ours-bench-$rep.time"
    rc=$?
  else
    /usr/bin/time -v taskset -c 0-19 "$LB" -m "$M" -p 128 -n 32 -pg 128,32 \
      -t $T -ngl 0 -r 3 -o json \
      > "$OUT/llama-bench-$rep.json" 2> "$OUT/llama-bench-$rep.time"
    rc=$?
  fi
  local bafter; bafter=$(builders)
  echo "$eng rep=$rep END exit=$rc load=$(loadall) builders=$bafter"
  if [ "$rc" -ne 0 ] || [ "$bafter" -ne 0 ]; then
    echo "$eng rep=$rep DISCARDED (exit=$rc builders_after=$bafter)"
    return 1
  fi
  return 0
}

R=1
attempt=0
while [ "$R" -le "$REPS" ]; do
  attempt=$((attempt + 1))
  if [ "$attempt" -gt 24 ]; then echo "GIVING_UP too many discards"; exit 2; fi
  if run_leg ours "$R" && run_leg llama "$R"; then
    echo "pair rep=$R ACCEPTED"
    R=$((R + 1))
  else
    echo "pair rep=$R RETRY"
    rm -f "$OUT/ours-bench-$R.txt" "$OUT/ours-bench-$R.time" \
          "$OUT/llama-bench-$R.json" "$OUT/llama-bench-$R.time"
  fi
done
echo "SERIES_DONE"
