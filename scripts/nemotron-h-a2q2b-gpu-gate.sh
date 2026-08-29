#!/usr/bin/env bash
# A2-Q2b (#810) dgx:gpu0 gate runner — the device `lm_head` on NVFP4 W4A16 g16.
#
# .agents/specs/nemotron-h-a2q2b-realckpt-lmhead.md.
#
# ── WHY THIS FILE EXISTS RATHER THAN A COMMAND LINE ────────────────────────
# The spec's §4 records that this row has lost FOUR GB10 windows to environment
# rather than to code — `121` instead of `121a`, an unconstrained build, a
# CUTLASS fetch with no egress, and an 8h19m outage. Every one of those is a
# precondition that can be checked before the expensive part starts, so they are
# checked here, once, and a violation VOIDS the run loudly instead of producing
# a number nobody can use.
#
# Env:
#   SRC        checkout to build (required)
#   BUILD      build directory (required; put it on /tmp, NOT /workspace — CIFS
#              holds no symlink and the link step fails there)
#   ARCH       CUDA arch, must be 121a on GB10 (default 121a)
#   LOG_ROOT   where to write the run log (required)
#   CKPT       Nemotron-3.5-Lightning NVFP4 snapshot (default from CHECKPOINT_ROOT)
set -uo pipefail

SRC="${SRC:?set SRC}"
BUILD="${BUILD:?set BUILD}"
ARCH="${ARCH:-121a}"
LOG_ROOT="${LOG_ROOT:?set LOG_ROOT}"
CKPT="${CKPT:-${CHECKPOINT_ROOT:-/usr/local/nas_share/checkpoints}/nemotron-3.5-lightning-30b-nvfp4}"
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/a2q2b-$(date -u +%Y%m%dT%H%M%SZ).log"
rc=0

say() { echo "=== $* ===" | tee -a "$LOG"; }

# `run` reports the COMMAND's exit status, never the pipeline's. `cmd | tail`
# reports tail's status, which is 0 essentially always — a whole gate series has
# read green that way in this tree.
run() {
  local name="$1"; shift
  say "$name"
  "$@" >>"$LOG" 2>&1
  local status=$?
  echo "### $name -> exit $status" | tee -a "$LOG"
  [ "$status" -ne 0 ] && rc=1
  return 0
}

say "A2Q2B GATE start $(date -u +%FT%TZ) host=$(hostname) src=$SRC arch=$ARCH"
(cd "$SRC" && git rev-parse HEAD && git status --short) | tee -a "$LOG"

# ── PRECONDITION 1: host headroom, sampled INSIDE the run, not before it ─────
free -g | tee -a "$LOG"
AVAIL=$(free -g | awk '/^Mem:/{print $7}')
if [ "${AVAIL:-0}" -lt 60 ]; then
  say "VOID: only ${AVAIL}G available, floor is 60G (spec §4.2). The box reboots \
rather than OOM-killing at gpu_memory_utilization; this is not a failure of the code."
  exit 3
fi

# ── the toolchain: the worker provisions ITSELF ──────────────────────────────
# No CUDA toolkit is preinstalled on an rc worker (AGENTS.md), and the first
# submission of this gate VOIDed in 60 seconds on exactly that. The recipe below
# is the one row A2-D1 proved on this box, reused verbatim rather than
# re-derived: `sbsa` (not x86) for GB10, and the 13-0 package set.
say "toolchain"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >>"$LOG" 2>&1
apt-get install -y -qq git cmake ninja-build g++ curl ca-certificates python3 python3-dev >>"$LOG" 2>&1
if ! command -v nvcc >/dev/null 2>&1 && [ ! -x /usr/local/cuda/bin/nvcc ]; then
  curl -fsSL -o /tmp/cuda-keyring.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb >>"$LOG" 2>&1
  dpkg -i /tmp/cuda-keyring.deb >>"$LOG" 2>&1
  apt-get update -qq >>"$LOG" 2>&1
  apt-get install -y -qq cuda-nvcc-13-0 cuda-cudart-dev-13-0 libcublas-dev-13-0 \
     cuda-nvrtc-dev-13-0 cuda-nvtx-13-0 cuda-profiler-api-13-0 libcurand-dev-13-0 >>"$LOG" 2>&1
fi
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version >>"$LOG" 2>&1 || { say "VOID: no nvcc after the install step"; exit 3; }

# ── PRECONDITION 2: the toolkit is proved by a LINK, not by --version ────────
# A2-D1's check, kept because `nvcc --version` succeeds on an install that
# cannot link, and this row has already lost windows to environment.
cat > /tmp/probe_a2q2b.cu <<'PROBE'
#include <cublasLt.h>
#include <cstdio>
int main() {
  cublasLtHandle_t h = nullptr;
  const auto s = cublasLtCreate(&h);
  std::printf("cublasLtCreate=%d\n", static_cast<int>(s));
  return s == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}
PROBE
if ! nvcc -arch="sm_$ARCH" /tmp/probe_a2q2b.cu -o /tmp/probe_a2q2b -lcublasLt >>"$LOG" 2>&1; then
  say "VOID: the CUDA toolkit does not link cublasLt at sm_$ARCH"
  exit 3
fi
/tmp/probe_a2q2b >>"$LOG" 2>&1

# ── CUTLASS: staged, never fetched. The HOST has no egress to github.com, and
# a CUTLASS fetch with no egress is one of the four windows this row has lost.
CUTLASS="${CUTLASS:-/root/cutlass-v4.5.0}"
if [ ! -f "$CUTLASS/include/cutlass/cutlass.h" ]; then
  if [ -f /workspace/cutlass-v4.5.0.tar.gz ]; then
    mkdir -p "$CUTLASS" && tar xzf /workspace/cutlass-v4.5.0.tar.gz -C "$CUTLASS" --strip-components=1 >>"$LOG" 2>&1
  else
    git clone --depth 1 --branch v4.5.0 https://github.com/NVIDIA/cutlass "$CUTLASS" >>"$LOG" 2>&1
  fi
fi
[ -f "$CUTLASS/include/cutlass/cutlass.h" ] || { say "VOID: no CUTLASS headers at $CUTLASS"; exit 3; }

# ── the build ────────────────────────────────────────────────────────────────
# -j 4: unconstrained parallelism has OOM-REBOOTED this box (AGENTS.md).
say "configure"
CFGLOG="$LOG_ROOT/configure-$(date -u +%H%M%SZ).log"
cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_CUDA=ON \
  -DVLLM_CPP_CUDA_ARCHITECTURES="$ARCH" \
  -DVLLM_CPP_CUTLASS_DIR="$CUTLASS" > "$CFGLOG" 2>&1
CFG=$?
cat "$CFGLOG" >>"$LOG"
echo "### configure -> exit $CFG" | tee -a "$LOG"
[ "$CFG" -ne 0 ] && { say "VOID: configure failed"; exit 3; }

# ── PRECONDITION 3: the arch the configure log actually RESOLVED ─────────────
# A `[121]` instead of `[121a]`, or a DISABLED marlin cell, VOIDS the run rather
# than failing it: the result would describe a different kernel set than the one
# this row is about. Both are spec §4.5 conditions.
grep -E "CUDA feature .*(ENABLED|DISABLED)" "$CFGLOG" | tee -a "$LOG"
if ! grep -qE "marlin-nvfp4: ENABLED for \[[^]]*$ARCH" "$CFGLOG"; then
  say "VOID: marlin-nvfp4 is not ENABLED for [$ARCH]; there is no NVFP4 GEMM to gate"
  grep -iE 'marlin' "$CFGLOG" | tee -a "$LOG"
  exit 3
fi

say "build (-j 4)"
cmake --build "$BUILD" -j 4 --target test_nemotron_h_moe_device nemotron-h-gen >>"$LOG" 2>&1
B=$?
echo "### build -> exit $B" | tee -a "$LOG"
[ "$B" -ne 0 ] && { say "BUILD FAILED — this IS a code result, not a void"; exit 1; }

# ── LEG 1: the synthetic device gate ─────────────────────────────────────────
# `-tc` filters are NOT used: doctest splits them on commas, and a comma in a
# case name yields `0 cases ran` + `SUCCESS!`. The whole TU runs, and the case
# count is asserted below.
run "LEG1 synthetic device MoE + lm_head" "$BUILD/tests/test_nemotron_h_moe_device" -s

# A run that executed NO cases prints SUCCESS and exits 0. Assert a non-zero
# case count from the summary line rather than trusting the exit status.
if ! grep -qE 'test cases:[[:space:]]+[1-9]' "$LOG"; then
  say "VOID: the device TU reported no executed test cases"
  rc=1
fi
grep -E 'test cases:|assertions:|Status:' "$LOG" | tail -6 | tee -a "$LOG"

# ── LEG 2: the REAL checkpoint, through the PRODUCTION ABI ───────────────────
# This is the leg that enters through the production entry point:
# `vllm_engine_load` -> the paged runner -> ForwardNemotronHForCausalLM ->
# NemotronHPagedForward -> the device lm_head branch. A unit test that calls
# NemotronHDeviceLmHead directly proves the projection; only this proves the
# capability is REACHED.
if [ -d "$CKPT" ]; then
  run "LEG2 real-checkpoint token identity (device lm_head)" \
    "$BUILD/examples/nemotron-h-gen" \
      --model "$CKPT" \
      --golden "$SRC/tests/parity/goldens/nemotron_35_lightning_greedy/oracle.json" \
      --steps 8 --prompts 3 --max-model-len 2048
else
  say "SKIP LEG2: no checkpoint at $CKPT — this is a SKIP with a named reason, never a pass"
  rc=1
fi

say "A2Q2B GATE overall exit: $rc  log: $LOG"
exit "$rc"
