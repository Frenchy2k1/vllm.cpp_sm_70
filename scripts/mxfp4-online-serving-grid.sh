#!/usr/bin/env bash
# One-command MXFP4 W4 throughput grid (row QUANT-CT-MXFP4-BENCH).
#
# Reproducibility wrapper around the fingerprinted online-serving harness for the
# "q3mxfp4" key (Yi30/Qwen3-8B-MXFP4, native Marlin W4A16 MXFP4 keep-quant). It
# adds only the two steps dgx-online-serving.sh does not own for a fresh key --
# the deterministic source-corpus generation and the dry-run manifest -- then
# delegates the locked, single-load, drop_caches grid to that tested driver and
# prints the per-axis summary. All GPU serialization, memory gating, oracle
# recording (VLLM_DISABLED_KERNELS=FlashInferMxFp4LinearKernel), the #44 smoke
# model gate and the c1/c2/c4/c8 x3 sequential legs live in dgx-online-serving.sh
# / online_gate.py; this script is intentionally thin so its own logic is trivial.
#
# The oracle arm needs ninja + nvcc on PATH (the DGX non-interactive quirk); run
# under a shell that has them, both flock locks free, free -g >= 90, and the
# production (profile-control-OFF) CUDA build already configured.
#
# BUILD CONTRACT (record-execution is strict — measured 2026-08-06): the --build-dir
# MUST be on a REAL disk, NOT tmpfs/dev/shm (tmpfs pages stay resident so the harness
# cache-drop POSIX_FADV_DONTNEED+mincore==0 proof fails on the server binary), and
# configured RelWithDebInfo with CMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc,
# CMAKE_MAKE_PROGRAM=<oracle venv>/bin/ninja, CMAKE_EXPORT_COMPILE_COMMANDS=ON,
# VLLM_CPP_BENCH_PROFILE_CONTROL=OFF, VLLM_CPP_BUILD_TESTS=ON, and VLLM_CPP_CUTLASS_DIR
# pointing at the ORACLE's flashinfer-bundled cutlass tree (4.5.0), not $HOME/cutlass-4.5.0.
# Full tree ~3 GiB. See .agents/benchmark-record.md for the exact configure line.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  mxfp4-online-serving-grid.sh --snapshot DIR --build-dir DIR --configure-log FILE
    [--client PATH] [--claim-root DIR] [--vllm-cpp-sha SHA] [--port N]

  --snapshot       HF snapshot dir; its basename MUST be the pinned revision
                   b3e7ab32f7225ca779b3dbf6ef4ecefeb6de9b47.
  --build-dir      configured production CMake build tree (server + vllm-cli).
  --configure-log  the non-empty log from that build configuration.
  --client         pinned vLLM oracle `vllm` (default ~/venvs/vllm-oracle/bin/vllm).
  --claim-root     evidence root (default ~/work/vllm.cpp-online-gate).
  --vllm-cpp-sha   defaults to the worktree HEAD.
  --port           server port (default 8001).
EOF
}

model=q3mxfp4
revision=b3e7ab32f7225ca779b3dbf6ef4ecefeb6de9b47
snapshot=""
build_dir=""
configure_log=""
client="${HOME}/venvs/vllm-oracle/bin/vllm"
claim_root="${HOME}/work/vllm.cpp-online-gate"
vllm_cpp_sha=""
port=8001

while (($#)); do
  case "$1" in
    --snapshot) snapshot=${2:?}; shift 2 ;;
    --build-dir) build_dir=${2:?}; shift 2 ;;
    --configure-log) configure_log=${2:?}; shift 2 ;;
    --client) client=${2:?}; shift 2 ;;
    --claim-root) claim_root=${2:?}; shift 2 ;;
    --vllm-cpp-sha) vllm_cpp_sha=${2:?}; shift 2 ;;
    --port) port=${2:?}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
driver="${repo_root}/scripts/dgx-online-serving.sh"

[[ -n ${snapshot} && -d ${snapshot} ]] || { echo "--snapshot directory is required" >&2; exit 2; }
[[ $(basename "${snapshot}") == "${revision}" ]] || {
  echo "--snapshot basename must be the pinned revision ${revision}" >&2; exit 2; }
[[ -n ${build_dir} && -f ${build_dir}/CMakeCache.txt ]] || {
  echo "--build-dir must name a configured CMake build tree" >&2; exit 2; }
[[ -n ${configure_log} && -s ${configure_log} ]] || {
  echo "--configure-log must name the non-empty configuration log" >&2; exit 2; }
[[ -x ${client} ]] || { echo "pinned vLLM client is not executable: ${client}" >&2; exit 2; }

if [[ -z ${vllm_cpp_sha} ]]; then
  vllm_cpp_sha=$(git -C "${repo_root}" rev-parse HEAD)
fi
evidence="${claim_root}/evidence/${vllm_cpp_sha}"
source_corpus="${evidence}/corpus/${model}"
oracle_python="$(dirname "${client}")/python"
[[ -x ${oracle_python} ]] || { echo "vLLM oracle Python is absent: ${oracle_python}" >&2; exit 2; }

# 1. Dry-run manifest FIRST: `online_gate.py plan` refuses to write into a
#    non-empty evidence root, so it must run before the corpus populates
#    ${evidence}/corpus. (A partially-populated evidence root from an aborted run
#    must be removed before re-running.)
if [[ ! -f ${evidence}/manifest.json ]]; then
  "${driver}" --dry-run \
    --claim-root "${claim_root}" \
    --client "${client}" \
    --vllm-cpp-sha "${vllm_cpp_sha}"
fi

# 2. Deterministic source corpus (exact 1024-token prompts) with the checkpoint's
#    own tokenizer, scoped to the c1/c2/c4/c8 low-concurrency sweep. Idempotent.
if [[ ! -f ${source_corpus}/manifest.json ]]; then
  mkdir -p "${source_corpus}"
  "${oracle_python}" -m tools.bench.make_serve_low_corpus \
    --tokenizer-json "${snapshot}/tokenizer.json" \
    --tokenizer-revision "${revision}" \
    --model-key "${model}" \
    --out "${source_corpus}" \
    --concurrencies 1,2,4,8 \
    --repetitions 3
fi

# 3. The locked, single-load, drop_caches c1/c2/c4/c8 x3 grid (both arms). The
#    driver runs the #44 smoke model gate, records the oracle, and emits the
#    per-axis summary (online_gate_summary --model q3mxfp4) as its final step.
"${driver}" --execute \
  --model "${model}" \
  --snapshot "${snapshot}" \
  --source-corpus "${source_corpus}" \
  --evidence "${evidence}" \
  --build-dir "${build_dir}" \
  --configure-log "${configure_log}" \
  --client "${client}" \
  --vllm-cpp-sha "${vllm_cpp_sha}" \
  --port "${port}"

echo "MXFP4 grid complete. Summary: ${evidence}/summary-${model}/report.md" >&2
