#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CUDA_ROOT RUNTIME_DIR" >&2
  exit 2
fi

cuda_root=$1
runtime_dir=$2
cuda_stub=$(find -L "$cuda_root" -type f -path '*/stubs/libcuda.so' -print -quit)
if [[ -z "$cuda_stub" ]]; then
  echo "CUDA driver stub is required for archive smoke validation" >&2
  exit 1
fi

cuda_stub=$(readlink -f "$cuda_stub")
mkdir -p "$runtime_dir"
runtime_dir=$(cd "$runtime_dir" && pwd -P)
ln -sfn "$cuda_stub" "$runtime_dir/libcuda.so.1"
printf '%s\n' "$runtime_dir"
