#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 6 ]]; then
  echo "usage: $0 ARTIFACT_ID CHANNEL BUILD_DIR [MLX_ROOT MLX_VERSION MLX_LICENSE]" >&2
  exit 2
fi

artifact_id=$1
channel=$2
build_dir=$3
mlx_root=${4:-}
mlx_version=${5:-}
mlx_license=${6:-}
: "${SOURCE_SHA:?SOURCE_SHA is required}"
: "${VERSION:?VERSION is required}"
: "${EVIDENCE_URL:?EVIDENCE_URL is required}"
: "${SOURCE_DATE_EPOCH:?SOURCE_DATE_EPOCH is required}"

mlx=OFF
if [[ "$artifact_id" == macos-arm64-metal-mlx ]]; then
  mlx=ON
  if [[ -z "$mlx_root" || -z "$mlx_version" || -z "$mlx_license" ]]; then
    echo "MLX artifact requires an exact root, version, and license" >&2
    exit 2
  fi
fi

cmake -S . -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_BUILD_TESTS=ON \
  -DVLLM_CPP_BUILD_VERSION="$VERSION" \
  -DVLLM_CPP_BUILD_EXAMPLES=ON \
  -DVLLM_CPP_SERVER=ON \
  -DVLLM_CPP_CUDA=OFF \
  -DVLLM_CPP_CUDA_ARCHITECTURES= \
  -DVLLM_CPP_HIP=OFF \
  -DVLLM_CPP_HIP_ARCHITECTURES= \
  -DVLLM_CPP_LITERAL_STATIC=OFF \
  -DVLLM_CPP_METAL=ON \
  -DVLLM_CPP_MLX="$mlx" \
  -DMLX_ROOT="$mlx_root" \
  -DVLLM_CPP_TRITON=OFF \
  -DVLLM_CPP_VULKAN=OFF
cmake --build "$build_dir" --target server test_metal_backend -j 2
"$build_dir/tests/test_metal_backend"

release_dir="$build_dir/release"
stage_dir="$release_dir/stage"
metadata_dir="$release_dir/metadata"
archive="$release_dir/vllm.cpp-$VERSION-$artifact_id.tar.gz"
mkdir -p "$release_dir"
python3 scripts/package-server.py --build-dir "$build_dir" --stage-dir "$stage_dir"

compiler=$(c++ --version | head -n 1)
toolchain="$(cmake --version | head -n 1); $(ninja --version)"
c_abi_version=$(sed -n 's/^#define VLLM_ABI_VERSION \([0-9][0-9]*\)$/\1/p' include/vllm.h)
python3 scripts/release_macos_metadata.py \
  --build-dir "$build_dir" \
  --stage-dir "$stage_dir" \
  --output-dir "$metadata_dir" \
  --artifact-id "$artifact_id" \
  --channel "$channel" \
  --version "$VERSION" \
  --c-abi-version "$c_abi_version" \
  --source-commit "$SOURCE_SHA" \
  --source-clean \
  --abi-version "$(sw_vers -productVersion)" \
  --mlx-version "$mlx_version" \
  --mlx-license "$mlx_license" \
  --compiler "$compiler" \
  --toolchain "$toolchain" \
  --evidence-url "$EVIDENCE_URL"
python3 scripts/package-server.py \
  --build-dir "$build_dir" \
  --stage-dir "$stage_dir" \
  --metadata-dir "$metadata_dir" \
  --archive "$archive" \
  --archive-format tar.gz
python3 scripts/validate-release-archive.py \
  --archive "$archive" \
  --archive-format tar.gz \
  --checksum "$archive.sha256" \
  --provenance "$archive.provenance.json" \
  --repo-root . \
  --forbid-path "$PWD/$build_dir"
