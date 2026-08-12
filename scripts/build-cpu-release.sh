#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 ARTIFACT_ID ARCH CHANNEL BUILD_DIR ABI_VERSION POOR_EMULATOR RICH_RUNNER" >&2
  exit 2
fi

artifact_id=$1
arch=$2
channel=$3
build_dir=$4
abi_version=$5
poor_emulator=$6
rich_runner=$7
: "${SOURCE_SHA:?SOURCE_SHA is required}"
: "${VERSION:?VERSION is required}"
: "${EVIDENCE_URL:?EVIDENCE_URL is required}"
: "${SOURCE_DATE_EPOCH:?SOURCE_DATE_EPOCH is required}"

literal_static=OFF
if [[ "$artifact_id" == linux-x86_64-musl-cpu-static ]]; then
  literal_static=ON
fi

cmake -S . -B "$build_dir" -G Ninja \
  -DVLLM_CPP_BUILD_TESTS=ON \
  -DVLLM_CPP_BUILD_EXAMPLES=ON \
  -DVLLM_CPP_SERVER=ON \
  -DVLLM_CPP_CUDA=OFF \
  -DVLLM_CPP_CUDA_ARCHITECTURES= \
  -DVLLM_CPP_HIP=OFF \
  -DVLLM_CPP_HIP_ARCHITECTURES= \
  -DVLLM_CPP_LITERAL_STATIC="$literal_static" \
  -DVLLM_CPP_METAL=OFF \
  -DVLLM_CPP_MLX=OFF \
  -DMLX_ROOT= \
  -DVLLM_CPP_TRITON=OFF \
  -DVLLM_CPP_VULKAN=OFF \
  -DCMAKE_BUILD_TYPE=Release

targets=(server test_ops_matmul_elem)
if [[ "$arch" == aarch64 ]]; then
  targets+=(test_cpu_isa_arm test_ops_quant_dot test_ops_quant_repack)
fi
cmake --build "$build_dir" --target "${targets[@]}" -j "${JOBS:-2}"

release_dir="$build_dir/release"
tier_report="$release_dir/cpu-tier-report.json"
stage_dir="$release_dir/stage"
metadata_dir="$release_dir/metadata"
archive="$release_dir/vllm.cpp-$VERSION-$artifact_id.tar.gz"
mkdir -p "$release_dir"

rich_runner_kind=qemu
if [[ "$arch" == x86_64 ]]; then
  rich_runner_kind=intel-sde
fi
python3 scripts/run-cpu-release-gates.py \
  --arch "$arch" \
  --tests-dir "$build_dir/tests" \
  --poor-emulator "$poor_emulator" \
  --rich-runner "$rich_runner" \
  --rich-runner-kind "$rich_runner_kind" \
  --rich-cpu max \
  --output "$tier_report" \
  --evidence-url "$EVIDENCE_URL"

python3 scripts/package-server.py \
  --build-dir "$build_dir" \
  --stage-dir "$stage_dir"

compiler=$(c++ --version | head -n 1)
toolchain="$(cmake --version | head -n 1); $(ninja --version)"
c_abi_version=$(sed -n 's/^#define VLLM_ABI_VERSION \([0-9][0-9]*\)$/\1/p' include/vllm.h)
if [[ -z "$c_abi_version" ]]; then
  echo "could not resolve VLLM_ABI_VERSION" >&2
  exit 1
fi

python3 scripts/release_metadata.py \
  --build-dir "$build_dir" \
  --stage-dir "$stage_dir" \
  --output-dir "$metadata_dir" \
  --tier-report "$tier_report" \
  --artifact-id "$artifact_id" \
  --channel "$channel" \
  --version "$VERSION" \
  --c-abi-version "$c_abi_version" \
  --source-commit "$SOURCE_SHA" \
  --source-clean \
  --abi-version "$abi_version" \
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
  --repo-root .
