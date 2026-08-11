#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 ARTIFACT_ID BACKEND BUILD_DIR" >&2
  exit 2
fi

artifact_id=$1
backend=$2
build_dir=$3
: "${SOURCE_SHA:?SOURCE_SHA is required}"
: "${VERSION:?VERSION is required}"
: "${EVIDENCE_URL:?EVIDENCE_URL is required}"
: "${SOURCE_DATE_EPOCH:?SOURCE_DATE_EPOCH is required}"

cuda=OFF
triton=OFF
vulkan=OFF
cuda_architectures=
if [[ "$backend" == cuda ]]; then
  cuda=ON
  triton=ON
  cuda_architectures='80;86;87;89;90a;100a;103a;110;120a;121a'
elif [[ "$backend" == vulkan ]]; then
  vulkan=ON
else
  echo "unsupported Linux accelerator backend: $backend" >&2
  exit 2
fi

cmake -S . -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DVLLM_CPP_BUILD_TESTS=ON \
  -DVLLM_CPP_BUILD_EXAMPLES=ON \
  -DVLLM_CPP_SERVER=ON \
  -DVLLM_CPP_CUDA="$cuda" \
  -DVLLM_CPP_CUDA_ARCHITECTURES="$cuda_architectures" \
  -DVLLM_CPP_CUTLASS_FETCH=ON \
  -DVLLM_CPP_HIP=OFF \
  -DVLLM_CPP_HIP_ARCHITECTURES= \
  -DVLLM_CPP_LITERAL_STATIC=OFF \
  -DVLLM_CPP_METAL=OFF \
  -DVLLM_CPP_MLX=OFF \
  -DMLX_ROOT= \
  -DVLLM_CPP_TRITON="$triton" \
  -DVLLM_CPP_VULKAN="$vulkan"

targets=(server)
if [[ "$backend" == vulkan ]]; then
  targets+=(test_vulkan_backend test_backend_cross_device)
fi
cmake --build "$build_dir" --target "${targets[@]}" -j "${JOBS:-2}"

if [[ "$backend" == cuda ]]; then
  python3 scripts/check-cuda-fat-gencode.py \
    --compile-commands "$build_dir/compile_commands.json" \
    --library "$build_dir/libvllm.a"
  python3 scripts/check-triton-aot-multiarch.py \
    --vendored-root src/vt/cuda/triton_aot_vendored \
    --library "$build_dir/libvllm.a"
else
  "$build_dir/tests/test_vulkan_backend"
  "$build_dir/tests/test_backend_cross_device"
fi

release_dir="$build_dir/release"
stage_dir="$release_dir/stage"
metadata_dir="$release_dir/metadata"
archive="$release_dir/vllm.cpp-$VERSION-$artifact_id.tar.gz"
mkdir -p "$release_dir"
python3 scripts/package-server.py --build-dir "$build_dir" --stage-dir "$stage_dir"

compiler=$(c++ --version | head -n 1)
toolchain="$(cmake --version | head -n 1); $(ninja --version)"
c_abi_version=$(sed -n 's/^#define VLLM_ABI_VERSION \([0-9][0-9]*\)$/\1/p' include/vllm.h)
abi_version=$(getconf GNU_LIBC_VERSION | awk '{print $2}')
python3 scripts/release_accelerator_metadata.py \
  --build-dir "$build_dir" \
  --stage-dir "$stage_dir" \
  --output-dir "$metadata_dir" \
  --artifact-id "$artifact_id" \
  --channel preview \
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
  --archive "$archive"
if [[ "$backend" == cuda ]]; then
  cuda_stub_validation_dir=$(mktemp -d)
  cleanup_cuda_stub_validation_dir() {
    rm -rf -- "$cuda_stub_validation_dir"
  }
  trap cleanup_cuda_stub_validation_dir EXIT
  cuda_stub_runtime_dir=$(scripts/prepare-cuda-driver-stub.sh /usr/local/cuda "$cuda_stub_validation_dir")
  export LD_LIBRARY_PATH="$cuda_stub_runtime_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
python3 scripts/validate-release-archive.py \
  --archive "$archive" \
  --checksum "$archive.sha256" \
  --provenance "$archive.provenance.json" \
  --repo-root . \
  --forbid-path "$PWD/$build_dir"
