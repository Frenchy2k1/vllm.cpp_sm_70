# Building vllm.cpp

vllm.cpp uses CMake (>= 3.24) and a C++20 compiler (gcc 13/14 and clang are
exercised; the tree builds -Werror-clean on gcc 14.2). The core has no ML
dependencies; the OpenAI server uses a vendored header-only HTTP transport
(cpp-httplib). The [README](../README.md) carries the two-line quickstart; this
page is the full build reference.

## CPU build (the correctness / CI reference)

```sh
cmake -S . -B build
cmake --build build -j
ctest --test-dir build
```

The server is ON by default. Example binaries land under `build/examples/`:
`vllm-cli`, `server`, `vllm-bench`, and `tokenize`.

## CUDA build (NVIDIA GB10 / DGX Spark)

```sh
cmake -S . -B build-cuda \
  -DVLLM_CPP_CUDA=ON \
  -DVLLM_CPP_TRITON=ON
cmake --build build-cuda -j
```

Triton-AOT cubins for the fast GDN path are **vendored**: Python and Triton are
needed only to regenerate them (`VLLM_CPP_TRITON_REGEN`), never to build or run
them.

For other CUDA families, set the arch explicitly:

```sh
cmake -S . -B build-cuda -DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=90a
```

## Metal build (Apple Silicon)

Metal is detected automatically on an Apple host with an ObjC++ compiler. The
optional MLX GEMM provider is a separate opt-in and needs an MLX install:

```sh
cmake -S . -B build-metal -DVLLM_CPP_MLX=ON -DMLX_ROOT=/path/to/mlx
cmake --build build-metal -j
```

MLX is shape-gated to prefill, where it wins; it declines the `m < 2` decode
GEMV by design. Note that an MLX build produces a **different greedy sequence**
than the default build (MLX's GEMM is not bit-identical), so goldens must not be
re-anchored to an MLX build. Details in [docs/BENCHMARKS.md](BENCHMARKS.md).

## Vulkan build

Headers are vendored and SPIR-V is committed, so no graphics toolchain is
needed. It is off unless requested:

```sh
cmake -S . -B build-vulkan -DVLLM_CPP_VULKAN=ON
cmake --build build-vulkan -j
```

## Nix shells

The checked-in flake pins CMake, Ninja and the CUDA toolchain, so nothing has
to be globally installed:

```sh
# CPU (correctness / CI reference)
nix develop .#default --command cmake -S . -B build-nix-cpu -G Ninja \
  -DVLLM_CPP_CUDA=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
nix develop .#default --command cmake --build build-nix-cpu -j4

# CUDA (set the arch for your GPU)
nix develop .#cuda --command bash -lc \
  'cmake -S . -B build-nix-cuda -G Ninja -DVLLM_CPP_CUDA=ON \
    -DCMAKE_CUDA_COMPILER="$CMAKE_CUDA_COMPILER" \
    -DCMAKE_CUDA_HOST_COMPILER="$CMAKE_CUDA_HOST_COMPILER" \
    -DVLLM_CPP_CUDA_ARCHITECTURES=120a -DCMAKE_BUILD_TYPE=RelWithDebInfo'
nix develop .#cuda --command cmake --build build-nix-cuda -j4
```

On NixOS the CUDA shell exports the driver-library path and
`TRITON_LIBCUDA_PATH=/run/opengl-driver/lib` so Triton finds `libcuda` without
`/sbin/ldconfig`.

## CMake options

Read from [`CMakeLists.txt`](../CMakeLists.txt). Defaults shown are the shipped
defaults.

| Option | Default | Purpose |
|---|---|---|
| `VLLM_CPP_CUDA` | `AUTO` | Build the CUDA backend: `ON`, `OFF`, or `AUTO` (on when a CUDA toolchain is found) |
| `VLLM_CPP_CUDA_ARCHITECTURES` | `121a` | Target CUDA arch(s): `121a` (GB10), `120a`/`120a;121a` (consumer Blackwell), and cross-family targets `90a`, `80`/`86`/`87`/`89`, `100a`/`103a`, `110`. The `a` suffix is required for the native fp4 MMA |
| `VLLM_CPP_METAL` | `AUTO` | Build the Metal backend: `ON`, `OFF`, or `AUTO` (on for an Apple host with an ObjC++ compiler) |
| `VLLM_CPP_VULKAN` | `AUTO` (= `OFF`) | Build the Vulkan backend. Opt-in with `-DVLLM_CPP_VULKAN=ON`; headers are vendored and SPIR-V is committed |
| `VLLM_CPP_MLX` | `OFF` | Build the optional MLX GEMM provider for Metal (needs `-DMLX_ROOT=<mlx install>`) |
| `MLX_ROOT` | (empty) | Root of an MLX install (`include/` + `lib/`) for `VLLM_CPP_MLX` |
| `VLLM_CPP_SERVER` | `ON` | Build the OpenAI HTTP server (needs `third_party/httplib/httplib.h`; disables itself with a warning if absent) |
| `VLLM_CPP_TRITON` | `OFF` | Consume the vendored per-arch Triton-AOT GDN cubins (CUDA only; no Python needed) |
| `VLLM_CPP_TRITON_REGEN` | `OFF` | Maintainer knob: regenerate the AOT cubins with Python + Triton |
| `VLLM_CPP_CUTLASS_DIR` | `third_party/cutlass` | CUTLASS source root (>= 4.5.0) for the sm120a NVFP4 GEMM |
| `VLLM_CPP_CUTLASS_FETCH` | `OFF` | FetchContent CUTLASS 4.5.0 if not found locally |
| `VLLM_CPP_MARLIN` | `ON` | Build the vendored Marlin NVFP4 W4A16 MoE GEMM (sm_12xa) |
| `VLLM_CPP_BUILD_TESTS` | `ON` | Compile and register ctest targets |
| `VLLM_CPP_BUILD_EXAMPLES` | `ON` | Build the example CLI, server, and bench binaries |
| `VLLM_CPP_BENCH_PROFILE_CONTROL` | `OFF` | Trace-only profiler replay control (never for production timing builds) |

## Backend and hardware state

| Backend | Hardware | State |
|---|---|---|
| CPU | x86-64 and arm64 | Correctness / CI reference; at or ahead of llama.cpp on every GGUF axis, with an Arm i8mm quant-GEMM tier |
| CUDA | GB10 / DGX Spark, sm_121a | Gate-model correctness passes; 27B at/above vLLM throughput, 35B prefill-pending. The only runtime-gated CUDA target |
| CUDA | Consumer Blackwell, sm_120a | Build-supported (compiles, emits real sm_120a code, all fast paths resolve) but not runtime-proven here (no such card) |
| CUDA | Hopper, sm_90a | Build-supported; the fast GDN (Triton-AOT) path is build-verified, not runtime-proven here |
| CUDA | Ampere/Ada (sm_80/86/87/89), datacenter Blackwell (sm_100a/103a), sm_110 | Build-supported; the fast GDN path is build-verified per-arch on sm_80/86/89/100a (plus FA2 on Ampere, sm_100a NVFP4 GEMM), not runtime-gated here. sm_70/sm_75 unsupported (no bf16 tensor cores) |
| Metal | Apple Silicon | Two models run end to end and pass correctness; 18 of 75 ops native. Warm b=1 throughput is 95.9% of MLX-LM, or 97.6% with the optional MLX provider gated to prefill (where we are 1.5% ahead). Indicative |
| Vulkan | Portable GPU | Skeleton: 8 ops plus the fusion catalogue run and cross-check against CPU and CUDA. No model runs yet |
| Intel XPU | Intel GPUs | Spiked, hardware-blocked |
| ROCm / ANE | AMD GPUs / Apple Neural Engine | Post-parity roadmap |

Only GB10 / sm_121a is a runtime-gated CUDA target today. Consumer Blackwell
(`120a`) plus the cross-family targets are build-supported (they compile and emit
real machine code, with the fast GDN path build-verified on several) but unproven
at runtime here (no such board), and non-Apple / non-NVIDIA backends run a subset
of operations. Per-op detail is in the
[backend matrix](../.agents/backend-matrix.md).

## Quantization formats

| Format | State |
|---|---|
| NVFP4 W4A4 / W4A16 | Both gate-model paths run on GB10, token-exact. FP4 tactics match vLLM; Marlin NVFP4 W4A16 grouped-MoE is the 35B expert path |
| compressed-tensors NVFP4A16 (W4A16), dense | Correctness-complete via the Marlin weight-only path; speed not yet measured |
| GGUF F32 / F16 / Q4_0 / Q8_0 / Q3_K / Q4_K / Q5_K / Q6_K | Supported. On CPU the six block encodings compute directly on the compressed blocks (`VT_GGUF_KEEP_QUANT=0` disables it). GPU builds still expand GGUF weights |
| FP8 (W8A8) | The 35B ModelOpt static per-tensor projection slice is implemented; generic FP8 modes and FP8 KV remain open |
| MXFP4 / MXFP8 | Planned |

## Environment variables

Runtime knobs (op providers, keep-quant, profiling) are documented in
[docs/ENVIRONMENT.md](ENVIRONMENT.md).
