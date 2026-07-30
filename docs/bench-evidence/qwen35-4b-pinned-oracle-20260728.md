# Qwen3.5-4B against an oracle AT the parity pin, 2026-07-28

Immutable evidence index for the first local comparison whose denominator is the
project's actual parity pin rather than a pip-installed PyPI release.

## Headline

**The published 0.9819x was PESSIMISTIC, not flattering.** vLLM at the pin is
1.25% SLOWER than 0.24.0 on this workload, so measuring against the older release
was understating us. Against the true pin we are **0.9970x** on total throughput
— a 0.3% gap, not 1.8%.

TPOT remains the failing axis and the real gap: **+12.41%**.

## Why the oracle was wrong before

The local oracle was pip-installed on 2026-07-09 and was vLLM **0.24.0**. The
parity pin is `555967922`, a vLLM **main** commit with no release tag and no
prebuilt wheel on ANY platform (x86_64 checked here, aarch64 checked by the pin
spec). `pip install vllm` reaches only PyPI releases, so the local oracle could
never have matched the pin by installation — it has to be built from source. The
records additionally mislabelled the venv as "0.25.0", which hid the drift.

## Oracle provenance

- Source: `/home/rich/c/vllm-upstream` at
  `5559679229bc961848b121ccdeaa8fa5d79bec98` (2026-07-26), the pin.
- Built: `vllm 0.23.1rc1.dev1511+g555967922.cu132`, editable, sm_120 only
  (`TORCH_CUDA_ARCH_LIST=12.0`), venv `.venv-vllm-pin`. The 0.24.0 venv is
  untouched and still present.
- Stack, all as the pin specifies: torch 2.13.0+cu130, torchvision 0.28.0,
  triton 3.7.1, transformers 5.14.1, flashinfer 0.6.15.post1 (+cubin),
  nvidia-cutlass-dsl 4.6.0, tilelang 0.1.9, quack-kernels 0.6.1,
  humming-kernels 0.1.10, tokenspeed-mla 0.1.8.
- Attention backend selected at runtime: **FLASH_ATTN**, out of
  `['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION']`.
- Binary SHA-256 (ours)
  `31a22725f99926e3ad533baefca2eba225c2cd569069b86979bdecde95b366fe`;
  dataset `9ea13603767c62c267e3f381fbccf42d0c9ca0c393655c37533eadca7aefca0c`.
- All 9 performance legs recorded **0% GPU utilization** before starting; one
  `flock /tmp/gpu` across the whole 18-leg series.

### The CUDA toolkit ceiling (recorded — it is not obvious and it cost two builds)

The usable CUDA version is set by **what the DRIVER can JIT**, not by what is
newest:

- vLLM ships FlashAttention-2 as `8.0+PTX`
  (`cuda_archs_loose_intersection(FA2_ARCHS "8.0+PTX" ...)`), so the driver
  JIT-compiles its PTX for Blackwell at load time. Driver 595.71.05 tops out at
  CUDA 13.2 and rejects nvcc-13.3 PTX with `cudaErrorUnsupportedPtxVersion` —
  which surfaces only at RUNTIME, after a completely clean build.
- CUDA **13.0** is separately unusable: its headers predate glibc 2.42's `rsqrt`
  declaration and collide with it (`exception specification is incompatible`).
  The fix (`_NV_RSQRT_SPECIFIER`) is present from 13.1 on.
- So **13.2** is the only version clearing both, and the whole toolkit must match
  — cccl hard-errors on a mixed compiler/header pair
  (`"CUDA compiler and CUDA toolkit headers are incompatible"`).
- Install vLLM with `--no-deps`, or pip re-resolves the CUDA runtime DOWN to
  13.0 after the build and reintroduces the mismatch.
- pip's CUDA wheels ship no unversioned `.so`, so `find_library` fails until dev
  symlinks are added (`CUDA_nvrtc_LIBRARY ... NOTFOUND`).

## Binding result

Three performance repetitions per arm, interleaved ON / vLLM / OFF.

| Axis | Direct ON | Direct OFF | vLLM @ pin | ON vs pin | Disposition |
|---|---:|---:|---:|---:|---|
| Total throughput (tok/s) | 6618.160 | 6504.9 | 6638.129 | 0.9970x | FAIL |
| Output throughput (tok/s) | 731.817 | — | 734.026 | 0.9970x | FAIL |
| Requests/s | 5.717 | — | 5.735 | 0.9969x | FAIL |
| Mean TTFT (ms) | 729.217 | — | 943.198 | 0.7731x | PASS, 22.7% lower |
| Mean TPOT (ms) | 38.107 | — | 33.900 | 1.1241x | FAIL, 12.4% higher |
| Peak PSS (GiB) | 2.488 | 8.595 | 8.149 | | PASS |
| Stable PSS (GiB) | 0.759 | 8.591 | 4.565 | | PASS |
| Peak VRAM (MiB) | 12850.7 | 12846.0 | 12826.7 | | ours +24 MiB |

Repetition spread is tight enough that these ratios are not noise: pin
6640.3 / 6633.4 / 6640.7 (0.11%), ours 6615.3 / 6618.7 / 6620.5 (0.08%).

## Oracle version delta, measured

| | vLLM 0.24.0 (2026-07-27) | vLLM @ pin (2026-07-28) |
|---|---:|---:|
| Total throughput (tok/s) | 6722.241 | 6638.129 |
| Mean TTFT (ms) | 913.552 | 943.198 |
| Mean TPOT (ms) | 33.526 | 33.900 |

The pin is **0.9875x** the 0.24.0 release on total throughput, and slightly worse
on both latency axes. Our own arm is unchanged between the two series
(1.0027x, i.e. reproduces itself), which is the control that makes the oracle
delta attributable to the oracle.

Consequently our published ratio moves **0.9819x -> 0.9970x** with no change to
our code.

## Token identity

| Comparison | Result |
|---|---|
| Ours ON vs OFF, per rep | 128/128, 128/128, 128/128 |
| Ours ON vs the 2026-07-27 series | 128/128, 128/128, 128/128 (our arm reproduces exactly) |
| vLLM @ pin vs vLLM 0.24.0 | 92/128, 92/128, 95/128 |
| Ours vs vLLM @ pin, per rep | 89/128, 89/128, 98/128 (the established near-tie) |

The oracle's own output moved between versions on ~28% of requests, which is
expected across two vLLM versions and consistent with the oracle not being
self-deterministic on this workload. Correctness remains grounded in the model
gate and the near-tie contract, not in forced benchmark token identity.

## Known non-fidelity, recorded rather than hidden

The oracle logs `Failed to import Triton kernels ... No module named
'triton_kernels.matmul_ogs'`. That package is not in vLLM's own
`requirements/cuda.txt`, so a stock install lacks it too — the oracle is faithful
to a stock vLLM, and the affected MoE paths are not on this dense 4B workload.
(This is NOT the repo's own `triton_kernels/` directory shadowing it: the harness
invokes the metrics script by path, so `sys.path[0]` is `tools/bench` and the
repo root is never on `sys.path`. Verified directly.)

## Reproduction

```sh
nix develop .#cuda --command bash -c 'REQUIRE_TRITON_AOT=1 \
  CPP_BENCH="$PWD/build-nix-cuda-transplant-triton/examples/vllm-bench" \
  CMAKE_CACHE="$PWD/build-nix-cuda-transplant-triton/CMakeCache.txt" \
  VLLM_PYTHON="$PWD/.venv-vllm-pin/bin/python" \
  VLLM_CUDA_HOME="$PWD/.venv-vllm-pin/lib/python3.12/site-packages/nvidia/cu13" \
  flock /tmp/gpu tools/bench/run_qwen35_4b_compare.sh /tmp/qwen35-pinned-oracle-<commit>'
```

Root `/tmp/qwen35-pinned-oracle-13729626`, aggregate `aggregate.json`.
No 4B result implies support or speed for the 27B/35B gate checkpoints.
