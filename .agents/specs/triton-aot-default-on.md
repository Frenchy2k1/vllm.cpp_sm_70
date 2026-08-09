# BUILD-TRITON-DEFAULT-ON — the vendored Triton AOT kernels ship by default

Issue: [#219](https://github.com/mudler/vllm.cpp/issues/219)
Row: `BUILD-TRITON-DEFAULT-ON`
Base: `origin/main` @`04069bd7`

## Scope

Make `VLLM_CPP_TRITON` default to `ON` where it is buildable, and prove which GDN
path actually executes on the two NVFP4 gate models.

**In scope:** the option's default in `cmake/TritonAOT.cmake`, the decline
diagnostics, the docs that describe the default, the release-metadata
expectations, and a runtime-dispatch evidence capture.

**Out of scope:** regenerating any vendored artifact, adding a new arch tree,
changing `VLLM_CPP_TRITON_REGEN`, and every kernel body.

## Why

`cmake/TritonAOT.cmake:57-58` defaults `OFF` while its own help string says
"CUDA-only; no Python needed". The vendored artifacts are pre-generated cubins
embedded in plain C and need only a C compiler, so the opt-in default protects
against no dependency. It does actively cause mismeasurement: a build that omits
the flag runs the hand WMMA path, which is how one NVFP4 sweep was recorded
against kernels the shipped CUDA release never executes
(`tests/scripts/test_release_accelerator_metadata.py:57` already asserts
`"ON" if cuda else "OFF"` for releases).

## Design

Replace the hardcoded `OFF` with a computed default, evaluated after
`VLLM_CPP_CUDA` and `VLLM_CPP_CUDA_ARCHITECTURES` are known:

`ON` iff **all** of:
1. CUDA is enabled;
2. the build is single-arch (`list(LENGTH VLLM_CPP_CUDA_ARCHITECTURES) == 1`),
   or `VLLM_CPP_TRITON_VENDORED_ARCH` pins one tree explicitly;
3. a vendored tree exists for the resolved arch.

Otherwise `OFF`, with one `message(STATUS ...)` naming the condition that
declined it. Use the `option(... ${_computed})` form so a user's
`-DVLLM_CPP_TRITON=...` on the command line still wins.

**The explicit path keeps failing loudly.** If the user passes
`-DVLLM_CPP_TRITON=ON` and the build is multi-arch or the tree is missing, the
existing `FATAL_ERROR` at `TritonAOT.cmake:116-127` must still fire, with its
current wording. Asking for something impossible is an error; only the *default*
degrades quietly. This distinction is the crux of the change — a default that
FATAL_ERRORs would break the fat build and every non-CUDA backend.

Must keep resolving `OFF`: CPU, Metal/MSL, Vulkan, ROCm, and `120a;121a`.

## Risks

- **Breaking the fat build.** `_triton_aot_arch_name` raises `FATAL_ERROR` on
  multi-arch. If the computed default reaches that function, `120a;121a` stops
  configuring. The arch-count test must happen *before* the default is applied.
- **Non-CUDA leakage.** `include(cmake/TritonAOT.cmake)` at `CMakeLists.txt:556`
  is unconditional; only the wiring at `:1483` is CUDA-gated. The default must
  therefore consult `VLLM_CPP_CUDA` itself.
- **Release metadata.** `test_release_metadata.py:69`,
  `test_release_accelerator_metadata.py:57,107`, `test_release_macos_metadata.py:63`
  and `tests/scripts/fixtures/release_manifest/v1/cpu-input.json` encode the
  matrix. Re-derive them from the new rule; **never** edit an expectation merely
  to make a red gate green.

## Tests

RED first:

1. CMake configure matrix test asserting the resolved default: `121a` CUDA → ON;
   `120a;121a` → OFF **and configure succeeds**; `VLLM_CPP_CUDA=OFF` → OFF; a CUDA
   arch with no vendored tree → OFF. Red today (default is unconditionally OFF, so
   the first case fails).
2. Assert an explicit `-DVLLM_CPP_TRITON=ON` on `120a;121a` still fails with the
   existing message. Guards against "fixing" the default by weakening the error.
3. Extend `tests/tools/test_gdn_packed_component.py`-style coverage so the
   default-built CUDA target carries `VLLM_CPP_TRITON=1` and
   `VLLM_CPP_TRITON_CHUNKO_BF16=1`.

## Dispatch evidence (the part that answers "are they used")

Building is not using. `TryTritonChunkO` (`src/vt/cuda/cuda_gdn.cu:4903-4917`)
has six silent bail-outs: device availability, `VT_GDN_CHUNKO_TRITON`,
`dk/dv/hk_n`, `hv_n not in {48,32}`, null `chunk_indices`, and a scale not
exactly `Dk^-0.5`. None warn.

Capture, on `sm_121a`, for **both** `nvidia/Qwen3.6-27B-NVFP4`@`0893e160` and
`nvidia/Qwen3.6-35B-A3B-NVFP4`@`491c2f1e`, at prefill and at decode:

- `GdnDebugCounters` (`chunk_o_f32_launches`, `chunk_o_bf16_launches`, and the
  delta_h / WU counterparts) with counters enabled;
- for any counter that reads zero, **which of the six conditions declined it**,
  established by reading the shapes actually passed — not inferred.

Record both models' head dims (`hv_n`) explicitly: if a gate model's `hv_n` is
outside `{48,32}` then Triton chunk_o never fires for it, and that is a finding
in its own right, not a configuration mistake.

Note these are the GDN **chunked-scan** kernels, so they bear on prefill/TTFT;
batch-1 decode goes through `GdnDecodeFusedKernel`. Do not claim a decode
throughput effect without a measurement that shows one.

## Gates

- Focused: the configure-matrix tests above.
- Full: `scripts/agent-preflight.sh` green; CUDA `ctest` on `sm_121a`; a CPU
  build and the `120a;121a` fat build both configure clean.
- No speed claim is required from this row. If a same-binary A/B is run, it needs
  3 reps per leg under one `flock` and medians, like any other.

## Evidence

`dgx:~/work/vllm.cpp-online-gate/evidence/<sha>/triton-default/` — configure logs
for each matrix cell, the counter dumps for both models, and the declined-reason
notes.

## Stop conditions

- Stop and report `NEEDS_DECISION` if making the default ON cannot preserve both
  a clean fat-build configure and the explicit-ON error.
- Do not regenerate vendored artifacts to make a cell pass.
- Do not edit a release-metadata expectation without re-deriving it from the rule.

## Outcome

Pending.
