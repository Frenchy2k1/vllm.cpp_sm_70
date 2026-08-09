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

### CORRECTION (implementation, 2026-08-09): conditions 2 and the "explicit path
### keeps failing loudly" clause are REFUTED

Conditions 1 and 3 stand. Condition 2 and the multi-arch `FATAL_ERROR`
requirement above were written from a stale reading of `TritonAOT.cmake` and are
false on `origin/main` @`785e2978`.

`_triton_aot_arch_name` — the function holding the multi-arch `FATAL_ERROR` — is
**unreachable from the builder path**. `add_triton_kernel` (`:226-228`) and
`triton_aot_finalize` (`:437-444`) both branch on `VLLM_CPP_TRITON_REGEN`, and
the non-regen branch calls `_triton_aot_arch_names`, which returns
`vt_triton_aot_available_arches` — *every* vendored tree — and never consults
`VLLM_CPP_CUDA_ARCHITECTURES`. W2 embeds all six trees and selects one by exact
SM at runtime. Only maintainer regeneration, which writes a cubin into one
directory, reaches the single-arch guard.

Measured, configure-only, on the gate host (`dgx.casa`, sm_121a, nvcc 13.0,
`-DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0`), against an unmodified
`origin/main` @`785e2978`:

| cell | exit | resolved |
|---|---|---|
| `-DVLLM_CPP_CUDA_ARCHITECTURES="120a;121a" -DVLLM_CPP_TRITON=ON` | **0** | `ON` |
| `-DVLLM_CPP_CUDA_ARCHITECTURES="80;…;121a" -DVLLM_CPP_TRITON=ON` | **0** | `ON` |

The fat cell printed `Triton AOT W2: embedded trees
[sm_80;sm_86;sm_89;sm_90a;sm_100a;sm_121a]` and configured clean. Two shipped
callers already depend on this: `scripts/build-linux-accelerator-release.sh:23,47`
builds the release archive as ten SMs *with* `-DVLLM_CPP_TRITON=ON`, and
`.github/workflows/ci.yml:344-346` does the same in `cuda-fat-build`. Making the
explicit multi-arch path fatal would break both, and
`test_release_accelerator_metadata.py:107` requires `VLLM_CPP_TRITON=ON` in a
CUDA release cache, so it would be self-contradictory as well.

**What was implemented instead.** The default is `ON` iff CUDA is enabled, every
vendored tree is present, and `VLLM_CPP_TRITON_REGEN` is `OFF` — the third
condition added because regeneration genuinely is single-arch, so the default
must not select it for you. The architecture **count is not a condition**;
`120a;121a` and the ten-SM set both resolve `ON` and configure clean. The
`FATAL_ERROR` at `TritonAOT.cmake:116-127` is **byte-unchanged** and still
guards the regen flow it actually protects. No new error was added, so the
default can never turn a configure that used to succeed into a failure.

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

Implemented on `row/BUILD-TRITON-DEFAULT-ON`, base `origin/main` @`bd6b3936`.
Gate host `dgx.casa` (GB10, sm_121a, nvcc 13.0.88, CUTLASS 4.5.0).

### The rule that shipped

`vt_triton_aot_computed_default` (`cmake/TritonAOTMultiArch.cmake`): `ON` iff
CUDA is enabled **and** every vendored tree is present **and**
`VLLM_CPP_TRITON_REGEN` is `OFF`. The **architecture count is not a condition** —
see the CORRECTION under Design. Third condition added because regeneration
genuinely writes into one tree, so the default must not select it for you.

### Configure matrix (configure-only, one script against both trees)

| cell | base | new | exit |
|---|---|---|---|
| `cpu-default` | OFF | OFF | 0 |
| `vulkan-default` | OFF | OFF | 0 |
| `cuda-121a-default` | OFF | **ON** | 0 |
| `cuda-120a-default` | OFF | **ON** | 0 |
| `cuda-110-default` | OFF | **ON** | 0 |
| `cuda-fat-default` (`120a;121a`) | OFF | **ON** | 0 |
| `cuda-121a-explicit-on` | ON | ON | 0 |
| `cuda-121a-explicit-off` | OFF | OFF | 0 |
| `cuda-fat-explicit-on` | ON | ON | 0 |
| `cuda-release-fat-explicit-on` (ten SM) | ON | ON | 0 |
| `cuda-121a-regen-default` | OFF | OFF | 0 |
| `cuda-fat-regen-explicit-on` | ON | ON | **1** |

The last row is the preserved `FATAL_ERROR`: byte-identical text on both trees
(only its line number moved, 116 → 136, from added comment lines).

A full CUDA gate build configured with **no** `-DVLLM_CPP_TRITON` prints
`Triton AOT W2: embedded trees [sm_80;sm_86;sm_89;sm_90a;sm_100a;sm_121a]`
beside the CUTLASS and FlashAttention-2 lines, and 1017 translation units carry
both `VLLM_CPP_TRITON=1` and `VLLM_CPP_TRITON_CHUNKO_BF16=1` (Tests §3).

### Dispatch evidence — the Triton chunk_o path RUNS, on both gate models

`GdnDebugCounters`, captured per phase in the real paged engine:

| model | `hv_n` | phase | `chunk_o_f32` | `chunk_o_bf16` | `chunk_o_hand` |
|---|---|---|---|---|---|
| 27B @`0893e160` | 48 | prefill (step 1) | 0 | **48** | **0** |
| 27B @`0893e160` | 48 | decode (step 2) | 0 | 0 | 0 |
| 35B-A3B @`491c2f1e` | 32 | prefill (step 1) | **30** | 0 | **0** |
| 35B-A3B @`491c2f1e` | 32 | decode (step 2) | 0 | 0 | 0 |

`hv_n` is `linear_num_value_heads` read from each checkpoint's `config.json`:
**48** (27B) and **32** (35B). Both are inside `{48,32}`, so the fourth bail-out
never fires — the spec's "if `hv_n` is outside `{48,32}` that is a finding" case
does not arise.

**Not one of the six bail-outs declined anything at prefill.** `chunk_o_hand`
is 0 in both models, so every GDN layer took the Triton realization: 48 of 48
GDN layers on the 27B, 30 of 30 on the 35B.

The zeros are not declines:

- `chunk_o_f32 = 0` on the 27B and `chunk_o_bf16 = 0` on the 35B are the *other*
  `Tout` template instantiation of the same call site. The 27B dense path runs
  the vLLM-faithful **bf16** GDN output; the 35B MoE path runs the **f32**
  recurrence output. Each model calls exactly one instantiation.
- **Every decode counter is 0, including `chunk_o_hand`.** The hand counter is
  incremented on the `!triton_chunko` branch of the same call site, so both being
  zero means `TryTritonChunkO` was never *called*: the chunked-scan path does not
  execute at batch-1 decode at all. That is the code path, not a bail-out —
  matching the spec's own note that decode goes through `GdnDecodeFusedKernel`.
  **No decode-throughput effect is claimed.**

Scratch pools behaved: 27B prefill 432 chunk reuses / 96 WU reuses with 0
growths; 35B 261 / 58 with 0 growths.

### Gates

- Focused: `cmake -P cmake/TritonAOTDefaultTest.cmake` — ALL PASS (RED before:
  `Unknown CMake command "vt_triton_aot_computed_default"`). Wired into the
  `cuda-arch-features` CI job beside `TritonAOTMultiArchTest.cmake`.
- `scripts/agent-preflight.sh --staged` — exit 0.
- CUDA `ctest` on sm_121a, **serial**, from the default-ON build:
  GDN/Triton subset **9/9 PASS**.
- 27B SACRED gate (`test_qwen27_paged_engine`, pinned unsloth @`890bdef7`):
  **235/235 assertions, SUCCESS** — the documented full-production-stack result,
  now reached with no build flag to remember.
- 35B gate (`test_qwen36_paged_engine`, pinned nvidia @`491c2f1e`):
  **2/2 cases, 315/315 assertions, SUCCESS**.
- Release metadata: all eleven tests pass **unedited**; every expectation
  re-derived from the new rule and unchanged (see the commit body).

### Pre-existing failures, attributed not assumed

The serial suite reached **379/395** with **nine** failures. Five are
**A/B-attributed to `origin/main`**: built there with its own default
(`VLLM_CPP_TRITON:BOOL=OFF`), same box, toolchain and CUTLASS, each fails with
the same assertion, the same line and the same pass/fail counts, so none is
caused by this change. A sixth shares an already-attributed root cause. The
remaining three are covered by a comprehensive same-set A/B against `origin/main`
that is queued to run when the suite finishes (`triton-abfull`, logs
`out-gate/abfull-*.log`).

**Attributed to main by direct A/B:**

- `test_linear_method` — `CHECK(after == before + 1)` at `:246`. Cause pinned
  exactly: `VT_MARLIN_DENSE` defaults **ON**
  (`dense_nvfp4_gemm.h:121-127` returns true unless the value is `0`), so
  `GateUpFusedMarlinD` takes the dense-GEMM branch and increments `dense_gemms`
  instead of `fused_gate_up`. Re-running the SAME binary with
  `VT_MARLIN_DENSE=0` passes 1/1. The comment at `dense_nvfp4_gemm.h:527` still
  says "VT_MARLIN_DENSE (default OFF)", which is now false. **Owed its own
  issue.**
- `test_minimax_h3` — `vt cuda: cudaFree: invalid argument` at `:3535` then
  SIGSEGV at `:3945`, byte-identical on main. **Owed its own issue.**
- `test_serve_low_tools` — `test_script_stays_shellcheck_clean`, a lint test
  with no CUDA involvement, failing on main too. **Owed its own issue.**
- `test_glm4_moe_lite_paged_engine` — the GLM/DSA G1 gate, token divergence
  `REQUIRE(got[j] == od[i*T+j])` at `:283`, 70 assertions / 69 passed. Attributed
  **twice**: (a) the SAME binary with `VT_GDN_CHUNKO_TRITON=0
  VT_GDN_DELTAH_TRITON=0 VT_GDN_WU_TRITON=0 VT_GDN_PACKED_DECODE_TRITON=0` fails
  identically, and (b) the `origin/main` build fails identically. Consistent with
  the model: GLM-4.7-Flash's `config.json` carries **no `linear_*` keys at all**
  (`qk_nope_head_dim` 192 / `qk_rope_head_dim` 64 / `v_head_dim` 256 — MLA +
  DeepSeek-MoE), so it has no GDN layers and the GDN-only Triton kernels cannot
  reach it. **Owed its own issue** — this is a live SACRED gate that is red on
  main.
- `test_gemma4_registry_e2e` — `vt::GeluMulSeparate: ROCm-only fast path in this
  build` thrown at `:163` (`fused_ops.cpp:70`); no CUDA realization of that op is
  registered. Identical on main. **Owed its own issue.**

**Same root cause as an attributed one:**

- `test_gemma4_paged_engine` — the identical
  `vt::GeluMulSeparate: ROCm-only fast path in this build` throw, at `:67`.
  Folded into the `GeluMulSeparate` issue above.

**Pending the queued comprehensive A/B** (none is plausibly reachable from a
GDN-only cubin set, but none is asserted attributed until measured):

- `test_capi` — SIGSEGV at `tests/capi/test_capi.cpp:480`, the ABI v8
  custom-logits-processor case on a *synthetic* engine; 47/47 assertions passed
  before the crash, 3 of 4 cases green. `.agents/environment.md:189` already
  calls this a known flake here, but this run was `-j 1`, so that explanation
  does not apply and it needs the A/B.
- `test_qwen3_apc_e2e` — `REQUIRE(anchor_ok)` at `:375`, 59 assertions / 58
  passed. Automatic prefix caching, no GDN layers involved.
- `test_minicpm3_paged_engine` — `REQUIRE(first_div < 0)` at `:224`, 26
  assertions / 25 passed.

### Operational note (cost a box)

`ctest -j 4` on this build **OOM-rebooted dgx.casa** at 00:55 (kernel
`NVRM ... Out of memory [NV_ERR_NO_MEMORY]`): unified memory means concurrent
model gates stack into host RAM. Run the CUDA suite with `-j 1` here.

The full serial suite is the remaining open item. It was started, stopped at
171/395 to free `$HOME/gpu.lock` for the attribution arms (which queue behind
it), and restarted from the beginning; the restarted run reached **379/395**
with the nine failures above. Logs, all under
`dgx:~/work/triton-default-on/out-gate/`: `ctest-serial-full-partial.log` (the
first run), `ctest-serial-full.log` (the restarted run, ends with
`full-serial-exit=`), and `abfull-armA.log` / `abfull-armB-main.log` (the
comprehensive attribution pass).

Serialising the whole suite behind one `flock` also means every probe queues
behind it; run attribution arms before, not during, a full suite.
