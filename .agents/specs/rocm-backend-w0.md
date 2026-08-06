# ROCm/HIP backend — W0 skeleton

**Rows:** `BACKEND-ROCM` (backend-matrix, `INVENTORIED` → `ACTIVE`),
`BACKEND-GATE-ROCM-VLLM` / `BACKEND-GATE-ROCM-SGLANG` (both stay `INVENTORIED`,
hardware-blocked here).
**Claim:** `CLAIM-ROCM-W0-1`.
**Upstream / port source:** `vllm/platforms/rocm.py` and `vllm/CMakeLists.txt`
@ pin `555967922` (the same pin the oracle runs). ROCm is a platform upstream
**already has**, so unlike Metal and Vulkan this is a MIRROR, not a recorded
extension.
**Origin:** [issue #41](https://github.com/mudler/vllm.cpp/issues/41) — three
contributors offered AMD hardware (gfx1151 Strix Halo, gfx1103 Radeon 780M,
4x gfx1100 7900 XTX). This spec is the skeleton that turns those offers into
bounded first tasks; the contributor-facing guide is [docs/ROCM.md](../../docs/ROCM.md).

---

## Spike contract

| Field | Value |
|---|---|
| Scope | IN: `kROCM` device type, a `vt::Backend` over the HIP runtime (6 virtuals + multi-device registrar), a `Platform` mirroring `vllm/platforms/rocm.py`, ONE registered op (RmsNorm), the `VLLM_CPP_HIP` build switch, and the tests. OUT: attention (empty priority list), hipGraph capture, `needs_weight_staging`, fp8/MXFP4/AITER/RCCL, and every kernel beyond the one. |
| Upstream chain | `vllm/platforms/rocm.py` (`_capability_from_gcn_arch:223-291`, `_get_backend_priorities:407`, `get_attn_backend_cls:545`), `vllm/CMakeLists.txt:20-59,196-211,335-341`, `csrc/rocm/` (RDNA3 kernels), `platforms/__init__.py:202-208` (probe order) @ pin `555967922`. ROCm is a platform upstream HAS, so this is a MIRROR, not an extension. |
| Our baseline | Nothing: no `kROCM`, no HIP build, no platform, no kernel before this change. The seams it lands through are all present (`vt::Backend`, `Platform`, the vt op table, the attention registry, the portable reference tier). |
| Port map | `rocm.py:223-291` -> `include/vt/rocm/rocm_arch.h` (1:1). `cuda_backend.cu` -> `src/vt/rocm/rocm_backend.hip` (virtual for virtual, incl. its unified-memory conjunction). `cuda_ops.cu:96-126` `RmsNormRowKernel` -> `src/vt/rocm/rocm_rmsnorm.hip` (hand-hipified; upstream root is `csrc/layernorm_kernels.cu`). `platforms/rocm.py` `class RocmPlatform` -> `src/vllm/platforms/rocm.cpp`. |
| Tests to port | vLLM has no C++ ROCm backend test to port. Newly authored: `tests/vt/test_rocm_arch.cpp` (upstream's own worked examples as cases) and `tests/vt/test_rocm_backend.cpp`. The numeric gate is INHERITED, not written: `test_backend_cross_device.cpp` covers any registered backend at NMSE <= 5e-4 vs the CPU oracle. |
| Gates | MET here (CPU tier, no GPU): clean `-Werror` build; `ctest` full suite; `test_rocm_arch` 40/40; `test_platform` incl. the new selection-walk case; `check-device-leakage` unchanged at 32; both non-HIP ROCm legs object-compiled. PENDING, hardware-blocked: every HIP compile, `test_rocm_backend`, the cross-device RmsNorm comparison, `BACKEND-GATE-ROCM-VLLM`, `BACKEND-GATE-ROCM-SGLANG`. See §8. |
| Dependencies | A contributor's AMD board (#41) is the only hard blocker; ROCm toolchain is contributor-side. Landed and depended on: the portable reference tier (`op_provider.h`), the cross-device harness, and the platform-selection walk gate. Nothing else in the tree depends on this row. Detail in §9. |
| Work breakdown | `W0` skeleton (this change, LANDED) -> `W1` first HIP compile -> `W2` on-device tests green -> `W3` first model e2e on a unified part -> `W4` kernel fan-out (one family per PR) -> `W5` attention backend -> `W6` vLLM-ROCm token gate -> `W7` throughput vs vLLM. Only W1/W2 are workable today, deliberately on three boards at once. Detail in §10. |
| Risks/decisions | R1 "never compiled" read as broken (said outright in 7 places); R2 a wrong `UnifiedMemory()` on a discrete board is memory corruption (probed conjunction + asserted); R3 a 64-lane wavefront breaking a 32-lane assumption (no warp primitives used); R4 platform-TU bit-rot (object-compiled in every build); R5 label drift to "build-supported" (labels defined, upgrade is report-gated); R6 three contributors colliding (milestones split by memory model). D1 empty attention priority, D2 `needs_weight_staging` base-false, D3 exactly one op. Detail in §11. |

## 0. Honesty statement — what is and is not claimed

**The three HIP-language translation units in this change have never been
compiled, by anyone.** No AMD GPU exists on the authoring machine and no ROCm
toolchain is installed on it. That is strictly weaker than this project's
"build-supported" label, which means the code compiles and emits real machine
code for an arch nobody has run. There is no such evidence here. The right
description is **UNBUILT**, and it appears in the header of every affected file,
in the CMake configure output, and in `docs/BUILD.md`.

The two NON-HIP legs (`src/vllm/platforms/rocm.cpp` and
`tests/vt/test_rocm_backend.cpp`) are a different case and are not covered by
that sentence: both are plain C++ by design, both compile `-Werror` clean here,
and a non-HIP build now object-compiles them as a bit-rot guard so a later change
to the `Platform` interface cannot break them silently for a contributor
mid-bring-up. The object is never linked, so a non-HIP build is unchanged.
Compiled is still not run: the test needs a GPU.

Specifically NOT claimed: that ROCm builds; that any model runs on an AMD GPU;
that the RmsNorm kernel produces correct numbers; that any speed relationship to
vLLM-ROCm, llama.cpp or anything else exists. No gate row moves to a passing
state and no benchmark number is recorded.

What IS claimed, and verified on this machine:

| Claim | Evidence |
|---|---|
| The `kROCM` device type does not break the tree | Clean `-Werror` CPU build; the enum addition forced exactly ONE switch site (`test_backend_cross_device.cpp:59`) |
| The GCN-arch capability parse is correct | `tests/vt/test_rocm_arch.cpp`, 7 cases / 40 assertions, green, CPU-only |
| Every DeviceType is reachable by platform selection | `tests/vllm/platforms/test_platform.cpp`, new case, green |
| The full existing suite is unaffected | `ctest` on the CPU tier |
| The two non-HIP ROCm legs compile `-Werror` | `vllm_rocm_platform_syntax_check` object target, built in every non-HIP build |

## 1. Why a skeleton at all, rather than waiting for hardware

The three volunteers each have an agent and a board. What they do not have is a
bounded first task: "write a backend" is a blank file, and a blank file written
three times in three different shapes is worse than nothing, because the seams
are what make a backend additive. The skeleton is the shape. It costs one
review here and removes the highest-variance part of the work from three people
who cannot see the record.

The second reason is that the memory-model decision is not obvious and gets
made ONCE, in `Backend::UnifiedMemory()`. Getting it wrong on a discrete card is
memory corruption rather than a slow path (see §3). It should not be rediscovered
per contributor.

## 2. What landed, and where the seams are

Seam numbering follows [backends.md](../backends.md).

| Seam | File | Language | Compiled here? |
|---|---|---|---|
| enum | `include/vt/device.h` | C++ | yes |
| 1, Platform | `src/vllm/platforms/rocm.cpp` | C++ (no HIP header) | **yes** — object-compiled as a bit-rot guard |
| — | `include/vt/rocm/rocm_arch.h` | C++ | **yes, and unit-tested** |
| — | `include/vt/rocm/rocm_runtime.h` | C++ | yes |
| runtime | `src/vt/rocm/rocm_backend.hip` | HIP | **no — UNBUILT** |
| 3, op table | `src/vt/rocm/rocm_ops.hip` | HIP | **no — UNBUILT** |
| 3, kernel | `src/vt/rocm/rocm_rmsnorm.hip` | HIP | **no — UNBUILT** |
| 2, attention | *(none — empty priority list)* | — | — |
| build | `CMakeLists.txt` (`VLLM_CPP_HIP`) | CMake | the OFF path and the fail-loudly path |
| test | `tests/vt/test_rocm_backend.cpp` | C++ | **compiled** (guard), never RUN |
| test | `tests/vt/test_rocm_arch.cpp` | C++ | **yes** |

**The split is the design.** Everything carrying a DECISION was pulled out of
the HIP translation units into plain C++ that this machine compiles and tests.
What remains unbuilt is API glue whose failure mode is a compile error on the
first contributor's machine — loud, immediate, and cheap to fix — rather than a
wrong answer.

## 3. The one decision that matters: `UnifiedMemory()`

The portable reference tier ([op_provider.h:186-224](../../include/vt/op_provider.h))
serves any op with no native kernel by running the CPU kernel, at a priority
strictly below every native one. It is gated on `Backend::UnifiedMemory()` and
never on `DeviceType`, because a CPU kernel dereferences HOST pointers, which is
valid only where host and device memory alias.

For ROCm this single bool splits the contributors' work:

- **gfx1151 / gfx1103 (integrated):** the tier installs, every unimplemented op
  falls back, and a model can run end to end against ONE registered kernel.
- **gfx1100 (discrete):** the tier must NOT install. `GetOp` throws on an
  unregistered op, which is the correct and safe behaviour.

The implementation answers `pageable_memory_access && integrated`, both probed
via `hipDeviceGetAttribute`. The conjunction is deliberate and is `CudaBackend`'s
own reasoning (`cuda_backend.cu:295-303`): pageable-memory-access alone is not
sufficient, because a discrete part can report it while the driver services host
pointers through page migration, which is not the zero-copy aliasing this
contract requires. Probed, never inferred from the gfx name.

## 4. Why RmsNorm is the one registered op

Not because it is easy. Because `tests/vt/test_backend_cross_device.cpp` runs
against every registered non-CPU backend, skips ops a device has not registered,
and compares the rest against the CPU backend at NMSE ≤ 5e-4 from the same
binary on the same inputs. Registering RmsNorm therefore hands a contributor a
real numeric gate on their own silicon without them writing a test.

RmsNorm specifically because it is a **cross-lane reduction**, which is where a
hipified kernel can actually be wrong: AMD's wavefront is 64 lanes where
NVIDIA's warp is 32, so a kernel that quietly assumed 32 breaks here. The ported
kernel does not assume — it reduces through `__syncthreads()` over shared memory,
so it is wavefront-width agnostic — and that property is now stated in the file
where the next person will read it. An elementwise op like `Add` would have
passed the same gate while proving nothing.

Ported from `cuda_ops.cu:96-126` (`RmsNormRowKernel`), whose own upstream
counterpart is `csrc/layernorm_kernels.cu`. The `VT_RMSNORM_DECODE_FAST` paths
are deliberately NOT ported: they exist to be bit-identical to this kernel while
being faster on sm_121a, and W0 is not a speed milestone.

## 5. The non-additive site this uncovered

`CurrentPlatform()` walks a hardcoded `kCurrentPriority` array
(`src/vllm/platforms/platform.cpp:46-56`). A platform missing from it registers
fine, answers every query correctly, and is **never selected** — and unlike a
missing enum case, no compiler diagnostic fires. `kROCM` was added there, after
`kCUDA`, mirroring upstream's own probe order (`platforms/__init__.py:202-208`
is `{tpu, cuda, rocm, xpu, cpu}`).

`CurrentPlatformPriority()` now exposes the walk so it can be asserted, and
`tests/vllm/platforms/test_platform.cpp` gates that EVERY `DeviceType` appears
in it and that CPU is last. That test is the actual deliverable of this section:
the next backend cannot reintroduce the silence.

This is worth recording against [backends.md](../backends.md)'s claim that
"adding a platform never touches engine code". The claim holds for the engine —
scheduler, KV, batch, sampler and serving are untouched — but the platform
SELECTION walk is a real exception, and it is now the only one, and it is tested.

## 6. Deliberately out of scope

- **Attention.** No kernel, so `get_attn_backend_priority()` returns EMPTY.
  Selection then throws loudly instead of naming a backend whose kernels do not
  exist. M3 fills it from `rocm.py:407` (`ROCM_ATTN`, `TRITON_ATTN` are the
  RDNA3-reachable entries; AITER is gfx9-only).
- **Graph capture.** `SupportsGraphCapture()` stays false. hipGraph is the
  mapping; a capture that bakes a wrong address is a silent correctness bug, and
  this project has been bitten by exactly that on CUDA twice.
- **`needs_weight_staging()`.** Stays base-false. HIP's programming model does
  stage, so a discrete card will eventually answer true, but W0 has one op, and
  answering true today routes a model into a path with no kernels.
- **fp8, MXFP4, AITER, RCCL, multi-GPU TP.** Registration of every visible
  device at its own `Device{kROCM,i}` slot is in (the 4-GPU box on #41 is why),
  but nothing consumes it yet.
- **CI.** No AMD runner exists. The HIP path is built by contributors, and the
  configure output says so.

## 7. What a contributor does next

```sh
cmake -S . -B build-hip -DVLLM_CPP_HIP=ON -DVLLM_CPP_HIP_ARCHITECTURES=gfx1100
cmake --build build-hip -j
ctest --test-dir build-hip -R 'rocm|cross_device'
```

Expected outcomes, in the order they will be hit:

1. **It does not compile.** Most likely, and the most useful possible report.
   The fix belongs in this skeleton, not in a fork.
2. **It compiles and `test_rocm_backend` fails.** Also a skeleton bug. The arch
   string is printed unconditionally so the report names the board.
3. **Both pass.** Then `test_backend_cross_device` has compared a real ROCm
   RmsNorm against the CPU oracle, seam 3 is proven end to end, and M2 (a model,
   via the reference tier on an integrated part) is the next row.

Milestones M0-M5 and their acceptance are in [docs/ROCM.md](../../docs/ROCM.md) §5.

## 8. Gates

| Gate | State |
|---|---|
| CPU tier `-Werror` clean, full `ctest` | PASS (this machine) |
| `test_rocm_arch` | PASS, 40 assertions |
| `test_platform` incl. the new priority-walk case | PASS |
| `check-device-leakage` ratchet | PASS, unchanged at 32 |
| HIP build on any arch | **PENDING — no hardware** |
| `test_rocm_backend` | **PENDING — no hardware** |
| Cross-device RmsNorm vs CPU oracle on AMD | **PENDING — no hardware** |
| `BACKEND-GATE-ROCM-VLLM` (vLLM-ROCm oracle, same box) | **PENDING — no hardware** |

A gate that cannot run here stays PENDING. None of these is failing; all of them
are unobserved, which is a different thing and is recorded as such.

## 9. Dependencies

| Depends on | State | Why it blocks |
|---|---|---|
| A contributor's AMD board (#41) | OFFERED, not yet run | Nothing here can be compiled without one; this is the only hard blocker |
| ROCm toolchain (hipcc), any recent version | Contributor-side | `-DVLLM_CPP_HIP=ON` fails the configure without it, by design |
| Portable reference tier (`op_provider.h`, S5) | LANDED | Is what lets a unified-memory board run a model against one kernel |
| Cross-device harness (`test_backend_cross_device.cpp`) | LANDED | Supplies the CPU-oracle numeric gate the first ROCm op inherits |
| Platform-selection walk gate (previous commit) | LANDED | Without it a registered ROCm platform would never be selected |

Depends on NOTHING that is unbuilt or unowned. Nothing else in the tree depends
on this row, so it cannot block anyone.

## 10. Work breakdown

Non-overlapping, in dependency order. W0 is this change; the rest are open.

| ID | Work | Owner | Blocked by |
|---|---|---|---|
| `W0` | Skeleton: enum, backend, platform, 1 op, build switch, tests, docs | LANDED (this change) | — |
| `W1` | First HIP compile on any gfx; fix what it finds | Contributor with a board | W0 |
| `W2` | `test_rocm_backend` + `test_backend_cross_device` green on device | Contributor | W1 |
| `W3` | First model e2e on a UNIFIED part via the reference tier; capture the `VT_OP_PROVIDER_STATS=1` fallback list | gfx1151 / gfx1103 | W2 |
| `W4` | Kernel fan-out, one family per PR, ordered by W3's fallback list | Any board | W3 (order), W2 (discrete boards start here) |
| `W5` | ROCm attention backend + its priority slot | Any board | W4 |
| `W6` | vLLM-ROCm oracle token gate on the same box | A board that can host vLLM-ROCm | W5 |
| `W7` | Throughput vs vLLM, quant-matched | Same box as W6 | W6 |

W1-W2 are the only ones that can be worked today, and they are the same task on
three different boards, which is deliberate: three independent first-compile
reports are worth more than one.

## 11. Risks and decisions

| # | Risk | Mitigation / decision |
|---|---|---|
| R1 | The .hip TUs do not compile, and a contributor reads that as "the project is broken" | Said outright everywhere a reader can land: file headers, CMake configure output, README, BUILD.md, STATUS.md, matrix row, and the PR. A compile error is named as the EXPECTED first outcome |
| R2 | `UnifiedMemory()` answers true on a discrete board → CPU kernels dereference device pointers → memory corruption, not a slow path | Probed as `pageable_memory_access && integrated`, never inferred from the gfx name; the conjunction is CudaBackend's own (`cuda_backend.cu:295-303`). `test_rocm_backend` asserts `ReferenceTierEligible(kROCM) == UnifiedMemory()` |
| R3 | The RmsNorm port silently assumes a 32-lane warp on 64-lane AMD hardware | The ported kernel uses NO warp-level primitive: a `__syncthreads()` shared-memory tree, width-agnostic. Stated in the file header so the next porter does not reintroduce the assumption |
| R4 | The platform TU bit-rots, since nothing here compiles it | Object-compiled in every non-HIP build (never linked), so a `Platform` interface change breaks CI here rather than a contributor's tree |
| R5 | "Never compiled" drifts into "build-supported" once someone reports success | The labels are distinct and defined in STATUS.md; a build report upgrades to build-supported ONLY, and running a model is a separate claim |
| R6 | Three contributors write the same file | Milestones are split by memory model in `docs/ROCM.md` §4 and claimed on #41 before work starts |

**D1 (decision).** Attention returns an EMPTY priority list rather than naming
`ROCM_ATTN`. An empty list makes selection throw loudly; a name we cannot honour
would fail deep in a model forward. Revisit at `W5`.

**D2 (decision).** `needs_weight_staging()` stays base-false even though HIP
stages, because W0 has one op and answering true routes a model into a path with
no kernels. Revisit at `W3` on a discrete board.

**D3 (decision).** One op, not zero and not ten. Zero proves no seam; ten is
unreviewable code nobody can compile. One, chosen so the existing cross-device
harness gates it for free.
