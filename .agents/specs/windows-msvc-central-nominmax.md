# Native MSVC central `NOMINMAX` contract repair

Identity: `ENG-RELEASE-WINDOWS`

Issue: [#462](https://github.com/mudler/vllm.cpp/issues/462)

Parent specification: [windows-binary-release.md](windows-binary-release.md)

Predecessor repair:
[windows-msvc-strict-build.md](windows-msvc-strict-build.md)

Status: `ACTIVE`. This repair starts from PR #446 head
`38100fc38ede50c8bc0b2e0f0f7f8deda46408a6` and preserves the central MSVC
`NOMINMAX` definition and `/W4 /WX` gate introduced by #459.

## Scope

Remove every project-owned, unguarded source-local `NOMINMAX` definition that
redefines the central CMake command-line contract under MSVC. Preserve guarded
fallback definitions that make isolated source use safe, preserve local
`WIN32_LEAN_AND_MEAN` definitions, and do not change runtime behavior, release
contents, public ABI, tuple identity, or warning strictness. Do not add a global
C4005 suppression and do not weaken the Windows portability checker.

## Observed baseline and root cause

Hosted run `31580273813`, native CPU job `94061520820`, failed under `/WX`
only with C4005 promoted through C2220. MSVC reported `NOMINMAX` as previously
declared on the command line and then redefined unconditionally at:

- `src/vllm/model_executor/models/minimax_h3_sharded.cpp:34`;
- `src/vllm/v1/kv_offload/fs_io.cpp:8`;
- `src/vllm/platform/console_shutdown.cpp:12`.

The same complete source scan finds an additional unguarded definition in
`tests/vllm/entrypoints/openai/test_api_server.cpp`; the CPU job stopped before
that test translation unit could become the next identical failure. Guarded
fallback definitions in the Vulkan loader and read-only mapping do not
redefine the command-line macro and are not defects.

The central definition is active from the start of preprocessing for every
project target. In each affected translation unit it therefore precedes both
the project/standard headers that appear before the local Win32 block and the
eventual `<windows.h>` include. Removing an unguarded local definition cannot
move protection later in the include chain; it removes only a redundant
second definition. `WIN32_LEAN_AND_MEAN` is not part of the central contract
and remains immediately before `<windows.h>`.

The native Vulkan job `94061520778` was still running when the spec was
committed. Its final log was inspected before implementation handoff and
contained the identical three C4005/C2220 paths with no distinct diagnostic.

## Test and implementation contract

Add one focused structural regression to the existing Windows portability
suite. It must derive the central-contract state from the real CMake source,
scan project-owned C++ sources and tests, and reject every active unguarded
`#define NOMINMAX` while accepting `#ifndef NOMINMAX` fallbacks. Before the
source repair it must fail with the exact four project-owned offending paths.

After RED, remove only those four redundant definitions. The test must then
pass without changing the central CMake definition, `/W4 /WX`, guarded
fallbacks, or `WIN32_LEAN_AND_MEAN`.

## Gates

1. RED focused regression at pinned head, with all four paths reported.
2. GREEN focused regression and the complete Windows portability suite.
3. Clean Linux CPU and Vulkan configure/build of the affected source closure,
   plus the OpenAI API, filesystem offload, Vulkan loader/backend, and
   cross-device tests available locally.
4. Full unstaged, staged, and post-commit `scripts/agent-preflight.sh`.
5. Exact PR #446 range gate from `a170c81c` to the immutable candidate.
6. Native CPU and Vulkan reruns under `/W4 /WX`; Linux cannot substitute for
   this compiler gate.

## Risks and stop conditions

The risk is deleting a definition that is the first protection before a
Windows header in an isolated build. The central target contract and include
ordering must be verified before editing each site. Stop with `NEEDS_CONTEXT`
if the final Vulkan log has a different diagnostic. Stop with `NEEDS_DECISION`
if a fix would require a warning suppression, compile-contract relocation,
runtime/public behavior change, or release-semantic change.

## Outcome

The focused regression failed at the pinned baseline with the exact four
project-owned unguarded definitions: the three translation units reached in
both hosted jobs and the OpenAI API test translation unit compilation had not
yet reached. The implementation removes only those four redundant definitions;
the central CMake `NOMINMAX`, `/W4 /WX`, every `WIN32_LEAN_AND_MEAN`, and the
guarded source-local fallbacks remain unchanged. The focused regression and the
complete Windows portability suite then passed, 1/1 and 63/63 respectively.

Clean local CPU and Vulkan source closures built 397/397 and 407/407 targets.
CPU API, filesystem, and Vulkan-loader tests passed. Vulkan filesystem,
loader, backend, and cross-device tests passed. The Vulkan-configured API test
retains the two embedding HTTP-500 failures already tracked independently by
#461; the same CPU-configured test passes 54/54. Native MSVC CPU/Vulkan reruns
remain required and are not inferred from Linux.
