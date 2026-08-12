# Native MSVC strict CPU and Vulkan build repair

Identity: `ENG-RELEASE-WINDOWS`

Issue: [#459](https://github.com/mudler/vllm.cpp/issues/459)

Parent specification: [windows-binary-release.md](windows-binary-release.md)

Status: `ACTIVE`. This repair starts from PR #446 head
`036c80c01c7094a7c45cdfbcfa746f462cbde31e` and preserves the required native
MSVC `/W4 /WX` release gate.

## Scope

Make the native Windows x86_64 CPU and Vulkan release configurations compile
cleanly with MSVC warnings treated as errors. Keep `/WX`, the static `/MT`
runtime, archive contents, tuple identities, preview status, and runtime gates
unchanged. This change may add narrowly valid Windows compilation definitions,
portable allocation/math helpers, and source-level type/name fixes. It must not
disable warning families globally, alter release assets, add Windows CUDA or
arm64, or modify issue #458's hermetic PR-size evidence-tool repair.

## Upstream and platform anchors

vLLM `555967922` remains the runtime-behavior oracle, but it has no native
Windows release lane. These fixes therefore follow the C++20/MSVC platform
contracts rather than porting vLLM behavior: Windows headers own the legacy
`min`/`max` macros unless `NOMINMAX` is defined; the UCRT deliberately marks
portable ISO CRT entry points such as `getenv`, `fopen`, and `sscanf` as unsafe
unless `_CRT_SECURE_NO_WARNINGS` declares the project's cross-platform CRT
contract; MSVC requires `/utf-8` for UTF-8 execution characters; and its C++
library does not provide `std::aligned_alloc`.

## Observed baseline

Hosted run `31574038913` reached real compilation and failed both
`windows-msvc-cpu` job `94042049971` and `windows-msvc-vulkan` job
`94042049929`. The complete unique warning-code set in both logs is C4003,
C4189, C4244, C4324, C4456, C4458, C4477, C4566, and C4996. The hard-error set
is C2039, C2059, C2065, C2143, C2589, C2737, C2760, C3861, and C3878; C2220 is
the expected `/WX` promotion wrapper.

The diagnostics reduce to these root causes:

- UCRT C4996 for the project's intentional portable `getenv`, `fopen`, and
  `sscanf` usage across the reachable source closure;
- Windows `min`/`max` macro expansion corrupting qualified `std::min` and
  `std::max` calls in LMCache and the CPU thread pool;
- absent `std::aligned_alloc` in `src/vt/cpu/cpu_backend.cpp`;
- absent `M_PI` in MLA and Minimax audio/video model sources;
- tokenizer Unicode literals compiled with the CP1252 execution character set;
- intentional cache-line alignment padding in `cpu_threadpool.h` diagnosed as
  C4324;
- genuine source warnings: unused `kKV`, integer narrowing in Gemma audio,
  LMCache, and range materialization, local/member shadowing in Minimax video
  and the single-type KV manager, and `%ld` scanning into `int64_t *`.

## Design

Define `NOMINMAX`, `_CRT_SECURE_NO_WARNINGS`, and UTF-8 compilation at the
central MSVC target contract because they describe one valid project-wide
Windows ABI/source policy. Keep `/W4 /WX`. Treat C4324 narrowly at the
intentional cache-line-aligned type rather than disabling the warning for the
target. Replace `std::aligned_alloc` with a small platform-owned aligned
allocation/deallocation pair whose Windows arm uses `_aligned_malloc` and
`_aligned_free`, preserving the existing alignment and lifetime semantics.
Use standard C++20 math constants rather than enabling non-standard `M_PI`.
Correct genuine narrowing, format, unused, and shadow diagnostics in source so
MSVC and non-MSVC builds express the same types and behavior.

## Tests and RED evidence

Extend the Windows portability mutation suite with focused assertions over the
real build/source contract. Before implementation, the new tests must fail for
the intended missing central MSVC definitions/UTF-8 contract, missing portable
aligned-allocation pairing, non-standard `M_PI`, and the observed genuine
warning patterns. Mutation coverage must remove or corrupt each critical guard
and make the focused suite fail. The hosted MSVC jobs remain the authoritative
compiler execution gate; Linux structural tests are not represented as native
compilation.

## Gates

1. RED: the focused portability regression tests fail at pinned head
   `036c80c0` for the observed contracts.
2. GREEN: `python3 -m unittest tests.scripts.test_check_windows_portability -v`.
3. Linux regression: clean CPU configure/build plus relevant focused C++ tests,
   and a Vulkan configure/build when the local Vulkan toolchain is available.
4. Full `scripts/agent-preflight.sh` before and after the implementation commit.
5. Exact PR base-to-candidate validation with the repository range checker when
   its independent #458 hermetic-tool fix is available; otherwise report that
   hosted-only boundary precisely.
6. Rerun the two native hosted jobs on the integrated PR head. They must compile
   with `/W4 /WX` before their runtime gates can be accepted.

## Risks and stop conditions

The main risk is hiding a genuine warning behind a broad compiler exception.
Only the three central platform definitions and the one intentional aligned
layout diagnostic are eligible for contract-level treatment; all semantic
type/name/format warnings are source fixes. Stop with `NEEDS_DECISION` if a
diagnostic requires changing runtime behavior, public ABI, tensor shapes,
release contents, or warning strictness. Stop with `NEEDS_CONTEXT` if the
hosted log exposes a diagnostic not present in the fetched CPU/Vulkan evidence.

## Outcome

The complete CPU and Vulkan job logs from hosted run `31574038913` were
reduced to the warning and hard-error classes in this spec. The implementation
keeps `/W4 /WX`, establishes the central Windows CRT/header/UTF-8 contract,
pairs the platform allocator correctly, narrows the one intentional aligned
layout warning to `Threadpool`, and fixes every source-specific diagnostic.

The three focused regression tests failed at the pinned baseline with 15
intended missing-contract/source-shape failures, then the complete Windows
portability suite passed 62/62. Clean local CPU and Vulkan configurations built
403/403 and 413/413 targets respectively. All selected CPU tests passed; all
selected non-API Vulkan tests passed. The Vulkan configuration newly exposed
two embedding API runtime cases returning HTTP 500 where the CPU configuration
passes; that behavior is outside this compile-only repair and requires its own
tracked decision rather than a silent scope expansion.

Fresh mutation review then restored
`src/vllm/v1/core/single_type_kv_cache_manager.cpp` byte-for-byte from the
parent and found that the real-tree source-warning test did not detect the
reintroduced C4458 declarations. The repaired test structurally scopes the
five affected manager methods after removing comments, literals, and complete
preprocessor directives. Restoring parent blob `98f8665d` now produces 12
focused failures across the `block_pool`, `kv_cache_spec`, and `block_size`
shadow families; candidate blob `db3d4f15` passes.

Native MSVC CPU/Vulkan reruns remain the authoritative `/W4 /WX` validation and
are not inferred from Linux. The integrated exact PR-range gate uses the
independent #458 hermetic evidence-tool repair carried by the candidate.
