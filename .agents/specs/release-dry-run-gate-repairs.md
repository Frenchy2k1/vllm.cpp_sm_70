# Merged-SHA release dry-run gate repairs

Identity: `ENG-RELEASE-WINDOWS`

Issues:
[#499](https://github.com/mudler/vllm.cpp/issues/499) and
[#500](https://github.com/mudler/vllm.cpp/issues/500)

Parent specifications:
[release-binary-matrix.md](release-binary-matrix.md) and
[windows-binary-release.md](windows-binary-release.md)

Related native compiler contract:
[windows-msvc-strict-build.md](windows-msvc-strict-build.md)

Status: `READY`. The developer approved this bounded design on 2026-08-12. It
starts from exact main commit `e1087a8812c9b7d96fca5a813981f378fcace638`
after PR #446 merged. Implementation, a fresh review, the operator gate, and a
new non-publishing ten-tuple dry run remain required.

## Scope

Repair the two independent failures exposed by the first exact-merged-SHA
release dry run after PR #446:

1. make the version compiled into every release `vllm-server` equal the exact
   release identity supplied by the release plan, including `-pre.1`, while
   preserving the existing backend suffix; and
2. make `tests/vt/test_cpu_isa_x86.cpp` own the standard stream definition its
   doctest `std::string_view` diagnostics instantiate under MSVC.

Keep CMake's numeric `project(vllm_cpp VERSION 0.0.3)` identity, archive names,
manifests, `VERSION` records, C ABI version, backend suffix policy, release
matrix, `/W4 /WX`, and the strict extracted-archive version validator
unchanged. Do not change production ISA selection, suppress a warning, alter a
release tuple, publish a tag or release, or fold an independent dry-run failure
into these issues.

## Hosted baseline and root causes

Manual workflow run
[`31607273683`](https://github.com/mudler/vllm.cpp/actions/runs/31607273683)
executed against exact merged SHA
`03ed25a5745ccbbfba7ce9e0f5a1c4a5251b7c30` with release identity
`0.0.3-pre.1`.

Issue #499 affected six jobs after their Linux or macOS build and packaging
work had completed:

| Tuple/job | Job ID | Observed gate |
|---|---:|---|
| Linux arm64 CPU | `94152936141` | extracted version mismatch |
| macOS arm64 Metal | `94152936210` | extracted version mismatch |
| Linux x86_64 Vulkan | `94152936237` | extracted version mismatch |
| Linux x86_64 CPU | `94152936310` | extracted version mismatch |
| Linux x86_64 musl CPU | `94152936346` | extracted version mismatch |
| macOS arm64 MLX | `94152936489` | extracted version mismatch |

`CMakeLists.txt:12` declares only numeric project version `0.0.3` and
`CMakeLists.txt:689` configures `include/vllm/version.h.in`. That template
exports only the numeric `PROJECT_VERSION_{MAJOR,MINOR,PATCH}` components, and
`src/vllm/version.cpp:5-12` reconstructs `0.0.3` before applying the current
backend suffix. The release builders require `VERSION`, but their CMake
configure calls do not pass it: `scripts/build-cpu-release.sh:26-40`,
`scripts/build-linux-accelerator-release.sh:32-48`,
`scripts/build-macos-release.sh:29-43`, and
`scripts/build-windows-release.ps1:180-198`. They use `VERSION` later for the
archive and metadata, so the same archive contains manifest/`VERSION` identity
`0.0.3-pre.1` and a server that reports `0.0.3`.

This is not a validator defect. `scripts/validate-release-archive.py:662-671`,
`:740-749`, and `:782-790` correctly require the extracted executable's output
to contain the manifest's exact version and the `VERSION` record's exact C ABI.
That comparison remains fail-closed.

Issue #500 affected native Windows CPU job `94152936312` and Vulkan job
`94152936445`. Both failed while compiling
`tests/vt/test_cpu_isa_x86.cpp:112` through doctest's binary-expression
stringification. The operands are `std::string_view`; the translation unit
includes `<string>` but not `<ostream>`. Under this MSVC include closure,
`std::basic_ostream` is only forward-declared, so the library's
`operator<<(basic_ostream&, string_view)` instantiation produces incomplete
type, missing `iostate`, and cascading C2146/C2027/C2065 diagnostics. The
production x86 ISA code compiled far enough to reach this test-only boundary.

## Design

### Exact compiled build version

Add a CMake cache string named `VLLM_CPP_BUILD_VERSION`. Its default is the
complete numeric `${PROJECT_VERSION}`, so developer and non-release builds keep
reporting `0.0.3` without a new required argument. Generate that string into
`include/vllm/version.h` and make `vllm::Version()` begin with the generated
build version instead of reconstructing the three numeric components.

Preserve the existing backend suffix exactly after that identity. For example,
the default CPU build reports `0.0.3`, a prerelease CPU artifact reports
`0.0.3-pre.1`, and a prerelease CUDA artifact reports
`0.0.3-pre.1+cuda`. Do not derive the prerelease from the tag inside CMake and
do not change the numeric `PROJECT_VERSION`, shared-library `VERSION`/`SOVERSION`,
or C ABI version.

All four release builder families pass their already-required `VERSION` value
to CMake as `-DVLLM_CPP_BUILD_VERSION=...`. The workflow remains responsible
for deriving the single release identity from `release/release-version.json`
and supplying it to each builder. CMake must reject an empty explicit build
version rather than silently reverting to the project version.

The generated value is build metadata, not a second release authority. The
release plan, manifest, archive name, `VERSION` record, executable output, and
post-publication audit must still agree, and the extracted-archive validator
continues enforcing that agreement.

### Explicit stream ownership in the MSVC test

Add the standard `<ostream>` definition directly to
`tests/vt/test_cpu_isa_x86.cpp`. This is a test translation-unit header-hygiene
repair: it makes doctest's diagnostic formatting valid without relying on a
transitive include. Do not change the compared `std::string_view` values,
doctest expressions, production x86 ISA code, compile flags, warning policy, or
standard-library formatting behavior.

## RED-first tests and mutation evidence

Implementation starts by capturing failures against the pinned baseline:

1. Extend the version test contract so a default configuration requires exact
   `0.0.3` plus the existing backend suffix, while a separate configure with
   `-DVLLM_CPP_BUILD_VERSION=0.0.3-pre.1` requires exact `0.0.3-pre.1` plus that
   same suffix. The prerelease arm must fail on the current numeric-only
   `Version()` implementation.
2. Extend `tests/scripts/test_release_pipeline.py` over the real four builder
   scripts. It must require each configure invocation to forward its existing
   `VERSION` input through `VLLM_CPP_BUILD_VERSION`; deleting the argument,
   substituting a literal, using `PROJECT_VERSION`, or omitting one builder
   must fail.
3. Preserve or extend the archive-validation mutation proving that a server
   output of `0.0.3` is rejected when manifest and `VERSION` declare
   `0.0.3-pre.1`. No skip-version path may be introduced into a release job.
4. Extend the Windows portability suite over the real
   `tests/vt/test_cpu_isa_x86.cpp` closure. It must require the explicit
   standard stream definition used by the `std::string_view` doctest
   expressions; removing it must fail the structural test and the native MSVC
   compile. Comments, literals, or unrelated transitive includes are not valid
   substitutes.

Every mutation runs in a scratch copy and restores the candidate tree
byte-for-byte. A structural Linux result does not substitute for executing the
native MSVC compiler.

## Gates

Focused local gates are:

1. the default and prerelease-configured `test_version` cases, including the
   current backend-suffix arm;
2. `python3 -m unittest tests.scripts.test_release_pipeline -v` and the focused
   release archive/metadata validator tests;
3. `python3 -m unittest tests.scripts.test_check_windows_portability -v` plus
   the direct portability checker; and
4. clean CPU configure/build and `test_cpu_isa_x86` execution on the available
   local host, followed by full unstaged, staged, and post-commit
   `scripts/agent-preflight.sh`.

Hosted acceptance requires native `windows-2022` CPU and Vulkan Release builds
to compile `test_cpu_isa_x86` under unchanged `/W4 /WX`, then execute their
existing focused runtime/ISA and archive gates. The six affected Linux,
macOS, and musl jobs must extract the archive and accept the exact
`0.0.3-pre.1` server identity. CUDA jobs must preserve their existing
`+cuda` suffix after the same prerelease identity.

After merge, rerun the complete ten-tuple non-publishing workflow at the exact
merged SHA. Every required tuple, aggregate handoff, and verify job must be
green, and each archive/manifest/`VERSION`/executable identity must agree.
Manual dispatch must continue skipping attest and publish. Only that immutable
dry run can authorize the already-planned developer-controlled
`v0.0.3-pre.1` tag flow; the tag run must rebuild, attest, publish exactly 32
assets, and pass the existing authenticated post-publication audit before any
binary-release claim advances.

## Risks and stop conditions

- A build-version string can accidentally become an independent release
  authority or be lost through one platform's quoting. The single declaration
  and all four builder mutations must keep the release graph closed.
- Replacing numeric component macros outright could break an internal consumer.
  Preserve them unless their removal is separately proved and reviewed; only
  `Version()` needs the full generated identity.
- Backend suffixes can be dropped or duplicated when the base string changes.
  Exact CPU and CUDA expectations are required.
- A transitive stream include may make a local compiler green while MSVC stays
  red. The test TU must own `<ostream>`, and native compilation remains binding.
- Stop with `NEEDS_DECISION` if #500 requires any production ISA, behavior,
  warning-level, or warning-suppression change.
- Stop with `NEEDS_CONTEXT` if a rerun exposes a direct diagnostic outside
  these two root causes. File and specify an independent issue rather than
  weakening a gate or silently expanding this repair.
- Do not tag or publish if any of the ten dry-run tuples, aggregate handoff,
  verify job, exact version checks, or Windows native gates is not green on the
  same merged SHA. A partial matrix is not release evidence.

## Written-spec self-review

- Scope is limited to exact compiled release identity and one test-TU stream
  definition; release topology and production ISA behavior do not change.
- The CMake default, release override, backend suffix, four builder inputs,
  strict validator, and native compiler boundary are explicit.
- RED tests and mutations fail for the two observed defects before code changes.
- Local, hosted, dry-run, tag-run, and post-publication evidence are separated;
  none is inferred from another.

## Outcome

Implemented both bounded dry-run repairs without changing release topology,
production ISA behavior, warning policy, or validator strictness.

- `VLLM_CPP_BUILD_VERSION` is now a non-empty CMake cache string defaulting to
  `PROJECT_VERSION`. The generated version header retains the numeric component
  macros and also carries that exact identity; `Version()` appends the existing
  `+cuda` qualifier to it unchanged.
- The CPU, Linux accelerator, macOS, and Windows release builders each forward
  their already-required `VERSION` value to CMake exactly once.
- `tests/vt/test_cpu_isa_x86.cpp` now owns the standard `<ostream>` definition
  required by its doctest diagnostics.

RED evidence on the pinned candidate preceded implementation. The builder
contract reported zero `VLLM_CPP_BUILD_VERSION` arguments for all four builder
families, the portability contract reported no active `<ostream>` include, and
a separately configured `0.0.3-pre.1` `test_version` failed because the binary
reported `0.0.3`. The retained strict-validator mutation rejects a numeric-only
server output when the manifest and `VERSION` declare `0.0.3-pre.1`.

Focused green evidence:

- release pipeline suite: 41 tests passed;
- Windows portability suite: 71 tests passed, followed by the direct checker;
- archive and platform metadata suites: 43 tests passed;
- an explicit empty build-version configure was rejected; and
- clean CPU builds passed the default and prerelease version tests. The clean
  prerelease build also passed all 8,242 x86 ISA assertions. Server smoke output
  was exactly `vllm.cpp 0.0.3 c-abi=17` by default and
  `vllm.cpp 0.0.3-pre.1 c-abi=17` with the release override.

The two hosted CUDA failures from dry run `31607273683` provide additional
root-cause evidence: jobs `94152936151` and `94152936356` both completed their
build and package phases and failed only because the extracted binary reported
the numeric project version. They did not expose a third repair within this
row's scope.

Native Windows compilation, the complete ten-tuple non-publishing workflow,
and the tag-run publication/audit remain post-merge acceptance gates. No tag or
release is authorized by the local evidence alone.
