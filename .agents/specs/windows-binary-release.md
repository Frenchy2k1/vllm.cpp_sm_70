# Native Windows binary release and pre-alpha publication

<!-- ENG-RELEASE-WINDOWS: state=ACTIVE publication=pending artifact=unpublished -->

Status: `ACTIVE`. Design approved by the developer on 2026-08-11. W14-W16 are
implemented locally; native hosted Windows evidence, the merged-SHA ten-tuple
dry run, prerelease publication, and the 32-asset audit remain pending.

Identity: `ENG-RELEASE-WINDOWS`

Issue: [#117](https://github.com/mudler/vllm.cpp/issues/117)

Documentation-checkpoint repair:
[#448](https://github.com/mudler/vllm.cpp/issues/448)

Archive-target documentation-checkpoint repair:
[#450](https://github.com/mudler/vllm.cpp/issues/450)

Exact-range checker-evidence repair:
[#453](https://github.com/mudler/vllm.cpp/issues/453)

Agent-record semantic-evidence repair:
[#455](https://github.com/mudler/vllm.cpp/issues/455)

Isolated evidence toolchain repair:
[#456](https://github.com/mudler/vllm.cpp/issues/456)

Parent contract: [release-binary-matrix.md](release-binary-matrix.md)

Planned publication: GitHub prerelease tag `v0.0.3-pre.1`.

## Scope

Extend the existing CI-built binary release from its eight Linux/macOS tuples
to ten tuples by adding native Windows x86_64 CPU and Vulkan downloads. The
result is a real Windows `vllm-server.exe`, not a Linux binary for WSL and not a
cross-compiled archive whose executable was never run. The first Windows-bearing
publication is deliberately pre-alpha: GitHub must report it as a prerelease,
and the generated index must say the Windows tuples are preview.

The two required new artifacts are:

| Artifact ID | Archive | Channel | Required evidence |
|---|---|---|---|
| `windows-x86_64-msvc-cpu` | `vllm.cpp-0.0.3-pre.1-windows-x86_64-msvc-cpu.zip` | preview | native MSVC build; extracted `--help`, `/health`, `/version`, clean shutdown; portable/SSE2 and AVX2 paths executed; PE dependency audit; model-loader and focused CPU correctness tests |
| `windows-x86_64-msvc-vulkan` | `vllm.cpp-0.0.3-pre.1-windows-x86_64-msvc-vulkan.zip` | preview | native MSVC Vulkan build; extracted `--help`, `/health`, `/version`, clean shutdown; Win32 loader contract; PE dependency audit; Vulkan runtime evidence remains false unless a real ICD runs the extracted binary |

All eight v0.0.2 tuples remain required and keep their current archive format,
channel, and evidence. This row does not add Windows CUDA, Windows arm64, an
installer, WSL artifacts, or per-ISA downloads.

## Upstream chain

Runtime behavior remains anchored to pinned vLLM `555967922`: its server and
model behavior are the oracle, while its `.buildkite/release-pipeline.yaml`
release lanes establish the production build/release comparison. That revision
has no native Windows lane, so Windows packaging and Win32 substrate are written
from scratch against the Windows API rather than attributed to vLLM. The model
mapping structure follows llama.cpp `src/llama-mmap.cpp:520-590` at
`237ad9b961f009ae19ac29dbce4cd0c1251f94b3`. The executing local chain is
GitHub Actions -> release plan -> native builder -> installed tree -> archive
validator -> immutable handoff -> aggregate verification -> GitHub attestation
and publication.

## Our baseline

The exact v0.0.2 SHA `7020de93652ca920424a10ac5255b34810dd2f24`
published eight Linux/macOS archive-checksum-provenance triplets plus two
indexes in workflow run `31466516224`; all build, verify, attest, and publish
jobs passed. The gaps are explicit in the current tree:
`release/release-matrix.json` has no Windows tuples,
`scripts/release_pipeline.py:49-54` and `scripts/release_index.py:77-86`
hard-code `.tar.gz`, `.github/workflows/release.yml` has no Windows builder, and
the shipped C++ sources still expose POSIX-only model mapping, process, signal,
socket, filesystem, and dynamic-library seams. No Windows artifact or runtime
evidence exists at this baseline.

The public lifecycle checker is itself stale at this baseline: it still
requires `docs/STATUS.md`, `docs/BENCHMARKS.md`, the parent engine row, and the
roadmap to claim that publication is pending. A staged preflight on this spec's
truthful v0.0.2 record correction failed in
`check-release-binary-contract` and its mutation suite for exactly that reason.
W16 must first add a red mutation for the published tag/SHA/run/asset facts,
then update the parent spec, records, checker, and existing mutation inventory
together. The spec-only commit deliberately leaves the checker-bound old text
unchanged rather than weakening the gate or committing a checker repair from
the coordinating session.

## Version and prerelease contract

`project(vllm_cpp VERSION 0.0.3)` remains a numeric CMake version. The release
version is the SemVer prerelease `0.0.3-pre.1`. It is one authenticated value
through the entire release graph:

1. the tag is exactly `v0.0.3-pre.1`;
2. the plan records `version=0.0.3-pre.1`, `project_version=0.0.3`, and
   `prerelease=true`;
3. every archive, checksum, provenance sidecar, embedded manifest, `VERSION`,
   handoff, and generated index uses `0.0.3-pre.1`;
4. publication passes GitHub's prerelease flag and the post-publish audit
   requires `isDraft=false` and `isPrerelease=true`; and
5. a stable tag, a suffix that does not match the declared release version, or
   a prerelease plan published without the prerelease flag fails before any
   release is created.

Manual dry runs consume the same committed release-version declaration as the
tag path. They cannot publish. The implementation may store the declaration in
the existing CMake/release configuration, but it must have one authoritative
value and a mutation test that makes any duplicate disagree.

The prerelease qualifier is not a title-only label. A release whose assets are
named `0.0.3` while its tag is `v0.0.3-pre.1`, or whose GitHub state is not a
prerelease, is a failed gate.

## Toolchain and ABI

The Windows host tuple is native x86_64 MSVC/UCRT on the pinned
`windows-2022` GitHub-hosted image. `windows-latest` is forbidden because its
Visual Studio generation moves independently of this repository. The build
uses the Visual Studio 2022 generator and Release configuration.

The server links the project core statically and uses the static MSVC runtime
(`/MT`). The archive may depend only on documented Windows system DLLs. The
Vulkan bundle dynamically discovers `vulkan-1.dll` from the host; the loader,
ICD, and device driver are external and are never copied into the archive.
`ffmpeg` remains an external executable exactly as on Linux/macOS.

The CPU artifact is one adaptive executable, not an AVX2-only binary and not a
separate binary per tier. Its x86_64 baseline is portable/SSE2. Existing F16C,
AVX2, and AVX-512 kernels remain individually compiled and selected by the same
CPUID plus XCR0 policy already used on Linux. MSVC's `/arch` options are scoped
to the corresponding translation units; no global `/arch:AVX2`,
`/arch:AVX512`, or host-native flag may contaminate the baseline.

## Dependencies

Build dependencies are the pinned `windows-2022` image, Visual Studio 2022
MSVC/UCRT, CMake, Python, Ninja where the chosen generator requires it, and the
project's existing vendored/source dependencies. The Vulkan artifact uses the
repository's existing Vulkan headers and links/discovers the host's
`vulkan-1.dll`; it does not download or bundle a loader, ICD, or driver. ffmpeg
and model weights remain external. Adding a package manager, installer runtime,
MinGW/MSYS runtime, proprietary SDK, or new release service requires a separate
decision.

## Port map and design

Windows support is implemented as narrow platform adapters rather than a
second server or model-loader implementation.

### Read-only model mappings

GGUF and safetensors use one shared RAII read-only mapping abstraction. The
POSIX implementation preserves `open`/`fstat`/`mmap`/`munmap`; Windows uses
`CreateFileW`, `CreateFileMappingW`, `MapViewOfFile`, `UnmapViewOfFile`, and
`CloseHandle`. The abstraction owns both file and mapping handles and supports
the existing borrowed-span keep-alive semantics. Unicode paths enter through
`std::filesystem::path`; no lossy ANSI conversion is accepted.

Tests use real temporary GGUF and safetensors files and prove parse parity,
borrow lifetime after the reader moves/dies, deterministic cleanup, empty-file
failure, and a non-ASCII path. The direct-upload lifetime contract remains
unchanged.

The structural reference is llama.cpp
`src/llama-mmap.cpp:520-590` at `237ad9b961f009ae19ac29dbce4cd0c1251f94b3`.
This is platform substrate only; vLLM remains the behavioral oracle for model
loading and serving.

### Process launch and shutdown

The existing argv-based ffmpeg boundary remains argv-based. Windows uses
`CreateProcessW` with one tested Windows-command-line quoting routine, waits for
the child, and returns the actual exit code. It never invokes `cmd.exe` and
never concatenates untrusted shell text.

The server installs a Win32 console control handler that requests the same
thread-safe server stop used by the POSIX self-pipe path. CTRL_C_EVENT and
CTRL_BREAK_EVENT must leave the extracted server with exit code 0 within the
gate timeout. POSIX signal behavior remains byte-for-byte unchanged.

### Networking and filesystem helpers

cpp-httplib supplies the HTTP transport's Windows socket layer. Project-owned
LMCache networking uses a small socket owner that initializes Winsock once,
uses `SOCKET`/`INVALID_SOCKET`, reports `WSAGetLastError`, and closes with
`closesocket`; the POSIX implementation stays unchanged. Focused loopback tests
exercise connect, read/write, peer close, refusal, and cleanup.

KV filesystem offload keeps its exact atomic-file contract. Windows uses
Win32 or CRT primitives that preserve exclusive temporary creation, complete
positional reads/writes, flush-before-publish where the existing contract
requires it, and atomic same-volume replacement. It is not silently compiled
out of the Windows server.

### Vulkan loading

The existing Vulkan dispatch table remains shared. Only the library handle
adapter changes: Windows tries `vulkan-1.dll` with `LoadLibraryW`, resolves
`vkGetInstanceProcAddr` with `GetProcAddress`, and releases with `FreeLibrary`.
A scratch fake DLL mutation proves the loader succeeds only when the mandatory
entry point exists. A host without Vulkan must make backend discovery return
unavailable without breaking CPU `--help` or server startup.

## Build-system contract

Compiler and linker options are compiler-aware:

- MSVC receives `/fp:strict` or the narrowest flag that preserves the existing
  no-contraction numerical contract; GCC/Clang keep `-ffp-contract=off`;
- warnings are `/W4 /WX` for MSVC and retain the existing flags elsewhere;
- visibility scripts, `--whole-archive`, `dl`, pthread, sanitizer flags, and
  literal-static Linux options never reach MSVC;
- Win32 system libraries are linked only by the components that need them;
- the installed executable is `bin/vllm-server.exe`; and
- existing Linux/macOS target names, install layouts, and archives are
  unchanged.

The first native configure/build is discovery evidence, not a license to omit
failing source files. Every source reachable from the shipped server either
builds and works on Windows or receives an explicit, user-visible refusal for
a genuinely unavailable external runtime. No CLI flag is silently accepted as
a no-op to make the build pass.

## Archive and manifest contract

Archive format is explicit per matrix tuple. Existing tuples declare
`tar.gz`; Windows declares `zip`. No code infers format from an artifact-name
prefix. `canonical_archive_name`, handoff inventory, release-index generation,
workflow upload paths, publication, and validators all consume that one matrix
field.

The Windows ZIP is deterministic: sorted normalized relative paths, fixed UTC
timestamps derived from `SOURCE_DATE_EPOCH`, stable compression settings, no
drive letters, no backslashes in member names, no absolute paths, and no
symlinks/reparse points. Extraction rejects traversal before writing. The
allowlisted tree is the same installed-server contract as other platforms:

- `bin/vllm-server.exe`;
- `VERSION` and the v1 release manifest;
- SPDX JSON SBOM, notices, and licenses; and
- declared runtime files only.

The manifest schema gains `host.os=windows`, `host.abi=msvc`, and the pinned
toolset/UCRT versions. CPU tiers retain their exact CPUID/XCR0 requirements.
The Vulkan dependencies name the external loader, ICD, and driver with false
bundled flags. Missing Windows evidence is false with a reason; it is never
derived from a Linux result.

PE validation uses `dumpbin /headers` or `llvm-readobj` to require AMD64 and
`dumpbin /dependents` or an equivalent structured reader to enumerate imports.
It rejects a non-system DLL, a missing import, a build-tree path, a developer
drive path, debug CRTs, MinGW/MSYS runtimes, and an unexpected bundled DLL.
The Vulkan preview may declare `vulkan-1.dll` external without requiring it to
be present for the CPU-only smoke; the validator must not suppress any other
unresolved import.

## Gates and publication flow

Two explicit read-only build jobs are added to `.github/workflows/release.yml`:

1. `cpu_windows` builds, tests, stages, ZIPs, extracts, validates, and uploads
   the exact CPU archive/checksum/provenance triplet.
2. `vulkan_windows` performs the corresponding Vulkan preview path and keeps
   runtime evidence false unless an extracted-archive Vulkan probe actually
   runs against an ICD.

Both use `windows-2022`, have no write token or OIDC, and upload immutable
SHA-bound artifact names. The aggregate, verify, attest, and publish jobs add
the two exact artifact IDs; no wildcard or artifact discovery is introduced.
The same structural checker that binds the existing eight jobs binds all ten.

The release sequence is:

1. PR CI proves the Windows focused tests and all existing release mutation
   suites.
2. A manual workflow dispatch on the exact merged SHA builds and verifies all
   ten tuples while publish, attest, and GitHub release creation stay skipped.
3. Only after that dry run is green, push the exact merged SHA as
   `v0.0.3-pre.1` through a pre-push full gate.
4. The tag workflow rebuilds all ten tuples, verifies, attests every archive
   digest, and publishes exactly 32 assets: ten archives, ten checksums, ten
   provenance sidecars, `release-index.json`, and `RELEASE_INDEX.md`.
5. Post-publication audit checks tag SHA, job conclusions, unique asset names,
   API digests, checksum/index agreement, one GitHub attestation per archive,
   `isDraft=false`, and `isPrerelease=true`.

The GitHub release is pre-alpha even where an inherited tuple's evidence-driven
channel remains stable. The release index states both levels: overall
`prerelease=true`; per-artifact channels remain exact. Windows stays preview in
this first release.

## Tests to port and adapt

Pinned vLLM has no Windows release or Win32 substrate suite to port. Its
applicable behavioral workload remains the identical server health/version and
model-serving contract already owned by the parent release spec. Existing
eight-tuple release-plan, archive, handoff, index, workflow, and publication
tests are extended without dropping parameters or mutations. Windows-specific
mapping, process, shutdown, socket, filesystem, loader, ZIP, and PE cases are
new local tests against the OS contract; each is recorded as written from
scratch rather than falsely attributed upstream.

## Red-first tests and mutations

Implementation begins with failing tests for each boundary:

1. release matrix accepts only an explicit `tar.gz` or `zip` per tuple and
   expects exactly ten required IDs;
2. canonical naming produces `.zip` for Windows and `.tar.gz` for existing
   tuples, with no version/extension aliases;
3. `v0.0.3-pre.1` produces a prerelease plan and any tag/version/state mismatch
   fails;
4. publication of a prerelease without GitHub's prerelease flag is rejected by
   the workflow checker;
5. deterministic ZIP creation and traversal/symlink mutations;
6. PE architecture/import/build-path/debug-runtime mutations;
7. mapped-file lifetime, Unicode path, and failure cleanup;
8. CreateProcess argv quoting and exit propagation;
9. console stop, Winsock loopback, KV file IO, and Vulkan loader mutations;
10. MSVC baseline/AVX2 forced-tier execution and an unsupported forced-tier
    refusal; and
11. workflow mutations deleting either Windows job, weakening permissions,
    changing runner/toolchain, omitting one exact handoff, changing the asset
    count, or publishing the tag as stable.

Focused gates run on Windows. Cross-platform Python/checker tests and the full
repository preflight run on Linux. The existing eight-tuple tests are
regressions, not inferred green from the two new jobs.

## Work breakdown

| Work | Deliverable | Exit gate |
|---|---|---|
| W14 | Win32 portability substrate and native MSVC CPU build | native Release `/W4 /WX`; model mapping, process, shutdown, socket, filesystem, focused CPU tests green; extracted CPU server smoke; portable/SSE2 and AVX2 executed |
| W15 | explicit ZIP/PE packaging and Windows Vulkan build | deterministic extracted ZIP validation; PE mutations red; Vulkan loader mutation green; exact CPU and Vulkan triplets produced |
| W16 | ten-tuple/prerelease CI, indexes, docs, and records | release mutation suites, full preflight, fresh immutable review, operator gate, ten-tuple dry run, then developer-authorized `v0.0.3-pre.1` publication and 32-asset audit |

W14, W15, and W16 land in one PR, with the spec commit preceding every
implementation commit. Each work unit keeps its own red/green evidence and is
reviewable independently inside that PR.

## Documentation and record updates

The implementation PR updates:

- `.agents/engine-matrix.md` and `.agents/roadmap_v1.md` with the Windows row,
  issue, lifecycle, tests, and evidence;
- `.agents/NOW.md` with the live release position;
- `docs/STATUS.md` with v0.0.2 as published and the Windows pre-alpha state;
- `docs/BENCHMARKS.md` with explicit pending Windows runtime/performance axes;
- `docs/FEATURES.md` with the ten-tuple prerelease only after publication;
- `docs/USAGE.md` with PowerShell download, checksum, extraction, dependency,
  and launch commands; and
- `README.md` only if the download becomes a headline/quick-start change.

No document claims the Windows artifacts exist before the tag workflow has
published and the API audit has passed.

## Risks and stop conditions

- Do not switch to MinGW, WSL, or cross-only artifacts to obtain a green lane.
  A required MSVC capability gap returns for a design decision.
- Do not label Windows CPU stable without its matching runtime, correctness,
  and ISA execution gates. The pre-alpha release may publish it as preview.
- Do not bundle `vulkan-1.dll`, an ICD, GPU driver, ffmpeg, Python, weights, a
  compiler, or source/build directories.
- Do not weaken the existing archive, dependency, handoff, or publication
  validators. Extend them with platform-specific evidence.
- Do not tag if any one of the ten dry-run tuples, aggregate handoff, or verify
  job is not green on the exact merged SHA.
- Do not create or mutate a stable `v0.0.3` release in this row. Its only
  publication target is the pre-alpha `v0.0.3-pre.1` prerelease.

## Written-spec self-review

- Placeholder scan: no TBD/TODO or deferred implementation requirement.
- Consistency: two new tuples, ten total triplets, and 32 release assets agree.
- Scope: native Windows x86_64 CPU/Vulkan only; CUDA, arm64, installer, and WSL
  are explicitly excluded.
- Ambiguity: tag, version, archive extension, toolchain, runner, channel,
  prerelease state, and post-publish evidence are exact.

## Outcome

W14 and W15 implement the Win32 portability substrate, native MSVC build,
deterministic ZIP/PE validation, and Vulkan loader/metadata path. W16 adds the
two read-only `windows-2022` jobs, a ten-tuple immutable handoff, and the single
`0.0.3-pre.1` prerelease identity while retaining CMake's numeric `0.0.3`
project version. The completed release workflow now triggers an authenticated,
fail-closed remote audit that re-downloads all 32 assets and binds the tag,
source run/jobs, API digests/sizes, checksums, provenance, indexes, and exactly
one GitHub-verified attestation per archive. Linux-hosted portability, release mutation suites, and local
operator gates are the accepted local evidence. Native MSVC `/W4 /WX`,
extracted Windows runtime/ISA execution, merged-SHA ten-tuple dry run,
`v0.0.3-pre.1` publication, attestations, and the exact 32-asset API audit are
not inferred from Linux and remain required hosted gates. Windows stays
`preview`; no stable `v0.0.3` release is authorized by this row.

PR #446's first local CI repair added the required device-seam annotation to
the server entrypoint without updating `docs/USAGE.md` in the same commit, so
the per-commit documentation checkpoint correctly rejected it ([#448](https://github.com/mudler/vllm.cpp/issues/448)).
The replacement commit keeps the annotation and native Windows PR gates intact
and adds the narrow usage projection atomically; neither checker is weakened.

The first local #447 repair also changed the public `vllm-server-archive`
target without its usage projection, which the same per-commit checkpoint
correctly rejected ([#450](https://github.com/mudler/vllm.cpp/issues/450)). The
replacement documents the numeric project-version tarball without conflating
it with the prerelease workflow's matrix-selected archive names and formats.

The exact PR range starts before `check-windows-portability.py` and
`check-windows-release-state.py` existed, so their otherwise recognized
mutation suites had no BASE checkers to falsify and the fail-closed PR-size
gate rejected PR #446 ([#453](https://github.com/mudler/vllm.cpp/issues/453)).
The repair registers the same closed disabled creation form used by other new
governance checkers. Both real suites stay unchanged and green at HEAD, then
fail with that stub substituted for each absent BASE implementation; path
classification and checker-evidence requirements remain unchanged.

The same full range proved that the `ENGINE_ROWS` ratchet change had no semantic
edit in its recognized agent-record suite ([#455](https://github.com/mudler/vllm.cpp/issues/455));
the added test binds the unique `ENG-RELEASE-WINDOWS` engine row to that suite.
It also proved the evidence subprocess's fixed `/bin:/usr/bin` PATH hid a valid
host Ninja installed elsewhere, falsely reddening the HEAD portability suite
([#456](https://github.com/mudler/vllm.cpp/issues/456)). The repair copies only
the explicitly required `ninja` executable into an isolated tool directory;
the ambient directory and sibling executables remain unreachable.

## Now

`ACTIVE`; W14-W16 implementation is assembled for one PR. Next: fresh review,
operator gate, native hosted PR CI, merge, and a non-publishing ten-tuple dry
run on the exact merged SHA. Only after those gates may the authorized
`v0.0.3-pre.1` prerelease be tagged and audited for exactly 32 assets.
