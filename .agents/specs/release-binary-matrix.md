# Downloadable server binary release matrix

Status: accepted spike for `ENG-RELEASE-BINARIES`. This document defines the
release contract and helper-sized implementation order. It does not claim that
an archive, release workflow, or runtime result exists.

Pins: vLLM parity source `555967922`; vllm.cpp spike baseline `f13c49ee`;
request [#117](https://github.com/mudler/vllm.cpp/issues/117); claim
`CLAIM-ENG-RELEASE-BINARIES-SPIKE` in draft PR
[#129](https://github.com/mudler/vllm.cpp/pull/129).

## Scope and product contract

The deliverable is a downloadable, backend-specific `vllm-server` bundle. The
server links the vllm.cpp core statically, while unavoidable platform runtimes
remain explicit dependencies. A release bundle is an installed staging tree,
not a copy from a build directory. It contains at least:

- `bin/vllm-server`, linked to the static `vllm` core;
- `VERSION`, a machine-readable release manifest, SHA256 checksum, SPDX JSON
  SBOM, build-provenance attestation, and third-party license notices;
- runtime files that the selected lane is licensed and designed to redistribute
  (for example MLX's dylib/metallib in the MLX preview lane); and
- no model weights, tokenizer assets, Python, PyTorch, Triton runtime, compiler,
  or source/build directory.

The bundle name includes project version, backend, target, OS, host ABI, and
archive format. CPU architecture, CUDA SM, libc floor, macOS deployment target,
and optional provider are never implicit. A future server install component and
package target must stage this exact tree. The current CMake installs
`libvllm.{a,so}` and `vllm.h` but does not install the server
(`CMakeLists.txt:1712-1783`; `examples/CMakeLists.txt:54-64`), so the install and
package targets are implementation work, not present-tense capability.

Out of scope for this spike: implementing CMake or CI, publishing a release,
bundling models, promising cross-libc portability, and widening any backend,
model, quantization, or kernel support claim.

## Upstream chain

Pinned vLLM's structural release reference is
`.buildkite/release-pipeline.yaml:1-18,34-170`; its CPU release-image dependency
boundary is `docker/Dockerfile.cpu:262-290`. Those lanes establish that release
construction, validation, and publication are separate concerns. vllm.cpp does
not copy Python wheels or containers: it applies the same separation to native
installed server archives while vLLM remains the runtime-behavior oracle.

## Our baseline

The current server is the CMake target `server`, linked to the static `vllm`
target and gated only by a help smoke (`examples/CMakeLists.txt:54-64`). The
library install rules package `libvllm.a`, the shared C ABI library, and
`include/vllm.h` (`CMakeLists.txt:1712-1783`). There is no server install rule,
archive layout, release manifest, staged dependency audit, provenance, or
publish workflow. Existing CUDA AOT trees are exactly the six directories named
below; ROCm's opt-in skeleton has never compiled on AMD hardware.

## Evidence classes and publication policy

The channel is evidence-driven, not backend-driven:

| Channel | Required evidence | User-facing promise |
|---|---|---|
| `stable` | clean build, staged-archive validation, matching-hardware runtime smoke, representative correctness gate, and release-tag rerun all pass for the exact target tuple | supported downloadable bundle for the named tuple |
| `preview` | clean build and staged-archive validation pass; runtime, correctness, or performance evidence is absent or incomplete | build-only/testing-welcome artifact; no runtime-support or performance claim |
| `blocked` | the backend does not yet compile or lacks the minimum packaging/runtime contract | no artifact is published |

Build success never sets runtime evidence true. Runtime smoke never sets
correctness or performance evidence true. The release manifest carries
independent booleans and evidence URLs for `build_verified`,
`archive_smoke_verified`, `dependency_audit_verified`, `runtime_verified`,
`correctness_verified`, and `performance_verified`. It also records the exact
commit, clean-tree status, compiler/toolchain, CMake cache options, target
architecture, host ABI, dependency versions, and test commands. Stable
publication fails closed unless every stable-required boolean is true;
preview publication preserves false values rather than deriving or omitting
them.

Performance evidence is informational in an archive manifest until the normal
same-box, same-workload vLLM/competitor gate has run. This release row cannot
turn a build result into a throughput claim.

## Release matrix

The initial matrix is deliberately hybrid. Runtime-gated tuples may graduate
to stable; build-only tuples remain downloadable previews so users can test
hardware the project does not own.

| Artifact tuple | Initial channel | Backend flags and evidence boundary |
|---|---|---|
| `linux-x86_64-glibc-cpu` | stable after matching x86_64 runtime gate | CPU core; CUDA, Metal, MLX, Vulkan, and HIP explicitly off; record glibc and libstdc++ floors |
| `linux-aarch64-glibc-cpu` | stable after matching arm64 runtime gate | same CPU contract, independent arm64 runtime evidence; never inferred from x86_64 |
| `macos-arm64-metal` | stable after M-series runtime gate | native Metal, MLX off; record deployment target and required system frameworks |
| `macos-arm64-metal-mlx` | preview until its exact bundled MLX tuple is runtime/correctness-gated | Metal plus opt-in MLX provider and redistribution/license audit; record MLX dylib and metallib versions |
| `linux-x86_64-glibc-vulkan` | preview | Vulkan explicitly on; loader/device/driver remain external; only the Vulkan-supported model/quant surface is declared |
| `linux-x86_64-musl-cpu-static` | experimental preview | literal-static feasibility lane; CPU only; see the static boundary below |
| ROCm/HIP | blocked | `VLLM_CPP_HIP` skeleton has never compiled on AMD hardware; no archive until compile, staged smoke, and matching-hardware gates exist |

`AUTO` is forbidden in release presets. Every backend/provider is selected
explicitly from the canonical CMake surface: `VLLM_CPP_CUDA`,
`VLLM_CPP_CUDA_ARCHITECTURES`, `VLLM_CPP_METAL`, `VLLM_CPP_MLX`, `MLX_ROOT`,
`VLLM_CPP_VULKAN`, `VLLM_CPP_HIP`, `VLLM_CPP_HIP_ARCHITECTURES`,
`VLLM_CPP_SERVER`, `VLLM_CPP_BUILD_EXAMPLES`, `VLLM_CPP_BUILD_TESTS`, and
`VLLM_CPP_TRITON` (`CMakeLists.txt:24-120`; `cmake/TritonAOT.cmake:52-87`).
The manifest records resolved feature-table output; it does not say “all CUDA
features” or “all models.”

### CUDA artifacts: one SM and one host ABI each

There is no generic CUDA archive and no cross-SM runtime claim. Every supported
release target is a separate artifact:

| CUDA SM | Host ABI | AOT | Initial channel and honest boundary |
|---|---|---:|---|
| `80` | Linux x86_64, glibc | on | preview; derived/build-verified, no matching runtime board |
| `86` | Linux x86_64, glibc | on | preview; same-major derived/build evidence, no matching runtime board |
| `87` | Linux aarch64, glibc/Jetson | off | preview; portable synchronous path runtime-verified, default async path remains a known bug |
| `89` | Linux x86_64, glibc | on | preview; derived/build-verified, no matching runtime board |
| `90a` | Linux x86_64, glibc | on | preview; derived/build-verified, no matching runtime board |
| `100a` | Linux x86_64, glibc | on | preview; derived/build-verified, no matching runtime board |
| `103a` | Linux x86_64, glibc | off | preview; portable build-only target |
| `110` | Linux aarch64, glibc/Thor | off | preview until the archive itself repeats the existing portable runtime proof |
| `120a` | Linux x86_64, glibc | off | preview; build-supported, no matching runtime board |
| `121a` | Linux aarch64, glibc/GB10 | on | stable candidate; must rerun archive smoke, both gate-model correctness, and release performance on GB10 |

The AOT split is exact: `80`, `86`, `89`, `90a`, `100a`, and `121a` consume
the complete vendored trees under `src/vt/cuda/triton_aot_vendored/`;
`87`, `103a`, `110`, and `120a` set `VLLM_CPP_TRITON=OFF`. No lane fabricates
or borrows another SM's cubin. AOT artifacts are single-architecture, so a
fat CUDA build with AOT is forbidden by both this matrix and
`cmake/TritonAOT.cmake:94-126`. This release program does not publish a
fat+AOT archive.

The host ABI is part of the tuple because a CUDA cubin does not make its ELF
host executable portable. Preview labels remain even when SASS exists if the
exact archive has not run on matching silicon. The matrix makes no blanket
claim about fast-path coverage: each manifest must preserve the resolved
`VT_CUDA_FEATURE_TABLE` cells from `cmake/CudaArchFeatures.cmake:213-357`.

## Static and external-runtime boundary

The normal bundles are **static-core**, not “one literal static executable.”
`vllm-server` contains the static project core but may dynamically depend on the
host C/C++ runtime, pthreads, platform frameworks, or a selected accelerator
runtime. Those dependencies must be enumerated and audited from the staged
binary.

The one literal-static experiment is
`linux-x86_64-musl-cpu-static`. It is CPU-only and passes only when `file`
identifies a static executable, `ldd` reports no dynamic interpreter, the server
help and loopback health smoke pass in a minimal container, and DNS/thread/file
loading behavior is exercised. It stays experimental preview even when green;
the result decides whether a second arm64 musl lane is warranted. The archive
must not silently disable server functionality to obtain a static link.

Accelerator drivers are honest external boundaries. NVIDIA's kernel driver and
CUDA driver ABI, the Vulkan loader/ICD and device driver, macOS Metal system
frameworks, and the ROCm kernel/user runtime cannot be made portable by
statically linking the vllm.cpp core. The archive manifest names the minimum
tested driver/runtime; it never claims to bundle a GPU driver. MLX is an
opt-in preview exception whose redistributable dylib/metallib may be carried
only with its exact license and version.

`ffmpeg` is also external. The server's video path defaults to the `ffmpeg`
executable and spawns it from the example boundary
(`examples/server/main.cpp:215,349-350,741-758,839,1090-1093`). Bundles do not
silently vendor it. The manifest and README say that text serving needs no
ffmpeg and video generation requires a compatible executable on `PATH` or an
explicit `--video-ffmpeg` path. Models, tokenizer data, certificates, and GPU
drivers are runtime inputs, not archive payloads.

## Gates: staged archive and supply chain

Every CI lane builds, installs into an empty staging prefix, creates the archive,
extracts it into a second empty directory, and validates only that extracted
tree. A build-tree smoke is not release evidence.

Required gates:

1. **Package contents:** exact allowlist, executable bit, no absolute build
   paths, no source/object files, no credentials, and version output matching
   the tag, commit, manifest, and C ABI version.
2. **Server smoke:** `vllm-server --help`; bind loopback on an ephemeral port;
   `/health`, `/version`, and clean shutdown; then a small representative model
   request on lanes with matching runtime hardware.
3. **Dependency audit:** Linux `readelf` plus `ldd`/`lddtree`, macOS `otool`, and
   platform equivalents reject an undeclared shared object, build-directory
   RPATH/RUNPATH, absolute developer path, or missing library. ELF RPATH must be
   absent or relative to the extracted bundle; Mach-O install names must use
   system paths, `@rpath`, or `@loader_path` as declared.
4. **Architecture audit:** `file`/ELF/Mach-O headers match the host ABI; CUDA
   `cuobjdump` shows only the named SM; an AOT-on artifact's manifest and cubins
   match that same SM; AOT-off artifacts contain no Triton AOT payload.
5. **Correctness:** CPU unit/conformance tests before packaging; after
   extraction, representative endpoint and model checks. Stable `sm_121a` owes
   both project gate models and the normal oracle comparison. Other stable
   lanes owe the row's available backend/model correctness contract.
6. **Supply chain:** archive SHA256, SPDX JSON SBOM, source/dependency/license
   inventory, third-party notices, immutable build provenance, and a `VERSION`
   record containing tag, commit, clean-tree bit, compiler, backend, target,
   host ABI, and C ABI version. The checksum and provenance refer to the final
   archive bytes, not the staging directory.

The release manifest schema and its checker are versioned together. Missing
evidence is `false` with a reason; command failure cannot collapse into “not
applicable.”

## Port map

| Responsibility | Local destination | Contract |
|---|---|---|
| server install/package component | top-level and `examples/CMakeLists.txt` | stage the existing static-core server under its canonical output name without changing server behavior |
| release tuple/preset | focused release CMake presets or matrix data | explicit backend/provider/host ABI; no `AUTO`, wildcard SM, or blanket feature claim |
| manifest and supply-chain metadata | release scripts plus a versioned schema | independent evidence values, final-archive SHA256, SPDX SBOM, provenance, version and licenses |
| staged archive validator | release checker tests/scripts | validate extracted bytes, dependencies, RPATH/install names, host architecture and AOT SM |
| dry-run/tag automation | release workflow | build, verify, attest and publish as isolated least-privilege stages |
| release index | generated public release notes/index | derive channel and limitations from verified manifests, never handwritten assumptions |

## Tests to port

The pinned upstream release pipeline is the executable structural spec; its
separate build/test/publish stages are mirrored by W4 rather than porting a
Python runtime test. Existing local executable specs remain binding:

- `test_server_help` (`examples/CMakeLists.txt:59-63`) runs from the extracted
  archive, not only the build tree;
- endpoint conformance under `tests/vllm/entrypoints/openai/` supplies the
  server protocol smoke before a lane can become stable;
- each backend row's existing correctness suite and matching-hardware model
  gate supplies runtime evidence; cross-builds record runtime false; and
- new manifest/archive fixtures mutate every required evidence field, RPATH,
  dependency allowlist, architecture and AOT association to prove the checker
  fails for the named reason.

## Least-privilege CI and release flow

The same build definition serves pull-request dry runs and tags, but authority
is separated by stage:

1. **Plan:** read-only checkout computes the matrix and validates the tag/version
   relationship. A pull request or manual dry run cannot create a release.
2. **Build:** per-tuple jobs have `contents: read`, no token write scope, no OIDC,
   and upload temporary workflow artifacts only. Cross builds produce build
   evidence, never runtime evidence.
3. **Verify:** fresh jobs download and extract archives, run the staged checks,
   and emit the independent evidence manifest. Hardware jobs receive only the
   exact tuple assigned to that runner.
4. **Attest:** only the provenance job receives `id-token: write`; it signs the
   verified archive digest, not arbitrary workspace content.
5. **Publish:** a protected tag/environment job alone receives
   `contents: write`. It downloads verified immutable archives, checks their
   digests and evidence channel, and attaches them to the matching release.

Fork pull requests never receive release secrets. Tag names are untrusted input
until the version gate passes. Artifact names are allowlisted, publish uses no
wildcards, and a failed lane cannot be replaced by an older workflow artifact.

## Dependencies

- the existing server and library install boundaries;
- canonical CMake backend flags and CUDA feature-table resolution;
- complete per-SM Triton AOT trees where AOT is enabled;
- matching runtime hosts for any stable tuple and the normal GPU contention
  protocol for correctness/performance gates;
- redistribution-compatible licenses for every bundled runtime file;
- platform dependency-inspection tools plus an SPDX SBOM/provenance generator;
  and
- protected release environments for attestation and publication authority.

## Work breakdown: helper-sized implementation plan

Each work unit is a separate claim with its own red-first checker change and
fresh review. No unit advances this row beyond `ACTIVE` until its gates pass;
the present checkpoint stays `SPIKE`.

| Work | Deliverable | Exit gate |
|---|---|---|
| W1 | canonical `vllm-server` output name, install component, and staging/package target for the existing static-core server | install into empty prefix; extracted help smoke; existing library install unchanged |
| W2 | versioned release-manifest generator and schema with independent evidence booleans | fixture tests distinguish absent, false, failed, and true evidence; mutation of each required field is red |
| W3 | staged archive validator: content allowlist, dependency/RPATH audit, architecture/AOT audit, SHA256, VERSION, licenses, SPDX SBOM | Linux fixture/archive tests red-first; no build-tree paths accepted |
| W4 | least-privilege dry-run/tag workflow skeleton, immutable artifact handoff, provenance and protected publish stages | permissions checker plus dry run proves no release is created and publish cannot consume unverified bytes |
| W5 | Linux glibc CPU x86_64 and arm64 bundles plus experimental x86_64 musl literal-static lane | both glibc tuples runtime-smoked on matching hosts before stable; musl static/loopback checks remain preview |
| W6 | macOS arm64 native-Metal bundle | extracted archive model smoke on M-series hardware; dependency/install-name audit; MLX absent |
| W7 | macOS arm64 MLX preview bundle | exact MLX dylib/metallib/license inventory and extracted runtime/correctness smoke; remains preview until all stable gates exist |
| W8 | Linux x86_64 Vulkan preview bundle | extracted archive runtime smoke on a Vulkan runner; loader/ICD stay external and declared |
| W9a | CUDA previews for `80`, `86`, `87`, `89` | four separate host/SM artifacts; exact AOT split; build/archive evidence never upgrades missing runtime booleans |
| W9b | CUDA previews for `90a`, `100a`, `103a`, `110` | same contract; `110` repeats matching-hardware archive smoke before any channel change |
| W9c | CUDA `120a` preview and `121a` stable candidate | separate artifacts; `121a` extracted archive reruns both gate models and release-performance gate on GB10 |
| W10 | release index/docs and retention policy generated from manifests | every link, checksum, channel, host ABI, driver boundary, and known limitation matches published bytes |

ROCm remains blocked outside these work units until its backend row first
compiles on AMD hardware. A new lane is added by changing this matrix and its
tests before workflow expansion, never by a wildcard build.

## Risks and decisions

- “Static” without the `static-core` qualifier is rejected for normal GPU and
  platform bundles; only the musl CPU experiment may say literal-static.
- Stable is a property of an exact artifact tuple and evidence set, not of a
  backend family. A later toolchain or dependency change reruns the gates.
- Cross-compilation can prove bytes and architecture, not execution. Preview is
  the honest publication channel for community hardware coverage.
- CUDA fast-path availability differs by SM. The feature-table output and AOT
  state are manifest data; archive names do not imply feature parity.
- ffmpeg and accelerator drivers are external operational dependencies. Their
  absence must fail the relevant feature actionably, not corrupt a general
  server smoke.
- Reproducibility means a recorded clean recipe plus immutable provenance first;
  byte-for-byte rebuild reproducibility is a separate evidence field until
  demonstrated.

## Spike verdict

The release program is feasible as backend-specific static-core bundles with a
hybrid stable/preview channel. Literal-static scope is limited to the
experimental musl CPU lane. ROCm is blocked. The first implementation slice is
W1, followed by the manifest/validator substrate before any publishing workflow.
`ENG-RELEASE-BINARIES` is `SPIKE`, not `READY`, `ACTIVE`, or `DONE`.
