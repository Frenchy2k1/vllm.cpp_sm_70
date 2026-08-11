# Release sanitizer and handoff blockers

Status: committed repair spec for the release follow-up on
`ENG-RELEASE-BINARIES`. Issues:
[#301](https://github.com/mudler/vllm.cpp/issues/301),
[#321](https://github.com/mudler/vllm.cpp/issues/321), and
[#322](https://github.com/mudler/vllm.cpp/issues/322).

Baseline: `origin/main` `60e71a0e55ad6e0f0dbaeb48630a83bb088be26d`.
Failing release PR head: `d1f9cf7730fbbbaecabeea63df14d38cece28a0b`.
The release version change did not introduce any of these failures: the same
code is on main, and each failure is independently grounded below.

## Scope and approved invariants

This is one integration follow-up because all three findings block the same
`v0.0.2` release gate. It contains three independently testable repairs:

1. CPU reads of borrowed safetensors bytes have defined unaligned semantics.
   The safetensors payload offset is byte-addressed and is not assumed to meet
   C++ alignment for `uint16_t` or `float`. Reads use `memcpy` through one small
   trivially-copyable load primitive. The mmap remains borrowed: the repair
   must not reintroduce a whole-weight host copy or weaken direct-upload
   accounting.
2. Attaching a non-owning `PrometheusStatLogger` to `AsyncLLM` has an explicit
   synchronous detach barrier. After `set_stat_logger(nullptr)` returns, no
   output-handler iteration can call the old logger. Production lifetime order
   keeps the logger alive until `LoadedEngine` destroys `AsyncLLM`; that
   destruction runs `shutdown()` and joins the output thread before the logger
   is destroyed. Sanitizer suppressions are forbidden.
3. Release artifacts use the dedicated transient directory name
   `release-assets` at every workflow stage: `release-assets`,
   `unverified/release-assets`, and `verified/release-assets`. The checker
   rejects any build, verify, attest, or publish consumer reverted to checkout
   `assets`. `merge-multiple: true`, immutable artifact IDs, exact inventory,
   canonical filenames, and fail-closed publication remain unchanged.

Explicitly out of scope: aligning or copying safetensors fixtures instead of
fixing consumers; weakening UBSan/ASan/TSan; allowlisting `favicon.png`;
deleting or renaming repository assets; changing handoff discovery or exact
inventory; changing metric values/catalogs; refactoring unrelated typed tensor
access; publishing or tagging the release.

## Root-cause evidence

### A. Borrowed mmap alignment

ASan/UBSan job
[`93622943433`](https://github.com/mudler/vllm.cpp/actions/runs/31440144513/job/93622943433)
fails six tests. Five are one defect class:

| Test | Failing site | Reduction |
|---|---|---|
| `test_load_direct_upload` | `tests/vllm/test_load_direct_upload.cpp:177` | test-only `reinterpret_cast<const uint16_t*>` read |
| `test_llama_embedding_fold` | `src/vt/cpu/cpu_ops.cpp:33` | shared `LoadF32` BF16 arm |
| `test_openai_api_server` | `src/vt/cpu/cpu_ops.cpp:33` | shared `LoadF32` BF16 arm |
| `test_capi` | `src/vt/cpu/cpu_ops.cpp:33` | shared `LoadF32` BF16 arm |
| `test_laguna_nvfp4_loader` | `src/vllm/model_executor/models/laguna.cpp:1028` | `LagunaEmbed` BF16 gather |

Every reported address ends in an odd byte. `ENG-LOAD-DIRECT-UPLOAD` makes the
weight view the mmap payload verbatim, so typed-pointer dereference creates a
misaligned C++ lvalue even though x86 hardware accepts the instruction. The
five failures therefore reduce to two production read helpers plus one
test-only reader; no fixture alignment assumption is valid.

Alternatives rejected: copying every borrowed tensor (undoes the measured load
and memory win) and requiring an aligned safetensors data offset (not part of
the file contract). The selected byte-copy load preserves both bytes and
borrowing.

### B. Async logger lifetime

The same ASan job reports `heap-use-after-free` in
`PromRegistry::Find` from `PrometheusStatLogger::Record`, called by
`AsyncLLM::RunOutputHandler`. The main test thread freed
`PromRegistry::families_` at `test_llm_engine.cpp:1140`; output thread T13 still
held the raw logger loaded before `OutputProcessor` published the terminal
collector value. TSan job
[`93622943476`](https://github.com/mudler/vllm.cpp/actions/runs/31440144513/job/93622943476)
reports the same read/free race, not an independent registry-lock bug.

The ported async-metrics spec already states the quiescence invariant: a
drained collector does not prove the subsequent logger fold retired; only
`AsyncLLM::shutdown()` joining the output handler does. The raw attach API also
promises that the logger outlives the engine. Two callers violate that order:
the failing test declares its logger after the harness, and `server_main.cpp`
declares `prom_logger` after `loaded`, so reverse destruction frees the logger
first.

Alternatives rejected: adding a sanitizer suppression, serializing only inside
`PromRegistry` (cannot protect an already-destroyed registry), or relying on a
sleep. The selected design makes detach a real barrier and makes production
declaration order satisfy shutdown/join-before-logger-destruction on every
normal return and stack-unwind path.

### C. Release download-root collision

Merged-main dry run
[`31435201833`](https://github.com/mudler/vllm.cpp/actions/runs/31435201833)
built all eight required artifact triplets. Aggregate job
[`93631280081`](https://github.com/mudler/vllm.cpp/actions/runs/31435201833/job/93631280081)
then ran `release_pipeline.py handoff --assets-dir assets`. Because checkout
already owns `assets/favicon.png`, exact inventory correctly rejected the extra
file. The validator is working; the download root is wrong.

Alternatives rejected: allowlisting the favicon, filtering discovered files,
or weakening exactness. A dedicated transient root removes the namespace
collision structurally while keeping the immutable handoff unchanged.

## Implementation design

### Defined loads

Add a narrow `vt` utility for loading a trivially-copyable value from an
arbitrary byte address via `std::memcpy`. `vt::cpu::LoadF32` computes the byte
address from `elem_offset` and uses it for F32/F16/BF16 inputs. `LagunaEmbed`
and Laguna's general BF16 `ReadF32` use the same primitive. The direct-upload
lifetime test reads expected and actual BF16 words without binding a typed
reference to the mmap.

Stores remain typed because their destinations are runtime-owned aligned
buffers; expanding the change to unrelated kernels is not justified by the
observed failures.

### Logger detach and teardown

Protect the attached logger pointer with an `AsyncLLM` mutex. An output-handler
iteration holds that attachment lock from deciding whether stats are needed
through its final `Record`, while still calling `Record` outside
`output_processor_mutex_`. `set_stat_logger` takes the same lock, so detaching
waits for any in-flight record and prevents future use of the old pointer.

The focused regression attaches a logger, completes a request, detaches it,
then destroys it while the engine remains alive; the current atomic-pointer
implementation permits the stale post-collector call and fails under
ASan/TSan. Production declares the logger before `LoadedEngine`, so reverse
destruction invokes `AsyncLLM::~AsyncLLM` -> `shutdown()` -> join before logger
destruction. Existing explicit `shutdown()` tests remain the primary join
contract.

### Release directory contract

Change only workflow transient paths and the checker/tests that pin them. The
Python handoff/index/publish interfaces continue accepting a caller-supplied
directory and keep exact file validation. The workflow checker validates each
required stage-specific path rather than a global word count, and mutations
revert each occurrence independently to prove no stage can drift back to the
tracked checkout directory.

## Tests and RED plan

RED is captured before production changes:

1. Add a focused unaligned BF16 input case to the smallest existing CPU-op
   target and run it plus `test_load_direct_upload` and
   `test_laguna_nvfp4_loader` under address+undefined sanitizers. It must fail
   on the typed production load, not fixture construction.
2. Add the detach-then-destroy AsyncLLM case and run the focused engine target
   under ASan and TSan. It must report the stale logger use on the current
   atomic-pointer implementation.
3. Add checker mutations that change each required `release-assets` stage back
   to `assets`. Before the checker change, the real workflow and/or mutants
   fail the new assertions. Also reproduce the exact checkout collision by
   running handoff against a temporary root containing one valid triplet plus
   `favicon.png`; exact inventory must continue rejecting it.

GREEN focused gates:

```sh
python3 -m unittest tests.scripts.test_release_pipeline
python3 scripts/check-release-workflow.py
cmake -S . -B build-sanitize -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_VULKAN=OFF -DVLLM_CPP_METAL=OFF \
  -DVLLM_CPP_SANITIZER=address,undefined
cmake --build build-sanitize --target test_load_direct_upload \
  test_llama_embedding_fold test_laguna_nvfp4_loader test_llm_engine
ctest --test-dir build-sanitize --output-on-failure \
  -R 'test_load_direct_upload|test_llama_embedding_fold|test_laguna_nvfp4_loader|test_llm_engine'
```

Use the repository's actual sanitizer option spelling discovered from CI/CMake
when configuring. Run a separate TSan build for the detach regression when the
local toolchain supports it. Also run the same focused tests non-sanitized, then
the full `scripts/agent-preflight.sh` and staged gate.

Mutation review must independently replace the unaligned load with a typed
dereference, remove the detach synchronization, and revert every transient
release path one at a time; the corresponding focused gate must fail.

## Risks and stop conditions

- Holding the attachment mutex across one output-processing iteration is
  intentionally narrow: it establishes the documented once-per-iteration
  logger snapshot and detach barrier. `Record` remains outside the output
  processor lock, preserving the existing lock-order decision.
- Byte-copy loads must compile to defined scalar loads without changing BF16
  bits. Tests compare the exact expected BF16 words/conversions.
- `release-assets` is a literal contract, not a discovery convention. Every
  handoff consumer stays explicitly named and wildcard-free except the existing
  attestation subject glob over already verified bytes.
- Stop only for a reproducible toolchain/resource blocker or if evidence shows
  a fourth independent defect. Do not broaden into unrelated tensor access,
  metric catalog work, release publication, or filesystem cleanup.

## Outcome

Implementation completed on `row/ENG-RELEASE-SANITIZER-BLOCKERS`; independent
immutable review and the operator's post-review gate remain pending.

- Arbitrary-address scalar reads now use one `std::memcpy`-based
  `vt::LoadUnaligned` primitive. The observed CPU-op and Laguna sites are
  defined without changing mmap borrowing. Replaying the full direct-upload
  target exposed one additional instance of the same defect in the shared BF16
  transpose fallback; fresh review then found the resident `LagunaGraph::Step`
  gather had retained its typed BF16 read. Both ordinary and resident-graph
  embedding now share one byte-addressed staging seam, directly covered with a
  borrowed BF16 row beginning at an odd byte offset.
- `set_stat_logger(nullptr)` is now a synchronous detach barrier. The output
  handler holds the attachment lock through its final `Record`, while `Record`
  remains outside `output_processor_mutex_`. The server's declaration order
  independently guarantees that `LoadedEngine` shuts down and joins before
  the non-owning logger is destroyed.
- Every release handoff stage now uses `release-assets` (nested as
  `unverified/release-assets` and `verified/release-assets`). Immutable artifact
  IDs, flat extraction, exact inventory, attestation, and publication remain
  unchanged. Fresh mutation review proved that global fragment counts could be
  compensated by changing an unrelated download; the checker now binds each
  exact artifact ID, job, step, and upload to its declared root. A second fresh
  mutation review proved that raw named-step fragments could still be supplied
  by comments or inert shell strings. Consumer validation now parses executable
  commands and continuations in the exact named step, binds a unique command's
  complete option map, and fails closed on malformed or duplicate commands and
  options. A checkout `favicon.png` in the handoff root is still rejected.
- Address+undefined sanitizer replay passed all six executables that failed in
  CI; the complete direct-upload target passed 14 cases / 183 assertions after
  the extra transpose repair. ThreadSanitizer passed the detach regression and
  the complete engine target (15 cases / 295 assertions) using
  `setarch x86_64 -R` to avoid this host's pre-main GCC TSan mapping failure.
  The non-sanitized focused suite passed all four targets. The post-review
  Laguna replay passed 4 cases / 63 assertions under combined address and
  undefined sanitizers, and the release pipeline suite passed all 23 tests,
  including cross-download, inert-shell-text, duplicate-binding, and direct
  `gh release` bypass mutations.
