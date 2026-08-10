# Load-time direct device upload — `ENG-LOAD-DIRECT-UPLOAD`

Row: `ENG-LOAD-DIRECT-UPLOAD` (issue
[#150](https://github.com/mudler/vllm.cpp/issues/150) — "Model load / cold start
time: measure it properly, then cut it"). Claim
`CLAIM-ENG-LOAD-DIRECT-UPLOAD`. Parity pin: vLLM `555967922` (0.26.0.dev0).

## Scope

**In.** Remove the intermediate owned host buffer from the safetensors load for
every weight the device consumes VERBATIM, so the byte moves from the file
mapping into the device allocation exactly once instead of twice. The mechanism
is a borrow (`OwnedBytes::Borrow`) whose keep-alive is the shard's mmap, plus a
post-upload adoption/page release so no second resident copy survives the load.
Also in scope: the measurement half of #150 — per-phase load timing and
host-copy / borrowed / device-upload byte counters behind `VT_LOAD_STATS`.

**Out.** Moving the upload itself into the load phase (design shape (b), see
Risks); merged/transposed/dequantized tensors, which are not verbatim and keep
their copy; the GGUF reader, which already borrows its own mapping; the
`LOAD-SAFETENSORS-DIRECT-DENSE` discrete-CUDA staging path, untouched.

**This is a LOAD-TIME lever, not a memory lever.** Peak resident memory is
~the model size either way. `#203`/`#204` already fixed the real memory defect
(the host copy was RETAINED alongside the device copy: 100.759 -> 53.413 GiB
VmRSS on the 27B). This row builds on that and does not re-litigate it.

## The gap

Loading a safetensors checkpoint moves the weights TWICE.

1. `LoadShards` mmaps every shard (`SafetensorsFile::Open`,
   `src/vllm/model_executor/model_loader/safetensors_reader.cpp:46`) and each
   loader `memcpy`s its tensors out of that mapping into an **owned anonymous**
   `OwnedBytes` buffer. The consumed source range is then dropped from residency
   (`MaybeReleaseSourcePages`, the `LOAD-SAFETENSORS` windowed release), so the
   PEAK is already bounded — but the COPY still happened.
2. Lazily, at first forward use, `ResidentWeight`
   (`include/vllm/model_executor/models/dense_attn_block.h:178`) allocates a
   device buffer, copies host -> device, and calls `AdoptDeviceBytesAsHost`,
   which on a host-addressable backend re-points the host handle at the device
   allocation and frees the anonymous copy.

The anonymous buffer exists only to be copied out of and thrown away. Every byte
of the model is written once into anonymous memory — which also costs the kernel
a page fault and a zero-fill per page — and read back once, for nothing.

## Upstream chain

vLLM never materializes an intermediate host mirror either: it streams one
tensor at a time out of `safe_open` and copies it straight into the destination
parameter, which for a GPU model is already device memory.

- `vllm/model_executor/model_loader/weight_utils.py:905-954` —
  `safetensors_weights_iterator` yields ONE tensor at a time from `safe_open`.
- `vllm/model_executor/model_loader/base_loader.py:43-82` — `load_weights`
  drives that iterator into the already-constructed model.
- `vllm/model_executor/models/utils.py:252-279` — `default_weight_loader` ->
  `param.data.copy_(loaded_weight)`, i.e. the yielded source is dead right
  after one copy into the destination.

One move per byte. Our two-move shape is the divergence. The container reader
itself is a vllm.cpp ORIGINAL (whole-file mmap, no 1:1 upstream mirror, recorded
in the porting inventory); what is mirrored here is the number of moves.

## Our baseline

- Copy helpers: `include/vllm/model_executor/models/dense_weight_loaders.h`
  (`LoadBf16Direct`, `LoadCtNvfp4W4A16`, `LoadCtMxfp4W4A16`, and the merged /
  transposed variants), `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp`
  (27B), `src/vllm/model_executor/models/qwen3_5_weights.cpp` (35B).
- Mapping owned outright by one object:
  `src/vllm/model_executor/model_loader/safetensors_reader.cpp:46-70,204-214`
  (`mmap` + `munmap` in `Release()`), so nothing could outlive it.
- Lazy upload + adoption:
  `include/vllm/model_executor/models/dense_attn_block.h:178`,
  `AdoptDeviceBytesAsHost` in `src/vllm/model_executor/models/qwen3_5_weights.cpp`.
- Shared-ownership shards and the release comment:
  `src/vllm/entrypoints/model_loader.cpp:1063`.
- Windowed source-page release (`LOAD-SAFETENSORS`, already landed):
  `safetensors_reader.cpp:317-339`, spec
  [safetensors-windowed-load.md](safetensors-windowed-load.md), whose
  page-lifetime table enumerates every copy helper.
- No load-time timing or byte accounting existed at all — the first half of
  #150 ("measure it properly") was simply missing.

## Port map

| Concern | Change |
|---|---|
| mapping lifetime | `SafetensorsFile` holds its `mmap`+`fd` in a `shared_ptr<Mapping>` whose destructor `munmap`s/`close`s; `StTensor::mapping` is an alias of it. `Release()` drops this object's reference and its tensor map, so with no borrower the unmap happens at exactly the same instant as before. |
| the borrow | `BorrowStTensorBytes(o, t, dtype, shape)` — ONE entry point, fails closed. Requires a live keep-alive, a non-null source, and `numel(shape) * sizeof(dtype) == t.nbytes`. |
| qualifying call sites | `dense_loaders::LoadBf16Direct` (hence `LoadBf16RawNK` and every arch that routes through the shared helpers), `LoadCtNvfp4W4A16`, `LoadCtMxfp4W4A16`; 27B `LoadModelBf16Direct` (BF16 arm only) and `LoadCtNvfp4Raw`; 35B `LoadBf16Direct`. |
| post-upload residency | `AdoptDeviceBytesAsHost` gains one branch keyed on `OwnedTensor::mmap_src`: host-addressable device memory adopts the device allocation as the host view and releases the source pages; otherwise the borrow stays (a valid, re-faultable `PROT_READ MAP_PRIVATE` view) and only the pages are released. The release is the branch's FIRST statement, ahead of both early returns and ahead of the assignment to `bytes` — that assignment drops this tensor's keep-alive on the mapping and, for the last adopted weight of a shard, munmaps synchronously, so releasing after it would madvise an unmapped range; and putting it ahead of the early returns keeps `VT_ADOPT_DEVICE_BYTES` from silently moving the release lever too. |
| fp4 residents | `ResidentNvfp4` (shared `dense_nvfp4_gemm.h`, and the private one in `qwen3_5.cpp`) is the upload for every borrowed compressed-tensors NVFP4/MXFP4 `packed`/`scale`, so it counts `load_stats::AddDeviceUpload` and runs the same `AdoptDeviceBytesAsHost` step by publishing the allocation on the `OwnedTensor`'s `d_dev` (one control block, still freed once). |
| merged loaders | `ReleaseBorrowedShardSource` releases a per-shard borrow once its bytes have been concatenated into the merged owned buffer. |
| measurement | `load_stats::{AddHostCopy,AddBorrowed,AddDeviceUpload,Snapshot,Reset}`; `VT_LOAD_STATS=1` prints phase timing + the three byte counts from `model_loader.cpp`. |

## Tests to port

vLLM has no loader-residency test to port (its loader has no second copy to
remove), so this is an ORIGINAL test, recorded as such:
`tests/vllm/test_load_direct_upload.cpp`. It asserts the MECHANISM, not a
timing — a timing test passes with the copy still in place.

- the loaded weight's `bytes.data()` IS the address inside the mapping, and
  `bytes.borrowed()` is true;
- the borrow outlives `~SafetensorsFile` and still reads the right values —
  the lifetime question the lazy upload poses, made executable;
- the byte accounting puts the range under `borrowed`, never `host_copy`;
- both arms (`VT_LOAD_DIRECT_UPLOAD` on/off) load byte-identical values;
- transpose, concatenation, size mismatch, dtype mismatch and a missing
  keep-alive each fail closed to the copy.

## Gates

`tests/test_vulkan_backend`, `tests/test_backend_cross_device`,
`tests/test_opt_paged_engine` (STRICT token-exact, with the device asserted from
the printed BACKEND PROOF line, not from an env var),
`scripts/gen-vulkan-spirv.py --check`, the CUDA test suite, and
`scripts/agent-preflight.sh --quiet`. Results in `## Outcome`.

## Dependencies

- `#203`/`#204` (`AdoptDeviceBytesAsHost`) must be on main first: this row
  extends that function rather than reintroducing a second copy.
- `LOAD-SAFETENSORS` windowed release, whose page-lifetime analysis is the
  by-construction evidence that every helper fully consumes its source range.
- GB10 (`dgx.casa`) for any number; llvmpipe proves nothing lifetime-shaped.

## Work breakdown

- **W1** refcounted mapping + `StTensor::mapping`.
- **W2** `BorrowStTensorBytes` (fail-closed) + `OwnedTensor::mmap_src`.
- **W3** qualifying call sites; every other helper untouched.
- **W4** post-upload adoption/page release; merged-shard release.
- **W5** `VT_LOAD_STATS` timing + byte counters (the measurement half of #150).
- **W6** mechanism test + two mutations; GB10 gates; the 27B A/B.

## Risks/decisions

- **The upload is lazy, the mapping was not.** Solved as design shape (a): keep
  the mapping alive to upload time, by refcounting it. Shape (b) — move the
  upload into the load phase — was NOT taken: it changes *when* the work happens
  for every model and backend, its seam
  (`ModelSource::FromSafetensorsOwned(shards, &queue)`) exists only for dense
  models, and on the unified GB10 box `DirectDeviceLoadEligible` deliberately
  returns false, so (b) would have had to be built before it could be measured.
  (a) is a strictly smaller change with the same byte-count result.
- **Not every weight is a straight copy.** Handled by construction, not by
  inspection: the single entry point re-checks the size identity and fails
  closed, so a caller that lies gets a copy rather than a wrong tensor. The
  three in-place repack flags (`repacked`, `q8_0_aligned`, `elem_kn_repacked`)
  that WOULD write through a read-only mapping are set only in
  `qwen3_5_gguf_weights.cpp`, never on the safetensors path.
- **Residency.** A borrow that survived the upload would keep the model resident
  in page cache; the post-upload release is what prevents that, and it is why
  the adoption branch is part of this row rather than a follow-up. It reaches
  every single-source upload: `ResidentWeight` for the bf16 weights and
  `ResidentNvfp4` (both copies) for the compressed-tensors NVFP4/MXFP4
  `packed`/`scale`. **Named residual, NOT claimed:** the MERGED device operands
  in `qwen3_5.cpp` (fp4 `qkv`/`gate_up` concatenation, the Marlin repack
  residents) build ONE device buffer out of SEVERAL borrowed host tensors, so an
  adoption is not representable there and those uploads are still neither
  counted nor followed by a release. They are the 27B's own residency path and
  predate this row; closing them is the next step, not part of this claim.
- **Blast radius.** This is the weight-loading path for every model and backend.
  Mitigated by the fail-closed helper, the same-binary A/B
  (`VT_LOAD_DIRECT_UPLOAD=0`), and running the token-exact engine gate.

## Outcome

**Landed, default ON.** Measured on GB10 (`dgx.casa`, GB10 unified 119 GiB),
Qwen/Qwen3.6-27B bf16, 15 shards, 50.098 GiB of weights, Vulkan build
(`-DVLLM_CPP_CUDA=OFF -DVLLM_CPP_VULKAN=ON`, Release), ONE binary for both arms
(`VT_LOAD_DIRECT_UPLOAD=1` vs `=0`), legs interleaved under one
`flock $HOME/gpu.lock`, `local-ai-worker` stopped and `--restart=no` for the
duration and restored afterwards, `MemAvailable >= 110 GiB` re-checked before
every load (measured 115-116 GiB at every guard).

### Load time

`weights` is the `ModelRegistry::Load` phase from `VT_LOAD_STATS`; `wall` is the
whole `vllm-cli` process — load, one greedy token, teardown.

| Page cache | Arm | `weights` (s) | `wall` (s) |
|---|---|---|---|
| warm | direct ON | **12.48** median (11.927 / 12.268 / 12.691 / 13.003) | **22.47** (21.255 / 22.205 / 22.732 / 22.766) |
| warm | OFF | **19.27** median (18.260 / 19.021 / 19.268 / 20.159 / 20.178) | **30.39** (29.853 / 29.973 / 30.390 / 30.852 / 31.162) |
| cold (`drop_caches` per leg) | direct ON | **32.75** median (32.697 / 32.807) | **55.60** (55.428 / 55.771) |
| cold | OFF | **52.62** median (52.570 / 52.661) | **62.98** (62.774 / 63.190) |

- warm: load phase **1.54x** (-35.2%, -6.79 s), whole process **1.35x** (-7.92 s);
- cold: load phase **1.61x** (-37.8%, -19.86 s), whole process **1.13x** (-7.38 s).

Nine warm legs and four cold legs across three interleaved series.

Every ON leg beat every OFF leg on both axes. ON spread is +/-4% warm and
+/-0.2% cold; OFF +/-5% warm and +/-0.1% cold. The wall ratio is smaller than
the phase ratio because the ON arm DEFERS the page faults for the borrowed
ranges to the (unchanged) upload, which happens after the load phase — the wall
column is the honest number, the phase column shows where the change acts.

The one leg excluded from the warm medians is `arm=1 rep=1` (35.014 s / 61.341 s):
it was the first read of the checkpoint after a `drop_caches`, so it paid the
whole cold disk cost. It is in the raw log, not deleted.

### Bytes moved (measured, not inferred)

`VT_LOAD_STATS` counters, at process exit:

| Arm | host materialization | borrowed in place | device upload | **total moved** |
|---|---|---|---|---|
| OFF | 50.098 GiB | 0 | 50.098 GiB | **100.196 GiB** |
| direct ON | 31.162 GiB | 18.936 GiB | 50.098 GiB | **81.260 GiB** |

The device upload is identical in both arms — the model still has to reach the
device once. What the row removes is 18.936 GiB of host materialization, plus
the anonymous pages and their zero-fill that the removed copy would have needed,
which is why the phase time falls by more than the byte share.

### What qualified, and what did NOT

**37.8% of this checkpoint** (18.936 of 50.098 GiB). Qualifying: `o_proj` and
`down_proj` (raw NK), the embedding table, and the norms — every tensor the 27B
loader takes verbatim. NOT qualifying, correctly: the merged `qkv` and
`gate_up` projections (`LoadMergedBf16RawNK` concatenates several sources into
one buffer) and `lm_head` (`LoadBf16Transposed`). Those are the majority of the
remaining bytes.

**The named next hypothesis:** a merged projection is still a plain copy, just
into an OFFSET of one destination. Uploading each constituent shard directly
into its slice of the device buffer would extend the lever to most of the
remaining 62% — but it needs the device at LOAD time, which is design shape (b)
above and a larger change. Not a ceiling, an unclaimed lever.

### Correctness

- Mechanism gate `tests/test_load_direct_upload` **14/14, 178 assertions**: the
  6 borrow-mechanism cases, 4 post-upload residency cases, 3 fp4-resident cases,
  and 1 general-branch RSS-reclaim case. RED under eleven mutations of the tree
  it guards, with the tree md5-verified restored and re-GREEN after each. EVERY
  count below was measured on THIS tree (rebased on `5023adec`), one mutation at
  a time, rebuilding the binary each time and aborting if the build failed — a
  stale binary reports the previous mutation's result. `origin/main` moved during
  the round, so all eleven were re-run on the FINAL base and reproduce
  identically; no recorded number predates the tree it describes.

  Mutations are named by the TEST CASE they land on rather than by a line
  number, because the line refs are what went wrong twice: they were carried
  forward from an older tree while the cases moved.

  | mutation | red |
  |---|---|
  | drop the size-identity check in `BorrowStTensorBytes` | 1 case / 5 assertions (`BorrowStTensorBytes FAILS CLOSED…`) |
  | skip the borrow in `LoadBf16Direct` | 2 cases / 7 assertions (`a verbatim BF16 weight VIEWS…`, `OFF copies…`) |
  | delete the whole adopt branch | 5 cases / 19 assertions (both fp4-borrow cases + all three `adopt:` release cases) |
  | move the release after the `bytes` reassignment | 5 cases / 7 assertions (incl. the ordering assertions read from inside the munmap) |
  | move the release after the host-addressable early return, BEFORE the env one | 2 cases / 3 assertions (`…NOT host-addressable`, `fp4 … NON-host-addressable`) |
  | move the release after the `VT_ADOPT_DEVICE_BYTES` early return (i.e. after BOTH early returns) | 3 cases / 4 assertions (the two above plus `VT_ADOPT_DEVICE_BYTES=0 moves ONLY the adoption`) |
  | delete the GENERAL adopt branch's `MADV_DONTNEED` | 1 case / 1 assertion (`the general branch DROPS the host mirror's resident pages`) |
  | `ResidentNvfp4`: drop both `AddDeviceUpload` calls | 3 cases / 3 assertions |
  | `ResidentNvfp4`: drop both `d_dev` publications | 3 cases / 20 assertions |
  | `ResidentNvfp4`: drop both `AdoptDeviceBytesAsHost` calls | 3 cases / 16 assertions |
  | `ResidentNvfp4`: drop all six statements (the reviewer's exact revert) | 3 cases / 23 assertions |

  FOUR of these rows were previously recorded wrong, all the same way: a count
  measured against an OLDER suite and copied forward after the suite grew. The
  two release-ordering rows (recorded 1/1 and 2/2, actually 2/3 and 3/4) and the
  two round-2 rows above them (`delete the whole adopt branch` recorded 3/7,
  actually 5/19; `release after the bytes reassignment` recorded 3/3, actually
  5/7) all gained the assertions the round-3 fp4 cases added to the same branch.
  The round-3 edit that claimed to have re-measured the first pair had in fact
  copied them, which its own line refs proved — they pointed at the PREVIOUS
  tree's `:479`/`:499`. Only the general-branch madvise row is genuinely new.
  All six of the `ResidentNvfp4` statements had shipped with NO test that bites
  — a fresh reviewer reverted them and 20/20 fp4-and-loader suites stayed green
  — which is what the three fp4-resident cases close, over the shared
  `dense_nvfp4::ResidentNvfp4` and a fake host-addressable backend.
- The general adopt branch's `MADV_DONTNEED` — the RSS half of this row's
  thesis for every OWNED host mirror — had no test that bites either: every
  other assertion reads VALUES, and the values survive in the device copy with
  or without it. The new case observes RESIDENCY instead, with `mincore()` over
  the mirror's interior pages before and after the adoption, pinning glibc to
  the sbrk arena (`M_MMAP_MAX=0`, trim off, a guard allocation above the mirror)
  so `free()` cannot return the pages by itself and pass the mutant. Linux +
  glibc only, `#if`-guarded: the mechanism is a glibc allocator behavior.
- Linux-only assumption, recorded rather than hidden: every `== 0` assertion in
  the residency section rests on `MADV_DONTNEED` DISCARDING a private anonymous
  mapping's contents. That is Linux semantics; on the BSDs and macOS the advice
  leaves contents intact and these assertions would read `kSrcPattern`. The
  already-merged adopt cases share the assumption, and so does the production
  behavior being pinned. vllm.cpp gates on Linux only.
- The `qwen3_5.cpp` duplicate of `ResidentNvfp4` sits in an anonymous namespace
  inside an 8.5k-line translation unit, so no test can call it. It is held to
  the same invariant structurally by `scripts/check-fp4-resident-consistency.py`
  (mutation suite `tests/scripts/test_check_fp4_resident_consistency.py`, **27
  cases**). The invariant is checked PER BUFFER, inside each buffer's own
  `if (!w.d_<buf>)` upload block. It was NOT: the first version matched
  `AddDeviceUpload` body-wide, so one surviving call satisfied both buffers and
  dropping exactly one counter passed — reproduced on this tree against the real
  `qwen3_5.cpp` text (old checker exit 0, new checker exit 1, for each of `pb`
  and `sb` alone). Nine drop-exactly-one and substitute-one mutations of the
  LIVE duplicate now go RED with the correct per-buffer message: dropping only
  `AddDeviceUpload(pb)`, only `(sb)`, `AddDeviceUpload(0)`, charging the scale
  upload to `pb`, `w.packed.d_dev = nullptr`, `AdoptDeviceBytesAsHost(d.b,
  other.packed)`, copying `w.packed.bytes.data()` into the scale buffer,
  dropping only the packed adoption, dropping only the scale publication.
- WHAT THAT CHECKER DOES NOT DO, stated so the record does not imply more: it is
  a STRUCTURAL check over text. It proves the six statements are present, bound
  to the right buffer of this function's own weight parameter, and in the right
  order. It cannot prove they are correct at run time — that the published
  pointer is the uploaded one, that the byte count matches the allocation, that
  the copy moved the right bytes. **The `qwen3_5.cpp` duplicate is guarded
  against DELETION and gross substitution, not against corruption.** Run-time
  proof exists only for the shared copy; extending it to the duplicate means
  making it reachable, which is the unification refactor this row does not
  attempt.
- GB10 Vulkan build, device asserted from the printed BACKEND PROOF line (not
  from an env var — `VLLM_CPP_DEVICE` is read nowhere in the tree):
  `test_vulkan_backend` **35/35 (2650)**, `test_backend_cross_device`
  **11/11 (132)**, `test_opt_paged_engine` **6/6 prompts token-exact (96/96),
  0 declines, device type 3 (VULKAN)**.
- `scripts/gen-vulkan-spirv.py --check`: `committed SPIR-V is up to date`.
- `scripts/agent-preflight.sh --quiet`: all gates green.

**CUDA, the same tree, GB10** (`-DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=121a
-DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0 -DVLLM_CPP_TRITON=ON`, configure log
confirms the sm120a NVFP4 CUTLASS GEMM and the vendored `sm_121a` Triton AOT
kernels): full serial `ctest` **383/393, 97%**. BOTH SACRED gates PASS —
`test_qwen36_paged_engine` (73.32 s) and `test_qwen27_paged_engine` (46.95 s) —
which is the strongest available statement that a change to the weight path did
not move a token.

All TEN failures were then re-run from a CLEAN `origin/main` (`375a471e`) CUDA
build in the same tree layout, and every one of them reproduces there with the
same signature, so none is a regression from this row:

| test | this tree | clean `origin/main` |
|---|---|---|
| `test_glm4_moe_lite_paged_engine` | anchor drift `REQUIRE(17148 == 43311)` | identical, same token pair |
| `test_gemma4_registry_e2e` | `vt::GeluMulSeparate: ROCm-only fast path in this build` | identical |
| `test_minimax_h3` | SIGSEGV at `:3945` | SIGSEGV at `:3945` (exit 139) |
| `test_linear_method` | `CHECK(0 == 1)` at `:246` (fused-path counter) | identical |
| `test_capi` | SIGSEGV | SIGSEGV (exit 139) |
| `test_qwen3_apc_e2e` | exit 1 | exit 1 |
| `test_gemma4_paged_engine` | exit 1 | exit 1 |
| `test_minicpm3_paged_engine` | exit 1 | exit 1 |
| `test_llama_paged_engine` | exit 1 | exit 1 |
| `test_serve_low_tools` | `FileNotFoundError: 'shellcheck'` | dgx has no `shellcheck`; the suite is 187/187 OK on the dev box |

`test_glm4_moe_lite_paged_engine`, `test_capi` and `test_qwen3_apc_e2e` were
additionally re-run from THIS binary with `VT_LOAD_DIRECT_UPLOAD=0` and failed
identically, so the borrow is not implicated even where the arch does route
through the shared loader helpers (glm4 and gemma4 both do; minimax-h3 does not).

**NOT run, stated plainly:** llvmpipe (the dev box is at 100% disk and cannot
build), Metal, ROCm, XPU, and any multi-GPU configuration. A fresh scoped review
of this head is owed and has not happened.

### Why the default is ON

Same bytes (the mechanism test asserts both arms load byte-identical values),
same tokens (the STRICT engine gate is green), strictly less work, and a
one-variable rollback for a same-binary A/B. There is no regime in which the
copy is faster: it is the same read plus an extra write into memory that is then
discarded.

### Round-3 follow-up on main (the two review findings)

The row merged to main as `8768c64e` on top of the round-2 repair `72548610`. A
fresh scoped review of that head then returned FAIL/narrow — no correctness
defect, no redesign — with two findings, both closed here as ordinary follow-up
defects. Base `9ec78b84`.

**Finding 1 (LOW-MEDIUM), the round-2 repair was pinned by no test.** The
reviewer reverted all six `ResidentNvfp4` statements (an `AddDeviceUpload`, a
`d_dev` publication and an `AdoptDeviceBytesAsHost` per buffer, in BOTH copies),
rebuilt clean, and ran every suite touching fp4 residency or the load counters —
`test_load_direct_upload`, `test_qwen36_weights`, `test_safetensors`,
`test_ops_nvfp4_*`, `test_ops_moe_grouped*`, `test_qwen3_32b_nvfp4a16_*`,
`test_laguna_nvfp4_loader`, `test_minimax_h3`, `test_linear_method`: **20/20
passed**. `test_load_direct_upload` is the only suite asserting on `load_stats`
and it never reached `ResidentNvfp4`; the sole `ResidentNvfp4` case
(`test_qwen36_weights.cpp:587`) is CUDA + 35B-shard gated and asserts nothing
about upload accounting or post-upload residency.

Closed with three cases driving the SHARED `dense_nvfp4::ResidentNvfp4` over the
`FakeBackend` + `ObservableMapping` harness the adopt cases already had, plus a
structural gate for the copy no test can reach. The mutation rows are in
§Correctness above; the estimate that ~30 lines would do it was low — the
harness needed a reusable `BorrowedWeight` split out of `BorrowedUploadedWeight`
and the three cases run ~230 lines with their reasoning.

The reviewer's remaining concern — a host-addressable backend reaching
`ResidentNvfp4` with **non-borrowed** `packed`/`scale`, where the new
`AdoptDeviceBytesAsHost` takes the GENERAL branch and madvises + frees the host
fp4 mirror — is the third case ("a host-addressable device adopts an OWNED fp4
mirror too"). It asserts the full byte content through `bytes` after the
adoption, which is the executable form of "the CPU dequant fallback still reads
valid bytes". A LATER reviewer showed that is only half the branch: deleting the
`MADV_DONTNEED` left the suite 13/13 green, because a value assertion cannot see
a residency change. Round 4 adds the residency observation (§Correctness).

**Finding 2 (LOW), the recorded mutation evidence was wrong — three rounds
running.** The failure mode never changed: a count measured against an older
suite, then copied forward instead of re-measured, while the suite kept growing
against the same branch.

- Round 2 recorded *"move the release after the `VT_ADOPT_DEVICE_BYTES` early
  return: 1 case / 1 assertion"*, pairing a description with the other
  mutation's count.
- Round 3 claimed to have re-measured both halves and recorded 1/1 and 2/2. It
  had not: the line refs it published, `:479` and `:499`, are where those cases
  sat in the tree BEFORE round 3's own edit moved them to `:487` and `:507`. The
  three fp4 cases round 3 added exercise the same release, so both counts had
  gone up under its feet.
- Round 4 re-ran every row, one mutation at a time, on the rebased tree. Four
  rows were wrong, not two.

| mutation | recorded at `e7d61020` | MEASURED (round 4) |
|---|---|---|
| release moved after the host-addressable return, BEFORE the env one | 1 case / 1 assertion | **2 cases / 3 assertions** |
| release moved after BOTH early returns | 2 cases / 2 assertions | **3 cases / 4 assertions** |
| delete the whole adopt branch | 3 cases / 7 assertions | **5 cases / 19 assertions** |
| move the release after the `bytes` reassignment | 3 cases / 3 assertions | **5 cases / 7 assertions** |

MEASURED vs INFERRED. The four right-hand numbers are measured on the round-4
tree (which adds one test case). INFERRED, from the fact that the new case does
not appear in any of those four failure lists: the same four numbers held at
`e7d61020` — the extra assertions came from round 3's own fp4 cases, not from
round 4's. The two rows the reviewer measured independently agree exactly with
this run.

Recording rule adopted so this cannot recur: mutations are named by the TEST
CASE they land on, not by a line number. Case names survive insertions; line
refs are what got copied forward twice.

**Gates for this follow-up** (test-and-record only: the sole compiled change is
`tests/vllm/test_load_direct_upload.cpp`; no `src/`, `include/` or `.cu` file
moved, md5-verified after every mutation). CLEAN Release build, `mudler-ubuntu-box`
dev box, `-DVLLM_CPP_CUDA=OFF -DVLLM_CPP_VULKAN=ON` (`CMakeCache` verified),
**Vulkan = llvmpipe**, which is a correctness device and proves nothing about
speed. Round 4 re-ran all of these on the branch rebased onto `5023adec`:

- `test_load_direct_upload` **14/14, 178 assertions**;
- `test_safetensors` **34/34 (79)**; `test_qwen36_weights` **7/7 (45)** (the
  35B-shard cases skip on this box, so 7/7 here is a skip and not a pass;
  `in_proj_qkv_fp8` did not go red on this build);
- `test_vulkan_backend` **35/35 (2107)**; `test_backend_cross_device`
  **11/11 (132)**; `test_opt_paged_engine` **6/6 prompts token-exact (96/96),
  0 declines, device type 3** from the printed BACKEND PROOF line (`all 9 OPT
  ops dispatched on device type 3 with 0 declines`);
- `scripts/gen-vulkan-spirv.py --check`: `committed SPIR-V is up to date`;
- `scripts/agent-preflight.sh --quiet`: **all gates green**, including
  `check-fp4-resident-consistency` and its now-**27**-case mutation suite.

**NOT run for this follow-up, stated plainly:** GB10, CUDA, and both SACRED
paged-engine gates. The change compiles into exactly one test binary, so a CUDA
re-run is owed to the operator rather than claimed here. Metal, ROCm and XPU
remain unrun as before.
