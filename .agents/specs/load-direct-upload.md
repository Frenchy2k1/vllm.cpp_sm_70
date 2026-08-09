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
| mapping lifetime | `SafetensorsFile` holds its `mmap`+`fd` in a `shared_ptr<Mapping>` whose destructor `munmap`s/`close`s; `StTensor::mapping` is an alias of it; `MappingKeepAlive()` exposes it. `Release()` drops this object's reference and its tensor map, so with no borrower the unmap happens at exactly the same instant as before. |
| the borrow | `BorrowStTensorBytes(o, t, dtype, shape)` — ONE entry point, fails closed. Requires a live keep-alive, a non-null source, and `numel(shape) * sizeof(dtype) == t.nbytes`. |
| qualifying call sites | `dense_loaders::LoadBf16Direct` (hence `LoadBf16RawNK` and every arch that routes through the shared helpers), `LoadCtNvfp4W4A16`, `LoadCtMxfp4W4A16`; 27B `LoadModelBf16Direct` (BF16 arm only) and `LoadCtNvfp4Raw`; 35B `LoadBf16Direct`. |
| post-upload residency | `AdoptDeviceBytesAsHost` gains one branch keyed on `OwnedTensor::mmap_src`: host-addressable device memory adopts the device allocation as the host view and releases the source pages; otherwise the borrow stays (a valid, re-faultable `PROT_READ MAP_PRIVATE` view) and only the pages are released. |
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

- **W1** refcounted mapping + `StTensor::mapping` + `MappingKeepAlive`.
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
  the adoption branch is part of this row rather than a follow-up.
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

- Mechanism gate `tests/test_load_direct_upload` **6/6, 77 assertions**, and RED
  under two independent mutations of the tree it guards: removing the size
  identity check in `BorrowStTensorBytes` fails 5 assertions; removing the
  borrow from `LoadBf16Direct` fails 7. Tree restored and md5-verified after
  each.
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
