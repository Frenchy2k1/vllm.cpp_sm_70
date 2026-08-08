# ROCm unified memory by construction — approach (b) (BACKEND-ROCM W1)

Row: `BACKEND-ROCM` (backend-matrix). Task #286, resolving issue #41's F6 fork.
Implemented blind (no AMD hardware, no hipcc here) under this lane's standing
policy: board owners provide compile evidence.

## The decision, verbatim

Maintainer call on #41 (2026-08-08), quoted in full because it is the binding
statement this change implements:

> Maintainer call on the F6 fork (@jimmykarily, @arch-btw): **approach (b)** —
> make unified memory true by construction. On a device reporting
> hipDeviceAttributeIntegrated=1 (and ManagedMemory=1 +
> ConcurrentManagedAccess=1), Backend::Alloc in the ROCm backend uses
> hipMallocManaged instead of hipMalloc, and UnifiedMemory() returns true
> exactly then — host access becomes API-guaranteed rather than architecturally
> incidental, which is the standard this gate exists to hold. The fix lives
> entirely in src/vt/rocm/ (the shared gate and the CUDA/GB10 path stay
> byte-untouched), costs one allocation-path branch, and (a) remains available
> as a fallback note if managed allocations measure slower on gfx1151 —
> measure, don't assume. A PR from a board owner with compile+ctest evidence
> (the M0/M1 tables you have both already posted are exactly the right shape)
> lands it; the reference-tier e2e (M2) should unblock immediately after. On
> the discrete lanes this decision changes nothing: UnifiedMemory()=false stays
> honest there and native kernels remain the path (#140 is doing exactly that
> for gfx1201).

## The measurements it rests on (jimmykarily, gfx1151; arch-btw, gfx1103)

F6, measured on Strix Halo and confirmed byte-for-byte in direction on the
780M:

```
hipDeviceAttributeIntegrated                             : 1
hipDeviceAttributePageableMemoryAccess                   : 0   <-- the W0 veto
hipDeviceAttributePageableMemoryAccessUsesHostPageTables : 0
hipDeviceAttributeManagedMemory                          : 1
hipDeviceAttributeConcurrentManagedAccess                : 1
hipDeviceAttributeUnifiedAddressing                      : 1
```

PageableMemoryAccess needs XNACK; RDNA3 has none (`XNACK enabled: NO`, and
`HSA_XNACK=1` changes nothing), so the W0 conjunction (integrated AND pageable,
carried over from CudaBackend) reads false on every RDNA3 APU — while the
aliasing the reference tier needs was separately measured to HOLD there (kernel
writes to a hipMalloc pointer, host reads it back with no memcpy). The
attribute answers the opposite direction (device reading pageable host memory)
from the one the tier requires (host reading device allocations); the two
coincide on NVIDIA integrated parts (GB10/Jetson report both 1) and come apart
on RDNA. hipPointerGetAttributes advertises no host alias for that working
pointer (type=device, hostPointer=nil), which is why "it worked in my test" was
not accepted as the probe and (b) was ratified over (a).

## What was built

- `src/vt/rocm/rocm_backend.hip`: `ProbeDevice` additionally probes
  `hipDeviceAttributeManagedMemory` + `hipDeviceAttributeConcurrentManagedAccess`
  (failed probes default to 0 — the safe direction, never unregistering the
  device). `UseManagedAlloc(caps) = integrated && managed &&
  concurrent_managed` selects the branch; `Backend::Alloc` then uses
  `hipMallocManaged(hipMemAttachGlobal)` (bytes==0 stays on `hipMalloc` to
  keep the zero-size contract byte-identical); `UnifiedMemory() =
  managed_branch || (integrated && pageable)` — the W0 conjunction kept intact
  as ground 1. `Free` stays `hipFree` for both branches: the HIP runtime API
  documents it as the release call for `hipMalloc` and `hipMallocManaged`
  alike (mirroring `cudaFree`; there is no `hipFreeManaged`). Certainty HIGH,
  API-documented, stated in the code as an assumption a board can falsify.
- Allocation-site audit (every site in the backend): `Alloc` branches;
  `Free` single-path by API contract; inherited `Backend::AllocPinned/
  FreePinned` delegate to `Alloc`/`Free` (`src/vt/backend.cpp:19-20`) and so
  ride the same branch — coherent with the pinned contract
  (`include/vt/backend.h:76-78`, "ordinary host memory via Alloc, correct on
  unified memory"); `rocm_rmsnorm.hip` has zero allocation sites; the skeleton
  has no pool paths. Discrete devices take `Integrated=0`, so the managed
  branch is provably dead and the discrete path is byte-identical to W0.
- Introspection seam (HIP-free): `vt::rocm::ManagedAllocActive(index)` /
  `IntegratedDevice(index)` in `include/vt/rocm/rocm_runtime.h`, so the tests
  and board reports can name the active path without a device header.
- Tests (`tests/vt/test_rocm_backend.cpp`, compile everywhere via the
  syntax-check object target, run only under `VLLM_CPP_HIP` with a device):
  the alloc-path/UnifiedMemory coupling case (discrete: managed branch dead
  AND unified false; integrated: managed active AND unified true, loud
  failure with the probe triple if a board class outside the fix appears),
  and F6's decisive experiment as a standing gate (host-writes inputs with no
  Copy, native RmsNorm kernel reads them, host-reads the kernel-written
  output with no Copy, against the golden row).
- CMake F1/F3 absorption (`CMakeLists.txt`, HIP branch, before
  `check_language(HIP)`): when `ROCM_PATH` exists on disk, derive
  `CMAKE_HIP_COMPILER_ROCM_ROOT`, seed `--rocm-path=${ROCM_PATH}` into
  `CMAKE_HIP_FLAGS`, and export `ROCM_PATH` into the environment — each ONLY
  when unset, so explicit flags/env win and non-ROCm machines are
  byte-identical. F2 is downstream of the unidentified compiler and needs no
  separate change.

## Upstream grounding

Upstream vLLM has NO analog: allocation is torch's job there, and
`vllm/platforms/rocm.py` knows the APUs only as device-name map entries
(`rocm.py:75-77`, Strix Point/Halo) plus `is_navi` (`rocm.py:909-910`);
`grep -rn hipMallocManaged csrc/` over the pin comes back empty. This is
therefore a recorded ADDITIVE deviation (porting-inventory §9), grounded in the
issue-41 measurements above rather than in an upstream file. What IS mirrored:
the `ROCM_PATH` build handling (`vllm/CMakeLists.txt:54-60` @ pin `555967922`)
and the W0 probe's CUDA lineage (`src/vt/cuda/cuda_backend.cu:295-303`).

## Verification the community owes (all PENDING, no AMD hardware here)

| Gate | Board | What to post on #41 |
|---|---|---|
| Configure with NO manual flags (F1/F3 absorbed) | Arch/TheRock: gfx1151, gfx1103 | configure log; any flag still needed is a miss |
| `.hip` compile of the (b) delta | any | first error text, or "clean" |
| `ctest -R 'rocm\|cross_device'` incl. the two new cases | all four boards | pass/fail + the printed `integrated/managed-alloc/UnifiedMemory` triple |
| Discrete branch provably dead | gfx1100, gfx1201 | the same triple: `0/0/false` |
| M2: small dense model, greedy parity vs `--device cpu` | gfx1151, gfx1103 | tokens + `VT_OP_PROVIDER_STATS=1` fallback list |
| (a)-fallback check: managed-alloc speed | gfx1151 | only if M5-era numbers regress vs hipMalloc — measure, don't assume |

STATUS/BENCHMARKS: docs/STATUS.md and docs/BENCHMARKS.md carry the
IMPLEMENTED-UNVERIFIED / PENDING-community rows for this change; docs/ROCM.md
§3.1 and §5.2 are the contributor-facing statement.
