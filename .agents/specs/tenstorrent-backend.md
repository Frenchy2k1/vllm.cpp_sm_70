# Tenstorrent (Blackhole) backend — spike spec (DRAFT, external proposal)

Status: **DRAFT, W0 skeleton LANDED 2026-08-09.** Filed through the claim
protocol (`CLAIM-BACKEND-TENSTORRENT-SPIKE`, draft PR mudler/vllm.cpp#197) but
not through the full `scripts/agent-start.py`/role-interview boot sequence —
role was claimed directly per the developer's explicit direction. Treat it as
a starting point for whoever reviews this row, not as a maintainer-accepted
artifact — no human review has happened yet. **`ACTIVE` MEANS A GATED
SKELETON, NOT A SUPPORTED BACKEND** — same caveat Metal/Vulkan's own W0 status
carries. See § Our baseline for exactly what that means here.

Proposed row id: `BACKEND-TENSTORRENT`. Not covered by any existing row —
`.agents/backends.md`'s per-platform table has no Tenstorrent entry.

## Scope

**In.** A new extension platform (`.agents/backends.md`'s "Extension
platforms" class, same as Metal/Vulkan): `DeviceType::kTENSTORRENT` as a thin
`vt::Backend` adapter over **ttnn** — Tenstorrent's own C++ tensor-op library
— rather than hand-written Tensix kernels, mirroring the Metal/MLX decision
(E1 in `backends.md`) over the hand-written-kernel alternative (E2). Landed
in this pass: the `DeviceType` + `Platform` + `Backend` seams, and ONE op
(`kMatmul`, F32, rank-2 only). Proposed follow-on (not yet built): a small
dense (non-MoE) decoder, decode-only path first — matmul, rmsnorm, RoPE,
embedding, `sdpa_decode` + paged KV cache, plus whatever `silu_and_mul` turns
out to need. That's roughly 8-10 `RegisterOp` calls total, not `vt::ops.h`'s
full ~130.

**Out.** MoE, quantized kernels (the CUDA/Vulkan-side marlin/nvfp4
equivalents), graph capture, and any model actually running — all
explicitly deferred past this row's minimal scope.

**Naming — read before writing any code.** Do not spell the `DeviceType`
enumerator `kBLACKHOLE`. This codebase's CUDA layer is saturated with NVIDIA
"Blackwell"-generation references (`sm_120`, `sm_121`, `GB10` = NVIDIA's
Grace-*Blackwell* Superchip, the T0 gate hardware) — `kBLACKHOLE` next to
those is a near-miss for both humans and grep. Landed as
`DeviceType::kTENSTORRENT` (mirrors `kROCM` naming after the vendor/stack,
not a specific chip) and namespace `vt::tenstorrent` (not the `vt::tt`
originally floated here — tt-metal's own top-level namespace is `tt::`, and
`vt::tt::Backend` calling into `::tt::tt_metal::...` read as an avoidable
collision once actually writing the code). "Blackhole" (the chip family:
P100/P150 PCIe cards) stays in comments and doc prose only.

**Why this is not "write Tensix kernels."** Every backend in this tree
targets a SIMT-shaped device (threads/warps, an implicit memory hierarchy, a
compiler taking portable kernel source). Tenstorrent Tensix cores are a
dataflow multicore chip: a `tt_metal::Program` is an explicit set of
reader/compute/writer kernels bound to specific cores, wired by circular
buffers over a NOC, dispatched through a `CommandQueue`. Reimplementing that
surface here would mean re-deriving a meaningful slice of `tt_metal`/`ttnn`
itself — hence the ttnn-adapter strategy instead.

## Upstream chain

**No upstream vLLM equivalent exists.** vLLM has no Tenstorrent platform
anywhere in its tree or dependency chain (same as Metal/Vulkan — see
`.agents/porting-inventory.md` §9 item 8 for those, item 15 for this one).
The loyal-port anchor is therefore the `Platform` interface itself, not a
Tenstorrent-specific upstream file: `vllm/platforms/interface.py:134-229`
(`class Platform`), the same seam Metal/Vulkan/ROCm implement against. This
backend adds a NEW leg of that already-faithfully-ported interface; it does
not deviate from anything vLLM defines.

Dependency chain anchors (not vLLM, but load-bearing for this row):
- `ttnn::operations::matmul::matmul(const Tensor&, const Tensor&, ...)`
  (tt-metal `ttnn/cpp/ttnn/operations/matmul/matmul.hpp`) — the one op
  landed.
- `TT-NN`/`TT-Metalium` CMake package configs (tt-metal
  `build_Release/lib64/cmake/{tt-nn,tt-metalium}/`), confirmed real and
  externally consumable — a standalone `find_package(TT-NN REQUIRED)` +
  `target_link_libraries(... TTNN::TTNN)` links clean with no vendoring of
  tt-metal's own build.

## Our baseline

**Landed 2026-08-09, real Blackhole hardware:**

- `DeviceType::kTENSTORRENT` (`include/vt/device.h`) + `kCurrentPriority`
  membership (`src/vllm/platforms/platform.cpp`, after `kMETAL` — newest and
  least proven among the accelerators, ahead of `kCPU`).
- `vt::tenstorrent::Backend` (`src/vt/tenstorrent/tenstorrent_backend.cpp`):
  `Alloc`/`Free`/`Copy`/`Memset` are HOST memory, identical to the CPU
  backend's `aligned_alloc` — Blackhole is genuinely discrete, unlike
  Vulkan's W0 here (unified on its GB10 target, so it can return a directly
  host-dereferenceable mapped pointer; that trick does not apply here).
  `UnifiedMemory()` is `false` regardless — the real hardware property,
  independent of this W0's host-staging implementation detail.
- `vt::tenstorrent::TenstorrentPlatform`
  (`src/vllm/platforms/tenstorrent.cpp`), mirroring `vulkan.cpp`'s registrar
  idiom.
- ONE op provider, `kMatmul`, F32, rank-2 only
  (`src/vt/tenstorrent/tenstorrent_ops.cpp`): `Tensor::from_vector` uploads
  both operands, `ttnn::operations::matmul::matmul` runs on-device,
  `to_vector` reads the result back.
- `tests/vt/test_tenstorrent_backend.cpp`: **3/3 test cases, 8/8 assertions,
  PASS on real Blackhole hardware** — registration, `Platform` mirrors
  `Backend`, and `kMatmul` matches a host F32 reference
  (`max_abs_diff=0.03375` vs `max_ref_mag=4.14`, ~0.8% relative — the
  rounding bf16 accumulation over K=32 predicts, not a correctness gap).

**Honest gaps.** One op, one dtype, one rank. No model runs. Every call pays
a host round-trip (§ Risks/decisions). `UnifiedMemory()==false` means
`op_provider.h`'s portable CPU reference tier stays gated off for this
device — an unregistered op throws, it does not silently slow-path, so
nothing here is "probably fine" by default the way it might be on a unified
backend.

Also worth flagging for whoever widens this: kernel JIT compile is expensive
on first run. Measured on this same box this week: ResNet-50 e2e perf
compile time went from ~120-140s (cold) to ~3.7s (warm cache) between two
runs of the same binary. Needs the same warmup discipline CUDA graph capture
already gets in this codebase, but the cliff is bigger — first-token latency
is a distinct, larger risk here than on CUDA.

## Port map

| `vt::Backend` method | Tenstorrent mapping |
|---|---|
| `Alloc`/`Free`/`Copy`/`Memset` | Host memory (`aligned_alloc`/`memcpy`), same as the CPU backend — see § Our baseline for why |
| `CreateQueue` | `Queue{Device{kTENSTORRENT,0}, nullptr}` — no native queue handle needed while every op stages through `from_vector`/`to_vector` |
| `UnifiedMemory()` | `false` — the real hardware property |
| `SupportsGraphCapture`/`BeginCapture`/`EndCaptureGraph`/`ReplayGraph` | Not implemented in W0. `tt_metal` trace capture (`begin_trace_capture`/`end_trace_capture`/`replay_trace`) is the eventual mapping — an existing, exercised tt-metal feature (`test_perf_trace_2cqs`-style paths already use it), the cleanest-looking piece of future work in this whole spec |

| `vt::op_provider` entry | ttnn call |
|---|---|
| `kMatmul` (landed) | `ttnn::operations::matmul::matmul(a, b)`, operands built via `Tensor::from_vector`, result read via `to_vector` |
| `kRmsNorm` (proposed) | `normalization/rmsnorm/` |
| RoPE (proposed) | `experimental/transformer/rotary_embedding_llama` (+ `_hf`, `_fused_qk` variants) |
| `kEmbedding` (proposed) | `embedding/` |
| Attention (proposed) | `transformer/sdpa/`, `transformer/sdpa_decode/` |
| Paged KV cache (proposed) | `experimental/paged_cache/` — the one that de-risks the whole follow-on plan: vLLM's block-table attention has a direct, already-implemented home in ttnn, not a from-scratch design problem |
| MoE routing (out of scope, later row) | `reduction/moe`, `data_movement/moe_expert_token_remap`, `data_movement/moe_routing_remap` |
| `kSiluAndMul` (unconfirmed) | No standalone fusion found at a glance in `ttnn/cpp/ttnn/operations/`; likely composable from existing elementwise ops, or a small custom-fused op later. Needs a hands-on pass to confirm, not assumed here |

## Tests to port

No upstream vLLM tests exist for this row — there is no upstream Tenstorrent
platform to port tests FROM (same as Metal/Vulkan). `tests/vt/
test_tenstorrent_backend.cpp` is newly authored, mirroring `test_vulkan_backend
.cpp`'s own rationale: every assertion goes through the public `vt::`/
`vllm::platforms::` seam, with no ttnn/tt-metal headers in the test file
itself — if the skeleton needed those headers in a test to be checkable, the
seam would be leaking. Every case SKIPS (not fails) when no Blackhole card is
present, so a Tenstorrent-enabled CI build with no card stays green.

## Gates

**Passing today:** `tests/vt/test_tenstorrent_backend.cpp` (3/3 cases, 8/8
assertions) on real Blackhole hardware — the correctness bar `kMatmul` alone
needs to clear.

**Not yet gated, and explicitly not claimed:**
- No e2e model run (no model has enough registered ops yet).
- No performance claim of any kind — the host-round-trip-per-call design
  (§ Risks/decisions) was never intended to be fast; `docs/BENCHMARKS.md`'s
  row for this backend says so explicitly ("NOT APPLICABLE: no number
  measured, claimed or owed").
- No cross-device oracle comparison (`test_backend_cross_device.cpp` was not
  extended to include `kTENSTORRENT` in this pass).
- No maintainer review.

## Dependencies

- `BACKEND-PLATFORM` (the `Platform` seam) — already `REALIZED`
  (`.agents/porting-inventory.md` §9 item 8's note), composed via, not
  reimplemented by, this row.
- Toolchain: a local tt-metal build with `TT-NN`/`TT-Metalium` installed and
  discoverable via `CMAKE_PREFIX_PATH` (both `lib64/cmake` and `share/cmake`
  — the latter carries `nlohmann_json`/`xtensor`, which `TT-NN`'s package
  config `find_dependency`s). `VLLM_CPP_TENSTORRENT=ON` fails loudly, not
  silently, if these aren't found (`CMakeLists.txt`).
- Hardware: a real Blackhole card for anything beyond the registrars
  compiling — `DeviceAvailable()` gates both the `Backend` and `Platform`
  registrars off cleanly (silent, not a throw) when none is present, mirroring
  Vulkan's `VulkanDeviceAvailable()` idiom.
- No model, dataset, or additional license dependency beyond ttnn/tt-metal's
  own (Apache-2.0).

## Work breakdown

Small, non-overlapping steps, each sized like `kMatmul` was:

1. `kRmsNorm` — next, since normalization is the next thing a decode step
   needs after a projection.
2. RoPE (`rotary_embedding_llama` family).
3. `kEmbedding`.
4. `sdpa_decode` + paged KV cache — the two together, since decode-time
   attention needs both.
5. Whatever `silu_and_mul` turns out to need (composable or a small custom
   op — § Port map).
6. Only once 1-5 land: revisit the host-round-trip design (§
   Risks/decisions) for a device-resident-tensor performance pass.

Each step is independently claimable and gateable — none blocks another
except that 4 wants 1-3 landed first for a meaningful decode-step test.

## Risks/decisions

Two real problems surfaced landing the W0 skeleton, both fixed and
documented in place rather than papered over — these are the load-bearing
facts for anyone extending this row, not just implementation trivia:

1. **A build-system integration hazard, not a Tenstorrent or vllm.cpp bug.**
   `tenstorrent_ops.cpp` includes `ttnn/operations/matmul/matmul.hpp`, which
   transitively pulls in tt-metal's `tt_stl/reflection.hpp`. That header
   properly includes the full `<nlohmann/json.hpp>` (verified — nothing
   missing on tt-metal's side). The break is that `vllm.cpp` ALSO vendors its
   own private copy of nlohmann-json (`third_party/nlohmann/json.hpp`, a
   different version) on the same include path for the whole `vllm` target.
   nlohmann-json tags each version's types in an ABI-versioned `inline
   namespace` specifically so mixing two versions in one translation unit is
   a loud compile error (ambiguous `basic_json`) rather than a silent ODR
   violation — which is exactly what happened the first time this file
   compiled. Fixed by isolating `tenstorrent_ops.cpp` as its own `OBJECT`
   library with its own include set (TT-NN's dirs, no `third_party`) — the
   same isolation pattern this codebase already uses for
   `vllm_rocm_platform_syntax_check`, not a new idiom. The real fix, if ever
   wanted, is on vllm.cpp's side (`find_package(nlohmann_json)` instead of a
   private vendored copy) — a cross-cutting build decision out of scope for
   this row, noted for the maintainer.
2. **A process-exit segfault, actually ours.** The first working version held
   the mesh device in a plain `static std::shared_ptr<MeshDevice>`. All test
   assertions passed, but the process then segfaulted during static
   destruction, inside `MeshDevice`'s own teardown chain
   (`GraphTracker::track_deallocate_cb`) — a cross-`libtt_metal.so`-boundary
   static destruction ordering hazard a standalone spike program (see below)
   never hit, because it held the device in a `main()`-local variable
   destroyed deterministically mid-program, not during static teardown.
   Fixed by deliberately leaking the device (heap-allocate, never `delete`)
   so its destructor never runs at process exit at all — documented in
   `tenstorrent_device.cpp`/`.h`, not a silent workaround.
3. **Product decision: host round-trip per call, not device-resident
   tensors.** `vt::Tensor` is a bare device-pointer view; before landing any
   code, it looked like this might force `vt::Backend::Alloc()`'s bare
   `void*` to somehow manufacture a raw device pointer `ttnn::Tensor` could
   attach to without copying. A standalone spike program (outside this
   repo's build — `find_package(TT-NN)`/`find_package(TT-Metalium)` against
   a local tt-metal install) proved that's not actually necessary:
   `Tensor::from_vector<T>(host_vec, spec, device)` uploads directly, no
   manual `DeviceStorage`/`MeshBuffer` wiring needed, and `to_vector<T>()`
   reads back — `max_abs_diff=0.033750` vs `max_ref_mag=4.140000` (~0.8%,
   the bf16-over-K=32 rounding expected) confirmed this on real hardware
   before any `vt::` code existed. The landed `kMatmul` provider uses exactly
   that sequence, which means every call pays a host round-trip rather than
   keeping tensors device-resident between ops — correct (proven by the
   spike and by the 3/3 test pass) but not fast, a deliberate W0 scope
   decision, not an oversight (`tenstorrent_backend.cpp`'s SCOPE note).
   Avoiding the round trip is real future work (§ Work breakdown step 6), not
   attempted here.
