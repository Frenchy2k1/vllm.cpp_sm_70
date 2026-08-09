# Tenstorrent (Blackhole) backend — spike spec (DRAFT, external proposal)

Status: **DRAFT.** Filed through the claim protocol (`CLAIM-BACKEND-TENSTORRENT-
SPIKE`, draft PR mudler/vllm.cpp#197) but not through the full
`scripts/agent-start.py`/role-interview boot sequence — role was claimed
directly per the developer's explicit direction. Treat it as a starting point
for whoever reviews/claims the row, not as a `READY` artifact. Still no `src/vt/`
code. The § Open risk item that gated this spec's core design claim **is now
hands-on resolved on real Blackhole hardware** — see § Resolved: hands-on spike
result.

Proposed row id: `BACKEND-TENSTORRENT`. Not covered by any existing row —
`.agents/backends.md`'s per-platform table has no Tenstorrent entry.

## Naming — read before writing any code

**Do not spell the `DeviceType` enumerator `kBLACKHOLE`.** This codebase's CUDA
layer is saturated with NVIDIA "Blackwell"-generation references (`sm_120`,
`sm_121`, `GB10` = NVIDIA's Grace-*Blackwell* Superchip, the T0 gate hardware) —
`kBLACKHOLE` next to those is a near-miss for both humans and grep. Use
`DeviceType::kTENSTORRENT` (mirrors `kROCM` naming after the vendor/stack, not
a specific chip); "Blackhole" (Tenstorrent's chip family: P100/P150 PCIe cards)
stays in comments and doc prose only, e.g. `vt::tt` as the namespace.

## Why this is not "write Tensix kernels"

Every backend in this tree targets a SIMT-shaped device (threads/warps, an
implicit memory hierarchy, a compiler taking portable kernel source). Tenstorrent
Tensix cores are a dataflow multicore chip: a `tt_metal::Program` is an explicit
set of reader/compute/writer kernels bound to specific cores, wired by circular
buffers over a NOC, dispatched through a `CommandQueue`. Reimplementing that
surface here would mean re-deriving a meaningful slice of `tt_metal`/`ttnn`
itself.

**Proposed strategy: mirror decision E1 (`.agents/backends.md`), not E2.** Apple
chose MLX (a mature vendor tensor library) over hand-written Metal kernels for
exactly this reason. The Tenstorrent equivalent is **ttnn** — Tenstorrent's own
C++ tensor-op library, which already ships the ops a minimal decode-only backend
needs (see § Evidence). `vt::tt` should be a thin adapter over ttnn calls, not a
from-scratch kernel backend.

## The three seams (`.agents/backends.md`)

**1. `DeviceType` (`vt/device.h`) + `Platform` (`vllm/platforms/interface.h`).**
Mechanical, but note `src/vllm/platforms/platform.cpp`'s `kCurrentPriority`
array is explicitly the ONE non-additive spot — a platform left out registers
fine and is simply never selected, silently. `tests/vllm/platforms/
test_platform.cpp` gates membership; do not skip updating both.

**2. `Backend` (`vt/backend.h`).**

| Method | Tenstorrent mapping |
|---|---|
| `Alloc`/`Free` | see § Open risk below — not a bare `tt_metal::Buffer` allocation |
| `Copy` | `EnqueueWriteBuffer`/`EnqueueReadBuffer` (host<->device; same-device copy is a device-side op) |
| `CreateQueue` | `tt_metal::CommandQueue` — the device already runs 2 CQs in existing perf paths |
| `UnifiedMemory()` | **`false`.** Blackhole is a discrete PCIe card (DRAM over PCIe, no shared address space). |
| `SupportsGraphCapture`/`BeginCapture`/`EndCaptureGraph`/`ReplayGraph` | `tt_metal` trace capture (`begin_trace_capture`/`end_trace_capture`/`replay_trace`) — an existing, exercised feature (`test_perf_trace_2cqs`-style paths already use it), the cleanest mapping in this whole spec |

`UnifiedMemory() == false` is load-bearing, not incidental: `op_provider.h`'s
portable CPU reference tier (§ "Portable reference tier") is gated on
`ReferenceTierEligible`, which requires unified memory. **A Tenstorrent backend
gets ZERO free correctness net for an unregistered op — it throws, it does not
silently slow-path.** Minimal scope must cover every op the chosen model
actually exercises; there is no partial-backend safety rail the way there would
be on an integrated/unified device.

Also flag for whoever picks this up: kernel JIT compile is expensive on first
run. Measured this week on a real Blackhole box: ResNet-50 e2e perf compile
time went from ~120–140s (cold) to ~3.7s (warm cache) between two runs of the
same binary. Needs the same warmup discipline CUDA graph capture already gets
in this codebase, but the cliff is bigger — treat first-token latency as a
distinct, larger risk than on CUDA.

**3. `vt::op_provider` registrations.** One `RegisterOp(OpId::kX,
DeviceType::kTENSTORRENT, ...)` per op, each a thin adapter into the matching
ttnn call (§ Evidence).

## Evidence (read-only checks against a local tt-metal checkout, this week)

- `find_package(TT-NN)` / `find_package(TT-Metalium)` are real, external-
  consumable CMake package configs — confirmed present at
  `build_Release/lib64/cmake/tt-nn/TT-NN.cmake` and `.../tt-metalium/
  Metalium.cmake`, exporting `TTNN::TTNN` as an `IMPORTED SHARED` target with
  `find_dependency(TT-Metalium)` / `find_dependency(xtensor)` wired in. A new
  CMake target can `find_package(TT-NN REQUIRED)` +
  `target_link_libraries(... TTNN::TTNN)` without vendoring tt-metal's build.
- `ttnn::operations::matmul::matmul(const Tensor&, const Tensor&, ...)`
  (`ttnn/cpp/ttnn/operations/matmul/matmul.hpp`) is a plain function, all
  parameters past the two input tensors defaulted — the minimal call is a
  one-liner.
- ttnn C++ op coverage relevant to a minimal decode-only model, confirmed
  present (not assumed) in `ttnn/cpp/ttnn/operations/`:
  - `matmul/` — matmul
  - `normalization/rmsnorm/`, `normalization/softmax/` — norm, softmax
  - `experimental/transformer/rotary_embedding_llama` (+ `_hf`, `_fused_qk`
    variants) — RoPE
  - `embedding/` — token embedding
  - `transformer/sdpa/`, `transformer/sdpa_decode/` — prefill/decode attention
  - **`experimental/paged_cache/`** — paged KV cache. This is the one that
    de-risks the whole plan: vLLM's block-table attention has a direct,
    already-implemented home, not a from-scratch design problem.
  - `reduction/moe`, `data_movement/moe_expert_token_remap`,
    `data_movement/moe_routing_remap` — MoE routing (post-minimal scope, listed
    for later rows)
  - No standalone `silu_and_mul`-shaped fusion found at a glance; likely
    composable from existing elementwise ops, or a small custom-fused op
    later. Needs the hands-on pass to confirm, not assumed here.

## Open risk (the actual unknown — was gating ACTIVE, now resolved)

`vt::Tensor` (`include/vt/tensor.h`) is a **non-owning raw-pointer view**:
`void* data`, `dtype`, `shape[4]`, `stride[4]`, nothing more — the same
convention CUDA/Vulkan/Metal kernels here already use.

`ttnn::Tensor` (`ttnn/api/ttnn/tensor/tensor.hpp`) is a richer, ref-counted
object. Its device-side constructors are `Tensor(DeviceStorage storage)` where
`DeviceStorage` wraps a `tt::tt_metal::MeshTensor`/`MeshBuffer` — i.e. ttnn
owns/allocates through its own mesh-buffer machinery, not a bare pointer.

This looked, from reading headers alone, like it might mean
`vt::Backend::Alloc()` returning a bare `void*` doesn't compose with ttnn's
ownership model. **It doesn't matter in practice — see § Resolved below.**
`Tensor::from_vector<T>(host_vec, spec, device)` uploads directly (no manual
`DeviceStorage`/`MeshBuffer` wiring needed at the call site), and
`to_vector<T>()` reads back. An adapter never needs to attach to a raw device
pointer at all.

## Resolved: hands-on spike result (2026-08-09, real Blackhole hardware)

Built a standalone program (outside this repo's build — `find_package(TT-NN)`
+ `find_package(TT-Metalium)` against a local tt-metal install, `TTNN::TTNN`
linked cleanly on the first try) that:

1. Opens the real device: `ttnn::open_mesh_device(0, DEFAULT_L1_SMALL_SIZE,
   DEFAULT_TRACE_REGION_SIZE)`.
2. Builds two 32×32 `std::vector<float>` host matrices with known values.
3. Uploads each via `Tensor::from_vector<float>(host_vec, spec, device.get())`
   — `spec` built from `tt::tt_metal::TensorSpec(Shape({32,32}),
   TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE),
   MemoryConfig{}))`. No manual `DeviceStorage`/`MeshBuffer` construction.
4. Runs `ttnn::operations::matmul::matmul(a, b)` on-device.
5. Reads the result back via `c.to_vector<float>()`.
6. Compares against a host-computed FP32 reference matmul.

**Result: `max_abs_diff=0.033750` against `max_ref_mag=4.140000` (~0.8%
relative) — PASS.** That's exactly the rounding expected from bf16
accumulation over K=32, not a correctness gap.

This confirms resolution (1) from the original two options below without
needing either of them literally: the adapter doesn't need to manufacture a
raw device pointer for `vt::Tensor.data` to point at. The natural design is
`vt::Tensor.data` as an opaque handle to a heap-boxed `ttnn::Tensor` (or a
thin wrapper owning one); `Alloc` creates it via `ttnn::empty(shape, dtype,
layout, device)` (confirmed present in `operations/creation/creation.hpp`)
or defers to first `Copy`; `Copy`/op adapters use `from_vector`/`to_vector`
for host transfers and pass the `ttnn::Tensor` directly, unwrapped, for
device-resident ops — no host round-trip on the hot path, no manual
`DeviceStorage` construction anywhere.

The two options originally listed here (kept for the record):

1. **Opaque handle, no host round-trip.** Confirmed reachable — see above.
2. **Host round-trip (correctness-only fallback).** Not needed; superseded by
   (1) actually working.

## Minimal scope proposal

One small dense (non-MoE) decoder, **decode-only path first** (prefill can
follow once decode is proven): matmul, rmsnorm, RoPE, embedding, `sdpa_decode`
+ paged KV cache, plus whatever silu_and_mul turns out to need. That's roughly
8-10 `RegisterOp` calls, not vt::ops.h's full ~130. MoE, quantized kernels
(the CUDA/Vulkan-side marlin/nvfp4-equivalents), and graph capture are
explicitly out of this row's minimal scope and follow once decode-only is
correctness-verified end to end.

## Next concrete step

The design-level unknown is resolved (§ Resolved above). What's left before
`src/vt/` code:

1. Maintainer review of this row and the adapter design.
2. First real `src/vt/` code: `DeviceType::kTENSTORRENT` + name (device.h),
   `vt::tt::Backend` (Alloc/Free/Copy/CreateQueue/UnifiedMemory=false,
   modeled on the opaque-`ttnn::Tensor`-handle design above), a `Platform`
   registrar, and ONE op provider (`kMatmul`) as the first vertical slice —
   mirroring how Metal's W0 skeleton landed one op before the rest.
3. Only then the remaining ~7-9 ops in the minimal scope below.

The standalone spike program that produced the § Resolved result is not part
of this repo (built against a local tt-metal checkout, not vllm.cpp's CMake)
and is not attached to this PR; its exact API calls are transcribed above in
full, which is what stays useful here.
