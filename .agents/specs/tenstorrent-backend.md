# Tenstorrent (Blackhole) backend — spike spec (DRAFT, external proposal)

Status: **DRAFT — not filed through the repo's own agent workflow.** This was
written by an assistant session at the developer's request, outside
`scripts/agent-start.py`/role declaration/`.agents/policy.csv` gates. Treat it
as a starting point for whoever claims the row, not as a `READY` artifact.
No code, no kernels, no benchmark. The only hands-on work done is read-only:
confirmed `TT-NN`'s CMake export resolves and inspected the exact C++ API
surface (see § Evidence).

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

## Open risk (the actual unknown — resolve this before claiming ACTIVE)

`vt::Tensor` (`include/vt/tensor.h`) is a **non-owning raw-pointer view**:
`void* data`, `dtype`, `shape[4]`, `stride[4]`, nothing more — the same
convention CUDA/Vulkan/Metal kernels here already use.

`ttnn::Tensor` (`ttnn/api/ttnn/tensor/tensor.hpp`) is a richer, ref-counted
object. Its device-side constructors are `Tensor(DeviceStorage storage)` where
`DeviceStorage` wraps a `tt::tt_metal::MeshTensor`/`MeshBuffer` — i.e. ttnn
owns/allocates through its own mesh-buffer machinery. The host-facing
`from_span`/`from_vector`/`from_borrowed_data` constructors are for **host**
data being pushed to device (they copy host->device); there is **no
"attach a Tensor to a raw device pointer I already own" constructor** at this
level.

This means `vt::Backend::Alloc()` returning a bare `void*` does not compose
directly with ttnn's ownership model — the two Tensor types have genuinely
different lifecycle assumptions, and this is the one place "thin adapter" could
turn out not to be thin.

Two ways to resolve it, in preference order:

1. **Opaque handle, no host round-trip (preferred).** `vt::tt::Alloc` allocates
   through ttnn/tt_metal's own buffer machinery (e.g. an owning
   `MeshBuffer`/`DeviceStorage`-compatible allocation) and returns an opaque
   `void*` handle to a small heap-allocated wrapper holding that real
   tt-metal-side handle — `vt::Tensor.data` is never a literal device address
   the way it is on CUDA, only an opaque token the Tenstorrent adapter knows
   how to unwrap back into a `ttnn::Tensor(DeviceStorage(...))` without a copy.
   `Copy`/`Free` follow the same indirection. Needs confirming that
   `DeviceStorage`'s constructor from an already-allocated `MeshBuffer` is
   actually reachable this way (not verified — this is exactly the "spend an
   hour" hands-on spike item).
2. **Host round-trip (correctness-only fallback for a first spike).** Adapter
   copies the vt-side device bytes to host, `Tensor::from_span(host_data, spec,
   device)` pushes to device, runs the ttnn op, copies the result back. Simple,
   provably correct, but defeats the entire point of a discrete-device backend
   and should never be presented as more than a first correctness checkpoint —
   explicitly not a shipping design.

## Minimal scope proposal

One small dense (non-MoE) decoder, **decode-only path first** (prefill can
follow once decode is proven): matmul, rmsnorm, RoPE, embedding, `sdpa_decode`
+ paged KV cache, plus whatever silu_and_mul turns out to need. That's roughly
8-10 `RegisterOp` calls, not vt::ops.h's full ~130. MoE, quantized kernels
(the CUDA/Vulkan-side marlin/nvfp4-equivalents), and graph capture are
explicitly out of this row's minimal scope and follow once decode-only is
correctness-verified end to end.

## Next concrete step

The § Open risk item is the one thing this spec could not resolve read-only.
Whoever claims `BACKEND-TENSTORRENT` should spend the hands-on hour proving
resolution (1) above with a throwaway standalone program linking `TTNN::TTNN`
(outside this repo's build, no vllm.cpp code touched yet) — allocate a device
buffer through ttnn's own path, wrap it in a `vt::Tensor`-shaped opaque handle,
round-trip it through `ttnn::operations::matmul::matmul` bit-exact against a
host-computed reference — before writing a single line inside `src/vt/`. If
resolution (1) turns out not to be reachable, this spec's whole "thin adapter"
premise needs revisiting before the row goes anywhere near `ACTIVE`.
