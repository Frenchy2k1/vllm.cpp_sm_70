# ROCm (AMD GPU) backend — contributor guide

**State today: the W0 skeleton is committed, and no HIP source in it has ever
been compiled.** `kROCM` exists, `-DVLLM_CPP_HIP=ON` exists, and there is a
`vt::Backend`, a `Platform`, and exactly one registered kernel (RmsNorm). The
plain-C++ parts compile and are tested here; the three `.hip` files have not been
built by anyone, because no maintainer machine has an AMD GPU. **Your first HIP
compile is genuinely the first**, and a failure is the expected outcome rather
than a sign you did something wrong.

This page exists because several people offered hardware in
[issue #41](https://github.com/mudler/vllm.cpp/issues/41), and it answers the
three questions that decide whether that goes anywhere: what a backend actually
*is* in this codebase, what to write first on the hardware you own, and what
"done" means. The design record behind the skeleton, including what was
deliberately left out, is
[.agents/specs/rocm-backend-w0.md](../.agents/specs/rocm-backend-w0.md).

Everything here is checked against the tree on 2026-08-06. Where a number is
counted, the command that counts it is given, because these numbers drift.

## 1. Why ROCm is the cheapest backend to add

Three structural facts, in the order they matter:

1. **The engine never learns about your device.** Scheduler, KV/block manager,
   persistent batch, sampler and serving are backend-agnostic, mirroring
   upstream vLLM. A backend lands as additive files through three seams
   ([.agents/backends.md](../.agents/backends.md)). Adding a platform touches
   one enum and one switch, both in `include/vt/device.h`. That is the whole
   core edit.
2. **Our CUDA kernels are ports of vLLM's `csrc/`, and upstream compiles that
   same `csrc/` for ROCm through a hipify pass.** So ROCm is not the
   Metal/Vulkan situation, where every kernel is written from scratch against a
   foreign API. Most of `src/vt/cuda/` is HIP source that has not been hipified
   yet. Upstream also ships RDNA3-specific kernels in `csrc/rocm/`
   (`q_gemm_rdna3.cu`, `moe_q_gemm_rdna3.cu`, `skinny_gemms.cu`), and every board
   offered in #41 so far is RDNA3.

   One caveat, so nobody loses an afternoon to it: vLLM's own `cmake/hipify.py`
   imports `torch.utils.hipify`, so it is a **torch-dependent** tool and we
   cannot reuse it. Your route is ROCm's `hipify-clang`, or hand-translation.
   `src/vt/rocm/rocm_rmsnorm.hip` is a hand-translation of
   `src/vt/cuda/cuda_ops.cu:96-126` written out deliberately so the two can be
   read side by side as a worked example of what the pass does.
3. **On unified-memory parts, a model can run correctly with zero ROCm
   kernels.** See §3. This is the single biggest lever for getting started, and
   it splits the work by hardware rather than by skill.

## 2. What a backend is, file by file

These now exist. The column that matters is the last one: what has been checked
on a real machine, and what has not.

| Seam | File | Verified? |
|---|---|---|
| Device enum | [`include/vt/device.h`](../include/vt/device.h) | ✅ compiled; the enum forced exactly one switch site tree-wide |
| — | [`include/vt/rocm/rocm_arch.h`](../include/vt/rocm/rocm_arch.h) — gfx name → `(major, minor)`, ported 1:1 from `rocm.py:223` | ✅ **unit-tested**, 40 assertions, no GPU needed |
| Runtime backend | [`src/vt/rocm/rocm_backend.hip`](../src/vt/rocm/rocm_backend.hip) — the 6 `vt::Backend` virtuals | ❌ **never compiled** |
| Op table | [`src/vt/rocm/rocm_ops.hip`](../src/vt/rocm/rocm_ops.hip) — one `RegisterOp` line | ❌ never compiled |
| Kernel | [`src/vt/rocm/rocm_rmsnorm.hip`](../src/vt/rocm/rocm_rmsnorm.hip) | ❌ never compiled |
| Platform | [`src/vllm/platforms/rocm.cpp`](../src/vllm/platforms/rocm.cpp) — mirrors `vllm/platforms/rocm.py` | ✅ compiles `-Werror` (plain C++, object-compiled in every build as a bit-rot guard); never *run* |
| Attention | *(none yet — `get_attn_backend_priority()` returns empty)* | — |
| Build | `VLLM_CPP_HIP` in [`CMakeLists.txt`](../CMakeLists.txt) | ✅ the OFF path and the fail-without-hipcc path |
| Test | [`tests/vt/test_rocm_backend.cpp`](../tests/vt/test_rocm_backend.cpp) | ✅ compiles (same guard) — ❌ never run |

So the shape is decided and the parts that hold a *decision* are tested; what
you are validating is the API glue. Adding your own op is one line in
`rocm_ops.hip` plus the kernel — no selector, model or runner edit anywhere.

Op coverage as of 2026-08-06 (`OpId` has 106 entries):

| Backend | Registered ops |
|---|---|
| CUDA | 103 |
| CPU | 83 |
| Metal | 19 |
| Vulkan | 8 |
| **ROCm** | **1** (RmsNorm) |

Recount before quoting:

```sh
grep -rho 'RegisterOp(OpId::[A-Za-z0-9_]*' src/vt/<backend>/ | sort -u | wc -l
```

The platform seam is deliberately plain C++ with no device headers: everything
device-specific is reached through the `vt::Backend` virtuals, which is why
`platforms/vulkan.cpp` compiles without a Vulkan header. Do the same, and the
engine-side tree stays free of HIP.

## 3. Correctness before kernels: the reference tier

`include/vt/op_provider.h:186-224` is our equivalent of vLLM's
`CustomOp.forward_native`. An op with no native kernel on your device falls back
to the CPU kernel, registered at a strictly-negative priority so a native kernel
always wins when it exists. A backend that implements **zero** kernels is
therefore correct, just slow.

**The gate is `Backend::UnifiedMemory()`, never `DeviceType`.** A CPU kernel
dereferences host pointers, which is only valid where host and device memory
alias. Consequences:

- **Unified memory** (Strix Halo, RDNA3 iGPU, anything where you report
  `UnifiedMemory() == true` honestly): a model runs end to end as soon as
  §2's first three rows exist. Kernels then replace the fallback one at a time,
  each one a measurable win with no correctness risk.
- **Discrete** (7900 XTX and every dGPU): the tier never installs, and it must
  not. `GetOp` throws on an unregistered op, so a model runs only once the ops
  it needs are registered. Your first milestone is the kernel path.

Two rules that keep this honest: `VT_OP_PROVIDER_STATS=1` prints the first time
each `(op, device)` falls back, and `GetReferenceTierHits()` **must be 0 in any
performance measurement**. A non-zero value means you benchmarked the CPU.

## 4. Pick your first task from your hardware

| Hardware | Arch | Memory | Start here |
|---|---|---|---|
| Strix Halo / GTR9 Pro 128GB | gfx1151 | unified | **Build M0/M1, then M2.** Once it compiles, the reference tier means a model runs with no further kernel written. Closest analogue to GB10, so the residency-policy question in §6 is yours |
| Radeon 780M iGPU | gfx1103 | shared | **Build M0/M1**, same path, smaller models. Best position to find every place a "CUDA" assumption is really an "NVIDIA" assumption. A vLLM-ROCm oracle is unlikely on this board, so M4 stays PENDING there — fine, and to be said rather than papered over |
| 4x 7900 XTX | gfx1100 | discrete | **Build M0/M1, then the kernel path**, since the reference tier cannot install on a dGPU and a model needs real kernels. The only board that can host a vLLM-ROCm oracle for M4 and, later, multi-GPU TP — the backend already registers all four at `Device{kROCM, i}` |

These do not collide. Two people can be on M0/M1/M2 on unified parts while a
third does the hipify pass, and the discrete board is what turns the result into
a gated backend.

## 5. Milestones as concrete PRs

**M0 — build. WRITTEN, unverified.** Tri-state `VLLM_CPP_HIP`, hipcc detection
that fails loudly, `VLLM_CPP_HIP_ARCHITECTURES`, `ROCM_PATH`. What remains is
for someone to run it. Acceptance: `cmake -DVLLM_CPP_HIP=ON` configures and
`cmake --build` produces a binary. **This is the open task.**

**M1 — platform + backend. WRITTEN, unverified.** `rocm_backend.hip`,
`platforms/rocm.cpp`, the `kROCM` enum, the capability parse (tested), and one
registered op. Acceptance: `ctest -R 'rocm|cross_device'` green on the device —
which also means the RmsNorm kernel matched the CPU oracle at NMSE ≤ 5e-4, so
seam 3 is proven end to end.

Expect M0/M1 to need fixes. A compile error in `rocm_backend.hip` is the single
most valuable thing anyone can report right now, and it belongs in this repo
rather than in a fork.

**M2 — first model end to end.** On a unified part this is mostly free: assert
`ReferenceTierEligible(kROCM)` and run a small dense model. Acceptance: greedy
token parity against the **CPU backend** on the same build, plus the
`VT_OP_PROVIDER_STATS=1` output showing which ops fell back, which is your
kernel to-do list, sorted by real usage rather than by guesswork.

**M3 — kernels + attention.** Hipify `src/vt/cuda/` family by family, starting
with what M2's fallback log actually hit: layernorm, rope, activations, glue,
reshape-cache, sampling, then paged attention. Register a ROCm attention backend
and put its name in the platform priority in the same change. For what upstream
selects on your arch, read `_get_backend_priorities` (`rocm.py:407`) and
`get_attn_backend_cls` (`rocm.py:545`): AITER FA is gfx9-only, RDNA3 goes down
the Triton/ROCm attention path.

**M4 — correctness gate.** Greedy token parity against a vLLM-ROCm oracle on the
same hardware, same workload, following
[.agents/gates.md](../.agents/gates.md) and the near-tie methodology. Where
vLLM's own greedy output is non-deterministic, the gate is distributional (ours
inside vLLM's K-run set), not token-exact.

**M5 — speed.** `vllm bench throughput` on the same box, quant-matched, against
the same model. The bar is vLLM, not llama.cpp. Method and honesty rules:
[.agents/benchmark-protocol.md](../.agents/benchmark-protocol.md) and
[docs/BENCHMARKS.md](BENCHMARKS.md).

Each milestone is a PR, or several. Do not stack M3 kernels into one change: one
kernel family per PR, each with its own correctness check, is what keeps review
from becoming the bottleneck.

## 6. What not to port

Do not spend time hipifying these. They are NVIDIA-specific and none of them is
on the path to a working AMD backend:

- NVFP4 (`cuda_matmul_nvfp4*.cu`, the `nvfp4_tactics` family) and Marlin. FP4
  tensor cores are a Blackwell thing; AMD's analogue is MXFP4 on gfx950 and is a
  separate project.
- CUTLASS-backed FA2 and the sm90/sm100 scaled-MM paths. The ROCm equivalents
  are Composable Kernel / AITER, and they are M3-and-later decisions.
- Vendored Triton-AOT cubins. Arch-specific NVIDIA binaries.
- NCCL transport. RCCL is API-compatible, but multi-GPU is post-M5.
- cuBLASLt plan caches. Route GEMM to hipBLASLt and measure before porting any
  caching strategy.

## 7. Working with the record

The project keeps an append-only engineering record under `.agents/`. Two things
matter for an outside contributor:

- **Machine paths in that record are not instructions.** They describe the
  developer's boxes. Yours go in an untracked `.env` (copy
  [`.env.example`](../.env.example), which already has the device-toolchain
  fields a ROCm bring-up needs: toolkit root, compiler, target arch) plus
  `.agents/developer-preferences.md`. A fresh agent session will generate both
  interactively. Register your AMD box as a profile in
  [.agents/environment.md](../.agents/environment.md) so it becomes the named
  gate environment for the ROCm rows.
- **A gate you cannot run stays PENDING.** That is a normal, publishable state.
  Claiming a pass you did not observe is the one thing that is not recoverable.
  The same applies to labels: "build-supported" means it compiles and emits real
  code, "runtime-gated" means a board here executed it. Do not upgrade one to
  the other on inference.

## 8. CI gates your PR will hit

All of these run on pull requests and are cheap to check locally first:

- **`FOLLOWING_AGENTS_PROTOCOL` trailer** on every non-merge commit, asserting
  you read [AGENTS.md](../AGENTS.md). Also add `Assisted-by: <tool>` if an AI
  assistant helped.
- **PR size**: 900 changed lines outside `.agents/`, `docs/`, `scripts/`,
  `tests/scripts/`, `.github/`. Enforced on `row/*` branches, reported on
  others.
- **Documentation checkpoint**: `python3 scripts/check-doc-checkpoint.py --base
  <base> --head <head>`. It validates the **committed** diff, so run it after
  committing, not before.
- **Device-leakage ratchet**: `python3 scripts/check-device-leakage.py`. It
  counts CUDA-specific references in the device-agnostic layer
  (`src/vllm/`, `include/vllm/`) and fails on any increase. It will not object to
  ROCm code under `src/vt/rocm/` or to your platform file; it will object if a
  model TU grows a device branch. Keep ROCm specifics below the seams.
- The full CPU test suite. Run `ctest --test-dir build` before pushing; some
  tests are flaky under `ctest -j` on a loaded box, so re-run failures serially
  before reporting them.

## 9. Asking

Comment on [#41](https://github.com/mudler/vllm.cpp/issues/41) with what you
picked and what you hit. Milestones get split into their own issues once work
starts, so say which one you are taking to avoid two people writing the same
platform file.
