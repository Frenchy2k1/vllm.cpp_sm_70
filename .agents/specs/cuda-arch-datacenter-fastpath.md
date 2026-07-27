# Hopper (sm_90a) + datacenter-Blackwell (sm_100/103/110) fast-path — scoping spike

Status: **SPIKE (scoping only; read-only on production code).** This spec scopes
porting the FAST-PATH kernel bodies for the datacenter CUDA architectures whose
FRAMEWORK is already done: `BACKEND-CUDA-SM090` (Hopper) and
`BACKEND-CUDA-SM100`/`BACKEND-CUDA-SM103`/`BACKEND-CUDA-SM110` (datacenter
Blackwell) all BUILD today as **single-arch PORTABLE-KERNELS-ONLY** (every
fast-path FEATURE-TABLE cell resolves EMPTY — no wgmma/tcgen05 body exists), per
[cuda-arch-additivity.md](cuda-arch-additivity.md) §W9/§W10. The additive seams
(`BACKEND-CUDA-ARCH-ADDITIVITY`) mean adding a fast path is a FEATURE-TABLE cell
widen + a tactic registration + the ported kernel body — not a framework change.

Owner: `CLAIM-CUDA-DATACENTER-SCOPE`. Rows scoped here (moved `INVENTORIED`/
build-supported → `SPIKE`): `BACKEND-CUDA-SM090`, `BACKEND-CUDA-SM100`,
`BACKEND-CUDA-SM103`, `BACKEND-CUDA-SM110`, `BACKEND-CUDA-COMP-MACHETE`,
`BACKEND-CUDA-COMP-DSV3`, `BACKEND-CUDA-COMP-MOE-CUTLASS`,
`BACKEND-CUDA-COMP-W4A8`, `BACKEND-CUDA-COMP-MLA`, `BACKEND-CUDA-COMP-FLASHMLA`,
`BACKEND-CUDA-COMP-DEEPGEMM`. Roadmap wiring: `ROAD-V1-D1` (NVIDIA target
fan-out). Multi-arch rows that ALREADY carry an sm_121 implementation
(`BACKEND-CUDA-COMP-SCALEDMM-C3X`, `BACKEND-CUDA-COMP-FP4`) stay `PARTIAL`; their
sm_90/sm_100 legs are scoped by this spec but the row state is not regressed.

Pins: vLLM `/home/mudler/_git/vllm` @ `555967922` (0.26.0.dev0); CUTLASS 4.5.0
(`$HOME/cutlass-4.5.0`, already the project's pinned CUTLASS — supports `Sm90`,
`Sm100` and `Sm120` ArchTags); FlashInfer reference `/home/mudler/_git/flashinfer-ref`;
vLLM-flash-attention fork `ed4b7342` (FA3); local base = `main` HEAD at authorship.

---

## 0. Doctrine — DERIVE-AND-SHIP, not scope-and-wait (user-directed)

We have no H100/H200 (sm_90) and no B100/B200/GB200 (sm_100) board here. The
intent is **NOT** to scope and wait for one. Because the arch-additivity
architecture is built for near-verbatim porting, the plan is the llama.cpp
model:

1. **PORT 1:1** each fast-path kernel that IS faithfully portable, citing the
   exact vLLM + dependency `file:line` it derives from.
2. **BUILD-VERIFY** it HERE on the existing GB10 box — see §1, the key enabler —
   by a single-arch cross-compile that emits real `sm_90a`/`sm_100a` SASS
   (`cuobjdump -lelf`), with the standard clean `-Werror` 0-warn gate.
3. **SHIP it LABELED** `DERIVED+BUILD-VERIFIED (testing-welcome)` — "derived 1:1
   from vLLM upstream, compiled and SASS-verified, NOT hardware-tested here;
   community hardware testing welcome." A board only ever UPGRADES the label to
   `RUNTIME-VERIFIED`; it is never a precondition to ship.
4. For a kernel that is **NOT** faithfully 1:1-portable (runtime JIT / autotune
   codegen, or a body too specialized to port without a real re-implementation),
   say so explicitly and mark it `NOT-YET-BUILDABLE / needs-real-port`. **Never
   ship a fake** (no stub cubin, no hand-waved body).

**Honesty rule (AGENTS.md).** "A green link is not execution evidence." A
DERIVED+BUILD-VERIFIED kernel is a faithful port with a compile+SASS proof and an
HONEST untested LABEL — never a correctness CLAIM. The correctness gate for a
DERIVED kernel is: (a) faithful 1:1 port with cited vLLM+dep `file:line`, (b)
clean `-Werror` compile, (c) real per-arch SASS in the archive. Token-exact vs
the vLLM oracle and every-axis perf are the RUNTIME-VERIFIED gate, gated on a
real board (cloud GPU is the realistic path — see §6).

### SIGNAL taxonomy (the "help wanted: hardware testing" matrix)

Every fast-path/arch cell carries exactly one SIGNAL:

| SIGNAL | Meaning |
|---|---|
| `RUNTIME-VERIFIED` | Ran a gate model token-exact on the real board + every-axis perf vs vLLM. The only "supported" tier. |
| `DERIVED+BUILD-VERIFIED (testing-welcome)` | 1:1 port from vLLM+dep, compiled + SASS-verified here, HONESTLY untested. Ship labeled; help wanted. |
| `NOT-YET-BUILDABLE / needs-real-port` | Codegen/JIT-blocked or too specialized for a faithful 1:1 port; a real re-implementation is required. No fake shipped. |

Today EVERY datacenter fast-path cell is EMPTY (no body). This spec's deliverable
is to move the portable ones to `DERIVED+BUILD-VERIFIED` and label the rest
honestly.

---

## 1. The key enabler — build-verify WITHOUT a datacenter board

A **single-arch** cross-compile on the existing dgx (GB10, nvcc 13.0, cutlass
4.5.0) compiles the `sm_90a` (or `sm_100a`) translation units and emits real
per-arch SASS, with NO Hopper/Blackwell silicon:

```
cmake -S . -B build -DVLLM_CPP_CUDA_ARCHITECTURES=90a \
  -DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0 -DVLLM_CPP_TRITON=OFF
cmake --build build -j
cuobjdump -lelf build/libvllm.a | grep sm_90a   # real sm_90a cubins
```

§W9 already PROVED this for the portable-only `sm_90a` build (16 TUs carrying
real `sm_90a` cubins, `-Werror` 0-warn). Adding a fast-path body widens that to
the fast-path TUs. **This is what makes derive-and-ship real here:** the build +
SASS proof (step 2 of the doctrine) needs only the toolkit + CUTLASS, both
present.

**The one build restriction (known, W7-owned).** The cross-family FAT build
(`"90a;121a"`) still cannot compile, because sources are gencode'd for the whole
`CMAKE_CUDA_ARCHITECTURES` list and `ptxas` rejects the sm12x fp4 PTX for
`compute_90a` (`cuda-arch-additivity.md` §W9, Risks #1). Per-source `-gencode`
narrowing (vLLM `set_gencode_flags_for_srcs`, `cmake/utils.cmake:265-345`) is the
named next row **W7** and is a prerequisite ONLY for a fat multi-arch binary. A
single-arch datacenter build — which is exactly what an H100/B200 owner uses —
does NOT need W7. So derive-and-ship is unblocked; only the "one fat wheel for
every GPU" convenience waits on W7.

---

## 2. Two sub-families, and what is shared vs distinct

- **Hopper `sm_90a`** — 4th-gen tensor core: `wgmma` (warpgroup async MMA) + TMA
  (tensor memory accelerator) + the warp-specialized producer/consumer pipeline.
  CUTLASS expresses this via `ArchTag = cutlass::arch::Sm90` and the
  `KernelTmaWarpSpecialized{Cooperative,Pingpong}` schedules.
- **Datacenter Blackwell `sm_100a`/`103a`/`110`** — 5th-gen tensor core:
  `tcgen05` (tensor-core-gen-05 UMMA with dedicated tensor memory) + TMA + 1SM/2SM
  cluster MMA. CUTLASS expresses this via `ArchTag = cutlass::arch::Sm100` and
  `KernelScheduleAuto` (CUTLASS selects the tcgen05 collective). `103a` shares
  the major-10 bodies; `110` is major-11, same collective family.

**Consumer Blackwell `sm_120a`/`121a` (already ported) vs datacenter `sm_100a` —
shared vs distinct.** This is the sharpest reuse story in the whole spec:

| Aspect | Shared sm_120a ↔ sm_100a | Distinct |
|---|---|---|
| FP4 quant/layout | e2m1 packing + block-scale (SFA/SFB) layout, and the shared quant/experts/KV entry kernels (`nvfp4_quant_kernels.cu`, `activation_nvfp4_quant_fusion_kernels.cu`, `nvfp4_experts_quant.cu`, `nvfp4_kv_cache_kernels.cu` — compiled by BOTH the `FP4_SM120_SRCS` and `FP4_SM100_SRCS` blocks, `CMakeLists.txt:967-984` / `:989-1002`) | — |
| FP4 matmul body | The CUTLASS block-scaled GEMM STRUCTURE (`OpClassBlockScaledTensorOp`, `CollectiveBuilder`, `Sm1xxBlkScaledConfig` SF layout) | `ArchTag = Sm120` (`nvfp4_scaled_mm_sm120_kernels.cu:94`) vs `ArchTag = Sm100` (`nvfp4_scaled_mm_kernels.cu:97`) + different tile/cluster configs (`Fp4GemmSm120` vs `Fp4GemmSm100`) |
| flashinfer kernel | block-scaled GEMM generator shape | `dense_blockscaled_gemm_sm120_b12x.py` ("blackwell_sm12x", mma) vs `dense_blockscaled_gemm_sm100.py` (tcgen05); csrc `fp4_gemm_cutlass_sm120.cu` vs `group_gemm_*_sm100.cu` |

**Consequence:** the datacenter fp4 body is our EXISTING `sm_12x` fp4 kernel with
the ArchTag/tile config swapped `Sm120 → Sm100`, reusing the shared quant/KV
entry kernels verbatim. This is the strongest derive-and-ship candidate — a
re-instantiation of an already-validated body.

---

## 3. Fast-path inventory — vLLM + dependency file:line, portability, SIGNAL

Grid target: the SIGNAL is what each item can become in ONE derive-and-ship pass
(build-verified here) vs. what needs a real port. "1:1?" is the portability
verdict (§4 details it). Every item is EMPTY (no body) today.

### 3a. Hopper `sm_90a` (`BACKEND-CUDA-SM090` + component rows)

| Fast path | vLLM dispatch `file:line` | Dep kernel `file:line` (structure) | 1:1? | Target SIGNAL |
|---|---|---|---|---|
| CUTLASS scaled-mm C3x FP8/INT8 (`SCALEDMM-C3X` sm90 leg) | `CMakeLists.txt:760-772` (`SCALED_MM_SM90_SRCS`); entry `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_fp8.cu:1-33` | `scaled_mm_sm90_fp8_dispatch.cuh:63,106-166` — `cutlass::arch::Sm90`, `KernelTmaWarpSpecialized{Cooperative,Pingpong}FP8FastAccum`, `EpilogueSchedule=TmaWarpSpecialized`; wgmma+TMA inside CUTLASS `CollectiveBuilder` | **YES** — CUTLASS template instantiation; wgmma/TMA live in cutlass 4.5.0, not in vLLM hand-code | `DERIVED+BUILD-VERIFIED` |
| CUTLASS blockwise-fp8 C3x (sm90) | `CMakeLists.txt:764` (`scaled_mm_blockwise_sm90_fp8.cu`) | `c3x/scaled_mm_blockwise_sm90_fp8_dispatch.cuh` — same Sm90 collective, blockwise SF epilogue | **YES** — CUTLASS template | `DERIVED+BUILD-VERIFIED` |
| CUTLASS grouped-MoE C3x (`MOE-CUTLASS` sm90) | `CMakeLists.txt:882-905` (`CUTLASS_MOE_SM90_SRCS = grouped_mm_c3x_sm90.cu`, `9.0a`, CUDA≥12.3) + `moe_data.cu` `:934-951` | `moe/grouped_mm_c3x_sm90.cu` — Sm90 grouped GEMM `CollectiveBuilder`; ptr-array group scheduling | **YES** — CUTLASS grouped-GEMM template | `DERIVED+BUILD-VERIFIED` |
| Machete low-bit mixed-input GEMM (`MACHETE`) | `CMakeLists.txt:485-529` (`MACHETE_ARCHS "9.0a"`, CUDA≥12.0); **Python codegen** `csrc/libtorch_stable/quantization/machete/generate.py` → `generated/*.cu` | generated CUTLASS Sm90 mixed-input instantiations (from `vllm_cutlass_library_extension`) | **PARTIAL** — the generated `.cu` ARE portable CUTLASS instantiations, but they come from a deterministic Python generator; port = vendor the generated sources (or port the generator) | `DERIVED+BUILD-VERIFIED` (generator-vendored; heavier) |
| CUTLASS W4A8 (`W4A8`) | `CMakeLists.txt:1022-1035` (`W4A8_ARCHS "9.0a"`, CUDA≥12.0); `cutlass_w4a8/w4a8_mm_entry.cu`, `w4a8_grouped_mm_entry.cu` | Sm90 mixed W4A8 GEMM `CollectiveBuilder` | **YES** — CUTLASS template; gated on a w4a8 model/quant being ported | `DERIVED+BUILD-VERIFIED` (model-gated) |
| DeepSeek-V3 fused-A GEMM + router (`DSV3`) | `CMakeLists.txt:687-719` (SM90+ family, CUDA≥12) | vLLM csrc fused-A GEMM + grouped-topk router (custom CUDA + cutlass) | **YES-ish** — hand CUDA + CUTLASS; gated on DeepSeek-V3 model | `DERIVED+BUILD-VERIFIED` (model-gated) |
| FlashAttention-3 (Hopper) (`BACKEND-CUDA-COMP-FA` FA3 leg) | gate `vllm/v1/attention/backends/flash_attn.py:122-123` (`fa_version>=3`), `:217` sink-cap `DeviceCapability(9,0)`, `fa_utils.py`; build `cmake/external_projects/vllm_flash_attn.cmake:38-46` (`GIT_TAG ed4b7342`) | vllm-project/flash-attention FA3 — hand-written Hopper wgmma/TMA warp-specialized CuTe kernel (`hopper/` tree) | **YES (large)** — buildable C++/CUDA CuTe, SAME category as the FA2 we already vendored (`src/vt/cuda/flash_attn/`); a LARGE vendoring effort, not a codegen block | `DERIVED+BUILD-VERIFIED` (large; the highest-value Hopper decode/prefill lever) |
| DeepGEMM fp8 blockwise (`DEEPGEMM` sm90 leg) | `cmake/external_projects/deepgemm.cmake:61-105` (`9.0a`); runtime `vllm/utils/deep_gemm.py:189,250,661` (JIT cache + `@torch.compile`) | DeepGEMM = **runtime NVRTC/JIT codegen + autotune** per shape | **NO** — JIT/autotune codegen; eager-C++ cannot 1:1 replicate (AGENTS.md names DeepGEMM-codegen explicitly). A faithful port = a from-scratch blockwise-fp8 GEMM, not a 1:1 port | `NOT-YET-BUILDABLE / needs-real-port` |
| FlashMLA sm90 dense+sparse (`FLASHMLA` sm90 leg) | `cmake/external_projects/flashmla.cmake:70-100` (sm90 dense/sparse decode + sparse prefill, `9.0a` CUDA≥12.3) | FlashMLA `csrc/sm90/**` — hand-written CuTe MLA kernels | **YES (large, model-gated)** — buildable CuTe, gated on a DeepSeek-MLA model using the FlashMLA path | `DERIVED+BUILD-VERIFIED` (large; model-gated) |

### 3b. Datacenter Blackwell `sm_100a`/`103a`/`110` (SM100/103/110 + component rows)

| Fast path | vLLM dispatch `file:line` | Dep kernel `file:line` (structure) | 1:1? | Target SIGNAL |
|---|---|---|---|---|
| CUTLASS scaled-mm C3x FP8 (`SCALEDMM-C3X` sm100 leg) | `CMakeLists.txt:820-837` (`SCALED_MM_SM100_SRCS`, `10.0a;10.1a;10.3a` / CUDA13 `10.0f;10.7f;11.0f`) | `c3x/scaled_mm_sm100_fp8_dispatch.cuh:63,103` — `cutlass::arch::Sm100`, `KernelScheduleAuto`/`EpilogueScheduleAuto` (CUTLASS picks the **tcgen05** collective) | **YES** — CUTLASS template; tcgen05 lives in cutlass 4.5.0 | `DERIVED+BUILD-VERIFIED` |
| CUTLASS blockwise-fp8 C3x (sm100) | `CMakeLists.txt:824` (`scaled_mm_blockwise_sm100_fp8.cu`) | `c3x/scaled_mm_blockwise_sm100_fp8_dispatch.cuh` — Sm100 collective + blockwise SF | **YES** — CUTLASS template | `DERIVED+BUILD-VERIFIED` |
| CUTLASS grouped-MoE C3x (`MOE-CUTLASS` sm100) | `CMakeLists.txt:918-940` (`CUTLASS_MOE_SM100_SRCS = grouped_mm_c3x_sm100.cu`) + shared `moe_data.cu` | `moe/grouped_mm_c3x_sm100.cu` — Sm100 grouped GEMM tcgen05 collective | **YES** — CUTLASS grouped-GEMM template | `DERIVED+BUILD-VERIFIED` |
| NVFP4 scaled-mm sm100 (tcgen05) (`FP4` sm100 leg) | `CMakeLists.txt:989-1002` (`FP4_SM100_SRCS = nvfp4_scaled_mm_kernels.cu` + shared quant/experts/KV entries) | `fp4/nvfp4_scaled_mm_kernels.cu:76-97` — `struct Fp4GemmSm100`, `ArchTag=Sm100`, `OpClassBlockScaledTensorOp`, `Sm100BlkScaledConfig` SF layout | **YES (strongest)** — our EXISTING `sm_120a` fp4 body (`Fp4GemmSm120`, `nvfp4_scaled_mm_sm120_kernels.cu:77-95`) with `ArchTag` swapped Sm120→Sm100 + tile config; shared quant/KV entries reused verbatim (§2) | `DERIVED+BUILD-VERIFIED` |
| MXFP4 experts + blockwise MoE (sm100) | `CMakeLists.txt:989-1002` (`mxfp4_experts_quant.cu`, `mxfp4_blockwise_moe_kernel.cu`; stubs before CUDA 12.9) | Sm100 MXFP4 CUTLASS block-scaled | **YES** — CUTLASS template (needs CUDA≥12.9 for real body, else stub — nvcc 13.0 here is fine) | `DERIVED+BUILD-VERIFIED` |
| CUTLASS MLA sm100 (`MLA`) | `CMakeLists.txt:1054-1074` (`CUTLASS_MLA_SRCS = sm100_cutlass_mla_kernel.cu`, `10.0a;10.1a;10.3a`); runtime gate `vllm/v1/attention/backends/mla/cutlass_mla.py` | `attention/mla/sm100_cutlass_mla_kernel.cu` + cutlass `examples/77_blackwell_fmha` (tcgen05 FMHA) | **YES (model-gated)** — CUTLASS-based; gated on an MLA model routing to CUTLASS-MLA (DeepSeek-V2/V3, GLM-4.7 already ported use the portable MLA block) | `DERIVED+BUILD-VERIFIED` (model-gated) |
| FlashMLA sm100 dense+sparse (`FLASHMLA` sm100 leg) | `cmake/external_projects/flashmla.cmake:101-120` (sm100 dense prefill/backward + sparse fwd/decode, `10.0a`/`10.0f` CUDA≥12.8/12.9) | FlashMLA `csrc/sm100/**` — hand-written tcgen05 CuTe MLA | **YES (large, model-gated)** — buildable CuTe | `DERIVED+BUILD-VERIFIED` (large; model-gated) |
| DeepGEMM fp8 blockwise (`DEEPGEMM` sm100 leg) | `deepgemm.cmake:61-105` (`10.0a`/`10.0f`) | runtime NVRTC/JIT + autotune | **NO** — same JIT-codegen block as §3a | `NOT-YET-BUILDABLE / needs-real-port` |

---

## 4. Portability honesty — 1:1-portable vs codegen-blocked

**The load-bearing finding: the CUTLASS C3x / FP4 / MoE / MLA kernels are
CUTLASS TEMPLATE INSTANTIATIONS, not vLLM hand-code.** vLLM writes a thin
dispatch wrapper (`cutlass_scaled_mm_sm90_fp8(...)`, `Fp4GemmSm100<...>`), and the
actual `wgmma` (Sm90) / `tcgen05` (Sm100) MMA + TMA pipeline is emitted by
CUTLASS's `CollectiveBuilder` selected by `ArchTag`. Since the project already
carries **cutlass 4.5.0** (which supports `Sm90`, `Sm100`, `Sm120` ArchTags),
these are genuinely 1:1-portable: port vLLM's dispatch wrapper, instantiate the
CUTLASS template with the right ArchTag/schedule, re-express the tensor ABI in
`vt::`. They compile and emit real SASS for the target — the derive-and-ship
sweet spot. Our existing `sm_121a` C3x FP8 (`cuda_matmul_fp8_cutlass.cu`,
`ArchTag=Sm120`) and NVFP4 (`cuda_matmul_nvfp4.cu`) prove the pattern already
runs in-tree; the datacenter legs are the same pattern with a different ArchTag.

**Genuinely codegen/JIT-blocked (mark honestly, ship no fake):**

- **DeepGEMM** (both sm90 and sm100 legs) — runtime NVRTC JIT + autotune +
  `@torch.compile` (`deep_gemm.py:189,250,661`). Eager C++ cannot 1:1 replicate a
  runtime code generator + tactic autotuner. AGENTS.md names this class
  explicitly. A faithful port is a from-scratch blockwise-fp8 GEMM (a kernel
  campaign), NOT a mechanical 1:1 — so `NOT-YET-BUILDABLE / needs-real-port`.

**Codegen-adjacent (portable, but heavier than a header instantiation):**

- **Machete** — the kernels themselves are CUTLASS Sm90 mixed-input
  instantiations, but they are emitted by a deterministic Python generator
  (`generate.py` + `vllm_cutlass_library_extension`). Two honest options: (a)
  vendor the GENERATED `.cu` sources 1:1 (portable, build-verifiable, brittle to
  a schedule change), or (b) port the generator. Lower priority than the direct
  C3x/FP4 wins; `DERIVED+BUILD-VERIFIED` once vendored.

**Buildable but large hand-kernels (not codegen, but a big vendoring lift):**

- **FA3** (Hopper) and **FlashMLA** (sm90 + sm100) — hand-written CuTe kernels,
  the SAME category as the FA2 body we already vendored under
  `src/vt/cuda/flash_attn/`. Faithfully portable by vendoring the dependency
  source and building it for the single target arch; `DERIVED+BUILD-VERIFIED`,
  but each is a large multi-file lift and FA3 is the single highest-value Hopper
  throughput lever (its absence is why a Hopper owner would get only portable
  attention today). FlashMLA is additionally MLA-model-gated.

---

## 5. Hardware-per-target — what unlocks each RUNTIME-VERIFIED upgrade

Cloud GPU is the realistic path for every datacenter arch (none is ownable on
this project's hardware).

| Row / target | Silicon that runtime-verifies | Realistic access |
|---|---|---|
| `BACKEND-CUDA-SM090` (`sm_90a`) | H100, H200 (Hopper) | Cloud (Lambda / RunPod / CoreWeave / GCP a3 / AWS p5) — widely available |
| `BACKEND-CUDA-SM100` (`sm_100a`) | B100, B200, GB200 (datacenter Blackwell) | Cloud (emerging: Lambda/CoreWeave B200, GCP a4, AWS p6) |
| `BACKEND-CUDA-SM103` (`sm_103a`) | B200-class major-10 variant | Cloud (as B200 fleets land) |
| `BACKEND-CUDA-SM110` (`sm_110`, major 11) | Blackwell major-11 (Thor / DRIVE-class) | Cloud/dev-kit (scarce; lowest priority) |
| `BACKEND-CUDA-SM120`/`121` (`sm_12x`, already ported) | RTX 50-series consumer Blackwell | Owned (GB10) + external RTX 5070 Ti evidence |

A RUNTIME-VERIFIED upgrade for any row is the four-step owner protocol from
`cuda-arch-additivity.md` §W9: single-arch build on the board → unit tier →
gate model end-to-end vs its golden (token counts reported) → every-axis perf
vs the vLLM oracle. Only step 3+ moves a cell past `DERIVED+BUILD-VERIFIED`.

---

## 6. Gate design

### 6a. DERIVED+BUILD-VERIFIED gate (achievable HERE, no board)

Per derived kernel, on dgx (GB10, nvcc 13.0, cutlass 4.5.0), single-arch build:

| Gate | Requirement |
|---|---|
| Faithful port | vLLM + dep `file:line` cited in the TU header (ground-every-impl rule); tensor ABI re-expressed, algorithm unchanged |
| Configure | FEATURE-TABLE cell for the arch widened; `-DVLLM_CPP_CUDA_ARCHITECTURES=<arch>` resolves the feature ENABLED for `[<arch>]` with no missing-tactic warning |
| Compile | clean full `-Werror` 0-warn single-arch build |
| SASS | `cuobjdump -lelf libvllm.a` shows the fast-path TU carrying a real `sm_<arch>` cubin |
| Tactic seam | the registered tactic SELECTS for a synthetic `DeviceCaps{sm_major=9 or 10}` (registry counter moves — a passing build is not proof the path is reachable; mirror §W8's `VT_ARCH_TACTIC_STATS` counter assertion) |
| Inertness | GB10 `sm_121a` production build byte-identical (the new body is arch-gated OFF on 12x); SACRED gates 27B 235/235, 35B 315/315 unchanged |
| LABEL | shipped as `DERIVED+BUILD-VERIFIED (testing-welcome)`, never as "supported" |

### 6b. RUNTIME-VERIFIED gate (board-gated; cloud)

The full acceptance bar (AGENTS.md): correctness FIRST (token-exact 16/16 vs the
pinned vLLM oracle on the identical workload), then every-axis perf (total +
output throughput, req/s ≥ vLLM; TTFT, TPOT/ITL, peak memory ≤ vLLM), ≥2-3 reps
on an idle board, same-binary A/B, `nsys` of BOTH stacks with
`--cuda-graph-trace=node`. SGLang floor where applicable. A board that runs a
gate model token-exact upgrades the row's SIGNAL and STATE.

---

## 7. Tests to port

Upstream tests are the executable spec; port or check in a traced skip (never
silently omit). Board-gated cases are checked in SKIPPED with a hardware reason.

| Rows | Upstream tests | Local tier |
|---|---|---|
| C3x sm90/sm100 FP8/INT8 + blockwise | `tests/kernels/quantization/test_cutlass_scaled_mm.py`, `test_scaled_mm_kernel_selection.py` | op parity vs a CPU/host reference (build-verifiable); numerical parity SKIP-until-board |
| FP4 sm100 (tcgen05) | `tests/kernels/quantization/test_nvfp4_scaled_mm.py`, `test_nvfp4_quant.py` | reuse the existing `tests/vt/test_ops_nvfp4_fp4.cpp` structure with an `Sm100` tactic; parity SKIP-until-board |
| grouped-MoE sm90/sm100 | `tests/kernels/moe/test_cutlass_moe.py` | grouped-MoE op parity (mirror `test_ops_moe_grouped.cpp`); SKIP-until-board for numerics |
| Machete | `tests/kernels/quantization/test_machete_mm.py` | mixed-input GEMM parity; compile-only until vendored+board |
| FA3 (Hopper) | `tests/kernels/attention/test_flash_attn.py` (fa_version 3 arm) | selector + FA3 numerics; SKIP-until-Hopper |
| CUTLASS MLA / FlashMLA | `tests/kernels/attention/test_flashmla.py`, `test_cutlass_mla_decode.py` | MLA decode parity; model + board-gated |
| arch dispatch | `tests/kernels/attention/test_attention_selector.py`, `tests/v1/attention/test_attention_backends_selection.py` | capability→backend priority unit table (build-verifiable now) |
| W4A8 | `tests/kernels/quantization/test_w4a8.py` (where present) | W4A8 GEMM parity; model + board-gated |

The DeepGEMM path has no 1:1 test to port until it is re-implemented; its row
carries the `needs-real-port` reason, not a skipped stub.

---

## 8. Dependencies

- CUTLASS 4.5.0 (present; `Sm90`/`Sm100`/`Sm120` ArchTags) — the whole
  C3x/FP4/MoE/MLA story rides on it.
- `cuda-arch-additivity.md` seams (`BACKEND-CUDA-ARCH-ADDITIVITY`): FEATURE
  TABLE, device-capability probe, tactic registry — all landed on `sm_121a`.
- **W7 (per-source `-gencode` narrowing)** — REQUIRED only for a cross-family FAT
  binary; NOT required for the single-arch datacenter builds this spec ships.
- FA3: vLLM-flash-attention fork `ed4b7342` source to vendor (large).
- FlashMLA + CUTLASS MLA: source to vendor; plus an MLA model routed to the
  CUTLASS/FlashMLA path (DeepSeek-V2/V3 / GLM-4.7 currently use the portable MLA).
- DeepGEMM: no faithful 1:1 dependency to vendor — a real re-implementation.
- Board access (cloud) for every RUNTIME-VERIFIED upgrade.
- The quantization/model matrices for the representative format/model gates
  (fp8 / nvfp4 / w4a8 / MLA) each datacenter leg needs to exercise its path.

---

## 9. Work breakdown

Ranked by value ÷ effort; every "port" item is derive-and-ship (port → build+SASS
verify HERE → ship `DERIVED+BUILD-VERIFIED`), then a board upgrades the label.

| # | Item | Rows | Effort | Notes |
|---|---|---|---|---|
| DC1 | **NVFP4 scaled-mm sm100 (tcgen05)** — swap our `sm_12x` fp4 body's ArchTag Sm120→Sm100 + tile config; reuse shared quant/KV entries | `SM100`, `FP4` (sm100 leg) | **S** | Strongest reuse (§2); we already own the sm120 body |
| DC2 | **CUTLASS scaled-mm C3x FP8/INT8 + blockwise, sm90 AND sm100** | `SM090`, `SM100`, `SCALEDMM-C3X` | S–M | Pure CUTLASS ArchTag instantiation both families; mirror our sm121 C3x FP8 |
| DC3 | **CUTLASS grouped-MoE C3x, sm90 + sm100** (+ shared `moe_data.cu`) | `SM090`, `SM100`, `MOE-CUTLASS` | M | Grouped-GEMM CUTLASS template; exercises the MoE gate models |
| DC4 | **MXFP4 experts + blockwise MoE, sm100** | `SM100`, `FP4` | S–M | CUTLASS block-scaled; nvcc 13.0 gives the real body (≥12.9) |
| DC5 | **FA3 (Hopper) vendoring** — the highest-value Hopper throughput lever | `SM090`, `BACKEND-CUDA-COMP-FA` | **L** | Large CuTe vendoring (FA2-category); selector already gates `fa_version>=3` |
| DC6 | **Machete** (vendor generated sm90 CUTLASS sources) | `SM090`, `MACHETE` | M–L | Codegen-adjacent; vendor the generated `.cu` |
| DC7 | **CUTLASS MLA sm100 + FlashMLA sm90/sm100** | `SM100`, `MLA`, `FLASHMLA` | L | Model-gated (MLA models currently use the portable block); large vendoring |
| DC8 | **W4A8 sm90 + DSV3 fused-A GEMM** | `SM090`, `W4A8`, `DSV3` | M–L | Model/quant-gated (w4a8, DeepSeek-V3) |
| DC9 | **DeepGEMM** — RE-IMPLEMENT (not a 1:1 port) or defer | `DEEPGEMM` | XL | `NOT-YET-BUILDABLE`; a from-scratch blockwise-fp8 GEMM campaign |
| DC10 | **W7 per-source gencode narrowing** — unlocks the one-fat-wheel build | `BACKEND-CUDA-ARCH-ADDITIVITY` | M | Only needed for a multi-arch binary; single-arch ships without it |
| DC11 | **RUNTIME-VERIFIED upgrades** on cloud H100/B200 | all datacenter rows | per-board | Token-exact gate + every-axis perf; upgrades SIGNAL/STATE |

---

## 10. Risks / decisions

1. **CUTLASS ArchTag is necessary but the tile/cluster config is load-bearing.**
   An `Sm100` collective with a bad tile shape compiles but is slow or invalid;
   the derive-and-ship port must carry vLLM's exact tile/cluster/schedule choices
   (`Fp4GemmSm100` config structs, C3x dispatch heuristics), not just the ArchTag.
2. **DeepGEMM stays honest.** Do NOT ship a stub or a mislabeled from-scratch
   kernel as "DeepGEMM". It is `NOT-YET-BUILDABLE`; a real blockwise-fp8 GEMM is a
   separate scoped campaign (DC9).
3. **FA3/FlashMLA size.** These are large vendoring lifts; a partial vendoring
   that does not compile is not shippable. Each ships only once it build-verifies
   `-Werror` clean with real SASS — otherwise it stays EMPTY, honestly.
4. **A build+SASS proof is NOT a correctness claim.** The LABEL discipline (§0) is
   the whole point: `DERIVED+BUILD-VERIFIED` is faithful-port + compile + SASS, no
   more. Only a board's token-exact run earns `RUNTIME-VERIFIED`.
5. **The fat cross-family build stays blocked (W7).** Shipping single-arch
   datacenter binaries is unaffected; the "one wheel for every GPU" convenience is
   the only thing that waits.
6. **No product call reopened.** Where vLLM has an answer (which arch gets which
   collective, the selector priority, the tile configs) it is mirrored, not
   re-decided.
