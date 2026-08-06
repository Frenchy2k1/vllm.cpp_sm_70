# Spike: close the 16-bit CPU GEMM gap, wide x86 ISA tiers + a tiled sgemm

Status: SPIKE (records only). No row goes `ACTIVE` until claimed in
`coordination.md`.

Grounded in the 2026-08-06 attribution in `.agents/benchmark-record.md` ("Why
vt's 16-bit CPU GEMM trails ggml"), which measured the cause instead of assuming
it: on Arm our kernel is AT PARITY with ggml's stock kernel and the whole
deficit is llamafile's tinyBLAS; on x86 there is a second, larger cause, namely
that our x86 tier is SSE2 while the hardware offers AVX2/AVX-512.

## Scope

**In**, two independent rows:

| Row ID | What | Measured headroom |
|---|---|---|
| `KERNEL-GEMM-CPU-ELEM-X86WIDE` | AVX2 and AVX-512 elementwise tiers for `cpu_matmul_elem`, runtime-dispatched | ~3.5x on x86 |
| `KERNEL-GEMM-CPU-TILED` | tinyBLAS-style M x N register-tiled sgemm, all architectures | ~1.9x Arm 16-bit, ~2.4x x86 |

**Out.** Quantized kernels (that is G5 and G8, a different family). The decode
threadpool cost (47% of decode per the same-day profile) which neither row
touches. Metal and Vulkan.

**Dispatch behavior.** Unchanged from today: `BuildTier()` returns one
`ElemGemmTierTable` of function pointers chosen once, at first use. New tiers
slot in as additional candidates; the portable tier stays the default and the
`VT_CPU_MATMUL_TIER` defeat switch keeps working, gaining `avx2` and `avx512`
as forceable values alongside `portable` and `ref`.

## Upstream chain

llama.cpp @ `237ad9b96`:

- `ggml/src/ggml-cpu/llamafile/sgemm.cpp` is the tiled GEMM whose absence the
  controls isolated. It tiles over M and N and loops K, which is the property
  that matters to us (see Risks).
- `ggml/src/ggml-cpu/arch/x86/quants.c` and the `ggml-cpu` per-variant build are
  the ISA-tier precedent.
- `ggml/src/CMakeLists.txt:371-401` `GGML_CPU_ALL_VARIANTS` builds the CPU
  backend once per ISA level (`haswell`, `skylakex`, `icelake`, `zen4`,
  `sapphirerapids`, ...) and selects at runtime via `GGML_BACKEND_DL`.

**Build-style decision, recorded.** We adopt llama.cpp's MODEL (same kernel
built at several ISA levels, chosen at runtime) but not its MECHANISM
(N shared-library backend variants behind `GGML_BACKEND_DL`), because we ship
one static library and one binary and have no backend-DL infrastructure. The
mechanism is our own existing surgical one: a separate TU per ISA level with
per-file `COMPILE_OPTIONS`, exactly as `CMakeLists.txt:744-760` already does for
the Arm i8mm tier, whose comment explicitly cites the x86 tier as its model.
This yields the same runtime-dispatch outcome with no new build machinery.

## Our baseline

- `src/vt/cpu/cpu_matmul_elem.cpp:226-231`, the x86 tier: SSE2 only, `MR=2`
  (16 XMM registers), plus a probed F16C pair for f16 conversion. The file's own
  comment says "The wider AVX2/AVX512 quant tier remains work row G5".
- `:80` the Arm tier: NEON, `MR=4` (`kMrNeon`), already at ggml-stock parity.
- `:432-491` `BuildPortableTier()` / `BuildTier()`, the runtime probe and tier
  table this work extends.
- `tests/vt/test_ops_matmul_elem.cpp`: 5 cases / 654 assertions, gate is
  `memcmp` byte-identity against an in-test scalar reference, green under all
  `VT_CPU_MATMUL_TIER` settings. This is the gate the new tiers must pass
  unchanged.

Measured now (GFLOP/s, f16 weight + f32 activations, conformer shapes):

| | vt | ggml no-llamafile | ggml stock |
|---|---:|---:|---:|
| Arm `131,2048,512` | 220.6 | 214.4 | 419.8 |
| x86 `131,2048,512` | 166.6 | 587.0 | 1403.2 |

## Port map

| Upstream idea | Local | Notes |
|---|---|---|
| ggml per-variant ISA build | new `src/vt/cpu/cpu_matmul_elem_avx2.cpp` | `-mavx2 -mfma -mf16c`, exports `BuildAvx2Tier()` |
| same | new `src/vt/cpu/cpu_matmul_elem_avx512.cpp` | `-mavx512f -mavx512bw -mavx512vl -mfma -mf16c`, exports `BuildAvx512Tier()` |
| CMake per-file flags | `CMakeLists.txt`, next to the i8mm block | Guarded on `CMAKE_SYSTEM_PROCESSOR MATCHES x86_64` |
| runtime selection | `cpu_matmul_elem.cpp` `BuildTier()` | `__builtin_cpu_supports("avx512f")` then `("avx2")`, widest wins; both TUs always compiled, never executed unless probed |
| `llamafile/sgemm.cpp` tiling | new `btm`-family kernels, per tier | Raise `MR` and add an `NR` dimension; no new op, no new seam |

## Tests to port

| Upstream | Local | Note |
|---|---|---|
| ggml `tests/test-backend-ops` MUL_MAT cases | already mirrored in `test_ops_matmul_elem.cpp` | Extend its tier loop to `avx2` and `avx512` so every existing case runs on every built tier |
| none (new) | `test_ops_matmul_elem.cpp` exhaustive f16 case | Already proves all 65,536 f16 patterns; must stay green on the wide tiers |

No new test FILE is required, and that is deliberate: the existing battery
already gates byte-identity, so the correct move is to widen its tier sweep
rather than write a parallel suite.

## Gates

- **Correctness, and it is the whole gate here:** `memcmp` byte-identity against
  the in-test scalar reference, on every tier the box can run, at thread counts
  1/2/4/8, including ragged K and N. **Byte-identity is achievable by
  construction, not by luck**, see Risks.
- **e2e:** SACRED regression set unchanged, goldens md5 identical before and
  after, on dgx.
- **Performance:** dgx aarch64 for the tiled row, one `flock`, same binary, 3
  reps, medians, idle box. **The x86 row cannot be speed-gated**: the x86 dev
  box is VOID for timing per `CLAIM-KERNEL-CPU-ELEM-GEMM-1`, so its numbers are
  reported as INDICATIVE and no ratio is binding until a qualified x86 host
  exists. This is a real limitation of the x86 row and is stated up front rather
  than discovered at review.
- **Build:** clean `-Werror` on x86-64, aarch64 and CUDA. The AVX-512 TU must
  compile and link on a machine WITHOUT AVX-512 and must not be entered there.

## Dependencies

- Rows: none. Both are independent of G5, G8 and of each other.
- Toolchain: GCC/Clang with `-mavx512*`; no new third-party.
- Hardware: an AVX-512 box to exercise that tier at all. The dev box reports
  `avx512f` and `avx512_bf16`, so the tier is testable here even though it is
  not timeable here. dgx cannot test the x86 tiers at all.
- Data/models: none for the unit gate.

## Work breakdown

| # | Row | Owns | Notes |
|---|---|---|---|
| W1 | `KERNEL-GEMM-CPU-ELEM-X86WIDE` | `cpu_matmul_elem_avx2.cpp` + CMake block + `BuildTier` probe | AVX2 first, self-contained |
| W2 | same row | `cpu_matmul_elem_avx512.cpp` | After W1 proves the seam |
| W3 | `KERNEL-GEMM-CPU-TILED` | `btm` kernels across portable/NEON/AVX2 | Independent of W1/W2, but lands cleaner after them |
| W4 | both | extend `test_ops_matmul_elem.cpp` tier sweep | Can land first, and should |

W4 first is the deliberate ordering: widen the gate before widening the kernels,
so the new tiers are born under the byte-identity check rather than retrofitted
into it.

## Risks and decisions

**The byte-identity argument, which is what makes this safe.** `Bt16Sse2`
(`cpu_matmul_elem.cpp:246-276`) computes `kElemLanes`=16 outputs as 4 groups of
4 lanes. Within a group it applies K elements `p..p+3` in strict order through
`_mm_shuffle_ps(av, av, 0x00/0x55/0xAA/0xFF)` after a 4x4 transpose, so **every
output keeps a strictly sequential K reduction**. That is the invariant behind
the E1-E4 byte-identity claim, and it constrains this work precisely:

> Widening MUST add output lanes (or M rows). It MUST NOT split the K reduction
> across accumulators.

Under that rule 16 outputs become 2 groups of 8 on AVX2 and 1 group of 16 on
AVX-512, K order untouched, and the result is byte-identical to the portable
tier BY CONSTRUCTION. The same rule applies to W3: tinyBLAS tiles M and N and
loops K, which is compatible; any tiling that splits K is not, and is rejected.

**Open engineering risk.** The 8x8 and 16x16 transposes are more expensive than
`_MM_TRANSPOSE4_PS`. If a transpose-based layout does not pay, the alternative
is to broadcast the activation and FMA into per-output accumulators, which needs
the weight's output dimension contiguous and it is not ([N,K] row-major, outputs
strided by K). Resolving that tradeoff is implementation work, and the memcmp
gate arbitrates correctness either way while a same-binary A/B arbitrates speed.

**Product calls: none.** Nothing here is vLLM-defined behavior; these are pure
kernel tiers under an existing op contract.

**Bounding the payoff honestly.** The same-day op-dispatch profile puts the
elementwise GEMM at 21.5% of prefill and 24.9% of decode on the GGUF bench
workload. A 1.9x on that term is therefore well under 1.9x end to end
(~1.15x prefill by Amdahl), and neither row touches the 47% decode threadpool
cost. These rows are worth doing and they are not a fix for CPU decode.
