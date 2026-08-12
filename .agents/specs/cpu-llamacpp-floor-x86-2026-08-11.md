# CPU vs llama.cpp — the x86_64 arm of the B4 floor (2026-08-11)

**Rows:** `BACKEND-GATE-CPU-LLAMACPP`, `BACKEND-CPU` ([backend
matrix](../backend-matrix.md)) · `QUANT-GGUF-COMPUTE`
([quantization matrix](../quantization-matrix.md)) ·
`ROAD-V1-D1` punch-list item 13, CPU half
([roadmap-v1-completion](roadmap-v1-completion.md)) ·
**issue:** [#433](https://github.com/mudler/vllm.cpp/issues/433) ·
**claim:** `CLAIM-CPU-X86-FLOOR-1` · **base:** `31a2b493`.

## Scope

Measure the **x86_64** arm of `BACKEND-GATE-CPU-LLAMACPP` on the B4 vehicle,
establish correctness first, and record every axis with its value **and** its
ratio. Reconcile the two matrix cells that still describe the pre-`G4` position
as current.

**In scope**

- One binding same-file, same-box A/B: vllm.cpp vs llama.cpp on
  `Qwen3.5-2B-UD-Q8_K_XL.gguf`, single stream, x86_64.
- Correctness before speed: same prompt, greedy, same token count, compared as
  text between the two engines.
- Three axes: prefill throughput, decode throughput, peak RSS. Plus E2E latency
  as the Pi arm records it.
- Record updates: `BACKEND-GATE-CPU-LLAMACPP`, `BACKEND-CPU`, the stale
  `QUANT-GGUF` / `QUANT-GGUF-COMPUTE` cells, `docs/BENCHMARKS.md`, `NOW.md`,
  the roadmap issue table, and a `docs/bench-evidence/` file.

**Explicitly out of scope**

- **No GPU.** Nothing in this row touches CUDA, and no leg queues on
  `$HOME/gpu.lock`.
- **No optimization.** This is measurement plus record repair. If an axis is
  below floor it is recorded as an open gap with a named next hypothesis, not
  fixed here.
- The **Metal/MLX half** of punch-list item 13. It needs an Apple M4, which
  this host is not.
- The RPi5 / Cortex-A76 arm (issue #284) and server-concurrency operating
  points. Both are separately owned and separately open.

## Upstream chain

llama.cpp is the competitor floor for the CPU/GGUF lane, not vLLM: pinned vLLM
has no GGUF load format. The executing chain on the oracle side is
`ggml/src/ggml-cpu/quants.c` (portable q8 dot),
`ggml/src/ggml-cpu/arch/x86/quants.c` (the AVX2/AVX-512 q8 dot this host will
actually run), `ggml/src/ggml-cpu/repack.cpp` (repack tiers) and
`src/models/qwen35.cpp`. Our side is `src/vt/cpu/cpu_backend.cpp`,
`src/vt/cpu/cpu_threadpool.cpp`, `src/vt/cpu/cpu_ops.cpp`, the
`kMatmulBTQuant` route in `src/vt/ops.cpp`, and
`src/vllm/model_executor/model_loader/gguf_keep_quant.cpp`.

## Our baseline

The record's CPU position is **ISA-split and only half measured**:

| Arm | Recorded position | Source |
|---|---|---|
| 20-core AArch64 + i8mm (`dgx.casa`) | **closed** — RSS 1.01x, prefill 1.18-1.26x ahead, decode parity | `BACKEND-GATE-CPU-LLAMACPP` |
| 4-core Cortex-A76 (RPi5), no i8mm | **open** — 0.461x prefill, 0.653x decode/E2E, 0.758x RSS | [Pi evidence](../../docs/bench-evidence/rpi5-a76-llamacpp-20260806.md) |
| **x86_64** | **none since 2026-07-10** | ledger B4 row |

Every lever that closed the first arm is Arm-scoped or Arm-measured: `CIQ` G6
(i8mm quant tier), `CIQ` G7 (q8_0 repack-at-load, the prefill crossing),
`KEEPQ` L7 (whose residual was an Arm repack-source double-count) and
`KERNEL-CPU-A76-Q8-DOT` (AArch64 SDOT). The only x86 numbers on file — 54-75x
decode, ~1,480x prefill, 2.7x RSS behind — predate the whole compute-in-quant
track, so they cannot be quoted as current and cannot be assumed stale either.

## Port map

Nothing is ported. This row executes an existing harness
(`examples/bench/main.cpp` → `vllm-bench`) against an existing pinned oracle
(`llama-bench`, `llama-cli`) and writes down what happens.

## Design

Single binding series on one idle host, interleaved so that neither engine owns
a quiet window the other does not get.

1. Build vllm.cpp CPU-only Release from the pinned base, on this host, with the
   recorded B4 flags. No CUDA, no tests, no server.
2. Reuse the already-present llama.cpp build at the recorded pin — same
   `CMAKE_BUILD_TYPE=Release`, `GGML_CUDA=OFF`, `GGML_NATIVE=ON`, OpenMP on.
3. Correctness leg first. Both engines, same raw prompt, greedy, 64 output
   tokens, compared as text. A speed number is not accepted until this passes.
4. Speed legs: three clean vllm.cpp process repetitions and one llama.cpp
   `-r 3` in-process series, interleaved, `taskset`-pinned to the same cores
   with the same thread count.
5. `uptime` recorded immediately before and immediately after **every** leg.
   Any leg taken above the declared load ceiling is discarded and re-run, not
   averaged in.

The comparison uses medians, and the spread across repetitions is reported so a
reader can see whether a stated gap survives the noise floor.

## Tests to port

None. This row adds no product code, so it adds no unit test. Its evidence is
the measurement file plus the record edits, and its "red before" is the absence
of any current x86_64 number.

## Gates

| # | Gate | Result form |
|---|---|---|
| G1 | Correctness: both engines emit the same greedy text for the same prompt at the same token count | pass / fail — **blocks every speed number** |
| G2 | Prefill throughput ratio, ours / llama.cpp | value + ratio |
| G3 | Decode throughput ratio, ours / llama.cpp | value + ratio |
| G4 | Peak RSS ratio, ours / llama.cpp | value + ratio |
| G5 | Every leg taken at load average below the declared ceiling, recorded before and after | recorded |

G2-G4 are **not** pass/fail for this row: the row's job is to produce the
number honestly. Any axis at or above 1.00x in our favour is met; any axis
below is recorded as an **open gap** on `BACKEND-GATE-CPU-LLAMACPP` with a
named next traceable hypothesis. Rounding an axis up, or reporting one
"pending" because it came out badly, fails the row.

## Dependencies

- Vehicle `Qwen3.5-2B-UD-Q8_K_XL.gguf`, present on this host.
- llama.cpp build at the recorded pin, present on this host.
- An idle x86_64 box. This one is shared and hit load 240 earlier today, which
  is why G5 exists.

## Work breakdown

| W | Item | State |
|---|---|---|
| W0 | Issue, spec, worktree, role claim | this commit |
| W1 | CPU-only Release build from base `31a2b493` | |
| W2 | G1 correctness leg | |
| W3 | G2-G4 interleaved binding series, G5 load discipline | |
| W4 | Evidence file under `docs/bench-evidence/` | |
| W5 | Record reconciliation: backend matrix, the two stale quant cells, `BENCHMARKS`, `NOW`, roadmap issue table | |

## Risks / decisions

- **The box is a 20-vCPU KVM guest, not bare metal.** Host-side contention is
  invisible from inside. Mitigated by the load ceiling, by interleaving the two
  arms, and by reporting spread; not eliminated. Recorded as a property of the
  measurement, not hidden.
- **The local vehicle is not byte-identical to the Pi arm's file.** Same name
  and quantization, different bytes. That is fine for this row — the binding
  requirement is that *both engines in this A/B read the same file* — but it
  means this arm's absolute numbers are not directly comparable to the Pi's.
  Both hashes are recorded.
- **A bad result is the expected outcome and is not a reason to stop.** The
  Arm-specific levers give a concrete prior that x86 is behind. If it is, that
  is the finding, and the gate stays open with a hypothesis.
- **No ceiling may be declared.** If x86 lands behind, the next lever is named
  in the Outcome, not written off as an ISA limit.

## Evidence

`docs/bench-evidence/cpu-x86-llamacpp-20260811.md` — commands, binary hashes,
model hash, per-repetition numbers, load averages, and the correctness hashes.

## Stop conditions

- Correctness leg fails → stop, report, do not publish a speed number.
- The box cannot be brought below the load ceiling → stop and report the axis
  as pending a quiet host, naming it. Do not average through contention.
- The work turns out to need the GPU → return `NEEDS_DECISION`; another session
  holds `$HOME/gpu.lock`.

## Now

**Peak RSS MET at 1.0022x (parity); prefill/decode/E2E `PENDING` a quiet host.**
The x86_64 arm of `BACKEND-GATE-CPU-LLAMACPP` now exists where it did not
before. Next: run the committed harness
`docs/bench-evidence/cpu-x86-llamacpp-20260811-harness.sh` on an idle x86_64 box
to close the three throughput axes, then CIQ `G5`.

## Outcome

**What was measured.** Correctness first: at the 32 output tokens the speed
recipe actually uses, our greedy continuation is byte-identical to llama.cpp's
(SHA-256 `e92cf4cd…`) and our own output is reproducible across processes. Peak
RSS is **2.8343 GiB against llama.cpp's 2.8281 GiB, ratio 1.0022x, parity**,
over five and three legs with 0.018% and 0.004% spread. Prefill, decode and E2E
are **`PENDING` a quiet host**: our single quiet-gated leg reads 42.39 / 5.99 /
19.53 tok/s but no llama.cpp leg ever passed the gate, so no ratio exists and
none was invented.

**What was rejected and why.** A complete five-repetition interleaved series
was thrown away rather than reported. A co-tenant build moved the one-minute
load average from 3.80 to 82.48 mid-series and the resulting spreads were 78.6%
to 248.2%; the ratios computable from it looked plausible and were meaningless.
This is the concrete re-confirmation of the record's existing `VOID`-for-binding
verdict on this box, and the reason the throughput axes are reported pending a
resource rather than satisfied.

**The near-tie was measured, not asserted.** At 64 output tokens the two engines
diverge once, at token ~57. Rather than wave that through as "probably a tie",
llama.cpp's own `--logit-bias` was bisected on the divergent token: the oracle
holds ` Japan` at -0.07 and flips to ` South` at -0.08, so its own top-2 margin
there is ~0.075 logits. The oracle was also checked for self-instability across
4/8/16/20 threads and does not flip, so this is recorded as a cross-engine
near-tie at a stated margin, not as oracle non-determinism.

**Why the gap, and no ceiling is declared.** The x86 position is not the Arm
position minus noise: `cpu_quant_dot.cpp:22` and `cpu_quant_repack.h:11` say in
their own words that the quant dot is portable-tier only on x86 (`G5` open) and
that the `G7` repacked layout has an i8mm-only consumer. The lever that crossed
prefill parity on Arm therefore does not exist on this ISA. Next traceable
hypothesis: CIQ **G5**, porting `ggml-cpu/arch/x86/quants.c` behind the existing
`cpu_isa_x86` probe, plus an AVX-512 consumer for the already-portable
`block_q8_0x4` layout.

**Records reconciled.** Two cells were quoting superseded positions as current
and are corrected here: `QUANT-GGUF` in the feature matrix still said the B4
speed/RSS checkpoint was "pending" with "no direct compute-in-quant or llama.cpp
speed parity", and `QUANT-GGUF-COMPUTE` in the quantization matrix still
presented G4's 3.38x/8.20x/2.29x as the live position. Both now point at
`BACKEND-GATE-CPU-LLAMACPP` as the single place that gate's position lives.

**Out of scope and stated as such.** The Metal/MLX half of `ROAD-V1-D1`
punch-list item 13 needs an Apple M4, which this host is not, and was not
touched.
