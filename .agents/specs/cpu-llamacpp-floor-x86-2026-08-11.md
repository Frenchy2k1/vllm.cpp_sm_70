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

## Outcome

To be filled when the series completes.
