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
| **x86_64** | **no A/B since 2026-07-10**; a one-thread control on 2026-07-22 | ledger B4 row; [#433](https://github.com/mudler/vllm.cpp/issues/433) |

Every lever that closed the first arm is Arm-scoped or Arm-measured: `CIQ` G6
(i8mm quant tier), `CIQ` G7 (q8_0 repack-at-load, the prefill crossing),
`KEEPQ` L7 (whose residual was an Arm repack-source double-count) and
`KERNEL-CPU-A76-Q8-DOT` (AArch64 SDOT). The x86 numbers on file — the B4 run's
54-75x decode, ~1,480x prefill, 2.7x RSS behind, plus the 2026-07-22 **one
thread** control that reproduced B4 within 4.8-5.5% and so attributed the Arm
movement to `W1-W3` — all predate the whole compute-in-quant track, so they
cannot be quoted as current and cannot be assumed stale either. The 07-22
control is one thread and one configuration; it is not a multi-thread A/B.

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
3. Correctness leg first. Both engines, same raw prompt, greedy, **32 output
   tokens — the length the speed recipe actually measures** — compared as text.
   A speed number is not accepted until this passes. The declared length was
   64 when this spec was written; it is 32 because that is the length the
   binding recipe uses and therefore the length the gate is about. The 64-token
   result is **not** dropped along with the length: it is a `FAIL`, reported as
   one in the evidence and in `G1` below.
4. Speed legs: three clean vllm.cpp process repetitions and one llama.cpp
   `-r 3` in-process series, interleaved, `taskset`-pinned to the same cores
   with the same thread count. (What ran was five and three; the evidence file
   says why.)
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

| # | Gate | Result form | Result |
|---|---|---|---|
| G1 | Correctness: both engines emit the same greedy text for the same prompt at the declared **32** output tokens | pass / fail — **blocks every speed number** | **PASS** at 32, byte-identical, sha256 `e92cf4cd…`. **FAIL at 64**, the length originally declared: one divergence at token 63 of 64, llama.cpp's own top-2 margin there ~0.075 logits, oracle self-stable across 4/8/16/20 threads |
| G2 | Prefill throughput ratio, ours / llama.cpp | value + ratio | **PENDING a quiet host** — ours 42.39 tok/s indicative, no clean denominator |
| G3 | Decode throughput ratio, ours / llama.cpp | value + ratio | **PENDING a quiet host** — ours 5.99 tok/s indicative, no clean denominator |
| G4 | Peak RSS ratio, ours / llama.cpp | value + ratio | **1.0022x = hairline OPEN GAP**, 6.33 MB against us on a lower-is-better axis |
| G5 | Every leg taken at load average below the declared ceiling, recorded before and after | recorded | **FAILING** — two load values survive, the per-leg raw `evi/*` were never committed and no longer exist |

G2-G4 are **not** pass/fail for this row: the row's job is to produce the
number honestly. Any axis at or above 1.00x in our favour is met; any axis
below is recorded as an **open gap** on `BACKEND-GATE-CPU-LLAMACPP` with a
named next traceable hypothesis. Rounding an axis up, or reporting one
"pending" because it came out badly, fails the row.

That criterion is applied to `G4` as written. 1.0022x on a lower-is-better axis
is 0.22% **against** us and 12-55x the leg-to-leg spread, so it is an open gap,
not a tie — and the llama.cpp arm's own recipe makes its figure an upper bound,
so the true ratio can only be worse for us. This spec deliberately does **not**
introduce a parity band: inventing one after seeing the number would be fitting
the criterion to the result.

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
- **Two review mutations SURVIVED, and they are recorded here as known gaps
  rather than closed.** (1) Making the harness read peak RSS from the wrong
  process produced a **1,688x** error that no checker caught. (2) Altering the
  recorded sha256 of the correctness output was likewise undetected. Nothing in
  the tree ties a number in a `docs/bench-evidence/` file to the artefact it
  came from, for any row — this is a property of the evidence surface, not of
  this measurement, and fixing it is a checker-semantics change that needs its
  own spec and its own red-before evidence. What this row did do is remove the
  half of the exposure it could: the correctness prompt, the literal output and
  the exact hash recipe are now committed, so `e92cf4cd…` is recomputable by
  anyone in one command instead of being an unverifiable assertion, and the
  harness now computes its own medians and ratios into `summary.md` instead of
  leaving every figure to hand transcription. The RSS provenance gap remains
  open: a wrong-process RSS in a future run would still land unchallenged.
- **Paying for this row exposed a structural blocker on `docs/BENCHMARKS.md`,
  filed as [#460](https://github.com/mudler/vllm.cpp/issues/460).** The surface
  is at its 45,000-character cap, so every new row must evict one, and the
  documented byte-for-byte eviction into `.agents/benchmark-record.md` is
  impossible for any row carrying a `docs/`-relative link, because
  `check_links` matches links inside fenced blocks and resolves them from the
  archive's directory. This row worked around it; the repair is a
  checker-semantics change and is deliberately not made here.
- **`G5`'s raw data is gone and cannot be recovered.** The per-leg `evi/*.time`
  and `evi/*.json` were never committed. `G5` is reported as failing. The
  harness now writes a per-leg `.load` file and renders the `G5` table from it,
  so the next run of this gate can satisfy it; this run cannot.

## Evidence

`docs/bench-evidence/cpu-x86-llamacpp-20260811.md` — commands, the literal
correctness prompt and continuation with the exact hash recipe, binary and
model hashes, per-repetition spreads, the two surviving load averages, and why
`G5` fails. `scripts/cpu-x86-llamacpp-floor.sh` is the re-runnable harness and
`tests/scripts/test_cpu_x86_llamacpp_floor.py` is its smoke contract.

## Stop conditions

- Correctness leg fails → stop, report, do not publish a speed number.
- The box cannot be brought below the load ceiling → stop and report the axis
  as pending a quiet host, naming it. Do not average through contention.
- The work turns out to need the GPU → return `NEEDS_DECISION`; another session
  holds `$HOME/gpu.lock`.

## Now

**Peak RSS 1.0022x = hairline OPEN GAP (6.33 MB against us); prefill/decode/E2E
`PENDING` a quiet host; `G5` FAILING, its raw data lost.** The x86_64 arm of
`BACKEND-GATE-CPU-LLAMACPP` now exists where it did not before, and no axis of
it is met. Next: run the repaired harness `scripts/cpu-x86-llamacpp-floor.sh`
on an idle x86_64 box to close the three throughput axes and to produce the
`G5` record this run cannot, then CIQ `G5` for throughput and a per-pool RSS
breakdown for the 6.33 MB.

## Outcome

**What was measured.** Correctness first: at the 32 output tokens the speed
recipe actually uses, our greedy continuation is byte-identical to llama.cpp's
(SHA-256 `e92cf4cd…` over the prompt `The capital of France is Paris. The
capital of Italy is`) and our own output is reproducible across processes. Peak
RSS is **2.8343 GiB against llama.cpp's 2.8281 GiB, ratio 1.0022x — a hairline
OPEN GAP of 6.33 MB against us**, over five and three legs with 0.018% and
0.004% spread, independently reproduced to the same four digits by the reviewer
on its own legs across loads 2 to 33. Prefill, decode and E2E are **`PENDING` a
quiet host**: our single quiet-gated leg reads 42.39 / 5.99 / 19.53 tok/s but no
llama.cpp leg ever passed the gate, so no ratio exists and none was invented.
`G5` is **failing**: its raw per-leg output was never committed and is gone.

**What was rejected and why.** A complete five-repetition interleaved series
was thrown away rather than reported. A co-tenant build moved the one-minute
load average from 3.80 to 82.48 mid-series and the resulting spreads were 78.6%
to 248.2%; the ratios computable from it looked plausible and were meaningless.
This is the concrete re-confirmation of the record's existing `VOID`-for-binding
verdict on this box, and the reason the throughput axes are reported pending a
resource rather than satisfied.

**The near-tie was measured, not asserted — and it is still a fail at 64.** At
64 output tokens the two engines diverge once, at token **63 of 64** (the first
62 are identical; the "~57" in the first draft of the evidence was wrong and is
corrected). Rather than wave that through as "probably a tie", llama.cpp's own
`--logit-bias` was bisected on the divergent token: the oracle holds ` Japan` at
-0.07 and flips to ` South` at -0.08, so its own top-2 margin there is ~0.075
logits. The oracle was also checked for self-instability across 4/8/16/20
threads and does not flip. That last check is what *removes* the distributional
escape hatch: `AGENTS.md` allows a distributional gate only where the oracle's
own greedy decode is non-deterministic, and here it demonstrably is not. So the
64-token result is reported as a **`FAIL` of `G1` as originally declared**, next
to the 32-token `PASS` at the length actually measured and now actually
declared. Every leg of this paragraph was re-run and reproduced on 2026-08-12
during the review repair.

**Why the gap, and no ceiling is declared.** The x86 position is not the Arm
position minus noise: `cpu_quant_dot.cpp:22` and `cpu_quant_repack.h:11` say in
their own words that the quant dot is portable-tier only on x86 (`G5` open) and
that the `G7` repacked layout has an i8mm-only consumer. The lever that crossed
prefill parity on Arm therefore does not exist on this ISA. Next traceable
hypothesis: CIQ **G5**, porting `ggml-cpu/arch/x86/quants.c` behind the existing
`cpu_isa_x86` probe, plus an AVX-512 consumer for the already-portable
`block_q8_0x4` layout. For the 6.33 MB RSS gap the named next step is the
per-pool attribution that closed the Arm arm under `KEEPQ` `L7`, run on a quiet
box against llama.cpp's own buffer report.

**The harness shipped unable to run, and now cannot again.** Review found the
committed `scripts/cpu-x86-llamacpp-floor.sh` set `OUT=evi` without creating it,
so from a clean checkout every redirection failed, every leg was discarded for a
non-zero exit, and it exited 2 blaming contention it had never reached; and its
quiet gate waited on a one-minute load average that its own 20-thread leg drives
to 20-33, so it could never re-arm after the first leg. Both are fixed, the gate
now measures foreign CPU share from `/proc/stat` and excludes our own process
tree by PID rather than by name, the post-leg check tests load and not just
compilers, and `tests/scripts/test_cpu_x86_llamacpp_floor.py` runs the real
script end to end against stub engines on every CI run.

**Records reconciled.** Two cells were quoting superseded positions as current
and are corrected here: `QUANT-GGUF` in the feature matrix still said the B4
speed/RSS checkpoint was "pending" with "no direct compute-in-quant or llama.cpp
speed parity", and `QUANT-GGUF-COMPUTE` in the quantization matrix still
presented G4's 3.38x/8.20x/2.29x as the live position. Both now point at
`BACKEND-GATE-CPU-LLAMACPP` as the single place that gate's position lives.

**Out of scope and stated as such.** The Metal/MLX half of `ROAD-V1-D1`
punch-list item 13 needs an Apple M4, which this host is not, and was not
touched.
