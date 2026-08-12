# x86_64 CPU vs llama.cpp, 2026-08-11

The x86_64 arm of `BACKEND-GATE-CPU-LLAMACPP`, issue
[#433](https://github.com/mudler/vllm.cpp/issues/433),
[spec](../../.agents/specs/cpu-llamacpp-floor-x86-2026-08-11.md).

Both recorded arms of this gate are AArch64: the 20-core i8mm arm is closed and
the four-core Cortex-A76 arm is open. Every lever that closed the first arm is
Arm-scoped or Arm-measured. This is the first x86_64 measurement of the whole
gate since the 2026-07-10 B4 decision run, which predates the compute-in-quant
track. It is not the first x86_64 datapoint of any kind: issue #433 records a
2026-07-22 **one-thread** x86 control that reproduced B4 within 4.8-5.5%, which
is what established that the Arm movement was `W1-W3` rather than the vehicle.
That control is one thread and one axis; it is not this A/B.

## Result

Nothing here is settled in our favour. One axis has a number, three do not.

| Axis | vllm.cpp | llama.cpp | Ratio | Result |
|---|---:|---:|---:|---|
| **Peak RSS** | 2.8343 GiB (2,971,992 KB) | 2.8281 GiB (2,965,508 KB) | **1.0022x** | **hairline OPEN GAP** (6.33 MB against us) |
| Prefill (pp128) | 42.39 tok/s indicative | not obtained clean | n/a | **PENDING a quiet host** |
| Decode (tg32) | 5.99 tok/s indicative | not obtained clean | n/a | **PENDING a quiet host** |
| E2E (128 + 32) | 19.53 tok/s indicative | not obtained clean | n/a | **PENDING a quiet host** |

**Peak RSS is an open gap, not a pass.** The row's own `G4` criterion is that an
axis at or above 1.00x in our favour is met and anything below is an open gap.
On a lower-is-better axis 1.0022x is 0.22% **against** us. That is 12-55x the
leg-to-leg spread on this axis (0.018% and 0.004%), so it is resolvable rather
than rounding, and the asymmetry below means the true ratio can only be worse
for us. 6.33 MB on a 2.83 GiB working set is hairline and it is still a gap, and
declaring a parity band after seeing the number would be fitting the criterion
to the result. It is recorded here as an open gap with a named next step, which
is what the criterion asks for. It does reproduce the shape of the 20-core Arm
arm's 1.01x.

The asymmetry, stated rather than buried: the llama.cpp legs ran `pp128`, `tg32`
*and* the combined `pg128,32` in one process while ours ran a single 128+32
request, so llama.cpp's peak is if anything an **upper bound** on what its
recipe needs, and the true ratio is `>= 1.0022x`.

**Where the RSS medians come from, plainly.** They are medians over **five**
vllm.cpp legs (2,971,664-2,972,196 KB, 0.018% spread) and **three** llama.cpp
legs (2,965,468-2,965,580 KB, 0.004% spread), and **most of those legs are from
the contended series described below** — the series whose *timing* was discarded
in full. That is deliberate, and it is the one axis where it is defensible:
peak RSS is the high-water mark of a process's own resident pages, and a
co-tenant burning CPU changes when our pages are touched, not how many. The
measurement bears that out from both directions. Across legs whose throughput
moved by 248%, RSS moved by at most 0.018%; and an independent reviewer
re-measured this ratio on its own fresh legs during review of
[PR #440](https://github.com/mudler/vllm.cpp/pull/440), reproducing **1.0022x**
with an RSS spread of 0.016-0.021% across one-minute loads from 2 to 33.
Two engines, two sessions, a 16x range of contention, same four digits.

**The three throughput axes are PENDING a quiet host, not failing and not
waived.** Those figures come from the single vllm.cpp leg that passed the quiet
gate (one-minute load 2.43, zero compiler processes). No llama.cpp leg ever
passed it, so no ratio is computed and none is guessed. Publishing a ratio
against a contended denominator is exactly the error this file documents below.
`scripts/cpu-x86-llamacpp-floor.sh` is committed and re-runnable.

## Correctness

Established before any speed number was accepted, and **re-verified from
scratch on 2026-08-12** while repairing this file, because the original run
recorded a hash without recording the prompt that produced it.

The prompt, literally, 12 tokens:

```text
The capital of France is Paris. The capital of Italy is
```

### G1 at 32 output tokens — the length the speed recipe measures: **PASS**

Both engines emit byte-identical greedy continuations:

```text
 Rome. The capital of Spain is Madrid. The capital of Germany is Berlin. The capital of the United States is Washington, D.C. The capital of the
```

SHA-256 `e92cf4cd8923e4a873600f1bf8f615e2478254eac4645aba3f18819808cdf30a`. The
hash is over the continuation text with leading whitespace and the trailing
newline stripped, which anyone can now recompute:

```sh
printf '%s' 'Rome. The capital of Spain is Madrid. The capital of Germany is Berlin. The capital of the United States is Washington, D.C. The capital of the' | sha256sum
```

Our engine reproduces its own output exactly across repeated processes.

### G1 at 64 output tokens — the length the spec declared: **FAIL**

The spec's `G1` was written at 64 output tokens. At 64 the two engines diverge
**once**, and one divergence against a token-exact gate is a fail. Reported as
a fail rather than folded into the 32-token pass:

- Both continuations are exactly 64 tokens and the first **62** are identical.
  (The first version of this file said "the first ~57"; re-tokenising both
  continuations with our own tokenizer gives 62.)
- At token 63 we emit `South` (id 4725) and llama.cpp emits ` Japan` (id 6124):
  `... New Zealand is Wellington. The capital of South Africa` against
  `... New Zealand is Wellington. The capital of Japan is`.
- The margin was **measured, not assumed**. Using llama.cpp's own
  `--logit-bias` as a probe on token 6124, the oracle keeps ` Japan` at a bias
  of -0.07 and flips to ` South` at -0.08, so **llama.cpp's own top-2 margin at
  that step is ~0.075 logits**, under 2% of probability mass between the two
  candidates. Both probe legs were re-run on 2026-08-12 and reproduce exactly.
- The oracle is **not** non-deterministic here: it was checked for
  self-instability at 4, 8, 16 and 20 threads and does not flip on its own.

That last point matters for how this is allowed to be reported. `AGENTS.md`
permits a distributional gate **only** where the oracle's own greedy decode is
non-deterministic, and the author's own thread sweep proves it is deterministic,
which removes the precondition. So this is a genuine cross-engine divergence at
a measured near-tie margin, decided by reduction order, which two independent
implementations are not required to share — and it is still a `FAIL` of the gate
as the spec declared it. The spec's declared length has been amended to 32, the
length actually gated and actually measured; the 64-token result stays on the
record as a fail rather than being deleted along with the length.

## Contention, read this before the numbers

This host is a shared 20-vCPU KVM guest, and the record already declares the
x86 dev box `VOID` for binding timing for exactly this reason (co-tenant load,
`CLAIM-KERNEL-CPU-ELEM-GEMM-1`). That declaration was re-confirmed the hard way
here.

A first five-repetition interleaved series was **discarded for timing**:
another session started a parallel build mid-series and the one-minute load
average went from 3.80 to 82.48 with 20 compiler processes running. The damage
is visible in the spreads across those repetitions:

| Quantity | Spread across contended reps |
|---|---:|
| our prefill | 78.6% |
| our TPOT | 107.6% |
| our E2EL | 127.2% |
| llama.cpp pp128 | 202.4% |
| llama.cpp tg32 | 248.2% |
| **peak RSS, both engines** | **≤ 0.02%** |

Not one throughput number above comes from those legs. Their **peak RSS does**,
for the reason given in the Result section: the axis moved ≤ 0.02% while
throughput moved 248%, so contention does not reach it. The spread is also the
finding in its own right.

**Why five and three legs, when the spec planned three and one.** The spec's
design called for three clean vllm.cpp process repetitions against one
llama.cpp `-r 3` in-process series. What actually ran was the five-repetition
interleaved series above, cut short by the co-tenant build after its third
llama.cpp leg, plus one later quiet-gated vllm.cpp leg. The RSS axis kept every
leg that produced a `/usr/bin/time -v` report, because contention does not
reach it; the timing axis kept only the quiet-gated one. The exact per-leg
attribution can no longer be reconstructed — see `G5`.

The replacement series then ran for over an hour without completing, because a
second session started `cmake --build build -j 20` and the one-minute load
average sat between 30 and 97. The gate did its job and refused every window;
that is why the throughput axes are `PENDING` rather than filled in with
whatever the contended box happened to print.

### G5: every leg's load recorded before and after — **FAILING**

`G5` has no result and cannot be given one. Two load values survive in this
file (3.80 -> 82.48 for the discarded series, and 2.43 for the accepted
vllm.cpp leg); the per-leg `evi/*.time` and `evi/*.json` raw outputs were never
committed and no longer exist on the host, so the before/after load pair for
each of the eight legs cannot be produced. `AGENTS.md` requires the raw output
and the contention state, and exactly one result per applicable rule, so this
is recorded as **failing**, not as pending and not as satisfied. The RSS number
stands on the reviewer's independent reproduction rather than on this row's
lost raw data.

The harness now writes `$OUT/<engine>-<rep>.load` — `uptime`, `/proc/loadavg`
and the foreign-CPU share, before and after every single leg — and its
`summary.md` renders the `G5` table directly from those files. The next run of
this gate produces the evidence `G5` asks for; this run cannot retroactively.

### Three harness defects, all of them the gate looking at itself

Worth recording because two of them each cost a full waiting cycle, and they
are the same bug three times.

1. The first quiet gate used `pgrep -f` on a pattern of compiler names, which
   matched **its own waiter shell**: the command line containing the pattern
   *is* a match for the pattern. It blocked on its own reflection while the box
   was idle. Fixed at the time by moving to `pgrep -x` on process names.
2. `pgrep -c` already prints `0` and exits non-zero on no match, so a
   `|| echo 0` fallback emits `"0\n0"` and every numeric test downstream fails.
3. **The committed gate blocked on its own footprint.** It waited for the
   one-minute load average to fall below 3, but a 20-thread leg drives the
   one-minute average to 20-33 by itself, so after the first leg the gate was
   waiting out its own decay. The reviewer's legs 2 and 3 started at 21.76 and
   22.61 with zero foreign builders. The gate now decides on **foreign CPU
   share measured from `/proc/stat` over a fresh window**, which nothing we
   already did can inflate, and it excludes our own process tree by walking
   `/proc/<pid>/stat` rather than by naming processes and hoping. Load averages
   are still recorded for `G5`; they are no longer what the gate decides on.

Two further defects in the committed harness, found in review:

- It set `OUT=evi` and never created the directory, so **from a clean checkout
  every redirection failed, every leg was discarded for a non-zero exit, and it
  exited 2 blaming contention — having never run a single leg.** Fixed, and
  `tests/scripts/test_cpu_x86_llamacpp_floor.py` now runs the real script end
  to end against stub engines on every CI run.
- Its post-leg check re-tested `builders` but never load, so a leg could start
  at 2.9, run through load 80 from any non-compiler source, and still be
  accepted. The post-leg check now computes foreign CPU consumed **during** the
  leg (machine busy minus the leg's own user+sys) and discards above a
  threshold. And the harness now computes the medians, spreads and ratios into
  `$OUT/summary.md` itself: every figure in the first version of this file was
  transcribed by hand.

## Provenance

- Host: 20-vCPU KVM guest on an AMD Ryzen 9 9950X3D, Linux 6.8.0-136-generic,
  84 GiB RAM, AVX-512 present (`avx512f/dq/ifma/cd/bw`).
- Model: `Qwen3.5-2B-UD-Q8_K_XL.gguf`, 2,893,114,784 bytes, SHA-256
  `1eb01bfc3fbb04323e03fe6123d1d396f531474985b5d06e851ddf0522192f52`,
  MD5 `4f3b2fa71c455bf54e9823230accc057`. llama.cpp reads it as `qwen35 2B
  Q8_0`, 1,942,653,248 parameters. **Both engines read this same file.** It is
  not byte-identical to the file the RPi5 arm used (SHA-256 `a53988df…`), so
  this arm's absolute numbers are not comparable to that arm's; the ratios
  within this file are what bind.
- vllm.cpp: base `31a2b493`, branch `row/D1-CPU-X86-FLOOR`, built on this host
  with `-DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
  -DVLLM_CPP_BUILD_TESTS=OFF -DVLLM_CPP_SERVER=OFF`, Ninja, GCC 13.
- llama.cpp: local fork `237ad9b96`, build number 9892, the recorded pin.
  Release, `GGML_CUDA=OFF`, `GGML_NATIVE=ON`, OpenMP on, `backends: CPU`.
  `llama-completion` was built from that same tree and cache for the
  correctness leg, because this build carried only `llama-bench` and
  `llama-cli`, and `llama-cli` at this pin refuses `--no-conversation`.
- No GPU was used, and no leg queued on `$HOME/gpu.lock`.

## Commands

`P` and `M` below are the literal prompt and the model path above.

vllm.cpp, per accepted repetition:

```sh
/usr/bin/time -v taskset -c 0-19 env VLLM_CPP_CPU_THREADS=20 \
  ./build-cpu/examples/vllm-bench \
  --model "$M" \
  --num-prompts 1 --input-len 128 --output-len 32 \
  --concurrency 1 --seed 0 --temperature 0
```

llama.cpp, per accepted repetition:

```sh
/usr/bin/time -v taskset -c 0-19 llama-bench \
  -m "$M" -p 128 -n 32 -pg 128,32 -t 20 -ngl 0 -r 3 -o json
```

Correctness:

```sh
P='The capital of France is Paris. The capital of Italy is'
vllm-cli   --model "$M" --prompt "$P" --max-tokens 32 --temperature 0 --seed 0 --device cpu
llama-completion -m "$M" -p "$P" -n 32 --temp 0 --top-k 1 --seed 0 -t 20 -ngl 0 -no-cnv --no-warmup
# margin probe, bisected on the divergent token at 64 output tokens
llama-completion ... -n 64 -l 6124-0.07   # keeps " Japan"
llama-completion ... -n 64 -l 6124-0.08   # flips to " South"
```

## Where the gap comes from, if there is one

Named before measuring, from source, so it cannot be fitted to the result.

`src/vt/cpu/cpu_quant_dot.cpp:22` states its own scope: *"THIS IS THE PORTABLE
TIER ONLY. The x86 AVX2/AVX512 variants (`arch/x86/quants.c`) are work row G5
and the Arm NEON/dotprod/i8mm variants are G6."* G6 is `DONE`; **G5 is open**
([CIQ spec](../../.agents/specs/gguf-compute-in-quant-gemm.md) row G5).

`src/vt/cpu/cpu_quant_repack.h:11` states the same for the repack tier that
crossed prefill parity on Arm: the byte permutation is portable, but *"only the
GEMM kernels that CONSUME the layout are i8mm-gated
(`cpu_quant_repack_arm.cpp`)"*. There is no x86 consumer, so **the G7 prefill
win does not exist on this ISA at all.**

So on x86_64 our quantized weights go through the portable auto-vectorized
scalar tier, against llama.cpp's hand-written `arch/x86` AVX2/AVX-512 quant
kernels and its x86 repack. The elementwise f16/bf16 GEMM *is* x86-tiered here
(`cpu_matmul_elem_avx512.cpp`, `cpu_matmul_elem_avx2.cpp`); the quantized path
is not. On this file roughly 1.06 GiB of weight bytes are `q8_0`, so that is
the mass running portable.

**Next traceable hypothesis, throughput:** `QUANT-GGUF-CIQ-GEMM` **G5**: port
`ggml-cpu/arch/x86/quants.c` to an AVX2/AVX-512 tier behind the existing
`cpu_isa_x86` probe, and add an AVX-512 consumer for the already-portable
`block_q8_0x4` repacked layout so the G7 lever reaches x86. The CIQ spec listed
G5 as blocked partly "on a non-`VOID` x86 host"; the profile-ranking half of
that block is what this measurement addresses.

**Next traceable hypothesis, the 6.33 MB RSS gap:** the same per-pool
attribution that closed the Arm arm. `KEEPQ` `L7` did not close RSS by guessing
either — it profiled per pool, disproved the "engine workspace" story, and found
a q8_0 repack-source double-count. Nobody has done that on x86: the next step is
a per-pool RSS breakdown on a quiet box against llama.cpp's own buffer report,
on legs whose `G5` load record actually survives. 6.33 MB is small enough that
one unreleased mapping or one allocator arena would account for all of it, and
"small" is not a reason to close an axis the row's own criterion calls open.
