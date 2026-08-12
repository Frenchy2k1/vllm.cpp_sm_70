# x86_64 CPU vs llama.cpp, 2026-08-11

The x86_64 arm of `BACKEND-GATE-CPU-LLAMACPP`, issue
[#433](https://github.com/mudler/vllm.cpp/issues/433),
[spec](../../.agents/specs/cpu-llamacpp-floor-x86-2026-08-11.md).

Both recorded arms of this gate are AArch64: the 20-core i8mm arm is closed and
the four-core Cortex-A76 arm is open. Every lever that closed the first arm is
Arm-scoped or Arm-measured. This file is the first x86_64 measurement since the
2026-07-10 B4 decision run, which predates the whole compute-in-quant track.

## Result

One axis is settled, three are not, and the reason is the host rather than the
engine.

| Axis | vllm.cpp | llama.cpp | Ratio | Result |
|---|---:|---:|---:|---|
| **Peak RSS** | **2.8343 GiB** (2,971,992 KB) | 2.8281 GiB (2,965,508 KB) | **1.0022x** | **PARITY, met** |
| Prefill (pp128) | 42.39 tok/s indicative | not obtained clean | n/a | **PENDING a quiet host** |
| Decode (tg32) | 5.99 tok/s indicative | not obtained clean | n/a | **PENDING a quiet host** |
| E2E (128 + 32) | 19.53 tok/s indicative | not obtained clean | n/a | **PENDING a quiet host** |

**Peak RSS is met and binding.** Medians over five vllm.cpp legs
(2,971,664-2,972,196 KB, 0.018% spread) and three llama.cpp legs
(2,965,468-2,965,580 KB, 0.004% spread). This axis is trustworthy on a busy box
in a way throughput is not: it moved by at most 0.018% across legs whose
throughput moved by 248%. One asymmetry is stated rather than buried: the
llama.cpp legs ran `pp128`, `tg32` *and* the combined `pg128,32` in one process
while ours ran a single 128+32 request, so llama.cpp's figure is if anything an
upper bound and the true ratio is not *better* for us than 1.0022x. At 6.5 MB
on a 2.83 GiB working set the axis is parity either way, and it reproduces the
20-core Arm arm's 1.01x.

**The three throughput axes are PENDING a quiet host, not failing and not
waived.** The vllm.cpp figures above come from the single leg that passed the
quiet gate (one-minute load 2.43, zero compiler processes); no llama.cpp leg
ever passed it, so no ratio is computed and none is guessed. Publishing a ratio
against a contended denominator is exactly the error this file documents below.
`scripts/cpu-x86-llamacpp-floor.sh` is committed and re-runnable: it blocks
each leg until the box is quiet and discards any leg a compiler appears during.

## Correctness

Established before any speed number was accepted.

At the **32 output tokens the speed recipe actually measures**, the two engines
produce **byte-identical** greedy continuations from the same 12-token prompt,
SHA-256 `e92cf4cd8923e4a873600f1bf8f615e2478254eac4645aba3f18819808cdf30a`.
Our engine reproduces its own output exactly across repeated processes.

Extending the same prompt to **64** output tokens exposes exactly one
divergence, at the second-to-last word: we continue `... Wellington. The
capital of South Africa`, llama.cpp continues `... Wellington. The capital of
Japan is`. The first ~57 tokens are identical.

That divergence was **quantified rather than assumed**. Using llama.cpp's own
`--logit-bias` as a margin probe on token 6124 (` Japan`), the oracle keeps
` Japan` at a bias of -0.07 and flips to ` South` at -0.08, so **llama.cpp's own
top-2 margin at that step is ~0.075 logits**, under 2% of probability mass
between the two candidates. A near-tie of that size is decided by reduction
order, which two independent implementations are not required to share. For
completeness the oracle was checked for self-instability at 4, 8, 16 and 20
threads and did **not** flip on its own, so this is recorded as a cross-engine
near-tie at a measured margin, not as oracle non-determinism and not as a
silent pass.

## Contention, read this before the numbers

This host is a shared 20-vCPU KVM guest, and the record already declares the
x86 dev box `VOID` for binding timing for exactly this reason (co-tenant load,
`CLAIM-KERNEL-CPU-ELEM-GEMM-1`). That declaration was re-confirmed the hard way
here.

A first five-repetition interleaved series was **discarded in full**: another
session started a parallel build mid-series and the one-minute load average
went from 3.80 to 82.48 with 20 compiler processes running. The damage is
visible in the spreads across those repetitions:

| Quantity | Spread across contended reps |
|---|---:|
| our prefill | 78.6% |
| our TPOT | 107.6% |
| our E2EL | 127.2% |
| llama.cpp pp128 | 202.4% |
| llama.cpp tg32 | 248.2% |
| **peak RSS, both engines** | **≤ 0.02%** |

None of those legs contributes to any number above; their spread is recorded
here because the spread is the finding. The replacement series gates every
single leg: it will not start one until the one-minute load average is below 3
with no compiler process running, and it discards and retries any leg during
which a compiler appears. That harness is committed as `scripts/cpu-x86-llamacpp-floor.sh`, so finishing
the pending axes is a matter of running it on a quiet box, not of rebuilding
the method.

Peak RSS is the one axis that is immune to this: it varied by at most 0.02%
across legs that varied by 248% in throughput.

The replacement series then ran for over an hour without completing, because a
second session started `cmake --build build -j 20` and the one-minute load
average sat between 30 and 97. The gate did its job and refused every window;
that is why the throughput axes are `PENDING` rather than filled in with
whatever the contended box happened to print.

One harness bug is worth recording because it wasted a full waiting cycle and
would recur: the first version of the quiet gate used `pgrep -f` on a pattern
of compiler names, which matched **its own waiter shell**: the command line
containing the pattern *is* a match for the pattern. The gate blocked on its
own reflection while the box was actually idle. It now uses `pgrep -x` against
process names. A second, quieter bug: `pgrep -c` already prints `0` and exits
non-zero on no match, so a `|| echo 0` fallback emits `"0\n0"` and every
numeric test downstream fails.

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

vllm.cpp, per accepted repetition:

```sh
/usr/bin/time -v taskset -c 0-19 env VLLM_CPP_CPU_THREADS=20 \
  ./build-cpu/examples/vllm-bench \
  --model Qwen3.5-2B-UD-Q8_K_XL.gguf \
  --num-prompts 1 --input-len 128 --output-len 32 \
  --concurrency 1 --seed 0 --temperature 0
```

llama.cpp, per accepted repetition:

```sh
/usr/bin/time -v taskset -c 0-19 llama-bench \
  -m Qwen3.5-2B-UD-Q8_K_XL.gguf -p 128 -n 32 -pg 128,32 -t 20 -ngl 0 -r 3 -o json
```

Correctness:

```sh
vllm-cli   --model $M --prompt "$P" --max-tokens 32 --temperature 0 --seed 0 --device cpu
llama-completion -m $M -p "$P" -n 32 --temp 0 --top-k 1 --seed 0 -t 20 -ngl 0 -no-cnv --no-warmup
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

**Next traceable hypothesis:** `QUANT-GGUF-CIQ-GEMM` **G5**: port
`ggml-cpu/arch/x86/quants.c` to an AVX2/AVX-512 tier behind the existing
`cpu_isa_x86` probe, and add an AVX-512 consumer for the already-portable
`block_q8_0x4` repacked layout so the G7 lever reaches x86. The CIQ spec listed
G5 as blocked partly "on a non-`VOID` x86 host"; the profile-ranking half of
that block is what this measurement addresses.
