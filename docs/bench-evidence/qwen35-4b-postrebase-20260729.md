# Qwen3.5-4B re-validated after rebasing onto 139 upstream commits, 2026-07-29

Immutable evidence index for the re-measurement of the 4B lever after the branch
was rebased from `main` `7f620e74` onto `f3ecbe70d`.

## Headline

**Nothing moved.** 139 upstream commits later, every axis reproduces the
2026-07-28 pinned-oracle series inside run-to-run noise. The published ratio
stands: **0.9972x** total throughput (was 0.9970x), TTFT **PASS**, TPOT still the
one failing axis at **+12.5%**.

This is a null result, and it is the useful kind: it says the rebase did not
regress the lever, and it says none of the upstream work in that window touched
this workload.

## What was measured

- Ours: `13906f189` (this branch rebased onto `main` `f3ecbe70d`), clean CUDA
  rebuild 925/925, 0 warnings.
- Oracle: `.venv-vllm-pin`, vLLM `0.23.1rc1.dev1511+g555967922` — built from
  source at the parity pin. Pin re-checked before the run and unchanged
  (`AGENTS.md` still specifies `555967922`), so the same denominator remains
  correct.
- Three performance repetitions per arm, interleaved, one `flock /tmp/gpu` across
  the whole 18-leg series.
- All nine performance legs recorded **0% GPU utilization** and 539 MiB resident
  (the X server) before starting.

## Binding result

| Axis | ours | vLLM @ pin | ratio | 2026-07-28 ratio | Disposition |
|---|---:|---:|---:|---:|---|
| Total throughput (tok/s) | 6610.270 | 6628.651 | 0.9972x | 0.9970x | FAIL |
| Output throughput (tok/s) | 730.947 | 732.978 | 0.9972x | 0.9970x | FAIL |
| Requests/s | 5.710 | 5.726 | 0.9971x | 0.9969x | FAIL |
| Mean TTFT (ms) | 729.963 | 947.850 | 0.7701x | 0.7731x | PASS |
| Mean TPOT (ms) | 38.153 | 33.923 | 1.1247x | 1.1241x | FAIL |
| Peak PSS (GiB) | 2.539 | 7.867 | | | PASS |

Direct-load OFF averages 6493.390 tok/s and 8.592 GiB peak PSS, so direct loading
still cuts host peak PSS by 70.4%.

Per-repetition total throughput: ours 6610.9 / 6611.6 / 6608.3 (spread 0.05%),
pin 6629.8 / 6625.7 / 6630.4 (0.07%), direct-OFF 6498.8 / 6496.4 / 6484.9 (0.21%).

## The control that makes "nothing moved" a measurement rather than an assumption

All three arms drifted down by the same ~0.13% against the previous series:

| Arm | now | 2026-07-28 | drift |
|---|---:|---:|---:|
| ours (direct ON) | 6610.270 | 6618.160 | 0.9988x |
| ours (direct OFF) | 6493.390 | 6504.600 | 0.9983x |
| vLLM @ pin | 6628.651 | 6638.129 | 0.9986x |

The vLLM arm is the same pinned binary running the same corpus — **its code did
not change between the two series**, yet it drifted by the same amount as ours.
A uniform drift across an arm that cannot have changed is ambient (thermal/clock
state of the box), not a code effect, and it is the same size as the arms' own
repetition spread. TPOT drifts identically small: ours 1.0012x, the pin 1.0007x.

So the ~0.1% is the noise floor of this vehicle, and every ratio above sits well
inside it.

## Token identity

| Comparison | Result |
|---|---|
| Ours ON vs OFF, per rep | 128/128, 128/128, 128/128 |
| Ours ON vs the 2026-07-28 series | 128/128, 128/128, 128/128 |
| Ours OFF vs the 2026-07-28 series | 128/128, 128/128, 128/128 |
| Ours ON vs vLLM @ pin, per rep | 89/128, 89/128, 89/128 (the established near-tie) |
| vLLM @ pin vs its own 2026-07-28 run | 128/128, 128/128, 95/128 |

Our output is bit-stable across a 139-commit rebase — the strongest available
statement that the merge resolutions were semantically correct, beyond the
1672/1672 model gate. The oracle's third repetition differing from its own prior
run is the already-recorded fact that vLLM is not self-deterministic on this
workload.

## Why no movement was the expected outcome

The one upstream commit in the window that names this gap —
`2b00866a4 explore(perf): decode TPOT/ITL lever` — is an exploration that
concluded the SGLang decode gap is **batch composition, not a decode-kernel
deficiency**. It changed records, not code. The rest of the window is breadth
(DeepSeek-V4-Flash, Gemma-4, Kimi K3, TP, LoRA, AWQ/GPTQ/MXFP4, fp8 KV, xgrammar)
on paths this dense 4B decode workload does not touch.

TPOT therefore remains owned by ENG-ASYNC-SCHED, and the W4 finding stands: the
lever needs the ASYNC engine loop, which `vllm-bench` does not drive.

## Reproduction

```sh
nix develop .#cuda --command bash -c 'REQUIRE_TRITON_AOT=1 \
  CPP_BENCH="$PWD/build-nix-cuda-transplant-triton/examples/vllm-bench" \
  CMAKE_CACHE="$PWD/build-nix-cuda-transplant-triton/CMakeCache.txt" \
  VLLM_PYTHON="$PWD/.venv-vllm-pin/bin/python" \
  VLLM_CUDA_HOME="$PWD/.venv-vllm-pin/lib/python3.12/site-packages/nvidia/cu13" \
  flock /tmp/gpu tools/bench/run_qwen35_4b_compare.sh /tmp/qwen35-postrebase-<commit>'

python3 tools/bench/summarize_qwen35_4b_compare.py \
  --root /tmp/qwen35-postrebase-<commit> \
  --historical-root /tmp/qwen35-pinned-oracle-13729626 \
  --output /tmp/qwen35-postrebase-<commit>/aggregate.json
```

Evidence root `/tmp/qwen35-postrebase-13906f189`, aggregate `aggregate.json`.
No 4B result implies support or speed for the 27B/35B gate checkpoints.
