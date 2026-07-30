# Qwen3.5-4B post-pull revalidation, 2026-07-27

Immutable evidence index for the local discrete-Blackwell
`LOAD-SAFETENSORS-DIRECT-DENSE` checkpoint after fast-forwarding the workspace
109 commits onto `upstream/main` at `7f620e74`. This is the revalidation
`docs/BENCHMARKS.md` recorded as PENDING for the `c317237a` transplant; that
transplant's two commits are now upstream (`a131de03`, `b6f1efc8`), so the
measured tree is plain current `main`.

The checkpoint remains **GATING / speed-pending**. Correctness and the
direct-loader memory goal pass. The oracle still wins total/output throughput
and TPOT/ITL, so this is not a parity claim.

## Headline

1. **No correctness regression.** Every generated token is IDENTICAL to the
   2026-07-25 run: 128/128 requests per repetition for both the direct-ON and
   direct-OFF arms. 109 upstream commits moved no token on this workload.
2. **No meaningful performance regression.** Against the oracle measured in the
   SAME series, total throughput went 0.9864x -> 0.9819x and the TPOT excess
   went +13.41% -> +14.00%.
3. **The 2026-07-25 series was CONTENDED and its absolute numbers are VOID.**
   All nine of its performance legs ran against a GPU already at 11-13%
   utilization with 611 MiB of extra resident VRAM. Today's nine legs ran at 0%.
   Both arms gained ~14% on the idle box, which is why the RATIO barely moved.
   No absolute number from that series may be published; the ratios remain usable
   because the contention hit both arms.

## Workload and environment

- GPU: NVIDIA GeForce RTX 5070 Ti, sm_120, 16 GiB, driver 595.71.05.
  **Discrete**: `is_integrated_gpu()` is FALSE, which selects the host fallback
  of the ENG-ASYNC-SCHED combine/scatter (see the residual, below).
- Model: cached `Qwen/Qwen3.5-4B` snapshot `851bf6e8...` under `.hf-cache`.
- Dataset: `/tmp/qwen35-4b-sharegpt-1024.json`, SHA-256
  `9ea13603767c62c267e3f381fbccf42d0c9ca0c393655c37533eadca7aefca0c`.
- Workload: 128 requests, 131,784 input tokens, 128 output tokens per request,
  concurrency 32, greedy, prefix caching disabled.
- Project build: `RelWithDebInfo`, CUDA arch `120a`, CUTLASS, FlashAttention-2,
  Triton AOT + AOT regeneration. Binary SHA-256
  `c95ecac92acf7f13c4c123de6917626a07fb00773f5fea22d68798522b48427a`;
  CMake cache `f03148416721d92ff4592fe271bf877773bec2385b461e575fb6744e8596b527`.
- Oracle: local `.venv-vllm`, **vLLM 0.24.0** (`vllm-0.24.0.dist-info`, and the
  series' own `vllm-version.txt`). NOTE: the 2026-07-25 evidence and the
  BENCHMARKS entry derived from it called this oracle "vLLM 0.25.0"; that label
  was wrong, as that series' own recorded `vllm-version.txt` also says 0.24.0.
  The project parity pin is now `555967922` / vLLM 0.26.0.dev0, so this local
  denominator is behind the pin and is labelled as such rather than treated as
  a pin-era result.
- The whole 18-leg series held one `flock /tmp/gpu`.
- Root: `/tmp/qwen35-postpull-7f620e74`, aggregate `aggregate.json`.

## Binding result

Three performance repetitions per arm, interleaved ON / vLLM / OFF.

| Axis | Direct ON | Direct OFF | vLLM 0.24.0 | ON vs vLLM | Disposition |
|---|---:|---:|---:|---:|---|
| Total throughput (tok/s) | 6600.663 | 6481.337 | 6722.241 | 0.9819x | FAIL |
| Output throughput (tok/s) | 729.883 | 716.687 | 743.327 | 0.9819x | FAIL |
| Requests/s | 5.700 | 5.600 | 5.807 | 0.9815x | FAIL |
| Mean TTFT (ms) | 729.240 | 835.007 | 913.552 | 0.7982x | PASS, 20.2% lower |
| Mean TPOT (ms) | 38.220 | 38.207 | 33.526 | 1.1400x | FAIL, 14.0% higher |
| Mean ITL (ms) | 38.220 | 38.207 | 33.526 | 1.1400x | FAIL |
| Peak PSS (GiB) | 2.282 | 8.594 | 7.666 | | PASS |
| Stable PSS (GiB) | 0.761 | 8.591 | 4.039 | | PASS |
| Peak VRAM (MiB) | 12850.0 | 12846.0 | 12933.3 | | PASS vs vLLM; ON is +4.0 MiB vs OFF |

Run-to-run spread is very tight, so the ratios above are not noise: our ON arm
measured 6602.6 / 6603.9 / 6595.5 tok/s (0.13% spread) and the vLLM arm
6716.3 / 6722.9 / 6727.6 tok/s (0.17% spread).

Direct loading still cuts peak host PSS by 73.4% and stable PSS by 91.1%
versus direct-OFF, and it now also LOWERS mean TTFT by 12.7% (835.0 -> 729.2 ms),
which the contended series could not resolve.

## Token identity

| Comparison | Result |
|---|---|
| Ours ON, rep-to-rep | 128/128, 128/128, 128/128 (deterministic) |
| Ours OFF, rep-to-rep | 128/128 (deterministic) |
| Ours ON vs OFF, per rep | 128/128, 128/128, 128/128 |
| Ours ON vs 2026-07-25 ON, per rep | **128/128, 128/128, 128/128** |
| Ours OFF vs 2026-07-25 OFF, per rep | **128/128, 128/128, 128/128** |
| vLLM, rep-to-rep | 128/128, 128/128, 102/128 (NOT deterministic) |
| Ours ON vs vLLM, per rep | 87/128, 87/128, 92/128 (the established near-tie) |

Project-vLLM correctness is grounded in the real model gate and the ratified
near-tie contract, not in forced benchmark token identity, because the oracle is
not self-deterministic on this workload.

## Contention finding, and the harness hole it came through

Per-leg `nvidia-smi` snapshots, all nine performance legs of each series:

| Series | VRAM used before | GPU utilization before |
|---|---|---|
| 2026-07-25 | 1150 MiB, every leg | 11-13%, every leg |
| 2026-07-27 | 539 MiB, every leg | 0%, every leg |

`prepare_leg` gated idleness on `nvidia-smi --query-compute-apps`, which lists
CUDA contexts ONLY. A graphics consumer does not appear there, so a persistently
busy GPU passed the gate for an entire series. `.agents/benchmark-protocol.md`
voids contended runs, so this was a live measurement defect, not a footnote.

Fixed in the same change: `prepare_leg` now also fails when `utilization.gpu`
exceeds `GPU_IDLE_UTIL_MAX` (default 2%). Re-parsed against both series, the gate
reads 0 for today (passes) and 12 for 2026-07-25 (would have refused to run).
The new same-binary A/B harness `tools/bench/run_qwen35_4b_ab.sh` carries the
same gate.

Consequence for the record: the 2026-07-25 ABSOLUTE numbers (5769.99 tok/s and
the rest) are VOID and are superseded by the table above. Its RATIOS, its
same-binary component attributions (H32 AOT +4.5906%, decode graph +0.3873%,
ratio-4 FA2 +1.6004%) and its profiling attribution remain valid, because each
was an internal comparison within one contended-but-uniform series.

## The residual is unchanged, and now has a spec

TPOT is the failing axis, at the same relative size as before. The mechanism
identified by the 2026-07-25 node-mode trace still applies verbatim: on a
DISCRETE GPU, `is_integrated_gpu()` is false, so both ENG-ASYNC-SCHED W3 device
call sites (`runner.cpp:821` combine, `runner.cpp:1763` scatter) take their host
fallback, and `sample_tokens_async` must synchronize the main stream before the
host can read the sampled ids. The async scheduler is engaged (the engine logs
`Asynchronous scheduling is enabled (max_concurrent_batches=2)`) but the runner
cannot overlap, so depth-2 buys nothing here.

Upstream has no such branch: `vllm/v1/worker/gpu/states.py:64` keeps
`last_sampled_tokens` as a GPU tensor unconditionally, and never condenses its
request slots (`states.py:132` uses a free-index pool), which is why it needs no
host round-trip. Scoped as `ENG-ASYNC-SCHED` W4 in
[.agents/specs/async-discrete-device-combine.md](../../.agents/specs/async-discrete-device-combine.md),
including the second per-step barrier that must go with it (the CUDA embedding
out-of-range flag's `cudaStreamSynchronize`).

## Reproduction

```sh
nix develop .#cuda --command cmake --build build-nix-cuda-transplant-triton -j6

nix develop .#cuda --command bash -c 'flock /tmp/gpu env HF_HOME="$PWD/.hf-cache" \
  LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH \
  build-nix-cuda-transplant-triton/tests/test_qwen35_plain_weights --no-skip'

nix develop .#cuda --command bash -c 'REQUIRE_TRITON_AOT=1 \
  CPP_BENCH="$PWD/build-nix-cuda-transplant-triton/examples/vllm-bench" \
  CMAKE_CACHE="$PWD/build-nix-cuda-transplant-triton/CMakeCache.txt" \
  flock /tmp/gpu tools/bench/run_qwen35_4b_compare.sh /tmp/qwen35-postpull-<commit>'

python3 -m tools.bench.summarize_qwen35_4b_compare \
  --root /tmp/qwen35-postpull-<commit> \
  --historical-root /tmp/qwen35-postpull-7f620e74 \
  --output /tmp/qwen35-postpull-<commit>/aggregate.json
```

The summarizer needed two repairs to run at all: it read the historical token
legs under a `perf-` prefix the harness has never written, and it required a
`vllm_production` key that its own current output does not contain. Both are
fixed in the same change, so a series can now be summarized against the previous
one.

## Correctness suites, same build, same lock

| Suite | Result |
|---|---|
| `test_qwen35_plain_weights --no-skip` | 3/3 cases, 1672/1672 assertions |
| `test_ops_gdn` | 66/66 cases, 4242/4242 assertions |
| `test_ops_paged_attn` | 25/25 cases, 454,474/454,474 assertions |
| `test_gdn_packed_decode_triton` | 1/1, 10/10 |
| `test_combine_tokens` | 7/7, 14/14 |
| `test_input_batch` | 22/22, 163/163 |

Every count matches the 2026-07-25 record exactly.

No 4B result implies support or speed for the 27B/35B gate checkpoints, which
remain hardware-unavailable on this host.
