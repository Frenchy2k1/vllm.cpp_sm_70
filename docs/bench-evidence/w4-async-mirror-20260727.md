# ENG-ASYNC-SCHED W4 evidence, 2026-07-27

Immutable evidence index for the discrete-CUDA device-resident sampled-token
mirror: its correctness gate, its same-binary A/B, and the profiling that
re-attributes the synchronizations the row was originally scoped from.

Commit `87160099` on branch `bench-lever-sampled-token-20260727`, off `main`
`7f620e74`. GPU: RTX 5070 Ti, sm_120, 16 GiB, driver 595.71.05, DISCRETE
(`is_integrated_gpu()` FALSE). Every run under one `flock /tmp/gpu` on an idle
box (the harness now refuses a leg above 2% utilization).

## Disposition

**Correctness GATED. Speed NEUTRAL on the only runnable vehicle, and
structurally unmeasurable there. Ships OPT-IN, DEFAULT OFF
(`VT_ASYNC_DEVICE_MIRROR=1`). The binding serving A/B is PENDING.**

## Correctness

| Gate | Result |
|---|---|
| Token identity, W4 ON vs OFF, 3 paired repetitions | **384/384 requests identical** |
| Earlier single pair, W4 OFF vs the pre-W4 baseline | 128/128 identical |
| `test_qwen35_plain_weights --no-skip` | 3/3 cases, 1672/1672 assertions |
| `test_input_batch` | 25/25 cases, 183/183 (includes 3 new W4 cases) |
| `test_combine_tokens` | 7/7, 14/14 |
| `test_ops_gdn` | 66/66, 4242/4242 |
| `test_ops_paged_attn` | 25/25, 454,474/454,474 |
| Clean `-Werror` CUDA rebuild | 0 warnings |
| ASan+UBSan lane, `test_input_batch` (the W4 op log) | 25/25, 183/183, leaks on |

The W4-OFF-vs-pre-W4-baseline row matters on its own: that binary already
contained W4e (the embedding barrier removal, which is NOT behind the opt-in), so
it establishes that W4e is output-identical to the pre-W4 engine.

## Speed, same-binary A/B (`/tmp/w4-ab-final`)

Three interleaved repetitions per arm, order flipped on even repetitions, 20 s
cooldown before each leg, identical corpus (128 requests, 128 output tokens,
concurrency 32, greedy).

| Axis | W4 ON | W4 OFF | ON/OFF |
|---|---:|---:|---:|
| Total throughput (tok/s) | 6618.047 | 6620.797 | 0.9996 |
| Output throughput (tok/s) | 731.807 | 732.107 | 0.9996 |
| Mean TPOT (ms) | 38.110 | 38.093 | 1.0004 |
| Mean TTFT (ms) | 728.603 | 728.423 | 1.0002 |

Per-repetition: ON 6610.4 / 6621.3 / 6622.4, OFF 6622.3 / 6622.6 / 6617.5. The
0.04% difference is an order of magnitude inside the arms' own spread.
**NEUTRAL**, and necessarily so — see below.

## Why it is neutral here, and what that says about the original attribution

`vllm-bench` drives the SYNCHRONOUS `LLMEngine::step()` loop, which calls
`GPUModelRunner::sample_tokens()`. Only `AsyncLLM`'s depth-2
`step_with_batch_queue` calls `sample_tokens_async()`. This was established
directly rather than inferred: instrumenting the W4 scatter branch inside
`sample_tokens_async` showed it **never executes** under `vllm-bench` (the first
gate run failed at 0/128 token identity precisely because the mirror was fed only
there and so stayed all-zero).

So on the benchmarked path there is no depth-2 overlap for W4 to unlock. It adds
four small uploads and two kernels per step and removes nothing, which is exactly
what the A/B measures.

**This falsifies the attribution the row was scoped from.** The 2026-07-25
evidence assigned 497 `cudaStreamSynchronize` calls (20.975 s, 42.20 ms/call) to
`sample_tokens_async`'s discrete host-bookkeeping path. That function is not on
the benchmarked path at all.

A fresh attribution-complete profile (`/tmp/w4-attrib.nsys-rep`,
`nsys profile --cuda-graph-trace=node --trace=cuda`, 32 requests x 64 output
tokens, concurrency 32) says what those synchronizations really are:

| CUDA API | Calls | Total | Avg |
|---|---:|---:|---:|
| `cudaStreamSynchronize` | 112 | 1.134 s | 10.12 ms |
| `cudaMemcpyAsync` | 1,228 | 0.398 s | 0.32 ms |
| `cudaMalloc` | 818 | 0.029 s | 0.035 ms |
| `cudaFree` | 512 | 0.028 s | 0.054 ms |

112 synchronizations over ~64 decode steps plus prefill and warm-up is **about
one per engine step**, at ~10 ms each on this smaller batch. Scaled to the
binding workload (~512 steps at ~38 ms TPOT) that is ~500 calls at ~40 ms — the
2026-07-25 numbers almost exactly. The COUNT and the TIME in that trace were
right; only the attribution was wrong. It is the depth-1 engine loop's per-step
wait for its own sampling, not an async-sampler defect.

The per-call embedding barrier is gone from this trace, which is W4e working:
before it, every step also paid a `cudaMalloc` + `cudaStreamSynchronize` +
`cudaFree` for the out-of-range flag.

## The lever this actually leaves

Overlap on this hardware needs the engine to run the ASYNC loop; W4 is what makes
that loop legal on a discrete GPU, and it is now built and gated. The measurement
that would bind it is a SERVING A/B over `AsyncLLM` (`examples/server`), which
`tools/bench/run_serve_low.py` cannot currently drive here — it requires a pinned
SGLang container image and accepts only the 27B/35B model keys, and neither is
available on this host. That harness gap, not W4, is what blocks the number.

## Reproduction

```sh
nix develop .#cuda --command cmake --build build-nix-cuda-transplant-triton -j6

nix develop .#cuda --command bash -c 'export \
  LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH HF_HOME=$PWD/.hf-cache; \
  CPP_BENCH="$PWD/build-nix-cuda-transplant-triton/examples/vllm-bench" \
  CMAKE_CACHE="$PWD/build-nix-cuda-transplant-triton/CMakeCache.txt" \
  flock /tmp/gpu tools/bench/run_qwen35_4b_ab.sh /tmp/w4-ab-<commit> \
    w4on VT_ASYNC_DEVICE_MIRROR=1 w4off ""'
```

No 4B result implies support or speed for the 27B/35B gate checkpoints.
