# Five-way 27B-dense benchmark on GB10 — plan and hazards

**Goal.** One replayable comparison of Qwen3.6-27B dense on dgx.casa (GB10),
1024 output tokens, across five engines, producing material for
`~/_git/social-devrel`.

| # | engine | backend | status |
|---|---|---|---|
| 1 | vllm.cpp | Vulkan | **blocked, see hazard 1** |
| 2 | vllm.cpp | CUDA | build was red until `cuda_gdn.cu` fix (below) |
| 3 | llama.cpp | Vulkan | harness rebuilt 2026-08-07, shaderc 2026.4-dev |
| 4 | llama.cpp | CUDA | building |
| 5 | vLLM | CUDA | **see hazard 2 — can reboot the box** |

Checkpoints already cached on dgx: `Qwen--Qwen3.6-27B` (bf16),
`unsloth--Qwen3.6-27B-NVFP4`, `z-lab--Qwen3.6-27B-DFlash`.

## Hazard 1 — arm 1 is exactly what issue #125 blocks, and the fix is UNRUN

`qwen3_5.cpp`'s `ResidentWeight` aliased host weight bytes into a device-tagged
tensor on every non-CUDA backend. Fixed in `19ddceb8` by gating on `is_cpu()`
instead of `!needs_weight_staging()` — but **no Qwen3.5/3.6 checkpoint has ever
been run off-CUDA**, so that fix is argued from the predicate and gated only on
what was reachable.

**This benchmark is therefore the first real test of the #125 fix.** Treat a
failure here as expected-and-informative, not as a surprise. If it throws
`embedding: table points outside every Vulkan allocation`, the fix is incomplete;
`ResidentWeightF32` (same file) is the next suspect, then the four
`needs_weight_staging()` feature gates that were deliberately left alone.

## Hazard 2 — vLLM on GB10 can hard-reboot the machine

GB10's ~119 GiB pool is UNIFIED, and vLLM's `gpu_memory_utilization` reserves
HOST RAM. `0.85` has hard-rebooted this box three times. Keep it LOW, force the
triton MoE path, and never run a large vLLM instance alongside a ctest or another
engine. This is the single most destructive step in the plan.

## Hazard 3 — quantization must be matched, or the comparison is meaningless

vllm.cpp and vLLM can both run NVFP4. llama.cpp cannot; it needs GGUF. bf16 27B is
~54 GB, so two engines cannot be resident at once on a 119 GiB unified pool — run
them strictly one at a time.

Decide and RECORD the matched arm before publishing anything. The 0.6B comparison
used byte-identical weights (llama.cpp's own converter over the safetensors we
load); replicate that discipline or the numbers are not comparable.

## Hazard 4 — the shapes must match, and at 0.6B they did not

`llama-bench` `tg32` generates 32 tokens from an EMPTY context. Our
`--input-len N --output-len M` prefills N then generates M. Those are different
workloads, and comparing our 32-in/8-out decode against `tg32` overstated our
position (1.75x) versus a matched 128-in/32-out shape (2.62x).

At 1024 output tokens use `llama-bench -p <prompt> -n 1024` against our
`--input-len <same> --output-len 1024`, and state the shape next to every number.

## Also fixed in passing: main's CUDA build was red

`cuda_gdn.cu` declared `hk_n` and never used it; nvcc `#177-D` under `-Werror`
fails **every** CUDA build. One-line `(void)hk_n`. This is why the earlier CUDA
verification of the #125 fix could only confirm the changed TU compiled — the
build died here.

## Replay

Every arm's command line, engine SHA and full stdout belongs in
`~/_git/social-devrel` alongside whatever gets published: that repo's rule is that
any artifact which informed a published result lives inside it. A number without
its command line is not replayable.
