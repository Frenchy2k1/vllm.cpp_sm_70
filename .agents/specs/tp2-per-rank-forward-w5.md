# W5 — per-rank forward dispatch (the runner seals the tp>1 gap)

Row: `BACKEND-DISTRIBUTED-TP`
Owner spec: `TP_PLAN.md` Step 3 (W5). Plate, not a narrative; every claim cites
a file:line and is either verified or PENDING the next gate.

## Scope

Make a `GPUModelRunner` armed with the W4-pre per-rank lanes (this branch,
`runner.h`/`runner.cpp`: `tp_queues_`, `tp_lane_count()`, lane-aware
`attach_tp_group`) actually run one dense decode step so its `input.tp` reaches
the tp-aware forward entry (`qwen3_5_dense.cpp:144-153`) and the per-layer
sharded traversal inside `DenseForwardBody` / `dense_attn::AttnBlock`
(`dense_attn_block.h:553`, `MlpDown`), ending at the existing
`vllm::TpAllReduceSum` seams. This is the last named piece of the TP-SERVE
refusal (`runner.cpp:434`).

## Already true on this branch (verified, do not re-derive)

- Loader-slice math (TpShard at the dense-weight chokepoint): `test_tp_forward`
  W4 case, 3/3, 133/133 CPU PASS.
- Per-rank placement + forward primitives on the real 2-V100 target:
  `test_nccl_group` 16/16, 197121/197121 at 2 GPU (and 196823 at 3 GPU after
  the world-modulo harness fix) — per-rank `TensorParallel` lane build,
  dense-body gate/up+down shard, device-sharded GEMM + group all-reduce, KV/GQA
  head-shard, MoE expert shard, lm_head all-gather, all == single-GPU.
- Runner accepts N per-rank queues (`tp_lane_count`), attach sizes/ranks from
  them, and **refuses** world>1 at construction (`attach_tp_group` G1) until
  this wave lands. `-tp N` flag parses and `-tp 2` is refused loudly (W7 first
  half, verified on the binary).

## Scope (this row)

1. Make the runner's `execute_model` dispatch per-rank when `tp_lane_count() >
   1`: stage each rank's device-resident weights (its own `ResidentWeight` on
   its own lane), run the tp-aware dense forward with `input.tp = &tp_`, and
   an all-reduced exit identical on every rank (the `exit-all-reduced`
   discipline `tp_shard_host.h` records). The forward ALREADY exists and takes
   `input.tp`; this is the runner wiring that reaches it.
2. Retain the G1 refusal UNTIL the tokens prove equality (see Gate), so a
   half-wired forward never silently runs tp1 on another rank's device.

## Gate (unlocks only if green / measured)

- **G5**: a TP2 forward (dense 1-token decode, Qwen3.5-27B-NVFP4 on 2 V100s,
  `CUDA_VISIBLE_DEVICES=1,2`) produces **token-exact** output equal to the
  single-GPU tp1 forward on the same prompt (greedy). The tp2==tp1 token gate
  is the acceptance; a token mismatch is a re-open, not a tolerance.
- G4 (each rank loads ONLY its TpShard slice to ITS device) is implied but the
  device-resident-upload half of W4-pre is separately owed (per `TP_PLAN.md`
  Step 1; its placement math is already proven, the runner upload wiring is
  not).

## Risks

- A collective that never returns on a step (rank never reaches its per-rank
  weights): the named-refusal must re-arm (never reduce to tp1 math).
- The `world>1` guard swallows any partial wiring — a dev that sees green
  without the forward is measuring the class, not the capability
  (`.agents/reachability.md`). Mutate: delete the `input.tp = &tp_` callsite in
  a scratch copy; a gate that stays green is not a gate.

## Evidence contract

- Red-first: capture a tp2 run that THROWS the current refusal, then the green
  tp2==tp1 token run on the same lattice config.
- Record the build/run recipe, revisions, model snapshot hash, and both raw
  output token streams (not a diff of a diff).

## Red-first evidence (measured 2026-08-22, scratch probe)

With `VT_TP_ALLOW=1` (a scratch env-gated probe added to `attach_tp_group` +
`FromModelDir`, default OFF, since REVERTED) a REAL `vllm-server --tp 2` on
Qwen3.8-27B-NVFP4 (GPUs 1,2) BOOTS and LISTENS, stages 17 GiB on GPU1 and
~0.5 GiB on GPU2, then the first generate never completes. A live trace pins the block
precisely (three progressive measurements, the last two correcting the
earlier ones):

1. `core-step begin unfinished=1` (VT_ENGINE_STEP_LOG heartbeat) — the request
   IS admitted to the scheduler (num_requests_waiting=0/running=0 that read
   earlier was a stale snapshot AFTER the client gave up).
2. `runner execute_model tokens=5 reqs=0` (runner trace) — the scheduled
   5-token PREFILL chunk REACHES `GPUModelRunner::execute_model`. The forward
   is entered, not skipped.
3. NO `core-step end`, NO fatal, NO exception, GPU 0% on all ranks, all other
   threads parked on futex/pipe — the engine thread enters `execute_model`
   once and NEVER returns.

So the block is INSIDE `execute_model`, on a host-only path, BEFORE the forward
is entered at all. Two decisive observations: (1) an instrumented `Forward`
entry (qwen3_5.cpp) with the same `VT_ENGINE_STEP_LOG` gate never printed
while `core-step begin` did — the stall is in the runner between the step and
`ModelRegistry::Forward`, not in the model forward; (2) GPU 0% on all ranks
and no CUDA/NCCL thread busy — no kernel, no shard primitive, no
fan-out thread was ever launched. The block is a lane-0-local host-side wait
in the runner's step-input assembly (the region between `execute_model` entry
and line ~1823 `ModelRegistry::Forward`), ahead of any collective.

Per-lane forward dispatch is a possible eventual fix but is NOT the diagnosis:
the stall never reaches the primitives that dispatch would feed. The next
implementer's first move should be instrumenting the runner body between the
step entry and `ModelRegistry::Forward` (the `prepare_inputs` /
`MakeCommonAttentionMetadata` / `forward_input` assembly) to find the exact
host wait, non multi-lane dispatch. `VT_TP_ALLOW` is a scratch scaffold (never
a production knob); production keeps the loud refusal until a tp>1 step
returns and the tp2==tp1 gate passes.

## tp1 ground-truth baseline (measured 2026-08-22, committed binary)

Served the real Qwen3.8-27B-NVFP4 (snapshot
`7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108`) tp=1 on GPU1
(`CUDA_VISIBLE_DEVICES=1`, `--tensor-parallel-size 1`), greedy temp 0:
prompt `"The capital of France is"` → **`" Paris.\nThe capital of Germany is
Berlin.\n"The` (10 tokens). G5's tp2 forward must reproduce this byte-for-byte;
the tp>1 path on the same checkpoint refuses at construction with the named W5
message. Capture tp=2 against THIS baseline, never a re-derived probe.