# TP_PLAN.md — make Qwen3.8-27B-NVFP4 TP=2 work

Row: `BACKEND-DISTRIBUTED-TP-SERVE`
Branch: `row/BACKEND-DISTRIBUTED-TP-CLEAN` (current working head)
Target: a real `vllm-server -tp 2` serves the 27B-NVFP4 on 2 V100s, token-exact
vs `-tp 1`, without the TP-SERVE construction refusal.

## Observed baseline (2026-08-21)

The NCCL transport is **built + linked + verified** in a Release arch-70 server
(`nm -D` shows `ncclAllReduce/AllGather/Send/Recv/CommInitAll`; `ldd` links the
conda `libnccl.so.2`). Running that server under `CUDA_VISIBLE_DEVICES=1,2`
exits loudly with the **named TP-SERVE refusal** (not a hang):

```
vllm.cpp distributed serving tp>1 (world_size=2): distributed weight load not
yet wired — tp>1 requires the per-rank loader (TpShard at the dense-weight-load
chokepoint) + the runner's per-rank device load + per-device forward, and the
paged full-attention tp>1 block currently serves pure-decode steps only
(prefill throws in qwen3_5.cpp)...
```

The parity gate `tests/vt/test_tp_forward.cpp` (60/60) already proves TpShard /
TpAllReduceSum + in-process shard collectives produce the tp1 result; the real
server never reaches those because of the three named gaps. This file plans and
tracks closing them. **Nothing that would deadlock the collective at tp>1 is
ever reduced to tp1 math silently; each gap keeps a named refuse until its wave
lands.**

## Verified so far (real hardware, 2026-08-21)

The gate below is now beyond the CPU parity test: **the W4 and W5 *primitive*
bricks are each proven on the actual 2-V100 TP target**. `test_nccl_group`
under `CUDA_VISIBLE_DEVICES=1,2` passes **15/15, 197118/197118** — covering, by
name, the dense-body merged gate/up column-slice + down row-parallel reduce
(`vt_cuda_dense_mlp_shard_selfcheck`), the device-sharded GEMM + group
all-reduce (`vt_sharded_forward_selfcheck`), KV/GQA head-shard paged attention,
MoE expert shard, lm_head vocab all-gather — each == the single-GPU result. The
**only unproven link is the runner**: it still binds ONE `queue_`/device and
stages every weight via `ResidentWeight` to that single device (GPUModelRunner
ctors take `vt::Queue queue`; `ModelRegistry::Prepare` runs on it). The W4-pre
construction gate (retained NCCL group, per-rank lane build, no forward) is now
**landed** (`vt_cuda_bridge_rank_lanes_selfcheck`, 16/16 green at 2 GPU) —
what the runner's guarded attach will call; the ctor/device-resident upload and
per-rank forward dispatch (W4-pre remainder / W5) remain open and unfeigned.

## Gates (all real, runnable, must pass before the step can be marked done)

- [ ] **G0 build**: `cmake -DVLLM_CPP_NCCL=ON -DVLLM_CPP_CUDA_ARCHITECTURES=70`
      produces a `vllm-server` that links `libnccl.so.2`.  → DONE (verified)
- [ ] **G1 refusal** (baseline): under 2 GPUs the server exits with the named
      TP-SERVE refuse (not a hang). Final on at start, removed by W7.
      **MEASURED (2026-08-22)**: with the refusal env-bypassed for a probe, a
      real tp2 server boots but STALLS on the first generate (curl timeout,
      GPU 0% both ranks) — the deadlock the refusal predicts is real, confirms
      the seam.
- [ ] **G2 parity** (exists): `tests/vt/test_tp_forward.cpp` — 3 cases, 133/133
      (was 60/60; +W4 loader-slice case). Sharded+AR == tp1, W4 slice are real.
      Must stay green across every wave.
- [x] **G2b device parity** (verified 2-GPU): `tests/vt/test_nccl_group` 16/16,
      197121/197121 on CUDA_VISIBLE_DEVICES=1,2 — per-rank placement, dense
      shard, sharded-forward, KV head-shard, MoE, lm_head + the W4-pre per-rank
      lane-build gate, all == single-GPU / correctly ranked.
- [ ] **G3 flagged 2GPU** (W7, partial): `--tensor-parallel-size N` parses and
      1 === byte-identical tp1; `-tp 2` REFUSED at construction (verified on
      the binary); `-tp` bad values refused. Full `1 <= tp <= visible` + dims-
      divisibility validation gated behind the W5 forward existing.

- [ ] **G4 rank-weight** (W4): each rank loads ONLY its `TpShard` slice of the
      fused bomb weights to ITS device (rank 2 receives its slice); verify via a
      device-memory footprint assert + a shard-checksum broadcast in `test_*`.
- [ ] **G5 forward equality** (W5): a TP2 forward of the 27B dense decoder over
      2 GPUs == tp1 token-exact on the same prompt (greedy). Covers dense +
      full-attn mating / KV — trust the existing SacredGate shape.
      **Forward-MATH half measured 2026-08-24** (`test_op_parity -tc
      'qwen27 dense logits tp==tp1 token gate'`, CUDA_VISIBLE_DEVICES=1,2):
      the in-process `TpThunkComm` gate (single queue + single weight copy,
      shard collectives self-NCCL world=2 in one process) PASSES —
      **"2-GPU dense == tp1 (8 tokens identical)"**. This proves the per-layer
      sharded traversal is token-exact at real 27B dims under a tp group, NOT
      that a 2-rank serve works. **M-B1b landed 2026-08-24** (`43f958fe1`):
      per-device weight residency (`d_dev_pd`/`d_packed_pd`/… keyed by device
      index) makes one weight resident on every lane at once, proven on GPUs
      1+2 by `vt_cuda_residency_selfcheck`. The G5 SERVE half (per-rank
      execute_model dispatch = M-B3) is still open.
      **tp1 baseline measured 2026-08-22** (committed binary, GPU1, real
      Qwen3.8-27B-NVFP4 snapshot 7d6f8d4d...): prompt `"The capital of France
      is"` → `" Paris.\nThe capital of Germany is Berlin.\nThe"` (10 tok,
      temp 0). TP2 must equal this byte-for-byte.
      **Device-leak fix 2026-08-25** (real TP2 serve on GPUs 0+1, scratch
      `VT_TP_ALLOW` probe, this branch): the o_proj `sigmoid_gate_bf16`
      `invalid resource handle` was root-caused to `~CudaCommGroup`'s
      `ncclCommDestroy` loop corrupting the caller's ambient CUDA device to the
      last comm's device (2-GPU group → device 1; `Create()` restores, the
      destructor never did); the destructor now save/restores the caller
      device. With it the tp>1 forward runs cleanly at the device level —
      held device 0 through paged→DBuf→Copy→gate-proj on the same thread —
      and `test_nccl_group` 18/18 / `test_tp_forward` 3/3 / `test_runner` 20/20
      stay green. **Single-lane host-roundtrip path measured UNVIABLE**: the
      tp>1 paged+K/V host-roundtrip forward (per-layer Create/delete g =
      ncclCommInitAll churn ×36/step) is host-bound, creeps host RSS
      3.1→14.6 GiB + threads/FDs + GPU mem, and dies silently (SIGSEGV at
      ~33 min; exit 1 at ~1415s) before a request returns a token — it never
      produced a completion. This is the definitive evidence that M-B3
      (persistent-group reuse + per-rank forward) is REQUIRED, not optional.
      **Memoized paged-group re-measurement 2026-08-25** (scratch, reverted):
      replacing the paged primitive's per-call `CudaCommGroup::Create()`+`delete g`
      with a process-wide memoized singleton (created once, retained) confirms the
      churn is the leak: threads/FDs hold at 58/132 in place of the monotonic
      climb, the server stays alive 21m50s (vs SIGSEGV at ~33 min / exit 1 at
      ~1415s) with no fatal memory death, and RSS moderates to ~7 GiB before
      drifting to ~12 under sustained 100% compute. **It still returns no token
      within 1200 s** (curl RC=28). The leak is only half the blocker: the
      host-bound per-layer host roundtrip is the binding constraint even leak-free.
      The memoized-group change is therefore a validated, re-usable component of
      M-B3 (the paged primitive must reuse a retained group, never Create per
      call), but a token-exact result still requires the per-rank forward dispatch.
      **CORRECTION 2026-08-25 — the "tp==tp1 token gate" is a TAUTOLOGY, not a
      compute proof.** Re-ran `test_op_parity.cpp:2433` with `VT_TP_TRACE=1`
      (GPUs 0+1, flock): it completes in ~112 s, "8 tokens identical", with
      ZERO `[TP-TRACE] mlp-shard` markers — the timers I added to
      `DenseMlpBlock`'s tp>1 fp4 branch are unconditional and compiled into the
      same `libvllm.a` the serve logs prove lit; their ABSENCE means the tp>1
      branch never executes in the gate. So the gate runs tp1 for both
      `run_greedy(nullptr)` and `run_greedy(&TpThunkComm{world=2})` — the
      thunk's no-op `AllReduce` sets `tp_size()==2` yet the sharded forward
      never engages — and "8 tokens identical" is trivially tp1==tp1. It proves
      NOTHING about the sharded compute. **The REAL measured blocker** (live
      G5 serve trace, VT_TP_TRACE, 2026-08-25): the paged attention shard is
      fast + correct (`vt_cuda_attn_kv_shard_paged_run` rc=0 in ~0.52 s/layer),
      but `DenseMlpBlock`'s tp>1 branch is a **host-intermediate per-token
      scaffold** — per layer it host-decodes the full 27B fp4 gate/up/down
      (~1.3 s) then for EACH token assembles + uploads the FULL merged
      gate/up (~411 MB) + down (~205 MB) to every rank's device
      (`vt_cuda_mlp_shard_run`, nccl_communicator.cu:756-766), ~4 s/token
      (measured `per-token-loop 20.2 s` for T=5). 25 layers × T tokens ≈
      ~100 s/layer → the T=5 prefill alone ≈ 2500 s ≫ the 1200 s curl timeout:
      THIS is the no-token-in-1200 s blocker, and it is the dense MLP scaffold,
      NOT the attention shard. M-B3's real fix: replace the host-intermediate
      fp4 MLP shard with a device-resident per-rank fp4 GEMM (each rank holds
      its `TpShard` slice on its own device via M-B1b residency; per-token
      activation; per-rank partial; ONE `TpAllReduceSum`), making the tp>1 MLP
      serve-viable the same way the attention shard already is.
      **M-B3 progress 2026-08-25 — `vt_cuda_mlp_shard_runT` landed (batched,
      process-lifetime group):** replaced the fp4 branch's per-token
      `vt_cuda_mlp_shard_run` with a batched primitive that reuses a
      process-lifetime `CudaCommGroup` and uploads each rank's gate/up/down
      slice ONCE per layer, then runs the T-token partial-GEMM + AllReduceSum
      loop (nccl_communicator.cu `vt_cuda_mlp_shard_runT`; fp4 branch rewired
      in qwen3_5.cpp `DenseMlpBlock`; declared qwen3_5.cpp + tp_shard_host.h).
      Validated model-free: **test_nccl_group 19/19 incl. a new T-token runT
      selfcheck** vs the host reference (GPUs 0+1). Reachability: a real tp2
      serve now GRINDS THROUGH the prefill (~870 s for the 5-token prompt)
      where the per-token scaffold previously never left the first layers —
      so runT removed the host-roundtrip wall. **Two walls remain, both
      precise:** (a) `DenseMlpRankPartial` is a NAIVE kernel — ONE thread per
      output row doing serial `per×H` MACs — so the tp>1 MLP is ~870 s per
      5-token prefill (unusable; needs a tiled/efficient per-rank GEMM, the
      real M-B3 kernel work); (b) the tp2 server silently `exit(1)` after the
      ~870 s forward (no stderr; silent abort, exit 1 — post-forward
      scheduler/sampling path or a CUDA async error), so the request returns
      RC=52 connection-close, never a body. Both must land before G5 serves
      tokens; runT is the validated transport fix, not yet a serving primitive.
- [ ] **G6 prefill** (W6): the tp>1 full-attn **prefill** (batched, variS)
      completes and == tp1; tp>1 pure-decode from E4 already equals. Dense
      prefill heads the "pending wave" throw and is replaced.
- [ ] **G7 full serve** (E2E): `CUDA_VISIBLE_DEVICES=1,2 ./vllm-server -tp 2
      --model SNAP` answer a *few* requests (1 short decode + 1 multi-token
      prefill+decode) with correct output; log shows sharded block active, not a
      refusal; no hang / no crash.

## Step 1 — W4: per-rank device weight load

Refusal naming: "per-rank loader (TpShard at the dense-weight-load chokepoint)
+ the runner's per-rank device load".

- [x] locate `dense_weight_loaders.h` merged-BF16/flow paths that carry `tp`
      (TpShard) — the chokepoint the refusal names.
- [x] confirm `TpShard` slice lands per-rank, not full-on-rank-0.
      (loader test_square, constituent-wise window, green)
- [x] add unit test: 2-rank in-process shard, each rank's loader slice == its
      TpShard window; concat reproduces the full tensor (test_tp_forward W4
      case, `test_tp_forward.cpp:96-97`, 133/133 SUCCESS). — CPU-only, done.
- [x] **device-side W4 verification on the real TP2 box**: 2 GPUs
      (CUDA_VISIBLE_DEVICES=1,2) `test_nccl_group` → **15/15, 197118/197118
      PASS** — per-rank placement (`loader_slice_selfcheck`), dense-body
      gate/up+down shard, device-sharded GEMM + group all-reduce, KV/GQA
      head-shard, MoE expert shard, lm_head all-gather all == single-GPU.
- [ ] thread `tp` (rank) into the **device-resident upload**, so each rank
      uploads its slice to its own device via the `DevicePool`/`ResidentWeight`.
      NOTE: the on-device loader selfcheck already proves the placement math;
      this is the runner wiring (W4-pre/W5) that reaches it at tp>1.

> **3-GPU harness (FIXED, 2026-08-21)**: `test_nccl_group` previously FAILED
> 8 cases at CUDA_VISIBLE_DEVICES=1,2,3 (loader-slice `reconstruction col 63
> 0.000`, MLP/MoE/lm_head rc) — a test-harness world-modulo bug, not engine:
> the harness hardcoded dims (N=64, I=16/32/512/1024, V=64) divisible by W=2/4
> but NOT by W=3, then asserted an exact shard and reported a mismatch. Fixed:
> every fixed-dim self-check/test now SKIPS (returns 2) at a world that does
> not divide its shard dimension — the same "<2 GPUs" skip discipline — so the
> suite passes at **2 (16/16, 197121) AND 3 (16/16, 196823) GPUs**. Product
> `TpShard` is an exact `dim/world` split (vLLM mirror), so a non-divisible
> world genuinely cannot shard — the harness now acknowledges rather than
> fabricates.

## Step 2 — W4-pre: per-rank device group / runner fan-out

Refusal naming: "runner's per-rank device load". The runner binds ONE device.

- [x] **runner ctor accepts N per-rank `Queue`/device** (landed): lanes ctor
      (LoadedModel& + owned-model + MoE/Dense weight overloads), `tp_queues_`
      member, public `tp_lane_count()`. tp1 (empty/one-lane) is byte-identical.
      Verifed: single-GPU host `test_runner` 20/20/548 PASS; multi-GPU host the
      ctor hits the G1 refusal (honest baseline, not silent tp1). The `-tp`
      flag (W7) will feed these; per-rank version of a TP-configured serve.
- [x] `attach_tp_group` **lane-aware + guarded** (landed): builds the retained
      group from the per-rank lanes when this runner was given >1 (`tp_queues_`
      size), else DeviceCount; the world>1 TP-SERVE refusal still fires for any
      multi-lane construction. tp1 (one lane / ≤1 device) is byte-identical.
      The per-rank comm ranking (`cq->Rank(queue_.device.index)`) is unchanged;
      the raw wrap is proven by `bridge_rank_lanes_selfcheck`. This is the
      seam W5's per-rank forward replaces the guard to reach.
- [x] **construction gate (landed)**: a retained NCCL group acquires + each
      rank lane builds its own `vllm::TensorParallel` (tp_size==W, rank==r,
      live per-lane TpShard window) with NO forward math. `vt_cuda_bridge_
      rank_lanes_selfcheck` + test case in `test_nccl_group.cpp`. 2-GPU:
      **16/16, 197121/197121 PASS**; tp<=1 host stays inert (skip). The
      engine's world>1 serve guard is untouched.
- [ ] **W5 (the real remaining seam)**: per-rank device forward — the tp-aware
      blocks already exist and take `input.tp`; wiring the runner to iterate
      its per-rank lanes for a step is the open gate. Guard stays until tokens
      prove equality (G4/G5/G6 not yet reachable).

## Step 3 — W5: per-device forward

Refusal naming: "per-device forward". The tp-aware blocks already exist.

**Dispatched-implementer contract**: `.agents/specs/tp2-per-rank-forward-w5.md`
(committed in this branch) names scope, the already-true verified bricks, the
G5 token-exact gate, risks (collective deadlock → re-arm refusal; delete-call-site
mutation), and the evidence contract. This is the W5 launch contract, not a
parallel same-tree edit — a fresh implementer executes it from the committed
spec and a fresh reviewer verifies (per `.agents/verification.md`).

- [ ] run the tp-aware  dense-block forward on its rank's device, reaching the
      existing `TpAllReduceSum` seams at `dense_attn_block.h:553` and
      `MlpDown`.
- [ ] head-sharded decode KV (`vt nv*cuda_attn_kv_shard_paged_run` + the
      num/den AllReduceSum) on rank devices; output full `[T,Hq,Dh]` exactly.
- [ ] G5: forward equality token-exact on a dense 1-token decode.

## Step 4 — W6: tp>1 prefill (the hardest)

Refusal naming: "paged full-attention tp>1 serves pure-decode steps only".

- [ ] head-shard prefill KV-mix + the num/den AllReduceSum for a batched
      variable-length prompt (the pure-decode path's twin, `S>1`).
- [ ] dims: `S = max(prompt lengths)`, per-head masked row; result full.
- [ ] G6: prefill tp2 == tp1 token-exact; DENSE prefill heads-up gone.

## Step 5 — W7: CLI / server flag

- [x] add `--tensor-parallel-size N` to `vllm_server` (params/`EngineParams`),
      default 1 (byte-identical tp1). **Landed + verified on the real binary**:
      `-tp 0`/`-tp -2` refused at parse (`must be >= 1` + usage);
      `-tp 1` (default) passes the gate and loads normally; **`-tp 2` is
      refused at construction** (`FromModelDir`, before path/weight I/O) with a
      message naming the open W5 forward seam — never a silent tp=1.
      `server_main.cpp` (Args field + parse + Usage), `model_loader.h`
      (EngineParams::tensor_parallel_size), `model_loader.cpp` (construction
      refusal).
- [ ] validate at construction: `1 <= tp <= visible devices`; divisible dims
      rejected loudly (mirroring `Utils::divide` assert). CAST: tp>1 is
      currently refused BEFORE the `<=visible` checks land; the full range /
      divisibility validation is wired when the forward (W5) exists to be
      validated against.
- [ ] re-enable the serve (remove G1 refusal only after W4+W5+W6 gates);
      G7 full lineage.

## Step 6 — docs / records

- [ ] `docs/FEATURES.md` + `docs/USAGE.md` note the `-tp 2` surface and the
      nvcc-collectives transport requirement.
- [ ] ledger / spec `## Outcome` measured at the milestone.
- [ ] keep TP_PLAN.md gates + step checkboxes current through the run.

## Stop conditions

- Any wall (W5, W6) that a test shows the collective never returns or a rank
  never reaches its per-rank weights: the tree re-arm the named refusal rather
  than reduce tp>1 to tp1 math. The affordance of the parity gate (shard AR) is
  never hoised into a "serve works" claim — the server gate is the server.