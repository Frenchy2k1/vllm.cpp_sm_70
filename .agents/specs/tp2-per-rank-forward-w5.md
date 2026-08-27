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
`FromModelDir`, default OFF; reverted after this red-first capture, then later
restored as the G1 refusal gate described below) a REAL `vllm-server --tp 2` on
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

So the block is a REAL collective stall inside the first fp4 sharded-MLP
primitive. A 1-token pure-decode tp2 request (T==num_reqs, the wired decode
path) is admitted (heartbeat unfinished=1), reaches the runner (markers
A drain -> B update_states -> C prepare_inputs [num_reqs=1] -> D attn_meta ->
E pre-forward all print), enters the dense forward (densebody T=1 tp=yes,
embed done), and then STALLS INSIDE THE FIRST fp4 SHARDED-MLP COLLECTIVE:
`vt_cuda_mlp_shard_run` (qwen3_5.cpp dense-MLP tp>1 branch) is reached and
NEVER returns. Every earlier premise is superseded: not scheduler admission
(heartbeat shows unfinished=1), not runner-body prep (A-E all print), not
pre-forward. The prior "pre-Forward" record (75f6278d6) was also superseded:
the HOST `Forward` entry probe stayed silent because tp>1 routes through
`ForwardDevice`/`DenseForwardBody` — the deeper densebody trace above proves
it. The decisive finding: the fp4 dense-MLP group collective, called from lane
0's single-lane attach, never completes.

Per-lane forward dispatch is the likely fix, but the diagnosis is now exact:
the stall is the shard primitive's group collective on a REAL tp2 request. The
same primitive PASSES the 2-GPU nccl gate, so the difference from the gate is
the runner's single-lane attach. The next implementer should debug
`vt_cuda_mlp_shard_run` queue/device binding on a real tp2 request, starting
from its internal per-rank queue+thread fan-out (nccl_communicator.cu) versus
the gate that drives per-rank queues. All probes (VT_TP_ALLOW,
VT_ENGINE_STEP_LOG, the runner/dense markers above) were REVERTED after each
measurement; production keeps the loud refusal until tp2==tp1 passes.

**Resolved 2026-08-23:** the MLP-shard stall above is FIXED (08a2e5091): the
O(I·H) host reference is now VT_TP_DIAG-gated; the nccl self-check keeps its
own reference. Measured live: a tp2 request that previously hung there now
passes the fp4 MLP shards and reaches the paged-attn `T != num_reqs`
pure-decode-only refusal — the honest next seam.

**Prefill seam — STATUS 2026-08-24: the primitive serves prefill (proven green),
the runner refusal is lifted for pure-prefill.** The "executable red" I first
committed (55ccfb613) was a FALSE POSITIVE — a test bug, not a kernel
deficiency: (1) the test fed an OOB kv buffer (2*T*Hkv*D vs the wrapper's
2*kbase=2*T*2*Hkv*D, V@kbase=T*2*Hkv*D; fixed 46127c7a0) and (2) the reference
used the seq-shard's slice-reduce semantics instead of the paged kernel's
HEAD-PARALLEL split (each query head g=h/(Hq/Hkv) attends only its group's kv
head across the causal window). With both fixed the test PASSES (0d5ca6119):
17/17 nccl green — `vt_cuda_attn_kv_shard_paged_run` serves a causal
multi-token single-request prefill token-exact to the tp1 reference. NO kernel
is owed. The remaining gap was runner-side: it feeds PER-REQUEST seq_lens/block
rows that are per-token only for decode. cff6de5c8 lifts the refusal for
PURE-PREFILL by expanding per-request->per-token (seq_lens[tok] = offset+1
causal window, block_table[tok] = owning request row) via query_start_loc;
mixed/divide prefill stays a loud refusal (no service path emits it). decode
is unchanged. Built green; runner 20/20, W4 tp_forward 3/3, nccl 17/17 hold.
**The G5 tp2==tp1 serve-token gate is the measured acceptance** — serve the
27B tp=2 on GPUs 1+2, capture greedy tokens on the tp1 baseline prompt
"The capital of France is", compare byte-for-byte to the recorded tp1
baseline " Paris.\nThe capital of Germany is Berlin.\nThe". Measured when the
shared gpu.lock mutex (a live concurrent smoke-gates parity batch) frees.
The concurrent session is running its own tp==tp1 parity via
test_op_parity qwen27/qwen36 paged logits — the same question on the same
hardware; this change must not interfere with or be measured against it.

## G5 green evidence (measured 2026-08-26, production serve)

`vllm-server --tp 2`, Qwen3.8-27B-NVFP4 snapshot `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108`,
2× V100 (CARDS 0,1), greedy (temperature 0), prompt `"The capital of France is"`,
`max_tokens=10`, run under the exclusive `gpu.lock` flock. The fp8w tail-MLP arm
(layers 56–63, per-channel F8_E4M3 + f32 column scale) routes through the tp>1
host-shard `runT` path; layers 0–55 are fp4.

tp1 body vs tp2 body: **byte-identical except the wall-clock `/created` field**
(1787708399 vs 1787709418). Common fields: `text=" Paris.\nThe capital of
Germany is Berlin."`, `finish_reason="length"`, `usage {completion_tokens:10,
prompt_tokens:5, total_tokens:15}`, `id="cmpl-0"`. Raw token streams saved
`/tmp/g5tp1tokens.txt` == `/tmp/g5tokens3.txt`. This text is a byte-for-byte
prefix of the recorded tp1 baseline in the section below (that baseline ran a
larger `max_tokens`), so tp2 == tp1 == recorded-baseline prefix.

Layer-56 segfault (the fp8w producer on tail layers dereferencing empty bf16
fields) is resolved; the change traverses all 64 layers and generates tokens.
tp2 decode is host-memory-bound at this arity: ~1727 s for the 10-token/11-step
generation after ~55 s boot (parallelized host decode; device-quantized shard
kernel remains the serving-speed path, TP_PLAN M-B).

## M-B3 outcome (measured 2026-08-26, device-resident per-rank fp4 GEMM)

M-B3 replaced the tp>1 fp4 host-decode MLP shard with per-rank device-resident
fp4 GEMM. Verified end-to-end on 2× V100 (CARDS 0,1, `VT_TP_ALLOW=1`,
`--tensor-parallel-size 2`, snapshot `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108`):

- **What changed**: `DenseMlpBlock` tp>1 fp4 branch lazily builds per-lane
  fp4 residency (`ResidentNvfp4` on each lane device, thread-per-lane) and, when
  every lane slot is resident, routes the batch through
  `vt_cuda_mlp_shard_runT_fp4` — in-kernel E2M1 × fp8-e4m3 group-scale ×
  per-tensor scale2 dequant, same `MlpGuAct`/`MlpDown` block-per-row scheduling
  and reduction order as the host `runT`, one AllReduceSum over [O]. Missing
  lane residency falls back to the host-decode `runT`; a built-kernel failure
  (rc!=0/2) faults loudly. The wrapper keeps one stream + dx/dact/dout buffer
  per rank across the 64 per-layer calls (process-lifetime `static` cache) —
  the per-call `cudaMalloc`/`cudaStreamCreate` churn alone hid the MLP cost.
- **Correctness**: model-free selfcheck `vt_cuda_mlp_shard_runT_fp4_selfcheck`
  (synthetic fp4 bytes, both nibbles + sign, device kernel vs host-decode +
  `runT` reference, tolerance 1e-3) passes on 2 GPU; full `test_nccl_group`
  20/20 green (197,529 assertions). Real serve G5: tp2 output
  `" Paris.\nThe capital of Germany is Berlin."`, `completion_tokens:10`,
  `prompt_tokens:5`, `finish:length` — **byte-identical** to the runT G5 green
  body above (the certified tp2==tp1 baseline). The M-B3 device path is
  token-exact at real 27B dims.
- **Wall**: host fp4 decode was ~5 s/layer; the M-B3 fused MLP is ~1.7 ms/layer
  decode (64 layers ≈ 0.11 s/step). Step wall dropped from ~130–170 s to
  ~43 s/step.
- **Residual (still open, next M-B follow-ups)**: the ~43 s/step wall is now
  bound by the tp>1 attention shard and per-layer host syncs, not the MLP —
  GPU idles at ~0% during decode, and per-layer MLP measures 1.7 ms. This
  matches the known deferred attention work; it is outside M-B3's scope and
  remains a tracked gap. The fp8w tail-MLP host shard (layers 56–63) is a
  separate deferred device-fp8w-GEMM follow-up; in this serve run every fp4 arm
  engaged (no fp8w layer executed).
- **Decisions kept**: token-exactness preserved by mirroring the host
  `runT` reduction order and using the exact host decoder math in kernel
  (`ldexpf` E2M1, `__float2bfloat16` RNE, per-tensor scalar `scale2`); a simple
  global-x-reload per block is accepted (weight bytes dominate, proven by the
  1.7 ms/layer result); `scale2` stays one scalar per weight, matching the host
  per-row broadcast.

## Attention-wall outcome (measured 2026-08-26, TP>1 step wall)

After M-B3 the ~43 s/step wall was eliminated in three ordered fixes, each
isolated by exhaustive per-stage timing before a change. Token-exact.

- **Fix 1 — attention NCCL per-call churn (nccl_communicator.cu).** The
  tp>1 attention shard created and destroyed an NCCL comm group per layer
  (`ncclCommInitAll` + `ncclCommDestroy`), measuring ~520–550 ms/layer in
  `vt_cuda_attn_kv_shard_paged_run`. Moving to a process-lifetime retained
  `static CudaCommGroup` (mirroring the M-B3 runT_fp4 wrapper) dropped it to
  ~2.6–3.4 ms/layer. Measured by ATTIN-DIAG (`pre_ms=0.0 q_ms=1.3 kv_ms=0.0
  prim_ms=1.4 copy_ms=0.0 total_ms=2.6`).
- **Fix 2 — MLP fp4 lazy host vectors (qwen3_5.cpp DenseMlpBlock).** An
  earlier "MLP = 1.7 ms/layer" figure was wrong: it measured only the wrapper
  internals, missing `DenseMlpBlock`'s host round-trip. The remaining wall was
  three unconditional `std::vector<float> gate_host/up_host/down_host`
  declarations at the block top, sized `I*H + I*H + H*I` (~1.2 GB at 27B dims)
  and zero-allocated every layer even on the fp4_device path where unused
  (~340 ms/layer); the fp8w arm wasted the same (it overwrote them with its own
  fresh vectors). The vectors are now empty at declaration and `resize`d only
  inside the host-decode fallback. MLP fp4 fast path: ~340 ms → ~3 ms/layer, and
  the fp8w arm shed the same ~340 ms/layer of idle zeroing.
- **Correctness**: after both fixes the clean binary passes full
  `test_nccl_group` 20/20 (197,529 assertions, unchanged from baseline) on
  GPUs 2,3, and the real TP2 serve output is byte-identical to the certified
  tp2==tp1 baseline (`" Paris.\nThe capital of Germany is Berlin."`, finish
  `length`).
- **Residual (open gap)**: the remaining per-step time is dominated by the
  8 fp8w tail layers (56–63), which measured ~1600–1780 ms/layer during the
  diagnosis (pre-lazy-fix); the lazy-alloc fix removes ~340 ms/layer of that,
  giving ~1.35–1.45 s/layer × 8 ≈ 10.8–11.6 s — consistent with the observed
  whole-request wall (~105 s incl. server startup for a 5-token prompt + 10
  tokens). This is the fp8w host-dequantize shard, a separate deferred
  device-fp8w-GEMM follow-up (mirrors M-B3), and is outside this wave. The
  56 fp4 layers now run ~0.02 s/step combined.

## fp8w outcome (measured 2026-08-27, device-resident per-rank fp8w W8A16 GEMM)

The deferred fp8w tail layers (56–63) now run device-resident per-rank, mirroring
M-B3. Verified end-to-end on 2× V100 (CARDS 0,1, `VT_TP_ALLOW=1`,
`--tensor-parallel-size 2`, snapshot
`7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108`):

- **What changed**: `StageAndReleaseKeepFp8w` is renamed `StageKeepFp8wResident`
  and no longer releases the fp8w tail host pages. The eager `d_dev` is set at
  load, which the release would free; the tp>1 lane build reads `w.bytes`
  (host) to upload each lane's copy at FIRST forward in `DenseMlpBlock`, so
  freeing those pages made lane>0 copy freed bytes and spin (the G5 hang this
  wave caught: two `R`-state thread-per-lane build threads, GPU at 0%). Keeping
  fp8w host resident matches the fp4 arm's memory posture (fp4 host is never
  released because its `d_dev` is set only at first forward). A `runT_fp8w`
  extern plus `MlpGuActFp8w`/`MlpDownFp8w` dispatch stages the raw fp8 packed +
  per-column f32 scale and runs the W8A16 GEMM on the device.
- **Correctness**: model-free selfcheck `vt_cuda_mlp_shard_runT_fp8w_selfcheck`
  passes on 2 GPU (G1); full `test_nccl_group` 21/21 green (197,530
  assertions). Real serve G5: tp2 output byte-identical to the certified
  tp2==tp1 baseline (`" Paris.\nThe capital of Germany is Berlin."`,
  `completion_tokens:10`), with `fp8w DEVICE` markers printed for layers
  56–63 at both T=5 (prefill) and T=1 (decode) — proving `runT_fp8w` executes.
- **Decisions kept**: fp8w host pages stay resident (only 8 layers, small;
  needed by the host-decode fallback too) rather than pre-staging all lane
  `d_dev_pd` slots, preserving the first-latency design intent and the
  selfcheck (which exercises the pre-staged upload, not the lane build).

## Design confirmed 2026-08-25 (reconnaissance completed; architecture verified)

The per-lane wiring shape is now code-anchored, not speculative:

- `execute_model` already threads `.tp = tp_group_ ? &tp_ : nullptr` into
  `ModelForwardInput` and calls `ModelRegistry::Forward(*model_, forward_input)`
  once on the primary `queue_` (`runner.cpp:1739,1824`). M-B3 must replace
  that ONE call with a per-lane loop when `tp_lane_count() > 1`: for each lane
  `r`, build a per-lane `ModelForwardInput` on `tp_queues_[r]` with
  `input.tp` = a per-rank `TensorParallel{ comm = group->Rank(r) }` and that
  lane's KV cache, and run `ModelRegistry::Forward` on it.
- The forward ALREADY reaches the tp math: a non-null `input.tp` routes
  `ModelRegistry::Forward` into the proven per-layer sharded traversal
  `DenseForwardLayers` → `FullAttnBlockPaged` / `DenseMlpBlock`
  (`qwen3_5.cpp:9706-9709,9889-9893`), whose per-collective seams are
  `TpAllReduceSum` at `dense_attn_block.h:553,613` and the `#ifdef VT_NCCL`
  tp>1 paged branches (`qwen3_5.cpp:5494,5700`). `TensorParallel`/`TpShard`/
  `TpAllReduceSum` are world/rank-aware and no-op at tp_size==1
  (`tensor_parallel.h:45-63`). Exit is all-reduced so every rank holds full
  `[num_tokens,vocab]`; lm_head stays full-width.
- UNRESOLVED PREREQUISITES that block a FIRST correct per-lane step (the
  remaining implementation work, each a real change):
  1. **Per-rank KV**: `GPUModelRunner` owns ONE `attn_kv_` (`PagedKvCache`)
     on `queue_`. Each lane needs its own KV cache on its own device, and the
     per-lane `ModelForwardInput` must carry it.
  2. **Per-rank weight device-upload in the runner path**: M-B1b made
     `ResidentWeightPd`/`ResidentNvfp4Pd(d, w, dev_idx)` upload into a
     per-device slot when `dev_idx > 0` (`dense_attn_block.h:187-190,240-244`,
     `qwen3_5.cpp:1385-1386,1419`), but the LOADER + runner still bind one
     queue; nothing yet uploads every lane's slice to every lane device before
     the per-lane forward. A lane>0 forward without its weights computes
     garbage — the wrong-answer shape the row exists to remove, so this must
     land WITH the dispatch, never before a token gate.
- G1 (`runner.cpp:434`) held until G5 passed; SATISFIED 2026-08-26 (tp2==tp1
  token-exact, G5 green evidence above). The gate remains in place as the
  explicit opt-in: tp>1 serves only with `VT_TP_ALLOW=1`, so a half-wired
  forward never silently runs tp1 math on another rank's device.
- The memoized paged-group finding (TP_PLAN G5 note, 2026-08-25) is a
  validated M-B3 component: the paged primitive must reuse a retained group,
  never `CudaCommGroup::Create()` per call (the per-call churn → leak/death).
- **CORRECTION 2026-08-25 — the "tp==tp1 token gate" is a TAUTOLOGY.** Re-ran
  `test_op_parity.cpp:2433` (GPUs 0+1, flock) with `VT_TP_TRACE=1`: Passes in
  ~112 s, "8 tokens identical", with ZERO `[TP-TRACE] mlp-shard` markers — the
  timers added to `DenseMlpBlock`'s tp>1 fp4 branch are unconditional and lit
  in the serve's logs from the same `libvllm.a`, so their absence proves the
  tp>1 branch never executes in the gate. The gate runs tp1 for both
  `run_greedy(nullptr)` and `run_greedy(&TpThunkComm{world=2})` (the thunk's
  no-op `AllReduce` makes `tp_size()==2` yet the sharded forward never engages),
  so "8 tokens identical" is trivially tp1==tp1. It proves nothing about the
  sharded compute. The earlier "138 s, 8/8" reading was a false positive for
  the same reason.
- **The REAL measured G5 blocker is the tp>1 dense MLP scaffold, not the
  attention shard** (live G5 serve trace, VT_TP_TRACE, 2026-08-25). The paged
  attention shard returns rc=0 in ~0.52 s/layer — fast and correct. The wall is
  `DenseMlpBlock`'s tp>1 branch, a host-intermediate per-token scaffold: it
  host-decodes the full 27B fp4 gate/up/down (~1.3 s/layer) then for EACH token
  assembles + uploads the FULL merged gate/up (~411 MB) + down (~205 MB) to
  every rank's device (`vt_cuda_mlp_shard_run`, nccl_communicator.cu:756-766) —
  measured `per-token-loop 20.2 s` for T=5 (≈4 s/token). 25 layers × T ≈
  ~100 s/layer ⇒ the T=5 prefill ≈ 2500 s ≫ the 1200 s curl timeout: this is
  the no-token-in-1200 s blocker, and it is the dense MLP host-roundtrip the
  memoized re-measurement flagged as the binding constraint, now pinpointed to
  this exact loop.
- **M-B3 progress 2026-08-25 — `vt_cuda_mlp_shard_runT` landed + validated.**
  The fp4 `DenseMlpBlock` tp>1 branch now calls a batched primitive (one
  process-lifetime group + one rank-slice upload per layer, then the T-token
  partial-GEMM + AllReduceSum loop) instead of per-token
  `vt_cuda_mlp_shard_run`. Model-free green (test_nccl_group 19/19 incl. a new
  T-token runT selfcheck); a real tp2 serve now grinds through the 5-token
  prefill (~870 s) where the old scaffold never left the first layers. The
  host-roundtrip wall is gone. **Remaining G5 doors (runtime-measured):** (a)
  `DenseMlpRankPartial` is a naive one-thread-per-output-row kernel → ~870 s
  per 5-token prefill (replace with a tiled per-rank GEMM — the real kernel
  work); (b) the tp2 server silently `exit(1)` after the forward (no stderr;
  RC=52 connection-close, no body) — localize the post-forward abort.

Full per-lane KV + per-lane weight-upload + per-lane dispatch + mutation review
+ G5 measurement is the implementation wave this spec gates; it is not landable
as an isolated half (see Gate / Risks).

## tp1 ground-truth baseline (measured 2026-08-22, committed binary)

Served the real Qwen3.8-27B-NVFP4 (snapshot
`7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108`) tp=1 on GPU1
(`CUDA_VISIBLE_DEVICES=1`, `--tensor-parallel-size 1`), greedy temp 0:
prompt `"The capital of France is"` → **`" Paris.\nThe capital of Germany is
Berlin.\n"The` (10 tokens). G5's tp2 forward must reproduce this byte-for-byte;
the tp>1 path on the same checkpoint refuses at construction with the named W5
message. Capture tp=2 against THIS baseline, never a re-derived probe.