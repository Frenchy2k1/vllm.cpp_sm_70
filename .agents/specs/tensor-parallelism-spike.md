# Tensor parallelism — end-to-end scope spike (task #287)

Records-only spike (NO build, NO GPU, NO download) · written 2026-08-08 ·
branch `row/SPIKE-TENSOR-PARALLELISM` (draft PR #143), base `main`
`b38f78a7` · owner claim `CLAIM-TP-SPIKE-287`. Pinned vLLM oracle
`555967922` (0.26.0.dev0) — `git -C ${VLLM_SOURCE} rev-parse HEAD` verified
`5559679229bc961848b121ccdeaa8fa5d79bec98` before any line below was read;
every `vllm/...` path is `file:line` in that tree. Our paths are `file:line`
at the spike base.

This is the spike that lets TP rows run as READY W-bricks: the at-pin upstream
inventory (S1), the our-side architecture decisions (S2), the test/hardware
strategy with a locally-runnable gate (S3), and the ranked W-plan (S4). It
BUILDS ON three prior deliverables rather than re-deriving them:

- [tensor-parallelism.md](tensor-parallelism.md) (task #50, 2026-07-10, at the
  OLD pin `e24d1b24`): the deep per-layer TP semantics for the GATE models —
  GDN sharding (§2.3), MoE-under-TP (§2.4), NVFP4 K%16 lockstep (§2.6), the
  divisibility table (§2.7), and the ~82/129 all-reduces-per-step decode
  inventory (§3). Those SEMANTICS are unchanged at `555967922`; where line
  numbers moved, this spec's S1 re-cites them at the current pin.
- [scale-out-distributed.md](scale-out-distributed.md): the W1/W2 lanes that
  LANDED — `vt::Communicator` + the CPU in-process transport, OpId routing,
  the multi-device registry, the derive-and-ship NCCL TU, and the first TP
  forward seams. §1.5 below is the landed-vs-claimed audit with our
  `file:line`.
- [parallelism-modes.md](parallelism-modes.md) (task #164): the mode
  enumeration at THIS pin — TP is P1, and its rank-group is
  `all_ranks.view(-1, TP)` (`vllm/distributed/parallel_state.py:1804`).

## Protocol compliance map

| Required field | Grounded content |
|---|---|
| Row IDs | `BACKEND-DISTRIBUTED-TP` (owning backend row), `PAR-TP` (engine inventory); rider `SPEC-DSPARK` |
| Scope | single-node TP=2 first on the classic-dense path, then gate models + 4/8; §S1, §S4 |
| Upstream chain | S1 table — every TP-touched upstream file at `555967922` |
| Our baseline | S1 "our seam" column + the §1.5 landed-vs-claimed audit |
| Port map | S1 verdicts + S2 decisions (process model, Forward seam, C-ABI, loader) |
| Tests to port | §S3 (upstream distributed tests → tiers; the TP2-on-CPU gate design) |
| Gates | §S3: G-CPU (locally runnable NOW), G-GPU/G-PERF (PENDING-HW) |
| Dependencies | ≥2-GPU host for G-GPU (absent); NCCL headers for the transport build; nothing else |
| Work breakdown | §S4 TP-W0..TP-W7 with effort/dependency/gate; CPU-completable bricks named |
| Risks/decisions | §S2 recommendations + §S5 risks (reduction-order numerics, upstream bar honesty) |

## S1 — upstream inventory at the pin (what TP touches, and our seam for each)

Verdicts: **REUSE** = our seam exists and carries the work; **PARTIAL** = seam
exists, TP>1 semantics missing; **NEW** = no seam.

| # | Upstream mechanism | vLLM @ `555967922` | Our seam @ `b38f78a7` | Verdict |
|---|---|---|---|---|
| 1 | `GroupCoordinator` — per-group rank/world + device communicator; every collective bypasses at `world_size==1`; `graph_capture` wraps capture in the custom-AR context | `vllm/distributed/parallel_state.py:358` (class), `:638` (`all_reduce`; the world-size-1 bypasses live in each entry point `:654,670,701,736,749`), `:595` (graph_capture), accessors `get_tp_group:1365`, `get_tensor_model_parallel_world_size:2039` / `_rank:2044` | `include/vt/communicator.h:47-76` (`Communicator`: rank/world_size + AllReduce/AllGather/Send/Recv, `world_size()==1` no-op documented `:44-46`); `vllm::TensorParallel` handle `include/vllm/model_executor/models/tensor_parallel.h:26-31` | **REUSE** — the abstraction is landed and CPU-gated; a `parallel_state`-style *group table* (TP group from the rank layout `parallel_state.py:1784-1804`) is a thin TP-W1 addition |
| 2 | TP group init — rank layout `ExternalDP×DP×PP×PCP×TP`, TP group = `all_ranks.view(-1, TP)` | `parallel_state.py:1718` (`initialize_model_parallel`), `:1784-1793` (layout comment + `arange(world_size).reshape`), `:1804` (TP unbind); `init_distributed_environment:1560` | none — our W1 communicator is constructed directly with (rank, world); no group math exists | **NEW (small)** — at TP-only world = TP, the layout collapses to one group; port the reshape math so PP/DP groups later are mechanical |
| 3 | Device communicators + all-reduce dispatch: symm-mem → QuickReduce(ROCm) → FlashInfer → aiter(ROCm) → **custom-AR** → **pynccl** → torch | `vllm/distributed/device_communicators/cuda_communicator.py:273-339`; `pynccl.py:60` (class), ops `all_reduce:166`, `all_gather:199`, `send:322`, `recv:349`; `pynccl_wrapper.py:168-341` (the 18-symbol NCCL C surface; capture-safety rationale `:4-23` — bare C calls are legal during CUDA-graph capture) | `src/vt/cuda/nccl_communicator.cu:85` (`NcclCommunicator`, mirrors pynccl symbol-for-symbol, built only under `-DVLLM_CPP_NCCL=ON`, CMakeLists.txt:1162; **never compiled** — no NCCL host); OpId dispatch `include/vt/ops.h:262-265` (`kAllReduce/kAllGather/kSend/kRecv`) via `OpProvider` keyed on DeviceType | **PARTIAL** — NCCL TU is derive-and-ship; build-verify + a per-op `cudaSetDevice` affinity (`src/vt/cuda/cuda_backend.cu:306-311` names it RESIDUAL) are TP-W5 |
| 4 | Custom all-reduce (the decode-latency fast path): worlds {2,4,6,8}, single-node, IPC P2P buffers, one-shot vs two-shot kernels, graph-buffer registration | `custom_all_reduce.py:51` (class), `:52` (worlds), `:105` (world gate), `:230` (`should_custom_ar`), `:211` (`register_graph_buffers`); kernels `csrc/custom_all_reduce.cuh:300` (`cross_device_reduce_1stage`), `:323` (`_2stage`), dispatch `:585-587` | none | **NEW** — TP-W6 (perf brick, not correctness); one-process threads make it SIMPLER (peer pointers via `cudaDeviceEnablePeerAccess`, no IPC handles — keep the registration interface shape so the port stays mechanical) |
| 5 | `ColumnParallelLinear` — output-dim shard; loader narrows checkpoint at `tp_rank*shard_size`; optional gather; **bias sharded with the output** | `vllm/model_executor/layers/linear.py:418` (class), `:478` (divide), `:559-572` (weight_loader narrow on `output_dim`), `:599-601` (gather_output all-gather — off on every path we port) | `TpShard` `tensor_parallel.h:45-52` (the divide + contiguous range, `dim % tp` asserted); applied at the merged-BF16 chokepoint `include/vllm/model_executor/models/dense_weight_loaders.h:137` | **REUSE** (bf16 dense); other dtypes/loaders per §S2d |
| 6 | `MergedColumnParallelLinear` / the `shard_id` machinery — each constituent (gate\|up, q\|k\|v) sharded INDEPENDENTLY then packed; loader called per-shard-id with offset math | `linear.py:660` (class), `:718` (`validate_shard_id`), `:745` (weight_loader, per-shard narrow `:822,871`); QKV variant `:1021`, heads `:1074`, **kv replication when `tp ≥ total_kv_heads`: `num_kv_heads=1`, replicas `:1075-1079`**, per-shard-id k/v offsets `:1308` | `dense_weight_loaders.h:100-137` (`LoadMergedBf16RawNK` shards each named constituent independently then concatenates — the same physical rule); **KV replication NOT modeled** (stated in-file `:129-132`) | **PARTIAL** — even-split done; head-aware KV replication (needed for 35B kv=2 at TP=4+, 27B kv=4 at TP=8) is TP-W2 |
| 7 | `RowParallelLinear` — input-dim shard; partial-product **all-reduce**; bias fused on rank 0 only | `linear.py:1612` (class), `:1665` (divide), `:1716-1728` (loader narrow on `input_dim`), `:1762` (bias rank-0), `:1766` (all-reduce) | all-reduce seams LANDED: o_proj `include/vllm/model_executor/models/dense_attn_block.h:544` (NVFP4) + `:555` (bf16), MLP-down `src/vllm/model_executor/models/qwen3.cpp:104` (`TpAllReduceSum`, `tensor_parallel.h:58-64`); **input-dim WEIGHT shard: none** (`TpShard` is only applied to output rows) | **PARTIAL** — the forward seam exists; the row-shard loader + rank-0 bias rule are TP-W2 |
| 8 | Vocab-parallel embedding + LM head — vocab-dim shard, masked lookup + **all-reduce**; logits **all-gather** | `vocab_parallel_embedding.py:198` (class), `:302` (per-partition), `:342-345` (shard indices), `:489-491` (masked_fill + all-reduce); `ParallelLMHead:505`; `logits_processor.py:85-96` (`_gather_logits`: all-gather `:93` when `use_all_gather`, else gather-to-rank-0 `:96`), trim `:152` | none — our embed/lm_head are whole-vocab (e.g. classic-dense forward + lm_head in `qwen3.cpp`) | **NEW** — TP-W2/W4; see S1#12 for who samples |
| 9 | Attention-head split + validation — heads%tp enforced at config; per-rank `num_heads`/`num_kv_heads` flow from QKVParallelLinear into the attention layer and the KV spec | validation `vllm/config/model.py:1280-1285` (raise "must be divisible by tensor parallel size"); head math `linear.py:1074-1079`; per-rank `num_kv_heads` reaches the KV cache spec via the attention module (`vllm/v1/worker/gpu_model_runner.py:7759`, `num_kv_heads=attn_module.num_kv_heads`, inside `get_kv_cache_spec:7774`) | `dense_attn_block.h:329` takes `tp` but does NOT divide `Hq/Hkv`; our KV spec takes `num_kv_heads` from the model spec (`src/vllm/v1/worker/gpu/runner.cpp:405` `initialize_kv_cache`, `:528-540` `Hkv = attn_spec->num_kv_heads`) | **PARTIAL** — a per-rank "sharded dims" view (divide at model-build time, don't mutate `HfConfig`) makes the runner + KV cache per-rank FOR FREE, since both already read the spec value. **KV-cache-per-rank implication for OUR paged runner: each rank's runner instance allocates `Hkv/tp` heads — same block tables, same page geometry, ZERO paging-logic change** |
| 10 | MoE under TP (vs EP) — TP mode: every rank ALL experts, each sharded on the intermediate dim; ONE combine all-reduce (shared expert folded); EP flips to whole-expert ownership | `use_ep` decision `vllm/model_executor/layers/fused_moe/config.py:1204` (needs `enable_expert_parallel`); TP-mode per-partition intermediate `routed_experts.py:127-137`; per-shard-id w1/w3/w2 loaders `:303-318` (+ w13 scale `:319-332`); final-reduce gate `layer.py:229` / `skip_final_all_reduce` `config.py:1308`; EP map `expert_map_manager.py:22` | our grouped MoE path (35B `qwen3_5.cpp`, keep-quant grouped GEMMs) is single-rank; `qwen3_moe_weights.cpp:103-111` loads per-expert tensors through the SAME merged loader | **NEW** (TP-W6) — the shard rule is w13 column + w2 row PER EXPERT, so S1#5/#7 machinery applies inside the expert loop; EP is out of scope here (`BACKEND-DISTRIBUTED-EP`) |
| 11 | Multiproc executor — one OS process per rank, shm `MessageQueue` RPC broadcast, ModelRunnerOutput from ONE rank | `vllm/v1/executor/multiproc_executor.py:103` (class), `:176-182` (per-rank spawn), `:151` (broadcast MQ), `:354-410` (`collective_rpc`, enqueue `:388`), `:247` + `:509-523` (`output_rank = world − tp×pcp` = TP-rank-0 of the last PP stage) | `include/vllm/v1/executor/executor.h:5-28` + `src/vllm/v1/executor/executor.cpp` — the collapsed direct-call executor whose header EXPLICITLY documents the collective_rpc collapse; this is the N-rank fan-out seam | **PARTIAL** — §S2a decides threads-not-processes; the fan-out (N runners, output from rank 0) is TP-W4 |
| 12 | Sampler / logits at TP — **who holds the full vocab row**: at this pin `Platform.use_all_gather()` defaults **True** (`vllm/platforms/interface.py:1096-1101`, no CUDA override) → `_gather_logits` ALL-GATHERS (`logits_processor.py:93`) → EVERY TP rank holds full logits and the V2 sampler runs identically on every rank; the executor returns rank 0's output (`multiproc_executor.py:509-523`). PP is different: sampled ids broadcast from the last stage (`vllm/v1/worker/gpu/model_runner.py:1439,1465`) | | our sampler is per-runner and on-GPU (`include/vllm/v1/worker/gpu/runner.h:380`; the born-on-runner seam) | **REUSE** — with all-gather mirrored, EVERY rank runs the existing sampler unchanged; rank 0's `ModelRunnerOutput` is returned. No cross-rank sampling protocol needed (greedy + seeded sampling are deterministic given identical logits) |
| 13 | Config plumbing — `tensor_parallel_size`, `world_size = PP×TP×PCP` | `vllm/config/parallel.py:124` (field), `:824-828` (world_size), heads validation S1#9 | `EngineParams` (`include/vllm/entrypoints/model_loader.h`) has no TP field; C-ABI §S2c | **NEW (small)** — TP-W7 |
| 14 | GDN / hybrid + NVFP4 sharding for the GATE models | semantics UNCHANGED from [tensor-parallelism.md](tensor-parallelism.md) §2.3 (GDN: all shards on the head/channel dim, state rank-local, ONE all-reduce at out_proj; max TP = num_k_heads = 16), §2.6 (NVFP4: weight uint8 [out, in/2] + fp8 scale [out, in/16] narrow in lockstep, K-shard %16 enforced; scale_2/alpha replicated; **shard FIRST in linear layout, THEN swizzle per rank**), §2.7 (both gate models cleanly TP=2/4/8) | GDN blocks + NVFP4 weights live in `qwen3_5.cpp` / `qwen3_5_weights.*`; no TP threading there yet | **NEW** (TP-W6) — the old spec's math carries over verbatim |

**S1 headline:** the communication layer is ~70% reusable TODAY (abstraction +
CPU transport + OpId routing + registry landed; NCCL TU written, unbuilt), the
forward seams are landed-but-partial on ONE model path, and the genuinely new
work concentrates in FOUR places: the row/input-dim + vocab weight sharding
(#7/#8), the per-rank dims view (#9), the executor fan-out (#11), and the gate
models' MoE/GDN/NVFP4 threading (#10/#14). Weighted by the S4 effort column,
roughly **40% of the end-to-end TP surface is already landed or directly
reusable**; the correctness-critical remainder is CPU-completable except the
NCCL build.

## §1.5 — landed vs claimed (the W1/W2 audit)

Verified in the tree at `b38f78a7` (spike base):

- **LANDED, real:** `include/vt/communicator.h` + `src/vt/communicator.cpp`
  (W1 CPU in-process multi-rank transport; gate `tests/vt/test_communicator.cpp`);
  OpId routing `include/vt/ops.h:262-265`; per-`Device{type,index}` registry
  (`include/vt/backend.h:191-193`, `kMaxDevicesPerType=16`) with the CUDA
  registrar enumerating every GPU (`src/vt/cuda/cuda_backend.cu:294-324` — the
  device-0 hardcode is gone); the NCCL TU `src/vt/cuda/nccl_communicator.cu:85`
  behind `-DVLLM_CPP_NCCL=ON` (CMakeLists.txt:1162); `TensorParallel`/`TpShard`/
  `TpAllReduceSum` (`tensor_parallel.h:26-64`); o_proj all-reduce seams
  `dense_attn_block.h:544,555`; MLP-down seam `qwen3.cpp:104`; merged-column
  loader shard `dense_weight_loaders.h:137`; gate
  `tests/vt/test_tp_forward.cpp:80` (TP-2 MLP == tp1, RED-verified) + `:181`
  (tp=1 inertness).
- **Claimed loosely, now stated precisely:** "TP in the Qwen3-dense forward"
  means the tp handle threads from `LayerForward` down
  (`qwen3.cpp:92,114,126,136`) — **it dead-ends there**: no caller above the
  layer (model forward, `ModelRegistry::Forward`, runner) passes anything but
  the default `nullptr`, and NO production loader passes `tp` into
  `LoadMergedBf16RawNK` (verified: every call site — `qwen3_moe_weights.cpp:103`,
  `gemma_weights.cpp:56`, `phi_weights.cpp:171`, … — omits it). The ONLY tp>1
  driver in the tree is the toy MLP test.
- **Missing entirely (no code):** row/input-dim weight sharding, per-rank
  Hq/Hkv division, vocab-parallel embed/LM-head/logits gather, the N-rank
  executor fan-out, any GDN/MoE/NVFP4 TP threading, config/ABI plumbing.
  `should_custom_ar`-class fast paths: nothing.

## S2 — our-side architecture decisions (analysis + recommendation; user-scope calls stay open)

### S2a — process-vs-thread orchestration

vLLM runs one OS process per rank because of Python: the GIL, one CUDA context
per process, and pickled RPC over a shm ring (`multiproc_executor.py:151,388`).
The entire apparatus (TCP-store rendezvous `parallel_state.py:1560`, unique-id
broadcast `pynccl.py`, shm MessageQueue, death pipes) exists to stitch
processes back together.

Our engine is one C++ process. **Recommendation: one process, one thread per
rank** for single-node TP:

- CUDA is explicitly multi-device-per-process: one context per device,
  `cudaSetDevice` per thread. Our registry already addresses
  `Device{kCUDA, i}` per rank (`cuda_backend.cu:294-324`); the named residual
  is per-op device affinity (`:306-311`).
- NCCL supports single-process multi-device: `ncclCommInitAll`, or grouped
  `ncclCommInitRank` per thread — no unique-id exchange, no TCP store. (The
  one NCCL rule: group the per-thread init calls, one comm per device.)
- CUDA-graph decode: each rank captures ITS OWN graph on its own device/stream;
  NCCL collectives are capture-legal as bare C calls (the pynccl design point,
  `pynccl_wrapper.py:4-23`), and our known trap — capture bakes stack addresses
  — is already doctrine (use pooled buffers, warmup all-reduce before first
  capture). Ranks capture/replay in lockstep because the executor fan-out
  synchronizes steps.
- `collective_rpc` collapses to an in-process fan-out: N rank-runners on N
  threads, `SchedulerOutput` passed by const ref (zero serialization), output
  taken from rank 0 — mirroring `output_rank` (`multiproc_executor.py:509-523`).
  This is exactly the collapse `executor.h:5-28` documents for world=1,
  extended to world=N.
- **Deviation to record** (porting-inventory §9, as the #50 spec already
  planned §4.2): WorkerProc/MessageQueue/TCP-store are NOT ported for
  single-node TP. They return with multi-node (2×Spark), where process
  isolation is forced. The `Executor` seam keeps vLLM's method surface so a
  multiproc executor can drop in without touching `EngineCore`.

### S2b — where TP slots into `ModelRegistry::Forward` and the shared runner

Today: one `GPUModelRunner` (`runner.h:128`) owns one device queue, one KV
cache, one sampler, and calls the registered forward. The tp handle already
threads INSIDE the classic-dense forward but dead-ends at the layer boundary
(§1.5).

**Recommendation — additive TP>1 branch, TP=1 byte-identical (structurally
guaranteed):**

1. `LoadedModel` carries an optional per-rank `TensorParallel` (comm + rank),
   set only when `tp_size>1`; `ModelRegistry::Forward`'s model-side plumbing
   passes it down to the already-landed layer seams. `tp==nullptr` enqueues
   ZERO extra ops and takes whole-tensor shards — the SACRED single-GPU paths
   never see a changed instruction stream (proven pattern:
   `test_tp_forward` inertness case).
2. One `GPUModelRunner` INSTANCE per rank (own device, queue, DevicePool, KV
   cache with `Hkv/tp` from its sharded spec, own decode graphs). The runner
   itself needs no TP knowledge beyond its spec values (S1#9) — attention,
   paging, sampling are rank-local; logits all-gather (S1#12) happens at the
   model's lm_head epilogue, INSIDE Forward, so every rank samples full-vocab
   identically.
3. The executor becomes the N-rank fan-out (S2a). `EngineCore::step` is
   unchanged.
4. MUST-route seams are unaffected: fusion catalog, merged-GEMM family and the
   born-on-runner decode all run rank-locally on tensors that are simply
   narrower; the collectives sit BETWEEN them at the four seam points S1
   enumerates.

### S2c — C-ABI surface

Mirror vLLM: add `int32_t tensor_parallel_size` to `vllm_model_params`
(`include/vllm.h:162`), `<=0 => 1`, next to the existing capacity knobs —
the exact analogue of `ParallelConfig.tensor_parallel_size`
(`vllm/config/parallel.py:124`) and `--tensor-parallel-size/-tp` on the CLI
and server (examples are ABI clients ONLY — the flag reaches them through the
ABI, never an internal header). ABI cascade: `VLLM_ABI_VERSION` is **14** at
the spike base (`include/vllm.h:122`) with a v15 bump in flight on another
lane; the TP field takes THE NEXT bump at whatever number is current when
TP-W7 lands (appending a field = minor-compatible by our convention, but the
version constant still moves; do not hardcode "v16" in code comments).
`world_size` validation (heads divisibility, S1#9) happens at engine load and
returns `VLLM_ERR_INVALID_ARGUMENT` — same failure the oracle raises
(`config/model.py:1280-1285`).

### S2d — checkpoint sharding at load (per weight class)

Our loaders are single-rank; upstream narrows every parallel weight at load
via param attributes (`output_dim`/`input_dim`) + `tp_rank*shard_size`
(`linear.py:559-572, 1716-1728`). Map, by weight class:

| Weight class | Upstream rule | Our chokepoint | Work |
|---|---|---|---|
| Merged column (gate_up, qkv, GDN qkvz/ba) | per-constituent dim-0 narrow, then pack (`linear.py:745-871`) | `LoadMergedBf16RawNK` (`dense_weight_loaders.h:100`) — DONE for bf16 | thread `tp` from the model loaders (today no caller passes it) |
| QKV with few KV heads | replicate: `num_kv_heads=1`, replicas `tp/total_kv` (`linear.py:1075-1079`, k/v offset `:1308`) | same chokepoint, marked RESIDUAL in-file (`:129-132`) | head-aware ranges instead of the even split |
| Row (o_proj, down_proj, GDN out_proj, MoE w2) | dim-1 (input) narrow (`:1716-1728`); bias rank-0 | none | NEW: an input-dim `TpShard` at the same loader family; bias loaded on rank 0 only |
| Vocab embed / lm_head | vocab-dim shard + shard indices (`vocab_parallel_embedding.py:302,342-345`) | none | NEW loader + masked-lookup forward |
| NVFP4 packed + block scale | narrow weight `[out, in/2]` and scale `[out, in/16]` in lockstep; K-shard %16; shard linear THEN swizzle ([tensor-parallelism.md](tensor-parallelism.md) §2.6) | `Nvfp4Weight` handling in `qwen3_5_weights.*` | NEW, mechanical once row/col rules exist |
| GGUF quant blocks | no upstream analogue (vLLM shards safetensors) | keep-quant loader | DEFER: gate TP on safetensors first (#50 spec's call, unchanged); GGUF×TP needs block-aligned narrowing (Q8_0 blocks of 32 divide our dims — flagged, not scoped) |
| Per-expert (MoE w13/w2) | same col/row rules inside the expert loop (`routed_experts.py:303-318`) | `qwen3_moe_weights.cpp:103-111` / 35B grouped slabs | reuse rules per expert; grouped-GEMM slabs narrow per rank at build |

Load cost note: each rank reads only its slice (the narrow happens on the
mmap'd shard view before the staging memcpy), so TP=2 load does NOT double IO;
the GB10 staging recipe (context-first, shard release) applies per rank.

## S3 — test / hardware strategy

### The locally-runnable gate: G-CPU, TP2-on-CPU token-exact

The only gate runnable on this box (and in CI) rides the W1 CPU in-process
transport (`src/vt/communicator.cpp`: N ranks = N host threads over one
generation-barrier + staging slots — a REAL cross-rank reduction). Design,
precisely:

- **Which model:** the classic-dense Qwen3 path (`Qwen3ForCausalLM`) — the ONE
  path whose forward seams are already TP-threaded (§1.5), with two tiers:
  - **G-CPU-1 (CI, no checkpoint):** a synthetic 2-layer Qwen3-dense config
    (small dims chosen from the real divisor structure: Hq=4, Hkv=2, Dh=8,
    ffn=64, vocab=128) built in-memory through the REAL loader chokepoints.
    Drive `ModelRegistry::Forward` at tp=1, then at tp=2 (2 ranks = 2 threads,
    each with its `TensorParallel{comm,rank}` and its own sharded
    `LoadedModel`), same token stream, greedy. Assert: per-step logits within
    f32 tolerance AND greedy tokens EXACT, both ranks. New
    `tests/vt/test_tp_model_forward.cpp` beside the existing toy gate; the
    RED-first mutation is deleting one of the four all-reduce seams.
  - **G-CPU-2 (dev-box, env-gated on the HF snapshot):** real Qwen3-0.6B bf16,
    the same 16-token greedy prompts as the existing SACRED gate
    (`tests/parity/test_qwen3_paged_engine.cpp:420`), tp=2 threads vs the
    committed tp=1 goldens — token-exact under the near-tie policy below.
- **How ranks exchange:** only through `Communicator` (AllReduce at the four
  seams, AllGather at logits); the test harness owns the barrier lifetime;
  rank 0's output is the gate subject (mirrors `output_rank`).
- **What it certifies:** the sharding algebra + collective placement over real
  model code — NOT device transport (that is G-GPU). The consistency-trap
  memory note is answered by anchoring: the tp=1 side of the comparison is the
  SAME path the committed oracle goldens gate, so TP2≡TP1 chains to
  TP1≡oracle.

### The honest correctness bar (verified at the pin)

The brief's premise "vLLM guarantees TP2 token-exact with TP1 for greedy" is
**not what upstream tests assert**:
`tests/basic_correctness/test_basic_correctness.py::test_models_distributed`
(`:204`) runs vLLM at `tensor_parallel_size=2` and compares greedy tokens
**against HuggingFace** (`check_outputs_equal`, 5 tokens, fp16) — TP2-vs-TP1
token equality is implied only transitively and only where HF-vs-vLLM
equality holds at ALL. Reduction-order reality: RowParallel splits K, so each
rank accumulates K/tp terms before the all-reduce adds partials — a different
f32/bf16 association than the full-K GEMM. Near-ties CAN flip. **Our bar,
therefore (mirrors our ratified near-tie doctrine):** TP2 vs TP1 token-exact
STRICT on G-CPU-1 (dims/weights chosen so drift cannot reach argmax ties —
enforced by asserting a minimum top1-top2 logit margin in the fixture);
near-tie DISTRIBUTIONAL on real checkpoints (G-CPU-2, G-GPU), exactly the
policy the 0.6B single-rank gate already applies. Where vLLM-TP2 output on the
same config is obtainable (G-GPU), ours-TP2 vs vLLM-TP2 token-exact is the
PRIMARY comparison (same sharding ⇒ same reduction structure), with TP2-vs-TP1
as the secondary check — the #50 spec's GATE-1 shape, kept.

### Real multi-GPU: PENDING-HW (honest options)

| Option | What it gates | Status |
|---|---|---|
| any 2× sm_80+ box (2×3090/4090, cloud 2×A100) | G-GPU: NCCL build-verify + ours-TP2 vs vLLM-TP2 token gate (bf16 + Marlin W4A16) | **PENDING-HW** — cheapest honest ask (#50 spec) |
| 2× sm_120 (5090/RTX PRO 6000) | G-PERF at NVFP4 W4A4 fidelity near GB10 | PENDING-HW |
| 2× DGX Spark over ConnectX-7 RoCE (#154 spike, Leg 2) | multi-NODE TP/PP — NOT a single-node substitute (custom-AR disabled multi-node; multi-node bootstrap in scope there) | PENDING-HW; belongs to `BACKEND-DISTRIBUTED-MULTINODE-SPARK` |
| cluster nodes (Thor sm_110, Orin sm_87, DGX sm_121) | each SINGLE-GPU — no TP gate possible | not viable |

### Tests to port (delta over the #50 spec §7 table, which stands)

`tests/distributed/test_pynccl.py` (incl. `_with_cudagraph`) → 2-GPU nightly;
`test_comm_ops.py` → CPU transport now / NCCL nightly; `test_custom_all_reduce.py`
→ with TP-W6; `test_multiproc_executor.py` semantics re-expressed for the
thread executor; shm/TCP-store cases SKIPPED-tracked to multi-node.

## S4 — ranked W-plan

Numbering continues the scale-out lane (its W1/W2 landed; W3 NCCL / W4 PP /
W5 Spark / W6 MLX remain there). TP bricks are `TP-W*` to avoid collision.
"CPU" = completable on this box with the in-process transport.

| Brick | Content | Effort | Depends on | Gate | CPU? |
|---|---|---|---|---|---|
| TP-W0 | this spec + rows (records) | done | — | record checkers | ✅ |
| TP-W1 | GroupCoordinator-analog: rank-layout math (`parallel_state.py:1784-1804`) → a TP group table over `vt::Communicator`; thread-local rank accessors; `LoadedModel` carries the per-rank `TensorParallel` | 1-2 d | — | unit: group math == upstream reshape for TP∈{1,2,4,8}; tp=1 inertness re-held | ✅ |
| TP-W2 | complete parallel-linear semantics + sharded loader on the classic-dense path: row/input-dim shard, rank-0 bias, per-rank Hq/Hkv dims view, QKV kv-replication ranges, vocab embed + lm_head shard + logits all-gather | 4-6 d | TP-W1 | unit per weight class (shard+concat == whole); the four seams RED-mutated | ✅ |
| TP-W3 | **G-CPU gate**: TP2-on-CPU token-exact model forward (design in S3), G-CPU-1 in CI + G-CPU-2 env-gated | 2-3 d | TP-W2 | G-CPU-1 strict; G-CPU-2 near-tie vs committed goldens | ✅ |
| TP-W4 | executor fan-out + per-rank runner instances + per-rank KV spec (`Hkv/tp`) + per-rank decode-graph capture discipline (comm-warmup before capture, pooled buffers) | 3-5 d | TP-W2 | engine-level TP2-on-CPU: full `EngineCore::step` loop, 2 ranks, tokens == tp1 engine | ✅ (CPU engine) |
| TP-W5 | NCCL real-GPU path: build-verify the TU, per-op `cudaSetDevice` affinity (`cuda_backend.cu:306-311`), `ncclCommInitAll` bring-up, graph-captured collectives | 2-4 d | TP-W4 + **2-GPU box** | G-GPU: ours-TP2 vs vLLM-TP2 token gate, then graphs-on re-hold | ❌ PENDING-HW (affinity code itself compiles CPU-side) |
| TP-W6 | gate models + MoE-TP + GDN + NVFP4 (the #50 §2.3-2.7 semantics) + custom-all-reduce one-shot port (`csrc/custom_all_reduce.cuh:300`) for decode latency | 1-2 wk | TP-W5 | GATE-1/2/3 of the #50 spec: token gates both models, then ≥ vLLM TP=2 every axis, same box | ❌ (MoE/GDN shard math unit-testable on CPU) |
| TP-W7 | ABI + server: `tensor_parallel_size` on `vllm_model_params` (+version bump), CLI/server flags, load-time validation | 1-2 d | TP-W4 | capi suite + conformance re-run with `-tp 2` (CPU threads) | ✅ |

Critical path to a MERGEABLE, gated increment: TP-W1→W2→W3 — entirely
CPU-completable, ~1.5-2 wk. TP-W4+W7 complete the CPU story; TP-W5/W6 wait on
hardware exactly where §S3 says.

## S5 — risks / decisions

- **Reduction-order numerics** (S3): strict-vs-distributional split is the
  decision; margin-asserting fixtures make the CI tier strict without lying.
- **Upstream-bar honesty:** upstream's own TP test is vs HF, not vs TP1 —
  recorded above so nobody cites a guarantee vLLM does not make.
- **Thread model is a recorded deviation** (S2a), returned to upstream shape
  at multi-node.
- **Do not shard GGUF in v1** (S2d) — safetensors first, GGUF×TP flagged.
- **Graph capture discipline** is the known-trap surface: comm init + warmup
  before first capture, pooled comm buffers, lockstep capture/replay.

---

## Appendix A — DSPARK speculator rider (INVENTORIED, grounded at the pin)

See [dspark-speculator-note.md](dspark-speculator-note.md) — the 5-line
grounding note (USER-requested 2026-08-08). Rows: `SPEC-DSPARK`
(engine-matrix, refreshed at pin `555967922`), the feature-matrix §8 DSpark
row, and the roadmap line under Scale-out/speculative breadth. Full scoping is
its own future spike (`planned: specs/dspark-spec-decode.md`).
