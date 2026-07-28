# Parallelism / distributed-execution modes — enumeration spike

User-directed enumeration spike (records-only; NO build, NO GPU, NO download).
Base `main` — see the closing ledger/state entries for the exact SHA. Pinned
vLLM oracle `555967922` (0.26.0.dev0); every `file:line` below is in that tree
(`${VLLM_SOURCE}` = `/home/mudler/_git/vllm/vllm`, verified
`git rev-parse HEAD == 5559679229bc961848b121ccdeaa8fa5d79bec98`). Owner claim:
`CLAIM-PARALLELISM-MODES-SPIKE`.

This spike **enumerates every parallelism / distributed-execution mode vLLM
has**, grounds each in real upstream source, and maps it onto our unifying
`vt::Communicator` seam (the process-group abstraction landed by
`CLAIM-SCALE-OUT-W1`) and the scale-out W-plan
([scale-out-distributed.md](scale-out-distributed.md)). It is the mode-level
companion to that spike: scale-out scoped the three *transport legs* (multi-GPU
/ multi-Spark / MLX); this scopes the *parallelism modes* that ride them.

**Honest headline.** vLLM has **five true world-partition dimensions**
(TP, PP, DP, PCP, DCP) laid out as one rank tensor
`ExternalDP × DP × PP × PCP × TP` (`parallel_state.py:1785`), plus **two modes
that are NOT their own world dimension**: **EP** (a re-grouping of the
DP×PCP×TP ranks that changes *which experts* a rank owns, `parallel_state.py:1892`)
and **SP** (a *compilation pass* that rewrites TP's all-reduce into
reduce-scatter→norm→all-gather, `compilation/passes/fusion/sequence_parallelism.py:498`;
NOT a process group). Both are flagged as such below — the user's mode list
item #5 (SP) is real in vLLM but is a TP-mode optimization, not a standalone
parallel axis.

`world_size = PP × TP × PCP` (`config/parallel.py:824-828`); DP is *outside*
world_size (`world_size_across_dp = world_size × DP`, `:549`); DCP reuses the TP
GPUs (must divide TP); EP spans DP×PCP×TP.

---

## The enumeration table

| # | Mode | What it does | vLLM `file:line` | Config flag | Comm pattern | Our-seam map (`vt::Communicator`) | Reuse-vs-new vs W1/W2 | Priority |
|---|---|---|---|---|---|---|---|---|
| 1 | **Tensor Parallel (TP)** | Megatron column/row-parallel: shard each linear's weight across ranks, split attention heads & KV heads, shard vocab embed / LM head; every rank runs every layer on a slice | `layers/linear.py:418` (Column, out-dim 0) / `:1612` (Row, in-dim 1) / `:1021` (QKV heads `:1074`) + `vocab_parallel_embedding.py:198` + attn split `llama.py:143-153` | `--tensor-parallel-size N` (`config/parallel.py`, `tensor_parallel_size`) | **all-reduce** after RowParallel (`linear.py:1766`), o_proj, MLP-down, MoE combine, and the vocab embed (`vocab_parallel_embedding.py:491`); **all-gather** of vocab-sharded logits (`logits_processor.py:85`) | `Communicator::AllReduce` after o_proj / MLP-down / combine; `AllGather` for logits; TP rank-group = the `all_ranks.view(-1, TP)` group (`parallel_state.py:1802`) | W2 (`BACKEND-DISTRIBUTED-TP`): ~70% new forward logic, insertion points UNAMBIGUOUS (`dense_attn_block.h:342,513-530`, `qwen3.cpp:83-90`); weight-slice chokepoint `dense_weight_loaders.h:131-138`. `AllReduce`/`AllGather` REUSE W1 | **P1** |
| 2 | **Pipeline Parallel (PP)** | Split the layer stack into contiguous stages, one stage per rank group; rank *r* runs `layers[start:end]`, sends the boundary hidden state to rank *r+1* | `models/utils.py:785` (`PPMissingLayer(nn.Identity)`) / `:798` (`make_layers`, `:823-827` pad with missing layers) + `distributed/utils.py:127` (`get_pp_indices`, env `VLLM_PP_LAYER_PARTITION`) + `parallel_state.py:957` (`send_tensor_dict`) | `--pipeline-parallel-size N` (`config/parallel.py`, `pipeline_parallel_size`) | point-to-point **send/recv** of `IntermediateTensors` (hidden+residual) between adjacent stages; no all-reduce added | `Communicator::Send`/`Recv` (landed in W1) on the PP rank-group `all_ranks.transpose(2,4).view(-1,PP)` (`parallel_state.py:1856`); executor fan-out to N stage-workers | W4 (`BACKEND-DISTRIBUTED-PP`): ~80% new — a `PPMissingLayer` analogue in the model construction + multi-worker executor (`executor.cpp:7-34` direct call → fan-out). `Send`/`Recv` REUSE W1 | **P3** |
| 3 | **Data Parallel (DP)** | Run N independent engine replicas over the SAME weights; a coordinator load-balances requests and keeps a global "request wave" so all DP ranks step together (required when DP wraps an EP MoE — dispatch is a cross-DP collective) | `v1/engine/coordinator.py:23` (`DPCoordinator`, wave `:33-56`) + `v1/worker/dp_utils.py:164` (`coordinate_batch_across_dp`) `:53` (per-step `num_tokens_across_dp` all-reduce) + group `parallel_state.py:1866` | `--data-parallel-size N` (+ `data_parallel_size_local`, `data_parallel_rank`, `data_parallel_master_ip/port`; `config/parallel.py:129-145`); external LB `data_parallel_external_lb` | per-step **all-reduce** of the token-count vector across DP ranks (batch coordination `dp_utils.py:53`); the request-wave sync is a coordinator RPC, not a tensor collective | DP is engine-level replication: N `LoadedEngine` instances + a coordinator; the per-step token all-reduce is `Communicator::AllReduce` on the DP rank-group `all_ranks.transpose(1,4).view(-1,DP)` (`parallel_state.py:1866`) | ~90% new (a NEW DP coordinator + engine-replica executor); the token-count `AllReduce` REUSES W1. **Depends on the multi-worker executor (W4)** | **P4** |
| 4 | **Expert Parallel (EP)** | For MoE: instead of every rank holding ALL experts sharded on the intermediate dim (TP mode), each rank owns a DISJOINT subset of WHOLE experts; tokens are all-to-all dispatched to the owning rank and all-to-all combined back | `fused_moe/expert_map_manager.py:22` (`determine_expert_map`, `local_num_experts` split `:69`, `-1` non-local `:72-87`) + `use_ep` decision `fused_moe/config.py:1204` + EP group `parallel_state.py:1892` (= DP×PCP×TP ranks, MoE-only `:1892`) + all-to-all backends `fused_moe/all2all_utils.py:44` (DeepEP HT/LL/V2) | `--enable-expert-parallel` (`config/parallel.py:165`) + `--all2all-backend` (`:188`, default `allgather_reducescatter`; `deepep_high_throughput`/`deepep_low_latency`/`deepep_v2`/`mori`/`nixl_ep`) | **all-to-all dispatch** (tokens→expert-owner) then **all-to-all combine** (results→token-owner) on the device communicator (`base_device_communicator.py` `dispatch:379`/`combine:398`); trailing TP all-reduce gated by `skip_final_all_reduce` | new collective `OpId::kAllToAll` (`dispatch`/`combine`) routed through the EP rank-group; the expert-subset selection is the loader/router edit (`qwen3_moe_weights.cpp:35-38` per-expert loop) | ~60% new (EP shard + all-to-all); rides the SAME `Communicator` with an added all-to-all primitive (NOT in W1 — W1 has AllReduce/AllGather/Send/Recv only) | **P2** |
| 5 | **Sequence Parallel (SP)** — *NOT a standalone dimension* | A **compilation pass** that, when TP is on, rewrites `AllReduce → RMSNorm (→Quant)` into `ReduceScatter → local RMSNorm (→Quant) → AllGather`, so the residual/norm is sharded along the sequence dim (each rank normalizes its 1/TP token slice). Enables downstream GEMM+RS / AG+GEMM fusions | `compilation/passes/fusion/sequence_parallelism.py:498` (`SequenceParallelismPass`) + enabled in `compilation/passes/pass_manager.py:147` + gated `config/compilation.py:1506` (`tp_size>1 and enable_sp`) | `compilation_config.pass_config.enable_sp` (`config/compilation.py:129`) + `sp_min_token_num` (`:179`) + `fuse_gemm_comms` (AsyncTP, `:133`) — **NOT a `--*-parallel-size` knob** | replaces one **all-reduce** with a **reduce-scatter + all-gather** pair over the TP group (same total bytes, different split); no new rank group | `Communicator::ReduceScatter` + `AllGather` on the TP group; for us this is a FORWARD-STRUCTURE choice (fuse RS/AG into the norm), not a new group. `ReduceScatter` is the one collective W1 does not have | ~50% new on top of W2-TP: needs `ReduceScatter` + the norm-residual split; a portable-fusion surpass-track candidate ([glue-fusion-2026-07-19](glue-fusion-2026-07-19.md)), NOT correctness-critical | **P5** |
| 6 | **Context / long-context parallel (PCP + DCP)** | Shard the SEQUENCE (KV/context) dimension across ranks for long context: **PCP** (prefill-context-parallel) is a real world dimension (part of `world_size`); **DCP** (decode-context-parallel) REUSES the TP GPUs to split the decode KV across ranks (ring/all-to-all attention) | PCP group `parallel_state.py:1836`; DCP group `:1817` (spans PCP then TP); `dcp_comm_backend` a2a `config/parallel.py:541` | `--prefill-context-parallel-size N` (`config/parallel.py:126`) + `--decode-context-parallel-size N` (`:342`, must divide TP) + `--dcp-comm-backend {a2a,...}` | **all-to-all** / ring exchange of KV blocks + partial-attention reduction across context ranks (DCP `a2a` backend `:541`); the context group is `all_ranks.transpose(3,4)` (PCP) / DCP transpose (`parallel_state.py:1817-1836`) | `Communicator` on the PCP/DCP rank-groups; the attention op gains a cross-rank KV-reduction (all-to-all + softmax-rescale), same primitive as EP's all-to-all | ~85% new (context-sharded paged attention is a genuinely new attention path); rides the same `Communicator`. Latent CP scaffolding pre-wired: `block_table.cpp:30-32` (`total_cp_world_size_(1)`), `kv_cache_coordinator.cpp:81-92` | **P6** |
| 7 | **Comm strategies (transport selection)** | ONE collective (`all_reduce`) has multiple backends; vLLM picks the fastest legal one per size/topology: intra-node NVLink one-shot **custom-all-reduce**, **pynccl** (thin NCCL C-API), **symmetric-memory**, **FlashInfer** fused AR, torch.distributed fallback | `device_communicators/cuda_communicator.py:273` (`all_reduce` dispatch, flags `:50-66`); `custom_all_reduce.py:51` (worlds {2,4,6,8}, single-node `:90-96`); `pynccl.py:60`; select in `base_device_communicator.py:147` (backend-plural interface) | `VLLM_ALLREDUCE_USE_SYMM_MEM`, `VLLM_ALLREDUCE_USE_FLASHINFER`, `_ENABLE_CUSTOM_ALL_REDUCE`; `--distributed-executor-backend {mp,ray,uni,external_launcher}` | selection logic, not a new collective: custom-AR (IPC P2P, capture-safe) before pynccl before torch fallback; all capture-safe as bare C calls | `OpProvider` keyed on `(OpId, DeviceType)` (`include/vt/op_provider.h:108`) IS this selection seam: a NCCL provider on `kCUDA`, custom-AR fast-path as a second provider, MLX on `kMETAL` | ~95% new per transport (W3 NCCL provider, custom-AR fast path later); the SELECTION shape (provider table) already exists in `OpProvider` | **P2** (with TP) |
| 8 | **Combos (TP×PP, TP×EP, DP×EP, +CP)** | The world is ONE rank tensor `ExternalDP × DP × PP × PCP × TP` (`parallel_state.py:1785`); each mode is a reshape/transpose+unbind of it into rank-groups. Large-scale DeepSeek serving = **DP×EP** (data-parallel attention + expert-parallel MoE, all-to-all across DP×TP) | rank layout `parallel_state.py:1785-1900` (TP `:1802`, DCP `:1817`, PCP `:1836`, PP `:1856`, DP `:1866`, EP `:1892`); `world_size=PP×TP×PCP` `config/parallel.py:824` | any combination of the above knobs (validated in `config/parallel.py:499-543`) | each group gets its own `GroupCoordinator`/communicator; a step touches multiple groups (TP all-reduce + PP send/recv + EP all-to-all + DP token-sync) | one `vt::Communicator` PER rank-group (TP-comm, PP-comm, EP-comm, DP-comm), each constructed from the same rank-tensor reshape; the model forward calls the right group's collective at each seam | rank-group construction is ~1 helper (the reshape/transpose math ports mechanically); the per-mode work is rows 1-6 | **derived** |

---

## Per-mode detail

### 1. Tensor Parallel (TP) — `PAR-TP` / `BACKEND-DISTRIBUTED-TP`
Fully scoped in [tensor-parallelism.md](tensor-parallelism.md) (heads/KV/GDN/MoE
divisibility for both gate models, NVFP4 K%16 shard rule, the ~82/129
all-reduces-per-step decode-latency inventory). **When vLLM uses it:** the
default multi-GPU knob — split a model too big for one GPU, or to cut latency.
**Comm pattern:** all-reduce is the workhorse (after every RowParallel); logits
all-gather once. **Our seam:** insertion points are unambiguous
(`dense_attn_block.h:513-530`, `qwen3.cpp:90`); `Communicator::AllReduce`
(landed W1) drops in. **Priority P1** — same-host multi-GPU TP is the first
scale-out payoff and the base every other mode builds on.

### 2. Pipeline Parallel (PP) — `PAR-PP` / `BACKEND-DISTRIBUTED-PP`
`make_layers` (`utils.py:798`) builds
`[PPMissingLayer]*start + real[start:end] + [PPMissingLayer]*(N-end)` from
`get_pp_indices` (`distributed/utils.py:127`). **When:** to span nodes (PP
tolerates slower inter-stage links — one send/recv per stage boundary vs TP's
per-layer all-reduce), or to fit a model across boxes. vLLM has **no
interleaved/virtual-pipeline** (1F1B GPipe-style) scheduler in v1 — it is a
simple contiguous stage split with `IntermediateTensors` passing; confirmed no
virtual-PP layer construction in the pinned tree. **Comm pattern:** point-to-point
send/recv only. **Our seam:** `Communicator::Send`/`Recv` (landed W1) + the
multi-worker executor fan-out (`executor.cpp:7-34` is a direct single-worker
call today). **Priority P3.**

### 3. Data Parallel (DP) — `PAR-DP` / `BACKEND-DISTRIBUTED-DP` (NEW row)
N full engine replicas; the `DPCoordinator` (`coordinator.py:23`) tracks a global
"request wave" so all DP ranks execute `generate` together (a stale-wave request
would deadlock — `:38-56`). **DP×EP is the large-scale DeepSeek serving mode:**
DP replicates the attention (each rank a full copy) while EP shards the experts,
so the MoE all-to-all spans DP×TP ranks. **When:** throughput scale-out (more
replicas = more concurrent requests) and to feed a wide EP MoE. **Comm pattern:**
a per-step all-reduce of the token-count vector (`dp_utils.py:53`,
`coordinate_batch_across_dp:164`) so ranks agree on padding/microbatching; the
wave sync is a coordinator RPC. **Our seam:** engine-level replication (N
`LoadedEngine`) + the token `AllReduce`; **depends on the multi-worker executor
(W4).** **Priority P4.**

### 4. Expert Parallel (EP) — `PAR-EP-EPLB` / `BACKEND-DISTRIBUTED-EP` (NEW row)
`determine_expert_map` (`expert_map_manager.py:22`) splits `global_num_experts`
into `local_num_experts` per EP rank (`:69`) with a `-1` sentinel for non-local
experts (`:72-87`). `use_ep` (`fused_moe/config.py:1204`) flips MoE from TP mode
(all experts, intermediate-dim shard) to EP mode (disjoint whole experts).
**When:** very wide MoE (e.g. DeepSeek 256 experts) where per-expert weights >
one GPU, or to remove the redundant all-experts replication of TP mode.
**Comm pattern:** all-to-all **dispatch** (route each token to its expert's owner
rank) then all-to-all **combine** — the DeepEP HT/LL/V2 backends
(`all2all_utils.py:44`) are the high-throughput / low-latency all-to-all kernels;
**DeepEP IS present** in the pinned tree (`prepare_finalize/deepep_{ht,ll,v2}.py`).
EPLB (expert-load-balancer) is a SEPARATE process group with the same ranks
(`parallel_state.py:1918`) to avoid deadlocking MoE-forward collectives against
rebalancing. **Our seam:** a new `OpId::kAllToAll` collective + the expert-subset
loader; rides the same `Communicator`. **Priority P2** (EP is the MoE payoff and
the gate models are MoE). *The scale-out spike currently folds the EP shard into
the TP row's note; this spike promotes EP to its own `BACKEND-DISTRIBUTED-EP`
row because the all-to-all comm pattern is genuinely distinct from TP's
all-reduce.*

### 5. Sequence Parallel (SP) — *compilation pass, not a world dimension*
**HONEST FINDING:** vLLM has **no `sequence_parallel_size` knob and no SP process
group.** SP is `SequenceParallelismPass` (`sequence_parallelism.py:498`), an
Inductor pattern-matcher pass enabled by `pass_config.enable_sp`
(`config/compilation.py:129`) and only when `tensor_parallel_size > 1`
(`:1506`). It rewrites the TP `all_reduce → RMSNorm` into
`reduce_scatter → local RMSNorm → all_gather`, so each rank normalizes its 1/TP
sequence slice; the pass docstring is explicit that this "does not directly
yield performance improvements" but "lays the groundwork for GEMM+ReduceScatter
and AllGather+GEMM fusions" (`fuse_gemm_comms`/AsyncTP, `pass_manager.py:149`).
It requires full-graph compilation (piecewise is unsupported — the residual gets
sequence-split, breaking subgraph boundaries). **The v0.25.0 `PAR-SEQUENCE-MOE`
inventory row** (feature-matrix §3, `PAR-SEQUENCE-MOE`) is the MoE-specific
non-DP SP variant reporting 1.9-5.0% E2E gain. **Our seam:** a forward-structure
choice + `Communicator::ReduceScatter` (the one collective W1 lacks); a
portable-fusion surpass-track item, NOT correctness-critical. **Priority P5.**

### 6. Context / long-context parallel (PCP + DCP)
vLLM's answer to the user's mode #6. **PCP** (`prefill_context_parallel_size`,
`config/parallel.py:126`) is a real world dimension (`world_size = PP×TP×PCP`);
**DCP** (`decode_context_parallel_size`, `:342`) reuses the TP GPUs and must
divide TP, splitting the decode KV cache across ranks with an all-to-all
attention backend (`dcp_comm_backend='a2a'`, `:541`). This is
ring/context-attention: each rank holds a slice of the KV sequence and the
attention output is reduced across context ranks. **When:** contexts too long for
one GPU's KV budget. **Our seam:** the paged-attention op gains a cross-rank
KV-reduction on the PCP/DCP group; latent CP scaffolding is already inert-wired
(`block_table.cpp:30-32`, `kv_cache_coordinator.cpp:81-92`,
`single_type_kv_cache_manager.cpp`, all fixed to `world_size 1, rank 0`).
**Priority P6** (long-context is a later payoff; needs a new attention path).

### 7. Comm strategies (transport selection)
Not a parallelism mode but the mechanism all modes share. `CudaCommunicator`
(`cuda_communicator.py:273`) dispatches one `all_reduce` across up to 5 backends
in order (custom-AR before pynccl before torch fallback); size gates decide
custom-AR eligibility. `pynccl.py` binds the 17-symbol NCCL C-API (capture-safe
by design — torch.distributed's all-reduce issues extra CUDA calls illegal
during graph capture, `pynccl_wrapper.py:4-23`). `custom_all_reduce.py:51` is the
intra-node NVLink one-shot kernel (worlds {2,4,6,8}, single-node `:90-96`,
world==2 exempt from the full-mesh requirement). **Our seam:** `OpProvider`
(`include/vt/op_provider.h:108`) keyed on `(OpId, DeviceType)` IS the
selection table — a NCCL provider on `kCUDA`, a custom-AR fast-path provider,
MLX on `kMETAL`. W3 lands the first (NCCL). **Priority P2** (ships with TP).

### 8. Combos & world partitioning
The world is ONE rank tensor `ExternalDP × DP × PP × PCP × TP`
(`parallel_state.py:1785`, note the layout comment `:1785-1793`). Each group is a
transpose+reshape+unbind: TP `all_ranks.view(-1,TP)` (`:1802`), DP
`.transpose(1,4)` (`:1866`), PP `.transpose(2,4)` (`:1856`), PCP `.transpose(3,4)`
(`:1836`), EP `.transpose(1,2)` over DP×PCP×TP (`:1892`, MoE-only). `world_size =
PP×TP×PCP` (`config/parallel.py:824`); DP is outside; EP has no size of its own.
**The three headline combos:** TP×PP (single big model across a node then across
nodes), TP×EP (dense-attn TP + MoE expert shard on one box), **DP×EP** (the
large-scale DeepSeek serving topology — DP-replicated attention + EP experts,
all-to-all across DP×TP). **Our seam:** one `vt::Communicator` per rank-group,
all built from the same reshape helper; the model forward calls the correct
group's collective at each seam. This is the payoff of ONE abstraction — a combo
is a rank-group table, not a forward rewrite.

---

## How each mode rides the unifying `vt::Communicator`

W1 landed `vt::Communicator` (`include/vt/communicator.h` + `src/vt/communicator.cpp`)
with `rank()`/`world_size()` + `AllReduce(sum/max/min/prod)`/`AllGather`/`Send`/`Recv`,
each stream-ordered on a `Queue&`, ported 1:1 from `DeviceCommunicatorBase`
(`base_device_communicator.py:147`) with the `world_size==1` byte-identical
bypass (`parallel_state.py:638`). The mode → collective map:

| Mode | Collective(s) needed | In W1? |
|---|---|---|
| TP | `AllReduce`, `AllGather` | ✅ both |
| PP | `Send`, `Recv` | ✅ both |
| DP | `AllReduce` (token-count sync) | ✅ |
| EP | `AllToAll` (dispatch/combine) | ❌ NEW primitive |
| SP | `ReduceScatter`, `AllGather` | ❌ `ReduceScatter` NEW |
| CP (PCP/DCP) | `AllToAll` + partial-attn reduce | ❌ NEW |
| Comm-strategy | provider selection, not a collective | `OpProvider` seam exists |
| Combos | one `Communicator` per rank-group | rank-group helper NEW |

**W2 residual on the abstraction:** two collectives (`AllToAll`, `ReduceScatter`)
plus routing collectives through `OpId`/`OpProvider` (deferred from W1). Each mode
constructs its `Communicator` from a rank-group = one reshape/transpose of the
world rank-tensor (the `parallel_state.py:1785-1900` math ports mechanically).
A new interconnect is one transport provider (`(OpId, DeviceType)`), never a
model-forward edit — the PR-#4 additive story.

---

## Priority ranking (user-directed order)

Per the user directive "same-host multi-GPU TP first, then EP/PP/DP, then
SP/context":

1. **P1 — TP** (`BACKEND-DISTRIBUTED-TP`, W2): the base multi-GPU payoff; every
   other mode reuses its sharded linears + rank-group + all-reduce.
2. **P2 — EP** (`BACKEND-DISTRIBUTED-EP`, new) + the **NCCL comm-strategy** (W3):
   the MoE payoff (gate models are MoE) and the transport TP needs to run on real
   hardware.
3. **P3 — PP** (`BACKEND-DISTRIBUTED-PP`, W4): stage split + send/recv +
   multi-worker executor — the cross-node reach.
4. **P4 — DP** (`BACKEND-DISTRIBUTED-DP`, new): engine-replica scale-out;
   completes DP×EP; depends on the W4 executor.
5. **P5 — SP** (`BACKEND-DISTRIBUTED-SP`, new): TP-mode fusion (reduce-scatter),
   surpass-track, not correctness-critical.
6. **P6 — Context (PCP/DCP)**: long-context KV sharding; a new attention path,
   later payoff.

---

## Rows & records

- **New `BACKEND-DISTRIBUTED-*` rows** added to [backend-matrix.md](../backend-matrix.md):
  `BACKEND-DISTRIBUTED-DP`, `-EP`, `-SP` (TP/PP already exist). Distributed table
  5 → 8 rows; BACKEND total 65 → 68 (`scripts/check-agent-record.py` bumped).
  Context/CP is documented here + roadmap sub-row but takes NO new backend row
  (it rides the same abstraction; the explicit row set is TP/PP/DP/EP/SP).
- **Engine-matrix `PAR-*` rows** (`PAR-TP`/`PAR-PP`/`PAR-EP-EPLB`/`PAR-DP`/`PAR-SEQUENCE-MOE`/`PAR-MULTINODE`,
  `feature-matrix.md:111-116`) are the pre-existing inventory; this spike does NOT
  add engine rows (`ENGINE_ROWS` unchanged) — the new rows are on the BACKEND
  (transport/execution) axis.
- **Roadmap** ([roadmap_v1.md](../roadmap_v1.md)) gains a mode-priority sub-table
  under the scale-out subsection.

## Cross-references

- **W2 (same-host multi-GPU):** [tensor-parallelism.md](tensor-parallelism.md)
  (the deep TP scope) + [scale-out-distributed.md](scale-out-distributed.md)
  Leg 1.
- **Multi-Spark:** [scale-out-distributed.md](scale-out-distributed.md) Leg 2
  (`BACKEND-DISTRIBUTED-MULTINODE-SPARK`) — TP×PP/EP over ConnectX-7 RoCE;
  DeepSeek-V4 fp8 across 2×119 GiB.
- **MLX:** [scale-out-distributed.md](scale-out-distributed.md) Leg 3
  (`BACKEND-DISTRIBUTED-MLX-RING`) — mlx-lm `--num-shards` is TP (heads+FFN) for
  dense, EP for MoE, so modes 1 & 4 map directly onto the Metal path.
- **Fusion surpass-track (SP):** [glue-fusion-2026-07-19.md](glue-fusion-2026-07-19.md).

## Gates / dependencies

No build, no GPU, no measurement — records-only enumeration
(`benchmark_binding=false`). Every mode's correctness gate is HW-blocked exactly
as in the scale-out spike (no ≥2-GPU box; GB10 and the cluster nodes are
single-GPU each). The W-plan (W2 TP-forward, W3 NCCL, W4 PP+executor, then
EP/DP/SP/CP) is in [scale-out-distributed.md](scale-out-distributed.md).

## Risks / honest flags

- **SP is not a parallel axis.** Any "sequence parallel" row is a TP-mode
  compilation-pass optimization (reduce-scatter fusion), not a world dimension —
  recorded above and in the `BACKEND-DISTRIBUTED-SP` row.
- **EP is a re-grouping, not a new world_size.** The EP group is DP×PCP×TP ranks;
  `enable_expert_parallel` flips the MoE shard strategy, it does not add a size
  knob.
- **No interleaved/virtual pipeline** in vLLM v1 — PP is a contiguous stage split
  (confirmed absent in the pinned tree). Do not scope a 1F1B scheduler we would
  not be mirroring.
- **Numerics:** all-reduce / reduce-scatter ordering perturbs bf16 near-ties; the
  sharded ≡ unsharded gate uses the near-tie distributional methodology where the
  model is non-deterministic, strict where deterministic (per the ratified
  near-tie distributional gate; see [tensor-parallelism.md](tensor-parallelism.md)
  §6 GATE-1).
