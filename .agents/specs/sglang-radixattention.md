# Spike: SGLang behavior parity — RadixAttention and the SGLang-alike runtime, fuse-or-flag

**Owning rows:** `KV-SGLANG-RADIX-CACHE` (RadixAttention scope vs our APC) and
`ENG-SGLANG-BEHAVIOR-FLAG` (the cache-aware-scheduling / overlap / jump-forward
survey + the enable/disable control). Both `SPIKE`.
**State:** scoping spike, read-only on production code. The deliverable is this
spec + matrix rows + a concrete fuse-or-flag recommendation per technique and a
flag design. No engine code is written here.

**User directive (2026-07-27):** "scope SGLang, look at how it implements
RadixAttention etc., fold into specs, and have at least a flag that
enables/disables RadixAttention and/or SGLang-alike behavior, if it can't be
fused somehow."

## Reference pins

| Reference | Pin | Local ground |
|---|---|---|
| SGLang source | tag `v0.5.15`, commit `f63458b5beaceabbd9d749b9fc956370e1b649e6` | cloned to `/home/mudler/_git/sglang` at that commit (the SGLang analogue of `/home/mudler/_git/vllm`); every SGLang `file:line` in this spec is from that tree |
| SGLang (older harness pin) | tag `v0.5.13`, commit `28b095c` | referenced by `BACKEND-BENCH-CUDA-SGLANG-PREFLIGHT`; not re-read here |
| vLLM oracle | `555967922`/0.26.0.dev0 | our MIRROR-vLLM reference; SGLang is a DIFFERENT engine, so the obligation here is BEHAVIOR PARITY + a toggle, not a blind 1:1 port |

**Policy note.** MIRROR-vLLM is the prime directive; vLLM has already made the
prefix-sharing design choice for us (block-hash APC). SGLang is a competitor
engine, not the mirror source. So this spike measures where SGLang's runtime
does something our vLLM-derived design cannot already express, and proposes a
toggle only there. It does NOT copy SGLang's Python data structures as a second,
incompatible cache abstraction — the sibling benchmark spike already recorded
that rule ([cuda-sglang-low-concurrency.md](cuda-sglang-low-concurrency.md#L594):
"do not copy the Python data structure as a new incompatible cache abstraction").

This spec is the IMPLEMENTATION-BEHAVIOR scope. The `BACKEND-GATE-CUDA-SGLANG*`
rows in [backend-matrix.md](../backend-matrix.md) are the sibling BENCHMARK
track (are-we-as-fast-as-SGLang); they are cross-referenced, not modified.

---

## 1. RadixAttention (the headline)

### How SGLang implements it

SGLang's automatic prefix sharing is a **radix TREE of KV prefixes**, not a
block-hash chain.

- **Data structure** — `RadixCache` (`python/sglang/srt/mem_cache/radix_cache.py:280`)
  is a trie of `TreeNode` (`:217`). Each node holds a `RadixKey` slice
  (`:60`, a run of token ids) and a `value` tensor of KV-cache slot indices for
  those tokens. Children are keyed by `child_key` = the first `page_size`
  logical units (`:198`). The tree namespaces branches by an `extra_key`
  (lora_id / cache_salt, `:75`) so different tenants never share a subtree.
- **Longest-prefix match** — `match_prefix` (`:355`) → `_match_prefix_helper`
  (`:648`) walks children, calling `RadixKey.match` (`:162`, a galloping
  exponential search) at each node and **splitting a node** (`_split_node`,
  `:674`) when the incoming request diverges mid-node. Match length is rounded
  DOWN to `page_size` (`page_aligned`, `:136`; `match(..., page_size)`, `:191`).
  With `page_size == 1` the tree shares a prefix down to a **single token**.
- **Insertion** — `cache_unfinished_req` (`:488`) and `cache_finished_req`
  (`:437`) write a running/finished request's tokens back into the tree via
  `_insert_helper` (`:704`), creating/splitting nodes as needed.
- **Eviction** — `evict` (`:563`) builds a min-heap over the **evictable leaves**
  keyed by `eviction_strategy.get_priority(node)`; the default strategy is LRU
  by `TreeNode.last_access_time` (`TreeNode.__lt__`, `:276`; `radix_eviction_policy`
  default `"lru"`, `server_args.py:739`, also `lfu`/`slru`/`priority`). Frees the
  least-valuable leaf, then re-heaps its parent if it became a childless leaf
  (`:583`). Nodes referenced by an in-flight request are protected by
  `lock_ref` (`inc_lock_ref`/`dec_lock_ref`, `:592`/`:607`); locked nodes are not
  in the evictable set (`evictable_size_`/`protected_size_` accounting).
- **Toggle** — `--disable-radix-cache` (default `False` = radix ON,
  `server_args.py:755`). When disabled (with chunked prefill), the factory
  (`mem_cache/registry.py:83-94`) falls back to `ChunkCache`, which retains KV
  only within a single request — i.e. NO cross-request sharing.
- **The attention kernel** — "RadixAttention" is the marketing name for the
  paged/flashinfer attention that reads the shared physical KV blocks the tree
  points at. The kernel itself is ordinary paged attention over a block table;
  the novelty is entirely in the tree that decides WHICH blocks are shared.

### What OUR engine already does (APC)

Our Automatic Prefix Caching is vLLM's **block-hash chain**, ported and shipped:

- **Data structure** — per-block SHA-256/CBOR chain hash over `block_size`
  (default 32) token windows, a `BlockPool`, and unitary / hybrid / no-prefix
  coordinators (`src/vllm/v1/core/kv_cache_utils.cpp:259,291`;
  `src/vllm/v1/core/kv_cache_coordinator.cpp:260`;
  `src/vllm/v1/core/kv_cache_manager.cpp:124`). A cached block is keyed by
  `(prefix_hash, block_token_ids, extra_keys)`; a repeated prefix hits the same
  pooled block.
- **Longest-prefix match** — `find_longest_cache_hit` walks the hash chain
  block-by-block (`FullAttentionManager`, `single_type_kv_cache_manager.cpp`);
  the match is BLOCK-granular (a multiple of `block_size`).
- **Extra keys / no-false-share** — `generate_block_hash_extra_keys` +
  `_gen_mm_extra_hash_keys` port (`kv_cache_utils.cpp`), with `cache_salt` +
  `lora_name` on the `Request` (`include/vllm/v1/request.h`), RED-proven to
  prevent cross-tenant false hits (`KV-PREFIX-CACHE` W2, `CLAIM-ROADMAP-D4APC`).
- **Eviction** — LRU over the free block pool (vLLM's block-level LRU), plus the
  `lock`/ref-count equivalent that keeps in-use blocks resident.
- **Toggle** — `--[no-]enable-prefix-caching` (`examples/server/main.cpp:185`);
  `EngineParams::enable_prefix_caching` tri-state
  (`include/vllm/entrypoints/model_loader.h:76`), resolved against the model
  default (dense = ON, hybrid/attention-free = OFF), mirroring vLLM.
- **Proof** — `KV-PREFIX-CACHE` is `DONE` for the dense APC path: the first-ever
  cache-ON model gate `tests/parity/test_qwen3_apc_e2e.cpp` on Qwen3-4B hit
  2240/2777 (0.807), TTFT 70.1→39.9 ms = 1.76×, token-exact vs APC-OFF and vs
  vLLM-APC-ON teacher-forced (`444ea9d7`).

### Where they DIFFER

| Axis | SGLang RadixAttention | Our APC (vLLM) | Material? |
|---|---|---|---|
| Sharing granularity | token-granular when `page_size==1`; else `page_size` units | block-granular (`block_size`, default 32) | **The only real difference.** A prefix that diverges mid-block is shared by SGLang (page 1) but not by us (rounds down to the block). In practice both engines run paged backends at page/block ≥ 16, and the benchmark protocol pins matched page/block sizes, so the divergence window is one block at most. |
| Overlap / partial | node split on mid-node divergence | partial-block primitives exist but throw (upstream's are DEAD CODE, no caller — `KV-PREFIX-CACHE`) | equivalent at block/page boundaries |
| Eviction unit | tree leaf (a run of tokens) | free block | equivalent LRU semantics; SGLang additionally offers lfu/slru/priority strategies (opt-in) |
| In-use protection | `lock_ref` per node | block ref-count | equivalent |
| Tenant isolation | `extra_key` on RadixKey | `extra_keys` in block hash (mm/lora/salt) | equivalent, both RED-proven-style |
| Cross-request sharing | default ON | default ON (dense) | equivalent |
| Off-path fallback | `ChunkCache` (intra-request only) | cache-off no-prefix coordinator | equivalent |

### VERDICT — RadixAttention is ALREADY FUSED into our APC

Our block-hash APC and SGLang's radix tree are **functionally equivalent
automatic prefix sharing**: both do longest-prefix match against previously
computed KV, share the physical blocks, LRU-evict, protect in-use entries, and
isolate tenants by an extra key. The ONLY behavioral delta is sharing
granularity (token/page vs block), and that delta:

1. is bounded to a single `block_size` window at the point of divergence;
2. is already parametrized by our `block_size` knob (set block/page equal on
   both engines and the granularity matches);
3. does not change token outputs (prefix caching is a compute-reuse
   optimization; the gated proof shows APC-ON == APC-OFF token-exact).

Building a second, token-granular trie cache would be a **redundant incompatible
abstraction** for a bounded, output-neutral granularity difference — explicitly
against the recorded rule and against MIRROR-vLLM (vLLM chose block-hash; we
mirror vLLM, not SGLang). Therefore:

> **`--enable-radix-attention` is an ALIAS for our APC toggle, not a distinct
> code path.** A genuinely-distinct token-granular radix path is NOT justified.

The flag switches exactly what `--enable-prefix-caching` switches. See §6.

---

## 2. Cache-aware scheduling

### How SGLang implements it

SGLang can REORDER the waiting queue to maximize radix-cache hits.
`SchedulePolicy` (`python/sglang/srt/managers/schedule_policy.py:155`) supports
cache-aware policies (`CacheAwarePolicy`, `:139`): **LPM** (longest prefix
match, `:142`) and **DFS-weight** (`:143`), plus in-batch prefix caching
(`_compute_prefix_matches`, `:253`, which computes each waiting request's matched
prefix length against the tree via `match_prefix_for_req`, `:91`, and
de-prioritizes requests that would collide on the same not-yet-cached prefix,
`IN_BATCH_PREFIX_CACHING_*_THRESHOLD`, `:76`/`:83`). `calc_priority` (`:176`)
sorts by longest matched prefix (`_sort_by_longest_prefix`, `:205`). The
expensive prefix sort is auto-disabled when the queue is large
(`_determine_active_policy`, `:229`, > 128 waiting).

**SGLang's own default is `fcfs`** (`schedule_policy` default `"fcfs"`,
`server_args.py:692`); LPM is opt-in.

### What OUR engine already does

Our scheduler (`src/vllm/v1/core/sched/scheduler.cpp`) admits from a `waiting`
queue that is FCFS (`request_queue.cpp:30`) or priority
(`SchedulingPolicy::kPriority`, `include/vllm/config/scheduler.h:56`;
`create_request_queue`, `request_queue.cpp:181`). It has NO cache-hit-priority
reordering — admission order is arrival/priority, independent of prefix-match
length. It DOES compute `num_common_prefix_blocks` among the RUNNING batch for
cascade attention (`scheduler.cpp:418`), but that is a kernel optimization, not
an admission-ordering policy.

### VERDICT — FLAG (a new scheduling-policy option)

This is a genuinely-distinct behavior our design cannot currently express: no
`lpm` waiting-queue ordering, no in-batch prefix-collision de-prioritization.
BUT SGLang's own default is `fcfs`, so mirroring SGLang's DEFAULT is already
covered — LPM is the opt-in delta. Flag-worthy (`--schedule-policy=lpm`),
scoped as a scheduling-policy option under `ENG-SGLANG-BEHAVIOR-FLAG`. See the
W-breakdown (§7). Lower priority than the flag alias since it is an opt-in
throughput lever, not a default.

---

## 3. Jump-forward decoding (constrained/grammar)

### How SGLang implements it

Compressed-FSM jump-forward: when the grammar FSM forces a deterministic run of
tokens, SGLang emits them WITHOUT model steps.
`OutlinesJumpForwardMap` (`python/sglang/srt/constrained/outlines_jump_forward.py:182`)
precomputes, per FSM state, the forced string / byte run
(`jump_forward_symbol`, `:146`; `jump_forward_byte`, `:159`). The scheduler
splices the forced tokens in and retokenizes across the boundary. Reference:
the lmsys "compressed FSM" blog (file header `:16`). Tied to the outlines
backend; xgrammar has its own token-level path.

### What OUR engine already does

We have structured output / grammar via a bitmask FSM: `StructuredOutputManager`
(`src/vllm/v1/structured_output/manager.cpp`) fills a per-request token bitmask
each step (`fill_bitmask_row`, `:50`), the backend `accept_tokens` /`rollback`
advance/undo the FSM (`include/vllm/v1/structured_output/backend_types.h:87,97`),
and `json_schema_to_gbnf` compiles schemas. This is vLLM's xgrammar-style
per-STEP masking — one model step per token, NO jump-forward token elision.

### VERDICT — FLAG (opt-in, distinct, low priority)

Jump-forward is a genuinely-distinct SGLang optimization our grammar path cannot
express (it skips model steps for FSM-forced runs). It is an output-neutral
speed optimization on constrained decoding. It is NOT a default in the mirror
engine (vLLM has no jump-forward either), so it is an OPT-IN flag
(`--enable-jump-forward`) if and when constrained-decoding throughput is
prioritized. It requires FSM-run precomputation + cross-boundary retokenization
— a substantial leaf, deferred behind the flag; not scheduled in this spike's
W-breakdown beyond naming it. Tracked as a survey item under
`ENG-SGLANG-BEHAVIOR-FLAG`.

---

## 4. Zero-overhead / overlap scheduler

### How SGLang implements it

`event_loop_overlap` (`python/sglang/srt/managers/scheduler.py:1563`) overlaps
CPU processing of batch N-1's results with GPU computation of batch N via a
`result_queue` deque + a `TpModelWorkerClient` future map
(`enable_overlap = not disable_overlap_schedule`, `:344`; default ON,
`server_args.py:776`). This is SGLang's "zero-overhead scheduler".

### What OUR engine already does

`ENG-ASYNC-SCHED` is `DONE` and default-ON (`6ea7856`): `AsyncScheduler` +
depth-2 `step_with_batch_queue` + async D2H on a copy stream +
`AsyncGPUModelRunnerOutput`, mirroring vLLM's async scheduler
(`src/vllm/v1/core/sched/async_scheduler.cpp`; `src/vllm/v1/engine/core.cpp`
`step_with_batch_queue`; `include/vllm/v1/worker/gpu/async_output.h`). It
overlaps the SAME CPU/GPU boundary SGLang's overlap loop targets, is
token-neutral, and carries a measured TPOT win. Rollback env
`VT_ASYNC_SCHED=0` / `VT_ASYNC_RUNNER=0`.

### VERDICT — ALREADY FUSED (covered by `ENG-ASYNC-SCHED`)

Our async/overlap scheduler is the fused equivalent of SGLang's zero-overhead
scheduler. Default-ON in both. No new flag; the existing rollback env vars are
the toggle. No-op alias only if an SGLang-compatible name is desired.

---

## 5. Survey of the rest (one line each)

| SGLang technique | SGLang ref | Our status | Disposition |
|---|---|---|---|
| Chunked prefill | `server_args.py:655` `chunked_prefill_size` (default on) | `ENG-CHUNKED-PREFILL` `ANCHOR-BACKFILL`, `scheduler.cpp:225,548` | **already covered** |
| Continuous batching | scheduler running-first | `ENG-SCHED-CORE` `ANCHOR-BACKFILL` | **already covered** |
| Speculative — EAGLE | `python/sglang/srt/speculative/` (eagle) | our `SPEC-MTP` DONE, `SPEC-DFLASH` DONE, ngram | **fuse** — same family (draft/verify); EAGLE-specific draft = future model port, not a new mechanism |
| Speculative — DFlash | `speculative/dflash_worker_v2.py`; `validate_dflash_request` (`scheduler.py:2163`) | `SPEC-DFLASH` DONE (ours) | **already covered** (both engines have DFlash) |
| Speculative — ngram | `speculative/cpp_ngram` | our ngram path | **already covered** |
| Radix eviction strategies (lfu/slru/priority) | `server_args.py:739`; `evict_policy.py` | our APC LRU only | **flag** (opt-in eviction-policy knob) — minor; folds under the APC/behavior flag if ever needed |
| Hierarchical / host-offload cache (HiCache) | `enable_hierarchical_cache` (`server_args.py:1991`); `hiradix_cache.py` | our `KV-OFFLOAD` / LMCache track | **already covered** by the KV-offload track (separate rows) |
| Mamba radix cache (hybrid SSM prefix) | `mamba_radix_cache.py`; `registry.py:129` | `KV-MAMBA-ALIGN` leaf (open) | **already scoped** under `KV-MAMBA-ALIGN` / `BACKEND-GATE-CUDA-SGLANG-PREFIX` |
| Custom attention/MoE kernels (flashinfer, sgl-kernel) | sgl-kernel | our vendored FA2 / Marlin / cutlass | **out of scope here** — kernel parity is the benchmark track, not a behavior toggle |
| Session radix cache | `session_radix_cache.py`; `enable_session_radix_cache` | none | **out of scope** (serving-session feature, not requested) |
| Priority scheduling | `enable_priority_scheduling` (`server_args.py:693`) | our `kPriority` policy (`GATING`) | **already covered** |

---

## 6. Flag design (the user's concrete ask)

The enable/disable control has THREE independent switches. Only the second is a
genuinely-new code path; the first and third are aliases/knobs over behavior we
already ship.

### 6.1 `--enable-radix-attention` — an ALIAS for APC (no new path)

- **What it switches:** exactly our existing APC on/off. `--enable-radix-attention`
  ≡ `--enable-prefix-caching`; `--disable-radix-attention` ≡
  `--no-enable-prefix-caching` (mirrors SGLang's `--disable-radix-cache`).
- **Surface:**
  - CLI: add the alias next to `--[no-]enable-prefix-caching`
    (`examples/server/main.cpp:185`). Pure argument aliasing; sets the same
    `EngineParams::enable_prefix_caching` tri-state.
  - C-ABI: `vllm_model_params` (`include/vllm.h:100`) currently has NO
    prefix-caching field (it exposes `block_size`/`num_blocks` only). Add a
    tri-state `int32_t enable_prefix_caching` (0=default/model, 1=on, 2=off) —
    additive, ABI-versioned — so the toggle is reachable from the C-ABI. The
    radix alias is documentation on that one field.
  - OpenAI server: no per-request field (prefix caching is engine-global in vLLM
    and SGLang alike); the server flag sets it at load.
- **Default:** unchanged — mirror our current default (APC ON for dense
  full-attention; OFF for hybrid/attention-free), which already mirrors vLLM.
- **Composition:** it IS the APC toggle. There is no second cache to compose
  with. Setting both `--enable-radix-attention` and `--no-enable-prefix-caching`
  is a contradiction → last-wins with a warning (same as passing the vLLM flag
  twice). Because it is an alias, it is a NO-OP wrapper — we do NOT invent a
  redundant token-granular path (§1 verdict).

### 6.2 `--schedule-policy={fcfs,priority,lpm}` — the genuinely-new path

- **What it switches:** the waiting-queue admission ordering. `fcfs` (default,
  today's behavior) and `priority` (our `kPriority`) exist; `lpm` is NEW —
  reorder the waiting queue by longest matched prefix against the APC block-hash
  index (the block-hash analogue of SGLang's tree match), with the large-queue
  auto-fallback to fcfs (mirror `schedule_policy.py:229`).
- **Surface:** CLI `--schedule-policy` (extend `SchedulerPolicy`,
  `include/vllm/config/scheduler.h:56`, and `create_request_queue`,
  `request_queue.cpp:181`); config field; NOT on the OpenAI per-request API.
- **Default:** `fcfs` — mirrors BOTH vLLM's and SGLang's own default. `lpm` is
  opt-in.
- **Composition:** independent of the APC toggle in principle, but `lpm` is
  meaningless with APC off (no cache to match against) → resolve `lpm` +
  cache-off to `fcfs` with a warning.

### 6.3 `--enable-jump-forward` — opt-in constrained-decode optimization (deferred)

- **What it switches:** grammar FSM-forced token runs are emitted without model
  steps (§3). Distinct, output-neutral, opt-in.
- **Surface:** CLI + config; interacts only with structured-output requests.
- **Default:** OFF (neither mirror engine defaults it on). Deferred — named, not
  scheduled here.

### 6.4 Umbrella `--sglang-compat` (optional convenience)

A single convenience flag that sets the SGLang-alike bundle: radix alias ON +
`--schedule-policy=lpm` + overlap scheduler ON (already default). Pure
composition of the switches above; adds no new behavior. Optional; only if a
user asks for "SGLang mode" in one flag.

### Summary — what each switch actually does

| Flag | New code path? | Maps to | Default | Notes |
|---|---|---|---|---|
| `--enable-radix-attention` | **No (alias)** | APC on/off (`enable_prefix_caching`) | model default (dense ON) | RadixAttention == our APC (§1) |
| `--schedule-policy=lpm` | **Yes** | new waiting-queue ordering | `fcfs` | the one genuinely-distinct behavior; W-breakdown §7 |
| `--enable-jump-forward` | Yes (deferred) | grammar FSM token elision | off | named, not scheduled |
| overlap scheduler | No (shipped) | `ENG-ASYNC-SCHED` (`VT_ASYNC_SCHED`) | on | already fused |
| `--radix-eviction-policy` | Yes (minor) | APC eviction strategy | `lru` | opt-in knob, lfu/slru/priority; folds under APC |
| `--sglang-compat` | No (composition) | the bundle above | off | convenience only |

---

## 7. W-breakdown (only the flag-worthy, genuinely-distinct work)

`KV-SGLANG-RADIX-CACHE` is **records-mostly**: its verdict is "already fused;
the flag is an alias." Its only implementation work is the alias plumbing +
the C-ABI field, both trivial and gated by existing APC tests.

| Work | Row | Deliverable | State |
|---|---|---|---|
| RW0 | `KV-SGLANG-RADIX-CACHE` | This spike + verdict (RadixAttention == APC, alias not path) | **this spec** |
| RW1 | `KV-SGLANG-RADIX-CACHE` | `--enable-radix-attention` CLI alias + C-ABI `enable_prefix_caching` tri-state field; reuses `test_qwen3_apc_e2e` + server-help contract; no engine change | scoped (trivial) |

| Work | Row | Deliverable | State |
|---|---|---|---|
| SW0 | `ENG-SGLANG-BEHAVIOR-FLAG` | This spike: cache-aware / overlap / jump-forward survey + flag design | **this spec** |
| SW1 | `ENG-SGLANG-BEHAVIOR-FLAG` | `SchedulingPolicy::kLPM` + `LPMRequestQueue` ordering by APC longest-match, large-queue fcfs fallback; port `schedule_policy.py` LPM cases as CPU behavioral tests; token-neutral A/B | scoped |
| SW2 | `ENG-SGLANG-BEHAVIOR-FLAG` | in-batch prefix-collision de-prioritization (`IN_BATCH_PREFIX_CACHING_*`) | scoped (after SW1) |
| SW3 | `ENG-SGLANG-BEHAVIOR-FLAG` | jump-forward decoding behind `--enable-jump-forward` (FSM-run precompute + retokenize) | **deferred** (named) |
| SW4 | `ENG-SGLANG-BEHAVIOR-FLAG` | `--radix-eviction-policy` lfu/slru/priority knob over the block pool | **deferred** (minor) |

Gates for any flag-worthy work: CPU `-Werror` + full CTest; token-neutral A/B
(an ordering/eviction change must not change greedy outputs — it is a
throughput/latency lever); the LPM lever additionally needs a GB10 cache-ON
throughput A/B against the SGLang floor (owned by `BACKEND-GATE-CUDA-SGLANG-PREFIX`,
not this row).

---

## 8. Honest already-covered / out-of-scope roll-up

- **Already fused (no flag owed):** RadixAttention automatic sharing (== APC),
  overlap scheduler (== `ENG-ASYNC-SCHED`), chunked prefill, continuous
  batching, DFlash/ngram speculation, priority scheduling, HiCache/host-offload
  (== `KV-OFFLOAD`), Mamba radix (== `KV-MAMBA-ALIGN`).
- **Flag-worthy (genuinely distinct):** cache-aware LPM scheduling (SW1/SW2);
  radix eviction strategies (SW4, minor); jump-forward decoding (SW3, deferred).
- **Out of scope for a behavior toggle:** SGLang custom kernels (benchmark
  track), session radix cache, PD-disaggregation radix, multi-item scoring.

## 9. Dependencies

- Rows: `KV-PREFIX-CACHE` (the APC the radix alias maps onto, `DONE` dense);
  `ENG-ASYNC-SCHED` (the overlap equivalent, `DONE`); `KV-MAMBA-ALIGN` and
  `BACKEND-GATE-CUDA-SGLANG-PREFIX` (the hybrid-prefix + benchmark siblings).
- No new library or data dependency. The alias + LPM ordering are CPU-testable;
  only the cache-ON throughput A/B needs GB10.

## 10. Risks and decisions

- **Do not build a second cache.** The token-granular tree is a bounded,
  output-neutral granularity difference over our block-hash APC; a distinct path
  would be a redundant incompatible abstraction against MIRROR-vLLM. Decision:
  alias only.
- **`lpm` without APC is meaningless** → resolve to `fcfs` + warn.
- **Naming.** The SGLang-compatible names are ergonomics; the engine behavior is
  vLLM's. Every alias documents the vLLM concept it maps to so the mirror stays
  legible.
