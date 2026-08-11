# KV-EVENTS — the KV-cache event stream (ROAD-V1-D4)

Row: [`KV-EVENTS`](../engine-matrix.md) · Claim: `CLAIM-ROADMAP-D4-KV-EVENTS` ·
State: `ACTIVE` (event generation + payload DONE 2026-07-27; engine/scheduler
envelope + `report_mode` DONE 2026-08-11; live ZMQ transport deferred).
Issues: [#352](https://github.com/mudler/vllm.cpp/issues/352) (W3, the batch
envelope + `kv_cache_report_mode`);
[#353](https://github.com/mudler/vllm.cpp/issues/353) (`KVEventsConfig` has no
`PostInit` — found during W3, filed rather than silently fixed).

## Scope

vLLM publishes a stream of KV-cache events (`BlockStored` / `BlockRemoved` /
`AllBlocksCleared`) so external consumers — prefix-cache-aware routers /
KV-cache load balancers — can track which prefix blocks each engine holds. Off
by default (`--kv-events-config enable_kv_cache_events=false`). This row ports,
additively, the event DATA TYPES, the EMISSION at the BlockPool store/remove/
clear sites, the publisher SEAM, and the `msgpack` PAYLOAD. The live ZMQ socket
transport is explicitly out of scope for W1 (deferred, see Risks/decisions).

**W3 (this pass, issue #352)** closes the CPU-reachable half of the residual: the
ENGINE/SCHEDULER wiring of the `KVEventBatch` envelope (the publisher built from
`--kv-events-config`, the per-step `take_events()` drain + `publish`, and
`shutdown`), and per-request `kv_cache_report_mode` including the
`report_mode == "full"` REUSE path that fires `emit_cached_block_events` on a
prefix-cache HIT. Both are gated through the already-landed
`CollectingEventPublisher`. The live ZMQ transport stays out of scope: it needs a
third-party socket dependency that does not exist under `third_party/`, which is
a dependency decision rather than an implementation one, so
`EventPublisherFactory` keeps throwing loudly on `"zmq"`.

## Upstream chain

Pin `555967922`, vLLM 0.26.0.dev0.

- **Config** — `vllm/config/kv_events.py:11-52` `KVEventsConfig`
  (`enable_kv_cache_events`, `publisher` ∈ {null, zmq}, `endpoint`,
  `replay_endpoint`, `buffer_steps`, `hwm`, `max_queue_size`, `topic`;
  `__post_init__` resolves publisher to "zmq" when enabled else "null").
- **Event types** — `vllm/distributed/kv_events.py`: `EventBatch` (`:25-33`,
  `array_like=True, omit_defaults=True`), `KVCacheEvent` (`:36-42`, `tag=True`)
  base, `BlockStored` (`:51-95`), `BlockRemoved` (`:98-113`), `AllBlocksCleared`
  (`:116-117`), `KVEventBatch` (`:120-121`).
- **Publisher seam** — `EventPublisher` (`:246-274`), `NullEventPublisher`
  (`:277-284`), `ZmqEventPublisher` (`:287-508`), `EventPublisherFactory`
  (`:511-543`). Multi-worker DP: `KVEventAggregator` (`:124-208`),
  `KVConnectorKVEvents` (`:211-243`).
- **Hash truncation** — `vllm/v1/core/kv_cache_utils.py:51-54` `ExternalBlockHash`
  + `:79-82` `maybe_convert_block_hash` (env `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES`,
  upstream default **True**, `envs.py:1816-1817` `bool(int(getenv(...,"1")))`).
- **Emission sites** — `vllm/v1/core/block_pool.py`: store `cache_full_blocks`
  (`:268,298-342`) via `_build_block_stored_event` (`:344-371`); reuse
  `emit_cached_block_events` (`:373-443`), fired only under
  `kv_cache_report_mode == "full"` (`kv_cache_manager.py:265-280`); remove
  `_emit_block_removed_events` (`:592-605`) from `_maybe_evict_cached_block`
  (`:679-700`) + the partial→full promotion (`:291-292`); clear
  `reset_prefix_cache` (`:794-795`); drain `take_events` (`:820-830`).
- **Scheduler wiring** (re-read at the pin for W3; the W1 line numbers above were
  carried from an older revision) — `vllm/v1/core/sched/scheduler.py:86`
  (`self.kv_events_config = vllm_config.kv_events_config`), `:116-119`
  (`self.enable_kv_cache_events = kv_events_config is not None and
  kv_events_config.enable_kv_cache_events`), `:155-158`
  (`EventPublisherFactory.create(self.kv_events_config,
  self.parallel_config.data_parallel_index)`), `:1901-1915` (drain
  `self.kv_cache_manager.take_events()`, then `if events:` build
  `KVEventBatch(ts=time.time(), events=events)` and
  `self.kv_event_publisher.publish(batch)` — inside `update_from_output`, just
  before the `EngineCoreOutputs` are assembled), and `:2456-2459`
  (`shutdown()` -> `self.kv_event_publisher.shutdown()`).
- **`report_mode` source** — `vllm/v1/request.py:118-127`: `Request.__init__`
  reads `sampling_params.extra_args.get("kv_cache_report_mode",
  "incremental")`, and falls back to `"incremental"` when `extra_args is None`.
- **`report_mode == "full"` reuse path** —
  `vllm/v1/core/kv_cache_manager.py:262-280`: inside `get_computed_blocks`,
  after `find_longest_cache_hit`, when `num_new_computed_tokens > 0 and
  self.enable_kv_cache_events and getattr(request, "kv_cache_report_mode",
  "incremental") == "full"`, iterate `computed_blocks` per group and call
  `self.block_pool.emit_cached_block_events(request, num_blocks, block_size,
  group_idx)` with `block_size` read from
  `self.kv_cache_config.kv_cache_groups[group_idx].kv_cache_spec.block_size`.
- **Payload wire** — `msgspec.msgpack.Encoder().encode(batch)`
  (`kv_events.py:445`).

## Our baseline

The SPIKE (2026-07-22) left scaffolding in place: `include/vllm/v1/core/block_pool.h`
carried an empty placeholder `struct KVCacheEvent {}`, `take_events()` returned an
always-empty queue, and the three emission points inside `cache_full_blocks` /
`_maybe_evict_cached_block` / `reset_prefix_cache` were marked-out no-ops. The
block hashes, `generate_block_hash_extra_keys` (text path returns `None`),
`get_block_hash`/`get_group_id`, and `AllTokenIds()`/`lora_name` on `Request`
already existed. So this was a bounded fill-in, not a fresh port.

**W3 baseline (re-verified against the tree at `7020de93`).** W1 landed the whole
producer side and left the consumer side empty:

- `Scheduler` hard-codes `/*enable_kv_cache_events=*/false` into its
  `KVCacheManager` (`src/vllm/v1/core/sched/scheduler.cpp:159`), owns no
  publisher, never calls `kv_cache_manager->take_events()`, and has no
  `shutdown()` at all. So no `KVEventBatch` is ever constructed anywhere outside
  a unit test, and no configuration reaches the pool: events were unreachable
  even with the pool flag flipped by hand.
- `KVCacheManager` accepts `enable_kv_cache_events` but only forwards it to the
  coordinator; it does not retain it, so it cannot gate a manager-level branch.
- `Request` has no `kv_cache_report_mode`, and `SamplingParams` has no
  `extra_args` (recorded DEFERRED at `include/vllm/sampling_params.h:18`), so the
  already-ported, already-tested `BlockPool::emit_cached_block_events` had no
  caller — `include/vllm/v1/core/block_pool.h:138-143` says so in as many words
  ("our Request has no report_mode field, so the MANAGER wiring is deferred").

## Port map

New: `include/vllm/distributed/kv_events.h` + `src/vllm/distributed/kv_events.cpp`.
- Event data types 1:1 (`BlockStored`, `BlockRemoved`, `AllBlocksCleared`,
  `KVEventBatch`); `KVCacheEvent` = a `std::variant` tagged union re-exported into
  `vllm::v1` as an alias replacing the placeholder.
- `ExternalBlockHash` (= `std::variant<std::string, uint64_t>`) +
  `maybe_convert_block_hash` added to `include/vllm/v1/core/kv_cache_utils.h` +
  `src/vllm/v1/core/kv_cache_utils.cpp` (env-parsed, default int-truncated).
- `encode_kv_event_batch` — byte-for-byte identical to
  `msgspec.msgpack.Encoder()` on the 1:1 upstream structs. The msgspec rules were
  reverse-engineered from a local `msgspec` 0.21.1 capture (`array_like` outer
  array KEEPS a trailing `data_parallel_rank` even when `None`; `tag=True` maps
  put `"type"` first; `omit_defaults` drops only fields equal to the (`None`)
  default; minimal-width ints; `bin` for byte hashes).
- Publisher seam: `EventPublisher` abstract, `NullEventPublisher` (default),
  `CollectingEventPublisher` (in-memory, for gating), `EventPublisherFactory` +
  `KVEventsConfig`.
- Emission wired in `src/vllm/v1/core/block_pool.cpp` at the store
  (`cache_full_blocks` + `_build_block_stored_event`), remove
  (`_emit_block_removed_events` from `_maybe_evict_cached_block`), clear
  (`reset_prefix_cache`), and reuse (`emit_cached_block_events`) sites, all
  guarded by `enable_kv_cache_events` (default OFF ⇒ default path byte-identical).

**W3 additions.**

| Upstream | Ours |
|---|---|
| `scheduler.py:86,116-119,155-158` | `Scheduler` ctor gains two trailing defaulted params, `const distributed::KVEventsConfig* kv_events_config = nullptr` and `int data_parallel_rank = 0`; members `enable_kv_cache_events_`, `kv_event_publisher_` |
| `scheduler.py:1901-1915` | `Scheduler::update_from_output` tail: drain `kv_cache_manager->take_events()`, and `if (!events.empty())` build `KVEventBatch{ts, events}` and `publish` |
| `scheduler.py:2456-2459` | `Scheduler::shutdown()` (virtual), forwards to `kv_event_publisher_->shutdown()` |
| `request.py:116-127` | `SamplingParams::extra_args`; `Request::kv_cache_report_mode` set in the ctor with the same `.get(key, "incremental")` / `None -> "incremental"` semantics |
| `kv_cache_manager.py:262-280` | `KVCacheManager` retains `enable_kv_cache_events`; `get_computed_blocks` gains the per-group `emit_cached_block_events` loop |

## Design

**Where the batch is built.** The envelope is assembled in `update_from_output`,
not in `schedule()`, because that is where upstream drains it — eviction events
raised by `allocate_slots` during a schedule and store events raised by
`cache_blocks` at the end of a step must land in the SAME batch, and
`update_from_output` is the last thing that runs in a step. Placing the drain
before the `EngineCoreOutputs` assembly mirrors `scheduler.py:1901-1915`
positionally.

**`ts` is wall clock.** Upstream uses `time.time()`, so ours is
`std::chrono::system_clock` seconds-since-epoch as a `double`, not
`steady_clock` — a consumer correlates the batch against its own wall clock.

**How a `CollectingEventPublisher` gets in without inventing a config value.**
`EventPublisherFactory::create` mirrors upstream exactly and knows only `"null"`
and (throwing) `"zmq"`; teaching it a `"collecting"` kind would invent a
`--kv-events-config publisher=` value vLLM does not have. Instead the Scheduler
gets `set_kv_event_publisher(std::unique_ptr<EventPublisher>)`, the same
injection shape the already-landed `set_kv_connector` uses
(`include/vllm/v1/core/sched/scheduler.h:196-199`) for exactly this
"non-upstream-reachable transport, still has to be gateable" case. Production
never calls it, so the config-driven path stays 1:1.

**Inertness.** `kv_events_config == nullptr` (the default, and what every
existing call site passes) resolves `enable_kv_cache_events_` to false, which is
the value already hard-coded today — so the default `KVCacheManager`,
`BlockPool`, and `update_from_output` behaviour is unchanged, and the publisher
is a `NullEventPublisher` whose `publish` is an empty body. `take_events()` is
still called each step on the disabled path (as upstream does), returning the
pool's empty queue; the `if (!events.empty())` guard means nothing else runs.
`kv_cache_report_mode` defaults to `"incremental"` on every request that does not
set `extra_args`, so the manager branch is dead on the default path too.

## Tests to port

Upstream: `tests/v1/core/test_prefix_caching.py:2040,2170,2228,2282,2368,2405`
(the event-emission cases). Re-expressed in `tests/vllm/v1/test_kv_events.cpp`
(6 cases / 62 assertions): byte-exact serialization vs the msgspec goldens (int
and bytes forms), the publisher seam, the store→reuse→evict→reset SEQUENCE, a
partial-store parent-linkage case, and the default-off inertness case. No
upstream test is dropped-without-reason; the ZMQ-transport socket cases are not
ported because the transport is deferred (see Risks/decisions).

**W3.** Upstream has NO test that drives the scheduler envelope or
`kv_cache_report_mode`: `grep -rn "kv_cache_report_mode" --include="*.py"` over
the pin returns exactly three sites, all of them source
(`kv_cache_manager.py:262,268`, `request.py:123-127`), and the event coverage in
`tests/v1/core/test_prefix_caching.py` calls `block_pool` /
`kv_cache_manager.take_events()` directly rather than going through a scheduler
step. So the W3 cases are written from scratch against the upstream SOURCE
anchors above and recorded as such (no upstream test is dropped — there is none
to drop). They live in the same `tests/vllm/v1/test_kv_events.cpp` target:

6. **Envelope through a real scheduler step** — build a `Scheduler` with
   `enable_kv_cache_events=true`, inject a `CollectingEventPublisher`, run
   `schedule()` + `update_from_output()` on a request long enough to fill blocks,
   and assert exactly one batch was published, that its `ts` is a plausible wall
   clock reading, that `data_parallel_rank` was annotated by the publisher, that
   the batch carries the `BlockStored` the pool raised, and that its encoded
   bytes round-trip through `encode_kv_event_batch`.
7. **A step that raises no event publishes NO batch** (the `if events:` guard).
8. **`shutdown()` reaches the publisher.**
9. **`report_mode == "full"` reuse** — prime the cache with one request, then
   admit a second request with the same prompt and
   `extra_args["kv_cache_report_mode"] = "full"`, and assert a `BlockStored`
   whose hashes are the REUSED prefix blocks is published on the hit; the same
   second request with the default `"incremental"` publishes nothing.
10. **Default-off inertness at the SCHEDULER level** — the same steps with the
    default ctor (no `KVEventsConfig`) publish nothing and take no events, and
    the request's `kv_cache_report_mode` is `"incremental"`.

## Gates

CPU, exact/behavioral, RED-first. `test_kv_events` 6/62:

1. Byte-exact `msgpack` payload vs `msgspec` 0.21.1 goldens — bytes-hash form
   (len 279) AND int-hash+parent form (len 170, exercising fixint/uint8/uint16
   widths).
2. Publisher seam — Null no-op; Collecting annotates `data_parallel_rank` and its
   encoded bytes match the encoder; factory returns Null when disabled, throws on
   `"zmq"`.
3. Emission SEQUENCE through a real BlockPool: store → reuse (touch emits
   nothing in default mode) → `emit_cached_block_events` (report_mode=full) →
   evict → reset emits `BlockStored` / (nothing) / `BlockStored` / `BlockRemoved`
   / `AllBlocksCleared` with correct hashes
   (`== maybe_convert_block_hash(request.block_hashes[i])`), token ids, parent,
   `group_idx`, `medium`.
4. Partial store (`num_cached_blocks > 0`) — `parent == hash[0]` + shifted range.
5. Default-off path emits nothing anywhere.

**RED-first:** mis-wiring emission (stored hash → `NONE_HASH` + `AllBlocksCleared`
dropped) fails 4 assertions (3 `block_hashes` + the vanished clear event); revert
→ GREEN. Default-off + APC unchanged: `test_block_pool` 132/132,
`test_prefix_cache_stats` 36/36, `test_kv_cache_manager` 74/74,
`test_kv_cache_coordinator` 106/106, `test_kv_cache_utils` 253/253. Clean
full-library CPU `-Werror`, 0 warnings.

**W3 gate.** CPU, behavioural, RED-first, no GPU. `test_kv_events` (cases 6-10
above) plus the unchanged W1 cases; regression on `test_scheduler`,
`test_llm_engine`, `test_block_pool`, `test_kv_cache_manager`, and the full
`ctest`. RED is captured BEFORE the wiring exists (the new cases must fail
because nothing publishes and `kv_cache_report_mode` does not compile/exist), and
the inertness case must be GREEN both before and after — it asserts the default
path never changed.

## Dependencies

None new at build time (no ZMQ, no msgspec runtime dep — the encoder is
hand-written). Depends on the already-landed prefix-cache block pool
(`KV-BLOCK-POOL`, `KV-PREFIX-CACHE`) and block hashing. Reference goldens were
captured from a local `msgspec` 0.21.1 pip in the scratchpad (build-time
verification only, not a runtime dependency). No dgx / GPU needed.

## Work breakdown

- **W1 (DONE 2026-07-27):** event types + `ExternalBlockHash`/
  `maybe_convert_block_hash` + `msgpack` encoder + publisher seam + BlockPool
  emission at store/remove/clear/reuse, guarded default-off; the CPU exactness
  gate above.
- **W2 (deferred, dependency-blocked):** live ZMQ publisher (sockets, replay
  buffer, thread, DP port offset) behind the seam. Blocked on a third-party
  socket library — there is no zmq under `third_party/` — which is a dependency
  decision, not an implementation one.
- **W3 (DONE 2026-08-11, issue #352, branch `row/KV-EVENTS-W2`):**
  engine/scheduler wiring of the `KVEventBatch` envelope + `publish` per step +
  `shutdown`, and the `report_mode == "full"` reuse call in the manager, with
  `Request::kv_cache_report_mode` sourced from `SamplingParams::extra_args`.
  (The branch is named `-W2` after the row's residual, not this spec's W
  numbering.)
- **W4 (deferred):** multi-worker DP aggregation (`KVEventAggregator` /
  `KVConnectorKVEvents`).

## Risks/decisions

- **Live ZMQ transport DEFERRED (honest residual).** `ZmqEventPublisher`'s
  PUB/ROUTER sockets, replay buffer + ROUTER service, background publisher
  thread, and DP port offset are external-router / HW-ish and not reachable on
  the CPU gate. Decision: gate event GENERATION + the wire PAYLOAD, keep the seam
  faithful (`EventPublisherFactory` throws loudly on `"zmq"` so a request is
  never silently downgraded), and drop a real ZMQ publisher in behind the seam
  later without touching the emission sites or the encoder.
- **`lora_id` is always `None`.** The T0 `Request` carries only `lora_name`
  (LORA-RUNTIME deferred); on the base-model text path both are `None`, so the
  event is byte-identical to upstream.
- **Non-`None` `extra_keys` tuple is a loud-throw in the encoder** — unreachable
  because `generate_block_hash_extra_keys` returns `None` on the text path (its
  MM/LoRA branches are themselves deferred). Consistent with the existing
  `extra_keys` deferral.
- **Byte-exact required capturing the real encoder.** msgspec keeps the trailing
  `data_parallel_rank` even when `None` and encodes the store path's `extra_keys`
  as a non-empty list of `None`; a hand-derived encoder would have been wrong.

### W3

- **`SamplingParams::extra_args` is a STRING-VALUED slice of upstream's
  `dict[str, Any]`.** Upstream's `extra_args` also carries `kv_transfer_params`
  and `ec_transfer_params`, which are themselves dicts and cannot fit a
  `map<string,string>`. Modelling the ported slice as
  `optional<map<string,string>>` matches the precedent already in the tree
  (`include/vllm/v1/kv_offload/kv_connector.h:109` models
  `kv_transfer_params: dict[str, Any]` the same way) and keeps
  `Request::kv_cache_report_mode`'s derivation a literal 1:1 port of
  `request.py:116-127`. The nested-dict keys stay DEFERRED with their own rows,
  and the restriction is recorded on the field.
- **`vllm_xargs` (the HTTP door to `extra_args`) stays DEFERRED.** It is already
  recorded as such at `include/vllm/entrypoints/openai/protocol.h:37`. So
  `report_mode == "full"` is reachable from the C++ engine API but not yet from a
  `/v1/completions` body. That is a separate, already-tracked protocol gap, not a
  KV-events one; wiring it here would grow this change into the OpenAI surface.
- **The connector half of the drain is NOT ported.**
  `scheduler.py:1903-1910` also merges `self.connector.take_events()` into the
  batch. Our `kv_offload::KVConnector` has no `take_events` returning
  `KVCacheEvent`; the nearest thing, `OffloadingManager::take_events`
  (`include/vllm/v1/kv_offload/base.h:175`), returns `OffloadingEvent`, a
  different type belonging to `KV-OFFLOAD`. Bridging the two is that row's work,
  so W3 drains the KV-cache manager only and the connector merge is recorded
  DEFERRED at the call site.
- **Nothing in production calls `Scheduler::shutdown()` yet.** Upstream calls it
  from `EngineCore.shutdown`; our `EngineCore` has no shutdown path at all
  (`grep -n shutdown include/vllm/v1/engine/core.h` is empty except for a
  comment). The method is ported so the seam exists and is gated directly; with
  the only shipped publishers being `Null` (empty body) and `Collecting` (a flag)
  there is nothing to leak. The real consumer of it is the deferred ZMQ
  publisher's thread, which arrives with W2.

## Evidence

- Issue [#352](https://github.com/mudler/vllm.cpp/issues/352).
- Branch `row/KV-EVENTS-W2`, based on `main` at `7020de93`, merged forward to
  `origin/main` `157080c8` (158 commits) before the PR.
- RED-first capture, focused GREEN, and the full CPU `ctest` are recorded in the
  W3 commit message and the `KV-EVENTS` row in
  [`engine-matrix.md`](../engine-matrix.md).
- Post-merge re-gate on the merged tree (2026-08-11, CPU-only Release
  `build-cpu`): build 0 warnings; `test_kv_events` 12/12 · 105,
  `test_scheduler` 36/36 · 423, `test_llm_engine` 24/24 · 493; full
  `ctest --test-dir build-cpu -j 6` **385/385, 0 failed** in 28.31 s, with
  `test_openai_conformance` passing in-sweep at 0.35 s — which settles the
  earlier single failure as load starvation (it failed at load 121+, passes in
  0.35-0.59 s on a quiet box), not a defect.

## Stop conditions

- Return `NEEDS_DECISION` rather than inventing a fake socket transport if the
  envelope or the reuse path turns out not to be gateable without a real ZMQ
  publisher. (It is: `CollectingEventPublisher` sits behind the same abstract
  seam the ZMQ publisher will, so nothing about the wiring depends on the
  transport.)
- Do not touch the OpenAI protocol to make `extra_args` settable over HTTP; that
  is `vllm_xargs`, a separate deferred surface.
- Stop and reconcile the record instead of implementing if the row or spec no
  longer matches the tree.

## Outcome

**W1 (2026-07-27).** The msgpack payload was NOT derivable by reading msgspec's
documentation: two behaviours had to be captured from the real 0.21.1 encoder
before the bytes matched. `array_like=True` plus `omit_defaults=True` do NOT
compose the way the docs imply — the trailing `data_parallel_rank` survives in
the array even when it is `None` — and the store path's `extra_keys` encodes as
a non-empty list of `None` rather than being omitted. A hand-derived encoder
would have shipped wrong bytes that no local test could have caught. The
int-truncated block-hash form is the DEFAULT because upstream's env default is
`VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=1` (`envs.py:1816-1817`), not because it is
cheaper; both forms are gated.

**W3 (2026-08-11).** MEASURED. RED came in two stages, because a C++ port of a
missing API fails to compile before it can fail an assertion, and only the
second stage proves the tests actually discriminate: stage 1, the API absent —
14 compile errors naming `extra_args`, `kv_cache_report_mode`,
`set_kv_event_publisher` and `shutdown`; stage 2, the API present with the four
behaviours absent — 5 of 12 cases and 5 assertions RED, with the
default-off inertness case GREEN in BOTH stages, which is the point of having
it. GREEN 12/12, 105 assertions. Regression: `test_scheduler` 36/423,
`test_llm_engine` 23/450, `test_block_pool` 14/132, `test_kv_cache_manager`
10/74, `test_prefix_cache_stats` 12/36, `test_kv_cache_coordinator` 16/106,
`test_kv_cache_utils` 29/253, `test_async_scheduler` 7/63, `test_scheduler_lpm`
6/47. Full CPU `ctest` 368/369 at `-j 6`; the one failure, `test_openai_conformance`,
is a known load-starver and is 23/23, 252/252 re-run serially, so 369/369.

One expectation in the gate was WRONG on the first run and is worth recording,
because it is the thing the test was written to find out: a `report_mode=="full"`
hit publishes TWO `BlockStored` events in one batch, not one — the reuse report
raised by `get_computed_blocks` during `schedule()`, and then the ordinary store
of the single block the request still had to compute, parented on the last
reused block. The `"incremental"` control publishes only the second. The gate now
asserts that whole shape, including the parent linkage that lets a consumer
chain the prefix, which is strictly more than the count check it started as.

The three judgement calls worth keeping: the publisher is INJECTED for gating rather than adding a
`publisher="collecting"` config value, because inventing a `--kv-events-config`
value vLLM does not have would make the config surface diverge from upstream for
the sake of a test; `extra_args` is ported as a string-valued map rather than a
variant tree, because the two other keys upstream puts there belong to rows that
are themselves deferred and would dictate the wrong shape; and the connector leg
of the drain is left out rather than adapted, because `OffloadingEvent` and
`KVCacheEvent` are different types owned by different rows and coercing one into
the other here would hide a real gap behind a plausible-looking merge.

A fourth is recorded as a filed gap rather than a decision: `KVEventsConfig` has
no `PostInit`, so `{enable_kv_cache_events: true}` with an unset publisher throws
`unknown event publisher ''` where upstream (`kv_events.py:50-52`) resolves it to
zmq — which here should surface as the deliberately loud zmq-deferral message.
W3's tests set `publisher = "null"` explicitly and point at
[#353](https://github.com/mudler/vllm.cpp/issues/353); fixing it inside this
change would have been a silent repair of a different row's surface.
