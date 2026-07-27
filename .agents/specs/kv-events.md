# KV-EVENTS — the KV-cache event stream (ROAD-V1-D4)

Row: [`KV-EVENTS`](../engine-matrix.md) · Claim: `CLAIM-ROADMAP-D4-KV-EVENTS` ·
State: `ACTIVE` (event generation + payload DONE 2026-07-27; live ZMQ transport deferred).

## Scope

vLLM publishes a stream of KV-cache events (`BlockStored` / `BlockRemoved` /
`AllBlocksCleared`) so external consumers — prefix-cache-aware routers /
KV-cache load balancers — can track which prefix blocks each engine holds. Off
by default (`--kv-events-config enable_kv_cache_events=false`). This row ports,
additively, the event DATA TYPES, the EMISSION at the BlockPool store/remove/
clear sites, the publisher SEAM, and the `msgpack` PAYLOAD. The live ZMQ socket
transport is explicitly out of scope for W1 (deferred, see Risks/decisions).

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
- **Scheduler wiring** — `vllm/v1/core/sched/scheduler.py:154,1791,1802-1805`
  builds `KVEventBatch(ts, take_events())` and `publisher.publish(batch)`.
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

## Tests to port

Upstream: `tests/v1/core/test_prefix_caching.py:2040,2170,2228,2282,2368,2405`
(the event-emission cases). Re-expressed in `tests/vllm/v1/test_kv_events.cpp`
(6 cases / 62 assertions): byte-exact serialization vs the msgspec goldens (int
and bytes forms), the publisher seam, the store→reuse→evict→reset SEQUENCE, a
partial-store parent-linkage case, and the default-off inertness case. No
upstream test is dropped-without-reason; the ZMQ-transport socket cases are not
ported because the transport is deferred (see Risks/decisions).

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
- **W2 (deferred):** live ZMQ publisher (sockets, replay buffer, thread, DP port
  offset) behind the seam.
- **W3 (deferred):** engine/scheduler wiring of the `KVEventBatch` envelope +
  `publish` per step, and the `report_mode == "full"` reuse call in the manager
  (needs `Request.kv_cache_report_mode`).
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
