# `MooncakeStoreConnector`: a native C++ client over the landed `KVConnector` seam

**Rows touched:** `KV-MOONCAKE-STORE` (new, primary), `KV-CONNECTORS` (the seam it
reuses; its Mooncake disposition is corrected by this spec).
**Claim:** `CLAIM-MOONCAKE-STORE`. **Issue:**
[#287](https://github.com/mudler/vllm.cpp/issues/287).
**Portfolio row:** `ROAD-V1-D4` (KV persistent state + external KV-cache provider
interoperability).

**Status: SPEC ONLY.** No implementation, no build, no GPU in this change. This
document is committed *before* any code, per the spec-before-code rule.

## Headline verdict

vLLM ships **two** Mooncake connectors and our record conflates them.
`.agents/engine-matrix.md` (`KV-CONNECTORS`, from the 2026-07-22 spike) lists
Mooncake among the connectors that are NOT SCHEDULED because each "needs an
external RDMA or store dependency absent from our boxes and ungateable on GB10".
Read at the pin, that verdict is **correct for `MooncakeConnector` and too strong
for `MooncakeStoreConnector`**.

| | `MooncakeConnector` | `MooncakeStoreConnector` |
|---|---|---|
| Shape | P2P prefiller→decoder push over the Mooncake Transfer Engine | Shared KV **object store** pool |
| Our analogue | NIXL | LMCache |
| Upstream | `v1/mooncake/mooncake_connector.py` (2179 lines) | `v1/mooncake/store/` (3554 lines) |
| To run it | 2 nodes + bootstrap FastAPI server + disagg proxy + RDMA | one `mooncake_master`; single node; `protocol: "tcp"` supported |
| Disposition | **stays NOT SCHEDULED** | **this spec** |

Two findings drive the reopening:

1. **Mooncake is native C++, so we link it rather than reimplement it.**
   `mooncake-store/include/client_service.h` exposes `mooncake::Client` with
   `Create` / `Get` / `BatchGet` / `Put` / `BatchPut` / `Upsert` / `Remove` /
   `Query` / `BatchQuery` / `IsExist` / `RegisterLocalMemory` / `MountSegment`.
   `mooncake.store.MooncakeDistributedStore`, the object vLLM imports, is a
   pybind wrapper (`pyclient.h`) over that class. The Transfer Engine
   additionally ships a pure C ABI (`transfer_engine_c.h`:
   `createTransferEngine`, `registerLocalMemory`, `allocateBatchID`,
   `submitTransfer`, `getTransferStatus`, …). We would call **the same native
   code vLLM calls**, one layer lower.

   This inverts the usual cost. LMCache is a Python library, so
   `KV-EXTERNAL-CACHE` W1/W2 had to reimplement the `lm://` wire from scratch
   (`src/vllm/v1/kv_offload/lmcache/remote_protocol.cpp`, 201 lines, a byte-exact
   port of a 186-byte `struct.pack` header; `remote_client.cpp`, 313 lines of
   socket framing). None of that class of work exists here.

2. **The single-node TCP configuration is gateable on hardware we already have.**
   `mooncake_master --port 50051` plus `"protocol": "tcp"`, `"mode": "embedded"`
   is a working store with no RDMA NIC. That is the same gate shape
   `KV-EXTERNAL-CACHE` W3/W5 already runs against a live `lmcache.v1.server`.

What remains genuinely ungateable is the *payoff*: `register_kv_caches`
registers the GPU KV tensors themselves and transfers run zero-copy (GPUDirect)
RDMA straight into the paged blocks. No box we have sits on an RDMA fabric, so
the speed axis stays **open and unmeasured** — recorded as a named gap, never as
a ceiling (§Gates G5).

## Scope

**In scope.** A `MooncakeStoreConnector` implementing the landed abstract
`KVConnector` (`include/vllm/v1/kv_offload/kv_connector.h`), backed by the native
`mooncake::Client`, selectable by name through the existing
`--kv-transfer-config` JSON surface, **default OFF and inert when unselected**.
Specifically: the pool key format, the scheduler-side lookup and load bookkeeping,
the worker-side buffer registration and batch get/put, the single-node TCP
correctness gate, and honest reporting of the unmeasured RDMA axis.

**Out of scope, dispositioned not deferred-by-omission.**

- `MooncakeConnector` (P/D disaggregation, bootstrap server, proxy). Stays NOT
  SCHEDULED; needs two nodes and a fabric.
- `MultiConnector` composition of the two (upstream's XpYd recipe). Follows P/D.
- The SSD/`enable_offload` tier and `standalone-store` mode (§Risks R4). Our own
  `fs_tier` already provides a local disk tier; the Mooncake disk tier duplicates
  it without adding a capability we can measure here.
- Cross-layer block packing (`enable_cross_layers_blocks`), multi-replica
  `ReplicateConfig` / `preferred_segment` steering, KV-event aggregation and the
  Prometheus metric surface (`store/metrics.py`). Each is additive on top of a
  working connector.
- TP > 1 / PCP / DCP rank-sharded keys. The key format reserves the fields from
  W1 (§Port map) so enabling them later is not a format change, but no multi-rank
  gate is in scope.

## Upstream chain

Pin `555967922` (vLLM 0.26.0.dev0), reference checkout `${VLLM_SOURCE}`.

| Concern | Upstream anchor |
|---|---|
| Registration (both connectors) | `vllm/distributed/kv_transfer/kv_connector/factory.py:219,224` |
| Connector shell, role split, HMA | `.../v1/mooncake/store/connector.py` |
| Key format | `.../store/data.py:100` (`KeyMetadata`), `:116` (`PoolKey`), `:137` (`build_prefix`), `:158` (`build_key_string`) |
| Block-size → chunk hash collapse | `.../store/data.py:80` (`chunk_hashes_for_block_size`) |
| Token → key/address mapping | `.../store/data.py:188` (`key_for`), `:197` (`prepare_value`), `:241` (`process_tokens`) |
| Scheduler hooks | `.../store/scheduler.py:73`, `:120`, `:155`, `:343`, `:367` |
| Store config schema | `.../store/worker.py:106` (`MooncakeStoreConfig`) |
| Client construction | `.../store/worker.py:1017-1037` (`store.setup(...)`) |
| Buffer registration + layout detection | `.../store/worker.py:1237` (`register_kv_caches`) |
| Batch transfer | `.../store/worker.py:698` (`batch_put_from_multi_buffers`), `:911` (`batch_get_into_multi_buffers`) |
| Existence probe | `.../store/worker.py:604`, `:1532` (`batch_is_exist`) |
| Async issue point | `.../store/worker.py:1367` (`get_finished`) |
| Lookup RPC | `.../store/worker.py:1607` (`LookupKeyServer`), `:1686` (`LookupKeyClient`) |
| Operator surface | `docs/features/mooncake_store_connector_usage.md` |

Mooncake side (`kvcache-ai/Mooncake`, revision to be pinned at W0):

- `mooncake-store/include/client_service.h` — `mooncake::Client`, the C++ API
- `mooncake-store/include/pyclient.h` — the pybind wrapper vLLM actually imports
- `mooncake-transfer-engine/include/transfer_engine_c.h` — the C ABI

**Both-sides rule.** Every claim below is read from upstream source. The running
oracle arm (execute pinned vLLM with `MooncakeStoreConnector` on the identical
workload) is owed at W4 and is what the key-agreement gate G3 depends on.

## Our baseline

`KV-CONNECTORS` W5 landed the abstract seam and `KV-EXTERNAL-CACHE` W1–W5 landed
the first real client over it. What already exists, and is reused unchanged:

- `include/vllm/v1/kv_offload/kv_connector.h` — the abstract `KVConnector`:
  `MatchResult` with the `std::optional<int>` **third state** ("not ready, re-ask
  next step") and the `load_async` flag, `update_state_after_alloc`,
  `build_connector_meta`, `RequestFinishedResult` with `delay_free` ownership,
  `SupportsHMA` multi-group finish, `supports_worker_transfer_on(DeviceType)`.
- `KVConnectorFactory` + `REGISTER_KV_CONNECTOR` /
  `REGISTER_KV_CONNECTOR_WITH_WORKER` — compile-time registration, the recorded
  C++ analogue of vLLM's `importlib` module path.
- `include/vllm/config/kv_transfer.h` — `KVTransferConfig`, `KVRole`,
  `KVLoadFailurePolicy`, and `ParseKVTransferConfigJson` behind vLLM's own
  `--kv-transfer-config` flag name and JSON shape.
- `GPUModelRunner::execute_model`'s `ConnectorLoadExternalKv` (before forward) /
  `ConnectorStorePromptKv` (after) worker wiring, from LMCache W5.
- Deterministic block hashes across processes (`KV-OFFLOAD` W1): `init_none_hash`
  resolves explicit arg > `$VLLM_PREFIX_CACHING_HASH_SEED` > `$PYTHONHASHSEED` >
  a fixed built-in default.

The seam needs **no change**. The `kv_connector.h` note that LMCache W3 was
"implement the abstract `KVConnector` with the landed W2 client — no further seam
change" holds verbatim for this connector.

## Port map

New subtree `include/vllm/v1/kv_offload/mooncake/` + `src/.../mooncake/`,
mirroring upstream's `store/` layout.

| New file | Mirrors | Content |
|---|---|---|
| `pool_key.{h,cpp}` | `store/data.py:80-163` | `KeyMetadata`, the prefix builder, `chunk_hashes_for_block_size` |
| `key_mapper.{h,cpp}` | `store/data.py:168-320` | `key_for`, `prepare_value(s)`, `process_tokens` |
| `store_config.{h,cpp}` | `store/worker.py:106-214` | the `MOONCAKE_CONFIG_PATH` JSON schema |
| `store_client.{h,cpp}` | `store/worker.py:1017-1037` | thin RAII wrapper over `mooncake::Client`, `tl::expected` → our error type |
| `mooncake_connector.{h,cpp}` | `store/connector.py`, `scheduler.py`, `worker.py` | the `KVConnector` implementation, both halves |

**Key format** (`data.py:137-163`), mirrored byte-for-byte so a key we write is a
key vLLM finds and vice versa:

```
{cache_prefix}@{model_name}@tp_rank:N@pcpN@dcpN@pp_rank:N@group:G@{chunk_hash_hex}
```

`cache_prefix` and its `@` separator are omitted entirely when empty, keeping
keys byte-identical to the unprefixed format. Chunks are keyed by the **last**
constituent sub-hash (`chunk_hashes_for_block_size`), which is what keeps a key
one digest regardless of `block_size`.

**Scheduler half.** `get_num_new_matched_tokens` probes the store for the longest
cached prefix and returns `need_to_allocate = hit − num_computed_tokens`, plus
`load_async`. Requests shorter than one block short-circuit to `(0, false)`
(`scheduler.py:80-81`). W1 ships the **synchronous** shape — `load_async=false`,
lookup inline — exactly as LMCache W3 did; the `std::nullopt` third state is
already on the seam and is what a later async lookup fills without a shape change.

**Worker half.** `RegisterLocalMemory` over each distinct KV cache storage
region, then `BatchGet` / `BatchPut` with `Slice`s pointing directly at paged
blocks. Layout is detected by byte-stride exactly as upstream
(`worker.py:1237`): a dim whose byte-stride exceeds the page size is an outer
segment dim (K/V-first, FlashAttn-style → split segments); no such dim means
blocks-first (FlashInfer/MLA → one segment).

### Recorded deviations

1. **No lookup RPC.** Upstream needs `LookupKeyClient`/`LookupKeyServer`
   (`worker.py:1607,1686`, ZMQ REQ/REP, 4-byte big-endian framing) purely because
   vLLM's scheduler and worker are separate *processes*. Ours are in one process,
   so the lookup collapses to a direct call. Same class of deviation as
   compile-time registration replacing `importlib`. The ZMQ wire is **not**
   reproduced.
2. **No `PYTHONHASHSEED` requirement.** Upstream's docs require
   `PYTHONHASHSEED=0` on every instance sharing a store, because Python
   randomizes hash seeds per process. Our hashes are deterministic by
   construction since `KV-OFFLOAD` W1. We are strictly better here and will say
   so in `docs/USAGE.md` rather than copying the caveat.
3. **`tl::expected` at the boundary.** `mooncake::Client` returns
   `tl::expected<T, ErrorCode>`. Converted to our error type at the
   `store_client` wrapper so `tl::expected` does not leak into the connector or
   the seam.
4. **Synchronous first.** Upstream issues loads and stores from `get_finished()`
   for compute/IO overlap, with `start_load_kv` / `save_kv_layer` /
   `wait_for_save` all no-ops. Our runner is synchronous; W1–W4 keep the
   documented no-op worker hooks and forfeit the overlap. Recorded as a named
   residual (§Risks R2), not silently dropped.

## Tests to port

Upstream `tests/v1/kv_connector/` has no `MooncakeStoreConnector` suite that runs
without the `mooncake` package. The tests below are therefore written by us and
recorded as from-scratch in the porting inventory §9, except the key-format and
`process_tokens` cases, which are ported against fixtures dumped from upstream's
own `data.py` driver (the same technique `test_lmcache_key_agreement` used).

| Test | What it pins |
|---|---|
| `test_mooncake_pool_key` | key string byte-for-byte vs fixtures from upstream `PoolKey.to_string()`, incl. empty vs non-empty `cache_prefix`, and `chunk_hashes_for_block_size` when `block_size > hash_block_size` |
| `test_mooncake_key_mapper` | `process_tokens` chunk boundaries, `mask_num` skipping, and the `put_step`/`put_step_rank` stride vs fixtures |
| `test_mooncake_store_config` | the `MOONCAKE_CONFIG_PATH` JSON schema: parse, defaults, `mode`/`global_segment_size` cross-validation, malformed refusal |
| `test_mooncake_connector` | store → fresh "restarted" connector → lookup shortcuts prefill → load byte-identical; foreign-key refusal; default-off inertness. Against an in-process mock store, always-on in CI, no external dep |
| `test_mooncake_output_invariance` | connector-ON generated tokens bit-identical to connector-OFF on a real OPT-125m loop against a live `mooncake_master` (opt-in, `VT_MOONCAKE_LIVE_*`) |

## Gates

| ID | Gate | Where |
|---|---|---|
| G1 | **Default-off inertness.** Every SACRED gate byte-identical with the connector unselected; the null-connector path unchanged | dgx.casa |
| G2 | **Connector round-trip.** Store a prefix → a fresh connector looks it up and shortcuts prefill through the REAL scheduler → loaded bytes byte-identical → foreign-key refusal | CI, mock store |
| G3 | **Key agreement with the real thing.** A key our C++ derives is byte-identical to the key upstream's `PoolKey`/`process_tokens` derives for the same tokens; and a value **vLLM PUT** into a live `mooncake_master` is one **we GET** byte-identical | live, TCP |
| G4 | **Output invariance.** Connector-ON tokens bit-identical to connector-OFF (cold full prefill), on both a store→restart→load cycle and a genuinely cold second process | live, TCP |
| G5 | **Speed: explicitly OPEN, not waived.** The TCP/loopback arm is expected to be noise-dominated or negative on a small model and **no speedup is claimed from it**. The RDMA/GPUDirect arm — the actual reason this connector exists — is **unmeasured for want of a fabric** and is recorded in `docs/BENCHMARKS.md` as a named open axis with the next traceable hypothesis, never as a ceiling | PENDING external resource |

G5 is the honest-reporting obligation for this row. Per the never-declare-a-ceiling
rule, "we cannot measure it here" is a resource gap with an owner, not a result.

## Dependencies

- **New external dependency: Mooncake.** Built from source, pinned by revision at
  W0. Behind a `-DVLLM_CPP_MOONCAKE=ON` build flag, **default OFF**, so no
  existing build recipe changes and no gate acquires a new dependency.
- `mooncake_master` binary for the live gates only.
- No RDMA NIC for G1–G4 (`protocol: "tcp"`). A fabric for G5, which we do not have.

## Work breakdown

| W | Deliverable | Gate |
|---|---|---|
| W0 | **Link spike.** Build Mooncake from source, pin the revision, link `mooncake::Client` from a throwaway C++ TU, confirm `Create`/`Put`/`Get`/`IsExist` work outside pybind against a local `mooncake_master` over TCP. **Go/no-go: if the C++ API is not usable standalone, the whole row stops here.** | a compiling, running TU |
| W1 | Key format + mapper + config schema, pure CPU, inert (no call site) | G2 fixtures, `test_mooncake_pool_key`, `test_mooncake_key_mapper`, `test_mooncake_store_config` |
| W2 | `store_client` RAII wrapper over `mooncake::Client` | round-trip vs a live master over TCP |
| W3 | The `KVConnector` implementation, both halves, registered, default OFF | G1, G2 |
| W4 | Live interop: run the pinned oracle with `MooncakeStoreConnector` on the identical workload, prove key agreement and cross-engine load | G3 |
| W5 | Worker wiring + full-model output invariance | G4 |
| W6 | Honest speed reporting: the TCP arm measured and reported as noise-dominated; the RDMA axis recorded OPEN | G5 |

W0 is a genuine stop point, not a formality (§Stop conditions S1).

## Risks / decisions

- **R1 — Mooncake is an unpinned moving target.** Same class of risk as LMCache:
  an interop feature carries a version-sync cost. Mitigated by pinning a revision
  at W0 and recording it alongside the vLLM pin. Their C++ API is not a declared
  stable ABI; a breaking change is a real maintenance cost, not a hypothetical.
- **R2 — We forfeit the compute/IO overlap.** Upstream's design issues transfers
  from `get_finished()` precisely so they overlap the forward pass. Our
  synchronous runner cannot, so even the TCP arm will look worse than upstream's
  own numbers. This must not be reported as a Mooncake result; it is our runner's
  shape (§Deviations 4).
- **R3 — GB10 unified memory.** `global_segment_size` reserves **host** RAM,
  which on GB10 is the same pool as GPU memory. The upstream doc's `80GB` would
  OOM-reboot a Spark. Gate configs use a small segment, and the spec's example
  config must not be copy-pasted from upstream.
- **R4 — Capability overlap with what we already have.** Our `fs_tier` +
  `tiering_manager` already give local CPU/disk tiering. The *marginal* capability
  Mooncake adds is (a) a prefix cache shared across processes and nodes, (b) RDMA
  zero-copy into GPU blocks, (c) an ecosystem-standard store other engines speak.
  Only (a) and (c) are demonstrable on our hardware. If the user's priority is
  local tiering, this row buys little; it is justified by interoperability.
- **R5 — HMA / hybrid models.** Upstream `_validate_kv_cache_config`
  (`connector.py:98-125`) refuses `CrossAttentionSpec`, `MambaSpec` with a
  block size mismatched to `cache_config.block_size`, and PCP/DCP > 1 with hybrid
  attention. Our gate models are two-group hybrids, so this refusal set must be
  mirrored exactly or we will silently produce wrong keys for a group.
- **R6 — Layout detection is a silent-wrong-bytes hazard.** The stride heuristic
  at `worker.py:1237` infers K/V-first vs blocks-first from byte strides. If our
  paged layout does not present the same stride relationship, registration
  succeeds and transfers move the wrong bytes. W3 must assert the detected
  segmentation against a known-good expectation per backend rather than trusting
  the heuristic. Compare the failure mode recorded in
  [gate-comparing-shared-helper-proves-consistency-not-correctness].
- **D1 — Why the store and not P/D.** P/D disaggregation needs two nodes, a
  fabric and a proxy. The store needs one process and a TCP socket. The store is
  also the direct analogue of the capability we already ship (LMCache), so it
  reuses the seam, the gate shape and the review knowledge.

## Stop conditions

- **S1.** W0 fails to link or run `mooncake::Client` standalone → stop, record the
  finding, close the row. Do not fall back to reimplementing the store wire from
  scratch: unlike `lm://`, this is a real distributed protocol with a master
  service, and a from-scratch client is not a proportionate response.
- **S2.** Key agreement (G3) cannot be established against the running oracle →
  stop before W5. A connector that cannot share a cache with vLLM has no reason
  to exist.
- **S3.** Any gate requires an RDMA fabric to pass → that gate is PENDING an
  external resource with a named owner, never waived and never reported as met.
- **S4.** The build flag cannot be kept default-OFF and inert → stop; no existing
  gate may acquire a new external dependency.
