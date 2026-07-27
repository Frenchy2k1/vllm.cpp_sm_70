// Ported from: vllm/distributed/kv_events.py @ 555967922 (vLLM 0.26.0.dev0)
//
// Scope (KV-EVENTS, ROAD-V1-D4): the KV-cache event stream vLLM publishes for
// external KV routers / prefix-cache-aware load balancers. This header ports the
// event DATA TYPES (BlockStored / BlockRemoved / AllBlocksCleared and the
// KVEventBatch envelope), the publisher SEAM (EventPublisher +
// NullEventPublisher + a collecting publisher for gating), and the msgpack
// SERIALIZATION of a KVEventBatch — byte-for-byte identical to msgspec's
// encoding of the upstream structs.
//
// GATED here: event GENERATION (the BlockStored/BlockRemoved/AllBlocksCleared
// values the BlockPool emits, see block_pool.cpp) + payload SERIALIZATION (the
// msgpack bytes, gated in tests against captures from msgspec 0.21.1 encoding
// the 1:1 upstream struct definitions).
//
// DEFERRED (documented, HW-ish — an external router process, not reachable on
// the CPU-only gate): the LIVE ZMQ transport (ZmqEventPublisher's PUB/ROUTER
// sockets, the replay buffer, the background publisher thread, the DP port
// offset). The seam is kept faithful to --kv-events-config
// (vllm/config/kv_events.py) via KVEventsConfig + EventPublisherFactory so a real
// ZMQ publisher drops in later WITHOUT touching the emission sites or the
// payload encoder. KVEventAggregator / KVConnectorKVEvents (multi-worker DP
// aggregation) are likewise deferred — single-process on the gate.
//
// ENCODING FIDELITY (verified against msgspec.msgpack.Encoder, 0.21.1):
//   - EventBatch is `array_like=True`: KVEventBatch encodes as a msgpack ARRAY
//     [ts (float64), events (array), data_parallel_rank]. Despite omit_defaults,
//     msgspec KEEPS the trailing data_parallel_rank even when None (verified),
//     so the array is ALWAYS length 3.
//   - KVCacheEvent is `tag=True`, not array_like: each event encodes as a
//     msgpack MAP whose FIRST pair is "type" -> <ClassName> (the tag), followed
//     by the struct fields in definition order.
//   - omit_defaults: fields WITH a default (all defaulting to None here —
//     extra_keys, group_idx, kv_cache_spec_kind, kv_cache_spec_sliding_window,
//     locality) are OMITTED when equal to the default. Fields WITHOUT a default
//     (block_hashes, parent_block_hash, token_ids, block_size, lora_id, medium,
//     lora_name) are ALWAYS emitted, even when None (encoded as nil).
//   - ints use minimal msgpack width; bytes use bin8/16/32; str uses fixstr/
//     str8/16/32; float uses float64 (cb). Big-endian throughout.
#ifndef VLLM_DISTRIBUTED_KV_EVENTS_H_
#define VLLM_DISTRIBUTED_KV_EVENTS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "vllm/v1/core/kv_cache_utils.h"  // ExternalBlockHash, ExtraKeys

namespace vllm::distributed {

// Medium constants (kv_events.py:45-48): where the cached blocks live relative
// to the publisher. The GPU BlockPool always emits MEDIUM_GPU.
inline constexpr const char* MEDIUM_GPU = "GPU";
inline constexpr const char* MEDIUM_CPU = "CPU";
inline constexpr const char* MEDIUM_FS = "FS";
inline constexpr const char* MEDIUM_OBJ = "OBJ";

// ExternalBlockHash is defined next to BlockHash in kv_cache_utils.h (mirroring
// upstream, where kv_events.py imports it from kv_cache_utils). Re-exported here
// for the event payload types below.
using vllm::v1::ExternalBlockHash;

// The per-block extra-key entry carried in a BlockStored event
// (kv_events.py:65-70). Each entry is `tuple[Any, ...] | None`, one per block.
// For the text path (no MM / LoRA / cache-salt) every entry is None (nullopt);
// the MM/LoRA/salt tuple SHAPE in the payload is DEFERRED with
// generate_block_hash_extra_keys itself (kv_cache_utils.h), so a present entry
// is a not-yet-reachable branch (the encoder throws on it).
using EventExtraKey = vllm::v1::ExtraKeys;  // std::optional<vector<ExtraKey>>

// BlockStored (kv_events.py:51-95): blocks that were just cached (or, via
// emit_cached_block_events, reused). Field order MATTERS for the payload.
struct BlockStored {
  static constexpr const char* kTag = "BlockStored";

  // --- required (no default) : ALWAYS emitted -------------------------------
  std::vector<vllm::v1::ExternalBlockHash> block_hashes;
  std::optional<vllm::v1::ExternalBlockHash> parent_block_hash;  // None -> nil
  std::vector<int64_t> token_ids;
  int64_t block_size = 0;
  // lora_id: DEPRECATED int adapter id (retained for back-compat). Our Request
  // carries only lora_name, so this is always None on the ported path.
  std::optional<int64_t> lora_id;
  std::optional<std::string> medium;     // "GPU" on the ported path
  std::optional<std::string> lora_name;  // None on the base-model text path

  // --- defaulted (None) : OMITTED when nullopt ------------------------------
  // extra_keys: one entry per block in block_hashes. Present (non-nullopt) even
  // for text (a vector of nullopt entries), because upstream builds a non-empty
  // list; nullopt here mirrors upstream `extra_keys=None`.
  std::optional<std::vector<EventExtraKey>> extra_keys;
  std::optional<int64_t> group_idx;
  std::optional<std::string> kv_cache_spec_kind;
  std::optional<int64_t> kv_cache_spec_sliding_window;
  std::optional<std::string> locality;
};

// BlockRemoved (kv_events.py:98-113): blocks evicted from / removed from the
// prefix cache.
struct BlockRemoved {
  static constexpr const char* kTag = "BlockRemoved";

  // --- required (no default) : ALWAYS emitted -------------------------------
  std::vector<vllm::v1::ExternalBlockHash> block_hashes;
  std::optional<std::string> medium;  // "GPU" on the ported path

  // --- defaulted (None) : OMITTED when nullopt ------------------------------
  std::optional<int64_t> group_idx;
  std::optional<std::string> locality;
};

// AllBlocksCleared (kv_events.py:116-117): the whole prefix cache was reset.
struct AllBlocksCleared {
  static constexpr const char* kTag = "AllBlocksCleared";
};

// KVCacheEvent — the tagged union of all KV-cache events (kv_events.py:36-42 +
// KVEventBatch.events, :120-121). Mirrors msgspec's tagged Struct hierarchy.
using KVCacheEvent = std::variant<BlockStored, BlockRemoved, AllBlocksCleared>;

// KVEventBatch (kv_events.py:25-33,120-121): the published envelope. `ts` is the
// batch timestamp; `data_parallel_rank` is annotated by the publisher (None
// until then). Encodes as a msgpack array [ts, events, data_parallel_rank].
struct KVEventBatch {
  double ts = 0.0;
  std::vector<KVCacheEvent> events;
  std::optional<int64_t> data_parallel_rank;
};

// Encode a KVEventBatch to its msgpack payload, byte-for-byte identical to
// `msgspec.msgpack.Encoder().encode(batch)` on the 1:1 upstream structs. This is
// the exact wire payload ZmqEventPublisher._publisher_thread packs
// (kv_events.py:445).
std::string encode_kv_event_batch(const KVEventBatch& batch);

// --- Publisher seam (kv_events.py:246-284, 511-543) --------------------------

// KVEventsConfig — the --kv-events-config surface (vllm/config/kv_events.py).
// Only the fields the seam needs; the ZMQ transport fields are carried so a real
// publisher can consume them later (endpoint/replay/hwm/... are DEFERRED here).
struct KVEventsConfig {
  bool enable_kv_cache_events = false;
  // "null" or "zmq". __post_init__ resolves an empty value to "zmq" when events
  // are enabled, else "null" (kv_events.py:50-52).
  std::string publisher;
  std::string endpoint = "tcp://*:5557";
  std::optional<std::string> replay_endpoint;
  int buffer_steps = 10000;
  int hwm = 100000;
  int max_queue_size = 100000;
  std::string topic;
};

// EventPublisher (kv_events.py:246-274): the abstract publisher seam.
class EventPublisher {
 public:
  explicit EventPublisher(int data_parallel_rank = 0)
      : data_parallel_rank_(data_parallel_rank) {}
  virtual ~EventPublisher() = default;

  EventPublisher(const EventPublisher&) = delete;
  EventPublisher& operator=(const EventPublisher&) = delete;

  // Emit a batch in order. Publishers annotate data_parallel_rank when unset.
  virtual void publish(const KVEventBatch& events) = 0;
  virtual void shutdown() = 0;

 protected:
  int data_parallel_rank_;
};

// NullEventPublisher (kv_events.py:277-284): the no-op default (events disabled).
class NullEventPublisher : public EventPublisher {
 public:
  void publish(const KVEventBatch&) override {}
  void shutdown() override {}
};

// CollectingEventPublisher — a NON-ZMQ concrete publisher that keeps every
// published batch in memory (both as the decoded struct and its msgpack bytes).
// Stands in for the ZMQ transport so event generation + payload can be gated
// without a live socket. (A ZmqEventPublisher slots in behind the same seam.)
class CollectingEventPublisher : public EventPublisher {
 public:
  explicit CollectingEventPublisher(int data_parallel_rank = 0)
      : EventPublisher(data_parallel_rank) {}

  void publish(const KVEventBatch& events) override {
    KVEventBatch annotated = events;
    if (!annotated.data_parallel_rank.has_value()) {
      annotated.data_parallel_rank = data_parallel_rank_;
    }
    encoded_.push_back(encode_kv_event_batch(annotated));
    batches_.push_back(std::move(annotated));
  }
  void shutdown() override { shut_down_ = true; }

  const std::vector<KVEventBatch>& batches() const { return batches_; }
  const std::vector<std::string>& encoded() const { return encoded_; }
  bool was_shut_down() const { return shut_down_; }

 private:
  std::vector<KVEventBatch> batches_;
  std::vector<std::string> encoded_;
  bool shut_down_ = false;
};

// EventPublisherFactory::create (kv_events.py:523-543): build a publisher from
// the config. Returns a NullEventPublisher when events are disabled or the
// publisher is "null". The "zmq" kind is DEFERRED (no live socket on the gate):
// requesting it throws so the deferral is loud, never silently wrong.
class EventPublisherFactory {
 public:
  static std::unique_ptr<EventPublisher> create(const KVEventsConfig* config,
                                                int data_parallel_rank = 0);
};

}  // namespace vllm::distributed

#endif  // VLLM_DISTRIBUTED_KV_EVENTS_H_
