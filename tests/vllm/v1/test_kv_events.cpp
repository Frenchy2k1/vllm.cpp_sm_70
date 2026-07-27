// Tests for the KV-cache event stream (KV-EVENTS, ROAD-V1-D4).
// Ported from vllm/tests/... event coverage of
// vllm/tests/v1/core/test_prefix_caching.py @ 555967922 (the BlockStored /
// BlockRemoved / AllBlocksCleared emission cases) plus a payload-serialization
// gate.
//
// TWO gate layers:
//   1. SERIALIZATION (byte-exact): the msgpack payload our encoder produces for
//      a fixed KVEventBatch equals msgspec.msgpack.Encoder()'s output on the 1:1
//      upstream struct definitions. The golden byte vectors below were captured
//      from msgspec 0.21.1 encoding the exact upstream structs (see the KV-events
//      spec for the capture script). Both the default INT-truncated block-hash
//      form and the raw-BYTES form are gated.
//   2. EMISSION SEQUENCE: driving a real BlockPool through
//      store -> reuse -> evict -> reset emits exactly the right event sequence,
//      with the correct block hashes / token_ids / parent linkage / group_idx /
//      ordering, mirroring block_pool.py. The default (events-disabled) path
//      emits NOTHING, so existing prefix-cache behaviour is byte-identical.
#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "vllm/distributed/kv_events.h"
#include "vllm/sampling_params.h"
#include "vllm/v1/core/block_pool.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/request.h"

using vllm::distributed::AllBlocksCleared;
using vllm::distributed::BlockRemoved;
using vllm::distributed::BlockStored;
using vllm::distributed::CollectingEventPublisher;
using vllm::distributed::encode_kv_event_batch;
using vllm::distributed::EventPublisherFactory;
using vllm::distributed::ExternalBlockHash;
using vllm::distributed::KVEventBatch;
using vllm::distributed::KVEventsConfig;
using vllm::distributed::NullEventPublisher;
using vllm::v1::BlockPool;
using vllm::v1::get_request_block_hasher;
using vllm::v1::init_none_hash;
using vllm::v1::KVCacheBlock;
using vllm::v1::maybe_convert_block_hash;
using vllm::v1::Request;
using vllm::v1::sha256_cbor;

namespace {

Request MakeRequest(const std::string& id, const std::vector<int32_t>& tokens,
                    int block_size) {
  return Request(id, tokens, vllm::SamplingParams{}, /*arrival_time=*/0.0,
                 get_request_block_hasher(block_size, sha256_cbor));
}

std::vector<int32_t> Iota(int n) {
  std::vector<int32_t> v;
  v.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) v.push_back(i);
  return v;
}

std::string ToHex(const std::string& bytes) {
  static const char* k = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2);
  for (unsigned char c : bytes) {
    out.push_back(k[c >> 4]);
    out.push_back(k[c & 0xf]);
  }
  return out;
}

std::string Bytes(std::initializer_list<int> vals) {
  std::string s;
  for (int v : vals) s.push_back(static_cast<char>(v));
  return s;
}

}  // namespace

// ---------------------------------------------------------------------------
// Layer 1: byte-exact serialization vs the msgspec golden captures.
// ---------------------------------------------------------------------------

// Golden A: bytes ExternalBlockHash (VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0),
// store(2 blocks, no parent, extra_keys=[None,None]) + remove + clear, ts=1234.5.
// Captured from msgspec 0.21.1 (len 279).
TEST_CASE("kv_events: msgpack payload is byte-exact (bytes hash form)") {
  const std::string h0 = "hash_block_0000";  // 15 raw bytes
  const std::string h1 = "hash_block_0001";

  KVEventBatch batch;
  batch.ts = 1234.5;

  BlockStored stored;
  stored.block_hashes = {ExternalBlockHash{h0}, ExternalBlockHash{h1}};
  stored.parent_block_hash = std::nullopt;
  stored.token_ids = {1, 2, 3, 4, 5, 6, 7, 8};
  stored.block_size = 4;
  stored.lora_id = std::nullopt;
  stored.medium = "GPU";
  stored.lora_name = std::nullopt;
  stored.extra_keys = std::vector<vllm::distributed::EventExtraKey>{
      std::nullopt, std::nullopt};
  stored.group_idx = 0;

  BlockRemoved removed;
  removed.block_hashes = {ExternalBlockHash{h0}};
  removed.medium = "GPU";
  removed.group_idx = 0;

  batch.events = {stored, removed, AllBlocksCleared{}};

  const std::string golden = Bytes(
      {0x93, 0xcb, 0x40, 0x93, 0x4a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x93, 0x8a,
       0xa4, 0x74, 0x79, 0x70, 0x65, 0xab, 0x42, 0x6c, 0x6f, 0x63, 0x6b, 0x53,
       0x74, 0x6f, 0x72, 0x65, 0x64, 0xac, 0x62, 0x6c, 0x6f, 0x63, 0x6b, 0x5f,
       0x68, 0x61, 0x73, 0x68, 0x65, 0x73, 0x92, 0xc4, 0x0f, 0x68, 0x61, 0x73,
       0x68, 0x5f, 0x62, 0x6c, 0x6f, 0x63, 0x6b, 0x5f, 0x30, 0x30, 0x30, 0x30,
       0xc4, 0x0f, 0x68, 0x61, 0x73, 0x68, 0x5f, 0x62, 0x6c, 0x6f, 0x63, 0x6b,
       0x5f, 0x30, 0x30, 0x30, 0x31, 0xb1, 0x70, 0x61, 0x72, 0x65, 0x6e, 0x74,
       0x5f, 0x62, 0x6c, 0x6f, 0x63, 0x6b, 0x5f, 0x68, 0x61, 0x73, 0x68, 0xc0,
       0xa9, 0x74, 0x6f, 0x6b, 0x65, 0x6e, 0x5f, 0x69, 0x64, 0x73, 0x98, 0x01,
       0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0xaa, 0x62, 0x6c, 0x6f, 0x63,
       0x6b, 0x5f, 0x73, 0x69, 0x7a, 0x65, 0x04, 0xa7, 0x6c, 0x6f, 0x72, 0x61,
       0x5f, 0x69, 0x64, 0xc0, 0xa6, 0x6d, 0x65, 0x64, 0x69, 0x75, 0x6d, 0xa3,
       0x47, 0x50, 0x55, 0xa9, 0x6c, 0x6f, 0x72, 0x61, 0x5f, 0x6e, 0x61, 0x6d,
       0x65, 0xc0, 0xaa, 0x65, 0x78, 0x74, 0x72, 0x61, 0x5f, 0x6b, 0x65, 0x79,
       0x73, 0x92, 0xc0, 0xc0, 0xa9, 0x67, 0x72, 0x6f, 0x75, 0x70, 0x5f, 0x69,
       0x64, 0x78, 0x00, 0x84, 0xa4, 0x74, 0x79, 0x70, 0x65, 0xac, 0x42, 0x6c,
       0x6f, 0x63, 0x6b, 0x52, 0x65, 0x6d, 0x6f, 0x76, 0x65, 0x64, 0xac, 0x62,
       0x6c, 0x6f, 0x63, 0x6b, 0x5f, 0x68, 0x61, 0x73, 0x68, 0x65, 0x73, 0x91,
       0xc4, 0x0f, 0x68, 0x61, 0x73, 0x68, 0x5f, 0x62, 0x6c, 0x6f, 0x63, 0x6b,
       0x5f, 0x30, 0x30, 0x30, 0x30, 0xa6, 0x6d, 0x65, 0x64, 0x69, 0x75, 0x6d,
       0xa3, 0x47, 0x50, 0x55, 0xa9, 0x67, 0x72, 0x6f, 0x75, 0x70, 0x5f, 0x69,
       0x64, 0x78, 0x00, 0x81, 0xa4, 0x74, 0x79, 0x70, 0x65, 0xb0, 0x41, 0x6c,
       0x6c, 0x42, 0x6c, 0x6f, 0x63, 0x6b, 0x73, 0x43, 0x6c, 0x65, 0x61, 0x72,
       0x65, 0x64, 0xc0});

  const std::string got = encode_kv_event_batch(batch);
  CHECK(got.size() == golden.size());
  CHECK(ToHex(got) == ToHex(golden));
}

// Golden B: int ExternalBlockHash (the DEFAULT form), parent set, varied
// token widths (100/200/300/128/127 exercise fixint/uint8/uint16), group_idx=3,
// block_size=2, ts=0.0. Captured from msgspec 0.21.1 (len 170).
TEST_CASE("kv_events: msgpack payload is byte-exact (int hash form + parent)") {
  KVEventBatch batch;
  batch.ts = 0.0;

  BlockStored stored;
  stored.block_hashes = {ExternalBlockHash{uint64_t{0x0102030405060708ULL}},
                         ExternalBlockHash{uint64_t{0xF0F1F2F3F4F5F6F7ULL}}};
  stored.parent_block_hash = ExternalBlockHash{uint64_t{0xABULL}};
  stored.token_ids = {100, 200, 300, 128, 127};
  stored.block_size = 2;
  stored.lora_id = std::nullopt;
  stored.medium = "GPU";
  stored.lora_name = std::nullopt;
  stored.extra_keys = std::vector<vllm::distributed::EventExtraKey>{
      std::nullopt, std::nullopt};
  stored.group_idx = 3;

  batch.events = {stored};

  const std::string golden = Bytes(
      {0x93, 0xcb, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x91, 0x8a,
       0xa4, 0x74, 0x79, 0x70, 0x65, 0xab, 0x42, 0x6c, 0x6f, 0x63, 0x6b, 0x53,
       0x74, 0x6f, 0x72, 0x65, 0x64, 0xac, 0x62, 0x6c, 0x6f, 0x63, 0x6b, 0x5f,
       0x68, 0x61, 0x73, 0x68, 0x65, 0x73, 0x92, 0xcf, 0x01, 0x02, 0x03, 0x04,
       0x05, 0x06, 0x07, 0x08, 0xcf, 0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6,
       0xf7, 0xb1, 0x70, 0x61, 0x72, 0x65, 0x6e, 0x74, 0x5f, 0x62, 0x6c, 0x6f,
       0x63, 0x6b, 0x5f, 0x68, 0x61, 0x73, 0x68, 0xcc, 0xab, 0xa9, 0x74, 0x6f,
       0x6b, 0x65, 0x6e, 0x5f, 0x69, 0x64, 0x73, 0x95, 0x64, 0xcc, 0xc8, 0xcd,
       0x01, 0x2c, 0xcc, 0x80, 0x7f, 0xaa, 0x62, 0x6c, 0x6f, 0x63, 0x6b, 0x5f,
       0x73, 0x69, 0x7a, 0x65, 0x02, 0xa7, 0x6c, 0x6f, 0x72, 0x61, 0x5f, 0x69,
       0x64, 0xc0, 0xa6, 0x6d, 0x65, 0x64, 0x69, 0x75, 0x6d, 0xa3, 0x47, 0x50,
       0x55, 0xa9, 0x6c, 0x6f, 0x72, 0x61, 0x5f, 0x6e, 0x61, 0x6d, 0x65, 0xc0,
       0xaa, 0x65, 0x78, 0x74, 0x72, 0x61, 0x5f, 0x6b, 0x65, 0x79, 0x73, 0x92,
       0xc0, 0xc0, 0xa9, 0x67, 0x72, 0x6f, 0x75, 0x70, 0x5f, 0x69, 0x64, 0x78,
       0x03, 0xc0});

  const std::string got = encode_kv_event_batch(batch);
  CHECK(got.size() == golden.size());
  CHECK(ToHex(got) == ToHex(golden));
}

// ---------------------------------------------------------------------------
// Publisher seam.
// ---------------------------------------------------------------------------

TEST_CASE("kv_events: publisher seam (null / collecting / factory)") {
  // NullEventPublisher: no-op, never throws.
  NullEventPublisher null_pub;
  KVEventBatch batch;
  batch.ts = 1.0;
  batch.events = {AllBlocksCleared{}};
  null_pub.publish(batch);
  null_pub.shutdown();

  // CollectingEventPublisher: keeps batches + encoded bytes, annotates dp_rank.
  CollectingEventPublisher pub(/*data_parallel_rank=*/2);
  pub.publish(batch);
  REQUIRE(pub.batches().size() == 1);
  CHECK(pub.batches()[0].data_parallel_rank.has_value());
  CHECK(pub.batches()[0].data_parallel_rank.value() == 2);
  // The encoded bytes match the encoder run over the annotated batch.
  KVEventBatch annotated = batch;
  annotated.data_parallel_rank = 2;
  CHECK(pub.encoded()[0] == encode_kv_event_batch(annotated));

  // Factory: Null when disabled or publisher == "null".
  KVEventsConfig off;
  off.enable_kv_cache_events = false;
  CHECK(EventPublisherFactory::create(&off) != nullptr);
  CHECK(EventPublisherFactory::create(nullptr) != nullptr);
  KVEventsConfig null_cfg;
  null_cfg.enable_kv_cache_events = true;
  null_cfg.publisher = "null";
  CHECK(EventPublisherFactory::create(&null_cfg) != nullptr);
  // zmq transport is DEFERRED -> loud throw (never silently downgraded).
  KVEventsConfig zmq_cfg;
  zmq_cfg.enable_kv_cache_events = true;
  zmq_cfg.publisher = "zmq";
  CHECK_THROWS_AS(EventPublisherFactory::create(&zmq_cfg), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Layer 2: emission SEQUENCE through a real BlockPool.
// ---------------------------------------------------------------------------

TEST_CASE("kv_events: store / reuse / evict / reset emit the right sequence") {
  init_none_hash(sha256_cbor, "seed42");
  const int block_size = 4;
  BlockPool pool(/*num_gpu_blocks=*/6, /*enable_caching=*/true, block_size,
                 /*enable_kv_cache_events=*/true);

  Request req = MakeRequest("0", Iota(12), block_size);  // 3 full blocks
  REQUIRE(req.block_hashes.size() == 3);

  // --- STORE: cache 3 full blocks from the start -> 1 BlockStored -----------
  auto blocks = pool.get_new_blocks(3);
  pool.cache_full_blocks(req, blocks, /*num_cached=*/0, /*num_full=*/3,
                         block_size, /*group=*/0);
  auto ev = pool.take_events();
  REQUIRE(ev.size() == 1);
  REQUIRE(std::holds_alternative<BlockStored>(ev[0]));
  {
    const BlockStored& s = std::get<BlockStored>(ev[0]);
    REQUIRE(s.block_hashes.size() == 3);
    for (int i = 0; i < 3; ++i) {
      CHECK(s.block_hashes[i] == maybe_convert_block_hash(req.block_hashes[i]));
    }
    CHECK_FALSE(s.parent_block_hash.has_value());  // num_cached == 0
    CHECK(s.token_ids == std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                              11});
    CHECK(s.block_size == 4);
    REQUIRE(s.group_idx.has_value());
    CHECK(s.group_idx.value() == 0);
    REQUIRE(s.medium.has_value());
    CHECK(s.medium.value() == "GPU");
    CHECK_FALSE(s.lora_id.has_value());
    CHECK_FALSE(s.lora_name.has_value());
    // extra_keys present (text path -> a list of None), one entry per block.
    REQUIRE(s.extra_keys.has_value());
    CHECK(s.extra_keys->size() == 3);
    for (const auto& e : *s.extra_keys) CHECK_FALSE(e.has_value());
  }
  CHECK(pool.take_events().empty());  // queue drained by take_events()

  // --- REUSE (default incremental mode): touch emits NOTHING ----------------
  // A prefix hit from another request touches the blocks (ref_cnt 1 -> 2).
  pool.touch(blocks);
  CHECK(pool.take_events().empty());
  // Release that extra ref again (ref_cnt 2 -> 1); free_blocks emits nothing and
  // no block reaches 0, so nothing is evicted.
  pool.free_blocks(blocks);
  CHECK(pool.take_events().empty());

  // --- REUSE-EMIT (report_mode == "full" path): emit_cached_block_events ----
  pool.emit_cached_block_events(req, /*num_cached_blocks=*/3, block_size,
                               /*group=*/0);
  ev = pool.take_events();
  REQUIRE(ev.size() == 1);
  REQUIRE(std::holds_alternative<BlockStored>(ev[0]));
  {
    const BlockStored& s = std::get<BlockStored>(ev[0]);
    CHECK(s.block_hashes.size() == 3);
    CHECK_FALSE(s.parent_block_hash.has_value());
    CHECK(s.token_ids.size() == 12);
  }

  // --- EVICT: drop one cached block -> exactly 1 BlockRemoved ---------------
  pool.evict_blocks({blocks[0]->block_id});
  ev = pool.take_events();
  REQUIRE(ev.size() == 1);
  REQUIRE(std::holds_alternative<BlockRemoved>(ev[0]));
  {
    const BlockRemoved& r = std::get<BlockRemoved>(ev[0]);
    REQUIRE(r.block_hashes.size() == 1);
    CHECK(r.block_hashes[0] == maybe_convert_block_hash(req.block_hashes[0]));
    REQUIRE(r.medium.has_value());
    CHECK(r.medium.value() == "GPU");
    REQUIRE(r.group_idx.has_value());
    CHECK(r.group_idx.value() == 0);
  }

  // --- RESET: free everything, then reset -> exactly 1 AllBlocksCleared -----
  pool.free_blocks({blocks[2], blocks[1], blocks[0]});
  CHECK(pool.take_events().empty());  // free_blocks itself emits nothing
  REQUIRE(pool.reset_prefix_cache());
  ev = pool.take_events();
  REQUIRE(ev.size() == 1);
  CHECK(std::holds_alternative<AllBlocksCleared>(ev[0]));
}

// Parent linkage + token range for a partial store (num_cached_blocks > 0).
TEST_CASE("kv_events: partial store carries parent hash + shifted token range") {
  init_none_hash(sha256_cbor, "seed42");
  const int block_size = 4;
  BlockPool pool(6, true, block_size, /*enable_kv_cache_events=*/true);

  Request req = MakeRequest("0", Iota(12), block_size);
  REQUIRE(req.block_hashes.size() == 3);

  auto blocks = pool.get_new_blocks(3);
  // Cache block 0 first (drains its own store event).
  pool.cache_full_blocks(req, blocks, 0, 1, block_size, 0);
  (void)pool.take_events();

  // Now cache blocks 1..2 with num_cached_blocks == 1: parent == hash[0].
  pool.cache_full_blocks(req, blocks, /*num_cached=*/1, /*num_full=*/3,
                         block_size, 0);
  auto ev = pool.take_events();
  REQUIRE(ev.size() == 1);
  const BlockStored& s = std::get<BlockStored>(ev[0]);
  CHECK(s.block_hashes.size() == 2);  // blocks 1 and 2
  REQUIRE(s.parent_block_hash.has_value());
  CHECK(s.parent_block_hash.value() == maybe_convert_block_hash(req.block_hashes[0]));
  // token range [1*4, 3*4) == tokens 4..11.
  CHECK(s.token_ids == std::vector<int64_t>{4, 5, 6, 7, 8, 9, 10, 11});
}

// The DEFAULT (events-disabled) path emits nothing -> prefix cache behaviour is
// byte-identical to the pre-KV-EVENTS pool.
TEST_CASE("kv_events: events-disabled path emits nothing") {
  init_none_hash(sha256_cbor, "seed42");
  const int block_size = 4;
  BlockPool pool(6, true, block_size, /*enable_kv_cache_events=*/false);

  Request req = MakeRequest("0", Iota(12), block_size);
  auto blocks = pool.get_new_blocks(3);
  pool.cache_full_blocks(req, blocks, 0, 3, block_size, 0);
  pool.emit_cached_block_events(req, 3, block_size, 0);
  pool.evict_blocks({blocks[0]->block_id});
  CHECK(pool.take_events().empty());

  pool.free_blocks({blocks[2], blocks[1], blocks[0]});
  CHECK(pool.reset_prefix_cache());
  CHECK(pool.take_events().empty());
}
