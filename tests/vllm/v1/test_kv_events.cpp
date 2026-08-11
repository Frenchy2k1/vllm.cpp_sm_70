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
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "vllm/config/scheduler.h"
#include "vllm/distributed/kv_events.h"
#include "vllm/sampling_params.h"
#include "vllm/v1/core/block_pool.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/engine/types.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/request.h"
#include "vt/dtype.h"

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

// ---------------------------------------------------------------------------
// Layer 3 (W3, issue #352): the ENGINE/SCHEDULER wiring of the batch envelope,
// and the per-request kv_cache_report_mode == "full" reuse path.
//
// Upstream has NO test that drives either: `grep -rn kv_cache_report_mode
// --include="*.py"` over the pin returns three SOURCE sites and no test, and the
// event coverage in tests/v1/core/test_prefix_caching.py calls the pool /
// manager directly rather than stepping a scheduler. These cases are therefore
// written against the upstream SOURCE anchors, recorded as written-from-scratch:
//   scheduler.py:86,116-119,155-158  ctor: config -> enable flag + publisher
//   scheduler.py:1901-1915           update_from_output: take_events + publish
//   scheduler.py:2456-2459           shutdown -> publisher.shutdown()
//   request.py:116-127               kv_cache_report_mode from extra_args
//   kv_cache_manager.py:262-280      report_mode == "full" reuse emission
// ---------------------------------------------------------------------------

namespace {

constexpr int kSchedBlockSize = 16;

// Mirror of test_scheduler.cpp's CreateScheduler, plus the KVEventsConfig the
// upstream ctor reads out of vllm_config (scheduler.py:86).
std::unique_ptr<vllm::v1::Scheduler> CreateEventScheduler(
    const KVEventsConfig* kv_events_config, int data_parallel_rank = 0,
    int num_blocks = 512) {
  // NOTE: every enabled config below sets publisher = "null" EXPLICITLY. Our
  // KVEventsConfig has no PostInit, so an unset publisher reaches the factory as
  // "" and throws "unknown event publisher ''" instead of resolving the way
  // kv_events.py:50-52 does. That is issue #353, deliberately not fixed inside
  // KV-EVENTS W3.
  vllm::SchedulerConfig cfg;
  cfg.max_num_seqs = 16;
  cfg.max_num_batched_tokens = 8192;
  cfg.enable_chunked_prefill = true;
  cfg.max_model_len = 8192;
  cfg.watermark = 0.0;

  vllm::v1::KVCacheConfig kv_cfg;
  kv_cfg.num_blocks = num_blocks;
  kv_cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<vllm::v1::FullAttentionSpec>(
          kSchedBlockSize, /*num_kv_heads=*/1, /*head_size=*/1, vt::DType::kF32));

  return std::make_unique<vllm::v1::Scheduler>(
      cfg, kv_cfg, kSchedBlockSize, /*enable_caching=*/true,
      /*structured_output_manager=*/nullptr,
      /*speculative_config=*/std::nullopt, kv_events_config,
      data_parallel_rank);
}

// A request whose prompt is `num_tokens` copies of `fill` (so two requests
// built with the same fill share a prefix, and different fills never do).
std::unique_ptr<Request> MakeSchedRequest(
    const std::string& id, int num_tokens, int32_t fill,
    const std::optional<std::string>& report_mode = std::nullopt) {
  init_none_hash(sha256_cbor);
  vllm::SamplingParams params;
  params.max_tokens = 16;
  if (report_mode.has_value()) {
    params.extra_args = std::map<std::string, std::string>{
        {"kv_cache_report_mode", *report_mode}};
  }
  return std::make_unique<Request>(
      id, std::vector<int32_t>(static_cast<std::size_t>(num_tokens), fill),
      params, /*arrival_time=*/0.0,
      get_request_block_hasher(kSchedBlockSize, sha256_cbor));
}

vllm::v1::ModelRunnerOutput RunnerOutput(
    const std::vector<std::pair<std::string, std::vector<int32_t>>>& per_req) {
  vllm::v1::ModelRunnerOutput mro;
  int idx = 0;
  for (const auto& [id, toks] : per_req) {
    mro.req_ids.push_back(id);
    mro.req_id_to_index[id] = idx++;
    mro.sampled_token_ids.push_back(toks);
  }
  return mro;
}

// Drive one full engine step for `ids` and return the scheduler output.
void Step(vllm::v1::Scheduler& sched, const std::vector<std::string>& ids) {
  auto out = sched.schedule();
  std::vector<std::pair<std::string, std::vector<int32_t>>> per_req;
  for (const auto& id : ids) per_req.emplace_back(id, std::vector<int32_t>{7});
  sched.update_from_output(out, RunnerOutput(per_req));
}

// Every BlockStored in every published batch, flattened in publish order.
std::vector<BlockStored> AllStored(const CollectingEventPublisher& pub) {
  std::vector<BlockStored> out;
  for (const auto& batch : pub.batches()) {
    for (const auto& ev : batch.events) {
      if (const auto* s = std::get_if<BlockStored>(&ev)) out.push_back(*s);
    }
  }
  return out;
}

}  // namespace

// scheduler.py:1901-1915 — a step that raised events publishes exactly ONE
// KVEventBatch, carrying the pool's events, annotated with the publisher's
// data_parallel_rank, stamped with a wall-clock ts, and byte-identical to the
// encoder's output for that batch.
TEST_CASE("kv_events: a scheduler step publishes the KVEventBatch envelope") {
  KVEventsConfig cfg;
  cfg.enable_kv_cache_events = true;
  cfg.publisher = "null";
  auto sched = CreateEventScheduler(&cfg, /*data_parallel_rank=*/3);

  auto* pub = new CollectingEventPublisher(/*data_parallel_rank=*/3);
  sched->set_kv_event_publisher(
      std::unique_ptr<vllm::distributed::EventPublisher>(pub));

  auto req = MakeSchedRequest("A", /*num_tokens=*/32, /*fill=*/1);
  Request* raw = req.get();
  sched->add_request(std::move(req));

  // schedule() alone must NOT publish: upstream drains in update_from_output.
  auto out = sched->schedule();
  CHECK(pub->batches().empty());

  sched->update_from_output(out, RunnerOutput({{"A", {7}}}));

  REQUIRE(pub->batches().size() == 1);
  const KVEventBatch& batch = pub->batches()[0];
  // ts is time.time(): wall-clock seconds since the epoch, not a steady clock.
  CHECK(batch.ts > 1.7e9);
  REQUIRE(batch.data_parallel_rank.has_value());
  CHECK(batch.data_parallel_rank.value() == 3);

  REQUIRE(batch.events.size() == 1);
  const BlockStored* stored = std::get_if<BlockStored>(&batch.events[0]);
  REQUIRE(stored != nullptr);
  CHECK(stored->block_size == kSchedBlockSize);
  REQUIRE(stored->medium.has_value());
  CHECK(stored->medium.value() == "GPU");
  CHECK_FALSE(stored->parent_block_hash.has_value());
  // 32 prompt tokens / 16 = the two full blocks cached by allocate_slots.
  REQUIRE(stored->block_hashes.size() == 2);
  for (std::size_t i = 0; i < stored->block_hashes.size(); ++i) {
    CHECK(stored->block_hashes[i] ==
          maybe_convert_block_hash(raw->block_hashes[i]));
  }

  // The wire bytes the (deferred) transport would ship are the encoder's.
  REQUIRE(pub->encoded().size() == 1);
  CHECK(pub->encoded()[0] == encode_kv_event_batch(batch));
}

// scheduler.py:1913 `if events:` — a step that raised nothing publishes nothing
// (no empty envelopes on the wire).
TEST_CASE("kv_events: a step with no events publishes no batch") {
  KVEventsConfig cfg;
  cfg.enable_kv_cache_events = true;
  cfg.publisher = "null";  // see CreateEventScheduler / issue #353
  auto sched = CreateEventScheduler(&cfg);
  auto* pub = new CollectingEventPublisher();
  sched->set_kv_event_publisher(
      std::unique_ptr<vllm::distributed::EventPublisher>(pub));

  sched->add_request(MakeSchedRequest("A", /*num_tokens=*/32, /*fill=*/1));
  Step(*sched, {"A"});
  REQUIRE(pub->batches().size() == 1);

  // A decode step appends one token: no block fills, so nothing is emitted.
  Step(*sched, {"A"});
  CHECK(pub->batches().size() == 1);
}

// scheduler.py:2456-2459 — shutdown reaches the publisher.
TEST_CASE("kv_events: Scheduler::shutdown shuts the publisher down") {
  KVEventsConfig cfg;
  cfg.enable_kv_cache_events = true;
  cfg.publisher = "null";  // see CreateEventScheduler / issue #353
  auto sched = CreateEventScheduler(&cfg);
  auto* pub = new CollectingEventPublisher();
  sched->set_kv_event_publisher(
      std::unique_ptr<vllm::distributed::EventPublisher>(pub));

  CHECK_FALSE(pub->was_shut_down());
  sched->shutdown();
  CHECK(pub->was_shut_down());
}

// request.py:116-127 — kv_cache_report_mode comes from
// sampling_params.extra_args and defaults to "incremental".
TEST_CASE("kv_events: Request::kv_cache_report_mode mirrors extra_args") {
  init_none_hash(sha256_cbor);
  vllm::SamplingParams bare;
  bare.max_tokens = 4;
  Request no_args("r0", {1, 2, 3}, bare, 0.0);
  CHECK(no_args.kv_cache_report_mode == "incremental");

  vllm::SamplingParams other;
  other.max_tokens = 4;
  other.extra_args =
      std::map<std::string, std::string>{{"something_else", "x"}};
  Request other_key("r1", {1, 2, 3}, other, 0.0);
  CHECK(other_key.kv_cache_report_mode == "incremental");

  vllm::SamplingParams full;
  full.max_tokens = 4;
  full.extra_args =
      std::map<std::string, std::string>{{"kv_cache_report_mode", "full"}};
  Request full_req("r2", {1, 2, 3}, full, 0.0);
  CHECK(full_req.kv_cache_report_mode == "full");
}

// kv_cache_manager.py:262-280 — under report_mode == "full" a prefix-cache HIT
// re-reports the REUSED blocks as BlockStored; under the default "incremental"
// it does not.
TEST_CASE("kv_events: report_mode==full re-reports reused prefix blocks") {
  KVEventsConfig cfg;
  cfg.enable_kv_cache_events = true;
  cfg.publisher = "null";  // see CreateEventScheduler / issue #353

  // Prime the cache with a 48-token prompt, then finish the request so its
  // blocks are free but still in the prefix cache.
  auto prime = [&](vllm::v1::Scheduler& sched) {
    sched.add_request(MakeSchedRequest("A", /*num_tokens=*/48, /*fill=*/5));
    Step(sched, {"A"});
    sched.finish_requests("A", vllm::v1::RequestStatus::kFinishedAborted);
  };

  std::vector<ExternalBlockHash> reused_hashes;
  {
    // report_mode == "full": the hit is re-reported.
    auto sched = CreateEventScheduler(&cfg);
    auto* pub = new CollectingEventPublisher();
    sched->set_kv_event_publisher(
        std::unique_ptr<vllm::distributed::EventPublisher>(pub));
    prime(*sched);
    const std::size_t after_prime = AllStored(*pub).size();
    REQUIRE(after_prime == 1);

    auto b = MakeSchedRequest("B", /*num_tokens=*/48, /*fill=*/5, "full");
    Request* raw_b = b.get();
    CHECK(raw_b->kv_cache_report_mode == "full");
    sched->add_request(std::move(b));
    Step(*sched, {"B"});

    auto stored = AllStored(*pub);
    // The prime's store, then TWO events for B in one batch: the REUSE report
    // raised by get_computed_blocks during schedule(), followed by the ordinary
    // store of the one block B actually had to compute. Both are drained by the
    // same end-of-step take_events, in that order.
    REQUIRE(stored.size() == after_prime + 2);

    const BlockStored& reuse = stored[after_prime];
    // max_cache_hit_length is num_tokens - 1 == 47, so 2 of the 3 blocks hit.
    REQUIRE(reuse.block_hashes.size() == 2);
    for (std::size_t i = 0; i < reuse.block_hashes.size(); ++i) {
      CHECK(reuse.block_hashes[i] ==
            maybe_convert_block_hash(raw_b->block_hashes[i]));
    }
    CHECK(reuse.block_size == kSchedBlockSize);
    REQUIRE(reuse.group_idx.has_value());
    CHECK(reuse.group_idx.value() == 0);

    // The follow-on store covers only the third block, parented on the last
    // REUSED one -- so a consumer can chain the whole prefix.
    const BlockStored& tail = stored[after_prime + 1];
    REQUIRE(tail.block_hashes.size() == 1);
    CHECK(tail.block_hashes[0] == maybe_convert_block_hash(raw_b->block_hashes[2]));
    REQUIRE(tail.parent_block_hash.has_value());
    CHECK(tail.parent_block_hash.value() ==
          maybe_convert_block_hash(raw_b->block_hashes[1]));

    reused_hashes = reuse.block_hashes;
  }

  {
    // Default "incremental": the identical hit reports NOTHING extra.
    auto sched = CreateEventScheduler(&cfg);
    auto* pub = new CollectingEventPublisher();
    sched->set_kv_event_publisher(
        std::unique_ptr<vllm::distributed::EventPublisher>(pub));
    prime(*sched);
    const std::size_t after_prime = AllStored(*pub).size();

    auto c = MakeSchedRequest("C", /*num_tokens=*/48, /*fill=*/5);
    CHECK(c->kv_cache_report_mode == "incremental");
    sched->add_request(std::move(c));
    Step(*sched, {"C"});

    auto stored = AllStored(*pub);
    // Only the ordinary store of C's third block -- ONE event where "full"
    // produced two. The reused prefix is never re-reported.
    REQUIRE(stored.size() == after_prime + 1);
    CHECK(stored[after_prime].block_hashes.size() == 1);
    for (const BlockStored& s : stored) {
      CHECK(s.block_hashes != reused_hashes);
    }
  }
}

// The DEFAULT scheduler (no KVEventsConfig, exactly what every existing call
// site constructs) is INERT: nothing is ever published, even with a publisher
// injected and a request explicitly asking for report_mode == "full".
TEST_CASE("kv_events: the default scheduler path publishes nothing") {
  auto sched = CreateEventScheduler(/*kv_events_config=*/nullptr);
  auto* pub = new CollectingEventPublisher();
  sched->set_kv_event_publisher(
      std::unique_ptr<vllm::distributed::EventPublisher>(pub));

  sched->add_request(MakeSchedRequest("A", /*num_tokens=*/48, /*fill=*/9));
  Step(*sched, {"A"});
  sched->finish_requests("A", vllm::v1::RequestStatus::kFinishedAborted);

  sched->add_request(MakeSchedRequest("B", /*num_tokens=*/48, /*fill=*/9, "full"));
  Step(*sched, {"B"});

  CHECK(pub->batches().empty());
  CHECK(pub->encoded().empty());
}
