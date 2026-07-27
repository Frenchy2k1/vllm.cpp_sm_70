// Ported from: vllm/distributed/kv_events.py @ 555967922 (vLLM 0.26.0.dev0)
// See include/vllm/distributed/kv_events.h for scope, encoding fidelity, and the
// recorded ZMQ-transport deferral.
//
// The msgpack writer below reproduces msgspec.msgpack.Encoder()'s output for the
// upstream struct definitions BYTE-FOR-BYTE (verified against msgspec 0.21.1;
// see tests/vllm/v1/test_kv_events.cpp for the captured golden vectors).
#include "vllm/distributed/kv_events.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace vllm::distributed {
namespace {

// --- Minimal-width msgpack primitives (matching msgspec's encoder) ----------

void put_byte(std::string& out, uint8_t b) {
  out.push_back(static_cast<char>(b));
}

// Big-endian append of the low `n` bytes of `v`.
void put_be(std::string& out, uint64_t v, int n) {
  for (int i = n - 1; i >= 0; --i) {
    put_byte(out, static_cast<uint8_t>((v >> (8 * i)) & 0xff));
  }
}

// Non-negative integer, minimal msgpack width (msgspec encodes all the ints that
// occur here — token ids, sizes, group ids, int block hashes, ts rank — as
// unsigned when they are >= 0).
void put_uint(std::string& out, uint64_t v) {
  if (v <= 0x7f) {
    put_byte(out, static_cast<uint8_t>(v));  // positive fixint
  } else if (v <= 0xff) {
    put_byte(out, 0xcc);
    put_be(out, v, 1);
  } else if (v <= 0xffff) {
    put_byte(out, 0xcd);
    put_be(out, v, 2);
  } else if (v <= 0xffffffffULL) {
    put_byte(out, 0xce);
    put_be(out, v, 4);
  } else {
    put_byte(out, 0xcf);
    put_be(out, v, 8);
  }
}

// Signed integer, minimal width. Non-negative values route through put_uint to
// match msgspec (which prefers the unsigned encoding for >= 0).
void put_int(std::string& out, int64_t v) {
  if (v >= 0) {
    put_uint(out, static_cast<uint64_t>(v));
    return;
  }
  if (v >= -32) {
    put_byte(out, static_cast<uint8_t>(v));  // negative fixint
  } else if (v >= -128) {
    put_byte(out, 0xd0);
    put_be(out, static_cast<uint64_t>(v), 1);
  } else if (v >= -32768) {
    put_byte(out, 0xd1);
    put_be(out, static_cast<uint64_t>(v), 2);
  } else if (v >= -2147483648LL) {
    put_byte(out, 0xd2);
    put_be(out, static_cast<uint64_t>(v), 4);
  } else {
    put_byte(out, 0xd3);
    put_be(out, static_cast<uint64_t>(v), 8);
  }
}

void put_float64(std::string& out, double d) {
  uint64_t bits;
  static_assert(sizeof(bits) == sizeof(d), "double must be 8 bytes");
  std::memcpy(&bits, &d, sizeof(bits));
  put_byte(out, 0xcb);
  put_be(out, bits, 8);
}

void put_str(std::string& out, const std::string& s) {
  const size_t n = s.size();
  if (n <= 31) {
    put_byte(out, static_cast<uint8_t>(0xa0 | n));  // fixstr
  } else if (n <= 0xff) {
    put_byte(out, 0xd9);
    put_be(out, n, 1);
  } else if (n <= 0xffff) {
    put_byte(out, 0xda);
    put_be(out, n, 2);
  } else {
    put_byte(out, 0xdb);
    put_be(out, n, 4);
  }
  out.append(s);
}

void put_bin(std::string& out, const std::string& s) {
  const size_t n = s.size();
  if (n <= 0xff) {
    put_byte(out, 0xc4);
    put_be(out, n, 1);
  } else if (n <= 0xffff) {
    put_byte(out, 0xc5);
    put_be(out, n, 2);
  } else {
    put_byte(out, 0xc6);
    put_be(out, n, 4);
  }
  out.append(s);
}

void put_nil(std::string& out) { put_byte(out, 0xc0); }

void put_array_header(std::string& out, size_t n) {
  if (n <= 15) {
    put_byte(out, static_cast<uint8_t>(0x90 | n));  // fixarray
  } else if (n <= 0xffff) {
    put_byte(out, 0xdc);
    put_be(out, n, 2);
  } else {
    put_byte(out, 0xdd);
    put_be(out, n, 4);
  }
}

void put_map_header(std::string& out, size_t n) {
  if (n <= 15) {
    put_byte(out, static_cast<uint8_t>(0x80 | n));  // fixmap
  } else if (n <= 0xffff) {
    put_byte(out, 0xde);
    put_be(out, n, 2);
  } else {
    put_byte(out, 0xdf);
    put_be(out, n, 4);
  }
}

// --- Value encoders ---------------------------------------------------------

void put_external_block_hash(std::string& out,
                             const vllm::v1::ExternalBlockHash& h) {
  if (std::holds_alternative<std::string>(h)) {
    put_bin(out, std::get<std::string>(h));  // raw sha256 bytes
  } else {
    put_uint(out, std::get<uint64_t>(h));  // int-truncated low 64 bits
  }
}

void put_block_hashes(std::string& out,
                      const std::vector<vllm::v1::ExternalBlockHash>& hs) {
  put_array_header(out, hs.size());
  for (const auto& h : hs) put_external_block_hash(out, h);
}

// One BlockStored.extra_keys entry: None -> nil. A present tuple (MM/LoRA/salt)
// is DEFERRED with generate_block_hash_extra_keys, so it is unreachable on the
// ported path; encode it loudly rather than silently wrong.
void put_extra_key_entry(std::string& out, const EventExtraKey& entry) {
  if (!entry.has_value()) {
    put_nil(out);
    return;
  }
  throw std::runtime_error(
      "encode_kv_event_batch: MM/LoRA/salt extra-key tuple payload not yet "
      "ported (generate_block_hash_extra_keys returns None on the text path)");
}

void put_str_field_key(std::string& out, const char* key) {
  put_str(out, std::string(key));
}

// A required optional<string> field: ALWAYS emitted (nil when nullopt).
void put_opt_str_value(std::string& out, const std::optional<std::string>& v) {
  if (v.has_value()) {
    put_str(out, *v);
  } else {
    put_nil(out);
  }
}

// Count the map size (tag + always-emitted required fields + present defaulted
// fields), then emit "type" first followed by the fields in definition order.
void put_block_stored(std::string& out, const BlockStored& e) {
  size_t n = 1 /*type*/ + 7 /*block_hashes,parent,token_ids,block_size,lora_id,
                                medium,lora_name*/;
  if (e.extra_keys.has_value()) ++n;
  if (e.group_idx.has_value()) ++n;
  if (e.kv_cache_spec_kind.has_value()) ++n;
  if (e.kv_cache_spec_sliding_window.has_value()) ++n;
  if (e.locality.has_value()) ++n;
  put_map_header(out, n);

  put_str_field_key(out, "type");
  put_str(out, std::string(BlockStored::kTag));

  put_str_field_key(out, "block_hashes");
  put_block_hashes(out, e.block_hashes);

  put_str_field_key(out, "parent_block_hash");
  if (e.parent_block_hash.has_value()) {
    put_external_block_hash(out, *e.parent_block_hash);
  } else {
    put_nil(out);
  }

  put_str_field_key(out, "token_ids");
  put_array_header(out, e.token_ids.size());
  for (int64_t t : e.token_ids) put_int(out, t);

  put_str_field_key(out, "block_size");
  put_int(out, e.block_size);

  put_str_field_key(out, "lora_id");
  if (e.lora_id.has_value()) {
    put_int(out, *e.lora_id);
  } else {
    put_nil(out);
  }

  put_str_field_key(out, "medium");
  put_opt_str_value(out, e.medium);

  put_str_field_key(out, "lora_name");
  put_opt_str_value(out, e.lora_name);

  if (e.extra_keys.has_value()) {
    put_str_field_key(out, "extra_keys");
    put_array_header(out, e.extra_keys->size());
    for (const auto& entry : *e.extra_keys) put_extra_key_entry(out, entry);
  }
  if (e.group_idx.has_value()) {
    put_str_field_key(out, "group_idx");
    put_int(out, *e.group_idx);
  }
  if (e.kv_cache_spec_kind.has_value()) {
    put_str_field_key(out, "kv_cache_spec_kind");
    put_str(out, *e.kv_cache_spec_kind);
  }
  if (e.kv_cache_spec_sliding_window.has_value()) {
    put_str_field_key(out, "kv_cache_spec_sliding_window");
    put_int(out, *e.kv_cache_spec_sliding_window);
  }
  if (e.locality.has_value()) {
    put_str_field_key(out, "locality");
    put_str(out, *e.locality);
  }
}

void put_block_removed(std::string& out, const BlockRemoved& e) {
  size_t n = 1 /*type*/ + 2 /*block_hashes, medium*/;
  if (e.group_idx.has_value()) ++n;
  if (e.locality.has_value()) ++n;
  put_map_header(out, n);

  put_str_field_key(out, "type");
  put_str(out, std::string(BlockRemoved::kTag));

  put_str_field_key(out, "block_hashes");
  put_block_hashes(out, e.block_hashes);

  put_str_field_key(out, "medium");
  put_opt_str_value(out, e.medium);

  if (e.group_idx.has_value()) {
    put_str_field_key(out, "group_idx");
    put_int(out, *e.group_idx);
  }
  if (e.locality.has_value()) {
    put_str_field_key(out, "locality");
    put_str(out, *e.locality);
  }
}

void put_all_blocks_cleared(std::string& out, const AllBlocksCleared&) {
  put_map_header(out, 1);
  put_str_field_key(out, "type");
  put_str(out, std::string(AllBlocksCleared::kTag));
}

void put_event(std::string& out, const KVCacheEvent& ev) {
  std::visit(
      [&out](const auto& e) {
        using T = std::decay_t<decltype(e)>;
        if constexpr (std::is_same_v<T, BlockStored>) {
          put_block_stored(out, e);
        } else if constexpr (std::is_same_v<T, BlockRemoved>) {
          put_block_removed(out, e);
        } else {
          put_all_blocks_cleared(out, e);
        }
      },
      ev);
}

}  // namespace

std::string encode_kv_event_batch(const KVEventBatch& batch) {
  std::string out;
  // EventBatch is array_like: [ts, events, data_parallel_rank]. msgspec keeps
  // the trailing data_parallel_rank even when None (verified), so length 3.
  put_array_header(out, 3);
  put_float64(out, batch.ts);
  put_array_header(out, batch.events.size());
  for (const auto& ev : batch.events) put_event(out, ev);
  if (batch.data_parallel_rank.has_value()) {
    put_int(out, *batch.data_parallel_rank);
  } else {
    put_nil(out);
  }
  return out;
}

std::unique_ptr<EventPublisher> EventPublisherFactory::create(
    const KVEventsConfig* config, int data_parallel_rank) {
  // kv_events.py:523-543: Null when disabled or publisher == "null".
  if (config == nullptr || !config->enable_kv_cache_events ||
      config->publisher == "null") {
    return std::make_unique<NullEventPublisher>();
  }
  // data_parallel_rank is consumed by the live ZMQ publisher (DP port offset +
  // event annotation); that transport is DEFERRED, so it is unused here.
  (void)data_parallel_rank;
  if (config->publisher == "zmq") {
    // DEFERRED: the live ZMQ PUB/ROUTER transport is not ported (see header).
    // Fail loudly so a request for it is never silently downgraded.
    throw std::runtime_error(
        "EventPublisherFactory: the 'zmq' publisher (live socket transport) is "
        "not yet ported; event generation + payload are gated, the ZMQ "
        "transport is deferred (KV-EVENTS spec).");
  }
  throw std::runtime_error("EventPublisherFactory: unknown event publisher '" +
                           config->publisher + "'");
}

}  // namespace vllm::distributed
