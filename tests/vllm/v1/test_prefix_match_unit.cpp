// Tests for resolve_kv_cache_block_sizes / the `--prefix-match-unit`
// (prefix_match_unit -> hash_block_size) semantics.
//
// Ported semantics from vllm/v1/core/kv_cache_utils.py:626-688
// (resolve_kv_cache_block_sizes) @ 555967922. Upstream has no direct unit test
// for the resolver (it is exercised indirectly by
// tests/v1/core/prefix_cache/test_partial_prefix_cache_primitives.py, which
// needs the deferred BlockHashListWithBlockSize fine-grained path). These are
// hand cases that pin the exact resolution — including the `prefix_match_unit=16`
// override vs the default gcd behaviour (RED-first) — plus a hasher-granularity
// case showing the resolved hash_block_size changes the matching unit.
//
// Row: KV-PREFIX-MATCH-UNIT. Spec: .agents/specs/prefix-match-unit.md.

#include <doctest/doctest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

using vllm::v1::AttentionSpec;
using vllm::v1::FullAttentionSpec;
using vllm::v1::hash_request_tokens;
using vllm::v1::init_none_hash;
using vllm::v1::KVCacheConfig;
using vllm::v1::MambaSpec;
using vllm::v1::resolve_kv_cache_block_sizes;
using vllm::v1::sha256_cbor;
using vt::DType;

namespace {

std::shared_ptr<FullAttentionSpec> MakeFull(int block_size) {
  return std::make_shared<FullAttentionSpec>(block_size, /*num_kv_heads=*/1,
                                             /*head_size=*/1, DType::kF32);
}

std::shared_ptr<MambaSpec> MakeMamba(int block_size) {
  return std::make_shared<MambaSpec>(
      block_size, std::vector<std::vector<int64_t>>{{1, 1}},
      std::vector<DType>{DType::kF32});
}

KVCacheConfig ConfigWith(std::vector<std::shared_ptr<vllm::v1::KVCacheSpec>> specs) {
  KVCacheConfig cfg;
  cfg.num_blocks = 100;
  int i = 0;
  for (auto& s : specs) {
    cfg.kv_cache_groups.emplace_back(
        std::vector<std::string>{"g" + std::to_string(i++)}, std::move(s));
  }
  return cfg;
}

}  // namespace

// Single group => the knob is inert: (block_size * dcp, block_size * dcp),
// regardless of prefix_match_unit / caching. (kv_cache_utils.py:649-651)
TEST_CASE("prefix_match_unit_single_group_is_inert") {
  KVCacheConfig cfg = ConfigWith({MakeFull(16)});
  auto [sched, hash] = resolve_kv_cache_block_sizes(
      cfg, /*cache_block_size=*/16, /*prefix_match_unit=*/std::nullopt,
      /*enable_prefix_caching=*/true, /*connector_enabled=*/false,
      /*dcp_world_size=*/1);
  CHECK(sched == 16);
  CHECK(hash == 16);

  // Even a set override is ignored on the single-group path.
  auto [sched2, hash2] = resolve_kv_cache_block_sizes(
      cfg, 16, /*prefix_match_unit=*/8, true, false, 1);
  CHECK(sched2 == 16);
  CHECK(hash2 == 16);

  // DCP scales the single-group block size.
  auto [sched3, hash3] = resolve_kv_cache_block_sizes(
      cfg, 16, std::nullopt, true, false, /*dcp_world_size=*/2);
  CHECK(sched3 == 32);
  CHECK(hash3 == 32);
}

// Multi-group default (prefix_match_unit unset) => hash_block_size = GCD of
// group block sizes; scheduler_block_size = LCM. (kv_cache_utils.py:659,679-681)
TEST_CASE("prefix_match_unit_multigroup_default_is_gcd") {
  KVCacheConfig cfg = ConfigWith({MakeFull(16), MakeFull(32)});
  auto [sched, hash] = resolve_kv_cache_block_sizes(
      cfg, /*cache_block_size=*/16, std::nullopt,
      /*enable_prefix_caching=*/true, /*connector_enabled=*/false, 1);
  CHECK(sched == 32);  // lcm(16, 32)
  CHECK(hash == 16);   // gcd(16, 32)
}

// RED-first: prefix_match_unit=16 forces a FINER matching unit than the default
// gcd (32) would give for two 32-token groups. (kv_cache_utils.py:678-679)
TEST_CASE("prefix_match_unit_override_is_finer_than_default") {
  KVCacheConfig cfg = ConfigWith({MakeFull(32), MakeFull(32)});

  // Default: gcd(32,32) == 32.
  auto [sched_d, hash_d] = resolve_kv_cache_block_sizes(
      cfg, /*cache_block_size=*/32, std::nullopt, true, false, 1);
  CHECK(sched_d == 32);
  CHECK(hash_d == 32);

  // Override =16 lands finer (16 divides 32); the two DIFFER — this is the
  // observable behaviour --prefix-match-unit 16 buys.
  auto [sched_o, hash_o] = resolve_kv_cache_block_sizes(
      cfg, 32, /*prefix_match_unit=*/16, true, false, 1);
  CHECK(sched_o == 32);
  CHECK(hash_o == 16);
  CHECK(hash_o != hash_d);
}

// prefix_match_unit set finer than a large physical block (the 32-vs-1024
// docstring example, generalized): 32 divides both group sizes.
TEST_CASE("prefix_match_unit_finer_than_large_block") {
  KVCacheConfig cfg = ConfigWith({MakeFull(1024), MakeFull(64)});
  auto [sched, hash] = resolve_kv_cache_block_sizes(
      cfg, /*cache_block_size=*/1024, /*prefix_match_unit=*/32, true, false, 1);
  CHECK(sched == 1024);  // lcm(1024, 64)
  CHECK(hash == 32);
}

// A prefix_match_unit that does not divide every group block size throws
// (mirrors upstream ValueError). (kv_cache_utils.py:682-687)
TEST_CASE("prefix_match_unit_nondivisible_throws") {
  KVCacheConfig cfg = ConfigWith({MakeFull(16), MakeFull(32)});
  CHECK_THROWS_AS(
      resolve_kv_cache_block_sizes(cfg, 16, /*prefix_match_unit=*/10, true,
                                   false, 1),
      std::invalid_argument);
  // 16 % 3 != 0 as well.
  CHECK_THROWS_AS(
      resolve_kv_cache_block_sizes(cfg, 16, /*prefix_match_unit=*/3, true, false,
                                   1),
      std::invalid_argument);
}

// No prefix caching AND no connector => hash_block_size is pinned to the
// scheduler block size (finer hashing disabled), even though gcd would be
// finer. (kv_cache_utils.py:661-666)
TEST_CASE("prefix_match_unit_no_consumer_backs_off") {
  KVCacheConfig cfg = ConfigWith({MakeFull(16), MakeFull(32)});
  auto [sched, hash] = resolve_kv_cache_block_sizes(
      cfg, 16, std::nullopt, /*enable_prefix_caching=*/false,
      /*connector_enabled=*/false, 1);
  CHECK(sched == 32);
  CHECK(hash == 32);  // NOT gcd(=16): no consumer, so no finer hashing.

  // A connector alone (no prefix caching) re-enables finer hashing.
  auto [sched2, hash2] = resolve_kv_cache_block_sizes(
      cfg, 16, std::nullopt, /*enable_prefix_caching=*/false,
      /*connector_enabled=*/true, 1);
  CHECK(sched2 == 32);
  CHECK(hash2 == 16);  // gcd(16, 32)
}

// A mamba group whose block size diverges from cache_block_size
// (mamba_cache_mode != "align") backs off to the scheduler block size; an
// aligned mamba group proceeds to the gcd. (kv_cache_utils.py:668-676)
TEST_CASE("prefix_match_unit_mamba_nonalign_backs_off") {
  // Non-align: attention 16 + mamba 32, cache_block_size 16 => back off to
  // lcm(16, 32) = 32 for BOTH, not gcd(=16).
  KVCacheConfig nonalign = ConfigWith({MakeFull(16), MakeMamba(32)});
  auto [sched_n, hash_n] = resolve_kv_cache_block_sizes(
      nonalign, /*cache_block_size=*/16, std::nullopt, true, false, 1);
  CHECK(sched_n == 32);
  CHECK(hash_n == 32);

  // Align: mamba block == cache_block_size == 16 => gcd(16, 16) = 16.
  KVCacheConfig align = ConfigWith({MakeFull(16), MakeMamba(16)});
  auto [sched_a, hash_a] = resolve_kv_cache_block_sizes(
      align, /*cache_block_size=*/16, std::nullopt, true, false, 1);
  CHECK(sched_a == 16);
  CHECK(hash_a == 16);
  CHECK(hash_a != hash_n);
}

// The resolved hash_block_size IS the matching unit: feeding it to the block
// hasher changes the number and granularity of prefix-cache hashes. A finer
// unit (16) yields more, finer hashes than the coarse unit (32) over the same
// tokens. This is what a prefix-cache hit can then land on.
TEST_CASE("prefix_match_unit_changes_hash_granularity") {
  init_none_hash(sha256_cbor, "seed42");
  std::vector<int32_t> tokens(64);
  for (int i = 0; i < 64; ++i) tokens[i] = i;

  // Coarse unit 32 (default gcd for two 32-blocks): 64/32 = 2 hashes.
  const auto coarse = hash_request_tokens(sha256_cbor, /*block_size=*/32, tokens);
  // Fine unit 16 (--prefix-match-unit 16): 64/16 = 4 hashes.
  const auto fine = hash_request_tokens(sha256_cbor, /*block_size=*/16, tokens);

  CHECK(coarse.size() == 2);
  CHECK(fine.size() == 4);
  // The finer unit exposes an extra intra-coarse-block boundary hash: the first
  // fine hash covers tokens [0,16) whereas the first coarse hash covers [0,32),
  // so they are distinct fingerprints.
  CHECK(fine[0] != coarse[0]);
}
