// Unit test for EncoderCacheManager (mm engine-cache seam).
// Mirrors the behaviors of vllm tests/v1/core/test_encoder_cache_manager.py:
// reference counting, freeable/reclaim, FIFO eviction, freed-hash reporting,
// and the budget helper.
#include "doctest/doctest.h"
#include "vllm/v1/core/encoder_cache_manager.h"

using vllm::v1::core::ComputeMmEncoderBudget;
using vllm::v1::core::EncoderCacheManager;

TEST_CASE("encoder-cache: allocate, reference-count, free, reclaim") {
  EncoderCacheManager m(/*cache_size=*/100);
  CHECK(m.num_free_slots() == 100);

  // r1 uses item0 (hashA, 40 embeds).
  CHECK_FALSE(m.CheckAndUpdateCache("r1", 0, "hashA"));
  CHECK(m.CanAllocate(40, /*compute_budget=*/100, /*to_schedule=*/0));
  m.Allocate("r1", 0, "hashA", 40);
  CHECK(m.num_free_slots() == 60);
  CHECK(m.num_freeable_slots() == 60);

  // r2 reuses hashA (already cached) -> no new allocation.
  CHECK(m.CheckAndUpdateCache("r2", 0, "hashA"));
  CHECK(m.num_free_slots() == 60);

  // Free r1's reference: still referenced by r2 -> not freeable.
  m.FreeEncoderInput("r1", 0, "hashA", 40);
  CHECK(m.num_freeable_slots() == 60);

  // Free r2's reference: now unreferenced -> becomes freeable.
  m.FreeEncoderInput("r2", 0, "hashA", 40);
  CHECK(m.num_freeable_slots() == 100);
  CHECK(m.num_free_slots() == 60);  // not physically freed yet

  // Re-reference before eviction reclaims it from freeable.
  CHECK(m.CheckAndUpdateCache("r3", 0, "hashA"));
  CHECK(m.num_freeable_slots() == 60);
  CHECK(m.GetFreedMmHashes().empty());
}

TEST_CASE("encoder-cache: FIFO eviction when space is needed") {
  EncoderCacheManager m(100);
  // Fill with two items, then free both (freeable, unreferenced).
  CHECK(m.CanAllocate(40, 100, 0));
  m.Allocate("r1", 0, "A", 40);
  CHECK(m.CanAllocate(40, 100, 0));
  m.Allocate("r1", 1, "B", 40);
  m.FreeEncoderInput("r1", 0, "A", 40);  // A freeable (older)
  m.FreeEncoderInput("r1", 1, "B", 40);  // B freeable (newer)
  CHECK(m.num_freeable_slots() == 100);
  CHECK(m.num_free_slots() == 20);

  // Need 40 free but only 20 free; evict oldest (A) to make room.
  CHECK(m.CanAllocate(40, 100, 0));
  auto freed = m.GetFreedMmHashes();
  REQUIRE(freed.size() == 1);
  CHECK(freed[0] == "A");           // FIFO: oldest first
  CHECK(m.GetFreedMmHashes().empty());  // cleared after read
}

TEST_CASE("encoder-cache: compute budget cap and capacity refusal") {
  EncoderCacheManager m(50);
  // Item exceeds the per-step compute budget -> refuse.
  CHECK_FALSE(m.CanAllocate(60, /*compute_budget=*/30, 0));
  // Exceeds combined free+freeable capacity -> refuse.
  CHECK_FALSE(m.CanAllocate(60, /*compute_budget=*/100, 0));
}

TEST_CASE("encoder-cache: reset") {
  EncoderCacheManager m(100);
  CHECK(m.CanAllocate(40, 100, 0));
  m.Allocate("r1", 0, "A", 40);
  m.Reset();
  CHECK(m.num_free_slots() == 100);
  CHECK(m.num_freeable_slots() == 100);
  CHECK(m.GetCachedInputIds("r1").empty());
}

TEST_CASE("compute_mm_encoder_budget") {
  // no mm modalities -> {0,0}
  auto b0 = ComputeMmEncoderBudget(4096, 4096, 8192, 0, false);
  CHECK(b0.first == 0);
  CHECK(b0.second == 0);
  // budgets are the max of config and per-item.
  auto b1 = ComputeMmEncoderBudget(1024, 2048, 8192, 3000, false);
  CHECK(b1.first == 3000);   // max(1024, 3000)
  CHECK(b1.second == 3000);  // max(2048, 3000)
  // disable_chunked_mm_input with item > max_num_batched_tokens -> throws.
  CHECK_THROWS(ComputeMmEncoderBudget(1024, 2048, 100, 3000, true));
}
