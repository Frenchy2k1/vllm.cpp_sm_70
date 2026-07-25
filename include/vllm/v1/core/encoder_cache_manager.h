// EncoderCacheManager — C++ mirror of vllm/v1/core/encoder_cache_manager.py:17.
//
// Ported from: vllm/v1/core/encoder_cache_manager.py (check_and_update_cache:94,
// can_allocate:123, allocate:184, free_encoder_input:214, free:243,
// get_freed_mm_hashes:255, compute_mm_encoder_budget:269) @ vLLM e24d1b24.
//
// Manages the lifecycle of multimodal encoder outputs (vision embeddings) keyed
// by mm-hash: memory-aware caching, reference counting across requests, and
// FIFO eviction of unreferenced entries at allocation time. Decoupled from the
// GPU/vision tower: the caller supplies (request_id, input_id, mm_hash,
// num_embeds); physical embedding storage lives in the (M2) encoder runner.
//
// INERT-WHEN-OFF: constructed only for mm-capable models; the text scheduler
// path never invokes it.
#pragma once

#include <cstdint>
#include <list>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace vllm::v1::core {

class EncoderCacheManager {
 public:
  explicit EncoderCacheManager(int64_t cache_size)
      : cache_size_(cache_size),
        num_free_slots_(cache_size),
        num_freeable_slots_(cache_size) {}

  int64_t cache_size() const { return cache_size_; }
  int64_t num_free_slots() const { return num_free_slots_; }
  int64_t num_freeable_slots() const { return num_freeable_slots_; }

  // Clear all cached encoder outputs and reset capacity (weights updated).
  void Reset();

  // If (mm_hash) is cached, add request_id/input_id references and reclaim it
  // from `freeable` if it was unreferenced. Returns true iff already cached.
  bool CheckAndUpdateCache(const std::string& request_id, int input_id,
                           const std::string& mm_hash);

  // Whether there is capacity (free or reclaimable) for `num_embeds`, given the
  // per-step encoder compute budget and embeds already scheduled this step.
  // Evicts oldest `freeable` entries into `freed_` as needed. Bookkeeping only.
  bool CanAllocate(int64_t num_embeds, int64_t encoder_compute_budget,
                   int64_t num_embeds_to_schedule);

  // Reserve cache space for this item's encoder output (assumes CanAllocate).
  void Allocate(const std::string& request_id, int input_id,
                const std::string& mm_hash, int64_t num_embeds);

  std::unordered_set<int> GetCachedInputIds(const std::string& request_id) const;

  // Drop this request's reference to (mm_hash); when the last reference goes,
  // the entry becomes `freeable` (not physically freed until space is needed).
  void FreeEncoderInput(const std::string& request_id, int input_id,
                        const std::string& mm_hash, int64_t num_embeds);

  // Get and clear the list of entries actually evicted since the last call
  // (the scheduler forwards these to workers to drop encoder outputs).
  std::vector<std::string> GetFreedMmHashes();

 private:
  int64_t cache_size_;
  int64_t num_free_slots_;
  int64_t num_freeable_slots_;

  // mm_hash => set of request_ids referencing it.
  std::unordered_map<std::string, std::unordered_set<std::string>> cached_;
  // request_id => set of input_ids cached for that request.
  std::unordered_map<std::string, std::unordered_set<int>> request_cached_ids_;

  // Insertion-ordered mm_hash => num_embeds (OrderedDict FIFO for eviction).
  std::list<std::pair<std::string, int64_t>> freeable_order_;
  std::unordered_map<std::string,
                     std::list<std::pair<std::string, int64_t>>::iterator>
      freeable_index_;
  std::vector<std::string> freed_;

  void FreeablePush(const std::string& mm_hash, int64_t num_embeds);
  int64_t FreeablePop(const std::string& mm_hash);       // by key
  std::pair<std::string, int64_t> FreeablePopFront();     // oldest
};

// compute_mm_encoder_budget (encoder_cache_manager.py:269): returns
// {compute_budget, cache_size} in encoder tokens. When `disable_chunked_mm_input`
// is set, a single item larger than `max_num_batched_tokens` is illegal
// (throws). When there are no mm modalities (max_tokens_per_mm_item == 0),
// returns {0, 0} (encoder cache not initialized).
std::pair<int64_t, int64_t> ComputeMmEncoderBudget(
    int64_t max_num_encoder_input_tokens, int64_t encoder_cache_size,
    int64_t max_num_batched_tokens, int64_t max_tokens_per_mm_item,
    bool disable_chunked_mm_input);

}  // namespace vllm::v1::core
