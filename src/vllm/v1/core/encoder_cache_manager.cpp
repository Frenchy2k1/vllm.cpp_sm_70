// Ported from: vllm/v1/core/encoder_cache_manager.py:17-316 @ vLLM e24d1b24.
// See encoder_cache_manager.h for provenance.
#include "vllm/v1/core/encoder_cache_manager.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace vllm::v1::core {

void EncoderCacheManager::FreeablePush(const std::string& mm_hash,
                                       int64_t num_embeds) {
  freeable_order_.emplace_back(mm_hash, num_embeds);
  freeable_index_[mm_hash] = std::prev(freeable_order_.end());
}

int64_t EncoderCacheManager::FreeablePop(const std::string& mm_hash) {
  auto it = freeable_index_.find(mm_hash);
  const int64_t n = it->second->second;
  freeable_order_.erase(it->second);
  freeable_index_.erase(it);
  return n;
}

std::pair<std::string, int64_t> EncoderCacheManager::FreeablePopFront() {
  auto pr = freeable_order_.front();
  freeable_order_.pop_front();
  freeable_index_.erase(pr.first);
  return pr;
}

void EncoderCacheManager::Reset() {
  cached_.clear();
  request_cached_ids_.clear();
  freeable_order_.clear();
  freeable_index_.clear();
  freed_.clear();
  num_free_slots_ = cache_size_;
  num_freeable_slots_ = cache_size_;
}

bool EncoderCacheManager::CheckAndUpdateCache(const std::string& request_id,
                                              int input_id,
                                              const std::string& mm_hash) {
  auto it = cached_.find(mm_hash);
  if (it == cached_.end()) return false;  // not cached at all

  // Cached but currently unreferenced: reclaim from freeable.
  if (it->second.empty()) {
    const int64_t num_embeds = FreeablePop(mm_hash);
    num_freeable_slots_ -= num_embeds;
  }
  it->second.insert(request_id);
  request_cached_ids_[request_id].insert(input_id);
  return true;
}

bool EncoderCacheManager::CanAllocate(int64_t num_embeds,
                                      int64_t encoder_compute_budget,
                                      int64_t num_embeds_to_schedule) {
  if (num_embeds > encoder_compute_budget) return false;  // over compute budget

  num_embeds += num_embeds_to_schedule;
  if (num_embeds <= num_free_slots_) return true;          // enough free
  if (num_embeds > num_freeable_slots_) return false;       // not reclaimable

  // Evict oldest freeable entries until there is room.
  while (num_embeds > num_free_slots_) {
    auto pr = FreeablePopFront();
    cached_.erase(pr.first);
    freed_.push_back(pr.first);
    num_free_slots_ += pr.second;
  }
  return true;
}

void EncoderCacheManager::Allocate(const std::string& request_id, int input_id,
                                   const std::string& mm_hash,
                                   int64_t num_embeds) {
  cached_.try_emplace(mm_hash);
  // Eviction happened in CanAllocate, so space is guaranteed.
  if (num_free_slots_ < num_embeds || num_freeable_slots_ < num_embeds) {
    throw std::runtime_error("EncoderCacheManager::Allocate without capacity");
  }
  cached_[mm_hash].insert(request_id);
  request_cached_ids_[request_id].insert(input_id);
  num_free_slots_ -= num_embeds;
  num_freeable_slots_ -= num_embeds;
}

std::unordered_set<int> EncoderCacheManager::GetCachedInputIds(
    const std::string& request_id) const {
  auto it = request_cached_ids_.find(request_id);
  if (it == request_cached_ids_.end()) return {};
  return it->second;
}

void EncoderCacheManager::FreeEncoderInput(const std::string& request_id,
                                           int input_id,
                                           const std::string& mm_hash,
                                           int64_t num_embeds) {
  // Always clean up request_cached_ids, even if mm_hash was already evicted.
  auto rit = request_cached_ids_.find(request_id);
  if (rit != request_cached_ids_.end()) {
    rit->second.erase(input_id);
    if (rit->second.empty()) request_cached_ids_.erase(rit);
  }
  auto cit = cached_.find(mm_hash);
  if (cit == cached_.end() || cit->second.empty()) return;
  cit->second.erase(request_id);
  if (cit->second.empty()) {
    FreeablePush(mm_hash, num_embeds);
    num_freeable_slots_ += num_embeds;
  }
}

std::vector<std::string> EncoderCacheManager::GetFreedMmHashes() {
  std::vector<std::string> freed;
  freed.swap(freed_);
  return freed;
}

std::pair<int64_t, int64_t> ComputeMmEncoderBudget(
    int64_t max_num_encoder_input_tokens, int64_t encoder_cache_size,
    int64_t max_num_batched_tokens, int64_t max_tokens_per_mm_item,
    bool disable_chunked_mm_input) {
  if (max_tokens_per_mm_item <= 0) return {0, 0};  // no mm modalities

  if (disable_chunked_mm_input &&
      max_tokens_per_mm_item > max_num_batched_tokens) {
    throw std::runtime_error(
        "Chunked MM input disabled but max_tokens_per_mm_item exceeds "
        "max_num_batched_tokens");
  }
  const int64_t compute_budget =
      std::max(max_num_encoder_input_tokens, max_tokens_per_mm_item);
  const int64_t cache = std::max(encoder_cache_size, max_tokens_per_mm_item);
  return {compute_budget, cache};
}

}  // namespace vllm::v1::core
