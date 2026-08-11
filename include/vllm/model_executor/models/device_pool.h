// Shared process-wide caching device allocator (DevicePool) — extracted VERBATIM
// from the Qwen3.6 forward (qwen3_5.cpp) so the dense Qwen3 forward (qwen3.cpp)
// reuses the SAME pooled-scratch machinery instead of raw per-op Backend
// Alloc/Free. This is a pure relocation: the class body, the Pool()/AuxPool()
// singletons, and the thread-local ActivePool()/ActivePoolScope are byte-for-byte
// the qwen3_5.cpp definitions, so the 27B/35B gate-model behavior is unchanged
// (the header is included by qwen3_5.cpp in place of its old inline copies).
//
// Rationale: both cudaMalloc AND cudaFree SYNCHRONIZE the whole device, so the
// per-op DBuf alloc/free churn in a forward (thousands of tiny scratch buffers per
// step) is itself a sync storm. This pool reuses freed blocks (size-class keyed)
// instead of hitting cudaMalloc/cudaFree, so after a brief warm-up almost every
// DBuf lifetime is sync-free. Reuse is safe under the forward's single-queue
// (single-stream) ordering: a block returned to the pool is only handed back out
// on the same queue, and CUDA stream ordering guarantees the op that last touched
// the block has completed before any reused op runs — no host sync needed. Blocks
// are never returned to the driver (leak at process exit, like the cublasLt
// workspace); the pool is bounded by the forward's peak concurrent scratch. The
// pool is backend-agnostic (CPU malloc/free too — a harmless bounded cache there).
#pragma once

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "vt/backend.h"

namespace vllm {

// SIZE-CLASS BUCKETING (prefill sync-cudaMalloc kill). Keying the pool on the
// EXACT byte size defeats reuse during prefill: under continuous batching each
// engine step processes a different token count T (prefill-chunk tokens + decode
// tokens), so every [T,*] scratch is a byte size that step-1 never saw — an
// exact-key MISS -> a SYNCHRONOUS cudaMalloc (device-serializing, does NOT
// overlap compute) on the forward's host thread, thousands per prefill. Rounding
// the request UP to a size class (keep the top `kClassBits` significant bits;
// <=1/2^kClassBits ~ 6.25% over-allocation) makes nearby-T scratch of the same
// op share ONE block, so after warm-up almost every prefill DBuf is a pool hit.
// The returned block is >= the requested bytes (the Tensor view uses only the
// logical prefix), and Put/Get round identically so a block always returns to
// its own class bucket. VT_POOL_EXACT=1 restores exact keying (A/B measurement).
class DevicePool {
 public:
  static size_t SizeClassForTest(size_t bytes) { return ClassOf(bytes); }

  void* Get(vt::Backend& b, size_t bytes) {
    // BYPASS lane (VT_POOL_BYPASS=1) — the pool is a DETECTOR BLIND SPOT and
    // this is how you see through it. Two ways it hides a real defect from
    // compute-sanitizer:
    //   1. size-class rounding hands back a block up to 6.25% LARGER than the
    //      logical tensor, so a write past the last row lands inside the same
    //      driver allocation and memcheck reports nothing;
    //   2. blocks are never returned to the driver, so a use-after-free of a
    //      released DBuf reads memory that is still legally mapped — and, worse,
    //      may already have been handed to an unrelated op.
    // Under bypass every Get is an EXACT-size driver allocation and every Put is
    // a real Free, which restores both boundaries for the detector. It is a
    // debugging lane only: it reinstates the per-op cudaMalloc/cudaFree sync
    // storm this pool exists to remove, so it is never a timing configuration.
    // The backend is remembered here so the no-backend Put overload (the
    // cross-step shared_ptr deleter) can free through it.
    if (Bypass()) {
      backend_ = &b;
      return b.Alloc(bytes);
    }
    const size_t key = ClassOf(bytes);
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = free_.find(key);
      if (it != free_.end() && !it->second.empty()) {
        void* p = it->second.back();
        it->second.pop_back();
        retained_ -= key;
        ++hits_;
        return p;
      }
      ++misses_;
    }
    return b.Alloc(key);
  }
  // Uncapped retention (deliberately-retained cross-step buffers: the device
  // logits / MTP hidden handed off via a shared_ptr deleter). Bytes are always
  // returned to the free list — the cross-step buffers are not cap-evicted.
  void Put(size_t bytes, void* p) {
    // Bypass: free for real so a later use-after-free traps. `backend_` is set by
    // the Get that produced `p`, so it is non-null whenever a Put can be reached;
    // the null guard keeps the lane from leaking a block if that ever stops
    // holding rather than dereferencing a null backend.
    if (Bypass()) {
      if (backend_ != nullptr) backend_->Free(p);
      return;
    }
    const size_t key = ClassOf(bytes);
    std::lock_guard<std::mutex> lk(mu_);
    retained_ += key;
    free_[key].push_back(p);
  }
  // Cap-aware retention for the high-frequency forward scratch (DBuf). The soft
  // cap comes from the platform's residency_policy().device_pool_cap_bytes
  // (BACKEND-PLATFORM item 2). cap == 0 is UNCAPPED — the identical fast path as
  // the uncapped Put above and thus behavior-preserving on GB10 (cap 0 today).
  // When a discrete GPU sets a bound, scratch over the cap is freed to the driver
  // rather than pooled, so the reuse pool self-limits without a model edit.
  void Put(vt::Backend& b, size_t bytes, void* p, size_t cap) {
    if (Bypass()) {
      b.Free(p);
      return;
    }
    const size_t key = ClassOf(bytes);
    std::lock_guard<std::mutex> lk(mu_);
    if (cap != 0 && retained_ + key > cap) {
      b.Free(p);
      return;
    }
    retained_ += key;
    free_[key].push_back(p);
  }

  // Release every RETAINED block back to the driver, and report the bytes freed.
  //
  // The pool earns its keep WITHIN a phase, where the same size classes recur
  // every step and cudaMalloc/cudaFree would be a sync storm. Across a PHASE
  // CHANGE the retained classes are the wrong shapes for what comes next, so on
  // an UNCAPPED pool (`device_pool_cap_bytes == 0`, which is GB10/Thor today)
  // they are pure headroom loss at exactly the moment the next phase wants its
  // own working set. Draining at that boundary costs one cudaFree per retained
  // block, once, and is what keeps a big-canvas MiniMax-H3 VAE decode from
  // meeting the driver's OOM on top of 50 steps of denoise scratch.
  //
  // SAFETY: `free_` only ever holds blocks a DBuf already returned, so nothing
  // live is touched. Under VT_POOL_BYPASS the free list is always empty (Put
  // frees straight through) and this is a no-op.
  size_t Drain(vt::Backend& b) {
    std::lock_guard<std::mutex> lk(mu_);
    size_t freed = 0;
    for (auto& entry : free_) {
      for (void* p : entry.second) {
        b.Free(p);
        freed += entry.first;
      }
    }
    free_.clear();
    retained_ = 0;
    return freed;
  }

  ~DevicePool() {
    if (std::getenv("VT_POOL_STATS") != nullptr) {
      const uint64_t h = hits_.load(), m = misses_.load();
      const double rate = (h + m) ? 100.0 * static_cast<double>(h) / static_cast<double>(h + m) : 0.0;
      std::fprintf(stderr,
                   "[DevicePool] hits=%llu misses(cudaMalloc)=%llu hit-rate=%.2f%% distinct-classes=%zu\n",
                   static_cast<unsigned long long>(h), static_cast<unsigned long long>(m),
                   rate, free_.size());
    }
  }

 private:
  // VT_POOL_BYPASS=1 turns every Get/Put into a raw driver Alloc/Free (see Get).
  // Read once: it must not change between an allocation and its matching free,
  // or a pooled block would be handed to Backend::Free (or a driver block leaked
  // into the free list).
  static bool Bypass() {
    static const bool on = [] {
      const char* e = std::getenv("VT_POOL_BYPASS");
      return e != nullptr && e[0] == '1';
    }();
    return on;
  }

  // Round `bytes` up so it keeps at most kClassBits leading significant bits.
  // Exact keying when VT_POOL_EXACT=1 (A/B). Small sizes (< 2^kClassBits) key
  // exactly — there are few of them and the waste would be proportionally large.
  static size_t ClassOf(size_t bytes) {
    static const bool exact = [] {
      const char* e = std::getenv("VT_POOL_EXACT");
      return e != nullptr && e[0] == '1';
    }();
    if (exact || bytes == 0) return bytes == 0 ? 1 : bytes;
    constexpr int kClassBits = 4;  // <=6.25% over-allocation per class
    const int msb = static_cast<int>(std::bit_width(bytes)) - 1;
    if (msb < kClassBits) return bytes;
    const int shift = msb - kClassBits;
    const size_t mask = (static_cast<size_t>(1) << shift) - 1;
    return (bytes + mask) & ~mask;  // round up to a multiple of 2^shift
  }

  std::mutex mu_;
  // Backend the last Get allocated through, so the no-backend Put overload can
  // free under bypass. One device per process (see ResolveDevicePoolPolicy), so
  // this is stable; unused when bypass is off.
  vt::Backend* backend_ = nullptr;
  std::unordered_map<size_t, std::vector<void*>> free_;
  size_t retained_ = 0;  // bytes (class-rounded) held in free_, for the soft cap
  std::atomic<uint64_t> hits_{0};
  std::atomic<uint64_t> misses_{0};
};

inline DevicePool& Pool() {
  static DevicePool p;
  return p;
}

// --- Aux-stream scratch pool (ENG-MOE-SHARED-AUX) ----------------------------
// The MoE shared-expert overlap (MoeBlockFusedMarlinCuda) issues the shared MLP
// on a SECOND CUDA stream concurrent with the routed experts on the main stream.
// The main `Pool()` above is only reuse-safe under SINGLE-stream ordering: a
// block returned to the free list is handed back out on the same stream, so CUDA
// stream ordering guarantees the block's last op has completed before any reuse.
// Two streams sharing one pool BREAKS that invariant — a scratch block the aux
// stream freed (e.g. the shared gate/up transient) could be handed to a routed
// GEMM on the main stream and written WHILE the aux kernel still reads it (a
// cross-stream RAW/WAR race compute-sanitizer flags). vLLM sidesteps this with
// its STREAM-AWARE caching allocator (record_stream); we keep the simple pool and
// instead give the aux stream its OWN pool. Each pool then only ever serves ONE
// stream, so single-stream ordering holds within it and the two streams never
// share a live block. Blocks are handed back to the pool they came from (DBuf
// stores its owning pool), so a buffer allocated in the aux region and destroyed
// after the join still returns to the aux pool.
#ifdef VT_MARLIN_NVFP4  // only the Marlin MoE overlap path draws from AuxPool
inline DevicePool& AuxPool() {
  static DevicePool p;
  return p;
}
#endif

// Thread-local "active" scratch pool a DBuf constructs from. Defaults to the main
// Pool(); the aux-stream overlap region swaps it to AuxPool() for the duration of
// the shared-expert issue via ActivePoolScope. Single host thread drives the
// forward, and the aux ops are issued in one contiguous block, so the swap is a
// simple RAII stack.
inline DevicePool*& ActivePool() {
  thread_local DevicePool* p = &Pool();
  return p;
}
struct ActivePoolScope {
  DevicePool* prev;
  explicit ActivePoolScope(DevicePool* p) : prev(ActivePool()) { ActivePool() = p; }
  ~ActivePoolScope() { ActivePool() = prev; }
  ActivePoolScope(const ActivePoolScope&) = delete;
  ActivePoolScope& operator=(const ActivePoolScope&) = delete;
};

}  // namespace vllm
