// vllm.cpp original -- portable ownership policy for CUDA argmax scratch.
#ifndef VT_CUDA_ARGMAX_SCRATCH_H_
#define VT_CUDA_ARGMAX_SCRATCH_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace vt::cuda {

inline bool ArgmaxGeometricScratchEnabled(const char* value) noexcept {
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

template <class Incumbent, class Candidate>
decltype(auto) DispatchArgmaxScratch(const char* value, Incumbent&& incumbent,
                                     Candidate&& candidate) {
  if (ArgmaxGeometricScratchEnabled(value)) {
    return std::forward<Candidate>(candidate)();
  }
  return std::forward<Incumbent>(incumbent)();
}

inline std::size_t RoundArgmaxScratchCapacity(std::size_t need) {
  if (need <= 1) return need;
  std::size_t capacity = 1;
  while (capacity < need) {
    if (capacity > std::numeric_limits<std::size_t>::max() / 2) {
      throw std::overflow_error("argmax geometric scratch capacity overflow");
    }
    capacity *= 2;
  }
  return capacity;
}

struct ArgmaxScratchKey {
  int device = 0;
  std::uintptr_t stream = 0;

  friend bool operator==(const ArgmaxScratchKey&, const ArgmaxScratchKey&) = default;
};

struct ArgmaxScratchView {
  float* values = nullptr;
  std::int64_t* indices = nullptr;
  std::size_t capacity = 0;
};

struct ArgmaxScratchDiagnostics {
  void* values = nullptr;
  void* indices = nullptr;
  std::size_t capacity = 0;
  std::size_t active_bytes = 0;
  std::size_t retired_bytes = 0;
  std::size_t growths = 0;
};

class ArgmaxScratchOwners {
 public:
  // The caller's submit callback launches both reduction kernels. It runs while
  // the per-(device, stream) owner mutex is held, preventing another host thread
  // from replacing or overwriting the pair between those submissions.
  template <class CaptureQuery, class Allocate, class CleanupPartial, class Retire, class Launch>
  void Submit(ArgmaxScratchKey key, std::size_t need, CaptureQuery&& is_capturing,
              Allocate&& allocate, CleanupPartial&& cleanup_partial, Retire&& retire,
              Launch&& submit) {
    Owner* owner = nullptr;
    bool capture_already_checked = false;
    {
      std::lock_guard lock(owners_mu_);
      auto it = owners_.find(key);
      if (it == owners_.end()) {
        if (std::forward<CaptureQuery>(is_capturing)()) ThrowCaptureMiss();
        capture_already_checked = true;
        it = owners_.emplace(key, std::make_unique<Owner>()).first;
      }
      owner = it->second.get();
    }

    std::lock_guard submit_lock(owner->submit_mu);
    if (need > owner->capacity) {
      if (!capture_already_checked && std::forward<CaptureQuery>(is_capturing)()) {
        ThrowCaptureMiss();
      }
      const std::size_t new_capacity = RoundArgmaxScratchCapacity(need);
      if (new_capacity > std::numeric_limits<std::size_t>::max() / sizeof(std::int64_t)) {
        throw std::overflow_error("argmax geometric scratch byte-size overflow");
      }

      void* new_values = nullptr;
      void* new_indices = nullptr;
      new_values = std::forward<Allocate>(allocate)(new_capacity * sizeof(float));
      if (new_values == nullptr) {
        throw std::runtime_error("argmax geometric scratch value allocation returned null");
      }
      try {
        new_indices = std::forward<Allocate>(allocate)(new_capacity * sizeof(std::int64_t));
        if (new_indices == nullptr) {
          throw std::runtime_error("argmax geometric scratch index allocation returned null");
        }
      } catch (...) {
        std::forward<CleanupPartial>(cleanup_partial)(new_values);
        throw;
      }

      void* old_values = owner->values;
      void* old_indices = owner->indices;
      const std::size_t old_capacity = owner->capacity;
      owner->values = new_values;
      owner->indices = new_indices;
      owner->capacity = new_capacity;
      ++owner->growths;
      if (old_values != nullptr) {
        owner->retired_bytes += old_capacity * sizeof(float);
        std::forward<Retire>(retire)(old_values);
      }
      if (old_indices != nullptr) {
        owner->retired_bytes += old_capacity * sizeof(std::int64_t);
        std::forward<Retire>(retire)(old_indices);
      }
    }

    std::forward<Launch>(submit)(ArgmaxScratchView{static_cast<float*>(owner->values),
                                                   static_cast<std::int64_t*>(owner->indices),
                                                   owner->capacity});
  }

  std::optional<ArgmaxScratchDiagnostics> Diagnostics(ArgmaxScratchKey key) const {
    Owner* owner = nullptr;
    {
      std::lock_guard lock(owners_mu_);
      const auto it = owners_.find(key);
      if (it == owners_.end()) return std::nullopt;
      owner = it->second.get();
    }
    std::lock_guard lock(owner->submit_mu);
    return ArgmaxScratchDiagnostics{
        owner->values,        owner->indices,
        owner->capacity,      owner->capacity * (sizeof(float) + sizeof(std::int64_t)),
        owner->retired_bytes, owner->growths};
  }

  std::size_t OwnerCount() const {
    std::lock_guard lock(owners_mu_);
    return owners_.size();
  }

 private:
  struct Owner {
    mutable std::mutex submit_mu;
    void* values = nullptr;
    void* indices = nullptr;
    std::size_t capacity = 0;
    std::size_t retired_bytes = 0;
    std::size_t growths = 0;
  };

  struct KeyHash {
    std::size_t operator()(ArgmaxScratchKey key) const noexcept {
      std::size_t hash = static_cast<std::size_t>(key.stream);
      hash ^= static_cast<std::size_t>(static_cast<unsigned>(key.device)) + 0x9e3779b97f4a7c15ULL +
              (hash << 6) + (hash >> 2);
      return hash;
    }
  };

  [[noreturn]] static void ThrowCaptureMiss() {
    throw std::runtime_error(
        "argmax geometric scratch capacity miss during CUDA graph capture; "
        "warm or reserve scratch before capture");
  }

  mutable std::mutex owners_mu_;
  std::unordered_map<ArgmaxScratchKey, std::unique_ptr<Owner>, KeyHash> owners_;
};

}  // namespace vt::cuda

#endif  // VT_CUDA_ARGMAX_SCRATCH_H_
