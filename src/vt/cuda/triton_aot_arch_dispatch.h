// Exact runtime selector for the six vendored Triton AOT CUDA trees shipped in
// the release fat binary. The other four release SMs use the portable CUDA
// implementation; a cubin is never tried on a merely similar architecture.
#pragma once

#include <array>
#include <cstddef>
#include <utility>

namespace vt::cuda {

inline constexpr int TritonAotTreeIndex(int major, int minor) {
  if (major == 8 && minor == 0) return 0;
  if (major == 8 && minor == 6) return 1;
  if (major == 8 && minor == 9) return 2;
  if (major == 9 && minor == 0) return 3;
  if (major == 10 && minor == 0) return 4;
  if (major == 12 && minor == 1) return 5;
  return -1;
}

template <typename Result, typename Function, typename... Args>
Result DispatchTritonAot(int major, int minor, Result fallback,
                         const std::array<Function, 6>& trees,
                         Args&&... args) {
  const int index = TritonAotTreeIndex(major, minor);
  if (index < 0) return fallback;
  return trees[static_cast<std::size_t>(index)](
      std::forward<Args>(args)...);
}

template <typename Function, typename... Args>
bool DispatchTritonAotVoid(int major, int minor,
                           const std::array<Function, 6>& trees,
                           Args&&... args) {
  const int index = TritonAotTreeIndex(major, minor);
  if (index < 0) return false;
  trees[static_cast<std::size_t>(index)](std::forward<Args>(args)...);
  return true;
}

}  // namespace vt::cuda
