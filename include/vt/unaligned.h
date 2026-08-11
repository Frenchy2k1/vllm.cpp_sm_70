#ifndef VT_UNALIGNED_H_
#define VT_UNALIGNED_H_

#include <cstring>
#include <type_traits>

namespace vt {

// Load a scalar from storage whose byte address is not required to satisfy the
// scalar type's alignment. mmap-backed tensor payloads may begin at any byte
// offset, so forming a typed pointer to them is undefined even on CPUs that
// tolerate unaligned instructions.
template <typename T>
T LoadUnaligned(const void* address) {
  static_assert(std::is_trivially_copyable_v<T>);
  T value;
  std::memcpy(&value, address, sizeof(value));
  return value;
}

}  // namespace vt

#endif  // VT_UNALIGNED_H_
