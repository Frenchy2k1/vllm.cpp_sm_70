// vllm.cpp original (portable read-only model-file mapping).
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

namespace vllm::detail {

// One immutable, whole-file mapping. The platform handles and mapped view are
// released together when the last shared owner dies, which lets model tensors
// safely borrow data() without retaining their reader object.
class ReadOnlyFileMapping final {
 public:
  static std::shared_ptr<ReadOnlyFileMapping> Open(
      const std::filesystem::path& path);

  ReadOnlyFileMapping(const ReadOnlyFileMapping&) = delete;
  ReadOnlyFileMapping& operator=(const ReadOnlyFileMapping&) = delete;
  ~ReadOnlyFileMapping() noexcept;

  const uint8_t* data() const noexcept { return data_; }
  size_t size() const noexcept { return size_; }

 private:
  ReadOnlyFileMapping() = default;

  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
#if defined(_WIN32)
  void* file_handle_ = nullptr;
  void* mapping_handle_ = nullptr;
#else
  int fd_ = -1;
#endif
};

}  // namespace vllm::detail
