// vllm.cpp original (portable read-only model-file mapping).
#include "vllm/model_executor/model_loader/read_only_file_mapping.h"

#include <limits>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace vllm::detail {
namespace {

#if defined(_WIN32)
std::runtime_error Win32Error(const char* operation) {
  return std::runtime_error(std::string(operation) + " failed (Win32 error " +
                           std::to_string(::GetLastError()) + ")");
}
#else
std::runtime_error PosixError(const char* operation) {
  return std::runtime_error(std::string(operation) + " failed: " +
                           std::strerror(errno));
}
#endif

}  // namespace

std::shared_ptr<ReadOnlyFileMapping> ReadOnlyFileMapping::Open(
    const std::filesystem::path& path) {
  auto result = std::shared_ptr<ReadOnlyFileMapping>(new ReadOnlyFileMapping());

#if defined(_WIN32)
  HANDLE file = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                              nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                              nullptr);
  if (file == INVALID_HANDLE_VALUE) throw Win32Error("CreateFileW");
  result->file_handle_ = file;

  LARGE_INTEGER file_size{};
  if (!::GetFileSizeEx(file, &file_size)) throw Win32Error("GetFileSizeEx");
  if (file_size.QuadPart <= 0) throw std::runtime_error("empty file");
  if (static_cast<unsigned long long>(file_size.QuadPart) >
      static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
    throw std::runtime_error("file is too large to map");
  }
  result->size_ = static_cast<size_t>(file_size.QuadPart);

  HANDLE mapping =
      ::CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (mapping == nullptr) throw Win32Error("CreateFileMappingW");
  result->mapping_handle_ = mapping;

  void* view = ::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
  if (view == nullptr) throw Win32Error("MapViewOfFile");
  result->data_ = static_cast<const uint8_t*>(view);
#else
  int flags = O_RDONLY;
#ifdef O_CLOEXEC
  flags |= O_CLOEXEC;
#endif
  result->fd_ = ::open(path.c_str(), flags);
  if (result->fd_ < 0) throw PosixError("open");

  struct stat st {};
  if (::fstat(result->fd_, &st) != 0) throw PosixError("fstat");
  if (st.st_size <= 0) throw std::runtime_error("empty file");
  if (static_cast<uintmax_t>(st.st_size) >
      static_cast<uintmax_t>(std::numeric_limits<size_t>::max())) {
    throw std::runtime_error("file is too large to map");
  }
  result->size_ = static_cast<size_t>(st.st_size);

  void* view =
      ::mmap(nullptr, result->size_, PROT_READ, MAP_PRIVATE, result->fd_, 0);
  if (view == MAP_FAILED) throw PosixError("mmap");
  result->data_ = static_cast<const uint8_t*>(view);
#endif

  return result;
}

ReadOnlyFileMapping::~ReadOnlyFileMapping() noexcept {
#if defined(_WIN32)
  if (data_ != nullptr) ::UnmapViewOfFile(data_);
  if (mapping_handle_ != nullptr)
    ::CloseHandle(static_cast<HANDLE>(mapping_handle_));
  if (file_handle_ != nullptr)
    ::CloseHandle(static_cast<HANDLE>(file_handle_));
#else
  if (data_ != nullptr) ::munmap(const_cast<uint8_t*>(data_), size_);
  if (fd_ >= 0) ::close(fd_);
#endif
}

}  // namespace vllm::detail
