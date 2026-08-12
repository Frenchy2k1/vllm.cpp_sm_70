// vllm.cpp original (container reader); no upstream mirror.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/read_only_file_mapping.h"

namespace vllm {

// One tensor entry from a .safetensors header. `data` points into the file's
// read-only mmap and stays valid for the lifetime of the owning
// SafetensorsFile. `dtype` is the raw header string ("F32", "BF16",
// "F8_E4M3", ...): unknown dtypes are kept as an opaque byte span.
struct StTensor {
  std::string dtype;
  std::vector<int64_t> shape;
  const uint8_t* data = nullptr;
  size_t nbytes = 0;
  // ENG-LOAD-DIRECT-UPLOAD (issue #150): keep-alive on the file's mmap, shared
  // by every tensor of the same shard. A loader that BORROWS `data` instead of
  // copying it out (OwnedBytes::Borrow) hands this to the borrow, so the
  // mapping outlives `SafetensorsFile` itself and the borrowed weight can be
  // read at device-upload time — which is what lets the upload copy straight
  // from the file mapping and skip the intermediate owned host buffer. Null
  // only for a default-constructed StTensor (tests).
  std::shared_ptr<const void> mapping;
};

// One .safetensors file, mmap'd read-only. All header metadata is treated as
// UNTRUSTED: every offset/size is bounds-checked against the actual file size
// and all size arithmetic is overflow-guarded, so Open() throws
// std::runtime_error (message includes the path) on any malformation instead
// of handing out an out-of-bounds span.
class SafetensorsFile {
 public:
  static SafetensorsFile Open(const std::string& path);

  SafetensorsFile(SafetensorsFile&& other) noexcept;
  SafetensorsFile& operator=(SafetensorsFile&& other) noexcept;
  SafetensorsFile(const SafetensorsFile&) = delete;
  SafetensorsFile& operator=(const SafetensorsFile&) = delete;
  ~SafetensorsFile();

  // Tensor names in header-appearance order ("__metadata__" excluded).
  const std::vector<std::string>& Names() const { return names_; }
  // Throws std::runtime_error if `name` is not present.
  const StTensor& Get(const std::string& name) const;
  // Contents of the optional "__metadata__" entry (string -> string).
  const std::map<std::string, std::string>& Metadata() const {
    return metadata_;
  }

 private:
  SafetensorsFile() = default;
  void Release() noexcept;

  std::string path_;
  std::shared_ptr<const detail::ReadOnlyFileMapping> map_;
  std::vector<std::string> names_;
  std::map<std::string, StTensor> tensors_;
  std::map<std::string, std::string> metadata_;
};

// Multi-file checkpoints: parses a model.safetensors.index.json and returns
// its "weight_map" (tensor name -> shard file name). Throws
// std::runtime_error (message includes the path) on missing file, malformed
// JSON, or a missing/non-string-valued "weight_map".
std::map<std::string, std::string> LoadSafetensorsIndex(
    const std::string& index_json_path);

// --- Windowed source-page release (LOAD-SAFETENSORS memory checkpoint) ---
//
// During a full model load every weight tensor is copied out of the read-only
// MAP_PRIVATE safetensors mmap into an owned host buffer; the source range is
// copied-then-dead the instant the copy returns (proven for every 27B/35B copy
// helper in .agents/specs/safetensors-windowed-load.md — no loaded tensor
// references the mmap after load, and entrypoints/model_loader.cpp munmaps the
// whole mapping right after the load). Without release the entire mmap stays
// resident alongside the growing owned mirror, so VmHWM peaks at ~2x the mirror
// (measured 27B: 48.3 GB peak vs 24.75 GB steady). Releasing each range as it is
// consumed mirrors vLLM's stream-shard-then-copy loader, whose source pages
// likewise never all sit resident (base_loader.py:43-82,
// weight_utils.py:905-954, models/utils.py:170-180,252-279).

// Unconditional primitive: drop the resident pages of the FULLY COVERED interior
// of [data, data+nbytes) via madvise(MADV_DONTNEED). Only interior pages (begin
// rounded up, end rounded down to page bounds) are released, so a partially
// copied neighbor tensor sharing an edge page is never dropped. The mapping is
// PROT_READ MAP_PRIVATE and never written, so its pages are clean and re-fault
// from the backing file on any later read — always correctness-safe. A too-small
// span (no whole interior page), a null pointer, or a madvise failure is a no-op.
void ReleaseSourcePages(const void* data, size_t nbytes);

// Process-cached gate: env VT_LOAD_WINDOWED_RELEASE, DEFAULT ON; "=0" rolls back
// to the retain-all-source-pages behavior (same-binary A/B). Read once per
// process (never per-tensor). A test override (below) can force either value.
bool LoadWindowedReleaseEnabled();

// The loaders' single call site after each tensor copy: release iff the gate is
// enabled. Cheap (one cached bool) when the gate is off.
void MaybeReleaseSourcePages(const void* data, size_t nbytes);

namespace detail {
// Test-only override of the windowed-release decision, bypassing the env cache
// so one test binary can exercise both arms. std::nullopt restores the
// env-driven default.
void SetLoadWindowedReleaseOverrideForTesting(std::optional<bool> value);
}  // namespace detail

// --- Load byte accounting (issue #150: measure it properly, then cut it) ------
//
// Process-global counters over the bytes a model load actually MOVES, so the
// cost of the load path is a measured number rather than an inferred one:
//
//   host_copy_bytes    source bytes materialized into an owned host buffer.
//     Counted at the single `MaybeReleaseSourcePages` call every copy helper
//     already makes right after it has consumed a source range — verbatim
//     copies, transposes, dtype conversions and dequants alike.
//   borrowed_bytes     source bytes a weight VIEWS in place (OwnedBytes::Borrow
//     over the safetensors mmap) instead of copying — the direct-upload path.
//   device_upload_bytes bytes copied host->device by ResidentWeight.
//
// Free-running and monotonic; `Reset()` exists for tests. Relaxed atomics: the
// loaders are single-threaded per tensor and these are diagnostics, so no
// ordering is required beyond not tearing.
namespace load_stats {

struct Counters {
  uint64_t host_copy_bytes = 0;
  uint64_t borrowed_bytes = 0;
  uint64_t device_upload_bytes = 0;
};

void AddHostCopy(uint64_t bytes);
void AddBorrowed(uint64_t bytes);
void AddDeviceUpload(uint64_t bytes);
Counters Snapshot();
void Reset();

}  // namespace load_stats

}  // namespace vllm
