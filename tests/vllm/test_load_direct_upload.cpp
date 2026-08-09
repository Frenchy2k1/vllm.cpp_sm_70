// ENG-LOAD-DIRECT-UPLOAD (issue #150) — the MECHANISM, not a timing.
//
// The claim under test is not "loading got faster"; it is that a weight the
// device consumes VERBATIM never materializes an owned host buffer at all. That
// is an observable, deterministic property of the loaded container:
//
//   * `bytes.data()` IS the address inside the safetensors mmap (so nothing was
//     copied), and `bytes.borrowed()` is true;
//   * the borrow keeps the mapping alive past ~SafetensorsFile, which is what
//     makes a LATER device upload legal (the whole design problem);
//   * the byte accounting shows the range under `borrowed`, and NOT under
//     `host_copy`;
//   * every non-verbatim path — transpose, dtype conversion, concatenation —
//     and every size/dtype mismatch still copies, i.e. the lever fails closed.
//
// A timing test would pass with the copy still in place. These do not.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"

namespace {

std::string U64Le(uint64_t v) {
  std::string s(8, '\0');
  for (int i = 0; i < 8; ++i) s[i] = static_cast<char>((v >> (8 * i)) & 0xff);
  return s;
}

class TempFile {
 public:
  explicit TempFile(const std::string& bytes) {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("vllm_direct_upload_test_" + std::to_string(counter++) +
              ".safetensors"))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  ~TempFile() { std::remove(path_.c_str()); }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

// Two BF16 [4,2] tensors ("w", "v") and one F32 [4] tensor ("f"), with distinct
// contents so a wrong copy is visible in the values, not only in the pointer.
constexpr int64_t kRows = 4;
constexpr int64_t kCols = 2;
constexpr size_t kBf16Bytes = static_cast<size_t>(kRows * kCols) * 2;
constexpr size_t kF32Bytes = static_cast<size_t>(kRows) * 4;

std::string Body() {
  std::string data(kBf16Bytes * 2 + kF32Bytes, '\0');
  std::vector<uint16_t> w(kRows * kCols);
  std::vector<uint16_t> v(kRows * kCols);
  for (size_t i = 0; i < w.size(); ++i) {
    w[i] = static_cast<uint16_t>(0x3f00 + i);
    v[i] = static_cast<uint16_t>(0x4100 + i);
  }
  std::memcpy(data.data(), w.data(), kBf16Bytes);
  std::memcpy(data.data() + kBf16Bytes, v.data(), kBf16Bytes);
  const float f[kRows] = {1.5F, 2.5F, 3.5F, 4.5F};
  std::memcpy(data.data() + 2 * kBf16Bytes, f, kF32Bytes);
  return data;
}

std::string Header() {
  return
      R"({"w":{"dtype":"BF16","shape":[4,2],"data_offsets":[0,16]},)"
      R"("v":{"dtype":"BF16","shape":[4,2],"data_offsets":[16,32]},)"
      R"("f":{"dtype":"F32","shape":[4],"data_offsets":[32,48]}})";
}

std::string File() {
  const std::string h = Header();
  return U64Le(h.size()) + h + Body();
}

vllm::TensorResolver ResolverFor(const vllm::SafetensorsFile& st) {
  return [&st](const std::string& name) -> const vllm::StTensor& {
    return st.Get(name);
  };
}

// RAII around the process-wide test seams so one failing CHECK cannot leak a
// forced decision into the next TEST_CASE.
class ForcedArm {
 public:
  explicit ForcedArm(bool direct) {
    vllm::detail::SetLoadDirectUploadOverrideForTesting(direct);
    // The windowed source-page release madvises the source range away after a
    // copy. Reading a copied tensor's SOURCE afterwards is legal (the pages
    // re-fault) but the assertions below compare against it, so keep it off to
    // isolate the one behavior under test.
    vllm::detail::SetLoadWindowedReleaseOverrideForTesting(false);
  }
  ~ForcedArm() {
    vllm::detail::SetLoadDirectUploadOverrideForTesting(std::nullopt);
    vllm::detail::SetLoadWindowedReleaseOverrideForTesting(std::nullopt);
  }
};

}  // namespace

TEST_CASE("direct upload: a verbatim BF16 weight VIEWS the mapping, no host copy") {
  ForcedArm arm(true);
  TempFile f(File());
  vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(f.path());
  const vllm::StTensor& src = st.Get("w");

  const vllm::load_stats::Counters before = vllm::load_stats::Snapshot();
  const vllm::OwnedTensor w = vllm::dense_loaders::LoadBf16Direct(ResolverFor(st), "w");
  const vllm::load_stats::Counters after = vllm::load_stats::Snapshot();

  // THE MECHANISM: the weight's bytes ARE the mapping's bytes. No owned buffer
  // was allocated, so there is nothing for a device upload to read but the file.
  CHECK(w.bytes.borrowed());
  CHECK(static_cast<const void*>(w.bytes.data()) ==
        static_cast<const void*>(src.data));
  CHECK(w.bytes.size() == kBf16Bytes);
  CHECK(w.mmap_src == static_cast<const void*>(src.data));
  CHECK(w.mmap_src_bytes == kBf16Bytes);

  // ... and the accounting agrees: these bytes were BORROWED, never copied.
  CHECK(after.borrowed_bytes - before.borrowed_bytes == kBf16Bytes);
  CHECK(after.host_copy_bytes == before.host_copy_bytes);

  // Metadata is the same as the copy arm would produce.
  CHECK(w.dtype == vt::DType::kBF16);
  CHECK(w.rank == 2);
  CHECK(w.shape[0] == kRows);
  CHECK(w.shape[1] == kCols);
}

TEST_CASE("direct upload: the borrow keeps the mapping alive past ~SafetensorsFile") {
  ForcedArm arm(true);
  TempFile f(File());
  std::vector<uint16_t> expected;
  vllm::OwnedTensor w;
  {
    vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(f.path());
    w = vllm::dense_loaders::LoadBf16Direct(ResolverFor(st), "w");
    expected.assign(reinterpret_cast<const uint16_t*>(w.bytes.data()),
                    reinterpret_cast<const uint16_t*>(w.bytes.data()) +
                        kRows * kCols);
  }
  // The SafetensorsFile is gone. This is the lifetime question the lazy upload
  // poses: the mapping must still be readable here, because ResidentWeight runs
  // long after the loader returned and the shards were released.
  REQUIRE(w.bytes.size() == kBf16Bytes);
  const auto* got = reinterpret_cast<const uint16_t*>(w.bytes.data());
  for (size_t i = 0; i < expected.size(); ++i) CHECK(got[i] == expected[i]);
  for (size_t i = 0; i < expected.size(); ++i)
    CHECK(got[i] == static_cast<uint16_t>(0x3f00 + i));
}

TEST_CASE("direct upload: OFF copies, and both arms load the SAME bytes") {
  TempFile f(File());
  std::vector<uint8_t> direct_bytes;
  std::vector<uint8_t> copied_bytes;
  {
    ForcedArm arm(true);
    vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(f.path());
    const vllm::OwnedTensor w =
        vllm::dense_loaders::LoadBf16Direct(ResolverFor(st), "w");
    REQUIRE(w.bytes.borrowed());
    direct_bytes.assign(w.bytes.begin(), w.bytes.end());
  }
  {
    ForcedArm arm(false);
    vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(f.path());
    const vllm::StTensor& src = st.Get("w");
    const vllm::load_stats::Counters before = vllm::load_stats::Snapshot();
    const vllm::OwnedTensor w =
        vllm::dense_loaders::LoadBf16Direct(ResolverFor(st), "w");
    const vllm::load_stats::Counters after = vllm::load_stats::Snapshot();
    CHECK_FALSE(w.bytes.borrowed());
    CHECK(static_cast<const void*>(w.bytes.data()) !=
          static_cast<const void*>(src.data));
    CHECK(w.mmap_src == nullptr);
    CHECK(after.host_copy_bytes - before.host_copy_bytes == kBf16Bytes);
    CHECK(after.borrowed_bytes == before.borrowed_bytes);
    copied_bytes.assign(w.bytes.begin(), w.bytes.end());
  }
  // A residency change may not change one byte of the model.
  CHECK(direct_bytes == copied_bytes);
}

TEST_CASE("direct upload: a TRANSPOSE is not a verbatim copy and still owns its buffer") {
  ForcedArm arm(true);
  TempFile f(File());
  vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(f.path());
  const vllm::StTensor& src = st.Get("w");
  const vllm::OwnedTensor t =
      vllm::dense_loaders::LoadBf16Transposed(ResolverFor(st), "w");
  CHECK_FALSE(t.bytes.borrowed());
  CHECK(t.mmap_src == nullptr);
  CHECK(static_cast<const void*>(t.bytes.data()) !=
        static_cast<const void*>(src.data));
  // [out=4, in=2] on disk -> [in=2, out=4] in memory, transposed values.
  REQUIRE(t.rank == 2);
  CHECK(t.shape[0] == kCols);
  CHECK(t.shape[1] == kRows);
  const auto* got = reinterpret_cast<const uint16_t*>(t.bytes.data());
  for (int64_t r = 0; r < kRows; ++r) {
    for (int64_t c = 0; c < kCols; ++c) {
      CHECK(got[c * kRows + r] == static_cast<uint16_t>(0x3f00 + r * kCols + c));
    }
  }
}

TEST_CASE("direct upload: a CONCATENATION of two shards owns its merged buffer") {
  ForcedArm arm(true);
  TempFile f(File());
  vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(f.path());
  const vllm::OwnedTensor merged =
      vllm::dense_loaders::LoadMergedBf16RawNK(ResolverFor(st), {"w", "v"});
  CHECK_FALSE(merged.bytes.borrowed());
  CHECK(merged.mmap_src == nullptr);
  REQUIRE(merged.bytes.size() == 2 * kBf16Bytes);
  const auto* got = reinterpret_cast<const uint16_t*>(merged.bytes.data());
  for (size_t i = 0; i < kRows * kCols; ++i) {
    CHECK(got[i] == static_cast<uint16_t>(0x3f00 + i));
    CHECK(got[kRows * kCols + i] == static_cast<uint16_t>(0x4100 + i));
  }
}

TEST_CASE("direct upload: BorrowStTensorBytes FAILS CLOSED on a size or dtype mismatch") {
  ForcedArm arm(true);
  TempFile f(File());
  vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(f.path());
  const vllm::StTensor& w = st.Get("w");
  const vllm::StTensor& fl = st.Get("f");

  // Right dtype, WRONG element count for the span.
  vllm::OwnedTensor too_small;
  CHECK_FALSE(vllm::BorrowStTensorBytes(too_small, w, vt::DType::kBF16, {2, 2}));
  CHECK(too_small.bytes.empty());
  CHECK(too_small.mmap_src == nullptr);

  // Right element count, WRONG dtype width: an f32 source is twice a bf16
  // destination, which is exactly the f32->bf16 conversion arm that must copy.
  vllm::OwnedTensor wrong_dtype;
  CHECK_FALSE(vllm::BorrowStTensorBytes(wrong_dtype, fl, vt::DType::kBF16, {4}));
  CHECK(wrong_dtype.bytes.empty());

  // A synthetic StTensor with no mapping keep-alive can never be borrowed: a
  // borrow with nothing holding the memory alive is the bug this fails closed on.
  vllm::StTensor detached;
  detached.dtype = "BF16";
  detached.shape = {4, 2};
  detached.nbytes = kBf16Bytes;
  detached.data = w.data;
  detached.mapping = nullptr;
  vllm::OwnedTensor no_owner;
  CHECK_FALSE(
      vllm::BorrowStTensorBytes(no_owner, detached, vt::DType::kBF16, {4, 2}));
  CHECK(no_owner.bytes.empty());

  // The matching case still borrows, so the checks above rejected for their
  // stated reason rather than because the arm was off.
  vllm::OwnedTensor ok;
  CHECK(vllm::BorrowStTensorBytes(ok, w, vt::DType::kBF16, {4, 2}));
  CHECK(ok.bytes.borrowed());
}
