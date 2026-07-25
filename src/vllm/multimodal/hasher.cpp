// Ported from: vllm/multimodal/hasher.py:50-162 @ vLLM e24d1b24.
//
// Reproduces the exact BLAKE3 byte stream of
//   MultiModalHasher.hash_kwargs(model_id=<str>, image=<PIL RGB>)
// for an image with no EXIF ImageID and no palette (the common case). Verified
// byte-exact against the vLLM 0.25.0 oracle for Qwen/Qwen3-VL-4B-Instruct
// (digest ef6f5bea...; see tests/vllm/multimodal/fixtures/qwen3vl/manifest.json).
//
// Serialization, in blake3.update() order (kwargs sorted: "image" < "model_id"):
//   "image"                                   (iter_item_to_bytes key)
//   "image.mode" , "RGB"                      (PIL .mode)
//   "image.data"                              (PIL .data dict key)
//     "ndarray.dtype" , "|u1"                 (np.asarray(img).dtype.str)
//     "ndarray.shape.{i}" , int64_le(dim)     (for i in 0..ndim; np default int64)
//     "ndarray.data" , <raw HWC uint8 bytes>  (c-contiguous view)
//   "model_id" , <utf-8 model_id>
#include "vllm/multimodal/hasher.h"

#include <array>

#include <blake3/blake3.h>

namespace vllm::multimodal {
namespace {

void Update(blake3_hasher& h, const char* s, size_t n) {
  blake3_hasher_update(&h, s, n);
}
void UpdateStr(blake3_hasher& h, const std::string& s) {
  blake3_hasher_update(&h, s.data(), s.size());
}
// numpy's np.array(<python int>).tobytes(): default integer dtype is int64 on
// LP64 platforms -> 8 little-endian bytes.
void UpdateInt64LE(blake3_hasher& h, int64_t v) {
  uint8_t b[8];
  uint64_t u = static_cast<uint64_t>(v);
  for (int i = 0; i < 8; ++i) b[i] = static_cast<uint8_t>((u >> (8 * i)) & 0xFF);
  blake3_hasher_update(&h, b, 8);
}

std::string FinalizeHex(blake3_hasher& h) {
  std::array<uint8_t, BLAKE3_OUT_LEN> out{};
  blake3_hasher_finalize(&h, out.data(), out.size());
  static const char* kHex = "0123456789abcdef";
  std::string hex;
  hex.reserve(out.size() * 2);
  for (uint8_t byte : out) {
    hex.push_back(kHex[byte >> 4]);
    hex.push_back(kHex[byte & 0xF]);
  }
  return hex;
}

}  // namespace

std::string MultiModalHasher::HashImageRGB(const std::string& model_id,
                                           const uint8_t* rgb, int64_t height,
                                           int64_t width) {
  blake3_hasher h;
  blake3_hasher_init(&h);

  // ---- key "image" (a PIL image) ----
  UpdateStr(h, "image");
  // serialize_item(PIL) -> iter_item_to_bytes("image", {"mode","data"}) (dict).
  //   "image.mode" -> "RGB"
  UpdateStr(h, "image.mode");
  UpdateStr(h, "RGB");
  //   "image.data" -> np.asarray(img) (an ndarray)
  UpdateStr(h, "image.data");
  //     serialize_item(ndarray) -> iter_item_to_bytes("ndarray", {dtype,shape,data})
  UpdateStr(h, "ndarray.dtype");
  UpdateStr(h, "|u1");  // uint8, endian-agnostic
  // shape = (H, W, 3), a tuple: each element under key "ndarray.shape.{i}"
  UpdateStr(h, "ndarray.shape.0");
  UpdateInt64LE(h, height);
  UpdateStr(h, "ndarray.shape.1");
  UpdateInt64LE(h, width);
  UpdateStr(h, "ndarray.shape.2");
  UpdateInt64LE(h, 3);
  // data = raw c-contiguous HWC uint8 bytes (memoryview -> yielded verbatim)
  UpdateStr(h, "ndarray.data");
  Update(h, reinterpret_cast<const char*>(rgb),
         static_cast<size_t>(height * width * 3));

  // ---- key "model_id" (a str) ----
  UpdateStr(h, "model_id");
  UpdateStr(h, model_id);

  return FinalizeHex(h);
}

std::string MultiModalHasher::HashAudioF32(const std::string& model_id,
                                           const float* samples,
                                           int64_t num_samples) {
  blake3_hasher h;
  blake3_hasher_init(&h);

  // ---- key "audio" (a float32 1-D np.ndarray) ----
  // hash_kwargs sorts kwargs => "audio" < "model_id"; iter_item_to_bytes("audio",
  // ndarray) yields the key then serialize_item(ndarray) (hasher.py:108-127).
  UpdateStr(h, "audio");
  //   serialize_item(ndarray) -> iter_item_to_bytes("ndarray", {dtype,shape,data})
  UpdateStr(h, "ndarray.dtype");
  UpdateStr(h, "<f4");  // little-endian float32 (numpy obj.dtype.str)
  // shape = (N,), a 1-element tuple -> one "ndarray.shape.0" entry.
  UpdateStr(h, "ndarray.shape.0");
  UpdateInt64LE(h, num_samples);
  // data = raw c-contiguous float32 bytes (obj.view(np.uint8).data memoryview).
  UpdateStr(h, "ndarray.data");
  Update(h, reinterpret_cast<const char*>(samples),
         static_cast<size_t>(num_samples) * sizeof(float));

  // ---- key "model_id" (a str) ----
  UpdateStr(h, "model_id");
  UpdateStr(h, model_id);

  return FinalizeHex(h);
}

}  // namespace vllm::multimodal
