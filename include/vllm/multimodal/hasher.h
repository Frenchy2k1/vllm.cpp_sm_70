// MultiModalHasher — C++ mirror of vllm/multimodal/hasher.py:50.
//
// Ported from: vllm/multimodal/hasher.py (MultiModalHasher.serialize_item :52,
// iter_item_to_bytes :133, hash_kwargs :153) @ vLLM e24d1b24, and
// vllm/multimodal/processing/inputs.py::get_mm_hashes:62 (the per-item call
// hash_kwargs(model_id=..., image=<PIL>)).
//
// Produces the per-item mm-hash (BLAKE3, vLLM's default VLLM_MM_HASHER_ALGORITHM)
// that keys the encoder cache AND flows into the LMCache KV key as `extra_keys`.
// Byte-exact against the vLLM oracle: hexdigest matches
// MultiModalHasher.hash_kwargs on the same (model_id, RGB image).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vllm::multimodal {

class MultiModalHasher {
 public:
  // Hash one RGB image item exactly as vLLM does for an image with no provided
  // UUID and empty hf_processor_mm_kwargs:
  //   hash_kwargs(model_id=<model_id>, image=<PIL RGB>)
  // where sorted kwargs => "image" is serialized before "model_id". `rgb` is the
  // HWC uint8 buffer (== np.asarray(PIL) for a mode-"RGB" image), height*width*3.
  //
  // Returns the lowercase hex BLAKE3 digest (64 chars).
  static std::string HashImageRGB(const std::string& model_id,
                                  const uint8_t* rgb, int64_t height,
                                  int64_t width);
};

}  // namespace vllm::multimodal
