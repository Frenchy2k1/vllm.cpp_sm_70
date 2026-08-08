// ROCm upload of Gemma-4 fused MoE expert stacks (multi-GPU capable).
#include <hip/hip_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/gemma4.h"
#include "vllm/model_executor/models/gemma4_moe.h"

namespace vllm {
namespace {

void Check(hipError_t e, const char* w) {
  if (e != hipSuccess)
    throw std::runtime_error(std::string("gemma4 resident: ") + w + ": " + hipGetErrorString(e));
}

}  // namespace

size_t UploadGemma4ExpertsResident(std::vector<Gemma4MoeLayerWeights>& layers,
                                   int num_gpus) {
  if (num_gpus < 1) num_gpus = 1;
  int available = 0;
  Check(hipGetDeviceCount(&available), "hipGetDeviceCount");
  if (available < 1) return 0;
  if (num_gpus > available) num_gpus = available;

  size_t total = 0;
  int ok_layers = 0;
  int max_layers = 100000;
  if (const char* ml = std::getenv("VT_GEMMA4_RESIDENT_MAX_LAYERS")) {
    max_layers = std::max(0, std::atoi(ml));
  }
  for (size_t li = 0; li < layers.size(); ++li) {
    if (ok_layers >= max_layers) break;
    auto& moe = layers[li];
    if (!moe.enabled || moe.experts.Empty()) continue;
    auto& ex = moe.experts;
    const size_t gu_bytes = ex.gate_up.bytes.size();
    const size_t dn_bytes = ex.down.bytes.size();
    const int dev = static_cast<int>(li % static_cast<size_t>(num_gpus));
    Check(hipSetDevice(dev), "hipSetDevice");
    void* gu = nullptr;
    void* dn = nullptr;
    hipError_t e1 = hipMalloc(&gu, gu_bytes);
    hipError_t e2 = (e1 == hipSuccess) ? hipMalloc(&dn, dn_bytes) : hipErrorMemoryAllocation;
    if (e1 != hipSuccess || e2 != hipSuccess) {
      if (gu) (void)hipFree(gu);
      if (dn) (void)hipFree(dn);
      std::fprintf(stderr,
                   "gemma4 moe: resident upload stopped at layer %zu on gpu %d (%s)\n",
                   li, dev, hipGetErrorString(e1 != hipSuccess ? e1 : e2));
      break;
    }
    Check(hipMemcpy(gu, ex.gate_up.bytes.data(), gu_bytes, hipMemcpyHostToDevice), "H2D gu");
    Check(hipMemcpy(dn, ex.down.bytes.data(), dn_bytes, hipMemcpyHostToDevice), "H2D dn");
    ex.gate_up_dev = gu;
    ex.down_dev = dn;
    ex.dev_id = dev;
    total += gu_bytes + dn_bytes;
    ++ok_layers;
  }
  Check(hipSetDevice(0), "hipSetDevice0");
  std::fprintf(stderr, "gemma4 moe: resident experts gpus=%d layers=%d %.2f GiB\n", num_gpus,
               ok_layers, total / (1024.0 * 1024.0 * 1024.0));
  return total;
}

size_t UploadGemma4ExpertsResidentForWeights(Gemma4Weights& weights, int num_gpus) {
  std::vector<Gemma4MoeLayerWeights> flat;
  std::vector<Gemma4MoeLayerWeights*> ptrs;
  for (auto& layer : weights.layers) {
    if (!layer.moe.enabled) continue;
    ptrs.push_back(&layer.moe);
    flat.push_back(std::move(layer.moe));
  }
  size_t n = UploadGemma4ExpertsResident(flat, num_gpus);
  for (size_t i = 0; i < ptrs.size(); ++i) *ptrs[i] = std::move(flat[i]);
  return n;
}

}  // namespace vllm
