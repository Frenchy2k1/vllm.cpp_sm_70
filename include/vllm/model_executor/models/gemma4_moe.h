// Gemma-4 MoE (26B-A4B) BF16 fused experts — mmap host + optional device-resident.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vt/device.h"
#include "vt/tensor.h"

namespace vllm {

struct Gemma4Weights;

struct Gemma4FusedExperts {
  OwnedTensor gate_up;  // bf16 [E, 2I, H] host mmap borrow
  OwnedTensor down;     // bf16 [E, H, I]
  int64_t num_experts = 0;
  int64_t intermediate = 0;
  int64_t hidden = 0;
  // Optional full-stack device copy (VT_GEMMA4_RESIDENT_EXPERTS=1).
  mutable void* gate_up_dev = nullptr;
  mutable void* down_dev = nullptr;
  mutable int dev_id = -1;  // HIP device; -1 = host stream
  bool Empty() const { return gate_up.Empty(); }
};

struct Gemma4MoeLayerWeights {
  bool enabled = false;
  OwnedTensor router_scale;
  OwnedTensor router_proj;
  OwnedTensor router_proj_fused;
  OwnedTensor per_expert_scale;
  OwnedTensor pre_feedforward_layernorm_2;
  OwnedTensor post_feedforward_layernorm_1;
  OwnedTensor post_feedforward_layernorm_2;
  Gemma4FusedExperts experts;
  int top_k = 8;
  int64_t moe_intermediate = 0;
};

struct Gemma4MoeScratch {
  vt::Tensor tensor;
  std::shared_ptr<void> storage;
};

Gemma4MoeScratch RunGemma4Moe(vt::Queue& q, const Gemma4MoeLayerWeights& moe,
                              const vt::Tensor& router_in, const vt::Tensor& expert_in,
                              int64_t T, int64_t H, float rms_eps);

// Upload each enabled layer's fused experts to a HIP device.
// num_gpus<=1 → all on device 0; else layer L → device (L % num_gpus).
// Returns total bytes placed on devices. Partial success still returns >0.
size_t UploadGemma4ExpertsResident(std::vector<Gemma4MoeLayerWeights>& layers,
                                   int num_gpus);
// Overload: upload each layer.moe in a full Gemma4Weights.
size_t UploadGemma4ExpertsResidentForWeights(Gemma4Weights& weights, int num_gpus);

}  // namespace vllm
