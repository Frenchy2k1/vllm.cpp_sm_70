// Gemma-4 MoE experts: BF16 fused (Google) or FP8 per-expert (Firworks) + resident.
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

// One FP8 expert (compressed-tensors channel scales). Host mmap borrows.
struct Gemma4Fp8ExpertMats {
  OwnedTensor gate_w;  // F8 as I8 [I,H]
  OwnedTensor gate_s;  // BF16 [I] or [I,1]
  OwnedTensor up_w;
  OwnedTensor up_s;
  OwnedTensor down_w;  // F8 [H,I]
  OwnedTensor down_s;  // BF16 [H]
  // Lazy host BF16 cache after first dequant (decode reuse).
  mutable std::vector<uint16_t> cached_gu;  // [2I,H]
  mutable std::vector<uint16_t> cached_dn;  // [H,I]
  // Lazy device BF16 copy on compute GPU (avoids H2D every token).
  mutable void* dev_gu = nullptr;  // [2I,H] bf16
  mutable void* dev_dn = nullptr;  // [H,I] bf16
};

struct Gemma4FusedExperts {
  // Google BF16 fused stacks (optional).
  OwnedTensor gate_up;  // bf16 [E, 2I, H]
  OwnedTensor down;     // bf16 [E, H, I]
  // Firworks FP8 per-expert (optional). size()==E when is_fp8.
  bool is_fp8 = false;
  std::vector<Gemma4Fp8ExpertMats> fp8;
  int64_t num_experts = 0;
  int64_t intermediate = 0;
  int64_t hidden = 0;
  // Optional device-resident BF16 fused stacks after Prepare.
  mutable void* gate_up_dev = nullptr;
  mutable void* down_dev = nullptr;
  mutable int dev_id = -1;
  bool Empty() const { return gate_up.Empty() && fp8.empty(); }
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

size_t UploadGemma4ExpertsResident(std::vector<Gemma4MoeLayerWeights>& layers,
                                   int num_gpus);
size_t UploadGemma4ExpertsResidentForWeights(Gemma4Weights& weights, int num_gpus);

// Dequant one FP8 expert into host BF16 gate_up[2I,H] and down[H,I] (caller-owned).
void DequantGemma4Fp8ExpertToBf16(const Gemma4Fp8ExpertMats& ex, int64_t I, int64_t H,
                                  uint16_t* gate_up_out, uint16_t* down_out);

}  // namespace vllm
