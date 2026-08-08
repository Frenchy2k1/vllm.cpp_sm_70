// Gemma-4 MoE (26B-A4B) BF16 fused experts — mmap-borrowed, top-k materialize.
// See LAB_GEMMA4_26B_MOE.md. Mirrors sglang gemma4_causal.py router + dual branch.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor
#include "vt/device.h"
#include "vt/tensor.h"

namespace vllm {

// Fused expert stacks as on google BF16 checkpoint:
//   gate_up [E, 2*I, H]  — rows [gate|up] per expert
//   down    [E, H, I]
// Bytes are BORROWED from safetensors mmap (keepalive on shards).
struct Gemma4FusedExperts {
  OwnedTensor gate_up;  // bf16 [E, 2I, H] borrowed (rank may be stored as flat)
  OwnedTensor down;     // bf16 [E, H, I]
  int64_t num_experts = 0;
  int64_t intermediate = 0;  // I
  int64_t hidden = 0;        // H
  bool Empty() const { return gate_up.Empty(); }
};

struct Gemma4MoeLayerWeights {
  bool enabled = false;
  OwnedTensor router_scale;      // bf16 [H] (kept for debug)
  OwnedTensor router_proj;       // bf16 raw-NK [E, H] unfused
  // scale * H^{-0.5} folded into proj columns at load (preferred at runtime).
  OwnedTensor router_proj_fused;  // bf16 raw-NK [E, H]
  OwnedTensor per_expert_scale;  // bf16 [E]
  OwnedTensor pre_feedforward_layernorm_2;
  OwnedTensor post_feedforward_layernorm_1;
  OwnedTensor post_feedforward_layernorm_2;
  Gemma4FusedExperts experts;
  int top_k = 8;
  int64_t moe_intermediate = 0;
};

struct Gemma4MoeScratch {
  vt::Tensor tensor;  // [T,H] bf16 device
  std::shared_ptr<void> storage;
};

// Router on residual; experts on expert_in; GeGLU top-k from fused BF16 stacks.
Gemma4MoeScratch RunGemma4Moe(vt::Queue& q, const Gemma4MoeLayerWeights& moe,
                              const vt::Tensor& router_in, const vt::Tensor& expert_in,
                              int64_t T, int64_t H, float rms_eps);

}  // namespace vllm
