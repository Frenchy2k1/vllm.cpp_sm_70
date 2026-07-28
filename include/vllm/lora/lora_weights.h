// LoRALayerWeights — the per-layer low-rank adapter weight container.
//
// UPSTREAM (ported FROM, ground-every-impl rule; ${VLLM_SOURCE} @ 555967922/
// vLLM 0.26.0.dev0):
//   vllm/lora/lora_weights.py:13-96  class LoRALayerWeights
//                                    (lora_a [rank,input], lora_b [output,rank],
//                                     scaling = alpha/rank, optimize(),
//                                     input_dim/output_dim, dummy weights)
//
// A LoRA fine-tunes a base linear W by adding a low-rank delta:
//   y = x @ Wᵀ  +  scaling * x @ lora_aᵀ @ lora_bᵀ
// where lora_a is [rank, input] and lora_b is [output, rank]. `scaling` is
// alpha/rank (or alpha/sqrt(rank) for rsLoRA, computed by PEFTHelper). This is
// the pure data container — the batched apply lives in punica.h. Weights are
// stored row-major as portable `float` (the CPU brick; the vt/GPU path is a
// later W in .agents/specs/lora-adapter.md).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vllm {
namespace lora {

// LoRALayerWeights (lora_weights.py:13). Two low-rank matrices for one module.
struct LoRALayerWeights {
  std::string module_name;
  int rank = 0;
  int lora_alpha = 0;
  // lora_a: [rank, input_dim] row-major (contracts the input; produces rank).
  std::vector<float> lora_a;
  // lora_b: [output_dim, rank] row-major (expands rank; produces output).
  std::vector<float> lora_b;
  int input_dim = 0;
  int output_dim = 0;
  // scaling defaults to alpha/rank (lora_weights.py:31-34). Callers that pass a
  // precomputed PEFT scaling (rsLoRA) set it explicitly.
  double scaling = 1.0;

  LoRALayerWeights() = default;

  LoRALayerWeights(std::string name, int rank_, int alpha,
                   std::vector<float> a, std::vector<float> b, int in_dim,
                   int out_dim)
      : module_name(std::move(name)),
        rank(rank_),
        lora_alpha(alpha),
        lora_a(std::move(a)),
        lora_b(std::move(b)),
        input_dim(in_dim),
        output_dim(out_dim),
        scaling(rank_ != 0 ? static_cast<double>(alpha) / rank_ : 1.0) {}

  // optimize() (lora_weights.py:36-42): fold scaling into lora_b so scaling==1.
  // A one-way transform; returns *this for chaining, mirroring upstream.
  LoRALayerWeights& Optimize();

  // input_dim / output_dim properties (lora_weights.py:44-50): derived from the
  // weight shapes. We keep them as explicit fields (portable layout has no
  // torch .shape) and expose the same accessor names for parity.
  int InputDim() const { return input_dim; }
  int OutputDim() const { return output_dim; }

  // create_dummy_lora_weights (lora_weights.py:72-96): zero-filled a/b of the
  // given dims, alpha=1 (scaling=1/rank). Used by the manager's dummy-slot path.
  static LoRALayerWeights CreateDummy(const std::string& module_name,
                                      int input_dim, int output_dim, int rank);
};

}  // namespace lora
}  // namespace vllm
