// LoRALayerWeights — the per-layer low-rank adapter weight container.
//
// UPSTREAM (ported FROM, ground-every-impl rule; ${VLLM_SOURCE} @ 555967922/
// vLLM 0.26.0.dev0):
//   vllm/lora/lora_weights.py:13-96  class LoRALayerWeights
//                                    (lora_a [rank,input], lora_b [output,rank],
//                                     scaling = alpha/rank, optimize(),
//                                     input_dim/output_dim, dummy weights)
//   vllm/lora/lora_weights.py:99-282 class PackedLoRALayerWeights
//                                    (pack(), per-slice optimize(), is_packed)
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
#include <optional>
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

  // is_packed (lora_weights.py:52-54): a plain adapter is never packed.
  static constexpr bool is_packed() { return false; }
};

// PackedLoRALayerWeights (lora_weights.py:99-282) — the adapter for a PACKED
// base layer: `qkv_proj` (q/k/v) and `gate_up_proj` (gate/up) are one fused
// linear whose output is a concatenation of sub-module windows, so one adapter
// carries one low-rank pair PER SUB-MODULE.
//
// Upstream stores four parallel lists (`lora_alphas`, `lora_a`, `lora_b`,
// `scaling`) with `None` marking a sub-module that has no LoRA
// (lora_weights.py:130-133). We store one optional LoRALayerWeights per
// sub-module: same information, and `pack()`'s "optimize every sub-LoRA, then
// declare every present slice's scaling to be 1" (`:138`, `:147-150`) is then
// exactly the per-element Optimize().
//
// DEVIATION (recorded): upstream subclasses LoRALayerWeights and overrides
// input_dim/output_dim to raise NotImplementedError (`:272-278`) because the
// packed lists have no single dim. We do not inherit, so those accessors are
// simply absent — the same "there is no one dim" contract, enforced at compile
// time instead of at call time.
//
// pack_moe / pack_moe_stacked (`:154-261`) are NOT ported here: they build the
// 3-D per-expert stacks consumed by `FusedMoEWithLoRA.set_lora`, which belongs
// with the fused-MoE LoRA layer (W7 in .agents/specs/lora-adapter.md).
struct PackedLoRALayerWeights {
  std::string module_name;
  int rank = 0;
  // One entry per packed sub-module, in output order. An empty optional is
  // upstream's `None`: that sub-module has no LoRA and contributes nothing.
  std::vector<std::optional<LoRALayerWeights>> subloras;

  // pack (lora_weights.py:126-152). Takes rank/module_name from the first
  // present sub-LoRA, optimize()s every present one (folding its scaling into
  // lora_b) and keeps the absent ones absent. Throws std::invalid_argument when
  // every entry is absent, mirroring upstream's StopIteration from
  // `next(lora for lora in loras if lora is not None)` (`:134`).
  static PackedLoRALayerWeights Pack(
      const std::vector<std::optional<LoRALayerWeights>>& loras);

  // optimize (lora_weights.py:263-270): fold each present slice's scaling into
  // its lora_b, skipping slices already at scaling == 1 and absent slices.
  PackedLoRALayerWeights& Optimize();

  // is_packed (lora_weights.py:280-282).
  static constexpr bool is_packed() { return true; }
};

}  // namespace lora
}  // namespace vllm
