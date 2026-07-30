// vt::MergedGemm — dispatch for the declarative MERGED-GEMM GROUP descriptor
// (merged_gemm.h). Realizes a group either through its fused fast op (the promoted
// shared kernel, selected by dtype/arity/epilogue) or through the byte-exact Tier-0
// composite of standalone vt:: ops — the same tiering FusedRecipe uses.
#include "vt/merged_gemm.h"

#include <cmath>
#include <vector>

#include "vt/dtype.h"
#include "vt/op_provider.h"
#include "vt/ops.h"

namespace vt {

void MergedGemm(Queue& q, const MergedGemmGroup& desc, Tensor& out, const Tensor& act,
                const Tensor& gate_w, const Tensor& up_w, const Tensor& expert_ids,
                float epilogue_scalar, bool force_composite) {
  VT_CHECK(desc.arity == 2 && desc.epilogue == MergedEpilogue::kSiluMulClamp,
           "MergedGemm: only the arity-2 clamped-SwiGLU group is realized today");

  // Fast path: the descriptor's fused kernel, registered for this device. This is the
  // one-launch realization every keep-quant MoE arch inherits automatically.
  const bool have_fast =
      desc.fast_op != kNoMergedFastOp &&
      OpRegistered(static_cast<OpId>(desc.fast_op), q.device.type);
  if (have_fast && !force_composite) {
    MoeGateUpSwiGLUGrouped(q, out, act, gate_w, up_w, expert_ids, epilogue_scalar);
    return;
  }

  // Tier-0 COMPOSITE (the portable reference golden): the standalone-op sequence
  //   g = MatmulBTQuantGrouped(gate_w);  u = MatmulBTQuantGrouped(up_w);
  //   out = min(g,limit)·sigmoid(min(g,limit))·clamp(u,±limit)
  // BYTE-EXACT to what the fused kernel computes. It runs on the CPU reference tier;
  // an accelerated backend registers the fast op above rather than expanding this.
  VT_CHECK(q.device.type == DeviceType::kCPU,
           "MergedGemm: the Tier-0 composite is the CPU reference tier — a non-CPU "
           "device must register the group's fast op (kMoeGateUpSwiGLUGrouped)");
  const int64_t P = out.shape[0];
  const int64_t N = out.shape[1];
  std::vector<float> g(static_cast<size_t>(P) * N);
  std::vector<float> u(static_cast<size_t>(P) * N);
  Tensor gt = Tensor::Contiguous(g.data(), DType::kF32, out.device, {P, N});
  Tensor ut = Tensor::Contiguous(u.data(), DType::kF32, out.device, {P, N});
  MatmulBTQuantGrouped(q, gt, act, gate_w, expert_ids);
  MatmulBTQuantGrouped(q, ut, act, up_w, expert_ids);
  float* o = static_cast<float*>(out.data);
  const float limit = epilogue_scalar;
  for (size_t i = 0; i < g.size(); ++i) {
    const float gate = std::fmin(g[i], limit);
    const float up = std::fmin(std::fmax(u[i], -limit), limit);
    o[i] = gate * (1.0F / (1.0F + std::exp(-gate))) * up;
  }
}

}  // namespace vt
