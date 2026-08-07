// CPU non-causal depthwise Conv1d — the portable-tier kernel for
// `vt::DepthwiseConv1d` (spike row P2,
// .agents/specs/parakeet-conformer-encoder.md).
//
// Ported FROM (semantics, 1:1): torch
// `nn.Conv1d(C, C, K, stride=1, padding=(K-1)//2, groups=C)` as instantiated by
//   transformers 5.3.0
//   transformers/models/parakeet/modeling_parakeet.py:116
//   `ParakeetEncoderConvolutionModule`
// — the module's `self.padding = (kernel_size - 1) // 2` at :136, the
// `self.depthwise_conv` constructor at :138-146, and its application to the
// [batch, channels, time] tensor at :180. That module is what vLLM runs
// (vllm/model_executor/models/parakeet.py:37,62 delegate to transformers).
// vLLM's own NATIVE conformer has the structurally identical layer at
// vllm/model_executor/models/conformer_encoder.py:229
// (`ConformerConvolution.depthwise_conv`), so one kernel serves both families.
//
// WHY A NEW OP RATHER THAN A FLAG ON THE EXISTING ONE. `vt::CausalConv1dFwd`
// (include/vt/ops.h kCausalConv1dFwd, kernel src/vt/cpu/cpu_ops.cpp) is the
// Mamba/GDN conv: CAUSAL, carrying a persistent `conv_state` across decode
// steps, keyed by `query_start_loc`/`has_initial_state`, and folding a SiLU.
// The conformer conv is centre-padded, stateless, activation-free and strided/
// dilatable. Widening the causal op would have put a branch in a hot decode
// kernel and risked its byte-exactness, so this is a SIBLING and the causal op
// is untouched — the spike's explicit instruction (port map, row P2).
//
// SELF-REGISTERING translation unit (the src/vt/cpu/cpu_ops.cpp Registrar
// idiom), like src/vt/cpu/cpu_layernorm.cpp.
//
// DETERMINISM CONTRACT (gate: tests/vt/test_ops_conv1d_depthwise.cpp). Every
// output element owns ONE f32 accumulator walked strictly in increasing tap
// order with the bias added LAST; the parallel dispatch partitions output
// (batch, channel) rows only. So the result is byte-identical across thread
// counts and byte-identical to the in-test scalar reference. Zero padding is
// realised by SKIPPING out-of-range taps, matching the reference exactly.
#include "cpu_threadpool.h"
#include "vt/ops.h"

namespace vt::cpu {
namespace {

float LoadF32At(const Tensor& t, int64_t i) {
  switch (t.dtype) {
    case DType::kF32: return t.Ptr<float>()[i];
    case DType::kF16: return F16ToF32(t.Ptr<uint16_t>()[i]);
    case DType::kBF16: return BF16ToF32(t.Ptr<uint16_t>()[i]);
    default: VT_CHECK(false, "cpu depthwise_conv1d: unsupported input dtype"); return 0.0f;
  }
}

void StoreF32At(const Tensor& t, int64_t i, float v) {
  switch (t.dtype) {
    case DType::kF32: t.Ptr<float>()[i] = v; break;
    case DType::kF16: t.Ptr<uint16_t>()[i] = F32ToF16(v); break;
    case DType::kBF16: t.Ptr<uint16_t>()[i] = F32ToBF16(v); break;
    default: VT_CHECK(false, "cpu depthwise_conv1d: unsupported output dtype");
  }
}

// out[n,c,ol] = bias[c] + Σ_k x[n, c, ol*stride - padding + k*dilation] * w[c,k],
// taps outside [0,Lin) skipped (zero padding). `groups == C` is implied by the
// op: channel c reads channel c only, which is what makes it depthwise.
void DepthwiseConv1dKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w,
                           const Tensor* bias, const DepthwiseConv1dArgs& args) {
  const int64_t c_dim = x.shape[1], lin = x.shape[2];
  const int64_t lout = out.shape[2];
  // [C,1,K] (torch's depthwise parameter) and [C,K] are the same contiguous
  // bytes; the per-channel filter starts at c*K either way.
  const int64_t k_n = w.shape[w.rank - 1];
  const int64_t stride = args.stride, pad = args.padding, dil = args.dilation;
  const int64_t rows = x.shape[0] * c_dim;  // one output (batch, channel) line per row
  ParallelForRows(CurrentThreadpool(), rows, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const int64_t c = r % c_dim;
      const float b = bias != nullptr ? LoadF32At(*bias, c) : 0.0f;
      const int64_t xbase = r * lin;   // r == bn * C + c, and x is contiguous [N,C,L]
      const int64_t obase = r * lout;
      const int64_t wbase = c * k_n;
      for (int64_t ol = 0; ol < lout; ++ol) {
        float acc = 0.0f;
        for (int64_t k = 0; k < k_n; ++k) {
          const int64_t il = ol * stride - pad + k * dil;
          if (il < 0 || il >= lin) continue;
          acc += LoadF32At(x, xbase + il) * LoadF32At(w, wbase + k);
        }
        if (bias != nullptr) acc += b;
        StoreF32At(out, obase + ol, acc);
      }
    }
  });
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kDepthwiseConv1d, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<DepthwiseConv1dFn>(&DepthwiseConv1dKernel)));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
