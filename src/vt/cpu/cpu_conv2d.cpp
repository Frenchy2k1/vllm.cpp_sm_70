// CPU Conv2d — the portable-tier kernel for `vt::Conv2d` (spike row P1,
// .agents/specs/parakeet-conformer-encoder.md).
//
// Ported FROM (semantics, 1:1): torch `nn.Conv2d` as instantiated by
//   transformers 5.3.0
//   transformers/models/parakeet/modeling_parakeet.py:357
//   `ParakeetEncoderSubsamplingConv2D`
// whose three constructor calls are the three configurations this kernel must
// serve:
//   * :369-371  nn.Conv2d(1, channels, kernel_size, stride, padding)   — dense
//   * :377-386  nn.Conv2d(C, C, kernel_size, stride, padding, groups=C) — depthwise
//   * :388      nn.Conv2d(C, C, kernel_size=1)                          — pointwise
// and whose output-length arithmetic (:392-400 `_get_output_length`) is the same
// `(L + pad_l + pad_r - kernel)/stride + 1` this kernel's shape contract uses.
// That module is what vLLM itself runs: vllm/model_executor/models/parakeet.py:37
// imports `ParakeetEncoder` from transformers and :62 instantiates it. vLLM's own
// NATIVE conformer front end, vllm/model_executor/models/conformer_encoder.py:18
// `Conv2dSubsampling`, builds the identical stack, so this one kernel serves both.
//
// It also supersedes the host `std::vector<float>` loop
// src/vllm/model_executor/models/gemma4_audio.cpp:92 `Conv2dK3S2P1`, which stays
// in place as an independent correctness reference for that model's prefix.
//
// Like src/vt/cpu/cpu_layernorm.cpp this is a SELF-REGISTERING translation unit
// (the src/vt/cpu/cpu_ops.cpp Registrar idiom), so adding the op edited no
// existing kernel file — only the op-table declaration in include/vt/ops.h and
// the validating wrapper in src/vt/ops.cpp.
//
// DETERMINISM CONTRACT (the gate in tests/vt/test_ops_conv2d.cpp). Every output
// element owns ONE f32 accumulator walked strictly in (ic, kh, kw) order with
// the bias added LAST, and the parallel dispatch partitions OUTPUT rows only —
// so the result is byte-identical to the single-threaded run by construction,
// and byte-identical to the in-test scalar reference that mirrors that order.
// Zero padding is realised by SKIPPING out-of-range taps rather than adding an
// explicit 0.0 product; the reference does the same, which keeps the two exact
// even for a -0.0 accumulator.
#include "cpu_threadpool.h"
#include "vt/ops.h"

namespace vt::cpu {
namespace {

float LoadF32At(const Tensor& t, int64_t i) {
  switch (t.dtype) {
    case DType::kF32: return t.Ptr<float>()[i];
    case DType::kF16: return F16ToF32(t.Ptr<uint16_t>()[i]);
    case DType::kBF16: return BF16ToF32(t.Ptr<uint16_t>()[i]);
    default: VT_CHECK(false, "cpu conv2d: unsupported input dtype"); return 0.0f;
  }
}

void StoreF32At(const Tensor& t, int64_t i, float v) {
  switch (t.dtype) {
    case DType::kF32: t.Ptr<float>()[i] = v; break;
    case DType::kF16: t.Ptr<uint16_t>()[i] = F32ToF16(v); break;
    case DType::kBF16: t.Ptr<uint16_t>()[i] = F32ToBF16(v); break;
    default: VT_CHECK(false, "cpu conv2d: unsupported output dtype");
  }
}

// out[n,oc,oh,ow] = bias[oc] + Σ_{ic,kh,kw} x[n, gc0+ic, oh*sh-ph+kh*dh,
//                                             ow*sw-pw+kw*dw] * w[oc,ic,kh,kw]
// with taps outside the input skipped (zero padding) and `gc0` the first input
// channel of output channel oc's group.
void Conv2dKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w, const Tensor* bias,
                  const Conv2dArgs& args) {
  const int64_t n = x.shape[0], cin = x.shape[1], hin = x.shape[2], win = x.shape[3];
  const int64_t cout = out.shape[1], hout = out.shape[2], wout = out.shape[3];
  const int64_t kh_n = w.shape[2], kw_n = w.shape[3];
  const int64_t cin_g = w.shape[1];          // Cin / groups
  const int64_t cout_g = cout / args.groups;  // Cout / groups
  const int64_t sh = args.stride_h, sw = args.stride_w;
  const int64_t ph = args.pad_h, pw = args.pad_w;
  const int64_t dh = args.dilation_h, dw = args.dilation_w;
  // One "row" is one output line (n, oc, oh) of `wout` elements: independent
  // outputs, so the partition can never change a reduction (spec W3 discipline).
  const int64_t rows = n * cout * hout;
  ParallelForRows(CurrentThreadpool(), rows, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const int64_t oh = r % hout;
      const int64_t oc = (r / hout) % cout;
      const int64_t bn = r / (hout * cout);
      const int64_t gc0 = (oc / cout_g) * cin_g;  // first input channel of oc's group
      const float b = bias != nullptr ? LoadF32At(*bias, oc) : 0.0f;
      for (int64_t ow = 0; ow < wout; ++ow) {
        float acc = 0.0f;
        for (int64_t ic = 0; ic < cin_g; ++ic) {
          const int64_t xc = ((bn * cin) + gc0 + ic) * hin;
          const int64_t wc = (oc * cin_g + ic) * kh_n;
          for (int64_t kh = 0; kh < kh_n; ++kh) {
            const int64_t ih = oh * sh - ph + kh * dh;
            if (ih < 0 || ih >= hin) continue;
            const int64_t xrow = (xc + ih) * win;
            const int64_t wrow = (wc + kh) * kw_n;
            for (int64_t kw = 0; kw < kw_n; ++kw) {
              const int64_t iw = ow * sw - pw + kw * dw;
              if (iw < 0 || iw >= win) continue;
              acc += LoadF32At(x, xrow + iw) * LoadF32At(w, wrow + kw);
            }
          }
        }
        if (bias != nullptr) acc += b;
        StoreF32At(out, ((bn * cout + oc) * hout + oh) * wout + ow, acc);
      }
    }
  });
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kConv2d, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<Conv2dFn>(&Conv2dKernel)));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
