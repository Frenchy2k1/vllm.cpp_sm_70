// CPU relative-position encoder self-attention — the portable-tier kernel for
// `vt::AttentionRelPos` (spike row P3,
// .agents/specs/parakeet-conformer-encoder.md).
//
// Ported FROM (semantics, 1:1):
//   transformers 5.3.0
//   transformers/models/parakeet/modeling_parakeet.py:259
//   `ParakeetEncoderAttention` — forward :302-347, `_rel_shift` :349-355,
//   and the softmax/value contraction it delegates to,
//   `eager_attention_forward` :225-255 (repeat_kv :212-222).
// That module is what vLLM runs (vllm/model_executor/models/parakeet.py:37
// imports transformers' `ParakeetEncoder`, :62 instantiates it). vLLM's own
// NATIVE conformer implements the same Transformer-XL attention at
// vllm/model_executor/models/conformer_encoder.py:170 `RelPosMultiHeadAttention`
// (forward :188-217, `_rel_shift` :179-186); its ONE arithmetic difference —
// scaling the summed score instead of the two terms separately — is exposed as
// `AttentionRelPosArgs::scale_after_sum` rather than chosen for it, so both
// upstreams get a byte-exact path.
//
// Section 3.3 of the Transformer-XL paper (arXiv 1901.02860) names the four
// terms; the kernel computes (a)+(c) as `ac` and (b)+(d) as `bd`.
//
// THE `_rel_shift` CLOSED FORM. Both upstreams left-pad matrix_bd [T, 2T-1] by
// one column, reinterpret as [2T, T], drop the first row, reinterpret as
// [T, 2T-1] and truncate to the first T columns. Element (i, p) of the result
// carries flat index i*(2T-1) + p + T in the padded [T, 2T] view; since
// 1 <= T - i + p <= 2T-1 for every i, p in [0, T), that index is row i,
// column T-i+p, and padded column c >= 1 is raw column c-1. Hence
//   shifted(i, j) = raw(i, T-1-i+j),
// i.e. the relative-position row is `T-1-i+j`. This kernel indexes rel_key
// directly by that expression instead of materialising the shift, which is why
// there is no [T, 2T-1] scratch buffer here. tests/vt/test_ops_attn_relpos.cpp
// gates the closed form against a literal pad/reshape/slice reference so the
// derivation is checked by execution, not by reading.
//
// SELF-REGISTERING translation unit (the src/vt/cpu/cpu_ops.cpp Registrar
// idiom), like src/vt/cpu/cpu_layernorm.cpp.
//
// DETERMINISM CONTRACT (gate: tests/vt/test_ops_attn_relpos.cpp). The three
// passes per (head, query) row — scores + max, exp + denominator, value
// contraction — are strictly sequential, and the parallel dispatch partitions
// (head, query) rows only. Byte-identical across thread counts, and
// byte-identical to the in-test scalar reference.
#include <cmath>
#include <limits>
#include <vector>

#include "cpu_threadpool.h"
#include "vt/ops.h"

namespace vt::cpu {
namespace {

float LoadF32At(const Tensor& t, int64_t i) {
  switch (t.dtype) {
    case DType::kF32: return t.Ptr<float>()[i];
    case DType::kF16: return F16ToF32(t.Ptr<uint16_t>()[i]);
    case DType::kBF16: return BF16ToF32(t.Ptr<uint16_t>()[i]);
    default: VT_CHECK(false, "cpu attention_relpos: unsupported input dtype"); return 0.0f;
  }
}

void StoreF32At(const Tensor& t, int64_t i, float v) {
  switch (t.dtype) {
    case DType::kF32: t.Ptr<float>()[i] = v; break;
    case DType::kF16: t.Ptr<uint16_t>()[i] = F32ToF16(v); break;
    case DType::kBF16: t.Ptr<uint16_t>()[i] = F32ToBF16(v); break;
    default: VT_CHECK(false, "cpu attention_relpos: unsupported output dtype");
  }
}

bool KeyValid(const Tensor* mask, int64_t j) {
  if (mask == nullptr) return true;
  return mask->dtype == DType::kI8 ? mask->Ptr<int8_t>()[j] != 0 : mask->Ptr<int32_t>()[j] != 0;
}

void AttentionRelPosKernel(Queue&, Tensor& out, const Tensor& query, const Tensor& key,
                           const Tensor& value, const Tensor& rel_key, const Tensor* bias_u,
                           const Tensor* bias_v, const Tensor* key_mask,
                           const AttentionRelPosArgs& args) {
  const int64_t t = query.shape[0], hq = query.shape[1], d = query.shape[2];
  const int64_t hk = key.shape[1];
  const int64_t qpk = hq / hk;  // q-heads per kv-head (GQA broadcast, repeat_kv :212)
  const float scale = args.scale;
  const float kNegInf = -std::numeric_limits<float>::infinity();
  // Row-chunked over (head, query) pairs, exactly as the dense CPU
  // AttentionKernel is (src/vt/cpu/cpu_ops.cpp) — each row owns its own output
  // slice, so the partition can never reassociate a reduction.
  ParallelForRows(CurrentThreadpool(), hq * t, [&](int64_t r0, int64_t r1) {
    std::vector<float> scores(static_cast<size_t>(t));
    std::vector<float> qu(static_cast<size_t>(d));
    std::vector<float> qv(static_cast<size_t>(d));
    std::vector<float> acc(static_cast<size_t>(d));
    for (int64_t r = r0; r < r1; ++r) {
      const int64_t h = r / t;
      const int64_t i = r % t;
      const int64_t g = h / qpk;
      const int64_t qoff = (i * hq + h) * d;
      const int64_t boff = h * d;
      // query + bias_u / bias_v, hoisted (:295-300 materialises these two
      // tensors; hoisting is the same value, computed once per query row).
      for (int64_t e = 0; e < d; ++e) {
        const float qe = LoadF32At(query, qoff + e);
        qu[static_cast<size_t>(e)] = bias_u != nullptr ? qe + LoadF32At(*bias_u, boff + e) : qe;
        qv[static_cast<size_t>(e)] = bias_v != nullptr ? qe + LoadF32At(*bias_v, boff + e) : qe;
      }
      // Pass 1: matrix_ac (a+c) and matrix_bd (b+d), combined and max-tracked.
      float m = kNegInf;
      for (int64_t j = 0; j < t; ++j) {
        if (!KeyValid(key_mask, j)) {
          scores[static_cast<size_t>(j)] = kNegInf;
          continue;
        }
        const int64_t koff = (j * hk + g) * d;
        const int64_t roff = ((t - 1 - i + j) * hq + h) * d;  // the _rel_shift index
        float ac = 0.0f;
        for (int64_t e = 0; e < d; ++e)
          ac += qu[static_cast<size_t>(e)] * LoadF32At(key, koff + e);
        float bd = 0.0f;
        for (int64_t e = 0; e < d; ++e)
          bd += qv[static_cast<size_t>(e)] * LoadF32At(rel_key, roff + e);
        const float s = args.scale_after_sum ? (ac + bd) * scale : scale * ac + scale * bd;
        scores[static_cast<size_t>(j)] = s;
        if (s > m) m = s;
      }
      if (m == kNegInf) {
        // Every key masked. Upstream would produce NaN (softmax of an all -inf
        // row); such a query row is itself padding and its output is discarded,
        // so zeros are emitted instead. Recorded deviation (include/vt/ops.h).
        for (int64_t e = 0; e < d; ++e) StoreF32At(out, qoff + e, 0.0f);
        continue;
      }
      // Pass 2: exp + denominator (f32, max-subtracted, as :251 softmax in f32).
      float denom = 0.0f;
      for (int64_t j = 0; j < t; ++j) {
        // exp(-inf - finite) == 0 exactly, so masked keys need no branch.
        const float e = std::exp(scores[static_cast<size_t>(j)] - m);
        scores[static_cast<size_t>(j)] = e;
        denom += e;
      }
      const float inv = 1.0f / denom;  // denom >= 1 (the argmax term is exp(0)==1)
      // Pass 3: p @ value (:253), f32 accumulation, one rounding on store.
      for (int64_t e = 0; e < d; ++e) acc[static_cast<size_t>(e)] = 0.0f;
      for (int64_t j = 0; j < t; ++j) {
        const float p = scores[static_cast<size_t>(j)] * inv;
        const int64_t voff = (j * hk + g) * d;
        for (int64_t e = 0; e < d; ++e)
          acc[static_cast<size_t>(e)] += p * LoadF32At(value, voff + e);
      }
      for (int64_t e = 0; e < d; ++e) StoreF32At(out, qoff + e, acc[static_cast<size_t>(e)]);
    }
  });
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kAttentionRelPos, DeviceType::kCPU,
               reinterpret_cast<void*>(static_cast<AttentionRelPosFn>(&AttentionRelPosKernel)));
  }
} registrar;

}  // namespace
}  // namespace vt::cpu
