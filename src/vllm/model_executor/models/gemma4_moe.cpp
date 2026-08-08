// Gemma-4 MoE: BF16 fused or FP8 per-expert + optional device resident.
#include "vllm/model_executor/models/gemma4_moe.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vllm/model_executor/models/dense_attn_block.h"
#include "vllm/model_executor/models/device_pool.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace vllm {
namespace {

using dense_attn::DBuf;
using dense_attn::Dev;
using dense_attn::ResidentWeight;
using vt::DType;
using vt::Tensor;

void ExpertGeGLUHost(Dev d, DBuf& out, const Tensor& x, const uint16_t* gate_up_e,
                     const uint16_t* down_e, int64_t I, int64_t H) {
  const int64_t T = x.shape[0];
  DBuf gate_w(d, DType::kBF16, {I, H}, gate_up_e);
  DBuf up_w(d, DType::kBF16, {I, H}, gate_up_e + I * H);
  DBuf down_w(d, DType::kBF16, {H, I}, down_e);
  DBuf gate(d, DType::kBF16, {T, I});
  DBuf up(d, DType::kBF16, {T, I});
  vt::MatmulBT(d.q, gate.t(), x, gate_w.t());
  vt::MatmulBT(d.q, up.t(), x, up_w.t());
  DBuf gu(d, DType::kBF16, {T, 2 * I});
  const size_t row = static_cast<size_t>(I) * sizeof(uint16_t);
  for (int64_t t = 0; t < T; ++t) {
    d.b.Copy(d.q, static_cast<char*>(gu.ptr()) + static_cast<size_t>(t) * 2 * row,
             static_cast<const char*>(gate.ptr()) + static_cast<size_t>(t) * row, row);
    d.b.Copy(d.q, static_cast<char*>(gu.ptr()) + static_cast<size_t>(t) * 2 * row + row,
             static_cast<const char*>(up.ptr()) + static_cast<size_t>(t) * row, row);
  }
  DBuf act(d, DType::kBF16, {T, I});
  vt::GeluAndMul(d.q, act.t(), gu.t());
  vt::MatmulBT(d.q, out.t(), act.t(), down_w.t());
}

void ExpertGeGLUDevice(Dev d, DBuf& out, const Tensor& x, const uint16_t* gate_up_e,
                       const uint16_t* down_e, int64_t I, int64_t H) {
  const int64_t T = x.shape[0];
  const vt::Device dev = d.q.device;
  Tensor gate_w =
      Tensor::Contiguous(const_cast<uint16_t*>(gate_up_e), DType::kBF16, dev, {I, H});
  Tensor up_w = Tensor::Contiguous(const_cast<uint16_t*>(gate_up_e + I * H), DType::kBF16,
                                   dev, {I, H});
  Tensor down_w =
      Tensor::Contiguous(const_cast<uint16_t*>(down_e), DType::kBF16, dev, {H, I});
  DBuf gate(d, DType::kBF16, {T, I});
  DBuf up(d, DType::kBF16, {T, I});
  vt::MatmulBT(d.q, gate.t(), x, gate_w);
  vt::MatmulBT(d.q, up.t(), x, up_w);
  DBuf gu(d, DType::kBF16, {T, 2 * I});
  const size_t row = static_cast<size_t>(I) * sizeof(uint16_t);
  for (int64_t t = 0; t < T; ++t) {
    d.b.Copy(d.q, static_cast<char*>(gu.ptr()) + static_cast<size_t>(t) * 2 * row,
             static_cast<const char*>(gate.ptr()) + static_cast<size_t>(t) * row, row);
    d.b.Copy(d.q, static_cast<char*>(gu.ptr()) + static_cast<size_t>(t) * 2 * row + row,
             static_cast<const char*>(up.ptr()) + static_cast<size_t>(t) * row, row);
  }
  DBuf act(d, DType::kBF16, {T, I});
  vt::GeluAndMul(d.q, act.t(), gu.t());
  vt::MatmulBT(d.q, out.t(), act.t(), down_w);
}

}  // namespace

// Ensure FP8 expert has BF16 cache filled (idempotent).
void EnsureGemma4Fp8ExpertCached(const Gemma4Fp8ExpertMats& ex, int64_t I, int64_t H) {
  if (!ex.cached_gu.empty() && !ex.cached_dn.empty() &&
      static_cast<int64_t>(ex.cached_gu.size()) == 2 * I * H &&
      static_cast<int64_t>(ex.cached_dn.size()) == H * I) {
    return;
  }
  ex.cached_gu.resize(static_cast<size_t>(2 * I * H));
  ex.cached_dn.resize(static_cast<size_t>(H * I));
  DequantFp8ChannelToBf16(ex.gate_w.bytes.data(),
                          reinterpret_cast<const uint16_t*>(ex.gate_s.bytes.data()), I, H,
                          ex.cached_gu.data());
  DequantFp8ChannelToBf16(ex.up_w.bytes.data(),
                          reinterpret_cast<const uint16_t*>(ex.up_s.bytes.data()), I, H,
                          ex.cached_gu.data() + I * H);
  DequantFp8ChannelToBf16(ex.down_w.bytes.data(),
                          reinterpret_cast<const uint16_t*>(ex.down_s.bytes.data()), H, I,
                          ex.cached_dn.data());
}

// Host BF16 cache + device upload once (subsequent tokens use device GEMM path).
bool EnsureGemma4Fp8ExpertOnDevice(Dev d, const Gemma4Fp8ExpertMats& ex, int64_t I,
                                   int64_t H) {
  EnsureGemma4Fp8ExpertCached(ex, I, H);
  if (ex.dev_gu != nullptr && ex.dev_dn != nullptr) return true;
  const size_t gu_b = static_cast<size_t>(2 * I * H) * sizeof(uint16_t);
  const size_t dn_b = static_cast<size_t>(H * I) * sizeof(uint16_t);
  void* gu = nullptr;
  void* dn = nullptr;
  try {
    gu = d.b.Alloc(gu_b);
    dn = d.b.Alloc(dn_b);
    d.b.Copy(d.q, gu, ex.cached_gu.data(), gu_b);
    d.b.Copy(d.q, dn, ex.cached_dn.data(), dn_b);
    d.b.Synchronize(d.q);
    ex.dev_gu = gu;
    ex.dev_dn = dn;
    return true;
  } catch (...) {
    if (gu) d.b.Free(gu);
    if (dn) d.b.Free(dn);
    return false;  // fall back to host H2D path
  }
}

void DequantGemma4Fp8ExpertToBf16(const Gemma4Fp8ExpertMats& ex, int64_t I, int64_t H,
                                  uint16_t* gate_up_out, uint16_t* down_out) {
  VT_CHECK(gate_up_out && down_out, "fp8 expert dequant null out");
  EnsureGemma4Fp8ExpertCached(ex, I, H);
  std::memcpy(gate_up_out, ex.cached_gu.data(), ex.cached_gu.size() * sizeof(uint16_t));
  std::memcpy(down_out, ex.cached_dn.data(), ex.cached_dn.size() * sizeof(uint16_t));
}

Gemma4MoeScratch RunGemma4Moe(vt::Queue& q, const Gemma4MoeLayerWeights& moe,
                              const vt::Tensor& router_in, const vt::Tensor& expert_in,
                              int64_t T, int64_t H, float rms_eps) {
  using dense_attn::DBuf;
  using dense_attn::Dev;
  using dense_attn::ResidentWeight;
  using vt::DType;
  using vt::Tensor;

  VT_CHECK(moe.enabled && !moe.experts.Empty(), "gemma4 moe: disabled");
  VT_CHECK(router_in.shape[0] == T && router_in.shape[1] == H, "gemma4 moe: router_in");
  VT_CHECK(expert_in.shape[0] == T && expert_in.shape[1] == H, "gemma4 moe: expert_in");
  const int64_t E = moe.experts.num_experts;
  const int64_t I = moe.experts.intermediate;
  const int top_k = moe.top_k;
  VT_CHECK(E > 0 && I > 0 && top_k > 0 && top_k <= E, "gemma4 moe: dims");
  VT_CHECK(moe.experts.hidden == H, "gemma4 moe: H mismatch");

  Dev d{vt::GetBackend(q.device.type), q};
  const vt::RmsNormArgs plain{rms_eps, false};
  const int compute_dev = q.device.index;

  DBuf rn(d, DType::kBF16, {T, H});
  {
    std::vector<uint16_t> ones(static_cast<size_t>(H), vt::F32ToBF16(1.f));
    DBuf w1(d, DType::kBF16, {H}, ones.data());
    vt::RmsNorm(d.q, rn.t(), router_in, w1.t(), plain);
  }

  const OwnedTensor& rproj =
      !moe.router_proj_fused.Empty() ? moe.router_proj_fused : moe.router_proj;
  VT_CHECK(rproj.HasHostBytes() && rproj.nk && rproj.shape[0] == E && rproj.shape[1] == H,
           "gemma4 moe: router proj");
  Tensor wp = ResidentWeight(d, rproj);
  DBuf logits(d, DType::kF32, {T, E});
  vt::MatmulBT(d.q, logits.t(), rn.t(), wp);
  d.b.Synchronize(d.q);
  std::vector<float> hlog(static_cast<size_t>(T * E));
  d.b.Copy(d.q, hlog.data(), logits.ptr(), hlog.size() * sizeof(float));
  d.b.Synchronize(d.q);

  std::vector<float> hscale(static_cast<size_t>(E), 1.f);
  if (moe.per_expert_scale.HasHostBytes()) {
    const auto* pe = reinterpret_cast<const uint16_t*>(moe.per_expert_scale.bytes.data());
    for (int64_t e = 0; e < E; ++e) hscale[static_cast<size_t>(e)] = vt::BF16ToF32(pe[e]);
  }

  const auto& ex = moe.experts;
  const int64_t gu_stride = 2 * I * H;
  const int64_t dn_stride = H * I;
  const bool same_dev =
      ex.gate_up_dev != nullptr && ex.down_dev != nullptr && ex.dev_id == compute_dev;

  const auto* gu_host = ex.gate_up.Empty()
                            ? nullptr
                            : reinterpret_cast<const uint16_t*>(ex.gate_up.bytes.data());
  const auto* dn_host =
      ex.down.Empty() ? nullptr : reinterpret_cast<const uint16_t*>(ex.down.bytes.data());

  DBuf acc(d, DType::kBF16, {T, H});
  acc.Zero(d);

  for (int64_t t = 0; t < T; ++t) {
    std::vector<int> idx(static_cast<size_t>(E));
    for (int e = 0; e < static_cast<int>(E); ++e) idx[static_cast<size_t>(e)] = e;
    std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(), [&](int a, int b) {
      return hlog[static_cast<size_t>(t * E + a)] > hlog[static_cast<size_t>(t * E + b)];
    });
    float mx = hlog[static_cast<size_t>(t * E + idx[0])];
    std::vector<float> wts(static_cast<size_t>(top_k));
    float sum = 0.f;
    for (int i = 0; i < top_k; ++i) {
      wts[static_cast<size_t>(i)] =
          std::exp(hlog[static_cast<size_t>(t * E + idx[static_cast<size_t>(i)])] - mx);
      sum += wts[static_cast<size_t>(i)];
    }
    for (int i = 0; i < top_k; ++i)
      wts[static_cast<size_t>(i)] =
          (wts[static_cast<size_t>(i)] / sum) *
          hscale[static_cast<size_t>(idx[static_cast<size_t>(i)])];

    DBuf xin(d, DType::kBF16, {1, H});
    d.b.Copy(d.q, xin.ptr(),
             static_cast<const char*>(expert_in.data) +
                 static_cast<size_t>(t) * static_cast<size_t>(H) * 2,
             static_cast<size_t>(H) * 2);

    DBuf ysum(d, DType::kBF16, {1, H});
    ysum.Zero(d);
    for (int i = 0; i < top_k; ++i) {
      const int e = idx[static_cast<size_t>(i)];
      DBuf y(d, DType::kBF16, {1, H});
      if (same_dev) {
        auto* gu = static_cast<const uint16_t*>(ex.gate_up_dev) +
                   static_cast<int64_t>(e) * gu_stride;
        auto* dn =
            static_cast<const uint16_t*>(ex.down_dev) + static_cast<int64_t>(e) * dn_stride;
        ExpertGeGLUDevice(d, y, xin.t(), gu, dn, I, H);
      } else if (ex.is_fp8) {
        const auto& fex = ex.fp8[static_cast<size_t>(e)];
        if (EnsureGemma4Fp8ExpertOnDevice(d, fex, I, H)) {
          ExpertGeGLUDevice(d, y, xin.t(), static_cast<const uint16_t*>(fex.dev_gu),
                            static_cast<const uint16_t*>(fex.dev_dn), I, H);
        } else {
          EnsureGemma4Fp8ExpertCached(fex, I, H);
          ExpertGeGLUHost(d, y, xin.t(), fex.cached_gu.data(), fex.cached_dn.data(), I, H);
        }
      } else {
        ExpertGeGLUHost(d, y, xin.t(), gu_host + static_cast<int64_t>(e) * gu_stride,
                        dn_host + static_cast<int64_t>(e) * dn_stride, I, H);
      }
      d.b.Synchronize(d.q);
      std::vector<uint16_t> hy(static_cast<size_t>(H)), hs(static_cast<size_t>(H));
      d.b.Copy(d.q, hy.data(), y.ptr(), hy.size() * 2);
      d.b.Copy(d.q, hs.data(), ysum.ptr(), hs.size() * 2);
      d.b.Synchronize(d.q);
      const float ww = wts[static_cast<size_t>(i)];
      for (int64_t j = 0; j < H; ++j)
        hs[static_cast<size_t>(j)] = vt::F32ToBF16(
            vt::BF16ToF32(hs[static_cast<size_t>(j)]) +
            ww * vt::BF16ToF32(hy[static_cast<size_t>(j)]));
      d.b.Copy(d.q, ysum.ptr(), hs.data(), hs.size() * 2);
    }
    d.b.Copy(d.q,
             static_cast<char*>(acc.ptr()) + static_cast<size_t>(t) * static_cast<size_t>(H) * 2,
             ysum.ptr(), static_cast<size_t>(H) * 2);
  }

  Gemma4MoeScratch r;
  r.tensor = acc.t();
  const size_t alloc = acc.alloc_bytes();
  void* p = acc.Release();
  r.storage = std::shared_ptr<void>(p, [alloc](void* q) { Pool().Put(alloc, q); });
  return r;
}

#ifndef VLLM_CPP_HIP
// Resident-expert preload is a discrete-ROCm optimization; its real
// implementation lives in src/vt/rocm/rocm_gemma4_experts.hip and is compiled
// only under -DVLLM_CPP_HIP. Non-HIP builds need these symbols to LINK (the
// call site in gemma4_registry.cpp is gated on VT_GEMMA4_RESIDENT_EXPERTS=1 and
// is never reached off-ROCm, but the reference must still resolve). Loud no-op:
// if a caller ever asks for resident experts without a HIP backend, say so.
size_t UploadGemma4ExpertsResident(std::vector<Gemma4MoeLayerWeights>& layers,
                                   int num_gpus) {
  (void)layers;
  (void)num_gpus;
  std::fprintf(stderr,
               "[gemma4] VT_GEMMA4_RESIDENT_EXPERTS requested but this binary "
               "was built without -DVLLM_CPP_HIP; resident preload is a no-op.\n");
  return 0;
}
size_t UploadGemma4ExpertsResidentForWeights(Gemma4Weights& weights,
                                             int num_gpus) {
  (void)weights;
  (void)num_gpus;
  std::fprintf(stderr,
               "[gemma4] VT_GEMMA4_RESIDENT_EXPERTS requested but this binary "
               "was built without -DVLLM_CPP_HIP; resident preload is a no-op.\n");
  return 0;
}
#endif  // VLLM_CPP_HIP

}  // namespace vllm
