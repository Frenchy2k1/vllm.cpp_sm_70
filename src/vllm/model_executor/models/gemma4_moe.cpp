// Gemma-4 MoE: BF16 fused or FP8 per-expert + optional device resident.
#include "vllm/model_executor/models/gemma4_moe.h"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vllm/model_executor/models/dense_attn_block.h"
#include "vllm/model_executor/models/device_pool.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/fused_ops.h"

namespace vllm {
namespace {

using dense_attn::DBuf;
using dense_attn::Dev;
using dense_attn::ResidentWeight;
using vt::DType;
using vt::Tensor;

// Scratch reused across top-k experts within a token (and host H2D weight slots).
struct ExpertScratch {
  DBuf gu;    // [T, 2I] fused gate|up activations
  DBuf act;   // [T, I]
  DBuf gu_w;  // host-path [2I, H] weight upload
  DBuf down_w;
  ExpertScratch(Dev d, int64_t T, int64_t I, int64_t H)
      : gu(d, DType::kBF16, {T, 2 * I}),
        act(d, DType::kBF16, {T, I}),
        gu_w(d, DType::kBF16, {2 * I, H}),
        down_w(d, DType::kBF16, {H, I}) {}
};

void ExpertGeGLUHost(Dev d, DBuf& out, const Tensor& x, const uint16_t* gate_up_e,
                     const uint16_t* down_e, int64_t I, int64_t H, ExpertScratch& s) {
  const size_t gu_b = static_cast<size_t>(2 * I * H) * sizeof(uint16_t);
  const size_t dn_b = static_cast<size_t>(H * I) * sizeof(uint16_t);
  d.b.Copy(d.q, s.gu_w.ptr(), gate_up_e, gu_b);
  d.b.Copy(d.q, s.down_w.ptr(), down_e, dn_b);
  // One GEMM: x @ W_gu^T -> [T, 2I], then GeluAndMul (interleaved gate|up).
  vt::MatmulBT(d.q, s.gu.t(), x, s.gu_w.t());
  vt::GeluAndMul(d.q, s.act.t(), s.gu.t());
  vt::MatmulBT(d.q, out.t(), s.act.t(), s.down_w.t());
}

void ExpertGeGLUDeviceAccum(Dev d, DBuf& out, const Tensor& x, const uint16_t* gate_up_e,
                            const uint16_t* down_e, int64_t I, int64_t H, ExpertScratch& s,
                            float alpha, float beta) {
  const int64_t T = x.shape[0];
  const vt::Device dev = d.q.device;
  // gate_up_e is contiguous [2I, H] — one BT GEMM instead of two.
  Tensor gu_w =
      Tensor::Contiguous(const_cast<uint16_t*>(gate_up_e), DType::kBF16, dev, {2 * I, H});
  vt::MatmulBT(d.q, s.gu.t(), x, gu_w);
  vt::GeluAndMul(d.q, s.act.t(), s.gu.t());
  vt::MatmulBTAlphaBeta(d.q, out.ptr(), s.act.ptr(), down_e, static_cast<int>(T),
                                  static_cast<int>(H), static_cast<int>(I), alpha, beta,
                                  DType::kBF16);
}

void ExpertGeGLUFp8Native(Dev d, DBuf& out, const Tensor& x, const void* fp8_gu,
                          const void* s_gu, const void* fp8_dn, const void* s_dn, int64_t I,
                          int64_t H, ExpertScratch& s, float alpha, float beta) {
  VT_CHECK(x.shape[0] == 1, "fp8 native: T==1 only");
  vt::MatmulBTFp8Channel(d.q, s.gu.ptr(), x.data, fp8_gu, s_gu, /*M=*/1,
                                   static_cast<int>(2 * I), static_cast<int>(H), 1.f, 0.f);
  vt::GeluAndMul(d.q, s.act.t(), s.gu.t());
  if (beta == 0.f) {
    vt::MatmulBTFp8Channel(d.q, out.ptr(), s.act.ptr(), fp8_dn, s_dn, /*M=*/1,
                                     static_cast<int>(H), static_cast<int>(I), alpha, 0.f);
  } else {
    DBuf ytmp(d, DType::kBF16, {1, H});
    vt::MatmulBTFp8Channel(d.q, ytmp.ptr(), s.act.ptr(), fp8_dn, s_dn, /*M=*/1,
                                     static_cast<int>(H), static_cast<int>(I), 1.f, 0.f);
    vt::MulScalar(d.q, out.t(), out.t(), static_cast<double>(beta));
    DBuf ysc(d, DType::kBF16, {1, H});
    vt::MulScalar(d.q, ysc.t(), ytmp.t(), static_cast<double>(alpha));
    vt::Add(d.q, out.t(), out.t(), ysc.t());
  }
}

// Top-k experts: all gate_up GEMMs → one GeluAndMul → all down GEMMs (alpha/beta mix).
// Cuts (top_k-1) Gelu launches vs per-expert ExpertGeGLUDeviceAccum.
// Uses MatmulBTAlphaBetaRocm directly (no vt::MatmulBT dispatch overhead).
bool ExpertGeGLUTopKFusedGelu(Dev d, DBuf& ysum, const Tensor& x, const uint16_t* const* gu_ptrs,
                              const uint16_t* const* dn_ptrs, const float* wts, int G, int64_t I,
                              int64_t H) {
  if (G <= 0 || x.shape[0] != 1) return false;
  struct Tls {
    int dev = -1;
    int Gcap = 0;
    int64_t I = 0, H = 0;
    std::optional<DBuf> gu;   // [G, 2I]
    std::optional<DBuf> act;  // [G, I]
  };
  static thread_local Tls tls;
  if (tls.dev != d.q.device.index || tls.Gcap < G || tls.I != I || tls.H != H) {
    tls.gu.emplace(d, DType::kBF16, std::vector<int64_t>{G, 2 * I});
    tls.act.emplace(d, DType::kBF16, std::vector<int64_t>{G, I});
    tls.dev = d.q.device.index;
    tls.Gcap = G;
    tls.I = I;
    tls.H = H;
  }
  const vt::Device dev = d.q.device;
  const size_t gu_row = static_cast<size_t>(2 * I) * 2;
  const size_t act_row = static_cast<size_t>(I) * 2;
  const int Ngu = static_cast<int>(2 * I);
  const int Nh = static_cast<int>(H);
  const int Ki = static_cast<int>(I);
  const int Kh = static_cast<int>(H);

  // Phase 1: gate_up GEMMs into packed [G, 2I]
  for (int g = 0; g < G; ++g) {
    void* gu_out = static_cast<char*>(tls.gu->ptr()) + static_cast<size_t>(g) * gu_row;
    vt::MatmulBTAlphaBeta(d.q, gu_out, x.data, gu_ptrs[g], /*M=*/1, Ngu, Kh, 1.f, 0.f,
                                    DType::kBF16);
  }

  // Phase 2: single GeluAndMul over all experts
  Tensor gu_all = Tensor::Contiguous(static_cast<uint16_t*>(tls.gu->ptr()), DType::kBF16, dev,
                                     {G, 2 * I});
  Tensor act_all = Tensor::Contiguous(static_cast<uint16_t*>(tls.act->ptr()), DType::kBF16, dev,
                                      {G, I});
  vt::GeluAndMul(d.q, act_all, gu_all);

  // Phase 3: down GEMMs with alpha/beta accumulate into ysum
  for (int g = 0; g < G; ++g) {
    const float alpha = wts[g];
    const float beta = (g == 0) ? 0.f : 1.f;
    void* act_g = static_cast<char*>(tls.act->ptr()) + static_cast<size_t>(g) * act_row;
    vt::MatmulBTAlphaBeta(d.q, ysum.ptr(), act_g, dn_ptrs[g], /*M=*/1, Nh, Ki, alpha,
                                    beta, DType::kBF16);
  }
  return true;
}

// Batched top-k path (gather+strided or pointer-batch): currently disabled.
// Lab: gather+strided produced wrong tokens (~23 t/s); pointer-batch ~0.8 t/s.
// Serial / fused-gelu top-k remains the correct path (~34 t/s).
bool ExpertGeGLUDeviceBatched(Dev /*d*/, DBuf& /*ysum*/, const Tensor& /*x*/,
                              const std::vector<const uint16_t*>& /*gu_ptrs*/,
                              const std::vector<const uint16_t*>& /*dn_ptrs*/,
                              const std::vector<float>& /*wts*/, int64_t /*I*/, int64_t /*H*/) {
  return false;
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
  PinGemma4Fp8ExpertHostCache(ex);
}

// Dequant into caller buffers without retaining a permanent host BF16 cache.
// Used by dual-GPU resident upload (must not pin ~1.5GiB/layer on host).
void DequantGemma4Fp8ExpertToBf16Ephemeral(const Gemma4Fp8ExpertMats& ex, int64_t I,
                                           int64_t H, uint16_t* gate_up_out,
                                           uint16_t* down_out) {
  VT_CHECK(gate_up_out && down_out, "fp8 expert ephemeral dequant null out");
  if (!ex.cached_gu.empty() && !ex.cached_dn.empty() &&
      static_cast<int64_t>(ex.cached_gu.size()) == 2 * I * H &&
      static_cast<int64_t>(ex.cached_dn.size()) == H * I) {
    std::memcpy(gate_up_out, ex.cached_gu.data(), ex.cached_gu.size() * sizeof(uint16_t));
    std::memcpy(down_out, ex.cached_dn.data(), ex.cached_dn.size() * sizeof(uint16_t));
    return;
  }
  DequantFp8ChannelToBf16(ex.gate_w.bytes.data(),
                          reinterpret_cast<const uint16_t*>(ex.gate_s.bytes.data()), I, H,
                          gate_up_out);
  DequantFp8ChannelToBf16(ex.up_w.bytes.data(),
                          reinterpret_cast<const uint16_t*>(ex.up_s.bytes.data()), I, H,
                          gate_up_out + I * H);
  DequantFp8ChannelToBf16(ex.down_w.bytes.data(),
                          reinterpret_cast<const uint16_t*>(ex.down_s.bytes.data()), H, I,
                          down_out);
}

// Host BF16 cache + device upload once (subsequent tokens use device GEMM path).
// H2D is async on d.q — later GEMMs on the same stream see the data without a
// device-wide Synchronize (was serializing every expert upload).
// VRAM budget: VT_GEMMA4_EXPERT_VRAM_MB (default 12288). LRU evicts oldest.
namespace {
struct DevExpertLru {
  struct Slot {
    const Gemma4Fp8ExpertMats* ex = nullptr;
    void* gu = nullptr;
    void* dn = nullptr;
    void* fp8_gu = nullptr;
    void* fp8_dn = nullptr;
    void* s_gu = nullptr;
    void* s_dn = nullptr;
    size_t bytes = 0;
    uint64_t tick = 0;
  };
  std::vector<Slot> slots;
  size_t used = 0;
  size_t budget = 0;
  uint64_t tick = 1;
  int dev = -1;

  size_t BudgetBytes() {
    if (budget) return budget;
    size_t mb = 0;
    if (const char* e = std::getenv("VT_GEMMA4_EXPERT_VRAM_MB")) {
      const long v = std::strtol(e, nullptr, 10);
      if (v >= 0) mb = static_cast<size_t>(v);
    }
    budget = mb == 0 ? static_cast<size_t>(-1) : mb * 1024ull * 1024ull;
    return budget;
  }

  void EvictOne(Dev d) {
    if (slots.empty()) return;
    size_t victim = 0;
    for (size_t i = 1; i < slots.size(); ++i)
      if (slots[i].tick < slots[victim].tick) victim = i;
    Slot s = slots[victim];
    if (s.gu) d.b.Free(s.gu);
    if (s.dn) d.b.Free(s.dn);
    if (s.fp8_gu) d.b.Free(s.fp8_gu);
    if (s.fp8_dn) d.b.Free(s.fp8_dn);
    if (s.s_gu) d.b.Free(s.s_gu);
    if (s.s_dn) d.b.Free(s.s_dn);
    if (s.ex) {
      s.ex->dev_gu = nullptr;
      s.ex->dev_dn = nullptr;
      s.ex->dev_fp8_gu = nullptr;
      s.ex->dev_fp8_dn = nullptr;
      s.ex->dev_s_gu = nullptr;
      s.ex->dev_s_dn = nullptr;
    }
    used = used >= s.bytes ? used - s.bytes : 0;
    slots.erase(slots.begin() + static_cast<std::ptrdiff_t>(victim));
  }

  void Note(const Gemma4Fp8ExpertMats* ex, void* gu, void* dn, size_t bytes, Dev d,
            void* fp8_gu = nullptr, void* fp8_dn = nullptr, void* s_gu = nullptr,
            void* s_dn = nullptr) {
    if (dev != d.q.device.index) {
      slots.clear();
      used = 0;
      dev = d.q.device.index;
    }
    const size_t bud = BudgetBytes();
    while (used + bytes > bud && !slots.empty()) EvictOne(d);
    if (used + bytes > bud) return;
    slots.push_back(Slot{ex, gu, dn, fp8_gu, fp8_dn, s_gu, s_dn, bytes, tick++});
    used += bytes;
  }

  void Touch(const Gemma4Fp8ExpertMats* ex) {
    for (auto& s : slots) {
      if (s.ex == ex) {
        s.tick = tick++;
        return;
      }
    }
  }
};

DevExpertLru& ExpertLru() {
  static DevExpertLru lru;
  return lru;
}
}  // namespace

bool EnsureGemma4Fp8ExpertOnDevice(Dev d, const Gemma4Fp8ExpertMats& ex, int64_t I,
                                   int64_t H) {
  EnsureGemma4Fp8ExpertCached(ex, I, H);
  if (ex.dev_gu != nullptr && ex.dev_dn != nullptr) {
    ExpertLru().Touch(&ex);
    return true;
  }
  const size_t gu_b = static_cast<size_t>(2 * I * H) * sizeof(uint16_t);
  const size_t dn_b = static_cast<size_t>(H * I) * sizeof(uint16_t);
  const size_t total = gu_b + dn_b;
  void* gu = nullptr;
  void* dn = nullptr;
  try {
    auto& lru = ExpertLru();
    while (lru.used + total > lru.BudgetBytes() && !lru.slots.empty()) lru.EvictOne(d);
    gu = d.b.Alloc(gu_b);
    dn = d.b.Alloc(dn_b);
    d.b.Copy(d.q, gu, ex.cached_gu.data(), gu_b);
    d.b.Copy(d.q, dn, ex.cached_dn.data(), dn_b);
    ex.dev_gu = gu;
    ex.dev_dn = dn;
    lru.Note(&ex, gu, dn, total, d);
    return true;
  } catch (...) {
    if (gu) d.b.Free(gu);
    if (dn) d.b.Free(dn);
    static std::atomic<int> fails{0};
    const int n = fails.fetch_add(1) + 1;
    if (n == 1 || n % 64 == 0)
      std::fprintf(stderr, "gemma4 moe: device expert upload fail #%d (falling back to H2D)\n",
                   n);
    return false;
  }
}

// Upload FP8 weights + channel scales (no BF16 dequant). Half weight VRAM vs BF16 path.
bool EnsureGemma4Fp8NativeOnDevice(Dev d, const Gemma4Fp8ExpertMats& ex, int64_t I, int64_t H) {
  if (ex.dev_fp8_gu && ex.dev_fp8_dn && ex.dev_s_gu && ex.dev_s_dn) {
    ExpertLru().Touch(&ex);
    return true;
  }
  VT_CHECK(ex.gate_w.HasHostBytes() && ex.up_w.HasHostBytes() && ex.down_w.HasHostBytes(),
           "fp8 native: missing weights");
  VT_CHECK(ex.gate_s.HasHostBytes() && ex.up_s.HasHostBytes() && ex.down_s.HasHostBytes(),
           "fp8 native: missing scales");
  const size_t gu_b = static_cast<size_t>(2 * I * H);       // u8
  const size_t dn_b = static_cast<size_t>(H * I);           // u8
  const size_t sgu_b = static_cast<size_t>(2 * I) * 2;      // bf16
  const size_t sdn_b = static_cast<size_t>(H) * 2;          // bf16
  const size_t total = gu_b + dn_b + sgu_b + sdn_b;
  void *fgu = nullptr, *fdn = nullptr, *sgu = nullptr, *sdn = nullptr;
  try {
    auto& lru = ExpertLru();
    while (lru.used + total > lru.BudgetBytes() && !lru.slots.empty()) lru.EvictOne(d);
    fgu = d.b.Alloc(gu_b);
    fdn = d.b.Alloc(dn_b);
    sgu = d.b.Alloc(sgu_b);
    sdn = d.b.Alloc(sdn_b);
    // Pack gate|up FP8 rows
    d.b.Copy(d.q, fgu, ex.gate_w.bytes.data(), static_cast<size_t>(I * H));
    d.b.Copy(d.q, static_cast<char*>(fgu) + static_cast<size_t>(I * H), ex.up_w.bytes.data(),
             static_cast<size_t>(I * H));
    d.b.Copy(d.q, fdn, ex.down_w.bytes.data(), dn_b);
    d.b.Copy(d.q, sgu, ex.gate_s.bytes.data(), static_cast<size_t>(I) * 2);
    d.b.Copy(d.q, static_cast<char*>(sgu) + static_cast<size_t>(I) * 2, ex.up_s.bytes.data(),
             static_cast<size_t>(I) * 2);
    d.b.Copy(d.q, sdn, ex.down_s.bytes.data(), sdn_b);
    ex.dev_fp8_gu = fgu;
    ex.dev_fp8_dn = fdn;
    ex.dev_s_gu = sgu;
    ex.dev_s_dn = sdn;
    lru.Note(&ex, nullptr, nullptr, total, d, fgu, fdn, sgu, sdn);
    return true;
  } catch (...) {
    if (fgu) d.b.Free(fgu);
    if (fdn) d.b.Free(fdn);
    if (sgu) d.b.Free(sgu);
    if (sdn) d.b.Free(sdn);
    return false;
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

  static const bool profile = [] {
    const char* e = std::getenv("VT_GEMMA4_PROFILE");
    return e && e[0] == '1';
  }();
  using clock = std::chrono::steady_clock;
  const auto t_all0 = profile ? clock::now() : clock::time_point{};

  DBuf rn(d, DType::kBF16, {T, H});
  // Identity RMS weight (ones) — TLS, upload once (was H2D every layer/token).
  {
    struct OnesTls {
      int dev = -1;
      int64_t H = 0;
      std::optional<DBuf> w;
    };
    static thread_local OnesTls ot;
    if (ot.dev != compute_dev || ot.H != H || !ot.w) {
      std::vector<uint16_t> ones(static_cast<size_t>(H), vt::F32ToBF16(1.f));
      ot.w.emplace(d, DType::kBF16, std::vector<int64_t>{H}, ones.data());
      ot.dev = compute_dev;
      ot.H = H;
    }
    vt::RmsNorm(d.q, rn.t(), router_in, ot.w->t(), plain);
  }

  const OwnedTensor& rproj =
      !moe.router_proj_fused.Empty() ? moe.router_proj_fused : moe.router_proj;
  VT_CHECK(rproj.HasHostBytes() && rproj.nk && rproj.shape[0] == E && rproj.shape[1] == H,
           "gemma4 moe: router proj");
  Tensor wp = ResidentWeight(d, rproj);
  DBuf logits(d, DType::kF32, {T, E});
  vt::MatmulBT(d.q, logits.t(), rn.t(), wp);

  // Device router top-k (softmax + greedy). Only D2H [T,K] weights/indices.
  DBuf rw(d, DType::kF32, {T, top_k});
  DBuf ri(d, DType::kI32, {T, top_k});
  vt::MoeRouterTopKArgs rargs;
  rargs.top_k = top_k;
  rargs.renormalize = true;
  vt::MoeRouterTopK(d.q, rw.t(), ri.t(), logits.t(), rargs);

  std::vector<float> hw(static_cast<size_t>(T * top_k));
  std::vector<int32_t> hi(static_cast<size_t>(T * top_k));
  d.b.Copy(d.q, hw.data(), rw.ptr(), hw.size() * sizeof(float));
  d.b.Copy(d.q, hi.data(), ri.ptr(), hi.size() * sizeof(int32_t));
  d.b.Synchronize(d.q);

  const auto t_router1 = profile ? clock::now() : clock::time_point{};

  std::vector<float> hscale(static_cast<size_t>(E), 1.f);
  if (moe.per_expert_scale.HasHostBytes()) {
    const auto* pe = reinterpret_cast<const uint16_t*>(moe.per_expert_scale.bytes.data());
    for (int64_t e = 0; e < E; ++e) hscale[static_cast<size_t>(e)] = vt::BF16ToF32(pe[e]);
  }
  // Apply per-expert scale to selected weights.
  for (int64_t t = 0; t < T; ++t) {
    for (int i = 0; i < top_k; ++i) {
      const size_t o = static_cast<size_t>(t * top_k + i);
      const int e = hi[o];
      if (e >= 0 && e < static_cast<int>(E)) hw[o] *= hscale[static_cast<size_t>(e)];
    }
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

  // Reuse MoE decode scratch across layers (30 layers × every token was thrashing the pool).
  struct MoeTlsScratch {
    int dev = -1;
    int64_t I = 0, H = 0;
    std::unique_ptr<ExpertScratch> esc;
    std::optional<DBuf> xin, ysum, y, ysc;
    std::optional<DBuf> gu_sc, dn_sc;
    bool have_peer = false;
    std::vector<uint16_t> gu_tmp, dn_tmp;
  };
  static thread_local MoeTlsScratch tls;
  if (tls.dev != compute_dev || tls.I != I || tls.H != H) {
    tls.esc = std::make_unique<ExpertScratch>(d, /*T=*/1, I, H);
    tls.xin.emplace(d, DType::kBF16, std::vector<int64_t>{1, H});
    tls.ysum.emplace(d, DType::kBF16, std::vector<int64_t>{1, H});
    tls.y.emplace(d, DType::kBF16, std::vector<int64_t>{1, H});
    tls.ysc.emplace(d, DType::kBF16, std::vector<int64_t>{1, H});
    tls.gu_sc.reset();
    tls.dn_sc.reset();
    tls.have_peer = false;
    tls.gu_tmp.clear();
    tls.dn_tmp.clear();
    tls.dev = compute_dev;
    tls.I = I;
    tls.H = H;
  }
  ExpertScratch& esc = *tls.esc;
  DBuf& xin = *tls.xin;
  DBuf& ysum = *tls.ysum;
  DBuf& y = *tls.y;
  DBuf& ysc = *tls.ysc;
  const bool need_peer_sc =
      ex.gate_up_dev != nullptr && ex.down_dev != nullptr && !same_dev;
  if (need_peer_sc && !tls.have_peer) {
    tls.gu_sc.emplace(d, DType::kBF16, std::vector<int64_t>{2 * I, H});
    tls.dn_sc.emplace(d, DType::kBF16, std::vector<int64_t>{H, I});
    tls.have_peer = true;
  }
  std::optional<DBuf>& gu_sc = tls.gu_sc;
  std::optional<DBuf>& dn_sc = tls.dn_sc;
  if (ex.is_fp8 && tls.gu_tmp.size() != static_cast<size_t>(gu_stride)) {
    tls.gu_tmp.resize(static_cast<size_t>(gu_stride));
    tls.dn_tmp.resize(static_cast<size_t>(dn_stride));
  }
  std::vector<uint16_t>& gu_tmp = tls.gu_tmp;
  std::vector<uint16_t>& dn_tmp = tls.dn_tmp;
  static const bool host_axpy = [] {
    const char* e = std::getenv("VT_GEMMA4_HOST_AXPY");
    return e && e[0] == '1';
  }();
  static const bool batch_experts = [] {
    const char* e = std::getenv("VT_GEMMA4_BATCH_EXPERTS");
    return e && e[0] == '1';
  }();
  static const bool fp8_native = [] {
    const char* e = std::getenv("VT_GEMMA4_FP8_NATIVE");
    return e && e[0] == '1';
  }();
  static const bool custom_expert = [] {
    const char* e = std::getenv("VT_GEMMA4_CUSTOM_EXPERT");
    return e && e[0] == '1';
  }();
  std::vector<uint16_t> hsum;
  if (host_axpy) hsum.assign(static_cast<size_t>(H), vt::F32ToBF16(0.f));

  for (int64_t t = 0; t < T; ++t) {
    std::vector<int> idx(static_cast<size_t>(top_k));
    std::vector<float> wts(static_cast<size_t>(top_k));
    for (int i = 0; i < top_k; ++i) {
      const size_t o = static_cast<size_t>(t * top_k + i);
      idx[static_cast<size_t>(i)] = static_cast<int>(hi[o]);
      wts[static_cast<size_t>(i)] = hw[o];
    }

    d.b.Copy(d.q, xin.ptr(),
             static_cast<const char*>(expert_in.data) +
                 static_cast<size_t>(t) * static_cast<size_t>(H) * 2,
             static_cast<size_t>(H) * 2);

    if (host_axpy) {
      std::fill(hsum.begin(), hsum.end(), vt::F32ToBF16(0.f));
    }
    // device path: first expert MulScalar writes ysum (no Zero needed)

    // Prefetch BF16 caches for this token's top-k experts in parallel (cold only).
    if (ex.is_fp8 && !same_dev && ex.gate_up_dev == nullptr) {
      bool any_cold = false;
      for (int i = 0; i < top_k; ++i) {
        const auto& fex = ex.fp8[static_cast<size_t>(idx[static_cast<size_t>(i)])];
        if (fex.cached_gu.empty() || fex.cached_dn.empty()) {
          any_cold = true;
          break;
        }
      }
      if (any_cold) {
        Fp8DequantBeginOuterParallel();
        std::vector<std::thread> pref;
        pref.reserve(static_cast<size_t>(top_k));
        for (int i = 0; i < top_k; ++i) {
          const int e = idx[static_cast<size_t>(i)];
          pref.emplace_back([&, e] {
            EnsureGemma4Fp8ExpertCached(ex.fp8[static_cast<size_t>(e)], I, H);
          });
        }
        for (auto& th : pref) th.join();
        Fp8DequantEndOuterParallel();
      }
    }

    // Prefetch: queue device expert H2D for all top-k before any GEMM (same stream).
    // Skip when fused resident packs exist (same_dev or peer) — those are the source of truth
    // and VRAM is already tight after full resident upload.
    if (ex.is_fp8 && !same_dev && ex.gate_up_dev == nullptr) {
      for (int i = 0; i < top_k; ++i) {
        const int e = idx[static_cast<size_t>(i)];
        if (e >= 0 && e < static_cast<int>(E)) {
          if (fp8_native)
            (void)EnsureGemma4Fp8NativeOnDevice(d, ex.fp8[static_cast<size_t>(e)], I, H);
          else
            (void)EnsureGemma4Fp8ExpertOnDevice(d, ex.fp8[static_cast<size_t>(e)], I, H);
        }
      }
    }

    // Fused top-k ExpertGeGLU (VT_GEMMA4_FUSED_EXPERTS=1).
    if (!host_axpy && ex.is_fp8 && T == 1) {
      std::vector<const uint16_t*> gu_ptrs, dn_ptrs;
      gu_ptrs.reserve(static_cast<size_t>(top_k));
      dn_ptrs.reserve(static_cast<size_t>(top_k));
      bool all_dev = true;
      for (int i = 0; i < top_k; ++i) {
        const int e = idx[static_cast<size_t>(i)];
        const auto& fex = ex.fp8[static_cast<size_t>(e)];
        if (!fex.dev_gu || !fex.dev_dn) {
          all_dev = false;
          break;
        }
        gu_ptrs.push_back(static_cast<const uint16_t*>(fex.dev_gu));
        dn_ptrs.push_back(static_cast<const uint16_t*>(fex.dev_dn));
      }
      if (all_dev &&
          RunGemma4FusedTopkExpertGeGLU(d.q, ysum.ptr(), xin.ptr(), gu_ptrs.data(),
                                        dn_ptrs.data(), wts.data(), top_k, I, H)) {
        d.b.Copy(
            d.q,
            static_cast<char*>(acc.ptr()) + static_cast<size_t>(t) * static_cast<size_t>(H) * 2,
            ysum.ptr(), static_cast<size_t>(H) * 2);
        continue;
      }
    }

    // Batched path: VT_GEMMA4_BATCH_EXPERTS=1 (default off).
    if (batch_experts && ex.is_fp8 && !host_axpy) {
      std::vector<const uint16_t*> gu_ptrs;
      std::vector<const uint16_t*> dn_ptrs;
      gu_ptrs.reserve(static_cast<size_t>(top_k));
      dn_ptrs.reserve(static_cast<size_t>(top_k));
      bool all_dev = true;
      for (int i = 0; i < top_k; ++i) {
        const int e = idx[static_cast<size_t>(i)];
        const auto& fex = ex.fp8[static_cast<size_t>(e)];
        if (!EnsureGemma4Fp8ExpertOnDevice(d, fex, I, H)) {
          all_dev = false;
          break;
        }
        gu_ptrs.push_back(static_cast<const uint16_t*>(fex.dev_gu));
        dn_ptrs.push_back(static_cast<const uint16_t*>(fex.dev_dn));
      }
      if (all_dev && ExpertGeGLUDeviceBatched(d, ysum, xin.t(), gu_ptrs, dn_ptrs, wts, I, H)) {
        d.b.Copy(
            d.q,
            static_cast<char*>(acc.ptr()) + static_cast<size_t>(t) * static_cast<size_t>(H) * 2,
            ysum.ptr(), static_cast<size_t>(H) * 2);
        continue;  // next token
      }
    }

    // Fused-Gelu top-k: gate_up×G → one GeluAndMul → down×G (default BF16 device path).
    // Optional custom RDNA4 expert kernels: VT_GEMMA4_CUSTOM_EXPERT=1
    if (!host_axpy && T == 1 && !fp8_native) {
      std::vector<const uint16_t*> gu_p, dn_p;
      gu_p.reserve(static_cast<size_t>(top_k));
      dn_p.reserve(static_cast<size_t>(top_k));
      bool ok = true;
      for (int i = 0; i < top_k && ok; ++i) {
        const int e = idx[static_cast<size_t>(i)];
        if (same_dev) {
          gu_p.push_back(static_cast<const uint16_t*>(ex.gate_up_dev) +
                         static_cast<int64_t>(e) * gu_stride);
          dn_p.push_back(static_cast<const uint16_t*>(ex.down_dev) +
                         static_cast<int64_t>(e) * dn_stride);
        } else if (ex.is_fp8) {
          const auto& fex = ex.fp8[static_cast<size_t>(e)];
          if (!fex.dev_gu || !fex.dev_dn) {
            if (!EnsureGemma4Fp8ExpertOnDevice(d, fex, I, H)) {
              ok = false;
              break;
            }
          }
          gu_p.push_back(static_cast<const uint16_t*>(fex.dev_gu));
          dn_p.push_back(static_cast<const uint16_t*>(fex.dev_dn));
        } else {
          ok = false;
        }
      }
      if (ok && static_cast<int>(gu_p.size()) == top_k) {
        bool ran = false;
        if (custom_expert) {
          std::vector<const void*> gu_v(static_cast<size_t>(top_k));
          std::vector<const void*> dn_v(static_cast<size_t>(top_k));
          for (int g = 0; g < top_k; ++g) {
            gu_v[static_cast<size_t>(g)] = gu_p[static_cast<size_t>(g)];
            dn_v[static_cast<size_t>(g)] = dn_p[static_cast<size_t>(g)];
          }
          ran = vt::ExpertGeGLUBf16TopKM1(d.q, ysum.ptr(), xin.ptr(), gu_v.data(),
                                                    dn_v.data(), wts.data(), top_k,
                                                    static_cast<int>(I), static_cast<int>(H));
        }
        if (!ran) {
          ran = ExpertGeGLUTopKFusedGelu(d, ysum, xin.t(), gu_p.data(), dn_p.data(), wts.data(),
                                         top_k, I, H);
        }
        if (ran) {
          d.b.Copy(
              d.q,
              static_cast<char*>(acc.ptr()) + static_cast<size_t>(t) * static_cast<size_t>(H) * 2,
              ysum.ptr(), static_cast<size_t>(H) * 2);
          continue;
        }
      }
    }

    for (int i = 0; i < top_k; ++i) {
      const int e = idx[static_cast<size_t>(i)];
      const float ww = wts[static_cast<size_t>(i)];
      const float beta = (i == 0) ? 0.f : 1.f;
      bool fused_mix = false;

      if (same_dev) {
        auto* gu = static_cast<const uint16_t*>(ex.gate_up_dev) +
                   static_cast<int64_t>(e) * gu_stride;
        auto* dn =
            static_cast<const uint16_t*>(ex.down_dev) + static_cast<int64_t>(e) * dn_stride;
        ExpertGeGLUDeviceAccum(d, ysum, xin.t(), gu, dn, I, H, esc, ww, beta);
        fused_mix = true;
      } else if (fp8_native && ex.is_fp8 && T == 1) {
        const auto& fex = ex.fp8[static_cast<size_t>(e)];
        if (EnsureGemma4Fp8NativeOnDevice(d, fex, I, H)) {
          ExpertGeGLUFp8Native(d, ysum, xin.t(), fex.dev_fp8_gu, fex.dev_s_gu, fex.dev_fp8_dn,
                               fex.dev_s_dn, I, H, esc, ww, beta);
          fused_mix = true;
        }
      } else if (need_peer_sc && gu_sc && dn_sc) {
        if (PeerCopyGemma4ExpertSlice(ex.dev_id, ex.gate_up_dev, ex.down_dev, e, I, H,
                                      compute_dev, gu_sc->ptr(), dn_sc->ptr())) {
          ExpertGeGLUDeviceAccum(d, ysum, xin.t(), static_cast<const uint16_t*>(gu_sc->ptr()),
                                 static_cast<const uint16_t*>(dn_sc->ptr()), I, H, esc, ww,
                                 beta);
          fused_mix = true;
        } else if (ex.is_fp8) {
          const auto& fex = ex.fp8[static_cast<size_t>(e)];
          DequantGemma4Fp8ExpertToBf16Ephemeral(fex, I, H, gu_tmp.data(), dn_tmp.data());
          ExpertGeGLUHost(d, y, xin.t(), gu_tmp.data(), dn_tmp.data(), I, H, esc);
        } else {
          VT_CHECK(gu_host && dn_host, "gemma4 moe: peer fail no host");
          ExpertGeGLUHost(d, y, xin.t(), gu_host + static_cast<int64_t>(e) * gu_stride,
                          dn_host + static_cast<int64_t>(e) * dn_stride, I, H, esc);
        }
      } else if (ex.is_fp8) {
        const auto& fex = ex.fp8[static_cast<size_t>(e)];
        if (EnsureGemma4Fp8ExpertOnDevice(d, fex, I, H)) {
          ExpertGeGLUDeviceAccum(d, ysum, xin.t(), static_cast<const uint16_t*>(fex.dev_gu),
                                 static_cast<const uint16_t*>(fex.dev_dn), I, H, esc, ww,
                                 beta);
          fused_mix = true;
        } else {
          EnsureGemma4Fp8ExpertCached(fex, I, H);
          ExpertGeGLUHost(d, y, xin.t(), fex.cached_gu.data(), fex.cached_dn.data(), I, H,
                          esc);
        }
      } else if (gu_host && dn_host) {
        ExpertGeGLUHost(d, y, xin.t(), gu_host + static_cast<int64_t>(e) * gu_stride,
                        dn_host + static_cast<int64_t>(e) * dn_stride, I, H, esc);
      } else {
        VT_CHECK(false, "gemma4 moe: no expert weights");
      }

      if (fused_mix) continue;  // already accumulated into ysum

      if (host_axpy) {
        d.b.Synchronize(d.q);
        std::vector<uint16_t> hy(static_cast<size_t>(H));
        d.b.Copy(d.q, hy.data(), y.ptr(), hy.size() * 2);
        d.b.Synchronize(d.q);
        for (int64_t j = 0; j < H; ++j)
          hsum[static_cast<size_t>(j)] = vt::F32ToBF16(
              vt::BF16ToF32(hsum[static_cast<size_t>(j)]) +
              ww * vt::BF16ToF32(hy[static_cast<size_t>(j)]));
      } else if (i == 0) {
        vt::MulScalar(d.q, ysum.t(), y.t(), static_cast<double>(ww));
      } else {
        vt::MulScalar(d.q, ysc.t(), y.t(), static_cast<double>(ww));
        vt::Add(d.q, ysum.t(), ysum.t(), ysc.t());
      }
    }
    if (host_axpy) {
      d.b.Copy(d.q, ysum.ptr(), hsum.data(), hsum.size() * 2);
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

  if (profile) {
    d.b.Synchronize(d.q);
    const auto t_all1 = clock::now();
    static std::atomic<uint64_t> ncalls{0};
    static std::atomic<uint64_t> us_router{0};
    static std::atomic<uint64_t> us_total{0};
    const auto ur = std::chrono::duration_cast<std::chrono::microseconds>(t_router1 - t_all0).count();
    const auto ut = std::chrono::duration_cast<std::chrono::microseconds>(t_all1 - t_all0).count();
    us_router.fetch_add(static_cast<uint64_t>(ur), std::memory_order_relaxed);
    us_total.fetch_add(static_cast<uint64_t>(ut), std::memory_order_relaxed);
    const uint64_t c = ncalls.fetch_add(1, std::memory_order_relaxed) + 1;
    if (c == 1 || c % 64 == 0) {
      const uint64_t tr = us_router.load(std::memory_order_relaxed);
      const uint64_t tt = us_total.load(std::memory_order_relaxed);
      std::fprintf(stderr,
                   "gemma4 moe profile: calls=%llu router_us/call=%.1f expert+rest_us/call=%.1f "
                   "total_us/call=%.1f (router%%=%.0f)\n",
                   static_cast<unsigned long long>(c), static_cast<double>(tr) / c,
                   static_cast<double>(tt - tr) / c, static_cast<double>(tt) / c,
                   tt ? 100.0 * static_cast<double>(tr) / static_cast<double>(tt) : 0.0);
    }
  }
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
bool RunGemma4FusedTopkExpertGeGLU(vt::Queue&, void*, const void*, const uint16_t* const*,
                                   const uint16_t* const*, const float*, int, int64_t, int64_t) {
  return false;
}
bool PeerCopyGemma4ExpertSlice(int, const void*, const void*, int, int64_t, int64_t, int, void*,
                               void*) {
  return false;
}
void PinGemma4Fp8ExpertHostCache(const Gemma4Fp8ExpertMats&) {}
#endif  // VLLM_CPP_HIP

}  // namespace vllm
