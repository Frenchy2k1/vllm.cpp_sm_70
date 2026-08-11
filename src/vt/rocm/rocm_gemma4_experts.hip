// ROCm upload of Gemma-4 MoE expert stacks (BF16 fused or FP8→BF16 dequant).
#include <cstring>
// Packs GPU0 first (compute device), then GPU1. FP8 streams per-expert to
// avoid ~30G host OOM from permanent BF16 caches / full-layer host buffers.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/gemma4.h"
#include "vllm/model_executor/models/gemma4_moe.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/fused_ops.h"
#include "vt/ops.h"

namespace vllm {
namespace {
// hipDeviceEnablePeerAccess leaves hipErrorPeerAccessAlreadyEnabled sticky;
// later Check(hipGetLastError()) would false-fail (seen on peer-act GeluAndMul).
inline void EnablePeerAccessQuiet(int peer_dev) {
  const hipError_t pe = hipDeviceEnablePeerAccess(peer_dev, 0);
  if (pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled) {
    // best-effort; unsupported P2P falls back to staging copies elsewhere
  }
  (void)hipGetLastError();  // clear sticky
}


void Check(hipError_t e, const char* w) {
  if (e != hipSuccess)
    throw std::runtime_error(std::string("gemma4 resident: ") + w + ": " + hipGetErrorString(e));
}

bool DeviceHasRoom(int dev, size_t need_bytes) {
  Check(hipSetDevice(dev), "hipSetDevice");
  size_t free_b = 0, tot_b = 0;
  if (hipMemGetInfo(&free_b, &tot_b) != hipSuccess) return true;
  // GPU0 is compute: dense weights + KV + activations need real headroom.
  // Default 12 GiB (lab dual-GPU). Override: VT_GEMMA4_GPU0_HEADROOM_GB (e.g. 4 for
  // single-GPU resident + short KV / graph experiments).
  static const size_t gpu0_headroom = []() -> size_t {
    const char* e = std::getenv("VT_GEMMA4_GPU0_HEADROOM_GB");
    if (e && e[0]) {
      const int g = std::atoi(e);
      if (g >= 0 && g <= 24) return static_cast<size_t>(g) << 30;
    }
    return static_cast<size_t>(12ull << 30);
  }();
  const size_t headroom = (dev == 0) ? gpu0_headroom : (512ull << 20);
  return free_b >= need_bytes + headroom;
}

bool AllocLayerDev(int dev, size_t gu_bytes, size_t dn_bytes, void** gu, void** dn) {
  if (!DeviceHasRoom(dev, gu_bytes + dn_bytes)) return false;
  Check(hipSetDevice(dev), "hipSetDevice");
  *gu = nullptr;
  *dn = nullptr;
  if (hipMalloc(gu, gu_bytes) != hipSuccess) return false;
  if (hipMalloc(dn, dn_bytes) != hipSuccess) {
    (void)hipFree(*gu);
    *gu = nullptr;
    return false;
  }
  return true;
}

bool AllocFour(int dev, size_t a, size_t b, size_t c, size_t d, void** pa, void** pb, void** pc,
               void** pd) {
  if (!DeviceHasRoom(dev, a + b + c + d)) return false;
  Check(hipSetDevice(dev), "hipSetDevice");
  *pa = *pb = *pc = *pd = nullptr;
  if (hipMalloc(pa, a) != hipSuccess) return false;
  if (hipMalloc(pb, b) != hipSuccess) {
    (void)hipFree(*pa);
    *pa = nullptr;
    return false;
  }
  if (hipMalloc(pc, c) != hipSuccess) {
    (void)hipFree(*pa);
    (void)hipFree(*pb);
    *pa = *pb = nullptr;
    return false;
  }
  if (hipMalloc(pd, d) != hipSuccess) {
    (void)hipFree(*pa);
    (void)hipFree(*pb);
    (void)hipFree(*pc);
    *pa = *pb = *pc = nullptr;
    return false;
  }
  return true;
}

// Native FP8 resident: keep u8 weights + BF16 channel scales on device (no BF16 expand).
// ~½ VRAM vs UploadFp8LayerStreaming BF16 path. Per-expert pointers alias layer packs.
bool UploadFp8NativeLayer(Gemma4FusedExperts& ex, int dev) {
  const int64_t E = ex.num_experts;
  const int64_t I = ex.intermediate;
  const int64_t H = ex.hidden;
  if (E <= 0 || I <= 0 || H <= 0 || ex.fp8.size() != static_cast<size_t>(E)) return false;

  const size_t gu_one = static_cast<size_t>(2 * I * H);       // u8
  const size_t dn_one = static_cast<size_t>(H * I);           // u8
  const size_t sgu_one = static_cast<size_t>(2 * I) * 2;      // bf16
  const size_t sdn_one = static_cast<size_t>(H) * 2;          // bf16
  const size_t gu_bytes = gu_one * static_cast<size_t>(E);
  const size_t dn_bytes = dn_one * static_cast<size_t>(E);
  const size_t sgu_bytes = sgu_one * static_cast<size_t>(E);
  const size_t sdn_bytes = sdn_one * static_cast<size_t>(E);

  void *gu = nullptr, *dn = nullptr, *sgu = nullptr, *sdn = nullptr;
  if (!AllocFour(dev, gu_bytes, dn_bytes, sgu_bytes, sdn_bytes, &gu, &dn, &sgu, &sdn))
    return false;

  std::vector<uint8_t> gu_pack(gu_one);
  for (int64_t e = 0; e < E; ++e) {
    auto& fex = ex.fp8[static_cast<size_t>(e)];
    VT_CHECK(fex.gate_w.HasHostBytes() && fex.up_w.HasHostBytes() && fex.down_w.HasHostBytes(),
             "fp8 native resident: missing weights");
    VT_CHECK(fex.gate_s.HasHostBytes() && fex.up_s.HasHostBytes() && fex.down_s.HasHostBytes(),
             "fp8 native resident: missing scales");
    // Pack gate|up FP8 contiguous [2I, H]
    const size_t half = static_cast<size_t>(I * H);
    std::memcpy(gu_pack.data(), fex.gate_w.bytes.data(), half);
    std::memcpy(gu_pack.data() + half, fex.up_w.bytes.data(), half);
    Check(hipMemcpy(static_cast<char*>(gu) + static_cast<size_t>(e) * gu_one, gu_pack.data(),
                    gu_one, hipMemcpyHostToDevice),
          "H2D fp8 gu");
    Check(hipMemcpy(static_cast<char*>(dn) + static_cast<size_t>(e) * dn_one,
                    fex.down_w.bytes.data(), dn_one, hipMemcpyHostToDevice),
          "H2D fp8 dn");
    // scales: gate then up → [2I]
    Check(hipMemcpy(static_cast<char*>(sgu) + static_cast<size_t>(e) * sgu_one,
                    fex.gate_s.bytes.data(), static_cast<size_t>(I) * 2, hipMemcpyHostToDevice),
          "H2D sgu g");
    Check(hipMemcpy(static_cast<char*>(sgu) + static_cast<size_t>(e) * sgu_one +
                        static_cast<size_t>(I) * 2,
                    fex.up_s.bytes.data(), static_cast<size_t>(I) * 2, hipMemcpyHostToDevice),
          "H2D sgu u");
    Check(hipMemcpy(static_cast<char*>(sdn) + static_cast<size_t>(e) * sdn_one,
                    fex.down_s.bytes.data(), sdn_one, hipMemcpyHostToDevice),
          "H2D sdn");

    // Drop host BF16 caches if any
    fex.cached_gu.clear();
    fex.cached_gu.shrink_to_fit();
    fex.cached_dn.clear();
    fex.cached_dn.shrink_to_fit();

    fex.dev_fp8_gu = static_cast<char*>(gu) + static_cast<size_t>(e) * gu_one;
    fex.dev_fp8_dn = static_cast<char*>(dn) + static_cast<size_t>(e) * dn_one;
    fex.dev_s_gu = static_cast<char*>(sgu) + static_cast<size_t>(e) * sgu_one;
    fex.dev_s_dn = static_cast<char*>(sdn) + static_cast<size_t>(e) * sdn_one;
    // Clear BF16 device expert slots (owned by LRU if any)
    fex.dev_gu = nullptr;
    fex.dev_dn = nullptr;
  }

  ex.fp8_gu_base = gu;
  ex.fp8_dn_base = dn;
  ex.fp8_sgu_base = sgu;
  ex.fp8_sdn_base = sdn;
  ex.fp8_native_resident = true;
  ex.gate_up_dev = nullptr;
  ex.down_dev = nullptr;
  ex.dev_id = dev;
  return true;
}

// Legacy: FP8 → BF16 expand on device (2× VRAM). Opt-in VT_GEMMA4_RESIDENT_BF16=1.
bool UploadFp8LayerStreaming(Gemma4FusedExperts& ex, int dev) {
  const int64_t E = ex.num_experts;
  const int64_t I = ex.intermediate;
  const int64_t H = ex.hidden;
  const size_t gu_one = static_cast<size_t>(2 * I * H) * sizeof(uint16_t);
  const size_t dn_one = static_cast<size_t>(H * I) * sizeof(uint16_t);
  const size_t gu_bytes = gu_one * static_cast<size_t>(E);
  const size_t dn_bytes = dn_one * static_cast<size_t>(E);

  void* gu = nullptr;
  void* dn = nullptr;
  if (!AllocLayerDev(dev, gu_bytes, dn_bytes, &gu, &dn)) return false;

  std::vector<uint16_t> gu_tmp(static_cast<size_t>(2 * I * H));
  std::vector<uint16_t> dn_tmp(static_cast<size_t>(H * I));
  for (int64_t e = 0; e < E; ++e) {
    DequantGemma4Fp8ExpertToBf16Ephemeral(ex.fp8[static_cast<size_t>(e)], I, H, gu_tmp.data(),
                                          dn_tmp.data());
    // Drop any accidental permanent cache on this expert (decode may have set it).
    ex.fp8[static_cast<size_t>(e)].cached_gu.clear();
    ex.fp8[static_cast<size_t>(e)].cached_gu.shrink_to_fit();
    ex.fp8[static_cast<size_t>(e)].cached_dn.clear();
    ex.fp8[static_cast<size_t>(e)].cached_dn.shrink_to_fit();

    Check(hipMemcpy(static_cast<char*>(gu) + static_cast<size_t>(e) * gu_one, gu_tmp.data(),
                    gu_one, hipMemcpyHostToDevice),
          "H2D gu e");
    Check(hipMemcpy(static_cast<char*>(dn) + static_cast<size_t>(e) * dn_one, dn_tmp.data(),
                    dn_one, hipMemcpyHostToDevice),
          "H2D dn e");
  }
  ex.gate_up_dev = gu;
  ex.down_dev = dn;
  ex.dev_id = dev;
  return true;
}

bool UploadBf16Layer(Gemma4FusedExperts& ex, int dev) {
  const int64_t E = ex.num_experts;
  const int64_t I = ex.intermediate;
  const int64_t H = ex.hidden;
  const size_t gu_bytes = static_cast<size_t>(E * 2 * I * H) * 2;
  const size_t dn_bytes = static_cast<size_t>(E * H * I) * 2;
  VT_CHECK(ex.gate_up.bytes.size() == gu_bytes && ex.down.bytes.size() == dn_bytes,
           "gemma4 resident: bf16 fused size");
  void* gu = nullptr;
  void* dn = nullptr;
  if (!AllocLayerDev(dev, gu_bytes, dn_bytes, &gu, &dn)) return false;
  Check(hipMemcpy(gu, ex.gate_up.bytes.data(), gu_bytes, hipMemcpyHostToDevice), "H2D gu");
  Check(hipMemcpy(dn, ex.down.bytes.data(), dn_bytes, hipMemcpyHostToDevice), "H2D dn");
  ex.gate_up_dev = gu;
  ex.down_dev = dn;
  ex.dev_id = dev;
  return true;
}

}  // namespace

size_t UploadGemma4ExpertsResident(std::vector<Gemma4MoeLayerWeights>& layers,
                                   int num_gpus) {
  if (num_gpus < 1) num_gpus = 1;
  int available = 0;
  Check(hipGetDeviceCount(&available), "hipGetDeviceCount");
  if (available < 1) return 0;
  if (num_gpus > available) num_gpus = available;

  // Enable peer access so off-device resident experts can D2D into compute GPU.
  for (int a = 0; a < num_gpus; ++a) {
    for (int b = 0; b < num_gpus; ++b) {
      if (a == b) continue;
      int can = 0;
      if (hipDeviceCanAccessPeer(&can, a, b) == hipSuccess && can) {
        Check(hipSetDevice(a), "peer set");
        hipError_t pe = hipDeviceEnablePeerAccess(b, 0);
        if (pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled) {
          std::fprintf(stderr, "gemma4 moe: peer access %d->%d failed: %s\n", a, b,
                       hipGetErrorString(pe));
        }
      }
    }
  }

  size_t total = 0;
  int ok_layers = 0;
  int max_layers = 100000;
  if (const char* ml = std::getenv("VT_GEMMA4_RESIDENT_MAX_LAYERS"))
    max_layers = std::max(0, std::atoi(ml));

  int fill_dev = 0;

  // FP8 resident mode:
  // Default: native FP8 packs (½ VRAM + fast fused ExpertGeGLU FP8 kernels).
  // VT_GEMMA4_RESIDENT_BF16=1 forces BF16 expand (hipBLAS reference / A/B).
  static const bool force_bf16 = [] {
    const char* e = std::getenv("VT_GEMMA4_RESIDENT_BF16");
    return e && e[0] == '1';
  }();
  static const bool force_native = [] {
    const char* e = std::getenv("VT_GEMMA4_RESIDENT_NATIVE");
    // Default ON when unset. =0 disables native (use BF16 expand).
    if (e == nullptr) return true;
    return e[0] == '1';
  }();

  for (size_t li = 0; li < layers.size(); ++li) {
    if (ok_layers >= max_layers) break;
    auto& moe = layers[li];
    if (!moe.enabled || moe.experts.Empty()) continue;
    auto& ex = moe.experts;
    const int64_t E = ex.num_experts;
    const int64_t I = ex.intermediate;
    const int64_t H = ex.hidden;
    const bool want_native = ex.is_fp8 && !force_bf16 && force_native;
    const size_t layer_bytes =
        want_native
            ? (static_cast<size_t>(E) * (static_cast<size_t>(2 * I * H) + static_cast<size_t>(H * I) +
                                         static_cast<size_t>(2 * I) * 2 + static_cast<size_t>(H) * 2))
            : (static_cast<size_t>(E * 2 * I * H) * 2 + static_cast<size_t>(E * H * I) * 2);

    bool ok = false;
    if (ex.is_fp8) {
      if (want_native)
        ok = UploadFp8NativeLayer(ex, fill_dev);
      else
        ok = UploadFp8LayerStreaming(ex, fill_dev);
    } else {
      ok = UploadBf16Layer(ex, fill_dev);
    }

    if (!ok && fill_dev + 1 < num_gpus) {
      ++fill_dev;
      std::fprintf(stderr, "gemma4 moe: GPU%d full after %d layers; continuing on GPU%d\n",
                   fill_dev - 1, ok_layers, fill_dev);
      if (ex.is_fp8) {
        if (want_native)
          ok = UploadFp8NativeLayer(ex, fill_dev);
        else
          ok = UploadFp8LayerStreaming(ex, fill_dev);
      } else {
        ok = UploadBf16Layer(ex, fill_dev);
      }
    }
    if (!ok) {
      std::fprintf(stderr, "gemma4 moe: resident upload stopped at layer %zu (OOM)\n", li);
      break;
    }
    total += layer_bytes;
    ++ok_layers;
    std::fprintf(stderr,
                 "gemma4 moe: resident layer %zu -> gpu %d %s (%.2f GiB cumulative)\n", li,
                 ex.dev_id, ex.fp8_native_resident ? "fp8-native" : "bf16",
                 total / (1024.0 * 1024.0 * 1024.0));
    std::fflush(stderr);
  }
  Check(hipSetDevice(0), "hipSetDevice0");
  std::fprintf(stderr, "gemma4 moe: RESIDENT DONE gpus=%d layers=%d %.2f GiB\n", num_gpus,
               ok_layers, total / (1024.0 * 1024.0 * 1024.0));
  std::fflush(stderr);
  return total;
}

size_t UploadGemma4ExpertsResidentForWeights(Gemma4Weights& weights, int num_gpus) {
  std::vector<Gemma4MoeLayerWeights> flat;
  std::vector<Gemma4MoeLayerWeights*> ptrs;
  for (auto& layer : weights.layers) {
    if (!layer.moe.enabled) continue;
    ptrs.push_back(&layer.moe);
    flat.push_back(std::move(layer.moe));
  }
  size_t n = UploadGemma4ExpertsResident(flat, num_gpus);
  for (size_t i = 0; i < ptrs.size(); ++i) *ptrs[i] = std::move(flat[i]);
  return n;
}

bool PeerCopyGemma4ExpertSlice(int src_dev, const void* gate_up_base, const void* down_base,
                               int expert_id, int64_t I, int64_t H, int compute_dev,
                               void* gate_up_dst, void* down_dst) {
  if (!gate_up_base || !down_base || !gate_up_dst || !down_dst) return false;
  if (src_dev < 0 || compute_dev < 0) return false;
  const size_t gu_one = static_cast<size_t>(2 * I * H) * sizeof(uint16_t);
  const size_t dn_one = static_cast<size_t>(H * I) * sizeof(uint16_t);
  const char* gu_src =
      static_cast<const char*>(gate_up_base) + static_cast<size_t>(expert_id) * gu_one;
  const char* dn_src =
      static_cast<const char*>(down_base) + static_cast<size_t>(expert_id) * dn_one;
  if (src_dev == compute_dev) {
    Check(hipSetDevice(compute_dev), "peer same set");
    Check(hipMemcpy(gate_up_dst, gu_src, gu_one, hipMemcpyDeviceToDevice), "D2D gu");
    Check(hipMemcpy(down_dst, dn_src, dn_one, hipMemcpyDeviceToDevice), "D2D dn");
    return true;
  }
  hipError_t e1 =
      hipMemcpyPeer(gate_up_dst, compute_dev, const_cast<char*>(gu_src), src_dev, gu_one);
  hipError_t e2 =
      hipMemcpyPeer(down_dst, compute_dev, const_cast<char*>(dn_src), src_dev, dn_one);
  if (e1 != hipSuccess || e2 != hipSuccess) {
    // Fallback: staged through host (still avoids FP8 dequant)
    std::vector<uint16_t> gu_tmp(gu_one / 2), dn_tmp(dn_one / 2);
    Check(hipSetDevice(src_dev), "peer fb src");
    Check(hipMemcpy(gu_tmp.data(), gu_src, gu_one, hipMemcpyDeviceToHost), "D2H gu");
    Check(hipMemcpy(dn_tmp.data(), dn_src, dn_one, hipMemcpyDeviceToHost), "D2H dn");
    Check(hipSetDevice(compute_dev), "peer fb dst");
    Check(hipMemcpy(gate_up_dst, gu_tmp.data(), gu_one, hipMemcpyHostToDevice), "H2D gu");
    Check(hipMemcpy(down_dst, dn_tmp.data(), dn_one, hipMemcpyHostToDevice), "H2D dn");
  }
  Check(hipSetDevice(compute_dev), "peer restore compute");
  return true;
}

bool PeerCopyGemma4Fp8ExpertSlice(int src_dev, const void* fp8_gu, const void* fp8_dn,
                                  const void* s_gu, const void* s_dn, int64_t I, int64_t H,
                                  int compute_dev, void* fp8_gu_dst, void* fp8_dn_dst,
                                  void* s_gu_dst, void* s_dn_dst) {
  if (!fp8_gu || !fp8_dn || !s_gu || !s_dn || !fp8_gu_dst || !fp8_dn_dst || !s_gu_dst ||
      !s_dn_dst)
    return false;
  if (src_dev < 0 || compute_dev < 0) return false;
  const size_t gu_b = static_cast<size_t>(2 * I * H);
  const size_t dn_b = static_cast<size_t>(H * I);
  const size_t sgu_b = static_cast<size_t>(2 * I) * 2;
  const size_t sdn_b = static_cast<size_t>(H) * 2;
  auto copy1 = [&](void* dst, const void* src, size_t n) -> bool {
    if (src_dev == compute_dev) {
      Check(hipSetDevice(compute_dev), "fp8 peer same");
      return hipMemcpy(dst, src, n, hipMemcpyDeviceToDevice) == hipSuccess;
    }
    if (hipMemcpyPeer(dst, compute_dev, const_cast<void*>(src), src_dev, n) == hipSuccess)
      return true;
    std::vector<char> tmp(n);
    Check(hipSetDevice(src_dev), "fp8 peer fb src");
    if (hipMemcpy(tmp.data(), src, n, hipMemcpyDeviceToHost) != hipSuccess) return false;
    Check(hipSetDevice(compute_dev), "fp8 peer fb dst");
    return hipMemcpy(dst, tmp.data(), n, hipMemcpyHostToDevice) == hipSuccess;
  };
  if (!copy1(fp8_gu_dst, fp8_gu, gu_b)) return false;
  if (!copy1(fp8_dn_dst, fp8_dn, dn_b)) return false;
  if (!copy1(s_gu_dst, s_gu, sgu_b)) return false;
  if (!copy1(s_dn_dst, s_dn, sdn_b)) return false;
  Check(hipSetDevice(compute_dev), "fp8 peer restore");
  return true;
}

bool RunGemma4Fp8TopKOnExpertDevice(vt::Queue& compute_q, int expert_dev, void* ysum_compute,
                                    const void* x_compute, const void* const* fp8_gu,
                                    const void* const* s_gu, const void* const* fp8_dn,
                                    const void* const* s_dn, const float* wts, int G, int I,
                                    int H) {
  const int compute_dev = compute_q.device.index;
  if (expert_dev < 0 || compute_dev < 0 || !ysum_compute || !x_compute || !fp8_gu || !s_gu ||
      !fp8_dn || !s_dn || !wts || G <= 0 || I <= 0 || H <= 0)
    return false;

  struct Tls {
    int cdev = -1;
    int edev = -1;
    vt::Queue cq{};  // only used when expert==compute and we need a stable q
    vt::Queue eq{};
    void* x = nullptr;
    void* y = nullptr;
    hipEvent_t ev_c = nullptr;
    hipEvent_t ev_e = nullptr;
    int H = 0;
    bool eq_live = false;
    bool cq_live = false;
    bool peer_ok = false;
  };
  static thread_local Tls tls;
  const size_t hb = static_cast<size_t>(H) * 2;  // bf16
  hipStream_t cst = static_cast<hipStream_t>(compute_q.handle);

  // Same device: run on compute_q directly (no Create/Destroy per call).
  if (expert_dev == compute_dev) {
    return vt::ExpertGeGLUFp8TopKM1(compute_q, ysum_compute, x_compute, fp8_gu, s_gu, fp8_dn, s_dn,
                                    wts, G, I, H);
  }

  vt::Device edev{vt::DeviceType::kROCM, expert_dev};
  if (tls.edev != expert_dev || tls.cdev != compute_dev || tls.H != H || !tls.x || !tls.y) {
    if (tls.x) {
      Check(hipSetDevice(tls.edev >= 0 ? tls.edev : expert_dev), "free x");
      (void)hipFree(tls.x);
      tls.x = nullptr;
    }
    if (tls.y) {
      Check(hipSetDevice(tls.edev >= 0 ? tls.edev : expert_dev), "free y");
      (void)hipFree(tls.y);
      tls.y = nullptr;
    }
    if (tls.eq_live) {
      vt::DestroyQueue(tls.eq);
      tls.eq_live = false;
    }
    if (tls.ev_c) {
      (void)hipEventDestroy(tls.ev_c);
      tls.ev_c = nullptr;
    }
    if (tls.ev_e) {
      (void)hipEventDestroy(tls.ev_e);
      tls.ev_e = nullptr;
    }
    Check(hipSetDevice(expert_dev), "expert alloc");
    if (hipMalloc(&tls.x, hb) != hipSuccess) return false;
    if (hipMalloc(&tls.y, hb) != hipSuccess) {
      (void)hipFree(tls.x);
      tls.x = nullptr;
      return false;
    }
    tls.eq = vt::CreateQueue(edev);
    tls.eq_live = true;
    Check(hipSetDevice(compute_dev), "ev c");
    Check(hipEventCreateWithFlags(&tls.ev_c, hipEventDisableTiming), "ev_c");
    Check(hipSetDevice(expert_dev), "ev e");
    Check(hipEventCreateWithFlags(&tls.ev_e, hipEventDisableTiming), "ev_e");
    // Best-effort P2P enable (ignore already-enabled / unsupported).
    Check(hipSetDevice(compute_dev), "p2p c");
    EnablePeerAccessQuiet(expert_dev);
    Check(hipSetDevice(expert_dev), "p2p e");
    EnablePeerAccessQuiet(compute_dev);
    tls.edev = expert_dev;
    tls.cdev = compute_dev;
    tls.H = H;
    tls.peer_ok = true;
  }

  hipStream_t est = static_cast<hipStream_t>(tls.eq.handle);

  // Steady-state: stream-ordered peer, no per-hop hipSetDevice.
  if (hipEventRecord(tls.ev_c, cst) != hipSuccess) return false;
  if (hipStreamWaitEvent(est, tls.ev_c, 0) != hipSuccess) return false;

  bool x_ok =
      hipMemcpyPeerAsync(tls.x, expert_dev, const_cast<void*>(x_compute), compute_dev, hb, est) ==
      hipSuccess;
  if (!x_ok) {
    // Host staging fallback (sync).
    std::vector<char> tmp(hb);
    Check(hipSetDevice(compute_dev), "x d2h");
    if (hipStreamSynchronize(cst) != hipSuccess) return false;
    if (hipMemcpy(tmp.data(), x_compute, hb, hipMemcpyDeviceToHost) != hipSuccess) return false;
    Check(hipSetDevice(expert_dev), "x h2d");
    if (hipMemcpyAsync(tls.x, tmp.data(), hb, hipMemcpyHostToDevice, est) != hipSuccess)
      return false;
  }

  if (!vt::ExpertGeGLUFp8TopKM1(tls.eq, tls.y, tls.x, fp8_gu, s_gu, fp8_dn, s_dn, wts, G, I, H))
    return false;

  // expert → compute: record expert done, wait on compute stream, async peer y.
  if (hipEventRecord(tls.ev_e, est) != hipSuccess) return false;
  if (hipStreamWaitEvent(cst, tls.ev_e, 0) != hipSuccess) return false;

  bool y_ok =
      hipMemcpyPeerAsync(ysum_compute, compute_dev, tls.y, expert_dev, hb, cst) == hipSuccess;
  if (!y_ok) {
    std::vector<char> tmp(hb);
    Check(hipSetDevice(expert_dev), "y d2h");
    if (hipStreamSynchronize(est) != hipSuccess) return false;
    if (hipMemcpy(tmp.data(), tls.y, hb, hipMemcpyDeviceToHost) != hipSuccess) return false;
    Check(hipSetDevice(compute_dev), "y h2d");
    if (hipMemcpyAsync(ysum_compute, tmp.data(), hb, hipMemcpyHostToDevice, cst) != hipSuccess)
      return false;
  }
  if (hipSetDevice(compute_dev) != hipSuccess) return false;
  return true;
}

// Decode T=1 indexed: bases stay on expert_dev; peer only x + tiny idx/wts + y.
bool RunGemma4Fp8TopKIndexedOnExpertDevice(vt::Queue& compute_q, int expert_dev, void* ysum_compute,
                                           const void* x_compute, const void* gu_base,
                                           const void* dn_base, const void* sgu_base,
                                           const void* sdn_base, const int32_t* idx_compute,
                                           const float* wts_compute, int G, int I, int H) {
  const int compute_dev = compute_q.device.index;
  if (expert_dev < 0 || compute_dev < 0 || !ysum_compute || !x_compute || !gu_base || !dn_base ||
      !sgu_base || !sdn_base || !idx_compute || !wts_compute || G <= 0 || I <= 0 || H <= 0)
    return false;

  if (expert_dev == compute_dev) {
    return vt::ExpertGeGLUFp8TopKIndexed(compute_q, ysum_compute, x_compute, gu_base, dn_base,
                                         sgu_base, sdn_base, idx_compute, wts_compute, G, I, H);
  }

  struct Tls {
    int cdev = -1, edev = -1, H = 0, Gcap = 0;
    vt::Queue eq{};
    void* x = nullptr;
    void* y = nullptr;
    int32_t* idx = nullptr;
    float* wts = nullptr;
    hipEvent_t ev_c = nullptr, ev_e = nullptr;
    bool eq_live = false;
  };
  static thread_local Tls tls;
  const size_t hb = static_cast<size_t>(H) * 2;
  hipStream_t cst = static_cast<hipStream_t>(compute_q.handle);
  vt::Device edev{vt::DeviceType::kROCM, expert_dev};

  if (tls.edev != expert_dev || tls.cdev != compute_dev || tls.H != H || tls.Gcap < G || !tls.x) {
    if (tls.x) {
      Check(hipSetDevice(tls.edev >= 0 ? tls.edev : expert_dev), "free");
      (void)hipFree(tls.x);
      (void)hipFree(tls.y);
      (void)hipFree(tls.idx);
      (void)hipFree(tls.wts);
      tls.x = tls.y = nullptr;
      tls.idx = nullptr;
      tls.wts = nullptr;
    }
    if (tls.eq_live) {
      vt::DestroyQueue(tls.eq);
      tls.eq_live = false;
    }
    if (tls.ev_c) {
      (void)hipEventDestroy(tls.ev_c);
      tls.ev_c = nullptr;
    }
    if (tls.ev_e) {
      (void)hipEventDestroy(tls.ev_e);
      tls.ev_e = nullptr;
    }
    Check(hipSetDevice(expert_dev), "idx alloc");
    if (hipMalloc(&tls.x, hb) != hipSuccess) return false;
    if (hipMalloc(&tls.y, hb) != hipSuccess) {
      (void)hipFree(tls.x);
      tls.x = nullptr;
      return false;
    }
    if (hipMalloc(&tls.idx, sizeof(int32_t) * 8) != hipSuccess) return false;
    if (hipMalloc(&tls.wts, sizeof(float) * 8) != hipSuccess) return false;
    tls.eq = vt::CreateQueue(edev);
    tls.eq_live = true;
    Check(hipSetDevice(compute_dev), "ev");
    Check(hipEventCreateWithFlags(&tls.ev_c, hipEventDisableTiming), "ev_c");
    Check(hipSetDevice(expert_dev), "ev e");
    Check(hipEventCreateWithFlags(&tls.ev_e, hipEventDisableTiming), "ev_e");
    Check(hipSetDevice(compute_dev), "p2p");
    EnablePeerAccessQuiet(expert_dev);
    Check(hipSetDevice(expert_dev), "p2p e");
    EnablePeerAccessQuiet(compute_dev);
    tls.edev = expert_dev;
    tls.cdev = compute_dev;
    tls.H = H;
    tls.Gcap = 8;
  }

  hipStream_t est = static_cast<hipStream_t>(tls.eq.handle);
  // Steady-state: no hipSetDevice between stream ops (graph-prep + less driver tax).
  if (hipEventRecord(tls.ev_c, cst) != hipSuccess) return false;
  if (hipStreamWaitEvent(est, tls.ev_c, 0) != hipSuccess) return false;

  auto peer = [&](void* dst, const void* src, size_t n) -> bool {
    return hipMemcpyPeerAsync(dst, expert_dev, const_cast<void*>(src), compute_dev, n, est) ==
           hipSuccess;
  };
  if (!peer(tls.x, x_compute, hb)) return false;
  if (!peer(tls.idx, idx_compute, sizeof(int32_t) * static_cast<size_t>(G))) return false;
  if (!peer(tls.wts, wts_compute, sizeof(float) * static_cast<size_t>(G))) return false;

  if (!vt::ExpertGeGLUFp8TopKIndexed(tls.eq, tls.y, tls.x, gu_base, dn_base, sgu_base, sdn_base,
                                     tls.idx, tls.wts, G, I, H))
    return false;

  if (hipEventRecord(tls.ev_e, est) != hipSuccess) return false;
  if (hipStreamWaitEvent(cst, tls.ev_e, 0) != hipSuccess) return false;
  if (hipMemcpyPeerAsync(ysum_compute, compute_dev, tls.y, expert_dev, hb, cst) != hipSuccess)
    return false;
  if (hipSetDevice(compute_dev) != hipSuccess) return false;
  return true;
}

// Prefill: run ExpertGeGLU for M rows on expert_dev (weights resident there).
// Peer only activations M×H bf16 — not multi-GB FP8 expert packs.
bool RunGemma4Fp8ExpertGeGLUPrefillOnExpertDevice(vt::Queue& compute_q, int expert_dev,
                                                  void* y_compute, const void* x_compute,
                                                  const void* fp8_gu, const void* s_gu,
                                                  const void* fp8_dn, const void* s_dn, int M,
                                                  int I, int H) {
  const int compute_dev = compute_q.device.index;
  if (expert_dev < 0 || compute_dev < 0 || !y_compute || !x_compute || !fp8_gu || !s_gu ||
      !fp8_dn || !s_dn || M <= 0 || I <= 0 || H <= 0)
    return false;

  // Same device: caller should use local ExpertGeGLU; still support for safety.
  if (expert_dev == compute_dev) {
    vt::Device dev = compute_q.device;
    // Dequant sticky into TLS then GEMM on compute_q.
    struct SameTls {
      int dev = -1;
      int I = 0, H = 0, Mcap = 0;
      const void* sticky = nullptr;
      void* gu_w = nullptr;
      void* dn_w = nullptr;
      void* gu_act = nullptr;
      void* act = nullptr;
    };
    static thread_local SameTls tls;
    const size_t gu_w_b = static_cast<size_t>(2 * I) * static_cast<size_t>(H) * 2;
    const size_t dn_w_b = static_cast<size_t>(H) * static_cast<size_t>(I) * 2;
    const size_t gu_act_b = static_cast<size_t>(M) * static_cast<size_t>(2 * I) * 2;
    const size_t act_b = static_cast<size_t>(M) * static_cast<size_t>(I) * 2;
    if (tls.dev != compute_dev || tls.I != I || tls.H != H || tls.Mcap < M || !tls.gu_w) {
      auto freep = [&](void*& p) {
        if (p) {
          (void)hipFree(p);
          p = nullptr;
        }
      };
      freep(tls.gu_w);
      freep(tls.dn_w);
      freep(tls.gu_act);
      freep(tls.act);
      Check(hipSetDevice(compute_dev), "same alloc");
      if (hipMalloc(&tls.gu_w, gu_w_b) != hipSuccess) return false;
      if (hipMalloc(&tls.dn_w, dn_w_b) != hipSuccess) return false;
      // Cap act buffers to at least M; grow only.
      size_t gu_act_cap = static_cast<size_t>(std::max(M, 64)) * static_cast<size_t>(2 * I) * 2;
      size_t act_cap = static_cast<size_t>(std::max(M, 64)) * static_cast<size_t>(I) * 2;
      if (hipMalloc(&tls.gu_act, gu_act_cap) != hipSuccess) return false;
      if (hipMalloc(&tls.act, act_cap) != hipSuccess) return false;
      tls.dev = compute_dev;
      tls.I = I;
      tls.H = H;
      tls.Mcap = std::max(M, 64);
      tls.sticky = nullptr;
    }
    if (tls.sticky != fp8_gu) {
      vt::DequantFp8ChannelBf16(compute_q, tls.gu_w, fp8_gu, s_gu, 2 * I, H);
      vt::DequantFp8ChannelBf16(compute_q, tls.dn_w, fp8_dn, s_dn, H, I);
      tls.sticky = fp8_gu;
    }
    vt::Tensor x = vt::Tensor::Contiguous(const_cast<void*>(x_compute), vt::DType::kBF16, dev,
                                          {M, H});
    vt::Tensor gu_w = vt::Tensor::Contiguous(tls.gu_w, vt::DType::kBF16, dev, {2 * I, H});
    vt::Tensor gu_act = vt::Tensor::Contiguous(tls.gu_act, vt::DType::kBF16, dev, {M, 2 * I});
    vt::Tensor act = vt::Tensor::Contiguous(tls.act, vt::DType::kBF16, dev, {M, I});
    vt::MatmulBT(compute_q, gu_act, x, gu_w);
    vt::GeluAndMul(compute_q, act, gu_act);
    vt::MatmulBTAlphaBeta(compute_q, y_compute, act.data, tls.dn_w, M, H, I, 1.f, 0.f,
                          vt::DType::kBF16);
    return true;
  }

  struct Tls {
    int cdev = -1, edev = -1;
    int I = 0, H = 0, Mcap = 0;
    const void* sticky = nullptr;
    vt::Queue eq{};
    bool eq_live = false;
    void* x = nullptr;
    void* y = nullptr;
    void* gu_w = nullptr;
    void* dn_w = nullptr;
    void* gu_act = nullptr;
    void* act = nullptr;
    hipEvent_t ev_c = nullptr, ev_e = nullptr;
  };
  static thread_local Tls tls;
  const size_t row_b = static_cast<size_t>(H) * 2;
  const size_t xy_need = static_cast<size_t>(M) * row_b;
  const size_t gu_w_b = static_cast<size_t>(2 * I) * static_cast<size_t>(H) * 2;
  const size_t dn_w_b = static_cast<size_t>(H) * static_cast<size_t>(I) * 2;
  hipStream_t cst = static_cast<hipStream_t>(compute_q.handle);
  vt::Device edev{vt::DeviceType::kROCM, expert_dev};

  auto freep = [](int dev, void*& p) {
    if (!p) return;
    (void)hipSetDevice(dev);
    (void)hipFree(p);
    p = nullptr;
  };

  if (tls.edev != expert_dev || tls.cdev != compute_dev || tls.I != I || tls.H != H ||
      tls.Mcap < M || !tls.x) {
    freep(tls.edev >= 0 ? tls.edev : expert_dev, tls.x);
    freep(tls.edev >= 0 ? tls.edev : expert_dev, tls.y);
    freep(tls.edev >= 0 ? tls.edev : expert_dev, tls.gu_w);
    freep(tls.edev >= 0 ? tls.edev : expert_dev, tls.dn_w);
    freep(tls.edev >= 0 ? tls.edev : expert_dev, tls.gu_act);
    freep(tls.edev >= 0 ? tls.edev : expert_dev, tls.act);
    if (tls.eq_live) {
      vt::DestroyQueue(tls.eq);
      tls.eq_live = false;
    }
    if (tls.ev_c) {
      (void)hipEventDestroy(tls.ev_c);
      tls.ev_c = nullptr;
    }
    if (tls.ev_e) {
      (void)hipEventDestroy(tls.ev_e);
      tls.ev_e = nullptr;
    }
    const int Mcap = std::max(M, 64);
    Check(hipSetDevice(expert_dev), "peer-act expert alloc");
    if (hipMalloc(&tls.x, static_cast<size_t>(Mcap) * row_b) != hipSuccess) return false;
    if (hipMalloc(&tls.y, static_cast<size_t>(Mcap) * row_b) != hipSuccess) return false;
    if (hipMalloc(&tls.gu_w, gu_w_b) != hipSuccess) return false;
    if (hipMalloc(&tls.dn_w, dn_w_b) != hipSuccess) return false;
    if (hipMalloc(&tls.gu_act, static_cast<size_t>(Mcap) * static_cast<size_t>(2 * I) * 2) !=
        hipSuccess)
      return false;
    if (hipMalloc(&tls.act, static_cast<size_t>(Mcap) * static_cast<size_t>(I) * 2) != hipSuccess)
      return false;
    tls.eq = vt::CreateQueue(edev);
    tls.eq_live = true;
    Check(hipSetDevice(compute_dev), "peer-act ev_c");
    Check(hipEventCreateWithFlags(&tls.ev_c, hipEventDisableTiming), "ev_c");
    Check(hipSetDevice(expert_dev), "peer-act ev_e");
    Check(hipEventCreateWithFlags(&tls.ev_e, hipEventDisableTiming), "ev_e");
    Check(hipSetDevice(compute_dev), "p2p c");
    EnablePeerAccessQuiet(expert_dev);
    Check(hipSetDevice(expert_dev), "p2p e");
    EnablePeerAccessQuiet(compute_dev);
    tls.edev = expert_dev;
    tls.cdev = compute_dev;
    tls.I = I;
    tls.H = H;
    tls.Mcap = Mcap;
    tls.sticky = nullptr;
  }

  hipStream_t est = static_cast<hipStream_t>(tls.eq.handle);
  if (hipEventRecord(tls.ev_c, cst) != hipSuccess) return false;
  if (hipStreamWaitEvent(est, tls.ev_c, 0) != hipSuccess) return false;

  bool x_ok = hipMemcpyPeerAsync(tls.x, expert_dev, const_cast<void*>(x_compute), compute_dev,
                                 xy_need, est) == hipSuccess;
  if (!x_ok) {
    std::vector<char> tmp(xy_need);
    Check(hipSetDevice(compute_dev), "x d2h");
    if (hipStreamSynchronize(cst) != hipSuccess) return false;
    if (hipMemcpy(tmp.data(), x_compute, xy_need, hipMemcpyDeviceToHost) != hipSuccess)
      return false;
    Check(hipSetDevice(expert_dev), "x h2d");
    if (hipMemcpyAsync(tls.x, tmp.data(), xy_need, hipMemcpyHostToDevice, est) != hipSuccess)
      return false;
  }

  if (tls.sticky != fp8_gu) {
    Check(hipSetDevice(expert_dev), "dequant setdev");
    vt::DequantFp8ChannelBf16(tls.eq, tls.gu_w, fp8_gu, s_gu, 2 * I, H);
    vt::DequantFp8ChannelBf16(tls.eq, tls.dn_w, fp8_dn, s_dn, H, I);
    tls.sticky = fp8_gu;
  }

  Check(hipSetDevice(expert_dev), "geglu setdev");
  vt::Tensor x = vt::Tensor::Contiguous(tls.x, vt::DType::kBF16, edev, {M, H});
  vt::Tensor gu_w = vt::Tensor::Contiguous(tls.gu_w, vt::DType::kBF16, edev, {2 * I, H});
  vt::Tensor gu_act = vt::Tensor::Contiguous(tls.gu_act, vt::DType::kBF16, edev, {M, 2 * I});
  vt::Tensor act = vt::Tensor::Contiguous(tls.act, vt::DType::kBF16, edev, {M, I});
  vt::MatmulBT(tls.eq, gu_act, x, gu_w);
  vt::GeluAndMul(tls.eq, act, gu_act);
  vt::MatmulBTAlphaBeta(tls.eq, tls.y, act.data, tls.dn_w, M, H, I, 1.f, 0.f, vt::DType::kBF16);

  if (hipEventRecord(tls.ev_e, est) != hipSuccess) return false;
  if (hipStreamWaitEvent(cst, tls.ev_e, 0) != hipSuccess) return false;
  bool y_ok =
      hipMemcpyPeerAsync(y_compute, compute_dev, tls.y, expert_dev, xy_need, cst) == hipSuccess;
  if (!y_ok) {
    std::vector<char> tmp(xy_need);
    Check(hipSetDevice(expert_dev), "y d2h");
    if (hipStreamSynchronize(est) != hipSuccess) return false;
    if (hipMemcpy(tmp.data(), tls.y, xy_need, hipMemcpyDeviceToHost) != hipSuccess) return false;
    Check(hipSetDevice(compute_dev), "y h2d");
    if (hipMemcpyAsync(y_compute, tmp.data(), xy_need, hipMemcpyHostToDevice, cst) != hipSuccess)
      return false;
  }
  if (hipSetDevice(compute_dev) != hipSuccess) return false;
  return true;
}

void PinGemma4Fp8ExpertHostCache(const Gemma4Fp8ExpertMats& ex) {
  if (ex.host_pinned) return;
  if (ex.cached_gu.empty() || ex.cached_dn.empty()) return;
  const size_t gu_b = ex.cached_gu.size() * sizeof(uint16_t);
  const size_t dn_b = ex.cached_dn.size() * sizeof(uint16_t);
  hipError_t e1 =
      hipHostRegister(const_cast<uint16_t*>(ex.cached_gu.data()), gu_b, hipHostRegisterDefault);
  hipError_t e2 =
      hipHostRegister(const_cast<uint16_t*>(ex.cached_dn.data()), dn_b, hipHostRegisterDefault);
  if ((e1 == hipSuccess || e1 == hipErrorHostMemoryAlreadyRegistered) &&
      (e2 == hipSuccess || e2 == hipErrorHostMemoryAlreadyRegistered)) {
    ex.host_pinned = true;
  }
}

void UnpinGemma4Fp8ExpertHostCache(const Gemma4Fp8ExpertMats& ex) {
  if (!ex.host_pinned) return;
  if (!ex.cached_gu.empty()) {
    (void)hipHostUnregister(const_cast<uint16_t*>(ex.cached_gu.data()));
  }
  if (!ex.cached_dn.empty()) {
    (void)hipHostUnregister(const_cast<uint16_t*>(ex.cached_dn.data()));
  }
  ex.host_pinned = false;
}

}  // namespace vllm
