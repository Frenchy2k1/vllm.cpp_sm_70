// Byte-exactness + determinism gate for `vt::Conv2d` (spike row P1,
// .agents/specs/parakeet-conformer-encoder.md; kernel src/vt/cpu/cpu_conv2d.cpp).
//
// The op mirrors torch `nn.Conv2d` as `ParakeetEncoderSubsamplingConv2D`
// instantiates it (transformers 5.3.0
// transformers/models/parakeet/modeling_parakeet.py:357 — dense :369-371,
// depthwise groups=C :377-386, pointwise 1x1 :388), which is the module vLLM
// runs (vllm/model_executor/models/parakeet.py:37,62), and which vLLM's own
// native `Conv2dSubsampling` (conformer_encoder.py:18) duplicates.
//
// GATE, matching tests/vt/test_ops_matmul_elem.cpp discipline: `memcmp` against
// an INDEPENDENT in-test scalar reference written from the convolution
// definition, NOT against the kernel. That is achievable exactly because the
// kernel keeps one f32 accumulator per output element, walked in (ic, kh, kw)
// order with the bias last (the contract stated in include/vt/ops.h). Coverage:
// every x/weight/out dtype combination over f32/f16/bf16, dense + depthwise +
// pointwise + grouped + dilated + non-square kernels, ragged extents that make
// the padded window fall off both edges, batch > 1, and thread counts 1/2/4/8.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "vt/cpu/cpu_threadpool.h"  // Threadpool::SwapForTesting (via -I src)
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Conv2dArgs;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue CpuQueue() { return Queue{Cpu(), nullptr}; }

// Deterministic LCG so every case is reproducible without a seed corpus.
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
  uint32_t Next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(s >> 33);
  }
  float Uniform() { return static_cast<float>(Next() % 20001) / 10000.0f - 1.0f; }
};

// Storage for one operand plus the EXACT f32 value of every element after
// rounding into the storage dtype (what the kernel will read back).
struct Buf {
  DType dtype;
  std::vector<uint8_t> bytes;
  std::vector<float> ref;

  Buf(DType dt, int64_t n, Rng& rng) : dtype(dt) {
    ref.resize(static_cast<size_t>(n));
    bytes.resize(static_cast<size_t>(n) * vt::SizeOf(dt));
    for (int64_t i = 0; i < n; ++i) {
      const float v = rng.Uniform();
      switch (dt) {
        case DType::kF32:
          reinterpret_cast<float*>(bytes.data())[i] = v;
          ref[i] = v;
          break;
        case DType::kF16: {
          const uint16_t h = vt::F32ToF16(v);
          reinterpret_cast<uint16_t*>(bytes.data())[i] = h;
          ref[i] = vt::F16ToF32(h);
          break;
        }
        case DType::kBF16: {
          const uint16_t h = vt::F32ToBF16(v);
          reinterpret_cast<uint16_t*>(bytes.data())[i] = h;
          ref[i] = vt::BF16ToF32(h);
          break;
        }
        default:
          break;
      }
    }
  }
  void* Data() { return bytes.data(); }
};

void StoreAt(std::vector<uint8_t>* out, int64_t i, DType dt, float v) {
  switch (dt) {
    case DType::kF32: reinterpret_cast<float*>(out->data())[i] = v; break;
    case DType::kF16: reinterpret_cast<uint16_t*>(out->data())[i] = vt::F32ToF16(v); break;
    case DType::kBF16: reinterpret_cast<uint16_t*>(out->data())[i] = vt::F32ToBF16(v); break;
    default: break;
  }
}

struct Case {
  const char* name;
  int64_t n, cin, hin, win;
  int64_t cout, kh, kw;
  int64_t sh, sw, ph, pw, dh, dw, groups;
  bool bias;
};

int64_t OutLen(int64_t in, int64_t pad, int64_t dil, int64_t k, int64_t stride) {
  return (in + 2 * pad - dil * (k - 1) - 1) / stride + 1;
}

// INDEPENDENT scalar reference. Written from the convolution definition:
// out[n,oc,oh,ow] = bias[oc] + Σ over the output channel's input group and the
// dilated kernel window, taps outside the input skipped (zero padding), one f32
// accumulator, bias added last.
void RefConv2d(const Case& c, const std::vector<float>& x, const std::vector<float>& w,
               const std::vector<float>* bias, DType odt, std::vector<uint8_t>* out) {
  const int64_t hout = OutLen(c.hin, c.ph, c.dh, c.kh, c.sh);
  const int64_t wout = OutLen(c.win, c.pw, c.dw, c.kw, c.sw);
  const int64_t cin_g = c.cin / c.groups;
  const int64_t cout_g = c.cout / c.groups;
  out->assign(static_cast<size_t>(c.n * c.cout * hout * wout) * vt::SizeOf(odt), 0);
  for (int64_t bn = 0; bn < c.n; ++bn) {
    for (int64_t oc = 0; oc < c.cout; ++oc) {
      const int64_t gc0 = (oc / cout_g) * cin_g;
      for (int64_t oh = 0; oh < hout; ++oh) {
        for (int64_t ow = 0; ow < wout; ++ow) {
          float acc = 0.0f;
          for (int64_t ic = 0; ic < cin_g; ++ic) {
            for (int64_t kh = 0; kh < c.kh; ++kh) {
              const int64_t ih = oh * c.sh - c.ph + kh * c.dh;
              if (ih < 0 || ih >= c.hin) continue;
              for (int64_t kw = 0; kw < c.kw; ++kw) {
                const int64_t iw = ow * c.sw - c.pw + kw * c.dw;
                if (iw < 0 || iw >= c.win) continue;
                const size_t xi =
                    static_cast<size_t>(((bn * c.cin + gc0 + ic) * c.hin + ih) * c.win + iw);
                const size_t wi =
                    static_cast<size_t>(((oc * cin_g + ic) * c.kh + kh) * c.kw + kw);
                acc += x[xi] * w[wi];
              }
            }
          }
          if (bias != nullptr) acc += (*bias)[static_cast<size_t>(oc)];
          StoreAt(out, ((bn * c.cout + oc) * hout + oh) * wout + ow, odt, acc);
        }
      }
    }
  }
}

// Shapes: the three ParakeetEncoderSubsamplingConv2D configurations first, then
// the generalizations (grouped, dilated, non-square, degenerate extents).
const std::vector<Case>& Cases() {
  static const std::vector<Case> c = {
      // dense front layer: nn.Conv2d(1, C, 3, stride 2, padding 1) (:369-371).
      {"dense k3s2p1", 1, 1, 7, 9, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1, true},
      {"dense k3s2p1 nobias", 2, 1, 8, 8, 3, 3, 3, 2, 2, 1, 1, 1, 1, 1, false},
      // depthwise stage: groups == Cin == Cout (:377-386).
      {"depthwise k3s2p1", 1, 4, 9, 5, 4, 3, 3, 2, 2, 1, 1, 1, 1, 4, true},
      {"depthwise k5s2p2", 2, 3, 11, 7, 3, 5, 5, 2, 2, 2, 2, 1, 1, 3, false},
      // pointwise stage: 1x1, stride 1, no padding (:388).
      {"pointwise 1x1", 2, 5, 4, 6, 7, 1, 1, 1, 1, 0, 0, 1, 1, 1, true},
      // generalizations
      {"grouped g2", 1, 4, 6, 6, 6, 3, 3, 1, 1, 1, 1, 1, 1, 2, true},
      {"dilated d2", 1, 2, 9, 9, 3, 3, 3, 1, 1, 2, 2, 2, 2, 1, true},
      {"non-square kh3 kw1", 1, 2, 7, 5, 3, 3, 1, 1, 1, 1, 0, 1, 1, 1, true},
      {"non-square kh1 kw5", 1, 2, 5, 9, 3, 1, 5, 1, 1, 0, 2, 1, 1, 1, false},
      {"asym stride", 1, 3, 10, 7, 5, 3, 3, 3, 1, 1, 1, 1, 1, 1, true},
      // degenerate / ragged extents: a single row, a single column, no padding
      // with a window that never leaves the input, and W not a multiple of the
      // stride so the last output column reads a truncated window.
      {"single row", 1, 2, 1, 8, 3, 1, 3, 1, 2, 0, 1, 1, 1, 1, true},
      {"single col", 1, 2, 8, 1, 3, 3, 1, 2, 1, 1, 0, 1, 1, 1, false},
      {"no padding", 1, 3, 6, 6, 4, 3, 3, 1, 1, 0, 0, 1, 1, 1, true},
      {"ragged stride", 1, 1, 13, 11, 2, 3, 3, 2, 2, 1, 1, 1, 1, 1, true},
      // batch > 1 with every knob on at once
      {"kitchen sink", 3, 6, 9, 10, 9, 3, 2, 2, 3, 1, 1, 2, 1, 3, true},
  };
  return c;
}

void RunCase(const Case& c, DType xdt, DType wdt, DType odt, uint64_t seed) {
  Queue q = CpuQueue();
  const int64_t hout = OutLen(c.hin, c.ph, c.dh, c.kh, c.sh);
  const int64_t wout = OutLen(c.win, c.pw, c.dw, c.kw, c.sw);
  REQUIRE(hout > 0);
  REQUIRE(wout > 0);
  Rng rng(seed);
  Buf x(xdt, c.n * c.cin * c.hin * c.win, rng);
  Buf w(wdt, c.cout * (c.cin / c.groups) * c.kh * c.kw, rng);
  Buf b(wdt, c.cout, rng);

  std::vector<uint8_t> got(static_cast<size_t>(c.n * c.cout * hout * wout) * vt::SizeOf(odt),
                           0xAB);
  Tensor tx = Tensor::Contiguous(x.Data(), xdt, Cpu(), {c.n, c.cin, c.hin, c.win});
  Tensor tw = Tensor::Contiguous(w.Data(), wdt, Cpu(), {c.cout, c.cin / c.groups, c.kh, c.kw});
  Tensor tb = Tensor::Contiguous(b.Data(), wdt, Cpu(), {c.cout});
  Tensor to = Tensor::Contiguous(got.data(), odt, Cpu(), {c.n, c.cout, hout, wout});
  Conv2dArgs args;
  args.stride_h = c.sh;
  args.stride_w = c.sw;
  args.pad_h = c.ph;
  args.pad_w = c.pw;
  args.dilation_h = c.dh;
  args.dilation_w = c.dw;
  args.groups = c.groups;
  vt::Conv2d(q, to, tx, tw, c.bias ? &tb : nullptr, args);

  std::vector<uint8_t> want;
  RefConv2d(c, x.ref, w.ref, c.bias ? &b.ref : nullptr, odt, &want);
  REQUIRE(want.size() == got.size());
  CHECK_MESSAGE(std::memcmp(got.data(), want.data(), want.size()) == 0, c.name);
}

}  // namespace

TEST_CASE("conv2d: byte-identical to the scalar reference over every dtype x shape") {
  const DType kDt[3] = {DType::kF32, DType::kF16, DType::kBF16};
  uint64_t seed = 1;
  for (DType xdt : kDt) {
    for (DType wdt : kDt) {
      for (DType odt : kDt) {
        for (const Case& c : Cases()) RunCase(c, xdt, wdt, odt, seed++);
      }
    }
  }
}

TEST_CASE("conv2d: byte-identical across thread counts") {
  // The determinism contract (src/vt/cpu/cpu_conv2d.cpp): parallelism partitions
  // output rows only, so the result must not depend on the worker count. The
  // shape is chosen so the (n, cout, hout) row count (3*9*4 == 108) does not
  // divide evenly by 8 -- a partition-dependent kernel would show up here.
  Queue q = CpuQueue();
  const Case c = {"threads", 3, 6, 9, 10, 9, 3, 2, 2, 3, 1, 1, 2, 1, 3, true};
  const int64_t hout = OutLen(c.hin, c.ph, c.dh, c.kh, c.sh);
  const int64_t wout = OutLen(c.win, c.pw, c.dw, c.kw, c.sw);
  Rng rng(9001);
  Buf x(DType::kBF16, c.n * c.cin * c.hin * c.win, rng);
  Buf w(DType::kBF16, c.cout * (c.cin / c.groups) * c.kh * c.kw, rng);
  Buf b(DType::kBF16, c.cout, rng);
  Conv2dArgs args;
  args.stride_h = c.sh;
  args.stride_w = c.sw;
  args.pad_h = c.ph;
  args.pad_w = c.pw;
  args.dilation_h = c.dh;
  args.dilation_w = c.dw;
  args.groups = c.groups;
  std::vector<uint8_t> base;
  for (int nth : {1, 2, 4, 8}) {
    vt::cpu::Threadpool tp(nth);
    vt::cpu::Threadpool* prev = vt::cpu::Threadpool::SwapForTesting(&tp);
    std::vector<uint8_t> got(static_cast<size_t>(c.n * c.cout * hout * wout) * sizeof(float),
                             0xAB);
    Tensor tx = Tensor::Contiguous(x.Data(), DType::kBF16, Cpu(), {c.n, c.cin, c.hin, c.win});
    Tensor tw =
        Tensor::Contiguous(w.Data(), DType::kBF16, Cpu(), {c.cout, c.cin / c.groups, c.kh, c.kw});
    Tensor tb = Tensor::Contiguous(b.Data(), DType::kBF16, Cpu(), {c.cout});
    Tensor to = Tensor::Contiguous(got.data(), DType::kF32, Cpu(), {c.n, c.cout, hout, wout});
    vt::Conv2d(q, to, tx, tw, &tb, args);
    vt::cpu::Threadpool::SwapForTesting(prev);
    if (base.empty()) {
      base = got;
      std::vector<uint8_t> want;
      RefConv2d(c, x.ref, w.ref, &b.ref, DType::kF32, &want);
      CHECK(std::memcmp(base.data(), want.data(), want.size()) == 0);
    } else {
      CHECK(std::memcmp(base.data(), got.data(), base.size()) == 0);
    }
  }
}

TEST_CASE("conv2d: the three ParakeetEncoderSubsamplingConv2D stages compose") {
  // modeling_parakeet.py:365-390 — Conv2d(1,C,k,s,p) -> ReLU -> [depthwise
  // Conv2d(C,C,k,s,p,groups=C) -> pointwise Conv2d(C,C,1) -> ReLU] * (L-1).
  // num_layers = log2(subsampling_factor); this exercises L == 2 (factor 4).
  // The gate is the same scalar reference applied stage by stage, so a wrong
  // group mapping or a wrong output-length rule cannot survive the chain.
  Queue q = CpuQueue();
  const int64_t mel = 16, tmax = 11, ch = 5;  // [N=1, 1, T, mel]
  Rng rng(1234);
  Buf x(DType::kF32, tmax * mel, rng);
  Buf w0(DType::kF32, ch * 1 * 3 * 3, rng);
  Buf wd(DType::kF32, ch * 1 * 3 * 3, rng);
  Buf wp(DType::kF32, ch * ch * 1 * 1, rng);

  Conv2dArgs sub;
  sub.stride_h = sub.stride_w = 2;
  sub.pad_h = sub.pad_w = 1;
  const int64_t h1 = OutLen(tmax, 1, 1, 3, 2), w1 = OutLen(mel, 1, 1, 3, 2);
  const int64_t h2 = OutLen(h1, 1, 1, 3, 2), w2 = OutLen(w1, 1, 1, 3, 2);

  std::vector<float> s1(static_cast<size_t>(ch * h1 * w1), 0.0f);
  Tensor tx = Tensor::Contiguous(x.Data(), DType::kF32, Cpu(), {1, 1, tmax, mel});
  Tensor tw0 = Tensor::Contiguous(w0.Data(), DType::kF32, Cpu(), {ch, 1, 3, 3});
  Tensor t1 = Tensor::Contiguous(s1.data(), DType::kF32, Cpu(), {1, ch, h1, w1});
  vt::Conv2d(q, t1, tx, tw0, nullptr, sub);
  vt::Relu(q, t1, t1);

  std::vector<float> s2(static_cast<size_t>(ch * h2 * w2), 0.0f);
  Conv2dArgs dws = sub;
  dws.groups = ch;
  Tensor twd = Tensor::Contiguous(wd.Data(), DType::kF32, Cpu(), {ch, 1, 3, 3});
  Tensor t2 = Tensor::Contiguous(s2.data(), DType::kF32, Cpu(), {1, ch, h2, w2});
  vt::Conv2d(q, t2, t1, twd, nullptr, dws);

  std::vector<float> s3(static_cast<size_t>(ch * h2 * w2), 0.0f);
  Conv2dArgs pws;  // 1x1, stride 1, no padding
  Tensor twp = Tensor::Contiguous(wp.Data(), DType::kF32, Cpu(), {ch, ch, 1, 1});
  Tensor t3 = Tensor::Contiguous(s3.data(), DType::kF32, Cpu(), {1, ch, h2, w2});
  vt::Conv2d(q, t3, t2, twp, nullptr, pws);
  vt::Relu(q, t3, t3);

  // Independent replay of the same chain through the in-test reference.
  const Case c0 = {"c0", 1, 1, tmax, mel, ch, 3, 3, 2, 2, 1, 1, 1, 1, 1, false};
  std::vector<uint8_t> r1b;
  RefConv2d(c0, x.ref, w0.ref, nullptr, DType::kF32, &r1b);
  std::vector<float> r1(reinterpret_cast<float*>(r1b.data()),
                        reinterpret_cast<float*>(r1b.data()) + s1.size());
  for (float& v : r1) v = v > 0.0f ? v : 0.0f;
  const Case c1 = {"c1", 1, ch, h1, w1, ch, 3, 3, 2, 2, 1, 1, 1, 1, ch, false};
  std::vector<uint8_t> r2b;
  RefConv2d(c1, r1, wd.ref, nullptr, DType::kF32, &r2b);
  std::vector<float> r2(reinterpret_cast<float*>(r2b.data()),
                        reinterpret_cast<float*>(r2b.data()) + s2.size());
  const Case c2 = {"c2", 1, ch, h2, w2, ch, 1, 1, 1, 1, 0, 0, 1, 1, 1, false};
  std::vector<uint8_t> r3b;
  RefConv2d(c2, r2, wp.ref, nullptr, DType::kF32, &r3b);
  std::vector<float> r3(reinterpret_cast<float*>(r3b.data()),
                        reinterpret_cast<float*>(r3b.data()) + s3.size());
  for (float& v : r3) v = v > 0.0f ? v : 0.0f;

  CHECK(std::memcmp(s1.data(), r1.data(), s1.size() * sizeof(float)) == 0);
  CHECK(std::memcmp(s2.data(), r2.data(), s2.size() * sizeof(float)) == 0);
  CHECK(std::memcmp(s3.data(), r3.data(), s3.size() * sizeof(float)) == 0);
}

TEST_CASE("conv2d: the shape contract is enforced, not assumed") {
  Queue q = CpuQueue();
  std::vector<float> x(1 * 4 * 6 * 6, 0.5f), w(6 * 2 * 3 * 3, 0.25f), o(1 * 6 * 6 * 6, 0.0f);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {1, 4, 6, 6});
  Tensor tw = Tensor::Contiguous(w.data(), DType::kF32, Cpu(), {6, 2, 3, 3});
  Tensor to = Tensor::Contiguous(o.data(), DType::kF32, Cpu(), {1, 6, 6, 6});
  Conv2dArgs g2;
  g2.groups = 2;
  g2.pad_h = g2.pad_w = 1;
  vt::Conv2d(q, to, tx, tw, nullptr, g2);  // the well-formed call

  // groups that does not divide Cout
  Conv2dArgs bad_g = g2;
  bad_g.groups = 3;  // Cin 4 % 3 != 0
  CHECK_THROWS(vt::Conv2d(q, to, tx, tw, nullptr, bad_g));
  // weight dim 1 must be Cin/groups
  Tensor tw_bad = Tensor::Contiguous(w.data(), DType::kF32, Cpu(), {6, 4, 3, 3});
  CHECK_THROWS(vt::Conv2d(q, to, tx, tw_bad, nullptr, g2));
  // out extents must follow from stride/padding/dilation
  Conv2dArgs nopad = g2;
  nopad.pad_h = nopad.pad_w = 0;
  CHECK_THROWS(vt::Conv2d(q, to, tx, tw, nullptr, nopad));
  // bias must be rank-1 [Cout]
  std::vector<float> b(3, 1.0f);
  Tensor tb = Tensor::Contiguous(b.data(), DType::kF32, Cpu(), {3});
  CHECK_THROWS(vt::Conv2d(q, to, tx, tw, &tb, g2));
}
