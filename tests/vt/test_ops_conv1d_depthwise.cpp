// Byte-exactness + determinism gate for `vt::DepthwiseConv1d` (spike row P2,
// .agents/specs/parakeet-conformer-encoder.md; kernel
// src/vt/cpu/cpu_conv1d_depthwise.cpp).
//
// The op mirrors torch `nn.Conv1d(C, C, K, stride=1, padding=(K-1)//2,
// groups=C)` as `ParakeetEncoderConvolutionModule` instantiates it
// (transformers 5.3.0 transformers/models/parakeet/modeling_parakeet.py:116,
// padding :136, constructor :138-146, applied :180) — the module vLLM runs
// (vllm/model_executor/models/parakeet.py:37,62) — and the structurally
// identical vLLM-native `ConformerConvolution.depthwise_conv`
// (conformer_encoder.py:229).
//
// GATE, matching tests/vt/test_ops_matmul_elem.cpp discipline: `memcmp` against
// an INDEPENDENT in-test scalar reference, NOT against the kernel. Coverage:
// every x/weight/out dtype combination over f32/f16/bf16; the SAME-padding
// odd-kernel shapes the conformer actually uses (K = 3/9/31/33) alongside
// strided, dilated and zero-padding forms; ragged lengths where the padded
// window falls off both ends; L < K; batch > 1; both accepted weight layouts
// ([C,1,K] torch and [C,K]); and thread counts 1/2/4/8.
//
// It also pins the SIBLING contract: this op must not perturb
// `vt::CausalConv1dFwd`, and with left-only padding + no initial state the two
// must agree element for element. That case is here because "add a non-causal
// sibling, do NOT modify the causal one" is the row's binding instruction.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "vt/cpu/cpu_threadpool.h"  // Threadpool::SwapForTesting (via -I src)
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::CausalConv1dArgs;
using vt::DepthwiseConv1dArgs;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue CpuQueue() { return Queue{Cpu(), nullptr}; }

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
  uint32_t Next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(s >> 33);
  }
  float Uniform() { return static_cast<float>(Next() % 20001) / 10000.0f - 1.0f; }
};

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
  int64_t n, c, lin, k, stride, pad, dil;
  bool bias;
};

int64_t OutLen(const Case& c) {
  return (c.lin + 2 * c.pad - c.dil * (c.k - 1) - 1) / c.stride + 1;
}

// INDEPENDENT scalar reference, written from the depthwise-conv1d definition:
// channel c reads channel c only; one f32 accumulator over increasing taps;
// taps outside [0, Lin) skipped (zero padding); bias added last.
void RefDepthwise(const Case& c, const std::vector<float>& x, const std::vector<float>& w,
                  const std::vector<float>* bias, DType odt, std::vector<uint8_t>* out) {
  const int64_t lout = OutLen(c);
  out->assign(static_cast<size_t>(c.n * c.c * lout) * vt::SizeOf(odt), 0);
  for (int64_t bn = 0; bn < c.n; ++bn) {
    for (int64_t ch = 0; ch < c.c; ++ch) {
      for (int64_t ol = 0; ol < lout; ++ol) {
        float acc = 0.0f;
        for (int64_t k = 0; k < c.k; ++k) {
          const int64_t il = ol * c.stride - c.pad + k * c.dil;
          if (il < 0 || il >= c.lin) continue;
          acc += x[static_cast<size_t>((bn * c.c + ch) * c.lin + il)] *
                 w[static_cast<size_t>(ch * c.k + k)];
        }
        if (bias != nullptr) acc += (*bias)[static_cast<size_t>(ch)];
        StoreAt(out, (bn * c.c + ch) * lout + ol, odt, acc);
      }
    }
  }
}

const std::vector<Case>& Cases() {
  static const std::vector<Case> c = {
      // SAME padding, odd kernel — the conformer configuration
      // (padding == (K-1)//2, modeling_parakeet.py:136). K=9 is the Parakeet
      // default conv_kernel_size; K=31/33 are the NeMo/FireRedASR conformer
      // values (conformer_encoder.py:221 default kernel_size=33).
      {"same k3", 1, 4, 12, 3, 1, 1, 1, true},
      {"same k9", 1, 3, 17, 9, 1, 4, 1, false},
      {"same k31", 2, 2, 40, 31, 1, 15, 1, true},
      {"same k33", 1, 2, 33, 33, 1, 16, 1, true},
      // L shorter than the kernel: every output tap is partly out of range.
      {"L < K", 1, 3, 5, 9, 1, 4, 1, true},
      {"L == 1", 2, 4, 1, 3, 1, 1, 1, true},
      // no padding at all, and asymmetric-ish padding via a stride that leaves
      // a truncated final window.
      {"valid k5", 1, 3, 11, 5, 1, 0, 1, false},
      {"stride 2 ragged", 1, 5, 13, 3, 2, 1, 1, true},
      {"stride 3", 2, 3, 20, 5, 3, 2, 1, false},
      // dilation (not used by Parakeet, but nn.Conv1d has it and the op takes it)
      {"dilated d2", 1, 4, 21, 3, 1, 2, 2, true},
      {"dilated d3 s2", 1, 2, 25, 4, 2, 3, 3, false},
      // wide channel count so the (batch, channel) row partition is non-trivial
      {"wide C", 3, 17, 19, 7, 1, 3, 1, true},
      {"single tap", 1, 6, 9, 1, 1, 0, 1, true},
  };
  return c;
}

void RunCase(const Case& c, DType xdt, DType wdt, DType odt, uint64_t seed, bool rank3_weight) {
  Queue q = CpuQueue();
  const int64_t lout = OutLen(c);
  REQUIRE(lout > 0);
  Rng rng(seed);
  Buf x(xdt, c.n * c.c * c.lin, rng);
  Buf w(wdt, c.c * c.k, rng);
  Buf b(wdt, c.c, rng);

  std::vector<uint8_t> got(static_cast<size_t>(c.n * c.c * lout) * vt::SizeOf(odt), 0xAB);
  Tensor tx = Tensor::Contiguous(x.Data(), xdt, Cpu(), {c.n, c.c, c.lin});
  Tensor tw = rank3_weight ? Tensor::Contiguous(w.Data(), wdt, Cpu(), {c.c, 1, c.k})
                           : Tensor::Contiguous(w.Data(), wdt, Cpu(), {c.c, c.k});
  Tensor tb = Tensor::Contiguous(b.Data(), wdt, Cpu(), {c.c});
  Tensor to = Tensor::Contiguous(got.data(), odt, Cpu(), {c.n, c.c, lout});
  DepthwiseConv1dArgs args;
  args.stride = c.stride;
  args.padding = c.pad;
  args.dilation = c.dil;
  vt::DepthwiseConv1d(q, to, tx, tw, c.bias ? &tb : nullptr, args);

  std::vector<uint8_t> want;
  RefDepthwise(c, x.ref, w.ref, c.bias ? &b.ref : nullptr, odt, &want);
  REQUIRE(want.size() == got.size());
  CHECK_MESSAGE(std::memcmp(got.data(), want.data(), want.size()) == 0, c.name);
}

}  // namespace

TEST_CASE("depthwise conv1d: byte-identical to the scalar reference over every dtype x shape") {
  const DType kDt[3] = {DType::kF32, DType::kF16, DType::kBF16};
  uint64_t seed = 1;
  for (DType xdt : kDt) {
    for (DType wdt : kDt) {
      for (DType odt : kDt) {
        for (const Case& c : Cases()) RunCase(c, xdt, wdt, odt, seed++, /*rank3_weight=*/true);
      }
    }
  }
}

TEST_CASE("depthwise conv1d: the [C,K] weight layout equals torch's [C,1,K]") {
  // Same contiguous bytes, so the two must produce identical output; the op
  // accepts both because vt::CausalConv1dFwd already carries [C,K] filters.
  uint64_t seed = 7000;
  for (const Case& c : Cases()) RunCase(c, DType::kF32, DType::kF32, DType::kF32, seed++, false);
}

TEST_CASE("depthwise conv1d: byte-identical across thread counts") {
  // Determinism contract (src/vt/cpu/cpu_conv1d_depthwise.cpp): parallelism
  // partitions (batch, channel) rows only. 3*17 == 51 rows does not divide by
  // 8, so an order-sensitive kernel would diverge here.
  Queue q = CpuQueue();
  const Case c = {"threads", 3, 17, 19, 7, 1, 3, 1, true};
  const int64_t lout = OutLen(c);
  Rng rng(31337);
  Buf x(DType::kBF16, c.n * c.c * c.lin, rng);
  Buf w(DType::kBF16, c.c * c.k, rng);
  Buf b(DType::kBF16, c.c, rng);
  DepthwiseConv1dArgs args;
  args.stride = c.stride;
  args.padding = c.pad;
  args.dilation = c.dil;
  std::vector<uint8_t> base;
  for (int nth : {1, 2, 4, 8}) {
    vt::cpu::Threadpool tp(nth);
    vt::cpu::Threadpool* prev = vt::cpu::Threadpool::SwapForTesting(&tp);
    std::vector<uint8_t> got(static_cast<size_t>(c.n * c.c * lout) * sizeof(float), 0xAB);
    Tensor tx = Tensor::Contiguous(x.Data(), DType::kBF16, Cpu(), {c.n, c.c, c.lin});
    Tensor tw = Tensor::Contiguous(w.Data(), DType::kBF16, Cpu(), {c.c, 1, c.k});
    Tensor tb = Tensor::Contiguous(b.Data(), DType::kBF16, Cpu(), {c.c});
    Tensor to = Tensor::Contiguous(got.data(), DType::kF32, Cpu(), {c.n, c.c, lout});
    vt::DepthwiseConv1d(q, to, tx, tw, &tb, args);
    vt::cpu::Threadpool::SwapForTesting(prev);
    if (base.empty()) {
      base = got;
      std::vector<uint8_t> want;
      RefDepthwise(c, x.ref, w.ref, &b.ref, DType::kF32, &want);
      CHECK(std::memcmp(base.data(), want.data(), want.size()) == 0);
    } else {
      CHECK(std::memcmp(base.data(), got.data(), base.size()) == 0);
    }
  }
}

TEST_CASE("depthwise conv1d: LEFT-only padding reproduces the causal conv1d") {
  // The row's binding instruction is "add a non-causal sibling, do NOT modify
  // the causal one". This is the cross-check that they are the same operator
  // under different padding: with padding == K-1 the non-causal window
  // [ol-(K-1), ol] is exactly the causal window, so with no initial state and
  // no SiLU the two ops must agree element for element -- proving the new op is
  // a generalization and, by running the causal op unchanged in the same
  // binary, that its behavior was not disturbed.
  //
  // Layout note: CausalConv1dFwd takes x as [T, C] token-major with a
  // query_start_loc, DepthwiseConv1d takes [N, C, L] channel-major, so the
  // inputs are transposed between the two calls.
  //
  // Both are run BIAS-FREE on purpose: the causal op seeds its accumulator with
  // the bias (cpu_ops.cpp CausalConv1dFwdKernel) while this op adds it last, so
  // with a bias the two agree in exact arithmetic but not bit-for-bit. That
  // difference is a deliberate consequence of keeping the causal kernel
  // untouched, and the bias-free comparison is the one that is byte-exact.
  Queue q = CpuQueue();
  const int64_t t = 14, c = 6, k = 4;
  Rng rng(4242);
  Buf xc(DType::kF32, t * c, rng);   // [T, C] for the causal op
  Buf w(DType::kF32, c * k, rng);

  std::vector<float> causal(static_cast<size_t>(t * c), 0.0f);
  std::vector<float> conv_state(static_cast<size_t>(c * (k - 1)), 0.0f);
  std::vector<int32_t> qsl = {0, static_cast<int32_t>(t)};
  std::vector<int32_t> has_init = {0};
  Tensor tx = Tensor::Contiguous(xc.Data(), DType::kF32, Cpu(), {t, c});
  Tensor tw = Tensor::Contiguous(w.Data(), DType::kF32, Cpu(), {c, k});
  Tensor tstate = Tensor::Contiguous(conv_state.data(), DType::kF32, Cpu(), {1, c, k - 1});
  Tensor tqsl = Tensor::Contiguous(qsl.data(), DType::kI32, Cpu(), {2});
  Tensor thi = Tensor::Contiguous(has_init.data(), DType::kI32, Cpu(), {1});
  Tensor tco = Tensor::Contiguous(causal.data(), DType::kF32, Cpu(), {t, c});
  CausalConv1dArgs cargs;
  cargs.silu_activation = false;  // the conformer applies its activation outside
  vt::CausalConv1dFwd(q, tco, tx, tw, nullptr, tstate, tqsl, thi, cargs);

  // The same data as [1, C, T] for the non-causal op, with padding == K-1.
  std::vector<float> xt(static_cast<size_t>(c * t), 0.0f);
  for (int64_t i = 0; i < t; ++i)
    for (int64_t j = 0; j < c; ++j)
      xt[static_cast<size_t>(j * t + i)] = xc.ref[static_cast<size_t>(i * c + j)];
  std::vector<float> nc(static_cast<size_t>(c * t), 0.0f);
  Tensor txt = Tensor::Contiguous(xt.data(), DType::kF32, Cpu(), {1, c, t});
  DepthwiseConv1dArgs dargs;
  dargs.padding = k - 1;  // left-only after the [..., :T] truncation below
  // (Lin + 2*(K-1) - (K-1) - 1) + 1 == T + (K-1) outputs; the first T are the
  // left-padded (causal) ones, the trailing K-1 are the right-padded tail.
  const int64_t lout = t + (k - 1);
  std::vector<float> full(static_cast<size_t>(c * lout), 0.0f);
  Tensor tfull = Tensor::Contiguous(full.data(), DType::kF32, Cpu(), {1, c, lout});
  vt::DepthwiseConv1d(q, tfull, txt, tw, nullptr, dargs);
  for (int64_t j = 0; j < c; ++j)
    for (int64_t i = 0; i < t; ++i)
      nc[static_cast<size_t>(j * t + i)] = full[static_cast<size_t>(j * lout + i)];

  for (int64_t i = 0; i < t; ++i) {
    for (int64_t j = 0; j < c; ++j) {
      CHECK(causal[static_cast<size_t>(i * c + j)] == nc[static_cast<size_t>(j * t + i)]);
    }
  }
}

TEST_CASE("depthwise conv1d: the shape contract is enforced, not assumed") {
  Queue q = CpuQueue();
  std::vector<float> x(2 * 4 * 10, 0.5f), w(4 * 3, 0.25f), o(2 * 4 * 10, 0.0f);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {2, 4, 10});
  Tensor tw = Tensor::Contiguous(w.data(), DType::kF32, Cpu(), {4, 1, 3});
  Tensor to = Tensor::Contiguous(o.data(), DType::kF32, Cpu(), {2, 4, 10});
  DepthwiseConv1dArgs same;
  same.padding = 1;
  vt::DepthwiseConv1d(q, to, tx, tw, nullptr, same);  // the well-formed call

  // weight dim 0 must be C (the depthwise/groups==C contract)
  std::vector<float> w2(3 * 3, 0.25f);
  Tensor tw2 = Tensor::Contiguous(w2.data(), DType::kF32, Cpu(), {3, 1, 3});
  CHECK_THROWS(vt::DepthwiseConv1d(q, to, tx, tw2, nullptr, same));
  // weight dim 1 must be 1 (in_channels/groups)
  std::vector<float> w3(4 * 2 * 3, 0.25f);
  Tensor tw3 = Tensor::Contiguous(w3.data(), DType::kF32, Cpu(), {4, 2, 3});
  CHECK_THROWS(vt::DepthwiseConv1d(q, to, tx, tw3, nullptr, same));
  // out length must follow from stride/padding/dilation
  DepthwiseConv1dArgs valid;  // padding 0 -> Lout == 8, not 10
  CHECK_THROWS(vt::DepthwiseConv1d(q, to, tx, tw, nullptr, valid));
  // bias must be rank-1 [C]
  std::vector<float> b(3, 1.0f);
  Tensor tb = Tensor::Contiguous(b.data(), DType::kF32, Cpu(), {3});
  CHECK_THROWS(vt::DepthwiseConv1d(q, to, tx, tw, &tb, same));
}
