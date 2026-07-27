// vllm.cpp original (vt runtime). Unit tests for vt::DFlashPagedBlockAttention —
// the CAPTURE-SAFE paged variant of DFlashBlockAttention (SPEC-DFLASH D12 Part B,
// the CUDA-graph draft-attention primitive). Semantics ref: the materialized
// [context; block] combined-buffer attention DFlashBlockAttention runs inside
// ForwardWithCtxKVDev (qwen3_dflash.cpp), generalized so the growing context
// enters as a PAGED K/V cache + per-request seq_lens + block_table instead of a
// variable-size combined buffer (the static-shape, device-only-metadata form a
// CUDA graph can capture).
//
// The load-bearing correctness proof is EQUIVALENCE: for every semantic corner
// (non-causal full, causal-SWA, per-request block isolation, GQA, multi-page
// context) the paged op's block-query outputs must equal DFlashBlockAttention run
// over an explicitly materialized [context; block] combined buffer with the block
// rows extracted — i.e. Part B computes exactly what the D9/D11 combined path did.
// Plus CUDA==CPU parity for the paged op itself (f32 + bf16).
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DFlashBlockAttentionArgs;
using vt::DFlashPagedBlockAttentionArgs;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {
Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue Q() { return Queue{Cpu(), nullptr}; }

Tensor Contig(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

std::vector<float> RandF32(size_t n, uint32_t seed) {
  // Deterministic LCG in [-2,2) (matches test_ops_dflash_block_attn).
  std::vector<float> v(n);
  uint32_t s = seed;
  for (auto& x : v) {
    s = s * 1664525u + 1013904223u;
    x = (static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) * 4.0f - 2.0f;
  }
  return v;
}

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Device Gpu() { return Device{DeviceType::kCUDA, 0}; }

struct QueueGuard {
  Backend& b;
  Queue q;
  explicit QueueGuard(Backend& backend) : b(backend), q(backend.CreateQueue()) {}
  ~QueueGuard() { b.DestroyQueue(q); }
  QueueGuard(const QueueGuard&) = delete;
  QueueGuard& operator=(const QueueGuard&) = delete;
};

class DeviceTensor {
 public:
  DeviceTensor(Backend& b, Queue& q, DType dt, const std::vector<int64_t>& shape,
               const void* host = nullptr)
      : b_(b) {
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p_ = b_.Alloc(bytes_ == 0 ? 1 : bytes_);
    if (host != nullptr) b_.Copy(q, p_, host, bytes_);
    t_ = Contig(p_, dt, Gpu(), shape);
  }
  ~DeviceTensor() { b_.Free(p_); }
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;
  Tensor& tensor() { return t_; }
  void Download(Queue& q, void* dst) {
    b_.Copy(q, dst, p_, bytes_);
    b_.Synchronize(q);
  }

 private:
  Backend& b_;
  void* p_ = nullptr;
  size_t bytes_ = 0;
  Tensor t_;
};

std::vector<uint16_t> ToBf16(const std::vector<float>& v) {
  // Round-to-nearest-even f32 -> bf16 (matches vt CastBf16 on device).
  std::vector<uint16_t> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    uint32_t bits;
    std::memcpy(&bits, &v[i], sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    o[i] = static_cast<uint16_t>(bits >> 16);
  }
  return o;
}
std::vector<float> FromBf16(const std::vector<uint16_t>& v) {
  std::vector<float> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    const uint32_t bits = static_cast<uint32_t>(v[i]) << 16;
    std::memcpy(&o[i], &bits, sizeof(float));
  }
  return o;
}

// A scenario: P requests, uniform (1+k) block per request, per-request context
// length ctxlen[r], packed into a paged cache with the given block_size.
struct Scenario {
  int64_t k, hq, hk, d, block_size, window;
  bool causal;
  std::vector<int64_t> ctxlen;
  uint32_t seed;
};

// Build the reference block-query outputs by running DFlashBlockAttention over an
// explicitly materialized [context; block] combined buffer per request (context
// query rows zeroed, then the block rows extracted). This is the SAME computation
// ForwardWithCtxKVDev performs, so the paged op must reproduce it.
std::vector<float> CombinedReference(const Scenario& sc, const std::vector<float>& ctx_k,
                                     const std::vector<float>& ctx_v,
                                     const std::vector<float>& blk_q,
                                     const std::vector<float>& blk_k,
                                     const std::vector<float>& blk_v) {
  const int P = static_cast<int>(sc.ctxlen.size());
  const int64_t blen = 1 + sc.k;
  const int64_t hq = sc.hq, hk = sc.hk, d = sc.d;
  // Per-request combined bounds.
  std::vector<int32_t> cu = {0};
  int64_t ncomb = 0;
  for (int r = 0; r < P; ++r) {
    ncomb += sc.ctxlen[static_cast<size_t>(r)] + blen;
    cu.push_back(static_cast<int32_t>(ncomb));
  }
  std::vector<float> cq(static_cast<size_t>(ncomb * hq * d), 0.0f);
  std::vector<float> ckv_k(static_cast<size_t>(ncomb * hk * d), 0.0f);
  std::vector<float> ckv_v(static_cast<size_t>(ncomb * hk * d), 0.0f);
  int64_t ctx_off = 0, blk_off = 0, comb = 0;
  for (int r = 0; r < P; ++r) {
    const int64_t C = sc.ctxlen[static_cast<size_t>(r)];
    // context rows -> combined [comb, comb+C): q zero, k/v = ctx.
    for (int64_t j = 0; j < C; ++j) {
      for (int64_t e = 0; e < hk * d; ++e) {
        ckv_k[static_cast<size_t>((comb + j) * hk * d + e)] =
            ctx_k[static_cast<size_t>((ctx_off + j) * hk * d + e)];
        ckv_v[static_cast<size_t>((comb + j) * hk * d + e)] =
            ctx_v[static_cast<size_t>((ctx_off + j) * hk * d + e)];
      }
    }
    // block rows -> combined [comb+C, comb+C+blen).
    for (int64_t j = 0; j < blen; ++j) {
      for (int64_t e = 0; e < hq * d; ++e)
        cq[static_cast<size_t>((comb + C + j) * hq * d + e)] =
            blk_q[static_cast<size_t>((blk_off + j) * hq * d + e)];
      for (int64_t e = 0; e < hk * d; ++e) {
        ckv_k[static_cast<size_t>((comb + C + j) * hk * d + e)] =
            blk_k[static_cast<size_t>((blk_off + j) * hk * d + e)];
        ckv_v[static_cast<size_t>((comb + C + j) * hk * d + e)] =
            blk_v[static_cast<size_t>((blk_off + j) * hk * d + e)];
      }
    }
    ctx_off += C;
    blk_off += blen;
    comb += C + blen;
  }
  std::vector<float> out_comb(static_cast<size_t>(ncomb * hq * d), 0.0f);
  Tensor tq = Contig(cq.data(), DType::kF32, Cpu(), {ncomb, hq, d});
  Tensor tk = Contig(ckv_k.data(), DType::kF32, Cpu(), {ncomb, hk, d});
  Tensor tv = Contig(ckv_v.data(), DType::kF32, Cpu(), {ncomb, hk, d});
  Tensor to = Contig(out_comb.data(), DType::kF32, Cpu(), {ncomb, hq, d});
  Queue cpuq = Q();
  DFlashBlockAttentionArgs a;
  a.scale = std::pow(static_cast<float>(d), -0.5f);
  a.causal = sc.causal;
  a.sliding_window = sc.window;
  a.cu_seqlens = cu.data();
  a.num_reqs = P;
  vt::DFlashBlockAttention(cpuq, to, tq, tk, tv, a);
  // Extract block rows -> [P*blen, hq, d].
  std::vector<float> ref(static_cast<size_t>(P * blen * hq * d), 0.0f);
  comb = 0;
  for (int r = 0; r < P; ++r) {
    const int64_t C = sc.ctxlen[static_cast<size_t>(r)];
    for (int64_t j = 0; j < blen; ++j)
      for (int64_t e = 0; e < hq * d; ++e)
        ref[static_cast<size_t>((r * blen + j) * hq * d + e)] =
            out_comb[static_cast<size_t>((comb + C + j) * hq * d + e)];
    comb += C + blen;
  }
  return ref;
}

// Pack the per-request context into a paged cache and build the metadata.
struct Paged {
  std::vector<float> ck, cv;  // [num_pages, block_size, hk, d]
  std::vector<int32_t> block_table;  // [P, max_pages]
  std::vector<int32_t> seq_lens;     // [P]
  std::vector<int32_t> cu;           // [P+1] block-row bounds
  int64_t num_pages = 0, max_pages = 0;
};

Paged PackPaged(const Scenario& sc, const std::vector<float>& ctx_k,
                const std::vector<float>& ctx_v) {
  const int P = static_cast<int>(sc.ctxlen.size());
  const int64_t bs = sc.block_size, hk = sc.hk, d = sc.d, blen = 1 + sc.k;
  Paged pg;
  std::vector<int64_t> pages_per(P);
  int64_t total_pages = 0, mx = 0;
  for (int r = 0; r < P; ++r) {
    pages_per[static_cast<size_t>(r)] = (sc.ctxlen[static_cast<size_t>(r)] + bs - 1) / bs;
    total_pages += pages_per[static_cast<size_t>(r)];
    if (pages_per[static_cast<size_t>(r)] > mx) mx = pages_per[static_cast<size_t>(r)];
  }
  if (mx == 0) mx = 1;  // block_table needs at least one column
  pg.num_pages = total_pages == 0 ? 1 : total_pages;
  pg.max_pages = mx;
  pg.ck.assign(static_cast<size_t>(pg.num_pages * bs * hk * d), 0.0f);
  pg.cv.assign(static_cast<size_t>(pg.num_pages * bs * hk * d), 0.0f);
  pg.block_table.assign(static_cast<size_t>(P * mx), 0);
  pg.seq_lens.assign(static_cast<size_t>(P), 0);
  pg.cu.assign(static_cast<size_t>(P) + 1, 0);
  int64_t page_base = 0, ctx_off = 0;
  for (int r = 0; r < P; ++r) {
    const int64_t C = sc.ctxlen[static_cast<size_t>(r)];
    pg.seq_lens[static_cast<size_t>(r)] = static_cast<int32_t>(C);
    pg.cu[static_cast<size_t>(r) + 1] = pg.cu[static_cast<size_t>(r)] + static_cast<int32_t>(blen);
    for (int64_t p = 0; p < pages_per[static_cast<size_t>(r)]; ++p)
      pg.block_table[static_cast<size_t>(r) * static_cast<size_t>(mx) + static_cast<size_t>(p)] =
          static_cast<int32_t>(page_base + p);
    for (int64_t j = 0; j < C; ++j) {
      const int64_t page = page_base + j / bs;
      const int64_t off = j % bs;
      for (int64_t e = 0; e < hk * d; ++e) {
        pg.ck[static_cast<size_t>((page * bs + off) * hk * d + e)] =
            ctx_k[static_cast<size_t>((ctx_off + j) * hk * d + e)];
        pg.cv[static_cast<size_t>((page * bs + off) * hk * d + e)] =
            ctx_v[static_cast<size_t>((ctx_off + j) * hk * d + e)];
      }
    }
    page_base += pages_per[static_cast<size_t>(r)];
    ctx_off += C;
  }
  return pg;
}

DFlashPagedBlockAttentionArgs PagedArgs(const Scenario& sc) {
  DFlashPagedBlockAttentionArgs a;
  a.scale = std::pow(static_cast<float>(sc.d), -0.5f);
  a.causal = sc.causal;
  a.sliding_window = sc.window;
  a.num_reqs = static_cast<int>(sc.ctxlen.size());
  a.block_size = sc.block_size;
  return a;
}

// Run one scenario: assert CPU-paged == combined reference, and (if CUDA) both the
// f32 and bf16 CUDA-paged == CPU-paged.
void RunScenario(const Scenario& sc) {
  const int P = static_cast<int>(sc.ctxlen.size());
  const int64_t blen = 1 + sc.k;
  const int64_t nq = P * blen;
  int64_t total_ctx = 0;
  for (auto c : sc.ctxlen) total_ctx += c;
  auto blk_q = RandF32(static_cast<size_t>(nq * sc.hq * sc.d), sc.seed);
  auto blk_k = RandF32(static_cast<size_t>(nq * sc.hk * sc.d), sc.seed + 1);
  auto blk_v = RandF32(static_cast<size_t>(nq * sc.hk * sc.d), sc.seed + 2);
  auto ctx_k = RandF32(static_cast<size_t>(total_ctx * sc.hk * sc.d), sc.seed + 3);
  auto ctx_v = RandF32(static_cast<size_t>(total_ctx * sc.hk * sc.d), sc.seed + 4);

  const std::vector<float> ref = CombinedReference(sc, ctx_k, ctx_v, blk_q, blk_k, blk_v);
  Paged pg = PackPaged(sc, ctx_k, ctx_v);

  // CPU paged.
  std::vector<float> cpu(static_cast<size_t>(nq * sc.hq * sc.d), 0.0f);
  {
    Queue cpuq = Q();
    Tensor tq = Contig(blk_q.data(), DType::kF32, Cpu(), {nq, sc.hq, sc.d});
    Tensor tbk = Contig(blk_k.data(), DType::kF32, Cpu(), {nq, sc.hk, sc.d});
    Tensor tbv = Contig(blk_v.data(), DType::kF32, Cpu(), {nq, sc.hk, sc.d});
    Tensor tck = Contig(pg.ck.data(), DType::kF32, Cpu(), {pg.num_pages, sc.block_size, sc.hk, sc.d});
    Tensor tcv = Contig(pg.cv.data(), DType::kF32, Cpu(), {pg.num_pages, sc.block_size, sc.hk, sc.d});
    Tensor tcu = Contig(pg.cu.data(), DType::kI32, Cpu(), {P + 1});
    Tensor tsl = Contig(pg.seq_lens.data(), DType::kI32, Cpu(), {P});
    Tensor tbt = Contig(pg.block_table.data(), DType::kI32, Cpu(), {P, pg.max_pages});
    Tensor to = Contig(cpu.data(), DType::kF32, Cpu(), {nq, sc.hq, sc.d});
    vt::DFlashPagedBlockAttention(cpuq, to, tq, tbk, tbv, tck, tcv, tcu, tsl, tbt, PagedArgs(sc));
  }
  // CPU paged == combined reference (both f32, same key order -> tight).
  for (size_t i = 0; i < ref.size(); ++i)
    CHECK(cpu[i] == doctest::Approx(ref[i]).epsilon(2e-5));

  if (!HasCuda()) return;
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);

  // f32 CUDA paged == CPU paged.
  {
    DeviceTensor dq(gpu, g.q, DType::kF32, {nq, sc.hq, sc.d}, blk_q.data());
    DeviceTensor dbk(gpu, g.q, DType::kF32, {nq, sc.hk, sc.d}, blk_k.data());
    DeviceTensor dbv(gpu, g.q, DType::kF32, {nq, sc.hk, sc.d}, blk_v.data());
    DeviceTensor dck(gpu, g.q, DType::kF32, {pg.num_pages, sc.block_size, sc.hk, sc.d}, pg.ck.data());
    DeviceTensor dcv(gpu, g.q, DType::kF32, {pg.num_pages, sc.block_size, sc.hk, sc.d}, pg.cv.data());
    DeviceTensor dcu(gpu, g.q, DType::kI32, {P + 1}, pg.cu.data());
    DeviceTensor dsl(gpu, g.q, DType::kI32, {P}, pg.seq_lens.data());
    DeviceTensor dbt(gpu, g.q, DType::kI32, {P, pg.max_pages}, pg.block_table.data());
    DeviceTensor dout(gpu, g.q, DType::kF32, {nq, sc.hq, sc.d});
    vt::DFlashPagedBlockAttention(g.q, dout.tensor(), dq.tensor(), dbk.tensor(), dbv.tensor(),
                                  dck.tensor(), dcv.tensor(), dcu.tensor(), dsl.tensor(),
                                  dbt.tensor(), PagedArgs(sc));
    std::vector<float> got(cpu.size(), 0.0f);
    dout.Download(g.q, got.data());
    for (size_t i = 0; i < cpu.size(); ++i)
      CHECK(got[i] == doctest::Approx(cpu[i]).epsilon(1e-4));
  }

  // bf16 CUDA paged == bf16 CPU paged (the production dtype). Feed bit-identical
  // bf16 inputs to both so the only divergence is the online-softmax rounding.
  {
    auto bq = ToBf16(blk_q), bk = ToBf16(blk_k), bv = ToBf16(blk_v);
    auto bck = ToBf16(pg.ck), bcv = ToBf16(pg.cv);
    // bf16 CPU paged reference.
    std::vector<uint16_t> bcpu(static_cast<size_t>(nq * sc.hq * sc.d), 0);
    {
      Queue cpuq = Q();
      Tensor tq = Contig(bq.data(), DType::kBF16, Cpu(), {nq, sc.hq, sc.d});
      Tensor tbk = Contig(bk.data(), DType::kBF16, Cpu(), {nq, sc.hk, sc.d});
      Tensor tbv = Contig(bv.data(), DType::kBF16, Cpu(), {nq, sc.hk, sc.d});
      Tensor tck = Contig(bck.data(), DType::kBF16, Cpu(), {pg.num_pages, sc.block_size, sc.hk, sc.d});
      Tensor tcv = Contig(bcv.data(), DType::kBF16, Cpu(), {pg.num_pages, sc.block_size, sc.hk, sc.d});
      Tensor tcu = Contig(pg.cu.data(), DType::kI32, Cpu(), {P + 1});
      Tensor tsl = Contig(pg.seq_lens.data(), DType::kI32, Cpu(), {P});
      Tensor tbt = Contig(pg.block_table.data(), DType::kI32, Cpu(), {P, pg.max_pages});
      Tensor to = Contig(bcpu.data(), DType::kBF16, Cpu(), {nq, sc.hq, sc.d});
      vt::DFlashPagedBlockAttention(cpuq, to, tq, tbk, tbv, tck, tcv, tcu, tsl, tbt, PagedArgs(sc));
    }
    DeviceTensor dq(gpu, g.q, DType::kBF16, {nq, sc.hq, sc.d}, bq.data());
    DeviceTensor dbk(gpu, g.q, DType::kBF16, {nq, sc.hk, sc.d}, bk.data());
    DeviceTensor dbv(gpu, g.q, DType::kBF16, {nq, sc.hk, sc.d}, bv.data());
    DeviceTensor dck(gpu, g.q, DType::kBF16, {pg.num_pages, sc.block_size, sc.hk, sc.d}, bck.data());
    DeviceTensor dcv(gpu, g.q, DType::kBF16, {pg.num_pages, sc.block_size, sc.hk, sc.d}, bcv.data());
    DeviceTensor dcu(gpu, g.q, DType::kI32, {P + 1}, pg.cu.data());
    DeviceTensor dsl(gpu, g.q, DType::kI32, {P}, pg.seq_lens.data());
    DeviceTensor dbt(gpu, g.q, DType::kI32, {P, pg.max_pages}, pg.block_table.data());
    DeviceTensor dout(gpu, g.q, DType::kBF16, {nq, sc.hq, sc.d});
    vt::DFlashPagedBlockAttention(g.q, dout.tensor(), dq.tensor(), dbk.tensor(), dbv.tensor(),
                                  dck.tensor(), dcv.tensor(), dcu.tensor(), dsl.tensor(),
                                  dbt.tensor(), PagedArgs(sc));
    std::vector<uint16_t> gotb(bcpu.size(), 0);
    dout.Download(g.q, gotb.data());
    auto cf = FromBf16(bcpu), gf = FromBf16(gotb);
    for (size_t i = 0; i < cf.size(); ++i)
      CHECK(gf[i] == doctest::Approx(cf[i]).epsilon(3e-2));
  }
}
}  // namespace

TEST_CASE("dflash-paged-block-attn == materialized combined DFlashBlockAttention (CPU)") {
  // (1) NON-CAUSAL full attention, GQA, real head_dim, k=16 block, single request,
  //     multi-page context (block_size 4, ctx 10 -> 3 pages).
  RunScenario({16, 32, 8, 128, 4, 0, false, {10}, 100});
  // (2) plain CAUSAL (window >> combined so it degenerates to causal), longer ctx.
  RunScenario({16, 32, 8, 128, 8, 4096, true, {23}, 200});
  // (3) per-request BLOCK isolation: 3 requests, ragged contexts, one page each big.
  RunScenario({16, 16, 4, 64, 16, 0, false, {5, 12, 0}, 300});
  // (4) SWA window strictly bounds the causal key range (window 6 < ctx+block).
  RunScenario({8, 8, 2, 32, 4, 6, true, {9}, 400});
  // (5) GQA extreme (8 q-heads share 1 kv-head), 2 ragged requests, small block_size.
  RunScenario({6, 8, 1, 16, 2, 0, false, {7, 3}, 500});
  // (6) ZERO context (first draft step): pure in-block attention, must still match.
  RunScenario({16, 32, 8, 128, 4, 0, false, {0}, 600});
}
