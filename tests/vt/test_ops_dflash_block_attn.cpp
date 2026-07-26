// vllm.cpp original (vt runtime). Unit tests for vt::DFlashBlockAttention — the
// DFlash draft's in-block attention (SPEC-DFLASH D2, DF-DRAFT-MODEL), the
// project's FIRST non-causal / bidirectional attention primitive. Semantics ref:
// DFlashQwen3Attention + _resolve_layer_attention (qwen3_dflash.py:86-146,
// 149-263 @ 555967922). These pin hand-computed values for the load-bearing
// corners: BIDIRECTIONAL (non-causal) full attention, causal-within-window SWA,
// per-request BLOCK isolation (cu_seqlens), GQA mapping, and — the RED proof —
// that causal != non-causal so a wrong mask is CAUGHT.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DFlashBlockAttentionArgs;
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
Tensor F32(std::vector<float>& v, const std::vector<int64_t>& shape) {
  return Contig(v.data(), DType::kF32, Cpu(), shape);
}
DFlashBlockAttentionArgs Args(const int32_t* cu, int num_reqs, bool causal, int64_t window) {
  DFlashBlockAttentionArgs a;
  a.scale = 1.0f;
  a.causal = causal;
  a.sliding_window = window;
  a.cu_seqlens = cu;
  a.num_reqs = num_reqs;
  return a;
}
}  // namespace

TEST_CASE("dflash-block-attn NON-CAUSAL: query 0 attends to the FUTURE key (bidirectional)") {
  // T=2, one head, head_dim=2, scale=1, ONE block [0,2), NON-causal (full layer).
  //   q = [[1,0],[0,1]], k = [[1,0],[0,1]], v = [[1,2],[3,4]]
  // query 0 (NON-causal → BOTH keys): scores {q0·k0, q0·k1} = {1,0};
  //   softmax({1,0}) = {e/(1+e), 1/(1+e)} = {0.73106, 0.26894};
  //   out0 = 0.73106*[1,2] + 0.26894*[3,4] = [1.53789, 2.53789].
  // (The causal op would give out0 = v0 = [1,2] — this is the RED-separating value.)
  std::vector<float> q = {1, 0, 0, 1};
  std::vector<float> k = {1, 0, 0, 1};
  std::vector<float> v = {1, 2, 3, 4};
  std::vector<float> out(4, 0.0f);
  Tensor tq = F32(q, {2, 1, 2}), tk = F32(k, {2, 1, 2}), tv = F32(v, {2, 1, 2});
  Tensor to = F32(out, {2, 1, 2});
  Queue qq = Q();
  const int32_t cu[] = {0, 2};
  vt::DFlashBlockAttention(qq, to, tq, tk, tv, Args(cu, 1, /*causal=*/false, 0));
  CHECK(out[0] == doctest::Approx(1.53789f).epsilon(1e-4));
  CHECK(out[1] == doctest::Approx(2.53789f).epsilon(1e-4));
}

TEST_CASE("dflash-block-attn RED: causal vs non-causal DIFFER for query 0 (mask is load-bearing)") {
  // The load-bearing invariant: a full-attention layer wrongly run CAUSAL diverges.
  std::vector<float> q = {1, 0, 0, 1};
  std::vector<float> k = {1, 0, 0, 1};
  std::vector<float> v = {1, 2, 3, 4};
  std::vector<float> out_nc(4, 0.0f), out_c(4, 0.0f);
  Tensor tq = F32(q, {2, 1, 2}), tk = F32(k, {2, 1, 2}), tv = F32(v, {2, 1, 2});
  Queue qq = Q();
  const int32_t cu[] = {0, 2};
  Tensor tnc = F32(out_nc, {2, 1, 2});
  Tensor tc = F32(out_c, {2, 1, 2});
  vt::DFlashBlockAttention(qq, tnc, tq, tk, tv, Args(cu, 1, /*causal=*/false, 0));
  vt::DFlashBlockAttention(qq, tc, tq, tk, tv, Args(cu, 1, /*causal=*/true, 0));
  // Non-causal query 0 sees the future key; causal query 0 sees only key 0 (=v0).
  CHECK(out_c[0] == doctest::Approx(1.0f));
  CHECK(out_c[1] == doctest::Approx(2.0f));
  CHECK(std::fabs(out_nc[0] - out_c[0]) > 0.4f);  // 1.538 vs 1.0 — caught
}

TEST_CASE("dflash-block-attn per-request BLOCK isolation (cu_seqlens)") {
  // Two blocks of 1 token each: [0,1) and [1,2). Even non-causal, block 0's query
  // must NOT see block 1's key (and vice versa) — each is its own softmax of 1 key,
  // so out == v of that row regardless of the other block.
  std::vector<float> q = {1, 0, 0, 1};
  std::vector<float> k = {1, 0, 0, 1};
  std::vector<float> v = {7, 8, 100, 200};
  std::vector<float> out(4, 0.0f);
  Tensor tq = F32(q, {2, 1, 2}), tk = F32(k, {2, 1, 2}), tv = F32(v, {2, 1, 2});
  Tensor to = F32(out, {2, 1, 2});
  Queue qq = Q();
  const int32_t cu[] = {0, 1, 2};  // two singleton blocks
  vt::DFlashBlockAttention(qq, to, tq, tk, tv, Args(cu, 2, /*causal=*/false, 0));
  CHECK(out[0] == doctest::Approx(7.0f));    // block 0 == v[0], not pulled by v[1]
  CHECK(out[1] == doctest::Approx(8.0f));
  CHECK(out[2] == doctest::Approx(100.0f));  // block 1 == v[1]
  CHECK(out[3] == doctest::Approx(200.0f));
}

TEST_CASE("dflash-block-attn SWA window bounds the causal key range") {
  // One block [0,3), causal, window=2: query 2 sees keys {1,2} only (not key 0).
  // q2=[0,0,1] picks key with e2 component. Make v distinctive per key.
  //   q = rows e0,e1,e2 (D=3); k = e0,e1,e2; scores are the identity → query i
  //   attends most to key i. window=2 for query 2 → keys {1,2}; key 0 excluded.
  std::vector<float> q = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<float> k = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<float> v = {10, 0, 0, 0, 20, 0, 0, 0, 30};  // v0,v1,v2 distinct axes
  std::vector<float> out_w(9, 0.0f), out_full(9, 0.0f);
  Tensor tq = F32(q, {3, 1, 3}), tk = F32(k, {3, 1, 3}), tv = F32(v, {3, 1, 3});
  Queue qq = Q();
  const int32_t cu[] = {0, 3};
  Tensor tw = F32(out_w, {3, 1, 3});
  Tensor tf = F32(out_full, {3, 1, 3});
  vt::DFlashBlockAttention(qq, tw, tq, tk, tv, Args(cu, 1, /*causal=*/true, /*window=*/2));
  vt::DFlashBlockAttention(qq, tf, tq, tk, tv, Args(cu, 1, /*causal=*/true, /*window=*/0));
  // query 2 row (out[6..8]): window=2 excludes key 0 (v0 on axis 0), so out_w[6]
  // (axis-0 component) must be strictly smaller than the full-causal case which
  // DOES mix in key 0's v0=10.
  CHECK(out_w[6] < out_full[6] - 1e-3f);
}

TEST_CASE("dflash-block-attn GQA: 2 q-heads share 1 kv-head") {
  // Hq=2, Hk=1, D=1, one block [0,1) (single token). Each q-head reads kv-head 0.
  std::vector<float> q = {1, 1};      // [T=1, Hq=2, D=1]
  std::vector<float> k = {2};         // [1,1,1]
  std::vector<float> v = {5};         // [1,1,1]
  std::vector<float> out(2, 0.0f);
  Tensor tq = F32(q, {1, 2, 1}), tk = F32(k, {1, 1, 1}), tv = F32(v, {1, 1, 1});
  Tensor to = F32(out, {1, 2, 1});
  Queue qq = Q();
  const int32_t cu[] = {0, 1};
  vt::DFlashBlockAttention(qq, to, tq, tk, tv, Args(cu, 1, false, 0));
  CHECK(out[0] == doctest::Approx(5.0f));  // single key → out == v
  CHECK(out[1] == doctest::Approx(5.0f));
}
