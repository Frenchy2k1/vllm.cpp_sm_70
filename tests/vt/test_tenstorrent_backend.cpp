// Tenstorrent backend skeleton unit gates (BACKEND-TENSTORRENT, W0). Newly
// authored — vLLM has no Tenstorrent tests to port. Mirrors the shape of
// tests/vt/test_vulkan_backend.cpp / test_metal_backend.cpp so the three are
// read side by side.
//
// This TU is COMPILED ONLY in a Tenstorrent build (tests/CMakeLists.txt gates
// it on VLLM_CPP_TENSTORRENT) and every assertion goes through the public
// vt:: seam — if the skeleton needed ttnn headers in a test to be checkable,
// the seam would be leaking. (This is also why this file needs none of the
// object-library include isolation tenstorrent_ops.cpp needed — it never
// touches ttnn/tt-metal headers at all.)
//
// Every case is SKIPPED, not failed, when no Blackhole card is present — the
// registrars stay silent by design (tenstorrent_backend.cpp/tenstorrent.cpp),
// and a Tenstorrent-enabled build legitimately runs in CI containers with no
// card. The skip is REPORTED so a silently-unregistered backend on a box that
// DOES have one cannot masquerade as a pass.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/ops.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::Queue;
using vt::Tensor;

namespace {

bool TenstorrentPresent() { return vt::TryGetBackend(DeviceType::kTENSTORRENT) != nullptr; }

}  // namespace

TEST_CASE("kTENSTORRENT backend registers iff a device is present") {
  Backend* b = vt::TryGetBackend(DeviceType::kTENSTORRENT);
  if (b == nullptr) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  CHECK_FALSE(b->UnifiedMemory());  // discrete PCIe card — see backend.h's SCOPE note
}

TEST_CASE("kTENSTORRENT Platform mirrors the registered Backend") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vllm::platforms::HasPlatform(DeviceType::kTENSTORRENT));
  vllm::platforms::Platform& p = vllm::platforms::GetPlatform(DeviceType::kTENSTORRENT);
  CHECK(p.device_type() == DeviceType::kTENSTORRENT);
  CHECK(&p.backend() == vt::TryGetBackend(DeviceType::kTENSTORRENT));
  // Registers the OPT-125m linear/residual/activation + vocab + norm slice
  // (kMatmul, kMatmulBT, kAdd, kRelu, kEmbedding, kLayerNorm) in F32 only —
  // see tenstorrent_ops.cpp.
  CHECK(p.supported_dtypes() == std::vector<vt::DType>{vt::DType::kF32});
  // No attention kernel exists yet; an empty priority list is the honest answer
  // (platforms/tenstorrent.cpp's comment on get_attn_backend_priority).
  CHECK(p.get_attn_backend_priority({}).empty());
}

TEST_CASE("kTENSTORRENT kMatmul matches a host F32 reference") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmul, DeviceType::kTENSTORRENT));

  constexpr int64_t M = 32, K = 32, N = 32;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);

  std::vector<float> host_a(M * K), host_b(K * N), host_out(M * N, 0.0f);
  for (size_t i = 0; i < host_a.size(); ++i) host_a[i] = static_cast<float>(i % 7) * 0.1f;
  for (size_t i = 0; i < host_b.size(); ++i) host_b[i] = static_cast<float>(i % 5) * 0.2f;

  void* mem_a = backend.Alloc(host_a.size() * sizeof(float));
  void* mem_b = backend.Alloc(host_b.size() * sizeof(float));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_a, host_a.data(), host_a.size() * sizeof(float));
  backend.Copy(q, mem_b, host_b.data(), host_b.size() * sizeof(float));

  Tensor a = Tensor::Contiguous(mem_a, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, K});
  Tensor b = Tensor::Contiguous(mem_b, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {K, N});
  Tensor out =
      Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, N});

  auto matmul = reinterpret_cast<vt::MatmulFn>(vt::GetOp(vt::OpId::kMatmul, DeviceType::kTENSTORRENT));
  matmul(q, out, a, b);

  backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
  backend.Free(mem_a);
  backend.Free(mem_b);
  backend.Free(mem_out);

  std::vector<float> ref(M * N, 0.0f);
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k) acc += host_a[i * K + k] * host_b[k * N + j];
      ref[i * N + j] = acc;
    }
  }

  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < ref.size(); ++i) max_abs_diff = std::max(max_abs_diff, std::fabs(host_out[i] - ref[i]));
  // bf16 accumulation over K=32 on-device — same tolerance the hands-on spike
  // measured (.agents/specs/tenstorrent-backend.md), not a rubber stamp.
  CHECK(max_abs_diff < 0.5f);
}

TEST_CASE("kTENSTORRENT kMatmulBT matches a host F32 reference (a @ b^T)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));

  // `a` is [M,K] activations; `b` is [N,K] nn.Linear weight (torch layout).
  constexpr int64_t M = 32, K = 32, N = 32;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);

  std::vector<float> host_a(M * K), host_b(N * K), host_out(M * N, 0.0f);
  for (size_t i = 0; i < host_a.size(); ++i) host_a[i] = static_cast<float>(i % 7) * 0.1f;
  for (size_t i = 0; i < host_b.size(); ++i) host_b[i] = static_cast<float>(i % 5) * 0.2f;

  void* mem_a = backend.Alloc(host_a.size() * sizeof(float));
  void* mem_b = backend.Alloc(host_b.size() * sizeof(float));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_a, host_a.data(), host_a.size() * sizeof(float));
  backend.Copy(q, mem_b, host_b.data(), host_b.size() * sizeof(float));

  Tensor a = Tensor::Contiguous(mem_a, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, K});
  Tensor b = Tensor::Contiguous(mem_b, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {N, K});
  Tensor out =
      Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, N});

  auto matmul_bt =
      reinterpret_cast<vt::MatmulFn>(vt::GetOp(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));
  matmul_bt(q, out, a, b);

  backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
  backend.Free(mem_a);
  backend.Free(mem_b);
  backend.Free(mem_out);

  std::vector<float> ref(M * N, 0.0f);
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k) acc += host_a[i * K + k] * host_b[j * K + k];
      ref[i * N + j] = acc;
    }
  }

  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < ref.size(); ++i) max_abs_diff = std::max(max_abs_diff, std::fabs(host_out[i] - ref[i]));
  CHECK(max_abs_diff < 0.5f);
}

TEST_CASE("kTENSTORRENT kAdd matches a host F32 reference (elementwise + bias broadcast)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kAdd, DeviceType::kTENSTORRENT));

  constexpr int64_t Rows = 32, D = 32;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  auto add = reinterpret_cast<vt::AddFn>(vt::GetOp(vt::OpId::kAdd, DeviceType::kTENSTORRENT));

  SUBCASE("elementwise, same rank") {
    std::vector<float> host_a(Rows * D), host_b(Rows * D), host_out(Rows * D, 0.0f);
    for (size_t i = 0; i < host_a.size(); ++i) host_a[i] = static_cast<float>(i % 7) * 0.1f;
    for (size_t i = 0; i < host_b.size(); ++i) host_b[i] = static_cast<float>(i % 5) * 0.2f;

    void* mem_a = backend.Alloc(host_a.size() * sizeof(float));
    void* mem_b = backend.Alloc(host_b.size() * sizeof(float));
    void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
    Queue q = backend.CreateQueue();
    backend.Copy(q, mem_a, host_a.data(), host_a.size() * sizeof(float));
    backend.Copy(q, mem_b, host_b.data(), host_b.size() * sizeof(float));

    Tensor a =
        Tensor::Contiguous(mem_a, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});
    Tensor b =
        Tensor::Contiguous(mem_b, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});
    Tensor out =
        Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});
    add(q, out, a, b);
    backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
    backend.Free(mem_a);
    backend.Free(mem_b);
    backend.Free(mem_out);

    float max_abs_diff = 0.0f;
    for (size_t i = 0; i < host_out.size(); ++i)
      max_abs_diff = std::max(max_abs_diff, std::fabs(host_out[i] - (host_a[i] + host_b[i])));
    CHECK(max_abs_diff < 0.1f);
  }

  SUBCASE("rank-1 bias broadcast over the last dim") {
    std::vector<float> host_a(Rows * D), host_bias(D), host_out(Rows * D, 0.0f);
    for (size_t i = 0; i < host_a.size(); ++i) host_a[i] = static_cast<float>(i % 7) * 0.1f;
    for (size_t i = 0; i < host_bias.size(); ++i) host_bias[i] = static_cast<float>(i) * 0.05f;

    void* mem_a = backend.Alloc(host_a.size() * sizeof(float));
    void* mem_bias = backend.Alloc(host_bias.size() * sizeof(float));
    void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
    Queue q = backend.CreateQueue();
    backend.Copy(q, mem_a, host_a.data(), host_a.size() * sizeof(float));
    backend.Copy(q, mem_bias, host_bias.data(), host_bias.size() * sizeof(float));

    Tensor a =
        Tensor::Contiguous(mem_a, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});
    Tensor bias =
        Tensor::Contiguous(mem_bias, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {D});
    Tensor out =
        Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});
    add(q, out, a, bias);
    backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
    backend.Free(mem_a);
    backend.Free(mem_bias);
    backend.Free(mem_out);

    float max_abs_diff = 0.0f;
    for (int64_t r = 0; r < Rows; ++r)
      for (int64_t c = 0; c < D; ++c)
        max_abs_diff = std::max(max_abs_diff,
                                 std::fabs(host_out[r * D + c] - (host_a[r * D + c] + host_bias[c])));
    CHECK(max_abs_diff < 0.1f);
  }
}

TEST_CASE("kTENSTORRENT kRelu matches a host F32 reference") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kRelu, DeviceType::kTENSTORRENT));

  constexpr int64_t Rows = 32, D = 32;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);

  std::vector<float> host_x(Rows * D), host_out(Rows * D, 0.0f);
  for (size_t i = 0; i < host_x.size(); ++i)
    host_x[i] = (static_cast<float>(i % 11) - 5.0f) * 0.3f;  // mix of signs

  void* mem_x = backend.Alloc(host_x.size() * sizeof(float));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_x, host_x.data(), host_x.size() * sizeof(float));

  Tensor x = Tensor::Contiguous(mem_x, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});
  Tensor out =
      Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});

  auto relu = reinterpret_cast<vt::ReluFn>(vt::GetOp(vt::OpId::kRelu, DeviceType::kTENSTORRENT));
  relu(q, out, x);

  backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
  backend.Free(mem_x);
  backend.Free(mem_out);

  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < host_x.size(); ++i)
    max_abs_diff = std::max(max_abs_diff, std::fabs(host_out[i] - std::max(0.0f, host_x[i])));
  CHECK(max_abs_diff < 0.1f);
}

TEST_CASE("kTENSTORRENT kEmbedding matches a host F32 reference (row gather)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kEmbedding, DeviceType::kTENSTORRENT));

  // Non-tile-aligned (t, h) on purpose: forces the ROW_MAJOR path and
  // proves download is dense without TILE padding. Vocab is modest so the
  // host oracle stays trivial.
  constexpr int64_t Vocab = 17, H = 24, T = 7;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);

  std::vector<float> host_table(Vocab * H);
  for (size_t i = 0; i < host_table.size(); ++i)
    host_table[i] = static_cast<float>(i % 13) * 0.1f - 0.5f;
  // i32 ids covering edges: first, last, and middle rows; repeats allowed.
  std::vector<int32_t> host_ids = {0, 3, 16, 1, 3, 8, 16};
  REQUIRE(static_cast<int64_t>(host_ids.size()) == T);

  std::vector<float> host_out(T * H, 0.0f);
  void* mem_table = backend.Alloc(host_table.size() * sizeof(float));
  void* mem_ids = backend.Alloc(host_ids.size() * sizeof(int32_t));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_table, host_table.data(), host_table.size() * sizeof(float));
  backend.Copy(q, mem_ids, host_ids.data(), host_ids.size() * sizeof(int32_t));

  Tensor table = Tensor::Contiguous(mem_table, vt::DType::kF32,
                                    Device{DeviceType::kTENSTORRENT, 0}, {Vocab, H});
  Tensor ids = Tensor::Contiguous(mem_ids, vt::DType::kI32,
                                  Device{DeviceType::kTENSTORRENT, 0}, {T});
  Tensor out = Tensor::Contiguous(mem_out, vt::DType::kF32,
                                  Device{DeviceType::kTENSTORRENT, 0}, {T, H});

  auto embedding =
      reinterpret_cast<vt::EmbeddingFn>(vt::GetOp(vt::OpId::kEmbedding, DeviceType::kTENSTORRENT));
  embedding(q, out, table, ids);

  backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
  backend.Free(mem_table);
  backend.Free(mem_ids);
  backend.Free(mem_out);

  // Host oracle: pure row gather. bf16 table storage means a modest abs tol.
  float max_abs_diff = 0.0f;
  for (int64_t i = 0; i < T; ++i) {
    const int32_t id = host_ids[static_cast<size_t>(i)];
    for (int64_t j = 0; j < H; ++j) {
      const float ref = host_table[static_cast<size_t>(id) * H + j];
      max_abs_diff = std::max(max_abs_diff, std::fabs(host_out[i * H + j] - ref));
    }
  }
  CHECK(max_abs_diff < 0.1f);
}

TEST_CASE("kTENSTORRENT kLayerNorm matches a host F32 reference (affine + plain)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kLayerNorm, DeviceType::kTENSTORRENT));

  // Tile-aligned so the TILE upload path is exercised cleanly (same as the
  // linear ops). Host oracle is the ATen/cpu_layernorm biased-variance form.
  constexpr int64_t Rows = 32, D = 32;
  constexpr float Eps = 1e-5f;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  auto layer_norm = reinterpret_cast<vt::LayerNormFn>(
      vt::GetOp(vt::OpId::kLayerNorm, DeviceType::kTENSTORRENT));

  auto run_case = [&](bool with_affine) {
    std::vector<float> host_x(Rows * D), host_w(D), host_b(D), host_out(Rows * D, 0.0f);
    for (size_t i = 0; i < host_x.size(); ++i)
      host_x[i] = (static_cast<float>(i % 17) - 8.0f) * 0.15f;
    for (int64_t j = 0; j < D; ++j) {
      host_w[static_cast<size_t>(j)] = 0.5f + static_cast<float>(j % 5) * 0.1f;
      host_b[static_cast<size_t>(j)] = static_cast<float>(j % 7) * 0.05f - 0.15f;
    }

    void* mem_x = backend.Alloc(host_x.size() * sizeof(float));
    void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
    void* mem_w = with_affine ? backend.Alloc(host_w.size() * sizeof(float)) : nullptr;
    void* mem_b = with_affine ? backend.Alloc(host_b.size() * sizeof(float)) : nullptr;
    Queue q = backend.CreateQueue();
    backend.Copy(q, mem_x, host_x.data(), host_x.size() * sizeof(float));
    if (with_affine) {
      backend.Copy(q, mem_w, host_w.data(), host_w.size() * sizeof(float));
      backend.Copy(q, mem_b, host_b.data(), host_b.size() * sizeof(float));
    }

    Tensor x =
        Tensor::Contiguous(mem_x, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});
    Tensor out =
        Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});
    Tensor w_t, b_t;
    const Tensor* w_ptr = nullptr;
    const Tensor* b_ptr = nullptr;
    if (with_affine) {
      w_t = Tensor::Contiguous(mem_w, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {D});
      b_t = Tensor::Contiguous(mem_b, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {D});
      w_ptr = &w_t;
      b_ptr = &b_t;
    }

    vt::LayerNormArgs args;
    args.eps = Eps;
    layer_norm(q, out, x, w_ptr, b_ptr, args);

    backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
    backend.Free(mem_x);
    backend.Free(mem_out);
    if (with_affine) {
      backend.Free(mem_w);
      backend.Free(mem_b);
    }

    float max_abs_diff = 0.0f;
    for (int64_t r = 0; r < Rows; ++r) {
      float sum = 0.0f;
      for (int64_t j = 0; j < D; ++j) sum += host_x[r * D + j];
      const float mean = sum / static_cast<float>(D);
      float sq = 0.0f;
      for (int64_t j = 0; j < D; ++j) {
        const float dv = host_x[r * D + j] - mean;
        sq += dv * dv;
      }
      const float rstd = 1.0f / std::sqrt(sq / static_cast<float>(D) + Eps);
      for (int64_t j = 0; j < D; ++j) {
        float ref = (host_x[r * D + j] - mean) * rstd;
        if (with_affine) ref = ref * host_w[static_cast<size_t>(j)] + host_b[static_cast<size_t>(j)];
        max_abs_diff = std::max(max_abs_diff, std::fabs(host_out[r * D + j] - ref));
      }
    }
    // bf16 storage + device reduction; not bit-exact vs f32 host mean/var.
    CHECK(max_abs_diff < 0.5f);
  };

  SUBCASE("elementwise_affine=True (weight + bias)") { run_case(true); }
  SUBCASE("elementwise_affine=False (no weight/bias)") { run_case(false); }
}

TEST_CASE("kTENSTORRENT kQkvSplit is BIT-EXACT vs a host reference (unequal widths)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kQkvSplit, DeviceType::kTENSTORRENT));

  // Independent q/k/v widths (kernel contract); not equal-width MHA only.
  constexpr int64_t T = 11, Qd = 24, Kd = 12, Vd = 12;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);

  std::vector<float> host_qkv(static_cast<size_t>(T * (Qd + Kd + Vd)));
  for (size_t i = 0; i < host_qkv.size(); ++i)
    host_qkv[i] = static_cast<float>(i % 19) * 0.1f - 0.7f;
  std::vector<float> host_q(static_cast<size_t>(T * Qd), 0.0f);
  std::vector<float> host_k(static_cast<size_t>(T * Kd), 0.0f);
  std::vector<float> host_v(static_cast<size_t>(T * Vd), 0.0f);

  void* mem_qkv = backend.Alloc(host_qkv.size() * sizeof(float));
  void* mem_q = backend.Alloc(host_q.size() * sizeof(float));
  void* mem_k = backend.Alloc(host_k.size() * sizeof(float));
  void* mem_v = backend.Alloc(host_v.size() * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_qkv, host_qkv.data(), host_qkv.size() * sizeof(float));

  Tensor qkv = Tensor::Contiguous(mem_qkv, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {T, Qd + Kd + Vd});
  Tensor tq =
      Tensor::Contiguous(mem_q, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {T, Qd});
  Tensor tk =
      Tensor::Contiguous(mem_k, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {T, Kd});
  Tensor tv =
      Tensor::Contiguous(mem_v, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {T, Vd});

  auto split =
      reinterpret_cast<vt::QkvSplitFn>(vt::GetOp(vt::OpId::kQkvSplit, DeviceType::kTENSTORRENT));
  split(q, tq, tk, tv, qkv);

  backend.Copy(q, host_q.data(), mem_q, host_q.size() * sizeof(float));
  backend.Copy(q, host_k.data(), mem_k, host_k.size() * sizeof(float));
  backend.Copy(q, host_v.data(), mem_v, host_v.size() * sizeof(float));
  backend.Free(mem_qkv);
  backend.Free(mem_q);
  backend.Free(mem_k);
  backend.Free(mem_v);

  // Pure column split — bit-exact.
  for (int64_t i = 0; i < T; ++i) {
    for (int64_t j = 0; j < Qd; ++j)
      CHECK(host_q[i * Qd + j] == host_qkv[i * (Qd + Kd + Vd) + j]);
    for (int64_t j = 0; j < Kd; ++j)
      CHECK(host_k[i * Kd + j] == host_qkv[i * (Qd + Kd + Vd) + Qd + j]);
    for (int64_t j = 0; j < Vd; ++j)
      CHECK(host_v[i * Vd + j] == host_qkv[i * (Qd + Kd + Vd) + Qd + Kd + j]);
  }
}
