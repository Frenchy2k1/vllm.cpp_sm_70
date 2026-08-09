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
#include <cstring>
#include <limits>
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
  // OPT-125m path: BF16 weights/activations + F32 logits — see tenstorrent_ops.cpp.
  CHECK(p.supported_dtypes() ==
        std::vector<vt::DType>{vt::DType::kBF16, vt::DType::kF32});
  // FLASH_ATTN is registered against the NHD layout our kPagedAttention uses.
  CHECK(p.get_attn_backend_priority({}) == std::vector<std::string>{"FLASH_ATTN"});
  CHECK(p.supports_model_architecture("OPTForCausalLM"));
  CHECK_FALSE(p.supports_model_architecture("Qwen3ForCausalLM"));
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

// First Qwen3-dense (`Qwen3ForCausalLM`) op beyond OPT's set. Host oracle is
// cpu_ops RmsNormKernel (no residual, gemma=false) — the Qwen3 default.
TEST_CASE("kTENSTORRENT kRmsNorm matches a host F32 reference (weight, no residual)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kRmsNorm, DeviceType::kTENSTORRENT));

  constexpr int64_t Rows = 32, D = 32;
  constexpr float Eps = 1e-6f;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  auto rms_norm = reinterpret_cast<vt::RmsNormFn>(
      vt::GetOp(vt::OpId::kRmsNorm, DeviceType::kTENSTORRENT));

  std::vector<float> host_x(Rows * D), host_w(D), host_out(Rows * D, 0.0f);
  for (size_t i = 0; i < host_x.size(); ++i)
    host_x[i] = (static_cast<float>(i % 17) - 8.0f) * 0.15f;
  for (int64_t j = 0; j < D; ++j)
    host_w[static_cast<size_t>(j)] = 0.5f + static_cast<float>(j % 5) * 0.1f;

  void* mem_x = backend.Alloc(host_x.size() * sizeof(float));
  void* mem_w = backend.Alloc(host_w.size() * sizeof(float));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_x, host_x.data(), host_x.size() * sizeof(float));
  backend.Copy(q, mem_w, host_w.data(), host_w.size() * sizeof(float));

  Tensor x =
      Tensor::Contiguous(mem_x, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});
  Tensor w =
      Tensor::Contiguous(mem_w, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {D});
  Tensor out =
      Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});

  vt::RmsNormArgs args;
  args.eps = Eps;
  args.gemma = false;
  rms_norm(q, out, x, w, args, /*residual=*/nullptr);

  backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
  backend.Free(mem_x);
  backend.Free(mem_w);
  backend.Free(mem_out);

  float max_abs_diff = 0.0f;
  for (int64_t r = 0; r < Rows; ++r) {
    float sumsq = 0.0f;
    for (int64_t j = 0; j < D; ++j) {
      const float v = host_x[r * D + j];
      sumsq += v * v;
    }
    const float inv = 1.0f / std::sqrt(sumsq / static_cast<float>(D) + Eps);
    for (int64_t j = 0; j < D; ++j) {
      const float ref = host_x[r * D + j] * inv * host_w[static_cast<size_t>(j)];
      max_abs_diff = std::max(max_abs_diff, std::fabs(host_out[r * D + j] - ref));
    }
  }
  // bf16 storage + device reduction; same envelope as kLayerNorm.
  CHECK(max_abs_diff < 0.5f);
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

TEST_CASE("kTENSTORRENT kReshapeAndCache is BIT-EXACT incl. slot<0 skip") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kReshapeAndCache, DeviceType::kTENSTORRENT));

  constexpr int64_t NBlocks = 6, Bsz = 8, Hkv = 3, Dh = 16, T = 10;
  constexpr int64_t Page = Hkv * Dh;
  const size_t cache_elems = static_cast<size_t>(NBlocks * Bsz * Hkv * Dh);
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);

  std::vector<float> host_k(static_cast<size_t>(T * Page)), host_v(static_cast<size_t>(T * Page));
  for (size_t i = 0; i < host_k.size(); ++i) {
    host_k[i] = static_cast<float>(i % 11) * 0.1f;
    host_v[i] = static_cast<float>(i % 13) * 0.05f - 0.2f;
  }
  // Scattered slots + one padded (-1) token that must leave its page untouched.
  std::vector<int64_t> slots{0, 9, 17, 3, -1, 40, 25, 8, 33, 11};
  REQUIRE(static_cast<int64_t>(slots.size()) == T);

  std::vector<float> seed(cache_elems);
  for (size_t i = 0; i < seed.size(); ++i) seed[i] = static_cast<float>(i % 977) * 0.001f;
  std::vector<float> host_kc = seed, host_vc = seed;

  void* mem_k = backend.Alloc(host_k.size() * sizeof(float));
  void* mem_v = backend.Alloc(host_v.size() * sizeof(float));
  void* mem_kc = backend.Alloc(cache_elems * sizeof(float));
  void* mem_vc = backend.Alloc(cache_elems * sizeof(float));
  void* mem_slots = backend.Alloc(slots.size() * sizeof(int64_t));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_k, host_k.data(), host_k.size() * sizeof(float));
  backend.Copy(q, mem_v, host_v.data(), host_v.size() * sizeof(float));
  backend.Copy(q, mem_kc, host_kc.data(), host_kc.size() * sizeof(float));
  backend.Copy(q, mem_vc, host_vc.data(), host_vc.size() * sizeof(float));
  backend.Copy(q, mem_slots, slots.data(), slots.size() * sizeof(int64_t));

  Tensor tk = Tensor::Contiguous(mem_k, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                 {T, Hkv, Dh});
  Tensor tv = Tensor::Contiguous(mem_v, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                 {T, Hkv, Dh});
  Tensor tkc = Tensor::Contiguous(mem_kc, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {NBlocks, Bsz, Hkv, Dh});
  Tensor tvc = Tensor::Contiguous(mem_vc, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {NBlocks, Bsz, Hkv, Dh});
  Tensor tsl = Tensor::Contiguous(mem_slots, vt::DType::kI64, Device{DeviceType::kTENSTORRENT, 0},
                                  {T});

  auto rac = reinterpret_cast<vt::ReshapeAndCacheFn>(
      vt::GetOp(vt::OpId::kReshapeAndCache, DeviceType::kTENSTORRENT));
  rac(q, tk, tv, tkc, tvc, tsl);

  backend.Copy(q, host_kc.data(), mem_kc, host_kc.size() * sizeof(float));
  backend.Copy(q, host_vc.data(), mem_vc, host_vc.size() * sizeof(float));
  backend.Free(mem_k);
  backend.Free(mem_v);
  backend.Free(mem_kc);
  backend.Free(mem_vc);
  backend.Free(mem_slots);

  // Host oracle: same stride math as cpu_cache.cpp.
  std::vector<float> ref_kc = seed, ref_vc = seed;
  for (int64_t t = 0; t < T; ++t) {
    const int64_t slot = slots[static_cast<size_t>(t)];
    if (slot < 0) continue;
    const int64_t block = slot / Bsz;
    const int64_t offset = slot % Bsz;
    const int64_t dst = (block * Bsz + offset) * Page;
    std::memcpy(ref_kc.data() + dst, host_k.data() + t * Page, static_cast<size_t>(Page) * sizeof(float));
    std::memcpy(ref_vc.data() + dst, host_v.data() + t * Page, static_cast<size_t>(Page) * sizeof(float));
  }
  CHECK(host_kc == ref_kc);
  CHECK(host_vc == ref_vc);
}

TEST_CASE("kTENSTORRENT kPagedAttention matches a host causal GQA reference") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kPagedAttention, DeviceType::kTENSTORRENT));

  // Single-request prefill: T=4 tokens, Hq=4, Hkv=2 (GQA 2:1), D=8, block=4.
  constexpr int64_t T = 4, Hq = 4, Hkv = 2, D = 8, Bsz = 4, NBlocks = 2;
  constexpr int64_t Page = Hkv * D;
  const size_t cache_elems = static_cast<size_t>(NBlocks * Bsz * Page);
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);

  std::vector<float> host_q(static_cast<size_t>(T * Hq * D));
  std::vector<float> host_k(static_cast<size_t>(T * Page)), host_v(static_cast<size_t>(T * Page));
  for (size_t i = 0; i < host_q.size(); ++i) host_q[i] = static_cast<float>(i % 7) * 0.1f - 0.3f;
  for (size_t i = 0; i < host_k.size(); ++i) {
    host_k[i] = static_cast<float>(i % 5) * 0.15f;
    host_v[i] = static_cast<float>(i % 9) * 0.05f - 0.1f;
  }
  // Write K/V into contiguous slots 0..T-1 of the cache first.
  std::vector<float> host_kc(cache_elems, 0.0f), host_vc(cache_elems, 0.0f);
  for (int64_t t = 0; t < T; ++t) {
    std::memcpy(host_kc.data() + t * Page, host_k.data() + t * Page,
                static_cast<size_t>(Page) * sizeof(float));
    std::memcpy(host_vc.data() + t * Page, host_v.data() + t * Page,
                static_cast<size_t>(Page) * sizeof(float));
  }
  std::vector<int32_t> block_table{0, 1};  // [num_reqs=1, max_blocks=2]
  std::vector<int32_t> seq_lens{static_cast<int32_t>(T)};
  std::vector<int32_t> qsl{0, static_cast<int32_t>(T)};
  std::vector<float> host_out(static_cast<size_t>(T * Hq * D), 0.0f);

  void* mem_q = backend.Alloc(host_q.size() * sizeof(float));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  void* mem_kc = backend.Alloc(host_kc.size() * sizeof(float));
  void* mem_vc = backend.Alloc(host_vc.size() * sizeof(float));
  void* mem_bt = backend.Alloc(block_table.size() * sizeof(int32_t));
  void* mem_sl = backend.Alloc(seq_lens.size() * sizeof(int32_t));
  void* mem_qsl = backend.Alloc(qsl.size() * sizeof(int32_t));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_q, host_q.data(), host_q.size() * sizeof(float));
  backend.Copy(q, mem_kc, host_kc.data(), host_kc.size() * sizeof(float));
  backend.Copy(q, mem_vc, host_vc.data(), host_vc.size() * sizeof(float));
  backend.Copy(q, mem_bt, block_table.data(), block_table.size() * sizeof(int32_t));
  backend.Copy(q, mem_sl, seq_lens.data(), seq_lens.size() * sizeof(int32_t));
  backend.Copy(q, mem_qsl, qsl.data(), qsl.size() * sizeof(int32_t));

  Tensor tq =
      Tensor::Contiguous(mem_q, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {T, Hq, D});
  Tensor tout = Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                   {T, Hq, D});
  Tensor tkc = Tensor::Contiguous(mem_kc, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {NBlocks, Bsz, Hkv, D});
  Tensor tvc = Tensor::Contiguous(mem_vc, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {NBlocks, Bsz, Hkv, D});
  Tensor tbt = Tensor::Contiguous(mem_bt, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0},
                                  {1, 2});
  Tensor tsl =
      Tensor::Contiguous(mem_sl, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0}, {1});
  Tensor tqsl =
      Tensor::Contiguous(mem_qsl, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0}, {2});

  vt::PagedAttentionArgs args;
  args.scale = 1.0f / std::sqrt(static_cast<float>(D));
  args.causal = true;
  auto pa = reinterpret_cast<vt::PagedAttentionFn>(
      vt::GetOp(vt::OpId::kPagedAttention, DeviceType::kTENSTORRENT));
  pa(q, tout, tq, tkc, tvc, tbt, tsl, tqsl, args);

  backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
  backend.Free(mem_q);
  backend.Free(mem_out);
  backend.Free(mem_kc);
  backend.Free(mem_vc);
  backend.Free(mem_bt);
  backend.Free(mem_sl);
  backend.Free(mem_qsl);

  // Host oracle: same two-pass max-subtracted softmax as cpu_paged_attn.cpp.
  std::vector<float> ref(static_cast<size_t>(T * Hq * D), 0.0f);
  const int64_t qpk = Hq / Hkv;
  for (int64_t t = 0; t < T; ++t) {
    const int64_t p = t;  // prefill, context=0
    for (int64_t h = 0; h < Hq; ++h) {
      const int64_t g = h / qpk;
      const int64_t qoff = (t * Hq + h) * D;
      float m = -std::numeric_limits<float>::infinity();
      std::vector<float> scores(static_cast<size_t>(p + 1));
      for (int64_t j = 0; j <= p; ++j) {
        float dot = 0.0f;
        for (int64_t e = 0; e < D; ++e)
          dot += host_q[static_cast<size_t>(qoff + e)] *
                 host_kc[static_cast<size_t>(j * Page + g * D + e)];
        scores[static_cast<size_t>(j)] = dot * args.scale;
        m = std::max(m, scores[static_cast<size_t>(j)]);
      }
      float denom = 0.0f;
      for (int64_t j = 0; j <= p; ++j) {
        scores[static_cast<size_t>(j)] = std::exp(scores[static_cast<size_t>(j)] - m);
        denom += scores[static_cast<size_t>(j)];
      }
      const float inv = 1.0f / denom;
      for (int64_t e = 0; e < D; ++e) {
        float acc = 0.0f;
        for (int64_t j = 0; j <= p; ++j)
          acc += scores[static_cast<size_t>(j)] * inv *
                 host_vc[static_cast<size_t>(j * Page + g * D + e)];
        ref[static_cast<size_t>(qoff + e)] = acc;
      }
    }
  }

  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < ref.size(); ++i)
    max_abs_diff = std::max(max_abs_diff, std::fabs(host_out[i] - ref[i]));
  // Host f32 path — should be essentially bit-exact; allow tiny float noise.
  CHECK(max_abs_diff < 1e-5f);
}
