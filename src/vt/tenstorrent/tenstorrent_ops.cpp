// Tenstorrent op providers — the ttnn adapter layer (BACKEND-TENSTORRENT,
// .agents/specs/tenstorrent-backend.md). vllm.cpp original; no upstream
// mirror (vLLM has no Tenstorrent platform). Op table covers OPT-125m's 9
// ops: ttnn for compute (matmul/add/relu/embedding/layernorm), host-staged
// pure data-movement / attention for the remainder (see HOST-STAGED OPS
// note below). Mirrors how Metal landed ops one seam at a time.
//
// SCOPE: F32 for the W0 path unless noted. kAdd allows rank-1 bias
// broadcast; kLayerNorm optional rank-1 weight/bias; kEmbedding i32/i64
// ids. Every other shape/dtype is a VT_CHECK failure — no CPU reference
// tier (UnifiedMemory()==false).
//
// HOST-STAGED OPS (kQkvSplit, kReshapeAndCache, kPagedAttention): this
// backend's Alloc is host memory (tenstorrent_backend.cpp). QkvSplit and
// ReshapeAndCache are pure contiguous / stride-aware copies — a device
// round-trip would only burn PCIe for bit-identical results. PagedAttention
// uses the CPU-oracle f32 softmax over the host-resident paged cache; mapping
// vLLM's block-table contract onto ttnn::sdpa_decode is deferred to the
// device-resident-tensor redesign the spec already names.
#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/tenstorrent/tenstorrent_device.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/embedding/embedding.hpp>
#include <ttnn/operations/normalization/layernorm/layernorm.hpp>

#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp>

namespace vt::tenstorrent {
namespace {

// Shared host round-trip helpers — see the kMatmul comment below for why the
// upload/compute/readback shape is correct (hardware evidence in
// .agents/specs/tenstorrent-backend.md's "Resolved: hands-on spike result").
tt::tt_metal::TensorSpec SpecOf(tt::tt_metal::Shape shape, ttnn::DataType dtype,
                                ttnn::Layout layout) {
  return tt::tt_metal::TensorSpec(
      std::move(shape),
      tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(layout),
                                 tt::tt_metal::MemoryConfig{}));
}

tt::tt_metal::TensorSpec TileSpecOf(uint32_t rows, uint32_t cols) {
  return SpecOf(tt::tt_metal::Shape({rows, cols}), ttnn::DataType::BFLOAT16,
                ttnn::Layout::TILE);
}

ttnn::Tensor UploadRows(const float* data, uint32_t rows, uint32_t cols, MeshDevice& device) {
  std::vector<float> host(data, data + static_cast<size_t>(rows) * cols);
  return ttnn::Tensor::from_vector<float>(host, TileSpecOf(rows, cols), &device);
}

void Download(ttnn::Tensor& dev, Tensor& out) {
  std::vector<float> result = dev.to_vector<float>();
  VT_CHECK(static_cast<int64_t>(result.size()) == out.Numel(),
           "tenstorrent: unexpected result size");
  std::copy(result.begin(), result.end(), out.Ptr<float>());
}

// Host round-trip per call — see this file's SCOPE note for why, and
// .agents/specs/tenstorrent-backend.md's "Resolved: hands-on spike result"
// for the hardware evidence this exact from_vector / matmul / to_vector
// sequence produces a correct answer (max_abs_diff 0.03375 vs max_ref_mag
// 4.14 on a real Blackhole, bf16 tolerance).
void MatmulKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  VT_CHECK(a.rank == 2 && b.rank == 2 && out.rank == 2,
           "tenstorrent kMatmul: only rank-2 tensors are supported in W0");
  VT_CHECK(a.dtype == DType::kF32 && b.dtype == DType::kF32 && out.dtype == DType::kF32,
           "tenstorrent kMatmul: only F32 is supported in W0");
  const uint32_t M = static_cast<uint32_t>(a.shape[0]);
  const uint32_t K = static_cast<uint32_t>(a.shape[1]);
  const uint32_t N = static_cast<uint32_t>(b.shape[1]);
  VT_CHECK(b.shape[0] == K, "tenstorrent kMatmul: a/b inner dimension mismatch");
  VT_CHECK(out.shape[0] == M && out.shape[1] == N, "tenstorrent kMatmul: out shape mismatch");
  VT_CHECK(a.IsContiguous() && b.IsContiguous() && out.IsContiguous(),
           "tenstorrent kMatmul: strided (non-contiguous) tensors are not supported in W0");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_a = UploadRows(a.Ptr<float>(), M, K, device);
  ttnn::Tensor dev_b = UploadRows(b.Ptr<float>(), K, N, device);
  ttnn::Tensor dev_c = ttnn::operations::matmul::matmul(dev_a, dev_b);
  Download(dev_c, out);
}

// kMatmulBT: `b` is a [N,K] row-major torch nn.Linear weight; computes
// `a @ b^T` (cpu_ops.cpp's MatmulBTKernel contract). ttnn's matmul() already
// exposes a transpose_b flag, so this is the same sequence as kMatmul with
// that flag flipped — no separate upload shape needed since `b` is uploaded
// in its native [N,K] layout and ttnn transposes on device.
void MatmulBTKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  VT_CHECK(a.rank == 2 && b.rank == 2 && out.rank == 2,
           "tenstorrent kMatmulBT: only rank-2 tensors are supported in W0");
  VT_CHECK(a.dtype == DType::kF32 && b.dtype == DType::kF32 && out.dtype == DType::kF32,
           "tenstorrent kMatmulBT: only F32 is supported in W0");
  const uint32_t M = static_cast<uint32_t>(a.shape[0]);
  const uint32_t K = static_cast<uint32_t>(a.shape[1]);
  const uint32_t N = static_cast<uint32_t>(b.shape[0]);
  VT_CHECK(b.shape[1] == K, "tenstorrent kMatmulBT: a/b inner dimension mismatch");
  VT_CHECK(out.shape[0] == M && out.shape[1] == N, "tenstorrent kMatmulBT: out shape mismatch");
  VT_CHECK(a.IsContiguous() && b.IsContiguous() && out.IsContiguous(),
           "tenstorrent kMatmulBT: strided (non-contiguous) tensors are not supported in W0");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_a = UploadRows(a.Ptr<float>(), M, K, device);
  ttnn::Tensor dev_b = UploadRows(b.Ptr<float>(), N, K, device);
  ttnn::Tensor dev_c =
      ttnn::operations::matmul::matmul(dev_a, dev_b, /*transpose_a=*/false, /*transpose_b=*/true);
  Download(dev_c, out);
}

// kAdd: elementwise add, plus the rank-1 `b` row-broadcast form used for
// nn.Linear bias (cpu_layernorm.cpp's AddKernel contract). ttnn::add needs
// same-rank operands, so the broadcast case uploads `b` replicated into a
// [rows, d] tile rather than relying on ttnn's own broadcast rules — keeps
// this kernel's behavior pinned to the CPU reference rather than to
// whatever ttnn::add happens to support today.
void AddKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  VT_CHECK(a.rank == 2 && out.rank == 2, "tenstorrent kAdd: `a`/`out` must be rank-2 in W0");
  VT_CHECK(b.rank == 2 || b.rank == 1, "tenstorrent kAdd: `b` must be rank-1 or rank-2 in W0");
  VT_CHECK(a.dtype == DType::kF32 && b.dtype == DType::kF32 && out.dtype == DType::kF32,
           "tenstorrent kAdd: only F32 is supported in W0");
  VT_CHECK(a.IsContiguous() && b.IsContiguous() && out.IsContiguous(),
           "tenstorrent kAdd: strided (non-contiguous) tensors are not supported in W0");
  const uint32_t rows = static_cast<uint32_t>(a.shape[0]);
  const uint32_t d = static_cast<uint32_t>(a.shape[1]);
  VT_CHECK(out.shape[0] == rows && out.shape[1] == d, "tenstorrent kAdd: out shape mismatch");
  const bool bcast = b.rank == 1;
  VT_CHECK(bcast ? b.shape[0] == d : (b.shape[0] == rows && b.shape[1] == d),
           "tenstorrent kAdd: `b` shape mismatch");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_a = UploadRows(a.Ptr<float>(), rows, d, device);
  ttnn::Tensor dev_b;
  if (bcast) {
    std::vector<float> replicated(static_cast<size_t>(rows) * d);
    const float* row = b.Ptr<float>();
    for (uint32_t r = 0; r < rows; ++r) std::copy(row, row + d, replicated.begin() + r * d);
    dev_b = ttnn::Tensor::from_vector<float>(replicated, TileSpecOf(rows, d), &device);
  } else {
    dev_b = UploadRows(b.Ptr<float>(), rows, d, device);
  }
  ttnn::Tensor dev_c = ttnn::add(dev_a, dev_b);
  Download(dev_c, out);
}

// kRelu: elementwise max(0, x) (cpu_layernorm.cpp's ReluKernel contract).
void ReluKernel(Queue&, Tensor& out, const Tensor& x) {
  VT_CHECK(x.rank == 2 && out.rank == 2, "tenstorrent kRelu: only rank-2 tensors are supported in W0");
  VT_CHECK(x.dtype == DType::kF32 && out.dtype == DType::kF32,
           "tenstorrent kRelu: only F32 is supported in W0");
  VT_CHECK(x.shape[0] == out.shape[0] && x.shape[1] == out.shape[1],
           "tenstorrent kRelu: out shape mismatch");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(),
           "tenstorrent kRelu: strided (non-contiguous) tensors are not supported in W0");
  const uint32_t rows = static_cast<uint32_t>(x.shape[0]);
  const uint32_t d = static_cast<uint32_t>(x.shape[1]);

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_x = UploadRows(x.Ptr<float>(), rows, d, device);
  ttnn::Tensor dev_y = ttnn::relu(dev_x);
  Download(dev_y, out);
}

// kEmbedding: row gather `out[i,:] = table[ids[i],:]` (cpu_ops.cpp
// EmbeddingKernel contract). Two layout departures from the TILE/BFLOAT16
// linear ops, forced by ttnn::embedding's validate path:
//   1. ids upload as ROW_MAJOR UINT32 (ttnn rejects i32/i64; vt still accepts
//      kI32/kI64 at the seam and converts host-side, matching Metal/Vulkan).
//   2. table upload as ROW_MAJOR BFLOAT16 (ttnn requires ROW_MAJOR weights;
//      TILE is converted inside ttnn::embedding, but starting RM is cheaper
//      and matches the unit tests in tt-metal).
// Parameter order at the ttnn call is (ids, table) — reversed from
// vt::EmbeddingFn's (table, ids). Output is requested ROW_MAJOR so
// to_vector is dense without tile padding for arbitrary (t, h).
void EmbeddingKernel(Queue&, Tensor& out, const Tensor& table, const Tensor& ids) {
  VT_CHECK(table.rank == 2 && ids.rank == 1 && out.rank == 2,
           "tenstorrent kEmbedding: table rank-2, ids rank-1, out rank-2");
  VT_CHECK(table.dtype == DType::kF32 && out.dtype == DType::kF32,
           "tenstorrent kEmbedding: only F32 table/out are supported in this step");
  VT_CHECK(ids.dtype == DType::kI32 || ids.dtype == DType::kI64,
           "tenstorrent kEmbedding: ids must be i32 or i64");
  VT_CHECK(table.IsContiguous() && ids.IsContiguous() && out.IsContiguous(),
           "tenstorrent kEmbedding: strided (non-contiguous) tensors are not supported");
  const uint32_t vocab = static_cast<uint32_t>(table.shape[0]);
  const uint32_t h = static_cast<uint32_t>(table.shape[1]);
  const uint32_t t = static_cast<uint32_t>(ids.shape[0]);
  VT_CHECK(out.shape[0] == t && out.shape[1] == h, "tenstorrent kEmbedding: out shape mismatch");

  std::vector<uint32_t> host_ids(t);
  if (ids.dtype == DType::kI32) {
    const int32_t* p = ids.Ptr<int32_t>();
    for (uint32_t i = 0; i < t; ++i) {
      VT_CHECK(p[i] >= 0 && static_cast<uint32_t>(p[i]) < vocab,
               "tenstorrent kEmbedding: id out of range");
      host_ids[i] = static_cast<uint32_t>(p[i]);
    }
  } else {
    const int64_t* p = ids.Ptr<int64_t>();
    for (uint32_t i = 0; i < t; ++i) {
      VT_CHECK(p[i] >= 0 && static_cast<uint64_t>(p[i]) < vocab,
               "tenstorrent kEmbedding: id out of range");
      host_ids[i] = static_cast<uint32_t>(p[i]);
    }
  }

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_ids = ttnn::Tensor::from_vector<uint32_t>(
      host_ids, SpecOf(tt::tt_metal::Shape({t}), ttnn::DataType::UINT32, ttnn::Layout::ROW_MAJOR),
      &device);
  std::vector<float> host_table(table.Ptr<float>(),
                                table.Ptr<float>() + static_cast<size_t>(vocab) * h);
  ttnn::Tensor dev_table = ttnn::Tensor::from_vector<float>(
      host_table,
      SpecOf(tt::tt_metal::Shape({vocab, h}), ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR),
      &device);
  // (ids, table) — reversed from vt::EmbeddingFn. Explicit ROW_MAJOR output
  // keeps download dense for non-tile-aligned (t, h).
  ttnn::Tensor dev_out = ttnn::embedding(dev_ids, dev_table, /*pad_token=*/std::nullopt,
                                         /*layout=*/ttnn::Layout::ROW_MAJOR);
  Download(dev_out, out);
}

// Upload a rank-1 F32 affine vector as TILE BFLOAT16 with logical shape
// [1, d]. ttnn::layer_norm's TILE-gamma path requires padded height ==
// tile_height (32); from_vector with TILE layout pads a [1,d] tensor to
// that. ROW_MAJOR gamma only works cleanly when d == tile_width in the
// ttnn unit tests, so TILE is the general path.
ttnn::Tensor UploadAffine1D(const float* data, uint32_t d, MeshDevice& device) {
  std::vector<float> host(data, data + d);
  return ttnn::Tensor::from_vector<float>(
      host, SpecOf(tt::tt_metal::Shape({1, d}), ttnn::DataType::BFLOAT16, ttnn::Layout::TILE),
      &device);
}

// kLayerNorm: per-row mean/var over the last dim (cpu_layernorm.cpp
// LayerNormKernel / ATen native_layer_norm). Biased (1/N) variance; optional
// rank-1 weight/bias (elementwise_affine). Uses ttnn::layer_norm with the
// same TILE/BFLOAT16 upload path as the linear ops; eps comes from
// LayerNormArgs (OPT default 1e-5, not ttnn's 1e-12 default).
void LayerNormKernel(Queue&, Tensor& out, const Tensor& x, const Tensor* weight,
                     const Tensor* bias, const LayerNormArgs& args) {
  VT_CHECK(x.rank == 2 && out.rank == 2,
           "tenstorrent kLayerNorm: only rank-2 tensors are supported in this step");
  VT_CHECK(x.dtype == DType::kF32 && out.dtype == DType::kF32,
           "tenstorrent kLayerNorm: only F32 is supported in this step");
  VT_CHECK(x.shape[0] == out.shape[0] && x.shape[1] == out.shape[1],
           "tenstorrent kLayerNorm: out shape mismatch");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(),
           "tenstorrent kLayerNorm: strided (non-contiguous) tensors are not supported");
  VT_CHECK(args.eps >= 0.0f, "tenstorrent kLayerNorm: eps must be non-negative");
  const uint32_t rows = static_cast<uint32_t>(x.shape[0]);
  const uint32_t d = static_cast<uint32_t>(x.shape[1]);
  for (const Tensor* p : {weight, bias}) {
    if (p == nullptr) continue;
    VT_CHECK(p->rank == 1 && p->shape[0] == d,
             "tenstorrent kLayerNorm: weight/bias must be rank-1 [D]");
    VT_CHECK(p->dtype == DType::kF32, "tenstorrent kLayerNorm: weight/bias must be F32");
    VT_CHECK(p->IsContiguous(), "tenstorrent kLayerNorm: weight/bias must be contiguous");
  }

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_x = UploadRows(x.Ptr<float>(), rows, d, device);
  std::optional<ttnn::Tensor> dev_w;
  std::optional<ttnn::Tensor> dev_b;
  if (weight != nullptr) dev_w = UploadAffine1D(weight->Ptr<float>(), d, device);
  if (bias != nullptr) dev_b = UploadAffine1D(bias->Ptr<float>(), d, device);
  ttnn::Tensor dev_y = ttnn::layer_norm(dev_x, args.eps, dev_w, dev_b);
  Download(dev_y, out);
}

// kQkvSplit: pure contiguous column split of merged [T, q+k+v] into q/k/v
// (cpu_ops.cpp QkvSplitKernel). Host-staged: bit-exact memcpy, no device.
void QkvSplitKernel(Queue&, Tensor& q_out, Tensor& k_out, Tensor& v_out, const Tensor& qkv) {
  VT_CHECK(qkv.rank == 2 && qkv.dtype == DType::kF32,
           "tenstorrent kQkvSplit: rank-2 F32 qkv required");
  VT_CHECK(q_out.dtype == DType::kF32 && k_out.dtype == DType::kF32 && v_out.dtype == DType::kF32,
           "tenstorrent kQkvSplit: only F32 out is supported in this step");
  VT_CHECK(q_out.IsContiguous() && k_out.IsContiguous() && v_out.IsContiguous() &&
               qkv.IsContiguous(),
           "tenstorrent kQkvSplit: contiguous required");
  const int64_t t = qkv.shape[0];
  const int64_t q_dim = q_out.Numel() / t;
  const int64_t k_dim = k_out.Numel() / t;
  const int64_t v_dim = v_out.Numel() / t;
  const int64_t total = q_dim + k_dim + v_dim;
  VT_CHECK(qkv.shape[1] == total, "tenstorrent kQkvSplit: inner dim mismatch");
  const float* src = qkv.Ptr<float>();
  float* qdst = q_out.Ptr<float>();
  float* kdst = k_out.Ptr<float>();
  float* vdst = v_out.Ptr<float>();
  for (int64_t i = 0; i < t; ++i) {
    const float* row = src + i * total;
    std::memcpy(qdst + i * q_dim, row, static_cast<size_t>(q_dim) * sizeof(float));
    std::memcpy(kdst + i * k_dim, row + q_dim, static_cast<size_t>(k_dim) * sizeof(float));
    std::memcpy(vdst + i * v_dim, row + q_dim + k_dim, static_cast<size_t>(v_dim) * sizeof(float));
  }
}

struct Registrar {
  Registrar() {
    if (!DeviceAvailable()) return;
    RegisterOp(OpId::kMatmul, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulKernel)));
    RegisterOp(OpId::kMatmulBT, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulBTKernel)));
    RegisterOp(OpId::kAdd, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<AddFn>(&AddKernel)));
    RegisterOp(OpId::kRelu, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<ReluFn>(&ReluKernel)));
    RegisterOp(OpId::kEmbedding, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<EmbeddingFn>(&EmbeddingKernel)));
    RegisterOp(OpId::kLayerNorm, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<LayerNormFn>(&LayerNormKernel)));
    RegisterOp(OpId::kQkvSplit, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<QkvSplitFn>(&QkvSplitKernel)));
  }
} registrar;

}  // namespace
}  // namespace vt::tenstorrent
