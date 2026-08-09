// Tenstorrent op providers — the ttnn adapter layer (BACKEND-TENSTORRENT,
// .agents/specs/tenstorrent-backend.md). vllm.cpp original; no upstream
// mirror (vLLM has no Tenstorrent platform). Op table: OPT-125m's 9 ops plus
// the Qwen3-dense deltas (kRmsNorm first; kSiluAndMul / RoPE / Cast next),
// matching Metal's OPT→Qwen3 sequencing. ttnn for compute where available;
// host-staged pure data-movement / attention for the remainder (see
// HOST-STAGED OPS note below).
//
// SCOPE: F32 for the W0 path unless noted. kAdd allows rank-1 bias
// broadcast; kLayerNorm optional rank-1 weight/bias; kRmsNorm weight +
// optional residual stream; kEmbedding i32/i64 ids. Every other shape/dtype
// is a VT_CHECK failure — no CPU reference tier (UnifiedMemory()==false).
//
// HOST-STAGED OPS (kQkvSplit, kReshapeAndCache, kPagedAttention): this
// backend's Alloc is host memory (tenstorrent_backend.cpp). QkvSplit and
// ReshapeAndCache are pure contiguous / stride-aware copies — a device
// round-trip would only burn PCIe for bit-identical results. PagedAttention
// uses the CPU-oracle f32 softmax over the host-resident paged cache; mapping
// vLLM's block-table contract onto ttnn::sdpa_decode is deferred to the
// device-resident-tensor redesign the spec already names.
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tenstorrent/tenstorrent_device.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/embedding/embedding.hpp>
#include <ttnn/operations/normalization/layernorm/layernorm.hpp>
#include <ttnn/operations/normalization/rmsnorm/rmsnorm.hpp>
#include <ttnn/operations/data_movement/slice/slice.hpp>

#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp>

namespace vt::tenstorrent {
namespace {

// ---- Host/device residency -------------------------------------------------
// vt::Tensor.data is always a host pointer from Backend::Alloc. A shadow map
// (Metal AllocMap shape) holds an optional device-resident ttnn::Tensor for
// that host base so multi-op chains need not download after every matmul.

struct BufferSlot {
  void* host = nullptr;
  size_t bytes = 0;
  std::optional<ttnn::Tensor> device;
  uint32_t dev_rows = 0;
  uint32_t dev_cols = 0;
  bool host_current = true;    // host bytes match the latest value
  bool device_current = false; // device tensor matches the latest value
};

std::mutex& SlotMutex() {
  static std::mutex m;
  return m;
}
std::map<uintptr_t, BufferSlot>& Slots() {
  static std::map<uintptr_t, BufferSlot> m;
  return m;
}

// Base slot for `p` or any interior pointer into a registered allocation.
BufferSlot* FindSlot(void* p) {
  if (p == nullptr) return nullptr;
  auto& m = Slots();
  const uintptr_t key = reinterpret_cast<uintptr_t>(p);
  auto it = m.upper_bound(key);
  if (it == m.begin()) return nullptr;
  --it;
  BufferSlot& s = it->second;
  const uintptr_t base = reinterpret_cast<uintptr_t>(s.host);
  if (key < base || key >= base + s.bytes) return nullptr;
  return &s;
}

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

// OPT-125m (and the rest of the dense path) runs BF16 weights/activations with
// F32 logits. Host-stage every float dtype to f32 for from_vector, then round
// back on download — ttnn already computes in BFLOAT16 tiles.
float LoadElemF32(const Tensor& t, int64_t i) {
  switch (t.dtype) {
    case DType::kF32: return t.Ptr<float>()[i];
    case DType::kBF16: return BF16ToF32(t.Ptr<uint16_t>()[i]);
    case DType::kF16: return F16ToF32(t.Ptr<uint16_t>()[i]);
    default: VT_CHECK(false, "tenstorrent: unsupported float dtype"); return 0.0f;
  }
}

void StoreElemF32(Tensor& t, int64_t i, float v) {
  switch (t.dtype) {
    case DType::kF32: t.Ptr<float>()[i] = v; break;
    case DType::kBF16: t.Ptr<uint16_t>()[i] = F32ToBF16(v); break;
    default: VT_CHECK(false, "tenstorrent: unsupported out dtype (f32/bf16)");
  }
}

bool IsFloatDType(DType d) {
  return d == DType::kF32 || d == DType::kBF16 || d == DType::kF16;
}

void DownloadToHost(ttnn::Tensor& dev, Tensor& out) {
  std::vector<float> result = dev.to_vector<float>();
  VT_CHECK(static_cast<int64_t>(result.size()) == out.Numel(),
           "tenstorrent: unexpected result size");
  for (int64_t i = 0; i < out.Numel(); ++i)
    StoreElemF32(out, i, result[static_cast<size_t>(i)]);
}

// Pull device → host if the host view is stale (required before host kernels).
void EnsureHost(Tensor& t) {
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(t.data);
  if (s == nullptr || s->host_current) return;
  VT_CHECK(s->device_current && s->device.has_value(),
           "tenstorrent: EnsureHost with no current device or host copy");
  DownloadToHost(*s->device, t);
  s->host_current = true;
}

void EnsureHost(const Tensor& t) {
  // const_cast: host bytes are filled in place; logical tensor is unchanged.
  EnsureHost(const_cast<Tensor&>(t));
}

std::vector<float> ToHostF32(const Tensor& t) {
  EnsureHost(t);
  const int64_t n = t.Numel();
  std::vector<float> host(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) host[static_cast<size_t>(i)] = LoadElemF32(t, i);
  return host;
}

ttnn::Tensor UploadRows(const float* data, uint32_t rows, uint32_t cols, MeshDevice& device) {
  std::vector<float> host(data, data + static_cast<size_t>(rows) * cols);
  return ttnn::Tensor::from_vector<float>(host, TileSpecOf(rows, cols), &device);
}

// Return a TILE BFLOAT16 device tensor for rank-2 `t`, uploading only when the
// device shadow is missing or stale.
ttnn::Tensor EnsureDevice2D(const Tensor& t, MeshDevice& device) {
  VT_CHECK(t.rank == 2 && t.IsContiguous(),
           "tenstorrent: EnsureDevice2D expects contiguous rank-2");
  const uint32_t rows = static_cast<uint32_t>(t.shape[0]);
  const uint32_t cols = static_cast<uint32_t>(t.shape[1]);
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(t.data);
    if (s != nullptr && s->device_current && s->device.has_value() &&
        s->dev_rows == rows && s->dev_cols == cols) {
      return *s->device;
    }
  }
  // Need host truth to upload (may download first if only device was current
  // under a different shape — rare).
  EnsureHost(t);
  const auto host = ToHostF32(t);
  ttnn::Tensor dev = UploadRows(host.data(), rows, cols, device);
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(t.data);
  if (s != nullptr) {
    s->device = dev;
    s->dev_rows = rows;
    s->dev_cols = cols;
    s->device_current = true;
    s->host_current = true;
  }
  return dev;
}

// Publish a device result as the current value of `out` WITHOUT downloading
// to host (the residency win). Host is marked stale until EnsureHost.
void CommitDevice2D(Tensor& out, ttnn::Tensor dev) {
  VT_CHECK(out.rank == 2 && out.IsContiguous(),
           "tenstorrent: CommitDevice2D expects contiguous rank-2 out");
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(out.data);
  if (s == nullptr) {
    // Untracked buffer (e.g. stack/test scratch): fall back to host write.
    DownloadToHost(dev, out);
    return;
  }
  s->device = std::move(dev);
  s->dev_rows = static_cast<uint32_t>(out.shape[0]);
  s->dev_cols = static_cast<uint32_t>(out.shape[1]);
  s->device_current = true;
  s->host_current = false;
}

// Host wrote `out` in place — drop any device shadow.
void CommitHost(Tensor& out) {
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(out.data);
  if (s == nullptr) return;
  s->host_current = true;
  s->device_current = false;
  s->device = std::nullopt;
}

// Legacy name used by a few call sites that still force a host materialization.
void Download(ttnn::Tensor& dev, Tensor& out) {
  DownloadToHost(dev, out);
  CommitHost(out);
  // Also keep device copy so a subsequent EnsureDevice can reuse it without
  // re-upload if host was not modified.
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(out.data);
  if (s != nullptr) {
    s->device = dev;
    s->dev_rows = out.rank >= 1 ? static_cast<uint32_t>(out.shape[0]) : 0;
    s->dev_cols = out.rank >= 2 ? static_cast<uint32_t>(out.shape[1]) : 0;
    s->device_current = (out.rank == 2);
    s->host_current = true;
  }
}

// Device compute: keep result on device (CommitDevice2D). Host round-trip only
// when the consumer is a host-staged op (EnsureHost) or an untracked buffer.
void MatmulKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  VT_CHECK(a.rank == 2 && b.rank == 2 && out.rank == 2,
           "tenstorrent kMatmul: only rank-2 tensors are supported in W0");
  VT_CHECK(IsFloatDType(a.dtype) && IsFloatDType(b.dtype) &&
               (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kMatmul: float in, f32/bf16 out");
  const uint32_t M = static_cast<uint32_t>(a.shape[0]);
  const uint32_t K = static_cast<uint32_t>(a.shape[1]);
  const uint32_t N = static_cast<uint32_t>(b.shape[1]);
  VT_CHECK(b.shape[0] == K, "tenstorrent kMatmul: a/b inner dimension mismatch");
  VT_CHECK(out.shape[0] == M && out.shape[1] == N, "tenstorrent kMatmul: out shape mismatch");
  VT_CHECK(a.IsContiguous() && b.IsContiguous() && out.IsContiguous(),
           "tenstorrent kMatmul: strided (non-contiguous) tensors are not supported in W0");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_a = EnsureDevice2D(a, device);
  ttnn::Tensor dev_b = EnsureDevice2D(b, device);
  ttnn::Tensor dev_c = ttnn::operations::matmul::matmul(dev_a, dev_b);
  CommitDevice2D(out, std::move(dev_c));
}

// kMatmulBT: `b` is a [N,K] row-major torch nn.Linear weight; computes
// `a @ b^T` (cpu_ops.cpp's MatmulBTKernel contract). ttnn's matmul() already
// exposes a transpose_b flag, so this is the same sequence as kMatmul with
// that flag flipped — no separate upload shape needed since `b` is uploaded
// in its native [N,K] layout and ttnn transposes on device.
void MatmulBTKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  VT_CHECK(a.rank == 2 && b.rank == 2 && out.rank == 2,
           "tenstorrent kMatmulBT: only rank-2 tensors are supported in W0");
  VT_CHECK(IsFloatDType(a.dtype) && IsFloatDType(b.dtype) &&
               (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kMatmulBT: float in, f32/bf16 out");
  const uint32_t M = static_cast<uint32_t>(a.shape[0]);
  const uint32_t K = static_cast<uint32_t>(a.shape[1]);
  const uint32_t N = static_cast<uint32_t>(b.shape[0]);
  VT_CHECK(b.shape[1] == K, "tenstorrent kMatmulBT: a/b inner dimension mismatch");
  VT_CHECK(out.shape[0] == M && out.shape[1] == N, "tenstorrent kMatmulBT: out shape mismatch");
  VT_CHECK(a.IsContiguous() && b.IsContiguous() && out.IsContiguous(),
           "tenstorrent kMatmulBT: strided (non-contiguous) tensors are not supported in W0");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_a = EnsureDevice2D(a, device);
  ttnn::Tensor dev_b = EnsureDevice2D(b, device);
  ttnn::Tensor dev_c =
      ttnn::operations::matmul::matmul(dev_a, dev_b, /*transpose_a=*/false, /*transpose_b=*/true);
  CommitDevice2D(out, std::move(dev_c));
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
  VT_CHECK(IsFloatDType(a.dtype) && IsFloatDType(b.dtype) &&
               (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kAdd: float in, f32/bf16 out");
  VT_CHECK(a.IsContiguous() && b.IsContiguous() && out.IsContiguous(),
           "tenstorrent kAdd: strided (non-contiguous) tensors are not supported in W0");
  const uint32_t rows = static_cast<uint32_t>(a.shape[0]);
  const uint32_t d = static_cast<uint32_t>(a.shape[1]);
  VT_CHECK(out.shape[0] == rows && out.shape[1] == d, "tenstorrent kAdd: out shape mismatch");
  const bool bcast = b.rank == 1;
  VT_CHECK(bcast ? b.shape[0] == d : (b.shape[0] == rows && b.shape[1] == d),
           "tenstorrent kAdd: `b` shape mismatch");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_a = EnsureDevice2D(a, device);
  ttnn::Tensor dev_b;
  if (bcast) {
    EnsureHost(b);
    std::vector<float> replicated(static_cast<size_t>(rows) * d);
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t c = 0; c < d; ++c)
        replicated[static_cast<size_t>(r) * d + c] = LoadElemF32(b, c);
    dev_b = ttnn::Tensor::from_vector<float>(replicated, TileSpecOf(rows, d), &device);
  } else {
    dev_b = EnsureDevice2D(b, device);
  }
  ttnn::Tensor dev_c = ttnn::add(dev_a, dev_b);
  CommitDevice2D(out, std::move(dev_c));
}

// kRelu: elementwise max(0, x) (cpu_layernorm.cpp's ReluKernel contract).
void ReluKernel(Queue&, Tensor& out, const Tensor& x) {
  VT_CHECK(x.rank == 2 && out.rank == 2, "tenstorrent kRelu: only rank-2 tensors are supported in W0");
  VT_CHECK(IsFloatDType(x.dtype) && (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kRelu: float in, f32/bf16 out");
  VT_CHECK(x.shape[0] == out.shape[0] && x.shape[1] == out.shape[1],
           "tenstorrent kRelu: out shape mismatch");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(),
           "tenstorrent kRelu: strided (non-contiguous) tensors are not supported in W0");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_x = EnsureDevice2D(x, device);
  ttnn::Tensor dev_y = ttnn::relu(dev_x);
  CommitDevice2D(out, std::move(dev_y));
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
  VT_CHECK(IsFloatDType(table.dtype) && (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kEmbedding: float table, f32/bf16 out");
  VT_CHECK(ids.dtype == DType::kI32 || ids.dtype == DType::kI64,
           "tenstorrent kEmbedding: ids must be i32 or i64");
  VT_CHECK(table.IsContiguous() && ids.IsContiguous() && out.IsContiguous(),
           "tenstorrent kEmbedding: strided (non-contiguous) tensors are not supported");
  const uint32_t vocab = static_cast<uint32_t>(table.shape[0]);
  const uint32_t h = static_cast<uint32_t>(table.shape[1]);
  const uint32_t t = static_cast<uint32_t>(ids.shape[0]);
  VT_CHECK(out.shape[0] == t && out.shape[1] == h, "tenstorrent kEmbedding: out shape mismatch");

  EnsureHost(ids);
  EnsureHost(table);
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
  std::vector<float> host_table = ToHostF32(table);
  ttnn::Tensor dev_table = ttnn::Tensor::from_vector<float>(
      host_table,
      SpecOf(tt::tt_metal::Shape({vocab, h}), ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR),
      &device);
  // (ids, table) — reversed from vt::EmbeddingFn. Explicit ROW_MAJOR output
  // keeps download dense for non-tile-aligned (t, h).
  ttnn::Tensor dev_out = ttnn::embedding(dev_ids, dev_table, /*pad_token=*/std::nullopt,
                                         /*layout=*/ttnn::Layout::ROW_MAJOR);
  // Embedding is ROW_MAJOR; materialize host then re-upload as TILE so the next
  // matmul hits the device cache. Once-per-forward cost; activations after
  // this stay device-resident via CommitDevice2D on matmul/norm.
  Download(dev_out, out);
}

// Upload a rank-1 F32 affine vector as TILE BFLOAT16 with logical shape
// [1, d]. ttnn::layer_norm's TILE-gamma path requires padded height ==
// tile_height (32); from_vector with TILE layout pads a [1,d] tensor to
// that. ROW_MAJOR gamma only works cleanly when d == tile_width in the
// ttnn unit tests, so TILE is the general path.
ttnn::Tensor UploadAffine1D(const Tensor& t, uint32_t d, MeshDevice& device) {
  std::vector<float> host(d);
  for (uint32_t i = 0; i < d; ++i) host[i] = LoadElemF32(t, i);
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
  VT_CHECK(IsFloatDType(x.dtype) && (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kLayerNorm: float in, f32/bf16 out");
  VT_CHECK(x.shape[0] == out.shape[0] && x.shape[1] == out.shape[1],
           "tenstorrent kLayerNorm: out shape mismatch");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(),
           "tenstorrent kLayerNorm: strided (non-contiguous) tensors are not supported");
  VT_CHECK(args.eps >= 0.0f, "tenstorrent kLayerNorm: eps must be non-negative");
  const uint32_t d = static_cast<uint32_t>(x.shape[1]);
  for (const Tensor* p : {weight, bias}) {
    if (p == nullptr) continue;
    VT_CHECK(p->rank == 1 && p->shape[0] == d,
             "tenstorrent kLayerNorm: weight/bias must be rank-1 [D]");
    VT_CHECK(IsFloatDType(p->dtype), "tenstorrent kLayerNorm: float weight/bias");
    VT_CHECK(p->IsContiguous(), "tenstorrent kLayerNorm: weight/bias must be contiguous");
  }

  MeshDevice& device = SharedMeshDevice();
  if (weight != nullptr) EnsureHost(*weight);
  if (bias != nullptr) EnsureHost(*bias);
  ttnn::Tensor dev_x = EnsureDevice2D(x, device);
  std::optional<ttnn::Tensor> dev_w;
  std::optional<ttnn::Tensor> dev_b;
  if (weight != nullptr) dev_w = UploadAffine1D(*weight, d, device);
  if (bias != nullptr) dev_b = UploadAffine1D(*bias, d, device);
  ttnn::Tensor dev_y = ttnn::layer_norm(dev_x, args.eps, dev_w, dev_b);
  CommitDevice2D(out, std::move(dev_y));
}

// kRmsNorm: per-row RMS over the last dim (cpu_ops.cpp RmsNormKernel). First
// Qwen3-dense (`Qwen3ForCausalLM`) op beyond OPT's LayerNorm set — Qwen3 uses
// RMSNorm for input/post-attn/final norms and per-head q/k norms. Weight is
// always present at the seam; optional residual is the residual stream
// (pre-norm sum written back, then normed), matching CPU residual round-trip
// for bf16 faithfulness. Gemma style (w+1) is host-only for now — Qwen3 does
// not set gemma=true.
//
// Device path: ttnn::rms_norm after residual merge (when any) and weight
// upload via the same TILE [1,D] affine helper as kLayerNorm.
void RmsNormKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& weight,
                   const RmsNormArgs& args, Tensor* residual) {
  VT_CHECK(x.rank == 2 && out.rank == 2,
           "tenstorrent kRmsNorm: only rank-2 tensors are supported in this step");
  VT_CHECK(IsFloatDType(x.dtype) && (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kRmsNorm: float in, f32/bf16 out");
  VT_CHECK(x.shape[0] == out.shape[0] && x.shape[1] == out.shape[1],
           "tenstorrent kRmsNorm: out shape mismatch");
  VT_CHECK(x.IsContiguous() && out.IsContiguous() && weight.IsContiguous(),
           "tenstorrent kRmsNorm: strided (non-contiguous) tensors are not supported");
  VT_CHECK(args.eps >= 0.0f, "tenstorrent kRmsNorm: eps must be non-negative");
  const uint32_t rows = static_cast<uint32_t>(x.shape[0]);
  const uint32_t d = static_cast<uint32_t>(x.shape[1]);
  VT_CHECK(weight.rank == 1 && weight.shape[0] == d && IsFloatDType(weight.dtype),
           "tenstorrent kRmsNorm: weight must be rank-1 float [D]");
  if (residual != nullptr) {
    VT_CHECK(residual->rank == 2 && residual->shape[0] == rows && residual->shape[1] == d,
             "tenstorrent kRmsNorm: residual shape must match x");
    VT_CHECK(IsFloatDType(residual->dtype) && residual->IsContiguous(),
             "tenstorrent kRmsNorm: residual must be contiguous float");
  }

  // Residual stream: x + residual -> residual (with dtype re-read), then norm
  // that value. Host-staged so bf16 round-trip matches cpu_ops RmsNormKernel.
  // Gemma (w+1) also stays on the host path with the same math as CPU.
  if (residual != nullptr || args.gemma) {
    EnsureHost(x);
    EnsureHost(weight);
    if (residual != nullptr) EnsureHost(*residual);
    for (int64_t r = 0; r < static_cast<int64_t>(rows); ++r) {
      float sumsq = 0.0f;
      for (int64_t j = 0; j < static_cast<int64_t>(d); ++j) {
        const int64_t idx = r * static_cast<int64_t>(d) + j;
        float v = LoadElemF32(x, idx);
        if (residual != nullptr) {
          v += LoadElemF32(*residual, idx);
          StoreElemF32(*residual, idx, v);
          v = LoadElemF32(*residual, idx);
        }
        sumsq += v * v;
      }
      const float inv =
          1.0f / std::sqrt(sumsq / static_cast<float>(d) + args.eps);
      for (int64_t j = 0; j < static_cast<int64_t>(d); ++j) {
        const int64_t idx = r * static_cast<int64_t>(d) + j;
        float v =
            residual != nullptr ? LoadElemF32(*residual, idx) : LoadElemF32(x, idx);
        float wj = LoadElemF32(weight, j);
        if (args.gemma) wj += 1.0f;
        StoreElemF32(out, idx, v * inv * wj);
      }
    }
    CommitHost(out);
    if (residual != nullptr) CommitHost(*residual);
    return;
  }

  MeshDevice& device = SharedMeshDevice();
  EnsureHost(weight);
  ttnn::Tensor dev_x = EnsureDevice2D(x, device);
  ttnn::Tensor dev_w = UploadAffine1D(weight, d, device);
  ttnn::Tensor dev_y = ttnn::rms_norm(dev_x, args.eps, dev_w);
  CommitDevice2D(out, std::move(dev_y));
}

// kSiluAndMul: SwiGLU gate half — out[i,j] = silu(x[i,j]) * x[i,j+d]
// with d = x.shape[1]/2 (cpu_ops.cpp SiluAndMulKernel). Second Qwen3-dense
// op beyond OPT (MLP: gate_up GEMM -> SiluAndMul -> down GEMM). Device path
// keeps the gate_up → SiluAndMul → down GEMM chain on-device: slice the
// last-dim halves, ttnn::silu(gate), ttnn::multiply by up. BF16 tile path
// (same envelope as matmul/norm); not bit-exact vs host f32.
void SiluAndMulKernel(Queue&, Tensor& out, const Tensor& x) {
  VT_CHECK(x.rank == 2 && out.rank == 2,
           "tenstorrent kSiluAndMul: only rank-2 tensors are supported");
  VT_CHECK(IsFloatDType(x.dtype) && (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kSiluAndMul: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(),
           "tenstorrent kSiluAndMul: strided (non-contiguous) tensors are not supported");
  VT_CHECK(x.shape[1] % 2 == 0, "tenstorrent kSiluAndMul: last dim must be even");
  const int64_t t = x.shape[0];
  const int64_t d = x.shape[1] / 2;
  VT_CHECK(out.shape[0] == t && out.shape[1] == d,
           "tenstorrent kSiluAndMul: out shape must be [T, D] with D = x.shape[1]/2");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_x = EnsureDevice2D(x, device);
  const uint32_t tu = static_cast<uint32_t>(t);
  const uint32_t du = static_cast<uint32_t>(d);
  // x = [gate | up] along last dim.
  ttnn::Tensor gate = ttnn::slice(dev_x, ttsl::SmallVector<uint32_t>{0, 0},
                                  ttsl::SmallVector<uint32_t>{tu, du},
                                  ttsl::SmallVector<uint32_t>{1, 1});
  ttnn::Tensor up = ttnn::slice(dev_x, ttsl::SmallVector<uint32_t>{0, du},
                                ttsl::SmallVector<uint32_t>{tu, 2 * du},
                                ttsl::SmallVector<uint32_t>{1, 1});
  ttnn::Tensor silu_gate = ttnn::silu(gate);
  ttnn::Tensor dev_y = ttnn::multiply(silu_gate, up);
  CommitDevice2D(out, std::move(dev_y));
}

// kCastBf16 / kCastF32: elementwise dtype convert via Load/Store (cpu_ops
// CastBf16Kernel / CastF32Kernel). Qwen3 uses these for K/V cache dtype and
// the logits / rope-cache paths. Host-staged; bit-exact for supported pairs.
void CastBf16Kernel(Queue&, Tensor& out, const Tensor& in) {
  VT_CHECK(out.dtype == DType::kBF16, "tenstorrent kCastBf16: out must be bf16");
  VT_CHECK(IsFloatDType(in.dtype), "tenstorrent kCastBf16: in must be float");
  VT_CHECK(out.Numel() == in.Numel(), "tenstorrent kCastBf16: numel mismatch");
  VT_CHECK(out.IsContiguous() && in.IsContiguous(),
           "tenstorrent kCastBf16: contiguous required");
  EnsureHost(in);
  const int64_t n = out.Numel();
  for (int64_t i = 0; i < n; ++i) StoreElemF32(out, i, LoadElemF32(in, i));
  CommitHost(out);
}

void CastF32Kernel(Queue&, Tensor& out, const Tensor& in) {
  VT_CHECK(out.dtype == DType::kF32, "tenstorrent kCastF32: out must be f32");
  VT_CHECK(IsFloatDType(in.dtype), "tenstorrent kCastF32: in must be float");
  VT_CHECK(out.Numel() == in.Numel(), "tenstorrent kCastF32: numel mismatch");
  VT_CHECK(out.IsContiguous() && in.IsContiguous(),
           "tenstorrent kCastF32: contiguous required");
  EnsureHost(in);
  const int64_t n = out.Numel();
  for (int64_t i = 0; i < n; ++i) StoreElemF32(out, i, LoadElemF32(in, i));
  CommitHost(out);
}

// Llama-3 frequency rescale (cpu_ops Llama3ScaleFreq); no-op when scaling_factor
// is unset. Kept so Qwen3 / Llama rope paths share one host implementation.
inline double Llama3ScaleFreq(double freq, const RopeArgs& a) {
  const double sf = static_cast<double>(a.llama3_scaling_factor);
  if (!(sf > 0.0)) return freq;
  constexpr double kTwoPi = 6.283185307179586476925286766559;
  const double lo = static_cast<double>(a.llama3_low_freq_factor);
  const double hi = static_cast<double>(a.llama3_high_freq_factor);
  const double omax = static_cast<double>(a.llama3_orig_max_position);
  const double low_freq_wavelen = omax / lo;
  const double high_freq_wavelen = omax / hi;
  const double wave_len = kTwoPi / freq;
  double smooth = 0.0;
  if (lo != hi) smooth = (omax / wave_len - lo) / (hi - lo);
  if (wave_len < high_freq_wavelen) return freq;
  if (wave_len > low_freq_wavelen) return freq / sf;
  return (1.0 - smooth) * freq / sf + smooth * freq;
}

// In-place NeoX rotation of one head (cpu_ops RopeRotateHead).
void RopeRotateHead(Tensor& t, int64_t head_off, int rot, double base, int64_t pos,
                    const RopeArgs& args) {
  const int half = rot / 2;
  for (int i = 0; i < half; ++i) {
    double freq = std::pow(base, -2.0 * static_cast<double>(i) / static_cast<double>(rot));
    freq = Llama3ScaleFreq(freq, args);
    const double angle = static_cast<double>(pos) * freq;
    const float c = static_cast<float>(std::cos(angle));
    const float s = static_cast<float>(std::sin(angle));
    const float x = LoadElemF32(t, head_off + i);
    const float y = LoadElemF32(t, head_off + i + half);
    StoreElemF32(t, head_off + i, x * c - y * s);
    StoreElemF32(t, head_off + i + half, x * s + y * c);
  }
}

// kRopeNeox: default Qwen3-dense RoPE (cpu_ops RopeNeoxKernel). Host-staged
// f32 math with dtype storeback — same contract as Metal M3b / CPU.
void RopeNeoxKernel(Queue&, Tensor& qs, Tensor& ks, const Tensor& pos, const RopeArgs& args) {
  VT_CHECK(qs.rank == 3 && ks.rank == 3, "tenstorrent kRopeNeox: qs/ks rank-3");
  VT_CHECK(IsFloatDType(qs.dtype) && qs.dtype == ks.dtype,
           "tenstorrent kRopeNeox: qs/ks float same dtype");
  VT_CHECK(pos.rank == 1 && (pos.dtype == DType::kI32 || pos.dtype == DType::kI64),
           "tenstorrent kRopeNeox: positions rank-1 i32/i64");
  VT_CHECK(qs.IsContiguous() && ks.IsContiguous() && pos.IsContiguous(),
           "tenstorrent kRopeNeox: contiguous required");
  VT_CHECK(args.rotary_dim > 0 && (args.rotary_dim % 2) == 0 &&
               args.rotary_dim <= qs.shape[2],
           "tenstorrent kRopeNeox: rotary_dim must be even and <= head_dim");
  const int64_t t = qs.shape[0], hq = qs.shape[1], hk = ks.shape[1], d = qs.shape[2];
  VT_CHECK(ks.shape[0] == t && ks.shape[2] == d, "tenstorrent kRopeNeox: ks shape");
  VT_CHECK(pos.shape[0] == t, "tenstorrent kRopeNeox: positions length");
  EnsureHost(qs);
  EnsureHost(ks);
  EnsureHost(pos);
  const int rot = args.rotary_dim;
  const double base = static_cast<double>(args.base);
  for (int64_t i = 0; i < t; ++i) {
    const int64_t p =
        pos.dtype == DType::kI32 ? pos.Ptr<int32_t>()[i] : pos.Ptr<int64_t>()[i];
    for (int64_t hh = 0; hh < hq; ++hh)
      RopeRotateHead(qs, (i * hq + hh) * d, rot, base, p, args);
    for (int64_t hh = 0; hh < hk; ++hh)
      RopeRotateHead(ks, (i * hk + hh) * d, rot, base, p, args);
  }
  CommitHost(qs);
  CommitHost(ks);
}

// kRopeCosSinCache: per-step cos|sin table [T, rot] (cpu_ops RopeCosSinCacheKernel).
void RopeCosSinCacheKernel(Queue&, Tensor& cos_sin, const Tensor& positions,
                           const RopeArgs& args) {
  VT_CHECK(cos_sin.rank == 2 && cos_sin.dtype == DType::kF32 && cos_sin.IsContiguous(),
           "tenstorrent kRopeCosSinCache: cos_sin contiguous f32 [T,rot]");
  VT_CHECK(positions.rank == 1 &&
               (positions.dtype == DType::kI32 || positions.dtype == DType::kI64) &&
               positions.IsContiguous(),
           "tenstorrent kRopeCosSinCache: positions rank-1 i32/i64");
  VT_CHECK(args.rotary_dim > 0 && (args.rotary_dim % 2) == 0,
           "tenstorrent kRopeCosSinCache: rotary_dim even > 0");
  const int64_t t = cos_sin.shape[0];
  const int rot = args.rotary_dim;
  VT_CHECK(cos_sin.shape[1] == rot && positions.shape[0] == t,
           "tenstorrent kRopeCosSinCache: shape mismatch");
  EnsureHost(positions);
  const int64_t half = rot / 2;
  const double base = static_cast<double>(args.base);
  for (int64_t i = 0; i < t; ++i) {
    const int64_t p = positions.dtype == DType::kI32 ? positions.Ptr<int32_t>()[i]
                                                     : positions.Ptr<int64_t>()[i];
    for (int64_t pair = 0; pair < half; ++pair) {
      double freq =
          std::pow(base, -2.0 * static_cast<double>(pair) / static_cast<double>(rot));
      freq = Llama3ScaleFreq(freq, args);
      const double angle = static_cast<double>(p) * freq;
      StoreElemF32(cos_sin, i * rot + pair, static_cast<float>(std::cos(angle)));
      StoreElemF32(cos_sin, i * rot + half + pair, static_cast<float>(std::sin(angle)));
    }
  }
  CommitHost(cos_sin);
}

// kRopeFromCache: apply precomputed cos|sin (cpu_ops RopeFromCacheKernel).
// Rank-1 positions only (Qwen3-dense); mrope deferred.
void RopeFromCacheKernel(Queue&, Tensor& qs, Tensor* ks, const Tensor& positions,
                         const Tensor& cache, const RopeArgs& args) {
  VT_CHECK(qs.rank == 3 && IsFloatDType(qs.dtype) && qs.IsContiguous(),
           "tenstorrent kRopeFromCache: qs rank-3 contiguous float");
  VT_CHECK(positions.rank == 1 &&
               (positions.dtype == DType::kI32 || positions.dtype == DType::kI64) &&
               positions.IsContiguous(),
           "tenstorrent kRopeFromCache: rank-1 positions only (no mrope yet)");
  VT_CHECK(cache.rank == 2 && IsFloatDType(cache.dtype) && cache.IsContiguous(),
           "tenstorrent kRopeFromCache: cache rank-2 contiguous float");
  VT_CHECK(args.rotary_dim > 0 && (args.rotary_dim % 2) == 0 &&
               args.rotary_dim <= qs.shape[2],
           "tenstorrent kRopeFromCache: rotary_dim");
  const int64_t tokens = qs.shape[0];
  const int64_t hq = qs.shape[1];
  const int64_t hk = ks == nullptr ? 0 : ks->shape[1];
  if (ks != nullptr) {
    VT_CHECK(ks->rank == 3 && ks->dtype == qs.dtype && ks->IsContiguous(),
             "tenstorrent kRopeFromCache: ks must match qs");
    VT_CHECK(ks->shape[0] == tokens && ks->shape[2] == qs.shape[2],
             "tenstorrent kRopeFromCache: ks shape");
  }
  VT_CHECK(positions.shape[0] == tokens, "tenstorrent kRopeFromCache: positions length");
  EnsureHost(qs);
  if (ks != nullptr) EnsureHost(*ks);
  EnsureHost(positions);
  EnsureHost(cache);
  const int64_t half = args.rotary_dim / 2;
  for (int64_t token = 0; token < tokens; ++token) {
    for (int64_t pair = 0; pair < half; ++pair) {
      const int64_t position = positions.dtype == DType::kI32
                                   ? static_cast<int64_t>(positions.Ptr<int32_t>()[token])
                                   : positions.Ptr<int64_t>()[token];
      VT_CHECK(position >= 0 && position < cache.shape[0],
               "tenstorrent kRopeFromCache: position outside cache");
      const int64_t cache_off = position * args.rotary_dim;
      const float c = LoadElemF32(cache, cache_off + pair);
      const float s = LoadElemF32(cache, cache_off + half + pair);
      const int64_t first = args.is_neox_style ? pair : pair * 2;
      const int64_t second = args.is_neox_style ? pair + half : pair * 2 + 1;
      for (int64_t head = 0; head < hq; ++head) {
        const int64_t off = token * qs.stride[0] + head * qs.stride[1];
        const float x = LoadElemF32(qs, off + first);
        const float y = LoadElemF32(qs, off + second);
        StoreElemF32(qs, off + first, x * c - y * s);
        StoreElemF32(qs, off + second, x * s + y * c);
      }
      if (ks != nullptr) {
        for (int64_t head = 0; head < hk; ++head) {
          const int64_t off = token * ks->stride[0] + head * ks->stride[1];
          const float x = LoadElemF32(*ks, off + first);
          const float y = LoadElemF32(*ks, off + second);
          StoreElemF32(*ks, off + first, x * c - y * s);
          StoreElemF32(*ks, off + second, x * s + y * c);
        }
      }
    }
  }
  CommitHost(qs);
  if (ks != nullptr) CommitHost(*ks);
}

// kQkvSplit: pure contiguous column split of merged [T, q+k+v] into q/k/v
// (cpu_ops.cpp QkvSplitKernel). Host-staged: bit-exact memcpy when dtypes match.
void QkvSplitKernel(Queue&, Tensor& q_out, Tensor& k_out, Tensor& v_out, const Tensor& qkv) {
  VT_CHECK(qkv.rank == 2 && IsFloatDType(qkv.dtype),
           "tenstorrent kQkvSplit: rank-2 float qkv required");
  VT_CHECK(q_out.dtype == qkv.dtype && k_out.dtype == qkv.dtype && v_out.dtype == qkv.dtype,
           "tenstorrent kQkvSplit: q/k/v out must match qkv dtype");
  VT_CHECK(q_out.IsContiguous() && k_out.IsContiguous() && v_out.IsContiguous() &&
               qkv.IsContiguous(),
           "tenstorrent kQkvSplit: contiguous required");
  EnsureHost(qkv);
  const int64_t t = qkv.shape[0];
  const int64_t q_dim = q_out.Numel() / t;
  const int64_t k_dim = k_out.Numel() / t;
  const int64_t v_dim = v_out.Numel() / t;
  const int64_t total = q_dim + k_dim + v_dim;
  VT_CHECK(qkv.shape[1] == total, "tenstorrent kQkvSplit: inner dim mismatch");
  const size_t esz = SizeOf(qkv.dtype);
  const auto* src = static_cast<const uint8_t*>(qkv.data);
  auto* qdst = static_cast<uint8_t*>(q_out.data);
  auto* kdst = static_cast<uint8_t*>(k_out.data);
  auto* vdst = static_cast<uint8_t*>(v_out.data);
  for (int64_t i = 0; i < t; ++i) {
    const uint8_t* row = src + static_cast<size_t>(i * total) * esz;
    std::memcpy(qdst + static_cast<size_t>(i * q_dim) * esz, row,
                static_cast<size_t>(q_dim) * esz);
    std::memcpy(kdst + static_cast<size_t>(i * k_dim) * esz, row + static_cast<size_t>(q_dim) * esz,
                static_cast<size_t>(k_dim) * esz);
    std::memcpy(vdst + static_cast<size_t>(i * v_dim) * esz,
                row + static_cast<size_t>(q_dim + k_dim) * esz, static_cast<size_t>(v_dim) * esz);
  }
  CommitHost(q_out);
  CommitHost(k_out);
  CommitHost(v_out);
}

// kReshapeAndCache: write per-token K/V into paged NHD cache slots
// (cpu_cache.cpp ReshapeAndCacheKernel). Stride-driven so unbind-style
// [num_blocks,2,bs,H,D] views work; slot < 0 is a padded-token skip.
// Host-staged pure element copy for F32.
void ReshapeAndCacheKernel(Queue&, const Tensor& k, const Tensor& v, Tensor& k_cache,
                           Tensor& v_cache, const Tensor& slot_mapping) {
  VT_CHECK(k.rank == 3 && v.rank == 3 && k_cache.rank == 4 && v_cache.rank == 4,
           "tenstorrent kReshapeAndCache: k/v rank-3, caches rank-4");
  VT_CHECK(IsFloatDType(k.dtype) && k.dtype == v.dtype && k_cache.dtype == k.dtype &&
               v_cache.dtype == k.dtype,
           "tenstorrent kReshapeAndCache: k/v/caches must share one float dtype");
  EnsureHost(k);
  EnsureHost(v);
  EnsureHost(k_cache);
  EnsureHost(v_cache);
  EnsureHost(slot_mapping);
  VT_CHECK(slot_mapping.rank == 1 && slot_mapping.dtype == DType::kI64,
           "tenstorrent kReshapeAndCache: slot_mapping rank-1 i64");
  const int64_t num_slots = slot_mapping.shape[0];
  const int64_t block_size = k_cache.shape[1];
  const int64_t num_kv_heads = k_cache.shape[2];
  const int64_t head_size = k_cache.shape[3];
  const int64_t n_elems = num_kv_heads * head_size;
  VT_CHECK(k.shape[1] == num_kv_heads && k.shape[2] == head_size && v.shape[1] == num_kv_heads &&
               v.shape[2] == head_size,
           "tenstorrent kReshapeAndCache: k/v head shape must match cache");
  VT_CHECK(k.shape[0] >= num_slots && v.shape[0] >= num_slots,
           "tenstorrent kReshapeAndCache: token count must cover slots");
  // Contiguous NHD page: head stride == head_size (ops.cpp contract).
  VT_CHECK(k_cache.stride[2] == head_size && v_cache.stride[2] == head_size &&
               k_cache.stride[3] == 1 && v_cache.stride[3] == 1,
           "tenstorrent kReshapeAndCache: cache pages must be dense NHD");
  VT_CHECK(k.stride[2] == 1 && v.stride[2] == 1,
           "tenstorrent kReshapeAndCache: k/v innermost stride must be 1");

  const int64_t k_block_stride = k_cache.stride[0];
  const int64_t k_page_stride = k_cache.stride[1];
  const int64_t v_block_stride = v_cache.stride[0];
  const int64_t v_page_stride = v_cache.stride[1];
  const int64_t k_tok_stride = k.stride[0];
  const int64_t v_tok_stride = v.stride[0];
  const int64_t* slots = slot_mapping.Ptr<int64_t>();
  const size_t esz = SizeOf(k.dtype);
  const auto* ksrc = static_cast<const uint8_t*>(k.data);
  const auto* vsrc = static_cast<const uint8_t*>(v.data);
  auto* kdst = static_cast<uint8_t*>(k_cache.data);
  auto* vdst = static_cast<uint8_t*>(v_cache.data);
  const size_t bytes = static_cast<size_t>(n_elems) * esz;

  for (int64_t t = 0; t < num_slots; ++t) {
    const int64_t slot = slots[t];
    if (slot < 0) continue;
    const int64_t block = slot / block_size;
    const int64_t offset = slot % block_size;
    const int64_t kdst_off = block * k_block_stride + offset * k_page_stride;
    const int64_t vdst_off = block * v_block_stride + offset * v_page_stride;
    std::memcpy(kdst + static_cast<size_t>(kdst_off) * esz,
                ksrc + static_cast<size_t>(t * k_tok_stride) * esz, bytes);
    std::memcpy(vdst + static_cast<size_t>(vdst_off) * esz,
                vsrc + static_cast<size_t>(t * v_tok_stride) * esz, bytes);
  }
  CommitHost(k_cache);
  CommitHost(v_cache);
}

// kPagedAttention: causal/non-causal GQA softmax over the paged NHD cache
// (cpu_paged_attn.cpp PagedAttentionKernel). Host-staged f32 oracle matching
// the CPU reference while Alloc is host memory. Device ttnn::sdpa_decode is
// deferred to the device-resident redesign — its layout contract does not map
// 1:1 onto vLLM's block_table without that work.
//
// This step supports: F32 query/out/cache, kAuto KV (no fp8), optional softcap
// and window_size (same math as CPU). OPT-125m uses causal + full window + no
// softcap.
void PagedAttentionKernel(Queue&, Tensor& out, const Tensor& query, const Tensor& k_cache,
                          const Tensor& v_cache, const Tensor& block_table,
                          const Tensor& seq_lens, const Tensor& query_start_loc,
                          const PagedAttentionArgs& args) {
  EnsureHost(query);
  EnsureHost(k_cache);
  EnsureHost(v_cache);
  EnsureHost(block_table);
  EnsureHost(seq_lens);
  EnsureHost(query_start_loc);
  VT_CHECK(query.rank == 3 && out.rank == 3 && k_cache.rank == 4 && v_cache.rank == 4,
           "tenstorrent kPagedAttention: query/out rank-3, caches rank-4");
  VT_CHECK(IsFloatDType(query.dtype) && (out.dtype == DType::kF32 || out.dtype == DType::kBF16) &&
               IsFloatDType(k_cache.dtype) && k_cache.dtype == v_cache.dtype,
           "tenstorrent kPagedAttention: float query/cache, f32/bf16 out");
  VT_CHECK(args.kv_cache_dtype == Fp8KVCacheDataType::kAuto,
           "tenstorrent kPagedAttention: fp8 KV not supported in this step");
  VT_CHECK(args.scale > 0.0f, "tenstorrent kPagedAttention: scale must be > 0");
  VT_CHECK(query.IsContiguous() && out.IsContiguous() && seq_lens.IsContiguous() &&
               query_start_loc.IsContiguous(),
           "tenstorrent kPagedAttention: query/out/seq_lens/query_start_loc contiguous");

  const int64_t total_q = query.shape[0];
  const int64_t hq = query.shape[1];
  const int64_t d = query.shape[2];
  const int64_t block_size = k_cache.shape[1];
  const int64_t num_kv_heads = k_cache.shape[2];
  VT_CHECK(d == k_cache.shape[3], "tenstorrent kPagedAttention: head_size mismatch");
  VT_CHECK(hq % num_kv_heads == 0, "tenstorrent kPagedAttention: GQA ratio");
  const int64_t qpk = hq / num_kv_heads;
  const float scale = args.scale;
  const float softcap = args.logits_soft_cap;
  const int64_t window_left = args.window_size.has_value() ? args.window_size->left : -1;
  const int64_t window_right = args.window_size.has_value() ? args.window_size->right : -1;

  const int64_t num_reqs = seq_lens.shape[0];
  const int32_t* qsl = query_start_loc.Ptr<int32_t>();
  const int32_t* slens = seq_lens.Ptr<int32_t>();
  const int32_t* btab = block_table.Ptr<int32_t>();
  const int64_t bt_row = block_table.stride[0], bt_col = block_table.stride[1];
  const int64_t kc_blk = k_cache.stride[0], kc_pg = k_cache.stride[1], kc_hd = k_cache.stride[2];
  const int64_t vc_blk = v_cache.stride[0], vc_pg = v_cache.stride[1], vc_hd = v_cache.stride[2];

  std::vector<int32_t> tok_pos(static_cast<size_t>(total_q));
  std::vector<int32_t> tok_slen(static_cast<size_t>(total_q));
  std::vector<int32_t> tok_req(static_cast<size_t>(total_q));
  for (int64_t r = 0; r < num_reqs; ++r) {
    const int64_t q0 = qsl[r], q1 = qsl[r + 1];
    const int64_t query_len = q1 - q0;
    if (query_len <= 0) continue;
    const int64_t seqlen = slens[r];
    const int64_t context = seqlen - query_len;
    for (int64_t local = 0; local < query_len; ++local) {
      tok_pos[static_cast<size_t>(q0 + local)] = static_cast<int32_t>(context + local);
      tok_slen[static_cast<size_t>(q0 + local)] = static_cast<int32_t>(seqlen);
      tok_req[static_cast<size_t>(q0 + local)] = static_cast<int32_t>(r);
    }
  }

  std::vector<float> probs;
  std::vector<float> acc(static_cast<size_t>(d));
  for (int64_t t = 0; t < total_q; ++t) {
    const int64_t r = tok_req[static_cast<size_t>(t)];
    const int64_t p = tok_pos[static_cast<size_t>(t)];
    const int64_t seqlen = tok_slen[static_cast<size_t>(t)];
    const int64_t jmin = window_left >= 0 ? std::max<int64_t>(0, p - window_left) : 0;
    int64_t jmax = args.causal ? p : seqlen - 1;
    if (window_right >= 0) jmax = std::min(jmax, p + window_right);
    jmax = std::min(jmax, seqlen - 1);
    if (jmax < jmin) continue;
    probs.assign(static_cast<size_t>(jmax - jmin + 1), 0.0f);
    for (int64_t h = 0; h < hq; ++h) {
      const int64_t g = h / qpk;
      const int64_t qoff = (t * hq + h) * d;
      float m = -std::numeric_limits<float>::infinity();
      for (int64_t j = jmin; j <= jmax; ++j) {
        const int64_t blk = btab[r * bt_row + (j / block_size) * bt_col];
        const int64_t off = j % block_size;
        const int64_t kbase = blk * kc_blk + off * kc_pg + g * kc_hd;
        float dot = 0.0f;
        for (int64_t e = 0; e < d; ++e)
          dot += LoadElemF32(query, qoff + e) * LoadElemF32(k_cache, kbase + e);
        dot *= scale;
        if (softcap > 0.0f) dot = softcap * std::tanh(dot / softcap);
        probs[static_cast<size_t>(j - jmin)] = dot;
        if (dot > m) m = dot;
      }
      float denom = 0.0f;
      for (int64_t j = jmin; j <= jmax; ++j) {
        const float e = std::exp(probs[static_cast<size_t>(j - jmin)] - m);
        probs[static_cast<size_t>(j - jmin)] = e;
        denom += e;
      }
      const float inv = 1.0f / denom;
      for (int64_t e = 0; e < d; ++e) acc[static_cast<size_t>(e)] = 0.0f;
      for (int64_t j = jmin; j <= jmax; ++j) {
        const float pw = probs[static_cast<size_t>(j - jmin)] * inv;
        const int64_t blk = btab[r * bt_row + (j / block_size) * bt_col];
        const int64_t off = j % block_size;
        const int64_t vbase = blk * vc_blk + off * vc_pg + g * vc_hd;
        for (int64_t e = 0; e < d; ++e)
          acc[static_cast<size_t>(e)] += pw * LoadElemF32(v_cache, vbase + e);
      }
      for (int64_t e = 0; e < d; ++e) StoreElemF32(out, qoff + e, acc[static_cast<size_t>(e)]);
    }
  }
  CommitHost(out);
}

// kGreedyArgmax: per-row lowest-index max of f32 logits (cpu_sample.cpp).
// OPT's lm_head produces F32 logits; host-staged, bit-exact with CPU.
void GreedyArgmaxKernel(Queue&, Tensor& token_ids, const Tensor& logits) {
  VT_CHECK(logits.rank == 2 && logits.dtype == DType::kF32 && logits.IsContiguous(),
           "tenstorrent kGreedyArgmax: logits must be contiguous f32 [N,V]");
  VT_CHECK(token_ids.rank == 1 && token_ids.dtype == DType::kI64 && token_ids.IsContiguous() &&
               token_ids.shape[0] == logits.shape[0],
           "tenstorrent kGreedyArgmax: token_ids must be i64 [N]");
  EnsureHost(logits);
  const int64_t n = logits.shape[0], v = logits.shape[1];
  const float* lp = logits.Ptr<float>();
  int64_t* out = token_ids.Ptr<int64_t>();
  for (int64_t i = 0; i < n; ++i) {
    const float* row = lp + i * v;
    int64_t best = 0;
    float best_v = row[0];
    for (int64_t j = 1; j < v; ++j) {
      if (row[j] > best_v) {
        best_v = row[j];
        best = j;
      }
    }
    out[i] = best;
  }
  CommitHost(token_ids);
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
    RegisterOp(OpId::kRmsNorm, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<RmsNormFn>(&RmsNormKernel)));
    RegisterOp(OpId::kSiluAndMul, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<SiluAndMulFn>(&SiluAndMulKernel)));
    RegisterOp(OpId::kCastBf16, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<CastBf16Fn>(&CastBf16Kernel)));
    RegisterOp(OpId::kCastF32, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<CastF32Fn>(&CastF32Kernel)));
    RegisterOp(OpId::kRopeNeox, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<RopeFn>(&RopeNeoxKernel)));
    RegisterOp(OpId::kRopeCosSinCache, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<RopeCosSinCacheFn>(&RopeCosSinCacheKernel)));
    RegisterOp(OpId::kRopeFromCache, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<RopeFromCacheFn>(&RopeFromCacheKernel)));
    RegisterOp(OpId::kQkvSplit, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<QkvSplitFn>(&QkvSplitKernel)));
    RegisterOp(OpId::kReshapeAndCache, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<ReshapeAndCacheFn>(&ReshapeAndCacheKernel)));
    RegisterOp(OpId::kPagedAttention, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<PagedAttentionFn>(&PagedAttentionKernel)));
    RegisterOp(OpId::kGreedyArgmax, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<GreedyArgmaxFn>(&GreedyArgmaxKernel)));
  }
} registrar;

}  // namespace

// ---- Called from TenstorrentBackend::Alloc/Free/Copy (no ttnn in that TU). ----
void RegisterHostBuffer(void* host, size_t bytes) {
  if (host == nullptr) return;
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot s;
  s.host = host;
  s.bytes = bytes;
  s.host_current = true;
  s.device_current = false;
  Slots()[reinterpret_cast<uintptr_t>(host)] = std::move(s);
}

void UnregisterHostBuffer(void* host) {
  if (host == nullptr) return;
  std::lock_guard<std::mutex> g(SlotMutex());
  Slots().erase(reinterpret_cast<uintptr_t>(host));
}

void MarkHostWritten(void* host) {
  if (host == nullptr) return;
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(host);
  if (s == nullptr) return;
  s->host_current = true;
  s->device_current = false;
  s->device = std::nullopt;
}

void EnsureHostBytes(void* host) {
  if (host == nullptr) return;
  ttnn::Tensor dev;
  size_t bytes = 0;
  void* base = nullptr;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(host);
    if (s == nullptr || s->host_current) return;
    VT_CHECK(s->device_current && s->device.has_value(),
             "tenstorrent: EnsureHostBytes with no current device data");
    dev = *s->device;
    bytes = s->bytes;
    base = s->host;
  }
  std::vector<float> result = dev.to_vector<float>();
  const size_t n = result.size();
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(host);
    if (s == nullptr || s->host_current) return;
    // Device results are f32 via to_vector. Host Alloc is typically
    // numel*sizeof(float) (tests/f32 path) or numel*2 (bf16 activations).
    if (bytes >= n * sizeof(float)) {
      std::memcpy(base, result.data(), n * sizeof(float));
    } else if (bytes >= n * sizeof(uint16_t)) {
      auto* dst = static_cast<uint16_t*>(base);
      for (size_t i = 0; i < n; ++i) dst[i] = F32ToBF16(result[i]);
    } else {
      VT_CHECK(false, "tenstorrent: EnsureHostBytes host buffer too small");
    }
    s->host_current = true;
  }
}

}  // namespace vt::tenstorrent
