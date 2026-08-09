// Tenstorrent op providers — the ttnn adapter layer (BACKEND-TENSTORRENT W0,
// .agents/specs/tenstorrent-backend.md). vllm.cpp original; no upstream
// mirror (vLLM has no Tenstorrent platform). Op table: kMatmul, kMatmulBT,
// kAdd, kRelu — the OPT-125m linear/residual/activation slice, mirroring how
// Metal's W0 skeleton landed one op before the rest (docs/STATUS.md Backend
// detail).
//
// SCOPE: 2D row-major F32 tensors only (kAdd also allows rank-1 `b` for the
// nn.Linear bias broadcast, matching cpu_layernorm.cpp's AddKernel). Every
// other shape/dtype is a VT_CHECK failure, not a silent wrong answer or a
// slow-path guess — this op table has no CPU reference-tier fallback to fall
// into anyway (UnifiedMemory()==false, tenstorrent_backend.cpp), so failing
// loudly here is the only honest option. Widening dtype/rank coverage is
// explicitly deferred, not attempted here.
#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/tenstorrent/tenstorrent_device.h"

#include <vector>

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>

#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp>

namespace vt::tenstorrent {
namespace {

// Shared host round-trip helpers — see the kMatmul comment below for why the
// upload/compute/readback shape is correct (hardware evidence in
// .agents/specs/tenstorrent-backend.md's "Resolved: hands-on spike result").
tt::tt_metal::TensorSpec TileSpecOf(uint32_t rows, uint32_t cols) {
  return tt::tt_metal::TensorSpec(
      tt::tt_metal::Shape({rows, cols}),
      tt::tt_metal::TensorLayout(
          ttnn::DataType::BFLOAT16, tt::tt_metal::PageConfig(ttnn::Layout::TILE),
          tt::tt_metal::MemoryConfig{}));
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
  }
} registrar;

}  // namespace
}  // namespace vt::tenstorrent
