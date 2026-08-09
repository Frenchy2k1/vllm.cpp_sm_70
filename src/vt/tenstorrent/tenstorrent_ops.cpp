// Tenstorrent op providers — the ttnn adapter layer (BACKEND-TENSTORRENT W0,
// .agents/specs/tenstorrent-backend.md). vllm.cpp original; no upstream
// mirror (vLLM has no Tenstorrent platform). ONE op registered: kMatmul, the
// first vertical slice, mirroring how Metal's W0 skeleton landed one op
// before the rest (docs/STATUS.md Backend detail).
//
// SCOPE: 2D row-major F32 tensors only. Every other shape/dtype is a VT_CHECK
// failure, not a silent wrong answer or a slow-path guess — this op table has
// no CPU reference-tier fallback to fall into anyway (UnifiedMemory()==false,
// tenstorrent_backend.cpp), so failing loudly here is the only honest option.
// Widening dtype/rank coverage is explicitly deferred, not attempted here.
#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/tenstorrent/tenstorrent_device.h"

#include <vector>

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/matmul/matmul.hpp>

#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp>

namespace vt::tenstorrent {
namespace {

// Host round-trip per call — see tenstorrent_backend.cpp's file-level SCOPE
// note for why, and .agents/specs/tenstorrent-backend.md's "Resolved:
// hands-on spike result" for the hardware evidence this exact from_vector /
// matmul / to_vector sequence produces a correct answer (max_abs_diff
// 0.03375 vs max_ref_mag 4.14 on a real Blackhole, bf16 tolerance).
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

  std::vector<float> host_a(a.Ptr<float>(), a.Ptr<float>() + a.Numel());
  std::vector<float> host_b(b.Ptr<float>(), b.Ptr<float>() + b.Numel());

  MeshDevice& device = SharedMeshDevice();
  auto layout_of = [](uint32_t rows, uint32_t cols) {
    return tt::tt_metal::TensorSpec(
        tt::tt_metal::Shape({rows, cols}),
        tt::tt_metal::TensorLayout(
            ttnn::DataType::BFLOAT16, tt::tt_metal::PageConfig(ttnn::Layout::TILE),
            tt::tt_metal::MemoryConfig{}));
  };

  ttnn::Tensor dev_a = ttnn::Tensor::from_vector<float>(host_a, layout_of(M, K), &device);
  ttnn::Tensor dev_b = ttnn::Tensor::from_vector<float>(host_b, layout_of(K, N), &device);
  ttnn::Tensor dev_c = ttnn::operations::matmul::matmul(dev_a, dev_b);

  std::vector<float> result = dev_c.to_vector<float>();
  VT_CHECK(static_cast<int64_t>(result.size()) == out.Numel(),
           "tenstorrent kMatmul: unexpected result size");
  std::copy(result.begin(), result.end(), out.Ptr<float>());
}

struct Registrar {
  Registrar() {
    if (!DeviceAvailable()) return;
    RegisterOp(OpId::kMatmul, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulKernel)));
  }
} registrar;

}  // namespace
}  // namespace vt::tenstorrent
