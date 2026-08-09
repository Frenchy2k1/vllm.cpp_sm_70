#pragma once

#include <cstdint>

namespace vt::cuda {

// Launch the optional sm12x packed BF16 SiLU->NVFP4 producer. Returns false
// when the device, shape, layout, alignment, or A/B flag selects the portable
// producer retained in cuda_matmul_nvfp4.cu.
bool TryLaunchSiluAndMulFp4QuantPackedSm12x(
    void* stream, uint8_t* packed, uint8_t* scale, const void* gate_up,
    float input_global_scale, int64_t m_rows, int64_t i_dim,
    int64_t scale_cols, int64_t scale_numel, bool approx_recip);

}  // namespace vt::cuda
