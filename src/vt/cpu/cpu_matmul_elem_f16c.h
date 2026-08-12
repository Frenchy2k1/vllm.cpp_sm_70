#pragma once

#include "vt/cpu/cpu_matmul_elem.h"

namespace vt::cpu {

// Populate only the f16 slots of the SSE2 tier. The implementation lives in a
// dedicated F16C translation unit so the portable dispatcher remains valid for
// the MSVC x64 baseline.
void FillF16cTier(ElemGemmTierTable* table);

}  // namespace vt::cpu
