#pragma once

#include <cstdint>
#include <vector>

namespace vllm::detail {

// Deterministic float-domain input used by the expert diagnostic probe.
std::vector<float> DeepseekV4ExpertProbeInput(int64_t size, float frequency);

}  // namespace vllm::detail
