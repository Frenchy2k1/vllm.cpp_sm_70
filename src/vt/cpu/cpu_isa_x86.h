#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace vt::cpu {

enum class X86IsaTier { kPortable, kSse2, kSse2F16c, kAvx2, kAvx512 };

struct X86IsaCaps {
  bool sse2 = false;
  bool f16c = false;
  bool avx = false;
  bool osxsave = false;
  bool avx2 = false;
  bool avx512f = false;
  bool avx512bw = false;
  bool avx512vl = false;
  uint64_t xcr0 = 0;
};

struct X86IsaTierRequirement {
  X86IsaTier tier;
  std::string_view name;
  std::string_view kernel_family;
  std::string_view cpu_features;
  std::string_view os_state;
  uint64_t xcr0_mask;
};

const std::array<X86IsaTierRequirement, 5>& X86IsaTierInventory();
const char* X86IsaTierName(X86IsaTier tier);
X86IsaCaps DetectX86IsaCaps();
bool X86IsaTierSupported(const X86IsaCaps& caps, X86IsaTier tier);
bool SelectX86IsaTier(const X86IsaCaps& caps, std::string_view forced,
                      X86IsaTier* selected, std::string* error);

}  // namespace vt::cpu
