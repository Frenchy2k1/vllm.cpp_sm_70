#include "vt/cpu/cpu_isa_x86.h"

#include <sstream>

#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace vt::cpu {
namespace {

constexpr uint64_t kAvxXcr0 = (1ULL << 1) | (1ULL << 2);
constexpr uint64_t kAvx512Xcr0 =
    kAvxXcr0 | (1ULL << 5) | (1ULL << 6) | (1ULL << 7);

constexpr std::array<X86IsaTierRequirement, 5> kInventory{{
    {X86IsaTier::kPortable, "portable", "elementwise-gemm", "none", "none", 0},
    {X86IsaTier::kSse2, "sse2", "elementwise-gemm", "sse2", "xmm", 0},
    {X86IsaTier::kSse2F16c, "sse2+f16c", "elementwise-gemm",
     "sse2,avx,f16c,osxsave", "xcr0:xmm,ymm", kAvxXcr0},
    {X86IsaTier::kAvx2, "avx2", "elementwise-gemm",
     "sse2,avx,f16c,avx2,osxsave", "xcr0:xmm,ymm", kAvxXcr0},
    {X86IsaTier::kAvx512, "avx512", "elementwise-gemm",
     "sse2,avx,f16c,avx2,avx512f,avx512bw,avx512vl,osxsave",
     "xcr0:xmm,ymm,opmask,zmm_hi256,hi16_zmm", kAvx512Xcr0},
}};

#if defined(__x86_64__) || defined(_M_X64)
uint64_t ReadXcr0() {
#if defined(_MSC_VER)
  return _xgetbv(0);
#else
  uint32_t eax = 0;
  uint32_t edx = 0;
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
  return (static_cast<uint64_t>(edx) << 32) | eax;
#endif
}

void ReadCpuid(uint32_t leaf, uint32_t subleaf, uint32_t* eax, uint32_t* ebx,
               uint32_t* ecx, uint32_t* edx) {
#if defined(_MSC_VER)
  int regs[4]{};
  __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
  *eax = static_cast<uint32_t>(regs[0]);
  *ebx = static_cast<uint32_t>(regs[1]);
  *ecx = static_cast<uint32_t>(regs[2]);
  *edx = static_cast<uint32_t>(regs[3]);
#else
  __cpuid_count(leaf, subleaf, *eax, *ebx, *ecx, *edx);
#endif
}
#endif

}  // namespace

const std::array<X86IsaTierRequirement, 5>& X86IsaTierInventory() {
  return kInventory;
}

const char* X86IsaTierName(X86IsaTier tier) {
  for (const auto& item : kInventory) {
    if (item.tier == tier) return item.name.data();
  }
  return "unknown";
}

X86IsaCaps DetectX86IsaCaps() {
  X86IsaCaps caps{};
#if defined(__x86_64__) || defined(_M_X64)
  uint32_t eax = 0;
  uint32_t ebx = 0;
  uint32_t ecx = 0;
  uint32_t edx = 0;
  ReadCpuid(0, 0, &eax, &ebx, &ecx, &edx);
  const uint32_t max_leaf = eax;
  if (max_leaf >= 1) {
    ReadCpuid(1, 0, &eax, &ebx, &ecx, &edx);
    caps.sse2 = (edx & (1U << 26)) != 0;
    caps.osxsave = (ecx & (1U << 27)) != 0;
    caps.avx = (ecx & (1U << 28)) != 0;
    caps.f16c = (ecx & (1U << 29)) != 0;
    if (caps.osxsave) caps.xcr0 = ReadXcr0();
  }
  if (max_leaf >= 7) {
    ReadCpuid(7, 0, &eax, &ebx, &ecx, &edx);
    caps.avx2 = (ebx & (1U << 5)) != 0;
    caps.avx512f = (ebx & (1U << 16)) != 0;
    caps.avx512bw = (ebx & (1U << 30)) != 0;
    caps.avx512vl = (ebx & (1U << 31)) != 0;
  }
#endif
  return caps;
}

bool X86IsaTierSupported(const X86IsaCaps& caps, X86IsaTier tier) {
  const bool avx_state = caps.osxsave && (caps.xcr0 & kAvxXcr0) == kAvxXcr0;
  const bool avx512_state =
      caps.osxsave && (caps.xcr0 & kAvx512Xcr0) == kAvx512Xcr0;
  switch (tier) {
    case X86IsaTier::kPortable:
      return true;
    case X86IsaTier::kSse2:
      return caps.sse2;
    case X86IsaTier::kSse2F16c:
      return caps.sse2 && caps.avx && caps.f16c && avx_state;
    case X86IsaTier::kAvx2:
      return caps.sse2 && caps.avx && caps.f16c && caps.avx2 && avx_state;
    case X86IsaTier::kAvx512:
      return caps.sse2 && caps.avx && caps.f16c && caps.avx2 && caps.avx512f &&
             caps.avx512bw && caps.avx512vl && avx512_state;
  }
  return false;
}

bool SelectX86IsaTier(const X86IsaCaps& caps, std::string_view forced,
                      X86IsaTier* selected, std::string* error) {
  X86IsaTier requested = X86IsaTier::kPortable;
  if (forced.empty()) {
    for (auto it = kInventory.rbegin(); it != kInventory.rend(); ++it) {
      if (X86IsaTierSupported(caps, it->tier)) {
        *selected = it->tier;
        return true;
      }
    }
  } else {
    bool found = false;
    for (const auto& item : kInventory) {
      if (item.name == forced) {
        requested = item.tier;
        found = true;
        break;
      }
    }
    if (!found) {
      if (error != nullptr) *error = "unknown x86 ISA tier '" + std::string(forced) + "'";
      return false;
    }
    if (X86IsaTierSupported(caps, requested)) {
      *selected = requested;
      return true;
    }
  }

  if (error != nullptr) {
    const auto& requirement = kInventory[static_cast<std::size_t>(requested)];
    std::ostringstream message;
    message << "unsupported x86 ISA tier '" << requirement.name << "': requires CPU "
            << requirement.cpu_features << " and OS state " << requirement.os_state;
    *error = message.str();
  }
  return false;
}

}  // namespace vt::cpu
