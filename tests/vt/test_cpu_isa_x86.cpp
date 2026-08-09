#include <array>
#include <string>

#include "doctest/doctest.h"
#include "vt/cpu/cpu_isa_x86.h"

namespace {

vt::cpu::X86IsaCaps FullAvx512() {
  return {.sse2 = true,
          .f16c = true,
          .avx = true,
          .osxsave = true,
          .avx2 = true,
          .avx512f = true,
          .avx512bw = true,
          .avx512vl = true,
          .xcr0 = 0xe6};
}

}  // namespace

TEST_CASE("x86 ISA selection requires CPU bits and exact OS-enabled state") {
  using vt::cpu::SelectX86IsaTier;
  using vt::cpu::X86IsaTier;
  X86IsaTier selected{};
  std::string error;

  vt::cpu::X86IsaCaps caps{.sse2 = true};
  CHECK(SelectX86IsaTier(caps, "", &selected, &error));
  CHECK(selected == X86IsaTier::kSse2);

  caps = FullAvx512();
  caps.xcr0 = 0x6;
  CHECK(SelectX86IsaTier(caps, "", &selected, &error));
  CHECK(selected == X86IsaTier::kAvx2);
  CHECK_FALSE(SelectX86IsaTier(caps, "avx512", &selected, &error));

  caps = FullAvx512();
  for (bool* bit : std::array{&caps.avx512f, &caps.avx512bw, &caps.avx512vl}) {
    *bit = false;
    CHECK_FALSE(SelectX86IsaTier(caps, "avx512", &selected, &error));
    *bit = true;
  }
  CHECK(SelectX86IsaTier(caps, "avx512", &selected, &error));
  CHECK(selected == X86IsaTier::kAvx512);
}

TEST_CASE("x86 forced tiers fail closed instead of silently narrowing") {
  using vt::cpu::SelectX86IsaTier;
  using vt::cpu::X86IsaTier;
  X86IsaTier selected{};
  std::string error;
  const vt::cpu::X86IsaCaps baseline{.sse2 = true};

  CHECK(SelectX86IsaTier(baseline, "portable", &selected, &error));
  CHECK(selected == X86IsaTier::kPortable);
  CHECK(SelectX86IsaTier(baseline, "sse2", &selected, &error));
  CHECK(selected == X86IsaTier::kSse2);
  CHECK_FALSE(SelectX86IsaTier(baseline, "sse2+f16c", &selected, &error));
  CHECK_FALSE(SelectX86IsaTier(baseline, "avx2", &selected, &error));
  CHECK_FALSE(SelectX86IsaTier(baseline, "avx512", &selected, &error));
  CHECK_FALSE(SelectX86IsaTier(baseline, "amx", &selected, &error));
        const bool has_reason = error.find("unsupported") != std::string::npos ||
                                error.find("unknown") != std::string::npos;
        CHECK(has_reason);
}

TEST_CASE("x86 release inventory lists only compiled elementwise GEMM tiers") {
  const auto inventory = vt::cpu::X86IsaTierInventory();
  REQUIRE(inventory.size() == 5);
  CHECK(inventory[0].name == "portable");
  CHECK(inventory[1].name == "sse2");
  CHECK(inventory[2].name == "sse2+f16c");
  CHECK(inventory[3].name == "avx2");
  CHECK(inventory[4].name == "avx512");
  for (const auto& tier : inventory) {
    CHECK(tier.kernel_family == "elementwise-gemm");
    CHECK(tier.cpu_features.find("vnni") == std::string_view::npos);
    CHECK(tier.cpu_features.find("amx") == std::string_view::npos);
  }
}

TEST_CASE("detected host resolves to a supported tier") {
  vt::cpu::X86IsaTier selected{};
  std::string error;
  const auto caps = vt::cpu::DetectX86IsaCaps();
  CHECK(vt::cpu::SelectX86IsaTier(caps, "", &selected, &error));
  CHECK(vt::cpu::X86IsaTierSupported(caps, selected));
}
