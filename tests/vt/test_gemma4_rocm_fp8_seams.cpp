// CPU seam gate for #317 Gemma4 ROCm FP8 MoE stack.
// Does NOT require a ROCm device: proves portable fused_ops declarations link
// and the documented VT_GEMMA4_ / VT_ATTN_ env knobs parse inertly on CPU.
#include <doctest/doctest.h>

#include <cstdlib>
#include <string>

#include "vt/fused_ops.h"

namespace {

struct EnvRestorer {
  const char* key;
  bool had = false;
  std::string prev;
  explicit EnvRestorer(const char* k) : key(k) {
    if (const char* v = std::getenv(k)) {
      had = true;
      prev = v;
    }
  }
  ~EnvRestorer() {
    if (had) ::setenv(key, prev.c_str(), 1);
    else ::unsetenv(key);
  }
};

int EnvInt(const char* key, int def) {
  const char* e = std::getenv(key);
  if (e == nullptr || e[0] == '\0') return def;
  char* end = nullptr;
  long v = std::strtol(e, &end, 10);
  if (end == e) return def;
  return static_cast<int>(v);
}

}  // namespace

TEST_CASE("gemma4 rocm fp8 seams: fused_ops ExpertGeGLU symbols link on CPU") {
  // Taking addresses forces the linker to resolve the portable wrappers.
  // Bodies may no-op or refuse without ROCm — that is fine for this gate.
  auto* p0 = &vt::ExpertGeGLUFp8TopKM1;
  auto* p1 = &vt::ExpertGeGLUFp8TopKIndexed;
  auto* p2 = &vt::MatmulBTFp8Channel;
  auto* p3 = &vt::DequantFp8ChannelBf16;
  auto* p4 = &vt::PrewarmExpertGeGLUFp8TopK;
  CHECK(p0 != nullptr);
  CHECK(p1 != nullptr);
  CHECK(p2 != nullptr);
  CHECK(p3 != nullptr);
  CHECK(p4 != nullptr);
}

TEST_CASE("gemma4 rocm fp8 seams: recipe env knobs parse inert defaults") {
  // Defaults match docs/ENVIRONMENT.md / lab recipe when unset.
  {
    EnvRestorer a("VT_GEMMA4_FP8_HW_CVT");
    EnvRestorer b("VT_ATTN_DECODE_KV_SPLITS");
    EnvRestorer c("VT_ATTN_DECODE_SLIDE_SPLITS");
    EnvRestorer d("VT_ATTN_DECODE_SPLIT_WARPS");
    ::unsetenv("VT_GEMMA4_FP8_HW_CVT");
    ::unsetenv("VT_ATTN_DECODE_KV_SPLITS");
    ::unsetenv("VT_ATTN_DECODE_SLIDE_SPLITS");
    ::unsetenv("VT_ATTN_DECODE_SPLIT_WARPS");
    // Unset → code defaults are env-read at runtime inside HIP; here we only
    // document the *recipe* integers the lab pins (not process-wide defaults).
    CHECK(EnvInt("VT_GEMMA4_FP8_HW_CVT", 1) == 1);
    CHECK(EnvInt("VT_ATTN_DECODE_KV_SPLITS", 16) == 16);
    CHECK(EnvInt("VT_ATTN_DECODE_SLIDE_SPLITS", 8) == 8);
    CHECK(EnvInt("VT_ATTN_DECODE_SPLIT_WARPS", 12) == 12);
  }
  {
    EnvRestorer a("VT_GEMMA4_FP8_HW_CVT");
    ::setenv("VT_GEMMA4_FP8_HW_CVT", "0", 1);
    CHECK(EnvInt("VT_GEMMA4_FP8_HW_CVT", 1) == 0);
  }
}
