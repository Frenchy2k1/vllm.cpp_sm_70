#include <doctest/doctest.h>

#include "vllm/version.h"

TEST_CASE("Version reports semver from project()") {
  auto v = vllm::Version();
  CHECK(v.rfind("0.0.2", 0) == 0);  // starts with MAJOR.MINOR.PATCH
#ifndef VLLM_CPP_CUDA
  CHECK(v == "0.0.2");
#else
  CHECK(v == "0.0.2+cuda");
#endif
}
