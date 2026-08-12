#include <doctest/doctest.h>

#include "vllm/version.h"

TEST_CASE("Version reports the exact configured build identity") {
  std::string expected = VLLM_CPP_EXPECTED_BUILD_VERSION;
#ifndef VLLM_CPP_CUDA
  CHECK(vllm::Version() == expected);
#else
  CHECK(vllm::Version() == expected + "+cuda");
#endif
}
