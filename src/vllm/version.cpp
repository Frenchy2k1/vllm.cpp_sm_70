#include "vllm/version.h"

namespace vllm {

std::string Version() {
  std::string v = VLLM_CPP_BUILD_VERSION;
#ifdef VLLM_CPP_CUDA
  v += "+cuda";
#endif
  return v;
}

}  // namespace vllm
