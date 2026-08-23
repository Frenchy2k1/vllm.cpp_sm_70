# Toolkit-free executable contract for W1 per-source CUDA gencode.
cmake_minimum_required(VERSION 3.24)

get_filename_component(_here "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
include("${_here}/CudaArchFeatures.cmake")

set(_ten_sms "80;86;87;89;90a;100a;103a;110;120a;121a")

function(expect_gencode ARCHS EXPECTED)
  vt_cuda_gencode_options(_got "${ARCHS}")
  if(NOT "${_got}" STREQUAL "${EXPECTED}")
    message(FATAL_ERROR
      "GENCODE MISMATCH for [${ARCHS}]: expected [${EXPECTED}], got [${_got}]")
  endif()
endfunction()

expect_gencode("90a;120a"
  "-gencode=arch=compute_90a,code=sm_90a;-gencode=arch=compute_120a,code=sm_120a")
expect_gencode("${_ten_sms}"
  "-gencode=arch=compute_80,code=sm_80;-gencode=arch=compute_86,code=sm_86;-gencode=arch=compute_87,code=sm_87;-gencode=arch=compute_89,code=sm_89;-gencode=arch=compute_90a,code=sm_90a;-gencode=arch=compute_100a,code=sm_100a;-gencode=arch=compute_103a,code=sm_103a;-gencode=arch=compute_110,code=sm_110;-gencode=arch=compute_120a,code=sm_120a;-gencode=arch=compute_121a,code=sm_121a")

set(VLLM_CPP_CUDA_ARCHITECTURES "${_ten_sms}")
vt_cuda_feature_archs(_fp4 "fp4-mma")
vt_cuda_feature_archs(_sm90 "scaledmm-c3x-sm90")
vt_cuda_feature_archs(_sm100 "scaledmm-c3x-sm100")
vt_cuda_feature_archs(_fa2 "fa2")
expect_gencode("${_fp4}"
  "-gencode=arch=compute_120a,code=sm_120a;-gencode=arch=compute_121a,code=sm_121a")
expect_gencode("${_sm90}" "-gencode=arch=compute_90a,code=sm_90a")
expect_gencode("${_sm100}" "-gencode=arch=compute_100a,code=sm_100a")
expect_gencode("${_fa2}"
  "-gencode=arch=compute_80,code=sm_80;-gencode=arch=compute_86,code=sm_86;-gencode=arch=compute_87,code=sm_87;-gencode=arch=compute_89,code=sm_89;-gencode=arch=compute_120a,code=sm_120a;-gencode=arch=compute_121a,code=sm_121a")

# --- SM70 / VOLTA (BACKEND-CUDA-SM070, W0) ----------------------------------
# The portable TUs must emit plain sm_70 SASS on request: no 'a' suffix exists
# for Volta, and the gencode formatter has to leave other targets alone when 70
# shares a (heterogeneous) fat build.
expect_gencode("70"
  "-gencode=arch=compute_70,code=sm_70")
expect_gencode("70;121a"
  "-gencode=arch=compute_70,code=sm_70;-gencode=arch=compute_121a,code=sm_121a")
# A 70-inclusive target list must NOT light any fast-path feature; the portable
# '70' arch coexists with the sm_12x features resolving for the sm_12x leg only.
set(VLLM_CPP_CUDA_ARCHITECTURES "70;121a")
vt_cuda_feature_archs(_fp4 "fp4-mma")
expect_gencode("${_fp4}" "-gencode=arch=compute_121a,code=sm_121a")
vt_cuda_feature_archs(_fa2 "fa2")
expect_gencode("${_fa2}" "-gencode=arch=compute_121a,code=sm_121a")

get_filename_component(_root "${_here}/.." ABSOLUTE)
file(READ "${_root}/CMakeLists.txt" _root_cmake)
foreach(_required IN ITEMS
    "set_property(TARGET vllm PROPERTY CUDA_ARCHITECTURES OFF)"
    "vt_cuda_set_source_gencode(\"\${VT_FP4_MMA_ARCHS}\""
    "src/vt/cuda/cuda_nvfp4_sm12x.cu"
    "vt_cuda_set_source_gencode(\"\${VT_CUTLASS_NVFP4_ARCHS}\""
    "vt_cuda_set_source_gencode(\"\${VT_CUTLASS_FP8_ARCHS}\""
    "vt_cuda_set_source_gencode(\"\${VT_SCALEDMM_C3X_SM90_ARCHS}\""
    "vt_cuda_set_source_gencode(\"\${VT_SCALEDMM_C3X_SM100_ARCHS}\""
    "vt_cuda_set_source_gencode(\"\${VT_MARLIN_NVFP4_ARCHS}\""
    "vt_cuda_set_source_gencode(\"\${VT_FA2_ARCHS}\""
    "vt_cuda_set_source_gencode(\"\${VLLM_CPP_CUDA_ARCHITECTURES}\"")
  string(FIND "${_root_cmake}" "${_required}" _position)
  if(_position EQUAL -1)
    message(FATAL_ERROR "missing W1 source-gencode wiring: ${_required}")
  endif()
endforeach()

file(READ "${_root}/src/vt/cuda/cuda_matmul_nvfp4.cu" _portable_nvfp4)
file(READ "${_root}/src/vt/cuda/cuda_nvfp4_sm12x.cu" _native_nvfp4)
string(FIND "${_portable_nvfp4}" "kind::mxf4nvf4" _portable_native_opcode)
string(FIND "${_native_nvfp4}" "kind::mxf4nvf4" _native_opcode)
if(NOT _portable_native_opcode EQUAL -1 OR _native_opcode EQUAL -1)
  message(FATAL_ERROR
    "sm12x native MMA must exist only in cuda_nvfp4_sm12x.cu")
endif()

message(STATUS "CUDA per-source gencode expectations: ALL PASS")
