cmake_minimum_required(VERSION 3.24)

# BUILD-TRITON-DEFAULT-ON (#219) — the resolved default of VLLM_CPP_TRITON.
#
# `vt_triton_aot_computed_default` is the whole rule, kept pure so this matrix
# runs under `cmake -P` with no compiler, no CUDA and no GPU (exactly like
# TritonAOTMultiArchTest.cmake). The configure-level cells that this cannot
# reach — a real CUDA configure resolving the option, and the fat build still
# configuring clean — are proven on the gate host and recorded in the row's
# spec.

get_filename_component(_here "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
get_filename_component(_root "${_here}/.." ABSOLUTE)
include("${_here}/TritonAOTMultiArch.cmake")

set(_vendored "${_root}/src/vt/cuda/triton_aot_vendored")
set(_scratch "${CMAKE_CURRENT_BINARY_DIR}/triton-aot-default-test")

function(_expect CASE EXPECTED_DEFAULT EXPECTED_REASON_SUBSTRING
         CUDA_ENABLED VENDORED_DIR REGEN)
  vt_triton_aot_computed_default(_default _reason
    "${CUDA_ENABLED}" "${VENDORED_DIR}" "${REGEN}")
  if(NOT _default STREQUAL EXPECTED_DEFAULT)
    message(FATAL_ERROR
      "${CASE}: expected default ${EXPECTED_DEFAULT}, got '${_default}'")
  endif()
  if(EXPECTED_REASON_SUBSTRING STREQUAL "")
    if(NOT _reason STREQUAL "")
      message(FATAL_ERROR
        "${CASE}: an enabling cell must carry no decline reason, got '${_reason}'")
    endif()
  else()
    if(_reason STREQUAL "")
      message(FATAL_ERROR
        "${CASE}: a declining cell must name the condition that declined it")
    endif()
    string(FIND "${_reason}" "${EXPECTED_REASON_SUBSTRING}" _pos)
    if(_pos LESS 0)
      message(FATAL_ERROR
        "${CASE}: decline reason must mention '${EXPECTED_REASON_SUBSTRING}', "
        "got '${_reason}'")
    endif()
  endif()
endfunction()

# 1. The vendored artifacts are CUDA cubins. Every non-CUDA backend (CPU,
#    Metal/MSL, Vulkan, ROCm) resolves OFF and must keep configuring clean.
_expect("cuda-off" OFF "VLLM_CPP_CUDA is OFF" OFF "${_vendored}" OFF)

# 2. A CUDA build with the vendored trees present is the case this row exists
#    for: the artifacts are pre-generated cubins embedded in plain C, so
#    consuming them needs a C compiler and nothing else.
_expect("cuda-vendored-present" ON "" ON "${_vendored}" OFF)

# 3. The arch COUNT is deliberately not a condition. The builder path embeds
#    every vendored tree (TritonAOT.cmake `_triton_aot_arch_names`) and
#    dispatches by exact SM at runtime, which is what the shipped ten-SM
#    release archive and the cuda-fat-gencode CI job already build with
#    -DVLLM_CPP_TRITON=ON. The default therefore cannot depend on
#    VLLM_CPP_CUDA_ARCHITECTURES, and the function takes no such argument;
#    this cell pins that the single-arch and fat answers are the same object.
_expect("cuda-fat-is-the-same-answer" ON "" ON "${_vendored}" OFF)

# 4. Maintainer regeneration pins ONE arch tree and runs the Python toolchain.
#    It is an explicit act, so the default never selects it for you.
_expect("regen" OFF "VLLM_CPP_TRITON_REGEN" ON "${_vendored}" ON)

# 5. A tree that is not there cannot be consumed: the builder path would
#    FATAL_ERROR in `_triton_aot_arch_names`. The default declines instead, so
#    adding it can never turn a configure that used to succeed into an error.
file(REMOVE_RECURSE "${_scratch}")
file(MAKE_DIRECTORY "${_scratch}/empty")
_expect("no-vendored-tree" OFF "sm_121a" ON "${_scratch}/empty" OFF)

vt_triton_aot_available_arches(_arches)
list(LENGTH _arches _n_arches)
if(NOT _n_arches EQUAL 6)
  message(FATAL_ERROR "expected six vendored trees, got ${_n_arches}")
endif()
file(MAKE_DIRECTORY "${_scratch}/partial")
foreach(_arch IN LISTS _arches)
  if(NOT _arch STREQUAL "sm_90a")
    file(MAKE_DIRECTORY "${_scratch}/partial/${_arch}")
    file(WRITE "${_scratch}/partial/${_arch}/MANIFEST" "arch ${_arch}\n")
  endif()
endforeach()
_expect("one-missing-tree" OFF "sm_90a" ON "${_scratch}/partial" OFF)
file(REMOVE_RECURSE "${_scratch}")

message(STATUS "Triton AOT default matrix: ALL PASS")
