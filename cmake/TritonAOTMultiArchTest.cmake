cmake_minimum_required(VERSION 3.24)

get_filename_component(_here "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
include("${_here}/TritonAOTMultiArch.cmake")

vt_triton_aot_available_arches(_arches)
if(NOT _arches STREQUAL "sm_80;sm_86;sm_89;sm_90a;sm_100a;sm_121a")
  message(FATAL_ERROR "unexpected W2 AOT tree order: [${_arches}]")
endif()

foreach(_case IN ITEMS
    "80=sm_80" "86=sm_86" "89=sm_89" "90a=sm_90a"
    "100a=sm_100a" "121a=sm_121a")
  string(REPLACE "=" ";" _parts "${_case}")
  list(GET _parts 0 _arch)
  list(GET _parts 1 _expected)
  vt_triton_aot_arch_tree(_actual "${_arch}")
  if(NOT _actual STREQUAL _expected)
    message(FATAL_ERROR "${_arch}: expected ${_expected}, got ${_actual}")
  endif()
endforeach()

foreach(_arch IN ITEMS 87 103a 110 120a)
  vt_triton_aot_arch_tree(_actual "${_arch}")
  if(_actual)
    message(FATAL_ERROR "${_arch}: unavailable tree must select fallback")
  endif()
endforeach()

vt_triton_aot_namespace_token(_token "sm_90a" "gdn_deltah_h48_default")
if(NOT _token STREQUAL "vt_aot_sm_90a_gdn_deltah_h48_default")
  message(FATAL_ERROR "unexpected namespaced token: ${_token}")
endif()

message(STATUS "Triton AOT multi-arch matrix: ALL PASS")
