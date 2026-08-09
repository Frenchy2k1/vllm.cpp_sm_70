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

vt_triton_aot_namespace_declaration(
  _declaration "sm_100a" "gdn_chunko_bf16_h48_default"
  "CUresult gdn_chunko_bf16_h48_default(CUstream stream);")
if(NOT _declaration STREQUAL
    "CUresult vt_aot_sm_100a_gdn_chunko_bf16_h48_default(CUstream stream)")
  message(FATAL_ERROR "unexpected normalized declaration: ${_declaration}")
endif()

# Exercise the same GLOBAL/list/write shape that lost a semicolon in the first
# hosted fat build. Every generated prototype must remain a declaration instead
# of running into the next line as a malformed function definition.
get_filename_component(_root "${_here}/.." ABSOLUTE)
set(_declarations)
foreach(_arch IN LISTS _arches)
  file(STRINGS
    "${_root}/src/vt/cuda/triton_aot_vendored/${_arch}/gdn_chunko_bf16_h48.h"
    _raw REGEX "^CUresult gdn_chunko_bf16_h48_default\\(")
  vt_triton_aot_namespace_declaration(
    _normalized "${_arch}" "gdn_chunko_bf16_h48_default" "${_raw}")
  list(APPEND _declarations "${_normalized}")
endforeach()
list(REMOVE_DUPLICATES _declarations)
list(SORT _declarations)
set(_generated "${CMAKE_CURRENT_BINARY_DIR}/triton-aot-declaration-test.h")
file(WRITE "${_generated}" "")
foreach(_decl IN LISTS _declarations)
  file(APPEND "${_generated}" "${_decl};\n")
endforeach()
file(STRINGS "${_generated}" _generated_lines)
list(LENGTH _generated_lines _generated_count)
if(NOT _generated_count EQUAL 6)
  message(FATAL_ERROR "expected six declarations, got ${_generated_count}")
endif()
foreach(_line IN LISTS _generated_lines)
  if(NOT _line MATCHES ";$")
    message(FATAL_ERROR "generated declaration lost semicolon: ${_line}")
  endif()
endforeach()
file(REMOVE "${_generated}")

message(STATUS "Triton AOT multi-arch matrix: ALL PASS")
