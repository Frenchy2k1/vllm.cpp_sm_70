# W2 helpers for embedding all available Triton AOT cubin trees in one binary.
# Kept separate so the exact matrix and symbol namespace can be tested with
# `cmake -P` without enabling a compiler or CUDA.

set(VT_TRITON_AOT_AVAILABLE_ARCHES
  sm_80 sm_86 sm_89 sm_90a sm_100a sm_121a)

function(vt_triton_aot_available_arches OUT_VAR)
  set(${OUT_VAR} "${VT_TRITON_AOT_AVAILABLE_ARCHES}" PARENT_SCOPE)
endfunction()

# vt_triton_aot_computed_default(OUT_DEFAULT OUT_REASON CUDA_ENABLED
#                                VENDORED_DIR REGEN)
#
# The default value of VLLM_CPP_TRITON (BUILD-TRITON-DEFAULT-ON, #219). ON
# wherever the build can actually consume the vendored artifacts; otherwise OFF
# with a one-line reason naming the condition that declined it.
#
# The artifacts are pre-generated cubins embedded in plain C, so consuming them
# needs a C compiler and nothing else — the opt-in default guarded no
# dependency, it only made the shipped kernels easy to omit by accident. The
# gate profile has required `-DVLLM_CPP_TRITON=ON` by hand for that reason
# (.agents/environment.md, MANDATORY gate-build flags).
#
# NOT a condition: the number of CUDA architectures. The builder path embeds
# EVERY vendored tree (`_triton_aot_arch_names` in TritonAOT.cmake) and the
# runtime picks a cubin by exact SM, which is precisely what the shipped ten-SM
# release archive (scripts/build-linux-accelerator-release.sh) and the
# cuda-fat-gencode CI job (.github/workflows/ci.yml) already build with
# -DVLLM_CPP_TRITON=ON. The single-arch FATAL_ERROR in `_triton_aot_arch_name`
# guards the MAINTAINER regen flow, which is why REGEN below declines instead.
#
# Declining is always quiet: the default must never turn a configure that used
# to succeed into an error. An explicit -DVLLM_CPP_TRITON=ON keeps every
# existing diagnostic, unchanged.
function(vt_triton_aot_computed_default OUT_DEFAULT OUT_REASON CUDA_ENABLED
         VENDORED_DIR REGEN)
  if(NOT CUDA_ENABLED)
    set(${OUT_DEFAULT} OFF PARENT_SCOPE)
    set(${OUT_REASON}
      "the vendored artifacts are CUDA cubins and VLLM_CPP_CUDA is OFF"
      PARENT_SCOPE)
    return()
  endif()
  if(REGEN)
    set(${OUT_DEFAULT} OFF PARENT_SCOPE)
    set(${OUT_REASON}
      "VLLM_CPP_TRITON_REGEN is ON — maintainer regeneration pins ONE arch tree \
and runs the Python toolchain, so it is never selected for you"
      PARENT_SCOPE)
    return()
  endif()
  vt_triton_aot_available_arches(_arches)
  set(_missing "")
  foreach(_arch IN LISTS _arches)
    if(NOT EXISTS "${VENDORED_DIR}/${_arch}/MANIFEST")
      list(APPEND _missing "${_arch}")
    endif()
  endforeach()
  if(_missing)
    list(JOIN _missing " " _missing_text)
    set(${OUT_DEFAULT} OFF PARENT_SCOPE)
    set(${OUT_REASON}
      "vendored artifact tree(s) [${_missing_text}] are absent from ${VENDORED_DIR}"
      PARENT_SCOPE)
    return()
  endif()
  set(${OUT_DEFAULT} ON PARENT_SCOPE)
  set(${OUT_REASON} "" PARENT_SCOPE)
endfunction()

function(vt_triton_aot_arch_tree OUT_VAR ARCH)
  set(_tree "sm_${ARCH}")
  if(_tree IN_LIST VT_TRITON_AOT_AVAILABLE_ARCHES)
    set(${OUT_VAR} "${_tree}" PARENT_SCOPE)
  else()
    set(${OUT_VAR} "" PARENT_SCOPE)
  endif()
endfunction()

function(vt_triton_aot_namespace_token OUT_VAR ARCH TOKEN)
  if(NOT ARCH MATCHES "^sm_[0-9]+a?$")
    message(FATAL_ERROR "invalid Triton AOT namespace architecture '${ARCH}'")
  endif()
  if(NOT TOKEN MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
    message(FATAL_ERROR "invalid Triton AOT namespace token '${TOKEN}'")
  endif()
  set(${OUT_VAR} "vt_aot_${ARCH}_${TOKEN}" PARENT_SCOPE)
endfunction()

# Return one namespaced declaration without its trailing semicolon. CMake uses
# semicolons as list separators, so storing raw C declarations in a GLOBAL
# property can drop the terminator and concatenate two declarations into a
# function definition. The final header writer adds exactly one semicolon.
function(vt_triton_aot_namespace_declaration OUT_VAR ARCH TOKEN DECLARATION)
  vt_triton_aot_namespace_token(_namespaced "${ARCH}" "${TOKEN}")
  string(REPLACE "${TOKEN}" "${_namespaced}" _declaration "${DECLARATION}")
  # file(STRINGS) escapes a source semicolon as `\;` in a CMake list. Remove
  # both characters before this value enters another list/property.
  string(REGEX REPLACE "[\\\\;]+$" "" _declaration "${_declaration}")
  set(${OUT_VAR} "${_declaration}" PARENT_SCOPE)
endfunction()

# Prefix every generated external identifier containing BASE before compiling a
# vendored C launcher. This includes stable dispatchers, hash dispatchers,
# module/function globals, cubin arrays, and load/unload functions. The embedded
# kernel name string remains untouched, as required by cuModuleGetFunction.
function(vt_triton_aot_namespace_sources ARCH BASE OUTPUT_DIR OUT_VAR)
  set(_sources ${ARGN})
  if(NOT _sources)
    message(FATAL_ERROR "no Triton AOT sources to namespace for ${ARCH}/${BASE}")
  endif()
  file(MAKE_DIRECTORY "${OUTPUT_DIR}")
  set(_tokens)
  foreach(_source IN LISTS _sources)
    file(STRINGS "${_source}" _symbol_lines REGEX "${BASE}")
    string(JOIN "\n" _symbol_text ${_symbol_lines})
    string(REGEX MATCHALL
      "[A-Za-z_][A-Za-z0-9_]*${BASE}[A-Za-z0-9_]*"
      _prefixed_tokens "${_symbol_text}")
    string(REGEX MATCHALL "${BASE}[A-Za-z0-9_]*"
      _base_tokens "${_symbol_text}")
    list(APPEND _tokens ${_prefixed_tokens} ${_base_tokens})
  endforeach()
  list(REMOVE_DUPLICATES _tokens)
  list(SORT _tokens)
  if(NOT _tokens)
    message(FATAL_ERROR "no generated symbols found for ${ARCH}/${BASE}")
  endif()

  set(_header "${OUTPUT_DIR}/${ARCH}_${BASE}_namespace.h")
  file(WRITE "${_header}"
    "/* Generated by TritonAOTMultiArch.cmake: collision-free W2 symbols. */\n")
  foreach(_token IN LISTS _tokens)
    vt_triton_aot_namespace_token(_namespaced "${ARCH}" "${_token}")
    file(APPEND "${_header}" "#define ${_token} ${_namespaced}\n")
  endforeach()
  set_property(SOURCE ${_sources} APPEND PROPERTY
    COMPILE_OPTIONS "-include${_header}")
  set(${OUT_VAR} "${_sources}" PARENT_SCOPE)
endfunction()
