# Configure-tier test for the in-source build guard (issue #85).
#
# Run standalone, with no compiler, no toolkit and no writes to the tree:
#   cmake -P cmake/InSourceGuardTest.cmake
# (CI runs it next to CudaArchFeaturesTest.cmake.)
#
# WHY THIS EXISTS. The guard's whole job is to turn a late, misattributed
# linker error into an early, actionable one, so the thing that must be pinned
# is exactly WHEN it fires. Two failure modes matter and neither is visible by
# reading it: firing on a legitimate out-of-source build would break every
# working build overnight, and NOT firing on a non-normalised path that resolves
# to the source directory would let the original `ld: Is a directory` failure
# straight back through. A plain STREQUAL on the unresolved strings passes the
# obvious case and misses the second one, which is why it is a case below.
#
# vllm_cpp_forbid_in_source_build() raises FATAL_ERROR, which cannot be caught
# in-process, so each case runs the guard in a CHILD `cmake -P` (this same file
# in PROBE mode) and asserts on the child's exit status.
cmake_minimum_required(VERSION 3.24)

get_filename_component(_here "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
include("${_here}/InSourceGuard.cmake")

# --- PROBE mode -------------------------------------------------------------
# Invoked by the parent below. Calls the guard and nothing else: exit 0 means
# the guard allowed the build, non-zero means it refused it.
if(DEFINED PROBE_SRC AND DEFINED PROBE_BIN)
  vllm_cpp_forbid_in_source_build("${PROBE_SRC}" "${PROBE_BIN}")
  return()
endif()

# --- test driver ------------------------------------------------------------
get_filename_component(_repo "${_here}/.." ABSOLUTE)
set(_failures 0)

# expect_guard(<expectation> <label> <source-dir> <binary-dir>)
#   <expectation> is REFUSES or ALLOWS.
function(expect_guard EXPECTATION LABEL SRC BIN)
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
            "-DPROBE_SRC=${SRC}" "-DPROBE_BIN=${BIN}"
            -P "${CMAKE_CURRENT_LIST_FILE}"
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err)
  if(EXPECTATION STREQUAL "REFUSES")
    if(_rc EQUAL 0)
      message(SEND_ERROR
        "IN-SOURCE GUARD DID NOT FIRE for ${LABEL}\n"
        "  source: ${SRC}\n  binary: ${BIN}\n"
        "This is the issue #85 build: it would fail later in the linker as "
        "'cannot open output file <target>: Is a directory'.")
      math(EXPR _failures "${_failures} + 1")
      set(_failures "${_failures}" PARENT_SCOPE)
      return()
    endif()
    # Refusing is necessary but not sufficient: the message is the deliverable,
    # so pin that it actually tells the reporter what to run instead.
    if(NOT _err MATCHES "cmake -S \\. -B build")
      message(SEND_ERROR
        "IN-SOURCE GUARD FIRED for ${LABEL} but its message does not carry the "
        "out-of-source command. Got:\n${_err}")
      math(EXPR _failures "${_failures} + 1")
      set(_failures "${_failures}" PARENT_SCOPE)
      return()
    endif()
    message(STATUS "ok  refuses  ${LABEL}")
  else()
    if(NOT _rc EQUAL 0)
      message(SEND_ERROR
        "IN-SOURCE GUARD FIRED ON A LEGITIMATE OUT-OF-SOURCE BUILD for ${LABEL}\n"
        "  source: ${SRC}\n  binary: ${BIN}\n${_err}")
      math(EXPR _failures "${_failures} + 1")
      set(_failures "${_failures}" PARENT_SCOPE)
      return()
    endif()
    message(STATUS "ok  allows   ${LABEL}")
  endif()
endfunction()

# The reported build: `cmake .` at the repository root.
expect_guard(REFUSES "identical source and binary directory" "${_repo}" "${_repo}")

# The case a STREQUAL on the raw strings gets WRONG: textually different paths
# that resolve to the same directory. `cmake -S . -B ./cmake/..` is the same
# in-source build and fails identically.
expect_guard(REFUSES "non-normalised path resolving to the source directory"
  "${_repo}" "${_repo}/cmake/..")

# The documented builds, which must stay untouched.
expect_guard(ALLOWS "in-tree build subdirectory"     "${_repo}" "${_repo}/build")
expect_guard(ALLOWS "nested build subdirectory"      "${_repo}" "${_repo}/build/cuda")
expect_guard(ALLOWS "build directory outside the tree" "${_repo}" "${_repo}/../vllm-cpp-build")
# A directory whose name merely STARTS with the source path is not the source
# path; the guard must compare directories, not string prefixes.
expect_guard(ALLOWS "sibling sharing the source-dir prefix" "${_repo}" "${_repo}-build")

if(_failures GREATER 0)
  message(FATAL_ERROR "in-source guard: ${_failures} case(s) failed")
endif()
message(STATUS "in-source guard: all cases passed")
