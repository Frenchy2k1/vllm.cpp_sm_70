# Refuse an in-source configure BEFORE anything else in the build can run.
#
# WHY THIS EXISTS (issue #85). `cmake .` in the checkout configures and builds
# most of the way, then fails in the LINKER with a message that names neither
# the cause nor the cure:
#
#   /usr/bin/ld: cannot open output file tokenize: Is a directory
#   /usr/bin/ld: cannot open output file dequant_nvfp4: Is a directory
#   /usr/bin/ld: cannot open output file dump_container: Is a directory
#
# Every example is a DIRECTORY under examples/ whose name is also the name of
# the executable target built from it (examples/tokenize -> target `tokenize`,
# examples/dequant_nvfp4 -> `dequant_nvfp4`, ...). In an in-source build the
# binary directory IS the source directory, so CMake asks the linker to write
# each executable exactly on top of the source directory it was built from, and
# `ld` reports the collision one target at a time with no mention of `cmake`.
#
# The failure is therefore neither the reporter's toolchain nor a missing
# dependency: it is structural, it is reproducible on every platform, and it is
# only ever fixed by configuring out-of-source. Diagnose it at configure time,
# where the fix can actually be stated.
#
# Kept as a module rather than inlined into CMakeLists.txt so the predicate is
# testable with `cmake -P cmake/InSourceGuardTest.cmake` — no compiler, no
# toolkit, no writes to the tree (the same shape as CudaArchFeaturesTest.cmake).

# vllm_cpp_forbid_in_source_build(<source-dir> <binary-dir>)
#   FATAL_ERROR when the two resolve to the same directory. Symlinks are
#   resolved first: a checkout reached through a symlinked path (or a build dir
#   symlinked back into it) is the same in-source build and fails the same way,
#   but a plain STREQUAL on the unresolved strings would not see it.
function(vllm_cpp_forbid_in_source_build source_dir binary_dir)
  get_filename_component(_src "${source_dir}" REALPATH)
  get_filename_component(_bin "${binary_dir}" REALPATH)
  if(NOT _src STREQUAL _bin)
    return()
  endif()
  # ONE argument: message() re-wraps and indents each argument separately, so a
  # per-line argument list renders with a blank line between every line.
  message(FATAL_ERROR
    "vllm.cpp does not support in-source builds. The example target names are "
    "also source directory names (examples/tokenize, examples/dequant_nvfp4, "
    "examples/dump_container, ...), so an in-source build asks the linker to "
    "write each executable on top of the directory it was built from, and "
    "fails late as 'cannot open output file <name>: Is a directory'."
    "\n\nConfigure into a separate build directory instead:"
    "\n    cmake -S . -B build"
    "\n    cmake --build build -j"
    "\n\nThen remove what this attempt already wrote into the source tree:"
    "\n    rm -rf CMakeCache.txt CMakeFiles")
endfunction()
