#!/usr/bin/env python3
"""Fail closed when the native Windows server contract regresses.

This is deliberately a source-contract gate on non-Windows hosts. Native MSVC
compile and runtime evidence are separate release gates; this checker prevents
the known POSIX-only or baseline-contaminating shapes from reaching them.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


SCOPED_CPP = (
    "src/vllm/entrypoints/openai/server_main.cpp",
    "src/vllm/platform/process.cpp",
    "src/vllm/platform/console_shutdown.cpp",
    "src/vllm/v1/kv_offload/lmcache/remote_client.cpp",
    "src/vllm/v1/kv_offload/fs_io.cpp",
)

POSIX_PATTERNS = (
    r"^\s*#\s*include\s*<(?:arpa/inet|netdb|netinet/[^>]+|sys/socket|sys/types|sys/wait|unistd|fcntl)\.h>",
    r"(?<![A-Za-z0-9_.>])(?:fork|execvp|waitpid|pipe|read|write|open|close|fsync|pread|pwrite|getpid)\s*\(",
)


def windows_possible_lines(text: str):
    """Yield (line number, line) for branches that can compile on Windows."""
    possible = True
    # (parent possible, condition known, condition possible on Windows)
    stack: list[tuple[bool, bool, bool]] = []
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if re.match(r"#\s*ifdef\s+_WIN32\b", stripped):
            stack.append((possible, True, True))
            possible = possible and True
        elif re.match(r"#\s*ifndef\s+_WIN32\b", stripped):
            stack.append((possible, True, False))
            possible = False
        elif re.match(r"#\s*if\s+defined\s*\(\s*_WIN32\s*\)", stripped):
            stack.append((possible, True, True))
            possible = possible and True
        elif re.match(r"#\s*if\b.*!\s*defined\s*\(\s*_WIN32\s*\)", stripped):
            stack.append((possible, True, False))
            possible = False
        elif re.match(r"#\s*if\b", stripped):
            stack.append((possible, False, possible))
        elif re.match(r"#\s*else\b", stripped) and stack:
            parent, known, condition = stack[-1]
            possible = parent and (not condition if known else True)
        elif re.match(r"#\s*elif\b", stripped) and stack:
            parent, _, _ = stack[-1]
            possible = parent
        elif re.match(r"#\s*endif\b", stripped) and stack:
            parent, _, _ = stack.pop()
            possible = parent
        elif possible:
            yield number, line


def without_set_source_properties(text: str) -> str:
    """Remove balanced set_source_files_properties commands."""
    lowered = text.lower()
    needle = "set_source_files_properties("
    out = list(text)
    start = 0
    while True:
        at = lowered.find(needle, start)
        if at < 0:
            break
        depth = 0
        end = at
        while end < len(text):
            if text[end] == "(":
                depth += 1
            elif text[end] == ")":
                depth -= 1
                if depth == 0:
                    end += 1
                    break
            end += 1
        out[at:end] = " " * (end - at)
        start = end
    return "".join(out)


def source_properties(text: str, source: str) -> str:
    """Return the balanced set_source_files_properties command for source."""
    lowered = text.lower()
    needle = "set_source_files_properties("
    start = 0
    while True:
        at = lowered.find(needle, start)
        if at < 0:
            return ""
        depth = 0
        end = at
        while end < len(text):
            if text[end] == "(":
                depth += 1
            elif text[end] == ")":
                depth -= 1
                if depth == 0:
                    end += 1
                    break
            end += 1
        command = text[at:end]
        if source in command:
            return command
        start = end


def require_file(root: Path, relative: str, errors: list[str]) -> str:
    path = root / relative
    if not path.is_file():
        errors.append(f"{relative}: required Windows portability surface is missing")
        return ""
    return path.read_text(encoding="utf-8")


def check(root: Path) -> list[str]:
    errors: list[str] = []
    texts = {relative: require_file(root, relative, errors) for relative in SCOPED_CPP}
    cmake = require_file(root, "CMakeLists.txt", errors)
    warnings_path = root / "cmake/CompilerWarnings.cmake"
    warnings = cmake
    if warnings_path.is_file():
        warnings += "\n" + warnings_path.read_text(encoding="utf-8")
    build_script = require_file(root, "scripts/build-windows-release.ps1", errors)
    cpu_baseline = require_file(root, "src/vt/cpu/cpu_matmul_elem.cpp", errors)

    for relative, source in texts.items():
        for number, line in windows_possible_lines(source):
            if any(re.search(pattern, line) for pattern in POSIX_PATTERNS):
                errors.append(f"{relative}:{number}: unguarded POSIX include/call reaches Windows")
            if re.search(r"\b(?:CreateFileA|LoadLibraryA|MoveFileExA|DeleteFileA)\b", line) or \
                    re.search(r"\.string\s*\(\)", line):
                errors.append(f"{relative}:{number}: lossy Windows path conversion/API is forbidden")

    all_source = "\n".join(texts.values())
    if re.search(r"\b(?:system|popen|_popen|ShellExecute[AW]?)\s*\(", all_source) or \
            re.search(r"(?i)\bcmd(?:\.exe)?\b", all_source):
        errors.append("server process launch: shell invocation is forbidden; execute argv directly")

    if not re.search(r"CMAKE_MSVC_RUNTIME_LIBRARY[^\n]*MultiThreaded", cmake):
        errors.append("CMakeLists.txt: static MSVC runtime (/MT) is required")
    global_options = without_set_source_properties(cmake)
    if re.search(r"(?i)/arch\s*:\s*AVX2", global_options):
        errors.append("CMakeLists.txt: global /arch:AVX2 contaminates the portable baseline")
    if not all(token in warnings for token in ("/W4", "/WX")):
        errors.append("CMakeLists.txt: MSVC /W4 /WX policy is required")
    if re.search(r"__attribute__\s*\(\(\s*target\s*\(\s*\"f16c\"", cpu_baseline):
        errors.append("cpu_matmul_elem.cpp: F16C must be isolated in a dedicated translation unit")
    f16c_properties = source_properties(cmake, "src/vt/cpu/cpu_matmul_elem_f16c.cpp")
    if not all(token in f16c_properties for token in ("/arch:AVX2", "-mf16c")):
        errors.append("CMakeLists.txt: dedicated F16C translation unit needs compiler-specific ISA flags")

    server = "\n".join(texts[name] for name in SCOPED_CPP[:3])
    for marker in ("CreateProcessW", "SetConsoleCtrlHandler"):
        if marker not in server:
            errors.append(f"server_main.cpp: required Win32 process/console support missing ({marker})")

    lmcache = texts["src/vllm/v1/kv_offload/lmcache/remote_client.cpp"]
    if not all(marker in lmcache for marker in ("WSAStartup", "WSAGetLastError", "closesocket")):
        errors.append("remote_client.cpp: LMCache Winsock support is missing or silently disabled")

    fs_io = texts["src/vllm/v1/kv_offload/fs_io.cpp"]
    if not all(marker in fs_io for marker in ("CreateFileW", "FlushFileBuffers", "MoveFileExW")):
        errors.append("fs_io.cpp: Windows KV filesystem support is missing or silently disabled")

    required_script_markers = (
        "Visual Studio 17 2022",
        "Release",
        "bin/vllm-server.exe",
        "--help",
        "/health",
        "/version",
        "CTRL_BREAK_EVENT",
        "VT_CPU_MATMUL_TIER",
        '"portable"',
        '"avx2"',
        '"amx"',
    )
    for marker in required_script_markers:
        if marker not in build_script:
            errors.append(f"build-windows-release.ps1: required native CPU gate missing ({marker})")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    errors = check(args.root.resolve())
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Windows portability contract OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
