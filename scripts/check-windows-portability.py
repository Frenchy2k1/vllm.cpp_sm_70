#!/usr/bin/env python3
"""Fail closed when the native Windows server contract regresses.

This is deliberately a source-contract gate on non-Windows hosts. Native MSVC
compile and runtime evidence are separate release gates; this checker prevents
the known POSIX-only or baseline-contaminating shapes from reaching them.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REQUIRED_CPP = (
    "src/vllm/entrypoints/openai/server_main.cpp",
    "src/vllm/platform/process.cpp",
    "src/vllm/platform/console_shutdown.cpp",
    "src/vllm/v1/kv_offload/lmcache/remote_client.cpp",
    "src/vllm/v1/kv_offload/fs_io.cpp",
)

POSIX_PATTERNS = (
    r"^\s*#\s*include\s*<(?:arpa/inet|netdb|netinet/[^>]+|sys/socket|sys/stat|sys/types|sys/wait|unistd|fcntl)\.h>",
    r"(?<![A-Za-z0-9_.>])(?:fork|execvp|waitpid|pipe|read|write|open|close|fsync|pread|pwrite|getpid|stat)\s*\(",
)


def _load_codemodel_sources(root: Path, build_dir: Path) -> set[str]:
    replies = build_dir / ".cmake/api/v1/reply"
    indexes = sorted(replies.glob("index-*.json"))
    if not indexes:
        raise RuntimeError(f"{build_dir}: CMake file-api codemodel reply is missing")
    index = json.loads(indexes[-1].read_text(encoding="utf-8"))
    reply = index.get("reply", {})
    codemodel_entry = next(
        (value for key, value in reply.items() if key.startswith("codemodel-v2")),
        None,
    )
    if not isinstance(codemodel_entry, dict) or "jsonFile" not in codemodel_entry:
        raise RuntimeError(f"{build_dir}: CMake codemodel-v2 reply is missing")
    codemodel = json.loads(
        (replies / codemodel_entry["jsonFile"]).read_text(encoding="utf-8")
    )
    configurations = codemodel.get("configurations", [])
    if not configurations:
        raise RuntimeError(f"{build_dir}: CMake codemodel has no configuration")
    targets: dict[str, dict] = {}
    target_files: dict[str, Path] = {}
    for target in configurations[0].get("targets", []):
        target_id = target.get("id")
        json_file = target.get("jsonFile")
        if target_id and json_file:
            targets[target_id] = target
            target_files[target_id] = replies / json_file
    roots = [target_id for target_id, value in targets.items()
             if value.get("name") == "server"]
    if len(roots) != 1:
        raise RuntimeError(
            f"{build_dir}: expected one shipped server target, found {len(roots)}"
        )
    pending = roots[:]
    seen: set[str] = set()
    sources: set[str] = set()
    while pending:
        target_id = pending.pop()
        if target_id in seen:
            continue
        seen.add(target_id)
        data = json.loads(target_files[target_id].read_text(encoding="utf-8"))
        for source in data.get("sources", []):
            path = Path(source.get("path", ""))
            absolute = path if path.is_absolute() else root / path
            try:
                sources.add(absolute.resolve().relative_to(root).as_posix())
            except ValueError:
                pass
        pending.extend(
            dependency["id"] for dependency in data.get("dependencies", [])
            if dependency.get("id") in targets
        )
    # CMake codemodel enumerates compiled translation units. Public/internal
    # project headers are part of those units but are not separate codemodel
    # entries, so include the whole shipped header surface conservatively.
    for pattern in ("include/vllm/**/*.h", "include/vllm/**/*.hpp"):
        sources.update(path.relative_to(root).as_posix() for path in root.glob(pattern))
    return sources


def shipped_server_sources(root: Path, build_dir: Path | None) -> set[str]:
    fixture = root / ".windows-portability-sources.json"
    if fixture.is_file():
        data = json.loads(fixture.read_text(encoding="utf-8"))
        return {str(item) for item in data.get("sources", [])}
    if build_dir is not None:
        return _load_codemodel_sources(root, build_dir.resolve())
    if shutil.which("cmake") is None:
        raise RuntimeError("cmake is required to derive the shipped-server source set")
    with tempfile.TemporaryDirectory(prefix="vllm-windows-codemodel-") as temp:
        generated = Path(temp)
        query = generated / ".cmake/api/v1/query"
        query.mkdir(parents=True)
        (query / "codemodel-v2").touch()
        command = [
            "cmake", "-S", str(root), "-B", str(generated), "-G", "Ninja",
            "-DVLLM_CPP_BUILD_TESTS=OFF", "-DVLLM_CPP_BUILD_EXAMPLES=ON",
            "-DVLLM_CPP_SERVER=ON", "-DVLLM_CPP_CUDA=OFF",
            "-DVLLM_CPP_HIP=OFF", "-DVLLM_CPP_METAL=OFF",
            "-DVLLM_CPP_MLX=OFF", "-DVLLM_CPP_TRITON=OFF",
            "-DVLLM_CPP_VULKAN=OFF", "-DCMAKE_BUILD_TYPE=Release",
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                "CMake configure failed while deriving shipped-server sources:\n" +
                result.stdout + result.stderr
            )
        return _load_codemodel_sources(root, generated)


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
        elif (re.match(r"#\s*if(?:n?def)?\b", stripped) and
              re.search(r"\b(?:__unix__|__APPLE__)\b", stripped) and
              not re.search(r"\b_WIN32\b", stripped)):
            stack.append((possible, True, False))
            possible = False
        elif re.match(r"#\s*if(?:n?def)?\b", stripped):
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


def without_cpp_comments(text: str) -> str:
    text = re.sub(r"(?s)/\*.*?\*/", "", text)
    return re.sub(r"//.*", "", text)


def _active_powershell(text: str) -> str:
    """Strip comments and literal `if ($false) { ... }` blocks locally.

    Native Windows additionally parses the file with PowerShell's AST below;
    this fallback keeps the Linux checker fail-closed for the mutations we own.
    """
    text = re.sub(r"(?s)<#.*?#>", "", text)
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    text = "\n".join(lines)
    return re.sub(r"(?is)if\s*\(\s*\$false\s*\)\s*\{.*?\}", "", text)


def _validate_powershell_ast(script: Path, errors: list[str]) -> None:
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        return
    parser = r'''
param([string]$Path)
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
  $Path, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -ne 0) {
  $parseErrors | ForEach-Object { [Console]::Error.WriteLine($_.Message) }
  exit 2
}
function Test-Dead([System.Management.Automation.Language.Ast]$Node) {
  $cursor = $Node
  while ($null -ne $cursor.Parent) {
    $parent = $cursor.Parent
    if ($parent -is [System.Management.Automation.Language.IfStatementAst]) {
      foreach ($clause in $parent.Clauses) {
        if ($clause.Item1.Extent.Text -match '^\s*\$false\s*$' -and
            $Node.Extent.StartOffset -ge $clause.Item2.Extent.StartOffset -and
            $Node.Extent.EndOffset -le $clause.Item2.Extent.EndOffset) {
          return $true
        }
      }
    }
    $cursor = $parent
  }
  return $false
}
$commands = @($ast.FindAll({
  param($node) $node -is [System.Management.Automation.Language.CommandAst]
}, $true) | Where-Object { -not (Test-Dead $_) } | ForEach-Object {
  [pscustomobject]@{ text = $_.Extent.Text; offset = $_.Extent.StartOffset }
})
$commands | ConvertTo-Json -Compress
'''
    result = subprocess.run(
        [pwsh, "-NoProfile", "-NonInteractive", "-Command", parser, "-Path", str(script)],
        text=True, capture_output=True, check=False,
    )
    if result.returncode != 0:
        errors.append("build-windows-release.ps1: PowerShell AST parse failed: " +
                      (result.stderr.strip() or result.stdout.strip()))
        return
    try:
        decoded = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as exc:
        errors.append(f"build-windows-release.ps1: PowerShell AST output invalid: {exc}")
        return
    if isinstance(decoded, dict):
        decoded = [decoded]
    command_text = "\n".join(item.get("text", "") for item in decoded)
    for description, pattern in (
        ("live --help smoke", r"(?s)Invoke-Checked\s+\$server.*?--help"),
        ("server smoke harness", r"(?s)Invoke-Checked\s+python.*?smokeHarness"),
        ("CRT audit", r"Invoke-CrtAudit"),
    ):
        if not re.search(pattern, command_text, re.IGNORECASE):
            errors.append(f"build-windows-release.ps1: AST missing active {description}")
    contract = subprocess.run(
        [pwsh, "-NoProfile", "-NonInteractive", "-File", str(script),
         "-SourceDir", str(script.parents[1]), "-ContractTest"],
        text=True, capture_output=True, check=False,
    )
    if contract.returncode != 0:
        errors.append("build-windows-release.ps1: injected fake-tool contract failed: " +
                      contract.stdout + contract.stderr)


def check(root: Path, build_dir: Path | None = None) -> list[str]:
    errors: list[str] = []
    try:
        source_paths = shipped_server_sources(root, build_dir)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        errors.append(f"shipped-server source discovery failed: {exc}")
        source_paths = set(REQUIRED_CPP)
    source_paths.update(REQUIRED_CPP)
    texts = {
        relative: require_file(root, relative, errors)
        for relative in sorted(source_paths)
        if Path(relative).suffix.lower() in {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"}
    }
    cmake = require_file(root, "CMakeLists.txt", errors)
    warnings_path = root / "cmake/CompilerWarnings.cmake"
    warnings = cmake
    if warnings_path.is_file():
        warnings += "\n" + warnings_path.read_text(encoding="utf-8")
    build_script = require_file(root, "scripts/build-windows-release.ps1", errors)
    cpu_baseline = require_file(root, "src/vt/cpu/cpu_matmul_elem.cpp", errors)

    for relative, source in texts.items():
        active_source = without_cpp_comments(source)
        for number, line in windows_possible_lines(active_source):
            full_source_posix = (
                re.search(POSIX_PATTERNS[0], line) or
                re.search(r"(?<![A-Za-z0-9_])::stat\s*\(", line) or
                re.search(r"\bS_IS(?:DIR|REG)\s*\(", line)
            )
            scoped_posix = (
                relative in REQUIRED_CPP and
                any(re.search(pattern, line) for pattern in POSIX_PATTERNS)
            )
            if full_source_posix or scoped_posix:
                errors.append(f"{relative}:{number}: unguarded POSIX include/call reaches Windows")
            if relative in REQUIRED_CPP and (
                    re.search(r"\b(?:CreateFileA|LoadLibraryA|MoveFileExA|DeleteFileA)\b", line) or
                    re.search(r"\.string\s*\(\)", line)):
                errors.append(f"{relative}:{number}: lossy Windows path conversion/API is forbidden")

    all_source = "\n".join(texts.values())
    if re.search(r"\b(?:system|popen|_popen|ShellExecute[AW]?)\s*\(", all_source) or \
            re.search(r"(?i)\bcmd(?:\.exe)?\b", all_source):
        errors.append("server process launch: shell invocation is forbidden; execute argv directly")

    runtime_values = re.findall(
        r"CMAKE_MSVC_RUNTIME_LIBRARY\s+(?:\"([^\"]+)\"|([^\s\)]+))", cmake
    )
    runtime_values = [quoted or bare for quoted, bare in runtime_values]
    if not runtime_values or any(
        value not in {"MultiThreaded", "MultiThreaded$<$<CONFIG:Debug>:Debug>"}
        for value in runtime_values
    ):
        errors.append("CMakeLists.txt: exact static MSVC runtime (/MT) is required")
    global_options = without_set_source_properties(cmake)
    if re.search(r"(?i)/arch\s*:\s*AVX2", global_options):
        errors.append("CMakeLists.txt: global /arch:AVX2 contaminates the portable baseline")
    if not all(token in warnings for token in ("/W4", "/WX")):
        errors.append("CMakeLists.txt: MSVC /W4 /WX policy is required")
    if re.search(r"__attribute__\s*\(\(\s*target\s*\(\s*\"f16c\"", cpu_baseline):
        errors.append("cpu_matmul_elem.cpp: F16C must be isolated in a dedicated translation unit")
    f16c_properties = source_properties(cmake, "src/vt/cpu/cpu_matmul_elem_f16c.cpp")
    if not all(token in f16c_properties for token in ("/arch:AVX", "-mf16c")):
        errors.append("CMakeLists.txt: dedicated F16C translation unit needs compiler-specific ISA flags")
    if re.search(r"(?i)/arch\s*:\s*AVX2", f16c_properties):
        errors.append("CMakeLists.txt: F16C translation unit must not require AVX2")

    for relative, source in texts.items():
        if "__builtin_clzll" in without_cpp_comments(source):
            errors.append(f"{relative}: non-portable compiler builtin reaches MSVC")

    server = "\n".join(texts[name] for name in REQUIRED_CPP[:3])
    for marker in ("CreateProcessW", "SetConsoleCtrlHandler"):
        if marker not in server:
            errors.append(f"server_main.cpp: required Win32 process/console support missing ({marker})")
    console = texts["src/vllm/platform/console_shutdown.cpp"]
    console_lifetime = (
        re.search(r"std::atomic\s*<[^>]+>\s+g?_?handler_acquisitions", console),
        re.search(r"std::atomic\s*<[^>]+>\s+in_flight", console),
        "SetEvent" in console,
        "WaitForSingleObject(impl_->win_state_->drained_event" in console,
    )
    if not all(console_lifetime):
        errors.append(
            "console_shutdown.cpp: stable event/in-flight handler lifetime protocol is required"
        )
    handler = re.search(
        r"(?s)BOOL\s+WINAPI\s+ConsoleControlHandler\s*\([^)]*\)\s*\{(.*?)\n\}",
        console,
    )
    dispatch = re.search(
        r"(?s)bool\s+DispatchControlEvent\s*\([^)]*\)\s*\{(.*?)\n\}",
        console,
    )
    handler_code = "\n".join(
        match.group(1) if match else "" for match in (handler, dispatch)
    )
    if not handler or not dispatch or re.search(
        r"RequestStop|std::mutex|condition_variable|stop_\s*\(", handler_code
    ):
        errors.append(
            "console_shutdown.cpp: OS handler may use only stable atomics and Win32 events"
        )

    lmcache = texts["src/vllm/v1/kv_offload/lmcache/remote_client.cpp"]
    if not all(marker in lmcache for marker in ("WSAStartup", "WSAGetLastError", "closesocket")):
        errors.append("remote_client.cpp: LMCache Winsock support is missing or silently disabled")
    if not re.search(r"(?s)if\s*\([^\)]*==\s*0\s*\).*?Close\s*\(\s*\).*?throw", lmcache):
        errors.append("remote_client.cpp: peer-close must invalidate the owned socket")

    fs_io = texts["src/vllm/v1/kv_offload/fs_io.cpp"]
    if not all(marker in fs_io for marker in ("CreateFileW", "FlushFileBuffers", "MoveFileExW")):
        errors.append("fs_io.cpp: Windows KV filesystem support is missing or silently disabled")
    if "#define NOMINMAX" not in fs_io:
        errors.append("fs_io.cpp: NOMINMAX must precede windows.h")
    if "CREATE_NEW" not in fs_io:
        errors.append("fs_io.cpp: exclusive temporary creation requires CREATE_NEW")
    if not all(flag in fs_io for flag in ("MOVEFILE_REPLACE_EXISTING", "MOVEFILE_WRITE_THROUGH")):
        errors.append("fs_io.cpp: atomic publish requires replace and MOVEFILE_WRITE_THROUGH")

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
    active_script = _active_powershell(build_script)
    for marker in required_script_markers:
        if marker not in active_script:
            errors.append(f"build-windows-release.ps1: required native CPU gate missing ({marker})")
    if not (re.search(r"(?i)\bdumpbin\b", active_script) and
            ('"/directives"' in active_script or
             re.search(r"(?i)\bdumpbin\s+/directives\b", active_script))):
        errors.append("build-windows-release.ps1: static-library CRT directive audit is missing")
    if not (re.search(r"(?i)\bdumpbin\b", active_script) and
            ('"/imports"' in active_script or
             re.search(r"(?i)\bdumpbin\s+/imports\b", active_script))):
        errors.append("build-windows-release.ps1: PE CRT import audit is missing")
    required_crt_rejections = (
        "VCRUNTIME", "MSVCP", "CONCRT", "UCRTBASE", "api-ms-win-crt-",
        "MSVCR", "LIBCMT",
    )
    if not all(token in active_script for token in required_crt_rejections):
        errors.append("build-windows-release.ps1: dynamic CRT rejection set is incomplete")
    if "--help" not in active_script:
        errors.append("build-windows-release.ps1: live --help smoke is missing")
    # Unit fixtures inject a generated source manifest and exercise the
    # cross-platform structural rules. Only the real repository script owns
    # the native AST/fake-tool execution contract.
    if not (root / ".windows-portability-sources.json").is_file():
        _validate_powershell_ast(root / "scripts/build-windows-release.ps1", errors)

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build-dir", type=Path)
    args = parser.parse_args()
    errors = check(args.root.resolve(), args.build_dir)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Windows portability contract OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
