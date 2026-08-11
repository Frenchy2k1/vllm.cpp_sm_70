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


CPP_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
UNSUPPORTED_TIER_FILTER = (
    "--test-case=elementwise CPU GEMM: the forced tier is the tier that actually ran"
)
UNSUPPORTED_TIER_DIAGNOSTIC = "unknown x86 ISA tier 'amx'"


def _project_header_closure(root: Path, sources: set[str],
                            include_roots: set[Path]) -> set[str]:
    """Resolve every project-local quoted include reachable from sources."""
    closure = set(sources)
    pending = list(sources)
    include_re = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)
    while pending:
        relative = pending.pop()
        source = root / relative
        if not source.is_file() or source.suffix.lower() not in CPP_SUFFIXES:
            continue
        for included in include_re.findall(source.read_text(encoding="utf-8")):
            candidates = [source.parent / included]
            candidates.extend(directory / included for directory in include_roots)
            for candidate in candidates:
                try:
                    resolved = candidate.resolve()
                    found = resolved.relative_to(root).as_posix()
                except (OSError, ValueError):
                    continue
                if resolved.is_file() and found not in closure:
                    closure.add(found)
                    pending.append(found)
                    break
    return closure


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
    include_roots = {root, root / "include", root / "src"}
    while pending:
        target_id = pending.pop()
        if target_id in seen:
            continue
        seen.add(target_id)
        data = json.loads(target_files[target_id].read_text(encoding="utf-8"))
        for group in data.get("compileGroups", []):
            for include in group.get("includes", []):
                path = Path(include.get("path", ""))
                absolute = path if path.is_absolute() else root / path
                try:
                    resolved = absolute.resolve()
                    resolved.relative_to(root)
                except (OSError, ValueError):
                    continue
                include_roots.add(resolved)
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
    return _project_header_closure(root, sources, include_roots)


def shipped_server_sources(root: Path, build_dir: Path | None,
                           source_manifest: Path | None = None) -> set[str]:
    if source_manifest is not None:
        data = json.loads(source_manifest.read_text(encoding="utf-8"))
        sources = {str(item) for item in data.get("sources", [])}
        return _project_header_closure(
            root, sources, {root, root / "include", root / "src"}
        )
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
        elif (re.match(r"#\s*if\b", stripped) and
              re.search(r"\b(?:__GNUC__|__clang__)\b", stripped) and
              not re.search(r"\b_MSC_VER\b", stripped)):
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
            if (re.search(r"\b(?:__GNUC__|__clang__)\b", stripped) and
                    not re.search(r"\b_MSC_VER\b", stripped)):
                possible = False
            else:
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


_CPP_INERT = re.compile(
    r'''(?P<block>/\*.*?\*/)|'''
    r'''(?P<line>//[^\n]*)|'''
    r'''(?P<raw>R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\(.*?\)(?P=delimiter)")|'''
    r'''(?P<string>"(?:\\.|[^"\\])*")|'''
    r'''(?P<char>'(?:\\.|[^'\\])*')''',
    re.DOTALL,
)


def without_cpp_comments_and_literals(text: str) -> str:
    """Blank inert C++ text while preserving offsets and line boundaries."""
    return _CPP_INERT.sub(
        lambda match: "".join("\n" if char == "\n" else " "
                              for char in match.group(0)),
        text,
    )


def _active_powershell(text: str) -> str:
    """Strip comments and literal `if ($false) { ... }` blocks locally.

    Native Windows additionally parses the file with PowerShell's AST below;
    this fallback keeps the Linux checker fail-closed for the mutations we own.
    """
    text = re.sub(r"(?s)<#.*?#>", "", text)
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    text = "\n".join(lines)
    text = re.sub(r"(?is)if\s*\(\s*\$false\s*\)\s*\{.*?\}", "", text)
    return re.sub(r"(?is)if\s*\(\s*\$ContractTest\s*\)\s*\{.*?\}", "", text)


def _ordered_matches(text: str, stages: tuple[tuple[str, str], ...],
                     errors: list[str], label: str) -> None:
    offsets: list[int] = []
    for description, pattern in stages:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if match is None:
            errors.append(
                f"build-windows-release.ps1: missing active {description} in {label}"
            )
            return
        offsets.append(match.start())
    if offsets != sorted(offsets) or len(offsets) != len(set(offsets)):
        errors.append(
            "build-windows-release.ps1: native gate order must be "
            "configure/codemodel -> checker -> build/focused tests -> install -> "
            "CRT audit -> help/tier/server smokes"
        )


def _validate_powershell_ast_order(commands: list[dict],
                                   errors: list[str]) -> None:
    stages = (
        ("codemodel query", r"\bNew-Item\b.*codemodel-v2"),
        ("configure", r"\bInvoke-Checked\s+cmake\b.*(?:\"-S\"|\s-S\s)"),
        ("portability checker", r"\bInvoke-Checked\s+python\b.*check-windows-portability\.py"),
        ("build", r"\bInvoke-Checked\s+cmake\b.*\"--build\""),
        ("focused tests", r"\bInvoke-Checked\b.*tests[/\\]Release[/\\]\$test"),
        ("install", r"\bInvoke-Checked\s+cmake\b.*\"--install\""),
        ("CRT audit", r"\bInvoke-CrtAudit\b"),
        ("live --help smoke", r"\bInvoke-Checked\s+\$server\b.*--help"),
        ("forced-tier smoke", r"\bInvoke-UnsupportedTierProbe\b.*\$tierTest\b"),
        ("server smoke harness", r"\bInvoke-Checked\s+python\b.*\$smokeHarness"),
    )
    offsets: list[int] = []
    for description, pattern in stages:
        candidates = [
            int(item.get("offset", -1)) for item in commands
            if re.search(pattern, item.get("text", ""), re.IGNORECASE | re.DOTALL)
        ]
        if not candidates:
            errors.append(
                f"build-windows-release.ps1: missing active {description} in PowerShell AST"
            )
            return
        offsets.append(min(candidates))
    if offsets != sorted(offsets) or len(offsets) != len(set(offsets)):
        errors.append(
            "build-windows-release.ps1: native gate order must be "
            "configure/codemodel -> checker -> build/focused tests -> install -> "
            "CRT audit -> help/tier/server smokes"
        )


def _validate_powershell_source_order(text: str, errors: list[str]) -> None:
    active = _active_powershell(text)
    pipeline = active.find("codemodel-v2")
    if pipeline >= 0:
        active = active[pipeline:]
    stages = (
        ("codemodel query", r"codemodel-v2"),
        ("configure", r"(?:^\s*cmake\s+-S\b|^\s*\"-S\"\s*,\s*\$SourceDir)"),
        ("portability checker", r"check-windows-portability\.py"),
        ("build", r"(?:^\s*cmake\s+--build\b|\"--build\"\s*,\s*\$BuildDir)"),
        ("focused tests", r"(?:^\s*foreach\s*\(\s*\$test\b|^\s*&\s+\"\$BuildDir/tests/test_openai_api_server\.exe\")"),
        ("install", r"(?:^\s*cmake\s+--install\b|\"--install\"\s*,\s*\$BuildDir)"),
        ("CRT audit", r"^\s*Invoke-CrtAudit\b"),
        ("live --help smoke", r"^\s*Invoke-Checked\s+\$server\b[^\n]*--help"),
        ("forced-tier smoke", r"^\s*Invoke-UnsupportedTierProbe\b[^\n]*\$tierTest\b"),
        ("server smoke harness", r"^\s*Invoke-Checked\s+python\b[^\n]*\$smokeHarness"),
    )
    _ordered_matches(active, stages, errors, "source contract")


def _cpp_braced_body(source: str, opening: int) -> tuple[str, int] | None:
    """Return a balanced braced body and its closing-brace offset."""
    if opening < 0 or opening >= len(source) or source[opening] != "{":
        return None
    depth = 0
    for offset in range(opening, len(source)):
        if source[offset] == "{":
            depth += 1
        elif source[offset] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1:offset], offset
    return None


def _cpp_function_body(source: str, signature: str) -> str:
    active = without_cpp_comments_and_literals(source)
    search_from = 0
    while True:
        match = re.search(signature, active[search_from:])
        if match is None:
            return ""
        match_start = search_from + match.start()
        match_end = search_from + match.end()
        parameters_open = active.rfind("(", match_start, match_end)
        if parameters_open < 0:
            return ""

        depth = 0
        parameters_close = -1
        for offset in range(parameters_open, len(active)):
            if active[offset] == "(":
                depth += 1
            elif active[offset] == ")":
                depth -= 1
                if depth == 0:
                    parameters_close = offset
                    break
        if parameters_close < 0:
            return ""

        suffix_depth: list[str] = []
        declaration = False
        pairs = {"(": ")", "[": "]"}
        offset = parameters_close + 1
        while offset < len(active):
            char = active[offset]
            requires = re.match(r"requires\b", active[offset:])
            if not suffix_depth and requires is not None:
                expression = offset + requires.end()
                while expression < len(active) and active[expression].isspace():
                    expression += 1
                if expression < len(active) and active[expression] == "(":
                    depth = 0
                    for cursor in range(expression, len(active)):
                        if active[cursor] == "(":
                            depth += 1
                        elif active[cursor] == ")":
                            depth -= 1
                            if depth == 0:
                                expression = cursor + 1
                                break
                    else:
                        return ""
                    while expression < len(active) and active[expression].isspace():
                        expression += 1
                if expression < len(active) and active[expression] == "{":
                    requires_body = _cpp_braced_body(active, expression)
                    if requires_body is None:
                        return ""
                    offset = requires_body[1] + 1
                    continue
            if char in pairs:
                suffix_depth.append(pairs[char])
            elif suffix_depth and char == suffix_depth[-1]:
                suffix_depth.pop()
            elif not suffix_depth and char == ";":
                declaration = True
                search_from = offset + 1
                break
            elif not suffix_depth and char == "{":
                result = _cpp_braced_body(active, offset)
                return "" if result is None else result[0]
            offset += 1
        if not declaration:
            return ""


def _active_cpp_macro_names(source: str) -> set[str]:
    """Return macros defined in source branches that can compile on Windows."""
    active = without_cpp_comments_and_literals(source)
    names: set[str] = set()
    for _, line in windows_possible_lines(active):
        match = re.match(r"\s*#\s*define\s+([A-Za-z_]\w*)", line)
        if match is not None:
            names.add(match.group(1))
    return names


def _validate_unsupported_tier_contract(text: str, errors: list[str]) -> None:
    active = _active_powershell(text)
    filter_count = len(re.findall(re.escape(UNSUPPORTED_TIER_FILTER), active))
    exact_filter_argument = re.search(
        r"\$arguments\s*=\s*@\(\s*(['\"])" +
        re.escape(UNSUPPORTED_TIER_FILTER) + r"\1\s*\)",
        active,
        re.IGNORECASE | re.MULTILINE,
    )
    if filter_count == 0:
        errors.append(
            "build-windows-release.ps1: missing active isolated "
            "unsupported-tier filter"
        )
    elif filter_count != 1 or exact_filter_argument is None:
        errors.append(
            "build-windows-release.ps1: unsupported-tier probe requires one "
            "exact unsupported-tier filter argument"
        )
    requirements = (
        (
            r"\$probeExitCode\s*-ne\s*1\b",
            "exact unsupported-tier exit status",
        ),
        (
            re.escape(UNSUPPORTED_TIER_DIAGNOSTIC),
            "unsupported-tier diagnostic",
        ),
        (
            r"@\(\s*&\s*\$TierTest\s+@arguments\s+2>&1\s*\)",
            "merged unsupported-tier stdout/stderr capture",
        ),
        (
            r"finally\s*\{[^}]*\$env:VT_CPU_MATMUL_TIER\s*=\s*\$savedTier",
            "unsupported-tier environment restoration",
        ),
    )
    for pattern, description in requirements:
        if re.search(pattern, active, re.IGNORECASE | re.MULTILINE) is None:
            errors.append(
                f"build-windows-release.ps1: missing active {description}"
            )

    for crash_status in ("134", "-1073741819", "3", "2"):
        if re.search(
                rf"ExitCode\s*=\s*{re.escape(crash_status)}\b", active,
                re.IGNORECASE) is None:
            errors.append(
                "build-windows-release.ps1: unsupported-tier crash contract "
                f"is missing status {crash_status}"
            )

    uncommented = re.sub(r"(?s)<#.*?#>", "", text)
    uncommented = "\n".join(
        line for line in uncommented.splitlines()
        if not line.lstrip().startswith("#")
    )
    contract_block = re.search(
        r"(?ms)^\s*if\s*\(\s*\$ContractTest\s*\)\s*\{(?P<body>.*?)^\s*\}",
        uncommented,
    )
    if (contract_block is None or
            re.search(r"(?m)^\s*Invoke-UnsupportedTierContractTests\s*$",
                      contract_block.group("body")) is None):
        errors.append(
            "build-windows-release.ps1: missing active unsupported-tier "
            "fake-tool contract"
        )

    amx = re.search(
        r"\$env:VT_CPU_MATMUL_TIER\s*=\s*[\"']amx[\"'](?P<body>.*?)"
        r"(?:\}\s*finally\b|\bfinally\b)",
        active,
        re.IGNORECASE | re.DOTALL,
    )
    if (amx is None or
            re.search(r"Invoke-UnsupportedTierProbe\b[^\n]*\$tierTest\b",
                      amx.group("body"), re.IGNORECASE) is None or
            re.search(r"(?:Invoke-Checked\s+|&\s*)\$tierTest\b",
                      amx.group("body"),
                      re.IGNORECASE) is not None):
        errors.append(
            "build-windows-release.ps1: AMX refusal must use only the isolated "
            "unsupported-tier probe"
        )


def _finite_timeout_expression(console: str, expression: str) -> bool:
    expression = expression.strip()
    if re.fullmatch(r"\d+(?:[uUlL]*)", expression):
        return True
    if not re.fullmatch(r"[A-Za-z_]\w*", expression):
        return False
    definitions = re.findall(
        rf"\b(?:constexpr|const)\b[^;=]*\b{re.escape(expression)}\s*=\s*([^;]+);",
        console,
    )
    return (len(definitions) == 1 and
            re.fullmatch(r"\s*\d+(?:[uUlL]*)\s*", definitions[0]) is not None)


def _validate_bounded_drain(console: str, function: str, counter: str,
                            errors: list[str]) -> None:
    active_console = without_cpp_comments_and_literals(console)
    body = _cpp_function_body(
        active_console,
        rf"\bbool\s+{re.escape(function)}\s*\(",
    )
    label = f"console_shutdown.cpp: {function} requires a finite timeout"
    if not body or re.search(r"\bINFINITE\b", body):
        errors.append(label)
        return

    starts = list(re.finditer(
        r"\b(?:const\s+)?ULONGLONG\s+start\s*=\s*"
        r"GetTickCount64\s*\(\s*\)\s*;",
        body,
    ))
    loop = re.search(
        rf"\bwhile\s*\(\s*{re.escape(counter)}\.load\s*\(\s*"
        r"std::memory_order_seq_cst\s*\)\s*!=\s*0\s*\)\s*\{",
        body,
    )
    if (len(starts) != 1 or loop is None or
            starts[0].start() >= (loop.start() if loop else 0)):
        errors.append(label)
        return

    opening = body.find("{", loop.start(), loop.end())
    loop_result = _cpp_braced_body(body, opening)
    if loop_result is None:
        errors.append(label)
        return
    loop_body, loop_close = loop_result
    timeout_branch = re.match(
        r"\s*if\s*\(\s*GetTickCount64\s*\(\s*\)\s*-\s*start\s*>=\s*"
        r"([A-Za-z_]\w*|\d+[uUlL]*)\s*\)"
        r"\s*(?:\{\s*)?return\s+false\s*;\s*(?:\}\s*)?",
        loop_body,
    )
    success_tail = body[loop_close + 1:]
    if (timeout_branch is None or
            not _finite_timeout_expression(active_console,
                                           timeout_branch.group(1)) or
            re.fullmatch(
                r"\s*return\s+true\s*;\s*", success_tail
            ) is None):
        errors.append(label)


def _validate_console_protocol(console: str, errors: list[str]) -> None:
    active_console = without_cpp_comments_and_literals(console)
    dispatch = _cpp_function_body(console, r"\bbool\s+DispatchControlEvent\s*\(")
    handler = _cpp_function_body(
        console, r"\bBOOL\s+WINAPI\s+ConsoleControlHandler\s*\("
    )
    stable_declarations = (
        re.search(r"std::atomic\s*<\s*WindowsHandlerState\s*\*\s*>\s+published", console),
        re.search(r"std::atomic\s*<\s*unsigned\s*>\s+entrants", console),
        re.search(r"std::atomic\s*<\s*unsigned\s*>\s+in_flight", console),
        re.search(
            r"static\s+auto\s*\*\s*registry\s*=\s*new\s+WindowsHandlerRegistry",
            console,
        ),
    )
    dispatch_steps = (
        r"entrants\.fetch_add\s*\(\s*1\s*,\s*std::memory_order_seq_cst\s*\)",
        r"published\.load\s*\(\s*std::memory_order_seq_cst\s*\)",
        r"in_flight\.fetch_add\s*\(\s*1\s*,\s*std::memory_order_seq_cst\s*\)",
        r"entrants\.fetch_sub\s*\(\s*1\s*,\s*std::memory_order_seq_cst\s*\)",
        r"SetEvent\s*\(\s*state->stop_event\s*\)",
        r"in_flight\.fetch_sub\s*\(\s*1\s*,\s*std::memory_order_seq_cst\s*\)",
    )
    dispatch_offsets = []
    for pattern in dispatch_steps:
        match = re.search(pattern, dispatch)
        dispatch_offsets.append(-1 if match is None else match.start())
    teardown = _cpp_function_body(
        console, r"ConsoleShutdown::~ConsoleShutdown\s*\("
    ) or console
    teardown_steps = (
        r"published\.store\s*\(\s*nullptr\s*,\s*std::memory_order_seq_cst\s*\)",
        r"DrainEntrantsWithTimeout\s*\(",
        r"DrainInFlightWithTimeout\s*\(",
        r"(?:(?:win_state_|state)\.reset\s*\(\s*\))",
    )
    teardown_offsets = []
    for index, pattern in enumerate(teardown_steps):
        matches = list(re.finditer(pattern, teardown))
        match = matches[-1] if index == len(teardown_steps) - 1 and matches else (
            matches[0] if matches else None
        )
        teardown_offsets.append(-1 if match is None else match.start())
    ordered_dispatch = (
        all(offset >= 0 for offset in dispatch_offsets) and
        dispatch_offsets == sorted(dispatch_offsets)
    )
    ordered_teardown = (
        all(offset >= 0 for offset in teardown_offsets) and
        teardown_offsets == sorted(teardown_offsets)
    )
    retire = re.search(r"RetireHandlerState\s*\(", teardown)
    reset = re.search(r"(?:win_state_|state)\.reset\s*\(", teardown)
    timeout_retained = (
        re.search(r"if\s*\(\s*!\s*safe_to_close\s*\)", teardown) is not None and
        retire is not None and reset is not None and retire.start() < reset.start()
    )
    if (not all(stable_declarations) or not ordered_dispatch or
            not ordered_teardown or not timeout_retained):
        errors.append(
            "console_shutdown.cpp: stable event/in-flight handler lifetime protocol is required"
        )
    forbidden = r"RequestStop|\bimpl_|\bthis\b|std::mutex|condition_variable|\bstop_\s*\("
    if not dispatch or not handler or re.search(forbidden, dispatch + "\n" + handler):
        errors.append(
            "console_shutdown.cpp: OS handler may use only stable atomics and Win32 events"
        )
    final_decrements = list(re.finditer(
        r"state->in_flight\.fetch_sub\s*\(\s*1\s*,\s*"
        r"std::memory_order_seq_cst\s*\)",
        dispatch,
    ))
    trusted_tail_tokens = {
        "state", "in_flight", "fetch_sub", "std", "memory_order_seq_cst",
        "return", "resumed",
    }
    macro_collisions = sorted(
        trusted_tail_tokens.intersection(_active_cpp_macro_names(console))
    )
    if macro_collisions:
        errors.append(
            "console_shutdown.cpp: trusted final-tail token must not be a macro "
            f"({', '.join(macro_collisions)})"
        )
    final_tail_ok = False
    if len(final_decrements) == 1:
        resumed_declarations = list(re.finditer(
            r"\b(?:const\s+)?bool\s+resumed\s*=\s*[^;]+;",
            dispatch[:final_decrements[0].start()],
        ))
        resumed_macro = re.search(
            r"(?m)^\s*#\s*define\s+resumed(?:\s|\()",
            active_console,
        )
        final_tail = dispatch[final_decrements[0].end():]
        final_tail_ok = (
            len(resumed_declarations) == 1 and resumed_macro is None and
            re.fullmatch(
                r"\s*;\s*return\s+resumed\s*;\s*",
                final_tail,
            ) is not None
        )
    if not final_tail_ok:
        errors.append(
            "console_shutdown.cpp: final in-flight decrement must be the last "
            "handler operation"
        )
    _validate_bounded_drain(
        console, "DrainEntrantsWithTimeout", "entrants", errors
    )
    _validate_bounded_drain(
        console, "DrainInFlightWithTimeout", "in_flight", errors
    )
    cleanup = _cpp_function_body(
        console, r"WindowsHandlerState::~WindowsHandlerState\s*\("
    ) or console
    if not all(re.search(rf"CloseHandle\s*\(\s*{event}\s*\)", cleanup)
               for event in ("stop_event", "quit_event")):
        errors.append(
            "console_shutdown.cpp: partial event creation cleanup must close every created handle"
        )


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
}, $true) | Where-Object {
  if (Test-Dead $_) { return $false }
  $cursor = $_.Parent
  while ($null -ne $cursor) {
    if ($cursor -is [System.Management.Automation.Language.FunctionDefinitionAst]) {
      return $false
    }
    $cursor = $cursor.Parent
  }
  return $true
} | ForEach-Object {
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
        ("isolated unsupported-tier smoke",
         r"Invoke-UnsupportedTierProbe\b.*?\$tierTest"),
        ("unsupported-tier fake-tool contract",
         r"Invoke-UnsupportedTierContractTests"),
    ):
        if not re.search(pattern, command_text, re.IGNORECASE):
            errors.append(f"build-windows-release.ps1: AST missing active {description}")
    _validate_powershell_ast_order(decoded, errors)
    contract = subprocess.run(
        [pwsh, "-NoProfile", "-NonInteractive", "-File", str(script),
         "-SourceDir", str(script.parents[1]), "-ContractTest"],
        text=True, capture_output=True, check=False,
    )
    if contract.returncode != 0:
        errors.append("build-windows-release.ps1: injected fake-tool contract failed: " +
                      contract.stdout + contract.stderr)


def check(root: Path, build_dir: Path | None = None,
          source_manifest: Path | None = None) -> list[str]:
    errors: list[str] = []
    try:
        source_paths = shipped_server_sources(root, build_dir, source_manifest)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        errors.append(f"shipped-server source discovery failed: {exc}")
        source_paths = set()
    for required in REQUIRED_CPP:
        if required not in source_paths:
            errors.append(
                f"{required}: required implementation is not reachable from "
                "the shipped server target"
            )
    texts = {
        relative: require_file(root, relative, errors)
        for relative in sorted(source_paths)
        if Path(relative).suffix.lower() in CPP_SUFFIXES
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
            platform_boundary = (
                relative in REQUIRED_CPP or
                relative.startswith("src/vllm/platform/")
            )
            scoped_posix = platform_boundary and any(
                re.search(pattern, line) for pattern in POSIX_PATTERNS
            )
            if full_source_posix or scoped_posix:
                errors.append(f"{relative}:{number}: unguarded POSIX include/call reaches Windows")
            if (re.search(
                    r"\b(?:CreateFileA|LoadLibraryA|MoveFileExA|DeleteFileA)\b",
                    line,
                ) or (platform_boundary and re.search(r"\.string\s*\(\)", line))):
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
        active_source = without_cpp_comments(source)
        if any("__builtin_clzll" in line
               for _, line in windows_possible_lines(active_source)):
            errors.append(f"{relative}: non-portable compiler builtin reaches MSVC")

    server = "\n".join(texts.get(name, "") for name in REQUIRED_CPP[:3])
    for marker in ("CreateProcessW", "SetConsoleCtrlHandler"):
        if marker not in server:
            errors.append(f"server_main.cpp: required Win32 process/console support missing ({marker})")
    console = texts.get("src/vllm/platform/console_shutdown.cpp", "")
    _validate_console_protocol(console, errors)

    lmcache = texts.get("src/vllm/v1/kv_offload/lmcache/remote_client.cpp", "")
    if not all(marker in lmcache for marker in ("WSAStartup", "WSAGetLastError", "closesocket")):
        errors.append("remote_client.cpp: LMCache Winsock support is missing or silently disabled")
    if not re.search(r"(?s)if\s*\([^\)]*==\s*0\s*\).*?Close\s*\(\s*\).*?throw", lmcache):
        errors.append("remote_client.cpp: peer-close must invalidate the owned socket")

    fs_io = texts.get("src/vllm/v1/kv_offload/fs_io.cpp", "")
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
    _validate_powershell_source_order(build_script, errors)
    _validate_unsupported_tier_contract(build_script, errors)
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
    if source_manifest is None:
        _validate_powershell_ast(root / "scripts/build-windows-release.ps1", errors)

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument(
        "--test-source-manifest", type=Path,
        help="explicit source fixture for checker tests; normal runs use CMake codemodel",
    )
    args = parser.parse_args()
    errors = check(args.root.resolve(), args.build_dir, args.test_source_manifest)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Windows portability contract OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
