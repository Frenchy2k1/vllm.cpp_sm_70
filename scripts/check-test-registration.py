#!/usr/bin/env python3
"""Require critical regression tests to remain buildable and CTest-registered.

The CPU CI lane builds every target and runs ``ctest``, but both promises become
vacuous when a regression test is accidentally removed from ``tests/CMakeLists.txt``.
This tree gate pins the small set of tests whose review explicitly requires a
non-vacuous registration guard.  It also verifies that the shared helper still
creates an executable *and* registers that executable with CTest.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS_CMAKE = ROOT / "tests/CMakeLists.txt"
PREFLIGHT = ROOT / "scripts/agent-preflight.sh"
CI = ROOT / ".github/workflows/ci.yml"

REQUIRED_TESTS = {
    "test_device_selection": "vllm/entrypoints/test_device_selection.cpp",
}

def _without_line_comments(text: str) -> str:
    """Remove ``#`` comments while preserving quoted ``#`` characters."""

    cleaned: list[str] = []
    for line in text.splitlines():
        quoted = False
        escaped = False
        kept: list[str] = []
        for char in line:
            if escaped:
                kept.append(char)
                escaped = False
                continue
            if char == "\\" and quoted:
                kept.append(char)
                escaped = True
                continue
            if char == '"':
                quoted = not quoted
                kept.append(char)
                continue
            if char == "#" and not quoted:
                break
            kept.append(char)
        cleaned.append("".join(kept))
    return "\n".join(cleaned)


def _configure(
    source_dir: Path, build_dir: Path, extra_args: list[str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Configure CMake after requesting its semantic target codemodel."""

    query = build_dir / ".cmake/api/v1/query/codemodel-v2"
    query.parent.mkdir(parents=True, exist_ok=True)
    query.touch()
    command = ["cmake", "-S", str(source_dir), "-B", str(build_dir)]
    if extra_args:
        command.extend(extra_args)
    return subprocess.run(command, text=True, capture_output=True, check=False)


def _codemodel_targets(build_dir: Path) -> dict[str, dict[str, object]]:
    """Return configured targets keyed by name from the CMake File API."""

    reply = build_dir / ".cmake/api/v1/reply"
    indexes = sorted(reply.glob("index-*.json"))
    if not indexes:
        return {}
    index = json.loads(indexes[-1].read_text(encoding="utf-8"))
    codemodel_file = index["reply"]["codemodel-v2"]["jsonFile"]
    codemodel = json.loads((reply / codemodel_file).read_text(encoding="utf-8"))
    targets: dict[str, dict[str, object]] = {}
    for configuration in codemodel.get("configurations", []):
        for summary in configuration.get("targets", []):
            detail = json.loads((reply / summary["jsonFile"]).read_text(encoding="utf-8"))
            targets[summary["name"]] = detail
    return targets


def _ctest_names(build_dir: Path) -> set[str]:
    result = subprocess.run(
        ["ctest", "--test-dir", str(build_dir), "--show-only=json-v1"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return set()
    document = json.loads(result.stdout)
    return {test["name"] for test in document.get("tests", [])}


def _configured_contract_errors(
    source_dir: Path,
    build_dir: Path,
    required: dict[str, str],
    extra_args: list[str] | None = None,
) -> list[str]:
    """Ask CMake/CTest what exists instead of interpreting CMake source text."""

    configured = _configure(source_dir, build_dir, extra_args)
    if configured.returncode != 0:
        transcript = configured.stdout + configured.stderr
        errors = ["CMake configure failed while proving required test registration"]
        for target in sorted(required):
            errors.append(f"missing required test target {target} in configured codemodel")
        if "already exists" in transcript:
            for target in sorted(required):
                errors.append(f"required test target {target} is registered 2 times")
        errors.append(
            "vllm_cpp_add_test does not create an executable with its configured sources"
        )
        return errors

    targets = _codemodel_targets(build_dir)
    tests = _ctest_names(build_dir)
    errors: list[str] = []
    for target, expected_source in sorted(required.items()):
        detail = targets.get(target)
        if detail is None:
            errors.append(f"missing required test target {target} in configured codemodel")
            errors.append(
                "vllm_cpp_add_test does not create an executable with its configured sources"
            )
            continue
        actual_sources = {
            Path(source["path"]).as_posix() for source in detail.get("sources", [])
        }
        if expected_source not in actual_sources:
            actual = ", ".join(sorted(actual_sources)) or "<no sources>"
            errors.append(
                f"required test target {target} must compile {expected_source}; got {actual}"
            )
        if target not in tests:
            errors.append(
                f"required test target {target} is not registered with CTest; "
                "vllm_cpp_add_test does not register that executable with CTest"
            )
    return errors


def registration_errors(
    cmake_text: str, required: dict[str, str] | None = None
) -> list[str]:
    """Return violations of the executable + CTest registration contract."""

    if required is None:
        required = REQUIRED_TESTS
    with tempfile.TemporaryDirectory(prefix="vllm-registration-unit-") as temporary:
        root = Path(temporary)
        (root / "CMakeLists.txt").write_text(cmake_text, encoding="utf-8")
        for source in {
            *required.values(),
            "vllm/entrypoints/other.cpp",
        }:
            path = root / source
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("int registration_guard_dummy;\n", encoding="utf-8")
        return _configured_contract_errors(root, root / "build", required)


def _yaml_literal_blocks(text: str) -> list[list[str]]:
    """Extract YAML literal ``run: |`` block lines without treating prose as code."""

    lines = text.splitlines()
    blocks: list[list[str]] = []
    for index, line in enumerate(lines):
        match = re.match(r"^(\s*)run:\s*\|[-+]?\s*$", line)
        if match is None:
            continue
        parent_indent = len(match.group(1))
        raw: list[str] = []
        for candidate in lines[index + 1 :]:
            if not candidate.strip():
                raw.append("")
                continue
            indent = len(candidate) - len(candidate.lstrip())
            if indent <= parent_indent:
                break
            raw.append(candidate)
        nonblank = [line for line in raw if line.strip()]
        if not nonblank:
            blocks.append([])
            continue
        content_indent = min(len(line) - len(line.lstrip()) for line in nonblank)
        blocks.append([line[content_indent:] if line.strip() else "" for line in raw])
    return blocks


def _direct_commands(block: list[str]) -> list[list[str]] | None:
    """Parse a literal block that contains only direct, unconditional commands."""

    commands: list[list[str]] = []
    for line in block:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line != line.lstrip():
            return None
        try:
            argv = shlex.split(stripped, comments=True, posix=True)
        except ValueError:
            return None
        if not argv or any(token in {";", "&&", "||", "|", "&"} for token in argv):
            return None
        commands.append(argv)
    return commands


def _ci_has_active_guard_step(ci_text: str) -> bool:
    expected = [
        ["python3", "scripts/check-test-registration.py"],
        ["python3", "tests/scripts/test_check_test_registration.py"],
    ]
    return any(_direct_commands(block) == expected for block in _yaml_literal_blocks(ci_text))


def _active_ci_commands(ci_text: str) -> set[tuple[str, ...]]:
    commands: set[tuple[str, ...]] = set()
    for block in _yaml_literal_blocks(ci_text):
        parsed = _direct_commands(block)
        if parsed is not None:
            commands.update(tuple(command) for command in parsed)
    return commands


def wiring_errors(preflight_text: str, ci_text: str) -> list[str]:
    """Return missing preflight/CI wiring for this checker and its mutations."""

    preflight_text = _without_line_comments(preflight_text)
    ci_text = _without_line_comments(ci_text)
    errors: list[str] = []
    if not re.search(r"^\s*check-test-registration\s*$", preflight_text, re.MULTILINE):
        errors.append("check-test-registration is missing from preflight CHECKERS")
    if not re.search(
        r"^\s*test_check_test_registration\s*$", preflight_text, re.MULTILINE
    ):
        errors.append("test_check_test_registration is missing from preflight SUITES")
    active_ci_commands = _active_ci_commands(ci_text)
    if ("python3", "scripts/check-test-registration.py") not in active_ci_commands:
        errors.append("check-test-registration is missing from the explicit CI checker step")
    if (
        "python3",
        "tests/scripts/test_check_test_registration.py",
    ) not in active_ci_commands:
        errors.append("test_check_test_registration is missing from the CI mutation suite")
    if not _ci_has_active_guard_step(ci_text):
        errors.append(
            "CI guard step must contain the checker and mutation suite as direct active commands"
        )
    return errors


def check_tree(root: Path = ROOT) -> list[str]:
    paths = {
        "tests/CMakeLists.txt": root / "tests/CMakeLists.txt",
        "scripts/agent-preflight.sh": root / "scripts/agent-preflight.sh",
        ".github/workflows/ci.yml": root / ".github/workflows/ci.yml",
    }
    missing = [relative for relative, path in paths.items() if not path.is_file()]
    if missing:
        return [f"required registration-guard input is missing: {path}" for path in missing]

    with tempfile.TemporaryDirectory(prefix="vllm-registration-tree-") as temporary:
        registration = _configured_contract_errors(
            root,
            Path(temporary) / "build",
            {
                target: f"tests/{source}"
                for target, source in REQUIRED_TESTS.items()
            },
            [
                "-DVLLM_CPP_CUDA=OFF",
                "-DVLLM_CPP_HIP=OFF",
                "-DVLLM_CPP_VULKAN=OFF",
                "-DVLLM_CPP_METAL=OFF",
                "-DVLLM_CPP_MLX=OFF",
                "-DVLLM_CPP_TRITON=OFF",
                "-DVLLM_CPP_BUILD_TESTS=ON",
                "-DVLLM_CPP_BUILD_EXAMPLES=OFF",
                "-DVLLM_CPP_SERVER=OFF",
                "-DCMAKE_BUILD_TYPE=Release",
            ],
        )
    return registration + wiring_errors(
        paths["scripts/agent-preflight.sh"].read_text(encoding="utf-8"),
        paths[".github/workflows/ci.yml"].read_text(encoding="utf-8"),
    )


def main() -> int:
    errors = check_tree()
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "OK: required regression tests have executable + CTest registration "
        "and the guard is wired into preflight/CI."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
