#!/usr/bin/env python3
"""Require critical regression tests to remain buildable and CTest-registered.

The CPU CI lane builds every target and runs ``ctest``, but both promises become
vacuous when a regression test is accidentally removed from ``tests/CMakeLists.txt``.
This tree gate pins the small set of tests whose review explicitly requires a
non-vacuous registration guard.  It also verifies that the shared helper still
creates an executable *and* registers that executable with CTest.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS_CMAKE = ROOT / "tests/CMakeLists.txt"
PREFLIGHT = ROOT / "scripts/agent-preflight.sh"
CI = ROOT / ".github/workflows/ci.yml"

REQUIRED_TESTS = {
    "test_device_selection": "vllm/entrypoints/test_device_selection.cpp",
}

HELPER_RE = re.compile(
    r"function\s*\(\s*vllm_cpp_add_test\s+name\s*\)(.*?)"
    r"endfunction\s*\(\s*\)",
    re.DOTALL,
)
INVOCATION_RE = re.compile(
    r"vllm_cpp_add_test\s*\(\s*(test_[A-Za-z0-9_]+)\s+([^)]*?)\)",
    re.DOTALL,
)
ADD_EXECUTABLE_RE = re.compile(
    r"add_executable\s*\(\s*\$\{name\}\s+\$\{ARGN\}\s*\)", re.DOTALL
)
ADD_CTEST_RE = re.compile(
    r"add_test\s*\(\s*NAME\s+\$\{name\}\s+COMMAND\s+\$\{name\}\s*\)",
    re.DOTALL,
)


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


def _sources(arguments: str) -> list[str]:
    return [token for token in re.split(r"\s+", arguments.strip()) if token]


def registration_errors(
    cmake_text: str, required: dict[str, str] | None = None
) -> list[str]:
    """Return violations of the executable + CTest registration contract."""

    if required is None:
        required = REQUIRED_TESTS
    cmake_text = _without_line_comments(cmake_text)
    errors: list[str] = []

    helper = HELPER_RE.search(cmake_text)
    if helper is None:
        errors.append(
            "tests/CMakeLists.txt has no vllm_cpp_add_test(name) helper; "
            "required tests cannot prove executable/CTest registration"
        )
    else:
        body = helper.group(1)
        if ADD_EXECUTABLE_RE.search(body) is None:
            errors.append(
                "vllm_cpp_add_test does not create an executable with "
                "add_executable(${name} ${ARGN})"
            )
        if ADD_CTEST_RE.search(body) is None:
            errors.append(
                "vllm_cpp_add_test does not register that executable with CTest "
                "using add_test(NAME ${name} COMMAND ${name})"
            )

    registrations: dict[str, list[list[str]]] = {}
    for match in INVOCATION_RE.finditer(cmake_text):
        registrations.setdefault(match.group(1), []).append(_sources(match.group(2)))

    for target, source in sorted(required.items()):
        found = registrations.get(target, [])
        if not found:
            errors.append(
                f"missing required test target {target}: expected "
                f"vllm_cpp_add_test({target} {source})"
            )
            continue
        if len(found) != 1:
            errors.append(f"required test target {target} is registered {len(found)} times")
            continue
        if found[0] != [source]:
            actual = " ".join(found[0]) or "<no sources>"
            errors.append(
                f"required test target {target} must compile {source}; got {actual}"
            )

    return errors


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
    if "python3 scripts/check-test-registration.py" not in ci_text:
        errors.append("check-test-registration is missing from the explicit CI checker step")
    if "python3 tests/scripts/test_check_test_registration.py" not in ci_text:
        errors.append("test_check_test_registration is missing from the CI mutation suite")
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

    return registration_errors(paths["tests/CMakeLists.txt"].read_text(encoding="utf-8")) + wiring_errors(
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
