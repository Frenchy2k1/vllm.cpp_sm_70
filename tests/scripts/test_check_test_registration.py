#!/usr/bin/env python3
"""Mutation checks for the required CMake/CTest registration guard."""

from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-test-registration.py"
SPEC = importlib.util.spec_from_file_location("check_test_registration", CHECKER)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


PASSING_CMAKE = """
cmake_minimum_required(VERSION 3.20)
project(registration_guard LANGUAGES CXX)
enable_testing()
add_library(vllm_core INTERFACE)
add_library(vllm::vllm ALIAS vllm_core)
add_library(vllm_test_main INTERFACE)

function(vllm_cpp_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE vllm::vllm vllm_test_main)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

vllm_cpp_add_test(test_device_selection
  vllm/entrypoints/test_device_selection.cpp)
"""

PASSING_CI = """jobs:
  checks:
    steps:
      - name: Critical regression tests remain executable and CTest-registered
        run: |
          python3 scripts/check-test-registration.py
          python3 tests/scripts/test_check_test_registration.py
"""

PASSING_PREFLIGHT = """CHECKERS=(
  check-test-registration
)
SUITES=(
  test_check_test_registration
)
for checker in "${CHECKERS[@]}"; do
  python3 "scripts/$checker.py"
done
for suite in "${SUITES[@]}"; do
  python3 "tests/scripts/$suite.py"
done
"""


def _suite_integrity_errors(source: str) -> list[str]:
    """Return semantic holes in this mutation suite's own proof machinery."""

    tree = ast.parse(source)
    methods = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    manifest = {
        line
        for line in (ROOT / "tests/scripts/check_test_registration_mutations.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line and not line.startswith("#")
    }
    errors: list[str] = []
    actual = {name for name in methods if name.startswith("test_M")}
    if actual != manifest:
        errors.append("mutation inventory differs from fixed manifest")

    for name in sorted(actual & manifest):
        calls = {
            call.func.attr
            for call in ast.walk(methods[name])
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        }
        if not {"assert_error", "assert_wiring_error"} & calls:
            errors.append(f"{name} has no semantic outcome assertion")

    for name, production_call in {
        "assert_error": "registration_errors",
        "assert_wiring_error": "wiring_errors",
    }.items():
        method = methods.get(name)
        if method is None:
            errors.append(f"{name} helper is missing")
            continue
        calls = [call for call in ast.walk(method) if isinstance(call, ast.Call)]
        if not any(
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "self"
            and call.func.attr == "assertTrue"
            for call in calls
        ):
            errors.append(f"{name} has no direct semantic assertion")
        if not any(
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "mod"
            and call.func.attr == production_call
            for call in calls
        ):
            errors.append(f"{name} does not call {production_call}")
    return errors


class RegistrationMutationTests(unittest.TestCase):
    def assert_error(self, text: str, needle: str) -> None:
        errors = mod.registration_errors(text)
        self.assertTrue(any(needle in error for error in errors), errors)

    def test_minimal_complete_registration_passes(self) -> None:
        self.assertEqual(mod.registration_errors(PASSING_CMAKE), [])

    def test_M1_deleting_test_invocation_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "vllm_cpp_add_test(test_device_selection\n"
            "  vllm/entrypoints/test_device_selection.cpp)\n",
            "",
        )
        self.assertNotEqual(mutated, PASSING_CMAKE)
        self.assert_error(mutated, "missing required test target test_device_selection")

    def test_M2_changing_test_source_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "vllm/entrypoints/test_device_selection.cpp", "vllm/entrypoints/other.cpp"
        )
        self.assertNotEqual(mutated, PASSING_CMAKE)
        self.assert_error(mutated, "must compile vllm/entrypoints/test_device_selection.cpp")

    def test_M3_deleting_add_executable_from_helper_fails(self) -> None:
        mutated = PASSING_CMAKE.replace("  add_executable(${name} ${ARGN})\n", "")
        self.assertNotEqual(mutated, PASSING_CMAKE)
        self.assert_error(mutated, "does not create an executable")

    def test_M4_deleting_add_test_from_helper_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "  add_test(NAME ${name} COMMAND ${name})\n", ""
        )
        self.assertNotEqual(mutated, PASSING_CMAKE)
        self.assert_error(mutated, "does not register that executable with CTest")

    def test_M5_duplicate_target_fails(self) -> None:
        mutated = PASSING_CMAKE + (
            "vllm_cpp_add_test(test_device_selection\n"
            "  vllm/entrypoints/test_device_selection.cpp)\n"
        )
        self.assert_error(mutated, "registered 2 times")

    def test_M6_commented_out_invocation_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "vllm_cpp_add_test(test_device_selection\n"
            "  vllm/entrypoints/test_device_selection.cpp)",
            "# vllm_cpp_add_test(test_device_selection "
            "vllm/entrypoints/test_device_selection.cpp)",
        )
        self.assert_error(mutated, "missing required test target test_device_selection")

    def test_M7_commented_out_add_executable_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "  add_executable(${name} ${ARGN})", "  # add_executable(${name} ${ARGN})"
        )
        self.assert_error(mutated, "does not create an executable")

    def test_M8_commented_out_add_test_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "  add_test(NAME ${name} COMMAND ${name})",
            "  # add_test(NAME ${name} COMMAND ${name})",
        )
        self.assert_error(mutated, "does not register that executable with CTest")

    def test_M13_bracket_commented_target_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "vllm_cpp_add_test(test_device_selection\n"
            "  vllm/entrypoints/test_device_selection.cpp)",
            "#[[\nvllm_cpp_add_test(test_device_selection\n"
            "  vllm/entrypoints/test_device_selection.cpp)\n]]",
        )
        self.assertNotEqual(mutated, PASSING_CMAKE)
        self.assert_error(mutated, "missing required test target test_device_selection")

    def test_M14_target_in_false_conditional_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "vllm_cpp_add_test(test_device_selection\n"
            "  vllm/entrypoints/test_device_selection.cpp)",
            "if(FALSE)\n"
            "  vllm_cpp_add_test(test_device_selection\n"
            "    vllm/entrypoints/test_device_selection.cpp)\n"
            "endif()",
        )
        self.assertNotEqual(mutated, PASSING_CMAKE)
        self.assert_error(mutated, "missing required test target test_device_selection")

    def test_M15_quoted_target_text_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "vllm_cpp_add_test(test_device_selection\n"
            "  vllm/entrypoints/test_device_selection.cpp)",
            'set(dead "vllm_cpp_add_test(test_device_selection '
            'vllm/entrypoints/test_device_selection.cpp)")',
        )
        self.assertNotEqual(mutated, PASSING_CMAKE)
        self.assert_error(mutated, "missing required test target test_device_selection")

    def test_M16_helper_registration_in_false_conditional_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "  add_test(NAME ${name} COMMAND ${name})",
            "  if(FALSE)\n"
            "    add_test(NAME ${name} COMMAND ${name})\n"
            "  endif()",
        )
        self.assertNotEqual(mutated, PASSING_CMAKE)
        self.assert_error(mutated, "is not registered with CTest")

    def test_M17_helper_executable_in_false_conditional_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "  add_executable(${name} ${ARGN})",
            "  if(FALSE)\n"
            "    add_executable(${name} ${ARGN})\n"
            "  endif()",
        )
        self.assertNotEqual(mutated, PASSING_CMAKE)
        self.assert_error(mutated, "missing required test target test_device_selection")

    def test_M20_ctest_noop_command_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "  add_test(NAME ${name} COMMAND ${name})",
            "  add_test(NAME ${name} COMMAND ${CMAKE_COMMAND} -E true)",
        )
        self.assertNotEqual(mutated, PASSING_CMAKE)
        self.assert_error(
            mutated, "must execute configured target test_device_selection exactly"
        )

    def test_M21_ctest_wrong_binary_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "  add_test(NAME ${name} COMMAND ${name})",
            "  add_test(NAME ${name} COMMAND other)",
        )
        self.assertNotEqual(mutated, PASSING_CMAKE)
        self.assert_error(
            mutated, "must execute configured target test_device_selection exactly"
        )

    def test_M22_ctest_extra_argument_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "  add_test(NAME ${name} COMMAND ${name})",
            "  add_test(NAME ${name} COMMAND ${name} --list-test-cases)",
        )
        self.assertNotEqual(mutated, PASSING_CMAKE)
        self.assert_error(
            mutated, "must execute configured target test_device_selection exactly"
        )


class WiringMutationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.preflight = PASSING_PREFLIGHT
        self.ci = PASSING_CI

    def assert_wiring_error(self, preflight: str, ci: str, needle: str) -> None:
        errors = mod.wiring_errors(preflight, ci)
        self.assertTrue(any(needle in error for error in errors), errors)

    def test_complete_preflight_and_ci_wiring_passes(self) -> None:
        self.assertEqual(mod.wiring_errors(self.preflight, self.ci), [])

    def test_M9_deleting_preflight_checker_fails(self) -> None:
        mutated = self.preflight.replace("  check-test-registration\n", "")
        self.assert_wiring_error(mutated, self.ci, "preflight CHECKERS")

    def test_M10_deleting_preflight_suite_fails(self) -> None:
        mutated = self.preflight.replace("  test_check_test_registration\n", "")
        self.assert_wiring_error(mutated, self.ci, "preflight SUITES")

    def test_M11_deleting_ci_checker_fails(self) -> None:
        mutated = self.ci.replace(
            "          python3 scripts/check-test-registration.py\n", ""
        )
        self.assert_wiring_error(self.preflight, mutated, "CI checker")

    def test_M12_deleting_ci_suite_fails(self) -> None:
        mutated = self.ci.replace(
            "          python3 tests/scripts/test_check_test_registration.py\n", ""
        )
        self.assert_wiring_error(self.preflight, mutated, "CI mutation suite")

    def test_M18_ci_commands_behind_false_shell_branch_fail(self) -> None:
        mutated = self.ci.replace(
            "          python3 scripts/check-test-registration.py\n"
            "          python3 tests/scripts/test_check_test_registration.py\n",
            "          if false; then\n"
            "            python3 scripts/check-test-registration.py\n"
            "            python3 tests/scripts/test_check_test_registration.py\n"
            "          fi\n",
        )
        self.assertNotEqual(mutated, self.ci)
        self.assert_wiring_error(self.preflight, mutated, "direct active commands")

    def test_M19_ci_commands_as_quoted_text_fail(self) -> None:
        mutated = self.ci.replace(
            "          python3 scripts/check-test-registration.py\n",
            '          echo "python3 scripts/check-test-registration.py"\n',
        )
        self.assertNotEqual(mutated, self.ci)
        self.assert_wiring_error(self.preflight, mutated, "direct active commands")

    def test_M23_ci_step_with_false_condition_fails(self) -> None:
        mutated = self.ci.replace(
            "      - name: Critical regression tests remain executable and CTest-registered\n"
            "        run: |\n",
            "      - name: Critical regression tests remain executable and CTest-registered\n"
            "        if: ${{ false }}\n"
            "        run: |\n",
        )
        self.assertNotEqual(mutated, self.ci)
        self.assert_wiring_error(self.preflight, mutated, "direct active commands")

    def test_M24_ci_job_with_false_condition_fails(self) -> None:
        mutated = self.ci.replace(
            "  checks:\n",
            "  checks:\n"
            "    if: ${{ false }}\n",
        )
        self.assertNotEqual(mutated, self.ci)
        self.assert_wiring_error(self.preflight, mutated, "direct active commands")

    def test_M25_preflight_names_in_inert_arrays_fail(self) -> None:
        mutated = self.preflight.replace(
            "  check-test-registration\n", ""
        ).replace(
            "  test_check_test_registration\n", ""
        ) + (
            "INERT=(\n"
            "  check-test-registration\n"
            "  test_check_test_registration\n"
            ")\n"
        )
        self.assertNotEqual(mutated, self.preflight)
        self.assert_wiring_error(mutated, self.ci, "preflight CHECKERS")
        self.assert_wiring_error(mutated, self.ci, "preflight SUITES")

    def test_M26_preflight_checker_loop_rebound_fails(self) -> None:
        mutated = self.preflight.replace(
            'for checker in "${CHECKERS[@]}"; do',
            'for checker in "${INERT[@]}"; do',
        )
        self.assertNotEqual(mutated, self.preflight)
        self.assert_wiring_error(mutated, self.ci, "execute CHECKERS")

    def test_M27_preflight_suite_loop_rebound_fails(self) -> None:
        mutated = self.preflight.replace(
            'for suite in "${SUITES[@]}"; do',
            'for suite in "${INERT[@]}"; do',
        )
        self.assertNotEqual(mutated, self.preflight)
        self.assert_wiring_error(mutated, self.ci, "execute SUITES")


class ShippedTreeTests(unittest.TestCase):
    def test_shipped_tree_is_registered_and_wired(self) -> None:
        self.assertEqual(mod.check_tree(ROOT), [])


class SuiteIntegrityTests(unittest.TestCase):
    def test_deleting_wrapper_assertions_is_caught(self) -> None:
        source = Path(__file__).read_text(encoding="utf-8")
        mutated = source.replace(
            "        self.assertTrue(any(needle in error for error in errors), errors)\n",
            "",
        )
        self.assertNotEqual(mutated, source)
        errors = _suite_integrity_errors(mutated)
        self.assertTrue(any("assert_error has no direct semantic assertion" in e for e in errors))
        self.assertTrue(
            any("assert_wiring_error has no direct semantic assertion" in e for e in errors)
        )

    def test_deleting_a_manifested_mutation_is_caught(self) -> None:
        source = Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "test_M19_ci_commands_as_quoted_text_fail"
        )
        lines = source.splitlines(keepends=True)
        mutated = "".join(lines[: method.lineno - 1] + lines[method.end_lineno :])
        errors = _suite_integrity_errors(mutated)
        self.assertTrue(any("fixed manifest" in error for error in errors), errors)

    def test_mutation_cases_keep_their_outcome_assertions(self) -> None:
        """Make deleting a mutation's only meaningful assertion turn RED."""

        source = Path(__file__).read_text(encoding="utf-8")
        self.assertEqual(_suite_integrity_errors(source), [])

    def test_assertion_helpers_keep_semantic_outcome_assertions(self) -> None:
        """The mutation wrappers must themselves retain a real assertion."""

        source = Path(__file__).read_text(encoding="utf-8")
        errors = _suite_integrity_errors(source)
        self.assertFalse(any("assert_error" in error for error in errors), errors)
        self.assertFalse(any("assert_wiring_error" in error for error in errors), errors)

        tree = ast.parse(source)
        methods = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name in {
            "test_minimal_complete_registration_passes",
            "test_complete_preflight_and_ci_wiring_passes",
            "test_shipped_tree_is_registered_and_wired",
        }:
            method = methods[name]
            calls = {
                call.func.attr
                for call in ast.walk(method)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            }
            self.assertIn("assertEqual", calls, name)


if __name__ == "__main__":
    unittest.main()
