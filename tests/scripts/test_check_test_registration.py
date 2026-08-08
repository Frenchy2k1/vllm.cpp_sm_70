#!/usr/bin/env python3
"""Mutation checks for the required CMake/CTest registration guard."""

from __future__ import annotations

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
function(vllm_cpp_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE vllm::vllm vllm_test_main)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

vllm_cpp_add_test(test_device_selection
  vllm/entrypoints/test_device_selection.cpp)
"""


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


class WiringMutationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.preflight = (
            "CHECKERS=(\n  check-test-registration\n)\n"
            "SUITES=(\n  test_check_test_registration\n)\n"
        )
        self.ci = (
            "python3 scripts/check-test-registration.py\n"
            "python3 tests/scripts/test_check_test_registration.py\n"
        )

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
        mutated = self.ci.replace("python3 scripts/check-test-registration.py\n", "")
        self.assert_wiring_error(self.preflight, mutated, "CI checker")

    def test_M12_deleting_ci_suite_fails(self) -> None:
        mutated = self.ci.replace(
            "python3 tests/scripts/test_check_test_registration.py\n", ""
        )
        self.assert_wiring_error(self.preflight, mutated, "CI mutation suite")


class ShippedTreeTests(unittest.TestCase):
    def test_shipped_tree_is_registered_and_wired(self) -> None:
        self.assertEqual(mod.check_tree(ROOT), [])


if __name__ == "__main__":
    unittest.main()
