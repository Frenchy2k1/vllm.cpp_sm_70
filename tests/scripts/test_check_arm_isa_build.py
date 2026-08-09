#!/usr/bin/env python3
"""Mutation tests for the W4 aarch64 ISA compile-command audit."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-arm-isa-build.py"


def entry(source: str, flags: str = "") -> dict[str, str]:
    return {
        "file": str(ROOT / source),
        "command": f"c++ {flags} -c {source}",
    }


def valid_commands() -> list[dict[str, str]]:
    return [
        entry("src/vt/cpu/cpu_isa_arm.cpp"),
        entry("src/vt/cpu/cpu_matmul_elem.cpp"),
        entry("src/vt/cpu/cpu_quant_dot.cpp"),
        entry("src/vt/cpu/cpu_quant_repack.cpp"),
        entry(
            "src/vt/cpu/cpu_quant_dot_sdot.cpp",
            "-march=armv8.2-a+dotprod+fp16",
        ),
        entry(
            "src/vt/cpu/cpu_quant_dot_arm.cpp",
            "-march=armv8.2-a+i8mm+dotprod",
        ),
        entry(
            "src/vt/cpu/cpu_quant_repack_arm.cpp",
            "-march=armv8.2-a+i8mm+dotprod",
        ),
    ]


class ArmIsaBuildContract(unittest.TestCase):
    def run_checker(
        self, entries: list[dict[str, str]]
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temporary:
            commands = Path(temporary) / "compile_commands.json"
            commands.write_text(json.dumps(entries), encoding="utf-8")
            return subprocess.run(
                [
                    sys.executable,
                    str(CHECKER),
                    "--compile-commands",
                    str(commands),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

    def test_exact_per_source_flags_pass(self) -> None:
        result = self.run_checker(valid_commands())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_march_native_anywhere_fails(self) -> None:
        entries = valid_commands()
        entries.append(entry("src/vt/cpu/cpu_ops.cpp", "-march=native"))
        result = self.run_checker(entries)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("-march=native", result.stdout + result.stderr)

    def test_baseline_source_rejects_arch_extension_leak(self) -> None:
        entries = valid_commands()
        entries[1] = entry("src/vt/cpu/cpu_matmul_elem.cpp", "-march=armv8.2-a+i8mm")
        result = self.run_checker(entries)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cpu_matmul_elem.cpp", result.stdout + result.stderr)

    def test_dotprod_source_requires_exact_tier(self) -> None:
        entries = valid_commands()
        entries[4] = entry("src/vt/cpu/cpu_quant_dot_sdot.cpp", "-march=armv8.2-a")
        result = self.run_checker(entries)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("dotprod+fp16", result.stdout + result.stderr)

    def test_i8mm_sources_require_exact_tier(self) -> None:
        for index in (5, 6):
            with self.subTest(index=index):
                entries = valid_commands()
                source = entries[index]["file"]
                entries[index] = {
                    "file": source,
                    "command": f"c++ -march=armv8.2-a+dotprod -c {source}",
                }
                result = self.run_checker(entries)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("i8mm+dotprod", result.stdout + result.stderr)

    def test_required_tier_source_must_be_compiled(self) -> None:
        result = self.run_checker(valid_commands()[:-1])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
