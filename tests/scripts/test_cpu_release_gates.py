#!/usr/bin/env python3
"""Executable contract for W9 CPU release-tier evidence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts/run-cpu-release-gates.py"


FAKE_TEST = """#!/bin/sh
if [ "${FAKE_POOR:-0}" = 1 ]; then
  case "${VT_CPU_MATMUL_TIER:-}${VT_CPU_Q8_DOT:-}${VT_CPU_QUANT_MMLA:-}${VT_CPU_QUANT_REPACK:-}" in
    *f16c*|*avx2*|*avx512*|*sdot*|*i8mm*) exit 17 ;;
  esac
fi
echo "selected ${VT_CPU_MATMUL_TIER:-${VT_CPU_Q8_DOT:-${VT_CPU_QUANT_MMLA:-${VT_CPU_QUANT_REPACK:-portable}}}}"
"""


FAKE_EMULATOR = """#!/bin/sh
export FAKE_POOR=1
test "$1" = -cpu
shift 2
exec "$@"
"""


class CpuReleaseGatesContract(unittest.TestCase):
    def fixture(self, scratch: Path) -> tuple[Path, Path]:
        tests = scratch / "tests"
        tests.mkdir(parents=True)
        for name in (
            "test_cpu_isa_arm",
            "test_ops_matmul_elem",
            "test_ops_quant_dot",
            "test_ops_quant_repack",
        ):
            path = tests / name
            path.write_text(FAKE_TEST, encoding="utf-8")
            path.chmod(0o755)
        emulator = scratch / "fake-emulator"
        emulator.write_text(FAKE_EMULATOR, encoding="utf-8")
        emulator.chmod(0o755)
        return tests, emulator

    def invoke(self, tests: Path, emulator: Path, arch: str, output: Path):
        return subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--arch",
                arch,
                "--tests-dir",
                str(tests),
                "--poor-emulator",
                str(emulator),
                "--output",
                str(output),
                "--evidence-url",
                "https://github.com/mudler/vllm.cpp/actions/runs/1",
            ],
            text=True,
            capture_output=True,
            env={**os.environ, "VLLM_CPP_RELEASE_GATE_TEST": "1"},
            check=False,
        )

    def test_x86_executes_all_tiers_and_poor_host_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tests, emulator = self.fixture(Path(temporary))
            output = Path(temporary) / "report.json"
            result = self.invoke(tests, emulator, "x86_64", output)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(
                list(report["tiers"]),
                ["portable-sse2", "sse2-f16c", "avx2-f16c", "avx512f"],
            )
            self.assertEqual({row["state"] for row in report["tiers"].values()}, {"passed"})
            self.assertTrue(any("Nehalem" in command for command in report["commands"]))
            self.assertTrue(any("expect-refusal" in command for command in report["commands"]))

    def test_arm_executes_dotprod_i8mm_and_poor_host_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tests, emulator = self.fixture(Path(temporary))
            output = Path(temporary) / "report.json"
            result = self.invoke(tests, emulator, "aarch64", output)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(list(report["tiers"]), ["portable-neon", "dotprod", "i8mm"])
            self.assertTrue(any("cortex-a53" in command for command in report["commands"]))
            self.assertIn("VT_CPU_Q8_DOT=sdot", report["tiers"]["dotprod"]["command"])
            self.assertIn("VT_CPU_QUANT_MMLA=i8mm", report["tiers"]["i8mm"]["command"])

    def test_missing_binary_or_accepted_unsupported_tier_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tests, emulator = self.fixture(Path(temporary))
            (tests / "test_ops_matmul_elem").unlink()
            result = self.invoke(tests, emulator, "x86_64", Path(temporary) / "report.json")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing test executable", result.stderr)
        with tempfile.TemporaryDirectory() as temporary:
            tests, emulator = self.fixture(Path(temporary))
            emulator.write_text("#!/bin/sh\nshift 2\nexec \"$@\"\n", encoding="utf-8")
            emulator.chmod(0o755)
            result = self.invoke(tests, emulator, "aarch64", Path(temporary) / "report.json")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unexpectedly accepted", result.stderr)


if __name__ == "__main__":
    unittest.main()
