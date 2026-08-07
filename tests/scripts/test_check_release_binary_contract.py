#!/usr/bin/env python3
"""Mutation tests for the accepted binary-release spike contract."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-release-binary-contract.py"

CONTRACT_PATHS = (
    ".agents/specs/release-binary-matrix.md",
    ".agents/engine-matrix.md",
    ".agents/roadmap_v1.md",
    ".agents/NOW.md",
    ".agents/coordination.md",
    ".agents/state.md",
    "docs/STATUS.md",
    "docs/BENCHMARKS.md",
)

ANCHORS = {
    ".agents/engine-matrix.md": "| `ENG-RELEASE-BINARIES` |",
    ".agents/roadmap_v1.md": "| REL | `ROAD-V1-RELEASE` |",
    ".agents/NOW.md": "| Release | SPIKE | #129 |",
    ".agents/coordination.md": (
        "| `CLAIM-ENG-RELEASE-BINARIES-SPIKE` | "
        "`ENG-RELEASE-BINARIES` |"
    ),
    ".agents/state.md": (
        "## 2026-08-07 — Release matrix revised: fat CUDA and adaptive CPU "
        "are the primary downloads"
    ),
    "docs/STATUS.md": (
        "[Release spike](../.agents/specs/release-binary-matrix.md)"
    ),
    "docs/BENCHMARKS.md": (
        "| **Binary release matrix (spiked)** | `ENG-RELEASE-BINARIES`:"
    ),
}

EXPECTED_DEPS = {
    "W1": "",
    "W2": "W1",
    "W3": "",
    "W4": "",
    "W5": "",
    "W6": "",
    "W7": "W1,W2,W3,W4,W5,W6",
    "W8": "W5,W7",
    "W9": "W3,W4,W5,W6,W7",
    "W10": "W1,W2,W5,W6,W7",
    "W11": "W5,W6,W7",
    "W12": "W1,W2,W5,W6,W7",
    "W13": "W5,W7,W8,W9,W10,W11",
}


def run_checker(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


class RepoCopy:
    def __enter__(self) -> Path:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        for relative in CONTRACT_PATHS:
            source = ROOT / relative
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        return root

    def __exit__(self, *unused: object) -> None:
        self._tmp.cleanup()


def mutate(root: Path, relative: str, before: str, after: str = "") -> None:
    path = root / relative
    text = path.read_text(encoding="utf-8")
    if before not in text:
        raise AssertionError(f"mutation target missing in {relative}: {before!r}")
    path.write_text(text.replace(before, after, 1), encoding="utf-8")


class LiveContract(unittest.TestCase):
    def test_repository_contract_passes(self) -> None:
        result = run_checker(ROOT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class AcceptedDesignMutations(unittest.TestCase):
    SPEC = ".agents/specs/release-binary-matrix.md"

    def assert_mutation_fails(self, before: str, after: str, reason: str) -> None:
        with RepoCopy() as root:
            mutate(root, self.SPEC, before, after)
            result = run_checker(root)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(reason, result.stdout + result.stderr)

    def test_spec_identity_is_fail_closed(self) -> None:
        self.assert_mutation_fails(
            "identity=ENG-RELEASE-BINARIES",
            "identity=ENG-RELEASE-ARCHIVES",
            "identity",
        )

    def test_each_primary_cuda_sm_is_required(self) -> None:
        for sm in ("80", "86", "87", "89", "90a", "100a", "103a", "110", "120a", "121a"):
            with self.subTest(sm=sm):
                self.assert_mutation_fails(
                    "primary_cuda_sms=80,86,87,89,90a,100a,103a,110,120a,121a",
                    "primary_cuda_sms=" + ",".join(
                        value
                        for value in ("80", "86", "87", "89", "90a", "100a", "103a", "110", "120a", "121a")
                        if value != sm
                    ),
                    "primary CUDA SM set",
                )

    def test_primary_cuda_must_stay_one_fat_binary_per_host_abi(self) -> None:
        self.assert_mutation_fails(
            "primary_cuda_artifact=one-fat-binary-per-os-host-abi",
            "primary_cuda_artifact=one-binary-per-sm",
            "primary CUDA artifact",
        )

    def test_per_sm_cuda_must_not_become_primary(self) -> None:
        self.assert_mutation_fails(
            "per_sm_cuda=optional-non-primary",
            "per_sm_cuda=primary",
            "per-SM CUDA",
        )

    def test_primary_cpu_must_stay_one_adaptive_binary_per_host_abi(self) -> None:
        self.assert_mutation_fails(
            "primary_cpu_artifact=one-adaptive-binary-per-os-host-abi",
            "primary_cpu_artifact=one-binary-per-isa",
            "primary CPU artifact",
        )

    def test_x86_64_baseline_must_not_require_avx2(self) -> None:
        self.assert_mutation_fails(
            "x86_64_baseline=portable-sse2-without-avx2",
            "x86_64_baseline=avx2-required",
            "x86_64 baseline",
        )


class WorkGraphMutations(unittest.TestCase):
    SPEC = ".agents/specs/release-binary-matrix.md"

    def test_work_table_has_explicit_deps_column(self) -> None:
        with RepoCopy() as root:
            mutate(root, self.SPEC, "| Work | Deps | Deliverable | Exit gate |", "| Work | Deliverable | Exit gate |")
            result = run_checker(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Deps column", result.stdout + result.stderr)

    def test_each_work_dependency_edge_is_pinned(self) -> None:
        for work, deps in EXPECTED_DEPS.items():
            with self.subTest(work=work), RepoCopy() as root:
                mutate(root, self.SPEC, f"work_{work}={deps}", f"work_{work}=BROKEN")
                result = run_checker(root)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"{work} dependencies", result.stdout + result.stderr)

    def test_optional_w12_does_not_block_w13(self) -> None:
        with RepoCopy() as root:
            mutate(root, self.SPEC, "work_W13=W5,W7,W8,W9,W10,W11", "work_W13=W5,W7,W8,W9,W10,W11,W12")
            result = run_checker(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("W13 dependencies", result.stdout + result.stderr)


class RecordAnchorMutations(unittest.TestCase):
    def test_each_required_record_anchor_is_fail_closed(self) -> None:
        for relative, anchor in ANCHORS.items():
            with self.subTest(path=relative), RepoCopy() as root:
                mutate(root, relative, anchor)
                result = run_checker(root)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(f"{relative} is missing required release anchor", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
