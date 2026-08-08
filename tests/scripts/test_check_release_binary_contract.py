#!/usr/bin/env python3
"""Mutation tests for the accepted binary-release spike contract."""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-release-binary-contract.py"

CONTRACT_PATHS = (
    "scripts/check-release-binary-contract.py",
    ".agents/specs/release-binary-matrix.md",
    ".agents/engine-matrix.md",
    ".agents/roadmap_v1.md",
    ".agents/NOW.md",
    ".agents/coordination.md",
    ".agents/state.md",
    "docs/STATUS.md",
    "docs/BENCHMARKS.md",
    "tests/scripts/test_check_release_binary_contract.py",
)

REQUIRED_TEST_METHODS = (
    "test_repository_contract_passes",
    "test_spec_identity_is_fail_closed",
    "test_each_primary_cuda_sm_is_required",
    "test_primary_cuda_must_stay_one_fat_binary_per_host_abi",
    "test_per_sm_cuda_must_not_become_primary",
    "test_primary_cpu_must_stay_one_adaptive_binary_per_host_abi",
    "test_x86_64_baseline_must_not_require_avx2",
    "test_work_table_has_explicit_deps_column",
    "test_each_work_dependency_edge_is_pinned",
    "test_optional_w12_does_not_block_w13",
    "test_each_required_record_anchor_is_fail_closed",
    "test_release_lifecycle_and_honesty_are_fail_closed",
    "test_public_release_rows_remain_pending",
    "test_human_w12_is_optional_and_cannot_replace_w10",
    "test_human_primary_artifact_contract_matches_machine_block",
    "test_primary_cuda_mutation_inventory_literal_is_pinned",
    "test_work_dependency_mutation_inventory_literal_is_pinned",
    "test_each_semantic_inventory_consumer_is_pinned",
    "test_checker_guard_map_keysets_are_exact",
    "test_required_mutation_test_inventory_is_pinned",
)

PRIMARY_CUDA_SMS = (
    "80",
    "86",
    "87",
    "89",
    "90a",
    "100a",
    "103a",
    "110",
    "120a",
    "121a",
)

GUARD_MAP_KEYS = {
    "TEST_LITERAL_INVENTORIES": (
        "PRIMARY_CUDA_SMS",
        "EXPECTED_DEPS",
        "RECORD_ANCHORS",
        "LIFECYCLE_RECORD_MUTATIONS",
        "PUBLIC_PENDING_MUTATIONS",
        "W10_W12_HUMAN_MUTATIONS",
        "PRIMARY_ARTIFACT_PROSE_MUTATIONS",
        "GUARD_MAP_KEYS",
    ),
    "TEST_INVENTORY_CONSUMERS": (
        "PRIMARY_CUDA_SMS",
        "EXPECTED_DEPS",
        "RECORD_ANCHORS",
        "LIFECYCLE_RECORD_MUTATIONS",
        "PUBLIC_PENDING_MUTATIONS",
        "W10_W12_HUMAN_MUTATIONS",
        "PRIMARY_ARTIFACT_PROSE_MUTATIONS",
        "GUARD_MAP_KEYS",
    ),
}

RECORD_ANCHORS = {
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

LIFECYCLE_RECORD_MUTATIONS = (
    (
        ".agents/engine-matrix.md",
        "`SPIKE` | `CLAIM-ENG-RELEASE-BINARIES-SPIKE` |",
        "`DONE` | `CLAIM-ENG-RELEASE-BINARIES-SPIKE` |",
        "engine-matrix release lifecycle",
    ),
    (
        ".agents/engine-matrix.md",
        "gaps remain; no install/archive/publish implementation",
        "gaps closed; install/archive/publish implementation complete",
        "engine-matrix release lifecycle",
    ),
    (
        ".agents/roadmap_v1.md",
        "`SPIKE` | Fresh review of PR #129",
        "`DONE` | Fresh review of PR #129",
        "roadmap release lifecycle",
    ),
    (
        ".agents/roadmap_v1.md",
        "bundle work; no archive exists",
        "bundle work complete; archive exists",
        "roadmap release lifecycle",
    ),
    (
        ".agents/coordination.md",
        "| `ACTIVE` | 2026-08-07 — user-reviewed revision complete: primary fat "
        "CUDA + adaptive CPU per host ABI, optional per-SM diagnostics; row stays "
        "`SPIKE`; awaiting fresh review |",
        "| `DONE` | 2026-08-07 — user-reviewed revision complete: primary fat "
        "CUDA + adaptive CPU per host ABI, optional per-SM diagnostics; row stays "
        "`SPIKE`; awaiting fresh review |",
        "coordination release lifecycle",
    ),
    (
        ".agents/coordination.md",
        "no CMake, workflow, source, test, or artifact implementation",
        "CMake, workflow, source, test, and artifact implementation complete",
        "coordination release lifecycle",
    ),
    (
        ".agents/coordination.md",
        "row stays `SPIKE`; awaiting fresh review",
        "row is `DONE`; release shipped",
        "coordination release lifecycle",
    ),
    (
        ".agents/state.md",
        "`ENG-RELEASE-BINARIES` remains `SPIKE`, and no archive or implementation "
        "is\nclaimed.",
        "`ENG-RELEASE-BINARIES` is `DONE`, with archive and implementation.",
        "state release lifecycle",
    ),
)

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

PUBLIC_PENDING_MUTATIONS = (
    (
        "docs/BENCHMARKS.md",
        "**PENDING:** pins 10-SM fat CUDA, adaptive no-AVX2 CPU, "
        "W1-W13/W10-W12 policy, public pending states, and 20 tests. No archive, "
        "staged smoke, runtime, correctness, or performance evidence",
        "**SHIPPED:** archive, runtime, correctness, and performance evidence "
        "complete",
        "docs/BENCHMARKS.md release row",
    ),
    (
        "docs/STATUS.md",
        "Supported subset; bundles SPIKED, no artifacts",
        "Supported; bundles SHIPPED with runtime evidence",
        "docs/STATUS.md release row",
    ),
)

W10_W12_HUMAN_MUTATIONS = (
    (
        "optional single-SM CUDA diagnostic/performance variants",
        "required primary single-SM CUDA release variants replacing W10",
        "W12 deliverable",
    ),
    (
        "generated from the same explicit matrix and evidence; never advertised "
        "as the primary KISS download or used to bypass W10",
        "the primary KISS download; W10 may be bypassed",
        "W12 exit gate",
    ),
)

PRIMARY_ARTIFACT_PROSE_MUTATIONS = (
    (
        "The primary CPU download is\none conservative-baseline, runtime-adaptive "
        "binary per OS+host ABI; the primary\nCUDA download is one fat binary per "
        "OS+host ABI containing every supported SM.",
        "The primary CPU download is one binary per ISA; the primary CUDA "
        "download is one binary per SM.",
        "human primary CPU/CUDA contract",
    ),
    (
        "For x86_64, the baseline must run without AVX2: portable/SSE2 code "
        "remains\ncallable, and higher instructions live only in per-function or "
        "per-TU tiers.",
        "For x86_64, AVX2 is required by the baseline.",
        "human x86_64 no-AVX2 contract",
    ),
)


def run_checker(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(root / "scripts/check-release-binary-contract.py"),
            "--root",
            str(root),
        ],
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


def delete_test_method(root: Path, method: str) -> None:
    relative = "tests/scripts/test_check_release_binary_contract.py"
    path = root / relative
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    matches = [
        node
        for node in ast.walk(ast.parse(text))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one method {method!r}, found {len(matches)}")
    node = matches[0]
    start = min(
        [node.lineno, *(decorator.lineno for decorator in node.decorator_list)]
    ) - 1
    del lines[start : node.end_lineno]
    path.write_text("".join(lines), encoding="utf-8")


def mutate_checker_guard_map(
    root: Path, guard_map: str, mutation: str, key: str
) -> None:
    relative = "scripts/check-release-binary-contract.py"
    path = root / relative
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == guard_map
            for target in node.targets
        )
    ]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Dict):
        raise AssertionError(f"expected one literal guard map {guard_map!r}")
    mapping = assignments[0].value
    keys = [ast.literal_eval(node) for node in mapping.keys]

    if mutation == "add":
        if key in keys:
            raise AssertionError(f"guard key already exists in {guard_map}: {key}")
        mapping.keys.append(ast.Constant(value=key))
        mapping.values.append(ast.Constant(value=None))
    else:
        if keys.count(key) != 1:
            raise AssertionError(
                f"expected one guard key {key!r} in {guard_map}, found {keys.count(key)}"
            )
        index = keys.index(key)
        if mutation == "delete":
            del mapping.keys[index]
            del mapping.values[index]
        elif mutation == "rename":
            mapping.keys[index] = ast.Constant(value=f"RENAMED_{key}")
        else:
            raise AssertionError(f"unknown guard-map mutation {mutation!r}")

    path.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n", encoding="utf-8")


def bypass_checker_keyset_enforcement(root: Path) -> None:
    relative = "scripts/check-release-binary-contract.py"
    path = root / relative
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        left = node.test.left
        if (
            isinstance(left, ast.Call)
            and isinstance(left.func, ast.Name)
            and left.func.id == "tuple"
            and len(left.args) == 1
            and isinstance(left.args[0], ast.Name)
            and left.args[0].id == "guard_map"
        ):
            matches.append(node)
    if len(matches) != 1:
        raise AssertionError(
            f"expected one guard-map keyset enforcement, found {len(matches)}"
        )
    matches[0].test = ast.Constant(value=False)
    path.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n", encoding="utf-8")


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
        for sm in PRIMARY_CUDA_SMS:
            with self.subTest(sm=sm):
                self.assert_mutation_fails(
                    "primary_cuda_sms=80,86,87,89,90a,100a,103a,110,120a,121a",
                    "primary_cuda_sms=" + ",".join(
                        value for value in PRIMARY_CUDA_SMS if value != sm
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
        for relative, anchor in RECORD_ANCHORS.items():
            with self.subTest(path=relative), RepoCopy() as root:
                mutate(root, relative, anchor)
                result = run_checker(root)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(f"{relative} is missing required release anchor", result.stdout + result.stderr)

    def test_release_lifecycle_and_honesty_are_fail_closed(self) -> None:
        for relative, before, after, reason in LIFECYCLE_RECORD_MUTATIONS:
            with self.subTest(path=relative, mutation=before), RepoCopy() as root:
                mutate(root, relative, before, after)
                result = run_checker(root)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(reason, result.stdout + result.stderr)

        with RepoCopy() as root:
            mutate(
                root,
                "tests/scripts/test_check_release_binary_contract.py",
                '        "engine-matrix release lifecycle",',
                '        "renamed engine lifecycle",',
            )
            result = run_checker(root)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("LIFECYCLE_RECORD_MUTATIONS", result.stdout + result.stderr)


class HumanContractMutations(unittest.TestCase):
    SPEC = ".agents/specs/release-binary-matrix.md"

    def test_public_release_rows_remain_pending(self) -> None:
        for relative, before, after, reason in PUBLIC_PENDING_MUTATIONS:
            with self.subTest(path=relative), RepoCopy() as root:
                mutate(root, relative, before, after)
                result = run_checker(root)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(reason, result.stdout + result.stderr)

    def test_human_w12_is_optional_and_cannot_replace_w10(self) -> None:
        for before, after, reason in W10_W12_HUMAN_MUTATIONS:
            with self.subTest(reason=reason), RepoCopy() as root:
                mutate(root, self.SPEC, before, after)
                result = run_checker(root)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(reason, result.stdout + result.stderr)

    def test_human_primary_artifact_contract_matches_machine_block(self) -> None:
        for before, after, reason in PRIMARY_ARTIFACT_PROSE_MUTATIONS:
            with self.subTest(reason=reason), RepoCopy() as root:
                mutate(root, self.SPEC, before, after)
                result = run_checker(root)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(reason, result.stdout + result.stderr)

    def test_primary_cuda_mutation_inventory_literal_is_pinned(self) -> None:
        with RepoCopy() as root:
            mutate(
                root,
                "tests/scripts/test_check_release_binary_contract.py",
                '    "120a",\n    "121a",\n)\n\nGUARD_MAP_KEYS',
                '    "120a",\n)\n\nGUARD_MAP_KEYS',
            )
            result = run_checker(root)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PRIMARY_CUDA_SMS", result.stdout + result.stderr)

    def test_work_dependency_mutation_inventory_literal_is_pinned(self) -> None:
        with RepoCopy() as root:
            mutate(
                root,
                "tests/scripts/test_check_release_binary_contract.py",
                '    "W2": "W1",',
                '    "W2": "",',
            )
            result = run_checker(root)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("EXPECTED_DEPS", result.stdout + result.stderr)

    def test_each_semantic_inventory_consumer_is_pinned(self) -> None:
        mutations = (
            ("for sm in PRIMARY_CUDA_SMS:", "for sm in ():", "PRIMARY_CUDA_SMS"),
            (
                "for work, deps in EXPECTED_DEPS.items():",
                "for work, deps in {}.items():",
                "EXPECTED_DEPS",
            ),
            (
                "for relative, anchor in RECORD_ANCHORS.items():",
                "for relative, anchor in {}.items():",
                "RECORD_ANCHORS",
            ),
            (
                "for relative, before, after, reason in LIFECYCLE_RECORD_MUTATIONS:",
                "for relative, before, after, reason in ():",
                "LIFECYCLE_RECORD_MUTATIONS",
            ),
            (
                "for relative, before, after, reason in PUBLIC_PENDING_MUTATIONS:",
                "for relative, before, after, reason in ():",
                "PUBLIC_PENDING_MUTATIONS",
            ),
            (
                "for before, after, reason in W10_W12_HUMAN_MUTATIONS:",
                "for before, after, reason in ():",
                "W10_W12_HUMAN_MUTATIONS",
            ),
            (
                "for before, after, reason in PRIMARY_ARTIFACT_PROSE_MUTATIONS:",
                "for before, after, reason in ():",
                "PRIMARY_ARTIFACT_PROSE_MUTATIONS",
            ),
        )
        for before, after, inventory in mutations:
            with self.subTest(inventory=inventory), RepoCopy() as root:
                mutate(
                    root,
                    "tests/scripts/test_check_release_binary_contract.py",
                    before,
                    after,
                )
                result = run_checker(root)
                self.assertNotEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )
                self.assertIn(inventory, result.stdout + result.stderr)

    def test_checker_guard_map_keysets_are_exact(self) -> None:
        for guard_map, keys in GUARD_MAP_KEYS.items():
            for key in keys:
                for mutation in ("delete", "rename"):
                    with (
                        self.subTest(
                            guard_map=guard_map,
                            key=key,
                            mutation=mutation,
                        ),
                        RepoCopy() as root,
                    ):
                        mutate_checker_guard_map(root, guard_map, mutation, key)
                        result = run_checker(root)
                        self.assertNotEqual(
                            result.returncode, 0, result.stdout + result.stderr
                        )
                        self.assertIn(
                            "semantic mutation guard map keyset",
                            result.stdout + result.stderr,
                        )

            with self.subTest(guard_map=guard_map, mutation="add"), RepoCopy() as root:
                mutate_checker_guard_map(root, guard_map, "add", "EXTRA_GUARD")
                result = run_checker(root)
                self.assertNotEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )
                self.assertIn(
                    "semantic mutation guard map keyset",
                    result.stdout + result.stderr,
                )

            with (
                self.subTest(guard_map=guard_map, mutation="bypass"),
                RepoCopy() as root,
            ):
                mutate_checker_guard_map(root, guard_map, "delete", keys[0])
                bypass_checker_keyset_enforcement(root)
                result = run_checker(root)
                self.assertNotEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )
                self.assertIn(keys[0], result.stdout + result.stderr)

    def test_required_mutation_test_inventory_is_pinned(self) -> None:
        for method in REQUIRED_TEST_METHODS:
            for mutation in ("delete", "rename"):
                with self.subTest(method=method, mutation=mutation), RepoCopy() as root:
                    if mutation == "delete":
                        delete_test_method(root, method)
                    else:
                        mutate(
                            root,
                            "tests/scripts/test_check_release_binary_contract.py",
                            f"    def {method}(",
                            f"    def renamed_{method}(",
                        )
                    result = run_checker(root)
                    self.assertNotEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    self.assertIn(
                        "required mutation-test inventory",
                        result.stdout + result.stderr,
                    )


if __name__ == "__main__":
    unittest.main()
