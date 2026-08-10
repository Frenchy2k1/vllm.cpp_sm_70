#!/usr/bin/env python3
"""Mutation tests for the arrival rule: every change lands on a task branch.

The checker's own docstring is the authority on what it enforces. These tests
pin the two regimes it now has:

* pre-worktree-cutover, integration paths (scripts/, .agents/, docs/, .github/,
  AGENTS.md) are EXEMPT, because that history was made under the direct-push
  rule and reddening it retroactively would be dishonest;
* post-cutover, nothing is exempt -- every tracked path must arrive on a task
  branch, which is what makes "the shared checkout is never a work surface"
  enforceable rather than merely written down.

The `govern_integration=True` cases are the red-before evidence: on the parent
commit the parameter does not exist and the exemption is unconditional.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location(
    "check_role_discipline", ROOT / "scripts/check-role-discipline.py"
)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


DIRECT = dict(parents=["a" * 40], subject="policy: tighten a rule", body="")


def violations(paths, *, govern_integration=False, **overrides):
    """Run the arrival rule over *paths* for a single-parent direct commit.

    The keyword is omitted unless it is needed, so the legacy-regime cases call
    the pre-change signature and stay green on both sides of the cutover; only
    the worktree-regime cases depend on the new parameter existing.
    """
    call = {**DIRECT, **overrides}
    extra = {"govern_integration": True} if govern_integration else {}
    return checker.policy_commit_violations(
        "deadbee",
        call["parents"],
        call["subject"],
        call["body"],
        paths,
        (),
        **extra,
    )


class IntegrationPathClassification(unittest.TestCase):
    def test_integration_trees_and_top_files_are_integration(self) -> None:
        for path in (
            "scripts/agent-start.py",
            "scripts/check-role-discipline.py",
            ".agents/NOW.md",
            ".agents/specs/worktree-isolation.md",
            "docs/STATUS.md",
            ".github/workflows/ci.yml",
            "tests/scripts/test_check_role_discipline.py",
            "AGENTS.md",
            "CLAUDE.md",
            "README.md",
        ):
            with self.subTest(path=path):
                self.assertTrue(checker.is_integration_path(path))

    def test_product_paths_are_not_integration(self) -> None:
        for path in (
            "src/engine.cpp",
            "include/vllm.h",
            "tests/vt/test_gemv.cpp",
            "CMakeLists.txt",
            "cmake/cuda.cmake",
        ):
            with self.subTest(path=path):
                self.assertFalse(checker.is_integration_path(path))
                self.assertTrue(checker.is_feature_path(path))

    def test_malformed_paths_fail_closed_as_feature(self) -> None:
        for path in ("/etc/passwd", "../escape.cpp", "a//b.cpp", "", "src\\win.cpp"):
            with self.subTest(path=path):
                self.assertTrue(checker.is_feature_path(path))


class LegacyRegimeKeepsTheIntegrationExemption(unittest.TestCase):
    """Before the worktree cutover, a direct integration push is allowed."""

    def test_integration_only_commit_is_exempt(self) -> None:
        self.assertEqual(violations(["AGENTS.md", ".agents/NOW.md"]), [])

    def test_feature_path_is_still_governed(self) -> None:
        problems = violations(["src/engine.cpp"])
        self.assertEqual(len(problems), 1)
        self.assertIn("src/engine.cpp", problems[0])


class WorktreeRegimeGovernsEverything(unittest.TestCase):
    """RED BEFORE: on the parent commit this parameter does not exist."""

    def test_integration_only_commit_is_now_governed(self) -> None:
        problems = violations(["AGENTS.md", ".agents/NOW.md"], govern_integration=True)
        self.assertEqual(len(problems), 1)
        self.assertIn("AGENTS.md", problems[0])
        self.assertIn("task branch", problems[0])

    def test_a_checker_repair_may_no_longer_go_straight_to_main(self) -> None:
        problems = violations(
            ["scripts/check-role-discipline.py"], govern_integration=True
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("shared checkout", problems[0])

    def test_arriving_on_a_row_branch_satisfies_the_rule(self) -> None:
        self.assertEqual(
            violations(
                ["AGENTS.md"],
                govern_integration=True,
                subject="policy: worktree isolation (#210)",
            ),
            [],
        )

    def test_a_merge_naming_the_row_branch_satisfies_the_rule(self) -> None:
        self.assertEqual(
            violations(
                ["AGENTS.md", "scripts/check-role-discipline.py"],
                govern_integration=True,
                parents=["a" * 40, "b" * 40],
                subject="Merge branch 'row/POLICY-WORK-WORKTREE'",
            ),
            [],
        )

    def test_preview_truncates_and_counts_the_remainder(self) -> None:
        paths = [f"docs/d{i}.md" for i in range(7)]
        problems = violations(paths, govern_integration=True)
        self.assertIn("... (+3)", problems[0])


class ArrivalDetection(unittest.TestCase):
    def test_squash_merge_carrying_a_pr_number_arrives(self) -> None:
        self.assertTrue(
            checker.arrives_via_row_pr(["a" * 40], "fix(ci): unbreak it (#194)", "")
        )

    def test_plain_direct_commit_does_not_arrive(self) -> None:
        self.assertFalse(
            checker.arrives_via_row_pr(["a" * 40], "fix: quick repair", "")
        )

    def test_synthetic_merge_inherits_arrival_from_its_second_parent(self) -> None:
        # GitHub's refs/pull/N/merge names neither the row nor the PR; the
        # reviewed content is the second parent, one hop away.
        self.assertTrue(
            checker.arrives_via_row_pr(
                ["a" * 40, "b" * 40],
                "Merge 1234567 into 89abcde",
                "",
                ("policy: tighten arrival\n\nRow: row/POLICY-WORK-WORKTREE",),
            )
        )

    def test_synthetic_merge_of_an_unnamed_branch_does_not_arrive(self) -> None:
        self.assertFalse(
            checker.arrives_via_row_pr(
                ["a" * 40, "b" * 40],
                "Merge 1234567 into 89abcde",
                "",
                ("wip: local scratch",),
            )
        )


class CutoverWiring(unittest.TestCase):
    def test_role_cutover_is_a_full_sha(self) -> None:
        self.assertRegex(checker.ROLE_DISCIPLINE_SINCE or "", r"\A[0-9a-f]{40}\Z")

    def test_worktree_cutover_is_unset_or_a_full_sha(self) -> None:
        since = checker.WORKTREE_DISCIPLINE_SINCE
        if since is not None:
            self.assertRegex(since, r"\A[0-9a-f]{40}\Z")

    def test_cutover_shas_resolve_to_real_commits(self) -> None:
        """A dangling cutover SHA fails OPEN, silently disabling the rule.

        `_since` swallows CalledProcessError and returns False, so a typo'd or
        rebased-away SHA turns enforcement off with a green gate -- exactly the
        "turn a red gate green by widening a scope" failure AGENTS.md forbids.
        """
        for name in ("ROLE_DISCIPLINE_SINCE", "WORKTREE_DISCIPLINE_SINCE"):
            since = getattr(checker, name)
            if since is None:
                continue
            with self.subTest(cutover=name):
                resolved = subprocess.run(
                    ["git", "cat-file", "-e", f"{since}^{{commit}}"],
                    cwd=ROOT,
                    capture_output=True,
                )
                self.assertEqual(
                    resolved.returncode, 0, f"{name}={since} is not a commit in this repo"
                )

    def test_unset_cutover_never_enforces(self) -> None:
        original = checker.WORKTREE_DISCIPLINE_SINCE
        checker.WORKTREE_DISCIPLINE_SINCE = None
        try:
            self.assertFalse(checker.worktree_enforced("HEAD"))
        finally:
            checker.WORKTREE_DISCIPLINE_SINCE = original


if __name__ == "__main__":
    unittest.main()
