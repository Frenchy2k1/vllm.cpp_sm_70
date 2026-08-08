#!/usr/bin/env python3
"""Exact waiver selection tests at the enforcement boundary."""

from __future__ import annotations

import datetime as dt
import importlib.util
import tempfile
import unittest
from pathlib import Path

from scripts.policy_contract import Waiver


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "check_commit_trailers", ROOT / "scripts/check-commit-trailers.py"
)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class ExactWaiverSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.waiver = Waiver(
            waiver_id="WAIVER-TRAILER-001",
            rule_id="POL-COMMIT-TRAILERS",
            scope="commit:" + "a" * 40,
            owner="maintainer",
            reason="bounded migration",
            evidence="PR-128",
            expires=dt.date(2026, 8, 9),
        )

    def test_rule_and_complete_scope_must_match_exactly(self) -> None:
        self.assertEqual(
            checker.exact_waiver(
                [self.waiver], "POL-COMMIT-TRAILERS", "commit:" + "a" * 40
            ),
            self.waiver,
        )
        for rule, scope in (
            ("POL-COMMIT-TRAILER", "commit:" + "a" * 40),
            ("POL-COMMIT-TRAILERS-EXTRA", "commit:" + "a" * 40),
            ("POL-COMMIT-TRAILERS", "commit:" + "a" * 39),
            ("POL-COMMIT-TRAILERS", "commit:" + "a" * 40 + "b"),
            ("POL-COMMIT-TRAILERS", "path:commit:" + "a" * 40),
        ):
            with self.subTest(rule=rule, scope=scope):
                self.assertIsNone(checker.exact_waiver([self.waiver], rule, scope))

    def test_duplicate_applicable_waivers_fail_closed(self) -> None:
        duplicate = Waiver(
            **{**self.waiver.__dict__, "waiver_id": "WAIVER-TRAILER-002"}
        )
        with self.assertRaises(ValueError):
            checker.exact_waiver(
                [self.waiver, duplicate],
                "POL-COMMIT-TRAILERS",
                "commit:" + "a" * 40,
            )

    def test_path_waiver_may_name_a_file_but_not_an_existing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src" / "one.cpp").write_text("x", encoding="utf-8")
            directory_waiver = Waiver(
                **{**self.waiver.__dict__, "scope": "path:src"}
            )
            file_waiver = Waiver(
                **{**self.waiver.__dict__, "scope": "path:src/one.cpp"}
            )
            with self.assertRaisesRegex(ValueError, "directory"):
                checker.validate_waiver_targets(root, [directory_waiver])
            checker.validate_waiver_targets(root, [file_waiver])


if __name__ == "__main__":
    unittest.main()
