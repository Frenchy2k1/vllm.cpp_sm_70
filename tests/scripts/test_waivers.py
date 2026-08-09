#!/usr/bin/env python3
"""Mutation tests for scripts/waivers.py.

A waiver is the one sanctioned way to pass a gate you did not satisfy, so every
guard here is proved by a mutation that makes the file invalid and asserts the
loader rejects it. A waiver loader that accepts anything is worse than none.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.waivers import HEADER, exact_waiver, load_waivers  # noqa: E402


VALID = (
    "waiver_id,checker,scope,owner,reason,evidence,expires\n"
    "W-1,scripts/check-pr-size.py,pr:128,maintainer,one-time migration,PR-128,2999-01-01\n"
)
TODAY = date(2026, 8, 9)


class WaiverLoaderTest(unittest.TestCase):
    def load(self, text: str):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".agents").mkdir()
            (root / ".agents/waivers.csv").write_text(text, encoding="utf-8")
            return load_waivers(root, today=TODAY)

    def test_valid_file_loads(self):
        waivers = self.load(VALID)
        self.assertEqual(len(waivers), 1)
        self.assertEqual(waivers[0].checker, "scripts/check-pr-size.py")

    def test_missing_file_is_empty_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_waivers(Path(tmp), today=TODAY), ())

    def test_expired_waiver_is_rejected(self):
        text = VALID.replace("2999-01-01", "2026-08-08")
        with self.assertRaisesRegex(ValueError, "expired"):
            self.load(text)

    def test_non_iso_expiry_is_rejected(self):
        text = VALID.replace("2999-01-01", "next tuesday")
        with self.assertRaisesRegex(ValueError, "not an ISO date"):
            self.load(text)

    def test_wildcard_scope_is_rejected(self):
        text = VALID.replace("pr:128", "pr:*")
        with self.assertRaisesRegex(ValueError, "wildcard"):
            self.load(text)

    def test_repository_wide_scope_is_rejected(self):
        text = VALID.replace("pr:128", "repository")
        with self.assertRaisesRegex(ValueError, "wildcard or repository-wide"):
            self.load(text)

    def test_empty_cell_is_rejected(self):
        text = VALID.replace("maintainer", "")
        with self.assertRaisesRegex(ValueError, "empty cell"):
            self.load(text)

    def test_duplicate_id_is_rejected(self):
        text = VALID + VALID.splitlines()[1] + "\n"
        with self.assertRaisesRegex(ValueError, "duplicate waiver id"):
            self.load(text)

    def test_wrong_header_is_rejected(self):
        text = VALID.replace("checker", "rule_id", 1)
        with self.assertRaisesRegex(ValueError, "header must be exactly"):
            self.load(text)

    def test_short_row_is_rejected(self):
        text = VALID + "W-2,scripts/check-pr-size.py,pr:1\n"
        with self.assertRaisesRegex(ValueError, "cells"):
            self.load(text)

    def test_exact_waiver_matches_checker_and_scope(self):
        waivers = self.load(VALID)
        self.assertIsNotNone(
            exact_waiver(waivers, "scripts/check-pr-size.py", "pr:128")
        )
        # A waiver for one checker never covers another.
        self.assertIsNone(exact_waiver(waivers, "scripts/check-policy.py", "pr:128"))
        # ... nor another target of the same checker.
        self.assertIsNone(
            exact_waiver(waivers, "scripts/check-pr-size.py", "pr:129")
        )


class RepositoryWaiversTest(unittest.TestCase):
    def test_tracked_waivers_file_is_valid(self):
        """The real .agents/waivers.csv must always load."""
        self.assertEqual(
            tuple(HEADER),
            tuple(
                (ROOT / ".agents/waivers.csv")
                .read_text(encoding="utf-8")
                .splitlines()[0]
                .split(",")
            ),
        )
        load_waivers(ROOT)


if __name__ == "__main__":
    unittest.main()
