#!/usr/bin/env python3
"""Regression gates for compiler-specific diagnostic pragmas."""

import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


class CompilerPragmaBoundaryTest(unittest.TestCase):
    def test_voxtral_gcc_only_warning_is_hidden_from_clang(self):
        source = (
            ROOT / "src/vllm/model_executor/models/voxtral.cpp"
        ).read_text(encoding="utf-8")
        guarded = re.search(
            r"#if defined\(__GNUC__\) && !defined\(__clang__\)"
            r"(?P<body>.*?)#endif",
            source,
            re.DOTALL,
        )
        self.assertIsNotNone(guarded)
        self.assertIn(
            '#pragma GCC diagnostic ignored "-Wstringop-overflow"',
            guarded.group("body"),
        )


if __name__ == "__main__":
    unittest.main()
