#!/usr/bin/env python3
"""Unit and mutation checks for scripts/checker_text.py.

The structural checkers match RAW source text, so anything that LOOKS like a
statement satisfies them. This module's three normalizations are what makes
"the statement is present" mean "the compiler keeps it": a commented-out step, an
`#if 0`-ed step and an `if (false)`-ed step are all deletions, and all three read
as the statement to a bare `re.search`. Every case below therefore checks BOTH
directions — the disguised deletion is removed, AND the live code beside it is not.

POSITION PRESERVATION IS PART OF THE CONTRACT, not an implementation detail: the
fp4 resident checker reports `file:line` and orders statements by OFFSET, so a
normalization that shortened the text would silently corrupt both. The two private
copies this module replaces collapsed each comment to a single space and could not
have been used for that.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "scripts/checker_text.py"
SPEC = importlib.util.spec_from_file_location("checker_text", MODULE)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)

blank_out = mod.blank_out
match_braces = mod.match_braces
strip_comments = mod.strip_comments
strip_never_taken_branches = mod.strip_never_taken_branches
strip_preprocessor_disabled = mod.strip_preprocessor_disabled
normalize_source = mod.normalize_source

STATEMENT = "AdoptDeviceBytesAsHost(d.b, w.packed);"


class PositionPreservationTests(unittest.TestCase):
    """Same length, same newlines — so offsets and line numbers survive."""

    def check_shape(self, raw: str, out: str) -> None:
        self.assertEqual(len(out), len(raw))
        self.assertEqual(out.count("\n"), raw.count("\n"))

    def test_line_comment_keeps_length_and_lines(self) -> None:
        raw = "a();\n// gone\nb();\n"
        out = strip_comments(raw)
        self.check_shape(raw, out)
        self.assertNotIn("gone", out)
        self.assertIn("a();", out)
        self.assertIn("b();", out)

    def test_multiline_block_comment_keeps_its_newlines(self) -> None:
        # The collapse-to-one-space copies dropped these newlines, which would move
        # every reported line number in the rest of the file.
        raw = "a();\n/* one\n   two\n   three */\nb();\n"
        out = strip_comments(raw)
        self.check_shape(raw, out)
        self.assertNotIn("two", out)
        self.assertEqual(out.splitlines()[4], "b();")

    def test_a_statements_offset_is_unchanged(self) -> None:
        raw = f"/* {STATEMENT} */\nreal();\n{STATEMENT}\n"
        out = normalize_source(raw)
        self.assertEqual(out.index(STATEMENT), raw.rindex(STATEMENT))

    def test_every_normalization_preserves_shape(self) -> None:
        raw = (
            "live_one();\n"
            "// commented\n"
            "#if 0\n"
            "disabled();\n"
            "#endif\n"
            "if (false) { dead(); }\n"
            "live_two();\n"
        )
        self.check_shape(raw, normalize_source(raw))


class StripCommentsTests(unittest.TestCase):
    def test_a_commented_statement_is_gone_but_the_real_one_stays(self) -> None:
        out = strip_comments(f"  // {STATEMENT}\n  {STATEMENT}\n")
        self.assertEqual(out.count(STATEMENT), 1)

    def test_block_comment_on_one_line(self) -> None:
        self.assertNotIn(STATEMENT, strip_comments(f"/* {STATEMENT} */"))

    def test_a_block_comment_containing_a_line_comment(self) -> None:
        self.assertNotIn("x", strip_comments("/* // x */"))

    def test_a_line_comment_containing_a_block_open_is_not_a_block(self) -> None:
        # `// /*` must not swallow the rest of the file looking for a `*/`.
        out = strip_comments("// /* not a block\nkeep_me();\n")
        self.assertIn("keep_me();", out)

    def test_code_without_comments_is_untouched(self) -> None:
        raw = "int a = b / c;\nint d = e * f;\n"
        self.assertEqual(strip_comments(raw), raw)

    def test_division_is_not_a_comment(self) -> None:
        self.assertIn("a / b", strip_comments("x = a / b;"))


class StripPreprocessorDisabledTests(unittest.TestCase):
    def test_if_0_region_is_removed(self) -> None:
        out = strip_preprocessor_disabled(f"#if 0\n{STATEMENT}\n#endif\n")
        self.assertNotIn(STATEMENT, out)

    def test_if_false_region_is_removed(self) -> None:
        out = strip_preprocessor_disabled(f"#if false\n{STATEMENT}\n#endif\n")
        self.assertNotIn(STATEMENT, out)

    def test_indented_and_spaced_directives(self) -> None:
        out = strip_preprocessor_disabled(f"  #  if 0\n{STATEMENT}\n  #  endif\n")
        self.assertNotIn(STATEMENT, out)

    def test_a_nested_ifdef_does_not_close_the_region_early(self) -> None:
        # The inner `#endif` belongs to the inner `#ifdef`; if it were taken for the
        # outer one, everything after it would be treated as live again.
        raw = f"#if 0\n#ifdef FOO\nx();\n#endif\n{STATEMENT}\n#endif\nlive();\n"
        out = strip_preprocessor_disabled(raw)
        self.assertNotIn(STATEMENT, out)
        self.assertNotIn("x();", out)
        self.assertIn("live();", out)

    def test_the_else_branch_of_if_0_is_live(self) -> None:
        raw = f"#if 0\ndead();\n#else\n{STATEMENT}\n#endif\n"
        out = strip_preprocessor_disabled(raw)
        self.assertNotIn("dead();", out)
        self.assertIn(STATEMENT, out)

    def test_an_elif_branch_is_treated_as_live(self) -> None:
        # Deliberately conservative: whether the elif is taken needs macro state.
        raw = f"#if 0\ndead();\n#elif defined(FOO)\n{STATEMENT}\n#endif\n"
        out = strip_preprocessor_disabled(raw)
        self.assertNotIn("dead();", out)
        self.assertIn(STATEMENT, out)

    def test_a_real_build_configuration_is_left_alone(self) -> None:
        # THE DECLINED CASE, pinned so it cannot drift into being removed: an
        # `#ifdef` region is a build configuration, not a disguised deletion, and
        # deciding it needs the build's macro state.
        raw = f"#ifdef VT_CUTLASS_NVFP4\n{STATEMENT}\n#endif\n"
        self.assertEqual(strip_preprocessor_disabled(raw), raw)
        raw2 = f"#if VLLM_CPP_VERSION > 2\n{STATEMENT}\n#endif\n"
        self.assertEqual(strip_preprocessor_disabled(raw2), raw2)

    def test_if_00_and_if_0_plus_x_are_not_literal_false(self) -> None:
        for cond in ("#if 00", "#if 0 + 1", "#if 0x0"):
            raw = f"{cond}\n{STATEMENT}\n#endif\n"
            self.assertEqual(strip_preprocessor_disabled(raw), raw, cond)

    def test_an_unterminated_if_0_disables_to_end_of_file(self) -> None:
        out = strip_preprocessor_disabled(f"#if 0\n{STATEMENT}\n")
        self.assertNotIn(STATEMENT, out)

    def test_two_disabled_regions(self) -> None:
        raw = f"#if 0\na();\n#endif\nlive();\n#if 0\n{STATEMENT}\n#endif\n"
        out = strip_preprocessor_disabled(raw)
        self.assertNotIn("a();", out)
        self.assertNotIn(STATEMENT, out)
        self.assertIn("live();", out)


class StripNeverTakenBranchesTests(unittest.TestCase):
    def test_if_false_body_is_removed(self) -> None:
        out = strip_never_taken_branches(f"if (false) {{ {STATEMENT} }}\n")
        self.assertNotIn(STATEMENT, out)

    def test_if_0_body_is_removed(self) -> None:
        self.assertNotIn(STATEMENT, strip_never_taken_branches(f"if (0) {{ {STATEMENT} }}"))

    def test_the_else_branch_survives(self) -> None:
        out = strip_never_taken_branches(f"if (false) {{ dead(); }} else {{ {STATEMENT} }}")
        self.assertNotIn("dead();", out)
        self.assertIn(STATEMENT, out)

    def test_nested_braces_inside_the_dead_branch(self) -> None:
        raw = f"if (false) {{ for (;;) {{ x(); }} {STATEMENT} }}\nlive();\n"
        out = strip_never_taken_branches(raw)
        self.assertNotIn(STATEMENT, out)
        self.assertNotIn("x();", out)
        self.assertIn("live();", out)

    def test_a_live_condition_is_untouched(self) -> None:
        for raw in (f"if (ok) {{ {STATEMENT} }}", f"if (!w.d_packed) {{ {STATEMENT} }}"):
            self.assertEqual(strip_never_taken_branches(raw), raw)

    def test_a_dead_branch_inside_a_dead_branch_terminates(self) -> None:
        out = strip_never_taken_branches(f"if (false) {{ if (false) {{ {STATEMENT} }} }}")
        self.assertNotIn(STATEMENT, out)


class MatchBracesTests(unittest.TestCase):
    def test_returns_the_index_past_the_closing_brace(self) -> None:
        text = "{a{b}c}tail"
        self.assertEqual(text[match_braces(text, 1) :], "tail")

    def test_unbalanced_input_stops_at_end(self) -> None:
        text = "{a{b}"
        self.assertEqual(match_braces(text, 1), len(text))


class NormalizeSourceTests(unittest.TestCase):
    def test_all_three_disguises_at_once(self) -> None:
        raw = (
            f"// {STATEMENT}\n"
            f"/* {STATEMENT} */\n"
            f"#if 0\n{STATEMENT}\n#endif\n"
            f"if (false) {{ {STATEMENT} }}\n"
            f"{STATEMENT}\n"
        )
        self.assertEqual(normalize_source(raw).count(STATEMENT), 1)

    def test_it_is_idempotent(self) -> None:
        # `file_violations` normalizes even text `main` already normalized.
        raw = (
            f"// {STATEMENT}\n#if 0\nx();\n#endif\nif (false) {{ y(); }}\n{STATEMENT}\n"
        )
        once = normalize_source(raw)
        self.assertEqual(normalize_source(once), once)

    def test_a_comment_cannot_hide_an_if_0(self) -> None:
        # Comments go first, so `// #if 0` never opens a disabled region.
        raw = f"// #if 0\n{STATEMENT}\n"
        self.assertIn(STATEMENT, normalize_source(raw))

    def test_blank_out_keeps_newlines(self) -> None:
        self.assertEqual(blank_out("ab\ncd"), "  \n  ")


class LiveTreeTests(unittest.TestCase):
    def test_every_importer_still_normalizes_the_real_tree_green(self) -> None:
        # The three checkers that share this helper must stay green on the tree it
        # normalizes; a change here that broke one of them would otherwise only
        # surface in preflight.
        import subprocess

        for checker in (
            "scripts/check-fp4-resident-consistency.py",
            "scripts/check-runner-routing-consistency.py",
            "scripts/check-surface-coverage.py",
        ):
            r = subprocess.run(
                [sys.executable, str(ROOT / checker)],
                capture_output=True,
                text=True,
                cwd=ROOT,
            )
            self.assertEqual(r.returncode, 0, checker + "\n" + r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
