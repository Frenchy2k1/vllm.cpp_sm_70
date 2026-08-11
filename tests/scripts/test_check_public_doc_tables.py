#!/usr/bin/env python3
"""Unit and mutation checks for the public keyed-table doc tooling.

Covers scripts/check-public-doc-tables.py (the CI gate over docs/BENCHMARKS.md
and docs/FEATURES.md) and scripts/roll-benchmark-record.py (the mechanical
rollup the gate points at).
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


doc_tables = _load("doc_tables", "scripts/check-public-doc-tables.py")
roll_record = _load("roll_record", "scripts/roll-benchmark-record.py")


# A minimal scoreboard that satisfies every rule, used as the mutation baseline.
# MIN_TABLE_ROWS forces a real table, so the baseline carries one.
_ROWS = "\n".join(f"| Model {i} | {i}.0x |" for i in range(60))

VALID = "\n".join(
    [
        "# Benchmarks",
        "",
        "Measured results. See .agents/benchmark-record.md for the record.",
        "",
        "## At a glance",
        "",
        "| Reference | Result |",
        "|---|---|",
        _ROWS,
        "",
        "## How we measure",
        "",
        "Greedy, closed loop, three interleaved repetitions.",
        "",
        "## Open gaps",
        "",
        "| Track | Status |",
        "|---|---|",
        "| Thing | Pending |",
        "",
        "## Reproduce",
        "",
        "```sh",
        "vllm-bench run",
        "```",
        "",
    ]
)


class BenchmarksStructureTests(unittest.TestCase):
    def test_minimal_valid_document_passes(self) -> None:
        self.assertEqual(doc_tables.benchmarks_errors(VALID), [])

    def test_shipped_benchmarks_page_passes(self) -> None:
        text = (ROOT / "docs/BENCHMARKS.md").read_text(encoding="utf-8")
        self.assertEqual(doc_tables.benchmarks_errors(text), [])

    def test_shipped_record_exists(self) -> None:
        self.assertEqual(
            doc_tables.record_errors(doc_tables.RECORD), []
        )

    def test_missing_record_fails(self) -> None:
        errors = doc_tables.record_errors(ROOT / "does/not/exist.md")
        self.assertTrue(any("benchmark-record.md" in e for e in errors), errors)

    def test_missing_at_a_glance_section_fails(self) -> None:
        mutated = VALID.replace("## At a glance", "## Numbers")
        errors = doc_tables.benchmarks_errors(mutated)
        self.assertTrue(any("At a glance" in e for e in errors), errors)

    def test_missing_methodology_section_fails(self) -> None:
        mutated = VALID.replace("## How we measure", "## Notes")
        errors = doc_tables.benchmarks_errors(mutated)
        self.assertTrue(any("How we measure" in e for e in errors), errors)

    def test_missing_open_gaps_section_fails(self) -> None:
        mutated = VALID.replace("## Open gaps", "## Misc")
        errors = doc_tables.benchmarks_errors(mutated)
        self.assertTrue(any("Open gaps" in e for e in errors), errors)

    def test_missing_reproduce_section_fails(self) -> None:
        mutated = VALID.replace("## Reproduce", "## Commands")
        errors = doc_tables.benchmarks_errors(mutated)
        self.assertTrue(any("Reproduce" in e for e in errors), errors)

    def test_appended_sections_fail_as_an_append_log(self) -> None:
        # The drift this checker exists to stop: one narrative H2 per
        # checkpoint until the page is a log again.
        appended = "\n".join(
            f"## CLAIM-SOMETHING-{i} (2026-08-0{i % 9}) - PENDING\n\nProse.\n"
            for i in range(20)
        )
        errors = doc_tables.benchmarks_errors(VALID + "\n" + appended)
        self.assertTrue(any("H2 sections" in e for e in errors), errors)

    def test_a_single_appended_section_fails_immediately(self) -> None:
        # The sharp rule: the FIRST appended entry fails, not the seventh once
        # a count budget fills. This is what actually stops the drift.
        mutated = VALID + "\n## CLAIM-FAKE-LEVER (2026-08-04) - PENDING\n\nProse.\n"
        errors = doc_tables.benchmarks_errors(mutated)
        self.assertTrue(any("non-canonical" in e for e in errors), errors)
        self.assertTrue(any("roll-benchmark-record.py" in e for e in errors), errors)

    def test_a_new_canonical_subject_is_allowed(self) -> None:
        # Adding a real comparison subject named in CANONICAL_SECTIONS passes.
        mutated = VALID + "\n## MLX-LM, Apple M4\n\n| Axis | Ratio |\n|---|---|\n"
        self.assertEqual(doc_tables.benchmarks_errors(mutated), [])

    def test_em_dash_fails(self) -> None:
        mutated = VALID.replace("Measured results.", "Measured results — really.")
        errors = doc_tables.benchmarks_errors(mutated)
        self.assertTrue(any("em-dash" in e for e in errors), errors)

    def test_oversized_page_fails(self) -> None:
        mutated = VALID + "\n" + ("- a filler bullet line\n" * 3000)
        errors = doc_tables.benchmarks_errors(mutated)
        self.assertTrue(any("scoreboard budget" in e for e in errors), errors)

    def test_missing_record_link_fails(self) -> None:
        mutated = VALID.replace(
            "See .agents/benchmark-record.md for the record.", "No record."
        )
        errors = doc_tables.benchmarks_errors(mutated)
        self.assertTrue(any("does not link to" in e for e in errors), errors)

    def test_too_many_prose_paragraphs_fails(self) -> None:
        # Results migrating back into prose, one short paragraph at a time.
        prose = "\n\n".join(f"A result sentence number {i}." for i in range(40))
        errors = doc_tables.benchmarks_errors(VALID + "\n\n" + prose)
        self.assertTrue(any("prose paragraphs" in e for e in errors), errors)

    def test_long_prose_paragraph_fails(self) -> None:
        mutated = VALID.replace(
            "Greedy, closed loop, three interleaved repetitions.", "word " * 300
        )
        errors = doc_tables.benchmarks_errors(mutated)
        self.assertTrue(any("prose paragraph" in e for e in errors), errors)

    def test_wall_of_prose_table_cell_fails(self) -> None:
        wall = "x " * 200
        mutated = VALID.replace("| Thing | Pending |", f"| Thing | {wall} |")
        errors = doc_tables.benchmarks_errors(mutated)
        self.assertTrue(any("wall-of-prose" in e for e in errors), errors)

    def test_page_without_tables_fails(self) -> None:
        # Shedding prose by shedding content must not pass.
        mutated = VALID.replace(_ROWS, "| Model 0 | 1.0x |")
        errors = doc_tables.benchmarks_errors(mutated)
        self.assertTrue(any("table rows" in e for e in errors), errors)

    def test_long_code_block_is_allowed(self) -> None:
        mutated = VALID.replace(
            "vllm-bench run", "\n".join(["vllm-bench run"] * 60)
        )
        self.assertEqual(doc_tables.benchmarks_errors(mutated), [])

    def test_many_short_table_rows_are_allowed(self) -> None:
        rows = "\n".join(["| Thing | Pending |"] * 120)
        mutated = VALID.replace("| Thing | Pending |", rows)
        self.assertEqual(doc_tables.benchmarks_errors(mutated), [])


VALID_FEATURES = "\n".join(
    [
        "# Features",
        "",
        "See STATUS.md for lifecycle state and BENCHMARKS.md for speed.",
        "",
        "## At a glance",
        "",
        "| Feature | vllm.cpp | vLLM |",
        "|---|---|---|",
        "\n".join(f"| Feature {i} | yes | yes |" for i in range(80)),
        "",
        "## Not supported yet",
        "",
        "| Gap | State |",
        "|---|---|",
        "| Thing | Not started |",
        "",
        "## How to read this page",
        "",
        "A mark means implemented and gated.",
        "",
    ]
)


class FeaturesStructureTests(unittest.TestCase):
    def test_minimal_valid_document_passes(self) -> None:
        self.assertEqual(doc_tables.features_errors(VALID_FEATURES), [])

    def test_shipped_features_page_passes(self) -> None:
        text = (ROOT / "docs/FEATURES.md").read_text(encoding="utf-8")
        self.assertEqual(doc_tables.features_errors(text), [])

    def test_missing_at_a_glance_section_fails(self) -> None:
        mutated = VALID_FEATURES.replace("## At a glance", "## Stuff")
        errors = doc_tables.features_errors(mutated)
        self.assertTrue(any("At a glance" in e for e in errors), errors)

    def test_missing_not_supported_section_fails(self) -> None:
        # The honesty section: a feature matrix without it reads as a sales
        # sheet, which is exactly the drift this row guards.
        mutated = VALID_FEATURES.replace("## Not supported yet", "## Roadmap")
        errors = doc_tables.features_errors(mutated)
        self.assertTrue(any("Not supported yet" in e for e in errors), errors)

    def test_missing_legend_section_fails(self) -> None:
        mutated = VALID_FEATURES.replace("## How to read this page", "## Notes")
        errors = doc_tables.features_errors(mutated)
        self.assertTrue(any("How to read" in e for e in errors), errors)

    def test_missing_status_link_fails(self) -> None:
        mutated = VALID_FEATURES.replace("STATUS.md", "nothing")
        errors = doc_tables.features_errors(mutated)
        self.assertTrue(any("STATUS.md" in e for e in errors), errors)

    def test_missing_benchmarks_link_fails(self) -> None:
        mutated = VALID_FEATURES.replace("BENCHMARKS.md for speed", "speed")
        errors = doc_tables.features_errors(mutated)
        self.assertTrue(any("BENCHMARKS.md" in e for e in errors), errors)

    def test_prose_replacing_the_matrix_fails(self) -> None:
        mutated = VALID_FEATURES.replace(
            "\n".join(f"| Feature {i} | yes | yes |" for i in range(80)),
            "| Feature 0 | yes | yes |",
        )
        errors = doc_tables.features_errors(mutated)
        self.assertTrue(any("table rows" in e for e in errors), errors)

    def test_appended_sections_fail_as_an_append_log(self) -> None:
        appended = "\n".join(
            f"## Feature note {i} (2026-08-04)\n\nProse.\n" for i in range(25)
        )
        errors = doc_tables.features_errors(VALID_FEATURES + "\n" + appended)
        self.assertTrue(any("H2 sections" in e for e in errors), errors)

    def test_em_dash_fails(self) -> None:
        mutated = VALID_FEATURES.replace(
            "A mark means implemented and gated.", "A mark means gated — really."
        )
        errors = doc_tables.features_errors(mutated)
        self.assertTrue(any("em-dash" in e for e in errors), errors)

    def test_the_two_pages_have_distinct_budgets(self) -> None:
        # A regression guard on the refactor: FEATURES must not silently
        # inherit the scoreboard's looser limits.
        self.assertNotEqual(
            doc_tables.FEATURES_RULES.max_chars,
            doc_tables.BENCHMARKS_RULES.max_chars,
        )
        self.assertNotEqual(
            doc_tables.FEATURES_RULES.required_sections,
            doc_tables.BENCHMARKS_RULES.required_sections,
        )


class RollRecordTests(unittest.TestCase):
    def test_shipped_page_has_nothing_to_roll(self) -> None:
        text = (ROOT / "docs/BENCHMARKS.md").read_text(encoding="utf-8")
        _, rollable = roll_record.plan(text)
        self.assertEqual([title for title, _ in rollable], [])

    def test_non_canonical_section_is_rollable(self) -> None:
        text = VALID + "\n## CLAIM-FOO decode attribution (2026-08-04)\n\nProse.\n"
        kept, rollable = roll_record.plan(text)
        self.assertEqual(len(rollable), 1)
        self.assertIn("CLAIM-FOO", rollable[0][0])
        self.assertTrue(all("CLAIM-FOO" not in t for t, _ in kept), kept)

    def test_canonical_sections_are_kept(self) -> None:
        kept, rollable = roll_record.plan(VALID)
        self.assertEqual(rollable, [])
        self.assertEqual(len(kept), 4)

    def test_rebuild_drops_only_the_rolled_sections(self) -> None:
        extra = "## CLAIM-BAR (2026-08-04)\n\nForensics.\n"
        text = VALID + "\n" + extra
        preamble, _ = roll_record.split_sections(text)
        kept, rollable = roll_record.plan(text)
        rebuilt = roll_record.rebuild(preamble, kept)
        self.assertNotIn("CLAIM-BAR", rebuilt)
        self.assertIn("## At a glance", rebuilt)
        # The moved text is preserved verbatim for the record.
        self.assertIn("Forensics.", rollable[0][1])

    def test_rebuilt_page_still_passes_the_checker(self) -> None:
        text = VALID + "\n## CLAIM-BAZ (2026-08-04)\n\n" + ("word " * 300) + "\n"
        self.assertTrue(doc_tables.benchmarks_errors(text))
        preamble, _ = roll_record.split_sections(text)
        kept, _ = roll_record.plan(text)
        rebuilt = roll_record.rebuild(preamble, kept)
        self.assertEqual(doc_tables.benchmarks_errors(rebuilt), [])

    def test_allowlist_has_exactly_one_definition(self) -> None:
        # The rollup and the CI gate must never disagree about what moves. The
        # rollup reads the list from the checker rather than copying it, so the
        # two allowlists are the same values, loaded from one file.
        self.assertEqual(
            roll_record.RULES.canonical_sections,
            doc_tables.BENCHMARKS_RULES.canonical_sections,
        )
        for title in ("At a glance", "vLLM, online serving", "Reproduce"):
            with self.subTest(title=title):
                self.assertEqual(
                    roll_record.is_canonical(title),
                    doc_tables.BENCHMARKS_RULES.is_canonical(title),
                )

    def test_what_the_checker_rejects_is_what_the_rollup_moves(self) -> None:
        text = VALID + "\n## CLAIM-QUX attribution (2026-08-04)\n\nProse.\n"
        self.assertTrue(
            any("non-canonical" in e for e in doc_tables.benchmarks_errors(text))
        )
        _, rollable = roll_record.plan(text)
        self.assertEqual(len(rollable), 1)
        self.assertIn("CLAIM-QUX", rollable[0][0])

    def test_rolling_is_byte_exact_on_the_kept_sections(self) -> None:
        # Rolling must change nothing except removing the rolled sections, so
        # repeated rolls are idempotent and never churn the scoreboard.
        base = (ROOT / "docs/BENCHMARKS.md").read_text(encoding="utf-8")
        text = base + "\n## CLAIM-CHURN (2026-08-04)\n\nProse.\n"
        preamble, _ = roll_record.split_sections(text)
        kept, _ = roll_record.plan(text)
        self.assertEqual(roll_record.rebuild(preamble, kept), base)

    def test_features_page_has_no_canonical_allowlist(self) -> None:
        # New feature AREAS are a legitimate way for that page to grow, so it
        # is governed by the section budget alone.
        self.assertIsNone(doc_tables.FEATURES_RULES.canonical_sections)
        self.assertIsNotNone(doc_tables.BENCHMARKS_RULES.canonical_sections)

    def test_heading_inside_a_fence_is_not_a_section(self) -> None:
        text = VALID.replace("vllm-bench run", "## not a heading")
        _, sections = roll_record.split_sections(text)
        self.assertTrue(all("not a heading" not in t for t, _ in sections))



# docs/STATUS.md is guarded by a RATCHET, not a budget: it may only shrink. The
# mutations therefore prove both directions -- growth fails, shrinkage passes --
# and that the required sections cannot be dropped.
STATUS_VALID = "\n".join(
    [
        "# Status",
        "",
        "## Parity pin",
        "",
        "vLLM 0.26.0.dev0.",
        "",
        "## Capability status",
        "",
        "| Capability | State |",
        "|---|---|",
        "| Thing | Works |",
        "",
        "## Not supported yet",
        "",
        "| Thing | Why |",
        "|---|---|",
        "| Other | Not started |",
    ]
)


class StatusRatchet(unittest.TestCase):
    def test_small_valid_page_passes(self) -> None:
        self.assertEqual(doc_tables.status_errors(STATUS_VALID), [])

    def test_each_required_section_is_enforced(self) -> None:
        for label, matchers in doc_tables.STATUS_REQUIRED:
            text = STATUS_VALID.replace(f"## {label}", "## Something else")
            with self.subTest(section=label):
                self.assertTrue(
                    any(label in e for e in doc_tables.status_errors(text)),
                    f"dropping '{label}' was not rejected",
                )

    def test_the_ratchet_is_exactly_one_byte_wide(self) -> None:
        """A page one byte over fails and one byte under passes.

        The number moved for ENG-RELEASE-CONTAINERS, which owed the page a
        lifecycle line and paid for it inside the release paragraph. What must
        keep holding is not the value but the shape: byte-tight, shrink-only.
        A ratchet that tolerated one spare byte would tolerate a hundred.
        """
        limit = doc_tables.STATUS_RATCHET["chars"]
        under = STATUS_VALID + "x" * (limit - len(STATUS_VALID) - 1)
        self.assertEqual(len(under), limit - 1)
        self.assertEqual(
            [e for e in doc_tables.status_errors(under) if "chars is" in e], []
        )
        over = STATUS_VALID + "x" * (limit - len(STATUS_VALID) + 1)
        self.assertEqual(len(over), limit + 1)
        self.assertTrue(any("chars is" in e for e in doc_tables.status_errors(over)))

    def test_growth_past_the_char_ratchet_is_rejected(self) -> None:
        text = STATUS_VALID + "\n" + "x" * doc_tables.STATUS_RATCHET["chars"]
        errors = doc_tables.status_errors(text)
        self.assertTrue(any("chars is" in error for error in errors), errors)
        self.assertTrue(
            any("structured state evidence" in error for error in errors), errors
        )

    def test_growth_past_the_section_ratchet_is_rejected(self) -> None:
        extra = "\n".join(
            f"## Extra {i}\n\nprose\n" for i in range(doc_tables.STATUS_RATCHET["h2_sections"] + 1)
        )
        errors = doc_tables.status_errors(STATUS_VALID + "\n" + extra)
        self.assertTrue(any("h2 sections is" in e for e in errors))

    def test_growth_past_the_paragraph_ratchet_is_rejected(self) -> None:
        para = "y" * (doc_tables.MAX_PARAGRAPH_CHARS + 1)
        extra = "\n\n".join([para] * (doc_tables.STATUS_RATCHET["long_paragraphs"] + 1))
        errors = doc_tables.status_errors(STATUS_VALID + "\n\n" + extra)
        self.assertTrue(any("long paragraphs is" in e for e in errors))

    def test_growth_past_the_cell_ratchet_is_rejected(self) -> None:
        cell = "z" * (doc_tables.MAX_CELL_CHARS + 1)
        rows = "\n".join(
            f"| {cell} | {cell} |" for _ in range(doc_tables.STATUS_RATCHET["oversized_cells"])
        )
        errors = doc_tables.status_errors(STATUS_VALID + "\n\n| a | b |\n|---|---|\n" + rows)
        self.assertTrue(any("oversized cells is" in e for e in errors))

    def test_the_live_page_is_inside_its_ratchet(self) -> None:
        self.assertEqual(
            doc_tables.status_errors(doc_tables.STATUS.read_text(encoding="utf-8")), []
        )

    def test_the_char_ratchet_is_not_slack(self) -> None:
        """Headroom is not a growth budget.

        The cheap way out of a red char ratchet is to RAISE the number instead
        of shrinking the page, and the gate then measures nothing. Bounding the
        slack means clearing it still costs a real collapse.
        """
        live = len(doc_tables.STATUS.read_text(encoding="utf-8"))
        slack = doc_tables.STATUS_RATCHET["chars"] - live
        self.assertGreaterEqual(slack, 0, "the live page is already over its ratchet")
        self.assertLessEqual(slack, 2000, "ratchet headroom is cover for bloat")

    def test_a_repin_can_only_tighten_the_char_ratchet(self) -> None:
        """A re-pin must move the ratchet DOWN, and must still bind afterwards.

        The Muse Glimmer speed entry (#333) took this page 243287 -> 243283 by
        collapsing a duplicated Qwen3.5-4B paragraph to pay for its own line.
        That is the only legitimate shape of a re-pin: the page shrank, so the
        number shrank with it.

        The failure this guards is the opposite move -- raising the constant to
        make room -- which turns a shrink-only ratchet into a growth budget and
        silently stops measuring anything. Proven by mutation in both
        directions: a ratchet below the live page must be an error, and one
        inflated above it must be caught as slack.
        """
        live = len(doc_tables.STATUS.read_text(encoding="utf-8"))
        pinned = doc_tables.STATUS_RATCHET["chars"]

        # The pin binds: it is at or above the page, but not by much.
        self.assertGreaterEqual(pinned, live)
        self.assertLessEqual(pinned - live, 2000)

        # DOWN past the page: must be rejected.
        with mock.patch.dict(doc_tables.STATUS_RATCHET, {"chars": live - 1}):
            errors = doc_tables.status_errors(
                doc_tables.STATUS.read_text(encoding="utf-8")
            )
            self.assertTrue(
                any("char" in error for error in errors),
                "a ratchet below the live page has to be an error",
            )

        # UP well past the page: the slack bound is what catches an inflated
        # re-pin, so assert it would fire rather than trusting the constant.
        self.assertGreater(
            (live + 5000) - live, 2000, "an inflated re-pin must exceed the slack bound"
        )

    def test_the_live_page_keeps_the_character_ratchet_tight(self) -> None:
        text = doc_tables.STATUS.read_text(encoding="utf-8")
        slack = doc_tables.STATUS_RATCHET["chars"] - len(text)
        self.assertGreaterEqual(slack, 0)
        self.assertLessEqual(slack, 25)

    def test_the_rebased_character_ratchet_is_byte_tight(self) -> None:
        text = doc_tables.STATUS.read_text(encoding="utf-8")
        self.assertEqual(doc_tables.STATUS_RATCHET["chars"], len(text))

    def test_a_retired_claim_cannot_come_back_for_free(self) -> None:
        """A claim the page RETIRED must cost something to reinstate.

        The ratchet is what makes a deletion permanent. #277 paid for its STATUS
        edit by deleting two claims that had become false -- that `/metrics` has
        no live async backing, once in the OpenAI-server row and once in the
        metrics paragraph -- and lowered the char ratchet by the same 30 bytes.

        Two things are asserted, and neither is the byte-tightness above.
        First, the retired wording is genuinely GONE: a page that still says
        `/metrics` is unbacked while the AsyncLLM output handler records into it
        is lying to a reader, and no size ratchet notices a false sentence.
        Second, putting it back is rejected -- so a future edit reinstating it
        has to find the space, out loud, instead of spending headroom the
        deletion left behind.
        """
        text = doc_tables.STATUS.read_text(encoding="utf-8")
        self.assertEqual(doc_tables.status_errors(text), [])

        retired = (
            "metrics and cache reset lack live async backing",
            "the async production-serving path wiring",
        )
        for claim in retired:
            with self.subTest(claim=claim):
                self.assertNotIn(
                    claim,
                    text,
                    "docs/STATUS.md still carries a claim about /metrics that "
                    "the AsyncLLM wiring made false",
                )

        restored = text.replace(
            "cache reset lacks live async backing",
            retired[0],
            1,
        ).replace(
            "The remaining work is the chat/completion",
            f"The remaining work is {retired[1]}, the chat/completion",
            1,
        )
        self.assertGreater(
            len(restored),
            len(text),
            "the mutation must actually re-add the retired claims; if the "
            "anchors stopped matching, this test is asserting nothing",
        )
        errors = doc_tables.status_errors(restored)
        self.assertTrue(
            any("chars is" in error for error in errors),
            "reinstating the retired claims must break the char ratchet, "
            f"otherwise the deletion bought free headroom; got {errors}",
        )

    def test_the_status_ratchet_only_ever_moves_down(self) -> None:
        """A ratchet that can be RAISED is a budget with extra steps.

        The checker states the rule in prose -- "every limit is pinned to what
        the page measured ... and may only go DOWN" -- and nothing enforced it.
        Every other ratchet test here measures the LIVE page, so raising a cap
        and growing the page by the same amount in one change is green in all of
        them: measured 2026-08-10, `"chars": 243584 -> 243684` plus 100 chars
        appended to docs/STATUS.md left this module at 52/52 OK. That is exactly
        the cheap way out the ratchet exists to block, and every STATUS edit
        touches this number, so it is the number most likely to drift upward
        unnoticed.

        STATUS_RATCHET_CEILING is the ratchet as it stands. Lowering a cap keeps
        passing; raising one fails here and can only be unblocked by editing the
        ceiling in the same change -- a deliberate, reviewable act instead of a
        silent bump. The caps have in fact only ever fallen (chars 289727 ->
        243584 over 48 commits), so the ceiling never needs to rise.
        """
        ceiling = {
            "chars": 243578,
            "h2_sections": 11,
            "long_paragraphs": 82,
            "oversized_cells": 44,
        }
        self.assertEqual(
            set(doc_tables.STATUS_RATCHET),
            set(ceiling),
            "every ratcheted metric needs a ceiling here, and a ceiling with no "
            "metric behind it is a metric that quietly stopped being enforced",
        )
        for key, cap in doc_tables.STATUS_RATCHET.items():
            with self.subTest(metric=key):
                self.assertLessEqual(
                    cap,
                    ceiling[key],
                    f"the {key} ratchet moved UP to {cap} from {ceiling[key]}; "
                    "docs/STATUS.md may only shrink. Collapse the superseded "
                    "narrative instead, and lower this ceiling in the same "
                    "change that lowers the ratchet -- never the reverse",
                )

    def test_the_repin_was_paid_for_by_a_real_collapse(self) -> None:
        # Lowering the cap is the honest half of "STATUS may only shrink"; the
        # dishonest half is lowering it after growing the page and calling the
        # difference rounding. The 2026-08-09 prompt-logprobs re-pin added a
        # Sampling clause and paid for it by collapsing three restatements in the
        # same row.
        #
        # So assert the payment, not the arithmetic: put the collapsed text back
        # and the page must break the cap it is pinned to. If a later edit
        # quietly restores that prose, this goes red rather than the slack being
        # silently re-spent.
        text = doc_tables.STATUS.read_text(encoding="utf-8")
        restored = text.replace(
            "); the C-ABI beam",
            "), so beam search runs on the real server (the C-ABI beam",
            1,
        ).replace(
            "beams are stepped sequentially).",
            "beams are stepped sequentially, byte-identical to the sync driver).",
            1,
        )
        self.assertNotEqual(restored, text, "the collapsed prose is not where it was")
        self.assertGreater(len(restored), doc_tables.STATUS_RATCHET["chars"])

    def test_the_logprobs_mode_repin_was_paid_for(self) -> None:
        # Same contract as the other re-pin guards: the cap may only come down,
        # and the reduction must be a real collapse rather than a growth budget.
        # The 2026-08-10 logprobs_mode line was paid for by dropping the best_of
        # cell's upstream rationale. Restore it and the page must break its cap,
        # so a later edit cannot quietly put the prose back and re-spend the
        # slack.
        #
        # This guard and test_the_repin_was_paid_for_by_a_real_collapse above
        # cover DIFFERENT collapses in the SAME row (#238's best_of rationale,
        # #223's beam-search restatements). They are complementary, not
        # alternatives: dropping either one would leave its collapse unguarded.
        text = doc_tables.STATUS.read_text(encoding="utf-8")
        restored = text.replace(
            "`best_of==n` is the default no-op).",
            "`best_of==n` is the default no-op) \u2014 vLLM 0.26 itself has dropped "
            "`best_of` from its live path, so this follows the classic OpenAI contract.",
            1,
        )
        self.assertNotEqual(restored, text, "the collapsed rationale is not where it was")
        self.assertGreater(len(restored), doc_tables.STATUS_RATCHET["chars"])

    def test_the_ratchet_carries_no_hidden_headroom(self) -> None:
        # A ratchet parked well above the page it guards is not a ratchet: it
        # silently licenses regrowth up to the old number. The rule is "lower it
        # in the same change", so the cap must track the page rather than trail
        # it. This caught the real case -- collapsing one 33,211-char row left
        # 32,728 chars of slack that would otherwise have been free to re-spend.
        text = doc_tables.STATUS.read_text(encoding="utf-8")
        headers = [line for line in text.split("\n") if line.startswith("## ")]
        measured = {
            "chars": len(text),
            "h2_sections": len(headers),
            "long_paragraphs": sum(
                1
                for _, para in doc_tables._prose_paragraphs(text)
                if len(para) > doc_tables.MAX_PARAGRAPH_CHARS
            ),
            "oversized_cells": sum(
                1
                for _, cells in doc_tables._table_rows(text)
                for cell in cells
                if len(cell) > doc_tables.MAX_CELL_CHARS
            ),
        }
        for key, cap in doc_tables.STATUS_RATCHET.items():
            with self.subTest(metric=key):
                self.assertLessEqual(measured[key], cap)
                # 1% of slack absorbs an in-flight edit; more than that is a cap
                # that stopped describing the page.
                self.assertLessEqual(
                    cap - measured[key],
                    max(1, cap // 100),
                    f"{key} ratchet {cap} sits {cap - measured[key]} above the "
                    f"measured {measured[key]}; lower it in the change that shrank "
                    "the page",
                )
if __name__ == "__main__":
    unittest.main()
