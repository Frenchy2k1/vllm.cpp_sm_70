#!/usr/bin/env python3
"""Behavior tests for strict Git trailer enforcement."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-commit-trailers.py"


def load_checker():
    # repository root on sys.path, so provide it here.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("check_commit_trailers", CHECKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STRICT_MESSAGE = """policy: enforce trailers

The body may explain the change.

FOLLOWING_AGENTS_PROTOCOL

Following-Agents-Protocol: true
AI-Assisted: true
Assisted-by: Codex:GPT-5 [Codex]
"""


class CommitMessageContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.checker = load_checker()

    def assertInvalid(self, message: str, needle: str) -> None:
        errors = self.checker.validate_commit_message(message, strict=True)
        self.assertTrue(any(needle in error for error in errors), errors)

    def test_valid_ai_assisted_and_human_only_messages(self) -> None:
        self.assertEqual(
            self.checker.validate_commit_message(STRICT_MESSAGE, strict=True), []
        )
        human = STRICT_MESSAGE.replace(
            "AI-Assisted: true\nAssisted-by: Codex:GPT-5 [Codex]\n",
            "AI-Assisted: false\n",
        )
        self.assertEqual(
            self.checker.validate_commit_message(human, strict=True), []
        )

    def test_substring_or_embedded_legacy_tag_is_not_a_raw_paragraph(self) -> None:
        self.assertInvalid(
            STRICT_MESSAGE.replace(
                "FOLLOWING_AGENTS_PROTOCOL\n\n",
                "We are FOLLOWING_AGENTS_PROTOCOL today.\n\n",
            ),
            "separate paragraph",
        )
        self.assertInvalid(
            STRICT_MESSAGE.replace(
                "FOLLOWING_AGENTS_PROTOCOL\n\n",
                "FOLLOWING_AGENTS_PROTOCOL extra\n\n",
            ),
            "separate paragraph",
        )

    def test_raw_tag_must_be_separate_from_the_trailer_paragraph(self) -> None:
        message = STRICT_MESSAGE.replace(
            "FOLLOWING_AGENTS_PROTOCOL\n\nFollowing-Agents",
            "FOLLOWING_AGENTS_PROTOCOL\nFollowing-Agents",
        )
        self.assertInvalid(message, "separate paragraph")

    def test_protocol_and_ai_declarations_are_unique_and_exact(self) -> None:
        self.assertInvalid(
            STRICT_MESSAGE.replace(
                "Following-Agents-Protocol: true",
                "Following-Agents-Protocol: false",
            ),
            "must be exactly true",
        )
        self.assertInvalid(
            STRICT_MESSAGE.replace(
                "Following-Agents-Protocol: true",
                "Following-Agents-Protocol: true\nFollowing-Agents-Protocol: true",
            ),
            "exactly once",
        )
        self.assertInvalid(
            STRICT_MESSAGE.replace("AI-Assisted: true", "AI-Assisted: maybe"),
            "true or false",
        )
        self.assertInvalid(
            STRICT_MESSAGE.replace(
                "AI-Assisted: true", "AI-Assisted: true\nAI-Assisted: false"
            ),
            "exactly once",
        )

    def test_assistance_attribution_is_required_and_has_closed_syntax(self) -> None:
        self.assertInvalid(
            STRICT_MESSAGE.replace("Assisted-by: Codex:GPT-5 [Codex]\n", ""),
            "Assisted-by",
        )
        for malformed in (
            "Codex GPT-5 [Codex]",
            "Codex:GPT-5",
            "Codex:GPT-5 []",
            "Codex: GPT-5 [Codex]",
            "Codex:GPT-5 [Codex] trailing",
        ):
            with self.subTest(malformed=malformed):
                self.assertInvalid(
                    STRICT_MESSAGE.replace("Codex:GPT-5 [Codex]", malformed),
                    "malformed Assisted-by",
                )

    def test_human_only_declaration_rejects_assistance_attribution(self) -> None:
        message = STRICT_MESSAGE.replace("AI-Assisted: true", "AI-Assisted: false")
        self.assertInvalid(message, "must omit Assisted-by")

    def test_ai_authorship_trailers_are_forbidden(self) -> None:
        for trailer in (
            "Signed-off-by: Codex:GPT-5 [Codex]",
            "Co-Authored-By: Claude Opus <noreply@anthropic.com>",
            "Signed-off-by: OpenAI Bot <bot@openai.com>",
        ):
            with self.subTest(trailer=trailer):
                self.assertInvalid(
                    STRICT_MESSAGE.rstrip() + "\n" + trailer + "\n",
                    "AI authorship",
                )

    def test_human_authorship_trailers_remain_allowed(self) -> None:
        for trailer in (
            "Signed-off-by: Alice Example <alice@example.com>",
            "Co-Authored-By: Bob Human <bob@example.org>",
        ):
            with self.subTest(trailer=trailer):
                message = STRICT_MESSAGE.rstrip() + "\n" + trailer + "\n"
                self.assertEqual(
                    self.checker.validate_commit_message(message, strict=True), []
                )

    def test_every_declared_agent_model_and_tool_identity_is_not_an_author(self) -> None:
        message = STRICT_MESSAGE.replace(
            "Codex:GPT-5 [Codex]",
            "Unfamiliar-Agent:Model-Seven [NeutralTool] [SecondTool]",
        )
        for identity in ("Model-Seven", "NeutralTool", "SecondTool"):
            with self.subTest(identity=identity):
                authored = (
                    message.rstrip()
                    + f"\nCo-Authored-By: {identity} <{identity.casefold()}@example.com>\n"
                )
                self.assertInvalid(authored, "AI authorship")

    def test_assisted_agent_identity_is_forbidden_as_an_authorship_trailer(self) -> None:
        message = STRICT_MESSAGE.replace(
            "Codex:GPT-5 [Codex]", "Unfamiliar-Agent:Model-7 [NeutralTool]"
        ).rstrip()
        message += "\nSigned-off-by: Unfamiliar-Agent <agent@example.com>\n"
        self.assertInvalid(message, "AI authorship")

    def test_legacy_is_only_accepted_in_non_strict_mode(self) -> None:
        legacy = "legacy change\n\nFOLLOWING_AGENTS_PROTOCOL\n"
        self.assertEqual(
            self.checker.validate_commit_message(legacy, strict=False), []
        )
        self.assertTrue(
            self.checker.validate_commit_message(legacy, strict=True)
        )

    def test_parser_matches_git_interpret_trailers(self) -> None:
        parsed = subprocess.check_output(
            ["git", "interpret-trailers", "--parse"],
            input=STRICT_MESSAGE,
            text=True,
        )
        self.assertEqual(self.checker.parsed_trailers(STRICT_MESSAGE), parsed)


class RangeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.checker = load_checker()

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "config", "user.email", "test@example.com"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.repo), "config", "user.name", "Test"],
            check=True,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def commit(self, message: str) -> str:
        marker = self.repo / "history"
        marker.write_text(marker.read_text() + "x" if marker.exists() else "x")
        subprocess.run(["git", "-C", str(self.repo), "add", "history"], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "commit", "-q", "-F", "-"],
            input=message,
            text=True,
            check=True,
        )
        return subprocess.check_output(
            ["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True
        ).strip()

    def test_cutover_commit_itself_is_strict_and_parent_is_legacy(self) -> None:
        base = self.commit("base\n\nFOLLOWING_AGENTS_PROTOCOL\n")
        before = self.commit("before\n\nFOLLOWING_AGENTS_PROTOCOL\n")
        cutover = self.commit(STRICT_MESSAGE)
        after = self.commit(STRICT_MESSAGE.replace("policy:", "policy after:"))
        errors = self.checker.validate_range(
            self.repo, base, after, cutover=cutover
        )
        self.assertEqual(errors, [])
        self.assertNotEqual(before, cutover)

    def test_post_cutover_legacy_commit_fails(self) -> None:
        base = self.commit("base\n\nFOLLOWING_AGENTS_PROTOCOL\n")
        cutover = self.commit(STRICT_MESSAGE)
        self.commit("after\n\nFOLLOWING_AGENTS_PROTOCOL\n")
        errors = self.checker.validate_range(
            self.repo, base, "HEAD", cutover=cutover
        )
        self.assertTrue(any("Following-Agents-Protocol" in error for error in errors))

    def test_missing_unreachable_and_non_ancestor_revisions_fail_closed(self) -> None:
        base = self.commit("base\n\nFOLLOWING_AGENTS_PROTOCOL\n")
        head = self.commit(STRICT_MESSAGE)
        with self.assertRaises(ValueError):
            self.checker.validate_range(
                self.repo, "missing", head, cutover=head
            )
        subprocess.run(["git", "-C", str(self.repo), "checkout", "-q", "--detach", base], check=True)
        side = self.commit(STRICT_MESSAGE.replace("policy:", "side:"))
        with self.assertRaises(ValueError):
            self.checker.validate_range(
                self.repo, side, head, cutover=head
            )

    def test_ambiguous_revision_name_fails_closed(self) -> None:
        base = self.commit("base\n\nFOLLOWING_AGENTS_PROTOCOL\n")
        head = self.commit(STRICT_MESSAGE)
        subprocess.run(
            ["git", "-C", str(self.repo), "branch", "collision", base], check=True
        )
        subprocess.run(
            ["git", "-C", str(self.repo), "tag", "collision", head], check=True
        )
        with self.assertRaises(ValueError):
            self.checker.validate_range(
                self.repo, base, "collision", cutover=head
            )


class MergeArtifacts(unittest.TestCase):
    """The trailer gate must judge the CLAIM, not the paragraph layout (#406).

    Every shape below was taken from a real commit on main. The gate was failing
    on how commits LAND rather than on how they are written, and 13 of the last
    30 commits on main failed it -- unnoticed only because those runs were
    cancelled (#274).
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.checker = load_checker()

    def test_github_co_authored_by_does_not_hide_the_block(self) -> None:
        """RED-BEFORE: this is dbd0d51c, 87308dea and f64f2b71 on main.

        Squash-merging through GitHub appends `Co-authored-by:` as a SEPARATE
        trailing paragraph. `git interpret-trailers --parse` reads only the last
        paragraph, so the protocol trailers become invisible and the gate counts
        zero. The commit is correct; the parse is what breaks.

        AGENTS.md forbids AI tools from adding Co-Authored-By. It does not
        forbid GitHub from recording a real human co-author, so this must pass.
        """
        message = STRICT_MESSAGE + "\nCo-authored-by: Ettore Di Giacinto <mudler@localai.io>\n"
        self.assertEqual(
            self.checker.validate_commit_message(message, strict=True), []
        )

    def test_a_squash_that_doubles_the_trailer_block_still_fails(self) -> None:
        """b8293c88, the commit that turned main red, must STAY red.

        Squashing a multi-commit PR concatenates each commit's trailer block, so
        every trailer appears twice. That was tempting to collapse -- two
        identical declarations do say the same thing -- but it is NOT what this
        change fixes, and relaxing it would delete a rule that catches genuinely
        malformed messages.

        The distinction that matters: the Co-authored-by case above is a correct
        commit REJECTED BY THE PARSE, while this one is a message that really is
        malformed and is fixable at the source by writing the squash body (or by
        landing a single-commit PR). AGENTS.md's "fix the cause, not the gate"
        puts this on the process side of the line.
        """
        _, _, trailers = STRICT_MESSAGE.rpartition("FOLLOWING_AGENTS_PROTOCOL\n")
        doubled = STRICT_MESSAGE + "\nFOLLOWING_AGENTS_PROTOCOL\n" + trailers
        errors = self.checker.validate_commit_message(doubled, strict=True)
        self.assertTrue(errors, "a doubled trailer block must still be rejected")

    def test_contradictory_declarations_fail(self) -> None:
        """A commit cannot both declare and deny AI assistance."""
        conflicting = STRICT_MESSAGE + "AI-Assisted: false\n"
        errors = self.checker.validate_commit_message(conflicting, strict=True)
        self.assertTrue(errors, "contradictory declarations must fail")

    def test_a_github_merge_commit_with_no_trailers_still_fails(self) -> None:
        """This is b580452d. It must STAY red -- a real violation, not a layout
        artifact. The relaxation is about placement and duplication only."""
        message = (
            "Merge pull request #386 from mudler/row/ENG-NOW-DERIVED-DONE\n"
            "\n"
            "close the row\n"
        )
        errors = self.checker.validate_commit_message(message, strict=True)
        self.assertTrue(errors, "a message with no trailers at all must fail")

    def test_prose_after_the_trailers_still_fails(self) -> None:
        """A prose paragraph must still terminate the block.

        Without this the relaxation would accept trailers buried anywhere near
        the end, which is exactly the looseness the gate exists to prevent.
        """
        message = STRICT_MESSAGE + "\nAnd then some closing prose about the change.\n"
        errors = self.checker.validate_commit_message(message, strict=True)
        self.assertTrue(errors, "prose after the trailer block must still fail")


if __name__ == "__main__":
    unittest.main()
