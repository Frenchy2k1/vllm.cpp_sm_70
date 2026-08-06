#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-protocol-consistency.py.

The gate exists to catch prose drifting away from the checker that enforces it,
so the mutations here are the real historical failure: a document that still
names README.md as a checkpoint surface, and a document whose contract silently
disagrees with scripts/check-doc-checkpoint.py.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
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


consistency = _load("protocol_consistency", "scripts/check-protocol-consistency.py")

EXPECTED = ("docs/STATUS.md", "docs/BENCHMARKS.md", "docs/FEATURES.md")


def _tracked_paths(prefix: str) -> set[str] | None:
    """Paths git knows under `prefix`, or None when this is not a checkout.

    A prompt that exists only in a working tree is precisely the thing this
    task exists to stop: an instruction nobody else can read. `git ls-files`
    sees staged files too, so it is honest before the commit as well as after.
    Exported trees (git archive) have no `.git`, so absence of git is a skip
    rather than a failure.
    """
    try:
        completed = subprocess.run(
            ["git", "ls-files", "--", prefix],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return {line for line in completed.stdout.splitlines() if line}


@contextlib.contextmanager
def _prompt_tree(files: dict[str, str]):
    """Point consistency.ROOT at a temp tree holding exactly `files`."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for relative, text in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        saved, consistency.ROOT = consistency.ROOT, root
        try:
            yield root
        finally:
            consistency.ROOT = saved


@contextlib.contextmanager
def _repo_copy(workflow_text: str, *, prompts: bool = True):
    """Run consistency.main() against a copy of the repo's own documents.

    Only `.agents/workflow.md` is substituted, so a red from this helper is
    attributable to the manual under test rather than to a hand-built fixture
    that never resembled the repository.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "scripts").mkdir()
        (root / ".agents").mkdir()
        shutil.copy(
            ROOT / "scripts/check-doc-checkpoint.py",
            root / "scripts/check-doc-checkpoint.py",
        )
        shutil.copy(ROOT / "AGENTS.md", root / "AGENTS.md")
        if prompts:
            shutil.copytree(ROOT / ".agents/prompts", root / ".agents/prompts")
        (root / ".agents/workflow.md").write_text(workflow_text, encoding="utf-8")
        saved, consistency.ROOT = consistency.ROOT, root
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                yield lambda: (consistency.main(), out.getvalue(), err.getvalue())
        finally:
            consistency.ROOT = saved


def document(*paths: str) -> str:
    rows = "\n".join(f"| `{path}` | every checkpoint |" for path in paths)
    return "\n".join(
        [
            "# Some normative document",
            "",
            consistency.BEGIN,
            "| Public surface | Owed by |",
            "|---|---|",
            rows,
            consistency.END,
            "",
            "Trailing prose.",
        ]
    )


class ContractParsing(unittest.TestCase):
    def test_extracts_paths_in_order(self) -> None:
        self.assertEqual(
            consistency.contract_paths(document(*EXPECTED)), list(EXPECTED)
        )

    def test_absent_block_is_none(self) -> None:
        self.assertIsNone(consistency.contract_paths("# No contract here"))

    def test_end_before_begin_is_none(self) -> None:
        text = f"{consistency.END}\n| `docs/STATUS.md` |\n{consistency.BEGIN}"
        self.assertIsNone(consistency.contract_paths(text))


class Mutations(unittest.TestCase):
    def test_baseline_passes(self) -> None:
        self.assertEqual(
            consistency.document_errors("doc", document(*EXPECTED), EXPECTED), []
        )

    def test_missing_block_is_rejected(self) -> None:
        errors = consistency.document_errors("doc", "# nothing", EXPECTED)
        self.assertTrue(any("missing the doc-obligation contract" in e for e in errors))

    def test_readme_in_contract_is_rejected_by_name(self) -> None:
        """The exact historical regression: README named as a checkpoint."""
        text = document("README.md", "docs/BENCHMARKS.md", "docs/FEATURES.md")
        errors = consistency.document_errors("doc", text, EXPECTED)
        self.assertTrue(any("README.md" in e and "landing page" in e for e in errors))

    def test_dropped_surface_is_rejected(self) -> None:
        text = document("docs/STATUS.md", "docs/BENCHMARKS.md")
        errors = consistency.document_errors("doc", text, EXPECTED)
        self.assertTrue(any("enforces" in e for e in errors))

    def test_reordered_surfaces_are_rejected(self) -> None:
        text = document("docs/BENCHMARKS.md", "docs/STATUS.md", "docs/FEATURES.md")
        self.assertNotEqual(
            consistency.document_errors("doc", text, EXPECTED), []
        )

    def test_extra_surface_is_rejected(self) -> None:
        text = document(*EXPECTED, "docs/USAGE.md")
        self.assertNotEqual(consistency.document_errors("doc", text, EXPECTED), [])


class LiveTree(unittest.TestCase):
    def test_expected_surfaces_come_from_the_checker(self) -> None:
        self.assertEqual(consistency.obligated_surfaces(), EXPECTED)

    def test_repository_contract_is_consistent(self) -> None:
        self.assertEqual(consistency.main(), 0)

    def test_every_contract_document_exists(self) -> None:
        for name in consistency.CONTRACT_DOCUMENTS:
            self.assertTrue((ROOT / name).exists(), name)


class InterviewBlockTests(unittest.TestCase):
    def test_workflow_carries_the_role_interview(self):
        text = (ROOT / ".agents/workflow.md").read_text(encoding="utf-8")
        self.assertIn(consistency.INTERVIEW_MARKER, text)
        self.assertIn("read-only", text)
        self.assertIn("claim helper --row", text)

    def test_checker_rejects_a_workflow_without_the_interview(self):
        # The mutation this gate exists to catch: the gate ships, the prose
        # does not, and agents never learn the precondition.
        errors = consistency.interview_errors("# workflow\n\nno interview here\n")
        self.assertTrue(errors)

    def test_every_declarable_role_is_named_in_the_interview(self):
        # INTERVIEW_REQUIRED is a hand-written tuple, so emptying or narrowing
        # it would leave every other assertion in this class green while the
        # checker quietly stopped looking at the answers. Bind it to the roles
        # agent-role.py actually accepts instead: a fourth answer must reach the
        # prose, and dropping one from the checker is a red build.
        role = _load("agent_role_for_interview", "scripts/agent-role.py")
        text = (ROOT / ".agents/workflow.md").read_text(encoding="utf-8")
        for name in role.DECLARABLE:
            with self.subTest(role=name):
                self.assertIn(f"claim {name}", text)
                self.assertTrue(
                    any(f"claim {name}" in n for n in consistency.INTERVIEW_REQUIRED),
                    f"INTERVIEW_REQUIRED does not cover 'claim {name}'",
                )

    def test_each_required_answer_is_pinned_individually(self):
        # A block that exists but has lost one of the three answers is the
        # likelier drift, and the marker alone would not see it.
        text = (ROOT / ".agents/workflow.md").read_text(encoding="utf-8")
        for needle in consistency.INTERVIEW_REQUIRED:
            with self.subTest(needle=needle):
                self.assertTrue(consistency.interview_errors(text.replace(needle, "")))


class InterviewWiring(unittest.TestCase):
    """The checker must CALL interview_errors, not merely define it.

    Every assertion above exercises the function directly, so a `main()` that
    never calls it leaves them all green while the gate enforces nothing --
    which is the exact shape of the drift this file exists to catch.
    """

    STRIP = re.compile(
        r"<!-- role-interview:begin -->.*?<!-- role-interview:end -->\n?", re.S
    )

    @contextlib.contextmanager
    def _tree(self, workflow_text: str, *, prompts: bool = True):
        with _repo_copy(workflow_text, prompts=prompts) as run:
            yield run

    def test_faithful_copy_passes(self):
        """Positive control: the temp tree itself is not what fails below."""
        text = (ROOT / ".agents/workflow.md").read_text(encoding="utf-8")
        self.assertIn(consistency.INTERVIEW_MARKER, text)
        with self._tree(text) as run:
            code, _, err = run()
        self.assertEqual(code, 0, err)

    def test_main_fails_when_the_interview_is_deleted(self):
        text = (ROOT / ".agents/workflow.md").read_text(encoding="utf-8")
        stripped = self.STRIP.sub("", text)
        self.assertNotIn(consistency.INTERVIEW_MARKER, stripped)
        self.assertNotEqual(stripped, text, "the strip pattern matched nothing")
        with self._tree(stripped) as run:
            code, _, err = run()
        self.assertEqual(code, 1)
        self.assertIn("role-interview", err)

    def test_main_fails_when_the_prompts_are_missing(self):
        """main() must CALL prompt_errors, not merely define it.

        Every prompt assertion above calls the function directly, so a main()
        that never wires it in leaves them all green while the gate enforces
        nothing -- the same drift, one function later.
        """
        text = (ROOT / ".agents/workflow.md").read_text(encoding="utf-8")
        with self._tree(text, prompts=False) as run:
            code, _, err = run()
        self.assertEqual(code, 1, err)
        self.assertIn(".agents/prompts/reviewer.md", err)


class PromptArtifactTests(unittest.TestCase):
    def test_both_prompts_exist_and_are_tracked(self):
        tracked = _tracked_paths(".agents/prompts")
        for name in ("reviewer.md", "implementer.md"):
            path = ROOT / ".agents/prompts" / name
            self.assertTrue(path.is_file(), f"{name} must exist")
            # A silent downgrade to existence-only is the failure/absence
            # confusion again, so say so in the run output rather than passing
            # quietly with half the assertion skipped.
            with self.subTest(tracked=name):
                if tracked is None:
                    self.skipTest("git unavailable: tracking not verifiable here")
                self.assertIn(
                    f".agents/prompts/{name}",
                    tracked,
                    f"{name} exists but is untracked; an operator-local prompt "
                    "is not a protocol",
                )

    def test_the_reviewer_prompt_carries_the_mutation_instruction(self):
        # The instruction IS the deliverable. A reviewer told only to "review"
        # reads the diff, and reading found none of the eleven tests that
        # passed with their subject deleted.
        text = (ROOT / ".agents/prompts/reviewer.md").read_text(encoding="utf-8")
        for needle in ("mutate", "delete or invert", "stays green"):
            self.assertIn(needle, text.lower(), needle)

    def test_the_reviewer_prompt_refuses_to_defer_to_the_plan(self):
        text = (ROOT / ".agents/prompts/reviewer.md").read_text(encoding="utf-8")
        self.assertIn("plan-mandated", text.lower())

    def test_checker_rejects_a_prompt_missing_its_instruction(self):
        # A missing FILE and a present file missing its INSTRUCTION are two
        # different failures. Asserting only the first would leave the needle
        # loop -- the part that carries the value -- entirely unpinned.
        errors = consistency.prompt_errors({"nonexistent-prompt.md": ("mutate",)})
        self.assertTrue(errors)
        self.assertTrue(any("missing" in e for e in errors), errors)

        present = ".agents/prompts/reviewer.md"
        self.assertEqual(consistency.prompt_errors({present: ("mutate",)}), [])
        omitted = consistency.prompt_errors(
            {present: ("no reviewer prompt would ever contain this phrase",)}
        )
        self.assertTrue(any("omits" in e for e in omitted), omitted)

    def test_an_explicitly_empty_spec_checks_nothing(self):
        # An empty spec must mean "nothing required", not silently fall back to
        # the live PROMPT_REQUIRED: an absence and a value that look the same is
        # the defect class the implementer prompt names.
        with _prompt_tree({}):
            self.assertEqual(consistency.prompt_errors({}), [])
            self.assertTrue(consistency.prompt_errors())

    def test_the_checker_enforces_the_phrases_these_tests_demand(self):
        # Every assertion above reads the prompt FILES, so emptying, narrowing
        # or widening a PROMPT_REQUIRED tuple would leave them all green while
        # the gate quietly stopped enforcing what this suite believes it does.
        #
        # The comparison is EQUALITY, deliberately, not "demanded is a substring
        # of enforced". That substring idiom is borrowed from
        # test_every_declarable_role_is_named_in_the_interview, where it is safe
        # because the demanded side is DERIVED from role.DECLARABLE. Here both
        # sides are hand-written literals, and a substring test cannot see the
        # one narrowing that matters: reverting the reviewer needle from
        # "mutate, don't read" to a bare "mutate" satisfies it while re-opening
        # the incidental-match hole check-protocol-consistency.py spends five
        # lines arguing is dangerous. Equality means changing what the gate
        # enforces is a deliberate two-file act.
        demanded = {
            ".agents/prompts/reviewer.md": (
                "mutate, don't read",
                "delete or invert",
                "stays green",
                "every mutation you make re-runs the suite",
                "plan-mandated",
            ),
            ".agents/prompts/implementer.md": (
                "failing test first",
                "mutate every test",
                "capture that failing set as a baseline",
                "escalate rather than guess",
            ),
        }
        self.assertEqual(
            set(demanded),
            set(consistency.PROMPT_REQUIRED),
            "PROMPT_REQUIRED covers a different set of prompts than this suite",
        )
        for relative, needles in demanded.items():
            with self.subTest(prompt=relative):
                self.assertEqual(
                    set(consistency.PROMPT_REQUIRED[relative]),
                    set(needles),
                    f"PROMPT_REQUIRED[{relative!r}] no longer enforces exactly "
                    "the phrases this suite demands; narrowing one is how the "
                    "gate stops catching what it was built for",
                )

    def test_a_bare_mutate_needle_would_not_pin_the_binding_instruction(self):
        # The executable justification for the full "mutate, don't read" needle.
        # Deleting the ENTIRE binding-instruction section still leaves the word
        # "mutate" in the file ("Never mutate the reviewed worktree" under What
        # you may not do), so a bare needle stays green through the exact
        # deletion it exists to catch. If this test ever goes red because the
        # incidental match is gone, the needle may safely be simplified.
        relative = ".agents/prompts/reviewer.md"
        text = (ROOT / relative).read_text(encoding="utf-8")
        without_section = re.sub(
            r"## The binding instruction.*?(?=\n## )", "", text, flags=re.S
        )
        self.assertNotEqual(without_section, text, "the strip pattern matched nothing")
        with _prompt_tree({relative: without_section}):
            self.assertEqual(
                consistency.prompt_errors({relative: ("mutate",)}),
                [],
                "a bare 'mutate' no longer matches incidentally",
            )
            self.assertTrue(
                consistency.prompt_errors({relative: ("mutate, don't read",)}),
                "the shipped needle failed to catch the section deletion",
            )

    def test_each_required_phrase_is_pinned_individually(self):
        # PROMPT_REQUIRED is a hand-written tuple, so a prompt that survives
        # losing one of its phrases means that phrase was never enforced. Strip
        # each one in turn from a copy of the real file and demand a red.
        for relative, needles in consistency.PROMPT_REQUIRED.items():
            text = (ROOT / relative).read_text(encoding="utf-8")
            for needle in needles:
                with self.subTest(prompt=relative, needle=needle):
                    damaged = re.sub(re.escape(needle), "", text, flags=re.I)
                    self.assertNotEqual(
                        damaged, text, f"{needle!r} does not appear in {relative}"
                    )
                    with _prompt_tree({relative: damaged}):
                        errors = consistency.prompt_errors({relative: needles})
                    self.assertTrue(any("omits" in e for e in errors), errors)


class OrchestrationLoopTests(unittest.TestCase):
    """The operator's loop must live in the manual agents actually read.

    A loop that exists only in an operator's head, or only in a sub-agent
    prompt the operator never opens, is not a protocol: nothing tells the next
    session that a reviewer must MUTATE, that the gate is run by the controller
    rather than reported by the author, or that findings are never fixed in the
    controller's own context.
    """

    def _manual(self) -> str:
        return (ROOT / consistency.LOOP_DOCUMENT).read_text(encoding="utf-8")

    def test_workflow_carries_the_loop_exactly_once(self):
        # Uniqueness, not mere presence. A duplicated block would make every
        # deletion mutation below silently invalid, because removing one copy
        # leaves the other behind and the gate stays green for the wrong reason.
        text = self._manual()
        self.assertEqual(text.count(consistency.LOOP_MARKER), 1)
        self.assertEqual(text.count(consistency.LOOP_END), 1)
        self.assertLess(
            text.index(consistency.LOOP_MARKER), text.index(consistency.LOOP_END)
        )

    def test_the_loop_states_the_rules_that_carry_it(self):
        block = consistency.loop_block(self._manual())
        self.assertIsNotNone(block, "the manual has no orchestration-loop block")
        lowered = block.lower()
        # A bare "reviewer" needle would be UNFALSIFIABLE here: the block links
        # `prompts/reviewer.md`, so the word is present no matter what the loop
        # says. The reviewer's INDEPENDENCE is the assertion worth making, and
        # it is the one rule below that LOOP_REQUIRED does not also pin.
        for needle in (
            "never the agent that wrote the code",
            "mutate, not read",
            "run the row's gate yourself",
            "never fix findings yourself",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, lowered)

    def test_the_loop_links_both_tracked_prompts(self):
        block = consistency.loop_block(self._manual())
        for name in ("implementer.md", "reviewer.md"):
            with self.subTest(prompt=name):
                self.assertIn(f"prompts/{name}", block)
                # The link is relative to `.agents/`, so a link that reads
                # perfectly can still resolve to nothing.
                self.assertTrue(
                    (ROOT / ".agents/prompts" / name).is_file(),
                    f"the loop links prompts/{name}, which does not exist",
                )

    def test_the_real_manual_satisfies_the_gate(self):
        # Positive control: every red below is the mutation, not the baseline.
        self.assertEqual(consistency.loop_errors(self._manual()), [])

    def test_checker_rejects_a_workflow_without_the_loop(self):
        errors = consistency.loop_errors("# workflow\n\nno loop here\n")
        self.assertTrue(errors)
        self.assertTrue(any("orchestration-loop" in e for e in errors), errors)

    def test_an_unterminated_block_is_rejected(self):
        # An opening marker with no `:end` is not a block. Without this the
        # scoping below could be satisfied by "everything after the marker".
        errors = consistency.loop_errors(
            f"# workflow\n{consistency.LOOP_MARKER}\n"
            + "\n".join(consistency.LOOP_REQUIRED)
            + "\n"
        )
        self.assertTrue(errors)

    def test_each_required_phrase_is_pinned_individually(self):
        # LOOP_REQUIRED is a hand-written tuple, so a manual that survives
        # losing one of its phrases means that phrase was never enforced.
        text = self._manual()
        for needle in consistency.LOOP_REQUIRED:
            with self.subTest(needle=needle):
                damaged = re.sub(re.escape(needle), "", text, flags=re.I)
                self.assertNotEqual(
                    damaged, text, f"{needle!r} does not appear in the manual"
                )
                self.assertTrue(consistency.loop_errors(damaged), needle)

    def test_the_needles_must_be_INSIDE_the_block(self):
        # Executable justification for scoping loop_errors to the block rather
        # than searching the whole file. `.agents/workflow.md` is a long manual
        # that already talks about gates, prompts and closing loops; a
        # whole-file search would keep a loop gutted down to its two markers
        # green on unrelated prose that happens to carry the phrases. That is
        # the "an unrelated line satisfied the assertion" failure this project
        # has now paid for repeatedly.
        gutted = "\n".join(
            [consistency.LOOP_MARKER, consistency.LOOP_END, *consistency.LOOP_REQUIRED]
        )
        self.assertTrue(consistency.loop_errors(gutted))

    def test_the_checker_enforces_the_phrases_this_suite_demands(self):
        # Every assertion above reads the MANUAL, so emptying or narrowing
        # LOOP_REQUIRED would leave them all green while the gate quietly
        # stopped looking. Equality, deliberately, not containment: a narrowing
        # (say, back to a bare "mutate") is exactly the failure to catch, and
        # containment cannot see it.
        self.assertEqual(
            set(consistency.LOOP_REQUIRED),
            {
                "prompts/implementer.md",
                "prompts/reviewer.md",
                "mutate, not read",
                "run the row's gate yourself",
                "never fix findings yourself",
            },
            "LOOP_REQUIRED no longer enforces exactly the phrases this suite "
            "demands; narrowing one is how the gate stops catching what it was "
            "built for",
        )


class OrchestrationLoopWiring(unittest.TestCase):
    """main() must CALL loop_errors, not merely define it.

    Every assertion in OrchestrationLoopTests exercises the function directly,
    so a main() that never wires it in leaves them all green while the gate
    enforces nothing -- the same drift this file exists to catch, one function
    later. InterviewWiring.test_faithful_copy_passes is the positive control
    for the temp tree these two tests run in.
    """

    STRIP = re.compile(
        r"<!-- orchestration-loop:begin -->.*?<!-- orchestration-loop:end -->\n?", re.S
    )

    def test_main_fails_when_the_loop_is_deleted(self):
        text = (ROOT / consistency.LOOP_DOCUMENT).read_text(encoding="utf-8")
        stripped = self.STRIP.sub("", text)
        self.assertNotEqual(stripped, text, "the strip pattern matched nothing")
        self.assertNotIn(consistency.LOOP_MARKER, stripped)
        # The interview must SURVIVE the strip: otherwise a red here would be
        # interview_errors firing and would prove nothing about the loop.
        self.assertEqual(consistency.interview_errors(stripped), [])
        with _repo_copy(stripped) as run:
            code, _, err = run()
        self.assertEqual(code, 1)
        self.assertIn("missing the orchestration-loop block", err)

    def test_main_fails_when_the_loop_loses_one_phrase(self):
        # The marker check and the needle loop are two different wirings. A
        # main() that only saw the marker would pass the test above and let a
        # block drift into saying nothing binding.
        text = (ROOT / consistency.LOOP_DOCUMENT).read_text(encoding="utf-8")
        needle = "run the row's gate yourself"
        self.assertEqual(
            text.lower().count(needle), 1, f"{needle!r} is not a unique anchor"
        )
        damaged = re.sub(re.escape(needle), "", text, flags=re.I)
        self.assertIn(consistency.LOOP_MARKER, damaged)
        with _repo_copy(damaged) as run:
            code, _, err = run()
        self.assertEqual(code, 1)
        self.assertIn("loop omits", err)


if __name__ == "__main__":
    unittest.main()
