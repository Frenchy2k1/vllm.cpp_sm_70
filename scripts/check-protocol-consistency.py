#!/usr/bin/env python3
"""Keep the protocol prose and the checkers that enforce it in agreement.

The failure this exists to prevent is real and already happened: the
same-change public-document obligation was migrated from README.md to
docs/STATUS.md in scripts/check-doc-checkpoint.py, AGENTS.md was updated, and
`.agents/workflow.md` -- the operating manual an agent is told to follow every
session -- was not. For a while the manual instructed agents to update README.md
at every checkpoint, which is exactly the drift the migration removed. Prose and
checker disagreed, and the prose is what agents actually read.

So the obligated surfaces are declared ONCE, as a machine-readable contract
block that both documents carry verbatim, and this gate asserts the block equals
the constants in scripts/check-doc-checkpoint.py. Changing the checker without
changing the prose (or the reverse) is a red build, not a silent divergence.

The contract block looks like this, and is a normal Markdown table to a reader:

    <!-- doc-obligation-contract:begin -->
    | Public surface | Owed by |
    |---|---|
    | `docs/STATUS.md` | every feature/iteration checkpoint |
    <!-- doc-obligation-contract:end -->

The same gate now also asserts that `.agents/workflow.md` carries the ROLE
INTERVIEW, between `<!-- role-interview:begin -->` and its `:end`. That is the
same failure with the polarity flipped: agent-preflight.sh refuses a session
that has not declared a role, so an agent who is never told the question, or
never told that `read-only` is one of the answers, meets a red gate with no
instructions -- and a gate people cannot satisfy is a gate people route around.

The same gate also asserts that `.agents/workflow.md` carries the ORCHESTRATION
LOOP, between `<!-- orchestration-loop:begin -->` and its `:end`. The prompts
below are what a sub-agent is handed; the loop is what the OPERATOR does with
them, and until it was written here it lived nowhere an agent reads -- neither
`AGENTS.md` nor this manual said that a reviewer must mutate rather than read,
that the controller runs the row's gate itself instead of taking the
implementer's word, or that findings are never fixed in the controller's own
session.

The same gate finally asserts that the sub-agent prompts under `.agents/prompts`
exist and still carry their binding instructions. Every Important finding across
two branches of this project came from an INDEPENDENT reviewer sub-agent, none
from an implementer's self-review, and the reviewers found them by MUTATING code
rather than reading diffs: eleven tests passed with the thing they named
deleted, and not one was visible by reading. That instruction is the deliverable,
so it is tracked and pinned phrase by phrase, not merely present as a file.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Both documents must carry the contract, because both are read as normative:
# AGENTS.md is the canonical index, workflow.md is the session operating manual.
CONTRACT_DOCUMENTS = ("AGENTS.md", ".agents/workflow.md")

BEGIN = "<!-- doc-obligation-contract:begin -->"
END = "<!-- doc-obligation-contract:end -->"

# The session manual must carry the role interview, because agent-preflight.sh
# now FAILS a session that has not declared a role. A gate whose precondition is
# written down nowhere is the same drift this file exists to prevent, with the
# polarity flipped: instead of prose demanding what the checker dropped, the
# checker demands what no prose ever taught.
INTERVIEW_DOCUMENT = ".agents/workflow.md"
INTERVIEW_MARKER = "<!-- role-interview:begin -->"
INTERVIEW_REQUIRED = ("claim operator", "claim helper --row", "claim read-only", "--headless")

# The same manual must carry the operator's LOOP. The prompts in PROMPT_REQUIRED
# below are handed to sub-agents; nothing told the operator how to run one, and
# the three rules that carry the whole return are exactly the ones an operator
# improvises away: dispatch a FRESH reviewer whose instruction is to MUTATE,
# run the row's gate YOURSELF rather than believing the author's report, and
# never repair a finding in the coordinating session.
#
# The needles are matched INSIDE the block, not across the whole document.
# workflow.md is a long manual that already discusses gates, prompts and
# "closing the loop", so a whole-file search would keep a loop gutted down to
# its two markers green on unrelated prose -- the same "an unrelated line
# satisfied the assertion" failure the reviewer prompt is written to catch.
LOOP_DOCUMENT = ".agents/workflow.md"
LOOP_MARKER = "<!-- orchestration-loop:begin -->"
LOOP_END = "<!-- orchestration-loop:end -->"
LOOP_REQUIRED = (
    "prompts/implementer.md",
    "prompts/reviewer.md",
    "mutate, not read",
    "run the row's gate yourself",
    "never fix findings yourself",
)

# The reviewer prompt's value is the MUTATION instruction; a reviewer told only
# to "review" reads the diff, and reading found none of the eleven tests that
# passed with their subject deleted. Pin the instruction, not the file.
#
# The reviewer needle is the full "mutate, don't read", not a bare "mutate":
# the prompt also says "never mutate the reviewed worktree" further down, so the
# short form would stay satisfied by an unrelated sentence after the binding
# instruction was deleted. That is the same "an unrelated line satisfied the
# assertion" failure the prompt itself is written to catch.
#
# Two needles pin REPAIRS to earlier drafts of these prompts, because a prompt
# that once contradicted itself can drift back: the reviewer prompt used to
# forbid re-running the full suite (which reads as a budget on the mutations it
# demands two sections earlier), and the implementer prompt used to demand a
# green gate with no answer for reds that were already there before the work
# started, whose only exits were stalling or an allowlist.
PROMPT_REQUIRED = {
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

# A path in a table cell, e.g. `docs/STATUS.md`.
CELL_PATH = re.compile(r"`([^`]+\.md)`")

# README.md is a landing page, not a checkpoint surface. Naming it inside the
# contract is the specific regression this gate was built after, so it earns a
# targeted message instead of a bare set-difference.
FORBIDDEN_IN_CONTRACT = {
    "README.md": (
        "README.md is a user-facing landing page, not a per-checkpoint status "
        "surface; it changes only when a user-visible headline shifts. The "
        "per-capability obligation belongs to docs/STATUS.md"
    ),
}


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def obligated_surfaces() -> tuple[str, ...]:
    """Return the surfaces check-doc-checkpoint.py actually enforces."""
    checkpoint = _load("doc_checkpoint", "scripts/check-doc-checkpoint.py")
    return tuple(checkpoint.PUBLIC_CHECKPOINTS) + (checkpoint.FEATURE_CHECKPOINT,)


def contract_paths(text: str) -> list[str] | None:
    """Return the paths declared in the contract block, or None if absent."""
    start = text.find(BEGIN)
    end = text.find(END)
    if start == -1 or end == -1 or end < start:
        return None
    block = text[start + len(BEGIN) : end]
    paths: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        found = CELL_PATH.findall(stripped)
        if found:
            paths.append(found[0])
    return paths


def document_errors(name: str, text: str, expected: tuple[str, ...]) -> list[str]:
    """Return contract problems for one normative document."""
    paths = contract_paths(text)
    if paths is None:
        return [
            f"{name} is missing the doc-obligation contract block "
            f"({BEGIN} ... {END}); it must declare the surfaces that "
            "scripts/check-doc-checkpoint.py enforces so prose cannot drift "
            "from the gate"
        ]

    errors: list[str] = []
    for path in paths:
        if path in FORBIDDEN_IN_CONTRACT:
            errors.append(f"{name} contract names {path}: {FORBIDDEN_IN_CONTRACT[path]}")

    if tuple(paths) != expected:
        errors.append(
            f"{name} contract declares {paths!r} but "
            f"scripts/check-doc-checkpoint.py enforces {list(expected)!r}; "
            "update the prose and the checker in the same change"
        )
    return errors


def interview_errors(text: str) -> list[str]:
    """The role interview must live where agents read it, not only in a gate."""
    if INTERVIEW_MARKER not in text:
        return [f"{INTERVIEW_DOCUMENT} is missing the role-interview block"]
    return [
        f"{INTERVIEW_DOCUMENT} role interview omits {needle!r}"
        for needle in INTERVIEW_REQUIRED
        if needle not in text
    ]


def loop_block(text: str) -> str | None:
    """Return the orchestration-loop block's body, or None if there isn't one.

    An opening marker with no `:end` is NOT a block. Treating it as "everything
    after the marker" would silently widen the scope back to the whole
    document, which is the incidental-match hole this scoping exists to close.
    """
    start = text.find(LOOP_MARKER)
    if start == -1:
        return None
    end = text.find(LOOP_END, start)
    if end == -1:
        return None
    return text[start + len(LOOP_MARKER) : end]


def loop_errors(text: str) -> list[str]:
    """The operator's loop must live where agents read it, not in a prompt."""
    block = loop_block(text)
    if block is None:
        return [
            f"{LOOP_DOCUMENT} is missing the orchestration-loop block "
            f"({LOOP_MARKER} ... {LOOP_END}); the sub-agent prompts say what a "
            "reviewer or implementer does, and nothing else says what the "
            "OPERATOR does with them"
        ]
    lowered = block.lower()
    return [
        f"{LOOP_DOCUMENT} loop omits {needle!r}"
        for needle in LOOP_REQUIRED
        if needle.lower() not in lowered
    ]


def prompt_errors(required: dict[str, tuple[str, ...]] | None = None) -> list[str]:
    """Each tracked prompt exists and carries its binding instruction."""
    # `required or PROMPT_REQUIRED` would silently promote an explicitly EMPTY
    # spec into the full live check, which is this repo's recurring defect
    # class: an absence and a value that look the same. Only a missing argument
    # means "use the default".
    errors: list[str] = []
    spec = PROMPT_REQUIRED if required is None else required
    for relative, needles in spec.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"{relative} is missing; the prompt is the protocol")
            continue
        text = path.read_text(encoding="utf-8").lower()
        errors.extend(
            f"{relative} omits {needle!r}"
            for needle in needles
            if needle.lower() not in text
        )
    return errors


def main() -> int:
    expected = obligated_surfaces()
    failures: list[str] = []
    blocks: dict[str, list[str] | None] = {}

    interview = ROOT / INTERVIEW_DOCUMENT
    if not interview.exists():
        failures.append(f"{INTERVIEW_DOCUMENT} does not exist")
    else:
        failures.extend(
            interview_errors(interview.read_text(encoding="utf-8"))
        )

    # INTERVIEW_DOCUMENT and LOOP_DOCUMENT are the same manual today, but the
    # two obligations are independent and either may move, so each resolves its
    # own path rather than sharing one read.
    loop_doc = ROOT / LOOP_DOCUMENT
    if not loop_doc.exists():
        failures.append(f"{LOOP_DOCUMENT} does not exist")
    else:
        failures.extend(loop_errors(loop_doc.read_text(encoding="utf-8")))

    failures.extend(prompt_errors())

    for name in CONTRACT_DOCUMENTS:
        path = ROOT / name
        if not path.exists():
            failures.append(f"{name} does not exist")
            continue
        text = path.read_text(encoding="utf-8")
        blocks[name] = contract_paths(text)
        failures.extend(document_errors(name, text, expected))

    present = {name: paths for name, paths in blocks.items() if paths is not None}
    if len(present) == len(CONTRACT_DOCUMENTS):
        distinct = {tuple(paths) for paths in present.values()}
        if len(distinct) > 1:
            failures.append(
                "the doc-obligation contract differs between "
                f"{' and '.join(CONTRACT_DOCUMENTS)}; both must carry the same "
                "block verbatim"
            )

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        print(
            "The obligated public surfaces are defined by PUBLIC_CHECKPOINTS and "
            "FEATURE_CHECKPOINT in scripts/check-doc-checkpoint.py. Mirror them "
            "in the contract block of every document listed in "
            "CONTRACT_DOCUMENTS. The role interview is the block between "
            f"{INTERVIEW_MARKER} and its :end in {INTERVIEW_DOCUMENT}; it must "
            "name every answer agent-role.py accepts. The operator's loop is "
            f"the block between {LOOP_MARKER} and its :end in {LOOP_DOCUMENT}; "
            f"it must carry {', '.join(repr(n) for n in LOOP_REQUIRED)} inside "
            "the block. The sub-agent prompts in "
            f"{', '.join(PROMPT_REQUIRED)} must carry their binding "
            "instructions verbatim; a prompt that lives only in an operator's "
            "head is not a protocol.",
            file=sys.stderr,
        )
        return 1

    print(
        "OK: the doc-obligation contract in "
        f"{' and '.join(CONTRACT_DOCUMENTS)} matches "
        f"scripts/check-doc-checkpoint.py, {INTERVIEW_DOCUMENT} carries the "
        f"role interview and the orchestration loop, and "
        f"{len(PROMPT_REQUIRED)} sub-agent prompts carry their binding "
        "instructions."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
