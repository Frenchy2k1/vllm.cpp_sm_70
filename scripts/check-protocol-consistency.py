#!/usr/bin/env python3
"""Keep structured policy consumers and their protocol artifacts consistent.

Public-document obligations are controlled policy rows. The document checker
fully parses their stable ID, exact scope, semantic trigger, positive action,
and enforcement list; explanatory details remain in procedures.

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

The same gate finally runs the closed runtime-prompt grammar. Phrase presence
cannot establish a method contract: contradictory prose may retain every
required phrase. The semantic validator therefore parses every nonempty line.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.policy_contract import PolicyRule, load_policy

# The session manual must carry the role interview, because agent-preflight.sh
# now FAILS a session that has not declared a role. A gate whose precondition is
# written down nowhere is the same drift this file exists to prevent, with the
# polarity flipped: instead of prose demanding what the checker dropped, the
# checker demands what no prose ever taught.
INTERVIEW_DOCUMENT = ".agents/workflow.md"
INTERVIEW_MARKER = "<!-- role-interview:begin -->"
INTERVIEW_REQUIRED = ("claim operator", "claim helper --row", "claim read-only", "--headless")

# The authoritative policy routes session start through one executable before
# role declaration and preflight. Welcome copy stays in source, not context.
ENTRYPOINT_DOCUMENT = ".agents/workflow.md"
ENTRYPOINT_MARKER = "<!-- session-entrypoint:begin -->"
ENTRYPOINT_END = "<!-- session-entrypoint:end -->"
ENTRYPOINT_REQUIRED = (
    "POL-BOOT-ENTRYPOINT",
    "scripts/agent-start.py",
    "--intent",
    "scripts/agent-preflight.sh",
)

# The same manual must carry the operator's LOOP. The prompts handed to
# sub-agents cannot by themselves tell the operator how to run one, and
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
    "repeat this cycle until pass",
    "attempt and retry budgets are scheduling controls",
    "never terminal blockers for correctable findings",
    "explicit developer direction",
    "precise external authority or resource blocker",
)

CUTOVER_WIRING = {
    "scripts/agent-preflight.sh": (
        "check-policy",
        "check-prompt-contract",
        'run "state-record range" python3 scripts/check-state-record.py',
        '--base origin/main\n  run "now-current range"',
        'run "now-current range" python3 scripts/check-now-current.py',
        '--base origin/main --head HEAD\n  run "doc-checkpoint range"',
        "test_state_record_core",
        "test_check_state_record",
        "test_agent_gates",
        "check-commit-trailers.py",
    ),
    ".agents/workflow.md": (
        "scripts/agent-ready.py",
        "scripts/agent-integration.py --base origin/main",
        "network-independent",
    ),
    ".github/workflows/ci.yml": (
        "scripts/check-policy.py",
        'check-state-record.py --base "$base"',
        'check-now-current.py --base "$base" --head "$head"',
        "tests/scripts/test_state_record_core.py",
        "tests/scripts/test_check_state_record.py",
        "tests/scripts/test_agent_gates.py",
        ".agents/policy-cutover",
    ),
    ".githooks/pre-push": (
        "CHECKERS=(check-policy.py",
        "check-prompt-contract.py",
        'check-state-record.py) args=(--base "$base")',
        'check-now-current.py) args=(--base "$base" --head "$sha")',
    ),
    "scripts/agent-ready.py": (
        "REMOTE_UNVERIFIED",
        "if not run_local_preflight():",
        "query_remote(expected)",
    ),
    "scripts/agent-integration.py": (
        "reviewDecision",
        "if not run_ready(args.pr_json):",
        'f"{args.base}..HEAD"',
        "errors = ready.ready_errors(payload, expected)",
        "check-commit-trailers.py",
        ".agents/policy-cutover",
    ),
}

RETIRED_STATE_WIRING = {
    "scripts/agent-preflight.sh": ("check-state-order", "test_check_state_order"),
    ".github/workflows/ci.yml": (
        "scripts/check-state-order.py",
        "tests/scripts/test_check_state_order.py",
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


def public_document_rule_errors(
    rules: dict[str, PolicyRule] | None = None,
) -> list[str]:
    """Validate every public rule through the document checker's closed parser."""

    checkpoint = _load(
        "doc_checkpoint_for_consistency", "scripts/check-doc-checkpoint.py"
    )
    policy = load_policy(ROOT) if rules is None else rules
    errors: list[str] = checkpoint.public_namespace_errors(policy)
    seen_surfaces: dict[str, str] = {}

    for rule_id in checkpoint.PUBLIC_RULE_IDS:
        rule = policy.get(rule_id)
        if rule is None:
            errors.append(f"public-document checker requires missing policy rule {rule_id}")
            continue
        try:
            binding = checkpoint.parse_public_rule(rule)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        previous = seen_surfaces.setdefault(binding.surface, rule_id)
        if previous != rule_id:
            errors.append(
                f"public surface {binding.surface!r} is duplicate in "
                f"{previous} and {rule_id}"
            )

    if rules is None and not errors:
        try:
            bindings = checkpoint.public_rule_bindings(ROOT)
        except ValueError as exc:
            errors.extend(str(exc).splitlines())
        else:
            if set(bindings) != set(checkpoint.PUBLIC_RULE_IDS):
                errors.append("document-checker bindings omit a required public rule")
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


def entrypoint_block(text: str) -> str | None:
    """Return the bounded universal-entrypoint procedure, when complete."""
    start = text.find(ENTRYPOINT_MARKER)
    if start == -1:
        return None
    end = text.find(ENTRYPOINT_END, start)
    if end == -1:
        return None
    return text[start + len(ENTRYPOINT_MARKER) : end]


def entrypoint_errors(text: str) -> list[str]:
    """Require the canonical start command to precede preflight."""
    block = entrypoint_block(text)
    if block is None:
        return [
            f"{ENTRYPOINT_DOCUMENT} is missing the session-entrypoint block "
            f"({ENTRYPOINT_MARKER} ... {ENTRYPOINT_END})"
        ]
    errors = [
        f"{ENTRYPOINT_DOCUMENT} session entrypoint omits {needle!r}"
        for needle in ENTRYPOINT_REQUIRED
        if needle not in block
    ]
    start = block.find("scripts/agent-start.py")
    preflight = block.find("scripts/agent-preflight.sh")
    if start != -1 and preflight != -1 and start > preflight:
        errors.append(
            f"{ENTRYPOINT_DOCUMENT} session entrypoint must route agent-start before preflight"
        )
    return errors


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


def prompt_contract_errors() -> list[str]:
    """Validate every runtime prompt through the closed semantic grammar."""

    checker = _load(
        "prompt_contract_for_consistency", "scripts/check-prompt-contract.py"
    )
    return checker.repository_errors(ROOT, set(load_policy(ROOT)))


def cutover_wiring_errors(root: Path | None = None) -> list[str]:
    """Bind the local/ready/integration separation and its backstops."""

    repository = root or ROOT
    errors: list[str] = []
    texts: dict[str, str] = {}
    for relative, needles in CUTOVER_WIRING.items():
        path = repository / relative
        if not path.is_file():
            errors.append(f"cutover wiring is missing {relative}")
            continue
        content = path.read_text(encoding="utf-8")
        texts[relative] = content
        for needle in needles:
            if needle not in content:
                errors.append(f"cutover wiring {relative} omits {needle!r}")
    for relative, retired_needles in RETIRED_STATE_WIRING.items():
        path = repository / relative
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        for needle in retired_needles:
            if needle in content:
                errors.append(f"cutover wiring {relative} retains retired {needle!r}")
    preflight = texts.get("scripts/agent-preflight.sh", "")
    for remote_entrypoint in ("agent-ready.py", "agent-integration.py", "gh pr"):
        if remote_entrypoint in preflight:
            errors.append(
                f"network-independent preflight invokes remote surface {remote_entrypoint!r}"
            )
    cutover = repository / ".agents/policy-cutover"
    if not cutover.is_file():
        errors.append("cutover wiring is missing .agents/policy-cutover")
    elif re.fullmatch(r"[0-9a-f]{40}\n", cutover.read_text(encoding="utf-8")) is None:
        errors.append(".agents/policy-cutover must contain one lowercase 40-hex commit")
    return errors


def main() -> int:
    failures: list[str] = []
    failures.extend(public_document_rule_errors())

    interview = ROOT / INTERVIEW_DOCUMENT
    if not interview.exists():
        failures.append(f"{INTERVIEW_DOCUMENT} does not exist")
    else:
        failures.extend(
            interview_errors(interview.read_text(encoding="utf-8"))
        )

    entrypoint = ROOT / ENTRYPOINT_DOCUMENT
    if not entrypoint.exists():
        failures.append(f"{ENTRYPOINT_DOCUMENT} does not exist")
    else:
        failures.extend(entrypoint_errors(entrypoint.read_text(encoding="utf-8")))

    # INTERVIEW_DOCUMENT and LOOP_DOCUMENT are the same manual today, but the
    # two obligations are independent and either may move, so each resolves its
    # own path rather than sharing one read.
    loop_doc = ROOT / LOOP_DOCUMENT
    if not loop_doc.exists():
        failures.append(f"{LOOP_DOCUMENT} does not exist")
    else:
        failures.extend(loop_errors(loop_doc.read_text(encoding="utf-8")))

    failures.extend(prompt_contract_errors())
    failures.extend(cutover_wiring_errors())

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        print(
            "Public-document policy rows must fully parse through "
            "scripts/check-doc-checkpoint.py. The role interview is the block between "
            f"{INTERVIEW_MARKER} and its :end in {INTERVIEW_DOCUMENT}; it must "
            "name every answer agent-role.py accepts. The session entrypoint is "
            f"the block between {ENTRYPOINT_MARKER} "
            f"and {ENTRYPOINT_END} in {ENTRYPOINT_DOCUMENT}; it must route "
            "scripts/agent-start.py before scripts/agent-preflight.sh. "
            f"The operator's loop is the block between {LOOP_MARKER} and its "
            f":end in {LOOP_DOCUMENT}; "
            f"it must carry {', '.join(repr(n) for n in LOOP_REQUIRED)} inside "
            "the block. Every runtime prompt must satisfy the closed grammar "
            "in scripts/check-prompt-contract.py; unknown prose is a failure.",
            file=sys.stderr,
        )
        return 1

    print(
        "OK: public-document policy matches scripts/check-doc-checkpoint.py, "
        f"{INTERVIEW_DOCUMENT} carries the "
        "role interview, universal session entrypoint, and orchestration loop, "
        "all runtime prompts satisfy "
        "the closed semantic contract, and cutover wiring is complete."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
