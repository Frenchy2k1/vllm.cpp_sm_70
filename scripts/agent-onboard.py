#!/usr/bin/env python3
"""Report what a session has not resolved yet, and record what it answered. (A)

This script REPORTS, and writes exactly one thing: a .env value it was handed.
It never asks and it never decides, because no harness-neutral mechanism exists
for a shell script to run an interactive prompt, and a hook injects text rather
than conversing. The split is fixed:

    this script   -> detect and report; record an answer it is given
    the agent     -> ask, using the interview in .agents/workflow.md
    agent-role.py -> make the answer a fact

    scripts/agent-onboard.py --probe             # human-readable state
    scripts/agent-onboard.py --probe --json      # machine-readable
    scripts/agent-onboard.py --env-set KEY=VALUE # record one answered value

`--env-set` never invents anything. Only keys .env.example declares may be
written, and an unanswered key stays EMPTY -- empty means the gates that need
it stay PENDING, which is an honest state and not a failure. Never infer a
value from a username, a filesystem path, a machine identity or another
developer's setup.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


role_mod = _load("agent_role", "scripts/agent-role.py")

ENV_EXAMPLE = ROOT / ".env.example"
ENV_FILE = ROOT / ".env"


def _example_keys() -> tuple[str, ...]:
    """Keys the tracked example declares. The ONLY source of legal keys."""
    keys = []
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and "=" in line:
            keys.append(line.split("=", 1)[0])
    return tuple(keys)


ENV_KEYS = _example_keys()


def env_state_from_text(text: str) -> tuple[str, list[str]]:
    """Classify .env content: present | incomplete, plus the unset keys."""
    values = {}
    for line in text.splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    missing = [key for key in ENV_KEYS if not values.get(key)]
    return ("incomplete" if missing else "present"), missing


def env_state(path: Path = ENV_FILE) -> tuple[str, list[str]]:
    """Classify the .env file. A missing FILE is distinct from missing VALUES.

    A file that exists but cannot be read is a THIRD case: reporting it as
    absent would send the agent to create a file that is already there, and
    reporting it as complete would hide every unresolved value. It gets its
    own status, and like a missing file it resolves nothing.
    """
    if not path.exists():
        return "missing", list(ENV_KEYS)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return "unreadable", list(ENV_KEYS)
    return env_state_from_text(text)


def queue_state() -> tuple[list[str], str | None]:
    """The READY queue, plus why it is unavailable when it is.

    ready-for-helper.py's `queue()` IS the computation, so the probe calls it
    instead of re-parsing that script's prose: its listing truncates at 40 rows
    and its header line ("READY-FOR-HELPER queue: N row(s)") reads like a row
    ID to any token filter, so an empty queue would report a phantom row.

    A queue that could not be computed is NOT an empty queue, exactly as an
    unreadable .env is not an absent one, so the failure is carried out rather
    than swallowed. The probe still never raises: it reports and does not gate.
    """
    try:
        helper = sys.modules.get("ready_for_helper") or _load(
            "ready_for_helper", "scripts/ready-for-helper.py"
        )
        pickable, _ = helper.queue()
    except Exception as error:  # a broken record must not crash the probe
        return [], f"{type(error).__name__}: {error}"
    return [row.item_id for row in pickable], None


def ready_rows() -> list[str]:
    """The READY queue alone. Callers that must tell empty from broken apart
    use queue_state()."""
    return queue_state()[0]


def probe() -> dict:
    state = role_mod.resolve()
    status, missing = env_state()
    rows, queue_error = queue_state()
    return {
        "role": state.get("role"),
        "row": state.get("row"),
        # Who else is coordinating right now, straight from resolve(). This
        # replaced `blocked_by_other_operator` when the operator lock became a
        # RECORD (issue #285): a recorded peer never blocks a claim, so
        # reporting it as a blocker sent sessions away from a command that
        # succeeds. Dropping it instead would make this front door LESS honest
        # than the tool it wraps -- "who is working where" is the whole reason
        # the file is kept.
        "operator_peers": state.get("operator_peers") or [],
        "reason": state.get("reason"),
        # The role tool owns both facts. Keep the branch from resolve() and
        # expose the same per-worktree identity used by role markers/locks so
        # downstream routers report real materialized state rather than a
        # renderer fixture's invented fields.
        "branch": state.get("branch"),
        "worktree": role_mod.worktree_id(),
        # resolve() now carries this (step 2). Still read with .get and still
        # rendered as a DEFAULT when absent: headless is never inferred, so a
        # state that carries no mode must not read as a declaration either.
        "mode": state.get("mode"),
        "env": status,
        "env_missing": missing,
        "queue": rows,
        "queue_error": queue_error,
    }


def render_probe(state: dict) -> str:
    role = state["role"] or "UNDECLARED"
    row = f" row={state['row']}" if state.get("row") else ""
    mode = state.get("mode") or "interactive (default, not declared)"
    if state.get("queue_error"):
        queue_line = f"queue: UNAVAILABLE ({state['queue_error']})"
    else:
        queue_line = f"queue: {len(state['queue'])} READY rows" + (
            f" — {', '.join(state['queue'][:5])}" if state["queue"] else ""
        )
    lines = [
        f"role: {role}{row}   mode: {mode}",
        f"branch: {state.get('branch') or 'unavailable'}   "
        f"worktree: {state.get('worktree') or 'unavailable'}",
        f".env: {state['env']}"
        + (f" (unset: {', '.join(state['env_missing'])})" if state["env_missing"] else ""),
        queue_line,
    ]
    # Rendered whatever this session's role is: any session may need to know
    # who else is coordinating. It is information and never an obstacle --
    # `claim operator` is never refused (issue #285).
    lines.extend(role_mod.render_peers(state.get("operator_peers") or []))
    if state["role"] is None:
        lines.append(
            "This session has not declared a role. Ask what the work is, then claim: "
            "a long campaign -> operator; one scoped change -> helper --row <ROW-ID>; "
            "just looking -> read-only. See .agents/workflow.md."
        )
    return "\n".join(lines)


def _render(value: str) -> str:
    """Render a value so the file's TWO readers agree on it.

    .env.example documents the loader as `set -a; . ./.env; set +a`, so the file
    is shell as well as data. An unquoted `/pa th` makes that loader run `th`
    ("command not found") and leave the variable EMPTY, while the probe's own
    parser reads the same line as a set value -- the probe reports PRESENT and
    the gate stays PENDING, which is the exact confusion this command exists to
    prevent. shlex.quote only adds quotes when they are needed, so ordinary
    paths are written unchanged.

    Empty is the one value left bare: shlex.quote("") is `''`, and the probe
    counts that two-character string as SET. Unanswered must keep reading as
    unanswered.
    """
    return shlex.quote(value) if value else ""


def cmd_env_set(pair: str) -> int:
    """Write one .env value. Refuses any key .env.example does not declare."""
    if "=" not in pair:
        print("ERROR: expected KEY=VALUE", file=sys.stderr)
        return 2
    key, value = pair.split("=", 1)
    key = key.strip()
    if key not in ENV_KEYS:
        # A typo would sit in .env doing nothing while the gate that wanted the
        # real key stays mysteriously PENDING.
        print(
            f"ERROR: {key} is not declared in .env.example. Never invent a key; "
            f"legal keys are: {', '.join(ENV_KEYS)}",
            file=sys.stderr,
        )
        return 2
    # The value is the one field nothing else validates. A line separator inside
    # it forges a whole extra .env line that no key check ever saw -- the same
    # silent clobber as an unrecognised key, through the back door.
    #
    # Ask the QUESTION rather than enumerate characters: "\n" and "\r" are 2 of
    # the 10 separators str.splitlines() breaks on, and it is splitlines() that
    # both env_state_from_text and the rewrite below use, so a "\v" or a U+2028
    # smuggled a forged pair past the key check and the probe then reported the
    # forged key as SET. An empty value splits to [] and is legal, so it is the
    # one case this cannot phrase as a round trip.
    if value and value.splitlines() != [value]:
        print(
            f"ERROR: the value for {key} contains a line separator, which would "
            "forge a second .env line. Pass a single-line value.",
            file=sys.stderr,
        )
        return 2
    if not ENV_FILE.exists():
        # Seed from the tracked example so every OTHER declared key survives,
        # commented and empty, instead of a one-line .env that hides the rest.
        ENV_FILE.write_text(ENV_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    try:
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        # env_state treats an existing-but-unreadable .env as its own third
        # case rather than as absent. Writing must agree: a bare traceback
        # tells the caller nothing, and "create one" is the wrong instruction
        # for a file that is already there.
        print(f"ERROR: .env exists but cannot be read ({error})", file=sys.stderr)
        return 2
    # Rewrite EVERY match, not just the first. A hand-maintained .env routinely
    # carries an override appended at the bottom, and both readers here and
    # `set -a; . ./.env` take the LAST assignment, so breaking on the first left
    # the file changed, the exit code 0, the message reassuring -- and the
    # effective value exactly what it was. Collapse to one line, keeping the
    # first line's position so the example's grouping and comments still read.
    matches = [
        index
        for index, line in enumerate(lines)
        if not line.startswith("#") and line.split("=", 1)[0].strip() == key
    ]
    if matches:
        lines[matches[0]] = f"{key}={_render(value)}"
        for index in reversed(matches[1:]):
            del lines[index]
    else:
        lines.append(f"{key}={_render(value)}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # An empty value is a legitimate answer -- it means UNAVAILABLE, and the
    # gates that need it stay PENDING. Say so rather than let it read as a win.
    print(f"set {key} in .env" + ("" if value else " (empty: gates stay PENDING)"))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report unresolved session state.")
    parser.add_argument("--probe", action="store_true", help="report session state")
    parser.add_argument("--json", action="store_true", help="machine-readable probe")
    parser.add_argument("--env-set", metavar="KEY=VALUE", help="write one .env value")
    args = parser.parse_args(argv)

    # `is not None`, not truthiness: `--env-set ''` is a malformed write, and
    # falling through to the probe would print a state report and exit 0 having
    # recorded nothing -- a silent no-op is the failure mode this command is
    # built to avoid. cmd_env_set refuses it out loud instead.
    if args.env_set is not None:
        return cmd_env_set(args.env_set)

    state = probe()
    print(json.dumps(state, indent=2, sort_keys=True) if args.json else render_probe(state))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
