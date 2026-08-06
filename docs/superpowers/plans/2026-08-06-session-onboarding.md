# Session Onboarding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing role obligation actually fire at session start, and make answering it pleasant — the agent asks about the work, the role follows, and a session that only reads never has to fake a claim.

**Architecture:** A fixed three-way split, because a shell script cannot ask a question and a hook cannot converse. `scripts/agent-onboard.py --probe` **reports** state and writes nothing; the **agent** asks, using the canonical interview in `.agents/workflow.md`; `scripts/agent-role.py` and `--env-set` **make the answer a fact**. `agent-preflight.sh` is the harness-neutral trigger: `--require-role` becomes the default, so the first command of a session demands a decision.

**Tech Stack:** Python 3 standard library only, `argparse`, `importlib.util` for loading hyphenated modules in tests, `unittest`. Bash for `agent-preflight.sh`. Matches the house style of `scripts/check-agent-record.py` and `scripts/agent-role.py`.

## Global Constraints

Copied from `AGENTS.md` and `.agents/specs/session-onboarding.md`. Every task's requirements implicitly include this section.

- **Every commit carries `FOLLOWING_AGENTS_PROTOCOL`** plus `Assisted-by: Claude Code:claude-opus-5 [ClaudeCode]`. **Never** `Signed-off-by` or `Co-Authored-By` from an AI.
- **Run `bash scripts/agent-preflight.sh` before committing; it must exit 0.** Never pipe it — redirect to a file and check `$?`.
- **Every commit touching `scripts/`, `tests/` or `.agents/specs/` also updates `docs/STATUS.md` and `docs/BENCHMARKS.md` in the SAME commit.** Verify the committed form explicitly with `python3 scripts/check-doc-checkpoint.py --commit <sha>` — preflight only runs that checker `--staged`, which passes vacuously after committing. `docs/STATUS.md` sits under a shrink-only char ratchet in `scripts/check-public-doc-tables.py`; if you add text there, stay under the cap or offset it and **lower** the ratchet. Never raise it. `docs/BENCHMARKS.md` is at its 35-prose-paragraph budget and a 700-char-per-paragraph limit — extend the existing "NOT APPLICABLE" paragraph rather than adding a new one.
- **Use a ROLLING doc surface, do not append per task.** Task 1 already added the entry on both pages and left `docs/STATUS.md` at 284,071 of a 284,081 cap and the BENCHMARKS paragraph at 676 of 700. There is no room for five separate additions. Every later task **edits the line Task 1 wrote** — rolling it forward **digit-only**: "step 1/5" → "2/5" → … → "step 5/5". `docs/STATUS.md` now sits at **exactly** its 284,081 cap with zero headroom and the BENCHMARKS paragraph at 699 of 700, so a digit roll is free and **anything longer is red** — never end the sequence with "all 5", which is two characters more and fails. If your entry needs more room, shorten *your own* sentence; never someone else's. Never compact unrelated evidence to buy room: on the previous branch that produced a cross-arm supersession claim that contradicted a paragraph two lines below it.
- **Python standard library only.** `from __future__ import annotations`, type hints, house style.
- **Never weaken a checker to make something pass. Repair the record.**
- **`read-only` is a declared ABSENCE of claim, not a third role.** The claimable roles stay exactly `("operator", "helper")`.
- **Headless is never inferred** — not from the hour, not from silence, not from a long task. It is declared.
- **Never infer a `.env` value** from a username, filesystem path, machine identity, or another developer's paths. Unanswered means empty, and empty means the gates that need it stay `PENDING`.
- Stage explicit paths. Never `git add -A`.

**Existing interfaces you build on** (read, do not reimplement):

- `scripts/agent-role.py`: `ROLES = ("operator", "helper")` (line 41), `UNDECLARED_EXIT = 3` (line 47), `session_id() -> str`, `marker_path() -> Path`, `lock_path() -> Path`, `read_json(path) -> dict | None`, `current_branch() -> str`, `resolve() -> dict`, `cmd_show`, `cmd_claim`, `cmd_heartbeat`, `cmd_release`, `main()`. The marker JSON is `{"role", "row", "session", "at"}`.
- `scripts/agent-preflight.sh`: arg loop at lines 24–32, `REQUIRE_ROLE=0` at line 23, `CHECKERS=(...)` at line 34, the `SUITES=(...)` list, and the role block at lines 84–90 which appends `role-undeclared` to `failed`.
- `scripts/ready-for-helper.py` already computes the `READY` queue.
- `.env.example` keys: `VLLM_SOURCE SGLANG_SOURCE LLAMACPP_SOURCE VLLM_ORACLE DEPENDENCY_SOURCE GATE_HOST CUTLASS_DIR GPU_LOCK DEVICE_ARCH DEVICE_TOOLKIT_ROOT DEVICE_COMPILER`.
- `.agents/workflow.md` sections: `## Session protocol` (line 6), `## Tabular lifecycle` (line 142).

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/agent-onboard.py` (create) | Report unresolved session state (`--probe`); write `.env` values (`--env-set`). Never asks. |
| `tests/scripts/test_agent_onboard.py` (create) | Unit + mutation suite for the probe and the env writer |
| `scripts/agent-role.py` (modify) | Accept `read-only` and `--headless`; carry mode in the marker |
| `tests/scripts/test_agent_role.py` (modify) | Cover the new declarations |
| `scripts/agent-preflight.sh` (modify) | `--require-role` default-on, `--no-require-role`, actionable failure text, refuse `--staged` for `read-only` |
| `scripts/check-protocol-consistency.py` (modify) | Assert the interview table appears in `.agents/workflow.md` |
| `AGENTS.md`, `.agents/workflow.md`, `.agents/specs/operator-helper-protocol.md` (modify) | The prose, moved in the same change as the gate |

---

### Task 1: The probe

**Files:**
- Create: `scripts/agent-onboard.py`
- Test: `tests/scripts/test_agent_onboard.py`

**Interfaces:**
- Consumes: `scripts/agent-role.py` — `resolve() -> dict` (keys `role`, `row`, `session`, `branch`, `reason`), loaded via `importlib.util` because the filename is hyphenated.
- Produces: `ENV_KEYS: tuple[str, ...]`; `env_state() -> tuple[str, list[str]]` returning `(status, missing_keys)` where status is `"present" | "missing" | "incomplete" | "unreadable"`; `ready_rows() -> list[str]`; `probe() -> dict` with keys `role`, `row`, `mode`, `env`, `env_missing`, `queue`; `render_probe(state: dict) -> str`; `main(argv=None) -> int`.

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_agent_onboard.py`:

```python
#!/usr/bin/env python3
"""Unit and mutation checks for scripts/agent-onboard.py.

The probe exists to report what is unresolved. Its one job is to be honest
about absence: a missing .env and an unreadable .env must not look the same as
a complete one, and an undeclared role must never render as a declared one.
"""

from __future__ import annotations

import importlib.util
import sys
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


onboard = _load("agent_onboard", "scripts/agent-onboard.py")


class EnvStateTests(unittest.TestCase):
    def test_env_keys_match_the_tracked_example(self):
        # The probe must never invent a key. .env.example is the only source.
        example = (ROOT / ".env.example").read_text(encoding="utf-8")
        declared = {
            line.split("=", 1)[0]
            for line in example.splitlines()
            if line and not line.startswith("#") and "=" in line
        }
        self.assertEqual(set(onboard.ENV_KEYS), declared)

    def test_missing_file_reports_missing_not_incomplete(self):
        status, missing = onboard.env_state(ROOT / "does-not-exist-.env")
        self.assertEqual(status, "missing")
        self.assertEqual(sorted(missing), sorted(onboard.ENV_KEYS))

    def test_blank_value_counts_as_missing_that_key(self):
        # An empty value is a legitimate "unavailable", but the probe still
        # has to report it so the agent knows what it may ask for.
        text = "\n".join(f"{k}=" for k in onboard.ENV_KEYS)
        status, missing = onboard.env_state_from_text(text)
        self.assertEqual(status, "incomplete")
        self.assertEqual(sorted(missing), sorted(onboard.ENV_KEYS))

    def test_all_values_present_reports_present(self):
        text = "\n".join(f"{k}=/some/path" for k in onboard.ENV_KEYS)
        status, missing = onboard.env_state_from_text(text)
        self.assertEqual(status, "present")
        self.assertEqual(missing, [])


class ProbeRenderTests(unittest.TestCase):
    UNDECLARED = {
        "role": None, "row": None, "mode": "interactive",
        "env": "missing", "env_missing": ["VLLM_ORACLE"], "queue": ["ENG-FOO"],
    }

    def test_undeclared_role_renders_as_undeclared(self):
        out = onboard.render_probe(self.UNDECLARED)
        self.assertIn("UNDECLARED", out)
        self.assertNotIn("operator", out.split("queue")[0])

    def test_declared_role_renders_with_its_row(self):
        # The row id must NOT be one the fixture queue already contains, or the
        # queue line satisfies the assertion and deleting row rendering stays
        # green. Assert the `row=` prefix, not the bare id.
        out = onboard.render_probe(dict(self.UNDECLARED, role="helper", row="KERNEL-BAR"))
        self.assertIn("helper", out)
        self.assertIn("row=KERNEL-BAR", out)

    def test_undeclared_render_carries_the_interview_hint(self):
        # The hint is the whole point of the probe: without it an agent sees a
        # state line and no instruction. Deleting the block must go red.
        out = onboard.render_probe(self.UNDECLARED)
        self.assertIn("claim", out)
        self.assertIn("read-only", out)
        self.assertNotIn("claim", onboard.render_probe(
            dict(self.UNDECLARED, role="helper", row="KERNEL-BAR")))

    def test_probe_never_exits_nonzero(self):
        # The probe reports; it does not gate. Preflight gates.
        self.assertEqual(onboard.main(["--probe"]), 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tests/scripts/test_agent_onboard.py -v`
Expected: FAIL — `FileNotFoundError`/`AssertionError` from `_load`, because `scripts/agent-onboard.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/agent-onboard.py`:

```python
#!/usr/bin/env python3
"""Report what a session has not resolved yet, and write .env values. (A)

This script REPORTS. It never asks and it never decides, because no
harness-neutral mechanism exists for a shell script to run an interactive
prompt, and a hook injects text rather than conversing. The split is fixed:

    this script   -> detect and report
    the agent     -> ask, using the interview in .agents/workflow.md
    agent-role.py -> make the answer a fact

    scripts/agent-onboard.py --probe            # human-readable state
    scripts/agent-onboard.py --probe --json     # machine-readable

--env-set arrives in step 4.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
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
    """Classify the .env file. A missing FILE is distinct from missing VALUES."""
    if not path.exists():
        return "missing", list(ENV_KEYS)
    return env_state_from_text(path.read_text(encoding="utf-8"))


def ready_rows() -> list[str]:
    """The READY queue, from the existing checker rather than a second parser."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/ready-for-helper.py")],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return []
    rows = []
    for line in result.stdout.splitlines():
        token = line.strip().strip("`").split()[0] if line.strip() else ""
        if token.isupper() and "-" in token:
            rows.append(token)
    return rows


def probe() -> dict:
    state = role_mod.resolve()
    status, missing = env_state()
    return {
        "role": state.get("role"),
        "row": state.get("row"),
        # resolve() distinguishes "never declared" from "the operator lock is
        # held by another live session" and from "operator marker without a
        # held lock; re-claim". Dropping those makes this front door LESS
        # honest than the tool it wraps, and sends a session toward `claim
        # operator` when that will fail.
        "blocked_by_other_operator": bool(state.get("operator_held_by_other")),
        "reason": state.get("reason"),
        # Absent until step 2 teaches resolve() about mode. Rendered as a
        # default rather than a declaration, because headless is never
        # inferred and neither is interactive.
        "mode": state.get("mode"),
        "env": status,
        "env_missing": missing,
        "queue": ready_rows(),
    }


def render_probe(state: dict) -> str:
    role = state["role"] or "UNDECLARED"
    row = f" row={state['row']}" if state.get("row") else ""
    mode = state.get("mode") or "interactive (default, not declared)"
    lines = [
        f"role: {role}{row}   mode: {mode}",
        f".env: {state['env']}"
        + (f" (unset: {', '.join(state['env_missing'])})" if state["env_missing"] else ""),
        f"queue: {len(state['queue'])} READY rows"
        + (f" — {', '.join(state['queue'][:5])}" if state["queue"] else ""),
    ]
    if state["role"] is None:
        if state.get("blocked_by_other_operator"):
            lines.append(
                "NOTE: the operator lock is held by another live session, so "
                "`claim operator` will fail. Take helper or read-only."
            )
        lines.append(
            "This session has not declared a role. Ask what the work is, then claim: "
            "a long campaign -> operator; one scoped change -> helper --row <ROW-ID>; "
            "just looking -> read-only. See .agents/workflow.md."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report unresolved session state.")
    parser.add_argument("--probe", action="store_true", help="report session state")
    parser.add_argument("--json", action="store_true", help="machine-readable probe")
    args = parser.parse_args(argv)

    state = probe()
    print(json.dumps(state, indent=2, sort_keys=True) if args.json else render_probe(state))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Make it executable: `chmod +x scripts/agent-onboard.py`

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 tests/scripts/test_agent_onboard.py -v`
Expected: PASS, 11 tests.

- [ ] **Step 5: Smoke-test against the real repo**

Run: `python3 scripts/agent-onboard.py --probe`
Expected: three lines plus the interview hint, since this session has no role marker and no `.env`.

Run: `python3 scripts/agent-onboard.py --probe --json | python3 -c "import json,sys; d=json.load(sys.stdin); print(sorted(d))"`
Expected 9 keys: `['blocked_by_other_operator', 'env', 'env_missing', 'mode', 'queue', 'queue_error', 'reason', 'role', 'row']`

- [ ] **Step 6: Update the owed doc surfaces, run preflight, commit**

Add one short line to `docs/STATUS.md` and extend the existing "NOT APPLICABLE" paragraph in `docs/BENCHMARKS.md` (do not add a new paragraph — the page is at its 35-paragraph budget). Check `docs/STATUS.md` stays under the ratchet in `scripts/check-public-doc-tables.py`.

```bash
bash scripts/agent-preflight.sh > /tmp/pf.log 2>&1; echo "EXIT=$?"
git add scripts/agent-onboard.py tests/scripts/test_agent_onboard.py docs/STATUS.md docs/BENCHMARKS.md
git commit -F - <<'EOF'
tools(onboard): probe the unresolved session state, ask nothing (A step 1)

A script cannot run an interactive prompt in any harness-neutral way, so the
probe reports and the agent asks. .env.example is the only source of legal
keys, and a missing FILE is reported distinctly from missing VALUES.

FOLLOWING_AGENTS_PROTOCOL
Assisted-by: Claude Code:claude-opus-5 [ClaudeCode]
EOF
python3 scripts/check-doc-checkpoint.py --commit "$(git rev-parse HEAD)"; echo "doc-checkpoint EXIT=$?"
```

---

### Task 2: `read-only` and `--headless`

**Files:**
- Modify: `scripts/agent-role.py:41` (`ROLES`), `cmd_claim`, `resolve`, `main`
- Test: `tests/scripts/test_agent_role.py`

**Interfaces:**
- Produces: `CLAIMABLE_ROLES = ("operator", "helper")`; `DECLARABLE = ("operator", "helper", "read-only")`; `resolve()` gains a `"mode"` key valued `"interactive"` or `"headless"`; `agent-role.py claim read-only` and `claim <role> --headless`.

**The distinction that matters:** `read-only` is a *declared absence of claim*, not a third role. It takes no lock and creates no worktree. `CLAIMABLE_ROLES` stays exactly two, and any code that asks "may this session write?" tests membership in `CLAIMABLE_ROLES`, never `DECLARABLE`.

- [ ] **Step 1: Write the failing test**

Append to `tests/scripts/test_agent_role.py`, above its `if __name__` block:

```python
class ReadOnlyAndModeTests(unittest.TestCase):
    def test_claimable_roles_stay_exactly_two(self):
        # read-only must never become a third claimable role: it takes no lock
        # and no worktree, and every write path keys on CLAIMABLE_ROLES.
        self.assertEqual(role.CLAIMABLE_ROLES, ("operator", "helper"))
        self.assertIn("read-only", role.DECLARABLE)
        self.assertNotIn("read-only", role.CLAIMABLE_ROLES)

    def test_read_only_is_declarable(self):
        self.assertIn("read-only", role.DECLARABLE)

    def test_the_roles_alias_is_not_widened(self):
        # ROLES means "may write". Widening it to DECLARABLE would silently let
        # read-only through every existing write-gating call site, and today
        # that mutation leaves the whole suite green.
        self.assertEqual(role.ROLES, role.CLAIMABLE_ROLES)

    def test_mode_defaults_to_interactive(self):
        # Headless is DECLARED, never inferred. Absent an explicit flag the
        # session is interactive.
        self.assertEqual(role.mode_from_marker({}), "interactive")
        self.assertEqual(role.mode_from_marker({"mode": "headless"}), "headless")
        self.assertEqual(role.mode_from_marker({"mode": "nonsense"}), "interactive")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tests/scripts/test_agent_role.py -v`
Expected: FAIL with `AttributeError: module 'agent_role' has no attribute 'CLAIMABLE_ROLES'`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/agent-role.py`, replace the `ROLES` definition at line 41:

```python
# read-only is a declared ABSENCE of claim, not a third role: it takes no lock
# and creates no worktree. Every "may this session write?" test keys on
# CLAIMABLE_ROLES. Without it, a session that only reads must either take the
# repo-wide operator lock or create a throwaway worktree, and faced with that
# people reach for --no-require-role until the gate means nothing.
CLAIMABLE_ROLES = ("operator", "helper")
DECLARABLE = (*CLAIMABLE_ROLES, "read-only")
ROLES = CLAIMABLE_ROLES  # retained: existing call sites mean "may write"


def mode_from_marker(marker: dict) -> str:
    """Interactive unless headless was DECLARED. Never inferred."""
    return "headless" if marker.get("mode") == "headless" else "interactive"
```

In `resolve()`, change the marker acceptance test from `marker.get("role") in ROLES` to `marker.get("role") in DECLARABLE`, add `"mode": mode_from_marker(marker)` to the returned dict on the declared path, and return `"mode": "interactive"` on the undeclared path. A `read-only` marker needs no lock, so skip the operator lock check for it:

```python
    if marker and marker.get("session") == me and marker.get("role") in DECLARABLE:
        declared = marker["role"]
        if declared == "operator":
            if not lock or lock.get("session") != me:
                return {
                    "role": None,
                    "session": me,
                    "mode": "interactive",
                    "reason": "operator marker without a held lock; re-claim",
                    "branch": current_branch(),
                }
        return {
            "role": declared,
            "row": marker.get("row"),
            "session": me,
            "branch": current_branch(),
            "mode": mode_from_marker(marker),
            "reason": "declared",
        }
```

In `cmd_claim`, write the mode into the marker and skip the lock for `read-only` (the existing `if role == "operator":` block already does this by construction — no change needed there):

```python
    marker_path().write_text(
        json.dumps({
            "role": role,
            "row": args.row,
            "session": me,
            "mode": "headless" if args.headless else "interactive",
            "at": time.time(),
        }),
        encoding="utf-8",
    )
```

In `main()`, widen the claim parser's choices and add the flag:

```python
    claim = sub.add_parser("claim", help="declare and materialize a role")
    claim.add_argument("role", choices=DECLARABLE)
    claim.add_argument("--row", help="the row a helper is taking")
    claim.add_argument(
        "--headless",
        action="store_true",
        help="unattended run: decide and record rather than ask (never inferred)",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 tests/scripts/test_agent_role.py -v`
Expected: PASS, all existing tests plus 3.

- [ ] **Step 5: Verify the real CLI end to end**

```bash
python3 scripts/agent-role.py claim read-only
python3 scripts/agent-role.py show
python3 scripts/agent-role.py release
```
Expected: claims without taking a lock, `show` prints `role=read-only`, release is clean. Confirm no file appeared in the git common dir: `ls "$(git rev-parse --git-common-dir)"/agent-operator.lock 2>/dev/null` prints nothing.

- [ ] **Step 6: Doc surfaces, preflight, commit**

Same doc obligation as Task 1.

```bash
bash scripts/agent-preflight.sh > /tmp/pf.log 2>&1; echo "EXIT=$?"
git add scripts/agent-role.py tests/scripts/test_agent_role.py docs/STATUS.md docs/BENCHMARKS.md
git commit -F - <<'EOF'
tools(role): declare read-only and headless (A step 2)

read-only is a declared ABSENCE of claim, not a third role: no lock, no
worktree, and CLAIMABLE_ROLES stays exactly two so every write path keeps
keying on it. Headless is declared with the role and never inferred.

FOLLOWING_AGENTS_PROTOCOL
Assisted-by: Claude Code:claude-opus-5 [ClaudeCode]
EOF
python3 scripts/check-doc-checkpoint.py --commit "$(git rev-parse HEAD)"; echo "doc-checkpoint EXIT=$?"
```

---

### Task 3: `--require-role` becomes the default

**Files:**
- Modify: `scripts/agent-preflight.sh` — line 23 (`REQUIRE_ROLE=0`), the arg loop at 24–32, the role block at 84–90, the `SUITES` list
- Test: `tests/scripts/test_agent_onboard.py`

**Interfaces:**
- Consumes: `agent-role.py`'s `resolve()` and its `--json` output; `CLAIMABLE_ROLES` from Task 2.
- Produces: preflight fails on an undeclared role by default; `--no-require-role` opts out; `--staged` fails for a `read-only` session.

**The two behaviours that are easy to get backwards:** a `read-only` session **passes** a plain preflight (that is the whole point of the third answer) but **fails** `--staged`, because staging means writing. And the failure message must carry the interview, not just an error code — a gate that tells you what to do next is the difference between a protocol people follow and one they route around.

- [ ] **Step 1: Write the failing test**

Append to `tests/scripts/test_agent_onboard.py`, above its `if __name__` block:

```python
class PreflightWiringTests(unittest.TestCase):
    TEXT = (ROOT / "scripts/agent-preflight.sh").read_text(encoding="utf-8")

    def test_require_role_defaults_on(self):
        self.assertIn("REQUIRE_ROLE=1", self.TEXT)
        self.assertNotIn("REQUIRE_ROLE=0", self.TEXT)

    def test_opt_out_flag_exists(self):
        self.assertIn("--no-require-role", self.TEXT)

    def test_failure_text_carries_the_interview(self):
        # An error code alone gets routed around. The gate must say what to ask.
        self.assertIn("claim read-only", self.TEXT)
        self.assertIn("claim helper --row", self.TEXT)

    def test_staged_refuses_read_only(self):
        self.assertIn("read-only", self.TEXT)
        self.assertIn("STAGED", self.TEXT)

    def test_onboard_suite_is_registered(self):
        self.assertIn("test_agent_onboard", self.TEXT)

    def test_read_only_alone_does_not_satisfy_a_write_gate(self):
        # agent-role.py show exits 0 for read-only, so --require-role is
        # satisfied by a declared ABSENCE of claim. That is correct for a plain
        # run and wrong for --staged; the refusal must be explicit.
        self.assertIn("read-only-cannot-stage", self.TEXT)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tests/scripts/test_agent_onboard.py -v`
Expected: FAIL on `test_require_role_defaults_on` (`REQUIRE_ROLE=0` is present) and on the flag/interview assertions.

- [ ] **Step 3: Write minimal implementation**

In `scripts/agent-preflight.sh`, change line 23 and the arg loop:

```sh
REQUIRE_ROLE=1
```

```sh
    --require-role) REQUIRE_ROLE=1 ;;
    --no-require-role) REQUIRE_ROLE=0 ;;
```

Replace the role block (lines 84–90) with:

```sh
else
  printf '  \033[33m--\033[0m   %s\n' "$(printf '%s' "$role_line" | head -1)"
  printf '       This session has not declared a role. Ask what the work is:\n'
  printf '         a long or multi-step campaign  -> scripts/agent-role.py claim operator\n'
  printf '         one scoped change              -> scripts/agent-role.py claim helper --row <ROW-ID>\n'
  printf '         just reading or answering      -> scripts/agent-role.py claim read-only\n'
  printf '       Add --headless to an unattended run. See .agents/workflow.md.\n'
  if [ "$REQUIRE_ROLE" -eq 1 ]; then
    failed+=("role-undeclared")
  fi
fi

# read-only PASSES a plain preflight -- that is the point of the third answer.
# It fails --staged, because staging is writing.
if [ "$STAGED" -eq 1 ] && printf '%s' "$role_line" | grep -q 'role=read-only'; then
  printf '  \033[31mFAIL\033[0m read-only sessions do not write. Claim operator or helper first.\n'
  failed+=("read-only-cannot-stage")
fi
```

Add `test_agent_onboard` to the `SUITES` list, and update the usage comment block at the top of the file so `--help` documents `--no-require-role`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 tests/scripts/test_agent_onboard.py -v`
Expected: PASS, 12 tests.

- [ ] **Step 5: Verify both directions on the real gate**

```bash
python3 scripts/agent-role.py release 2>/dev/null
bash scripts/agent-preflight.sh > /tmp/pf-undeclared.log 2>&1; echo "undeclared EXIT=$?"   # expect 1
bash scripts/agent-preflight.sh --no-require-role > /tmp/pf-optout.log 2>&1; echo "opt-out EXIT=$?"  # expect 0
python3 scripts/agent-role.py claim read-only
bash scripts/agent-preflight.sh > /tmp/pf-readonly.log 2>&1; echo "read-only EXIT=$?"      # expect 0
bash scripts/agent-preflight.sh --staged > /tmp/pf-staged.log 2>&1; echo "read-only staged EXIT=$?"  # expect 1
```

Expected exactly: `1`, `0`, `0`, `1`. If the read-only plain run fails, the third answer is broken and the gate will be routed around — fix it before continuing.

- [ ] **Step 6: Claim a real role, then doc surfaces, preflight, commit**

You now need a claimable role to commit. `python3 scripts/agent-role.py claim helper --row <the row you are on>` — or `operator` if this session holds `main`.

```bash
bash scripts/agent-preflight.sh > /tmp/pf.log 2>&1; echo "EXIT=$?"
git add scripts/agent-preflight.sh tests/scripts/test_agent_onboard.py docs/STATUS.md docs/BENCHMARKS.md
git commit -F - <<'EOF'
gate(preflight): demand a role by default (A step 3)

The obligation already existed in prose and in an opt-in flag, and neither
fired. --require-role is now the default with --no-require-role to opt out,
and the failure carries the interview rather than an error code, because a
gate that does not say what to do next is a gate people route around.

read-only passes a plain preflight and fails --staged: reading is free,
writing is a claim.

FOLLOWING_AGENTS_PROTOCOL
Assisted-by: Claude Code:claude-opus-5 [ClaudeCode]
EOF
python3 scripts/check-doc-checkpoint.py --commit "$(git rev-parse HEAD)"; echo "doc-checkpoint EXIT=$?"
```

---

### Task 4: Just-in-time `.env`

**Files:**
- Modify: `scripts/agent-onboard.py` (`cmd_env_set` already exists from Task 1 — this task tests and hardens it)
- Test: `tests/scripts/test_agent_onboard.py`

**Interfaces:**
- Consumes: `ENV_KEYS`, `ENV_FILE`, `ENV_EXAMPLE` from Task 1.
- Produces: `cmd_env_set(pair: str) -> int`, and `--env-set KEY=VALUE` on `main()`.

**Why this is its own task:** `--env-set` writes to an untracked file that gates read, so an unrecognised key or a clobbered line fails silently and surfaces later as a mysteriously `PENDING` gate. Task 1 deliberately ships the probe WITHOUT it, so these tests have a real RED to go green from.

- [ ] **Step 1: Write the failing test**

Append to `tests/scripts/test_agent_onboard.py`, above its `if __name__` block:

```python
class EnvSetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.env = self.tmp / ".env"
        self._real = onboard.ENV_FILE
        onboard.ENV_FILE = self.env

    def tearDown(self):
        onboard.ENV_FILE = self._real
        shutil.rmtree(self.tmp)

    def test_unknown_key_is_refused(self):
        # Never invent a key: a typo'd name would sit in .env doing nothing
        # while the gate that wanted it stays mysteriously PENDING.
        self.assertEqual(onboard.cmd_env_set("NOT_A_REAL_KEY=/x"), 2)
        self.assertFalse(self.env.exists())

    def test_missing_pair_is_refused(self):
        self.assertEqual(onboard.cmd_env_set("VLLM_ORACLE"), 2)

    def test_first_write_seeds_from_the_example(self):
        self.assertEqual(onboard.cmd_env_set(f"{onboard.ENV_KEYS[0]}=/oracle"), 0)
        text = self.env.read_text(encoding="utf-8")
        self.assertIn(f"{onboard.ENV_KEYS[0]}=/oracle", text)
        # every other declared key survives, so nothing is silently dropped
        for key in onboard.ENV_KEYS:
            self.assertIn(key, text)

    def test_second_write_updates_in_place_without_duplicating(self):
        key = onboard.ENV_KEYS[0]
        onboard.cmd_env_set(f"{key}=/first")
        onboard.cmd_env_set(f"{key}=/second")
        text = self.env.read_text(encoding="utf-8")
        self.assertIn(f"{key}=/second", text)
        self.assertNotIn("/first", text)
        self.assertEqual(sum(1 for l in text.splitlines() if l.startswith(f"{key}=")), 1)

    def test_other_keys_are_not_disturbed(self):
        a, b = onboard.ENV_KEYS[0], onboard.ENV_KEYS[1]
        onboard.cmd_env_set(f"{a}=/aaa")
        onboard.cmd_env_set(f"{b}=/bbb")
        text = self.env.read_text(encoding="utf-8")
        self.assertIn(f"{a}=/aaa", text)
        self.assertIn(f"{b}=/bbb", text)
```

Add `import shutil` and `import tempfile` to the test file's import block.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tests/scripts/test_agent_onboard.py -v`
Expected: FAIL, 5 errors — `AttributeError: module 'agent_onboard' has no attribute 'cmd_env_set'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/agent-onboard.py`, above `main()`:

```python
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
    if not ENV_FILE.exists():
        ENV_FILE.write_text(ENV_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("#") and line.split("=", 1)[0].strip() == key:
            lines[index] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"set {key} in .env")
    return 0
```

In `main()`, add the flag and dispatch before the probe:

```python
    parser.add_argument("--env-set", metavar="KEY=VALUE", help="write one .env value")
```

```python
    if args.env_set:
        return cmd_env_set(args.env_set)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 tests/scripts/test_agent_onboard.py -v`
Expected: PASS, 17 tests.

- [ ] **Step 5: Doc surfaces, preflight, commit**

```bash
bash scripts/agent-preflight.sh > /tmp/pf.log 2>&1; echo "EXIT=$?"
git add scripts/agent-onboard.py tests/scripts/test_agent_onboard.py docs/STATUS.md docs/BENCHMARKS.md
git commit -F - <<'EOF'
tools(onboard): make --env-set safe to call repeatedly (A step 4)

It writes an untracked file that gates read, so an unrecognised key or a
clobbered line fails silently and surfaces later as a mysteriously PENDING
gate. Unknown keys are refused, the first write seeds from .env.example, and
repeat writes update in place without duplicating or disturbing other keys.

FOLLOWING_AGENTS_PROTOCOL
Assisted-by: Claude Code:claude-opus-5 [ClaudeCode]
EOF
python3 scripts/check-doc-checkpoint.py --commit "$(git rev-parse HEAD)"; echo "doc-checkpoint EXIT=$?"
```

---

### Task 5: The prose, moved with the gate

**Files:**
- Modify: `.agents/workflow.md` (after `## Session protocol`, line 6)
- Modify: `AGENTS.md` (the T0 role bullet)
- Modify: `.agents/specs/operator-helper-protocol.md`
- Modify: `scripts/check-protocol-consistency.py`
- Test: `tests/scripts/test_check_protocol_consistency.py`

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces: `INTERVIEW_MARKER = "<!-- role-interview:begin -->"` in `check-protocol-consistency.py`, asserting the interview block exists in `.agents/workflow.md`.

**Why the checker moves in the same commit:** `check-protocol-consistency.py` exists because an obligation was once migrated in `AGENTS.md` and the checker but not in the manual, which went on instructing agents to do the thing the migration had removed. Prose is what agents actually read. A gate whose prose lives nowhere is the same failure with the polarity flipped.

- [ ] **Step 1: Write the failing test**

Append to `tests/scripts/test_check_protocol_consistency.py`, above its `if __name__` block:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tests/scripts/test_check_protocol_consistency.py -v`
Expected: FAIL with `AttributeError: module … has no attribute 'INTERVIEW_MARKER'`.

- [ ] **Step 3: Write the prose and the checker**

In `.agents/workflow.md`, immediately after the `## Session protocol` heading, insert:

```markdown
<!-- role-interview:begin -->
### First question of every session

`scripts/agent-preflight.sh` fails until this session has declared a role. Ask
what the work is — not which role the developer wants, which is vocabulary they
should not have to learn first.

| What are you here to do? | Claim | What it means |
|---|---|---|
| A long or multi-step campaign — several changes, a benchmark grid, a whole row block | `scripts/agent-role.py claim operator` | Owns `main` and the GPU. Merges PRs first. Drives feature work through sub-agents rather than writing it. One at a time, repo-wide. |
| One scoped change — a fix, a port, a single row | `scripts/agent-role.py claim helper --row <ROW-ID>` | Isolated worktree on `row/<ROW-ID>`, draft PR opened at the START. That PR **is** the claim. Never touches `main`. |
| Just looking — reading code, answering a question | `scripts/agent-role.py claim read-only` | No lock, no worktree, no claim. Passes preflight; every write path refuses until you claim a real role. |

`read-only` is a declared **absence** of claim, not a third role. Escalating is
one command.

Add `--headless` when the developer has said the run is unattended: decide,
record each decision in `.agents/state.md`, never block, never merge, park what
will not go green. Headless is **declared, never inferred** — not from the hour,
not from silence, not from a long task.

`.env` is asked **just in time**: when a gate needs a value, ask for that value
and write it with `scripts/agent-onboard.py --env-set KEY=VALUE`. Never walk the
whole template up front, and never infer a value from a username, a path or a
machine identity. Unanswered means empty, and empty means the gates that need it
stay `PENDING`.

Run `scripts/agent-onboard.py --probe` to see what is still unresolved.
<!-- role-interview:end -->
```

In `scripts/check-protocol-consistency.py`, add:

```python
INTERVIEW_MARKER = "<!-- role-interview:begin -->"
INTERVIEW_REQUIRED = ("claim operator", "claim helper --row", "claim read-only", "--headless")


def interview_errors(text: str) -> list[str]:
    """The role interview must live where agents read it, not only in a gate."""
    if INTERVIEW_MARKER not in text:
        return [".agents/workflow.md is missing the role-interview block"]
    return [
        f".agents/workflow.md role interview omits {needle!r}"
        for needle in INTERVIEW_REQUIRED
        if needle not in text
    ]
```

Call `interview_errors` from `main()` against `.agents/workflow.md` and add its output to the error list.

**Two residual false claims must die in this task** (found by Task 3's re-review, both outside the lines Task 3 corrected, both passing every existing gate):

- `.agents/NOW.md` still says "role discipline ENFORCING, `--require-role` still opt-in" — directly contradicted by Task 3's deliverable.
- `.agents/specs/operator-helper-protocol.md:203`'s W0 table row still says "session-scoped role marker" — the exact claim corrected 130 lines earlier in the same file.

**And add the behavioural pin the text anchors cannot give.** Every assertion in `PreflightWiringTests` greps text, so an *override* still slips through even though a *rewrite* of the default is caught: keeping `REQUIRE_ROLE=1` and adding `REQUIRE_ROLE=0 ` (one trailing space) on the next line leaves the suite green while the gate stops failing. Add a test that RUNS the script:

```python
    def test_preflight_actually_fails_on_an_undeclared_role(self):
        # Every other assertion here greps text, so an override on a later line
        # slips through. Only executing the script closes the class.
        env = dict(os.environ, VLLM_CPP_AGENT_SESSION="probe-no-such-session")
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/agent-preflight.sh"), "--quiet"],
            cwd=ROOT, capture_output=True, text=True, check=False, env=env,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("role-undeclared", result.stdout + result.stderr)
```

This test must run in a state with no resolvable role. If the worktree carries a marker, the test is meaningless — assert the precondition or skip loudly, never silently pass.

Update `AGENTS.md`'s T0 role bullet to say the role is asked as the first question of the session, that `read-only` is available, and that preflight demands it by default. Update `.agents/specs/operator-helper-protocol.md` § "Determining the role" to record `read-only` and the mode declaration.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 tests/scripts/test_check_protocol_consistency.py -v`
Expected: PASS.

- [ ] **Step 5: Verify the whole gate suite**

```bash
python3 scripts/check-protocol-consistency.py; echo "consistency EXIT=$?"
bash scripts/agent-preflight.sh > /tmp/pf.log 2>&1; echo "preflight EXIT=$?"
```
Expected: both `EXIT=0`.

- [ ] **Step 6: Doc surfaces, preflight, commit**

```bash
git add AGENTS.md .agents/workflow.md .agents/specs/operator-helper-protocol.md \
        scripts/check-protocol-consistency.py tests/scripts/test_check_protocol_consistency.py \
        docs/STATUS.md docs/BENCHMARKS.md
git commit -F - <<'EOF'
docs(protocol): the role interview ships with the gate that demands it (A step 5)

check-protocol-consistency.py exists because an obligation was once migrated in
AGENTS.md and the checker but not in the manual, which went on instructing
agents to do the thing the migration had removed. A gate whose prose lives
nowhere is that failure with the polarity flipped, so the interview lands in
workflow.md and the checker asserts it is there.

FOLLOWING_AGENTS_PROTOCOL
Assisted-by: Claude Code:claude-opus-5 [ClaudeCode]
EOF
python3 scripts/check-doc-checkpoint.py --commit "$(git rev-parse HEAD)"; echo "doc-checkpoint EXIT=$?"
```

---

## Done when

- `scripts/agent-onboard.py --probe` reports role, mode, `.env` state and the `READY` queue, and writes nothing.
- `agent-preflight.sh` fails on an undeclared role by default; `--no-require-role` opts out; a `read-only` session passes plainly and fails `--staged`.
- `agent-role.py claim read-only` takes no lock and creates no worktree; `--headless` is recorded in the marker.
- `--env-set` refuses unknown keys, seeds from `.env.example`, and updates in place.
- The interview lives in `.agents/workflow.md` and `check-protocol-consistency.py` fails without it.
- Every commit passes `check-doc-checkpoint.py --commit <sha>` in its committed form.

## Out of scope

Subsystem B — the orchestration harness: how an operator runs a row through implementer subagents with an independent reviewer, the gate-command discipline that makes a row's `Gates` field a real Verify, and the headless execution loop. It gets its own spec and plan. Also out of scope: generating `.agents/developer-preferences.md`, and any Claude-Code-specific hook.
