# P0 Live-State Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconcile the 188 live-state matrix rows against Git reality — above all the 54 rows simultaneously claiming `ACTIVE`, which cannot all be true — so the record is truthful before any issue backfill mints ~160 public issues from it.

**Architecture:** A new reporting tool, `scripts/audit-live-rows.py`, reuses the row parser already inside `scripts/check-agent-record.py` (never reimplements it) and cross-references each live row against local/remote `row/<ID>` branches and `main` commits mentioning the ID. Classification is a **pure function** over already-gathered evidence, so it is unit-testable without Git. The tool **proposes and reports; it never rewrites a matrix.** A human/agent applies corrections per matrix in reviewable commits. Only after the record is corrected does the tool become a CI gate, so the rot cannot return.

**Tech Stack:** Python 3 standard library only (no new dependencies), `argparse`, `importlib.util` for loading the hyphenated checker module, `subprocess` for Git, `unittest` for the mutation suite. Matches the existing `scripts/check-*.py` house style.

**Deliberate divergence from the spec.** The spec's P0 says "reconcile against branches, PRs and commits". This plan uses **Git evidence only — no `gh`**. An open PR always has a `row/<ID>` head branch with unmerged commits, so the branch check already covers it, while staying fully offline-capable and keeping P0 free of the issue machinery that P1 introduces. If a row is ever worked without a `row/<ID>` branch, the audit reports it `ABANDONED` and the human review in Task 5 Step 3 catches it.

## Global Constraints

Copied from `AGENTS.md`, `.agents/coordination.md` and `.agents/specs/issue-native-tracking.md`. Every task's requirements implicitly include this section.

- **Every commit carries the trailer `FOLLOWING_AGENTS_PROTOCOL`** plus `Assisted-by: <AGENT>:<MODEL> [TOOL]`. **Never** `Signed-off-by` or `Co-Authored-By` from an AI. CI rejects commits lacking the protocol trailer.
- **Run `scripts/agent-preflight.sh` before every commit.** It must exit 0. Never pipe it (`cmd | tail` masks the exit status); redirect to a file and check `$?`.
- **Never weaken a checker to make a transition pass. Repair the record.**
- **Never three-way merge a keyed record.** The matrices, `docs/STATUS.md`, `docs/BENCHMARKS.md`, `docs/FEATURES.md` and `.agents/NOW.md` are merged by taking `main`'s version wholesale, re-applying your edit, and verifying the other side is byte-identical.
- **Evidence is moved, never deleted.**
- **Python: standard library only.** No new dependencies. Use `from __future__ import annotations`, dataclasses and type hints, matching `scripts/check-agent-record.py`.
- **DRY across P0/P1:** row parsing lives in `scripts/check-agent-record.py` and is imported, never copied. `scripts/sync-rows.py` (P1) will import the same helpers.
- **Heuristics report; they never gate.** The `PARTIAL` missing-modes detector is a flag for human review and must never become a hard failure.
- **State-transition legality** (enforced by `check-agent-record.py`, so violating it breaks the build):
  - `READY`, `ACTIVE`, `GATING`, `DONE`, `BLOCKED` require a real `.agents/specs/<slug>.md` link.
  - `PARTIAL`, `ANCHOR-BACKFILL`, `GATING`, `DONE` require resolving code/test evidence anchors.
  - Therefore an abandoned `ACTIVE` row moves to **`READY` if it has a real spec, otherwise `INVENTORIED`** — it may not simply be blanked.
- The live set is exactly: `SPIKE`, `READY`, `ACTIVE`, `GATING`, `PARTIAL`, `BLOCKED`.

**Baseline census on `origin/main` @ `027af9b0`**, measured with `parse_claim_rows` itself (an earlier ad-hoc regex estimate of 160 was wrong and is superseded):

| Source | Rows | Live rows | ACTIVE | PARTIAL | SPIKE | GATING | BLOCKED | READY |
|---|---|---|---|---|---|---|---|---|
| The 5 files in `MATRIX_PATHS` | 695 | 177 | 51 | 60 | 43 | 10 | 7 | 6 |
| `feature-matrix.md` + `sglang-matrix.md` | 19 | 11 | 3 | 8 | 0 | 0 | 0 | 0 |
| **Total across all 7 matrices** | **714** | **188** | **54** | **68** | **43** | **10** | **7** | **6** |

**`check-agent-record.py`'s `MATRIX_PATHS` covers only 5 of the 7 matrices.** The audit must cover all 7: `feature-matrix.md` and `sglang-matrix.md` hold 11 live rows that would otherwise become unaudited public issues in P2. The audit therefore defines its own `AUDIT_MATRIX_PATHS`. It does **not** widen `MATRIX_PATHS` itself — that would change what the repo-wide CI gate validates and could turn CI red on rows never held to the row contract. The DRY constraint is about the *parser*, which is still imported and never reimplemented.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/audit-live-rows.py` (create) | Load live rows, gather Git evidence, classify, render report/JSON, and (from Task 7) gate |
| `tests/scripts/test_audit_live_rows.py` (create) | Unit + mutation suite for the classifier and the shipped-matrix integration |
| `.agents/specs/live-state-audit-2026-08-06.md` (create) | The audit findings artifact — the evidence justifying every correction |
| `.agents/*-matrix.md` (modify, Task 6) | The corrections themselves, one commit per matrix |
| `.agents/state-index/<writable-shard>.csv`, `.agents/state-events/<YYYY-MM>/<event-id>.md` (create/modify, Task 6) | Ordered index row plus immutable checkpoint evidence |
| `scripts/agent-preflight.sh:50-63` (modify, Task 7) | Register the new mutation suite and the gate |
| `.github/workflows/ci.yml:42-46` (modify, Task 7) | Run the gate and its mutation suite in CI |

---

### Task 1: Row loading and Git evidence collection

**Files:**
- Create: `scripts/audit-live-rows.py`
- Test: `tests/scripts/test_audit_live_rows.py`

**Interfaces:**
- Consumes: `scripts/check-agent-record.py` — `ClaimRow` (frozen dataclass with fields `path: Path`, `line_no: int`, `item_id: str`, `state: str`, `header: tuple[str, ...]`, `cells: tuple[str, ...]`, `raw: str`, and method `field(name: str) -> str`); `parse_claim_rows(path: Path, errors: list[str]) -> list[ClaimRow]`; `MATRIX_PATHS: list[Path]`.
- Produces: `LIVE_STATES: frozenset[str]`; `AUDIT_MATRIX_PATHS: list[Path]` (= `record.MATRIX_PATHS` plus `.agents/feature-matrix.md` and `.agents/sglang-matrix.md`); `live_rows(errors: list[str] | None = None) -> list[record.ClaimRow]`; `row_branches() -> dict[str, list[str]]`; `main_commits(item_id: str) -> list[str]` (**anchored** on ID boundaries); `unmerged(branch: str) -> list[str]`; `require_origin_main() -> None`.

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_audit_live_rows.py`:

```python
#!/usr/bin/env python3
"""Unit and mutation checks for scripts/audit-live-rows.py.

The audit only helps if it is honest in both directions: it must not call a
live row abandoned when work is really in flight, and it must not call an
abandoned row live because a branch name happens to exist.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
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


audit = _load("audit_live_rows", "scripts/audit-live-rows.py")


class LiveRowLoadingTests(unittest.TestCase):
    def test_live_states_are_exactly_the_six(self):
        self.assertEqual(
            audit.LIVE_STATES,
            frozenset({"SPIKE", "READY", "ACTIVE", "GATING", "PARTIAL", "BLOCKED"}),
        )

    def test_all_seven_matrices_are_audited(self):
        names = {path.name for path in audit.AUDIT_MATRIX_PATHS}
        self.assertIn("feature-matrix.md", names)
        self.assertIn("sglang-matrix.md", names)
        self.assertEqual(len(names), 7)

    def test_newly_covered_feature_matrix_actually_yields_live_rows(self):
        # Asserting the path is in a list proves nothing: every other assertion
        # here still passes if feature-matrix.md contributes zero rows. This is
        # what the seventh-matrix commit actually bought.
        rows = audit.live_rows()
        self.assertTrue([r for r in rows if r.path.name == "feature-matrix.md"])

    def test_shipped_record_parses_without_errors(self):
        # parse_claim_rows DROPS a row it cannot parse. If a malformed row ever
        # lands, the census silently shrinks -- so the sink must be surfaced,
        # and it must be empty today.
        errors: list[str] = []
        audit.live_rows(errors)
        self.assertEqual(errors, [])

    def test_shipped_matrices_yield_only_live_rows(self):
        rows = audit.live_rows()
        self.assertTrue(rows, "the shipped matrices must contain live rows")
        for row in rows:
            self.assertIn(row.state, audit.LIVE_STATES)

    def test_every_live_state_is_represented_in_the_shipped_matrices(self):
        # Guards the loader against silently dropping a whole state: if a
        # header rename made one state unparseable, its count would go to
        # zero while the other five still looked healthy.
        rows = audit.live_rows()
        present = {row.state for row in rows}
        self.assertEqual(present, set(audit.LIVE_STATES))
        self.assertGreater(len(rows), 100, "the live set is ~188 rows")


class IdGrepPatternTests(unittest.TestCase):
    """The ID match must be a whole-token match, never a prefix match."""

    def test_pattern_matches_the_id_as_a_whole_token(self):
        pattern = re.compile(audit.id_grep_pattern("MODEL-MM"))
        for message in (
            "MODEL-MM",
            "feat(mm): MODEL-MM decoder lands",
            "closes MODEL-MM.",
            "(MODEL-MM) golden captured",
        ):
            self.assertTrue(pattern.search(message), message)

    def test_pattern_rejects_a_longer_id_that_merely_starts_with_it(self):
        # 55 pairs of live row IDs are prefixes of longer ones. A substring
        # match would credit MODEL-MM with MODEL-MM-voxtral's commits, and the
        # classifier calls any commit LANDED -- so an abandoned row would
        # report as finished, the exact false negative this tool prevents.
        pattern = re.compile(audit.id_grep_pattern("MODEL-MM"))
        for message in (
            "feat(mm): MODEL-MM-voxtral audio tower lands",
            "MODEL-MM-QWEN3VL golden captured",
            "record(mm): MODEL-MM_SUFFIX bookkeeping",
        ):
            self.assertIsNone(pattern.search(message), message)


class CommandLineGuardTests(unittest.TestCase):
    def test_check_flag_does_not_silently_exit_zero(self):
        # The file is executable and its docstring advertises --check, but the
        # real CLI arrives in P0 step 4. Until then --check must NOT exit 0:
        # a gate that reports success because it ignored its own flag is the
        # worst possible answer.
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/audit-live-rows.py"), "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
```

**`CommandLineGuardTests` is transitional.** Task 4 lands the real CLI and replaces it with the `exit_code` tests — delete it there, do not leave both.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tests/scripts/test_audit_live_rows.py -v`
Expected: FAIL — `FileNotFoundError` / `AssertionError` from `_load`, because `scripts/audit-live-rows.py` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/audit-live-rows.py`:

```python
#!/usr/bin/env python3
"""Audit the live-state matrix rows against Git reality. (P0)

54 rows claim ACTIVE at once, which cannot be true: a stale ACTIVE cell inside
a several-hundred-row table is invisible rot. This tool makes it visible.

It PROPOSES and REPORTS. It never rewrites a matrix -- corrections are applied
by a human/agent in reviewable per-matrix commits, because a state transition
carries contract obligations (a spec link, evidence anchors) that only a reader
of the row can satisfy.

Row parsing is imported from scripts/check-agent-record.py rather than
reimplemented, so the audit and the gate can never disagree about what a row is.

    scripts/audit-live-rows.py                 # markdown report
    scripts/audit-live-rows.py --json          # machine-readable
    scripts/audit-live-rows.py --check         # exit 1 if an ACTIVE row is abandoned
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
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


record = _load("agent_record", "scripts/check-agent-record.py")

LIVE_STATES = frozenset({"SPIKE", "READY", "ACTIVE", "GATING", "PARTIAL", "BLOCKED"})

# check-agent-record.py's MATRIX_PATHS omits feature-matrix.md and
# sglang-matrix.md, which together hold 11 live rows. The audit covers all
# seven matrices so no live row escapes it, but deliberately does NOT widen
# MATRIX_PATHS itself: that governs a repo-wide CI gate whose row contract
# these two files have never been held to.
AUDIT_MATRIX_PATHS = [
    *record.MATRIX_PATHS,
    record.AGENTS / "feature-matrix.md",
    record.AGENTS / "sglang-matrix.md",
]


def live_rows(errors: list[str] | None = None) -> list[record.ClaimRow]:
    """Every row in the audited matrices whose state is in LIVE_STATES.

    Parse errors are appended to `errors` when a list is supplied. They must
    not be swallowed: parse_claim_rows DROPS a row it cannot parse, so a
    malformed row would vanish from a census whose whole point is
    completeness -- and it bites hardest on feature-matrix.md and
    sglang-matrix.md, which no CI gate parses today.
    """
    sink = errors if errors is not None else []
    rows = []
    for path in AUDIT_MATRIX_PATHS:
        for row in record.parse_claim_rows(path, sink):
            if row.state in LIVE_STATES:
                rows.append(row)
    return rows


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout if result.returncode == 0 else ""


def row_branches() -> dict[str, list[str]]:
    """Map row ID -> every local or remote branch named row/<ID>."""
    mapping: dict[str, list[str]] = {}
    out = git("for-each-ref", "--format=%(refname:short)", "refs/heads", "refs/remotes")
    for line in out.splitlines():
        name = line.strip()
        if name.startswith("row/"):
            item_id = name[len("row/") :]
        elif "/row/" in name:
            item_id = name.split("/row/", 1)[1]
        else:
            continue
        mapping.setdefault(item_id, []).append(name)
    return mapping


def id_grep_pattern(item_id: str) -> str:
    """POSIX-ERE pattern matching this row ID as a whole token, never a prefix."""
    return r"(^|[^A-Za-z0-9_-])" + re.escape(item_id) + r"([^A-Za-z0-9_-]|$)"


def main_commits(item_id: str) -> list[str]:
    """Commits on origin/main whose message mentions this row ID as a whole token.

    The match is ANCHORED on ID boundaries, never a substring: 55 pairs of live
    row IDs are prefixes of longer ones (MODEL-MM of seven MODEL-MM-* rows,
    LOAD-SAFETENSORS of LOAD-SAFETENSORS-DIRECT-DENSE, ...). A substring match
    would credit the short row with the long row's commits, and the classifier
    calls any commit LANDED -- so an abandoned row would silently report as
    finished, the exact false negative this tool exists to prevent.
    """
    out = git(
        "log", "--oneline", "-E", f"--grep={id_grep_pattern(item_id)}", "-n", "20",
        "origin/main",
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def require_origin_main() -> None:
    """Abort unless origin/main resolves.

    git() maps any failure to "", which downstream is indistinguishable from
    "no evidence". An unfetched or missing origin/main would therefore make
    EVERY row look abandoned and the audit would propose downgrading all 54
    ACTIVE rows at once. Absence of work and absence of information must never
    look the same.
    """
    if not git("rev-parse", "--verify", "--quiet", "origin/main").strip():
        raise SystemExit(
            "origin/main does not resolve -- run `git fetch origin main` first. "
            "Without it every row reports no Git evidence and this audit would "
            "propose downgrading every ACTIVE row."
        )


def unmerged(branch: str) -> list[str]:
    """Commits on branch that are not yet on origin/main."""
    out = git("log", "--oneline", f"origin/main..{branch}")
    return [line.strip() for line in out.splitlines() if line.strip()]
```

Make it executable: `chmod +x scripts/audit-live-rows.py`

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 tests/scripts/test_audit_live_rows.py -v`
Expected: PASS, 9 tests.

- [ ] **Step 5: Verify the loader sees the real census**

Run: `python3 -c "import importlib.util,sys; s=importlib.util.spec_from_file_location('a','scripts/audit-live-rows.py'); m=importlib.util.module_from_spec(s); sys.modules['a']=m; s.loader.exec_module(m); rows=m.live_rows(); import collections; print(len(rows), collections.Counter(r.state for r in rows))"`
Expected: `188` total, with `ACTIVE` = 54, `PARTIAL` = 68, `SPIKE` = 43, `GATING` = 10, `BLOCKED` = 7, `READY` = 6.

If the numbers differ, do **not** adjust the test to match — `main` has moved. Re-read the current census, record the new baseline in the report, and continue.

- [ ] **Step 6: Run preflight and commit**

```bash
bash scripts/agent-preflight.sh > /tmp/preflight.log 2>&1; echo "EXIT=$?"
git add scripts/audit-live-rows.py tests/scripts/test_audit_live_rows.py
git commit -F - <<'EOF'
tools(audit): load live matrix rows and gather Git evidence (P0 step 1)

Reuses the row parser in check-agent-record.py rather than reimplementing it,
so the audit and the gate can never disagree about what a row is.

FOLLOWING_AGENTS_PROTOCOL
Assisted-by: Claude Code:claude-opus-5 [ClaudeCode]
EOF
```

---

### Task 2: The classifier

**Files:**
- Modify: `scripts/audit-live-rows.py`
- Test: `tests/scripts/test_audit_live_rows.py`

**Interfaces:**
- Consumes: `row_branches()`, `main_commits()`, `unmerged()` from Task 1.
- Produces: `classify_active(branches: list[str], unmerged_by_branch: dict[str, list[str]], commits: list[str]) -> tuple[str, str]` returning `(verdict, reason)` where verdict is one of `"IN-FLIGHT"`, `"LANDED"`, `"ABANDONED"`. Also `VERDICTS: frozenset[str]`.

The classifier is pure — it takes evidence, not a repository — so it is testable without Git and cannot be flaky.

- [ ] **Step 1: Write the failing test**

Append to `tests/scripts/test_audit_live_rows.py`, above the `if __name__` block:

```python
class ClassifierTests(unittest.TestCase):
    def test_unmerged_branch_commits_mean_in_flight(self):
        verdict, reason = audit.classify_active(
            branches=["row/ENG-FOO"],
            unmerged_by_branch={"row/ENG-FOO": ["abc1234 wip"]},
            commits=[],
        )
        self.assertEqual(verdict, "IN-FLIGHT")
        self.assertIn("row/ENG-FOO", reason)

    def test_fully_merged_branch_means_landed(self):
        verdict, reason = audit.classify_active(
            branches=["row/ENG-FOO"],
            unmerged_by_branch={"row/ENG-FOO": []},
            commits=[],
        )
        self.assertEqual(verdict, "LANDED")
        # "unmerged" contains "merged", so a substring check on the bare word
        # would pass for the IN-FLIGHT reason too.
        self.assertIn("fully merged", reason)

    def test_main_commits_without_branch_mean_landed(self):
        verdict, reason = audit.classify_active(
            branches=[],
            unmerged_by_branch={},
            commits=["def5678 feat(eng): ENG-FOO"],
        )
        self.assertEqual(verdict, "LANDED")
        self.assertIn("def5678", reason)

    def test_no_evidence_at_all_means_abandoned(self):
        verdict, reason = audit.classify_active(
            branches=[], unmerged_by_branch={}, commits=[]
        )
        self.assertEqual(verdict, "ABANDONED")
        self.assertIn("no branch", reason.lower())

    def test_in_flight_wins_over_landed_when_both_present(self):
        # A row can have landed groundwork AND active follow-up work.
        # Claiming it is finished would silently steal an open claim.
        verdict, _ = audit.classify_active(
            branches=["row/ENG-FOO"],
            unmerged_by_branch={"row/ENG-FOO": ["abc1234 wip"]},
            commits=["def5678 feat(eng): ENG-FOO groundwork"],
        )
        self.assertEqual(verdict, "IN-FLIGHT")

    def test_reason_names_only_the_branches_with_unmerged_commits(self):
        # With more than one branch, the reason must name the live one and not
        # the merged one, and must be order-independent so a re-run does not
        # produce a spuriously different report.
        verdict, reason = audit.classify_active(
            branches=["row/B-LIVE", "row/A-MERGED"],
            unmerged_by_branch={"row/B-LIVE": ["abc1234 wip"], "row/A-MERGED": []},
            commits=[],
        )
        self.assertEqual(verdict, "IN-FLIGHT")
        self.assertIn("row/B-LIVE", reason)
        self.assertNotIn("row/A-MERGED", reason)

    def test_reason_is_order_independent(self):
        # The mixed case above has exactly ONE live branch, so sorted() is a
        # no-op there and deleting it survives. Two branches on the same side
        # of the filter are what pin determinism, on both reason paths: a
        # report that reshuffles its own evidence between runs cannot be
        # diffed by the human who has to act on it.
        for label, by_branch in [
            ("in-flight", {"row/B": ["abc1234 wip"], "row/A": ["def5678 wip"]}),
            ("landed", {"row/B": [], "row/A": []}),
        ]:
            with self.subTest(label):
                forward = audit.classify_active(["row/A", "row/B"], by_branch, [])
                reverse = audit.classify_active(["row/B", "row/A"], by_branch, [])
                self.assertEqual(forward, reverse)
                self.assertIn("row/A, row/B", forward[1])

    def test_missing_branch_key_is_a_loud_caller_bug(self):
        # Silently treating an ungathered branch as merged would report a live
        # claim as finished -- the exact false negative this tool prevents.
        with self.assertRaises(KeyError):
            audit.classify_active(
                branches=["row/NEVER-GATHERED"], unmerged_by_branch={}, commits=[]
            )

    def test_every_verdict_is_declared(self):
        for branches, by_branch, commits in [
            (["row/X"], {"row/X": ["a b"]}, []),
            (["row/X"], {"row/X": []}, []),
            ([], {}, ["a b"]),
            ([], {}, []),
        ]:
            verdict, _ = audit.classify_active(branches, by_branch, commits)
            self.assertIn(verdict, audit.VERDICTS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tests/scripts/test_audit_live_rows.py -v`
Expected: FAIL with `AttributeError: module 'audit_live_rows' has no attribute 'classify_active'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/audit-live-rows.py`:

```python
VERDICTS = frozenset({"IN-FLIGHT", "LANDED", "ABANDONED"})


def classify_active(
    branches: list[str],
    unmerged_by_branch: dict[str, list[str]],
    commits: list[str],
) -> tuple[str, str]:
    """Classify one ACTIVE row from already-gathered evidence.

    IN-FLIGHT wins over LANDED whenever both are present: a row can have landed
    groundwork and still have open follow-up work, and calling that finished
    would silently steal a live claim.
    """
    # Indexed, never .get(): `branches` is the authority for which keys must
    # exist, so a missing key is a CALLER bug, not data. .get() would return
    # None -> falsy -> the row reports LANDED, a live claim reported as
    # finished. Absence of work and absence of information must never look the
    # same; a KeyError at the audit's own boundary is the loud alternative.
    live_branches = [b for b in branches if unmerged_by_branch[b]]
    if live_branches:
        joined = ", ".join(sorted(live_branches))
        return "IN-FLIGHT", f"unmerged commits on {joined}"
    if branches:
        joined = ", ".join(sorted(branches))
        return "LANDED", f"branch {joined} exists and is fully merged into main"
    if commits:
        return "LANDED", f"on main: {commits[0]}"
    return "ABANDONED", "no branch, no commit on main mentioning the row ID"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 tests/scripts/test_audit_live_rows.py -v`
Expected: PASS, 18 tests.

- [ ] **Step 5: Run preflight and commit**

```bash
bash scripts/agent-preflight.sh > /tmp/preflight.log 2>&1; echo "EXIT=$?"
git add scripts/audit-live-rows.py tests/scripts/test_audit_live_rows.py
git commit -F - <<'EOF'
tools(audit): pure evidence classifier for ACTIVE rows (P0 step 2)

IN-FLIGHT deliberately outranks LANDED: a row can have landed groundwork and
still have open follow-up, and calling that finished would steal a live claim.

FOLLOWING_AGENTS_PROTOCOL
Assisted-by: Claude Code:claude-opus-5 [ClaudeCode]
EOF
```

---

### Task 3: PARTIAL missing-modes flag

**Files:**
- Modify: `scripts/audit-live-rows.py`
- Test: `tests/scripts/test_audit_live_rows.py`

**Interfaces:**
- Produces: `GAP_MARKERS: tuple[str, ...]`; `gap_pattern(markers: tuple[str, ...]) -> re.Pattern[str]`; `GAP_RE`; `CHECK_FAILS_ON: frozenset[str]`; `names_missing_modes(row_text: str) -> bool`; `matched_marker(row_text: str) -> str`.

The row contract already requires a `PARTIAL` row to make its missing modes explicit. Because `PARTIAL` rows become **public** issues in P2, a vague one becomes a vague public issue. This is a **report-only flag for human review** and must never become a hard failure — the detector is a keyword heuristic and gating on it would be exactly the fragile-checker trap the protocol warns about.

- [ ] **Step 1: Write the failing test**

Append to `tests/scripts/test_audit_live_rows.py`, above the `if __name__` block:

```python
class PartialGapTests(unittest.TestCase):
    def test_explicit_gap_language_is_recognised(self):
        for text in [
            "Works for bf16; fp8 is missing",
            "Prefill only, decode not yet ported",
            "Dense path supported, MoE unsupported",
            "Image works; audio pending",
        ]:
            self.assertTrue(audit.names_missing_modes(text), text)

    def test_row_without_gap_language_is_flagged(self):
        self.assertFalse(audit.names_missing_modes("Ported and gated on GB10"))

    def test_detection_is_case_insensitive(self):
        self.assertTrue(audit.names_missing_modes("FP8 IS MISSING"))

    def test_markers_match_whole_words_not_substrings(self):
        # "commonly" contains "only" and "node" contains "no". A substring
        # match would mark these rows explicit and hide them from review.
        self.assertFalse(audit.names_missing_modes("Commonly used decode node"))
        # Both halves need their marker pinned as live, or the assertFalse
        # above passes for the wrong reason: dropping "no" from GAP_MARKERS
        # entirely also stops "node" matching, and nothing would notice.
        self.assertTrue(audit.names_missing_modes("Decode only"))
        self.assertTrue(audit.names_missing_modes("No fp8 path"))

    def test_flag_is_advisory_and_never_gates(self):
        # check mode fails only on abandoned ACTIVE rows, never on a vague
        # PARTIAL row -- the detector is a keyword heuristic.
        self.assertNotIn("PARTIAL", audit.CHECK_FAILS_ON)

    def test_check_fails_on_active_and_nothing_else(self):
        # assertNotIn above passes for frozenset() -- a check mode that fails
        # on NOTHING -- and even for the bare string "ACTIVE", since "PARTIAL"
        # is not a substring of it. Neither is what "report-only" means: the
        # flag must be excluded from a set that still gates something. Pin the
        # membership exactly, and pin that the gated state is a real live one.
        self.assertEqual(audit.CHECK_FAILS_ON, frozenset({"ACTIVE"}))
        self.assertTrue(audit.CHECK_FAILS_ON <= audit.LIVE_STATES)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tests/scripts/test_audit_live_rows.py -v`
Expected: FAIL with `AttributeError: module 'audit_live_rows' has no attribute 'names_missing_modes'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/audit-live-rows.py`:

```python
GAP_MARKERS = (
    "missing",
    "not yet",
    "unsupported",
    "pending",
    "only",
    "absent",
    "gap",
    "no",
    "without",
    "blocked",
    "todo",
)

# Whole words, never substrings: "only" must not match "commonly" and "no"
# must not match "node". A substring match would silently mark a vague row as
# explicit, which is the exact failure this flag exists to catch.
def gap_pattern(markers: tuple[str, ...]) -> re.Pattern[str]:
    """Compile markers into a whole-word, case-insensitive alternation.

    Whole words, never substrings: "only" must not match "commonly" and "no"
    must not match "node". A substring match would silently mark a vague row as
    explicit, which is the exact failure this flag exists to catch.

    Markers are ESCAPED before interpolation. GAP_MARKERS invites human tuning,
    and a raw marker fails two ways: "fp4(" raises re.error at IMPORT time and
    takes this whole module down with it, while "not.yet" compiles silently
    into a wildcard that also matches "notXyet". re.escape("not yet") is
    "not\\ yet", so widening that escaped space to \\s+ still works.

    Taking the markers as an argument is what makes the escaping testable: no
    shipped marker needs escaping today, so an inline expression could drop
    re.escape with nothing to notice.
    """
    alternation = "|".join(
        re.escape(marker).replace("\\ ", r"\s+") for marker in markers
    )
    return re.compile(r"\b(?:" + alternation + r")\b", re.IGNORECASE)


GAP_RE = gap_pattern(GAP_MARKERS)

# check mode fails on abandoned ACTIVE rows and nothing else. The PARTIAL flag
# is a keyword heuristic for human review; gating on it would be the fragile
# checker the protocol warns against.
CHECK_FAILS_ON = frozenset({"ACTIVE"})


def names_missing_modes(row_text: str) -> bool:
    """True when a PARTIAL row states what is NOT supported."""
    return GAP_RE.search(row_text) is not None


def matched_marker(row_text: str) -> str:
    """The gap marker that fired, or "" -- so a human can discount a bad hit.

    The heuristic under-flags: 11 of the 48 rows it reads as explicit qualify
    only via bare `no` or `gap`, on prose asserting GOODNESS rather than
    absence ("no longer double-resides", "max gap 0.0 nats", "CLOSED the CPU
    RSS gap"). Naming the marker lets a reviewer dismiss those at a glance
    instead of trusting the verdict.
    """
    match = GAP_RE.search(row_text)
    return match.group(0) if match else ""
```

`re` is already imported at the top of the module from Task 1; if it is not, add it there rather than mid-file.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 tests/scripts/test_audit_live_rows.py -v`
Expected: PASS, 26 tests.

- [ ] **Step 5: Run preflight and commit**

```bash
bash scripts/agent-preflight.sh > /tmp/preflight.log 2>&1; echo "EXIT=$?"
git add scripts/audit-live-rows.py tests/scripts/test_audit_live_rows.py
git commit -F - <<'EOF'
tools(audit): advisory PARTIAL missing-modes flag (P0 step 3)

Report-only by construction: CHECK_FAILS_ON is ACTIVE alone, so the keyword
heuristic can never fail a build.

FOLLOWING_AGENTS_PROTOCOL
Assisted-by: Claude Code:claude-opus-5 [ClaudeCode]
EOF
```

---

### Task 4: Report rendering and CLI

**Files:**
- Modify: `scripts/audit-live-rows.py`
- Test: `tests/scripts/test_audit_live_rows.py`

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces: `audit() -> list[dict]` (one record per live row, keys `id`, `state`, `path`, `line`, `verdict`, `reason`, `flag`, `duplicate`); `duplicate_live_ids(rows: list) -> dict[str, list[str]]`; `render_markdown(records: list[dict]) -> str`; `main(argv: list[str] | None = None) -> int`. `audit()` calls `require_origin_main()` first and aborts on any parse error.

- [ ] **Step 1: Write the failing test**

Append to `tests/scripts/test_audit_live_rows.py`, above the `if __name__` block:

```python
class ReportTests(unittest.TestCase):
    RECORDS = [
        {
            "id": "ENG-FOO",
            "state": "ACTIVE",
            "path": ".agents/engine-matrix.md",
            "line": 42,
            "verdict": "ABANDONED",
            "reason": "no branch, no commit on main mentioning the row ID",
            "flag": "",
        },
        {
            "id": "MODEL-BAR",
            "state": "PARTIAL",
            "path": ".agents/model-matrix.md",
            "line": 7,
            "verdict": "",
            "reason": "",
            "flag": "does not name its missing modes",
        },
    ]

    def test_markdown_lists_every_record(self):
        out = audit.render_markdown(self.RECORDS)
        self.assertIn("ENG-FOO", out)
        self.assertIn("MODEL-BAR", out)
        self.assertIn("ABANDONED", out)
        self.assertIn("does not name its missing modes", out)

    def test_markdown_cells_do_not_break_the_table(self):
        records = [dict(self.RECORDS[0], reason="a | b")]
        out = audit.render_markdown(records)
        body = [ln for ln in out.splitlines() if "ENG-FOO" in ln]
        self.assertEqual(len(body), 1)
        self.assertNotIn("a | b", body[0])

    def test_check_mode_fails_when_an_active_row_is_abandoned(self):
        self.assertEqual(audit.exit_code(self.RECORDS, check=True), 1)

    def test_check_mode_passes_when_no_active_row_is_abandoned(self):
        clean = [dict(self.RECORDS[0], verdict="IN-FLIGHT")] + self.RECORDS[1:]
        self.assertEqual(audit.exit_code(clean, check=True), 0)

    def test_only_the_vague_flag_counts_as_needing_review(self):
        # Every PARTIAL row carries a flag: the marker that fired, or the vague
        # string. Counting non-empty flags would report all 68 as vague.
        explicit = dict(self.RECORDS[1], flag="explicit via 'missing'")
        self.assertNotEqual(explicit["flag"], audit.VAGUE_FLAG)
        self.assertEqual(self.RECORDS[1]["flag"], audit.VAGUE_FLAG)

    def test_report_mode_always_exits_zero(self):
        self.assertEqual(audit.exit_code(self.RECORDS, check=False), 0)

    def test_vague_partial_alone_never_fails_check_mode(self):
        only_flag = [self.RECORDS[1]]
        self.assertEqual(audit.exit_code(only_flag, check=True), 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tests/scripts/test_audit_live_rows.py -v`
Expected: FAIL with `AttributeError: module 'audit_live_rows' has no attribute 'render_markdown'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/audit-live-rows.py`:

```python
import argparse
import json


# Every PARTIAL row carries a flag string now -- either the marker that fired
# or this. So "needs review" is THIS string, never merely a non-empty flag;
# counting non-empty flags would report all 68 PARTIAL rows as vague.
VAGUE_FLAG = "does not name its missing modes"


def duplicate_live_ids(rows: list) -> dict[str, list[str]]:
    """Row IDs that appear live in more than one matrix.

    BACKEND-CUDA-SM121 and BACKEND-CPU are PARTIAL in BOTH backend-matrix.md
    and feature-matrix.md, so 188 live rows carry only 186 unique IDs.
    check-agent-record.py's duplicate check only walks MATRIX_PATHS, so it has
    never seen these. Left unresolved, the backfill would mint two issues for
    one item and this audit would report each twice with identical evidence.
    """
    seen: dict[str, list[str]] = {}
    for row in rows:
        seen.setdefault(row.item_id, []).append(f"{row.path.name}:{row.line_no}")
    return {k: v for k, v in seen.items() if len(v) > 1}


def audit() -> list[dict]:
    """One record per live row, with verdict (ACTIVE) and flag (PARTIAL)."""
    require_origin_main()
    parse_errors: list[str] = []
    rows = live_rows(parse_errors)
    if parse_errors:
        raise SystemExit(
            "the matrices do not parse cleanly, so the census is incomplete:\n"
            + "\n".join(parse_errors)
        )
    duplicates = duplicate_live_ids(rows)
    branches_by_id = row_branches()
    records: list[dict] = []
    for row in rows:
        verdict = ""
        reason = ""
        flag = ""
        if row.state == "ACTIVE":
            branches = branches_by_id.get(row.item_id, [])
            unmerged_by_branch = {b: unmerged(b) for b in branches}
            verdict, reason = classify_active(
                branches, unmerged_by_branch, main_commits(row.item_id)
            )
        if row.state == "PARTIAL":
            marker = matched_marker(row.raw)
            flag = f"explicit via {marker!r}" if marker else VAGUE_FLAG
        records.append(
            {
                "duplicate": ", ".join(duplicates.get(row.item_id, [])),
                "id": row.item_id,
                "state": row.state,
                "path": str(row.path.relative_to(ROOT)),
                "line": row.line_no,
                "verdict": verdict,
                "reason": reason,
                "flag": flag,
            }
        )
    return records


def _cell(value: object) -> str:
    """Table cells never contain a raw pipe -- it would split the row."""
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def render_markdown(records: list[dict]) -> str:
    lines = [
        "| Row | State | Location | Verdict | Evidence | Flag |",
        "|---|---|---|---|---|---|",
    ]
    for item in records:
        lines.append(
            "| `{id}` | `{state}` | {path}:{line} | {verdict} | {reason} | {flag} |".format(
                id=_cell(item["id"]),
                state=_cell(item["state"]),
                path=_cell(item["path"]),
                line=_cell(item["line"]),
                verdict=_cell(item["verdict"]) or "-",
                reason=_cell(item["reason"]) or "-",
                flag=_cell(item["flag"]) or "-",
            )
        )
    return "\n".join(lines)


def exit_code(records: list[dict], check: bool) -> int:
    if not check:
        return 0
    abandoned = [
        item
        for item in records
        if item["state"] in CHECK_FAILS_ON and item["verdict"] == "ABANDONED"
    ]
    return 1 if abandoned else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if any ACTIVE row is abandoned",
    )
    args = parser.parse_args(argv)

    records = audit()
    if args.json:
        print(json.dumps(records, indent=2, sort_keys=True))
    else:
        print(render_markdown(records))
        stale = [i for i in records if i["verdict"] == "ABANDONED"]
        vague = [i for i in records if i["flag"] == VAGUE_FLAG]
        dupes = sorted({i["id"] for i in records if i["duplicate"]})
        print(
            f"\n{len(records)} live rows; {len(stale)} abandoned ACTIVE; "
            f"{len(vague)} PARTIAL rows to review; "
            f"{len(dupes)} IDs live in two matrices: {', '.join(dupes) or 'none'}."
        )
    return exit_code(records, args.check)


if __name__ == "__main__":
    raise SystemExit(main())
```

Move the `import argparse` and `import json` lines up into the module's import block at the top of the file so imports are not scattered mid-module.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 tests/scripts/test_audit_live_rows.py -v`
Expected: PASS, 38 tests (39 added minus the transitional CLI-guard test you delete here).

- [ ] **Step 5: Smoke-test the CLI against the real repository**

Run: `git fetch -q origin && python3 scripts/audit-live-rows.py | tail -5`
Expected: a summary line reading `188 live rows; N abandoned ACTIVE; M PARTIAL rows to review.`

Run: `python3 scripts/audit-live-rows.py --json | python3 -c "import json,sys; print(len(json.load(sys.stdin)))"`
Expected: `188`.

**Note:** `--check` is expected to exit 1 right now. That is the finding, not a bug. Do not wire it into preflight or CI until Task 7.

- [ ] **Step 6: Run preflight and commit**

```bash
bash scripts/agent-preflight.sh > /tmp/preflight.log 2>&1; echo "EXIT=$?"
git add scripts/audit-live-rows.py tests/scripts/test_audit_live_rows.py
git commit -F - <<'EOF'
tools(audit): report, JSON and check modes (P0 step 4)

check mode is deliberately NOT wired into preflight or CI yet -- it fails
today, and that failure is the audit finding the record has to absorb first.

FOLLOWING_AGENTS_PROTOCOL
Assisted-by: Claude Code:claude-opus-5 [ClaudeCode]
EOF
```

---

### Task 5: Run the audit and record the findings

**Files:**
- Create: `.agents/specs/live-state-audit-2026-08-06.md`

**Interfaces:**
- Consumes: `scripts/audit-live-rows.py --json`.
- Produces: the findings artifact that Task 6 cites as justification for every correction.

**No matrix is edited in this task.** Findings land first, corrections second, so the reasoning is reviewable independently of the churn.

- [ ] **Step 1: Refresh remotes and capture the audit**

```bash
git fetch -q origin
python3 scripts/audit-live-rows.py --json > /tmp/audit.json
python3 scripts/audit-live-rows.py > /tmp/audit.md
tail -3 /tmp/audit.md
```

- [ ] **Step 2: Summarise the verdict distribution**

```bash
python3 -c "
import json, collections
records = json.load(open('/tmp/audit.json'))
print('ACTIVE verdicts:', collections.Counter(r['verdict'] for r in records if r['state']=='ACTIVE'))
print('PARTIAL flagged:', sum(1 for r in records if r['flag']))
print('by matrix:', collections.Counter(r['path'] for r in records if r['verdict']=='ABANDONED'))
"
```

- [ ] **Step 3: Hand-verify a sample before trusting the tool**

Pick three rows the tool called `ABANDONED` and one it called `IN-FLIGHT`. For each, run:

```bash
git log --oneline --all --fixed-strings --grep="<ROW-ID>" | head -5
git branch -a --list "*row/<ROW-ID>"
```

Confirm the tool's verdict matches what you see. A classifier that is wrong on a sample is wrong on all 49 — fix it and re-run rather than proceeding. Record the sample and its outcome in the artifact.

- [ ] **Step 4: Write the findings artifact**

Create `.agents/specs/live-state-audit-2026-08-06.md` with these sections:

- **Scope** — the 188 live rows on `origin/main` @ `<SHA>`; what the audit does and does not decide.
- **Method** — `scripts/audit-live-rows.py`, the classification rules verbatim, and the hand-verified sample from Step 3 with its result.
- **Findings** — the full report table from `/tmp/audit.md`, plus the verdict distribution from Step 2.
- **Proposed corrections** — one line per row needing a change, with its target state and the contract obligation that target carries. Apply the legality rule from Global Constraints: an abandoned `ACTIVE` row goes to `READY` if it has a real spec link, otherwise to `INVENTORIED`.
- **Duplicate live IDs** — `BACKEND-CUDA-SM121` and `BACKEND-CPU` are `PARTIAL` in both `backend-matrix.md` and `feature-matrix.md`. Decide which matrix OWNS each row and what the other becomes (a non-claimable reference, a differently-keyed row, or deleted), and say why. This must be settled here: the backfill would otherwise mint two issues for one item.
- **Rows left alone** — every row the audit did not propose changing, named, so the next reader can see it was considered and kept. There are **no `IN-FLIGHT` rows**: not one of the 54 `ACTIVE` claims has an unmerged `row/<ID>` branch.
- **The `LANDED` caveat** — all 44 `LANDED` verdicts rest on a commit *mentioning* the row ID, and **8 of them on records-only commits that changed no code**. `LANDED` means "has evidence worth reading", never "finished". No row may be proposed `DONE` on a commit mention alone.
- **Risks/decisions** — every verdict the tool could not decide, and the human call made.

- [ ] **Step 5: Run preflight and commit**

```bash
bash scripts/agent-preflight.sh > /tmp/preflight.log 2>&1; echo "EXIT=$?"
git add .agents/specs/live-state-audit-2026-08-06.md
git commit -F - <<'EOF'
record(audit): live-state audit findings, no corrections applied yet (P0 step 5)

Findings land before corrections so the reasoning is reviewable independently
of the churn. Includes the hand-verified sample that validates the classifier.

FOLLOWING_AGENTS_PROTOCOL
Assisted-by: Claude Code:claude-opus-5 [ClaudeCode]
EOF
```

---

### Task 6: Apply the corrections, one matrix per commit

**Files:**
- Modify: `.agents/engine-matrix.md`, `.agents/model-matrix.md`, `.agents/kernel-matrix.md`, `.agents/quantization-matrix.md`, `.agents/backend-matrix.md`, `.agents/feature-matrix.md`, `.agents/sglang-matrix.md` (only those with corrections)
- Modify: `.agents/roadmap_v1.md` (only if a corrected row has a portfolio row)
- Modify: `.agents/state-index/<writable-shard>.csv`, `.agents/NOW.md`, `docs/STATUS.md`
- Create: `.agents/state-events/<YYYY-MM>/<event-id>.md`

**Interfaces:**
- Consumes: the **Proposed corrections** section of `.agents/specs/live-state-audit-2026-08-06.md`.
- Produces: matrices whose live states are true, so P2 can mint issues from a corrected record.

**One commit per matrix.** A single sweeping commit is unreviewable, and a bad
transition in one matrix should be revertible without losing the others. The 10
rows sit in 5 matrices: engine 3, model 3, kernel 2, quantization 1, backend 1.

**SCOPE, user-directed 2026-08-06 — the 10 abandoned `ACTIVE` rows and nothing
else.** Explicitly OUT of scope, left as-is and already documented in the
artifact:

- the **44 `LANDED`** rows — every verdict rests on a commit merely mentioning
  the ID, 8 of them on commits that changed no code, so none is touched;
- the **~30 vague `PARTIAL`** rows — the repair is editorial and needs per-row
  knowledge of what actually works, which the tool cannot supply;
- the **2 duplicate IDs** (`BACKEND-CPU`, `BACKEND-CUDA-SM121`) — the ownership
  decision is recorded in the artifact and deliberately not applied here.

Touching any of those is scope creep. The artifact keeps them visible for a
later, separately-decided change.

- [ ] **Step 0: Re-fetch and re-run the audit**

The artifact pins `origin/main` at `cf32c619` and the remote has since advanced.
Run `git fetch -q origin` and re-run `python3 scripts/audit-live-rows.py --json`
before editing anything. If the verdict set has changed, the artifact's proposals
are stale — re-verify the affected rows rather than applying a stale list.

- [ ] **Step 1: Correct the first matrix**

Work one matrix at a time, starting with the one holding the most corrections. For each of the 10 rows in the artifact's corrections list, edit **only** the `State` and `Owner` cells. Do not touch `Our code`, `Tests/evidence`, `Upstream` or `Spike/spec` — those are durable anchors and are not what this audit is about.

Apply the legality rule: `READY` if the row has a real spec link, otherwise `INVENTORIED`. Clear the `Owner` cell to `-` for any row leaving `ACTIVE`.

- [ ] **Step 2: Verify the record still validates**

Run: `python3 scripts/check-agent-record.py; echo "EXIT=$?"`
Expected: `EXIT=0`.

If it fails with a spec or evidence-anchor complaint, the target state was illegal for that row — re-read the legality rule and pick the correct target. **Never** weaken the checker or strip an anchor to make the transition pass.

- [ ] **Step 3: Confirm the audit agrees**

Run: `python3 scripts/audit-live-rows.py --json | python3 -c "
import json,sys
records=json.load(sys.stdin)
print('remaining abandoned ACTIVE:', sum(1 for r in records if r['verdict']=='ABANDONED'))
"`
Expected: the count has dropped by exactly the number of rows corrected in this matrix.

- [ ] **Step 4: Run preflight and commit this matrix**

```bash
bash scripts/agent-preflight.sh > /tmp/preflight.log 2>&1; echo "EXIT=$?"
git add .agents/<name>-matrix.md
git commit -F - <<'EOF'
record(<area>): live-state audit corrections — N rows off stale ACTIVE (P0 step 6)

Evidence: .agents/specs/live-state-audit-2026-08-06.md. Only State and Owner
cells move; durable anchors are untouched.

FOLLOWING_AGENTS_PROTOCOL
Assisted-by: Claude Code:claude-opus-5 [ClaudeCode]
EOF
```

- [ ] **Step 4b: Retire the coordination claims in the SAME commit**

Every abandoned row sits in an active claim in `.agents/coordination.md`, and
`check_row_contracts` cross-checks a row's `Owner` against that claim table.
**11 claims must be RETIRED, not emptied** — moved to the completed block with
their outcome — and it has to happen in the same commit as the matrix edit, or
`scripts/check-agent-record.py` goes red between the two.

**Retirement has an exact mechanic.** `parse_active_claims` keys on
`line.startswith("| \`CLAIM-")`, so a claim moved to a table row in another
section is *still parsed as active* and still red. The repo's existing archival
convention (`.agents/coordination.md:1643+`) uses **prose bullets** — follow it.

Run `python3 scripts/check-agent-record.py; echo "EXIT=$?"` and confirm `EXIT=0`
before committing. If it complains about a claim/owner mismatch, the claim
retirement is missing or partial — repair the record, never the checker.

- [ ] **Step 5: Repeat Steps 1–4b for each remaining matrix with corrections**

- [ ] **Step 6: Update the roadmap, structured state record and public status**

The record obligation is that the roadmap portfolio row and its owning area matrix row move in the **same change** as the state they describe. If any corrected row has a portfolio row in `.agents/roadmap_v1.md`, update it now.

Append one checkpoint row to the writable `.agents/state-index/` shard and add
its matching immutable Markdown evidence under `.agents/state-events/`, using
the schema and event contract in the structured-state design. Refresh
`.agents/NOW.md` in the same commit (the freshness coupling is CI-gated) and
update `docs/STATUS.md`. Leave the `.agents/state.md` compatibility stub
unchanged.

- [ ] **Step 7: Verify chronology and doc obligations, then commit**

```bash
python3 scripts/check-state-record.py; echo "state-record EXIT=$?"
bash scripts/agent-preflight.sh > /tmp/preflight.log 2>&1; echo "EXIT=$?"
git add .agents/roadmap_v1.md .agents/state-index/<writable-shard>.csv \
  .agents/state-events/<YYYY-MM>/<event-id>.md .agents/NOW.md docs/STATUS.md
git commit -F - <<'EOF'
record(state): live-state audit checkpoint — the ACTIVE claim set is now true

49 rows claimed ACTIVE simultaneously; the audit reconciled them against
branches and commits. Portfolio, structured state indexes, immutable event
evidence, NOW and STATUS move together.

FOLLOWING_AGENTS_PROTOCOL
Assisted-by: Claude Code:claude-opus-5 [ClaudeCode]
EOF
```

---

### Task 7: Turn the audit into a gate

**Files:**
- Modify: `scripts/agent-preflight.sh:50-63` (suite list) and its gate list
- Modify: `.github/workflows/ci.yml:42-46`
- Test: `tests/scripts/test_audit_live_rows.py`

**Interfaces:**
- Consumes: a corrected record where no `ACTIVE` row classifies `ABANDONED`.
- Produces: a standing gate, so the rot cannot silently return before P1–P5 land.

- [ ] **Step 1: Write the failing mutation test**

Append to `tests/scripts/test_audit_live_rows.py`, above the `if __name__` block:

```python
class GateWiringTests(unittest.TestCase):
    def test_preflight_runs_the_audit_suite(self):
        text = (ROOT / "scripts/agent-preflight.sh").read_text(encoding="utf-8")
        self.assertIn("test_audit_live_rows", text)
        self.assertIn("audit-live-rows.py", text)

    def test_ci_runs_the_gate_and_its_suite(self):
        text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("scripts/audit-live-rows.py --check", text)
        self.assertIn("tests/scripts/test_audit_live_rows.py", text)

    def test_shipped_record_has_no_abandoned_active_row(self):
        records = audit.audit()
        stale = [r["id"] for r in records if r["verdict"] == "ABANDONED"]
        self.assertEqual(stale, [], f"stale ACTIVE rows remain: {stale}")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tests/scripts/test_audit_live_rows.py -v`
Expected: FAIL on `test_preflight_runs_the_audit_suite` and `test_ci_runs_the_gate_and_its_suite`.

`test_shipped_record_has_no_abandoned_active_row` must **already pass** — Task 6 corrected the record. If it fails, Task 6 is incomplete; finish it before wiring the gate. Never wire a gate around an unrepaired record.

- [ ] **Step 3: Wire preflight**

In `scripts/agent-preflight.sh`, add `test_audit_live_rows` to the suite list at lines 50–63 (after `test_check_now_current`), and add the gate itself alongside the other checkers:

```sh
run audit-live-rows python3 scripts/audit-live-rows.py --check
```

- [ ] **Step 4: Wire CI**

In `.github/workflows/ci.yml`, extend the record job (lines 42–46):

```yaml
      - name: Live-state rows are reconciled against Git reality
        run: |
          git fetch -q origin main
          python3 scripts/audit-live-rows.py --check
          python3 tests/scripts/test_audit_live_rows.py
```

`git fetch` is required because the classifier compares against `origin/main`, and Actions checkouts are shallow by default.

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 tests/scripts/test_audit_live_rows.py -v`
Expected: PASS, 41 tests.

- [ ] **Step 6: Verify the whole gate is green**

```bash
python3 scripts/audit-live-rows.py --check; echo "gate EXIT=$?"
bash scripts/agent-preflight.sh > /tmp/preflight.log 2>&1; echo "preflight EXIT=$?"
```
Expected: both `EXIT=0`.

- [ ] **Step 7: Commit**

```bash
git add scripts/agent-preflight.sh .github/workflows/ci.yml tests/scripts/test_audit_live_rows.py
git commit -F - <<'EOF'
gate(audit): ACTIVE rows stay reconciled with Git reality (P0 step 7)

Wired only after the record was repaired, so the gate never had to be relaxed
to pass. Keeps the 49-row rot from returning before P1-P5 land.

FOLLOWING_AGENTS_PROTOCOL
Assisted-by: Claude Code:claude-opus-5 [ClaudeCode]
EOF
```

---

## Branch-level obligation: the doc checkpoint

`scripts/check-doc-checkpoint.py` requires that any commit touching `scripts/`,
`tests/` or `.agents/specs/` **also updates `docs/STATUS.md` and
`docs/BENCHMARKS.md` in the same commit**. CI enforces it **per commit** over the
PR range (`.github/workflows/ci.yml:126`), but `agent-preflight.sh` only runs it
`--staged` (line 109) — which passes vacuously once the work is already
committed. That asymmetry is why per-task commits accumulate violations while
every preflight reports green.

**Squashing alone does NOT clear it.** That was an early assumption and it is
disproven: feeding the union of every path this branch changes through
`checkpoint_errors()` still fails, because `docs/BENCHMARKS.md` and
`docs/FEATURES.md` are never touched anywhere on the branch. A squash fixes the
*per-commit* spread; it cannot conjure a file the branch never edits.

**The fix is a record repair, not a merge trick.** The owed surfaces must
actually be written. Both accept an explicit "nothing moved" entry — that is the
point of the obligation, which covers pending, failed and void checkpoints, not
only externally visible closure:

- `docs/BENCHMARKS.md` — a line recording that this checkpoint moved lifecycle
  bookkeeping and produced **no benchmark movement**.
- `docs/FEATURES.md` — a line recording that **no capability changed**.

That second one matters for honesty, not just for the gate. Moving a row to
`READY` is a statement about *evidence of in-flight work*, never about capability;
`scripts/check-model-checklist.py` forces `✅`→`🚧` at `READY`, which makes the
internal checklist claim less support than reality. Do **not** demote the public
`docs/FEATURES.md` marks to match — that would assert a regression that did not
happen. Explain the divergence instead.

This branch still lands as a **squash** (the house practice: `gh pr merge` would
attribute the squash to `localai-bot`, so squashes are landed locally via
`commit-tree` and a direct push), but only once the owed surfaces exist.

Verify before pushing, over the exact range that will be pushed:

```bash
python3 scripts/check-doc-checkpoint.py --base <merge-base> --head HEAD
```

This is a real obligation, not a formality to route around: never weaken the
checker to pass it.

## Done when

- `python3 scripts/audit-live-rows.py --check` exits 0 on a record where every `ACTIVE` row has real Git evidence behind it.
- `.agents/specs/live-state-audit-2026-08-06.md` justifies every correction, names every row left alone, and records the hand-verified classifier sample.
- `bash scripts/agent-preflight.sh` exits 0, with the new suite and gate registered.
- CI runs the gate and its mutation suite.
- The squashed commit updates `docs/STATUS.md` and `docs/BENCHMARKS.md`, and `check-doc-checkpoint.py` passes over the pushed range.
- P2's backfill can mint ~188 issues from a record that is true.

## Out of scope

Everything after P0. No labels, no milestones, no issue template, no `sync-rows.py`, no `check-issue-record.py`, no `State`/`Owner` column removal, no `coordination.md` retirement, no prose changes to `AGENTS.md`. This plan leaves the file-based protocol fully intact and merely truthful — which is why it is worth landing even if the rest of the migration never happens.
