# The operator lock becomes a RECORD of who is coordinating where

User-directed 2026-08-10, issue
[#285](https://github.com/mudler/vllm.cpp/issues/285). Row `ENG-OPERATOR-RECORD`
(tooling and policy; it owns no claim-matrix row and no product code).

The developer's words: *"I dont want push force main, never. the operator have
at max the way to merge directly PRs and heavily dispatch sub-agents with
separate worktrees ( == coordinator )"*, and, on the lock itself: *"let's keep
it as a record for who is working where"*.

## Scope

`scripts/agent-role.py` stops refusing a second operator. The file in the git
common dir stays, and stops being a lock: it becomes a set of records, one per
worktree, that says who is coordinating where. `show` reports the other live
coordinators. `release` removes only the caller's record. The 2-hour TTL and
stale pruning stay exactly as they are.

Out of scope: the helper role, the role marker, the `read-only` answer,
`--headless`, `check-role-discipline.py`'s row/PR rule, and anything about how
work lands. Nothing here weakens a gate: no refusal is added, and the one
refusal removed is the subject of the issue.

## Our baseline — why the exclusivity no longer holds

`scripts/agent-role.py:28-37` justifies a repo-wide exclusive lock with *"the
shared case is the operator's primary checkout, where one role is the correct
answer anyway."* That premise is gone. `AGENTS.md` § "Work happens in a
worktree" now requires **every** unit of work to take its own linked worktree,
and § "Landing work" requires everything to reach `main` from a task branch. So
there is no shared checkout to protect and no unsynchronised writer to exclude.

What an operator actually is, per the developer: a **coordinator**. Its maximum
powers are merging PRs directly and dispatching sub-agents into separate
worktrees. It never rewrites shared history — `main` is never force-pushed, with
no `--force` and no `--force-with-lease` — so a plain `git push` refuses any
non-fast-forward and **git itself is the interlock**. Two coordinators racing to
land serialise on that refusal: the loser fetches, re-merges, re-gates and
pushes again. A JSON file in `.git/` never provided that guarantee and cannot.

What the exclusivity costs is measured, not hypothetical.
`LOCK_TTL_SECONDS = 2 * 60 * 60`, so a session killed mid-flight — one was on
2026-08-10, by a host disk cleanup, leaving a dead pid and a frozen heartbeat —
blocks **all** coordination for up to two hours, and the only remedy is
hand-deleting a file inside `.git/`. A claim refused at 78 minutes is the TTL
working correctly; that it was refused at all is the defect.

## Design

**Representation.** One file per worktree in a directory,
`<git-common-dir>/vllm-cpp-operators/<sha256(worktree)[:16]>.json`, instead of
one shared `vllm-cpp-operator.lock`. Still in the git common dir, so it is
shared by every worktree and can never be committed.

That shape is what makes concurrency safe: **a writer only ever touches its own
record**, because the filename is derived from the identity ownership already
keys on. Two coordinators claiming at the same instant write two different
paths, so neither can lose the other's record — there is no read-modify-write of
a shared file anywhere in the design. Each write is `write temp + os.replace`
inside the same directory, which is atomic on POSIX, so a reader sees the old
record or the new one and never a half-written one. A single JSON array or a
JSONL log would both have required rewriting a shared file to release or prune,
which is exactly where a concurrent writer's record gets dropped.

`O_CREAT|O_EXCL` goes: it existed to make the second claimant fail, and the
second claimant must now succeed.

**Ownership still keys on the worktree** (`git rev-parse --absolute-git-dir`),
unchanged from the 2026-08-06 correction, and `record_is_ours` keeps the legacy
session fallback for a record written before that correction.

**Staleness.** `RECORD_TTL_SECONDS` keeps the 2-hour value and the heartbeat
semantics. Another worktree's stale record is filtered out of every display and
unlinked on the next `claim`, which is a write path; `show` and `resolve` never
unlink, because `agent-preflight.sh` documents itself as never writing anything.
A stale record can no longer refuse anybody, so breaking one is no longer an
event: the NOTE that announced it goes with the refusal it explained.

The prune skips THIS worktree's own record (`keep_canonical`, third review
round). Our own record is stale on any ordinary re-claim — the TTL is two hours
and a session re-claims at the top of its next tool call — and pruning it before
republishing it is the same unlink-then-create the second round removed from
`drop_our_record`, only reached by the commoner path. It costs nothing to skip:
`write_our_record` republishes that exact path immediately, and `resolve` matches
our own record by ownership with no staleness filter, so an aged own record was
never displayed as a live coordinator in the first place. The consequence of not
skipping is measured, not theoretical: a session killed inside the window turned
a worktree that resolved `role=operator` into `role=UNDECLARED` exit 3.

**Migration.** A pre-#285 `vllm-cpp-operator.lock` is read as one more record,
so a session that claimed operator before this change still resolves as operator
instead of silently becoming UNDECLARED mid-flight. Its next `claim` or
`release` removes it, which heals the repo the first time either runs.

**The front doors stop blocking.** `scripts/agent-onboard.py` and
`scripts/agent-start.py` told a session that `claim operator` would fail and
instructed it not to run "a known-failing claim". That claim no longer fails, so
`blocked_by_other_operator` is replaced by `operator_peers` — the live records
that are not this worktree's — and both surfaces report them as information
beside the ordinary claim command.

## Port map

None. This is repository tooling with no upstream vLLM counterpart; the porting
inventory's §9 (written from scratch) is where role machinery has always sat.

## Upstream chain

Not applicable — vLLM has no agent-role protocol. The authority for this change
is the developer's direction in issue #285.

## Tests to port

None to port. `tests/scripts/test_agent_role.py` grows the new behaviour, and
the tests that pinned the removed refusal are rewritten to pin its replacement
rather than deleted:

| Was | Becomes |
|---|---|
| `test_second_operator_is_refused` | `test_a_second_coordinator_is_recorded_not_refused` |
| `test_one_operator_per_repo_holds_across_worktrees` | `test_show_lists_the_other_live_coordinators` |
| `test_stale_lock_is_broken_but_reported` | `test_a_stale_record_is_pruned_and_never_blocks` |
| `test_a_legacy_lock_cannot_produce_two_operators` | `test_a_legacy_lock_file_is_adopted_as_this_worktrees_record` |
| `test_an_operator_marker_beaten_to_the_lock_reports_the_lockout` | `test_an_operator_whose_record_vanished_is_told_to_re_claim` |

New: concurrent claims from many worktrees lose no record; `release` removes
only the caller's; `show` reports worktree, session, host and heartbeat age for
each peer.

## Gates

- `python3 tests/scripts/test_agent_role.py` (focused, RED first)
- `python3 tests/scripts/test_agent_onboard.py`, `test_agent_start.py`
- `scripts/agent-preflight.sh --quiet` green before and after the commit
- Mutation: every guard the new tests claim to pin is deleted or inverted, the
  focused suite is shown RED, and the guard is restored and shown GREEN.

## Dependencies

None. Python, docs and tests only; no GPU, no build, no network.

## Work breakdown

Single unit — the tool, its suite, the two front doors that consumed the
refusal, and the documents that asserted it. Splitting it would leave the repo
in a state where the tool permits a second coordinator and the docs still forbid
one.

## Risks/decisions

- **Risk: a directory of files is harder to inspect than one file.** Accepted;
  `show` renders it, which is the point of keeping records at all.
- **Risk: nothing now prevents two coordinators merging to `main` at once.**
  That is the decision, not an oversight: git's non-fast-forward refusal is the
  real interlock and the lock never was one. It holds only while `main` is never
  force-pushed, which `AGENTS.md` now states as a rule.
- **Decision: keep the TTL.** It is correct and already works. Staleness now
  only prunes a display entry instead of gating a claim.
- **Decision: no `--force` variant is ever added to any script here.**

## Outcome

Landed 2026-08-10 on `row/ENG-OPERATOR-RECORD`. `claim operator` records and
succeeds alongside live peers; `show` lists every other live coordinator with
worktree, session, host and heartbeat age; `release` removes only the caller's
record; a stale record is pruned from the display and blocks nothing. The
refusal path and its `already held` message are gone from the tool, the two
front doors, and the documents.

Three review rounds. Round 2 scoped `drop_our_record` so a re-claim replaces its
record instead of unlinking it, and pinned the atomic publish. Round 3 found the
same window still open through `prune_stale_records`, which the round-2 comment
had asserted was closed: the prune runs first and unlinks any stale record,
including ours. It is now scoped the same way, and
`test_a_reclaim_never_unlinks_its_own_record` runs both ages of record because
only the fresh one had been covered. Round 3 also pinned what the publish tests
missed — a publish that unlinks the name and recreates it leaves the hardlink
witness and the residue check both green, so the NAME is now watched directly —
and pinned the `legacy` field a pre-#285 peer adds to `--probe --json`.

Still declined, unchanged: `prune_stale_records` targets a path rather than the
inode it read, so a PEER that republishes inside the window loses that record.
Its remedy is `claim operator`, which is never refused. With `keep_canonical`
that declination is now confined to peers; this worktree's own record is no
longer exposed to it.

## Follow-up: issue [#296](https://github.com/mudler/vllm.cpp/issues/296)

Round 3's final review passed with two LOW findings recorded rather than
blocked. Both are closed on `row/ENG-OPERATOR-RECORD-COVERAGE`. No product
behaviour changes: `scripts/agent-role.py`'s publish is unchanged apart from one
comment.

**1. The `RECORD_TTL_SECONDS` comment was falsified by its own fix.** It still
said a stale record is *"unlinked by the next `claim`"*, which `keep_canonical`
made untrue for THIS worktree's record — that one is replaced, never unlinked.
The identical sentence at line 74 above was corrected when the fix landed and
the source copy, one screen above the function it describes, was not. It now
carries the same qualification, and a tree-wide search found no third copy.

**2. The publish-NAME pin was escapable, and widening it again would not have
been the fix.** `test_a_publish_never_leaves_the_record_NAME_absent` watched
`Path.write_text`, `Path.unlink`, `os.unlink` and `os.remove`. A publish written
`os.rename(target, aside)` then `open(target, "w")` escaped it: MEASURED, the
pre-#296 suite is 59/59 green under that mutation. That is the third round in a
row where one publish-coverage residual was fixed and the next one opened, and
the reason is structural — **any monkeypatch-watcher pin is escapable by one more
I/O primitive**, so widening the list chases a fixed point.

The pin is now two tests that fail differently, and the evidence says neither
subsumes the other:

- `_watch_publish` (deterministic, in-process) adds `os.rename`/`os.replace` and
  `Path.rename`/`Path.replace` as SOURCE, plus `builtins.open`/`io.open`/`os.open`
  in write modes. It catches every primitive it names, on every run, in
  milliseconds — and only those.
- `test_a_concurrent_observer_never_sees_the_record_name_absent` (probabilistic,
  out-of-process) republishes the record 200 times in a subprocess while the test
  process does nothing but poll `exists()`. It enumerates nothing, so it is the
  only half that can catch a primitive nobody listed. It is ONE-SIDED — a hit
  proves absence, a miss proves nothing — and therefore cannot go red on correct
  code however the two processes are scheduled.

Measured 2026-08-10 on a 20-core box already at load average 263, ~1.5ms per
publish against ~1us per poll, 150k-300k polls per run:

| Publish under test | Watcher | Observer |
|---|---|---|
| `os.rename` + `open` (the documented escape) | RED | 10/10, and 10/10 at 60 publishes |
| `unlink` + `write_text` (the previous escape) | RED | 10/10 |
| byte-at-a-time in-place rewrite (the escape before that) | green — caught instead by the hardlink test | 0/6 |
| shipped `write temp + os.replace` | green | 40/40 green, zero absent readings |
| shipped, both processes pinned to ONE core | green | 20/20 green |
| `os.rename` + `open`, pinned to ONE core | RED | 1/15 |

The two pinned rows are why both halves ship. The observer's sensitivity needs
real parallelism — on a single core the publisher is rarely preempted inside a
window that lasts microseconds — and it is blind by construction to a publish
that never makes the name absent. The watcher is unaffected by scheduling and
covers exactly the named primitives. Sizing follows from that: 200 publishes is
~0.39s, chosen for margin over the 60 that already detected 10/10, and the only
non-assertion guard is a floor of 200 polls, ~0.1% of the measured count, whose
sole job is to notice a poll loop that never ran. A loaded machine costs
detection probability, never a red run.

The test's stated claim was weakened to match what it proves: it no longer says
*"at the instant **any** file content is written … **nothing** may unlink it"*,
but bounds itself to the primitives `_watch_publish` names, and points at the
observer for the rest.

**Still declined here:** the observer is a backstop, not a gate — a single-core
CI runner reduces it to ~1/15 per run on the mutation it is aimed at. The
deterministic half is what holds the line there, and the residual is recorded
rather than closed with a longer run, because buying meaningful single-core
sensitivity costs seconds per suite invocation for a probability that still is
not one.
