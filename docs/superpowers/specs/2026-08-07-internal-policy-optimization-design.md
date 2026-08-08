# Internal policy optimization 1

Status: accepted design for PR #128.  This specification covers the policy and
agent-prompt layer only.  Converting project records into structured sources is
a separate follow-up campaign.

## Goal

Make the repository's agent policy smaller, unambiguous, agent-agnostic, and
mechanically enforced.  A green gate must mean either that the applicable rule
was satisfied or that a narrow, attributable, unexpired waiver exists.  It must
never mean that prose was present, a remote snapshot was merely recent, or a
failure was silently treated as absence.

The root `AGENTS.md` remains the agent-facing entrypoint and index.  It stops
being a second full copy of the policy.

## Scope

This PR owns:

- the authority and precedence of repository policy;
- a compact, structured policy-rule registry;
- the root agent index and the active policy/procedure file set;
- role, worktree, branch, task, PR, trailer, and waiver enforcement;
- the implementer, reviewer, and operator prompt contracts;
- bounded migration from the current protocol;
- mutation tests proving that the new gates fail for the defects they name.

This PR does not convert matrices, state, ledgers, benchmark records, or public
status pages into structured data.  That is the explicitly deferred
"structured records" PR.  This PR may archive obsolete policy rationale and
legacy claim narratives, but it must not reinterpret feature evidence or
rewrite matrix lifecycle state.

## Why the current system is not strict

The current preflight is green, but several binding claims are false or
contradict one another.

### Authority and context

- `AGENTS.md:3-4` says to read `AGENTS.md` first, while `AGENTS.md:39` and
  `.agents/workflow.md:59` say `NOW.md` is first.  There is no single boot order.
- `AGENTS.md:28-34` requires a full interactive environment/preferences setup,
  while `.agents/workflow.md:40-44` requires just-in-time, one-value onboarding
  and explicitly forbids walking the whole template.
- `AGENTS.md:348` requires a state entry every working session, while
  `AGENTS.md:168-171` exempts routine review, Git housekeeping, and protocol
  discussion.
- `.agents/coordination.md:72-80` still defines hand-maintained claims, while
  `.agents/specs/operator-helper-protocol.md:175-187` says PR-derived claims
  replace that table and must never be hand-edited.
- `.agents/coordination.md:84-87` still requires `README.md` at every checkpoint,
  contradicting the public-document contract at `AGENTS.md:175-202` and
  `.agents/workflow.md:141-158`.
- `AGENTS.md:88-90` calls 16/16 token identity an exceptionless precondition,
  while `.agents/NOW.md` explicitly recognizes a ratified distributional gate.
  Exceptions exist in practice but are not represented as structured policy.
- `.agents/specs/operator-helper-protocol.md:3-5` says the design is not yet
  enforced and `AGENTS.md` is untouched, while its own work table at lines
  256-261 says W0-W5 landed and the current `AGENTS.md` contains the protocol.

The root `AGENTS.md` is 23,181 bytes.  It is below Codex's default 32 KiB
combined project-instruction limit by itself, but leaves little room for global
or nested instructions and points to more than 70 KiB of additional binding
prose.  Linked files are not automatically part of an agent's instruction
chain.  Current official guidance recommends a concise, practical `AGENTS.md`,
task prompts with goal/context/constraints/completion criteria, and reusable
workflows outside the root instruction file:

- <https://learn.chatgpt.com/docs/agent-configuration/agents-md>
- <https://learn.chatgpt.com/docs/prompting>

The design remains agent-agnostic; these sources validate the general context
and prompt principles rather than defining tool-specific policy.

### Enforcement gaps

- `scripts/agent-role.py:220-291` writes a marker for a helper.  It does not
  create a worktree, create the exact branch, validate a known task, push a
  reservation, or open a PR.  It accepts helper mode in the primary checkout.
- `scripts/claim-view.py:42-45,113-145` checks only that a committed PR snapshot
  is syntactically valid, references known matrix rows, and is no more than 14
  days old.  On 2026-08-07 its view said `_none_` while six live `row/*` PRs
  existed; preflight still passed.
- `scripts/ready-for-helper.py:50-96` does not test the documented unmerged-PR
  dependency condition.  It tests file existence, not that the spec is
  committed; approximates runnable gates through an anchor heuristic; and
  treats the absence of hardware words as proof of CPU gateability.
- `.github/workflows/ci.yml:180-214` searches for the string
  `FOLLOWING_AGENTS_PROTOCOL` anywhere in a commit message.  It is not a Git
  trailer and the gate does not validate the `Assisted-by:` contract.
- `scripts/check-role-discipline.py:43-60` and
  `scripts/check-pr-size.py:24-25` exempt broad directory classes.  Material
  feature or enforcement logic under `scripts/`, `docs/`, or `.agents/` can
  bypass the path and size rules.
- `scripts/check-protocol-consistency.py` pins a few literal phrases but does not
  check the semantics above.  Contradictory prose can contain every required
  phrase and remain green.
- The new `internal-policy-optimization-1` helper claim is itself unknown to the
  matrix-row registry.  Refreshing the current claim view would reject the PR
  that repairs the claim system.

### Live-context growth

The tracked `.agents/` tree is about 131,000 lines.  `state.md` is about 41,000
lines, `benchmark-record.md` about 15,000, and `coordination.md` about 2,000.
This conflicts with the stated goal that the active `.agents/` surface contains
only current context.  `NOW.md` records that line-number evidence anchors now
block record-era rollover.  The full structural repair belongs to the follow-up
PR; this PR must avoid adding another narrative policy log and must archive
superseded policy material.

## Authority model

### Root index

`AGENTS.md` remains the automatically discovered, agent-facing index.  It is
hand-curated outside delimited generated blocks and has a hard budget of 12 KiB.
It contains only:

1. policy precedence and the exact boot sequence;
2. the small T0 rule subset required under context pressure;
3. exact role/preflight/ready commands;
4. the task-based map to current procedures and records;
5. the commit and PR handoff summary.

The T0 block is rendered from the structured policy registry.  Manual edits to
that block fail `--check`.

### Canonical rule registry

`.agents/policy.csv` is the sole repository-policy authority.  It is an RFC 4180
CSV file with this header:

```csv
rule_id,scope,trigger,requirement,enforcement,waiver_class,procedure
```

Contract:

- one rule per physical line; multiline fields are forbidden;
- `rule_id` is stable and unique;
- `scope` and `trigger` say when the rule applies;
- `requirement` is one short, testable statement;
- `enforcement` names one or more real checker entrypoints;
- `waiver_class` is `never`, `expiring`, or `migration-only`;
- `procedure` links to the instructions for satisfying the rule;
- rationale, incidents, dated counts, and attempt history are forbidden;
- maximum 60 rules and 16 KiB.  Growth beyond either budget fails and requires
  consolidation or moving procedural detail.

The registry is machine-readable and human-reviewable.  There is no second
generated full policy-reference file.

### Procedures

The active cross-project procedure set becomes:

- `.agents/workflow.md` — session, role, task, PR, and handoff procedure;
- `.agents/verification.md` — correctness, performance, oracle, trace, and
  reproduction procedures, consolidating `gates.md` and
  `benchmark-protocol.md`;
- `.agents/porting.md` — upstream grounding, design discipline, test porting,
  and deviation procedure, consolidating `discipline.md` and `test-porting.md`.

Procedures explain how to satisfy a rule.  A normative paragraph must cite its
`POL-*` rule ID and cannot create a new independent obligation.  Specialized
technical documents such as parity-lever, upstream-sync, backend, environment,
and matrix documents remain task-specific references rather than global policy
authorities.

The active policy set has explicit size budgets.  Proposed initial limits are
12 KiB for `AGENTS.md`, 16 KiB for `policy.csv`, 16 KiB for `workflow.md`, 24
KiB for `verification.md`, 16 KiB for `porting.md`, and 4 KiB per runtime prompt.

### Public documentation projections

The redesign preserves and strengthens the current public-document obligations.
It does not require every document on every change.  Each surface has one
purpose-specific policy rule and trigger:

- `docs/STATUS.md` changes at every feature or iteration checkpoint and states
  the current lifecycle, active gap, evidence, and next gate;
- `docs/BENCHMARKS.md` changes at every feature or iteration checkpoint with an
  accepted result or explicit `PENDING`, `NOT APPLICABLE`, `FAILED`, or `VOID`
  disposition and reproduction action;
- `docs/FEATURES.md` changes whenever a feature, model, backend, or quantization
  support surface changes;
- `docs/USAGE.md` changes whenever a user-facing command, flag, API, endpoint,
  configuration key, installation step, or runnable workflow changes;
- `README.md` remains the human-friendly landing page and changes only when a
  user-visible headline, quickstart, installation path, positioning statement,
  or headline benchmark changes;
- `.agents/NOW.md` changes when live claims, the current gate, next actions, or
  a `state.md` append changes.  A policy-only design that moves none of those
  does not manufacture a NOW entry.

The repository has no `NEWS` or changelog surface at this design point.  The
word "news" is interpreted as `.agents/NOW.md`; adding a release-news surface
would be a separate product/documentation decision.

The policy registry owns these triggers.  A documentation checker classifies
the semantic change class from explicit path/rule mappings, not from the mere
presence of any file under `.agents/specs/`.  This fixes the observed failure
where staging this governance design falsely demanded public feature and
benchmark updates.

The public files are projections, not duplicate narratives: README links to
USAGE, STATUS, FEATURES, and BENCHMARKS; each detail lives in exactly one of
those destinations.

### Precedence and boot order

External system, developer, and user instructions retain their platform-defined
precedence.  Within the repository:

1. `AGENTS.md` selects the applicable rule IDs and references;
2. `policy.csv` defines the rule;
3. the linked procedure explains how to comply;
4. task records and evidence describe the current instance but do not amend the
   rule.

The one boot sequence is:

1. resolve the existing worktree role; if undeclared, ask what work is being
   done and materialize the answer;
2. read `NOW.md`;
3. read shared developer preferences and only the environment values needed by
   the current gate;
4. read the claimed task/row and only its linked procedures/evidence;
5. run preflight before edits and before every commit/push transition.

Untracked developer configuration must resolve from one shared location across
linked worktrees.  A helper must not silently lose the primary checkout's
preferences or `.env`, and it must never infer another developer's values.

## Consolidation and archival

The following active policy sources are retired or folded:

- `.agents/directives.md` -> rules in `policy.csv`, procedures in the three
  procedure documents, historical rationale in `completed/`;
- `.agents/ai-coding-assistants.md` -> contribution rules in `policy.csv` and
  commit procedure in `workflow.md`;
- `.agents/specs/operator-helper-protocol.md` -> current procedure in
  `workflow.md`, design/incidents archived under `completed/`;
- `.agents/gates.md` + `.agents/benchmark-protocol.md` ->
  `.agents/verification.md`;
- `.agents/discipline.md` + `.agents/test-porting.md` -> `.agents/porting.md`;
- the policy/procedure preamble of `.agents/coordination.md` -> `workflow.md`.

Legacy/manual claim narratives needed to interpret pre-cutover owners move to
an era-stamped completed record.  New claims are PR-derived.  Feature lifecycle
rows, append-only evidence, and public projections are not structurally migrated
in this PR.

Every move repairs repository links in the same change.  No evidence is deleted.

## Role and claim enforcement

### Known tasks

Feature/model/kernel/backend IDs continue to come from their matrices.  A small
`.agents/governance-tasks.csv` admits non-feature tasks such as this policy PR.
It uses stable IDs and contains only current governance work.  It is not the
general structured-record system deferred to the follow-up PR.

### Helper start

The supported helper entrypoint is a transactional orchestration command rather
than a marker-only `claim helper`:

```text
scripts/agent-role.py start-helper --task <ID> [--headless] [--open-pr]
```

It must:

1. validate the task against a matrix or governance registry;
2. refuse the primary checkout as the helper's final workspace;
3. create a linked worktree and exact `row/<ID>` branch from the selected base;
4. write the role marker in the linked worktree's Git directory;
5. when remote operations are authorized, push a reservation commit and open a
   draft PR before implementation;
6. return the worktree path, branch, task, role, PR, and verification state.

If a remote step fails, the command must either roll back safe local reservation
state or retain it as an explicit `INCOMPLETE_RESERVATION`.  It must never print
success for a marker-only helper.  Without remote authority the session may
prepare locally, but it cannot claim `PR_VERIFIED` or mark the task ready.

Plain role resolution refuses:

- helper markers in the primary checkout;
- branch/task mismatch;
- unknown tasks;
- missing or conflicting operator locks;
- a supposedly ready helper with no matching open PR.

### Live claims

The committed timestamp snapshot and its 14-day TTL are removed.  Local,
network-independent preflight validates the marker/worktree/branch/task tuple.
The ready/integration gate and CI query live PR state and reject:

- two open PRs claiming one task;
- an unknown task;
- a `row/*` PR whose head does not exactly match its task;
- a task marked ready without a draft/ready PR;
- a merged/closed PR still represented as live.

Remote unavailability is `REMOTE_UNVERIFIED`, not "no claim" and not green for
integration.

### Ready-for-helper

The helper queue must actually prove its advertised conditions:

- the spec is tracked and reachable from the chosen base commit;
- the gate command exists, is executable, and has a nonzero failure mutation;
- CPU gateability is explicit, otherwise exact hardware is declared;
- dependencies are parsed and merged/satisfied;
- live PR state contains no claim;
- the lifecycle state is pickable.

Absence of a hardware keyword is not evidence of CPU gateability.

## PR, path, and commit enforcement

All new feature, policy, checker, documentation, and record changes arrive
through a PR.  The operator integrates; it does not directly push new work to
`main`.  Emergency repairs use an exact, expiring waiver.

Path classification must not exempt whole mutable trees.  Policy/checker work
under `scripts/`, `tests/scripts/`, `.agents/`, `docs/`, or `.github/` remains
reviewed and size-bounded.  Generated evidence and append-only records may use
separate budgets, but they are explicit classes rather than blanket exemptions.

New commits use real Git trailers:

```text
Following-Agents-Protocol: true
AI-Assisted: true
Assisted-by: Codex:GPT-5 [Codex]
```

`AI-Assisted: false` omits `Assisted-by`.  When true, at least one syntactically
valid `Assisted-by` is required.  An AI agent never adds its own `Signed-off-by`
or `Co-authored-by`.  The checker uses `git interpret-trailers` semantics rather
than substring search.

The old magic line is accepted only for commits before the cutover.  PR #128 is
squashed or amended so its final range satisfies the new contract.

## Waivers and policy changes

`.agents/waivers.csv` has this schema:

```csv
waiver_id,rule_id,scope,owner,reason,evidence,expires
```

Rules:

- `rule_id` must exist and permit the requested waiver class;
- `scope` identifies an exact task, PR, commit, path, gate, or hardware leg;
- `owner`, `reason`, and `evidence` are nonempty;
- `expires` is an ISO date and must be in the future;
- wildcard or repository-wide scope is forbidden;
- expired, unused, duplicated, or unknown waivers fail;
- migration-only waivers cannot be created after the cutover window;
- `waiver_class=never` cannot be waived.

Permanent `REPORT-ONLY` enforcement is removed.  Diagnostics may exist, but a
rule is either not applicable, satisfied, waived, pending a named external gate,
or failing.

"Never weaken a checker" becomes a testable change-control rule.  A checker
semantic change requires:

1. a dedicated policy PR/task;
2. the affected rule IDs;
3. a before/after mutation that demonstrates the old and intended behavior;
4. synchronized registry and procedure updates;
5. no unrelated gate relaxation.

This permits legitimate checker repairs without allowing a red transition to be
made green by assertion deletion or scope widening.

## Agent-agnostic prompt contracts

The runtime prompts remain plain Markdown and tool-neutral.  Each is versioned,
under 4 KiB, and contains four sections: required inputs, role method, required
output, and stop conditions.  Optional tool adapters may explain invocation but
cannot alter policy or acceptance criteria.

### Common task envelope

Every dispatched task supplies:

- `Goal` — the outcome, not a preselected implementation;
- `Context` — exact task/row, files, sources, baseline, and dependencies;
- `Constraints` — boundaries, allowed actions, policy IDs, and prohibitions;
- `Done when` — observable acceptance criteria and exact gates;
- `Required evidence` — commands, mutations, traces, or comparisons owed;
- `Authority` — remote, hardware, dependency, download, and service permissions.

A missing required field is `NEEDS_CONTEXT`; the agent does not guess a
materially different task.

### Implementer

The implementer uses test-first development where applicable, makes the minimum
in-scope change, runs focused gates, and then runs the required project gate.  A
material disagreement with the brief returns `NEEDS_DECISION`; it does not
silently expand scope.  Baseline red gates are reported and block ready status
unless covered by an applicable waiver.

Required output:

- status: `COMPLETE`, `BLOCKED`, `NEEDS_CONTEXT`, or `NEEDS_DECISION`;
- summary and changed files;
- commands and exit results;
- negative/mutation evidence for new tests;
- deviations and applicable waiver IDs;
- unresolved risks and omitted gates;
- commit SHA when committed.

### Reviewer

Review has three independent lenses:

1. static contract review for missing requirements, architecture, security,
   unsupported claims, absent tests, and scope drift;
2. targeted mutation/negative tests for the important changed claims;
3. relevant full-gate verification once focused checks are complete.

Mutation supplements static review; it never replaces it.  Each mutation runs
the smallest relevant test in a scratch copy.  The reviewer does not mutate the
reviewed worktree and does not repair findings.

Output starts with findings ordered by severity.  Every finding includes
`file:line`, evidence, the violated requirement or policy ID, and required
remediation.  There is no mandatory praise preamble and no dated anecdote in the
runtime prompt.

### Operator

The operator contract receives the task, implementer report, reviewer report,
and exact integration gates.  It verifies rather than trusts reports, returns
findings to an implementer, and dispositions the PR according to policy.  It
does not implement feature fixes inside the coordinating context.

### Prompt enforcement

`scripts/check-prompt-contract.py` parses structural fields and role boundaries.
It must not pin incidental sentences.  Mutation tests delete each required
section, remove a role boundary, and weaken a required output to prove the gate
fails semantically.

## Preflight and CI states

One command must not blur local and remote evidence.

- Local preflight: role/task/worktree/branch consistency, policy schema,
  generated blocks, local gates, staged change, and committed range.  No network.
- Ready gate: all local checks plus live PR identity, collision, and current CI
  requirements.  Remote failure is explicit and non-green.
- Integration gate: ready gate plus review disposition, commit trailers,
  waivers, and base freshness.
- CI: re-runs repository checks from a clean checkout and queries the current PR
  event rather than trusting committed remote snapshots.

Hooks remain convenience backstops.  They are never described as proof because
`--no-verify` can bypass them.

## Migration and cutover

PR #128 consumes `WAIVER-PR-SIZE-001` only for the one-time consolidation and
evidence-preserving archive migration. It expires after the migration window;
subsequent policy changes remain within the ordinary per-class budgets.
Pull-request CI supplies its exact event head to role discipline so a detached
clean checkout cannot misreport unmerged row commits as already landed on main.
Trailer validation also walks the exact event base-to-head range; it never
mistakes GitHub's synthetic checkout merge for contributor history.
The tree-scoped role suite uses that same range and exact pending-head evidence.

1. Inventory every active normative statement and map it to a `POL-*` rule,
   procedure, archived rationale, or deletion as duplicate.
2. Land `policy.csv`, schemas, render/check tooling, and mutation tests without
   changing project feature claims.
3. Render the compact `AGENTS.md`; consolidate procedures and repair links.
4. Land prompt contracts and their checker.
5. Land role/task/claim/PR/trailer/waiver enforcement.
6. Register the governance task for PR #128 and make the PR pass its own rules.
7. Enumerate existing open-PR migration waivers with short expirations.
8. Declare one cutover commit.  Historical commits remain grandfathered; every
   new commit and PR is strict.
9. Archive superseded policy and pre-cutover claim rationale.

No permanent legacy switch remains after cutover.  An open PR that does not
finish within its migration waiver must rebase and comply.

## Verification and mutation matrix

The new test suite must prove failure for at least:

| Surface | Required red mutation |
|---|---|
| policy CSV | missing column, duplicate/unknown ID, multiline cell, bad enum, unknown checker, over size/count budget |
| AGENTS index | stale generated T0, unknown link/rule, over 12 KiB, contradictory boot-order marker |
| procedure authority | normative paragraph without exactly one applicable rule ID, unknown rule ID, stale procedure back-reference |
| public documentation | missing STATUS/BENCHMARKS checkpoint, missing FEATURES support update, missing USAGE change, README churn without a headline trigger, stale NOW after live-state change, governance-only design misclassified as feature work |
| helper role | primary checkout, wrong branch, unknown task, marker-only success, missing worktree |
| live claim | missing PR, duplicate PR, wrong head, closed PR treated live, remote failure treated absent |
| helper queue | uncommitted spec, no failing gate mutation, undeclared hardware, unmet dependency, reserved row |
| PR/path size | oversized policy/checker change, material script change misclassified as exempt |
| trailers | substring-only legacy tag, missing protocol trailer, malformed AI declaration, missing assistance attribution |
| waivers | unknown/non-waivable rule, wildcard scope, expired/unused/duplicate waiver, missing evidence |
| prompts | missing envelope field, missing output field, removed stop condition, reviewer mutation replacing static review |
| migration | post-cutover legacy commit/PR, expired migration waiver, new report-only mode |

Integration tests use throwaway Git repositories and fixture PR payloads.  They
must not mutate the real checkout or require live GitHub access.  One authorized
online smoke test verifies the GitHub query against PR #128 before the PR is
marked ready.

Existing record, document, source, and unit gates remain green throughout.  A
policy-only change does not fabricate a feature checkpoint or benchmark result.

## Work breakdown for PR #128

The implementation plan may split commits, but each commit must satisfy the
same-change obligations applicable after its point in the migration.

1. Policy inventory and CSV schema.
2. Policy parser, renderer, authority checker, and mutation tests.
3. Compact root index and procedure consolidation/archive.
4. Prompt contracts and semantic checker.
5. Governance task registry and helper materialization.
6. Live PR claim and ready-queue enforcement.
7. Trailer, path-classification, PR-size, and waiver enforcement.
8. Purpose-specific public-document trigger enforcement.
9. Preflight/CI integration and cutover migration.
10. Full self-hosting verification, link audit, final compactness audit, and PR
   evidence.

PR #128 remains one reviewable PR as requested.  Reorder commits or trim
incidental cleanup if necessary, but do not split these acceptance criteria into
a series and do not leave prose claiming enforcement before its gate lands.

## Risks and decisions

- **Bootstrap recursion:** the new rules govern their own implementation.
  Resolve with ordered commits and a single explicit cutover, not a permanent
  bypass.
- **Remote dependence:** local work must remain possible, but remote absence
  cannot prove a claim.  Keep local and ready states separate.
- **CSV abuse:** long quoted prose would recreate the current problem.  Forbid
  multiline fields and enforce rule/file budgets.
- **False deduplication:** similar procedures may encode distinct constraints.
  Inventory by stable rule semantics before archiving text.
- **Historical link breakage:** archive with link repair and link checking in
  the same change.
- **Cross-agent capability differences:** the core prompt specifies outcomes
  and evidence, while adapters only describe tool invocation.
- **Waiver normalization:** a waiver is visible debt, not success.  Expiry and
  exact scope are mandatory.

## Deferred structured-record PR

The follow-up PR will separately design and migrate structured sources for:

- roadmap and area matrices;
- claims and dependencies;
- state and benchmark events;
- parity-ledger evidence with stable IDs instead of line anchors;
- `NOW.md` and public Markdown projections;
- era rollover and archival automation.

Its migration must preserve every evidence item, replace fragile line-number
anchors with stable IDs, generate human-readable projections, and prove
byte/semantic equivalence before deleting any legacy source.  None of those
record-format decisions are smuggled into PR #128.

## Acceptance criteria

PR #128 is complete when:

- one compact policy CSV is authoritative and all normative statements in the
  designated active policy surfaces map to it;
- `AGENTS.md` remains a compact index under its budget;
- active policy/procedure files are fewer and duplicates are archived;
- role, claim, PR, trailer, waiver, and prompt promises are executable;
- STATUS, BENCHMARKS, FEATURES, USAGE, README, and NOW update exactly on their
  defined triggers without duplicating one another;
- all new gates have red-first mutation evidence;
- existing project gates remain green;
- current open PR migration is explicit and expiring;
- PR #128 passes the new local, ready, and integration gates on its own head;
- the structured-record migration remains a named, separate follow-up.
