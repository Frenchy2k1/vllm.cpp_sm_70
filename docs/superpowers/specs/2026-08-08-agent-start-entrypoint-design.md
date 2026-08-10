# Agent session entrypoint design

**Date:** 2026-08-08
**Status:** Approved for implementation
**Scope:** Contributor onboarding and session routing only

## Problem

The repository already has the right low-level pieces for session onboarding:
`agent-role.py` declares roles, `agent-onboard.py` probes state, and
`agent-preflight.sh` runs the gates. The first minute still feels procedural.
An agent must assemble those pieces from Markdown, a new contributor meets role
vocabulary before learning why it exists, and the undeclared-role failure
duplicates instructions instead of presenting one intentional entrypoint.

The contributor experience should be welcoming without permanently spending
context on decoration. A first-time contributor should see a compact project
welcome and choose the shape of the work. An experienced contributor who has
already stated the intended role should go directly to the exact next command.
Every session, including a session whose worktree already carries a role,
should receive one state-aware route into preflight and the resume record.

## Goals

- Establish one mandatory first command for every agent session.
- Show a compact ASCII welcome only when the worktree is undeclared and the
  user has not already made their intent explicit.
- Explain helper, operator, and read-only in terms of contributor intent.
- Keep the welcome out of `AGENTS.md`, workflow prose, and other documents
  loaded into the agent's immediate context.
- Give the agent exact, state-aware next commands without performing hidden
  writes or role decisions.
- Preserve the existing ownership boundaries between role declaration, state
  probing, and preflight.

## Non-goals

- Replacing `agent-role.py`, `agent-onboard.py`, or `agent-preflight.sh`.
- Adding an interactive stdin prompt.
- Guessing a role, row, environment value, or headless mode.
- Changing helper worktree, pull-request, operator-lock, or gate semantics.
- Adding a harness-specific startup hook.

## Approaches considered

### 1. A state-aware standalone entrypoint — selected

Add `scripts/agent-start.py`. It consumes the existing onboarding probe,
renders the welcome when appropriate, and prints exact next actions. The agent
relays a delimited welcome block verbatim when directed.

This keeps presentation in executable source, produces consistent behavior
across agent harnesses, and leaves every state transition with its current
owner.

### 2. Treat terminal or tool output as the welcome

This avoids the relay instruction, but tool output is collapsed, styled, or
hidden differently by each harness. A contributor may never see it as the
agent's welcome.

### 3. Put the banner in agent Markdown

This makes the banner easy to reproduce but loads decoration into context on
every session, including sessions that do not need onboarding. It also invites
the prose and executable behavior to drift.

## Command contract

The canonical first command is:

```console
scripts/agent-start.py [--intent operator|helper|read-only] [--row ROW-ID] [--headless]
```

The agent runs it before preflight on every session. An explicit intent is
passed only when the user's opening request already makes the work shape clear.
`--headless` is passed only when the user explicitly declared an unattended
run. The command never infers either value.

`--row` is valid only with `--intent helper`; supplying it for operator or
read-only is an argument error. `--headless` without an explicit intent is also
an argument error because there is no role claim to receive the mode. When a
declared worktree is inspected, mode comes exclusively from its materialized
marker rather than from command-line overrides.

The entrypoint is non-interactive and instruction-only. It does not claim a
role, create a worktree, acquire a lock, edit `.env`, or run preflight. A
successful routing result exits zero even when more action is required.
Invalid arguments and failures that prevent truthful routing exit nonzero.

## State machine

### Declared worktree

The banner is suppressed. The output reports the inherited role, row when
applicable, mode, and worktree. If an explicit intent conflicts with the
inherited declaration, the entrypoint calls out the mismatch and gives the
exact re-declaration or escalation instruction; it never silently changes the
role.

With no conflict, the next-action block instructs the agent to:

1. confirm the inherited role fits the current request;
2. run `scripts/agent-preflight.sh`;
3. use the `NOW.md` surface printed by preflight;
4. read developer preferences when present; and
5. resume from the declared row's coordination and state anchors.

### Undeclared worktree with explicit intent

The banner is suppressed. The entrypoint emits the exact claim command:

- operator: `scripts/agent-role.py claim operator`;
- read-only: `scripts/agent-role.py claim read-only`;
- helper with a row: `scripts/agent-role.py claim helper --row ROW-ID`;
- helper without a row: identify or create the scoped row before claiming; the
  READY helper queue may be shown, but no row is selected automatically.

The action block then instructs the agent to rerun `agent-start.py` so routing
is re-derived from materialized state, followed by preflight.

An operator lock held by another worktree is reported as a blocker rather than
converted into a different role. The agent reports it and obtains direction.

**SUPERSEDED 2026-08-10 by issue #285** (`.agents/specs/operator-record.md`):
there is no lock and no blocker. A live coordinator in another worktree is
reported as status — `other coordinators: N recorded (claim is allowed)` — and
`claim operator` is offered as normal, because it is never refused.

### Undeclared worktree without explicit intent

The entrypoint emits the selected compact-frame welcome. The source constant,
not a Markdown copy, is the canonical rendering. It is ASCII-only, contains no
ANSI color, fits within 72 columns, and gives the three paths this wording:

- **Helper:** contribute one focused task that helps move the project forward.
- **Operator:** maintain the project or coordinate a long, multi-agent
  campaign.
- **Looking:** read, review, or ask questions without claiming project work.

The welcome is wrapped in stable `WELCOME: RELAY VERBATIM` delimiters. The
following action block instructs the agent to relay only that block verbatim,
ask what the contributor is here to do, invoke the matching role claimer,
rerun `agent-start.py`, and then run preflight.

## Output contract

Human-readable output has stable sections:

```text
--- WELCOME: RELAY VERBATIM ---   # present only for first-time routing
<source-owned compact frame and paths>
--- END WELCOME ---

--- AGENT NEXT ACTIONS ---        # always present
<state-aware numbered actions and exact commands>
--- END ACTIONS ---
```

The action block is written for the agent, while the welcome block is written
for the contributor. The separation prevents an agent from relaying internal
commands or paraphrasing the visual welcome.

Only status is printed for environment configuration. Values, paths, tokens,
and other machine-specific content are never echoed. Missing environment
values retain the existing just-in-time policy: they do not block startup and
are asked only when a later gate needs them.

## Component boundaries

| Component | Responsibility after this change |
|---|---|
| `scripts/agent-start.py` | First-command presentation and state-aware next-action routing |
| `scripts/agent-onboard.py` | Read-only role/environment/helper-queue state probe |
| `scripts/agent-role.py` | Role marker, helper row declaration, mode, and operator lock |
| `scripts/agent-preflight.sh` | Repository correctness gates and undeclared-role backstop |

`agent-start.py` consumes the probe's computation rather than reparsing record
files or duplicating role resolution. Preflight's undeclared-role failure stops
duplicating the interview and points to the canonical entrypoint.

## Failure handling

- An unreadable environment file or unavailable helper queue is reported
  honestly and does not become an invented empty state.
- A missing helper row produces a specific next action rather than a malformed
  claim command.
- A held operator lock reports the holder conflict and does not recommend an
  unauthorized fallback. (SUPERSEDED by #285: recorded coordinators are
  reported as status and refuse nothing.)
- A declared-role/explicit-intent mismatch is visible and never mutates state.
- Headless mode is propagated into the exact claim command only when explicitly
  supplied.
- Unexpected probe failures produce a concise error and a nonzero exit.

## Documentation and enforcement

`AGENTS.md` and `.agents/workflow.md` will name `agent-start.py` as the first
command of every session and describe the intent flag at a high level. They
will not contain the banner. The existing session-onboarding and
operator/helper protocol specs will be updated in the same change so prose and
behavior remain aligned.

`scripts/check-protocol-consistency.py` will assert that the canonical
entrypoint and ordering agree across the binding protocol surfaces. It will
not duplicate the banner text.

`agent-preflight.sh` remains the backstop: when a session is undeclared, it
fails with the exact instruction to run `scripts/agent-start.py`. CI and
scripted callers retain the existing explicit preflight escape behavior.

## Test strategy

Add `tests/scripts/test_agent_start.py` and register it with preflight. Tests
cover:

- declared operator, helper, and read-only routing;
- declared interactive and headless modes;
- undeclared routing with no intent, each explicit intent, and helper with and
  without a row;
- suppression of the welcome for every declared or explicit-intent path;
- the compact banner's ASCII-only and maximum-width guarantees;
- stable welcome/action delimiters and exact role-claim commands;
- inherited-role/explicit-intent conflicts;
- a live coordinator recorded by another worktree (a blocking lock until #285);
- unavailable helper queue and unreadable/incomplete environment states;
- absence of environment values and secrets in rendered output;
- nonzero exits only for invalid input or an inability to route truthfully;
- preflight pointing to `agent-start.py`; and
- protocol-consistency mutations proving that stale entrypoint prose fails.

## Acceptance criteria

1. Every binding session-start surface directs agents to `agent-start.py`
   before preflight.
2. A genuinely first-time, non-explicit session receives the compact welcome
   and a verbatim-relay instruction.
3. Declared sessions and sessions with explicit intent never render the
   welcome.
4. Every valid state receives exact, non-mutating next actions.
5. Existing role, lock, environment, helper queue, and preflight semantics are
   unchanged.
6. The banner exists only in executable/test source, not in immediately loaded
   agent Markdown; executable source remains the one canonical rendering and
   tests assert properties or reference that constant rather than copying it.
7. The focused unit, mutation, protocol-consistency, and preflight suites pass.
