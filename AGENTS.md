# AGENTS.md — repository bootstrap

This is the small, stable entry point for work in `vllm.cpp`. Repository rules
live only in [`.agents/policy.csv`](.agents/policy.csv). Procedure documents
explain how to satisfy those rules; records and matrices contain project state
and evidence. If prose conflicts with the registry, the registry wins.

## Boot order

<!-- policy-boot:begin -->
1. Resolve the worktree role with `python3 scripts/agent-role.py show`; claim the developer-selected role only when no valid role exists.
2. Resolve `.env` and `.agents/developer-preferences.md` from the shared checkout, requesting only values required by the current task.
3. Read `.agents/NOW.md` for the live snapshot.
4. Read `.agents/policy.csv`, then the procedure named by each applicable rule.
5. Read only the claimed task's spec, owning row, evidence, and coordination entry.
<!-- policy-boot:end -->

Do not infer a role, environment, host, permission, or developer preference.
When configuration is absent, start from [`.env.example`](.env.example) and
[`.agents/developer-preferences.example.md`](.agents/developer-preferences.example.md),
ask only for the value required by the next gate, and leave unavailable gates
`PENDING`. Preferences control operations, never project truth.

## Compact T0

The following block is generated from the authoritative registry and checked
byte-for-byte. Do not edit it independently.

<!-- policy-t0:begin -->
- `POL-AUTH-REGISTRY` — Use policy.csv as the sole repository-policy authority.
- `POL-ROLE-DECLARED` — Declare operator helper or read-only and keep the role state consistent with the worktree.
- `POL-SPIKE-FIRST` — Commit a complete spike spec before a row enters READY or ACTIVE.
- `POL-MIRROR-VLLM` — Mirror every applicable vLLM mode instead of inventing product behavior.
- `POL-PORT-TESTS` — Port the applicable upstream tests in the same change.
- `POL-CORRECTNESS-GATE` — Pass the declared token-exact or ratified distributional correctness gate before performance acceptance.
- `POL-PREFLIGHT` — Run the applicable preflight and prevent a failed gate from being followed by a push.
- `POL-REVIEW-FRESH` — Use a fresh reviewer that performs static review and targeted scratch mutation.
- `POL-REVIEW-NO-REPAIR` — Return findings to a fresh implementer and do not repair them in the coordinating session.
- `POL-OPERATOR-VERIFY` — Run the claimed gate instead of trusting an implementer report.
- `POL-ONE-SURFACE` — Expose every shipped capability through include/vllm.h and keep examples as thin clients of the same library surface.
- `POL-EVIDENCE-PRESERVE` — Move evidence without deleting it.
- `POL-PR-DISPOSITION` — Merge a verified PR in-session or close an obsolete PR with the reason recorded.
<!-- policy-t0:end -->

## Procedure routing

| Task | Read |
|---|---|
| session, role, claims, implementation, prompts, records, public docs, handoff | [`.agents/workflow.md`](.agents/workflow.md) |
| tests, gates, review, mutations, correctness, benchmarks, evidence | [`.agents/verification.md`](.agents/verification.md) |
| vLLM mirroring, upstream sync, model seams, test ports, parity inventory | [`.agents/porting.md`](.agents/porting.md) |

Task-specific technical protocols remain authoritative evidence for their
domain, including [parity lever analysis](.agents/parity-lever-protocol.md),
[upstream synchronization](.agents/upstream-sync.md), and backend or environment
records linked by the selected task. They do not create repository rules.

## Execution method

Every implementation follows the same lifecycle: verify the gap; commit a
spike/spec; implement from that spec with a red test before the fix; run the
focused gate and then the full gate; send the immutable result to a fresh
reviewer for static and targeted mutation review; return findings to a fresh
implementer; have the operator independently rerun the declared gate; then
merge, close, or record a precise external blocker. See the routed procedures
for the binding details and policy IDs.

Delegated prompts use the tracked, tool-neutral contracts in
[`.agents/prompts/`](.agents/prompts/): goal, context, constraints, done-when,
evidence, authority, output, and stop conditions are explicit. An agent lacking
required context returns `NEEDS_CONTEXT` rather than guessing.

## Public and record surfaces

The human-facing documents have distinct purposes:

| Surface | Purpose |
|---|---|
| `docs/STATUS.md` | capability status; update at every feature/iteration checkpoint |
| `docs/BENCHMARKS.md` | accepted and pending measurements; update at every feature/iteration checkpoint |
| `docs/FEATURES.md` | feature, model, backend, and quantization surface changes |
| `docs/USAGE.md` | commands, API, configuration, installation, and user workflows |
| `README.md` | landing-page and user-visible headline changes only |
| `.agents/NOW.md` | live snapshot; refresh in the same change as a state-log append |

Keyed records are updated in place. Append-only evidence remains append-only.
During concurrent reconciliation, take the current target branch version of a
keyed record and reapply the scoped edit; never accept an automatic three-way
combination. Historical policy and evidence are archived under
`.agents/completed/`, not deleted.

## Essential commands

```sh
python3 scripts/agent-role.py show
scripts/agent-preflight.sh
scripts/agent-preflight.sh --staged
python3 scripts/check-policy.py
python3 scripts/ready-for-helper.py
python3 scripts/agent-ready.py
python3 scripts/agent-integration.py --base origin/main
```

Before any push, run the applicable gate and chain that successful command to
the exact SHA push. Never push, merge, manage services, use external compute,
or download large assets without the authority recorded in developer
preferences or provided by the developer for the task.

## Commit and handoff

Post-cutover commits carry Git-parsed `Following-Agents-Protocol: true` and,
when assisted, `AI-Assisted: true` plus `Assisted-by:`. AI tools never add
`Signed-off-by` or `Co-Authored-By`. The human submitter owns and reviews the
change.

For unfinished work, append an exact dated checkpoint below the enforced marker
in [`.agents/state.md`](.agents/state.md), refresh [`.agents/NOW.md`](.agents/NOW.md)
in the same change, and update the live claim in
[`.agents/coordination.md`](.agents/coordination.md). Record the immutable head,
exact gates run, evidence roots, prohibitions, blocker, and first resume
command. Routine discussion and Git housekeeping do not create state entries.
