# Workflow — session, coordination, and delivery

This is the operating method. [Policy](policy.csv) is the sole rule registry;
this file gives executable procedure without restating a second policy source.
Each controlled paragraph names exactly the rule it implements.

## Bootstrap and authority

<!-- policy-procedure:begin -->
[POL-AUTH-REGISTRY] Treat `.agents/policy.csv` as the complete rule inventory. Use this procedure for method and the task records for facts; neither can add or weaken a registry rule.

[POL-AUTH-PRECEDENCE] Resolve conflicts in this order: repository `AGENTS.md`, the applicable registry row, that row's procedure, then task evidence. Report an unresolved contradiction before changing state.

<!-- role-interview:begin -->
[POL-BOOT-ROLE] Begin with `scripts/agent-role.py show`. If no valid role exists, ask what work the developer intends and run `scripts/agent-role.py claim operator` for a multi-step integration campaign, `scripts/agent-role.py claim helper --row <ID>` for one scoped task, or `scripts/agent-role.py claim read-only` for inspection. Add `--headless` only when the developer explicitly declares an unattended run; never infer it.
<!-- role-interview:end -->

[POL-BOOT-NOW] After role resolution, read `.agents/NOW.md` as the one-read live snapshot; consult the append-only state tail only when the task needs deeper history.

[POL-BOOT-TASK] After NOW, read only the claimed spec, owning matrix/roadmap row, coordination entry, linked evidence, and the procedures selected by applicable policy rows.

[POL-INTAKE-REVALIDATE] Before claiming or implementing, search current code and tests plus open issues, pull requests, NOW, coordination, and owning rows. Record exact anchors in the spike. If the gap already landed, is claimed, or no longer matches its record, stop implementation and reconcile the task first.

[POL-CONFIG-SHARED] Resolve `.env` and `.agents/developer-preferences.md` through the shared checkout configuration described by their tracked examples; do not manufacture per-worktree alternatives.

[POL-CONFIG-JIT] Request only a missing environment or permission value needed by the current gate. An empty value remains unavailable and its dependent gate remains `PENDING`.

[POL-CONFIG-NO-WEAKEN] Apply preferences only to operational choices such as remote use, downloads, hosts, services, Git integration, or delegation; never use preferences to reduce correctness, evidence, attribution, lifecycle, testing, or documentation obligations.

[POL-ROLE-DECLARED] Keep role state consistent with the worktree. Read-only performs no writes; a helper changes only its task branch/worktree; the operator owns integration and shared resources.

[POL-HELPER-TASK] A helper claim uses one known matrix row ID or one explicit governance task ID. Reject invented, duplicate, closed, or ambiguous identifiers.

[POL-HELPER-WORKTREE] Materialize helper work in the exact linked worktree and `row/<ID>` branch created for the claim. Do not reuse a foreign worktree, branch, marker, or task identity.

[POL-HELPER-PR] When remote operations are authorized, open a draft PR at helper start and use its exact repository, branch, and head as the remote claim. Without authority, report the remote step as pending.

[POL-REMOTE-UNKNOWN] If the remote cannot be queried or its identity cannot be established, report `REMOTE_UNVERIFIED`; unknown state is neither absence nor success and authorizes no destructive cleanup.

[POL-OPERATOR-BOUNDARY] The operator coordinates feature implementation through bounded implementer and reviewer tasks, owns main integration and the GPU, and avoids writing an implementation that should be independently reviewed.

[POL-PREFLIGHT] Run network-independent `scripts/agent-preflight.sh` at session start and its staged form before committing. Before remote handoff run `scripts/agent-ready.py`; before integration run `scripts/agent-integration.py --base origin/main`. A remote failure is never converted to local success. Before pushing, rerun the applicable local gate and chain success directly to the exact-SHA push; hooks are only bypassable convenience backstops, never proof.
<!-- policy-procedure:end -->

## Intake, claims, and readiness

<!-- policy-procedure:begin -->
[POL-SPIKE-FIRST] Before a row becomes `READY` or `ACTIVE`, verify the gap against current code, tests, issues, PRs, NOW, coordination, and owning records, then commit `.agents/specs/<slug>.md` with scope, upstream anchors, design, risks, tests, gates, evidence, and stop conditions.

[POL-READY-SPEC] Advertise helper work only when the committed spec is reachable from the advertised base and its declared executable gate has a demonstrated failing mutation that the unmodified tree passes.

[POL-READY-HARDWARE] Mark readiness with either a CPU-runnable gate or the exact required hardware, architecture, toolchain, host authorization, and contention protocol.

[POL-READY-DEPS] Parse and prove dependencies complete, reject duplicate or unknown rows, and establish that no local or remote live claim already owns the task before offering it.

[POL-ROW-SYNC] Whenever lifecycle state changes, update the portfolio roadmap row and its owning area matrix row in the same change with exact source, code, test, evidence, spec, state, and owner anchors.

[POL-KEYED-MERGE] Reconcile keyed documents by taking the target branch version wholesale and reapplying the scoped edit. Verify unrelated keys byte-for-byte; union-append only append-only logs.

[POL-EVIDENCE-PRESERVE] Compact by moving superseded detail to a clearly named file under `.agents/completed/` while retaining links and provenance. Never discard evidence to reduce context.
<!-- policy-procedure:end -->

## Implementation and delegated prompts

<!-- policy-procedure:begin -->
[POL-PROMPT-ENVELOPE] Every delegated task supplies the goal, relevant context, exact scope and exclusions, constraints, done-when conditions, required evidence, allowed authority, output contract, and stop conditions. Missing binding context produces `NEEDS_CONTEXT`.

[POL-PROMPT-BOUNDARIES] Use the versioned tool-neutral contracts in `.agents/prompts/`; keep method, output fields, authority, and stop rows machine-parseable, bounded, and independent of a named agent vendor or runtime.

[POL-CHECKER-CHANGE] A checker-semantics change uses a governance task naming every affected policy ID and includes a red-before test or mutation, green-after evidence, and synchronized registry and procedure changes.

[POL-NO-REPORT-ONLY] For every applicable rule, record exactly one result: satisfied, narrowly waived, pending external authority/resource, or failing. A permanent report-only result never completes a task.

[POL-WAIVER-EXACT] A waiver identifies one rule and one concrete task, PR, commit, path, gate, or hardware target; it names an owner, reason, evidence, and future expiry, and is used only when that rule's waiver class permits it.

[POL-PATH-CLASSIFICATION] Classify policy, checker, documentation, script, test, CI, generated, binary, and product paths explicitly when computing scope. Never hide mutable files behind a blanket directory exclusion.

[POL-PR-SIZE] Keep every explicit path class within its reviewed budget. Split unrelated work or cite one valid exact waiver before marking a PR ready.

[POL-ONE-SURFACE] A capability reaches `DONE` only when `include/vllm.h` exposes it through the stable C ABI, implementation state flows through the shared library entry point, and CLI/server examples are thin clients of that same surface. Run `scripts/check-surface-coverage.py` and its mutation suite; a feature reachable only through example internals remains open.
<!-- policy-procedure:end -->

Implementation starts from the committed spike. Write or port the smallest
test that fails for the intended reason, capture the red result, implement the
minimal complete behavior, and obtain green focused tests before broader
validation. The verification and porting procedures define their respective
gates.

## Review and disposition

<!-- policy-procedure:begin -->
<!-- orchestration-loop:begin -->
[POL-REVIEW-FRESH] After focused and full gates pass on an immutable head from a fresh [implementer](prompts/implementer.md), dispatch a fresh [reviewer](prompts/reviewer.md), never the agent that wrote the code, to perform both static inspection and targeted scratch mutations of the claimed guarantees—mutate, not read. Review output identifies commands, mutations, findings, and the reviewed SHA.

[POL-REVIEW-NO-REPAIR] A coordinating/operator session never repairs a reviewer finding: never fix findings yourself. Send the bounded finding and evidence to a fresh implementer, rerun focused and full gates, then use a fresh scoped reviewer.

[POL-OPERATOR-VERIFY] The operator independently checks the immutable head: run the row's gate yourself. Implementer or reviewer summaries are evidence inputs, not gate results.

[POL-PR-DISPOSITION] After independent verification and clean review, merge the PR in that session when authorized. Close obsolete or superseded PRs with a recorded reason; otherwise record the exact external blocker. Only an explicitly declared headless run may record `READY-TO-MERGE at <sha>` for later disposition.
<!-- orchestration-loop:end -->

[POL-PR-REQUIRED] Deliver feature, policy, checker, documentation, or record changes through a PR unless one exact unexpired emergency waiver authorizes the named change.
<!-- policy-procedure:end -->

## Commit, documents, and handoff

<!-- policy-procedure:begin -->
[POL-COMMIT-TRAILERS] Every post-cutover commit uses Git-parsed `Following-Agents-Protocol: true`; assisted commits also use `AI-Assisted: true` and a syntactically valid `Assisted-by:` trailer.

[POL-AI-ATTRIBUTION] Attribute assistance with `Assisted-by: AGENT:MODEL [TOOL]`. AI tools never add `Signed-off-by` or `Co-Authored-By`; the human contributor remains responsible for review and understanding.

[POL-DOC-STATUS] Update `docs/STATUS.md` at every feature or iteration checkpoint so each capability reports the current accepted, pending, failed, or blocked state with anchors.

[POL-DOC-BENCHMARKS] Update the keyed rows in `docs/BENCHMARKS.md` at every feature or iteration checkpoint, including accepted and current pending/failed/void results plus reproducible recipes.

[POL-DOC-FEATURES] Update keyed `docs/FEATURES.md` rows whenever a feature, model, backend, or quantization surface changes; keep forensic detail in the evidence record.

[POL-DOC-USAGE] Update `docs/USAGE.md` when commands, C API, configuration, installation, or user workflows change, with runnable examples that use the public surface.

[POL-DOC-README] Change `README.md` only for a user-visible landing-page headline, positioning, primary quick start, or top-level capability shift; routine checkpoints belong in purpose-specific docs.

[POL-NOW-COUPLING] Treat `.agents/NOW.md` as a bounded live snapshot, not a log. Rewrite it in the same change as every `.agents/state.md` append.

[POL-STATE-ORDER] Append state entries below `<!-- state-order:enforced-below -->` with a sortable date anchor in oldest-to-newest order; repair interleaving with `scripts/sort-state-tail.py --apply`.
<!-- policy-procedure:end -->

An unfinished-work handoff records the immutable head, live claim, exact source
and evidence roots, commands and results, prohibitions, blocker, and first
resume command in state/coordination, with NOW refreshed when state changes.
