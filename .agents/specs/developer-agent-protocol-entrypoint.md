# Developer agent protocol entry point

User-directed 2026-08-07. Row:
`DOCS-AGENT-PROTOCOL-ENTRYPOINT`. Status: accepted design.

## Purpose

Give a new contributor one friendly, public place to learn what vllm.cpp is,
how to start an agent coding tool, and how to choose work without duplicating an
open effort or trusting stale roadmap text.

The entry point does not restate the project policy. `AGENTS.md` remains the
canonical index and `.agents/workflow.md` remains the operating manual.

## Audience and tone

The audience is a developer arriving from the repository landing page who wants
to contribute but has not learned the project's operator/helper vocabulary.
The page welcomes them before introducing process. It opens with "Here be
dragons. Welcome!", explains the project's ambition in a few sentences, and
presents the agent protocol as guidance rather than a test the developer must
memorize.

The project description stays aligned with `README.md`:

- vllm.cpp is a from-scratch C++20 implementation of vLLM aiming at 1:1
  behavior and feature parity without Python or PyTorch at inference time;
- it also carries useful capabilities beyond the vLLM baseline, including
  SGLang scheduling ideas, llama.cpp-style deployment, and text, image, video,
  and audio support in one engine;
- every architecture, model family, feature, and backend must be tested against
  its reference and benchmarked on the same workload.

The public page receives a humanizer pass after the factual draft. The pass
preserves every claim, code block, and link target, but may reshape the prose.
It uses plain technical language, adds no facts, and removes promotional filler,
vague attribution, em dashes, and en dashes.

## Public entry point

Create a concise root-level `CONTRIBUTING.md` and link it from the README's
documentation table. Its first action is a copyable instruction:

> Read `AGENTS.md` completely and follow it before doing any work. Start by
> running `scripts/agent-preflight.sh`.

The page briefly explains that the agent asks about the kind of work, selects
the appropriate operator/helper/read-only path, and requests machine-specific
settings only when needed. It links to `AGENTS.md` and
`.agents/workflow.md` for the binding detail.

## Finding work

The public page gives contributors this ordered intake:

1. Search open issues and pull requests for the topic and candidate row ID.
2. Read `.agents/NOW.md` for live claims and the current gate.
3. Run `scripts/ready-for-helper.py` to see rows that satisfy the helper-ready
   conditions.
4. Read the relevant roadmap and owning matrix row.
5. Inspect the current implementation, tests, and recorded evidence. Confirm
   that the described gap still exists at the current branch head.
6. Claim the row only after those checks show that the work remains open and is
   not already owned.

The page must not advertise an issue, roadmap row, or helper-queue result as
sufficient on its own. Repository state and code are authoritative for what is
already implemented; the canonical records explain what has been verified.

## Binding protocol change

Add a mandatory intake rule to `AGENTS.md` T0 and the session protocol in
`.agents/workflow.md`. Before starting implementation or claiming a row, every
agent must check:

- open issues and pull requests for duplicates and active ownership;
- `.agents/NOW.md`, the helper queue when applicable, the roadmap, the owning
  matrix row, and coordination state;
- the current code, tests, and relevant evidence anchors.

The agent records concrete current-code and test anchors in its spike or PR.
If the gap has already landed, is claimed, or no longer matches the record, the
agent stops and reconciles the task instead of starting duplicate work.

The binding obligation appears verbatim in `AGENTS.md` and
`.agents/workflow.md`. `scripts/check-protocol-consistency.py` owns the
constant and asserts both copies match. Its mutation suite proves that deleting
or changing either copy fails.

This is the strongest reliable offline enforcement for a read-before-work
obligation. The checker can prove that the policy is present and synchronized;
it cannot prove that a person understood a file. Network-dependent issue checks
must not make local work impossible. CI and review inspect the recorded intake
evidence instead.

## Pull request evidence

Extend `.github/pull_request_template.md` with a short "Before starting"
section that asks for:

- the issue/PR search performed and any linked issue or existing claim;
- the selected roadmap or matrix row and helper-queue result, when applicable;
- exact current-code and test anchors inspected before implementation.

The template accepts "no matching issue or PR" as an honest result. It does not
require inventing an issue for small governance or documentation work.

## Files and boundaries

Expected implementation files:

- `CONTRIBUTING.md` (new public entry point);
- `README.md` (add the contributor row and compact prose only inside the
  Documentation table and the canonical-record paragraph immediately below it;
  preserve every link and its meaning in that bounded region, make no other
  README edits, and finish at or below 29,900 characters);
- `AGENTS.md` and `.agents/workflow.md` (the synchronized intake obligation);
- `.github/pull_request_template.md` (recorded intake evidence);
- `scripts/check-protocol-consistency.py` and
  `tests/scripts/test_check_protocol_consistency.py` (drift enforcement and
  mutation proof);
- `scripts/check-readme-structure.py` and
  `tests/scripts/test_check_readme_structure.py` (contributor-link enforcement
  and mutation proof);
- `docs/STATUS.md`, `docs/BENCHMARKS.md`, `.agents/NOW.md`, and
  `.agents/state.md` (same-change checkpoint record).

No engine, model, kernel, benchmark number, capability state, or lifecycle row
changes. `docs/FEATURES.md` remains unchanged because the feature, model,
backend, and quantization surfaces do not move.

## Verification

The implementation is complete when:

- the new protocol-consistency mutation fails after either synchronized intake
  block is removed or changed, and passes for the committed copies;
- `python3 scripts/check-protocol-consistency.py` passes;
- `python3 tests/scripts/test_check_protocol_consistency.py -v` passes;
- all links in `CONTRIBUTING.md` and the new README link resolve locally;
- `scripts/agent-preflight.sh --staged` passes before the implementation
  commit;
- the branch-wide documentation checkpoint passes against `origin/main`.

## Rejected alternatives

**Public prose only.** It would welcome contributors but would leave the
read-before-work requirement outside the canonical policy and vulnerable to
drift.

**A network-backed intake receipt in `agent-role.py`.** It could prove that a
command ran, but not that the code was understood. It would also make role
declaration depend on GitHub availability. The synchronized policy block plus
reviewable PR evidence is stricter where the repository can prove facts and
remains usable offline.
