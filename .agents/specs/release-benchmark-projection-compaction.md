# Release benchmark projection compaction

Identity: `ENG-RELEASE-WINDOWS`

Issue: [#475](https://github.com/mudler/vllm.cpp/issues/475)

Delivery PR: [#446](https://github.com/mudler/vllm.cpp/pull/446)

Parent specification: [windows-binary-release.md](windows-binary-release.md)

Status: `ACTIVE`. The developer approved this bounded repair on 2026-08-12.
It starts from exact PR head
`7eeeb56f433310f250885b661b9bfbbca14f8068` and evaluates integration against
`origin/main` commit `bbc482a2`.

## Scope

Compact only the two adjacent `ROAD-V1-RELEASE` rows projected into the
`docs/BENCHMARKS.md` at-a-glance table. Preserve every current release fact:

- `v0.0.2` shipped eight primary archive/checksum/provenance triplets plus two
  indexes, totaling 26 assets, from source SHA
  `7020de93652ca920424a10ac5255b34810dd2f24` in run `31466516224`;
- Windows W14-W16 are implemented;
- native hosted Windows gates, matching-hardware evidence, the merged-SHA
  ten-tuple dry run, `v0.0.3-pre.1` publication, and the 32-asset audit remain
  pending;
- W12 remains optional/non-primary.

Do not compact another benchmark row, move forensic content from another row,
raise the 45,000-character budget, weaken `check-public-doc-tables.py`, or
change any release workflow, artifact, lifecycle, or publication state.

## Observed baseline

At the pinned release head, `docs/BENCHMARKS.md` is 44,859 characters. At
`bbc482a2` it is 44,942 characters. `git merge-tree --write-tree bbc482a2 HEAD`
merges this page cleanly but produces a 45,100-character result, which fails
the unchanged 45,000-character scoreboard budget. The PR-owned delta in this
page is confined to the two adjacent release rows at the top-level scoreboard.

## Design

Use one keyed `Binary release` table row instead of separately projecting the
matrix and delivery topology. Its workload cell carries the shipped v0.0.2
inventory and provenance; its headline carries implemented Windows scope and
every pending gate; its token cell records W12 optionality. This removes the
duplicated release subject and topology phrasing while leaving the detailed
release design and forensic evidence in the owning specifications.

Alternatives rejected:

1. Raising or adding slack to the global page cap would weaken the gate and
   preserve the merge-coupling defect.
2. Trimming unrelated benchmark rows would make #475 evict facts owned by
   concurrent work.
3. Shortening the two rows independently retains duplicate projection shape
   and gives less deterministic headroom than one complete keyed row.

## Tests and gates

Add a focused real-file contract in the existing public-document suite. It
must isolate the at-a-glance release projection, require exactly one release
row, independently require every fact listed in Scope, and reproduce the
integration budget by applying the release projection to `bbc482a2`'s
`docs/BENCHMARKS.md`. The pinned two-row form must fail before the document is
changed. Mutating any required fact or restoring the second release row must
fail.

After GREEN, run the focused test, the complete public-document suite, the
direct public-document checker, full unstaged/staged/post-commit preflight, and
the exact PR range gate from `a170c81c9b73de084cc5db7f6f1d37a19664d91f`
to the immutable candidate. Also reproduce the merge-tree result against
`bbc482a2` and record its exact character count.

## Risks and stop conditions

The primary risk is satisfying the character budget by erasing a state or
provenance fact. The focused contract therefore checks the release projection
semantically before checking its merged size. Stop with `NEEDS_DECISION` if
the merge cannot fit without changing another row, weakening the checker, or
changing a release claim. Stop with `NEEDS_CONTEXT` if `bbc482a2` no longer
resolves or the page no longer merges cleanly enough to isolate this projection.

## Outcome

The pinned two-row projection failed the focused contract with
`expected one keyed Binary release row, found 2`; applying that projection to
`bbc482a2` reproduced the 45,100-character merge result. The implementation
replaces only those two rows with one keyed release row. The branch page is
44,579 characters and the projected main merge is 44,820, leaving 180
characters below the unchanged 45,000-character cap.

The focused contract independently requires all 13 release fact anchors from
Scope, rejects deletion of each one, rejects restoration of a second release
row, and validates the projected merged page with the production public-doc
checker. The two existing release-state checkers initially rejected their
old exact wording, providing a second RED boundary. Their mutation inventories
and expected anchors now require the compact row and pending facts; neither
budget nor lifecycle assertion was relaxed. Full immutable verification is
recorded on the implementation commit.

## Now

`ACTIVE`; W14-W16 implementation remains assembled in PR #446. This repair
only restores integration headroom for its public benchmark projection; native
hosted gates and the release-publication sequence remain pending.
