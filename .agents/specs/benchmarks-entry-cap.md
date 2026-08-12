# The BENCHMARKS cap is a file cap, and a file cap is a lock

Issue: [#460](https://github.com/mudler/vllm.cpp/issues/460).
Row: `ENG-RECORD-CONFLICT-SURFACES`.
Measured against `origin/main` `918c568a` on 2026-08-12.

`docs/BENCHMARKS.md` is gated by a 45,000-character budget on the whole file.
The page measures 44,795 characters, so 205 characters are free. Adding a
measurement row therefore means deleting somebody else's row, and the documented
way to delete one, moving it byte-for-byte into `.agents/benchmark-record.md`,
is broken for exactly the rows that carry evidence links.

This is the same defect this row already retired twice on 2026-08-11: the
`MAX_CHARS` budget in `scripts/check-now-current.py` and the `chars` key of
`STATUS_RATCHET` in `scripts/check-public-doc-tables.py`, both removed by
`87308dea` under #364 with the reasoning recorded in place. The scoreboard's own
`max_chars` was left standing in that pass. It is now the binding constraint on
every remaining roadmap measurement.

AGENTS.md, Records, states the rule this spec applies: **cap the entry, never
the file**, and **a gate is what usually creates the lock: if a checker requires
every change to touch a shared file, that is the defect**.

## Scope

**In scope.**

1. `max_chars` on `PageRules` in `scripts/check-public-doc-tables.py`, which
   applies to `docs/BENCHMARKS.md` (45,000) and `docs/FEATURES.md` (30,000).
2. The entry-scoped rules that take over its obligation: a per-row character
   cap, and a regrowth guard on per-attempt headings at any depth.
3. `check_links` in `scripts/check-agent-record.py`, so the archive path the
   scoreboard points at works for a row that carries a relative link.
4. The mutation suites `tests/scripts/test_check_public_doc_tables.py` and
   `tests/scripts/test_agent_record.py`.
5. One new row on `docs/BENCHMARKS.md`, the owed 35B canonical regrid named by
   #481, added with nothing evicted, as the acceptance demonstration.

**Out of scope, deliberately.** The `STATUS_RATCHET` keys kept by #364; the
required-section, canonical-section, prose-paragraph, paragraph-length and
cell-length rules, all of which are kept and none of which are widened; the
content of any existing row; `scripts/roll-benchmark-record.py`'s move logic;
and rebuilding the public scoreboard as a derived or globbed surface, which is
argued against in Design and deferred in Work breakdown.

**Not a correctness change.** No product source, kernel, ABI or model path
moves. Nothing measured changes value.

## Upstream chain

**No vLLM counterpart. This is project infrastructure.** vLLM has no equivalent
of `docs/BENCHMARKS.md`, of `.agents/benchmark-record.md`, or of the checker
suite that gates them: they exist to serve this project's protocol, which vLLM
does not run. The authority for this change is AGENTS.md, Records, and the
precedent set by `87308dea` (#364) on the two sibling budgets.

## Our baseline

**The page has had no usable headroom for 25 commits.** Free characters against
the 45,000 cap at each of the last 25 commits that touched `docs/BENCHMARKS.md`,
with the page's table-row count:

| commit | rows | chars | free | subject |
|---|---:|---:|---:|---|
| `918c568a` | 162 | 44,795 | 205 | measure(SPEC-DSPARK) fibonacci gap |
| `523b8a6f` | 162 | 44,826 | 174 | measure(SPEC-DSPARK) 5-rep interleaved |
| `887e04ff` | 163 | 44,579 | 421 | **docs(release): compact benchmark projection (#475)** |
| `1c9dbe08` | 163 | 44,931 | 69 | perf(SPEC-DSPARK) sync-free Markov chain |
| `bbc482a2` | 163 | 44,942 | 58 | merge: Whisper encoder FA-2 |
| `c5615cfe` | 164 | 44,692 | 308 | perf(SPEC-DSPARK) speculative verify |
| `4112ac8c` | 165 | 44,968 | 32 | fix(mm-speed) review findings |
| `425abf7c` | 163 | 44,936 | 64 | bench(cpu) x86_64 floor |
| `93613baa` | 165 | 44,964 | 36 | **docs(benchmarks): trim the Voxtral encoder row back inside** |
| `04b2b9fa` | 165 | **45,007** | **-7** | **merge: origin/main into row/MM-SPEED-ENC-FA2** |

Three facts follow, and each of them is the thing AGENTS.md names.

**The success mode is unsafe.** `04b2b9fa` is a clean automatic merge that
landed the page at 45,007 characters, 7 over the cap. Two PRs each paid for
their row by evicting a different one; the three-way merge applied both
additions and neither eviction cancelled the other. That is verbatim the
corollary in Records: "merging two such edits cleanly is worse than
conflicting". It has already happened here, in the tree, not in theory.

**The eviction is real and it is winning.** Row count fell from 165 to 162 over
these 25 commits while the project gained measurements. Two commits,
`93613baa` and `887e04ff`, exist for no purpose but to pay rent: their subjects
are "trim the Voxtral encoder row back inside" and "compact benchmark
projection".

**The payment mechanism does not work for the rows that carry evidence.**
`check_links` (`scripts/check-agent-record.py:599-610`) runs `LINK_RE.findall`
over the raw file with no fenced-span stripping and resolves every hit from
`source.parent`. Three of the 162 rows carry a `docs/`-relative link
(`bench-evidence/qwen35-4b-sm120-main-20260807.md`,
`bench-evidence/rpi5-a76-q8-dot-20260806.md`,
`bench-evidence/rpi5-a76-llamacpp-20260806.md`), and none of them can be moved
into `.agents/benchmark-record.md` byte-for-byte: the target resolves from
`.agents/` and dangles. #433 hit this and had to archive a shorter link-free row
instead. The subset of payable rows shrinks every time one is spent.

**What the cap is actually still catching.** `_h2_headers` matches `## ` only,
so `### ` subsections are ungoverned by the canonical-section allowlist. The
live page carries six of them. An appended per-attempt `### ` section is
rejected today by nothing except the character budget. This is the one real
obligation `max_chars` still discharges, and it is the one the replacement has
to pick up.

## Port map

**No upstream file to port from.** The local anchors this change edits:

| Anchor | What changes |
|---|---|
| `scripts/check-public-doc-tables.py` | `max_chars` removed from `PageRules`; `MAX_ROW_CHARS` and `DATED_HEADING_RE` added; `page_errors` gains the two entry-scoped checks |
| `scripts/check-agent-record.py` | `check_links` gains `extract_links`, a pure fenced/inline-code-aware link scanner; `link_base` becomes `link_bases` |
| `tests/scripts/test_check_public_doc_tables.py` | three `max_chars` tests replaced by entry-cap, regrowth and no-eviction tests |
| `tests/scripts/test_agent_record.py` | new `LinkExtractionTests` |
| `docs/BENCHMARKS.md` | one row added, nothing removed |

## Design

**Remove the file cap. Relocate its obligation to two entry-scoped rules.**

*1. `max_chars` is deleted from `PageRules`, with the reason recorded in place.*
A byte budget on a shared page is a lock by construction: every addition is a
read-modify-write of one global, the conflict is the lucky outcome, and the
clean merge is the unsafe one. `87308dea` removed the two sibling budgets on
exactly this argument; this is the third, left behind in that pass.

*2. `MAX_ROW_CHARS = 600` caps one table row.* This is the literal "cap the
entry": a measurement's cost is bounded locally, by the author of that
measurement, and never by deleting a row somebody else owns. It is the same
shape as `MAX_ENTRY_CHARS` in `check-now-current.py`. 600 is set from the live
pages: the longest row on `docs/BENCHMARKS.md` is 520 characters and on
`docs/FEATURES.md` 580. It is a genuine constraint, and a tighter one than what
it joins: `MAX_CELL_CHARS = 220` alone permits a five-column row of 1,100
characters.

*3. `DATED_HEADING_RE` rejects a per-attempt heading at any depth.* This is the
regrowth guard, the same shape as the `ROW_TABLE_LINE` guard `87308dea` added to
`check-now-current.py`, and it closes the `### ` hole the character cap was
covering. The shape is measured, not invented: of the **301 sections already
rolled into `.agents/benchmark-record.md`, 278 name a date in their heading**
(`2026-08-07`, `2026-07-31`, ...), which is what a per-attempt entry looks like
here. **Zero of the 16 live headings on `docs/BENCHMARKS.md` and zero of the 16
on `docs/FEATURES.md` name a date.** So the guard fires on the first appended
checkpoint section, at `##` or `###`, and never on a subject section. This is
strictly tighter than the status quo, which checks `##` only.

*4. `check_links` stops validating text that is not a link, and the archive
resolves what it archived.* Two changes, both narrow:

- Fenced code blocks and inline code spans are stripped before `LINK_RE`. A
  target inside a fence is not a link: CommonMark renders it as literal text, no
  reader can follow it, and there is nothing for the checker's rule ("every link
  resolves") to be about. Today the checker forbids any document in the tree
  from *showing* a link in sample output, which is why #460's own reproduction
  is a fence.
- `.agents/benchmark-record.md` resolves a target from `docs/` as well as from
  `.agents/`. It is the declared archive of `docs/BENCHMARKS.md`, so content
  moved into it verbatim was written against `docs/`. `link_base` already
  carries one such special case for migrated legacy payloads; this generalises
  it to `link_bases`, a tuple, and a target still has to exist under one of
  them. Fence-stripping alone does not cover this case, because
  `roll-benchmark-record.py` moves sections as live markdown, not fenced.

**Why not (a) alone, the `check_links` fix.** It unblocks payment and leaves the
ratchet. Every measurement would still evict a row somebody else owns, and the
clean merge of two such payments would still land the page over budget, as
`04b2b9fa` did. It treats the symptom the issue was filed from and not the
defect the issue describes.

**Why not (b), per-row files or a derived page.** The three admissible shapes in
Records govern *record* surfaces. `docs/BENCHMARKS.md` is not one: AGENTS.md,
Public documents, defines it as a projection whose purpose is to be one readable
page a user reaches from the README badge. "Derived at read time, so nobody
writes it" removes the lock only when the rendered artifact is not committed,
and GitHub renders committed markdown with no build step, so the generated page
would still be committed and still be a file every measurement PR writes. The
lock would move from the author to the generator, not die. It is also not what
is blocking: the binding pain today is the *cap*, not conflicts. Deferred as W4
with the condition that would justify it.

**Why not simply raise the cap.** Prohibited by the task, and the checker's own
comment already records why it does not work: the previous occupant of this line
"answered it by adding slack to the constant, which only postponed it to the
next cadence of parallel work".

## Tests to port

**No upstream tests exist.** These are written against this project's checkers.

`tests/scripts/test_check_public_doc_tables.py`:

- `test_a_new_row_costs_no_eviction`: a page sized just under the retired
  45,000 budget gains one row and stays valid. RED on BASE (over budget),
  green on HEAD. This is the acceptance test.
- `test_oversized_row_fails`: a single row past `MAX_ROW_CHARS` is rejected.
  RED on BASE, which has no row cap.
- `test_row_cap_is_tighter_than_the_cell_cap`: a row of legal cells whose sum is
  illegal is rejected, so the entry cap is not subsumed by `MAX_CELL_CHARS`.
- `test_dated_h2_is_rejected` and `test_dated_h3_is_rejected`: an appended
  per-attempt section fails at the first one, at either depth. RED on BASE for
  `###`, which no rule covered.
- `test_subject_headings_are_allowed`: a genuinely new subject subsection
  passes, so the guard is not a section freeze.
- `test_the_live_pages_carry_no_dated_heading`: the two shipped pages satisfy
  the new guard.
- Replaced: `test_oversized_page_fails`,
  `test_release_projection_fits_after_current_main_merge`'s `max_chars`
  assertion, `test_the_two_pages_have_distinct_budgets`'s `max_chars`
  comparison. Each is replaced by an assertion on the rule that took the
  obligation over, never deleted outright.

`tests/scripts/test_agent_record.py`, new `LinkExtractionTests`:

- `test_fenced_link_is_not_extracted` and `test_inline_code_link_is_not_extracted`:
  RED on BASE, which extracts both.
- `test_live_link_is_still_extracted` and
  `test_link_after_a_closed_fence_is_still_extracted`: the narrowing does not
  swallow real links.
- `test_archived_docs_relative_link_resolves`: the #460 reproduction, moved into
  the record as live markdown, resolves.
- `test_the_tree_has_no_dangling_link`: the whole-tree run stays green.

## Gates

1. `python3 -m pytest tests/scripts/ -q`, unbounded, full run.
2. `python3 scripts/check-public-doc-tables.py`.
3. `python3 scripts/check-agent-record.py`.
4. `scripts/agent-preflight.sh --staged`.
5. `python3 scripts/check-pr-size.py` red-before/green-after harness on both
   changed checkers: it reruns each HEAD test module against the BASE checker in
   an isolated worktree and refuses the PR unless BASE goes red.
6. Acceptance: `docs/BENCHMARKS.md` carries one more row than `918c568a` and no
   fewer, and every checker is green.

**No GPU. Nothing here measures.**

## Dependencies

- #364 / `87308dea`, which set the precedent and removed the two sibling
  budgets, is on `main`.
- #481 is open and edits `docs/BENCHMARKS.md`. The row added here is additive
  and placed in `## Open gaps`; a conflict resolves by taking the target
  branch's page wholesale and reapplying the one added line, per Records.
- Nothing else blocks.

## Work breakdown

| W | Work | State |
|---|---|---|
| W1 | This spec, committed alone | this commit |
| W2 | `check-public-doc-tables.py`: retire `max_chars`, add `MAX_ROW_CHARS` and `DATED_HEADING_RE`, with tests | in this PR |
| W3 | `check-agent-record.py`: fenced/inline-code-aware `extract_links`, `link_bases` for the archive, with tests | in this PR |
| W4 | Rebuild `docs/BENCHMARKS.md` as a derived index over per-row files | **DEFERRED.** Justified only if the page becomes a *conflict* hotspot after the cap is gone. Trigger: `git merge-tree` shows it conflicting in 3 or more concurrently open PRs, measured, as #364 measured its three surfaces. Not justified by the cap, which W2 removes. |
| W5 | Teach `roll-benchmark-record.py` to record the archived section's origin explicitly, rather than relying on W3's two-base resolution | **DEFERRED.** W3 makes the move work; W5 would make it self-describing. No blocker depends on it. |
| W6 | Make `_h2_headers` fence-aware, so the gate and the rollup agree on what a section is ([#495](https://github.com/mudler/vllm.cpp/issues/495)) | **DEFERRED, filed not fixed.** Found doing W2 and reproduced: `_h2_headers` is a bare `startswith("## ")` scan while `split_sections` tracks fences, so a heading-shaped line inside a fence is a section to the gate and not to the script the gate tells you to run. It changes what an existing gate counts, so it takes its own spec and red-before rather than riding along here. Neither page has a fenced heading today. W2's `_headings` is already fence-aware and is the natural basis for the repair. |

## Risks/decisions

**Risk: removing a size gate lets the page bloat.** Answered by measurement, not
assertion. What bloated the page to 11,405 lines was per-attempt sections, and
after W2 the first one fails at either heading depth, where today only `##` is
covered. Prose is still capped at 35 paragraphs and 700 characters each, cells
at 220, and rows now at 600. The one growth W2 permits that the cap forbade is
*more subject rows*, which is the page doing its job.

**Risk: `DATED_HEADING_RE` fires on a legitimate heading.** A pinned-date
subject would trip it, for example "vLLM 0.26.0 as of 2026-08-12". Measured
against both live pages: zero of 32 headings match. If one is ever wanted, the
date belongs in the row or the prose, which are unaffected. The error message
says so.

**Risk: `MAX_ROW_CHARS = 600` is tuned to current content.** It is, and that is
sound for an entry cap in a way it is not for a file cap: an author who needs a
longer row shortens their own row, and never anyone else's. The FEATURES margin
is thin (580 of 600). Accepted; the alternative, no row cap, leaves
`MAX_CELL_CHARS` permitting a 1,100-character row.

**Risk: skipping fenced links narrows a checker.** It narrows it to what the
rule was always about. A fenced target is not a link under CommonMark, so no
reader can follow it and no rendering can dangle. Real links, including one on
the same line after a closed inline span, stay checked, and
`test_the_tree_has_no_dangling_link` holds the whole tree.

**Risk: two-base resolution in the archive hides a genuine dangling link.** The
second base applies to one file, `.agents/benchmark-record.md`, the declared
archive of `docs/BENCHMARKS.md`, and the target must still exist under one of
the two bases. Every other file keeps single-base resolution.

**Decision: `docs/FEATURES.md` loses its `max_chars` too.** The constant lives on
the shared `PageRules`, the argument is identical, FEATURES is at 29,740 of
30,000 with 260 characters free, and leaving it would leave a known lock armed
on the page that grows every time a feature ships. It gains the same two
entry-scoped rules.

## Evidence

- Char and row history: reproduce with `git show <rev>:docs/BENCHMARKS.md | wc -c`
  over `git log --format=%h -25 -- docs/BENCHMARKS.md`. Table in Our baseline.
- `04b2b9fa` over the cap: `git show 04b2b9fa:docs/BENCHMARKS.md | wc -c` gives
  45,007.
- Heading-shape survey: 278 of 301 archived section titles carry a date, 0 of 32
  live headings do.
- #460's reproduction, red before and green after W3.
- `check-pr-size.py`'s own red-before/green-after harness, for both checkers.

## Stop conditions

- **Stop if** removing `max_chars` cannot be shown to leave the append-log class
  caught. The regrowth guard is the condition of the removal, not a nicety.
- **Stop if** the entry cap or the heading guard fires on either live page as
  shipped. That would mean the replacement is a different rule, not a relocated
  one, and the page would owe an edit this row is not authorised to make.
- **Stop and return `NEEDS_DECISION`** if closing #460 turns out to require
  rewriting an archived link, which would break the byte-for-byte guarantee the
  archive exists to give.
- **Never** raise a cap to pass, and never delete an assertion to turn a gate
  green.

## Outcome

Pending. Filled on `DONE` with the measured before/after, what was rejected, and
why each constant is set where it is.
