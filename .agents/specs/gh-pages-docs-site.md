# GitHub Pages docs site — render `docs/` without owning a second copy

Row: `ENG-DOCS-SITE` ([engine-matrix](../engine-matrix.md)).
Issue: [#224](https://github.com/mudler/vllm.cpp/issues/224).
Status: accepted design, user-directed 2026-08-09.

Publishes the existing `docs/*.md` as a browsable site at
`https://mudler.github.io/vllm.cpp/`. Inspired by
[LocalAI's](https://github.com/mudler/LocalAI) two-Hugo-site + one-Pages-artifact
setup (`website/` marketing site, `docs/` relearn site, merged by
`.github/workflows/gh-pages.yml`), reduced to the docs half.

## The principle

**Anything that must be kept in sync is the defect.**

`docs/*.md` are protocol-governed projections. `AGENTS.md` fixes exactly when each
one changes — `docs/STATUS.md` on a lifecycle state change, `docs/BENCHMARKS.md`
on an accepted or pending measurement — and `scripts/check-doc-checkpoint.py`
validates them at those literal paths. A site that *copies* those files into a
`content/` tree creates a second surface that can disagree with the first, and
nothing would notice.

So the site owns no content. It mounts `docs/` read-only and derives everything
it needs — titles, ordering, links — from what is already in the files.

| Concern | Single source | Duplicate this design refuses to create |
|---|---|---|
| Doc prose | `docs/*.md` | a `website/content/docs/` copy |
| Page title | the file's first `# H1` | a `title:` front-matter key |
| Sidebar order | `website/data/nav.yaml` | a `weight:` in every file |
| Protocol links | `.agents/**` on GitHub | a mirrored `.agents/` on the site |

No file under `docs/` is modified, moved, renamed, or gains front matter.

## Scope

**In:** a Hugo site at `website/`, custom lean layouts, a Pages deploy workflow,
a `scripts/check-site.py` guard, a `website/**` path class in
`scripts/check-pr-size.py` with its test, a README pointer.

**Out:** a marketing landing page (the README stays the front door); migrating or
restructuring `docs/`; a custom domain; publishing `.agents/**`; search over
`docs/bench-evidence/`.

## Upstream chain

**There is no vLLM analogue, and this row carries no parity obligation.**
Upstream's documentation is a separate mkdocs site describing a Python package;
nothing here mirrors upstream *behavior*, so the "mirror vLLM, never ask" rule
has nothing to bind to. Recorded as from-scratch work rather than a port.

The **structural** reference is LocalAI, read at
`~/_git/LocalAI` on 2026-08-09:

| Reference | What is taken |
|---|---|
| `.github/workflows/gh-pages.yml` | job shape: checkout → `peaceiris/actions-hugo` → `actions/configure-pages` → build with `--baseURL ${{ steps.pages.outputs.base_url }}` → `upload-pages-artifact` → `deploy-pages`; `concurrency: pages` with `cancel-in-progress: false` |
| `docs/hugo.toml` | Hugo `0.146.3` extended as the pinned version, and `[markup.goldmark.renderer] unsafe = true` for HTML in markdown |
| `website/static/CNAME` | the dormant custom-domain mechanism |

What is deliberately **not** taken: the relearn theme (front-matter-driven, see
fact 7), the second marketing Hugo site, and the merge-two-`public/`-trees step
that only exists because LocalAI publishes both.

## Our baseline

On `main` at `64b94471`:

- **No site of any kind.** No `website/`, no `.github/workflows/gh-pages.yml`,
  no Hugo config anywhere in the tree. `docs/` is 11 top-level markdown files
  plus `bench-evidence/` (18 files) and `superpowers/` (36 specs and plans),
  readable only through GitHub's blob view.
- **`README.md`** is the de facto landing page (521 lines) and stays so.
- **`scripts/check-pr-size.py`** classifies every repository path and raises on
  `website/**` (fact 6).
- **No checker asserts anything about doc *content* shape** — that every doc
  opens with an `# H1` is true today by convention only, which is why
  `check-site.py` exists.

## Port map

Nothing is ported; every artifact is new. Recorded here so the
[porting inventory](../porting-inventory.md) §9 (from-scratch) entry is
unambiguous:

| New artifact | Derived from | Kind |
|---|---|---|
| `.github/workflows/gh-pages.yml` | LocalAI's `gh-pages.yml`, single-site | adapted |
| `website/hugo.toml` | LocalAI's `docs/hugo.toml` for the pin and goldmark block; mounts are new | adapted |
| `website/layouts/**`, `website/assets/css/site.css` | none | from scratch |
| `website/data/nav.yaml`, `website/content/_index.md` | none | from scratch |
| `scripts/check-site.py` | the repo's existing `check-*.py` idiom | from scratch |

## Tests to port

**None — there is no upstream test suite to port.** LocalAI's site has no tests;
vLLM has no analogous surface. The gates below are therefore all locally
authored, and each is a *mutation* gate rather than a comparison against a
reference implementation:

| New test | Proves | Red-before mutation |
|---|---|---|
| `scripts/check-site.py` invariant 1 | every mounted doc opens with `# ` | strip the `# H1` from `docs/USAGE.md` |
| `scripts/check-site.py` invariant 2 | `nav.yaml` ↔ mounted set bijection | delete one `nav.yaml` entry; add a stray one |
| `tests/scripts/test_check_pr_size.py` | `website/**` → `public_document` | the assertion fails on today's `main`, where `classify_path` raises |
| the PR-time `gh-pages.yml` build job | the site compiles | any Hugo template error |

## Dependencies

- **Hugo `0.146.3` extended** — already installed locally; CI installs it via
  `peaceiris/actions-hugo`. No theme, no git submodule, no Go module, no npm.
  Verified: the `../docs` mount needs no `go.mod`.
- **GitHub Pages enabled on `mudler/vllm.cpp`** with source = GitHub Actions.
  Outside the repository and outside this session's authority; a stop condition.
- **`scripts/check-pr-size.py`** must learn `website/**` *before* the PR can
  pass its own gate — an ordering dependency inside the same change, not a
  separate PR.
- No dependency on any other row. Nothing depends on this one.

## Work breakdown

Five non-overlapping units, in dependency order. W1 must land first or nothing
else can be reviewed.

| # | Unit | Files | Done when |
|---|---|---|---|
| W1 | Teach the size gate the path | `scripts/check-pr-size.py`, `tests/scripts/test_check_pr_size.py` | red-before test green-after; no other assertion touched |
| W2 | The site skeleton that builds | `website/hugo.toml`, `content/_index.md`, `data/nav.yaml`, minimal layouts | `hugo --minify` emits 11 doc pages + home; `bench-evidence/` and `superpowers/` absent from `public/` |
| W3 | Titles and links | `layouts/partials/title.html`, `layouts/_default/_markup/render-link.html` | no `href` ending in `.md` in `public/`; `docs/status/` links to the `.agents/roadmap_v1.md` blob URL |
| W4 | The guard | `scripts/check-site.py`, wired into `ci.yml` | exits 0 on `main`, non-zero under both mutations |
| W5 | Style, search, deploy | `assets/css/site.css`, `index.json`, `.github/workflows/gh-pages.yml`, `website/README.md`, `README.md` pointer | site legible in both themes; workflow builds on PR, deploys on `main` |

W2 and W3 are separable but land together — W2 alone publishes a site whose
every cross-link 404s, which is not a state worth having on `main`.

## Verified facts this design rests on

Established by measurement on `main` at `64b94471`, not assumption:

1. **A `../docs` mount works.** A scratch site with
   `[[module.mounts]] source = "../docs", target = "content/docs"` built
   successfully on the locally installed Hugo `0.146.3+extended`, produced
   `/docs/status/` and `/docs/benchmarks/`, and honored `excludeFiles` for
   `bench-evidence/**`. No `go.mod` was required.
2. **Front-matter-less pages render but have an empty `.Title`.** The same build
   emitted `/docs/status/ :: ` — the title is blank. Titles must be derived.
3. **All 11 top-level docs open with a `# H1`.** Checked file by file, so
   deriving the title from `.RawContent` is sound — and worth guarding, because
   it is an invariant of *content* that no existing checker holds.
4. **No doc references an image.** `grep -ohE '!\[[^]]*\]\([^)]*\)' docs/*.md`
   is empty, so there is no asset-path rewriting problem.
5. **139 markdown cross-links, and the overwhelming majority leave `docs/`.**
   They point at `../.agents/**` and `../AGENTS.md`; `docs/STATUS.md` alone has
   70. Left alone, every one of them 404s on the site. The count is not stable —
   it was 135 ten commits earlier, at `4cba292d` — which is the argument for
   rewriting the whole class mechanically rather than fixing links one by one.
6. **`classify_path` in `scripts/check-pr-size.py` fails closed on `website/`.**
   Called directly, `classify_path("website/hugo.toml")` raises
   `ValueError: unclassified repository path 'website/hugo.toml'`, as do
   `website/layouts/index.html` and `website/assets/css/site.css`. A PR adding
   `website/` therefore cannot pass the repo's own size gate until the
   classifier learns the path. This is a hard prerequisite, not a nicety. The
   other new paths already classify correctly and need no change:
   `.github/workflows/gh-pages.yml` → `ci`, `scripts/check-site.py` →
   `governance_checker`, `tests/scripts/test_check_site.py` →
   `governance_test`, and this spec → `procedure`.
7. **hugo-book now requires Hugo ≥ 0.158.** The local binary and LocalAI's CI
   both pin `0.146.3`. Combined with (2) — themes read titles, weights and menus
   out of front matter this design does not have — every candidate theme would
   need its title partial, menu and link hook overridden anyway. Hence custom
   layouts rather than a theme.

## Design

### Layout

```
website/
  hugo.toml                       # mounts, baseURL, params
  data/nav.yaml                   # sidebar order + labels, 11 entries
  content/_index.md               # home: short intro + card list
  layouts/
    _default/baseof.html
    _default/single.html
    index.html
    _default/_markup/render-link.html
    partials/{head,nav,sidebar,footer,title}.html
  static/                         # logo + favicon copied from assets/
  assets/css/site.css
  README.md                       # local preview, CNAME path
```

### Mount

```toml
[[module.mounts]]
source = "content"
target = "content"

[[module.mounts]]
source = "../docs"
target = "content/docs"
excludeFiles = ["bench-evidence/**", "superpowers/**"]
```

Excluding `bench-evidence/` (18 raw benchmark logs) and `superpowers/` (36 specs
and plans) leaves the 11 user-facing docs. `excludeFiles` is deprecated in Hugo
0.153 in favor of `files`; at the pinned 0.146.3 it is the correct key, and the
pin is what CI installs, so the two agree. A Hugo bump past 0.153 must migrate
this key — recorded here because nothing else will say it.

### Title from the first H1

`layouts/partials/title.html` extracts the first `^# ` line from `.RawContent`
and falls back to `.File.ContentBaseName` if absent. Fact (3) says the fallback
should never fire; `scripts/check-site.py` makes that a gate rather than a hope.

### Link rewriting

`layouts/_default/_markup/render-link.html`, two rules:

| Link in `docs/*.md` | Rewritten to |
|---|---|
| `BENCHMARKS.md`, `./USAGE.md` | `/vllm.cpp/docs/benchmarks/`, `/vllm.cpp/docs/usage/` |
| `../.agents/**`, `../AGENTS.md`, `../include/**`, any other `../` escape | `https://github.com/mudler/vllm.cpp/blob/main/<path>` |

Everything else (absolute URLs, in-page anchors) passes through untouched. The
second rule is what keeps fact (5)'s 139 links alive: protocol documents are not
published, but they remain one click away at their source.

Pinning the GitHub URL to `main` rather than a SHA is deliberate: the site tracks
`main`, and a link into a frozen SHA would rot in the opposite direction.

### Navigation

`website/data/nav.yaml` lists the 11 docs in reading order with display labels —
Usage, Build, Features, Benchmarks, Status, Environment, Releases, Speculative
decoding, KV offload, SGLang compat, ROCm. Alphabetical order would open on
`BENCHMARKS.md`; this opens on `USAGE.md`.

### Look

Custom CSS, roughly 250 lines, on CSS custom properties with a
`prefers-color-scheme` dark/light pair, using the palette the README badges
already establish (cyan `#3ec8e0`, green `#7ee787`). Two-column: sticky sidebar,
content column with a max measure. No JS framework. Client-side search is a
~30-line filter over a Hugo-generated `index.json` of the 11 page titles and
headings — at this corpus size that is the whole feature.

### Deploy

`.github/workflows/gh-pages.yml`:

- **Trigger** — push to `main` filtered to `docs/**`, `website/**`, `assets/**`,
  and the workflow file, plus `workflow_dispatch`.
  `concurrency: {group: pages, cancel-in-progress: false}`.
- **Runner** — `ubuntu-latest`. This repo's only self-hosted routing is the GPU
  job in `triton-aot-sync.yml`; a Hugo build has no claim on it.
- **Steps** — `actions/checkout` → `peaceiris/actions-hugo` pinned `0.146.3`
  extended → `actions/configure-pages` → `hugo --minify --baseURL
  ${{ steps.pages.outputs.base_url }}/` in `website/` →
  `upload-pages-artifact` → `deploy-pages` in a `github-pages` environment.
  `permissions: {contents: read, pages: write, id-token: write}`.
- **PR guard** — on pull requests touching the same paths, build without
  deploying, so a site that does not compile fails review instead of `main`.

Taking `baseURL` from `configure-pages` rather than hardcoding
`https://mudler.github.io/vllm.cpp/` means a later custom domain needs only a
`website/static/CNAME`; `website/README.md` records that path.

### `scripts/check-site.py`

In the repo's existing checker idiom, holding the two invariants the site
silently depends on:

1. Every `docs/*.md` reachable through the mount begins with a `# ` line —
   fact (3), which nothing else enforces and a future doc could break.
2. `website/data/nav.yaml` and the mounted set are in bijection — no nav entry
   pointing at a deleted file, no new doc invisible in the sidebar.

Wired into `ci.yml` beside the other document checkers. Its own message is the
authority on what it enforces; this paragraph is prose about it.

### `check-pr-size.py`

`website/**` is classified `public_document` (budget 2500). It is prose, layout
and style for a published surface; that is what the class is for, and the site is
about 600 lines. `.github/workflows/gh-pages.yml` keeps class `ci` and
`scripts/check-site.py` keeps `governance_checker`, both by existing rules.

This is a checker semantics change, so per `AGENTS.md` it lands with a
red-before test in `tests/scripts/test_check_pr_size.py` asserting
`classify_path("website/hugo.toml") == "public_document"` — failing on today's
`main`, where the call raises — and green-after evidence. No assertion is
deleted and no scope is widened to make it pass.

## Risks

| Risk | Mitigation |
|---|---|
| A future doc lands without an `# H1` and publishes titleless | `check-site.py` invariant 1 |
| A new doc never appears in the sidebar | `check-site.py` invariant 2 |
| Hugo bump past 0.153 silently ignores `excludeFiles`, publishing 18 benchmark logs and 36 specs and plans | Version pinned in the workflow; migration noted above; a bump is a deliberate PR |
| `docs/` restructuring breaks the mount | The mount is a glob over one directory; `check-site.py` fails loudly rather than publishing a partial site |
| Publishing `.agents/**` by accident | Excluded at the mount, and `.agents/` is outside the mounted tree entirely |

## Tests and evidence

- `hugo --minify` builds clean in `website/` — 11 doc pages plus home.
- The generated `docs/status/index.html` contains a resolvable link to
  `github.com/mudler/vllm.cpp/blob/main/.agents/roadmap_v1.md` (fact 5's largest
  cluster) and no `href` ending in `.md`.
- `python3 scripts/check-site.py` exits 0 on `main` and non-zero under a
  mutation that strips the `# H1` from a doc, and under a mutation that removes
  one entry from `data/nav.yaml`.
- `tests/scripts/test_check_pr_size.py` red before the classifier change, green
  after; the rest of the file unchanged.
- `scripts/agent-preflight.sh --staged` clean before commit.

## Gates

| Gate | Command | Result |
|---|---|---|
| The site builds | `hugo --minify -s website` | required |
| Site invariants hold | `python3 scripts/check-site.py` | required |
| Size gate learns the path | `python3 tests/scripts/test_check_pr_size.py` | required, red-before / green-after |
| Record is consistent | `python3 scripts/check-agent-record.py` | required |
| Staged protocol gate | `scripts/agent-preflight.sh --staged` | required |
| `docs/**` byte-for-byte unchanged | `git diff --stat origin/main -- docs/` is empty except this spec's own path | required |
| Open GitHub issue linked in the PR body | [#224](https://github.com/mudler/vllm.cpp/issues/224) | done |
| Site reachable at `https://mudler.github.io/vllm.cpp/` | manual, after merge | required |

## Doc surfaces owed

Adding a website is not a lifecycle change, so it owes no `STATUS.md`,
`BENCHMARKS.md` or `NOW.md` update. It owes:

- `README.md` — one line pointing at the published site, once it is live.
- `website/README.md` — local preview (`hugo server -s website`), the Hugo pin,
  and the dormant-CNAME path for a future custom domain.

## Stop conditions

- GitHub Pages is not enabled for the repository and cannot be enabled from this
  session — report and stop; the workflow is inert without it.
- The `../docs` mount fails on the CI Hugo despite working locally — stop rather
  than falling back to copying files, which would reintroduce the duplicate this
  design exists to avoid.
- A doc turns out to need per-page front matter to render correctly — return
  `NEEDS_DECISION` rather than editing anything under `docs/`.

## Open question deferred

A custom domain is entangled with the pending vLLM trademark outreach
(`collaboration@vllm.ai`, fallback name `ingot.cpp`). The site ships on the
default Pages URL; the CNAME mechanism is documented and dormant.
