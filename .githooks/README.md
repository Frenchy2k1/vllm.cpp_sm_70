# Git hooks

Repo-tracked hooks. They are not active until a clone opts in:

```sh
git config core.hooksPath .githooks
```

`git push --no-verify` bypasses them for one push.

## pre-push

Runs the two public-doc page gates against the **commits being pushed**, not the
working tree:

- `scripts/check-readme-structure.py`
- `scripts/check-public-doc-tables.py`

Both enforce limits that otherwise surface only in CI: `docs/STATUS.md` has a
shrink-only size ratchet, and `docs/BENCHMARKS.md` / `docs/FEATURES.md` have a
per-cell budget and a no-em-dash house rule. A push that overshoots them turns
main red for whoever pushes next.

`scripts/agent-preflight.sh` runs these same two checkers plus the rest of the
record gates and the tool suites, and it stays the thing to run before
committing. The hook is the backstop for the run that gets skipped: it checks
the pushed commit's tree, so a dirty checkout cannot fail a clean push and an
uncommitted fix cannot let a broken commit through.
