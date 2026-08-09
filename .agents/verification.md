# Task guide — gates, evidence, and review

How to prove something, and how to review someone else's proof. The rules are in
[`AGENTS.md`](../AGENTS.md); this is the method.

## Running a gate

Start with the smallest deterministic test that can falsify the spec. Preserve
the red result — a test that was never seen failing has proven nothing. Make it
green, run the declared focused gate, then run the full preflight.

A gate report records the immutable SHA, the exact command, the environment,
the exit status, and the evidence path. Not a summary of them.

Two traps that have produced false greens here:

- **Incremental builds mask `-Werror`.** Clean-rebuild after any header change;
  an incremental green is not a clean green.
- **A copied build directory rebuilds the original sources.** CMake caches
  absolute source paths, so `cp -a` of a build tree can produce a
  byte-identical binary from unmodified code. Build in place and verify source
  and binary checksums.

Release builds define `NDEBUG`, so `assert` is compiled out. A green Release
gate over an assert-firing bug is a latent failure, not a pass — check the build
type before believing a surprising green.

Tests that starve under `ctest -j` are re-run serially before being called a
regression.

## Reviewing

Review happens only after the implementation's own gates pass, and only on an
immutable head, and never by the agent that wrote the code.

**Static pass:** the spec, the diff, the tests, the error paths, ownership
boundaries, and whether the claims are actually supported.

**Mutation pass:** for each critical guard, temporarily remove or corrupt it in
a *scratch copy* and prove the focused test fails. Mutate, don't just read — a
test that passes with the guard deleted was testing nothing. Restore the tree
byte-for-byte after every mutation, and never mutate the reviewed worktree.

Report `PASS` only after both passes on the same head. Every finding carries
severity, the violated requirement, a reproduction, and the expected behavior.

Do not take another agent's report at face value; the operator reruns the gate
regardless of how confident the report sounded.

## Evidence

Separate what you observed from what you inferred. Name source roots, versions,
`file:line` anchors, commands, artifacts, and limitations.

A negative result is a result: record refuted hypotheses and failed attempts,
including the regime they were measured in. "Not established" usually means
"not resolvable against the current noise floor" — the same code can read
differently once the bottleneck moves, so a discarded lever is worth re-testing
after the surrounding performance picture changes.

Public documents carry only the keyed current projection. Forensic detail stays
in the row's spec and the append-only records.
