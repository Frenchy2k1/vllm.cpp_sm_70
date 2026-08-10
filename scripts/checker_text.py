#!/usr/bin/env python3
"""Shared text normalization for the structural source checkers.

The `check-*-consistency.py` / `check-surface-coverage.py` family all match RAW
C/C++ source text. Raw text has three ways of carrying a statement that the
COMPILER never sees, and every one of them is exactly the silent divergence those
checkers exist to catch:

  * a `//` or `/* */` COMMENT — `// AdoptDeviceBytesAsHost(d.b, w.packed);`
    reads as the statement to `re.search` and as a deletion to the compiler;
  * a `#if 0` … `#endif` REGION — same, at preprocessing time;
  * an `if (false) { … }` / `if (0) { … }` BRANCH — same, at run time.

`strip_comments` already existed as two byte-identical private copies (in
check-runner-routing-consistency.py and check-surface-coverage.py). This module is
that helper, made shareable rather than copied a third time, plus the two
normalizations the comment strip alone does not cover.

EVERY FUNCTION HERE IS POSITION-PRESERVING: removed text is overwritten with
spaces and newlines are kept, so byte offsets and line numbers in the returned
string are identical to the input's. Callers that report `file:line` or that
compare offsets to check STATEMENT ORDER therefore keep working on the normalized
text, which the earlier collapse-to-one-space copies did not allow.

WHAT IS DELIBERATELY NOT DONE. These are lexical normalizations, not a
preprocessor. `#ifdef FOO` / `#if VERSION > 2` regions are left ALONE, because a
genuinely conditional region is a real build configuration rather than a disguised
deletion, and deciding it needs the build's macro state. Only a literally-constant
false condition (`#if 0`, `#if false`, `if (false)`, `if (0)`) is removed. String
literals are not tracked either, so a `"//"` inside a string is treated as a
comment start — the same caveat the two original copies carried.

Pure functions over text; unit-tested by tests/scripts/test_checker_text.py.
"""

from __future__ import annotations

import re

__all__ = [
    "blank_out",
    "match_braces",
    "strip_comments",
    "strip_never_taken_branches",
    "strip_preprocessor_disabled",
    "normalize_source",
]

_NOT_NEWLINE = re.compile(r"[^\n]")

# One pass with an alternation, so `//` inside a block comment and `/*` inside a
# line comment are each consumed by the construct that really encloses them.
_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)

# A conditional-compilation directive, at the start of its line.
_CPP_DIRECTIVE = re.compile(r"^[ \t]*#[ \t]*(if|ifdef|ifndef|elif|else|endif)\b[^\n]*", re.M)

# `#if 0` / `#if false`, with nothing else on the line.
_FALSE_COND = re.compile(r"^[ \t]*#[ \t]*if[ \t]+(?:0|false)[ \t]*$")

# `if (false) {` / `if (0) {` — a branch the compiler keeps and never runs.
_DEAD_BRANCH = re.compile(r"\bif\s*\(\s*(?:false|0)\s*\)\s*\{")


def blank_out(text: str) -> str:
    """`text` with every non-newline character replaced by a space."""
    return _NOT_NEWLINE.sub(" ", text)


def match_braces(text: str, start: int) -> int:
    """Index just past the `}` closing the block whose `{` ended at `start`."""
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return i


def strip_comments(text: str) -> str:
    """Blank `//` line and `/* */` block comments so a statement that survives only
    as a COMMENT never satisfies a clause that means "this statement runs"."""
    return _COMMENT.sub(lambda m: blank_out(m.group(0)), text)


def strip_preprocessor_disabled(text: str) -> str:
    """Blank the body of every `#if 0` / `#if false` region.

    Nesting is tracked so an inner `#ifdef`'s `#endif` cannot close the disabled
    region early, and an `#else` / `#elif` at the disabled region's own depth ENDS
    it — that branch is the one the compiler keeps. The directive lines themselves
    are left in place, which makes this idempotent.

    Conditions other than a literal `0` / `false` are not evaluated; see the module
    docstring for why.
    """
    out = text
    depth = 0
    start: int | None = None
    start_depth: int | None = None
    spans: list[tuple[int, int]] = []

    for m in _CPP_DIRECTIVE.finditer(text):
        kw = m.group(1)
        if kw in ("if", "ifdef", "ifndef"):
            depth += 1
            if start is None and kw == "if" and _FALSE_COND.match(m.group(0)):
                start, start_depth = m.end(), depth
        elif kw in ("else", "elif"):
            if start is not None and depth == start_depth:
                spans.append((start, m.start()))
                start = None
        elif kw == "endif":
            if start is not None and depth == start_depth:
                spans.append((start, m.start()))
                start = None
            depth = max(0, depth - 1)

    if start is not None:  # unterminated `#if 0`: disabled through end of file
        spans.append((start, len(text)))

    for a, b in reversed(spans):
        out = out[:a] + blank_out(out[a:b]) + out[b:]
    return out


def strip_never_taken_branches(text: str) -> str:
    """Blank every `if (false) { … }` / `if (0) { … }` branch, braces and head
    included. A following `else` branch is LIVE and is left alone."""
    out = text
    while True:
        m = _DEAD_BRANCH.search(out)
        if m is None:
            return out
        end = match_braces(out, m.end())
        out = out[: m.start()] + blank_out(out[m.start() : end]) + out[end:]


def normalize_source(text: str) -> str:
    """All three normalizations, in the order the toolchain applies them. Idempotent
    and position-preserving, so `file:line` reports and offset comparisons over the
    result still describe the ORIGINAL file."""
    return strip_never_taken_branches(strip_preprocessor_disabled(strip_comments(text)))
