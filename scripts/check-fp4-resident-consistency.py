#!/usr/bin/env python3
"""Fail if an fp4 resident-upload copy drops the ENG-LOAD-DIRECT-UPLOAD post-upload step.

THE DEFECT THIS EXISTS TO PREVENT ALREADY HAPPENED, TWICE OVER. `ResidentNvfp4` is
the ONE host->device move of a compressed-tensors fp4 weight's `packed`/`scale`,
because `LoadCtNvfp4W4A16` / `LoadCtMxfp4W4A16` / `LoadCtNvfp4Raw` BORROW those
buffers from the safetensors mmap (issue #150) instead of copying them. Round 2 of
ENG-LOAD-DIRECT-UPLOAD therefore gave every fp4 upload three statements per buffer:

  1. `load_stats::AddDeviceUpload(nb)`         — account the move. These bytes are
     counted NOWHERE else: they were borrowed on the way in, so the host-copy
     counter never saw them.
  2. `w.<buf>.d_dev = w.d_<handle>`            — PUBLISH the allocation on the
     OwnedTensor. `AdoptDeviceBytesAsHost` keys on `d_dev` and returns immediately
     when it is null, so without this the next statement is a silent no-op.
  3. `AdoptDeviceBytesAsHost(d.b, w.<buf>)`    — the post-upload residency step:
     release the consumed source pages, and on a host-addressable device (GB10
     unified memory under Vulkan) adopt the device allocation as the host view so
     the model is not resident TWICE.

A fresh reviewer then reverted all six statements — three per buffer, in BOTH copies
of the function — rebuilt clean, and ran every suite touching fp4 residency or the
load counters: 20/20 still passed. `tests/vllm/test_load_direct_upload.cpp` now pins
the SHARED copy at run time (it drives `dense_nvfp4::ResidentNvfp4` over a fake
host-addressable backend and an observable stand-in mapping, and goes red under each
of those mutations). This checker exists for the OTHER copy.

WHY A SOURCE CHECK FOR THAT ONE. `src/vllm/model_executor/models/qwen3_5.cpp` keeps
its own `ResidentNvfp4` — recorded, deliberate duplication (see the preamble of
include/vllm/model_executor/models/dense_nvfp4_gemm.h: unifying the two device-glue
families is a separate refactor that would touch the 27B/35B gate models' hot path).
It lives in an ANONYMOUS namespace inside an 8.5k-line translation unit, so no test
can call it and no runtime assertion can reach it. The only mechanism that bites when
it silently diverges from the copy the runtime gate pins is a structural one.

The invariant, over each `ResidentNvfp4` body in both files, for each of `packed` and
`scale`:

  (a) the upload is COUNTED     — a `load_stats::AddDeviceUpload(...)` call;
  (b) the allocation is PUBLISHED on the OwnedTensor — `w.<buf>.d_dev = ...`;
  (c) the residency step RUNS   — `AdoptDeviceBytesAsHost(..., w.<buf>)`;
  (d) (b) precedes (c)          — the ordering that makes (c) more than a no-op.

Pure functions over text, so they are unit- and mutation-testable
(tests/scripts/test_check_fp4_resident_consistency.py), mirroring
check-fusion-consistency.py and check-gemv-invocation-consistency.py.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The two copies of the fp4 resident upload. Both are in scope: the shared one is
# also pinned at run time, and checking it here keeps the two descriptions of the
# invariant from drifting apart in opposite directions.
SOURCES = (
    Path("include/vllm/model_executor/models/dense_nvfp4_gemm.h"),
    Path("src/vllm/model_executor/models/qwen3_5.cpp"),
)

# The buffers an Nvfp4Weight uploads. Each needs the full three-statement step.
BUFFERS = ("packed", "scale")

# The function whose body carries the invariant.
FUNCTION = "ResidentNvfp4"

# A definition head `... ResidentNvfp4(Dev d, const Nvfp4Weight& w) {`. Anchored on
# the parameter list so the many CALL sites (`ResidentNvfp4(d, w)`) never match, and
# so the neighbouring `ResidentNvfp4Qkv` / `ResidentNvfp4GateUp` / `ResidentNvfp4Alpha`
# / `ResidentNvfp4ScaleSwizzled` overloads — which build MERGED device operands out of
# several borrowed tensors and are explicitly out of this row's scope — do not either.
_DEF = re.compile(
    r"\b" + re.escape(FUNCTION) + r"\s*\(\s*Dev\s+\w+\s*,\s*const\s+Nvfp4Weight\s*&\s*\w+\s*\)\s*\{"
)

_COUNT = re.compile(r"\bload_stats::AddDeviceUpload\s*\(")
_ADOPT = re.compile(r"\bAdoptDeviceBytesAsHost\s*\(\s*[^,)]+,\s*(?P<arg>[\w.]+)\s*\)")


def _publish(buffer: str) -> re.Pattern[str]:
    """`w.packed.d_dev = <anything>;` — the publication of the device allocation
    onto the OwnedTensor that AdoptDeviceBytesAsHost keys on."""
    return re.compile(r"\.\s*" + re.escape(buffer) + r"\s*\.\s*d_dev\s*=")


def _line_no(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def function_bodies(text: str) -> list[tuple[int, str]]:
    """Every `ResidentNvfp4(Dev, const Nvfp4Weight&)` body as (line_no, body_text),
    delimited by brace matching from the definition's opening brace."""
    bodies: list[tuple[int, str]] = []
    for m in _DEF.finditer(text):
        start = m.end()  # just past the opening brace
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        bodies.append((_line_no(text, m.start()), text[start : i - 1]))
    return bodies


def adopted_buffers(body: str) -> dict[str, int]:
    """Buffer name -> offset of the AdoptDeviceBytesAsHost call that adopts it.
    `AdoptDeviceBytesAsHost(d.b, w.packed)` yields {"packed": ...}."""
    found: dict[str, int] = {}
    for m in _ADOPT.finditer(body):
        name = m.group("arg").split(".")[-1]
        found.setdefault(name, m.start())
    return found


def body_violations(body: str) -> list[str]:
    """Every way ONE ResidentNvfp4 body breaks the invariant. Empty == it holds."""
    problems: list[str] = []

    if not _COUNT.search(body):
        problems.append(
            "the host->device upload is NOT counted: no load_stats::AddDeviceUpload("
            ") call. These bytes were BORROWED from the shard mmap, so this is the "
            "only counter that can ever see them"
        )

    adopted = adopted_buffers(body)
    for buffer in BUFFERS:
        pub = _publish(buffer).search(body)
        if pub is None:
            problems.append(
                f"`{buffer}` does not PUBLISH its device allocation on the OwnedTensor "
                f"(`w.{buffer}.d_dev = ...`); AdoptDeviceBytesAsHost keys on `d_dev` "
                f"and would return immediately"
            )
        if buffer not in adopted:
            problems.append(
                f"`{buffer}` skips the post-upload residency step "
                f"(`AdoptDeviceBytesAsHost(d.b, w.{buffer})`): its consumed source "
                f"pages are never released and, on a host-addressable device, the "
                f"model stays resident twice"
            )
        elif pub is not None and pub.start() > adopted[buffer]:
            problems.append(
                f"`{buffer}` publishes `d_dev` AFTER its AdoptDeviceBytesAsHost call, "
                f"so the adoption sees a null `d_dev` and silently does nothing"
            )
    return problems


def file_violations(text: str, label: str) -> list[str]:
    """Every violation in one file, each already prefixed with `label:line`."""
    bodies = function_bodies(text)
    if not bodies:
        return [
            f"{label}: no `{FUNCTION}(Dev, const Nvfp4Weight&)` definition found. "
            f"If this copy was deliberately removed or renamed, update SOURCES in "
            f"scripts/check-fp4-resident-consistency.py in the same change."
        ]
    return [
        f"{label}:{line_no}: {problem}"
        for line_no, body in bodies
        for problem in body_violations(body)
    ]


def main() -> int:
    violations: list[str] = []
    checked = 0
    for rel in SOURCES:
        path = ROOT / rel
        if not path.exists():
            print(f"ERROR: {rel} not found", file=sys.stderr)
            return 1
        text = path.read_text(encoding="utf-8", errors="ignore")
        checked += len(function_bodies(text))
        violations.extend(file_violations(text, str(rel)))

    if violations:
        print(
            "ERROR: an fp4 resident upload drops part of the ENG-LOAD-DIRECT-UPLOAD "
            "post-upload step (issue #150):",
            file=sys.stderr,
        )
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        print(
            "Every ResidentNvfp4 must COUNT its upload, PUBLISH the allocation on the "
            "OwnedTensor, and then run AdoptDeviceBytesAsHost, for BOTH `packed` and "
            "`scale`. The shared copy is pinned at run time by "
            "tests/vllm/test_load_direct_upload.cpp; this gate exists because the "
            "qwen3_5.cpp duplicate sits in an anonymous namespace no test can reach.",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: {checked} ResidentNvfp4 definition(s) across {len(SOURCES)} file(s) count "
        f"the upload, publish d_dev, and adopt both packed and scale."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
