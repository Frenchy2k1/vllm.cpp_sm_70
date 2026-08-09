#!/usr/bin/env python3
"""Load and validate .agents/waivers.csv.

A waiver is visible debt, not success. It names ONE checker and ONE exact
target, with an owner, a reason, evidence, and a future expiry.

Waivers used to be keyed by a POL-* rule ID from .agents/policy.csv. That
registry is gone -- the checker is now the rule -- so a waiver keys on the
checker's own path. That also removes a whole failure mode: a waiver can no
longer name a rule that no code enforces.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable


HEADER = ("waiver_id", "checker", "scope", "owner", "reason", "evidence", "expires")
WAIVERS_PATH = ".agents/waivers.csv"


@dataclass(frozen=True)
class Waiver:
    waiver_id: str
    checker: str
    scope: str
    owner: str
    reason: str
    evidence: str
    expires: str

    def is_expired(self, today: date) -> bool:
        return date.fromisoformat(self.expires) < today


def load_waivers(root: Path, *, today: date | None = None) -> tuple[Waiver, ...]:
    """Return every valid waiver, raising ValueError on a malformed file."""

    path = root / WAIVERS_PATH
    if not path.exists():
        return ()
    today = today or date.today()

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = tuple(next(reader, ()))
        if header != HEADER:
            raise ValueError(f"{WAIVERS_PATH}: header must be exactly {HEADER}")
        rows = [row for row in reader if row]

    waivers: list[Waiver] = []
    seen: set[str] = set()
    for row in rows:
        if len(row) != len(HEADER):
            raise ValueError(f"{WAIVERS_PATH}: row {row!r} must have {len(HEADER)} cells")
        waiver = Waiver(*(cell.strip() for cell in row))
        if not all(
            (waiver.waiver_id, waiver.checker, waiver.scope, waiver.owner,
             waiver.reason, waiver.evidence, waiver.expires)
        ):
            raise ValueError(f"{WAIVERS_PATH}: {waiver.waiver_id or row!r} has an empty cell")
        if waiver.waiver_id in seen:
            raise ValueError(f"{WAIVERS_PATH}: duplicate waiver id {waiver.waiver_id}")
        seen.add(waiver.waiver_id)
        if "*" in waiver.scope or waiver.scope in {".", "/", "all", "repository"}:
            raise ValueError(
                f"{WAIVERS_PATH}: {waiver.waiver_id} has a wildcard or repository-wide "
                "scope; a waiver names one exact target"
            )
        try:
            expired = waiver.is_expired(today)
        except ValueError as exc:
            raise ValueError(
                f"{WAIVERS_PATH}: {waiver.waiver_id} expires {waiver.expires!r}, "
                "which is not an ISO date"
            ) from exc
        if expired:
            raise ValueError(
                f"{WAIVERS_PATH}: {waiver.waiver_id} expired on {waiver.expires}; "
                "remove it or fix the underlying change"
            )
        waivers.append(waiver)
    return tuple(waivers)


def exact_waiver(
    waivers: Iterable[Waiver], checker: str, scope: str
) -> Waiver | None:
    """Return the waiver covering exactly this checker and target, if any."""

    for waiver in waivers:
        if waiver.checker == checker and waiver.scope == scope:
            return waiver
    return None
