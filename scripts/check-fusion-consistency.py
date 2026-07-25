#!/usr/bin/env python3
"""Fail if a model forward HAND-FUSES an add+residual+RMSNorm chain without
routing it through the portable fusion catalog (`vt::FusedChain`).

The `KERNEL-FUSION-FRAMEWORK` (`.agents/specs/portable-fusion-framework.md`)
declares each fusable op-chain ONCE as a `constexpr FusedRecipe`
(`include/vt/recipes.h`) and realizes it per-backend through the `vt::` op table
via `vt::FusedChain(recipe, ...)`. The residual add+RMSNorm chain is the most
common fusable glue in a model forward; the catalog already carries a recipe for
both variants (`kFusedAddRmsNorm` gemma-(1+w), `kFusedAddRmsNormStd` plain-w).

A model that hand-calls the residual `vt::RmsNorm(..., &residual)` overload
DIRECTLY, instead of `vt::FusedChain(kFusedAddRmsNorm{,Std}, ...)`, DRIFTS from
the catalog: a new backend cannot inherit that fusion, and a new vLLM fusion PR
cannot port as one declaration for that site. This is the exact drift the
framework exists to prevent (spike §4, the PR-#4 additivity test).

This checker is a coarse FLOOR, not a per-site proof: it flags a model file that
contains ONE OR MORE residual add+RMSNorm hand-calls yet NEVER references
`vt::FusedChain` at all (i.e. it has not engaged the catalog for that glue). A
fully-migrated model keeps its same-binary rollback hand-call in the `else`
branch of a `FusedChainAdoptEnabled()` guard, so the presence of `FusedChain`
in the file is the adoption signal. Models that are deliberately-not-fusable
(post-norm / LayerNorm architectures with no add+RMSNorm chain never trip the
detector) or a KNOWN, TRACKED not-yet-migrated model are carried on
`scripts/fusion-consistency-allowlist.txt` with a reason, so the gate stays
green while any NEW unadopted hand-fusion model must be a conscious,
reviewable allowlist entry.

The validation logic is a pure function `drift_models(...) -> list[str]` so it
is unit- and mutation-testable (tests/scripts/test_check_fusion_consistency.py),
mirroring check-env-doc.py.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "src/vllm/model_executor/models"
ALLOWLIST = ROOT / "scripts/fusion-consistency-allowlist.txt"

# The residual overload of vt::RmsNorm — the fused add+RMSNorm chain — is called
# with the residual tensor as the trailing &pointer argument, e.g.
#   vt::RmsNorm(d.q, out.t(), x.t(), w, gemma, &res.t());
# The standalone (non-fused) RmsNorm has NO trailing &pointer, so this pattern
# selects exactly the fusable add+RMSNorm sites.
_RESIDUAL_RMSNORM = re.compile(
    r"vt::RmsNorm\([^;]*,\s*&[A-Za-z_][A-Za-z0-9_]*(?:\.t\(\))?\s*\)\s*;"
)
# The adoption signal: the file routes at least some glue through the catalog.
_FUSEDCHAIN = re.compile(r"FusedChain")


def count_residual_rmsnorm(text: str) -> int:
    """Number of hand-called residual add+RMSNorm sites in a model TU."""
    return len(_RESIDUAL_RMSNORM.findall(text))


def uses_catalog(text: str) -> bool:
    """True if the model TU references the vt::FusedChain catalog seam."""
    return bool(_FUSEDCHAIN.search(text))


def scan_models(models_dir: Path) -> dict[str, tuple[int, bool]]:
    """Map model-file stem -> (residual_rmsnorm_sites, uses_catalog)."""
    out: dict[str, tuple[int, bool]] = {}
    if not models_dir.is_dir():
        return out
    for path in sorted(models_dir.glob("*.cpp")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        n = count_residual_rmsnorm(text)
        if n:
            out[path.stem] = (n, uses_catalog(text))
    return out


def allowlisted_names(text: str) -> set[str]:
    """Model stems accepted as known-drift / deliberately-deferred (one per line,
    # comments ignored)."""
    names: set[str] = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(line)
    return names


def drift_models(
    scanned: dict[str, tuple[int, bool]], allowlisted: set[str]
) -> list[str]:
    """Model stems that hand-fuse add+RMSNorm (>=1 site) yet never reference the
    catalog and are not allowlisted. Empty == the check passes."""
    return sorted(
        stem
        for stem, (n, catalog) in scanned.items()
        if n > 0 and not catalog and stem not in allowlisted
    )


def main() -> int:
    scanned = scan_models(MODELS_DIR)
    allowlisted = (
        allowlisted_names(ALLOWLIST.read_text(encoding="utf-8"))
        if ALLOWLIST.exists()
        else set()
    )
    drift = drift_models(scanned, allowlisted)
    if drift:
        print(
            "ERROR: model forward(s) hand-fuse an add+residual+RMSNorm chain "
            "without routing it through the vt::FusedChain catalog "
            "(KERNEL-FUSION-FRAMEWORK), and are not on "
            "scripts/fusion-consistency-allowlist.txt:",
            file=sys.stderr,
        )
        for stem in drift:
            n = scanned[stem][0]
            print(f"  - {stem}.cpp ({n} residual add+RMSNorm hand-call site(s))",
                  file=sys.stderr)
        print(
            "Route the site through vt::FusedChain(kFusedAddRmsNorm{,Std}, ...) "
            "behind FusedChainAdoptEnabled() (see qwen3.cpp / deepseek_v2.cpp), "
            "or add the model to scripts/fusion-consistency-allowlist.txt with a "
            "reason (known-drift pending migration, or deliberately-not-fused).",
            file=sys.stderr,
        )
        return 1
    n_catalog = sum(1 for _, c in scanned.values() if c)
    print(
        f"OK: {len(scanned)} model TUs carry add+RMSNorm glue; {n_catalog} route "
        f"it through the vt::FusedChain catalog, {len(allowlisted)} are "
        "allowlisted (known-drift / deferred)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
