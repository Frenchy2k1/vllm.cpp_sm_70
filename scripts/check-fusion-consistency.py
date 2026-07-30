#!/usr/bin/env python3
"""Fail if a model forward drifts from the portable fusion / merged-GEMM frameworks.

Two independent floors, one per framework the codebase has built:

1. GLUE (vt::FusedChain catalog, `.agents/specs/portable-fusion-framework.md`):
   a model that hand-calls the residual `vt::RmsNorm(..., &residual)` add+RMSNorm
   overload DIRECTLY, instead of `vt::FusedChain(kFusedAddRmsNorm{,Std}, ...)`,
   cannot inherit the fusion on a new backend and cannot absorb a new vLLM fusion
   PR as one declaration. Allowlist: scripts/fusion-consistency-allowlist.txt.

2. MERGED-GEMM (the gate-up MLP method seam + `vt::MergedGemmGroup` descriptor +
   the shared fused ops, `.agents/specs/arch-fusion-fold-plan-2026-07-30.md`,
   `.agents/specs/keepquant-shared-ops-2026-07-30.md`): a model that hand-rolls a
   gated MLP epilogue (`vt::SiluAndMul` / `vt::GeluAndMul`) WITHOUT routing its
   gate-up through the shared `MlpGateUpMethodBase` seam (or a tuned quant arm:
   GateUpFusedMarlinD / ResidentNvfp4GateUp), the `MergedGemmGroup` descriptor, or
   the shared MoE op (`MoeGateUpSwiGLU`), re-fragments the exact structure the
   framework unified — a new arch does not inherit the tuned kernel, and a new
   backend cannot pick it up by registration. Allowlist:
   scripts/merged-gemm-consistency-allowlist.txt.

Both are coarse FLOORS, not per-site proofs: they flag a model file that carries
the fusable pattern yet NEVER references the corresponding shared seam. A
fully-migrated model keeps its same-binary rollback hand-call in the `else` branch
of an adoption guard, so the presence of the seam token in the file is the
adoption signal. Models deliberately-not-fusable (no matching pattern) never trip
the detector; a KNOWN, TRACKED not-yet-migrated model is carried on the matching
allowlist with a reason, so the gate stays green while any NEW unadopted
hand-fusion model must be a conscious, reviewable allowlist entry.

The validation logic is pure functions (`drift_models`, `gemm_merge_drift_models`)
so both are unit- and mutation-testable
(tests/scripts/test_check_fusion_consistency.py), mirroring check-env-doc.py.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "src/vllm/model_executor/models"
ALLOWLIST = ROOT / "scripts/fusion-consistency-allowlist.txt"
GEMM_ALLOWLIST = ROOT / "scripts/merged-gemm-consistency-allowlist.txt"

# --- Check 1: GLUE (add+residual+RMSNorm -> vt::FusedChain) --------------------

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

# --- Check 2: MERGED-GEMM (gate-up MLP -> shared seam / descriptor / MoE op) ---

# The gated-MLP epilogue — the tell that a forward computes a gate-up MLP
# (SwiGLU / GeGLU) or a grouped-MoE gate+up. A model that has this yet routes its
# gate-up through none of the shared seams below is hand-rolling the merge.
_GATED_MLP_ACT = re.compile(r"vt::(?:SiluAndMul|GeluAndMul)\b")
# The adoption signal: any of the shared merged-GEMM constructs the frameworks
# provide — the generic method seam, its tuned quant arms, the declarative
# descriptor, or the shared fused-MoE op.
# MlpGateUp[A-Za-z]*Method covers the base, the factory (MakeMlpGateUpMethod), and
# BOTH activation arms (UnquantizedMlpGateUpMethod SwiGLU, UnquantizedMlpGateUpGelu
# Method GeGLU — the "Gelu" infix would otherwise break a fixed "MlpGateUpMethod").
_MERGED_GEMM_SEAM = re.compile(
    r"MlpGateUp[A-Za-z]*Method|MergedGemm|MoeGateUpSwiGLU"
    r"|GateUpFusedMarlin|Nvfp4GateUp|ResidentNvfp4GateUp"
)


def count_residual_rmsnorm(text: str) -> int:
    """Number of hand-called residual add+RMSNorm sites in a model TU."""
    return len(_RESIDUAL_RMSNORM.findall(text))


def uses_catalog(text: str) -> bool:
    """True if the model TU references the vt::FusedChain catalog seam."""
    return bool(_FUSEDCHAIN.search(text))


def count_gated_mlp_act(text: str) -> int:
    """Number of gated-MLP epilogue (SiluAndMul/GeluAndMul) hand-calls in a TU."""
    return len(_GATED_MLP_ACT.findall(text))


def uses_merged_gemm_seam(text: str) -> bool:
    """True if the model TU routes gate-up through a shared merged-GEMM seam."""
    return bool(_MERGED_GEMM_SEAM.search(text))


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


def scan_models_gemm(models_dir: Path) -> dict[str, tuple[int, bool]]:
    """Map model-file stem -> (gated_mlp_act_sites, uses_merged_gemm_seam)."""
    out: dict[str, tuple[int, bool]] = {}
    if not models_dir.is_dir():
        return out
    for path in sorted(models_dir.glob("*.cpp")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        n = count_gated_mlp_act(text)
        if n:
            out[path.stem] = (n, uses_merged_gemm_seam(text))
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


def gemm_merge_drift_models(
    scanned: dict[str, tuple[int, bool]], allowlisted: set[str]
) -> list[str]:
    """Model stems that hand-roll a gated MLP epilogue (>=1 site) yet route their
    gate-up through none of the shared merged-GEMM seams and are not allowlisted.
    Empty == the check passes."""
    return sorted(
        stem
        for stem, (n, seam) in scanned.items()
        if n > 0 and not seam and stem not in allowlisted
    )


def _load_allowlist(path: Path) -> set[str]:
    return (
        allowlisted_names(path.read_text(encoding="utf-8"))
        if path.exists()
        else set()
    )


def main() -> int:
    rc = 0

    # Check 1 — glue add+RMSNorm -> FusedChain.
    scanned = scan_models(MODELS_DIR)
    allowlisted = _load_allowlist(ALLOWLIST)
    drift = drift_models(scanned, allowlisted)
    if drift:
        rc = 1
        print(
            "ERROR: model forward(s) hand-fuse an add+residual+RMSNorm chain "
            "without routing it through the vt::FusedChain catalog "
            "(KERNEL-FUSION-FRAMEWORK), and are not on "
            "scripts/fusion-consistency-allowlist.txt:",
            file=sys.stderr,
        )
        for stem in drift:
            print(f"  - {stem}.cpp ({scanned[stem][0]} residual add+RMSNorm "
                  "hand-call site(s))", file=sys.stderr)
        print(
            "Route the site through vt::FusedChain(kFusedAddRmsNorm{,Std}, ...) "
            "behind FusedChainAdoptEnabled() (see qwen3.cpp / deepseek_v2.cpp), "
            "or add the model to scripts/fusion-consistency-allowlist.txt with a "
            "reason (known-drift pending migration, or deliberately-not-fused).",
            file=sys.stderr,
        )
    else:
        n_catalog = sum(1 for _, c in scanned.values() if c)
        print(
            f"OK (glue): {len(scanned)} model TUs carry add+RMSNorm glue; "
            f"{n_catalog} route it through the vt::FusedChain catalog, "
            f"{len(allowlisted)} allowlisted."
        )

    # Check 2 — merged-GEMM gate-up -> shared seam / descriptor / MoE op.
    scanned_g = scan_models_gemm(MODELS_DIR)
    allowlisted_g = _load_allowlist(GEMM_ALLOWLIST)
    drift_g = gemm_merge_drift_models(scanned_g, allowlisted_g)
    if drift_g:
        rc = 1
        print(
            "ERROR: model forward(s) hand-roll a gated MLP epilogue "
            "(vt::SiluAndMul / vt::GeluAndMul) without routing gate-up through a "
            "shared merged-GEMM seam (MlpGateUpMethodBase / MergedGemmGroup "
            "descriptor / MoeGateUpSwiGLU shared op), and are not on "
            "scripts/merged-gemm-consistency-allowlist.txt:",
            file=sys.stderr,
        )
        for stem in drift_g:
            print(f"  - {stem}.cpp ({scanned_g[stem][0]} gated-MLP epilogue "
                  "hand-call site(s))", file=sys.stderr)
        print(
            "Fold the gate-up onto layers::UnquantizedMlpGateUp{,Gelu}Method / "
            "MakeMlpGateUpMethod (dense; see qwen3.cpp / gemma4.cpp), the "
            "vt::MergedGemmGroup descriptor, or vt::MoeGateUpSwiGLUGrouped (MoE; "
            "see deepseek_v4.cpp), or add the model to "
            "scripts/merged-gemm-consistency-allowlist.txt with a reason "
            "(known-drift pending fold, or deliberately-not-merged).",
            file=sys.stderr,
        )
    else:
        n_seam = sum(1 for _, s in scanned_g.values() if s)
        print(
            f"OK (merged-gemm): {len(scanned_g)} model TUs carry a gated MLP "
            f"epilogue; {n_seam} route gate-up through a shared merged-GEMM seam, "
            f"{len(allowlisted_g)} allowlisted."
        )

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
