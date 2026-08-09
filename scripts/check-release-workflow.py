#!/usr/bin/env python3
"""Static least-privilege and immutable-handoff gate for W8 release.yml."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/release.yml"


def job_block(text: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n(.*?)(?=^  [a-zA-Z0-9_-]+:\n|\Z)", text
    )
    return match.group(1) if match else ""


def validate(text: str) -> list[str]:
    errors: list[str] = []
    required_global = (
        "  workflow_dispatch: {}",
        "  push:\n    tags: ['v*']",
        "permissions:\n  contents: read",
    )
    for fragment in required_global:
        if fragment not in text:
            errors.append(f"workflow is missing required global contract: {fragment!r}")
    if re.search(r"(?m)^\s*pull_request:", text):
        errors.append("release workflow must not run for pull requests")
    if "continue-on-error" in text:
        errors.append("release workflow may not continue after an error")

    blocks = {name: job_block(text, name) for name in ("plan", "build", "verify", "attest", "publish")}
    for name, block in blocks.items():
        if not block:
            errors.append(f"release workflow is missing {name} job")
    for name in ("plan", "build", "verify"):
        block = blocks[name]
        if "    permissions:\n      contents: read" not in block:
            errors.append(f"{name} job must have contents: read only")
        permission_block = re.search(r"(?m)^    permissions:\n((?:      [^\n]*\n)+)", block)
        if permission_block and "write" in permission_block.group(1):
            errors.append(f"{name} job unexpectedly has write permission")

    attest = blocks["attest"]
    for permission in (
        "      contents: read",
        "      id-token: write",
        "      attestations: write",
        "      artifact-metadata: write",
    ):
        if permission not in attest:
            errors.append(f"attest job is missing {permission.strip()}")
    if "uses: actions/attest@v4" not in attest:
        errors.append("attest job must use actions/attest@v4")
    tag_gate = "startsWith(github.ref, 'refs/tags/v')"
    if tag_gate not in attest or "needs.plan.outputs.publish == 'true'" not in attest:
        errors.append("attest job must require an exact tag and approved publish plan")

    publish = blocks["publish"]
    if "    permissions:\n      contents: write" not in publish:
        errors.append("publish job alone must receive contents: write")
    if any(permission in publish for permission in ("id-token: write", "attestations: write")):
        errors.append("publish job must not receive attestation authority")
    if "    environment: release" not in publish:
        errors.append("publish job must use the protected release environment")
    if "    needs: [plan, verify, attest]" not in publish:
        errors.append("publish job must consume only plan plus verified and attested handoffs")
    if tag_gate not in publish or "needs.plan.outputs.publish == 'true'" not in publish:
        errors.append("publish job must require an exact tag and approved publish plan")

    required_handoff = (
        "name: release-plan-${{ github.sha }}",
        "name: release-unverified-${{ github.sha }}",
        "name: release-verified-${{ github.sha }}",
        "artifact-ids: ${{ needs.plan.outputs.artifact_id }}",
        "artifact-ids: ${{ needs.build.outputs.artifact_id }}",
        "artifact-ids: ${{ needs.verify.outputs.artifact_id }}",
        "overwrite: false",
        "if-no-files-found: error",
    )
    for fragment in required_handoff:
        if fragment not in text:
            errors.append(f"immutable artifact handoff is missing {fragment!r}")
    if text.count("overwrite: false") != 3:
        errors.append("every one of the three artifact uploads must refuse overwrite")
    if text.count("if-no-files-found: error") != 3:
        errors.append("every artifact upload must fail when its explicit file is missing")
    if text.count("uses: actions/upload-artifact@v4") != 3:
        errors.append("release workflow requires exactly three immutable uploads")
    if text.count("uses: actions/download-artifact@v4") != 5:
        errors.append("every cross-job handoff must use the exact v4 downloader")
    if re.search(r"(?m)^\s+path:\s*[^\n]*[*?]", text):
        errors.append("release workflow artifact paths must not use wildcards")
    if re.search(r"gh release (?:create|upload)[^\n]*[*?]", text):
        errors.append("publish command must enumerate release assets without wildcards")
    return errors


def main() -> int:
    errors = validate(WORKFLOW.read_text(encoding="utf-8"))
    if errors:
        print("release workflow policy FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("release workflow policy: least privilege and immutable handoff OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
