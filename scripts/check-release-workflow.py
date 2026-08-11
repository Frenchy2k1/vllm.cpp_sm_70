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


def action_steps(text: str, action: str) -> list[str]:
    starts = [match.start() for match in re.finditer(r"(?m)^      - ", text)]
    starts.append(len(text))
    return [
        text[start:end]
        for start, end in zip(starts, starts[1:])
        if re.search(rf"(?m)^        uses: {re.escape(action)}$", text[start:end])
    ]


def step_mapping_blocks(step: str, name: str) -> list[str]:
    lines = step.splitlines()
    blocks: list[str] = []
    for index, line in enumerate(lines):
        if line != f"        {name}:":
            continue
        end = index + 1
        while end < len(lines) and not re.match(r"^        \S", lines[end]):
            end += 1
        blocks.append("\n".join(lines[index + 1 : end]))
    return blocks


def mapping_value(step: str, mapping: str, key: str) -> str | None:
    blocks = step_mapping_blocks(step, mapping)
    if len(blocks) != 1:
        return None
    values = re.findall(
        rf"(?m)^          {re.escape(key)}:\s*([^\n]+?)\s*$", blocks[0]
    )
    return values[0] if len(values) == 1 else None


def mapping_paths(step: str) -> list[str]:
    blocks = step_mapping_blocks(step, "with")
    if len(blocks) != 1:
        return []
    lines = blocks[0].splitlines()
    for index, line in enumerate(lines):
        match = re.fullmatch(r"          path:\s*(.*)", line)
        if match is None:
            continue
        if match.group(1) != "|":
            return [match.group(1)]
        paths: list[str] = []
        for candidate in lines[index + 1 :]:
            if not candidate.startswith("            "):
                break
            paths.append(candidate.strip())
        return paths
    return []


def workflow_steps(block: str) -> list[str]:
    starts = [match.start() for match in re.finditer(r"(?m)^      - ", block)]
    starts.append(len(block))
    return [block[start:end] for start, end in zip(starts, starts[1:])]


def named_step(block: str, name: str) -> str | None:
    matches = [
        step
        for step in workflow_steps(block)
        if re.search(rf"(?m)^      - name: {re.escape(name)}$", step)
    ]
    return matches[0] if len(matches) == 1 else None


def download_is_bound(block: str, artifact_ids: str, path: str) -> bool:
    matches = [
        step
        for step in action_steps(block, "actions/download-artifact@v4")
        if mapping_value(step, "with", "artifact-ids") == artifact_ids
    ]
    return len(matches) == 1 and mapping_value(matches[0], "with", "path") == path


def upload_is_bound(block: str, artifact_name: str, paths: list[str]) -> bool:
    matches = [
        step
        for step in action_steps(block, "actions/upload-artifact@v4")
        if mapping_value(step, "with", "name") == artifact_name
    ]
    return len(matches) == 1 and mapping_paths(matches[0]) == paths


def step_has_fragments(block: str, name: str, fragments: tuple[str, ...]) -> bool:
    step = named_step(block, name)
    return step is not None and all(fragment in step for fragment in fragments)


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

    primary_build_jobs = (
        "cpu_x86",
        "cpu_arm64",
        "cpu_musl",
        "cuda_x86",
        "cuda_arm64",
        "vulkan_x86",
        "metal_arm64",
        "mlx_arm64",
    )
    release_outputs = (
        ("build-release-cpu-x86", "linux-x86_64-glibc-cpu"),
        ("build-release-cpu-arm64", "linux-aarch64-glibc-cpu"),
        ("build-release-cpu-musl", "linux-x86_64-musl-cpu-static"),
        ("build-release-cuda-x86", "linux-x86_64-glibc-cuda-fat"),
        ("build-release-cuda-arm64", "linux-aarch64-glibc-cuda-fat"),
        ("build-release-vulkan-x86", "linux-x86_64-glibc-vulkan"),
        ("build-release-metal-arm64", "macos-arm64-metal"),
        ("build-release-mlx-arm64", "macos-arm64-metal-mlx"),
    )
    read_only_jobs = (
        "plan",
        *primary_build_jobs,
        "build",
        "verify",
    )
    blocks = {
        name: job_block(text, name)
        for name in (*read_only_jobs, "attest", "publish")
    }
    for name, block in blocks.items():
        if not block:
            errors.append(f"release workflow is missing {name} job")
    for name in read_only_jobs:
        block = blocks[name]
        if "    permissions:\n      contents: read" not in block:
            errors.append(f"{name} job must have contents: read only")
        permission_block = re.search(r"(?m)^    permissions:\n((?:      [^\n]*\n)+)", block)
        if permission_block and "write" in permission_block.group(1):
            errors.append(f"{name} job unexpectedly has write permission")

    build = blocks["build"]
    expected_needs = "    needs: [plan, " + ", ".join(primary_build_jobs) + "]"
    if expected_needs not in build:
        errors.append("handoff build job must depend on every primary release tuple")
    for name in primary_build_jobs:
        reference = f"${{{{ needs.{name}.outputs.artifact_id }}}}"
        if reference not in build:
            errors.append(f"handoff build job does not consume immutable {name} output")
    for build_dir, artifact_id in release_outputs:
        archive = (
            f"{build_dir}/release/vllm.cpp-${{{{ needs.plan.outputs.version }}}}-"
            f"{artifact_id}.tar.gz"
        )
        if any(
            f"            {archive}{suffix}\n" not in text
            for suffix in ("", ".sha256", ".provenance.json")
        ):
            errors.append(
                "every release upload path must use its canonical versioned archive name"
            )

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
    if "uses: actions/checkout@v4" not in publish:
        errors.append("publish job must check out the exact tagged publisher implementation")
    if tag_gate not in publish or "needs.plan.outputs.publish == 'true'" not in publish:
        errors.append("publish job must require an exact tag and approved publish plan")
    if "scripts/release_pipeline.py publish" not in publish:
        errors.append("publish job must use the byte-bound release publisher")
    if "verified/release-assets" not in publish or "verified/release-index.json" not in publish:
        errors.append("publish job must release verified assets and generated indexes")
    if "gh release create" in publish:
        errors.append("release workflow must not bypass the byte-bound publisher")

    required_handoff = (
        "name: release-plan-${{ github.sha }}",
        "name: release-unverified-${{ github.sha }}",
        "name: release-verified-${{ github.sha }}",
        "artifact-ids: ${{ needs.plan.outputs.artifact_id }}",
        "artifact-ids: ${{ needs.build.outputs.artifact_id }}",
        "artifact-ids: ${{ needs.verify.outputs.artifact_id }}",
        "overwrite: false",
        "if-no-files-found: error",
        "python3 scripts/release_index.py",
    )
    for fragment in required_handoff:
        if fragment not in text:
            errors.append(f"immutable artifact handoff is missing {fragment!r}")
    plan_artifact = "${{ needs.plan.outputs.artifact_id }}"
    primary_artifacts = ",".join(
        f"${{{{ needs.{name}.outputs.artifact_id }}}}"
        for name in primary_build_jobs
    )
    verified_artifact = "${{ needs.verify.outputs.artifact_id }}"
    verify = blocks["verify"]
    root_contracts = (
        download_is_bound(build, plan_artifact, "plan"),
        download_is_bound(build, primary_artifacts, "release-assets"),
        step_has_fragments(
            build,
            "Produce the byte-bound release handoff",
            (
                "--plan plan/release-plan.json",
                "--assets-dir release-assets",
                "--output release-handoff.json",
            ),
        ),
        upload_is_bound(
            build,
            "release-unverified-${{ github.sha }}",
            ["release-handoff.json", "release-assets"],
        ),
        download_is_bound(verify, plan_artifact, "plan"),
        download_is_bound(
            verify, "${{ needs.build.outputs.artifact_id }}", "unverified"
        ),
        step_has_fragments(
            verify,
            "Verify handoff against plan and workflow SHA",
            (
                "cp -a unverified/release-assets verified/release-assets",
                "--plan plan/release-plan.json",
                "--handoff unverified/release-handoff.json",
                "--assets-dir verified/release-assets",
                "--output verified/verified-handoff.json",
            ),
        ),
        step_has_fragments(
            verify,
            "Generate byte-derived release indexes",
            (
                "--assets-dir verified/release-assets",
                "--handoff verified/verified-handoff.json",
                "--json-output verified/release-index.json",
                "--markdown-output verified/RELEASE_INDEX.md",
            ),
        ),
        upload_is_bound(
            verify, "release-verified-${{ github.sha }}", ["verified"]
        ),
        download_is_bound(attest, verified_artifact, "verified"),
        step_has_fragments(
            attest,
            "Attest verified bytes",
            ("subject-path: verified/release-assets/**",),
        ),
        download_is_bound(publish, verified_artifact, "verified"),
        step_has_fragments(
            publish,
            "Publish the exact verified release assets",
            (
                "--handoff verified/verified-handoff.json",
                "--assets-dir verified/release-assets",
                "--index-json verified/release-index.json",
                "--index-markdown verified/RELEASE_INDEX.md",
            ),
        ),
    )
    if not all(root_contracts):
        errors.append(
            "release workflow must bind each handoff stage to its declared root"
        )
    uploads = text.count("uses: actions/upload-artifact@v4")
    download_steps = action_steps(text, "actions/download-artifact@v4")
    downloads = len(download_steps)
    if uploads < 3:
        errors.append("release workflow requires immutable plan, asset, and verified uploads")
    if text.count("overwrite: false") != uploads:
        errors.append("every artifact upload must refuse overwrite")
    if text.count("if-no-files-found: error") != uploads:
        errors.append("every artifact upload must fail when its explicit file is missing")
    if downloads < 5 or text.count("artifact-ids:") != downloads:
        errors.append("every cross-job handoff must use an exact immutable artifact ID")
    flatten_values = []
    for step in download_steps:
        with_blocks = step_mapping_blocks(step, "with")
        flatten_values.append(
            re.findall(
                r"(?m)^          merge-multiple:\s*(true|'true'|\"true\")\s*$",
                with_blocks[0],
            )
            if len(with_blocks) == 1
            else []
        )
    if any(values not in (["true"], ["'true'"], ['"true"']) for values in flatten_values):
        errors.append("every artifact download must flatten into its declared path")
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
