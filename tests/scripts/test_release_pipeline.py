#!/usr/bin/env python3
"""W8 dry-run/tag planning, immutable handoff, and permission mutations."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "scripts/release_pipeline.py"
CHECKER = ROOT / "scripts/check-release-workflow.py"
WORKFLOW = ROOT / ".github/workflows/release.yml"
BUILD_DRIVERS = (
    ROOT / "scripts/build-cpu-release.sh",
    ROOT / "scripts/build-linux-accelerator-release.sh",
    ROOT / "scripts/build-macos-release.sh",
)
MATRIX = ROOT / "release/release-matrix.json"
SHA = "0123456789abcdef0123456789abcdef01234567"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleasePipelineContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = load(PIPELINE, "release_pipeline")
        cls.checker = load(CHECKER, "check_release_workflow")

    def plan(self, event: str, ref: str, release_ready: bool = False):
        matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
        matrix["release_ready"] = release_ready
        return self.pipeline.make_plan(event, ref, SHA, "0.0.1", matrix)

    def test_manual_dispatch_is_always_a_non_publishing_dry_run(self) -> None:
        plan = self.plan("workflow_dispatch", "refs/heads/main", release_ready=True)
        self.assertFalse(plan["publish"])
        self.assertEqual(plan["release_tag"], f"dry-run-{SHA[:12]}")
        self.assertEqual(plan["source_sha"], SHA)

    def test_tag_publish_requires_exact_version_and_ready_matrix(self) -> None:
        self.assertFalse(self.plan("push", "refs/tags/v0.0.1")["publish"])
        self.assertTrue(self.plan("push", "refs/tags/v0.0.1", release_ready=True)["publish"])
        for ref in ("refs/tags/v0.0.2", "refs/heads/v0.0.1", "refs/tags/0.0.1"):
            with self.subTest(ref=ref):
                with self.assertRaises(ValueError):
                    self.plan("push", ref, release_ready=True)

    def test_only_explicit_matrix_artifacts_enter_the_plan(self) -> None:
        plan = self.plan("workflow_dispatch", "refs/heads/topic")
        ids = [item["id"] for item in plan["artifacts"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("linux-x86_64-glibc-cpu", ids)
        self.assertIn("linux-aarch64-glibc-cuda-fat", ids)
        self.assertNotIn("rocm", " ".join(ids))
        self.assertEqual(plan["retention"]["ci_artifacts_days"], 7)

    def test_matrix_retention_policy_is_exact(self) -> None:
        matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
        matrix["retention"]["ci_artifacts_days"] = 90
        with self.assertRaises(ValueError):
            self.pipeline.validate_matrix(matrix)

    def test_publish_matrix_contains_all_eight_primary_bundles(self) -> None:
        matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
        artifacts = self.pipeline.validate_matrix(matrix)
        self.assertTrue(matrix["release_ready"])
        self.assertEqual(
            {item["id"]: item["channel"] for item in artifacts},
            {
                "linux-x86_64-glibc-cpu": "stable",
                "linux-aarch64-glibc-cpu": "stable",
                "linux-x86_64-musl-cpu-static": "experimental-preview",
                "linux-x86_64-glibc-cuda-fat": "preview",
                "linux-aarch64-glibc-cuda-fat": "preview",
                "linux-x86_64-glibc-vulkan": "preview",
                "macos-arm64-metal": "stable",
                "macos-arm64-metal-mlx": "preview",
            },
        )
        self.assertTrue(all(item["required"] is True for item in artifacts))

    def test_handoff_digest_and_source_sha_are_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "release-plan.json"
            handoff_path = root / "release-handoff.json"
            verified_path = root / "verified-handoff.json"
            assets = root / "assets"
            assets.mkdir()
            archive = assets / "vllm.cpp-0.0.1-linux-x86_64-glibc-cpu.tar.gz"
            archive.write_bytes(b"release bytes")
            digest = self.pipeline.file_sha256(archive)
            (assets / f"{archive.name}.sha256").write_text(
                f"{digest}  {archive.name}\n", encoding="utf-8"
            )
            (assets / f"{archive.name}.provenance.json").write_text(
                json.dumps({"subject": [{"name": archive.name, "digest": {"sha256": digest}}]}),
                encoding="utf-8",
            )
            self.pipeline.write_json(plan_path, self.plan("workflow_dispatch", "refs/heads/main"))
            self.pipeline.make_handoff(plan_path, assets, handoff_path)
            self.pipeline.verify_handoff(plan_path, handoff_path, assets, verified_path, SHA)
            verified = json.loads(verified_path.read_text())
            self.assertTrue(verified["verified"])
            self.assertEqual([item["name"] for item in verified["files"]], sorted(path.name for path in assets.iterdir()))
            mutant = json.loads(handoff_path.read_text())
            mutant["source_sha"] = "f" * 40
            self.pipeline.write_json(handoff_path, mutant)
            with self.assertRaises(ValueError):
                self.pipeline.verify_handoff(plan_path, handoff_path, assets, verified_path, SHA)
            self.pipeline.make_handoff(plan_path, assets, handoff_path)
            archive.write_bytes(b"mutated")
            with self.assertRaises(ValueError):
                self.pipeline.verify_handoff(plan_path, handoff_path, assets, verified_path, SHA)

    def test_publish_ready_plan_requires_every_required_asset_triplet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = self.plan("push", "refs/tags/v0.0.1", release_ready=True)
            plan_path = root / "plan.json"
            self.pipeline.write_json(plan_path, plan)
            assets = root / "assets"
            assets.mkdir()
            with self.assertRaises(ValueError):
                self.pipeline.make_handoff(plan_path, assets, root / "handoff.json")

    def test_handoff_accepts_only_versioned_matrix_asset_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = self.plan("workflow_dispatch", "refs/heads/main")
            plan_path = root / "plan.json"
            self.pipeline.write_json(plan_path, plan)
            assets = root / "assets"
            assets.mkdir()
            artifact_id = "linux-x86_64-glibc-cpu"
            archive = assets / f"vllm.cpp-{plan['version']}-{artifact_id}.tar.gz"
            archive.write_bytes(b"release bytes")
            digest = self.pipeline.file_sha256(archive)
            (assets / f"{archive.name}.sha256").write_text(
                f"{digest}  {archive.name}\n", encoding="utf-8"
            )
            (assets / f"{archive.name}.provenance.json").write_text(
                "{}\n", encoding="utf-8"
            )
            handoff_path = root / "handoff.json"
            self.pipeline.make_handoff(plan_path, assets, handoff_path)
            files = json.loads(handoff_path.read_text(encoding="utf-8"))["files"]
            self.assertEqual({item["artifact_id"] for item in files}, {artifact_id})

            # A checkout-owned file copied into the transient root is not a
            # release asset and must remain a hard failure. The workflow fixes
            # that collision by using a disjoint root, not by weakening this
            # exact-inventory validator.
            (assets / "favicon.png").write_bytes(b"checkout asset")
            with self.assertRaises(ValueError):
                self.pipeline.make_handoff(plan_path, assets, handoff_path)

    def test_publish_enumerates_only_verified_assets_without_shell_globs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = root / "assets"
            assets.mkdir()
            artifact_id = "linux-x86_64-glibc-cpu"
            archive = assets / f"vllm.cpp-0.0.1-{artifact_id}.tar.gz"
            archive.write_bytes(b"release bytes")
            digest = self.pipeline.file_sha256(archive)
            for suffix, content in (
                (".sha256", f"{digest}  {archive.name}\n"),
                (".provenance.json", "{}\n"),
            ):
                (assets / f"{archive.name}{suffix}").write_text(content)
            handoff = {
                "files": [
                    {"name": path.name, "sha256": self.pipeline.file_sha256(path), "size": path.stat().st_size}
                    for path in sorted(assets.iterdir())
                ],
                "publish": True,
                "release_tag": "v0.0.1",
                "source_sha": SHA,
                "verified": True,
                "version": "0.0.1",
            }
            handoff_path = root / "verified-handoff.json"
            index_json = root / "release-index.json"
            index_md = root / "RELEASE_INDEX.md"
            self.pipeline.write_json(handoff_path, handoff)
            self.pipeline.write_json(index_json, {
                "artifacts": [{
                    "archive": archive.name,
                    "id": artifact_id,
                    "sha256": digest,
                }],
                "release_tag": "v0.0.1",
                "schema": "vllm.cpp.release-index.v1",
                "source_sha": SHA,
            })
            index_md.write_text(
                f"# release v0.0.1\n\nSource: `{SHA}`\n\n{archive.name}\n"
            )
            with mock.patch.object(self.pipeline.subprocess, "run") as run:
                self.pipeline.publish_release(
                    handoff_path, assets, index_json, index_md, "v0.0.1"
                )
            argv = run.call_args.args[0]
            self.assertEqual(argv[:4], ["gh", "release", "create", "v0.0.1"])
            self.assertFalse(any("*" in value or "?" in value for value in argv))
            self.assertIn(str(archive), argv)
            run.assert_called_once()

            index = json.loads(index_json.read_text())
            index["source_sha"] = "f" * 40
            self.pipeline.write_json(index_json, index)
            with self.assertRaises(ValueError):
                self.pipeline.publish_release(
                    handoff_path, assets, index_json, index_md, "v0.0.1"
                )

    def test_publish_rejects_unverified_drift_and_extra_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = root / "assets"
            assets.mkdir()
            archive = assets / "asset.tar.gz"
            archive.write_bytes(b"bytes")
            handoff = {
                "files": [{
                    "name": archive.name,
                    "sha256": self.pipeline.file_sha256(archive),
                    "size": archive.stat().st_size,
                }],
                "publish": True,
                "release_tag": "v0.0.1",
                "verified": True,
                "version": "0.0.1",
            }
            handoff_path = root / "verified-handoff.json"
            index_json = root / "release-index.json"
            index_md = root / "RELEASE_INDEX.md"
            self.pipeline.write_json(handoff_path, handoff)
            index_json.write_text("{}\n")
            index_md.write_text("# release\n")
            (assets / "unexpected").write_text("no")
            with self.assertRaises(ValueError):
                self.pipeline.publish_release(
                    handoff_path, assets, index_json, index_md, "v0.0.1"
                )
            (assets / "unexpected").unlink()
            archive.write_bytes(b"drift")
            with self.assertRaises(ValueError):
                self.pipeline.publish_release(
                    handoff_path, assets, index_json, index_md, "v0.0.1"
                )

    def test_cli_dry_run_never_calls_github_or_creates_a_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "plan.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(PIPELINE),
                    "plan",
                    "--event",
                    "workflow_dispatch",
                    "--ref",
                    "refs/heads/main",
                    "--sha",
                    SHA,
                    "--version",
                    "0.0.1",
                    "--matrix",
                    str(MATRIX),
                    "--output",
                    str(output),
                ],
                text=True,
                capture_output=True,
                check=False,
                env={"PATH": "/nonexistent"},
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse(json.loads(output.read_text())["publish"])

    def test_workflow_has_exact_least_privilege_stage_boundaries(self) -> None:
        errors = self.checker.validate(WORKFLOW.read_text(encoding="utf-8"))
        self.assertEqual(errors, [])

    def test_all_archive_producers_use_the_canonical_versioned_name(self) -> None:
        assignment = 'archive="$release_dir/vllm.cpp-$VERSION-$artifact_id.tar.gz"'
        for driver in BUILD_DRIVERS:
            with self.subTest(driver=driver.name):
                self.assertIn(assignment, driver.read_text(encoding="utf-8"))

        workflow = WORKFLOW.read_text(encoding="utf-8")
        artifact_id = "linux-x86_64-glibc-cpu"
        canonical = (
            "build-release-cpu-x86/release/"
            "vllm.cpp-${{ needs.plan.outputs.version }}-"
            f"{artifact_id}.tar.gz"
        )
        self.assertIn(canonical, workflow)
        mutant = workflow.replace(canonical, canonical.replace("vllm.cpp-${{ needs.plan.outputs.version }}-", ""), 1)
        self.assertIn(
            "every release upload path must use its canonical versioned archive name",
            self.checker.validate(mutant),
        )

    def test_every_artifact_download_uses_flat_extraction(self) -> None:
        original = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("          merge-multiple: true", original)
        mutant = original.replace(
            "          merge-multiple: true",
            "          merge-multiple: false",
            1,
        )
        self.assertIn(
            "every artifact download must flatten into its declared path",
            self.checker.validate(mutant),
        )

    def test_every_release_stage_uses_a_checkout_disjoint_asset_root(self) -> None:
        original = WORKFLOW.read_text(encoding="utf-8")
        required = (
            "          path: release-assets",
            "            --assets-dir release-assets \\",
            "            release-assets",
            "          cp -a unverified/release-assets verified/release-assets",
            "            --assets-dir verified/release-assets \\",
            "          subject-path: verified/release-assets/**",
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, original)
                mutant = original.replace(fragment, fragment.replace("release-assets", "assets"), 1)
                self.assertIn(
                    "release workflow must isolate transient release assets from checkout assets",
                    self.checker.validate(mutant),
                )

    def test_flat_extraction_cannot_be_compensated_by_an_upload(self) -> None:
        original = WORKFLOW.read_text(encoding="utf-8")
        mutant = original.replace(
            "          merge-multiple: true",
            "          merge-multiple: false",
            1,
        ).replace(
            "          overwrite: false",
            "          merge-multiple: true\n          overwrite: false",
            1,
        )
        self.assertIn(
            "every artifact download must flatten into its declared path",
            self.checker.validate(mutant),
        )

    def test_flat_extraction_cannot_be_spoofed_by_sibling_env(self) -> None:
        original = WORKFLOW.read_text(encoding="utf-8")
        download = (
            "      - name: Download the exact immutable plan by ID\n"
            "        uses: actions/download-artifact@v4\n"
            "        with:\n"
            "          artifact-ids: ${{ needs.plan.outputs.artifact_id }}\n"
            "          path: plan\n"
            "          merge-multiple: true"
        )
        spoofed = (
            "      - name: Download the exact immutable plan by ID\n"
            "        uses: actions/download-artifact@v4\n"
            "        env:\n"
            "          merge-multiple: true\n"
            "        with:\n"
            "          artifact-ids: ${{ needs.plan.outputs.artifact_id }}\n"
            "          path: plan"
        )
        self.assertEqual(original.count(download), 1)
        mutant = original.replace(download, spoofed, 1)
        self.assertIn(
            "every artifact download must flatten into its declared path",
            self.checker.validate(mutant),
        )

    def test_hosted_packagers_resolve_their_runtime_dependencies(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn(
            "apk add --no-cache bash binutils build-base cmake file gcompat ninja python3 qemu-x86_64",
            workflow,
        )
        self.assertNotIn("mlx.__file__", workflow)
        self.assertIn('d.locate_file("mlx")', workflow)
        musl = self.checker.job_block(workflow, "cpu_musl")
        x86 = self.checker.job_block(workflow, "cpu_x86")
        self.assertIn('scripts/install-intel-sde.sh "$RUNNER_TEMP/intel-sde"', x86)
        self.assertIn('"$RUNNER_TEMP/intel-sde/sde64"', x86)
        self.assertIn('scripts/install-intel-sde.sh "$RUNNER_TEMP/intel-sde"', musl)
        self.assertIn('-v "$RUNNER_TEMP/intel-sde:/intel-sde:ro"', musl)
        self.assertIn("/intel-sde/sde64", musl)
        self.assertIn(
            "SOURCE_DATE_EPOCH=$(git show -s --format=%ct HEAD)\n"
            "          export SOURCE_DATE_EPOCH",
            musl,
        )
        for job in ("cuda_x86", "cuda_arm64"):
            with self.subTest(job=job):
                block = self.checker.job_block(workflow, job)
                self.assertIn(
                    'git config --global --add safe.directory "$GITHUB_WORKSPACE"',
                    block,
                )

    def test_security_critical_workflow_mutations_fail(self) -> None:
        original = WORKFLOW.read_text(encoding="utf-8")
        mutations = {
            "pull request trigger": ("  workflow_dispatch: {}", "  pull_request: {}\n  workflow_dispatch: {}"),
            "global write": ("permissions:\n  contents: read", "permissions:\n  contents: write"),
            "mutable upload": ("overwrite: false", "overwrite: true"),
            "primary tuple omitted from handoff": (
                "needs: [plan, cpu_x86, cpu_arm64, cpu_musl, cuda_x86, cuda_arm64, vulkan_x86, metal_arm64, mlx_arm64]",
                "needs: [plan, cpu_x86, cpu_arm64, cpu_musl, cuda_x86, cuda_arm64, vulkan_x86, metal_arm64]",
            ),
            "name not SHA-bound": ("release-unverified-${{ github.sha }}", "release-unverified"),
            "name download": ("artifact-ids: ${{ needs.build.outputs.artifact_id }}", "name: release-unverified"),
            "attest no OIDC": ("      id-token: write", "      id-token: none"),
            "attest no repository grant": ("      attestations: write", "      attestations: none"),
            "attest no metadata grant": ("      artifact-metadata: write", "      artifact-metadata: none"),
            "publish no environment": ("    environment: release", "    # environment removed"),
            "publish broad dependency": ("    needs: [plan, verify, attest]", "    needs: [plan, build, verify, attest]"),
            "publish no checkout": (
                "    steps:\n      - uses: actions/checkout@v4\n      - name: Download only the verified handoff by ID",
                "    steps:\n      - name: Download only the verified handoff by ID",
            ),
            "publish bypasses byte binding": (
                "python3 scripts/release_pipeline.py publish",
                "python3 -c 'pass'",
            ),
            "publish not tag gated": (
                "startsWith(github.ref, 'refs/tags/v')",
                "startsWith(github.ref, 'refs/heads/')",
            ),
            "continue on error": ("    runs-on: ubuntu-latest", "    continue-on-error: true\n    runs-on: ubuntu-latest"),
        }
        for label, (before, after) in mutations.items():
            with self.subTest(label=label):
                self.assertIn(before, original)
                mutant = original.replace(before, after, 1)
                self.assertTrue(self.checker.validate(mutant), label)


if __name__ == "__main__":
    unittest.main()
