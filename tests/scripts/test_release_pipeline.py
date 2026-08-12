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
RELEASE_VERSION = ROOT / "release/release-version.json"
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

    def test_release_version_declaration_is_the_single_prerelease_identity(self) -> None:
        declaration = json.loads(RELEASE_VERSION.read_text(encoding="utf-8"))
        self.assertEqual(
            declaration,
            {
                "prerelease": True,
                "project_version": "0.0.3",
                "schema": "vllm.cpp.release-version.v1",
                "tag": "v0.0.3-pre.1",
                "version": "0.0.3-pre.1",
            },
        )
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("project(vllm_cpp VERSION 0.0.3 LANGUAGES CXX)", cmake)

    def test_workflow_has_two_exact_native_windows_preview_lanes(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        for job, backend, build_dir in (
            ("cpu_windows", "cpu", "build-release-windows-cpu"),
            ("vulkan_windows", "vulkan", "build-release-windows-vulkan"),
        ):
            with self.subTest(job=job):
                block = self.checker.job_block(workflow, job)
                self.assertTrue(block)
                self.assertIn("    permissions:\n      contents: read", block)
                self.assertNotIn("write", block.split("    runs-on:", 1)[0])
                self.assertIn("    runs-on: windows-2022", block)
                self.assertNotIn("windows-latest", block)
                self.assertIn("SOURCE_SHA: ${{ github.sha }}", block)
                self.assertIn("VERSION: ${{ needs.plan.outputs.version }}", block)
                self.assertIn("./scripts/build-windows-release.ps1 -ContractTest", block)
                self.assertIn(
                    f"./scripts/build-windows-release.ps1 `\n"
                    f"            -Backend {backend} `\n"
                    f"            -ArtifactId windows-x86_64-msvc-{backend} `\n"
                    f"            -BuildDir $env:GITHUB_WORKSPACE/{build_dir}",
                    block,
                )

    def test_workflow_collects_exactly_ten_triplets_and_two_indexes(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        primary_jobs = (
            "cpu_x86", "cpu_arm64", "cpu_musl", "cuda_x86", "cuda_arm64",
            "vulkan_x86", "metal_arm64", "mlx_arm64", "cpu_windows",
            "vulkan_windows",
        )
        build = self.checker.job_block(workflow, "build")
        self.assertIn(
            "    needs: [plan, " + ", ".join(primary_jobs) + "]",
            build,
        )
        for job in primary_jobs:
            self.assertEqual(
                build.count(f"${{{{ needs.{job}.outputs.artifact_id }}}}"), 1
            )
        self.assertEqual(workflow.count(".provenance.json"), 10)
        self.assertEqual(workflow.count(".sha256"), 10)
        self.assertEqual(workflow.count("vllm.cpp-${{ needs.plan.outputs.version }}-"), 30)
        verify = self.checker.job_block(workflow, "verify")
        self.assertIn("--json-output verified/release-index.json", verify)
        self.assertIn("--markdown-output verified/RELEASE_INDEX.md", verify)

    def test_windows_workflow_contract_mutations_are_rejected(self) -> None:
        original = WORKFLOW.read_text(encoding="utf-8")
        self.assertEqual(self.checker.validate(original), [])
        mutations = {
            "missing CPU job": ("  cpu_windows:\n", "  cpu_windows_removed:\n"),
            "moving runner": ("    runs-on: windows-2022\n", "    runs-on: windows-latest\n"),
            "write authority": (
                "  cpu_windows:\n    needs: plan\n    permissions:\n      contents: read",
                "  cpu_windows:\n    needs: plan\n    permissions:\n      contents: write",
            ),
            "wrong backend": ("            -Backend cpu `", "            -Backend vulkan `"),
            "wrong tuple": (
                "            -ArtifactId windows-x86_64-msvc-cpu `",
                "            -ArtifactId windows-x86_64-msvc-cuda `",
            ),
            "source SHA removed": (
                "      - name: Build, execute, package, and validate native Windows CPU preview\n"
                "        env:\n"
                "          EVIDENCE_URL: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}\n"
                "          SOURCE_SHA: ${{ github.sha }}",
                "      - name: Build, execute, package, and validate native Windows CPU preview\n"
                "        env:\n"
                "          EVIDENCE_URL: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}\n"
                "          SOURCE_SHA: unbound",
            ),
            "ZIP renamed": (
                "windows-x86_64-msvc-cpu.zip",
                "windows-x86_64-msvc-cpu.tar.gz",
            ),
            "handoff omits Windows": (
                ",${{ needs.cpu_windows.outputs.artifact_id }},${{ needs.vulkan_windows.outputs.artifact_id }}",
                "",
            ),
        }
        for label, (before, after) in mutations.items():
            with self.subTest(label=label):
                self.assertIn(before, original)
                mutant = original.replace(before, after, 1)
                self.assertTrue(self.checker.validate(mutant), label)

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
        self.assertIn("linux-aarch64-glibc-cuda", ids)
        self.assertNotIn("rocm", " ".join(ids))
        self.assertEqual(plan["retention"]["ci_artifacts_days"], 7)

    def test_matrix_retention_policy_is_exact(self) -> None:
        matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
        matrix["retention"]["ci_artifacts_days"] = 90
        with self.assertRaises(ValueError):
            self.pipeline.validate_matrix(matrix)

    def test_existing_build_drivers_pass_explicit_tar_format(self) -> None:
        for driver in BUILD_DRIVERS:
            with self.subTest(driver=driver.name):
                text = driver.read_text(encoding="utf-8")
                self.assertIn("--archive-format tar.gz", text)

    def test_publish_matrix_contains_all_ten_primary_bundles_with_explicit_formats(self) -> None:
        matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
        artifacts = self.pipeline.validate_matrix(matrix)
        self.assertTrue(matrix["release_ready"])
        self.assertEqual(
            {item["id"]: item["channel"] for item in artifacts},
            {
                "linux-x86_64-glibc-cpu": "stable",
                "linux-aarch64-glibc-cpu": "stable",
                "linux-x86_64-musl-cpu-static": "experimental-preview",
                "linux-x86_64-glibc-cuda": "preview",
                "linux-aarch64-glibc-cuda": "preview",
                "linux-x86_64-glibc-vulkan": "preview",
                "macos-arm64-metal": "stable",
                "macos-arm64-metal-mlx": "preview",
                "windows-x86_64-msvc-cpu": "preview",
                "windows-x86_64-msvc-vulkan": "preview",
            },
        )
        self.assertTrue(all(item["required"] is True for item in artifacts))
        self.assertEqual(
            {item["archive_format"] for item in artifacts if item["id"].startswith("windows-")},
            {"zip"},
        )
        self.assertEqual(
            {item["archive_format"] for item in artifacts if not item["id"].startswith("windows-")},
            {"tar.gz"},
        )

    def test_matrix_rejects_missing_or_aliased_archive_formats(self) -> None:
        matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
        for value in (None, "tgz", "ZIP"):
            with self.subTest(value=value):
                mutant = copy.deepcopy(matrix)
                if value is None:
                    mutant["artifacts"][0].pop("archive_format")
                else:
                    mutant["artifacts"][0]["archive_format"] = value
                with self.assertRaises(ValueError):
                    self.pipeline.validate_matrix(mutant)

    def test_matrix_rejects_tuple_set_and_platform_format_mutations(self) -> None:
        matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
        mutations = []
        removed = copy.deepcopy(matrix)
        removed["artifacts"].pop()
        mutations.append(removed)
        added = copy.deepcopy(matrix)
        added["artifacts"].append({
            "archive_format": "tar.gz", "channel": "preview",
            "id": "linux-x86_64-glibc-extra", "required": True,
        })
        mutations.append(added)
        swapped = copy.deepcopy(matrix)
        swapped["artifacts"][0]["archive_format"] = "zip"
        swapped["artifacts"][-1]["archive_format"] = "tar.gz"
        mutations.append(swapped)
        for mutant in mutations:
            with self.subTest(ids=[item["id"] for item in mutant["artifacts"]]):
                with self.assertRaises(ValueError):
                    self.pipeline.validate_matrix(mutant)

    def test_windows_archive_triplet_uses_zip(self) -> None:
        self.assertEqual(
            self.pipeline.canonical_archive_name(
                "0.0.1", "windows-x86_64-msvc-cpu", "zip"
            ),
            "vllm.cpp-0.0.1-windows-x86_64-msvc-cpu.zip",
        )
        with self.assertRaises(ValueError):
            self.pipeline.canonical_archive_name("0.0.1", "artifact", "tgz")

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
                "artifacts": [{
                    "archive_format": "tar.gz", "channel": "stable",
                    "id": artifact_id, "required": True,
                }],
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

    def test_publish_rejects_missing_or_mismatched_explicit_archive_format(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = root / "assets"
            assets.mkdir()
            artifact_id = "linux-x86_64-glibc-cpu"
            archive = assets / f"vllm.cpp-0.0.1-{artifact_id}.tar.gz"
            archive.write_bytes(b"release bytes")
            digest = self.pipeline.file_sha256(archive)
            for suffix in (".sha256", ".provenance.json"):
                (assets / f"{archive.name}{suffix}").write_text("sidecar\n")
            files = [
                {"name": path.name, "sha256": self.pipeline.file_sha256(path),
                 "size": path.stat().st_size}
                for path in sorted(assets.iterdir())
            ]
            index_json = root / "release-index.json"
            self.pipeline.write_json(index_json, {
                "artifacts": [{"archive": archive.name, "id": artifact_id,
                               "sha256": digest}],
                "release_tag": "v0.0.1", "schema": "vllm.cpp.release-index.v1",
                "source_sha": SHA,
            })
            index_md = root / "RELEASE_INDEX.md"
            index_md.write_text(f"v0.0.1\n{SHA}\n{archive.name}\n")
            for artifacts in ([], [{
                "archive_format": "zip", "channel": "stable",
                "id": artifact_id, "required": True,
            }]):
                handoff = root / "handoff.json"
                self.pipeline.write_json(handoff, {
                    "artifacts": artifacts, "files": files, "publish": True,
                    "release_tag": "v0.0.1", "source_sha": SHA,
                    "verified": True, "version": "0.0.1",
                })
                with self.subTest(artifacts=artifacts), \
                        mock.patch.object(self.pipeline.subprocess, "run") as run, \
                        self.assertRaises(ValueError):
                    self.pipeline.publish_release(handoff, assets, index_json, index_md, "v0.0.1")
                run.assert_not_called()

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
                    "release workflow must bind each handoff stage to its declared root",
                    self.checker.validate(mutant),
                )

    def test_asset_download_root_cannot_be_compensated_by_plan_download(self) -> None:
        original = WORKFLOW.read_text(encoding="utf-8")
        plan_download = """          artifact-ids: ${{ needs.plan.outputs.artifact_id }}
          path: plan
          merge-multiple: true"""
        asset_download = """          artifact-ids: ${{ needs.cpu_x86.outputs.artifact_id }},${{ needs.cpu_arm64.outputs.artifact_id }},${{ needs.cpu_musl.outputs.artifact_id }},${{ needs.cuda_x86.outputs.artifact_id }},${{ needs.cuda_arm64.outputs.artifact_id }},${{ needs.vulkan_x86.outputs.artifact_id }},${{ needs.metal_arm64.outputs.artifact_id }},${{ needs.mlx_arm64.outputs.artifact_id }},${{ needs.cpu_windows.outputs.artifact_id }},${{ needs.vulkan_windows.outputs.artifact_id }}
          path: release-assets
          merge-multiple: true"""
        self.assertIn(plan_download, original)
        self.assertIn(asset_download, original)

        mutant = original.replace(
            plan_download,
            plan_download.replace("path: plan", "path: release-assets"),
            1,
        ).replace(
            asset_download,
            asset_download.replace("path: release-assets", "path: assets"),
            1,
        )
        self.assertIn(
            "release workflow must bind each handoff stage to its declared root",
            self.checker.validate(mutant),
        )

    def test_publish_asset_root_cannot_be_compensated_by_an_inert_comment(self) -> None:
        original = WORKFLOW.read_text(encoding="utf-8")
        publish_command = (
            "          python3 scripts/release_pipeline.py publish \\\n"
            "            --handoff verified/verified-handoff.json \\\n"
            "            --assets-dir verified/release-assets \\\n"
            "            --index-json verified/release-index.json \\\n"
            "            --index-markdown verified/RELEASE_INDEX.md \\\n"
            '            --tag "$tag"'
        )
        self.assertIn(publish_command, original)
        mutant = original.replace(
            publish_command,
            publish_command.replace(
                "            --assets-dir verified/release-assets \\",
                "            # Inert compensation: --assets-dir verified/release-assets\n"
                "            --assets-dir verified/assets \\",
            ),
            1,
        )
        self.assertIn(
            "release workflow must bind each handoff stage to its declared root",
            self.checker.validate(mutant),
        )

    def test_publish_binding_ignores_inert_shell_text_and_duplicates(self) -> None:
        original = WORKFLOW.read_text(encoding="utf-8")
        publish_command = (
            "          python3 scripts/release_pipeline.py publish \\\n"
            "            --handoff verified/verified-handoff.json \\\n"
            "            --assets-dir verified/release-assets \\\n"
            "            --index-json verified/release-index.json \\\n"
            "            --index-markdown verified/RELEASE_INDEX.md \\\n"
            '            --tag "$tag"'
        )
        good_option = "            --assets-dir verified/release-assets \\"
        bad_option = "            --assets-dir verified/assets \\"
        wrong_publish = publish_command.replace(good_option, bad_option)
        plan_command = "          python3 scripts/release_pipeline.py plan \\"
        flat_publish = (
            "python3 scripts/release_pipeline.py publish "
            "--handoff verified/verified-handoff.json "
            "--assets-dir verified/release-assets "
            "--index-json verified/release-index.json "
            "--index-markdown verified/RELEASE_INDEX.md --tag \"$tag\""
        )
        self.assertIn(publish_command, original)
        self.assertIn(plan_command, original)

        mutants = {
            "inline comment": original.replace(
                publish_command,
                publish_command.replace(
                    good_option,
                    "            --assets-dir verified/assets"
                    " # --assets-dir verified/release-assets",
                ),
                1,
            ),
            "echo": original.replace(
                publish_command,
                wrong_publish
                + '\n          echo "--assets-dir verified/release-assets"',
                1,
            ),
            "quoted assignment": original.replace(
                publish_command,
                wrong_publish
                + "\n          note='--assets-dir verified/release-assets'",
                1,
            ),
            "unrelated command": original.replace(
                publish_command,
                wrong_publish
                + "\n          python3 scripts/other.py"
                " --assets-dir verified/release-assets",
                1,
            ),
            "duplicate option": original.replace(
                publish_command,
                publish_command.replace(
                    good_option, good_option + "\n" + bad_option
                ),
                1,
            ),
            "duplicate publisher": original.replace(
                publish_command, publish_command + "\n" + publish_command, 1
            ),
            "plan step": original.replace(
                publish_command, wrong_publish, 1
            ).replace(
                plan_command, publish_command + "\n" + plan_command, 1
            ),
            "later control segment": original.replace(
                publish_command,
                wrong_publish + "\n          true && " + flat_publish,
                1,
            ),
            "publisher followed by control": original.replace(
                publish_command, publish_command + " && true", 1
            ),
        }
        for attack, mutant in mutants.items():
            with self.subTest(attack=attack):
                self.assertIn(
                    "release workflow must bind each handoff stage to its declared root",
                    self.checker.validate(mutant),
                )

    def test_gh_release_bypass_check_only_matches_executable_commands(self) -> None:
        original = WORKFLOW.read_text(encoding="utf-8")
        anchor = '            --tag "$tag"'
        self.assertIn(anchor, original)
        for inert in (
            "          # gh release create --notes harmless",
            '          echo "gh release upload --clobber harmless"',
            "          note='gh release create harmless'",
            '          note="gh release upload harmless"',
            "          note='$(gh release create harmless)'",
            "          note='`gh release upload harmless`'",
            '          note="\\$(gh release create harmless)"',
        ):
            with self.subTest(inert=inert):
                mutant = original.replace(anchor, anchor + "\n" + inert, 1)
                self.assertEqual(self.checker.validate(mutant), [])

        for command in (
            '          gh release create "$tag" verified/release-index.json',
            '          gh release upload "$tag" verified/release-index.json',
        ):
            with self.subTest(command=command):
                mutant = original.replace(anchor, anchor + "\n" + command, 1)
                self.assertIn(
                    "release workflow must not bypass the byte-bound publisher",
                    self.checker.validate(mutant),
                )

    def test_gh_release_bypass_cannot_hide_in_shell_control_or_substitution(self) -> None:
        original = WORKFLOW.read_text(encoding="utf-8")
        anchor = '            --tag "$tag"'
        self.assertIn(anchor, original)
        for command in (
            '          true && gh release create "$tag" verified/release-index.json',
            '          true; gh release upload "$tag" verified/release-index.json',
            '          note="$(gh release create "$tag")"',
            '          false || gh release upload "$tag" verified/release-index.json',
            '          printf x | gh release create "$tag" verified/release-index.json',
            '          (gh release create "$tag" verified/release-index.json)',
            '          { gh release upload "$tag" verified/release-index.json; }',
            '          gh release create "$tag" verified/release-index.json && true',
            '          true && gh release create "$tag" one; gh release upload "$tag" two',
            '          note=`gh release upload "$tag" verified/release-index.json`',
            '          cat <(gh release create "$tag" verified/release-index.json)',
        ):
            with self.subTest(command=command):
                mutant = original.replace(anchor, anchor + "\n" + command, 1)
                self.assertIn(
                    "release workflow must not bypass the byte-bound publisher",
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
                "needs: [plan, cpu_x86, cpu_arm64, cpu_musl, cuda_x86, cuda_arm64, vulkan_x86, metal_arm64, mlx_arm64, cpu_windows, vulkan_windows]",
                "needs: [plan, cpu_x86, cpu_arm64, cpu_musl, cuda_x86, cuda_arm64, vulkan_x86, metal_arm64, mlx_arm64, cpu_windows]",
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
