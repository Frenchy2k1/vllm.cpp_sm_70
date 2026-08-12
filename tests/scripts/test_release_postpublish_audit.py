#!/usr/bin/env python3
"""Offline contract tests for the authenticated post-publication audit."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "scripts/release_postpublish_audit.py"
MATRIX = ROOT / "release/release-matrix.json"
VERSION = ROOT / "release/release-version.json"
SHA = "0123456789abcdef0123456789abcdef01234567"
RUN_ID = "31415926535"
REPO = "mudler/vllm.cpp"


def load():
    spec = importlib.util.spec_from_file_location("release_postpublish_audit", AUDIT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {AUDIT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PostPublishAuditContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = load()
        cls.matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
        cls.version = json.loads(VERSION.read_text(encoding="utf-8"))

    def fixture(self):
        remote_bytes: dict[str, bytes] = {}
        rows = []
        attestations = {}
        for item in self.matrix["artifacts"]:
            archive = self.audit.canonical_archive_name(
                self.version["version"], item["id"], item["archive_format"]
            )
            body = f"archive:{item['id']}".encode()
            digest = hashlib.sha256(body).hexdigest()
            checksum = f"{digest}  {archive}\n".encode()
            provenance = json.dumps(
                {"subject": [{"name": archive, "digest": {"sha256": digest}}]}
            ).encode()
            for name, data in (
                (archive, body),
                (archive + ".sha256", checksum),
                (archive + ".provenance.json", provenance),
            ):
                remote_bytes[name] = data
            rows.append(
                {
                    "archive": archive,
                    "checksum": archive + ".sha256",
                    "id": item["id"],
                    "provenance": archive + ".provenance.json",
                    "sha256": digest,
                    "size": len(body),
                }
            )
            attestations[archive] = [{
                "digest": digest,
                "repository": REPO,
                "source_sha": SHA,
                "run_id": RUN_ID,
                "verified": True,
            }]
        index = {
            "artifacts": rows,
            "prerelease": True,
            "project_version": "0.0.3",
            "release_tag": "v0.0.3-pre.1",
            "schema": "vllm.cpp.release-index.v1",
            "source_sha": SHA,
            "version": "0.0.3-pre.1",
        }
        remote_bytes["release-index.json"] = json.dumps(index).encode()
        remote_bytes["RELEASE_INDEX.md"] = (
            "# vllm.cpp v0.0.3-pre.1 binary index\n"
            f"Source: `{SHA}`\n" + "\n".join(row["archive"] for row in rows)
        ).encode()
        assets = [
            {
                "id": index + 100,
                "name": name,
                "size": len(data),
                "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
            }
            for index, (name, data) in enumerate(sorted(remote_bytes.items()))
        ]
        snapshot = {
            "tag": {"object": {"sha": SHA, "type": "commit"}},
            "release": {
                "draft": False,
                "prerelease": True,
                "tag_name": "v0.0.3-pre.1",
                "assets": assets,
            },
            "run": {
                "head_sha": SHA,
                "id": int(RUN_ID),
                "path": ".github/workflows/release.yml",
                "conclusion": "success",
            },
            "jobs": [
                {"name": name, "conclusion": "success"}
                for name in self.audit.REQUIRED_RELEASE_JOBS
            ],
        }
        return snapshot, remote_bytes, attestations

    def test_complete_authenticated_remote_fixture_passes(self) -> None:
        snapshot, remote_bytes, attestations = self.fixture()
        result = self.audit.validate_remote_release(
            snapshot, remote_bytes, attestations, self.matrix, self.version,
            REPO, SHA, RUN_ID,
        )
        self.assertEqual(result["asset_count"], 32)
        self.assertEqual(result["archive_count"], 10)

    def test_every_remote_invariant_fails_closed(self) -> None:
        mutations = {}
        snapshot, remote_bytes, attestations = self.fixture()
        mutations["tag SHA"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        mutations["tag SHA"][0]["tag"]["object"]["sha"] = "f" * 40
        mutations["release prerelease"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        mutations["release prerelease"][0]["release"]["prerelease"] = False
        mutations["release draft"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        mutations["release draft"][0]["release"]["draft"] = True
        mutations["run SHA"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        mutations["run SHA"][0]["run"]["head_sha"] = "e" * 40
        mutations["job conclusion"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        mutations["job conclusion"][0]["jobs"][0]["conclusion"] = "failure"
        mutations["extra asset"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        mutations["extra asset"][0]["release"]["assets"].append({"id": 999, "name": "extra", "size": 0, "digest": "sha256:" + "0" * 64})
        mutations["duplicate asset"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        mutations["duplicate asset"][0]["release"]["assets"].append(copy.deepcopy(snapshot["release"]["assets"][0]))
        mutations["API digest"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        mutations["API digest"][0]["release"]["assets"][0]["digest"] = None
        mutations["download bytes"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        first_name = mutations["download bytes"][0]["release"]["assets"][0]["name"]
        mutations["download bytes"][1][first_name] += b"drift"
        archive = next(name for name in remote_bytes if name.endswith((".tar.gz", ".zip")))
        mutations["checksum"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        mutations["checksum"][1][archive + ".sha256"] = ("0" * 64 + f"  {archive}\n").encode()
        mutations["provenance"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        mutations["provenance"][1][archive + ".provenance.json"] = b'{"subject": []}'
        mutations["index"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        index = json.loads(mutations["index"][1]["release-index.json"])
        index["source_sha"] = "d" * 40
        mutations["index"][1]["release-index.json"] = json.dumps(index).encode()
        mutations["attestation missing"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        mutations["attestation missing"][2][archive] = []
        mutations["attestation duplicate"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        mutations["attestation duplicate"][2][archive] *= 2
        mutations["attestation identity"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        mutations["attestation identity"][2][archive][0]["source_sha"] = "c" * 40

        for name, (mutant_snapshot, mutant_bytes, mutant_attestations) in mutations.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.audit.validate_remote_release(
                    mutant_snapshot, mutant_bytes, mutant_attestations,
                    self.matrix, self.version, REPO, SHA, RUN_ID,
                )

    def test_live_collector_uses_mocked_authenticated_apis_and_downloads(self) -> None:
        snapshot, remote_bytes, _ = self.fixture()
        release = copy.deepcopy(snapshot["release"])
        by_id = {row["id"]: row["name"] for row in release["assets"]}

        def json_side_effect(args):
            endpoint = args[-1]
            if endpoint == f"repos/{REPO}/releases/tags/v0.0.3-pre.1":
                return release
            if endpoint == f"repos/{REPO}/actions/runs/{RUN_ID}":
                return snapshot["run"]
            if endpoint.endswith("/jobs?per_page=100"):
                return [{"jobs": snapshot["jobs"]}]
            if endpoint == f"repos/{REPO}/git/ref/tags/v0.0.3-pre.1":
                return {"object": {"sha": SHA, "type": "commit"}}
            if args[:2] == ["attestation", "verify"]:
                self.assertIn(f"{REPO}/.github/workflows/release.yml", args)
                self.assertIn(SHA, args)
                self.assertIn("refs/tags/v0.0.3-pre.1", args)
                return [{"verificationResult": {"statement": {"subject": []}, "certificate": {
                    "sourceRepositoryURI": f"https://github.com/{REPO}",
                    "sourceRepositoryDigest": SHA,
                    "runInvocationURI": f"https://github.com/{REPO}/actions/runs/{RUN_ID}",
                }}}]
            raise AssertionError(args)

        def bytes_side_effect(args):
            asset_id = int(args[-1].rsplit("/", 1)[1])
            return remote_bytes[by_id[asset_id]]

        with mock.patch.object(self.audit, "gh_json", side_effect=json_side_effect), mock.patch.object(
            self.audit, "gh_bytes", side_effect=bytes_side_effect
        ):
            observed, downloads, attestations = self.audit.collect_remote(
                REPO, "v0.0.3-pre.1", SHA, RUN_ID
            )
        self.assertEqual(downloads, remote_bytes)
        result = self.audit.validate_remote_release(
            observed, downloads, attestations, self.matrix, self.version,
            REPO, SHA, RUN_ID,
        )
        self.assertEqual(result["asset_count"], 32)

    def test_annotated_and_lightweight_tags_resolve_to_the_commit(self) -> None:
        with mock.patch.object(
            self.audit, "gh_json", return_value={"object": {"sha": SHA, "type": "commit"}}
        ):
            self.assertEqual(
                self.audit.resolve_tag(REPO, "v0.0.3-pre.1"),
                {"sha": SHA, "type": "commit"},
            )
        responses = iter((
            {"object": {"sha": "a" * 40, "type": "tag"}},
            {"tag": "v0.0.3-pre.1", "object": {"sha": SHA, "type": "commit"}},
        ))
        with mock.patch.object(self.audit, "gh_json", side_effect=lambda args: next(responses)):
            self.assertEqual(
                self.audit.resolve_tag(REPO, "v0.0.3-pre.1"),
                {"sha": SHA, "type": "commit"},
            )


if __name__ == "__main__":
    unittest.main()
