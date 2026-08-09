#!/usr/bin/env python3
"""Plan and verify immutable W8 release workflow handoffs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


PLAN_SCHEMA = "vllm.cpp.release-plan.v1"
HANDOFF_SCHEMA = "vllm.cpp.release-handoff.v1"
MATRIX_SCHEMA = "vllm.cpp.release-matrix.v1"
CHANNELS = {"stable", "preview", "experimental-preview"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_matrix(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    if matrix.get("schema") != MATRIX_SCHEMA:
        raise ValueError(f"release matrix schema must be {MATRIX_SCHEMA}")
    if type(matrix.get("release_ready")) is not bool:
        raise ValueError("release matrix release_ready must be boolean")
    artifacts = matrix.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("release matrix artifacts must be a non-empty array")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(artifacts):
        if not isinstance(item, dict) or set(item) != {"id", "channel", "required"}:
            raise ValueError(f"release matrix artifact {index} has unknown or missing fields")
        artifact_id = item.get("id")
        if not isinstance(artifact_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9_.-]+", artifact_id) is None:
            raise ValueError(f"release matrix artifact {index} has unsafe id")
        if item.get("channel") not in CHANNELS or type(item.get("required")) is not bool:
            raise ValueError(f"release matrix artifact {artifact_id} has invalid policy")
        normalized.append(dict(item))
    ids = [item["id"] for item in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("release matrix artifact ids must be unique")
    return normalized


def make_plan(
    event: str, ref: str, source_sha: str, version: str, matrix: dict[str, Any]
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise ValueError("source SHA must be a full lowercase 40-hex commit")
    if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?", version) is None:
        raise ValueError("version must be a semantic version")
    artifacts = validate_matrix(matrix)
    if event == "workflow_dispatch":
        if not ref.startswith("refs/heads/"):
            raise ValueError("manual dry runs must resolve a branch ref")
        release_tag = f"dry-run-{source_sha[:12]}"
        publish = False
    elif event == "push":
        release_tag = f"v{version}"
        if ref != f"refs/tags/{release_tag}":
            raise ValueError(f"release tag must exactly equal {release_tag}")
        publish = matrix["release_ready"]
    else:
        raise ValueError(f"unsupported release event {event!r}")
    return {
        "artifacts": artifacts,
        "event": event,
        "publish": publish,
        "release_tag": release_tag,
        "schema": PLAN_SCHEMA,
        "source_sha": source_sha,
        "version": version,
    }


def make_handoff(plan_path: Path, output: Path) -> None:
    plan = read_json(plan_path)
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("handoff input is not a release plan")
    write_json(
        output,
        {
            "artifacts": plan.get("artifacts"),
            "plan_sha256": file_sha256(plan_path),
            "publish": plan.get("publish"),
            "release_tag": plan.get("release_tag"),
            "schema": HANDOFF_SCHEMA,
            "source_sha": plan.get("source_sha"),
        },
    )


def verify_handoff(
    plan_path: Path, handoff_path: Path, output: Path, expected_sha: str
) -> None:
    plan = read_json(plan_path)
    handoff = read_json(handoff_path)
    expected = {
        "artifacts": plan.get("artifacts"),
        "plan_sha256": file_sha256(plan_path),
        "publish": plan.get("publish"),
        "release_tag": plan.get("release_tag"),
        "schema": HANDOFF_SCHEMA,
        "source_sha": expected_sha,
    }
    if handoff != expected or plan.get("source_sha") != expected_sha:
        raise ValueError("release handoff does not match the immutable plan and workflow SHA")
    write_json(output, {**handoff, "verified": True})


def write_outputs(path: Path, values: dict[str, str | bool]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            rendered = str(value).lower() if isinstance(value, bool) else value
            if "\n" in rendered:
                raise ValueError("workflow outputs must be single-line values")
            handle.write(f"{key}={rendered}\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--event", required=True)
    plan.add_argument("--ref", required=True)
    plan.add_argument("--sha", required=True)
    plan.add_argument("--version", required=True)
    plan.add_argument("--matrix", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--github-output", type=Path)
    handoff = commands.add_parser("handoff")
    handoff.add_argument("--plan", type=Path, required=True)
    handoff.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--handoff", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--sha", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "plan":
            plan = make_plan(args.event, args.ref, args.sha, args.version, read_json(args.matrix))
            write_json(args.output, plan)
            if args.github_output:
                write_outputs(
                    args.github_output,
                    {
                        "artifact_name": f"release-plan-{args.sha}",
                        "publish": plan["publish"],
                        "release_tag": plan["release_tag"],
                    },
                )
        elif args.command == "handoff":
            make_handoff(args.plan, args.output)
        else:
            verify_handoff(args.plan, args.handoff, args.output, args.sha)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"release pipeline error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
