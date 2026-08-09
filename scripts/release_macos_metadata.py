#!/usr/bin/env python3
"""Generate native macOS Metal/MLX release metadata from staged bytes."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import release_manifest  # noqa: E402
import release_metadata  # noqa: E402


ARTIFACTS = {
    "macos-arm64-metal": ("metal", "stable"),
    "macos-arm64-metal-mlx": ("mlx", "preview"),
}


def otool_dependencies(server: Path) -> list[str]:
    result = subprocess.run(
        ["otool", "-L", str(server)], text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise ValueError(f"otool dependency inspection failed: {result.stderr.strip()}")
    return [line.strip().split(" (", 1)[0] for line in result.stdout.splitlines()[1:] if line.strip()]


def dependency_name(install_name: str) -> tuple[str, str]:
    for component in install_name.split("/"):
        if component.endswith(".framework"):
            return component, "framework"
    return Path(install_name).name, "library"


def dependency_rows(
    binaries: list[Path], backend: str, abi_version: str, mlx_version: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    install_names = sorted(
        {name for binary in binaries for name in otool_dependencies(binary)}
    )
    for install_name in install_names:
        name, kind = dependency_name(install_name)
        if name == "libmlx.dylib":
            rows.append(
                {
                    "bundled": True,
                    "kind": "library",
                    "linkage": "dynamic",
                    "name": name,
                    "role": "runtime",
                    "version": mlx_version,
                }
            )
        elif kind == "framework":
            rows.append(
                {
                    "bundled": False,
                    "kind": kind,
                    "linkage": "external",
                    "name": name,
                    "role": "external-runtime",
                    "version": abi_version,
                }
            )
        else:
            rows.append(
                {
                    "bundled": False,
                    "kind": kind,
                    "linkage": "dynamic",
                    "name": name,
                    "role": "runtime",
                    "version": abi_version,
                }
            )
    if backend == "mlx":
        rows.append(
            {
                "bundled": True,
                "kind": "library",
                "linkage": "dynamic",
                "name": "mlx.metallib",
                "role": "runtime",
                "version": mlx_version,
            }
        )
    names = [row["name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Mach-O dependency inventory contains duplicate names")
    return rows


def prepare_macos_metadata(args: argparse.Namespace) -> dict[str, Any]:
    if args.artifact_id not in ARTIFACTS:
        raise ValueError(f"unsupported macOS artifact {args.artifact_id!r}")
    backend, required_channel = ARTIFACTS[args.artifact_id]
    if args.channel != required_channel:
        raise ValueError(f"{args.artifact_id} requires channel {required_channel}")
    server = args.stage_dir / "bin/vllm-server"
    if not server.is_file():
        raise ValueError("staged bin/vllm-server is missing")
    flags = release_metadata.backend_flags(
        release_metadata.parse_cache(args.build_dir / "CMakeCache.txt")
    )
    mlx = backend == "mlx"
    if flags["VLLM_CPP_METAL"] is not True or flags["VLLM_CPP_MLX"] is not mlx:
        raise ValueError("configured Metal/MLX flags do not match the artifact")
    if mlx:
        if not args.mlx_version:
            raise ValueError("MLX preview requires an exact MLX version")
        for relative in (
            "lib/libmlx.dylib",
            "lib/mlx.metallib",
        ):
            if not (args.stage_dir / relative).is_file():
                raise ValueError(f"MLX staged runtime is missing {relative}")
        if args.mlx_license is None or not args.mlx_license.is_file():
            raise ValueError("MLX preview requires the exact installed distribution license")
    elif flags["MLX_ROOT"] != "":
        raise ValueError("native Metal artifact must not record an MLX root")
    binaries = [server]
    if mlx:
        binaries.append(args.stage_dir / "lib/libmlx.dylib")
    dependencies = dependency_rows(binaries, backend, args.abi_version, args.mlx_version)
    runtime_command = "test_metal_backend"
    if mlx:
        runtime_command += " (MLX provider enabled)"
    facts: dict[str, Any] = {
        "artifact": {
            "c_abi_version": args.c_abi_version,
            "channel": args.channel,
            "id": args.artifact_id,
            "kind": "primary",
            "static_boundary": "static-core",
            "version": args.version,
        },
        "backend": {
            "flags": flags,
            "gpu_driver_boundary": "external-host-never-bundled",
            "name": backend,
        },
        "build": {
            "compiler": args.compiler,
            "resolved_cmake_options": flags,
            "source_clean": args.source_clean,
            "source_commit": args.source_commit,
            "test_commands": [runtime_command, "python3 scripts/validate-release-archive.py"],
            "toolchain": args.toolchain,
        },
        "dependencies": dependencies,
        "evidence": {
            "archive_smoke": release_metadata.passed(
                "extracted vllm-server --help && --version", args.evidence_url
            ),
            "build": release_metadata.passed("cmake --build <build> --target server", args.evidence_url),
            "correctness": release_metadata.passed(runtime_command, args.evidence_url),
            "dependency_audit": release_metadata.passed("file && otool -L && otool -l", args.evidence_url),
            "performance": release_metadata.absent("release packaging makes no performance claim"),
            "runtime": release_metadata.passed(runtime_command, args.evidence_url),
        },
        "host": {
            "abi": "macos",
            "abi_version": args.abi_version,
            "arch": "aarch64",
            "os": "macos",
        },
        "supply_chain": {
            "archive_checksum": release_metadata.passed("shasum -a 256 <final-archive>", args.evidence_url),
            "licenses": release_metadata.passed("validate vllm.cpp and MLX licenses", args.evidence_url),
            "provenance": release_metadata.passed("validate detached SLSA subject", args.evidence_url),
            "sbom": release_metadata.passed("validate SPDX-2.3 inventory", args.evidence_url),
        },
    }
    schema = release_manifest.load_schema(args.repo_root / "release/manifest-v1.schema.json")
    manifest = release_manifest.generate_manifest(facts, args.repo_root, schema)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "release-manifest.json").write_text(
        release_metadata.canonical_json(manifest), encoding="utf-8"
    )
    version_values = {
        "version": args.version,
        "commit": args.source_commit,
        "artifact_id": args.artifact_id,
        "backend": backend,
        "host_os": "macos",
        "host_arch": "aarch64",
        "host_abi": "macos",
        "source_clean": "true" if args.source_clean else "false",
        "c_abi_version": str(args.c_abi_version),
    }
    (output / "VERSION").write_text(
        "".join(f"{key}={value}\n" for key, value in version_values.items()),
        encoding="utf-8",
    )
    bundled_files: list[tuple[str, Path, str]] = []
    if mlx:
        bundled_files = [
            ("lib/libmlx.dylib", args.stage_dir / "lib/libmlx.dylib", "MIT"),
            ("lib/mlx.metallib", args.stage_dir / "lib/mlx.metallib", "MIT"),
        ]
    (output / "sbom.spdx.json").write_text(
        release_metadata.canonical_json(
            release_metadata.spdx_document(
                args.artifact_id,
                args.version,
                args.source_commit,
                server,
                dependencies,
                bundled_files,
            )
        ),
        encoding="utf-8",
    )
    notices = ["vllm.cpp release dependency notices", ""]
    notices.extend(
        f"- {row['name']} {row['version']} ({row['linkage']})" for row in dependencies
    )
    (output / "THIRD_PARTY_NOTICES").write_text("\n".join(notices) + "\n", encoding="utf-8")
    license_dir = output / "share/licenses/vllm.cpp"
    license_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.repo_root / "LICENSE", license_dir / "LICENSE")
    if mlx:
        mlx_license_dir = output / "share/licenses/MLX"
        mlx_license_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.mlx_license, mlx_license_dir / "LICENSE")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=SCRIPT_DIR.parent)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--channel", choices=("stable", "preview"), required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--c-abi-version", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-clean", action="store_true")
    parser.add_argument("--abi-version", required=True)
    parser.add_argument("--mlx-version", default="")
    parser.add_argument("--mlx-license", type=Path)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--toolchain", required=True)
    parser.add_argument("--evidence-url", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        prepare_macos_metadata(args)
    except (OSError, json.JSONDecodeError, KeyError, ValueError, release_manifest.ManifestError) as exc:
        print(f"macOS release metadata error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
