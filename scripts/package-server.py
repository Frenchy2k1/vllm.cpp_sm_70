#!/usr/bin/env python3
"""Stage the installed server component and optionally archive it reproducibly."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--config", default="")
    return parser.parse_args()


def source_date_epoch() -> int:
    raw = os.environ.get("SOURCE_DATE_EPOCH", "0")
    try:
        epoch = int(raw)
    except ValueError as exc:
        raise SystemExit("SOURCE_DATE_EPOCH must be a non-negative integer") from exc
    if epoch < 0:
        raise SystemExit("SOURCE_DATE_EPOCH must be a non-negative integer")
    return epoch


def require_build_output(path: Path, build_dir: Path, label: str) -> None:
    try:
        relative = path.relative_to(build_dir)
    except ValueError as exc:
        raise SystemExit(f"{label} must be inside the configured build directory") from exc
    if relative == Path("."):
        raise SystemExit(f"{label} must not replace the configured build directory")


def install_component(build_dir: Path, prefix: Path, config: str) -> None:
    command = [
        "cmake",
        "--install",
        str(build_dir),
        "--prefix",
        str(prefix),
        "--component",
        "vllm-server",
    ]
    if config:
        command.extend(("--config", config))
    subprocess.run(command, check=True)


def normalized_info(tar: tarfile.TarFile, path: Path, arcname: str, epoch: int) -> tarfile.TarInfo:
    info = tar.gettarinfo(str(path), arcname=arcname)
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mtime = epoch
    if info.isdir():
        info.mode = 0o755
    elif info.isfile():
        info.mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
    return info


def write_archive(stage_dir: Path, archive: Path, epoch: int) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=archive.parent, prefix=f".{archive.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.GNU_FORMAT) as bundle:
                    for path in sorted(stage_dir.rglob("*"), key=lambda item: item.as_posix()):
                        arcname = path.relative_to(stage_dir).as_posix()
                        info = normalized_info(bundle, path, arcname, epoch)
                        if info.isfile():
                            with path.open("rb") as source:
                                bundle.addfile(info, source)
                        else:
                            bundle.addfile(info)
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)


def install_metadata(metadata_dir: Path, stage_dir: Path) -> None:
    required = {
        "THIRD_PARTY_NOTICES",
        "VERSION",
        "release-manifest.json",
        "sbom.spdx.json",
    }
    files = {
        path.relative_to(metadata_dir).as_posix(): path
        for path in metadata_dir.rglob("*")
        if path.is_file()
    }
    missing = sorted(required - files.keys())
    if missing:
        raise SystemExit(f"release metadata is missing required files: {missing}")
    licenses = [name for name in files if name.startswith("share/licenses/")]
    if not licenses:
        raise SystemExit("release metadata must include share/licenses entries")
    allowed = required | set(licenses)
    extra = sorted(set(files) - allowed)
    if extra:
        raise SystemExit(f"release metadata contains undeclared files: {extra}")
    for relative, source in files.items():
        destination = stage_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def write_archive_sidecars(archive: Path, stage_dir: Path) -> None:
    hasher = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    Path(f"{archive}.sha256").write_text(
        f"{digest}  {archive.name}\n", encoding="utf-8"
    )
    manifest = json.loads((stage_dir / "release-manifest.json").read_text(encoding="utf-8"))
    provenance = {
        "_type": "https://in-toto.io/Statement/v1",
        "predicateType": "https://slsa.dev/provenance/v1",
        "subject": [{"name": archive.name, "digest": {"sha256": digest}}],
        "predicate": {
            "buildDefinition": {
                "externalParameters": {
                    "artifact_id": manifest["artifact"]["id"],
                    "source_commit": manifest["build"]["source_commit"],
                }
            }
        },
    }
    Path(f"{archive}.provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    build_dir = args.build_dir.resolve()
    stage_dir = args.stage_dir.resolve()
    if not (build_dir / "CMakeCache.txt").is_file():
        raise SystemExit("--build-dir must name a configured CMake build")
    require_build_output(stage_dir, build_dir, "--stage-dir")
    if args.archive is not None:
        require_build_output(args.archive.resolve(), build_dir, "--archive")
    metadata_dir = args.metadata_dir.resolve() if args.metadata_dir is not None else None
    if metadata_dir is not None and not metadata_dir.is_dir():
        raise SystemExit("--metadata-dir must name a prepared metadata directory")
    stage_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".vllm-server-stage-", dir=stage_dir.parent) as raw:
        fresh_stage = Path(raw)
        install_component(build_dir, fresh_stage, args.config)
        server_name = "vllm-server.exe" if os.name == "nt" else "vllm-server"
        server = fresh_stage / "bin" / server_name
        if not server.is_file():
            raise SystemExit(f"installed server component is missing {server.relative_to(fresh_stage)}")
        if metadata_dir is not None:
            install_metadata(metadata_dir, fresh_stage)
        if args.archive is not None:
            write_archive(fresh_stage, args.archive.resolve(), source_date_epoch())
            if metadata_dir is not None:
                write_archive_sidecars(args.archive.resolve(), fresh_stage)

        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        shutil.copytree(fresh_stage, stage_dir, symlinks=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
