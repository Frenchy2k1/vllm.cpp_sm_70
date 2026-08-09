#!/usr/bin/env python3
"""W11 Metal and MLX release metadata contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts/release_macos_metadata.py"
SHA = "0123456789abcdef0123456789abcdef01234567"


def load():
    spec = importlib.util.spec_from_file_location("release_macos_metadata", TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {TOOL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MacosMetadataContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = load()

    def fixture(self, scratch: Path, mlx: bool):
        build = scratch / "build"
        stage = scratch / "stage"
        output = scratch / "metadata"
        build.mkdir()
        (stage / "bin").mkdir(parents=True)
        shutil.copy2("/bin/true", stage / "bin/vllm-server")
        mlx_root = "/opt/mlx" if mlx else ""
        if mlx:
            (stage / "lib").mkdir()
            (stage / "lib/libmlx.dylib").write_bytes(b"mlx")
            (stage / "lib/mlx.metallib").write_bytes(b"metal")
            license_dir = stage / "share/licenses/MLX"
            license_dir.mkdir(parents=True)
            (license_dir / "LICENSE").write_text("MIT\n", encoding="utf-8")
        cache = {
            "MLX_ROOT": ("PATH", mlx_root),
            "VLLM_CPP_BUILD_EXAMPLES": ("BOOL", "ON"),
            "VLLM_CPP_BUILD_TESTS": ("BOOL", "ON"),
            "VLLM_CPP_CUDA": ("BOOL", "OFF"),
            "VLLM_CPP_CUDA_ARCHITECTURES": ("STRING", ""),
            "VLLM_CPP_HIP": ("BOOL", "OFF"),
            "VLLM_CPP_HIP_ARCHITECTURES": ("STRING", ""),
            "VLLM_CPP_LITERAL_STATIC": ("BOOL", "OFF"),
            "VLLM_CPP_METAL": ("BOOL", "ON"),
            "VLLM_CPP_MLX": ("BOOL", "ON" if mlx else "OFF"),
            "VLLM_CPP_SERVER": ("BOOL", "ON"),
            "VLLM_CPP_TRITON": ("BOOL", "OFF"),
            "VLLM_CPP_VULKAN": ("BOOL", "OFF"),
        }
        (build / "CMakeCache.txt").write_text(
            "".join(f"{key}:{kind}={value}\n" for key, (kind, value) in cache.items()),
            encoding="utf-8",
        )
        return argparse.Namespace(
            abi_version="15.0",
            artifact_id="macos-arm64-metal-mlx" if mlx else "macos-arm64-metal",
            build_dir=build,
            c_abi_version=17,
            channel="preview" if mlx else "stable",
            compiler="AppleClang 16",
            evidence_url="https://github.com/mudler/vllm.cpp/actions/runs/1",
            mlx_version="0.32.0" if mlx else "",
            mlx_license=(stage / "share/licenses/MLX/LICENSE") if mlx else None,
            output_dir=output,
            repo_root=ROOT,
            source_clean=True,
            source_commit=SHA,
            stage_dir=stage,
            toolchain="cmake-3.30+ninja-1.12",
            version="0.0.1",
        )

    def otool(self, mlx: bool):
        names = [
            "/System/Library/Frameworks/Metal.framework/Versions/A/Metal",
            "/System/Library/Frameworks/Foundation.framework/Versions/C/Foundation",
            "/usr/lib/libc++.1.dylib",
            "/usr/lib/libSystem.B.dylib",
        ]
        if mlx:
            names.insert(0, "@rpath/libmlx.dylib")
        return names

    def test_native_metal_stable_has_runtime_correctness_and_frameworks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary), False)
            with mock.patch.object(self.tool, "otool_dependencies", return_value=self.otool(False)):
                manifest = self.tool.prepare_macos_metadata(args)
            self.assertEqual(manifest["artifact"]["channel"], "stable")
            self.assertEqual(manifest["evidence"]["runtime"]["state"], "passed")
            self.assertEqual(manifest["evidence"]["correctness"]["state"], "passed")
            names = {row["name"] for row in manifest["dependencies"]}
            self.assertTrue({"Metal.framework", "Foundation.framework"} <= names)

    def test_mlx_preview_requires_bundled_versioned_runtime_and_license(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary), True)
            with mock.patch.object(self.tool, "otool_dependencies", return_value=self.otool(True)):
                manifest = self.tool.prepare_macos_metadata(args)
            dependencies = {row["name"]: row for row in manifest["dependencies"]}
            for name in ("libmlx.dylib", "mlx.metallib"):
                self.assertTrue(dependencies[name]["bundled"])
                self.assertEqual(dependencies[name]["version"], "0.32.0")
            (args.stage_dir / "lib/mlx.metallib").unlink()
            with mock.patch.object(self.tool, "otool_dependencies", return_value=self.otool(True)):
                with self.assertRaises(ValueError):
                    self.tool.prepare_macos_metadata(args)


if __name__ == "__main__":
    unittest.main()
