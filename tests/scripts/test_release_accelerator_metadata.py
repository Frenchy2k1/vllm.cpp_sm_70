#!/usr/bin/env python3
"""W10/W11 Linux accelerator release metadata contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts/release_accelerator_metadata.py"
SHA = "0123456789abcdef0123456789abcdef01234567"
SMS = ["80", "86", "87", "89", "90a", "100a", "103a", "110", "120a", "121a"]


def load():
    spec = importlib.util.spec_from_file_location("release_accelerator_metadata", TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {TOOL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AcceleratorMetadataContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = load()

    def fixture(self, scratch: Path, artifact_id: str):
        build = scratch / "build"
        stage = scratch / "stage"
        output = scratch / "metadata"
        build.mkdir()
        (stage / "bin").mkdir(parents=True)
        shutil.copy2("/bin/true", stage / "bin/vllm-server")
        cuda = "cuda" in artifact_id
        vulkan = "vulkan" in artifact_id
        cache = {
            "MLX_ROOT": ("PATH", ""),
            "VLLM_CPP_BUILD_EXAMPLES": ("BOOL", "ON"),
            "VLLM_CPP_BUILD_TESTS": ("BOOL", "ON"),
            "VLLM_CPP_CUDA": ("BOOL", "ON" if cuda else "OFF"),
            "VLLM_CPP_CUDA_ARCHITECTURES": ("STRING", ";".join(SMS) if cuda else ""),
            "VLLM_CPP_HIP": ("BOOL", "OFF"),
            "VLLM_CPP_HIP_ARCHITECTURES": ("STRING", ""),
            "VLLM_CPP_LITERAL_STATIC": ("BOOL", "OFF"),
            "VLLM_CPP_METAL": ("BOOL", "OFF"),
            "VLLM_CPP_MLX": ("BOOL", "OFF"),
            "VLLM_CPP_SERVER": ("BOOL", "ON"),
            "VLLM_CPP_TRITON": ("BOOL", "ON" if cuda else "OFF"),
            "VLLM_CPP_VULKAN": ("BOOL", "ON" if vulkan else "OFF"),
        }
        (build / "CMakeCache.txt").write_text(
            "".join(f"{key}:{kind}={value}\n" for key, (kind, value) in cache.items()),
            encoding="utf-8",
        )
        return argparse.Namespace(
            abi_version="2.39",
            artifact_id=artifact_id,
            build_dir=build,
            c_abi_version=17,
            channel="preview",
            compiler="GNU C++ 13.2.0",
            evidence_url="https://github.com/mudler/vllm.cpp/actions/runs/1",
            output_dir=output,
            repo_root=ROOT,
            source_clean=True,
            source_commit=SHA,
            stage_dir=stage,
            toolchain="cmake-3.30+ninja-1.12",
            version="0.0.1",
        )

    def test_cuda_manifest_carries_all_sms_aot_and_external_driver(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary), "linux-x86_64-glibc-cuda-fat")
            manifest = self.tool.prepare_accelerator_metadata(args)
            self.assertEqual(manifest["cuda"]["compiled_sms"], SMS)
            self.assertEqual(
                [row["aot_available"] for row in manifest["cuda"]["sm_evidence"]],
                [True, True, False, True, True, True, False, False, False, True],
            )
            driver = next(row for row in manifest["dependencies"] if row["name"] == "nvidia-driver")
            self.assertEqual((driver["linkage"], driver["bundled"]), ("external", False))

    def test_vulkan_manifest_has_all_three_external_runtime_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary), "linux-x86_64-glibc-vulkan")
            manifest = self.tool.prepare_accelerator_metadata(args)
            names = {row["name"] for row in manifest["dependencies"]}
            self.assertTrue({"vulkan-loader", "vulkan-icd", "vulkan-driver"} <= names)
            self.assertNotIn("cuda", manifest)

    def test_cuda_refuses_partial_sm_or_disabled_triton_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary), "linux-x86_64-glibc-cuda-fat")
            cache = (args.build_dir / "CMakeCache.txt").read_text()
            for before, after in (
                ("80;86;87;89;90a;100a;103a;110;120a;121a", "80;86"),
                ("VLLM_CPP_TRITON:BOOL=ON", "VLLM_CPP_TRITON:BOOL=OFF"),
            ):
                (args.build_dir / "CMakeCache.txt").write_text(cache.replace(before, after), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.tool.prepare_accelerator_metadata(args)
                (args.build_dir / "CMakeCache.txt").write_text(cache, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
