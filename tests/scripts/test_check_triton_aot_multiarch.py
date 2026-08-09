#!/usr/bin/env python3
"""Mutation tests for W2 multi-SM Triton AOT artifact auditing."""

from __future__ import annotations

import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-triton-aot-multiarch.py"
FLAGS = {
    "sm_80": 0x00500550,
    "sm_86": 0x00560556,
    "sm_89": 0x00590559,
    "sm_90a": 0x005A0D5A,
    "sm_100a": 0x0600640A,
    "sm_121a": 0x0600790A,
}


def cubin_source(flags: int) -> str:
    data = bytearray(64)
    data[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<I", data, 48, flags)
    values = ", ".join(f"0x{byte:02x}" for byte in data)
    return f"unsigned char CUBIN_NAME[64] = {{ {values} }};\n"


class TritonAotMultiArchContract(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, Path]:
        vendored = root / "vendored"
        nm = root / "nm.txt"
        symbols = []
        for arch, flags in FLAGS.items():
            tree = vendored / arch
            tree.mkdir(parents=True)
            (tree / "MANIFEST").write_text(
                f"arch {arch}\nbase toy py=toy.py kernel=toy\n",
                encoding="utf-8",
            )
            (tree / "toy.hash.c").write_text(cubin_source(flags), encoding="utf-8")
            (tree / "toy.c").write_text("int toy_default(void);\n", encoding="utf-8")
            symbols.append(f"00000000 T vt_aot_{arch}_toy_default")
            symbols.append(f"00000000 T vt_aot_{arch}_load_toy")
        nm.write_text("\n".join(symbols) + "\n", encoding="utf-8")
        return vendored, nm

    def run_checker(self, mutate=None) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temporary:
            vendored, nm = self.make_fixture(Path(temporary))
            if mutate is not None:
                mutate(vendored, nm)
            return subprocess.run(
                [
                    sys.executable,
                    str(CHECKER),
                    "--vendored-root",
                    str(vendored),
                    "--nm-list",
                    str(nm),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

    def test_exact_six_trees_and_namespaced_symbols_pass(self) -> None:
        result = self.run_checker()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_wrong_tree_cubin_fails(self) -> None:
        def mutate(vendored: Path, _nm: Path) -> None:
            (vendored / "sm_80" / "toy.hash.c").write_text(
                cubin_source(FLAGS["sm_86"]), encoding="utf-8"
            )

        result = self.run_checker(mutate)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("wrong cubin", result.stdout + result.stderr)

    def test_missing_tree_fails(self) -> None:
        def mutate(vendored: Path, _nm: Path) -> None:
            for path in (vendored / "sm_89").iterdir():
                path.unlink()
            (vendored / "sm_89").rmdir()

        result = self.run_checker(mutate)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("tree set", result.stdout + result.stderr)

    def test_unnamespaced_or_missing_symbol_fails(self) -> None:
        def mutate(_vendored: Path, nm: Path) -> None:
            text = nm.read_text(encoding="utf-8")
            text = text.replace("vt_aot_sm_80_toy_default", "toy_default")
            nm.write_text(text, encoding="utf-8")

        result = self.run_checker(mutate)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("namespace", result.stdout + result.stderr)

    def test_missing_namespaced_loader_fails(self) -> None:
        def mutate(_vendored: Path, nm: Path) -> None:
            text = nm.read_text(encoding="utf-8")
            text = text.replace("00000000 T vt_aot_sm_121a_load_toy\n", "")
            nm.write_text(text, encoding="utf-8")

        result = self.run_checker(mutate)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("namespace is missing", result.stdout + result.stderr)

    def test_forced_namespace_headers_compile_collision_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = "int toy_default(void) { return 0; }\nvoid load_toy(void) {}\n"
            (root / "sm80.c").write_text(source, encoding="utf-8")
            (root / "sm86.c").write_text(source, encoding="utf-8")
            helper = (ROOT / "cmake" / "TritonAOTMultiArch.cmake").as_posix()
            (root / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.24)\n"
                "project(triton_namespace C)\n"
                f'include("{helper}")\n'
                "set(sm80 ${CMAKE_CURRENT_SOURCE_DIR}/sm80.c)\n"
                "set(sm86 ${CMAKE_CURRENT_SOURCE_DIR}/sm86.c)\n"
                "vt_triton_aot_namespace_sources(sm_80 toy "
                '"${CMAKE_CURRENT_BINARY_DIR}/namespaces" out80 ${sm80})\n'
                "vt_triton_aot_namespace_sources(sm_86 toy "
                '"${CMAKE_CURRENT_BINARY_DIR}/namespaces" out86 ${sm86})\n'
                "add_library(toy STATIC ${out80} ${out86})\n",
                encoding="utf-8",
            )
            configure = subprocess.run(
                ["cmake", "-S", str(root), "-B", str(root / "build")],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(configure.returncode, 0, configure.stdout + configure.stderr)
            build = subprocess.run(
                ["cmake", "--build", str(root / "build")],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            listing = subprocess.run(
                ["nm", "--defined-only", "--extern-only", str(root / "build" / "libtoy.a")],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(listing.returncode, 0, listing.stderr)
            for arch in ("sm_80", "sm_86"):
                self.assertIn(f"vt_aot_{arch}_toy_default", listing.stdout)
                self.assertIn(f"vt_aot_{arch}_load_toy", listing.stdout)
            self.assertNotRegex(listing.stdout, r"\btoy_default\b")
            self.assertNotRegex(listing.stdout, r"\bload_toy\b")


if __name__ == "__main__":
    unittest.main()
