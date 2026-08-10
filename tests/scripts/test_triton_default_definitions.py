#!/usr/bin/env python3
"""BUILD-TRITON-DEFAULT-ON (#219), spec `## Tests` item 3.

The row's whole claim is that a CUDA build with **no** `-DVLLM_CPP_TRITON` on
the command line still compiles the vendored Triton AOT kernels in. Resolving
the option to `ON` in the cache is only half of that: the claim is only true if
the resolved value reaches the CUDA translation units as
`VLLM_CPP_TRITON=1` **and** `VLLM_CPP_TRITON_CHUNKO_BF16=1`, which is what
`src/vt/cuda/cuda_gdn.cu` compiles against.

That end-to-end fact was previously discharged by counting translation units by
hand on the gate host, which is not a gate. `definition_errors` below is the
gate; the `cuda-fat-build` CI job feeds it a real ten-SM CUDA configure made
without the flag, and the synthetic cases here pin that each way of breaking the
claim is actually rejected.

Run standalone (skips the live case):

    python3 -m unittest tests.scripts.test_triton_default_definitions

Run against a real configure:

    VLLM_CPP_DEFAULT_BUILD_DIR=build-cuda-default \
        python3 -m unittest tests.scripts.test_triton_default_definitions
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

# The two definitions the CUDA sources compile against. Both are set by
# `target_compile_definitions(vllm PUBLIC ...)` inside the single
# `if(VLLM_CPP_TRITON)` block in CMakeLists.txt, so a build that carries one but
# not the other means the wiring was split, not that the option resolved.
REQUIRED_DEFINITIONS = ("VLLM_CPP_TRITON=1", "VLLM_CPP_TRITON_CHUNKO_BF16=1")


def cache_value(cache_text: str, key: str) -> str | None:
    """Return the value of a CMakeCache.txt entry, or None if it is absent."""

    for line in cache_text.splitlines():
        name, _, value = line.partition("=")
        base, _, _kind = name.partition(":")
        if base == key:
            return value
    return None


def _command_of(entry: dict) -> str:
    if "command" in entry:
        return str(entry["command"])
    return " ".join(str(argument) for argument in entry.get("arguments", ()))


def cuda_translation_units(commands: list[dict]) -> list[dict]:
    """Return the compile_commands.json entries that compile a .cu source."""

    return [entry for entry in commands if str(entry.get("file", "")).endswith(".cu")]


def definition_errors(cache_text: str, commands: list[dict]) -> list[str]:
    """Return why this configure fails the row's claim, empty if it holds."""

    errors: list[str] = []
    if cache_value(cache_text, "VLLM_CPP_CUDA") != "ON":
        errors.append(
            "VLLM_CPP_CUDA is not ON in this cache, so it cannot show anything "
            "about the CUDA default"
        )
        return errors
    resolved = cache_value(cache_text, "VLLM_CPP_TRITON")
    if resolved != "ON":
        errors.append(
            f"VLLM_CPP_TRITON resolved to {resolved!r} in a CUDA configure that "
            "passed no -DVLLM_CPP_TRITON; the computed default did not take"
        )
    units = cuda_translation_units(commands)
    if not units:
        errors.append(
            "no .cu translation unit in compile_commands.json; this configure "
            "cannot witness the CUDA compile definitions"
        )
        return errors
    for definition in REQUIRED_DEFINITIONS:
        without = [
            entry["file"] for entry in units if f"-D{definition}" not in _command_of(entry)
        ]
        if without:
            errors.append(
                f"{len(without)} of {len(units)} CUDA translation units do not "
                f"carry -D{definition} (first: {without[0]}); the default-built "
                "CUDA target does not compile the vendored Triton AOT kernels in"
            )
    return errors


def _unit(path: str, *definitions: str) -> dict:
    flags = " ".join(f"-D{definition}" for definition in definitions)
    return {"directory": "/b", "file": path, "command": f"nvcc {flags} -c {path}"}


CACHE_OK = "VLLM_CPP_CUDA:BOOL=ON\nVLLM_CPP_TRITON:BOOL=ON\nCMAKE_BUILD_TYPE:STRING=Release\n"
UNITS_OK = [
    _unit("/s/src/vt/cuda/cuda_gdn.cu", *REQUIRED_DEFINITIONS),
    _unit("/s/src/vt/cuda/cuda_backend.cu", *REQUIRED_DEFINITIONS),
    {"directory": "/b", "file": "/s/src/vt/cpu/cpu_backend.cpp", "command": "c++ -c x"},
]


class TritonDefaultDefinitions(unittest.TestCase):
    def test_a_default_on_cuda_configure_passes(self) -> None:
        self.assertEqual(definition_errors(CACHE_OK, list(UNITS_OK)), [])

    def test_the_option_resolving_off_is_rejected(self) -> None:
        cache = CACHE_OK.replace("VLLM_CPP_TRITON:BOOL=ON", "VLLM_CPP_TRITON:BOOL=OFF")
        errors = definition_errors(cache, list(UNITS_OK))
        self.assertTrue(any("computed default did not take" in e for e in errors), errors)

    def test_a_missing_option_is_rejected(self) -> None:
        cache = "VLLM_CPP_CUDA:BOOL=ON\n"
        errors = definition_errors(cache, list(UNITS_OK))
        self.assertTrue(any("VLLM_CPP_TRITON resolved to None" in e for e in errors), errors)

    def test_each_required_definition_is_enforced(self) -> None:
        for definition in REQUIRED_DEFINITIONS:
            kept = [d for d in REQUIRED_DEFINITIONS if d != definition]
            units = [
                _unit("/s/src/vt/cuda/cuda_gdn.cu", *REQUIRED_DEFINITIONS),
                _unit("/s/src/vt/cuda/cuda_backend.cu", *kept),
            ]
            with self.subTest(definition=definition):
                errors = definition_errors(CACHE_OK, units)
                self.assertTrue(
                    any(f"-D{definition}" in e for e in errors),
                    f"dropping {definition} from one CUDA unit was not rejected: {errors}",
                )

    def test_a_configure_with_no_cuda_unit_cannot_witness_the_claim(self) -> None:
        units = [{"directory": "/b", "file": "/s/a.cpp", "command": "c++ -c /s/a.cpp"}]
        errors = definition_errors(CACHE_OK, units)
        self.assertTrue(any("no .cu translation unit" in e for e in errors), errors)

    def test_arguments_form_is_read_too(self) -> None:
        # CMake emits `arguments` rather than `command` for some generators.
        units = [
            {
                "directory": "/b",
                "file": "/s/x.cu",
                "arguments": ["nvcc", "-DVLLM_CPP_TRITON=1", "-c", "/s/x.cu"],
            }
        ]
        errors = definition_errors(CACHE_OK, units)
        self.assertTrue(
            any("VLLM_CPP_TRITON_CHUNKO_BF16=1" in e for e in errors), errors
        )
        self.assertFalse(any("-DVLLM_CPP_TRITON=1 " in e for e in errors), errors)

    def test_the_live_default_configure(self) -> None:
        build = os.environ.get("VLLM_CPP_DEFAULT_BUILD_DIR")
        if not build:
            self.skipTest(
                "set VLLM_CPP_DEFAULT_BUILD_DIR to a CUDA configure made with no "
                "-DVLLM_CPP_TRITON (the cuda-fat-build CI job does)"
            )
        root = Path(build)
        cache = (root / "CMakeCache.txt").read_text(encoding="utf-8")
        commands = json.loads((root / "compile_commands.json").read_text(encoding="utf-8"))
        self.assertEqual(definition_errors(cache, commands), [])


if __name__ == "__main__":
    unittest.main()
