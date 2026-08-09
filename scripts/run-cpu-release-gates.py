#!/usr/bin/env python3
"""Execute every adaptive CPU tier plus a feature-poor refusal gate."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


TIER_ENV = (
    "VT_CPU_MATMUL_TIER",
    "VT_CPU_Q8_DOT",
    "VT_CPU_QUANT_MMLA",
    "VT_CPU_QUANT_REPACK",
)
MATMUL_GATE_ARG = (
    "--test-case=elementwise CPU GEMM: row-strided activation stays bit-exact"
)


class GateError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("x86_64", "aarch64"), required=True)
    parser.add_argument("--tests-dir", type=Path, required=True)
    parser.add_argument("--poor-emulator", type=Path, required=True)
    parser.add_argument("--rich-runner", type=Path)
    parser.add_argument("--rich-runner-kind", choices=("qemu", "intel-sde"))
    parser.add_argument("--rich-cpu", default="max")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-url", required=True)
    return parser.parse_args()


def executable(tests_dir: Path, name: str) -> Path:
    path = (tests_dir / name).resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise GateError(f"missing test executable: {path}")
    return path


def test_args(binary: Path, targeted: bool) -> list[str]:
    return [MATMUL_GATE_ARG] if targeted and binary.name == "test_ops_matmul_elem" else []


def display_command(
    prefix: list[str], env_values: dict[str, str], binary: Path, targeted: bool
) -> str:
    assignments = " ".join(f"{name}={shlex.quote(value)}" for name, value in env_values.items())
    command = shlex.join([*prefix, str(binary), *test_args(binary, targeted)])
    return f"{assignments} {command}".strip()


def run_command(
    prefix: list[str],
    env_values: dict[str, str],
    binary: Path,
    expect_failure: bool = False,
    targeted: bool = False,
) -> tuple[str, str]:
    command = display_command(prefix, env_values, binary, targeted)
    environment = os.environ.copy()
    for name in TIER_ENV:
        environment.pop(name, None)
    environment.update(env_values)
    result = subprocess.run(
        [*prefix, str(binary), *test_args(binary, targeted)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = result.stdout.strip()[-4000:]
    if expect_failure:
        if result.returncode == 0:
            raise GateError(f"feature-poor host unexpectedly accepted: {command}")
        return f"expect-refusal {command}", f"exit {result.returncode}: {output}"
    if result.returncode != 0:
        raise GateError(f"CPU release gate failed ({result.returncode}): {command}\n{output}")
    return command, f"exit 0: {output}"


def evidence(commands: list[str], results: list[str], url: str) -> dict[str, str]:
    return {
        "command": " && ".join(commands),
        "reason": "",
        "result": " | ".join(results),
        "state": "passed",
        "url": url,
    }


def run_group(
    prefix: list[str], rows: list[tuple[dict[str, str], Path]], url: str
) -> tuple[dict[str, str], list[str]]:
    commands: list[str] = []
    results: list[str] = []
    for env_values, binary in rows:
        command, result = run_command(prefix, env_values, binary)
        commands.append(command)
        results.append(result)
    return evidence(commands, results, url), commands


def gate(args: argparse.Namespace) -> dict[str, object]:
    tests_dir = args.tests_dir.resolve()
    matmul = executable(tests_dir, "test_ops_matmul_elem")
    if (args.rich_runner is None) != (args.rich_runner_kind is None):
        raise GateError("rich runner and kind must be provided together")
    rich_prefix: list[str] = []
    if args.rich_runner is not None:
        rich_runner = str(args.rich_runner.resolve())
        rich_prefix = (
            [rich_runner, "-cpu", args.rich_cpu]
            if args.rich_runner_kind == "qemu"
            else [rich_runner, "-skx", "--"]
        )
    poor_model = "Nehalem" if args.arch == "x86_64" else "cortex-a53"
    poor_prefix = [str(args.poor_emulator.resolve()), "-cpu", poor_model]
    tiers: dict[str, dict[str, str]] = {}
    commands: list[str] = []

    if args.arch == "x86_64":
        definitions = (
            ("portable-sse2", [({"VT_CPU_MATMUL_TIER": "portable"}, matmul), ({"VT_CPU_MATMUL_TIER": "sse2"}, matmul)]),
            ("sse2-f16c", [({"VT_CPU_MATMUL_TIER": "sse2+f16c"}, matmul)]),
            ("avx2-f16c", [({"VT_CPU_MATMUL_TIER": "avx2"}, matmul)]),
            ("avx512f", [({"VT_CPU_MATMUL_TIER": "avx512"}, matmul)]),
        )
        for name, rows in definitions:
            tiers[name], ran = run_group(rich_prefix, rows, args.evidence_url)
            commands.extend(ran)
        for env_values in (
            {"VT_CPU_MATMUL_TIER": "portable"},
            {"VT_CPU_MATMUL_TIER": "sse2"},
        ):
            command, _ = run_command(poor_prefix, env_values, matmul, targeted=True)
            commands.append(command)
        for env_values in (
            {"VT_CPU_MATMUL_TIER": "sse2+f16c"},
            {"VT_CPU_MATMUL_TIER": "avx2"},
        ):
            command, _ = run_command(
                poor_prefix, env_values, matmul, expect_failure=True, targeted=True
            )
            commands.append(command)
        selected_tier = "avx512f"
    else:
        cpu_isa = executable(tests_dir, "test_cpu_isa_arm")
        quant_dot = executable(tests_dir, "test_ops_quant_dot")
        quant_repack = executable(tests_dir, "test_ops_quant_repack")
        definitions = (
            (
                "portable-neon",
                [
                    ({}, cpu_isa),
                    ({"VT_CPU_MATMUL_TIER": "portable"}, matmul),
                    ({"VT_CPU_MATMUL_TIER": "neon"}, matmul),
                    ({"VT_CPU_Q8_DOT": "portable", "VT_CPU_QUANT_MMLA": "portable"}, quant_dot),
                    ({"VT_CPU_QUANT_REPACK": "portable"}, quant_repack),
                ],
            ),
            ("dotprod", [({"VT_CPU_Q8_DOT": "sdot"}, quant_dot)]),
            (
                "i8mm",
                [
                    ({"VT_CPU_QUANT_MMLA": "i8mm"}, quant_dot),
                    ({"VT_CPU_QUANT_REPACK": "i8mm"}, quant_repack),
                ],
            ),
        )
        for name, rows in definitions:
            tiers[name], ran = run_group(rich_prefix, rows, args.evidence_url)
            commands.extend(ran)
        poor_pass = (
            ({}, cpu_isa),
            ({"VT_CPU_MATMUL_TIER": "portable"}, matmul),
            ({"VT_CPU_MATMUL_TIER": "neon"}, matmul),
            ({"VT_CPU_Q8_DOT": "portable", "VT_CPU_QUANT_MMLA": "portable"}, quant_dot),
            ({"VT_CPU_QUANT_REPACK": "portable"}, quant_repack),
        )
        for env_values, binary in poor_pass:
            command, _ = run_command(poor_prefix, env_values, binary, targeted=True)
            commands.append(command)
        poor_refuse = (
            ({"VT_CPU_Q8_DOT": "sdot"}, quant_dot),
            ({"VT_CPU_QUANT_MMLA": "i8mm"}, quant_dot),
            ({"VT_CPU_QUANT_REPACK": "i8mm"}, quant_repack),
        )
        for env_values, binary in poor_refuse:
            command, _ = run_command(
                poor_prefix, env_values, binary, expect_failure=True, targeted=True
            )
            commands.append(command)
        selected_tier = "i8mm"

    return {
        "commands": commands,
        "schema": "vllm.cpp.cpu-tier-report.v1",
        "selected_tier": selected_tier,
        "tiers": tiers,
    }


def main() -> int:
    args = parse_args()
    try:
        report = gate(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    except (GateError, OSError) as exc:
        print(f"CPU release gate error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
