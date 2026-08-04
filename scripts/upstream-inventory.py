#!/usr/bin/env python3
"""Derive what still needs porting from the reference code. (W1)

The agent record only tracks rows we created, so it cannot see what upstream has
that we never inventoried. The first pass of this enumeration found 62 vLLM
architectures with no row at all, and CORRECTED a wrong conclusion: component
rows with no local files are not dead, they are unported.

Doing that once is an audit; doing it reproducibly makes matrix-vs-upstream
drift a checkable condition. Every finding cites `file:line` in the reference, so
a row derived from it satisfies the same evidence bar the matrices already
demand.

    scripts/upstream-inventory.py            # report
    scripts/upstream-inventory.py --emit     # refresh the committed snapshot
    scripts/upstream-inventory.py --check    # fail on drift vs the snapshot

Reference checkouts come from `${VLLM_SOURCE}` etc., defaulting to ~/_git/*.
They are NOT available in CI, so `--check` skips cleanly when they are absent
rather than pretending to have verified something.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / ".agents/upstream-inventory.json"

REFERENCES = {
    "vllm": (os.environ.get("VLLM_SOURCE", str(Path.home() / "_git/vllm")), "555967922"),
    "sglang": (os.environ.get("SGLANG_SOURCE", str(Path.home() / "_git/sglang")), None),
    "llamacpp": (os.environ.get("LLAMACPP_SOURCE", str(Path.home() / "_git/llama.cpp")), None),
}

# vLLM csrc components our backend matrix tracks as port targets.
COMPONENTS = (
    "machete", "allspark", "scaled_mm_c2x", "cutlass_moe",
    "w4a8", "flashmla", "deepgemm", "mla",
)

ARCH_ROW = re.compile(r"BACKEND-CUDA-SM(\d{3})")
REGISTRY_ENTRY = re.compile(r'^\s+"([A-Za-z0-9_]+)":\s*\(', re.M)
SUPPORTED_ARCHS = re.compile(r'set\(CUDA_SUPPORTED_ARCHS\s+"([^"]+)"')


def head_of(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError):
        return None


def references_available() -> dict[str, Path]:
    found = {}
    for name, (path, _) in REFERENCES.items():
        candidate = Path(path)
        if candidate.is_dir() and head_of(candidate):
            found[name] = candidate
    return found


def parse_supported_archs(text: str) -> tuple[list[str], int]:
    """Every arch vLLM builds, plus the line the widest list came from."""
    best: list[str] = []
    best_line = 0
    for match in SUPPORTED_ARCHS.finditer(text):
        archs = [a.strip() for a in match.group(1).split(";") if a.strip()]
        if len(archs) > len(best):
            best, best_line = archs, text[: match.start()].count("\n") + 1
    return best, best_line


def arch_key(arch: str) -> int:
    """'7.5' -> 75, so it compares against our SM0NN row ids."""
    major, _, minor = arch.partition(".")
    return int(major) * 10 + int(minor or 0)


def below_floor(our_rows: set[str], archs: list[str]) -> list[str]:
    """Our arch rows that sit under vLLM's lowest supported arch."""
    if not archs:
        return []
    floor = min(arch_key(a) for a in archs)
    out = []
    for row in sorted(our_rows):
        match = ARCH_ROW.search(row)
        if match and int(match.group(1)) < floor:
            out.append(row)
    return out


def registry_archs(text: str) -> dict[str, int]:
    """Architecture name -> line in vLLM's registry."""
    found: dict[str, int] = {}
    for match in REGISTRY_ENTRY.finditer(text):
        found.setdefault(match.group(1), text[: match.start()].count("\n") + 1)
    return found


def our_named_archs() -> set[str]:
    text = (ROOT / ".agents/model-matrix.md").read_text(encoding="utf-8")
    return set(re.findall(
        r"`([A-Za-z0-9_]+(?:ForCausalLM|ForConditionalGeneration|Model|"
        r"ForSequenceClassification|ForMaskedLM|ForTokenClassification|"
        r"ForRetrieval))`", text))


def our_arch_rows() -> set[str]:
    text = (ROOT / ".agents/backend-matrix.md").read_text(encoding="utf-8")
    return set(re.findall(r"BACKEND-CUDA-SM\d{3}", text))


def component_presence(vllm: Path) -> dict[str, dict]:
    out = {}
    for name in COMPONENTS:
        try:
            files = subprocess.run(
                ["find", str(vllm / "csrc"), str(vllm / "vllm"), "-iname", f"*{name}*"],
                capture_output=True, text=True).stdout.splitlines()
        except OSError:
            files = []
        first = ""
        if files:
            rel = Path(files[0])
            try:
                first = f"{rel.relative_to(vllm)}:1"
            except ValueError:
                first = f"{rel}:1"
        out[name] = {"files": len(files), "anchor": first}
    return out


def build() -> dict:
    refs = references_available()
    if "vllm" not in refs:
        return {"available": False}
    vllm = refs["vllm"]

    cmake = (vllm / "CMakeLists.txt").read_text(encoding="utf-8", errors="replace")
    archs, arch_line = parse_supported_archs(cmake)

    registry_path = vllm / "vllm/model_executor/models/registry.py"
    registry = registry_archs(registry_path.read_text(encoding="utf-8", errors="replace"))
    ours = our_named_archs()
    missing = sorted(set(registry) - ours)

    return {
        "available": True,
        "pins": {name: head_of(path)[:9] for name, path in refs.items()},
        "arch_floor": {
            "supported": archs,
            "anchor": f"CMakeLists.txt:{arch_line}",
            "our_rows_below_floor": below_floor(our_arch_rows(), archs),
        },
        "components": component_presence(vllm),
        "registry": {
            "upstream_total": len(registry),
            "named_by_us": len(set(registry) & ours),
            "missing_count": len(missing),
            "missing": [
                {"arch": a, "anchor": f"vllm/model_executor/models/registry.py:{registry[a]}"}
                for a in missing
            ],
        },
    }


def render(data: dict) -> str:
    if not data.get("available"):
        return "Reference checkouts are not available; nothing derived."
    lines = [
        "Upstream-derived inventory",
        f"  pins: {data['pins']}",
        f"  vLLM supported archs ({data['arch_floor']['anchor']}): "
        f"{' '.join(data['arch_floor']['supported'])}",
        f"  our arch rows BELOW that floor: "
        f"{', '.join(data['arch_floor']['our_rows_below_floor']) or 'none'}",
        "  components:",
    ]
    for name, info in data["components"].items():
        verdict = "NOT in vLLM (external dep)" if info["files"] == 0 else "real, unported"
        lines.append(f"    {name:16} files={info['files']:<4} {verdict}")
    reg = data["registry"]
    lines += [
        f"  registry: {reg['upstream_total']} upstream archs, "
        f"{reg['named_by_us']} named by us, {reg['missing_count']} with NO ROW",
    ]
    for entry in reg["missing"][:10]:
        lines.append(f"    missing {entry['arch']:38} {entry['anchor']}")
    if reg["missing_count"] > 10:
        lines.append(f"    ... (+{reg['missing_count'] - 10} more)")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--emit", action="store_true", help="refresh the snapshot")
    mode.add_argument("--check", action="store_true", help="fail on drift")
    args = parser.parse_args()

    data = build()

    if args.emit:
        SNAPSHOT.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        print(render(data))
        print(f"\nsnapshot written: {SNAPSHOT.relative_to(ROOT)}")
        return 0

    if args.check:
        if not data.get("available"):
            print("SKIP: reference checkouts unavailable; upstream drift not checked "
                  "(set VLLM_SOURCE to enable).")
            return 0
        if not SNAPSHOT.exists():
            print(f"ERROR: {SNAPSHOT.relative_to(ROOT)} is missing; run --emit",
                  file=sys.stderr)
            return 1
        stored = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        drift = []
        if stored.get("registry", {}).get("missing_count") != data["registry"]["missing_count"]:
            drift.append(
                f"uninventoried upstream architectures moved "
                f"{stored['registry']['missing_count']} -> "
                f"{data['registry']['missing_count']}")
        if stored.get("arch_floor", {}).get("supported") != data["arch_floor"]["supported"]:
            drift.append("vLLM's supported CUDA arch list changed")
        if drift:
            for item in drift:
                print(f"ERROR: {item}", file=sys.stderr)
            print("Upstream moved. Re-run --emit and inventory the delta.",
                  file=sys.stderr)
            return 1
        print("OK: the agent record matches the upstream inventory snapshot.")
        return 0

    print(render(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
