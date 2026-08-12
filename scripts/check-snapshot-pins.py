#!/usr/bin/env python3
"""GATE-PIN-UNPINNED-SNAPSHOTS (#471) — no NEW unpinned checkpoint resolution.

A gate that resolves its checkpoint by `std::filesystem::directory_iterator`
over `<repo>/snapshots/` measures whichever revision readdir happens to yield
first. That is a property of the box, not of the gate. `unsloth/Qwen3.6-27B-NVFP4`
caches two materially different models under one repo name, so the failure mode
is not hypothetical.

The fix is `parity::HfSnapshot` in tests/parity/hf_snapshot.h, which names the
revision and returns "" — the caller's loud skip — when the cache holds a
different one.

This checker fails on any unpinned resolution that is not in LEDGER below.

ON THE LEDGER. It exists because ~48 gates cannot be pinned today: their goldens
record no revision at all (19 `*_greedy*` corpora have no manifest file
whatsoever), and pinning to "whatever is cached here" is the defect wearing a
constant's name. #472 owes the re-capture. Every line is a debt:

  * DELETING a line is the work.
  * ADDING a line is a review event and needs an argument in the commit message
    that excuses it, per the no-waiver-registry rule in AGENTS.md.

The ledger is written only when an unpinned resolver is added or removed, never
by every PR, so it is not the shared-lock shape AGENTS.md forbids.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# An unpinned resolution: a directory_iterator whose subject is a `snapshots`
# path. Matched on the statement, then attributed to the nearest `snapshots`
# mention above it, because the two are usually a few lines apart.
ITER_RE = re.compile(r"\bdirectory_iterator\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)")
SNAPSHOT_ASSIGN_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=[^;]*?snapshots", re.DOTALL
)

# file -> why it cannot be pinned yet. Sorted, one line each, SHRINKING.
LEDGER: dict[str, str] = {
    # --- tests/parity: *_greedy corpora with no manifest file at all (#472) ---
    "tests/parity/test_commandr_paged_engine.cpp": "goldens/commandr_greedy has no manifest (#472)",
    "tests/parity/test_gemma4_paged_engine.cpp": "goldens/gemma4_e4b_text records model_id only (#472)",
    "tests/parity/test_glm4_paged_engine.cpp": "goldens/glm4_greedy_9b has no manifest (#472)",
    "tests/parity/test_granite_paged_engine.cpp": "goldens/granite_greedy_2b has no manifest (#472)",
    "tests/parity/test_internlm2_paged_engine.cpp": "goldens/internlm2_greedy_1_8b has no manifest (#472)",
    "tests/parity/test_internlm3_paged_engine.cpp": "goldens/internlm3_greedy_8b has no manifest (#472)",
    "tests/parity/test_llama_paged_engine.cpp": "goldens/llama_greedy_1b has no manifest (#472)",
    "tests/parity/test_minicpm3_paged_engine.cpp": "goldens/minicpm3_greedy_4b has no manifest (#472)",
    "tests/parity/test_minicpm_paged_engine.cpp": "goldens/minicpm_greedy_2b has no manifest (#472)",
    "tests/parity/test_mistral_paged_engine.cpp": "goldens/mistral_greedy_7b has no manifest (#472)",
    "tests/parity/test_olmo2_paged_engine.cpp": "goldens/olmo2_greedy_1b has no manifest (#472)",
    "tests/parity/test_olmo3_paged_engine.cpp": "goldens/olmo3_greedy_7b has no manifest (#472)",
    "tests/parity/test_phi3_paged_engine.cpp": "goldens/phi4_*_greedy have no manifest (#472)",
    "tests/parity/test_phi_paged_engine.cpp": "goldens/phi2_greedy_2_7b has no manifest (#472)",
    "tests/parity/test_qwen3_apc_e2e.cpp": "goldens/qwen3_apc_4b has no manifest (#472)",
    "tests/parity/test_qwen3_dense_async_serving.cpp": "no goldens at all (#472)",
    "tests/parity/test_qwen3_paged_engine.cpp": "goldens/qwen3_greedy_* have no manifest (#472)",
    "tests/parity/test_stablelm_paged_engine.cpp": "goldens/stablelm_greedy_1_6b has no manifest (#472)",
    "tests/parity/test_yi_paged_engine.cpp": "goldens/yi_greedy_coder_1_5b has no manifest (#472)",
    # --- tests/vllm/models: load/forward tests, no goldens to derive from ---
    "tests/vllm/models/test_deepseek_v2_forward.cpp": "no goldens (#472)",
    "tests/vllm/models/test_deepseek_v2_load.cpp": "no goldens (#472)",
    "tests/vllm/models/test_deepseek_v2_paged_engine.cpp": "goldens/deepseek_v2_greedy has no manifest (#472)",
    "tests/vllm/models/test_gemma2_forward.cpp": "no goldens (#472)",
    "tests/vllm/models/test_gemma2_load.cpp": "no goldens (#472)",
    "tests/vllm/models/test_gemma3_forward.cpp": "no goldens (#472)",
    "tests/vllm/models/test_gemma3_load.cpp": "no goldens (#472)",
    "tests/vllm/models/test_gemma_forward.cpp": "no goldens (#472)",
    "tests/vllm/models/test_gemma_load.cpp": "no goldens (#472)",
    "tests/vllm/models/test_glm4_moe_lite_load.cpp": "no goldens (#472)",
    "tests/vllm/models/test_glm4_moe_lite_paged_engine.cpp": "goldens/glm4_moe_lite_greedy has no manifest (#472)",
    "tests/vllm/models/test_llama_forward.cpp": "no goldens (#472)",
    "tests/vllm/models/test_llama_load.cpp": "no goldens (#472)",
    "tests/vllm/models/test_mistral_forward.cpp": "no goldens (#472)",
    "tests/vllm/models/test_mistral_load.cpp": "no goldens (#472)",
    "tests/vllm/models/test_moe_two_engines.cpp": "no goldens (#472)",
    "tests/vllm/models/test_qwen35_plain_weights.cpp": "no goldens (#472)",
    "tests/vllm/models/test_qwen3_32b_nvfp4a16_load.cpp": "no goldens (#472)",
    "tests/vllm/models/test_qwen3_32b_nvfp4a16_paged_engine.cpp": "goldens/qwen3_32b_nvfp4a16_greedy has no manifest (#472)",
    "tests/vllm/models/test_qwen3_forward.cpp": "no goldens (#472)",
    "tests/vllm/models/test_qwen3_load.cpp": "no goldens (#472)",
    "tests/vllm/models/test_qwen3_moe_forward.cpp": "no goldens (#472)",
    "tests/vllm/models/test_qwen3_moe_load.cpp": "no goldens (#472)",
    "tests/vllm/models/test_qwen3coder_paged_engine.cpp": "goldens/qwen3coder_greedy has no manifest (#472)",
    # --- tests/vllm/multimodal ---
    "tests/vllm/multimodal/bench_qwen3_5_vl_tower.cpp": "benchmark, no goldens (#472)",
    "tests/vllm/multimodal/test_gemma4_registry_e2e.cpp": "goldens/gemma4_e4b_image records model_id only (#472)",
    "tests/vllm/multimodal/test_qwen3_5_vl_e2e.cpp": "manifest records no revision (#472)",
    "tests/vllm/multimodal/test_qwen3_5_vl_video_e2e.cpp": "manifest records no revision (#472)",
    "tests/vllm/multimodal/test_qwen3vl_e2e.cpp": "manifest records no revision (#472)",
    "tests/vllm/multimodal/test_qwen3vl_registry_e2e.cpp": "no goldens (#472)",
    "tests/vllm/multimodal/test_qwen3vl_video_e2e.cpp": "manifest records no revision (#472)",
}


def unpinned_resolutions(root: pathlib.Path) -> dict[str, list[int]]:
    """{relpath: [line numbers]} for every unpinned checkpoint resolution."""
    found: dict[str, list[int]] = {}
    tests = root / "tests"
    if not tests.is_dir():
        return found
    for path in sorted(tests.rglob("*")):
        if path.suffix not in (".cpp", ".h") or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "directory_iterator" not in text or "snapshots" not in text:
            continue
        # Variables that were assigned a path mentioning `snapshots`.
        snapshot_vars = set(SNAPSHOT_ASSIGN_RE.findall(text))
        rel = path.relative_to(root).as_posix()
        for lineno, line in enumerate(text.splitlines(), start=1):
            if line.lstrip().startswith("//"):
                continue
            match = ITER_RE.search(line)
            if match is None:
                continue
            subject = match.group(1)
            if subject in snapshot_vars or "snapshots" in line:
                found.setdefault(rel, []).append(lineno)
    return found


def check(root: pathlib.Path) -> list[str]:
    findings = unpinned_resolutions(root)
    problems: list[str] = []
    for rel, lines in sorted(findings.items()):
        if rel not in LEDGER:
            where = ", ".join(f"{rel}:{n}" for n in lines)
            problems.append(
                f"UNPINNED checkpoint resolution not in the ledger: {where}\n"
                "    A gate may not choose its own subject. Resolve through\n"
                "    parity::HfSnapshot (tests/parity/hf_snapshot.h) with the\n"
                "    revision your goldens record. If your goldens record none,\n"
                "    they must be re-captured (#472) -- do NOT pin to whatever is\n"
                "    cached on your box, and do NOT add a ledger line without an\n"
                "    argument for it in the commit message."
            )
    for rel in sorted(LEDGER):
        if rel not in findings:
            problems.append(
                f"STALE ledger entry: {rel} no longer resolves unpinned.\n"
                "    Delete the line. A ledger that outlives its debt starts\n"
                "    excusing resolutions nobody reviewed."
            )
    return problems


UNPINNED_FIXTURE = """#include <filesystem>
#include <string>
namespace fs = std::filesystem;
std::string Resolve() {
  const fs::path snaps = fs::path("/x") / "snapshots";
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(snaps, ec))
    if (fs::exists(e.path() / "config.json", ec)) return e.path().string();
  return "";
}
"""

PINNED_FIXTURE = """#include <filesystem>
#include <string>
#include "hf_snapshot.h"
namespace fs = std::filesystem;
std::string Resolve() { return parity::Qwen27NvfP4Snapshot(); }
"""


def self_test() -> int:
    """RED-before proof: the checker must FAIL on a synthesised unpinned resolver.

    Without this, the checker could be turned green by weakening its own pattern
    and nothing would notice.
    """
    failures = 0
    with tempfile.TemporaryDirectory() as raw:
        scratch = pathlib.Path(raw)
        target = scratch / "tests" / "parity" / "test_synthetic_gate.cpp"
        target.parent.mkdir(parents=True)

        target.write_text(UNPINNED_FIXTURE, encoding="utf-8")
        red = check(scratch)
        red_hit = any("UNPINNED checkpoint resolution" in p for p in red)
        print(f"self-test RED  (unpinned fixture): {'DETECTED' if red_hit else 'MISSED'}")
        if not red_hit:
            failures += 1

        target.write_text(PINNED_FIXTURE, encoding="utf-8")
        green = [p for p in check(scratch) if "UNPINNED" in p]
        print(f"self-test GREEN (pinned fixture):   {'clean' if not green else green}")
        if green:
            failures += 1

        # A ledger line must actually excuse the file it names, and only that one.
        target.write_text(UNPINNED_FIXTURE, encoding="utf-8")
        LEDGER["tests/parity/test_synthetic_gate.cpp"] = "self-test"
        try:
            excused = [p for p in check(scratch) if "UNPINNED" in p]
            print(f"self-test LEDGER (excused):         {'clean' if not excused else excused}")
            if excused:
                failures += 1
        finally:
            del LEDGER["tests/parity/test_synthetic_gate.cpp"]
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="prove the checker fails on a synthesised unpinned resolver",
    )
    args = parser.parse_args()

    if args.self_test:
        failures = self_test()
        print("SELF-TEST OK" if failures == 0 else f"SELF-TEST FAILED ({failures})")
        return 1 if failures else 0

    problems = check(REPO_ROOT)
    if problems:
        print("check-snapshot-pins: FAIL")
        for problem in problems:
            print(f"  {problem}")
        return 1
    ledger = len(LEDGER)
    print(
        f"check-snapshot-pins: OK — no unpinned checkpoint resolution outside the "
        f"ledger ({ledger} file(s) still owed on #472)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
