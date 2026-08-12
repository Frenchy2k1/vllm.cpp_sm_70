#!/usr/bin/env python3
"""GATE-PIN-UNPINNED-SNAPSHOTS (#471) — no NEW unpinned checkpoint resolution.

A gate that resolves its checkpoint by enumerating `<repo>/snapshots/` measures
whichever revision readdir happens to yield first. That is a property of the box,
not of the gate. `unsloth/Qwen3.6-27B-NVFP4` caches two materially different
models under one repo name, so the failure mode is not hypothetical.

The fix is `parity::HfSnapshot` in tests/parity/hf_snapshot.h, which names the
revision and returns "" — the caller's loud skip — when the cache holds a
different one.

This checker fails on any unpinned resolution that is not in LEDGER below.

WHAT IT GUARANTEES, EXACTLY. Read this before citing it; the first version of
this file claimed more than it delivered.

  * IT IS A LEXICAL CHECKER. It matches normalized source text. It does not
    parse C++ or Python, does not resolve types, and does not follow a value
    across a file boundary. A resolution assembled in one file and enumerated in
    another is NOT caught.
  * WHAT IT MODELS is a *class*, not one idiom: any of the enumerators in
    `ENUMERATORS` (`directory_iterator`, `recursive_directory_iterator`,
    `opendir`, `scandir`, `readdir`, `listdir`, `iterdir`, `walk`, `glob`,
    `iglob`, `rglob`) whose call text — receiver chain, name and *balanced*
    argument list, so line wrapping is irrelevant — mentions an HF-cache marker
    (`CACHE_MARKERS`) or names a value this file bound to one. Binding is
    followed to a fixpoint through assignments, through struct members, and
    through a helper that RETURNS the path, so renaming the variable or hiding
    the literal behind a `constexpr` does not evade it.
  * SCOPE IS BY ROOT AND SUFFIX: `SCAN_ROOTS` x `SCANNED_SUFFIXES`. `src/` is
    deliberately NOT scanned — `src/vllm/entrypoints/model_loader.cpp` resolves a
    user's cache at run time, which is the product working, not a gate choosing
    its own subject. `examples/` likewise.
  * IT IS DEFEATABLE ON PURPOSE. Shelling out to `find`, reading a path from an
    environment variable whose name carries no marker, or a `#define`d enumerator
    all evade it. It raises the cost of the ACCIDENT — the ordinary
    copy-paste that reintroduces readdir-order resolution — and it is not an
    adversarial control.

  * THE SELF-TEST (`--self-test`) sweeps the `FIXTURES` corpus below, in BOTH
    directions: every `unpinned=True` idiom must be reported and every
    `unpinned=False` one must not. That is what makes "the checker cannot be
    greened by narrowing its own pattern" a testable claim rather than an
    assertion — narrowing any single branch (the enumerator alternation, the
    `fs::` spelling, the scanned roots, the scanned suffixes, the subject-binding
    rule) drops at least one positive fixture, and dropping the binding rule
    entirely raises at least one negative. It is still only as good as the
    corpus: an idiom nobody wrote a fixture for is not covered by the claim.

ON THE LEDGER. It exists because the files listed in LEDGER cannot be pinned
today: their goldens record no revision at all (19 `*_greedy*` corpora have no
manifest file whatsoever), and pinning to "whatever is cached here" is the defect
wearing a constant's name. #472 owes the re-capture. The last five entries are
`scripts/` staging and reference-dump helpers rather than gates — they FEED
goldens instead of asserting them, which is the same hazard one step upstream.
Every line is a debt:

  * DELETING a line is the work. The STALE check makes it a one-way ratchet: once
    a file is pinned, its ledger line MUST go or the checker fails.
  * ADDING a line is a review event and needs an argument in the commit message
    that excuses it, per the no-waiver-registry rule in AGENTS.md. That argument
    IS the control, and it is deliberately the only one: a machine-checkable
    admission test would be a waiver registry with extra steps, which AGENTS.md
    forbids. `check_ledger_shape` therefore enforces only what is mechanical —
    a non-empty reason naming a tracking issue — and the question of whether the
    debt is legitimate stays with the reviewer who reads the diff.

The ledger is written only when an unpinned resolver is added or removed, never
by every PR, so it is not the shared-lock shape AGENTS.md forbids.
"""

from __future__ import annotations

import argparse
import collections
import io
import pathlib
import re
import sys
import tempfile
import tokenize

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# `scripts/` is sys.path[0] when this file is RUN, but not when a test loads it by
# path, so pin it either way.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from checker_text import blank_out, match_braces, normalize_source  # noqa: E402

# --------------------------------------------------------------------------- #
# Scope
# --------------------------------------------------------------------------- #

SCAN_ROOTS = ("tests", "tools", "scripts")
CXX_SUFFIXES = frozenset({".cpp", ".cc", ".cxx", ".h", ".hpp", ".hh"})
PY_SUFFIXES = frozenset({".py"})
SCANNED_SUFFIXES = CXX_SUFFIXES | PY_SUFFIXES

# This checker and its suite carry unpinned resolvers as FIXTURE TEXT, by
# construction. Excluded BY EXACT PATH — never by a directory rule, which would
# also excuse every future file dropped beside them.
SELF_EXCLUDED = frozenset(
    {
        "scripts/check-snapshot-pins.py",
        "tests/scripts/test_check_snapshot_pins.py",
    }
)

# --------------------------------------------------------------------------- #
# The pattern
# --------------------------------------------------------------------------- #

# Text that says "this names an HF snapshot cache". `models--` is the cache's own
# repo-directory prefix, so a path built from it is a resolution even when the
# word `snapshots` never appears.
CACHE_MARKERS = (
    "snapshots",
    "models--",
    "HF_HUB_CACHE",
    "HF_HOME",
    "TRANSFORMERS_CACHE",
)
CACHE_MARKER_RE = re.compile("|".join(re.escape(m) for m in CACHE_MARKERS))

# Every way this tree enumerates a directory, C++ and Python. `readdir`/`scandir`
# are listed with `opendir` because the POSIX idiom resolves at the `opendir`.
ENUMERATORS = (
    "recursive_directory_iterator",
    "directory_iterator",
    "opendir",
    "scandir",
    "readdir",
    "listdir",
    "iterdir",
    "walk",
    "rglob",
    "iglob",
    "glob",
)
ENUMERATOR_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:" + "|".join(ENUMERATORS) + r")[ \t\r\n]*\("
)

RECEIVER_SEPARATORS = (".", "::", "->")
RECEIVER_CLOSERS = {")": "(", "]": "["}

IDENT_RE = re.compile(r"[A-Za-z_]\w*")
ASSIGN_LHS_RE = re.compile(r"([A-Za-z_]\w*)\s*=(?!=)")
RETURN_RE = re.compile(r"\breturn\b([^;\n]*)")

# A C++ function definition: a return type, the name, a parameter list, `{`.
# Requiring the return type keeps `if (`, `for (` and `TEST_CASE("…") {` out.
CXX_FUNCDEF_RE = re.compile(
    r"[A-Za-z_][\w:<>,\s]*[\s&*]([A-Za-z_]\w*)\s*\([^;{}]*\)\s*"
    r"(?:const\s*)?(?:noexcept\s*)?\{"
)
PY_FUNCDEF_RE = re.compile(r"^([ \t]*)def[ \t]+([A-Za-z_]\w*)\s*\(", re.M)

_MARK_FIXPOINT_ROUNDS = 6


def normalize_python(text: str) -> str:
    """Blank `#` comments and docstrings, position-preserving.

    The Python analogue of scripts/checker_text.py: a resolver that survives only
    inside a comment or a usage-example docstring is text the interpreter never
    runs, and must not read as a resolution. String literals that are ARGUMENTS
    are kept — `glob("…/snapshots/*")` is the very thing being detected.

    A file that does not tokenize is returned unchanged: a syntax error is a
    different problem, and the conservative reading is to keep all the text.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return text

    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))

    def offset(row: int, col: int) -> int:
        if row - 1 >= len(offsets):
            return len(text)
        return min(offsets[row - 1] + col, len(text))

    spans: list[tuple[int, int]] = []
    line_start = True
    for token in tokens:
        if token.type == tokenize.COMMENT:
            spans.append((offset(*token.start), offset(*token.end)))
            continue
        if (
            token.type == tokenize.STRING
            and line_start
            and re.match(r"^[a-zA-Z]*('''|\"\"\")", token.string)
        ):
            spans.append((offset(*token.start), offset(*token.end)))
        if token.type in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT):
            line_start = True
        elif token.type != tokenize.COMMENT:
            line_start = False

    out = text
    for start, end in reversed(spans):
        out = out[:start] + blank_out(out[start:end]) + out[end:]
    return out


def normalize(text: str, python: bool) -> str:
    """Source with everything the toolchain never sees blanked out, in place."""
    return normalize_python(text) if python else normalize_source(text)


def _logical_statements(text: str, python: bool) -> list[str]:
    """Statements, joined across physical lines.

    A C++ statement ends at `;`, `{` or `}`. A Python statement ends at a newline
    that closes every bracket. Either way an assignment spread over three lines by
    clang-format or black is ONE string here, which is what stops line wrapping
    from being an evasion.
    """
    if not python:
        return re.split(r"[;{}]", text)
    statements: list[str] = []
    buffer: list[str] = []
    depth = 0
    for line in text.splitlines():
        buffer.append(line)
        depth += line.count("(") + line.count("[") + line.count("{")
        depth -= line.count(")") + line.count("]") + line.count("}")
        if depth <= 0 and not line.rstrip().endswith("\\"):
            statements.append(" ".join(buffer))
            buffer = []
            depth = 0
    if buffer:
        statements.append(" ".join(buffer))
    return statements


def _function_bodies(text: str, python: bool) -> list[tuple[str, str]]:
    """`[(name, body)]` for every function defined in `text`."""
    bodies: list[tuple[str, str]] = []
    if python:
        lines = text.splitlines(keepends=True)
        starts = [0]
        for line in lines:
            starts.append(starts[-1] + len(line))
        for match in PY_FUNCDEF_RE.finditer(text):
            indent = len(match.group(1).expandtabs(8))
            row = text.count("\n", 0, match.start())
            end = len(text)
            for index in range(row + 1, len(lines)):
                line = lines[index]
                if not line.strip():
                    continue
                if len(line[: len(line) - len(line.lstrip())].expandtabs(8)) <= indent:
                    end = starts[index]
                    break
            bodies.append((match.group(2), text[match.end() : end]))
        return bodies
    for match in CXX_FUNCDEF_RE.finditer(text):
        end = match_braces(text, match.end())
        bodies.append((match.group(1), text[match.end() : end]))
    return bodies


def marked_names(text: str, python: bool) -> set[str]:
    """Every name this text binds, directly or transitively, to a cache path.

    Three producers, iterated to a fixpoint so an indirection chain
    (`kSnapDir = "snapshots"` -> `root = base / kSnapDir` -> `SnapRoot()`) is
    followed all the way:

      * an assignment whose right-hand side carries a marker or a marked name —
        this also covers `cache.snaps_dir = …`, whose left-hand identifier is the
        member;
      * a function whose body carries a marker;
      * a function that RETURNS a marked name.
    """
    marked: set[str] = set()
    statements = _logical_statements(text, python)
    bodies = _function_bodies(text, python)
    for _ in range(_MARK_FIXPOINT_ROUNDS):
        before = len(marked)
        for statement in statements:
            assignment = ASSIGN_LHS_RE.search(statement)
            if assignment is None:
                continue
            rhs = statement[assignment.end() :]
            if CACHE_MARKER_RE.search(rhs) or marked & set(IDENT_RE.findall(rhs)):
                marked.add(assignment.group(1))
        for name, body in bodies:
            if name in marked:
                continue
            if CACHE_MARKER_RE.search(body):
                marked.add(name)
                continue
            returned = " ".join(RETURN_RE.findall(body))
            if marked & set(IDENT_RE.findall(returned)):
                marked.add(name)
        if len(marked) == before:
            break
    return marked


def _receiver(text: str, start: int) -> str:
    """The qualifier/receiver chain immediately left of the call, or "".

    `fs::`, `std::filesystem::`, `glob.`, `cache.snaps_dir.` — and
    `snap_root(cache).`, which is why this walks backwards over BALANCED
    brackets rather than matching a regex: a Python method call on the result of
    a helper is ordinary code, and a chain that stops at `)` loses the one
    identifier that names the subject.
    """
    index = start
    if not any(text[:index].endswith(sep) for sep in RECEIVER_SEPARATORS):
        return ""
    while index > 0:
        character = text[index - 1]
        if character in RECEIVER_CLOSERS:
            depth = 0
            while index > 0:
                index -= 1
                if text[index] in RECEIVER_CLOSERS:
                    depth += 1
                elif text[index] in RECEIVER_CLOSERS.values():
                    depth -= 1
                    if depth == 0:
                        break
            else:
                break
        elif character.isalnum() or character == "_" or character == ".":
            index -= 1
        elif text[index - 2 : index] in ("::", "->"):
            index -= 2
        else:
            break
    return text[index:start]


def _call_text(text: str, start: int) -> str:
    """The receiver chain, the enumerator name and its BALANCED argument list.

    Balanced rather than line-scoped: `directory_iterator(\\n    snaps, ec)` is
    what clang-format produces, and a per-line match misses it.
    """
    receiver = _receiver(text, start)
    open_paren = text.find("(", start)
    if open_paren < 0:
        return text[start:]
    depth = 0
    index = open_paren
    while index < len(text):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                index += 1
                break
        index += 1
    return receiver + text[start:index]


def resolutions_in(raw: str, python: bool) -> list[int]:
    """Line numbers of every unpinned checkpoint resolution in one file's text."""
    text = normalize(raw, python)
    if not CACHE_MARKER_RE.search(text) or not ENUMERATOR_RE.search(text):
        return []
    names = marked_names(text, python)
    hits: set[int] = set()
    for match in ENUMERATOR_RE.finditer(text):
        call = _call_text(text, match.start())
        if CACHE_MARKER_RE.search(call) or names & set(IDENT_RE.findall(call)):
            hits.add(text.count("\n", 0, match.start()) + 1)
    return sorted(hits)


# --------------------------------------------------------------------------- #
# The ledger
# --------------------------------------------------------------------------- #

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
    # --- scripts/: staging and reference-dump helpers, reached by widening the
    #     scan to `scripts/` and `.py`. These FEED goldens rather than assert
    #     them, which is the same hazard one step upstream (#472) ---
    "scripts/minicpm-convert-safetensors.py": "stages a checkpoint by glob order, no revision recorded (#472)",
    "scripts/minicpm3-convert-safetensors.py": "stages a checkpoint by glob order, no revision recorded (#472)",
    "scripts/mm/a3_voxtral_wcheck.py": "one-off Voxtral weight dump, glob(snapshots/*) (#472)",
    "scripts/mm/g3_audio_tower_ref.py": "gemma-4 audio reference dump, glob(snapshots/*)[0] (#472)",
    "scripts/stage-tokenizer-metaspace.py": "stages a tokenizer by glob order, no revision recorded (#472)",
}

LEDGER_ISSUE_RE = re.compile(r"\(#\d+\)")


def check_ledger_shape() -> list[str]:
    """Every ledger line must at least name the issue that owes its removal.

    Deliberately the ONLY mechanical admission test. See the module docstring:
    the real control on a new line is the argument in the commit message, and a
    machine-checkable admission rule here would be the waiver registry AGENTS.md
    forbids.
    """
    problems = []
    for rel, reason in sorted(LEDGER.items()):
        if not LEDGER_ISSUE_RE.search(reason or ""):
            problems.append(
                f"LEDGER line without a tracking issue: {rel}\n"
                "    A debt with no issue is an exemption. Name the issue that\n"
                "    owes its removal."
            )
    return problems


def unpinned_resolutions(root: pathlib.Path) -> dict[str, list[int]]:
    """{relpath: [line numbers]} for every unpinned checkpoint resolution."""
    found: dict[str, list[int]] = {}
    for base in SCAN_ROOTS:
        directory = root / base
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix not in SCANNED_SUFFIXES or not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            if rel in SELF_EXCLUDED:
                continue
            lines = resolutions_in(
                path.read_text(encoding="utf-8", errors="replace"),
                python=path.suffix in PY_SUFFIXES,
            )
            if lines:
                found[rel] = lines
    return found


def check(root: pathlib.Path) -> list[str]:
    findings = unpinned_resolutions(root)
    problems: list[str] = check_ledger_shape()
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


# --------------------------------------------------------------------------- #
# The fixture corpus — the self-test's whole substance
# --------------------------------------------------------------------------- #


# One synthesised source file and what the checker owes on it. `kills` names the
# narrowing each POSITIVE fixture exists to catch, so the corpus reads as a list
# of claims rather than a pile of files.
#
# A namedtuple rather than a dataclass DELIBERATELY: this module is loaded by
# `spec_from_file_location` without a `sys.modules` entry (tests/scripts/, and
# any other tool that wants the LEDGER), and `@dataclass` resolves annotations
# through `sys.modules[cls.__module__]`, so it raises under exactly that loader.
Fixture = collections.namedtuple(
    "Fixture", ("name", "rel", "body", "unpinned", "kills")
)


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

FIXTURES: tuple[Fixture, ...] = (
    Fixture(
        name="baseline",
        rel="tests/parity/test_synthetic_gate.cpp",
        body=UNPINNED_FIXTURE,
        unpinned=True,
        kills="the idiom the row actually removed",
    ),
    Fixture(
        name="recursive_iterator",
        rel="tests/parity/test_recursive.cpp",
        body="""#include <filesystem>
namespace fs = std::filesystem;
std::string Resolve() {
  const fs::path snaps = fs::path("/x") / "snapshots";
  std::error_code ec;
  for (const auto& e : fs::recursive_directory_iterator(snaps, ec))
    return e.path().string();
  return "";
}
""",
        unpinned=True,
        kills="a `\\b`-anchored `directory_iterator` never matches after `_`",
    ),
    Fixture(
        name="line_wrapped_call",
        rel="tests/parity/test_wrapped.cpp",
        body="""#include <filesystem>
namespace fs = std::filesystem;
std::string Resolve() {
  const fs::path snapshot_root =
      fs::path("/x") / "models--org--name" / "snapshots";
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(
           snapshot_root, ec))
    return e.path().string();
  return "";
}
""",
        unpinned=True,
        kills="matching PER LINE — this is what clang-format produces",
    ),
    Fixture(
        name="call_subject",
        rel="tests/parity/test_call_subject.cpp",
        body="""#include <filesystem>
namespace fs = std::filesystem;
static fs::path SnapRoot() { return fs::path("/x") / "snapshots"; }
std::string Resolve() {
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(SnapRoot(), ec))
    return e.path().string();
  return "";
}
""",
        unpinned=True,
        kills="requiring the subject to be a `snapshots`-assigned VARIABLE",
    ),
    Fixture(
        name="helper_returns_a_marked_value",
        rel="tests/parity/test_returned_subject.cpp",
        body="""#include <filesystem>
namespace fs = std::filesystem;
static const fs::path kRoot = fs::path("/x") / "snapshots";
static fs::path SnapRoot() { return kRoot; }
std::string Resolve() {
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(SnapRoot(), ec))
    return e.path().string();
  return "";
}
""",
        unpinned=True,
        kills="marking a helper only when its own BODY carries the literal",
    ),
    Fixture(
        name="member_subject",
        rel="tests/parity/test_member_subject.cpp",
        body="""#include <filesystem>
namespace fs = std::filesystem;
struct Cache { fs::path snaps_dir; };
std::string Resolve(Cache& cache) {
  cache.snaps_dir = fs::path("/x") / "snapshots";
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(cache.snaps_dir, ec))
    return e.path().string();
  return "";
}
""",
        unpinned=True,
        kills="tracking bare variables only, never a struct member",
    ),
    Fixture(
        name="constant_indirection",
        rel="tests/parity/test_indirect_constant.cpp",
        body="""#include <filesystem>
namespace fs = std::filesystem;
static constexpr const char* kSnapDir = "snapshots";
std::string Resolve(const fs::path& base) {
  const fs::path root = base / kSnapDir;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(root, ec))
    return e.path().string();
  return "";
}
""",
        unpinned=True,
        kills="requiring the literal in the ASSIGNMENT that feeds the call",
    ),
    Fixture(
        name="fully_qualified_spelling",
        rel="tests/vllm/models/test_qualified.cpp",
        body="""#include <filesystem>
std::string Resolve() {
  const std::filesystem::path snaps =
      std::filesystem::path("/x") / "snapshots";
  std::error_code ec;
  for (const auto& e : std::filesystem::directory_iterator(snaps, ec))
    return e.path().string();
  return "";
}
""",
        unpinned=True,
        kills="requiring the `fs::` spelling; also that only tests/parity is scanned",
    ),
    Fixture(
        name="header_hpp",
        rel="tests/vllm/multimodal/resolve_snapshot.hpp",
        body="""#pragma once
#include <filesystem>
namespace fs = std::filesystem;
inline std::string Resolve() {
  const fs::path snaps = fs::path("/x") / "snapshots";
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(snaps, ec))
    return e.path().string();
  return "";
}
""",
        unpinned=True,
        kills="scanning only `.cpp`/`.h`",
    ),
    Fixture(
        name="translation_unit_cc",
        rel="tests/vllm/v1/resolve_snapshot.cc",
        body=UNPINNED_FIXTURE,
        unpinned=True,
        kills="scanning only `.cpp`/`.h`",
    ),
    Fixture(
        name="posix_opendir",
        rel="tests/parity/test_opendir.cpp",
        body="""#include <dirent.h>
#include <string>
std::string Resolve(const std::string& base) {
  const std::string snaps = base + "/snapshots";
  DIR* d = opendir(snaps.c_str());
  struct dirent* e = readdir(d);
  return e ? e->d_name : "";
}
""",
        unpinned=True,
        kills="modelling `directory_iterator` only, not the POSIX idiom",
    ),
    Fixture(
        name="python_glob_under_tools",
        rel="tools/bench/resolve_gate_snapshot.py",
        body='''"""Docstring mentioning snapshots, which is not code."""
import glob
import os


def resolve(cache):
    base = os.path.join(cache, "snapshots")
    return sorted(glob.glob(os.path.join(base, "*")))[0]
''',
        unpinned=True,
        kills="scanning C++ only; also that only `tests/` is scanned",
    ),
    Fixture(
        name="python_iterdir_under_tests",
        rel="tests/tools/resolve_gate_snapshot.py",
        body='''import pathlib


def resolve(cache: pathlib.Path) -> pathlib.Path:
    snap_root = cache / "models--org--name" / "snapshots"
    return next(iter(sorted(snap_root.iterdir())))
''',
        unpinned=True,
        kills="scanning C++ only; also a receiver-side subject",
    ),
    Fixture(
        name="python_indirection_defined_after_use",
        rel="tools/bench/late_helpers.py",
        body='''import pathlib


def resolve(cache: pathlib.Path) -> pathlib.Path:
    return sorted(snap_root(cache).iterdir())[0]


def snap_root(cache: pathlib.Path) -> pathlib.Path:
    return _hub(cache)


def _hub(cache: pathlib.Path) -> pathlib.Path:
    return cache / "snapshots"
''',
        unpinned=True,
        kills="marking in ONE pass — the chain is only closed on the second",
    ),
    Fixture(
        name="models_dash_marker_only",
        rel="tests/parity/test_models_dash.cpp",
        body="""#include <filesystem>
namespace fs = std::filesystem;
std::string Resolve(const fs::path& hub) {
  const fs::path repo = hub / "models--unsloth--Qwen3.6-27B-NVFP4";
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(repo, ec))
    return e.path().string();
  return "";
}
""",
        unpinned=True,
        kills="requiring the literal word `snapshots`",
    ),
    # ---- NEGATIVES. These are what a WIDENING breaks. ----
    Fixture(
        name="pinned_resolver",
        rel="tests/parity/test_pinned.cpp",
        body=PINNED_FIXTURE,
        unpinned=False,
        kills="",
    ),
    Fixture(
        name="shard_iteration_in_a_resolved_dir",
        rel="tests/parity/test_shards.cpp",
        body="""#include <filesystem>
namespace fs = std::filesystem;
// resolved via parity::HfSnapshot, then enumerate shards under snapshots/<rev>:
void Shards(const std::string& dir) {
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(dir, ec)) (void)e;
}
""",
        unpinned=False,
        kills="",
    ),
    Fixture(
        name="commented_out_resolver",
        rel="tests/parity/test_comment.cpp",
        body="""#include <filesystem>
namespace fs = std::filesystem;
// const fs::path snaps = fs::path("/x") / "snapshots";
// for (const auto& e : fs::directory_iterator(snaps, ec)) {}
void Nothing() {}
""",
        unpinned=False,
        kills="",
    ),
    Fixture(
        name="marker_in_prose_string",
        rel="tests/vt/test_cache_prose.cpp",
        body="""#include <filesystem>
namespace fs = std::filesystem;
TEST_CASE("ready-map import snapshots deterministically") {
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(fs::path("/out"), ec)) (void)e;
}
""",
        unpinned=False,
        kills="",
    ),
    Fixture(
        name="python_shards_inside_a_pinned_snapshot",
        rel="tools/bench/pinned_shards.py",
        body='''import pathlib

SNAPSHOT = pathlib.Path.home() / "models--org--name/snapshots/890bdef7"


def shards(snapshot: pathlib.Path):
    return sorted(snapshot.glob("*.safetensors"))
''',
        unpinned=False,
        kills="",
    ),
    Fixture(
        name="out_of_scope_src_tree",
        rel="src/vllm/entrypoints/resolve.cpp",
        body=UNPINNED_FIXTURE,
        unpinned=False,
        kills="",
    ),
)


def self_test() -> int:
    """Sweep FIXTURES in both directions.

    The RED-before proof, generalised. A single fixture proves only that the
    checker matches THAT fixture, which is how the first version of this file
    came to claim more than it enforced. Every positive here is an idiom a
    plausible narrowing would drop, and every negative is code a plausible
    widening would falsely flag.
    """
    failures = 0
    for fixture in FIXTURES:
        with tempfile.TemporaryDirectory() as raw:
            scratch = pathlib.Path(raw)
            target = scratch / fixture.rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(fixture.body, encoding="utf-8")
            reported = [
                p for p in check(scratch) if "UNPINNED checkpoint resolution" in p
            ]
        ok = bool(reported) == fixture.unpinned
        verdict = "DETECTED" if reported else "clean"
        expected = "unpinned" if fixture.unpinned else "must stay clean"
        print(
            f"self-test {'ok  ' if ok else 'FAIL'} "
            f"{fixture.name:<38} {expected:<16} -> {verdict}"
        )
        if not ok:
            failures += 1

    # A ledger line must actually excuse the file it names, and only that one.
    with tempfile.TemporaryDirectory() as raw:
        scratch = pathlib.Path(raw)
        target = scratch / "tests/parity/test_synthetic_gate.cpp"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(UNPINNED_FIXTURE, encoding="utf-8")
        LEDGER["tests/parity/test_synthetic_gate.cpp"] = "self-test (#471)"
        try:
            excused = [p for p in check(scratch) if "UNPINNED" in p]
        finally:
            del LEDGER["tests/parity/test_synthetic_gate.cpp"]
    print(f"self-test {'ok  ' if not excused else 'FAIL'} {'ledger excuses its own file':<38}")
    if excused:
        failures += 1

    # The STALE arm: an empty tree must not leave the ledger standing.
    with tempfile.TemporaryDirectory() as raw:
        scratch = pathlib.Path(raw)
        (scratch / "tests").mkdir()
        stale = [p for p in check(scratch) if "STALE ledger entry" in p]
    print(f"self-test {'ok  ' if stale else 'FAIL'} {'stale ledger line is refused':<38}")
    if not stale:
        failures += 1

    shape = check_ledger_shape()
    print(f"self-test {'ok  ' if not shape else 'FAIL'} {'every ledger line names an issue':<38}")
    if shape:
        failures += 1
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="sweep the fixture corpus in both directions",
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
