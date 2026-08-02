#!/usr/bin/env python3
"""Fail if a registered model's decode goes off the production runner/decode seam.

The THIRD "MUST route through, not re-implement" seam (alongside the glue
`vt::FusedChain` catalog and the merged-GEMM `MlpGateUpMethodBase` family, both
policed by scripts/check-fusion-consistency.py): the **decode/runtime path**. A new
model's decode MUST enter the production runner (`ModelRegistry::Forward`, the
registry factory `.forward` hook) and return device-resident logits
(`ForwardLogits.on_device()==true`) on the default `gather_logits` path — handed
straight to the on-GPU sampler — NOT a private generate/argmax loop with a host
logit download. A model born off the runner inherits none of the parity-enablers
(paged bf16-KV attention, on-GPU sampling, shared MoE builders, device RoPE,
decode-graph capture) and forces per-model rediscovery — the "Laguna anti-pattern"
(AGENTS.md, the decode/runtime seam added in `c5c872e6`).

Two invariants, checked per `REGISTER_VLLM_MODEL` (`.forward = &Fn`):

(a) **ON-DEVICE LOGITS** — the model's DEFAULT (`gather_logits`) production forward
    returns a device-resident `ForwardLogits` (reaches the device-logits seam
    `WrapDeviceLogits` / `ViewDeviceLogits` / a `device_storage`/`device_tensor`
    assignment). A registered model whose production forward instead returns
    `HostLogits` / a host `logits.Download` off the on-GPU sampler is DRIFT.

(b) **RUNNER-ROUTED (no private host generate loop)** — the same model must not
    ALSO ship a private `*GenerateCore` host greedy-argmax decode loop as its real
    decode path (the Qwen3-VL escape: `VLGenerateCore` + host `ArgMax`). This is an
    ENRICHMENT of (a): it is only reported for a model already classified HOST, so a
    CLEAN device-resident model that merely keeps a `*GenerateCore` example helper
    around (e.g. qwen3_5's `VLGenerateCoreGdn`) is NOT flagged — the runner path,
    not the presence of a helper, is what makes a model "born on the runner".

Like check-fusion-consistency.py this is a coarse FLOOR, not a per-site proof, and it
resolves through the codebase's real seams so the false-positive/negative rate stays
low:
  * the registered `.forward` hook usually DELEGATES on its `gather_logits` branch to
    `SomeModel::ForwardDevice(...)`; the classification follows that call into the
    ForwardDevice IMPL body (so laguna/qwen3_vl, whose ForwardDevice is a HOST stub
    returning `HostLogits`/`out.host`, are caught even though their registry hook looks
    identical to a clean model's);
  * and ONE further hop into a file-local `ForwardLogits`-returning helper the impl
    calls, so a model that builds its device carrier in its OWN wrapper rather than
    the shared `WrapDeviceLogits` still reads as DEVICE (deepseek_v4's
    `WrapV4DeviceLogits`). Without the hop such a model matched neither seam and
    classified NONE — not an error state, so it dropped out of the drift check
    entirely and the gate stayed green while silently exempting it;
  * `using LlamaModel = Qwen3DenseModel;` aliases are resolved (llama/mistral/internlm2
    reuse the qwen3 dense device forward);
  * a REFUSE-by-name stub (`KimiK3Model::ForwardDevice` is `VT_CHECK(false)`) decodes
    nothing and is SKIPPED, not flagged;
  * a multimodal model with a device text-decode path plus a host mm-PREFILL path
    (gemma4) is CLEAN — its default decode still reaches ForwardDevice.

A model that legitimately cannot route yet is a CONSCIOUS, reviewable allowlist entry
(scripts/runner-routing-allowlist.txt) with a reason; the 3 known off-framework models
(laguna, deepseek_v4, qwen3_vl) keep the gate GREEN while a NEW off-runner model must be
a deliberate allowlist landing. Removing an entry after the model is routed through the
runner (device-resident logits) is the enforcement gate closing.

The validation logic is pure functions (`classify_body`, `drift_models`) so it is unit-
and mutation-testable (tests/scripts/test_check_runner_routing_consistency.py), mirroring
check-fusion-consistency.py.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "src/vllm/model_executor/models"
INCLUDE_DIR = ROOT / "include/vllm/model_executor/models"
ALLOWLIST = ROOT / "scripts/runner-routing-allowlist.txt"

# --- Seam / signal regexes ----------------------------------------------------

# The DEVICE-resident-logits seam: a ForwardLogits carrying a pool-backed device
# buffer (on_device()==true). WrapDeviceLogits/ViewDeviceLogits are the shared
# helpers every clean model funnels through; the raw field assignments cover a
# model that builds the carrier inline.
_DEVICE_SEAM = re.compile(
    r"\bWrapDeviceLogits\b|\bViewDeviceLogits\b"
    r"|\.device_storage\s*=|\.device_tensor\s*="
)
# The HOST-logits producer: a ForwardLogits with only a host [rows,vocab] buffer
# (on_device()==false) handed off the on-GPU sampler, or a host download/argmax.
_HOST_SEAM = re.compile(
    r"\bHostLogits\s*\(|\bout\.host\b|\.host\s*=\s*std::move|\.Download\s*\("
)
# A REFUSE-by-name stub decodes nothing (VT_CHECK(false)); out of scope.
_REFUSE = re.compile(r"VT_CHECK\(\s*false")

# `.forward = &Fn` — the registry factory decode hook (ModelRegistry::Forward).
_FORWARD_FIELD = re.compile(r"\.forward\s*=\s*&\s*([A-Za-z_]\w*)")
# `using Alias = Target;` — model-class aliases (llama/mistral/internlm2).
_ALIAS = re.compile(r"\busing\s+([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*;")
# `SomeModel::ForwardDevice(...)` (+ tap/mm/step variants) the hook delegates to.
_DELEGATE = re.compile(
    r"\b([A-Za-z_]\w*)::(?:ForwardDevice|ForwardDeviceTap|ForwardDeviceMultiTap"
    r"|ForwardMm|Step)\b"
)
# A private greedy host generate loop (the runner-bypass anti-pattern).
_GENERATE_CORE_CALL = re.compile(r"\b([A-Za-z_]\w*GenerateCore)\s*\(")
_GENERATE_CORE_DEF = re.compile(
    r"\b(?:std::vector<[^;{}]+>|ForwardLogits|void)\s+([A-Za-z_]\w*GenerateCore)\s*\("
)
# A free function the hook calls whose definition file carries the real decode body
# (Qwen3VLForwardStepLastLogits -> qwen3_vl.cpp, which also holds VLGenerateCore).
_FREE_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def strip_comments(text: str) -> str:
    """Drop // line and /* */ block comments so a comment mentioning a seam name
    never flips a classification."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", " ", text)
    return text


def extract_fn_body(text: str, fn: str) -> str | None:
    """Return the brace-matched body `{...}` of a `ForwardLogits <fn>(...)` DEFINITION
    in `text`, or None. Matches the definition (return type ForwardLogits) rather than
    a call site, and brace-balances params then body."""
    for m in re.finditer(r"\b" + re.escape(fn) + r"\s*\(", text):
        head = text[max(0, m.start() - 48):m.start()]
        if "ForwardLogits" not in head:
            continue
        # Balance the parameter list starting at the '(' we matched.
        i = m.end() - 1
        depth = 0
        while i < len(text):
            c = text[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        j = text.find("{", i)
        semi = text.find(";", i)
        if j < 0 or (0 <= semi < j):
            continue  # a forward-declaration, not a definition
        depth = 0
        k = j
        while k < len(text):
            c = text[k]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[j:k + 1]
            k += 1
    return None


def classify_body(body: str | None) -> str:
    """Classify a production-forward body: DEVICE (device-resident logits, clean),
    HOST (host logits off the sampler, drift), REFUSE (VT_CHECK(false) stub, skip),
    or NONE (no recognizable logit producer). DEVICE wins over HOST wins over REFUSE."""
    if body is None:
        return "NONE"
    body = strip_comments(body)
    if _DEVICE_SEAM.search(body):
        return "DEVICE"
    if _HOST_SEAM.search(body):
        return "HOST"
    if _REFUSE.search(body):
        return "REFUSE"
    return "NONE"


def classify_with_helpers(body: str | None, defining_text: str) -> str:
    """`classify_body`, but following ONE level of file-local `ForwardLogits`-returning
    helper that the body CALLS.

    A model may build its device-resident carrier in its own privately-named wrapper
    instead of the shared `WrapDeviceLogits` — deepseek_v4.cpp's `WrapV4DeviceLogits`
    is the case that motivated this. Without the hop, `ForwardDevice` matched NEITHER
    seam and classified NONE, which is not an error state: the model dropped out of
    the HOST-drift check entirely and the gate went green while claiming it had "1
    no-logit-producer". A hole that silently EXEMPTS a model is worse than a red gate,
    so the resolution follows the call the same way the delegate hop already follows
    `Class::ForwardDevice`. Conservative by construction: `extract_fn_body` only
    matches a definition whose return type is `ForwardLogits`, so a helper that
    produces something else is skipped rather than guessed at."""
    direct = classify_body(body)
    if direct == "DEVICE" or body is None:
        return direct
    ranked = {direct}
    for called in _FREE_CALL.findall(strip_comments(body)):
        helper = extract_fn_body(defining_text, called)
        if helper is not None:
            ranked.add(classify_body(helper))
    for level in ("DEVICE", "HOST", "REFUSE"):
        if level in ranked:
            return level
    return "NONE"


def resolve_alias(cls: str, alias: dict[str, str]) -> str:
    seen: set[str] = set()
    while cls in alias and cls not in seen:
        seen.add(cls)
        cls = alias[cls]
    return cls


@dataclass(frozen=True)
class ModelRoute:
    """The decode-routing verdict for one REGISTER_VLLM_MODEL registration."""
    name: str                      # allowlist key (registry stem minus _registry)
    reg_file: str
    forward_fn: str
    classification: str            # DEVICE | HOST | REFUSE | NONE
    private_generate_loop: bool    # invariant (b): ships a *GenerateCore host loop
    device_source: str = ""        # which delegated class supplied the device seam


def build_alias_map(files: list[Path]) -> dict[str, str]:
    alias: dict[str, str] = {}
    for p in files:
        for a, tgt in _ALIAS.findall(strip_comments(read(p))):
            alias[a] = tgt
    return alias


def collect_forwarddevice_bodies(cpp_files: list[Path]) -> dict[str, tuple[str, str]]:
    """Map ModelClass -> (defining file name, ForwardDevice body) for every
    `ForwardLogits Class::ForwardDevice(...) {...}` definition."""
    out: dict[str, tuple[str, str]] = {}
    for p in cpp_files:
        text = read(p)
        for m in re.finditer(
            r"ForwardLogits\s+([A-Za-z_]\w*)::ForwardDevice\s*\(", text
        ):
            cls = m.group(1)
            j = text.find("{", m.end())
            semi = text.find(";", m.end())
            if j < 0 or (0 <= semi < j):
                continue
            depth = 0
            k = j
            while k < len(text):
                c = text[k]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        out.setdefault(cls, (p.name, text[j:k + 1]))
                        break
                k += 1
    return out


def _model_name(reg_stem: str) -> str:
    return reg_stem[:-len("_registry")] if reg_stem.endswith("_registry") else reg_stem


def scan_registrations(
    models_dir: Path, include_dir: Path
) -> dict[str, ModelRoute]:
    """Scan every registered model and classify its decode routing. Returns a map
    name -> ModelRoute."""
    routes: dict[str, ModelRoute] = {}
    if not models_dir.is_dir():
        return routes
    cpp_files = sorted(models_dir.glob("*.cpp"))
    header_files = sorted(models_dir.glob("*.h"))
    if include_dir.is_dir():
        header_files += sorted(include_dir.glob("*.h"))

    alias = build_alias_map(cpp_files + header_files)
    fd_bodies = collect_forwarddevice_bodies(cpp_files)
    # free-function definition -> file (for resolving the hook's decode body file)
    free_fn_file: dict[str, str] = {}
    file_text: dict[str, str] = {}
    for p in cpp_files:
        file_text[p.name] = read(p)
        for m in re.finditer(
            r"\b(?:std::vector<[^;{}]+>|ForwardLogits)\s+([A-Za-z_]\w*)\s*\(", file_text[p.name]
        ):
            free_fn_file.setdefault(m.group(1), p.name)

    for p in cpp_files:
        text = file_text[p.name]
        fm = _FORWARD_FIELD.search(strip_comments(text))
        if not fm:
            continue
        fn = fm.group(1)
        name = _model_name(p.stem)
        body = extract_fn_body(text, fn)
        clean_body = strip_comments(body) if body else ""

        # Files that carry this model's real decode body: the registry hook file,
        # the ForwardDevice impl files of delegated classes, and the def files of
        # free functions the hook calls.
        impl_files: set[str] = {p.name}
        delegated_classes = {
            resolve_alias(c, alias) for c in _DELEGATE.findall(clean_body)
        }
        device_source = ""
        classification = "NONE"
        if classify_with_helpers(body, text) == "DEVICE":
            classification, device_source = "DEVICE", fn
        for cls in delegated_classes:
            impl = fd_bodies.get(cls)
            if impl:
                impl_files.add(impl[0])
                impl_class = classify_with_helpers(impl[1], file_text.get(impl[0], ""))
                if impl_class == "DEVICE" and classification != "DEVICE":
                    classification, device_source = "DEVICE", cls
        if classification != "DEVICE":
            # Not device-reachable: rank the delegated ForwardDevice impls, else the
            # hook body itself, as HOST > REFUSE > NONE.
            impl_classes = [
                classify_with_helpers(fd_bodies[c][1], file_text.get(fd_bodies[c][0], ""))
                for c in delegated_classes
                if c in fd_bodies
            ]
            if "HOST" in impl_classes:
                classification = "HOST"
            elif not impl_classes and _HOST_SEAM.search(clean_body):
                classification = "HOST"
            elif "REFUSE" in impl_classes:
                classification = "REFUSE"
            elif _REFUSE.search(clean_body):
                classification = "REFUSE"

        # Invariant (b): does this model's decode-body file set define a private
        # *GenerateCore host loop? (Only meaningful for a HOST model — a DEVICE model
        # that keeps a GenerateCore example helper is legitimately clean.)
        for called in _FREE_CALL.findall(clean_body):
            f = free_fn_file.get(called)
            if f:
                impl_files.add(f)
        private_loop = any(
            _GENERATE_CORE_DEF.search(strip_comments(file_text.get(f, "")))
            for f in impl_files
        )

        routes[name] = ModelRoute(
            name=name,
            reg_file=p.name,
            forward_fn=fn,
            classification=classification,
            private_generate_loop=private_loop,
            device_source=device_source,
        )
    return routes


def allowlisted_names(text: str) -> set[str]:
    """Model names accepted as known off-framework / deliberately-deferred (one per
    line, # comments ignored) — mirrors check-fusion-consistency.py."""
    names: set[str] = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(line)
    return names


def drift_models(
    scanned: dict[str, ModelRoute], allowlisted: set[str]
) -> list[str]:
    """Model names whose production decode returns HOST logits off the runner's
    on-GPU sampler (invariant a) and are not allowlisted. Empty == the check passes.
    REFUSE stubs and DEVICE-clean models never drift."""
    return sorted(
        name
        for name, route in scanned.items()
        if route.classification == "HOST" and name not in allowlisted
    )


def _load_allowlist(path: Path) -> set[str]:
    return allowlisted_names(read(path)) if path.exists() else set()


def main() -> int:
    scanned = scan_registrations(MODELS_DIR, INCLUDE_DIR)
    allowlisted = _load_allowlist(ALLOWLIST)
    drift = drift_models(scanned, allowlisted)

    n_device = sum(1 for r in scanned.values() if r.classification == "DEVICE")
    n_host = sum(1 for r in scanned.values() if r.classification == "HOST")
    n_refuse = sum(1 for r in scanned.values() if r.classification == "REFUSE")
    n_none = sum(1 for r in scanned.values() if r.classification == "NONE")

    if drift:
        print(
            "ERROR: registered model decode(s) return HOST logits off the production "
            "runner / on-GPU sampler instead of a device-resident ForwardLogits "
            "(on_device()==true) on the default gather_logits path — the "
            "decode/runtime seam (AGENTS.md 'born on the runner') — and are not on "
            "scripts/runner-routing-allowlist.txt:",
            file=sys.stderr,
        )
        for name in drift:
            r = scanned[name]
            extra = (
                " + ships a private *GenerateCore host generate loop off the runner "
                "(invariant b)" if r.private_generate_loop else ""
            )
            print(
                f"  - {name} ({r.reg_file}: {r.forward_fn} returns HostLogits"
                f"{extra})",
                file=sys.stderr,
            )
        print(
            "Route decode through ModelRegistry::Forward returning "
            "ForwardLogits.on_device()==true (WrapDeviceLogits / ViewDeviceLogits on "
            "the gather_logits path; see qwen3_dense.cpp / gemma.cpp / opt.cpp), or "
            "add the model to scripts/runner-routing-allowlist.txt with a reason "
            "(pending framework-routing, see AGENTS.md decode/runtime seam).",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK (runner-routing): {len(scanned)} registered model(s); "
        f"{n_device} return device-resident logits on the runner, "
        f"{n_host} host-logits off-framework ({len(allowlisted)} allowlisted), "
        f"{n_refuse} refuse-by-name stub(s) skipped, {n_none} no-logit-producer."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
