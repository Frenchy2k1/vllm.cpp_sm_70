#!/usr/bin/env python3
"""Enforce purpose-specific public-document projections for each change."""

from __future__ import annotations

import argparse
import re
import string
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.policy_contract import PolicyRule, load_policy


CHECKER_PATH = "scripts/check-doc-checkpoint.py"


@dataclass(frozen=True)
class PublicRuleContract:
    surface: str
    change_class: str
    action: str
    enforcement: tuple[str, ...]


@dataclass(frozen=True)
class PublicRuleBinding:
    rule_id: str
    surface: str
    change_class: str
    action: str
    enforcement: tuple[str, ...]


# Public rules are a closed machine interface. Their procedures carry the
# explanatory detail; these cells contain no prose for a checker to interpret.
PUBLIC_RULE_CONTRACTS = {
    "POL-DOC-STATUS": PublicRuleContract(
        "docs/STATUS.md",
        "feature_checkpoint",
        "Update",
        (CHECKER_PATH, "scripts/check-policy.py"),
    ),
    "POL-DOC-BENCHMARKS": PublicRuleContract(
        "docs/BENCHMARKS.md",
        "feature_checkpoint",
        "Update",
        (
            CHECKER_PATH,
            "scripts/check-public-doc-tables.py",
            "scripts/check-policy.py",
        ),
    ),
    "POL-DOC-FEATURES": PublicRuleContract(
        "docs/FEATURES.md",
        "feature_surface",
        "Update",
        (
            CHECKER_PATH,
            "scripts/check-public-doc-tables.py",
            "scripts/check-policy.py",
        ),
    ),
    "POL-DOC-USAGE": PublicRuleContract(
        "docs/USAGE.md",
        "user_usage",
        "Update",
        (CHECKER_PATH, "scripts/check-policy.py"),
    ),
    "POL-DOC-README": PublicRuleContract(
        "README.md",
        "landing_page",
        "Update",
        (
            CHECKER_PATH,
            "scripts/check-readme-structure.py",
            "scripts/check-policy.py",
        ),
    ),
    "POL-NOW-COUPLING": PublicRuleContract(
        ".agents/NOW.md",
        "live_state",
        "Refresh",
        (
            CHECKER_PATH,
            "scripts/check-now-current.py",
            "scripts/check-policy.py",
        ),
    ),
}
PUBLIC_RULE_IDS = tuple(PUBLIC_RULE_CONTRACTS)
PUBLIC_CHANGE_CLASSES = frozenset(
    contract.change_class for contract in PUBLIC_RULE_CONTRACTS.values()
)
PUBLIC_SURFACES = frozenset(
    contract.surface for contract in PUBLIC_RULE_CONTRACTS.values()
)
POSITIVE_ACTIONS = frozenset({"Update", "Refresh"})
REPOSITORY_PATH_CHARACTERS = frozenset(
    string.ascii_letters + string.digits + "._-"
)
REPOSITORY_PATH_LEXICAL_CHARACTERS = REPOSITORY_PATH_CHARACTERS | frozenset("/")


def parse_requirement(requirement: str) -> tuple[str, str]:
    """Parse one positive action and repository-relative target exactly.

    The final period is requirement syntax, not part of the path. Validate it
    separately so a greedy target cannot consume an extra period and evade a
    target-ownership comparison.
    """

    if not requirement.endswith("."):
        raise ValueError("must end with exactly one terminal period")

    body = requirement[:-1]
    fields = body.split(" ")
    if len(fields) != 2 or any(not field for field in fields):
        raise ValueError("must contain one action and one path token")
    action, target = fields
    if action not in POSITIVE_ACTIONS:
        raise ValueError("action must be Update or Refresh")

    if target.startswith("/"):
        raise ValueError("target must be repository-relative")
    segments = target.split("/")
    if any(not segment for segment in segments):
        raise ValueError("target path segments must be nonempty")
    for segment in segments:
        if segment in {".", ".."}:
            raise ValueError("target must not contain dot path segments")
        if segment.endswith("."):
            raise ValueError("target path segments must not end with a period")
        if any(
            character not in REPOSITORY_PATH_CHARACTERS for character in segment
        ):
            raise ValueError("target contains a non-portable repository-path character")
    return action, target


def _uses_repository_target_form(requirement: str) -> bool:
    """Distinguish controlled path requirements from ordinary policy prose."""

    for action in POSITIVE_ACTIONS:
        prefix = f"{action} "
        if not requirement.startswith(prefix):
            continue
        first_field = requirement[len(prefix) :].split(" ", 1)[0]
        return any(marker in first_field for marker in ("/", "\\", "."))
    return False


def reserved_public_targets(requirement: str) -> set[str]:
    """Find exact public-surface tokens without interpreting the sentence.

    Ownership is independent of the positive requirement grammar. A non-owner
    cannot hide a reserved target behind another verb, an inserted word, or
    malformed whitespace. Boundaries use the repository-path alphabet plus
    ``/`` so path prefixes such as ``docs/STATUS.md.extra`` and nested paths do
    not accidentally claim the exact ``docs/STATUS.md`` token.
    """

    found: set[str] = set()
    for surface in PUBLIC_SURFACES:
        start = 0
        while True:
            index = requirement.find(surface, start)
            if index == -1:
                break
            end = index + len(surface)
            left_is_path = (
                index > 0
                and requirement[index - 1] in REPOSITORY_PATH_LEXICAL_CHARACTERS
            )
            right_is_path = False
            if end < len(requirement):
                right = requirement[end]
                if right != ".":
                    right_is_path = right in REPOSITORY_PATH_LEXICAL_CHARACTERS
                else:
                    # A run of sentence periods delimits the exact token. A
                    # following path character makes it a longer path instead
                    # (for example docs/STATUS.md.extra).
                    cursor = end
                    while cursor < len(requirement) and requirement[cursor] == ".":
                        cursor += 1
                    right_is_path = (
                        cursor < len(requirement)
                        and requirement[cursor]
                        in REPOSITORY_PATH_LEXICAL_CHARACTERS
                    )
            if not left_is_path and not right_is_path:
                found.add(surface)
            start = index + 1
    return found


def public_namespace_errors(rules: dict[str, PolicyRule]) -> list[str]:
    """Reject any rule that trespasses on the closed public-rule namespace."""

    trigger_owners: dict[str, set[str]] = {}
    scope_owners: dict[str, str] = {}
    target_owners: dict[str, str] = {}
    semantic_owners: dict[tuple[str, str, str, str], str] = {}
    for rule_id, contract in PUBLIC_RULE_CONTRACTS.items():
        trigger_owners.setdefault(contract.change_class, set()).add(rule_id)
        scope_owners[contract.surface] = rule_id
        target_owners[contract.surface] = rule_id
        semantic_owners[
            (contract.surface, contract.change_class, contract.action, contract.surface)
        ] = rule_id

    errors: list[str] = []
    for rule_id, rule in sorted(rules.items()):
        if rule_id.startswith("POL-DOC-") and rule_id not in PUBLIC_RULE_CONTRACTS:
            errors.append(
                f"{rule_id}: reserved POL-DOC-* namespace belongs to the public "
                "document contract"
            )

        allowed_trigger_owners = trigger_owners.get(rule.trigger)
        if allowed_trigger_owners is not None and rule_id not in allowed_trigger_owners:
            errors.append(
                f"{rule_id}: trigger {rule.trigger!r} is reserved for "
                + ", ".join(sorted(allowed_trigger_owners))
            )

        scope_owner = scope_owners.get(rule.scope)
        if scope_owner is not None and rule_id != scope_owner:
            errors.append(
                f"{rule_id}: scope {rule.scope!r} is reserved for {scope_owner}"
            )

        claimed_targets = reserved_public_targets(rule.requirement)
        for target in sorted(claimed_targets):
            target_owner = target_owners[target]
            if rule_id != target_owner:
                errors.append(
                    f"{rule_id}: requirement target {target!r} is reserved for "
                    f"{target_owner}"
                )

        try:
            action, target = parse_requirement(rule.requirement)
        except ValueError as exc:
            if _uses_repository_target_form(rule.requirement):
                errors.append(f"{rule_id}: requirement is invalid: {exc}")
            continue
        target_owner = target_owners.get(target)
        if (
            target_owner is not None
            and rule_id != target_owner
            and target not in claimed_targets
        ):
            errors.append(
                f"{rule_id}: requirement target {target!r} is reserved for "
                f"{target_owner}"
            )

        semantic_owner = semantic_owners.get(
            (rule.scope, rule.trigger, action, target)
        )
        if semantic_owner is not None and rule_id != semantic_owner:
            errors.append(
                f"{rule_id}: duplicate public semantic binding owned by "
                f"{semantic_owner}"
            )
    return errors


def parse_public_rule(rule: PolicyRule) -> PublicRuleBinding:
    """Parse one public rule completely or reject it as non-canonical."""

    contract = PUBLIC_RULE_CONTRACTS.get(rule.rule_id)
    if contract is None:
        raise ValueError(f"{rule.rule_id}: unknown public rule")
    if rule.scope != contract.surface:
        raise ValueError(
            f"{rule.rule_id}: scope {rule.scope!r} must be {contract.surface!r}"
        )
    if rule.trigger != contract.change_class:
        raise ValueError(
            f"{rule.rule_id}: trigger {rule.trigger!r} must be "
            f"{contract.change_class!r}"
        )

    try:
        action, target = parse_requirement(rule.requirement)
    except ValueError as exc:
        raise ValueError(
            f"{rule.rule_id}: requirement must fully match "
            f"{contract.action} {contract.surface}."
        ) from exc
    if action != contract.action or target != contract.surface:
        raise ValueError(
            f"{rule.rule_id}: requirement must be exactly "
            f"{contract.action} {contract.surface}."
        )

    enforcement = tuple(item.strip() for item in rule.enforcement.split(";"))
    if any(not item for item in enforcement) or enforcement != contract.enforcement:
        raise ValueError(
            f"{rule.rule_id}: enforcement must be exactly "
            + "; ".join(contract.enforcement)
        )
    return PublicRuleBinding(
        rule_id=rule.rule_id,
        surface=contract.surface,
        change_class=contract.change_class,
        action=contract.action,
        enforcement=enforcement,
    )


def public_document_rules(root: Path = ROOT) -> dict[str, PolicyRule]:
    """Load the exact public-projection rules from the policy authority."""

    rules = load_policy(root)
    namespace_errors = public_namespace_errors(rules)
    if namespace_errors:
        raise ValueError("\n".join(namespace_errors))
    missing = [rule_id for rule_id in PUBLIC_RULE_IDS if rule_id not in rules]
    if missing:
        raise ValueError("policy.csv is missing public rules: " + ", ".join(missing))
    return {rule_id: rules[rule_id] for rule_id in PUBLIC_RULE_IDS}


def public_rule_bindings(root: Path = ROOT) -> dict[str, PublicRuleBinding]:
    """Return parsed public-rule bindings keyed by stable rule ID."""

    return {
        rule_id: parse_public_rule(rule)
        for rule_id, rule in public_document_rules(root).items()
    }


# Governance paths are exact. Unlisted scripts and tests remain checkpoints,
# including technical runtime consistency checkers with similar names.
GOVERNANCE_FILES = frozenset(
    {
        ".agents/prompts/implementer.md",
        ".agents/prompts/operator.md",
        ".agents/prompts/reviewer.md",
        ".agents/governance-tasks.csv",
        ".agents/policy.csv",
        ".agents/waivers.csv",
        ".agents/state.md",
        ".agents/state.csv",
        ".agents/completed/state-migration-manifest.csv",
        "scripts/agent-role.py",
        "scripts/check-doc-checkpoint.py",
        "scripts/check-commit-trailers.py",
        "scripts/check-gate-commands.py",
        "scripts/check-now-current.py",
        "scripts/policy_contract.py",
        "scripts/check-policy.py",
        "scripts/check-prompt-contract.py",
        "scripts/check-pr-size.py",
        "scripts/check-protocol-consistency.py",
        "scripts/check-public-doc-tables.py",
        "scripts/check-state-record.py",
        "scripts/check-role-discipline.py",
        "scripts/claim-view.py",
        "scripts/ready-for-helper.py",
        "scripts/state_record.py",
        "scripts/migrate-state-record.py",
        "tests/scripts/test_agent_role.py",
        "tests/scripts/test_claim_view.py",
        "tests/scripts/test_ready_for_helper.py",
        "tests/scripts/test_policy_contract.py",
        "tests/scripts/test_check_prompt_contract.py",
        "tests/scripts/test_doc_checkpoint.py",
        "tests/scripts/test_check_commit_trailers.py",
        "tests/scripts/test_check_now_current.py",
        "tests/scripts/test_policy_waivers.py",
        "tests/scripts/test_check_pr_size.py",
        "tests/scripts/test_check_protocol_consistency.py",
        "tests/scripts/test_check_public_doc_tables.py",
        "tests/scripts/test_check_state_record.py",
        "tests/scripts/test_state_record_core.py",
        "tests/scripts/test_migrate_state_record.py",
        "tests/scripts/test_state_record_cutover.py",
        "docs/superpowers/specs/2026-08-07-internal-policy-optimization-design.md",
        ".agents/completed/pre-cutover-claim-protocol.md",
    }
)
STRUCTURED_STATE_PATH = re.compile(
    r"\.agents/(?:state-index/\d{4}-\d{2}-\d{3}\.csv|"
    r"state-events/\d{4}-\d{2}/STATE-[A-Za-z0-9-]+\.md)\Z"
)

# coordination.md normally moves live feature state and therefore owes NOW plus
# the checkpoint projections.  This one closed migration removes only its
# obsolete generated-claim snapshot.  Scope the exception to the complete,
# exact cutover transaction so a later coordination edit -- alone or alongside
# feature work -- cannot inherit a blanket governance bypass.
CLAIM_CUTOVER_FILES = frozenset(
    {
        "scripts/claim-view.py",
        "scripts/ready-for-helper.py",
        "tests/scripts/test_claim_view.py",
        "tests/scripts/test_ready_for_helper.py",
        ".agents/coordination.md",
        ".agents/completed/pre-cutover-claim-protocol.md",
        "scripts/check-doc-checkpoint.py",
        "tests/scripts/test_doc_checkpoint.py",
    }
)
# Policy consolidation repairs links in records that ordinarily signal feature
# work.  Exempt only the complete, closed migration transaction: removing any
# path or adding any unrelated path restores the ordinary semantic classes.
POLICY_CONSOLIDATION_FILES = frozenset(
    {
        ".agents/NOW.md",
        ".agents/backend-matrix.md",
        ".agents/backends.md",
        ".agents/benchmark-record.md",
        ".agents/completed/ai-coding-assistants-legacy.md",
        ".agents/completed/benchmark-protocol-legacy.md",
        ".agents/completed/mvp-gates-legacy.md",
        ".agents/completed/operator-helper-protocol-legacy.md",
        ".agents/completed/policy-directives-legacy.md",
        ".agents/completed/porting-discipline-legacy.md",
        ".agents/completed/roadmap_mvp_v0.md",
        ".agents/completed/roadmap_v1_inventory_spikes_2026-07-10.md",
        ".agents/completed/test-porting-legacy.md",
        ".agents/parity-ledger.md",
        ".agents/policy.csv",
        ".agents/porting.md",
        ".agents/roadmap_v1.md",
        ".agents/specs/backend-fanout-metal-vulkan-xpu.md",
        ".agents/specs/cuda-sglang-low-concurrency.md",
        ".agents/specs/gemma4-multimodal.md",
        ".agents/specs/glm-dsa-latest-deepseek.md",
        ".agents/specs/kv-persistence-lmcache.md",
        ".agents/specs/lmcache-cpp-client-connector.md",
        ".agents/specs/mla-deepseek-campaign.md",
        ".agents/specs/model-factory-registry.md",
        ".agents/specs/multimodal-track.md",
        ".agents/specs/prefix-prompt-caching-parity.md",
        ".agents/specs/quantization-coverage.md",
        ".agents/specs/session-onboarding.md",
        ".agents/specs/sglang-parity-oracle.md",
        ".agents/specs/sweep-gemma.md",
        ".agents/specs/sweep-olmo2.md",
        ".agents/specs/sweep-recent-dense-batch.md",
        ".agents/specs/vulkan-full-support.md",
        ".agents/state.md",
        ".agents/verification.md",
        ".agents/workflow.md",
        "AGENTS.md",
        "docs/ROCM.md",
        "docs/STATUS.md",
        "scripts/check-doc-checkpoint.py",
        "scripts/policy_contract.py",
        "tests/scripts/test_doc_checkpoint.py",
        "tests/scripts/test_policy_contract.py",
    }
)

# Final policy cutover wires policy-only local/remote/integration enforcement.
# CI and scripts normally signal product work, so exempt only this complete,
# closed transaction; any missing or extra path restores ordinary classes.
POLICY_CUTOVER_FILES = frozenset(
    {
        ".agents/policy-cutover",
        ".agents/workflow.md",
        ".github/workflows/ci.yml",
        ".githooks/pre-push",
        "scripts/agent-integration.py",
        "scripts/agent-preflight.sh",
        "scripts/agent-ready.py",
        "scripts/check-doc-checkpoint.py",
        "scripts/check-protocol-consistency.py",
        "tests/scripts/test_agent_gates.py",
        "tests/scripts/test_check_protocol_consistency.py",
        "tests/scripts/test_doc_checkpoint.py",
    }
)
# The cutover PR must prove its own finite review budget against a base where
# the size checker did not exist, and must distinguish a detached PR head from
# landed main history. This exact transaction supplies closed creation
# mutations, one expiring PR-scoped migration waiver, and the exact PR-head
# signal. CI normally signals feature work, so any missing or additional path
# restores the usual product-document obligations.
PR_SIZE_BOOTSTRAP_FILES = frozenset(
    {
        ".agents/waivers.csv",
        ".github/workflows/ci.yml",
        "docs/superpowers/specs/2026-08-07-internal-policy-optimization-design.md",
        "scripts/check-doc-checkpoint.py",
        "scripts/check-pr-size.py",
        "scripts/check-role-discipline.py",
        "tests/scripts/test_check_pr_size.py",
        "tests/scripts/test_doc_checkpoint.py",
    }
)
# Follow-up found only in a detached clean checkout: role discipline must receive
# the exact event head so it reports unmerged PR commits instead of treating
# detached HEAD as landed main. Keep this corrective transaction closed too.
PENDING_PR_RANGE_FILES = frozenset(
    {
        ".github/workflows/ci.yml",
        "docs/superpowers/specs/2026-08-07-internal-policy-optimization-design.md",
        "scripts/check-doc-checkpoint.py",
        "scripts/check-role-discipline.py",
        "tests/scripts/test_agent_role.py",
        "tests/scripts/test_check_pr_size.py",
        "tests/scripts/test_doc_checkpoint.py",
    }
)
# Actions checks out a synthetic PR merge, which is not contributor history.
# These closed follow-ups make both in-job role checks and trailer validation
# walk the exact event base..head range instead of detached HEAD.
SYNTHETIC_MERGE_RANGE_FILES = frozenset(
    {
        ".github/workflows/ci.yml",
        "docs/superpowers/specs/2026-08-07-internal-policy-optimization-design.md",
        "scripts/check-doc-checkpoint.py",
        "tests/scripts/test_agent_gates.py",
        "tests/scripts/test_doc_checkpoint.py",
    }
)
FEATURE_CHECKPOINT_FILES = frozenset(
    {
        "CMakeLists.txt",
        ".agents/backend-matrix.md",
        ".agents/coordination.md",
        ".agents/engine-matrix.md",
        ".agents/feature-matrix.md",
        ".agents/kernel-matrix.md",
        ".agents/model-matrix.md",
        ".agents/parity-ledger.md",
        ".agents/porting-inventory.md",
        ".agents/quantization-matrix.md",
        ".agents/roadmap_v1.md",
    }
)
FEATURE_CHECKPOINT_PREFIXES = (
    ".github/workflows/",
    ".agents/specs/",
    "cmake/",
    "examples/",
    "include/",
    "scripts/",
    "src/",
    "tests/",
    "tools/",
)
FEATURE_SURFACE_FILES = frozenset(
    {
        ".agents/backend-matrix.md",
        ".agents/feature-matrix.md",
        ".agents/model-matrix.md",
        ".agents/quantization-matrix.md",
    }
)
FEATURE_SURFACE_PREFIXES = ("src/vllm/model_executor/models/",)

# These are exact user-facing configuration/build/install entrypoints. Do not
# broaden this to all cmake files: toolchain internals are checkpoints but do
# not necessarily alter installation instructions.
USER_USAGE_FILES = frozenset(
    {
        ".env.example",
        "CMakeLists.txt",
        "cmake/install.cmake",
        "examples/CMakeLists.txt",
        "examples/cli/main.cpp",
        "examples/server/main.cpp",
        "include/vllm.h",
    }
)
USER_USAGE_PREFIXES = (
    "include/vllm/",
    "src/vllm/entrypoints/",
    "examples/cli/",
    "examples/server/",
)
LIVE_STATE_FILES = frozenset({".agents/coordination.md"})

# README permission and obligation come only from underlying landing sources.
# Co-edited public projections can never justify README churn.
LANDING_SOURCE_FILES = frozenset(
    {
        ".agents/mission.md",
        "CMakeLists.txt",
        "benchmarks/demo/footprint_gb10.json",
        "benchmarks/demo/qwen36_27b_c1_c32.json",
        "examples/cli/main.cpp",
        "examples/server/main.cpp",
        # The Vulkan-vs-llama.cpp headline, added 2026-08-09. README carried two
        # FALSE Vulkan claims ("24 native ops", "llama.cpp Vulkan stays 2.62x
        # ahead") that no permitted change could correct, because the backend it
        # describes had no headline-benchmark source in this list while the CUDA
        # comparison had two.
        #
        # Adding docs/STATUS.md instead was considered and REJECTED: the rule
        # above -- co-edited public projections never justify README churn -- is
        # deliberate and directly tested
        # (test_readme_is_not_justified_by_coedited_public_projections names
        # docs/STATUS.md), and overturning it to unblock a README edit would be
        # weakening a checker to make a change pass. The gap was not that the rule
        # was wrong; it was that a real headline benchmark had no source file.
        "benchmarks/demo/vulkan_27b_llamacpp.json",
    }
)


def classify_changed_paths(paths: list[str]) -> set[str]:
    """Classify changed paths by semantic obligation, not mutable directory."""

    classes: set[str] = set()
    path_set = set(paths)
    if path_set == POLICY_CONSOLIDATION_FILES:
        return {"governance"}
    if path_set == POLICY_CUTOVER_FILES:
        return {"governance"}
    if path_set == PR_SIZE_BOOTSTRAP_FILES:
        return {"governance"}
    if path_set == PENDING_PR_RANGE_FILES:
        return {"governance"}
    if path_set == SYNTHETIC_MERGE_RANGE_FILES:
        return {"governance"}
    pr_size_bootstrap_members = path_set & PR_SIZE_BOOTSTRAP_FILES
    incomplete_pr_size_bootstrap = (
        ".agents/waivers.csv" in pr_size_bootstrap_members
        and len(pr_size_bootstrap_members) > 1
    )
    if incomplete_pr_size_bootstrap:
        classes.add("feature_checkpoint")
    pending_pr_range_members = path_set & PENDING_PR_RANGE_FILES
    incomplete_pending_pr_range = (
        "docs/superpowers/specs/2026-08-07-internal-policy-optimization-design.md"
        in pending_pr_range_members
        and len(pending_pr_range_members) > 1
    )
    if incomplete_pending_pr_range:
        classes.add("feature_checkpoint")
    synthetic_merge_range_members = path_set & SYNTHETIC_MERGE_RANGE_FILES
    incomplete_synthetic_merge_range = (
        "docs/superpowers/specs/2026-08-07-internal-policy-optimization-design.md"
        in synthetic_merge_range_members
        and len(synthetic_merge_range_members) > 1
    )
    if incomplete_synthetic_merge_range:
        classes.add("feature_checkpoint")
    exact_claim_cutover = (
        path_set == CLAIM_CUTOVER_FILES
    )
    for path in sorted(path_set):
        if path in PUBLIC_SURFACES:
            continue
        if exact_claim_cutover and path == ".agents/coordination.md":
            classes.add("governance")
            continue
        if path in GOVERNANCE_FILES or STRUCTURED_STATE_PATH.fullmatch(path):
            classes.add("governance")
            continue
        if path in LIVE_STATE_FILES:
            classes.update({"feature_checkpoint", "live_state"})
        if path in FEATURE_SURFACE_FILES or path.startswith(FEATURE_SURFACE_PREFIXES):
            classes.update({"feature_checkpoint", "feature_surface"})
        if path in USER_USAGE_FILES or path.startswith(USER_USAGE_PREFIXES):
            classes.update({"feature_checkpoint", "user_usage"})
        if path in LANDING_SOURCE_FILES:
            classes.add("landing_page")
        if path in FEATURE_CHECKPOINT_FILES or path.startswith(
            FEATURE_CHECKPOINT_PREFIXES
        ):
            classes.add("feature_checkpoint")

    if "README.md" in path_set:
        classes.add("readme_changed")
    return classes


def required_public_surfaces(change_classes: set[str]) -> set[str]:
    """Project semantic classes to surfaces through fully parsed policy rules."""

    return {
        binding.surface
        for binding in public_rule_bindings().values()
        if binding.change_class in change_classes
    }


def _preview(paths: set[str]) -> str:
    ordered = sorted(paths)
    preview = ", ".join(ordered[:5])
    if len(ordered) > 5:
        preview += f", ... (+{len(ordered) - 5})"
    return preview


def checkpoint_errors(paths: set[str]) -> list[str]:
    """Return missing or unjustified public-projection errors for one change."""

    classes = classify_changed_paths(sorted(paths))
    required = required_public_surfaces(classes)
    bindings = public_rule_bindings()
    errors: list[str] = []
    for rule_id in PUBLIC_RULE_IDS:
        binding = bindings[rule_id]
        if binding.surface in required and binding.surface not in paths:
            errors.append(
                f"{binding.change_class} change ({_preview(paths)}) requires "
                f"{binding.surface} in the same change under {rule_id}"
            )

    if "readme_changed" in classes and "landing_page" not in classes:
        errors.append(
            "README.md changed without a landing-page trigger; change it only "
            "with an underlying quickstart, install, positioning, or headline "
            "benchmark source"
        )
    return errors


# Compatibility for existing callers.
def is_checkpoint_path(path: str) -> bool:
    return "feature_checkpoint" in classify_changed_paths([path])


def is_feature_path(path: str) -> bool:
    return "feature_surface" in classify_changed_paths([path])


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def commit_paths(commit: str) -> set[str]:
    parents = git("rev-list", "--parents", "-n", "1", commit).split()[1:]
    if parents:
        output = git("diff", "--name-only", parents[0], commit)
    else:
        output = git(
            "diff-tree", "--root", "--no-commit-id", "--name-only", "-r", commit
        )
    return {line for line in output.splitlines() if line}


def commits_in_range(base: str, head: str) -> list[str]:
    try:
        git("cat-file", "-e", f"{base}^{{commit}}")
    except subprocess.CalledProcessError:
        return [head]
    output = git("rev-list", "--reverse", f"{base}..{head}")
    return [line for line in output.splitlines() if line]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--commit", default=None, help="check one commit")
    source.add_argument(
        "--staged", action="store_true", help="check the current staged change"
    )
    source.add_argument("--range", dest="revision_range", help="check BASE..HEAD")
    parser.add_argument("--base", help="check every commit after this revision")
    parser.add_argument("--head", help="range endpoint (requires --base)")
    args = parser.parse_args()
    if (args.base is None) != (args.head is None):
        parser.error("--base and --head must be supplied together")
    if args.base is not None and (
        args.commit is not None or args.staged or args.revision_range is not None
    ):
        parser.error("a revision range cannot be combined with --commit/--staged")
    if args.revision_range is not None:
        parts = args.revision_range.split("..")
        if len(parts) != 2 or not all(parts):
            parser.error("--range must be BASE..HEAD")
        args.base, args.head = parts
    return args


def main() -> int:
    args = parse_args()
    failures: list[str] = []
    if args.staged:
        paths = set(git("diff", "--cached", "--name-only").splitlines())
        failures.extend(f"staged change: {error}" for error in checkpoint_errors(paths))
    else:
        commits = (
            commits_in_range(args.base, args.head)
            if args.base is not None
            else [args.commit or "HEAD"]
        )
        for commit in commits:
            short = git("rev-parse", "--short", commit)
            failures.extend(
                f"commit {short}: {error}"
                for error in checkpoint_errors(commit_paths(commit))
            )
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    print("OK: public documentation matches the change's semantic policy triggers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
