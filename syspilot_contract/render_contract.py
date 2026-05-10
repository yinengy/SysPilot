from __future__ import annotations

from syspilot_contract.compiler import compile_lpcl
from syspilot_contract.schemas import ComposedContract, LibraryContract


def render_contract_lpcl(
    contract: LibraryContract | ComposedContract,
    *,
    include_evidence: bool = False,
) -> str:
    return compile_lpcl(contract, include_evidence=include_evidence)


def _describe_demand(demand: dict[str, object]) -> str:
    dimension = demand.get("dimension")
    if dimension == "workers":
        return f"workers [{demand.get('min')}, {demand.get('max')}]"
    if dimension == "core_set":
        return f"core_set from {demand.get('source')}"
    if dimension == "devices":
        return f"devices [{demand.get('min')}, {demand.get('max')}]"
    if dimension == "partition":
        return f"partition from {demand.get('source')}"
    if dimension == "bytes":
        return f"bytes [{demand.get('min')}, {demand.get('max')}]"
    if dimension == "intensity":
        return f"intensity {demand.get('value')}"
    if dimension == "bandwidth":
        return f"bandwidth [{demand.get('min')}, {demand.get('max')}]"
    if dimension == "depth":
        return f"depth [{demand.get('min')}, {demand.get('max')}]"
    return "unknown"


def render_contract_markdown(contract: LibraryContract | ComposedContract) -> str:
    if isinstance(contract, LibraryContract):
        lines = [
            f"# Library Performance Contract: {contract.contract}",
            "",
            "## Components",
            "",
        ]
        for item in contract.components:
            attrs = ", ".join(f"{key}={value}" for key, value in item.attrs.items()) or "(none)"
            lines.append(f"- `{item.id}` (`{item.kind}`): {attrs}")
        lines.extend(["", "## Shared Resource Uses", ""])
        for item in contract.uses:
            lines.append(
                f"- `{item.id}`: component `{item.by}` uses `{item.resource}` during `{item.phase}` at `{item.scope}` scope; share `{item.share}`; demand `{_describe_demand(item.demand)}`"
            )
        lines.extend(["", "## Assumptions", ""])
        for item in contract.assumptions:
            predicate = item.predicate
            value_suffix = f" value={predicate.get('value')}" if "value" in predicate else ""
            lines.append(
                f"- `{item.id}`: `{predicate.get('kind')}` on `{item.resource}`{value_suffix}"
            )
        lines.extend(["", "## Controls", ""])
        for item in contract.controls:
            effect = item.effect.model_dump(mode="json", exclude_none=True)
            lines.append(
                f"- `{item.name}` (`{item.kind}`, `{item.type}`) targets `{item.target}` with `{effect}` interacts_with={item.interacts_with or '(none)'}`"
            )
        lines.extend(["", "## Signals", ""])
        for item in contract.signals:
            lines.append(
                f"- `{item.id}`: `{item.metric}` targets `{item.target}` via `{item.virtual_probe}` role=`{item.role}` expect=`{item.expected_direction_when_mitigated}` → `{item.indicates}`"
            )
        lines.extend(["", "## Recipes", ""])
        for item in contract.recipes:
            action_text = ", ".join(action.summary for action in item.actions)
            diagnose_text = ", ".join(diagnose.label for diagnose in item.diagnose)
            lines.append(
                f"- `{item.id}` for `{item.hypothesis_kind}`: collect {', '.join(item.collect)}; diagnose: {diagnose_text}; config_checks: {', '.join(item.config_checks)}; outcomes: {', '.join(item.outcomes)}; guardrails: {', '.join(item.guardrails)}; actions: {action_text}"
            )
        return "\n".join(lines).strip() + "\n"

    lines = [
        f"# Composed Performance Contract: {contract.contract}",
        "",
        "## Selected Contracts",
        "",
    ]
    for item in contract.contract_instances:
        lines.append(f"- `{item.instance_id}` → `{item.contract_id}`")
    lines.extend(["", "## Derived Conflict Hypotheses", ""])
    for item in contract.hypotheses:
        lines.append(
            f"- `{item.id}`: `{item.kind}` on `{item.resource}` during `{item.phase}` across {', '.join(item.contracts)}"
        )
    lines.extend(["", "## Shared Resources", ""])
    for item in contract.shared_resources:
        lines.append(f"- `{item.resource}` / `{item.phase}` shared by {', '.join(item.contracts)}")
    lines.extend(["", "## Validation Plan", ""])
    for item in contract.hypotheses:
        lines.append(f"- `{item.id}` collect {', '.join(item.validation.collect)}")
    lines.extend(["", "## Candidate Actions", ""])
    for item in contract.hypotheses:
        action_text = ", ".join(item.candidate_actions) or "(none)"
        lines.append(f"- `{item.id}`: {action_text}")
    return "\n".join(lines).strip() + "\n"
