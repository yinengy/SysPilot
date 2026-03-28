from __future__ import annotations

from syspilot_contract.schemas import ComposedContract, LibraryContract


def _describe_demand(use: LibraryContract | ComposedContract, demand: object) -> str:
    if isinstance(demand, dict):
        kind = demand.get("kind", "unknown")
        if kind == "fixed":
            return f"fixed({demand.get('value')})"
        if kind == "range":
            return f"range({demand.get('lower')}, {demand.get('upper')})"
        if kind == "scaled_by":
            return f"scaled_by({demand.get('knob')}, {demand.get('lower')}, {demand.get('upper')})"
        if kind == "proportional_to":
            return f"proportional_to({demand.get('metric')})"
        if kind == "burst":
            return f"burst({demand.get('trigger')})"
    return "unknown"


def render_contract_markdown(contract: LibraryContract | ComposedContract) -> str:
    lines = [f"# {contract.contract} LPCL Contract", "", "Source of truth: contract.yaml", ""]
    if isinstance(contract, LibraryContract):
        lines.extend(["## Uses", ""])
        for item in contract.uses:
            demand = _describe_demand(contract, item.demand.model_dump(mode="json"))
            lines.append(
                f"- `{item.id}` → `{item.resource}` during `{item.phase}` at `{item.scope}` scope as `{item.role}`; sharing `{item.sharing}`; demand `{demand}`"
            )
        lines.extend(["", "## Assumptions", ""])
        for item in contract.assumptions:
            lines.append(
                f"- `{item.id}` → `{item.predicate.kind}` on `{item.predicate.resource}` during `{item.predicate.phase}`"
            )
        lines.extend(["", "## Controls", ""])
        for item in contract.controls:
            affects = ", ".join(item.affects)
            activation = (
                f", activation={item.activation_value}" if item.activation_value is not None else ""
            )
            lines.append(
                f"- `{item.name}` ({item.kind}, {item.type}{activation}) affects `{affects}` with `{item.effect.op}` on `{item.effect.resource}` at `{item.scope}` scope"
            )
        lines.extend(["", "## Signals", ""])
        for item in contract.signals:
            lines.append(
                f"- `{item.metric}` via `{item.probe.kind}:{item.probe.target}` watches `{item.resource}` and indicates `{item.indicates}`"
            )
        lines.extend(["", "## Recipes", ""])
        for item in contract.recipes:
            lines.append(
                f"- `{item.id}` → collect {', '.join(item.collect)}; controls: {', '.join(item.control_refs)}; actions: {', '.join(item.actions)}"
            )
    else:
        lines.extend(["## Shared Resources", ""])
        for item in contract.shared_resources:
            lines.append(
                f"- `{item.resource}` / `{item.phase}` shared by {', '.join(item.libraries)}"
            )
        lines.extend(["", "## Conflicts", ""])
        for item in contract.hypotheses:
            lines.append(
                f"- `{item.id}` → `{item.resource}` / `{item.phase}` across {', '.join(item.libraries)}; collect {', '.join(item.validation.collect)}"
            )
            if item.candidate_control_refs:
                lines.append(f"  - controls: {', '.join(item.candidate_control_refs)}")
            if item.candidate_recipe_refs:
                lines.append(f"  - recipes: {', '.join(item.candidate_recipe_refs)}")
    return "\n".join(lines).strip() + "\n"
