from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


PROMPT_JSON_LIMIT_MARKER = "__syspilot_prompt_note__"
DEFAULT_PROMPT_JSON_CHAR_LIMIT = 80_000
DEFAULT_CONTRACT_CONTEXT_CHAR_LIMIT = 60_000
DEFAULT_PROFILING_JSON_CHAR_LIMIT = 45_000
_NO_COMPACTED_VALUE = object()

_DROP_KEYS = {
    "evidence",
    "source_artifacts",
    "source_refs",
    "resolved_components",
    "resolved_uses",
}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def truncate_prompt_text(text: str, max_chars: int, *, label: str = "text") -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return (
        text[:max_chars]
        + f"\n...[truncated {omitted} chars from {label}; full data remains in run artifacts]"
    )


def _compact_scalar(value: Any, max_string_chars: int) -> Any:
    if isinstance(value, str):
        return truncate_prompt_text(value, max_string_chars, label="field")
    return value


def compact_for_prompt(
    value: Any,
    *,
    max_depth: int = 6,
    max_list_items: int = 8,
    max_string_chars: int = 1_200,
) -> Any:
    if max_depth <= 0:
        if isinstance(value, Mapping):
            return {PROMPT_JSON_LIMIT_MARKER: "nested object omitted for prompt budget"}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [f"... {len(value)} nested item(s) omitted for prompt budget"]
        return _compact_scalar(value, max_string_chars)

    if isinstance(value, Mapping):
        compacted: dict[str, Any] = {}
        for raw_key, raw_item in value.items():
            key = str(raw_key)
            if key in _DROP_KEYS:
                continue
            compacted[key] = compact_for_prompt(
                raw_item,
                max_depth=max_depth - 1,
                max_list_items=max_list_items,
                max_string_chars=max_string_chars,
            )
        return compacted

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        if len(items) <= max_list_items:
            selected: list[Any] = items
            omitted = 0
        else:
            head_count = max(1, max_list_items // 2)
            tail_count = max(1, max_list_items - head_count)
            selected = [*items[:head_count], *items[-tail_count:]]
            omitted = len(items) - len(selected)
        compacted_items = [
            compact_for_prompt(
                item,
                max_depth=max_depth - 1,
                max_list_items=max_list_items,
                max_string_chars=max_string_chars,
            )
            for item in selected
        ]
        if omitted:
            compacted_items.insert(
                max(1, len(compacted_items) // 2),
                {PROMPT_JSON_LIMIT_MARKER: f"{omitted} list item(s) omitted for prompt budget"},
            )
        return compacted_items

    return _compact_scalar(value, max_string_chars)


def _compact_control_entry(entry: Any) -> Any:
    if not isinstance(entry, Mapping):
        return compact_for_prompt(entry, max_depth=3)
    control = entry.get("control")
    if not isinstance(control, Mapping):
        return compact_for_prompt(entry, max_depth=3)
    payload: dict[str, Any] = {
        "ref": entry.get("ref") or entry.get("control_ref"),
        "contract": entry.get("contract") or entry.get("library"),
        "name": control.get("name"),
        "kind": control.get("kind"),
        "type": control.get("type"),
        "target": control.get("target"),
        "activation_value": control.get("activation_value"),
        "effect": compact_for_prompt(control.get("effect"), max_depth=2),
    }
    return {key: value for key, value in payload.items() if value not in (None, "")}


def _compact_signal_entry(entry: Any) -> Any:
    if not isinstance(entry, Mapping):
        return compact_for_prompt(entry, max_depth=3)
    signal = entry.get("signal")
    if not isinstance(signal, Mapping):
        return compact_for_prompt(entry, max_depth=3)
    payload: dict[str, Any] = {
        "ref": entry.get("ref") or entry.get("signal_ref"),
        "contract": entry.get("contract") or entry.get("library"),
        "role": entry.get("role") or signal.get("role"),
        "metric": signal.get("metric"),
        "virtual_probe": entry.get("virtual_probe") or signal.get("virtual_probe"),
        "expected_direction_when_mitigated": entry.get("expected_direction_when_mitigated")
        or signal.get("expected_direction_when_mitigated"),
        "candidate_controls": entry.get("candidate_controls") or signal.get("candidate_controls"),
        "resolved_probe": compact_for_prompt(entry.get("resolved_probe"), max_depth=2),
    }
    return {key: value for key, value in payload.items() if value not in (None, "", [])}


def _compact_recipe_entry(entry: Any) -> Any:
    if not isinstance(entry, Mapping):
        return compact_for_prompt(entry, max_depth=3)
    recipe = entry.get("recipe")
    if not isinstance(recipe, Mapping):
        return compact_for_prompt(entry, max_depth=3)
    actions = _as_list(recipe.get("actions"))
    diagnose = _as_list(recipe.get("diagnose"))
    return {
        "ref": entry.get("ref") or entry.get("recipe_ref"),
        "contract": entry.get("contract") or entry.get("library"),
        "hypothesis_kind": recipe.get("hypothesis_kind"),
        "collect": recipe.get("collect"),
        "diagnose": [compact_for_prompt(item, max_depth=2) for item in diagnose[:4]],
        "actions": [compact_for_prompt(item, max_depth=2) for item in actions[:6]],
        "guardrails": _as_list(recipe.get("guardrails"))[:6],
        "search_space_hints": _as_list(recipe.get("search_space_hints"))[:6],
    }


def _compact_hypothesis_profile(profile: Any) -> Any:
    if not isinstance(profile, Mapping):
        return compact_for_prompt(profile, max_depth=4)
    return {
        "hypothesis_id": profile.get("hypothesis_id"),
        "hypothesis_name": profile.get("hypothesis_name"),
        "kind": profile.get("kind"),
        "resource": profile.get("resource"),
        "contracts": profile.get("contracts"),
        "validation_signals": [
            _compact_signal_entry(item) for item in _as_list(profile.get("validation_signals"))[:8]
        ],
        "signals_to_collect": _as_list(profile.get("signals_to_collect"))[:8],
        "report_controls": [
            _compact_control_entry(item) for item in _as_list(profile.get("report_controls"))[:8]
        ],
        "candidate_controls": [
            _compact_control_entry(item) for item in _as_list(profile.get("candidate_controls"))[:10]
        ],
        "candidate_recipes": [
            _compact_recipe_entry(item) for item in _as_list(profile.get("candidate_recipes"))[:4]
        ],
        "diagnose": compact_for_prompt(profile.get("diagnose", []), max_depth=3, max_list_items=8),
        "expected_signal_changes": compact_for_prompt(
            profile.get("expected_signal_changes", []), max_depth=3, max_list_items=8
        ),
        "guardrails": _as_list(profile.get("guardrails"))[:8],
        "search_space_hints": _as_list(profile.get("search_space_hints"))[:8],
        "explanation": truncate_prompt_text(str(profile.get("explanation", "")), 1_500, label="explanation"),
    }


def compact_contract_context_for_prompt(bundle: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(bundle, Mapping) or not bundle:
        return {}
    compacted: dict[str, Any] = {
        "summary": bundle.get("summary"),
        "relevant_resources": bundle.get("relevant_resources", []),
        "relevant_components": compact_for_prompt(
            bundle.get("relevant_components", []), max_depth=4, max_list_items=8
        ),
        "profile_first": [_compact_signal_entry(item) for item in _as_list(bundle.get("profile_first"))[:8]],
        "candidate_actions": compact_for_prompt(
            bundle.get("candidate_actions", []), max_depth=4, max_list_items=8
        ),
        "candidate_controls": [
            _compact_control_entry(item) for item in _as_list(bundle.get("candidate_controls"))[:12]
        ],
        "recipes": [
            _compact_recipe_entry(item)
            for item in _as_list(bundle.get("recipes"))[:6]
        ],
        "hypothesis_profiles": [
            _compact_hypothesis_profile(item)
            for item in _as_list(bundle.get("hypothesis_profiles"))[:4]
        ],
        "conflicts": compact_for_prompt(bundle.get("conflicts", []), max_depth=4, max_list_items=4),
        "validation": compact_for_prompt(bundle.get("validation", []), max_depth=4, max_list_items=6),
    }
    return {key: value for key, value in compacted.items() if value not in (None, "", [])}


def json_for_prompt(
    value: Any,
    *,
    max_chars: int = DEFAULT_PROMPT_JSON_CHAR_LIMIT,
    label: str = "JSON payload",
    compacted_value: object = _NO_COMPACTED_VALUE,
) -> str:
    raw_text = json.dumps(value, indent=2, sort_keys=True, default=str)
    if len(raw_text) <= max_chars:
        return raw_text

    payload = compact_for_prompt(value) if compacted_value is _NO_COMPACTED_VALUE else compacted_value
    compacted_text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if len(compacted_text) <= max_chars:
        return compacted_text

    compact_text = json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str)
    return truncate_prompt_text(compact_text, max_chars, label=label)


def contract_context_json_for_prompt(bundle: Mapping[str, Any] | None) -> str:
    if not isinstance(bundle, Mapping) or not bundle:
        return ""
    return json_for_prompt(
        bundle,
        max_chars=DEFAULT_CONTRACT_CONTEXT_CHAR_LIMIT,
        label="contract context",
        compacted_value=compact_contract_context_for_prompt(bundle),
    )


def profiling_json_for_prompt(value: Any) -> str:
    return json_for_prompt(
        value,
        max_chars=DEFAULT_PROFILING_JSON_CHAR_LIMIT,
        label="profiling JSON",
    )
