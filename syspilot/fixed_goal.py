from __future__ import annotations


def fixed_goal_reached(
    value: float | None,
    threshold: float | None,
    objective_mode: str,
) -> bool:
    if threshold is None or value is None:
        return False
    tolerance = 1e-12
    if objective_mode == "minimize":
        return value <= threshold + tolerance
    return value >= threshold - tolerance


def fixed_goal_payload(
    *,
    threshold: float | None,
    value: float | None,
    objective_mode: str,
    stop_reason: str = "",
) -> dict[str, object]:
    if threshold is None:
        return {}
    return {
        "fixed_goal_threshold": threshold,
        "fixed_goal_current_value": value,
        "fixed_goal_mode": objective_mode,
        "fixed_goal_reached": fixed_goal_reached(value, threshold, objective_mode),
        "fixed_goal_stop_reason": stop_reason,
    }
