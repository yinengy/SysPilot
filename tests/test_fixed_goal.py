from __future__ import annotations

from syspilot.fixed_goal import fixed_goal_payload, fixed_goal_reached


def test_fixed_goal_reached_uses_objective_direction() -> None:
    assert fixed_goal_reached(1231.1, 1231.1, "maximize") is True
    assert fixed_goal_reached(1230.0, 1231.1, "maximize") is False
    assert fixed_goal_reached(0.8, 0.85, "minimize") is True
    assert fixed_goal_reached(0.9, 0.85, "minimize") is False


def test_fixed_goal_payload_is_empty_when_disabled() -> None:
    assert fixed_goal_payload(threshold=None, value=1.0, objective_mode="maximize") == {}


def test_fixed_goal_payload_includes_status() -> None:
    payload = fixed_goal_payload(
        threshold=10.0,
        value=12.0,
        objective_mode="maximize",
        stop_reason="fixed_goal_reached",
    )

    assert payload["fixed_goal_threshold"] == 10.0
    assert payload["fixed_goal_current_value"] == 12.0
    assert payload["fixed_goal_mode"] == "maximize"
    assert payload["fixed_goal_reached"] is True
    assert payload["fixed_goal_stop_reason"] == "fixed_goal_reached"
