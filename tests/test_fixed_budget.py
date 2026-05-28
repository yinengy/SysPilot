from __future__ import annotations

import pytest

from syspilot.fixed_budget import FixedBudgetState


def test_fixed_budget_disabled_state_is_unbounded() -> None:
    budget = FixedBudgetState(total=None)

    assert budget.enabled is False
    assert budget.remaining is None
    assert budget.can_consume()
    assert budget.consume("candidate", "trial-1", "disabled budget bookkeeping") is True
    assert budget.used == 1


def test_fixed_budget_enabled_state_tracks_remaining_and_blocks_overconsumption() -> None:
    budget = FixedBudgetState(total=2)

    assert budget.enabled is True
    assert budget.remaining == 2
    assert budget.consume("candidate", "trial-1", "first") is True
    assert budget.used == 1
    assert budget.remaining == 1
    assert budget.consume("candidate", "trial-2", "second") is True
    assert budget.used == 2
    assert budget.remaining == 0
    assert budget.can_consume() is False
    assert budget.consume("candidate", "trial-3", "exhausted") is False
    assert budget.used == 2


def test_fixed_budget_rejects_non_positive_total() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _ = FixedBudgetState(total=0)

    with pytest.raises(ValueError, match="positive integer"):
        _ = FixedBudgetState(total=-1)


def test_fixed_budget_payload_includes_no_action_reprompt_state() -> None:
    budget = FixedBudgetState(total=3, max_no_action_reprompts=2)

    assert budget.can_reprompt_no_action() is True
    assert budget.record_no_action_reprompt() is True
    assert budget.can_reprompt_no_action() is True
    assert budget.record_no_action_reprompt() is True
    assert budget.can_reprompt_no_action() is False
    assert budget.record_no_action_reprompt() is False

    payload = budget.to_payload()

    assert payload["total"] == 3
    assert payload["used"] == 0
    assert payload["remaining"] == 3
    assert payload["no_action_reprompts_used"] == 2
    assert payload["max_no_action_reprompts"] == 2


def test_fixed_budget_no_action_reprompt_cap_can_be_checked_per_stall() -> None:
    budget = FixedBudgetState(total=4, max_no_action_reprompts=2)

    assert budget.record_no_action_reprompt(stall_reprompts_used=0) is True
    assert budget.record_no_action_reprompt(stall_reprompts_used=1) is True
    assert budget.record_no_action_reprompt(stall_reprompts_used=2) is False
    assert budget.no_action_reprompts_used == 2

    assert budget.record_no_action_reprompt(stall_reprompts_used=0) is True
    assert budget.no_action_reprompts_used == 3
