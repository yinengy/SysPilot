from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FixedBudgetConsumption:
    kind: str
    label: str
    reason: str
    count: int = 1


@dataclass
class FixedBudgetState:
    total: int | None
    used: int = 0
    no_action_reprompts_used: int = 0
    max_no_action_reprompts: int = 3
    consumptions: list[FixedBudgetConsumption] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.total is not None and self.total <= 0:
            raise ValueError("fixed budget total must be a positive integer")
        if self.used < 0:
            raise ValueError("fixed budget used cannot be negative")
        if self.max_no_action_reprompts < 0:
            raise ValueError("max_no_action_reprompts cannot be negative")
        if self.no_action_reprompts_used < 0:
            raise ValueError("no_action_reprompts_used cannot be negative")
        if self.total is not None and self.used > self.total:
            raise ValueError("fixed budget used cannot exceed total")

    @property
    def enabled(self) -> bool:
        return self.total is not None

    @property
    def remaining(self) -> int | None:
        if self.total is None:
            return None
        return self.total - self.used

    def can_consume(self, count: int = 1) -> bool:
        if count < 0:
            raise ValueError("count must be non-negative")
        if self.total is None:
            return True
        return (self.used + count) <= self.total

    def consume(self, kind: str, label: str, reason: str, count: int = 1) -> bool:
        if count <= 0:
            raise ValueError("count must be a positive integer")
        if not self.can_consume(count):
            return False
        self.used += count
        self.consumptions.append(
            FixedBudgetConsumption(kind=kind, label=label, reason=reason, count=count)
        )
        return True

    def can_reprompt_no_action(self, stall_reprompts_used: int | None = None) -> bool:
        if stall_reprompts_used is not None and stall_reprompts_used < 0:
            raise ValueError("stall_reprompts_used cannot be negative")
        if stall_reprompts_used is None:
            stall_reprompts_used = self.no_action_reprompts_used
        return stall_reprompts_used < self.max_no_action_reprompts

    def record_no_action_reprompt(self, stall_reprompts_used: int | None = None) -> bool:
        if not self.can_reprompt_no_action(stall_reprompts_used):
            return False
        self.no_action_reprompts_used += 1
        return True

    def to_payload(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "total": self.total,
            "used": self.used,
            "remaining": self.remaining,
            "no_action_reprompts_used": self.no_action_reprompts_used,
            "max_no_action_reprompts": self.max_no_action_reprompts,
            "consumptions": [
                {
                    "kind": item.kind,
                    "label": item.label,
                    "reason": item.reason,
                    "count": item.count,
                }
                for item in self.consumptions
            ],
        }
