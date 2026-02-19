from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

EVENT_SCHEMA_VERSION = "1.0"


class EventPriority(int, Enum):
    CRITICAL = 30
    NORMAL = 20
    LOW = 10


CRITICAL_EVENT_NAMES = frozenset(
    {
        "error",
        "run.started",
        "run.finished",
        "phase.started",
        "phase.finished",
    }
)

COALESCIBLE_EVENT_NAMES = frozenset({"progress.update", "phase.progress"})


@dataclass(frozen=True)
class EventPolicy:
    priority: EventPriority
    droppable: bool
    coalescible: bool
    event_type: str


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def is_critical_event(event_name: str) -> bool:
    return event_name in CRITICAL_EVENT_NAMES


def classify_event_type(event_name: str) -> str:
    if event_name == "error":
        return "error"
    if event_name.startswith("log."):
        return "log"
    if event_name.startswith("run."):
        if event_name == "run.finished":
            return "result"
        return "lifecycle"
    if event_name.startswith("phase."):
        return "lifecycle"
    if event_name.startswith("result."):
        return "result"
    if event_name.startswith("progress.") or event_name.startswith("candidate."):
        return "progress"
    if event_name.startswith("profile."):
        return "metrics"
    if event_name.startswith("recommendation."):
        return "recommendation"
    return "info"


def tune_event_policy(event_name: str) -> EventPolicy:
    if is_critical_event(event_name):
        return EventPolicy(
            priority=EventPriority.CRITICAL,
            droppable=False,
            coalescible=False,
            event_type=classify_event_type(event_name),
        )

    return EventPolicy(
        priority=EventPriority.LOW
        if event_name in COALESCIBLE_EVENT_NAMES
        else EventPriority.NORMAL,
        droppable=True,
        coalescible=event_name in COALESCIBLE_EVENT_NAMES,
        event_type=classify_event_type(event_name),
    )
