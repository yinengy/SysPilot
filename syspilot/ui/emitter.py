from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from uuid import uuid4

from syspilot.ui.events import (
    EVENT_SCHEMA_VERSION,
    EventPriority,
    tune_event_policy,
    utc_timestamp,
)


@dataclass(frozen=True)
class TuneEmitterStats:
    published_count: int
    dropped_count: int
    coalesced_count: int
    critical_sent_count: int
    buffered_count: int


@dataclass
class _BufferedEvent:
    envelope: dict[str, object]
    event_name: str
    priority: EventPriority
    droppable: bool
    coalescible: bool


class TuneEventEmitter:
    def __init__(
        self,
        *,
        run_id: str | None = None,
        capacity: int = 256,
        critical_reserve: int = 16,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if critical_reserve < 1:
            raise ValueError("critical_reserve must be >= 1")

        self._run_id: str = run_id or f"tune-{uuid4().hex[:12]}"
        self._capacity: int = capacity
        self._critical_reserve: int = min(critical_reserve, capacity)
        self._noncritical_budget: int = max(0, capacity - self._critical_reserve)

        self._seq: int = 0
        self._buffer: deque[_BufferedEvent] = deque()
        self._noncritical_count: int = 0

        self._published_count: int = 0
        self._dropped_count: int = 0
        self._coalesced_count: int = 0
        self._critical_sent_count: int = 0

    @property
    def run_id(self) -> str:
        return self._run_id

    def publish(
        self,
        event_name: str,
        payload: dict[str, object] | None = None,
        *,
        event_type: str | None = None,
    ) -> bool:
        policy = tune_event_policy(event_name)
        resolved_event_type = event_type or policy.event_type
        next_seq = self._next_seq()

        item = _BufferedEvent(
            envelope={
                "schema_version": EVENT_SCHEMA_VERSION,
                "run_id": self._run_id,
                "seq": next_seq,
                "ts": utc_timestamp(),
                "event_type": resolved_event_type,
                "event": event_name,
                "payload": payload or {},
            },
            event_name=event_name,
            priority=policy.priority,
            droppable=policy.droppable,
            coalescible=policy.coalescible,
        )

        if item.coalescible and self._replace_coalescible(item):
            self._coalesced_count += 1
            self._published_count += 1
            return True

        if item.priority != EventPriority.CRITICAL and self._noncritical_budget == 0:
            self._dropped_count += 1
            return False

        if (
            item.priority != EventPriority.CRITICAL
            and self._noncritical_count >= self._noncritical_budget
        ):
            self._dropped_count += 1
            return False

        if len(self._buffer) >= self._capacity:
            if item.priority == EventPriority.CRITICAL:
                self._evict_oldest_noncritical()
            elif item.droppable:
                self._dropped_count += 1
                return False

        if len(self._buffer) >= self._capacity:
            if item.priority == EventPriority.CRITICAL:
                self._evict_oldest_any()
            else:
                self._dropped_count += 1
                return False

        self._buffer.append(item)
        if item.priority == EventPriority.CRITICAL:
            self._critical_sent_count += 1
        else:
            self._noncritical_count += 1
        self._published_count += 1
        return True

    def drain(self) -> list[dict[str, object]]:
        events = [item.envelope for item in self._buffer]
        self._buffer.clear()
        self._noncritical_count = 0
        return events

    def snapshot(self) -> list[dict[str, object]]:
        return [item.envelope for item in self._buffer]

    def stats(self) -> TuneEmitterStats:
        return TuneEmitterStats(
            published_count=self._published_count,
            dropped_count=self._dropped_count,
            coalesced_count=self._coalesced_count,
            critical_sent_count=self._critical_sent_count,
            buffered_count=len(self._buffer),
        )

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _replace_coalescible(self, item: _BufferedEvent) -> bool:
        for index in range(len(self._buffer) - 1, -1, -1):
            existing = self._buffer[index]
            if existing.event_name == item.event_name and existing.coalescible:
                del self._buffer[index]
                self._buffer.append(item)
                return True
        return False

    def _evict_oldest_noncritical(self) -> None:
        for index, existing in enumerate(self._buffer):
            if existing.priority == EventPriority.CRITICAL:
                continue
            del self._buffer[index]
            self._noncritical_count -= 1
            self._dropped_count += 1
            return

    def _evict_oldest_any(self) -> None:
        if not self._buffer:
            return
        removed = self._buffer.popleft()
        if removed.priority != EventPriority.CRITICAL:
            self._noncritical_count -= 1
        self._dropped_count += 1
