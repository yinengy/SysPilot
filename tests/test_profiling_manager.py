from __future__ import annotations

from datetime import datetime

import pytest

from syspilot.profiling.collectors.base import BaseCollector, CollectorResult
from syspilot.profiling.manager import ProfilingManager


class StubCollector(BaseCollector):
    def __init__(
        self,
        name: str,
        result: CollectorResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self._result = result
        self._error = error
        self.calls: list[int | None] = []

    async def collect(self, pid: int | None = None) -> CollectorResult:
        self.calls.append(pid)
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result

    def is_available(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_collect_ignores_unknown_collectors() -> None:
    manager = ProfilingManager()
    cpu_result = CollectorResult(
        timestamp=datetime.now(),
        collector_name="cpu",
        data={"cpu_percent": 10.0},
    )
    cpu_collector = StubCollector("cpu", result=cpu_result)
    manager._collectors = {"cpu": cpu_collector}

    results = await manager.collect(["missing", "cpu"], pid=4321)

    assert list(results.keys()) == ["cpu"]
    assert results["cpu"] is cpu_result
    assert cpu_collector.calls == [4321]


@pytest.mark.asyncio
async def test_collect_skips_failed_collectors() -> None:
    manager = ProfilingManager()
    failing_collector = StubCollector("cpu", error=RuntimeError("boom"))
    manager._collectors = {"cpu": failing_collector}

    results = await manager.collect(["cpu"], pid=123)

    assert results == {}
    assert failing_collector.calls == [123]


def test_summarize_results_empty_returns_empty_string() -> None:
    manager = ProfilingManager()
    assert manager.summarize_results({}) == ""


def test_summarize_results_respects_max_length() -> None:
    manager = ProfilingManager()
    result = CollectorResult(
        timestamp=datetime.now(),
        collector_name="cpu",
        data={"long": "x" * 400},
    )

    summary = manager.summarize_results({"cpu": result}, max_length=60)

    assert len(summary) <= 60
    assert summary.startswith("[cpu]")
