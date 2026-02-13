from __future__ import annotations

import asyncio
import logging

from syspilot.profiling.collectors.base import BaseCollector, CollectorResult
from syspilot.profiling.collectors.cpu import CPUCollector
from syspilot.profiling.collectors.memory import MemoryCollector
from syspilot.profiling.collectors.numa import NUMACollector
from syspilot.profiling.collectors.process import ProcessCollector

logger = logging.getLogger(__name__)


class ProfilingManager:
    def __init__(self) -> None:
        self._collectors: dict[str, BaseCollector] = {}
        self._register_default_collectors()
    
    def _register_default_collectors(self) -> None:
        for collector_cls in [CPUCollector, MemoryCollector, ProcessCollector, NUMACollector]:
            collector = collector_cls()
            if collector.is_available():
                self._collectors[collector.name] = collector
                logger.debug(f"Registered collector: {collector.name}")
    
    def register_collector(self, collector: BaseCollector) -> None:
        if collector.is_available():
            self._collectors[collector.name] = collector
            logger.info(f"Registered custom collector: {collector.name}")
    
    async def collect(
        self,
        collector_names: list[str] | None = None,
        pid: int | None = None,
    ) -> dict[str, CollectorResult]:
        requested_names = collector_names or list(self._collectors.keys())

        selected_names = [name for name in requested_names if name in self._collectors]
        for name in requested_names:
            if name not in self._collectors:
                logger.warning("Unknown collector requested: %s", name)

        if not selected_names:
            return {}

        tasks = [self._collect_one(name, pid) for name in selected_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output: dict[str, CollectorResult] = {}
        for name, result in zip(selected_names, results):
            if isinstance(result, BaseException):
                logger.error("Collector %s failed: %s", name, result)
            else:
                output[name] = result

        return output
    
    async def _collect_one(self, name: str, pid: int | None) -> CollectorResult:
        return await self._collectors[name].collect(pid)
    
    async def collect_all(self, pid: int | None = None) -> dict[str, CollectorResult]:
        return await self.collect(None, pid)
    
    def get_available_collectors(self) -> list[str]:
        return list(self._collectors.keys())
    
    def summarize_results(
        self,
        results: dict[str, CollectorResult],
        max_length: int = 2000,
    ) -> str:
        if not results:
            return ""

        per_result_budget = max(1, max_length // len(results))
        summaries = [result.summarize(per_result_budget) for result in results.values()]
        combined = "\n\n".join(summaries)
        if len(combined) > max_length:
            return combined[: max_length - 3] + "..."
        return combined
