from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from syspilot.profiling.collectors.base import BaseCollector, CollectorResult
from syspilot.profiling.collectors.cpu import CPUCollector
from syspilot.profiling.collectors.memory import MemoryCollector
from syspilot.profiling.collectors.process import ProcessCollector

logger = logging.getLogger(__name__)


class ProfilingManager:
    def __init__(self) -> None:
        self._collectors: dict[str, BaseCollector] = {}
        self._register_default_collectors()
    
    def _register_default_collectors(self) -> None:
        for collector_cls in [CPUCollector, MemoryCollector, ProcessCollector]:
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
        names = collector_names or list(self._collectors.keys())
        
        tasks = []
        for name in names:
            if name in self._collectors:
                tasks.append(self._collect_one(name, pid))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        output = {}
        for name, result in zip(names, results):
            if isinstance(result, Exception):
                logger.error(f"Collector {name} failed: {result}")
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
        summaries = []
        for name, result in results.items():
            summaries.append(result.summarize(max_length // len(results)))
        return "\n\n".join(summaries)
