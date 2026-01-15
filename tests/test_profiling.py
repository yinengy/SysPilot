import pytest
from syspilot.profiling.collectors.base import CollectorResult
from syspilot.profiling.collectors.cpu import CPUCollector
from syspilot.profiling.collectors.memory import MemoryCollector
from datetime import datetime


@pytest.mark.asyncio
async def test_cpu_collector_system():
    collector = CPUCollector()
    assert collector.is_available()
    
    result = await collector.collect()
    assert result.collector_name == "cpu"
    assert "cpu_percent_avg" in result.data
    assert "cpu_count" in result.data


@pytest.mark.asyncio
async def test_memory_collector_system():
    collector = MemoryCollector()
    assert collector.is_available()
    
    result = await collector.collect()
    assert result.collector_name == "memory"
    assert "total_bytes" in result.data
    assert "percent_used" in result.data


def test_collector_result_summarize():
    result = CollectorResult(
        timestamp=datetime.now(),
        collector_name="test",
        data={
            "metric1": 100,
            "metric2": 200.5,
            "nested": {"a": 1, "b": 2},
            "list_data": [1, 2, 3, 4, 5],
        },
        raw_size_bytes=100,
    )
    
    summary = result.summarize()
    assert "[test]" in summary
    assert "metric1: 100" in summary
    assert "metric2: 200.5" in summary
