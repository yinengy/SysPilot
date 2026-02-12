import pytest
from pathlib import Path
from unittest.mock import patch
from datetime import datetime

from syspilot.profiling.collectors.numa import NUMACollector, NUMA_SYSFS


@pytest.mark.asyncio
async def test_numa_collector_system():
    collector = NUMACollector()
    if not collector.is_available():
        pytest.skip("NUMA not available on this system")

    result = await collector.collect()
    assert result.collector_name == "numa"
    assert "num_nodes" in result.data
    assert result.data["num_nodes"] > 0
    assert "topology" in result.data

    node0 = result.data["topology"].get("node0")
    assert node0 is not None
    assert "cpulist" in node0
    assert "mem_total_kb" in node0
    assert "distance" in node0


@pytest.mark.asyncio
async def test_numa_collector_with_pid():
    collector = NUMACollector()
    if not collector.is_available():
        pytest.skip("NUMA not available on this system")

    import os
    pid = os.getpid()
    result = await collector.collect(pid=pid)
    assert result.data.get("pid") == pid
    assert "numa_maps_summary" in result.data


def test_numa_discover_nodes():
    collector = NUMACollector()
    if not NUMA_SYSFS.exists():
        pytest.skip("NUMA sysfs not available")

    nodes = collector._discover_nodes()
    assert isinstance(nodes, list)
    assert nodes == sorted(nodes)


def test_numa_collector_availability():
    collector = NUMACollector()
    available = collector.is_available()
    assert isinstance(available, bool)
