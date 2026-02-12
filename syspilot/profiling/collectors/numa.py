from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from syspilot.profiling.collectors.base import BaseCollector, CollectorResult

logger = logging.getLogger(__name__)

# Base sysfs path for NUMA topology
NUMA_SYSFS = Path("/sys/devices/system/node")


class NUMACollector(BaseCollector):
    """Collector for NUMA topology and per-process memory placement data."""

    name = "numa"

    async def collect(self, pid: int | None = None) -> CollectorResult:
        data: dict[str, Any] = {}

        # -- System-level NUMA topology (always collected) --
        nodes = self._discover_nodes()
        data["num_nodes"] = len(nodes)

        topology: dict[str, Any] = {}
        for node_id in nodes:
            node_info: dict[str, Any] = {}
            node_info["cpulist"] = self._read_cpulist(node_id)
            node_info.update(self._read_meminfo(node_id))
            node_info["distance"] = self._read_distance(node_id)
            topology[f"node{node_id}"] = node_info
        data["topology"] = topology

        # -- Per-process NUMA placement (when pid given) --
        if pid is not None:
            numa_maps = self._read_numa_maps(pid)
            if numa_maps is not None:
                data["pid"] = pid
                data["numa_maps_summary"] = numa_maps
            else:
                data["pid"] = pid
                data["numa_maps_summary"] = {"error": f"Cannot read numa_maps for PID {pid}"}

        return CollectorResult(
            timestamp=datetime.now(),
            collector_name=self.name,
            data=data,
            raw_size_bytes=len(str(data)),
        )

    def is_available(self) -> bool:
        return NUMA_SYSFS.exists() and len(self._discover_nodes()) > 0

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _discover_nodes() -> list[int]:
        """Return sorted list of NUMA node IDs present on the system."""
        if not NUMA_SYSFS.exists():
            return []
        nodes: list[int] = []
        for entry in NUMA_SYSFS.iterdir():
            match = re.match(r"^node(\d+)$", entry.name)
            if match and entry.is_dir():
                nodes.append(int(match.group(1)))
        return sorted(nodes)

    @staticmethod
    def _read_cpulist(node_id: int) -> str:
        """Read the CPU list string for a given node (e.g. '0-11')."""
        path = NUMA_SYSFS / f"node{node_id}" / "cpulist"
        try:
            return path.read_text().strip()
        except OSError:
            return ""

    @staticmethod
    def _read_meminfo(node_id: int) -> dict[str, int]:
        """Parse MemTotal and MemFree (in bytes) from node meminfo."""
        path = NUMA_SYSFS / f"node{node_id}" / "meminfo"
        result: dict[str, int] = {}
        try:
            for line in path.read_text().splitlines():
                if "MemTotal" in line:
                    result["mem_total_kb"] = int(line.split()[-2])
                elif "MemFree" in line:
                    result["mem_free_kb"] = int(line.split()[-2])
        except (OSError, ValueError, IndexError):
            pass
        return result

    @staticmethod
    def _read_distance(node_id: int) -> list[int]:
        """Read NUMA distance vector for *node_id*."""
        path = NUMA_SYSFS / f"node{node_id}" / "distance"
        try:
            return [int(d) for d in path.read_text().split()]
        except (OSError, ValueError):
            return []

    @staticmethod
    def _read_numa_maps(pid: int) -> dict[str, Any] | None:
        """Summarise /proc/{pid}/numa_maps into per-node page counts."""
        maps_path = Path(f"/proc/{pid}/numa_maps")
        if not maps_path.exists():
            return None

        per_node_pages: dict[str, int] = {}
        total_pages = 0
        num_mappings = 0

        try:
            for line in maps_path.read_text().splitlines():
                num_mappings += 1
                # Tokens like N0=1234 N1=567
                for token in line.split():
                    m = re.match(r"^N(\d+)=(\d+)$", token)
                    if m:
                        node_key = f"node{m.group(1)}"
                        pages = int(m.group(2))
                        per_node_pages[node_key] = per_node_pages.get(node_key, 0) + pages
                        total_pages += pages
        except OSError:
            return None

        return {
            "num_mappings": num_mappings,
            "total_pages": total_pages,
            "per_node_pages": per_node_pages,
        }
