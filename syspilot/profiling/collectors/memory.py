from __future__ import annotations

from datetime import datetime

import psutil

from syspilot.profiling.collectors.base import BaseCollector, CollectorResult


class MemoryCollector(BaseCollector):
    name = "memory"
    
    async def collect(self, pid: int | None = None) -> CollectorResult:
        if pid:
            try:
                proc = psutil.Process(pid)
                mem_info = proc.memory_info()
                mem_percent = proc.memory_percent()
                
                data = {
                    "pid": pid,
                    "rss_bytes": mem_info.rss,
                    "vms_bytes": mem_info.vms,
                    "rss_mb": mem_info.rss / (1024 * 1024),
                    "vms_mb": mem_info.vms / (1024 * 1024),
                    "memory_percent": mem_percent,
                    "shared_bytes": getattr(mem_info, "shared", 0),
                }
            except psutil.NoSuchProcess:
                data = {"error": f"Process {pid} not found"}
        else:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            data = {
                "total_bytes": mem.total,
                "available_bytes": mem.available,
                "used_bytes": mem.used,
                "free_bytes": mem.free,
                "percent_used": mem.percent,
                "total_gb": mem.total / (1024**3),
                "available_gb": mem.available / (1024**3),
                "used_gb": mem.used / (1024**3),
                "cached_bytes": getattr(mem, "cached", 0),
                "buffers_bytes": getattr(mem, "buffers", 0),
                "swap_total_bytes": swap.total,
                "swap_used_bytes": swap.used,
                "swap_percent": swap.percent,
            }
        
        return CollectorResult(
            timestamp=datetime.now(),
            collector_name=self.name,
            data=data,
            raw_size_bytes=len(str(data)),
        )
    
    def is_available(self) -> bool:
        return True
