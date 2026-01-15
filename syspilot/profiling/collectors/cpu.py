from __future__ import annotations

from datetime import datetime

import psutil

from syspilot.profiling.collectors.base import BaseCollector, CollectorResult


class CPUCollector(BaseCollector):
    name = "cpu"
    
    async def collect(self, pid: int | None = None) -> CollectorResult:
        if pid:
            try:
                proc = psutil.Process(pid)
                cpu_percent = proc.cpu_percent(interval=0.1)
                cpu_times = proc.cpu_times()
                data = {
                    "pid": pid,
                    "cpu_percent": cpu_percent,
                    "user_time": cpu_times.user,
                    "system_time": cpu_times.system,
                    "num_threads": proc.num_threads(),
                }
            except psutil.NoSuchProcess:
                data = {"error": f"Process {pid} not found"}
        else:
            cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
            cpu_times = psutil.cpu_times()
            cpu_freq = psutil.cpu_freq()
            
            data = {
                "cpu_percent_per_core": cpu_percent,
                "cpu_percent_avg": sum(cpu_percent) / len(cpu_percent),
                "cpu_count": psutil.cpu_count(),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "user_time": cpu_times.user,
                "system_time": cpu_times.system,
                "idle_time": cpu_times.idle,
                "iowait_time": getattr(cpu_times, "iowait", 0),
                "freq_current": cpu_freq.current if cpu_freq else None,
                "freq_max": cpu_freq.max if cpu_freq else None,
            }
        
        return CollectorResult(
            timestamp=datetime.now(),
            collector_name=self.name,
            data=data,
            raw_size_bytes=len(str(data)),
        )
    
    def is_available(self) -> bool:
        return True
