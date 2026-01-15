from __future__ import annotations

from datetime import datetime

import psutil

from syspilot.profiling.collectors.base import BaseCollector, CollectorResult


class ProcessCollector(BaseCollector):
    name = "process"
    
    async def collect(self, pid: int | None = None) -> CollectorResult:
        if not pid:
            return CollectorResult(
                timestamp=datetime.now(),
                collector_name=self.name,
                data={"error": "PID required for process collector"},
                raw_size_bytes=0,
            )
        
        try:
            proc = psutil.Process(pid)
            
            with proc.oneshot():
                data = {
                    "pid": pid,
                    "name": proc.name(),
                    "status": proc.status(),
                    "create_time": proc.create_time(),
                    "cpu_percent": proc.cpu_percent(),
                    "memory_percent": proc.memory_percent(),
                    "num_threads": proc.num_threads(),
                    "num_fds": proc.num_fds() if hasattr(proc, "num_fds") else None,
                    "cmdline": proc.cmdline(),
                    "cwd": proc.cwd(),
                    "username": proc.username(),
                    "nice": proc.nice(),
                }
                
                try:
                    io_counters = proc.io_counters()
                    data["io_read_bytes"] = io_counters.read_bytes
                    data["io_write_bytes"] = io_counters.write_bytes
                    data["io_read_count"] = io_counters.read_count
                    data["io_write_count"] = io_counters.write_count
                except (psutil.AccessDenied, AttributeError):
                    pass
                
                try:
                    ctx_switches = proc.num_ctx_switches()
                    data["ctx_switches_voluntary"] = ctx_switches.voluntary
                    data["ctx_switches_involuntary"] = ctx_switches.involuntary
                except psutil.AccessDenied:
                    pass
                
                try:
                    connections = proc.connections()
                    data["num_connections"] = len(connections)
                    data["connection_statuses"] = {}
                    for conn in connections:
                        status = conn.status
                        data["connection_statuses"][status] = data["connection_statuses"].get(status, 0) + 1
                except psutil.AccessDenied:
                    pass
                    
        except psutil.NoSuchProcess:
            data = {"error": f"Process {pid} not found"}
        except psutil.AccessDenied:
            data = {"error": f"Access denied to process {pid}"}
        
        return CollectorResult(
            timestamp=datetime.now(),
            collector_name=self.name,
            data=data,
            raw_size_bytes=len(str(data)),
        )
    
    def is_available(self) -> bool:
        return True
