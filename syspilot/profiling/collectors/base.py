from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class CollectorResult:
    timestamp: datetime
    collector_name: str
    data: dict[str, Any]
    raw_size_bytes: int = 0
    
    def summarize(self, max_length: int = 500) -> str:
        """Generate a condensed summary for LLM context."""
        lines = [f"[{self.collector_name}] {self.timestamp.isoformat()}"]
        
        for key, value in self.data.items():
            if isinstance(value, (int, float)):
                lines.append(f"  {key}: {value}")
            elif isinstance(value, dict):
                lines.append(f"  {key}: {len(value)} items")
            elif isinstance(value, list):
                lines.append(f"  {key}: {len(value)} entries")
            else:
                str_val = str(value)
                if len(str_val) > 50:
                    str_val = str_val[:47] + "..."
                lines.append(f"  {key}: {str_val}")
        
        summary = "\n".join(lines)
        if len(summary) > max_length:
            summary = summary[:max_length - 3] + "..."
        return summary


class BaseCollector(ABC):
    name: str = "base"
    
    @abstractmethod
    async def collect(self, pid: int | None = None) -> CollectorResult:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass
