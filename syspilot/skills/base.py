from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SkillResult:
    success: bool
    output: str
    summary: str
    execution_time_seconds: float
    error: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


class BaseSkill(ABC):
    name: str = "base_skill"
    description: str = "Base skill class"
    
    @abstractmethod
    async def execute(self, **kwargs: Any) -> SkillResult:
        pass
    
    @abstractmethod
    def get_parameters_schema(self) -> dict[str, Any]:
        """Return JSON schema for skill parameters."""
        pass
    
    def validate_parameters(self, params: dict[str, Any]) -> bool:
        return True


class SkillRegistry:
    _instance: SkillRegistry | None = None
    
    def __new__(cls) -> SkillRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._skills = {}
        return cls._instance
    
    def __init__(self) -> None:
        if not hasattr(self, "_skills"):
            self._skills: dict[str, BaseSkill] = {}
    
    def register(self, skill: BaseSkill) -> None:
        self._skills[skill.name] = skill
        logger.info(f"Registered skill: {skill.name}")
    
    def get(self, name: str) -> BaseSkill | None:
        return self._skills.get(name)
    
    def list_skills(self) -> list[dict[str, Any]]:
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "parameters": skill.get_parameters_schema(),
            }
            for skill in self._skills.values()
        ]
    
    def get_tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": f"skill_{skill.name}",
                "description": skill.description,
                "parameters": skill.get_parameters_schema(),
            }
            for skill in self._skills.values()
        ]
