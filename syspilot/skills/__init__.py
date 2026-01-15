"""Skill system for dynamic script generation and execution."""

from syspilot.skills.base import BaseSkill, SkillResult, SkillRegistry
from syspilot.skills.executor import ScriptExecutor

__all__ = ["BaseSkill", "SkillResult", "SkillRegistry", "ScriptExecutor"]
