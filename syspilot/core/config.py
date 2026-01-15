"""
SysPilot Configuration Module

Handles all configuration for the SysPilot agent including LLM settings,
monitoring parameters, and skill configurations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


@dataclass
class LLMConfig:
    """LLM configuration settings."""
    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4o"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.1
    max_tokens: int = 4096
    
    def __post_init__(self) -> None:
        """Load API key from environment if not provided."""
        if self.api_key is None:
            if self.provider == LLMProvider.OPENAI:
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == LLMProvider.ANTHROPIC:
                self.api_key = os.getenv("ANTHROPIC_API_KEY")


@dataclass
class MonitoringConfig:
    """System monitoring configuration."""
    interval_seconds: float = 5.0
    cpu_threshold_percent: float = 80.0
    memory_threshold_percent: float = 85.0
    io_threshold_percent: float = 90.0
    enable_ebpf: bool = True
    profile_duration_seconds: float = 10.0


@dataclass
class SkillConfig:
    """Skill system configuration."""
    skills_dir: Path = field(default_factory=lambda: Path("~/.syspilot/skills").expanduser())
    enable_dynamic_scripts: bool = True
    script_timeout_seconds: float = 60.0
    sandbox_enabled: bool = True
    allowed_commands: list[str] = field(default_factory=lambda: [
        "perf", "strace", "bpftrace", "python3", "bash", "awk", "grep", "sed"
    ])


@dataclass
class TuningConfig:
    """Application tuning configuration."""
    auto_tune: bool = False  # Require confirmation by default
    backup_configs: bool = True
    dry_run: bool = True  # Show changes without applying
    max_tuning_iterations: int = 5


@dataclass
class SysPilotConfig:
    """Main configuration for SysPilot agent."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    skills: SkillConfig = field(default_factory=SkillConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    
    # Target application settings
    target_pid: int | None = None
    target_name: str | None = None
    
    # Logging
    log_level: str = "INFO"
    log_file: Path | None = None
    
    @classmethod
    def from_yaml(cls, path: Path) -> SysPilotConfig:
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SysPilotConfig:
        """Create configuration from dictionary."""
        config = cls()
        
        if "llm" in data:
            llm_data = data["llm"]
            if "provider" in llm_data:
                llm_data["provider"] = LLMProvider(llm_data["provider"])
            config.llm = LLMConfig(**llm_data)
        
        if "monitoring" in data:
            config.monitoring = MonitoringConfig(**data["monitoring"])
        
        if "skills" in data:
            skills_data = data["skills"]
            if "skills_dir" in skills_data:
                skills_data["skills_dir"] = Path(skills_data["skills_dir"]).expanduser()
            config.skills = SkillConfig(**skills_data)
        
        if "tuning" in data:
            config.tuning = TuningConfig(**data["tuning"])
        
        if "target_pid" in data:
            config.target_pid = data["target_pid"]
        if "target_name" in data:
            config.target_name = data["target_name"]
        if "log_level" in data:
            config.log_level = data["log_level"]
        if "log_file" in data:
            config.log_file = Path(data["log_file"])
        
        return config
    
    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "llm": {
                "provider": self.llm.provider.value,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
            },
            "monitoring": {
                "interval_seconds": self.monitoring.interval_seconds,
                "cpu_threshold_percent": self.monitoring.cpu_threshold_percent,
                "memory_threshold_percent": self.monitoring.memory_threshold_percent,
                "io_threshold_percent": self.monitoring.io_threshold_percent,
                "enable_ebpf": self.monitoring.enable_ebpf,
                "profile_duration_seconds": self.monitoring.profile_duration_seconds,
            },
            "skills": {
                "skills_dir": str(self.skills.skills_dir),
                "enable_dynamic_scripts": self.skills.enable_dynamic_scripts,
                "script_timeout_seconds": self.skills.script_timeout_seconds,
                "sandbox_enabled": self.skills.sandbox_enabled,
            },
            "tuning": {
                "auto_tune": self.tuning.auto_tune,
                "backup_configs": self.tuning.backup_configs,
                "dry_run": self.tuning.dry_run,
                "max_tuning_iterations": self.tuning.max_tuning_iterations,
            },
            "target_pid": self.target_pid,
            "target_name": self.target_name,
            "log_level": self.log_level,
        }
    
    def save_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
