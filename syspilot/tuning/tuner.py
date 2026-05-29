from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from syspilot.core.config import TuningConfig
from syspilot.tuning.detector import ConfigDetector, DetectedConfig

logger = logging.getLogger(__name__)


@dataclass
class TuningAction:
    config_type: str
    parameter: str
    current_value: Any
    new_value: Any
    reason: str
    pid: int | None = None
    applied: bool = False
    backup_path: Path | None = None


class ApplicationTuner:
    def __init__(self, config: TuningConfig) -> None:
        self.config = config
        self.detector = ConfigDetector()
        self._history: list[TuningAction] = []
        self._backup_dir = Path("~/.syspilot/backups").expanduser()
    
    def analyze(self, pid: int) -> dict[str, Any]:
        detected_configs = self.detector.detect_for_process(pid)
        
        analysis = {
            "pid": pid,
            "detected_configs": [
                {
                    "type": c.config_type,
                    "path": str(c.path) if c.path else None,
                    "parameters": c.parameters,
                    "source": c.source,
                }
                for c in detected_configs
            ],
            "recommendations": self._generate_recommendations(detected_configs),
        }
        
        return analysis
    
    def _generate_recommendations(
        self,
        configs: list[DetectedConfig],
    ) -> list[dict[str, Any]]:
        recommendations = []
        
        for config in configs:
            if config.config_type == "java_cmdline":
                recommendations.extend(self._java_recommendations(config))
            elif config.config_type == "nodejs_cmdline":
                recommendations.extend(self._nodejs_recommendations(config))
            elif config.config_type == "system_limits":
                recommendations.extend(self._limits_recommendations(config))
        
        return recommendations
    
    def _java_recommendations(self, config: DetectedConfig) -> list[dict[str, Any]]:
        recs = []
        params = config.parameters
        
        has_xmx = any("-Xmx" in p for p in params)
        has_xms = any("-Xms" in p for p in params)
        
        if not has_xmx:
            recs.append({
                "parameter": "-Xmx",
                "recommendation": "Set explicit max heap size",
                "reason": "Prevents unbounded memory growth",
                "priority": "high",
            })
        
        if not has_xms:
            recs.append({
                "parameter": "-Xms",
                "recommendation": "Set initial heap size equal to max",
                "reason": "Reduces GC overhead from heap resizing",
                "priority": "medium",
            })
        
        return recs
    
    def _nodejs_recommendations(self, config: DetectedConfig) -> list[dict[str, Any]]:
        recs = []
        params = config.parameters
        
        has_max_old_space = any("--max-old-space-size" in p for p in params)
        
        if not has_max_old_space:
            recs.append({
                "parameter": "--max-old-space-size",
                "recommendation": "Set explicit V8 heap limit",
                "reason": "Prevents OOM in memory-intensive applications",
                "priority": "medium",
            })
        
        return recs
    
    def _limits_recommendations(self, config: DetectedConfig) -> list[dict[str, Any]]:
        recs = []
        params = config.parameters
        
        nofile = params.get("Max open files", {})
        if nofile:
            soft = nofile.get("soft", "0")
            if soft != "unlimited" and int(soft) < 65536:
                recs.append({
                    "parameter": "nofile",
                    "current": soft,
                    "recommendation": "Increase to at least 65536",
                    "reason": "Low file descriptor limit can cause connection failures",
                    "priority": "high",
                })
        
        return recs
    
    def apply_tuning(
        self,
        pid: int,
        actions: list[dict[str, Any]],
    ) -> list[TuningAction]:
        results = []
        
        for action_dict in actions:
            action = TuningAction(
                config_type=action_dict.get("config_type", "unknown"),
                parameter=action_dict.get("parameter", ""),
                current_value=action_dict.get("current_value"),
                new_value=action_dict.get("new_value"),
                reason=action_dict.get("reason", ""),
                pid=pid,
            )
            
            if self.config.dry_run:
                logger.info(f"[DRY RUN] Would apply: {action}")
                results.append(action)
                continue
            
            if self.config.backup_configs:
                action.backup_path = self._create_backup(action)
            
            success = self._apply_action(pid, action)
            action.applied = success
            results.append(action)
            self._history.append(action)
        
        return results
    
    def _create_backup(self, action: TuningAction) -> Path | None:
        self._backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self._backup_dir / f"{action.config_type}_{timestamp}.json"
        
        backup_data = {
            "timestamp": timestamp,
            "config_type": action.config_type,
            "parameter": action.parameter,
            "original_value": action.current_value,
            "pid": action.pid,
        }
        
        backup_path.write_text(json.dumps(backup_data, indent=2))
        return backup_path
    
    def _apply_action(self, pid: int | None, action: TuningAction) -> bool:
        logger.info(f"Applying tuning action: {action.parameter} = {action.new_value}")

        dispatch = {
            "sysctl": self._apply_sysctl,
            "prlimit": self._apply_prlimit,
            "command": self._apply_command,
            "env_var": self._apply_env_var,
        }

        handler = dispatch.get(action.config_type)
        if handler is None:
            logger.warning(f"No handler for config_type '{action.config_type}'")
            return False

        try:
            return handler(pid, action)
        except Exception as e:
            logger.error(f"Failed to apply action {action.parameter}: {e}")
            return False

    def _apply_sysctl(self, pid: int | None, action: TuningAction) -> bool:
        param = action.parameter
        value = str(action.new_value)
        result = subprocess.run(
            ["sudo", "sysctl", "-w", f"{param}={value}"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            logger.error(f"sysctl failed: {result.stderr}")
            return False
        return True

    def _apply_prlimit(self, pid: int | None, action: TuningAction) -> bool:
        if pid is None:
            logger.error("prlimit action requires target PID")
            return False

        resource = action.parameter
        value = str(action.new_value)
        result = subprocess.run(
            ["sudo", "prlimit", f"--pid={pid}", f"--{resource}={value}:{value}"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            logger.error(f"prlimit failed: {result.stderr}")
            return False
        return True

    def _apply_command(self, pid: int | None, action: TuningAction) -> bool:
        cmd = action.new_value
        if not isinstance(cmd, str):
            logger.error("command config_type expects new_value to be a shell command string")
            return False
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            logger.error(f"Command failed: {result.stderr}")
            return False
        return True

    def _apply_env_var(self, pid: int | None, action: TuningAction) -> bool:
        logger.info(
            f"Environment variable '{action.parameter}' should be set to "
            f"'{action.new_value}' on next process restart."
        )
        return True

    def rollback(self, action: TuningAction) -> bool:
        if not action.backup_path or not action.backup_path.exists():
            logger.error("No backup available for rollback")
            return False

        logger.info(f"Rolling back: {action.parameter}")
        try:
            backup_data = json.loads(action.backup_path.read_text())
            original = backup_data.get("original_value")
            if original is None:
                logger.error("Backup does not contain original_value")
                return False

            target_pid = action.pid
            if target_pid is None:
                pid_from_backup = backup_data.get("pid")
                if isinstance(pid_from_backup, int):
                    target_pid = pid_from_backup

            if action.config_type == "prlimit" and target_pid is None:
                logger.error("Rollback for prlimit requires a target PID")
                return False

            rollback_action = TuningAction(
                config_type=action.config_type,
                parameter=action.parameter,
                current_value=action.new_value,
                new_value=original,
                reason=f"rollback of {action.parameter}",
                pid=target_pid,
            )

            success = self._apply_action(target_pid, rollback_action)
            if success:
                action.applied = False
                logger.info(f"Rollback successful: {action.parameter}")
            return success
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def get_history(self) -> list[TuningAction]:
        return self._history.copy()
