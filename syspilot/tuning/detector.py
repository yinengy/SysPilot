from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class DetectedConfig:
    config_type: str
    path: Path | None
    parameters: dict[str, Any]
    source: str


class ConfigDetector:
    KNOWN_CONFIG_PATTERNS = {
        "java": {
            "env_vars": ["JAVA_OPTS", "JVM_OPTS", "_JAVA_OPTIONS"],
            "file_patterns": ["*.properties", "jvm.options"],
            "cmdline_patterns": [r"-Xmx\d+[kmgKMG]?", r"-Xms\d+[kmgKMG]?", r"-XX:\w+"],
        },
        "python": {
            "env_vars": ["PYTHONPATH", "PYTHONHASHSEED", "PYTHONOPTIMIZE"],
            "file_patterns": ["*.ini", "*.cfg", "pyproject.toml"],
            "cmdline_patterns": [r"-O", r"-u", r"-B"],
        },
        "nodejs": {
            "env_vars": ["NODE_OPTIONS", "NODE_ENV", "UV_THREADPOOL_SIZE"],
            "file_patterns": ["package.json", ".npmrc"],
            "cmdline_patterns": [r"--max-old-space-size=\d+", r"--max-semi-space-size=\d+"],
        },
        "nginx": {
            "file_patterns": ["nginx.conf", "*.nginx"],
            "config_dirs": ["/etc/nginx", "/usr/local/nginx/conf"],
        },
        "mysql": {
            "file_patterns": ["my.cnf", "my.ini"],
            "config_dirs": ["/etc/mysql", "/etc/my.cnf.d"],
        },
        "postgresql": {
            "file_patterns": ["postgresql.conf", "pg_hba.conf"],
            "config_dirs": ["/etc/postgresql", "/var/lib/pgsql/data"],
        },
        "redis": {
            "file_patterns": ["redis.conf"],
            "config_dirs": ["/etc/redis"],
        },
    }
    
    def __init__(self) -> None:
        self._cache: dict[int, list[DetectedConfig]] = {}
    
    def detect_for_process(self, pid: int) -> list[DetectedConfig]:
        if pid in self._cache:
            return self._cache[pid]
        
        configs: list[DetectedConfig] = []
        
        try:
            proc = psutil.Process(pid)
            cmdline = proc.cmdline()
            environ = proc.environ()
            exe = proc.exe()
            
            app_type = self._detect_app_type(exe, cmdline)
            
            if app_type:
                configs.extend(self._detect_cmdline_config(app_type, cmdline))
                configs.extend(self._detect_env_config(app_type, environ))
                configs.extend(self._detect_file_config(app_type, proc.cwd()))
            
            configs.extend(self._detect_system_limits(pid))
            
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning(f"Cannot detect config for PID {pid}: {e}")
        
        self._cache[pid] = configs
        return configs
    
    def _detect_app_type(self, exe: str, cmdline: list[str]) -> str | None:
        exe_lower = exe.lower()
        cmdline_str = " ".join(cmdline).lower()
        
        if "java" in exe_lower or "java" in cmdline_str:
            return "java"
        if "python" in exe_lower:
            return "python"
        if "node" in exe_lower:
            return "nodejs"
        if "nginx" in exe_lower:
            return "nginx"
        if "mysql" in exe_lower or "mysqld" in exe_lower:
            return "mysql"
        if "postgres" in exe_lower:
            return "postgresql"
        if "redis" in exe_lower:
            return "redis"
        
        return None
    
    def _detect_cmdline_config(self, app_type: str, cmdline: list[str]) -> list[DetectedConfig]:
        configs = []
        patterns = self.KNOWN_CONFIG_PATTERNS.get(app_type, {}).get("cmdline_patterns", [])
        
        cmdline_str = " ".join(cmdline)
        params = {}
        
        for pattern in patterns:
            matches = re.findall(pattern, cmdline_str)
            for match in matches:
                params[match] = True
        
        if params:
            configs.append(DetectedConfig(
                config_type=f"{app_type}_cmdline",
                path=None,
                parameters=params,
                source="command_line",
            ))
        
        return configs
    
    def _detect_env_config(self, app_type: str, environ: dict[str, str]) -> list[DetectedConfig]:
        configs = []
        env_vars = self.KNOWN_CONFIG_PATTERNS.get(app_type, {}).get("env_vars", [])
        
        params = {}
        for var in env_vars:
            if var in environ:
                params[var] = environ[var]
        
        if params:
            configs.append(DetectedConfig(
                config_type=f"{app_type}_env",
                path=None,
                parameters=params,
                source="environment",
            ))
        
        return configs
    
    def _detect_file_config(self, app_type: str, cwd: str) -> list[DetectedConfig]:
        configs = []
        patterns = self.KNOWN_CONFIG_PATTERNS.get(app_type, {})
        
        search_dirs = [Path(cwd)]
        search_dirs.extend([Path(d) for d in patterns.get("config_dirs", [])])
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            
            for file_pattern in patterns.get("file_patterns", []):
                for config_file in search_dir.glob(file_pattern):
                    if config_file.is_file():
                        configs.append(DetectedConfig(
                            config_type=f"{app_type}_file",
                            path=config_file,
                            parameters={"file": str(config_file)},
                            source="config_file",
                        ))
        
        return configs
    
    def _detect_system_limits(self, pid: int) -> list[DetectedConfig]:
        configs = []
        limits_path = Path(f"/proc/{pid}/limits")
        
        if limits_path.exists():
            try:
                content = limits_path.read_text()
                params = {}
                
                for line in content.strip().split("\n")[1:]:
                    parts = line.split()
                    if len(parts) >= 4:
                        limit_name = " ".join(parts[:-3])
                        soft = parts[-3]
                        hard = parts[-2]
                        params[limit_name] = {"soft": soft, "hard": hard}
                
                configs.append(DetectedConfig(
                    config_type="system_limits",
                    path=limits_path,
                    parameters=params,
                    source="proc_limits",
                ))
            except Exception as e:
                logger.debug(f"Cannot read limits for PID {pid}: {e}")
        
        return configs
    
    def clear_cache(self, pid: int | None = None) -> None:
        if pid:
            self._cache.pop(pid, None)
        else:
            self._cache.clear()
