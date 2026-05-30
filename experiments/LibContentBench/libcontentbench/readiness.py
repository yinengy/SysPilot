from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any

from .common import write_json


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def has_command(name: str) -> bool:
    return shutil.which(name) is not None


def write_readiness(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)
