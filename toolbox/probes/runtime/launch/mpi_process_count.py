from __future__ import annotations

import re
from typing import Any

from toolbox.shared.log_artifacts import payload_text_sources
from toolbox.shared.probe_payload import emit_result, load_payload

_TEXT_PATTERNS = (
    re.compile(r"total\s+number\s+of\s+message\s+passing\s+processes\s+(\d+)", re.IGNORECASE),
    re.compile(r"using\s+(\d+)\s+mpi\s+process(?:es)?", re.IGNORECASE),
    re.compile(r"mpirun\b.*(?:-n|--np|-np)\s+(\d+)", re.IGNORECASE),
    re.compile(r"srun\b.*(?:-n|--ntasks)\s+(\d+)", re.IGNORECASE),
)
_COMMAND_FLAGS = {"-n", "--np", "-np", "--ntasks"}


def _value_from_command_parts(payload: dict[str, Any]) -> tuple[int | None, str | None, str | None]:
    command_parts = payload.get("command_parts")
    if not isinstance(command_parts, list):
        return None, None, None
    parts = [str(part).strip() for part in command_parts if str(part).strip()]
    for index, part in enumerate(parts[:-1]):
        if part in _COMMAND_FLAGS and parts[index + 1].isdigit():
            return int(parts[index + 1]), "command_parts", " ".join(parts)
    return None, None, None


def _value_from_text(payload: dict[str, Any]) -> tuple[int | None, str | None, str | None]:
    for source, text in payload_text_sources(payload):
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            for pattern in _TEXT_PATTERNS:
                match = pattern.search(line)
                if match:
                    return int(match.group(1)), source, line
    return None, None, None


def collect(payload: dict[str, Any]) -> dict[str, Any]:
    value, source, excerpt = _value_from_text(payload)
    if value is None:
        value, source, excerpt = _value_from_command_parts(payload)
    observed = value is not None
    return {
        "observed": observed,
        "source": source or "runtime_output_missing",
        "value": value,
        "excerpt": [excerpt] if excerpt else [],
        "flat": {"runtime.launch.mpi_process_count": value} if observed else {},
        "structured": {"mpi_process_count": value} if observed else {},
    }


if __name__ == "__main__":
    emit_result(collect(load_payload()))
