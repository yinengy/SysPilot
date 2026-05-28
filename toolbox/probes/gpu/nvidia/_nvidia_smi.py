from __future__ import annotations

import csv
import re
import subprocess
from dataclasses import dataclass

NVIDIA_SMI_TIMEOUT_S = 2.0
MIB = 1024 * 1024


@dataclass(frozen=True)
class NvidiaSmiOutput:
    observed: bool
    source: str
    stdout: str
    error: str = ""


def numeric_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text in {"-", "N/A", "[Not Supported]"}:
        return None
    match = re.search(r"([-+]?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    return float(match.group(1))


def numeric_int(value: object) -> int | None:
    number = numeric_float(value)
    if number is None:
        return None
    return int(number)


def target_pids(payload: dict[str, object]) -> set[int]:
    raw_pids_obj = payload.get("target_pids", [])
    raw_pids: list[object] = raw_pids_obj if isinstance(raw_pids_obj, list) else []
    primary_pid = payload.get("primary_pid")
    candidates: list[object] = [*raw_pids]
    if primary_pid is not None:
        candidates.append(primary_pid)
    pids: set[int] = set()
    for candidate in candidates:
        pid = numeric_int(candidate)
        if pid is not None:
            pids.add(pid)
    return pids


def run_nvidia_smi(args: list[str]) -> NvidiaSmiOutput:
    try:
        completed = subprocess.run(
            ["nvidia-smi", *args],
            capture_output=True,
            check=False,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_S,
        )
    except FileNotFoundError:
        return NvidiaSmiOutput(False, "os_missing", "", "nvidia-smi not found")
    except subprocess.TimeoutExpired:
        return NvidiaSmiOutput(False, "timeout", "", "nvidia-smi timed out")

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        return NvidiaSmiOutput(
            False,
            "nvidia_smi_error",
            stdout,
            stderr or f"nvidia-smi exited with code {completed.returncode}",
        )
    return NvidiaSmiOutput(bool(stdout), "nvidia_smi", stdout, stderr)


def parse_csv_rows(stdout: str, fields: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parsed: list[str] = list(next(csv.reader([line], skipinitialspace=True), []))
        if len(parsed) < len(fields):
            continue
        rows.append({field: parsed[index].strip() for index, field in enumerate(fields)})
    return rows


def query_gpu(fields: list[str]) -> tuple[NvidiaSmiOutput, list[dict[str, str]]]:
    result = run_nvidia_smi(
        [
            "--query-gpu=" + ",".join(fields),
            "--format=csv,noheader,nounits",
        ]
    )
    return result, parse_csv_rows(result.stdout, fields) if result.observed else []


def query_compute_apps(fields: list[str]) -> tuple[NvidiaSmiOutput, list[dict[str, str]]]:
    result = run_nvidia_smi(
        [
            "--query-compute-apps=" + ",".join(fields),
            "--format=csv,noheader,nounits",
        ]
    )
    return result, parse_csv_rows(result.stdout, fields) if result.observed else []


def parse_pmon_rows(stdout: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        gpu = numeric_int(parts[0])
        pid = numeric_int(parts[1])
        sm = numeric_float(parts[3])
        mem = numeric_float(parts[4])
        if gpu is None or pid is None:
            continue
        rows.append(
            {
                "gpu": gpu,
                "pid": pid,
                "type": parts[2],
                "sm_utilization_pct": sm,
                "memory_utilization_pct": mem,
                "command": " ".join(parts[7:]) if len(parts) > 7 else parts[6],
            }
        )
    return rows


def query_pmon() -> tuple[NvidiaSmiOutput, list[dict[str, object]]]:
    result = run_nvidia_smi(["pmon", "-c", "1", "-s", "um"])
    return result, parse_pmon_rows(result.stdout) if result.observed else []


def unavailable_result(virtual_probe: str, output: NvidiaSmiOutput) -> dict[str, object]:
    return {
        "observed": False,
        "source": output.source,
        "value": None,
        "flat": {},
        "structured": {
            "schema_version": "gpu_probe/v1",
            "virtual_probe": virtual_probe,
            "error": output.error,
        },
        "error": output.error,
    }
