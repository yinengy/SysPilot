from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any


_VALID_PROBE_SEGMENT = re.compile(r"^[A-Za-z0-9_]+$")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def toolbox_root() -> Path:
    return repo_root() / "toolbox"


def probes_root() -> Path:
    return toolbox_root() / "probes"


def probe_script_path(virtual_probe: str, toolbox: Path | None = None) -> Path | None:
    probe = str(virtual_probe).strip()
    if not probe:
        return None
    parts = probe.split(".")
    if not parts or any(not _VALID_PROBE_SEGMENT.fullmatch(part) for part in parts):
        return None
    root = toolbox if toolbox is not None else probes_root()
    return root.joinpath(*parts).with_suffix(".py")


def resolve_toolbox_probe(virtual_probe: str, toolbox: Path | None = None) -> dict[str, object]:
    probe = str(virtual_probe).strip()
    script_path = probe_script_path(probe, toolbox)
    if script_path is None:
        return {
            "virtual_probe": probe,
            "availability": "unavailable",
            "collector": "unavailable",
            "requirements": ["virtual_probe name must map to toolbox/probes/<probe>.py"],
        }

    base_root = toolbox_root()
    if script_path.exists():
        try:
            relative = script_path.relative_to(repo_root())
        except ValueError:
            relative = script_path
        return {
            "virtual_probe": probe,
            "availability": "available",
            "collector": "toolbox_script",
            "requirements": [f"toolbox script {relative}"],
            "script_path": str(script_path),
        }

    try:
        relative_missing = script_path.relative_to(repo_root())
    except ValueError:
        relative_missing = script_path
    return {
        "virtual_probe": probe,
        "availability": "unavailable",
        "collector": "unavailable",
        "requirements": [f"missing toolbox script {relative_missing}"],
        "script_path": str(script_path),
        "toolbox_root": str(base_root),
    }


def missing_toolbox_probe_issues(
    virtual_probes: list[tuple[str, str]],
    toolbox: Path | None = None,
) -> list[str]:
    issues: list[str] = []
    for signal_id, virtual_probe in virtual_probes:
        resolved = resolve_toolbox_probe(virtual_probe, toolbox)
        if resolved.get("availability") == "available":
            continue
        requirements = resolved.get("requirements", [])
        message = requirements[0] if isinstance(requirements, list) and requirements else "missing toolbox script"
        issues.append(f"signals:{signal_id} virtual_probe '{virtual_probe}' is unavailable: {message}")
    return issues


def extract_declared_virtual_probes(
    hypothesis_profiles: list[dict[str, Any]],
    *,
    memory_high: str | None = None,
) -> list[str]:
    probes: list[str] = []
    for profile in hypothesis_profiles:
        signals = profile.get("validation_signals", []) if isinstance(profile, dict) else []
        for entry in signals:
            if not isinstance(entry, dict):
                continue
            probe = entry.get("virtual_probe")
            if not isinstance(probe, str) or not probe.strip():
                signal = entry.get("signal", {})
                if isinstance(signal, dict):
                    candidate = signal.get("virtual_probe")
                    if isinstance(candidate, str):
                        probe = candidate
            if isinstance(probe, str) and probe.strip():
                probes.append(probe.strip())
    if memory_high:
        probes.append("cgroup.memory.events.high")
    return list(dict.fromkeys(probes))


@dataclass
class ToolboxProbeSession:
    virtual_probe: str
    resolved_probe: dict[str, object]
    module: ModuleType | None
    state: Any = None


def _load_probe_module(virtual_probe: str, script_path: Path) -> ModuleType:
    module_name = "syspilot_toolbox_" + virtual_probe.replace(".", "_")
    spec = spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load toolbox probe module for {virtual_probe}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _normalize_probe_result(
    virtual_probe: str,
    resolved_probe: dict[str, object],
    raw_result: Any,
) -> dict[str, Any]:
    if not isinstance(raw_result, dict):
        raw_result = {
            "observed": False,
            "source": "probe_error",
            "error": f"toolbox probe {virtual_probe} returned non-object result",
            "flat": {},
            "structured": {},
        }

    flat = raw_result.get("flat", {})
    structured = raw_result.get("structured", {})
    excerpt = raw_result.get("excerpt", [])
    value = raw_result.get("value")

    if not isinstance(flat, dict):
        flat = {}
    if not isinstance(structured, dict):
        structured = {}
    if isinstance(excerpt, str):
        excerpt = [excerpt]
    elif not isinstance(excerpt, list):
        excerpt = []

    observed = raw_result.get("observed")
    if not isinstance(observed, bool):
        observed = bool(flat or structured or value is not None or excerpt)

    source = raw_result.get("source")
    if not isinstance(source, str) or not source.strip():
        source = str(resolved_probe.get("collector", "toolbox_script"))

    normalized: dict[str, Any] = {
        "virtual_probe": virtual_probe,
        "observed": observed,
        "source": source,
        "value": value,
        "excerpt": excerpt,
        "flat": flat,
        "structured": structured,
        "resolved_probe": resolved_probe,
    }
    error = raw_result.get("error")
    if isinstance(error, str) and error.strip():
        normalized["error"] = error.strip()
    return normalized


async def start_toolbox_probe_sessions(
    virtual_probes: list[str],
    payload: dict[str, Any],
    toolbox: Path | None = None,
) -> dict[str, ToolboxProbeSession]:
    sessions: dict[str, ToolboxProbeSession] = {}
    for probe in dict.fromkeys(virtual_probes):
        resolved = resolve_toolbox_probe(probe, toolbox)
        if resolved.get("availability") != "available":
            sessions[probe] = ToolboxProbeSession(probe, resolved, None)
            continue

        script_path_raw = resolved.get("script_path")
        if not isinstance(script_path_raw, str):
            sessions[probe] = ToolboxProbeSession(probe, resolved, None)
            continue

        script_path = Path(script_path_raw)
        try:
            module = _load_probe_module(probe, script_path)
            begin = getattr(module, "begin", None)
            state = await _maybe_await(begin(dict(payload))) if callable(begin) else None
            sessions[probe] = ToolboxProbeSession(probe, resolved, module, state)
        except Exception as exc:
            sessions[probe] = ToolboxProbeSession(
                probe,
                {
                    **resolved,
                    "availability": "unavailable",
                    "collector": "probe_error",
                    "requirements": [str(exc)],
                },
                None,
            )
    return sessions


async def finish_toolbox_probe_sessions(
    sessions: dict[str, ToolboxProbeSession],
    payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for probe, session in sessions.items():
        if session.module is None or session.resolved_probe.get("availability") != "available":
            requirements = session.resolved_probe.get("requirements", [])
            requirement_list = requirements if isinstance(requirements, list) else [requirements]
            results[probe] = _normalize_probe_result(
                probe,
                session.resolved_probe,
                {
                    "observed": False,
                    "source": "unavailable",
                    "flat": {},
                    "structured": {},
                    "error": "; ".join(str(item) for item in requirement_list),
                },
            )
            continue

        try:
            finish = getattr(session.module, "finish", None)
            collect = getattr(session.module, "collect", None)
            if callable(finish):
                raw = await _maybe_await(finish(dict(payload), session.state))
            elif callable(collect):
                raw = await _maybe_await(collect(dict(payload)))
            else:
                raise RuntimeError("toolbox probe script must define collect() or finish()")
        except Exception as exc:
            raw = {
                "observed": False,
                "source": "probe_error",
                "flat": {},
                "structured": {},
                "error": str(exc),
            }
        results[probe] = _normalize_probe_result(probe, session.resolved_probe, raw)
    return results
