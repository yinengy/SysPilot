from __future__ import annotations

import asyncio
import difflib
import json
import logging
import os
import re
import shlex
import subprocess
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol, cast

import click
import psutil
import yaml
from rich.console import Console
from rich.markup import escape as rich_escape
from rich.table import Table

from syspilot import __version__
from syspilot.core.agent import SysPilotAgent
from syspilot.core.config import LLMConfig, SysPilotConfig
from syspilot.core.llm import LLMError, LLMInterface, Message
from syspilot.core.openai_auth import (
    build_authorization_url,
    create_pkce_verifier_and_challenge,
    create_state,
    exchange_authorization_code,
    parse_authorization_input,
    save_auth,
    syspilot_openai_auth_file,
    wait_for_callback_code,
)
from syspilot.fixed_budget import FixedBudgetState
from syspilot.fixed_goal import fixed_goal_payload, fixed_goal_reached
from syspilot.profiling import ProfilingManager
from syspilot.prompt_budget import (
    CONTRACT_YAML_CONTEXT_KEY,
    hypothesis_profiles_yaml_for_prompt,
    profiling_json_for_prompt,
    truncate_prompt_text,
)
from syspilot.quick_mode import QuickModeDefaults
from syspilot.skills import SkillRegistry
from syspilot.skills.builtin import (
    DataProcessorSkill,
    DynamicScriptSkill,
    PerfAnalyzerSkill,
    StraceAnalyzerSkill,
)
from syspilot.spawn_runtime import execute_profiled_spawn_run, execute_unprofiled_spawn_run
from syspilot.study_trace import (
    NestedCpuBlasStudyTrace,
    RepositoryStudySourceUnavailableError,
    StudyProposal,
)
from syspilot.toolbox_runtime import (
    finish_toolbox_probe_sessions,
    sample_toolbox_probe_sessions,
    start_toolbox_probe_sessions,
)
from syspilot.tuning import ApplicationTuner
from syspilot.ui.emitter import TuneEventEmitter
from syspilot.ui.events import is_critical_event
from syspilot.ui.output_sink import OutputSink, SwitchableOutputSink, TuneUIMode, make_output_sink
from syspilot.ui.tui_sidecar import TuneTuiOrchestrator, should_wait_for_user_on_close
from syspilot.utils import setup_logging

console = Console()
logger = logging.getLogger(__name__)
DEFAULT_QUICK_MODE_DEFAULTS = QuickModeDefaults()

METRIC_EXTRACTION_SYSTEM_PROMPT = (
    "You extract numeric workload performance metrics from program output.\n"
    "Return exactly one flat JSON object and no explanation.\n"
    "Include only measured performance outcomes: throughput/rate, latency percentiles, "
    "runtime/duration, or bandwidth.\n"
    "Exclude configuration and static metadata such as thread counts, batch sizes, token "
    "counts, topology, hardware details, model size, or parameter counts.\n"
    'If no measured performance metrics are present, return {"no_metrics": true}.'
)


def _enable_llm_debug_options(
    config: SysPilotConfig,
    *,
    debug_llm: bool,
    debug_llm_md: str | None,
) -> None:
    if debug_llm or debug_llm_md:
        config.llm.debug_conversation = True
    if debug_llm_md:
        config.llm.debug_markdown_file = debug_llm_md


async def _agent_step_for_prompt(agent: Any, prompt: str) -> Any:
    stateless_step = getattr(agent, "step_stateless", None)
    if callable(stateless_step):
        typed_step = cast(Callable[..., Awaitable[Any]], stateless_step)
        return await typed_step(prompt, include_tools=False)
    return await agent.step(prompt)


def _default_llm_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "llm_config.json"


def _load_default_llm_config() -> LLMConfig:
    default_path = _default_llm_config_path()
    if default_path.exists():
        return LLMConfig.from_file(default_path)
    return LLMConfig.from_file()


def _merge_llm_config_with_repo_defaults(data: dict[str, Any]) -> LLMConfig:
    repo_llm = _load_default_llm_config()
    llm_data = data.get("llm")
    if not isinstance(llm_data, dict):
        return repo_llm

    merged = {
        "provider": repo_llm.provider.value,
        "model": repo_llm.model,
        "metric_extraction_model": repo_llm.metric_extraction_model,
        "api_key": repo_llm.api_key,
        "base_url": repo_llm.base_url,
        "openai_auth_file": repo_llm.openai_auth_file,
        "temperature": repo_llm.temperature,
        "max_tokens": repo_llm.max_tokens,
        "timeout_seconds": repo_llm.timeout_seconds,
        "max_retries": repo_llm.max_retries,
        "retry_cooldown_seconds": repo_llm.retry_cooldown_seconds,
        "debug_conversation": repo_llm.debug_conversation,
        "debug_markdown_file": repo_llm.debug_markdown_file,
        "debug_markdown_append": repo_llm.debug_markdown_append,
    }
    merged.update(llm_data)
    return LLMConfig.from_dict(merged)


def _load_cli_config(config_path: str | None) -> SysPilotConfig:
    if not config_path:
        sys_config = SysPilotConfig()
        sys_config.llm = _load_default_llm_config()
        return sys_config

    path = Path(config_path)
    with open(path) as f:
        raw_data = yaml.safe_load(f) or {}

    if not isinstance(raw_data, dict):
        raise ValueError(f"Config file must contain a mapping at top level: {path}")

    config = SysPilotConfig.from_dict(raw_data)
    config.llm = _merge_llm_config_with_repo_defaults(raw_data)
    return config


def _manual_pipeline_llm_config(base: LLMConfig, minimum_timeout_seconds: float) -> LLMConfig:
    return replace(
        base,
        timeout_seconds=max(base.timeout_seconds, minimum_timeout_seconds),
        max_retries=max(base.max_retries, 1),
    )


def _metric_extraction_llm_config(base: LLMConfig) -> LLMConfig:
    metric_model = (base.metric_extraction_model or "").strip()
    if not metric_model:
        return replace(base, debug_markdown_append=True)
    return replace(base, model=metric_model, debug_markdown_append=True)


class _TuneEmitter(Protocol):
    def publish(
        self,
        event_name: str,
        payload: dict[str, object] | None = None,
        *,
        event_type: str | None = None,
    ) -> bool: ...


def _is_env_flag_enabled(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _register_default_skills(agent: SysPilotAgent) -> None:
    registry = SkillRegistry()

    for skill_cls in [
        DynamicScriptSkill,
        StraceAnalyzerSkill,
        PerfAnalyzerSkill,
        DataProcessorSkill,
    ]:
        skill = skill_cls()
        registry.register(skill)

        async def skill_executor(skill_instance: Any = skill, **kwargs: Any) -> dict[str, Any]:
            result = await skill_instance.execute(**kwargs)
            return {
                "success": result.success,
                "output": result.output,
                "summary": result.summary,
                "error": result.error,
                "data": result.data,
            }

        agent.register_tool(
            name=f"skill_{skill.name}",
            func=skill_executor,
            description=skill.description,
            parameters=skill.get_parameters_schema(),
        )


def _register_profiling_tools(agent: SysPilotAgent, manager: ProfilingManager) -> None:
    async def collect_profile(
        collectors: list[str] | None = None,
        pid: int | None = None,
    ) -> dict[str, Any]:
        results = await manager.collect(collectors, pid)
        return {
            name: {
                "data": result.data,
                "summary": result.summarize(),
            }
            for name, result in results.items()
        }

    agent.register_tool(
        name="collect_profile",
        func=collect_profile,
        description="Collect system profiling data using available collectors",
        parameters={
            "type": "object",
            "properties": {
                "collectors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"Collectors to use: {manager.get_available_collectors()}",
                },
                "pid": {
                    "type": "integer",
                    "description": "Target process ID (optional for system-wide)",
                },
            },
        },
    )


def _register_tuning_tools(agent: SysPilotAgent, tuner: ApplicationTuner) -> None:
    def analyze_config(pid: int) -> dict[str, Any]:
        return tuner.analyze(pid)

    agent.register_tool(
        name="analyze_config",
        func=analyze_config,
        description="Analyze application configuration and get tuning recommendations",
        parameters={
            "type": "object",
            "properties": {
                "pid": {"type": "integer", "description": "Target process ID"},
            },
            "required": ["pid"],
        },
    )

    def apply_tuning(pid: int, actions: list[dict[str, Any]]) -> dict[str, Any]:
        results = tuner.apply_tuning(pid, actions)
        return {
            "applied": [
                {
                    "parameter": a.parameter,
                    "new_value": a.new_value,
                    "applied": a.applied,
                    "reason": a.reason,
                }
                for a in results
            ],
        }

    agent.register_tool(
        name="apply_tuning",
        func=apply_tuning,
        description=(
            "Apply tuning actions to a process. Each action must specify config_type "
            "(sysctl|prlimit|command|env_var), parameter, current_value, new_value, reason."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pid": {"type": "integer", "description": "Target process ID"},
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "config_type": {
                                "type": "string",
                                "enum": ["sysctl", "prlimit", "command", "env_var"],
                            },
                            "parameter": {"type": "string"},
                            "current_value": {},
                            "new_value": {},
                            "reason": {"type": "string"},
                        },
                        "required": ["config_type", "parameter", "new_value", "reason"],
                    },
                    "description": "List of tuning actions to apply",
                },
            },
            "required": ["pid", "actions"],
        },
    )


@click.group()
@click.version_option(version=__version__)
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file path")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--debug-llm", is_flag=True, help="Show full LLM conversation input/output")
@click.option(
    "--debug-llm-md", type=click.Path(), help="Write LLM conversation transcript to markdown file"
)
@click.pass_context
def cli(
    ctx: click.Context,
    config: str | None,
    log_level: str,
    debug_llm: bool,
    debug_llm_md: str | None,
) -> None:
    ctx.ensure_object(dict)

    ctx.obj["config"] = _load_cli_config(config)

    ctx.obj["config"].log_level = log_level
    _enable_llm_debug_options(
        ctx.obj["config"],
        debug_llm=debug_llm,
        debug_llm_md=debug_llm_md,
    )
    setup_logging(log_level)


@cli.group()
def auth() -> None:
    pass


@auth.command("login")
@click.option(
    "--auto/--manual",
    "auto",
    default=False,
    show_default=True,
    help="Use automatic browser callback flow (default: manual URL paste)",
)
@click.option(
    "--timeout", default=180, type=int, show_default=True, help="OAuth callback timeout seconds"
)
@click.option("--auth-file", type=click.Path(), help="Path to save OpenAI auth credentials")
def auth_login(auto: bool, timeout: int, auth_file: str | None) -> None:
    verifier, challenge = create_pkce_verifier_and_challenge()
    state = create_state()
    auth_url = build_authorization_url(state, challenge)

    code: str | None = None
    if auto:
        code, opened = wait_for_callback_code(
            state=state,
            timeout_seconds=timeout,
            authorize_url=auth_url,
        )
        if opened:
            console.print("[cyan]Browser opened for OpenAI login. Waiting for callback...[/cyan]")
        else:
            console.print("[yellow]Could not open browser automatically.[/yellow]")

        if not code:
            console.print(
                "[yellow]Automatic callback unavailable or timed out. Using manual mode.[/yellow]"
            )

    if not code:
        console.print("[cyan]Open this URL and complete login:[/cyan]")
        console.print(auth_url)
        pasted = click.prompt("Paste full redirect URL (or authorization code)")
        parsed_code, parsed_state = parse_authorization_input(pasted)

        if not parsed_code:
            console.print("[red]Could not parse authorization code from input.[/red]")
            raise SystemExit(1)
        if parsed_state and parsed_state != state:
            console.print("[red]State mismatch in OAuth callback.[/red]")
            raise SystemExit(1)

        code = parsed_code

    try:
        auth_data = exchange_authorization_code(code, verifier)
    except Exception as exc:
        console.print(f"[red]OpenAI login failed: {exc}[/red]")
        raise SystemExit(1)

    output_path = Path(auth_file).expanduser() if auth_file else syspilot_openai_auth_file()
    save_auth(auth_data, output_path)
    console.print(f"[green]OpenAI auth saved to {output_path}[/green]")


@cli.command()
@click.argument("pid", type=int)
@click.option("--collector", "-c", multiple=True, help="Specific collectors to use")
@click.pass_context
def profile(ctx: click.Context, pid: int, collector: tuple[str, ...]) -> None:
    manager = ProfilingManager()

    collectors = list(collector) if collector else None
    results = asyncio.run(manager.collect(collectors, pid))

    for name, result in results.items():
        table = Table(title=f"{name.upper()} Profile")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for key, value in result.data.items():
            if isinstance(value, dict):
                value = json.dumps(value, indent=2)
            table.add_row(str(key), str(value))

        console.print(table)


@cli.command()
def manuals() -> None:
    """List available tuning manuals."""
    manual_dir = _discover_manual_root()

    if not manual_dir:
        console.print(
            "[yellow]No manuals directory found (checked source tree and "
            "~/.syspilot/manual)[/yellow]"
        )
        return

    manuals_map: dict[str, Path] = {p.stem: p for p in manual_dir.glob("*.md")}
    for bundle_dir in (path for path in manual_dir.iterdir() if path.is_dir()):
        for candidate_name in ("manual.md", "manual.json"):
            candidate = bundle_dir / candidate_name
            if candidate.exists() and bundle_dir.name not in manuals_map:
                manuals_map[bundle_dir.name] = candidate
                break
    manuals_list = sorted(manuals_map)

    if not manuals_list:
        console.print("[yellow]No manuals available[/yellow]")
        return

    table = Table(title="Available Tuning Manuals")
    table.add_column("Manual", style="cyan")
    table.add_column("Path", style="green")

    for manual_name in sorted(manuals_list):
        table.add_row(manual_name, str(manuals_map[manual_name]))

    console.print(table)
    console.print(
        "\n[dim]Use --manual/-m for spawn-mode application hints or "
        "agent/manual tooling[/dim]"
    )
    console.print(
        "[dim]Example: syspilot tune --command \"./benchmark\" --contract "
        "numactl --manual \"worker pool expands after warmup\"[/dim]"
    )


def _discover_manual_root() -> Path | None:
    manual_candidates = [
        Path(__file__).resolve().parents[1] / "manual",
        Path("manual"),
        Path("~/.syspilot/manual").expanduser(),
    ]
    for path in manual_candidates:
        if path.exists() and path.is_dir():
            return path
    return None


def _slugify_manual_id(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or "manual"


def _resolve_manual_root(manual_root: Path | None, *, create: bool = False) -> Path:
    if manual_root is not None:
        return manual_root

    discovered = _discover_manual_root()
    if discovered is not None:
        return discovered

    fallback = Path("manual")
    if create:
        fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _discover_contract_root() -> Path | None:
    contract_candidates = [
        Path(__file__).resolve().parents[1] / "contracts",
        Path("contracts"),
        Path("~/.syspilot/contracts").expanduser(),
    ]
    for path in contract_candidates:
        if path.exists() and path.is_dir():
            return path
    return None


def _resolve_contract_root(contract_root: Path | None, *, create: bool = False) -> Path:
    if contract_root is not None:
        return contract_root

    discovered = _discover_contract_root()
    if discovered is not None:
        return discovered

    fallback = Path("contracts")
    if create:
        fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _contract_prompt_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "syspilot_contract" / "prompts"


def _resolve_contract_path(contract_ref: str, contract_root: Path | None = None) -> Path:
    candidate = Path(contract_ref).expanduser()
    if candidate.exists():
        return candidate
    root = _resolve_contract_root(contract_root)
    direct = root / contract_ref / "contract.yaml"
    if direct.exists():
        return direct
    normalized_ref = _slugify_manual_id(contract_ref)
    normalized = root / normalized_ref / "contract.yaml"
    if normalized.exists():
        return normalized
    return direct


def _has_structured_context(bundle: dict[str, object]) -> bool:
    return any(
        bundle.get(key) for key in ("profile_first", "candidate_actions", "conflicts", "validation")
    )


def _split_escaped_csv_values(value: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    escaping = False
    for char in value:
        if escaping:
            current.append(char)
            escaping = False
            continue
        if char == "\\":
            escaping = True
            continue
        if char == ",":
            item = "".join(current).strip()
            if item:
                parts.append(item)
            current = []
            continue
        current.append(char)

    if escaping:
        current.append("\\")
    item = "".join(current).strip()
    if item:
        parts.append(item)
    return parts


def _normalize_multi_value_option(values: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        for item in _split_escaped_csv_values(raw_value):
            if item in seen:
                continue
            normalized.append(item)
            seen.add(item)
    return normalized


def _load_tunable_whitelist(path: Path) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name or name in seen:
            continue
        if "|" in name:
            raise ValueError(
                f"invalid tunable whitelist entry {name!r}; use one exact control name per line"
            )
        names.append(name)
        seen.add(name)
    return names


def _looks_like_inline_manual_hint(value: str) -> bool:
    return any(char.isspace() for char in value)


def _load_spawn_manual_inputs(
    agent: SysPilotAgent,
    manual_inputs: Sequence[str],
) -> tuple[list[str], list[str], list[str]]:
    loaded_manuals: list[str] = []
    inline_hints: list[str] = []
    missing_manuals: list[str] = []

    for manual_input in manual_inputs:
        if _looks_like_inline_manual_hint(manual_input):
            inline_hints.append(manual_input)
            continue

        load_results = agent.load_manuals([manual_input])
        if load_results.get(manual_input):
            loaded_manuals.append(manual_input)
        else:
            missing_manuals.append(manual_input)
            inline_hints.append(manual_input)

    return loaded_manuals, inline_hints, missing_manuals


def _render_excluded_config_names(names: Sequence[str]) -> str:
    if not names:
        return "(none)"
    joined = ", ".join(names)
    return (
        f"Keep exactly these controls unchanged: {joined}. Only these exact listed "
        "environment variables, command-line parameters, or config controls are immutable; "
        "related controls with different names remain tunable when supported by evidence."
    )


def _render_tunable_whitelist_names(names: Sequence[str]) -> str:
    if not names:
        return (
            "A tunable whitelist file was provided, but it contained no controls. "
            "Return no tuning actions."
        )
    joined = ", ".join(names)
    return (
        f"Change only these exact tunable controls: {joined}. Environment variables, "
        "command-line parameters, command-prefix parameters, or config controls not listed here "
        "must remain unchanged even when evidence suggests they may help."
    )


def _render_loaded_manual_context(loaded_manuals: dict[str, str]) -> str:
    if not loaded_manuals:
        return ""
    parts: list[str] = []
    for manual_name, manual_content in loaded_manuals.items():
        parts.append(f"### Manual: {manual_name}\n{manual_content.strip()}")
    return "\n\n".join(parts)


def _build_structured_context_bundle(
    manuals: list[str],
    *,
    goal: str | None,
    symptom: str | None,
    max_items_per_section: int,
    manual_root: Path | None,
    contract_root: Path | None,
) -> dict[str, object]:
    if contract_root is not None:
        try:
            from syspilot_contract.integration import (
                build_agent_context_bundle as build_contract_context_bundle,
            )

            contract_bundle = build_contract_context_bundle(
                manuals,
                goal=goal,
                symptom=symptom,
                max_items_per_section=max_items_per_section,
                contract_root=contract_root,
            )
            if _has_structured_context(contract_bundle):
                return contract_bundle
        except KeyError:
            pass
        except Exception as exc:
            logger.debug("Structured contract context unavailable: %s", exc)

    if manual_root is not None:
        try:
            from syspilot_manual.integration import (
                build_agent_context_bundle as build_manual_context_bundle,
            )

            manual_bundle = build_manual_context_bundle(
                manuals,
                goal=goal,
                symptom=symptom,
                max_items_per_section=max_items_per_section,
                manual_root=manual_root,
            )
            if _has_structured_context(manual_bundle):
                return manual_bundle
        except Exception as exc:
            logger.debug("Structured manual context unavailable: %s", exc)

    return {}


def _build_contract_context_bundle(
    contracts: list[str],
    *,
    goal: str | None,
    symptom: str | None,
    max_items_per_section: int,
    contract_root: Path | None,
) -> dict[str, object]:
    if contract_root is None:
        return {}
    try:
        from syspilot_contract.integration import (
            build_agent_context_bundle as build_contract_context_bundle,
        )

        bundle = build_contract_context_bundle(
            contracts,
            goal=goal,
            symptom=symptom,
            max_items_per_section=max_items_per_section,
            contract_root=contract_root,
        )
        yaml_context = _render_contract_yaml_context(contracts, contract_root)
        if yaml_context:
            bundle = {**bundle, CONTRACT_YAML_CONTEXT_KEY: yaml_context}
        return bundle if _has_structured_context(bundle) else {}
    except KeyError:
        return {}
    except Exception as exc:
        logger.debug("Structured contract context unavailable: %s", exc)
        return {}


def _contract_yaml_paths_for_prompt(
    contracts: Sequence[str],
    contract_root: Path,
) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()

    def add_path(path: Path) -> None:
        resolved = path.resolve() if path.exists() else path
        if resolved in seen or not path.exists():
            return
        seen.add(resolved)
        paths.append(path)

    for contract_ref in contracts:
        path = _resolve_contract_path(contract_ref, contract_root)
        add_path(path)
        if not path.exists():
            continue
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if not isinstance(payload, dict) or payload.get("type") != "composed":
            continue
        raw_instances = payload.get("contract_instances", [])
        if not isinstance(raw_instances, list):
            continue
        for item in raw_instances:
            if not isinstance(item, dict):
                continue
            contract_id = item.get("contract_id")
            if isinstance(contract_id, str) and contract_id.strip():
                add_path(_resolve_contract_path(contract_id, contract_root))

    return paths


def _strip_contract_prompt_evidence(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _strip_contract_prompt_evidence(item)
            for key, item in value.items()
            if str(key) != "evidence"
        }
    if isinstance(value, list):
        return [_strip_contract_prompt_evidence(item) for item in value]
    return value


def _contract_yaml_text_for_prompt(path: Path) -> str:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    stripped_payload = _strip_contract_prompt_evidence(payload)
    return yaml.safe_dump(
        stripped_payload,
        sort_keys=False,
        allow_unicode=True,
    ).strip()


def _render_contract_yaml_context(
    contracts: Sequence[str],
    contract_root: Path | None,
) -> str:
    if contract_root is None or not contracts:
        return ""
    parts: list[str] = []
    for path in _contract_yaml_paths_for_prompt(contracts, contract_root):
        if not path.exists():
            continue
        yaml_text = _contract_yaml_text_for_prompt(path)
        if not yaml_text:
            continue
        label = path.parent.name
        parts.append(f"### Contract YAML: {label}\n```yaml\n{yaml_text}\n```")
    return truncate_prompt_text(
        "\n\n".join(parts),
        60_000,
        label="contract YAML context",
    )


def _default_tune_prompt_log_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("artifacts") / "tune_llm_prompts" / f"{stamp}.jsonl"


@cli.command("ingest")
@click.option("--library", required=True, help="Library name or manual ID")
@click.option(
    "--docs",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Docs file or directory",
)
@click.option(
    "--code",
    type=click.Path(exists=True, path_type=Path),
    help="Optional code file or directory for lightweight hints",
)
@click.option(
    "--metadata",
    type=click.Path(exists=True, path_type=Path),
    help="Optional metadata file (json/yaml/txt)",
)
@click.option(
    "--artifact-root",
    type=click.Path(path_type=Path),
    default=Path("artifacts"),
    show_default=True,
    help="Artifact output root",
)
def ingest(
    library: str, docs: Path, code: Path | None, metadata: Path | None, artifact_root: Path
) -> None:
    from syspilot_manual.integration import ingest_library_inputs

    chunks = ingest_library_inputs(
        library_name=library,
        docs_path=docs,
        artifact_root=artifact_root,
        code_path=code,
        metadata_path=metadata,
    )
    click.echo(
        json.dumps(
            {
                "library": library,
                "artifact_root": str(artifact_root),
                "chunks_written": len(chunks),
                "chunk_manifest": str(artifact_root / "chunks" / library / "chunks.jsonl"),
            },
            indent=2,
        )
    )


@cli.command("extract-evidence")
@click.option("--library", required=True, help="Library name used during ingest")
@click.option(
    "--artifact-root",
    type=click.Path(path_type=Path),
    default=Path("artifacts"),
    show_default=True,
    help="Artifact root containing chunks",
)
@click.pass_context
def extract_evidence_cmd(ctx: click.Context, library: str, artifact_root: Path) -> None:
    from syspilot_manual.integration import extract_library_evidence
    from syspilot_manual.llm_client import ManualLLMClient

    llm_client = ManualLLMClient(
        config=_manual_pipeline_llm_config(ctx.obj["config"].llm, minimum_timeout_seconds=90.0)
    )
    evidence = asyncio.run(
        extract_library_evidence(
            library,
            artifact_root,
            llm_client,
            progress_callback=_manual_extract_progress_callback,
        )
    )
    click.echo(
        json.dumps(
            {
                "library": library,
                "artifact_root": str(artifact_root),
                "evidence_items": len(evidence),
                "evidence_path": str(artifact_root / "evidence" / library / "evidence.jsonl"),
            },
            indent=2,
        )
    )


@cli.command("synthesize-manual")
@click.option("--library", required=True, help="Library name used during ingest")
@click.option("--manual-id", help="Override manual ID (defaults to slugified library name)")
@click.option(
    "--artifact-root",
    type=click.Path(path_type=Path),
    default=Path("artifacts"),
    show_default=True,
    help="Artifact root containing extracted evidence",
)
@click.option(
    "--manual-root",
    type=click.Path(path_type=Path),
    help="Compatibility manual root",
)
@click.pass_context
def synthesize_manual_cmd(
    ctx: click.Context,
    library: str,
    manual_id: str | None,
    artifact_root: Path,
    manual_root: Path | None,
) -> None:
    from syspilot_manual.integration import synthesize_library_bundle
    from syspilot_manual.llm_client import ManualLLMClient

    llm_client = ManualLLMClient(
        config=_manual_pipeline_llm_config(ctx.obj["config"].llm, minimum_timeout_seconds=180.0)
    )
    resolved_manual_id = manual_id or _slugify_manual_id(library)
    resolved_manual_root = _resolve_manual_root(manual_root, create=True)
    manual = asyncio.run(
        synthesize_library_bundle(
            manual_id=resolved_manual_id,
            library_name=library,
            artifact_root=artifact_root,
            llm_client=llm_client,
            manual_root=resolved_manual_root,
            progress_callback=_manual_synthesis_progress_callback,
        )
    )
    click.echo(
        json.dumps(
            {
                "manual_id": manual.manual_id,
                "manual_root": str(resolved_manual_root),
                "bundle_dir": str(resolved_manual_root / manual.manual_id),
                "compat_markdown": str(resolved_manual_root / f"{manual.manual_id}.md"),
            },
            indent=2,
        )
    )


@cli.command("synthesize-contract")
@click.option("--library", required=True, help="Library name used during ingest")
@click.option("--contract-id", help="Override contract ID (defaults to slugified library name)")
@click.option(
    "--artifact-root",
    type=click.Path(path_type=Path),
    default=Path("artifacts"),
    show_default=True,
    help="Artifact root containing extracted evidence",
)
@click.option(
    "--contract-root",
    type=click.Path(path_type=Path),
    help="Runtime contract root",
)
@click.option(
    "--include-evidence/--no-include-evidence",
    default=False,
    show_default=True,
    help="Include evidence blocks in generated LPCL output",
)
@click.pass_context
def synthesize_contract_cmd(
    ctx: click.Context,
    library: str,
    contract_id: str | None,
    artifact_root: Path,
    contract_root: Path | None,
    include_evidence: bool,
) -> None:
    from syspilot_contract.integration import synthesize_library_contract_bundle
    from syspilot_manual.llm_client import ManualLLMClient

    llm_client = ManualLLMClient(
        config=_manual_pipeline_llm_config(ctx.obj["config"].llm, minimum_timeout_seconds=420.0),
        prompt_dir=_contract_prompt_dir(),
    )
    resolved_contract_id = contract_id or _slugify_manual_id(library)
    resolved_contract_root = _resolve_contract_root(contract_root, create=True)
    contract = asyncio.run(
        synthesize_library_contract_bundle(
            contract_id=resolved_contract_id,
            library_name=library,
            artifact_root=artifact_root,
            llm_client=llm_client,
            contract_root=resolved_contract_root,
            include_evidence=include_evidence,
            progress_callback=_contract_synthesis_progress_callback,
        )
    )
    click.echo(
        json.dumps(
            {
                "contract_id": resolved_contract_id,
                "contract": contract.contract,
                "contract_root": str(resolved_contract_root),
                "bundle_dir": str(resolved_contract_root / resolved_contract_id),
                "contract_yaml": str(
                    resolved_contract_root / resolved_contract_id / "contract.yaml"
                ),
                "contract_lpcl": str(
                    resolved_contract_root / resolved_contract_id / "contract.lpcl"
                ),
            },
            indent=2,
        )
    )


@cli.command("compile-lpcl")
@click.option("--input", "input_path", required=True, type=click.Path(path_type=Path))
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option(
    "--include-evidence/--no-include-evidence",
    default=False,
    show_default=True,
    help="Include evidence blocks in compiled LPCL output",
)
def compile_lpcl_cmd(input_path: Path, output_path: Path, include_evidence: bool) -> None:
    from syspilot_contract.compiler import compile_lpcl_file

    try:
        compile_lpcl_file(input_path, output_path, include_evidence=include_evidence)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
            },
            indent=2,
        )
    )


@cli.command("check-contract")
@click.argument("contract_ref")
@click.option(
    "--contract-root",
    type=click.Path(path_type=Path),
    help="Optional runtime contract root for ID lookup",
)
def check_contract_cmd(
    contract_ref: str,
    contract_root: Path | None,
) -> None:
    from syspilot_contract.checker import validate_contract_file

    contract_path = _resolve_contract_path(contract_ref, contract_root)
    if not contract_path.exists():
        raise click.ClickException(f"Contract not found: {contract_path}")
    contract, result = validate_contract_file(contract_path)
    click.echo(
        json.dumps(
            {
                "contract_path": str(contract_path),
                "valid": result.valid,
                "issues": result.issues,
                "type": getattr(contract, "contract_type", None),
                "contract": getattr(contract, "contract", None),
            },
            indent=2,
        )
    )
    if not result.valid:
        raise SystemExit(1)


@cli.command("render-manual")
@click.option("--manual", "manual_id", required=True, help="Manual ID to render")
@click.option(
    "--artifact-root",
    type=click.Path(path_type=Path),
    default=Path("artifacts"),
    show_default=True,
)
@click.option(
    "--manual-root",
    type=click.Path(path_type=Path),
)
def render_manual_cmd(manual_id: str, artifact_root: Path, manual_root: Path | None) -> None:
    from syspilot_manual.integration import render_existing_manual

    bundle_dir = render_existing_manual(manual_id, artifact_root, _resolve_manual_root(manual_root))
    click.echo(json.dumps({"manual_id": manual_id, "bundle_dir": str(bundle_dir)}, indent=2))


@cli.command("build-index")
@click.option("--manual", "manual_id", required=True, help="Manual ID to index")
@click.option(
    "--artifact-root",
    type=click.Path(path_type=Path),
    default=Path("artifacts"),
    show_default=True,
)
@click.option(
    "--manual-root",
    type=click.Path(path_type=Path),
)
def build_index_cmd(manual_id: str, artifact_root: Path, manual_root: Path | None) -> None:
    from syspilot_manual.integration import build_existing_index

    bundle_dir = build_existing_index(manual_id, artifact_root, _resolve_manual_root(manual_root))
    click.echo(
        json.dumps(
            {
                "manual_id": manual_id,
                "index_path": str(bundle_dir / "manual_index.jsonl"),
            },
            indent=2,
        )
    )


@cli.command("compose-manuals")
@click.option("--app", required=True, help="Application or workload name")
@click.option(
    "--libraries", multiple=True, required=True, help="Selected manual IDs or manual.json paths"
)
@click.option("--manual-id", help="Override composed manual ID (defaults to <app>_composed)")
@click.option("--objective", help="Optional app objective")
@click.option("--symptom", help="Optional current symptom")
@click.option(
    "--artifact-root",
    type=click.Path(path_type=Path),
    default=Path("artifacts"),
    show_default=True,
)
@click.option(
    "--manual-root",
    type=click.Path(path_type=Path),
)
@click.pass_context
def compose_manuals_cmd(
    ctx: click.Context,
    app: str,
    libraries: tuple[str, ...],
    manual_id: str | None,
    objective: str | None,
    symptom: str | None,
    artifact_root: Path,
    manual_root: Path | None,
) -> None:
    from syspilot_manual.integration import compose_and_write_manual
    from syspilot_manual.llm_client import ManualLLMClient

    llm_client = ManualLLMClient(
        config=_manual_pipeline_llm_config(ctx.obj["config"].llm, minimum_timeout_seconds=180.0)
    )
    resolved_manual_id = manual_id or f"{_slugify_manual_id(app)}_composed"
    resolved_manual_root = _resolve_manual_root(manual_root, create=True)
    composed = asyncio.run(
        compose_and_write_manual(
            manual_id=resolved_manual_id,
            app_name=app,
            manual_refs=list(libraries),
            artifact_root=artifact_root,
            llm_client=llm_client,
            manual_root=resolved_manual_root,
            objective=objective,
            symptom=symptom,
            progress_callback=_manual_composition_progress_callback,
        )
    )
    click.echo(
        json.dumps(
            {
                "manual_id": composed.manual_id,
                "bundle_dir": str(resolved_manual_root / composed.manual_id),
                "compat_markdown": str(resolved_manual_root / f"{composed.manual_id}.md"),
            },
            indent=2,
        )
    )


@cli.command("compose-contracts")
@click.option("--app", required=True, help="Application or workload name")
@click.option(
    "--contracts",
    multiple=True,
    required=True,
    help="Selected contract IDs or contract.yaml paths",
)
@click.option("--contract-id", help="Override composed contract ID (defaults to <app>_composed)")
@click.option(
    "--artifact-root",
    type=click.Path(path_type=Path),
    default=Path("artifacts"),
    show_default=True,
)
@click.option(
    "--contract-root",
    type=click.Path(path_type=Path),
)
@click.option(
    "--include-evidence/--no-include-evidence",
    default=False,
    show_default=True,
    help="Include evidence blocks in generated LPCL output",
)
def compose_contracts_cmd(
    app: str,
    contracts: tuple[str, ...],
    contract_id: str | None,
    artifact_root: Path,
    contract_root: Path | None,
    include_evidence: bool,
) -> None:
    from syspilot_contract.integration import compose_and_write_contract

    resolved_contract_id = contract_id or f"{_slugify_manual_id(app)}_composed"
    resolved_contract_root = _resolve_contract_root(contract_root, create=True)
    composed = compose_and_write_contract(
        contract_id=resolved_contract_id,
        app_name=app,
        contract_refs=list(contracts),
        artifact_root=artifact_root,
        contract_root=resolved_contract_root,
        include_evidence=include_evidence,
        progress_callback=_contract_composition_progress_callback,
    )
    click.echo(
        json.dumps(
            {
                "contract_id": resolved_contract_id,
                "contract": composed.contract,
                "contract_root": str(resolved_contract_root),
                "bundle_dir": str(resolved_contract_root / resolved_contract_id),
                "contract_yaml": str(
                    resolved_contract_root / resolved_contract_id / "contract.yaml"
                ),
                "contract_lpcl": str(
                    resolved_contract_root / resolved_contract_id / "contract.lpcl"
                ),
            },
            indent=2,
        )
    )


@cli.command("query")
@click.option("--manual", "manual_id", required=True, help="Manual ID to query")
@click.option("--section", help="Section name")
@click.option("--keywords", multiple=True, help="Keyword filters")
@click.option("--symptom", help="Symptom filter")
@click.option("--goal", help="Goal filter")
@click.option(
    "--manual-root",
    type=click.Path(path_type=Path),
)
def query_manual_cmd(
    manual_id: str,
    section: str | None,
    keywords: tuple[str, ...],
    symptom: str | None,
    goal: str | None,
    manual_root: Path | None,
) -> None:
    from syspilot_manual.integration import load_store

    store = load_store(_resolve_manual_root(manual_root))
    if section:
        result = store.query_section(manual_id, section, list(keywords) or None)
    elif symptom:
        result = store.query_by_symptom([manual_id], symptom)
    elif goal:
        result = store.query_by_goal([manual_id], goal)
    else:
        raise click.UsageError("Provide --section, --symptom, or --goal")
    click.echo(json.dumps(result, indent=2))


@cli.command("build-agent-context")
@click.option("--manuals", multiple=True, required=True, help="Manual IDs to query")
@click.option("--goal", help="Goal filter")
@click.option("--symptom", help="Symptom filter")
@click.option("--max-items-per-section", default=5, show_default=True, type=int)
@click.option(
    "--manual-root",
    type=click.Path(path_type=Path),
)
@click.option(
    "--contract-root",
    type=click.Path(path_type=Path),
)
def build_agent_context_cmd(
    manuals: tuple[str, ...],
    goal: str | None,
    symptom: str | None,
    max_items_per_section: int,
    manual_root: Path | None,
    contract_root: Path | None,
) -> None:
    bundle = _build_structured_context_bundle(
        list(manuals),
        goal=goal,
        symptom=symptom,
        max_items_per_section=max_items_per_section,
        manual_root=manual_root if manual_root is not None else _discover_manual_root(),
        contract_root=contract_root if contract_root is not None else _discover_contract_root(),
    )
    click.echo(json.dumps(bundle, indent=2))


@cli.command()
def skills() -> None:
    registry = SkillRegistry()

    for skill_cls in [
        DynamicScriptSkill,
        StraceAnalyzerSkill,
        PerfAnalyzerSkill,
        DataProcessorSkill,
    ]:
        registry.register(skill_cls())

    table = Table(title="Available Skills")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="green")

    for skill_info in registry.list_skills():
        table.add_row(skill_info["name"], skill_info["description"])

    console.print(table)


@cli.command()
@click.option("--command", "-c", type=str, help="Command to run and tune (spawn mode)")
@click.option(
    "--ui-mode",
    type=click.Choice(["tui", "plain"], case_sensitive=False),
    default="tui",
    show_default=True,
    help="Tune UI mode (tui-first with safe fallback to plain)",
)
@click.option(
    "--tui-bin",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Override path to syspilot-tui binary (or set SYSPILOT_TUI_BIN)",
)
@click.option(
    "--workdir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Working directory for spawn mode command execution",
)
@click.option(
    "--memory-high",
    type=str,
    help=(
        "If set in spawn mode, launch the command via systemd-run --user "
        "--scope -p MemoryHigh=..."
    ),
)
@click.option(
    "--profile/--no-profile",
    "profile_enabled",
    default=True,
    show_default=True,
    help="Enable profiling evidence collection; --no-profile uses raw command output only.",
)
@click.option(
    "--contract",
    multiple=True,
    help=(
        "Library contract context(s) for spawn-mode tune. Repeat the option or use comma-separated "
        "values (escape literal commas as \\,)."
    ),
)
@click.option(
    "--manual",
    "-m",
    multiple=True,
    help=(
        "Application-specific natural-language tune hint(s). Repeat the "
        "option or use comma-separated "
        "values (escape literal commas as \\,)."
    ),
)
@click.option(
    "--exclude-config",
    multiple=True,
    help=(
        "Environment variable, command-line parameter, or config control "
        "that tune must keep unchanged. "
        "Repeat the option or use comma-separated values (escape literal commas as \\,)."
    ),
)
@click.option(
    "--tunable-whitelist",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "File containing newline-separated environment variables, command-line parameters, "
        "or config controls that tune is allowed to change."
    ),
)
@click.option(
    "--fixed-budget",
    type=click.IntRange(min=1),
    help="Positive cap on total candidate trial budget across tune mode.",
)
@click.option(
    "--fixed-goal",
    type=click.FloatRange(min=0.0, min_open=True),
    help=(
        "Positive optimization-target value to reach before stopping. Uses <= for "
        "minimize objectives and >= for maximize objectives."
    ),
)
@click.option(
    "--study-ledger",
    type=click.Path(dir_okay=False, path_type=Path),
    help=(
        "Opt-in nestedcpu-blas deep-trace JSONL ledger path for spawn-mode tune; "
        "requires a repository source checkout."
    ),
)
@click.option(
    "--study-proposal-cap",
    type=click.IntRange(min=1),
    help="Deep-trace model proposal cap; requires --study-ledger (default: manifest value 60).",
)
@click.option(
    "--study-unique-execution-target",
    type=click.IntRange(min=1),
    help=(
        "Deep-trace unique non-baseline execution target; requires --study-ledger "
        "(default: manifest value 30)."
    ),
)
@click.option(
    "--max-iterations",
    default=20,
    show_default=True,
    help="Maximum materialized candidate trial attempts, excluding the baseline run.",
)
@click.option("--dry-run/--no-dry-run", default=True, help="Show changes without applying")
@click.option("--timeout", default=120, help="Timeout per command run in seconds")
@click.option(
    "--primary-objective",
    type=str,
    help=(
        "Free-form workload objective to optimize during spawn-mode tune "
        "(for example: wall time, throughput, latency). SysPilot extracts "
        "runtime outcome metrics from program output, resolves this intent to "
        "a numeric metric, and uses orchestration wall time only as a tie-breaker."
    ),
)
@click.option(
    "--primary-objective-mode",
    type=click.Choice(["minimize", "maximize"], case_sensitive=False),
    help=(
        "How to optimize --primary-objective. If omitted, SysPilot infers a direction "
        "for common metric names."
    ),
)
@click.option(
    "--quick-mode/--no-quick-mode",
    default=False,
    show_default=True,
    help=(
        "Enable quick tuning mode with one full sampled baseline, short signal-screening runs, "
        "and capped full validation."
    ),
)
@click.option(
    "--quick-sample-interval-sec",
    type=click.FloatRange(min=0.0, min_open=True),
    default=DEFAULT_QUICK_MODE_DEFAULTS.sample_interval_sec,
    show_default=True,
    help="Quick-mode periodic probe sampling interval in seconds.",
)
@click.option(
    "--quick-short-measurement-sec",
    type=click.FloatRange(min=0.0, min_open=True),
    default=DEFAULT_QUICK_MODE_DEFAULTS.short_measurement_sec,
    show_default=True,
    help="Quick-mode short-screen measurement budget in seconds.",
)
@click.option(
    "--quick-full-run-ratio",
    type=click.FloatRange(min=0.0, max=1.0),
    default=DEFAULT_QUICK_MODE_DEFAULTS.full_run_ratio,
    show_default=True,
    help="Fraction of short-screen winners to promote to full validation within each signal group.",
)
@click.option(
    "--quick-max-full-runs-per-group",
    type=click.IntRange(min=1),
    default=DEFAULT_QUICK_MODE_DEFAULTS.max_full_runs_per_group,
    show_default=True,
    help="Quick-mode cap on promoted full validations per checked signal group.",
)
@click.option(
    "--quick-min-signal-samples",
    type=click.IntRange(min=1),
    default=DEFAULT_QUICK_MODE_DEFAULTS.min_signal_samples,
    show_default=True,
    help="Minimum baseline samples a quick-mode signal needs before it can be screened.",
)
@click.option(
    "--quick-neutral-change-pct",
    type=click.FloatRange(min=0.0),
    default=DEFAULT_QUICK_MODE_DEFAULTS.neutral_change_pct,
    show_default=True,
    help="Quick-mode threshold below which signal or objective change counts as neutral.",
)
@click.option(
    "--quick-significant-improvement-pct",
    type=click.FloatRange(min=0.0),
    default=DEFAULT_QUICK_MODE_DEFAULTS.significant_improvement_pct,
    show_default=True,
    help="Quick-mode full-run improvement threshold for a significant win.",
)
@click.option(
    "--quick-max-windows-per-signal",
    type=click.IntRange(min=1),
    default=DEFAULT_QUICK_MODE_DEFAULTS.max_windows_per_signal,
    show_default=True,
    help="Maximum ranked abnormal windows to keep per signal during quick mode.",
)
@click.option(
    "--quick-max-signal-groups",
    type=click.IntRange(min=1),
    default=DEFAULT_QUICK_MODE_DEFAULTS.max_signal_groups,
    show_default=True,
    help="Maximum ranked signal windows quick mode will check overall.",
)
@click.option(
    "--quick-max-total-short-runs",
    type=click.IntRange(min=1),
    default=DEFAULT_QUICK_MODE_DEFAULTS.max_total_short_runs,
    show_default=True,
    help="Quick-mode total cap on short-screen candidate runs.",
)
@click.option(
    "--quick-max-total-full-runs",
    type=click.IntRange(min=1),
    default=DEFAULT_QUICK_MODE_DEFAULTS.max_total_full_runs,
    show_default=True,
    help="Quick-mode total cap on promoted full validation runs.",
)
@click.option("--debug-llm", is_flag=True, help="Show full LLM conversation input/output")
@click.option(
    "--debug-llm-md", type=click.Path(), help="Write LLM conversation transcript to markdown file"
)
@click.pass_context
def tune(
    ctx: click.Context,
    command: str | None,
    ui_mode: str,
    tui_bin: Path | None,
    workdir: Path | None,
    memory_high: str | None,
    profile_enabled: bool,
    contract: tuple[str, ...],
    manual: tuple[str, ...],
    exclude_config: tuple[str, ...],
    tunable_whitelist: Path | None,
    fixed_budget: int | None,
    fixed_goal: float | None,
    study_ledger: Path | None,
    study_proposal_cap: int | None,
    study_unique_execution_target: int | None,
    max_iterations: int,
    dry_run: bool,
    timeout: int,
    primary_objective: str | None,
    primary_objective_mode: str | None,
    quick_mode: bool,
    quick_sample_interval_sec: float,
    quick_short_measurement_sec: float,
    quick_full_run_ratio: float,
    quick_max_full_runs_per_group: int,
    quick_min_signal_samples: int,
    quick_neutral_change_pct: float,
    quick_significant_improvement_pct: float,
    quick_max_windows_per_signal: int,
    quick_max_signal_groups: int,
    quick_max_total_short_runs: int,
    quick_max_total_full_runs: int,
    debug_llm: bool,
    debug_llm_md: str | None,
) -> None:
    """Iteratively tune a spawned command.

    Example:
      syspilot tune -c "./app --arg value" --workdir /path/to/app --contract numactl
    """
    requested_mode = TuneUIMode(ui_mode.lower())
    plain_sink = make_output_sink(ui_mode=TuneUIMode.PLAIN, console=console)
    sink: OutputSink = plain_sink

    if not command:
        sink.print("[red]Error: --command is required; tune is spawn-mode only[/red]")
        raise SystemExit(1)
    if quick_mode and not profile_enabled:
        sink.print("[red]Error: --no-profile is not supported with --quick-mode[/red]")
        raise SystemExit(1)
    if quick_mode and study_ledger is not None:
        sink.print("[red]Error: --study-ledger is supported only by standard spawn-mode tune[/red]")
        raise SystemExit(1)
    if study_ledger is None and (
        study_proposal_cap is not None or study_unique_execution_target is not None
    ):
        sink.print(
            "[red]Error: --study-proposal-cap and --study-unique-execution-target "
            "require --study-ledger[/red]"
        )
        raise SystemExit(1)

    study_trace: NestedCpuBlasStudyTrace | None = None
    if study_ledger is not None:
        try:
            study_trace = NestedCpuBlasStudyTrace(
                study_ledger,
                proposal_cap=study_proposal_cap,
                unique_execution_target=study_unique_execution_target,
            )
        except (RepositoryStudySourceUnavailableError, OSError, ValueError) as exc:
            sink.print("[red]Error: could not initialize --study-ledger[/red]")
            sink.print(f"[red]{exc}[/red]")
            raise SystemExit(1) from exc

    resolved_primary_objective: str | None = None
    resolved_primary_objective_mode: str | None = None
    contract_names = _normalize_multi_value_option(contract)
    manual_inputs = _normalize_multi_value_option(manual)
    excluded_config_names = _normalize_multi_value_option(exclude_config)
    tunable_whitelist_names: list[str] | None = None
    if tunable_whitelist is not None:
        try:
            tunable_whitelist_names = _load_tunable_whitelist(tunable_whitelist)
        except (OSError, ValueError) as exc:
            sink.print(f"[red]Error: could not read --tunable-whitelist: {exc}[/red]")
            raise SystemExit(1) from exc
    if primary_objective is not None:
        resolved_primary_objective = primary_objective.strip()
        if not resolved_primary_objective:
            sink.print("[red]Error: --primary-objective cannot be empty[/red]")
            raise SystemExit(1)
        resolved_primary_objective_mode = (
            primary_objective_mode.lower() if primary_objective_mode else None
        )
    elif primary_objective_mode is not None:
        sink.print("[red]Error: --primary-objective-mode requires --primary-objective[/red]")
        raise SystemExit(1)

    normalized_memory_high: str | None = None
    if memory_high is not None:
        normalized_memory_high = memory_high.strip()
        if not normalized_memory_high:
            sink.print("[red]Error: --memory-high cannot be empty[/red]")
            raise SystemExit(1)

    quick_mode_defaults = QuickModeDefaults(
        sample_interval_sec=quick_sample_interval_sec,
        short_measurement_sec=quick_short_measurement_sec,
        full_run_ratio=quick_full_run_ratio,
        max_full_runs_per_group=quick_max_full_runs_per_group,
        min_signal_samples=quick_min_signal_samples,
        neutral_change_pct=quick_neutral_change_pct,
        significant_improvement_pct=quick_significant_improvement_pct,
        max_windows_per_signal=quick_max_windows_per_signal,
        max_signal_groups=quick_max_signal_groups,
        max_total_short_runs=quick_max_total_short_runs,
        max_total_full_runs=quick_max_total_full_runs,
    )

    tui_orchestrator: TuneTuiOrchestrator | None = None
    tui_sink: SwitchableOutputSink | None = None
    if requested_mode == TuneUIMode.TUI:
        tui_orchestrator = TuneTuiOrchestrator(requested="tui", tui_bin_override=tui_bin)
        started, warning = tui_orchestrator.try_start()
        if warning:
            plain_sink.print(f"[yellow]{warning}[/yellow]")
        if started:
            tui_sink = SwitchableOutputSink(
                make_output_sink(ui_mode=TuneUIMode.TUI, console=console)
            )
            sink = _TuiAwareOutputSink(
                sink=tui_sink,
                orchestrator=tui_orchestrator,
                plain_sink=plain_sink,
            )

    config: SysPilotConfig = ctx.obj["config"]
    _enable_llm_debug_options(config, debug_llm=debug_llm, debug_llm_md=debug_llm_md)
    config.tuning.dry_run = dry_run
    config.tuning.max_tuning_iterations = max_iterations

    agent = SysPilotAgent(config)
    if debug_llm or debug_llm_md:
        prompt_log_path = agent.enable_llm_prompt_logging(_default_tune_prompt_log_path())
        sink.print(f"[cyan]Saving LLM prompts to: {prompt_log_path}[/cyan]")
    loaded_manual_refs, manual_hints, missing_manual_refs = _load_spawn_manual_inputs(
        agent,
        manual_inputs,
    )

    contract_root = _discover_contract_root() if contract_names else None
    structured_context_bundle = _build_contract_context_bundle(
        contract_names,
        goal="improve throughput",
        symptom=None,
        max_items_per_section=5,
        contract_root=contract_root,
    )
    initial_hypothesis_profiles = _extract_hypothesis_profiles(
        structured_context_bundle,
        max_profiles=None,
    )
    agent.configure_tuning_prompt(
        contract_enabled=bool(contract_names),
        manual_enabled=bool(loaded_manual_refs or manual_hints),
        quick_mode=quick_mode,
        profiling_enabled=profile_enabled,
    )

    if contract_names:
        if _has_structured_context(structured_context_bundle):
            sink.print(
                "[cyan]Using structured contract context: "
                f"{', '.join(contract_names)}[/cyan]"
            )
        else:
            sink.print(
                "[yellow]No structured contract context found for: "
                f"{', '.join(contract_names)}[/yellow]"
            )

    if manual_hints:
        sink.print(f"[cyan]Using application manual hint(s): {', '.join(manual_hints)}[/cyan]")
    _ = loaded_manual_refs, missing_manual_refs

    if excluded_config_names:
        sink.print(
            "[cyan]Keeping config control(s) unchanged: "
            f"{', '.join(excluded_config_names)}[/cyan]"
        )

    if tunable_whitelist_names is not None:
        whitelist_display = ", ".join(tunable_whitelist_names) or "(empty)"
        sink.print(f"[cyan]Restricting tunable controls to: {whitelist_display}[/cyan]")

    if initial_hypothesis_profiles:
        sink.print(
            "[cyan]Generated "
            f"{len(initial_hypothesis_profiles)} contract hypothesis profile(s) "
            "at startup[/cyan]"
        )

    manager = ProfilingManager()
    tuner = ApplicationTuner(config.tuning)

    _register_default_skills(agent)
    _register_profiling_tools(agent, manager)
    _register_tuning_tools(agent, tuner)

    base_emitter = TuneEventEmitter()
    event_emitter: _TuneEmitter = base_emitter
    if tui_orchestrator is not None and tui_sink is not None:
        event_emitter = _TuiForwardingEmitterAdapter(
            emitter=base_emitter,
            orchestrator=tui_orchestrator,
        )

    if isinstance(sink, _TuiAwareOutputSink):
        sink.set_event_emitter(event_emitter)

    logger_forward_handler: _TuiLogForwardHandler | None = None
    syspilot_logger = logging.getLogger("syspilot")
    if isinstance(sink, _TuiAwareOutputSink):
        logger_forward_handler = _TuiLogForwardHandler(sink)
        syspilot_logger.addHandler(logger_forward_handler)

    try:
        if quick_mode:
            _run_quick_tune_loop(
                agent=agent,
                command=command,
                contracts=contract_names,
                manual_hints=manual_hints,
                fixed_budget=fixed_budget,
                fixed_goal=fixed_goal,
                max_iterations=max_iterations,
                timeout=timeout,
                primary_objective=resolved_primary_objective,
                primary_objective_mode=resolved_primary_objective_mode,
                workdir=workdir,
                memory_high=normalized_memory_high,
                sink=sink,
                event_emitter=event_emitter,
                structured_context_bundle=(
                    structured_context_bundle
                    if _has_structured_context(structured_context_bundle)
                    else None
                ),
                initial_hypothesis_profiles=initial_hypothesis_profiles,
                defaults=quick_mode_defaults,
                excluded_config_names=excluded_config_names,
                tunable_whitelist_names=tunable_whitelist_names,
            )
        else:
            _run_spawn_tune_loop(
                agent,
                command,
                contract_names,
                manual_hints,
                fixed_budget,
                fixed_goal,
                max_iterations,
                timeout,
                resolved_primary_objective,
                resolved_primary_objective_mode,
                workdir,
                normalized_memory_high,
                profile_enabled,
                sink,
                event_emitter,
                (
                    structured_context_bundle
                    if _has_structured_context(structured_context_bundle)
                    else None
                ),
                initial_hypothesis_profiles,
                excluded_config_names,
                tunable_whitelist_names,
                study_trace,
            )
    finally:
        if logger_forward_handler is not None:
            syspilot_logger.removeHandler(logger_forward_handler)
        if isinstance(event_emitter, _TuiForwardingEmitterAdapter):
            event_emitter.flush()
        if tui_orchestrator is not None:
            tui_orchestrator.close(wait_for_user=should_wait_for_user_on_close())


class _TuiAwareOutputSink:
    def __init__(
        self,
        *,
        sink: SwitchableOutputSink,
        orchestrator: TuneTuiOrchestrator,
        plain_sink: OutputSink,
    ) -> None:
        self._sink: SwitchableOutputSink = sink
        self._orchestrator: TuneTuiOrchestrator = orchestrator
        self._plain_sink: OutputSink = plain_sink
        self._event_emitter: _TuneEmitter | None = None

    def set_event_emitter(self, event_emitter: _TuneEmitter) -> None:
        self._event_emitter = event_emitter

    def is_tui_active(self) -> bool:
        return self._orchestrator.using_tui

    def _ensure_fallback(self) -> None:
        warning = self._orchestrator.maybe_fallback_warning()
        if not warning:
            return
        self._sink.switch(self._plain_sink)
        self._plain_sink.print(f"[yellow]{warning}[/yellow]")

    def _emit_log_line(self, message: str) -> None:
        if not self._orchestrator.using_tui:
            return

        normalized = message.strip()
        if not normalized:
            return

        _emit_tune_event(
            self._event_emitter,
            "log.line",
            {"message": normalized},
        )

    def print(self, message: str = "") -> None:
        self._ensure_fallback()
        self._emit_log_line(message)
        self._sink.print(message)

    def panel(self, body: str, *, title: str) -> None:
        self._ensure_fallback()
        self._emit_log_line(title)
        for line in body.splitlines():
            self._emit_log_line(line)
        self._sink.panel(body, title=title)

    def table(
        self,
        *,
        title: str,
        columns: Sequence[tuple[str, str | None]],
        rows: Sequence[Sequence[str]],
    ) -> None:
        self._ensure_fallback()
        header = " | ".join(name for name, _ in columns)
        if title:
            self._emit_log_line(title)
        self._emit_log_line(header)
        for row in rows:
            self._emit_log_line(" | ".join(str(value) for value in row))
        self._sink.table(title=title, columns=columns, rows=rows)


class _TuiForwardingEmitterAdapter:
    def __init__(
        self,
        *,
        emitter: TuneEventEmitter,
        orchestrator: TuneTuiOrchestrator,
    ) -> None:
        self._emitter: TuneEventEmitter = emitter
        self._orchestrator: TuneTuiOrchestrator = orchestrator
        self._pending_noncritical_count = 0
        self._last_forward_ts = time.monotonic()

    def _forward(self) -> None:
        self._orchestrator.forward_emitter(self._emitter)
        self._pending_noncritical_count = 0
        self._last_forward_ts = time.monotonic()

    def flush(self) -> None:
        self._forward()

    def publish(
        self,
        event_name: str,
        payload: dict[str, object] | None = None,
        *,
        event_type: str | None = None,
    ) -> bool:
        ok = self._emitter.publish(event_name, payload, event_type=event_type)
        if not ok:
            return False

        if is_critical_event(event_name):
            self._forward()
            return True

        self._pending_noncritical_count += 1
        now = time.monotonic()
        if self._pending_noncritical_count >= 64 or (now - self._last_forward_ts) >= 0.25:
            self._forward()
        return ok


class _TuiLogForwardHandler(logging.Handler):
    def __init__(self, sink: _TuiAwareOutputSink) -> None:
        super().__init__(level=logging.INFO)
        self._sink = sink
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        if not self._sink.is_tui_active():
            return

        try:
            message = self.format(record)
        except Exception:
            return

        if record.levelno >= logging.ERROR:
            message = f"[red]{message}[/red]"
        elif record.levelno >= logging.WARNING:
            message = f"[yellow]{message}[/yellow]"
        elif record.levelno <= logging.DEBUG:
            message = f"[dim]{message}[/dim]"

        self._sink.print(message)


def _emit_tune_event(
    event_emitter: _TuneEmitter | None,
    event_name: str,
    payload: dict[str, object],
) -> None:
    if event_emitter is None:
        return
    try:
        _ = event_emitter.publish(event_name, payload)
    except Exception:
        return


def _run_command(
    cmd: list[str],
    timeout: int,
    workdir: str | None = None,
    env_overrides: dict[str, str] | None = None,
) -> tuple[float, str, str, int]:
    """Run command and return (wall_time, stdout, stderr, returncode)."""
    t0 = time.monotonic()
    try:
        env = None
        if env_overrides:
            env = os.environ.copy()
            env.update({k: str(v) for k, v in env_overrides.items()})

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir,
            env=env,
        )
        wall = time.monotonic() - t0
        return wall, result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return timeout, "", "Command timed out", -1


def _event_output_payload(
    stdout: str,
    stderr: str,
    max_chars: int = 8000,
) -> dict[str, Any]:
    stdout_text = stdout or ""
    stderr_text = stderr or ""

    stdout_truncated = len(stdout_text) > max_chars
    stderr_truncated = len(stderr_text) > max_chars

    return {
        "stdout": stdout_text[:max_chars],
        "stderr": stderr_text[:max_chars],
        "stdout_len": len(stdout_text),
        "stderr_len": len(stderr_text),
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }


def _fixed_budget_event_payload(
    fixed_budget_state: FixedBudgetState,
    *,
    stop_reason: str = "",
) -> dict[str, object]:
    if not fixed_budget_state.enabled:
        return {}
    payload = fixed_budget_state.to_payload()
    return {
        "fixed_budget": payload,
        "fixed_budget_total": payload["total"],
        "fixed_budget_used": payload["used"],
        "fixed_budget_remaining": payload["remaining"],
        "fixed_budget_stop_reason": stop_reason,
        "no_action_reprompts_used": payload["no_action_reprompts_used"],
    }


def _profile_run_event_payload(
    *,
    phase: str,
    iteration: int | None = None,
    candidate_id: int | None = None,
    candidate_label: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"mode": "spawn", "phase": phase}
    if iteration is not None:
        payload["iteration"] = iteration
    if candidate_id is not None:
        payload["candidate_id"] = candidate_id
    if candidate_label:
        payload["candidate_label"] = candidate_label
        payload["run_label"] = candidate_label
    elif phase == "spawn.baseline":
        payload["run_label"] = "baseline"
    return payload


def _emit_profile_artifact_events(
    event_emitter: _TuneEmitter | None,
    profiling_artifacts: dict[str, Any],
    *,
    phase: str,
    iteration: int | None = None,
    candidate_id: int | None = None,
    candidate_label: str | None = None,
) -> None:
    profiling_json = profiling_artifacts.get("profiling_json", {})
    profiling_text = str(profiling_artifacts.get("profiling_text", ""))
    profile_json_payload: dict[str, Any] = {}
    profile_tools: list[str] = []
    if isinstance(profiling_json, dict):
        profile_json_payload = dict(profiling_json)
        raw_tools = profiling_json.get("tools_used")
        if isinstance(raw_tools, list):
            profile_tools = [str(tool) for tool in raw_tools if str(tool).strip()]
    profiling_enabled = bool(profile_json_payload.get("pipeline_used", True))

    base_payload = _profile_run_event_payload(
        phase=phase,
        iteration=iteration,
        candidate_id=candidate_id,
        candidate_label=candidate_label,
    )
    _emit_tune_event(
        event_emitter,
        "profile.snapshot",
        {
            **base_payload,
            "profile_tools": profile_tools,
            "profiling_enabled": profiling_enabled,
            "profiling_skipped": not profiling_enabled,
            "profile_json": profile_json_payload,
            "profile_summary": profiling_text,
            "metrics": {
                "pid": profiling_artifacts["pid"],
                "wall_time_s": profiling_artifacts["wall_time"],
                "returncode": profiling_artifacts["returncode"],
                "tool_calls_used": profiling_artifacts.get("tool_calls_used", 0),
            },
        },
    )
    _emit_tune_event(
        event_emitter,
        "profile.finished",
        {
            **base_payload,
            "status": "success" if profiling_artifacts["returncode"] == 0 else "failed",
            "profiling_enabled": profiling_enabled,
            "profiling_skipped": not profiling_enabled,
            "returncode": profiling_artifacts["returncode"],
            "wall_time_s": profiling_artifacts["wall_time"],
            **_event_output_payload(
                str(profiling_artifacts.get("stdout", "")),
                str(profiling_artifacts.get("stderr", "")),
            ),
        },
    )


def _print_profile_artifact_summary(
    sink: OutputSink,
    profiling_artifacts: dict[str, Any],
    *,
    prefix: str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    profiling_text = str(profiling_artifacts.get("profiling_text", ""))
    profiling_json = profiling_artifacts.get("profiling_json", {})
    if not isinstance(profiling_json, dict):
        profiling_json = {}

    sink.print(f"  {prefix} PID: {profiling_artifacts['pid']}")
    sink.print(f"  {prefix} wall time: {profiling_artifacts['wall_time']:.2f}s")
    sink.print(f"  Agent tool calls used: {profiling_artifacts.get('tool_calls_used', 0)}")
    if profiling_artifacts["returncode"] != 0:
        sink.print(f"  [yellow]{prefix} return code: {profiling_artifacts['returncode']}[/yellow]")

    target_pids = profiling_json.get("target_pids", []) if profiling_json else []
    primary_pid = profiling_json.get("primary_pid") if profiling_json else None
    if target_pids:
        sink.print(f"  Profiling targets: {target_pids} (primary={primary_pid})")

    cgroup_memory = profiling_json.get("cgroup_memory_evidence", {}) if profiling_json else {}
    if isinstance(cgroup_memory, dict) and cgroup_memory.get("control_group"):
        memory_events = (
            cgroup_memory.get("memory_events", {})
            if isinstance(cgroup_memory.get("memory_events"), dict)
            else {}
        )
        sink.print(
            "  Cgroup memory envelope: "
            "limit="
            f"{_format_bytes_human(cast(int | None, cgroup_memory.get('effective_limit_bytes')))} "
            f"current={_format_bytes_human(cast(int | None, cgroup_memory.get('current_bytes')))} "
            f"peak={_format_bytes_human(cast(int | None, cgroup_memory.get('peak_bytes')))} "
            "headroom="
            f"{_format_bytes_human(cast(int | None, cgroup_memory.get('min_headroom_bytes')))} "
            f"status={cgroup_memory.get('status', 'unknown')}"
        )
        sink.print(
            "  Cgroup memory signals: "
            f"high={memory_events.get('high', 0)} "
            f"max={memory_events.get('max', 0)} "
            f"oom={memory_events.get('oom', 0)} "
            f"oom_kill={memory_events.get('oom_kill', 0)}"
        )

    contract_validation = (
        profiling_json.get("contract_hypothesis_validation", {}) if profiling_json else {}
    )
    if isinstance(contract_validation, dict) and contract_validation.get("profiles"):
        profiles = contract_validation.get("profiles", [])
        sink.print(f"  Contract-guided validation profiles: {len(profiles)}")
        if isinstance(profiles, list):
            for profile in profiles[:3]:
                if not isinstance(profile, dict):
                    continue
                signal_results = profile.get("signal_results", [])
                observed = (
                    sum(
                        1
                        for item in signal_results
                        if isinstance(item, dict) and bool(item.get("observed"))
                    )
                    if isinstance(signal_results, list)
                    else 0
                )
                total = len(signal_results) if isinstance(signal_results, list) else 0
                sink.print(
                    f"    - {profile.get('hypothesis_id')}: observed {observed}/{total} signal(s)"
                )

    return (
        profiling_text,
        profiling_json,
        contract_validation if isinstance(contract_validation, dict) else {},
    )


def _render_spawn_manual_hints(manuals: Sequence[str]) -> str:
    if not manuals:
        return ""
    return "\n".join(f"- {manual}" for manual in manuals)


def _format_progress_duration(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _format_progress_finish_time(eta_s: float) -> str:
    finish_time = datetime.now(timezone.utc) + timedelta(seconds=max(eta_s, 0.0))
    return finish_time.strftime("%H:%M:%S UTC")


def _manual_extract_progress_callback(progress: object) -> None:
    from syspilot_manual.extract_evidence import ExtractionProgress

    if not isinstance(progress, ExtractionProgress):
        return
    if progress.status == "started":
        if progress.resumed_chunks:
            click.echo(
                "[extract-evidence] "
                f"{progress.library_name}: resuming from checkpoint at "
                f"{progress.resumed_chunks}/{progress.total_chunks} chunks"
            )
        else:
            click.echo(
                f"[extract-evidence] {progress.library_name}: starting "
                f"{progress.total_chunks} chunks"
            )
        return
    if progress.status == "completed":
        click.echo(
            "[extract-evidence] "
            f"{progress.library_name}: completed "
            f"{progress.processed_chunks}/{progress.total_chunks} "
            f"chunks, ok={progress.succeeded_chunks}, failed={progress.failed_chunks}, "
            f"elapsed={_format_progress_duration(progress.elapsed_s)}"
        )
        return

    if progress.processed_chunks != 1 and progress.processed_chunks % 10 != 0:
        return

    pct = (
        (progress.processed_chunks / progress.total_chunks * 100.0)
        if progress.total_chunks
        else 100.0
    )
    click.echo(
        "[extract-evidence] "
        f"{progress.processed_chunks}/{progress.total_chunks} ({pct:.1f}%) "
        f"ok={progress.succeeded_chunks} failed={progress.failed_chunks} "
        f"items={progress.current_items} elapsed={_format_progress_duration(progress.elapsed_s)} "
        f"avg/chunk={progress.avg_chunk_s:.1f}s eta={_format_progress_duration(progress.eta_s)} "
        f"finish~{_format_progress_finish_time(progress.eta_s)} current={progress.current_chunk_id}"
    )


def _manual_synthesis_progress_callback(progress: object) -> None:
    from syspilot_manual.integration import ManualSynthesisProgress

    if not isinstance(progress, ManualSynthesisProgress):
        return
    click.echo(
        "[synthesize-manual] "
        f"{progress.library_name}/{progress.manual_id} phase={progress.status} "
        f"evidence={progress.evidence_items} "
        f"elapsed={_format_progress_duration(progress.elapsed_s)} "
        f"eta={_format_progress_duration(progress.eta_s)} "
        f"finish~{_format_progress_finish_time(progress.eta_s)}"
    )


def _manual_composition_progress_callback(progress: object) -> None:
    from syspilot_manual.integration import ManualCompositionProgress

    if not isinstance(progress, ManualCompositionProgress):
        return
    click.echo(
        "[compose-manuals] "
        f"{progress.app_name}/{progress.manual_id} phase={progress.status} "
        f"libraries={progress.library_count} "
        f"elapsed={_format_progress_duration(progress.elapsed_s)} "
        f"eta={_format_progress_duration(progress.eta_s)} "
        f"finish~{_format_progress_finish_time(progress.eta_s)}"
    )


def _contract_synthesis_progress_callback(progress: object) -> None:
    from syspilot_contract.integration import ContractSynthesisProgress

    if not isinstance(progress, ContractSynthesisProgress):
        return
    click.echo(
        "[synthesize-contract] "
        f"{progress.library_name}/{progress.contract_id} phase={progress.status} "
        f"evidence={progress.evidence_items} "
        f"elapsed={_format_progress_duration(progress.elapsed_s)} "
        f"eta={_format_progress_duration(progress.eta_s)} "
        f"finish~{_format_progress_finish_time(progress.eta_s)}"
    )


def _contract_composition_progress_callback(progress: object) -> None:
    from syspilot_contract.integration import ContractCompositionProgress

    if not isinstance(progress, ContractCompositionProgress):
        return
    click.echo(
        "[compose-contracts] "
        f"{progress.app_name}/{progress.contract_id} phase={progress.status} "
        f"libraries={progress.library_count} "
        f"elapsed={_format_progress_duration(progress.elapsed_s)} "
        f"eta={_format_progress_duration(progress.eta_s)} "
        f"finish~{_format_progress_finish_time(progress.eta_s)}"
    )


def _stringify_env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _build_spawn_profiling_env(
    env_overrides: dict[str, str] | None,
    guided_env_overrides: dict[str, str] | None = None,
) -> dict[str, str] | None:
    if not env_overrides:
        profiling_env = os.environ.copy()
    else:
        profiling_env = os.environ.copy()
        profiling_env.update({key: str(value) for key, value in env_overrides.items()})

    profiling_env.setdefault("OMP_DISPLAY_ENV", "true")
    profiling_env.setdefault("OMP_DISPLAY_AFFINITY", "true")
    if guided_env_overrides:
        for key, value in guided_env_overrides.items():
            profiling_env.setdefault(key, _stringify_env_value(value))
    return profiling_env


def _extract_hypothesis_profiles(
    structured_context_bundle: dict[str, object] | None,
    selected_hypothesis_refs: list[str] | None = None,
    *,
    max_profiles: int | None = 3,
) -> list[dict[str, Any]]:
    if not isinstance(structured_context_bundle, dict):
        return []
    raw_profiles = structured_context_bundle.get("hypothesis_profiles")
    if not isinstance(raw_profiles, list):
        return []
    profiles = [item for item in raw_profiles if isinstance(item, dict)]
    if selected_hypothesis_refs:
        selected = {item.strip() for item in selected_hypothesis_refs if item.strip()}
        prioritized = [
            item for item in profiles if str(item.get("hypothesis_id", "")).strip() in selected
        ]
        if prioritized:
            return prioritized if max_profiles is None else prioritized[:max_profiles]
    return profiles if max_profiles is None else profiles[:max_profiles]


def _collect_contract_report_env(hypothesis_profiles: list[dict[str, Any]]) -> dict[str, str]:
    report_env: dict[str, str] = {}
    for profile in hypothesis_profiles:
        report_controls = profile.get("report_controls")
        if not isinstance(report_controls, list):
            continue
        for control_entry in report_controls:
            if not isinstance(control_entry, dict):
                continue
            control = control_entry.get("control")
            if not isinstance(control, dict):
                continue
            name = str(control.get("name", "")).strip()
            activation_value = control.get("activation_value")
            if activation_value is None:
                effect = control.get("effect")
                if isinstance(effect, dict):
                    effect_kind = str(effect.get("kind", "")).strip()
                    if effect_kind == "enable":
                        activation_value = True
                    elif effect_kind == "disable":
                        activation_value = False
                    elif effect_kind == "set":
                        effect_value = effect.get("value")
                        if effect_value != "input":
                            activation_value = effect_value
            if not name or activation_value is None:
                continue
            report_env.setdefault(name, _stringify_env_value(activation_value))
    return report_env


def _virtual_probe_search_terms(signal: dict[str, Any], virtual_probe: str) -> list[str]:
    terms = [
        virtual_probe,
        str(signal.get("metric", "")).strip(),
        str(signal.get("indicates", "")).strip(),
    ]
    return [term for term in terms if term]


def _build_contract_hypothesis_validation_script(
    hypothesis_profiles: list[dict[str, Any]],
    profiling_json: dict[str, Any],
    stdout: str,
    stderr: str,
    report_env: dict[str, str],
    active_env: dict[str, str],
) -> str:
    payload = {
        "plans": hypothesis_profiles,
        "profiling": {
            "runtime_output": profiling_json.get("runtime_output", {}),
            "refined_runtime_maps": profiling_json.get("refined_runtime_maps", {}),
            "collect_profile_data": profiling_json.get("collect_profile_data", {}),
            "virtual_probe_results": profiling_json.get("virtual_probe_results", {}),
        },
        "combined_output": (stdout or "") + ("\n" + stderr if stderr else ""),
        "report_env": report_env,
        "active_env": active_env,
    }
    payload_json = json.dumps(payload)
    return (
        "import json\n"
        f"payload = json.loads({json.dumps(payload_json)})\n"
        "plans = payload.get('plans', [])\n"
        "profiling = payload.get('profiling', {})\n"
        "runtime_output = profiling.get('runtime_output', {}) if isinstance(\n"
        "    profiling, dict\n"
        ") else {}\n"
        "refined_runtime_maps = profiling.get('refined_runtime_maps', {}) if isinstance(\n"
        "    profiling, dict\n"
        ") else {}\n"
        "collect_profile_data = profiling.get('collect_profile_data', {}) if isinstance(\n"
        "    profiling, dict\n"
        ") else {}\n"
        "probe_results = profiling.get('virtual_probe_results', {}) if isinstance(\n"
        "    profiling, dict\n"
        ") else {}\n"
        "if isinstance(probe_results, list):\n"
        "    simplified_probe_results = {}\n"
        "    for item in probe_results:\n"
        "        if isinstance(item, dict):\n"
        "            simplified_probe_results.update(item)\n"
        "    probe_results = simplified_probe_results\n"
        "cpu_collector = collect_profile_data.get('cpu', {}) if isinstance(\n"
        "    collect_profile_data, dict\n"
        ") else {}\n"
        "cpu_data = cpu_collector.get('data', {}) if isinstance(cpu_collector, dict) else {}\n"
        "memory_collector = collect_profile_data.get('memory', {}) if isinstance(\n"
        "    collect_profile_data, dict\n"
        ") else {}\n"
        "memory_data = memory_collector.get('data', {}) if isinstance(\n"
        "    memory_collector, dict\n"
        ") else {}\n"
        "combined_output = str(payload.get('combined_output', ''))\n"
        "report_env = payload.get('report_env', {}) if isinstance(\n"
        "    payload.get('report_env', {}), dict\n"
        ") else {}\n"
        "active_env = payload.get('active_env', {}) if isinstance(\n"
        "    payload.get('active_env', {}), dict\n"
        ") else {}\n"
        "def _parse_cpu_list(text):\n"
        "    if not isinstance(text, str) or not text.strip():\n"
        "        return None\n"
        "    total = 0\n"
        "    for part in text.split(','):\n"
        "        part = part.strip()\n"
        "        if not part:\n"
        "            continue\n"
        "        if '-' in part:\n"
        "            start, end = part.split('-', 1)\n"
        "            if start.strip().isdigit() and end.strip().isdigit():\n"
        "                total += int(end) - int(start) + 1\n"
        "        elif part.isdigit():\n"
        "            total += 1\n"
        "    return total or None\n"
        "def _signal_result(signal_entry):\n"
        "    signal = signal_entry.get('signal', {}) if isinstance(signal_entry, dict) else {}\n"
        "    resolved_probe = signal_entry.get('resolved_probe', {}) if isinstance(\n"
        "        signal_entry, dict\n"
        "    ) else {}\n"
        "    virtual_probe = str(signal.get('virtual_probe', ''))\n"
        "    result = {\n"
        "        'ref': signal_entry.get('ref'),\n"
        "        'metric': signal.get('metric'),\n"
        "        'virtual_probe': virtual_probe,\n"
        "        'observed': False,\n"
        "        'source': 'unsupported',\n"
        "        'value': None,\n"
        "        'excerpt': [],\n"
        "    }\n"
        "    lines = combined_output.splitlines()\n"
        "    probe_result = probe_results.get(virtual_probe, {}) if isinstance(\n"
        "        probe_results, dict\n"
        "    ) else {}\n"
        "    if probe_result is not None and not isinstance(probe_result, dict):\n"
        "        result['observed'] = True\n"
        "        result['source'] = 'virtual_probe_results'\n"
        "        result['value'] = probe_result\n"
        "        return result\n"
        "    if isinstance(probe_result, dict) and probe_result:\n"
        "        result['observed'] = bool(probe_result.get('observed', False))\n"
        "        result['source'] = str(probe_result.get('source', 'toolbox_script'))\n"
        "        result['value'] = probe_result.get('value')\n"
        "        excerpt = probe_result.get('excerpt', [])\n"
        "        result['excerpt'] = excerpt[:8] if isinstance(excerpt, list) else []\n"
        "        if result['observed'] or result['source'] in (\n"
        "            'unavailable', 'probe_error', 'os_missing', 'runtime_output_missing'\n"
        "        ):\n"
        "            return result\n"
        "    terms = [\n"
        "        virtual_probe.lower(),\n"
        "        str(signal.get('metric', '')).lower(),\n"
        "        str(signal.get('indicates', '')).lower(),\n"
        "    ]\n"
        "    searchable_terms = [term for term in terms if term]\n"
        "    if isinstance(resolved_probe, dict) and resolved_probe.get(\n"
        "        'availability'\n"
        "    ) == 'unavailable':\n"
        "        result['source'] = 'unavailable'\n"
        "        return result\n"
        "    matches = [\n"
        "        line.strip()\n"
        "        for line in lines\n"
        "        if any(term in line.lower() for term in searchable_terms)\n"
        "    ]\n"
        "    result['observed'] = bool(matches)\n"
        "    result['source'] = 'runtime_output' if matches else 'unsupported'\n"
        "    result['value'] = matches[:8] if matches else None\n"
        "    result['excerpt'] = matches[:8]\n"
        "    return result\n"
        "profiles = []\n"
        "for plan in plans:\n"
        "    signals = plan.get('validation_signals', []) if isinstance(plan, dict) else []\n"
        "    signal_results = [\n"
        "        _signal_result(signal) for signal in signals if isinstance(signal, dict)\n"
        "    ]\n"
        "    findings = [\n"
        "        f\"{item['ref']} observed via {item['source']}\"\n"
        "        for item in signal_results\n"
        "        if item.get('observed')\n"
        "    ]\n"
        "    missing = [item.get('ref') for item in signal_results if not item.get('observed')]\n"
        "    profiles.append({\n"
        "        'hypothesis_id': plan.get('hypothesis_id'),\n"
        "        'resource': plan.get('resource'),\n"
        "        'libraries': plan.get('contracts'),\n"
        "        'report_env': plan.get('report_controls', []),\n"
        "        'signal_results': signal_results,\n"
        "        'findings': findings,\n"
        "        'missing_signals': missing,\n"
        "    })\n"
        "print(json.dumps({'profiles': profiles, 'report_env': report_env}))\n"
    )


async def _run_contract_hypothesis_validation(
    agent: SysPilotAgent,
    hypothesis_profiles: list[dict[str, Any]],
    profiling_json: dict[str, Any],
    stdout: str,
    stderr: str,
    report_env: dict[str, str],
    active_env: dict[str, str],
) -> tuple[dict[str, Any], int, list[str]]:
    if not hypothesis_profiles:
        return {}, 0, []

    script = _build_contract_hypothesis_validation_script(
        hypothesis_profiles,
        profiling_json,
        stdout,
        stderr,
        report_env,
        active_env,
    )
    try:
        script_result = await agent._execute_tool(
            "skill_dynamic_script",
            {"script_type": "python", "script": script},
        )
    except Exception as exc:
        return (
            {
                "success": False,
                "error": str(exc),
                "profiles": [],
                "report_env": report_env,
            },
            1,
            ["skill_dynamic_script.contract_validation"],
        )

    result_payload: dict[str, Any] = {
        "success": bool(script_result.get("success", False))
        if isinstance(script_result, dict)
        else False,
        "error": str(script_result.get("error", "")).strip()
        if isinstance(script_result, dict)
        else "",
        "summary": str(script_result.get("summary", "")).strip()
        if isinstance(script_result, dict)
        else "",
        "profiles": [],
        "report_env": report_env,
    }
    if isinstance(script_result, dict):
        output = str(script_result.get("output", "")).strip()
        if output:
            parsed = _extract_json_object(output)
            if parsed and isinstance(parsed.get("profiles"), list):
                result_payload["profiles"] = parsed.get("profiles", [])
                parsed_report_env = parsed.get("report_env")
                if isinstance(parsed_report_env, dict):
                    result_payload["report_env"] = {
                        str(key): _stringify_env_value(value)
                        for key, value in parsed_report_env.items()
                        if str(key).strip()
                    }
        if not result_payload["profiles"] and output:
            result_payload["raw_output_excerpt"] = output[:1000]
    return result_payload, 1, ["skill_dynamic_script.contract_validation"]


def _summarize_spawn_runtime_output(
    stdout: str, stderr: str, *, max_chars: int = 2000
) -> dict[str, object]:
    combined = (stdout or "") + ("\n" + stderr if stderr else "")
    excerpt = combined[:max_chars]
    thread_requested = re.search(r"Number of Threads requested\s*=\s*(\d+)", combined)
    thread_counted = re.search(r"Number of Threads counted\s*=\s*(\d+)", combined)
    openmp_env_present = "OPENMP DISPLAY ENVIRONMENT BEGIN" in combined
    affinity_lines = [
        line.strip()
        for line in combined.splitlines()
        if "affinity" in line.lower() or "place" in line.lower()
    ]
    return {
        "stdout_excerpt": excerpt,
        "threads_requested": int(thread_requested.group(1)) if thread_requested else None,
        "threads_counted": int(thread_counted.group(1)) if thread_counted else None,
        "openmp_display_env_present": openmp_env_present,
        "affinity_excerpt": "\n".join(affinity_lines[:20]),
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    if not candidate:
        return None

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def _to_token_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return shlex.split(value)

    if isinstance(value, list):
        tokens: list[str] = []
        for item in value:
            token = str(item).strip()
            if token:
                tokens.append(token)
        return tokens

    return []


_SPAWN_HELP_OPTION_CACHE: dict[tuple[str, ...], set[str] | None] = {}

_LLAMA_BENCH_OPTION_ALIASES: dict[str, str] = {
    "-m": "model",
    "--model": "model",
    "-p": "n_prompt",
    "--n-prompt": "n_prompt",
    "-n": "n_gen",
    "--n-gen": "n_gen",
    "-pg": "pp_tg",
    "-d": "n_depth",
    "--n-depth": "n_depth",
    "-embd": "embeddings",
    "--embeddings": "embeddings",
    "-r": "repetitions",
    "--repetitions": "repetitions",
    "-o": "output",
    "--output": "output",
    "-oe": "output_err",
    "--output-err": "output_err",
    "-t": "threads",
    "--threads": "threads",
    "-C": "cpu_mask",
    "--cpu-mask": "cpu_mask",
    "--cpu-strict": "cpu_strict",
    "--poll": "poll",
    "--numa": "numa",
    "--prio": "prio",
    "-b": "batch_size",
    "--batch-size": "batch_size",
    "-ub": "ubatch_size",
    "--ubatch-size": "ubatch_size",
    "-ctk": "cache_type_k",
    "--cache-type-k": "cache_type_k",
    "-ctv": "cache_type_v",
    "--cache-type-v": "cache_type_v",
    "-ngl": "n_gpu_layers",
    "--n-gpu-layers": "n_gpu_layers",
    "-ncmoe": "n_cpu_moe",
    "--n-cpu-moe": "n_cpu_moe",
    "-sm": "split_mode",
    "--split-mode": "split_mode",
    "-mg": "main_gpu",
    "--main-gpu": "main_gpu",
    "-nkvo": "no_kv_offload",
    "--no-kv-offload": "no_kv_offload",
    "-fa": "flash_attn",
    "--flash-attn": "flash_attn",
    "-dev": "device",
    "--device": "device",
    "-mmp": "mmap",
    "--mmap": "mmap",
    "-ts": "tensor_split",
    "--tensor-split": "tensor_split",
    "-ot": "override_tensors",
    "--override-tensors": "override_tensors",
    "-nopo": "no_op_offload",
    "--no-op-offload": "no_op_offload",
    "-rpc": "rpc",
    "--rpc": "rpc",
}

_LLAMA_BENCH_FIXED_OPTION_KEYS = {
    "model",
    "n_prompt",
    "n_gen",
    "pp_tg",
    "n_depth",
    "embeddings",
    "repetitions",
    "output",
    "output_err",
}

_LLAMA_BENCH_MUTABLE_OPTION_KEYS = {
    "threads",
    "cpu_mask",
    "cpu_strict",
    "poll",
    "numa",
    "prio",
    "batch_size",
    "ubatch_size",
    "cache_type_k",
    "cache_type_v",
    "n_gpu_layers",
    "n_cpu_moe",
    "split_mode",
    "main_gpu",
    "no_kv_offload",
    "flash_attn",
    "device",
    "mmap",
    "tensor_split",
    "override_tensors",
    "no_op_offload",
    "rpc",
}

_LLAMA_BENCH_SINGLE_VALUE_OPTION_KEYS = {
    "threads",
    "batch_size",
    "ubatch_size",
}


def _looks_like_numeric_token(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _is_option_token(token: str) -> bool:
    return token.startswith("-") and token != "-" and not _looks_like_numeric_token(token)


def _is_benchmark_sweep_value(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    if "," in stripped:
        return True
    return bool(re.fullmatch(r"-?\d+\s*-\s*-?\d+(?:[+*]-?\d+)?", stripped))


def _parse_cli_command_signature(
    command_tokens: list[str],
) -> tuple[dict[str, list[str]], list[str]]:
    options: dict[str, list[str]] = {}
    positionals: list[str] = []

    idx = 1
    while idx < len(command_tokens):
        token = command_tokens[idx]
        if not _is_option_token(token):
            positionals.append(token)
            idx += 1
            continue

        option_name = token
        option_value = ""
        if token.startswith("--") and "=" in token:
            option_name, option_value = token.split("=", 1)
        elif idx + 1 < len(command_tokens):
            next_token = command_tokens[idx + 1]
            if not _is_option_token(next_token) or _looks_like_numeric_token(next_token):
                option_value = next_token
                idx += 1

        options.setdefault(option_name, []).append(option_value)
        idx += 1

    return options, positionals


def _baseline_control_order(
    baseline_command_tokens: Sequence[str] | None = None,
    baseline_env_overrides: dict[str, str] | None = None,
) -> dict[str, int]:
    order: dict[str, int] = {}

    def _remember(name: str) -> None:
        if name and name not in order:
            order[name] = len(order)

    if baseline_env_overrides:
        for env_name in baseline_env_overrides:
            _remember(str(env_name))

    command_tokens = list(baseline_command_tokens or [])
    normalized_command, inline_env = _extract_inline_env_assignments(command_tokens)
    for env_name in inline_env:
        _remember(str(env_name))

    idx = 1
    while idx < len(normalized_command):
        token = normalized_command[idx]
        if not _is_option_token(token):
            idx += 1
            continue
        option_name = token.split("=", 1)[0]
        _remember(option_name)
        idx += 1
        if idx < len(normalized_command):
            next_token = normalized_command[idx]
            if not _is_option_token(next_token) or _looks_like_numeric_token(next_token):
                idx += 1

    return order


def _control_sort_key(
    control_name: str,
    baseline_order: dict[str, int],
    fallback_index: int = 0,
) -> tuple[int, int | str, int]:
    normalized = control_name.strip()
    if normalized.startswith("env."):
        normalized = normalized[4:]
    normalized = normalized.split("=", 1)[0]
    if normalized in baseline_order:
        return (0, baseline_order[normalized], fallback_index)
    return (1, normalized, fallback_index)


def _help_target_tokens(command_tokens: list[str]) -> tuple[str, ...]:
    if not command_tokens:
        return tuple()

    help_target = [command_tokens[0]]
    idx = 1
    while idx < len(command_tokens):
        token = command_tokens[idx]
        if _is_option_token(token):
            break
        help_target.append(token)
        idx += 1
    return tuple(help_target)


def _first_positional_index(command_tokens: list[str]) -> int | None:
    idx = 1
    while idx < len(command_tokens):
        token = command_tokens[idx]
        if not _is_option_token(token):
            return idx

        idx += 1
        if idx < len(command_tokens):
            next_token = command_tokens[idx]
            if not _is_option_token(next_token):
                idx += 1

    return None


def _wrapper_child_start_index(command_tokens: list[str]) -> int | None:
    if len(_help_target_tokens(command_tokens)) > 1:
        return None
    return _first_positional_index(command_tokens)


def _is_env_assignment_token(token: str) -> bool:
    if "=" not in token:
        return False
    name, _ = token.split("=", 1)
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name))


def _extract_inline_env_assignments(command_tokens: list[str]) -> tuple[list[str], dict[str, str]]:
    if not command_tokens:
        return [], {}

    saw_env_prefix = command_tokens[0] == "env"
    idx = 1 if saw_env_prefix else 0
    env_overrides: dict[str, str] = {}
    assignment_start = idx
    while idx < len(command_tokens) and _is_env_assignment_token(command_tokens[idx]):
        name, value = command_tokens[idx].split("=", 1)
        env_overrides[name] = value
        idx += 1

    remaining = command_tokens[idx:]
    if not remaining or remaining[0].startswith("-"):
        return list(command_tokens), {}

    if not saw_env_prefix and idx == assignment_start:
        return list(command_tokens), {}

    return remaining, env_overrides


def _extract_runtime_unsupported_options(stdout: str, stderr: str) -> list[str]:
    text = f"{stderr}\n{stdout}"
    patterns = [
        r"option\s+['\"]?(--?[A-Za-z0-9][A-Za-z0-9-]*)['\"]?\s+is\s+not\s+supported",
        r"(?:unknown|unrecognized|invalid)\s+(?:command\s+line\s+)?option\s+['\"]?(--?[A-Za-z0-9][A-Za-z0-9-]*)['\"]?",
        r"option\s+['\"]?(--?[A-Za-z0-9][A-Za-z0-9-]*)['\"]?\s+(?:not\s+recognized|unrecognized|unknown|invalid)",
        r"unrecognized\s+argument[s]?:\s+(--?[A-Za-z0-9][A-Za-z0-9-]*)",
    ]
    matches: list[str] = []
    for pattern in patterns:
        for option_name in re.findall(pattern, text, flags=re.IGNORECASE):
            normalized = str(option_name).strip()
            if normalized and normalized not in matches:
                matches.append(normalized)
    return matches


def _strip_option_from_tokens(tokens: list[str], option_name: str) -> tuple[list[str], bool]:
    stripped: list[str] = []
    idx = 0
    removed = False
    while idx < len(tokens):
        token = tokens[idx]
        if not removed and token == option_name:
            removed = True
            idx += 1
            while idx < len(tokens) and not _is_option_token(tokens[idx]):
                idx += 1
            continue
        stripped.append(token)
        idx += 1
    return stripped, removed


def _candidate_actions_without_unsupported_options(
    base_command: list[str],
    actions: list[dict[str, Any]],
    unsupported_options: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    updated_actions = [dict(action) for action in actions]
    removed_options: list[str] = []
    for option_name in unsupported_options:
        for idx, action in enumerate(updated_actions):
            config_type = str(action.get("config_type", "")).strip().lower()
            if config_type not in {"command", "command_prefix"}:
                continue
            tokens = _to_token_list(action.get("new_value"))
            if not tokens or option_name not in tokens:
                continue

            if config_type == "command":
                base_child_start = _wrapper_child_start_index(base_command)
                candidate_child_start = _wrapper_child_start_index(tokens)
                removed = False
                new_tokens = list(tokens)
                if base_child_start is not None and candidate_child_start is not None:
                    wrapper_tokens = tokens[:candidate_child_start]
                    child_tokens = tokens[candidate_child_start:]
                    if option_name in wrapper_tokens:
                        new_wrapper, removed = _strip_option_from_tokens(
                            wrapper_tokens, option_name
                        )
                        new_tokens = new_wrapper + child_tokens
                    elif option_name in child_tokens:
                        new_child, removed = _strip_option_from_tokens(child_tokens, option_name)
                        new_tokens = wrapper_tokens + new_child
                else:
                    new_tokens, removed = _strip_option_from_tokens(tokens, option_name)
            else:
                new_tokens, removed = _strip_option_from_tokens(tokens, option_name)

            if not removed:
                continue
            removed_options.append(option_name)
            if not new_tokens:
                updated_actions.pop(idx)
            else:
                action["new_value"] = new_tokens
            break
    return updated_actions, removed_options


def _retry_spawn_candidate_without_unsupported_options(
    base_command: list[str],
    actions: list[dict[str, Any]],
    *,
    base_env_overrides: dict[str, str] | None,
    unsupported_options: list[str],
    excluded_config_names: Sequence[str] | None = None,
    tunable_whitelist_names: Sequence[str] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not unsupported_options:
        return None, []

    baseline_options, _ = _parse_cli_command_signature(base_command)
    retryable_options = [option for option in unsupported_options if option not in baseline_options]
    if not retryable_options:
        return None, []

    retry_actions, removed_options = _candidate_actions_without_unsupported_options(
        base_command,
        actions,
        retryable_options,
    )
    if not removed_options:
        return None, []

    tuned_cmd, env_overrides, applied_actions, skipped_actions = _build_spawn_execution_plan(
        base_command,
        retry_actions,
        base_env_overrides=base_env_overrides,
        excluded_config_names=excluded_config_names,
        tunable_whitelist_names=tunable_whitelist_names,
    )
    if not tuned_cmd:
        return None, removed_options

    return (
        {
            "cmd": tuned_cmd,
            "env_overrides": env_overrides,
            "label": "; ".join(applied_actions) if applied_actions else "(retry)",
            "skipped_actions": skipped_actions,
            "actions": retry_actions,
        },
        removed_options,
    )


def _help_probe_commands(help_target: tuple[str, ...]) -> list[list[str]]:
    if not help_target:
        return []

    commands: list[list[str]] = []
    if len(help_target) == 1:
        commands.append([*help_target, "--help", "all"])
    commands.extend([[*help_target, "--help"], [*help_target, "-h"]])
    return commands


def _discover_supported_cli_options(help_target: tuple[str, ...]) -> set[str] | None:
    cached = _SPAWN_HELP_OPTION_CACHE.get(help_target)
    if help_target in _SPAWN_HELP_OPTION_CACHE:
        return cached

    if not help_target:
        return None

    discovered_options: set[str] = set()

    for help_command in _help_probe_commands(help_target):
        try:
            completed = subprocess.run(
                help_command,
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (FileNotFoundError, PermissionError, OSError, subprocess.TimeoutExpired):
            continue

        help_text = (completed.stdout or "") + "\n" + (completed.stderr or "")
        if not help_text.strip():
            continue

        matches = re.findall(
            r"(?<![A-Za-z0-9_-])(\-\-[A-Za-z0-9][A-Za-z0-9-]*|\-[A-Za-z][A-Za-z0-9-]*)",
            help_text,
        )
        options = set(matches)
        if options:
            discovered_options.update(options)

    if discovered_options:
        _SPAWN_HELP_OPTION_CACHE[help_target] = discovered_options
        return discovered_options

    _SPAWN_HELP_OPTION_CACHE[help_target] = None
    return None


def _validate_cli_options_against_help(
    command_tokens: list[str],
    baseline_command_tokens: list[str] | None = None,
) -> list[str]:
    if not command_tokens:
        return ["candidate command was empty"]

    help_target = _help_target_tokens(command_tokens)
    supported_options = _discover_supported_cli_options(help_target)
    if not supported_options and len(help_target) > 1:
        supported_options = _discover_supported_cli_options((command_tokens[0],))
    if not supported_options:
        return []

    candidate_options, _ = _parse_cli_command_signature(command_tokens)
    baseline_options, _ = (
        _parse_cli_command_signature(baseline_command_tokens)
        if baseline_command_tokens
        else ({}, [])
    )
    executable_name = Path(command_tokens[0]).name

    issues: list[str] = []
    for option_name in sorted(candidate_options.keys()):
        if option_name in baseline_options:
            continue
        if option_name not in supported_options:
            issues.append(
                f"option '{option_name}' is not supported by executable '{executable_name}'"
            )

    return issues


def _validate_wrapper_command_options(
    base_command: list[str],
    candidate_command: list[str],
) -> list[str]:
    base_child_start = _wrapper_child_start_index(base_command)
    candidate_child_start = _wrapper_child_start_index(candidate_command)
    if base_child_start is None or candidate_child_start is None:
        return _validate_cli_options_against_help(candidate_command, base_command)

    base_wrapper = base_command[:base_child_start]
    base_child = base_command[base_child_start:]
    candidate_wrapper = candidate_command[:candidate_child_start]
    candidate_child = candidate_command[candidate_child_start:]

    issues = _validate_cli_options_against_help(candidate_wrapper, base_wrapper)
    issues.extend(_validate_cli_options_against_help(candidate_child, base_child))
    return issues


def _canonicalize_llama_bench_options(
    option_map: dict[str, list[str]],
) -> tuple[dict[str, list[str]], list[str]]:
    canonical: dict[str, list[str]] = {}
    unknown: list[str] = []

    for option_name, values in option_map.items():
        canonical_name = _LLAMA_BENCH_OPTION_ALIASES.get(option_name)
        if not canonical_name:
            unknown.append(option_name)
            continue
        canonical.setdefault(canonical_name, []).extend(values)

    return canonical, unknown


def _validate_llama_bench_fairness(
    base_command: list[str],
    candidate_command: list[str],
) -> list[str]:
    base_options, base_positionals = _parse_cli_command_signature(base_command)
    candidate_options, candidate_positionals = _parse_cli_command_signature(candidate_command)

    issues: list[str] = []

    if candidate_positionals != base_positionals:
        issues.append("must keep positional arguments unchanged for fair benchmark comparison")

    base_canonical, _ = _canonicalize_llama_bench_options(base_options)
    candidate_canonical, unknown_candidate_options = _canonicalize_llama_bench_options(
        candidate_options
    )

    for option_name in sorted(unknown_candidate_options):
        issues.append(f"uses unsupported llama-bench option '{option_name}'")

    for fixed_key in sorted(_LLAMA_BENCH_FIXED_OPTION_KEYS):
        baseline_values = base_canonical.get(fixed_key)
        candidate_values = candidate_canonical.get(fixed_key)

        if baseline_values is None:
            if candidate_values is not None:
                issues.append(f"must not add workload-defining option '{fixed_key}'")
            continue

        if candidate_values is None:
            issues.append(f"must keep '{fixed_key}' fixed to baseline value(s)")
            continue

        if candidate_values != baseline_values:
            issues.append(f"must keep '{fixed_key}' fixed to baseline value(s)")

    for canonical_key in sorted(candidate_canonical.keys()):
        if canonical_key in base_canonical:
            continue
        if canonical_key not in _LLAMA_BENCH_MUTABLE_OPTION_KEYS:
            issues.append(f"must not add non-tuning option '{canonical_key}'")

    for option_key in sorted(_LLAMA_BENCH_SINGLE_VALUE_OPTION_KEYS):
        for option_value in candidate_canonical.get(option_key, []):
            if _is_benchmark_sweep_value(option_value):
                issues.append(f"must use a single concrete value for '{option_key}' per trial")
                break

    return issues


def _validate_spawn_command_rewrite(
    base_command: list[str],
    candidate_command: list[str],
) -> list[str]:
    if not base_command or not candidate_command:
        return ["command rewrite validation received an empty command"]

    issues: list[str] = []
    _, base_positionals = _parse_cli_command_signature(base_command)
    _, candidate_positionals = _parse_cli_command_signature(candidate_command)
    if candidate_positionals != base_positionals:
        issues.append("must keep positional arguments unchanged for fair benchmark comparison")
    issues.extend(_validate_wrapper_command_options(base_command, candidate_command))

    executable_name = Path(base_command[0]).name
    if executable_name == "llama-bench":
        issues.extend(_validate_llama_bench_fairness(base_command, candidate_command))

    return issues


def _extract_raw_spawn_candidates(response: str) -> list[object]:
    parsed = _extract_json_object(response)
    if not parsed:
        return []
    candidates = parsed.get("candidates")
    return list(candidates) if isinstance(candidates, list) else []


def _parse_spawn_actions(
    response: str,
    *,
    baseline_command_tokens: Sequence[str] | None = None,
    baseline_env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    parsed = _extract_json_object(response)
    if not parsed:
        return {
            "valid": False,
            "error": "Agent response did not contain a valid JSON object",
            "summary": "",
            "manual_citation": "",
            "evidence": [],
            "actions": [],
            "candidates": [],
            "raw_candidates": [],
            "skipped_actions": [],
        }

    summary = str(parsed.get("summary", "")).strip()
    manual_citation = str(parsed.get("manual_citation", "")).strip()
    evidence_raw = parsed.get("evidence", [])

    evidence: list[str] = []
    if isinstance(evidence_raw, list):
        for item in evidence_raw:
            value = str(item).strip()
            if value:
                evidence.append(value)

    hypothesis_refs_raw = parsed.get("hypothesis_refs", [])
    hypothesis_refs: list[str] = []
    if isinstance(hypothesis_refs_raw, list):
        for item in hypothesis_refs_raw:
            value = str(item).strip()
            if value:
                hypothesis_refs.append(value)

    if "candidates" not in parsed:
        return {
            "valid": False,
            "error": "Field 'candidates' must be present and must be a list",
            "summary": summary,
            "manual_citation": manual_citation,
            "evidence": evidence,
            "hypothesis_refs": hypothesis_refs,
            "actions": [],
            "candidates": [],
            "raw_candidates": [],
            "skipped_actions": [],
        }

    candidates_raw = parsed.get("candidates", [])
    if not isinstance(candidates_raw, list):
        return {
            "valid": False,
            "error": "Field 'candidates' must be a list",
            "summary": summary,
            "manual_citation": manual_citation,
            "evidence": evidence,
            "hypothesis_refs": hypothesis_refs,
            "actions": [],
            "candidates": [],
            "raw_candidates": [],
            "skipped_actions": [],
        }

    candidate_fields = {"changes"}
    normalized_actions: list[dict[str, Any]] = []
    normalized_candidates: list[dict[str, Any]] = []
    skipped_actions: list[str] = []
    baseline_order = _baseline_control_order(baseline_command_tokens, baseline_env_overrides)

    for candidate_idx, candidate in enumerate(candidates_raw):
        if not isinstance(candidate, dict):
            skipped_actions.append(f"candidate[{candidate_idx}] is not an object")
            continue

        candidate_reason = str(candidate.get("reason", "")).strip()
        candidate_manual_citation = str(candidate.get("manual_citation", manual_citation)).strip()
        candidate_actions: list[dict[str, Any]] = []

        unknown_candidate_fields = sorted(
            key
            for key in candidate.keys()
            if key not in {"reason", "manual_citation", *candidate_fields}
        )
        for field_name in unknown_candidate_fields:
            skipped_actions.append(
                f"candidate[{candidate_idx}] has unsupported field '{field_name}'"
            )

        changes_raw = candidate.get("changes", {})
        if not isinstance(changes_raw, dict):
            skipped_actions.append(f"candidate[{candidate_idx}].changes must be an object")
            changes_raw = {}

        command_delta_tokens: list[str] = []
        sorted_changes = sorted(
            enumerate(changes_raw.items()),
            key=lambda item: _control_sort_key(str(item[1][0]).strip(), baseline_order, item[0]),
        )
        for _, (raw_name, raw_value) in sorted_changes:
            control_name = str(raw_name).strip()
            if not control_name or "|" in control_name:
                skipped_actions.append(
                    f"candidate[{candidate_idx}].changes has invalid control name '{control_name}'"
                )
                continue
            if isinstance(raw_value, (dict, list)):
                skipped_actions.append(
                    f"candidate[{candidate_idx}].changes['{control_name}'] must be a "
                    "concrete scalar value"
                )
                continue
            control_value = str(raw_value).strip()
            if "|" in control_value:
                skipped_actions.append(
                    f"candidate[{candidate_idx}].changes['{control_name}'] uses unsupported "
                    "bundled value syntax"
                )
                continue
            if _is_option_token(control_name):
                command_delta_tokens.extend([control_name, control_value])
                continue

            env_name = control_name
            if "=" in env_name or " " in env_name:
                skipped_actions.append(
                    f"candidate[{candidate_idx}].changes has invalid env var name '{env_name}'"
                )
                continue
            action = {
                "config_type": "env_var",
                "parameter": env_name,
                "new_value": control_value,
                "reason": candidate_reason,
                "manual_citation": candidate_manual_citation,
            }
            candidate_actions.append(action)
            normalized_actions.append(action)

        if command_delta_tokens:
            action = {
                "config_type": "command",
                "parameter": "",
                "new_value": command_delta_tokens,
                "reason": candidate_reason,
                "manual_citation": candidate_manual_citation,
            }
            candidate_actions.append(action)
            normalized_actions.append(action)

        if not candidate_actions:
            skipped_actions.append(f"candidate[{candidate_idx}] has no valid changes")
            continue

        normalized_candidates.append(
            {
                "source_candidate_index": candidate_idx,
                "reason": candidate_reason,
                "manual_citation": candidate_manual_citation,
                "actions": candidate_actions,
            }
        )

    return {
        "valid": True,
        "error": "",
        "summary": summary,
        "manual_citation": manual_citation,
        "evidence": evidence,
        "hypothesis_refs": hypothesis_refs,
        "actions": normalized_actions,
        "candidates": normalized_candidates,
        "raw_candidates": _extract_raw_spawn_candidates(response),
        "skipped_actions": skipped_actions,
    }


def _format_spawn_action_preview(action: dict[str, Any]) -> str:
    config_type = str(action.get("config_type", "")).strip().lower()
    parameter = str(action.get("parameter", "")).strip()
    new_value = action.get("new_value")

    if config_type == "env_var":
        value = "" if new_value is None else str(new_value).strip()
        if parameter and value:
            return f"{parameter}={value}"
        if parameter:
            return parameter
        return value

    tokens = _to_token_list(new_value)
    token_display = " ".join(tokens)
    fallback = "" if new_value is None else str(new_value).strip()
    value_display = token_display or fallback

    if config_type == "command":
        return value_display
    if config_type == "command_prefix":
        return f"prefix {value_display}".strip()
    if parameter and value_display:
        return f"{parameter}={value_display}"
    return value_display or parameter or config_type


def _recommendation_next_action_display(actions: list[dict[str, Any]]) -> str:
    if not actions:
        return ""

    preview = _format_spawn_action_preview(actions[0]).strip()
    if not preview:
        return ""

    remaining = len(actions) - 1
    if remaining > 0:
        return f"{preview} (+{remaining} more)"
    return preview


def _matching_excluded_config_name(name: str, excluded_config_names: Sequence[str]) -> str | None:
    stripped_name = name.strip()
    if not stripped_name:
        return None
    comparable_name = stripped_name.lstrip("-")
    for excluded_name in excluded_config_names:
        stripped_excluded = str(excluded_name).strip()
        if not stripped_excluded:
            continue
        if stripped_name == stripped_excluded:
            return stripped_excluded
        if (stripped_name.startswith("-") or stripped_excluded.startswith("-")) and (
            comparable_name == stripped_excluded.lstrip("-")
        ):
            return stripped_excluded
    return None


def _excluded_inline_env_changes(
    env_updates: dict[str, str],
    base_env_overrides: dict[str, str],
    excluded_config_names: Sequence[str],
) -> list[str]:
    changed: list[str] = []
    for name, value in env_updates.items():
        matched = _matching_excluded_config_name(name, excluded_config_names)
        if matched is None:
            continue
        if base_env_overrides.get(name) == value:
            continue
        changed.append(name)
    return changed


def _excluded_env_assignments_in_tokens(
    tokens: list[str],
    excluded_config_names: Sequence[str],
) -> list[str]:
    changed: list[str] = []
    idx = 1 if tokens and tokens[0] == "env" else 0
    while idx < len(tokens) and _is_env_assignment_token(tokens[idx]):
        name, _ = tokens[idx].split("=", 1)
        if _matching_excluded_config_name(name, excluded_config_names) is not None:
            changed.append(name)
        idx += 1
    return changed


def _excluded_option_tokens_in_tokens(
    tokens: list[str],
    excluded_config_names: Sequence[str],
) -> list[str]:
    changed: list[str] = []
    for token in tokens:
        if not _is_option_token(token):
            continue
        option_name = token.split("=", 1)[0]
        if _matching_excluded_config_name(option_name, excluded_config_names) is not None:
            changed.append(option_name)
    return changed


def _excluded_command_option_changes(
    base_command: list[str],
    candidate_command: list[str],
    excluded_config_names: Sequence[str],
) -> list[str]:
    base_options, _ = _parse_cli_command_signature(base_command)
    candidate_options, _ = _parse_cli_command_signature(candidate_command)
    changed: list[str] = []
    for option_name in sorted(set(base_options) | set(candidate_options)):
        if _matching_excluded_config_name(option_name, excluded_config_names) is None:
            continue
        if base_options.get(option_name) != candidate_options.get(option_name):
            changed.append(option_name)
    return changed


def _config_name_is_whitelisted(name: str, tunable_whitelist_names: Sequence[str] | None) -> bool:
    if tunable_whitelist_names is None:
        return True
    return _matching_excluded_config_name(name, tunable_whitelist_names) is not None


def _tokens_with_disallowed_alternation_syntax(tokens: Sequence[str]) -> list[str]:
    return [token for token in tokens if "|" in token]


def _non_whitelisted_inline_env_changes(
    env_updates: dict[str, str],
    base_env_overrides: dict[str, str],
    tunable_whitelist_names: Sequence[str] | None,
) -> list[str]:
    if tunable_whitelist_names is None:
        return []
    changed: list[str] = []
    for name, value in env_updates.items():
        if base_env_overrides.get(name) == value:
            continue
        if not _config_name_is_whitelisted(name, tunable_whitelist_names):
            changed.append(name)
    return changed


def _non_whitelisted_env_assignments_in_tokens(
    tokens: list[str],
    tunable_whitelist_names: Sequence[str] | None,
) -> list[str]:
    if tunable_whitelist_names is None:
        return []
    changed: list[str] = []
    idx = 1 if tokens and tokens[0] == "env" else 0
    while idx < len(tokens) and _is_env_assignment_token(tokens[idx]):
        name, _ = tokens[idx].split("=", 1)
        if not _config_name_is_whitelisted(name, tunable_whitelist_names):
            changed.append(name)
        idx += 1
    return changed


def _non_whitelisted_option_tokens_in_tokens(
    tokens: list[str],
    tunable_whitelist_names: Sequence[str] | None,
) -> list[str]:
    if tunable_whitelist_names is None:
        return []
    changed: list[str] = []
    for token in tokens:
        if not _is_option_token(token):
            continue
        option_name = token.split("=", 1)[0]
        if not _config_name_is_whitelisted(option_name, tunable_whitelist_names):
            changed.append(option_name)
    return changed


def _non_whitelisted_command_changes(
    base_command: list[str],
    candidate_command: list[str],
    tunable_whitelist_names: Sequence[str] | None,
) -> list[str]:
    if tunable_whitelist_names is None:
        return []
    base_options, base_positionals = _parse_cli_command_signature(base_command)
    candidate_options, candidate_positionals = _parse_cli_command_signature(candidate_command)
    changed: list[str] = []
    for option_name in sorted(set(base_options) | set(candidate_options)):
        if base_options.get(option_name) == candidate_options.get(option_name):
            continue
        if not _config_name_is_whitelisted(option_name, tunable_whitelist_names):
            changed.append(option_name)
    if candidate_positionals != base_positionals and not _config_name_is_whitelisted(
        "positionals",
        tunable_whitelist_names,
    ):
        changed.append("positionals")
    return changed


def _replace_command_option_delta(
    base_command: list[str],
    option_name: str,
    replacement_tokens: list[str],
) -> list[str]:
    idx = 1
    while idx < len(base_command):
        token = base_command[idx]
        if token == option_name or token.startswith(f"{option_name}="):
            end_idx = idx + 1
            if (
                token == option_name
                and end_idx < len(base_command)
                and (
                    not _is_option_token(base_command[end_idx])
                    or _looks_like_numeric_token(base_command[end_idx])
                )
            ):
                end_idx += 1
            return [*base_command[:idx], *replacement_tokens, *base_command[end_idx:]]
        idx += 1
    return [*base_command, *replacement_tokens]


def _split_command_option_delta_groups(tokens: list[str]) -> list[list[str]]:
    groups: list[list[str]] = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if not _is_option_token(token):
            return []
        group = [token]
        idx += 1
        while idx < len(tokens) and (
            not _is_option_token(tokens[idx]) or _looks_like_numeric_token(tokens[idx])
        ):
            group.append(tokens[idx])
            idx += 1
        groups.append(group)
    return groups


def _apply_command_option_delta_groups(
    base_command: list[str],
    delta_tokens: list[str],
) -> list[str]:
    groups = _split_command_option_delta_groups(delta_tokens)
    if not groups:
        return []
    updated_command = list(base_command)
    for group in groups:
        option_name = group[0].split("=", 1)[0]
        updated_command = _replace_command_option_delta(updated_command, option_name, group)
    return updated_command


def _apply_command_delta(
    base_command: list[str],
    parameter: str,
    value_tokens: list[str],
) -> list[str]:
    if not value_tokens:
        return []
    if parameter and _is_option_token(parameter):
        replacement_tokens = (
            value_tokens
            if value_tokens[0] == parameter or value_tokens[0].startswith(f"{parameter}=")
            else [parameter, *value_tokens]
        )
        if any(_is_option_token(token) for token in replacement_tokens[1:]):
            return _apply_command_option_delta_groups(base_command, replacement_tokens)
        return _replace_command_option_delta(base_command, parameter, replacement_tokens)
    if not parameter and _is_option_token(value_tokens[0]):
        return _apply_command_option_delta_groups(base_command, value_tokens)
    if parameter:
        try:
            token_index = base_command.index(parameter, 1)
        except ValueError:
            return [*base_command, *value_tokens]
        return [*base_command[:token_index], *value_tokens, *base_command[token_index + 1 :]]
    return [*base_command, *value_tokens]


def _build_spawn_execution_plan(
    base_command: list[str],
    actions: list[dict[str, Any]],
    base_env_overrides: dict[str, str] | None = None,
    excluded_config_names: Sequence[str] | None = None,
    tunable_whitelist_names: Sequence[str] | None = None,
) -> tuple[list[str], dict[str, str], list[str], list[str]]:
    if not base_command:
        return [], {}, [], ["base command was empty"]

    prefix_tokens: list[str] = []
    command_tokens = list(base_command)
    command_replaced = False
    env_overrides: dict[str, str] = dict(base_env_overrides or {})
    applied: list[str] = []
    skipped: list[str] = []
    excluded_controls = list(excluded_config_names or [])
    tunable_whitelist = None if tunable_whitelist_names is None else list(tunable_whitelist_names)

    for idx, action in enumerate(actions):
        config_type = str(action.get("config_type", "")).strip().lower()
        parameter = str(action.get("parameter", "")).strip()
        new_value = action.get("new_value")
        if "|" in parameter:
            skipped.append(
                f"action[{idx}] parameter uses unsupported bundled control syntax '{parameter}'"
            )
            continue

        if config_type == "env_var":
            matched_excluded = _matching_excluded_config_name(parameter, excluded_controls)
            if matched_excluded is not None:
                skipped.append(
                    f"action[{idx}] changes excluded config '{matched_excluded}'"
                )
                continue
            if not _config_name_is_whitelisted(parameter, tunable_whitelist):
                skipped.append(
                    f"action[{idx}] changes non-whitelisted config '{parameter}'"
                )
                continue
            if new_value is None:
                skipped.append(f"action[{idx}] env_var '{parameter}' missing new_value")
                continue
            env_value = str(new_value)
            env_overrides[parameter] = env_value
            applied.append(f"env {parameter}={env_value}")
            continue

        tokens = _to_token_list(new_value)
        if not tokens:
            skipped.append(f"action[{idx}] produced no command tokens")
            continue
        disallowed_tokens = _tokens_with_disallowed_alternation_syntax(tokens)
        if disallowed_tokens:
            skipped.append(
                f"action[{idx}] new_value uses unsupported bundled token syntax: "
                f"{', '.join(disallowed_tokens)}"
            )
            continue

        if config_type == "command":
            if command_replaced:
                skipped.append(f"action[{idx}] additional command rewrite skipped")
                continue

            candidate_executable_tokens, _ = _extract_inline_env_assignments(tokens)
            if (
                not candidate_executable_tokens
                or Path(candidate_executable_tokens[0]).name != Path(base_command[0]).name
            ):
                tokens = _apply_command_delta(base_command, parameter, tokens)
                if not tokens:
                    skipped.append(f"action[{idx}] command delta produced no executable tokens")
                    continue

            normalized_tokens, inline_env_overrides = _extract_inline_env_assignments(tokens)
            if not normalized_tokens:
                skipped.append(f"action[{idx}] command rewrite produced no executable tokens")
                continue

            excluded_env_changes = _excluded_inline_env_changes(
                inline_env_overrides,
                env_overrides,
                excluded_controls,
            )
            if excluded_env_changes:
                skipped.append(
                    f"action[{idx}] command rewrite changes excluded config(s): "
                    f"{', '.join(excluded_env_changes)}"
                )
                continue

            non_whitelisted_env_changes = _non_whitelisted_inline_env_changes(
                inline_env_overrides,
                env_overrides,
                tunable_whitelist,
            )
            if non_whitelisted_env_changes:
                skipped.append(
                    f"action[{idx}] command rewrite changes non-whitelisted config(s): "
                    f"{', '.join(non_whitelisted_env_changes)}"
                )
                continue

            if Path(normalized_tokens[0]).name != Path(base_command[0]).name:
                skipped.append(
                    f"action[{idx}] command replacement must keep executable '{base_command[0]}'"
                )
                continue

            rewrite_issues = _validate_spawn_command_rewrite(base_command, normalized_tokens)
            if rewrite_issues:
                for issue in rewrite_issues:
                    skipped.append(f"action[{idx}] {issue}")
                continue

            excluded_option_changes = _excluded_command_option_changes(
                base_command,
                normalized_tokens,
                excluded_controls,
            )
            if excluded_option_changes:
                skipped.append(
                    f"action[{idx}] command rewrite changes excluded config(s): "
                    f"{', '.join(excluded_option_changes)}"
                )
                continue

            non_whitelisted_command_changes = _non_whitelisted_command_changes(
                base_command,
                normalized_tokens,
                tunable_whitelist,
            )
            if non_whitelisted_command_changes:
                skipped.append(
                    f"action[{idx}] command rewrite changes non-whitelisted config(s): "
                    f"{', '.join(non_whitelisted_command_changes)}"
                )
                continue

            command_tokens = normalized_tokens
            env_overrides.update(inline_env_overrides)
            command_replaced = True
            applied.append(f"command {' '.join(normalized_tokens)}")
            continue

        if config_type == "command_prefix":
            excluded_env_changes = _excluded_env_assignments_in_tokens(tokens, excluded_controls)
            excluded_options = _excluded_option_tokens_in_tokens(tokens, excluded_controls)
            if excluded_env_changes or excluded_options:
                excluded_changes = [*excluded_env_changes, *excluded_options]
                skipped.append(
                    f"action[{idx}] command prefix changes excluded config(s): "
                    f"{', '.join(excluded_changes)}"
                )
                continue
            non_whitelisted_env_changes = _non_whitelisted_env_assignments_in_tokens(
                tokens,
                tunable_whitelist,
            )
            non_whitelisted_options = _non_whitelisted_option_tokens_in_tokens(
                tokens,
                tunable_whitelist,
            )
            if non_whitelisted_env_changes or non_whitelisted_options:
                non_whitelisted_changes = [
                    *non_whitelisted_env_changes,
                    *non_whitelisted_options,
                ]
                skipped.append(
                    f"action[{idx}] command prefix changes non-whitelisted config(s): "
                    f"{', '.join(non_whitelisted_changes)}"
                )
                continue
            if tunable_whitelist is not None and not (
                non_whitelisted_env_changes
                or non_whitelisted_options
                or _excluded_env_assignments_in_tokens(tokens, tunable_whitelist)
                or _excluded_option_tokens_in_tokens(tokens, tunable_whitelist)
            ):
                skipped.append(
                    f"action[{idx}] command prefix contains no whitelisted tunable controls"
                )
                continue
            prefix_tokens.extend(tokens)
            applied.append(f"prefix {' '.join(tokens)}")
            continue

        skipped.append(f"action[{idx}] unsupported at execution time")

    return prefix_tokens + command_tokens, env_overrides, applied, skipped


def _command_key(
    command_parts: list[str],
    env_overrides: dict[str, str],
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    return (
        tuple(command_parts),
        tuple(sorted((k, str(v)) for k, v in env_overrides.items())),
    )


def _build_spawn_candidates(
    base_command: list[str],
    actions: list[dict[str, Any]],
    max_candidates: int,
    base_env_overrides: dict[str, str] | None = None,
    excluded_config_names: Sequence[str] | None = None,
    tunable_whitelist_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if max_candidates < 1:
        return []

    if actions and all(
        isinstance(action, dict)
        and "config_type" not in action
        and isinstance(action.get("actions"), list)
        for action in actions
    ):
        attempts = _materialize_spawn_candidate_attempts_from_candidate_sets(
            base_command,
            actions,
            max_attempts=max_candidates,
            base_env_overrides=base_env_overrides,
            excluded_config_names=excluded_config_names,
            tunable_whitelist_names=tunable_whitelist_names,
        )
        return [attempt for attempt in attempts if attempt.get("materialized_status") == "ready"]

    alternative_action_types = {"command", "command_prefix"}
    shared_actions: list[dict[str, Any]] = []
    alternative_actions: list[dict[str, Any]] = []

    for action in actions:
        config_type = str(action.get("config_type", "")).strip().lower()
        if config_type in alternative_action_types:
            alternative_actions.append(action)
        else:
            shared_actions.append(action)

    candidate_action_sets: list[list[dict[str, Any]]] = []
    if alternative_actions:
        candidate_action_sets.extend([shared_actions + [action] for action in alternative_actions])
    elif shared_actions:
        candidate_action_sets.extend(
            [shared_actions[:idx] for idx in range(1, len(shared_actions) + 1)]
        )

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, ...], tuple[tuple[str, str], ...]]] = set()

    for action_set in candidate_action_sets:
        tuned_cmd, env_overrides, applied_actions, skipped_actions = _build_spawn_execution_plan(
            base_command,
            action_set,
            base_env_overrides=base_env_overrides,
            excluded_config_names=excluded_config_names,
            tunable_whitelist_names=tunable_whitelist_names,
        )
        if not tuned_cmd:
            continue

        if tuned_cmd == base_command and env_overrides == dict(base_env_overrides or {}):
            continue

        key = _command_key(tuned_cmd, env_overrides)
        if key in seen:
            continue
        seen.add(key)

        label = ", ".join(applied_actions) if applied_actions else "(none)"
        candidates.append(
            {
                "cmd": tuned_cmd,
                "env_overrides": env_overrides,
                "label": label,
                "skipped_actions": skipped_actions,
                "actions": [dict(action) for action in action_set],
            }
        )
        if len(candidates) >= max_candidates:
            break

    return candidates


def _materialize_spawn_candidate_attempts_from_candidate_sets(
    base_command: list[str],
    candidates: list[dict[str, Any]],
    max_attempts: int,
    base_env_overrides: dict[str, str] | None = None,
    excluded_config_names: Sequence[str] | None = None,
    tunable_whitelist_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if max_attempts < 1:
        return []

    attempts: list[dict[str, Any]] = []
    seen_materialized: set[tuple[tuple[str, ...], tuple[tuple[str, str], ...]]] = set()
    base_env = dict(base_env_overrides or {})

    for candidate_idx, candidate in enumerate(candidates):
        action_set = [
            dict(action)
            for action in candidate.get("actions", [])
            if isinstance(action, dict)
        ]
        tuned_cmd, env_overrides, applied_actions, skipped_actions = _build_spawn_execution_plan(
            base_command,
            action_set,
            base_env_overrides=base_env_overrides,
            excluded_config_names=excluded_config_names,
            tunable_whitelist_names=tunable_whitelist_names,
        )
        label = str(candidate.get("label", "")).strip()
        if not label:
            label = (
                ", ".join(applied_actions)
                if applied_actions
                else f"candidate {candidate_idx + 1}"
            )
        key = _command_key(tuned_cmd, env_overrides) if tuned_cmd else None
        materialized_status = "ready"
        materialized_reason = "candidate materialized"

        if not tuned_cmd:
            materialized_status = "rejected"
            materialized_reason = "candidate materialization produced no executable command"
        elif tuned_cmd == base_command and env_overrides == base_env:
            materialized_status = "rejected"
            materialized_reason = "candidate materialized with no effective change"
        elif key in seen_materialized:
            materialized_status = "duplicate"
            materialized_reason = "candidate duplicated an earlier materialized attempt"
        else:
            seen_materialized.add(cast(tuple[tuple[str, ...], tuple[tuple[str, str], ...]], key))

        attempts.append(
            {
                "cmd": tuned_cmd,
                "env_overrides": env_overrides,
                "label": label,
                "skipped_actions": skipped_actions,
                "actions": action_set,
                "source_candidate_index": candidate.get("source_candidate_index"),
                "materialized_status": materialized_status,
                "materialized_reason": materialized_reason,
            }
        )
        if len(attempts) >= max_attempts:
            break

    return attempts


def _materialize_spawn_candidate_attempts(
    base_command: list[str],
    actions: list[dict[str, Any]],
    max_attempts: int,
    base_env_overrides: dict[str, str] | None = None,
    excluded_config_names: Sequence[str] | None = None,
    tunable_whitelist_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if max_attempts < 1:
        return []

    alternative_action_types = {"command", "command_prefix"}
    shared_actions: list[dict[str, Any]] = []
    alternative_actions: list[dict[str, Any]] = []

    for action in actions:
        config_type = str(action.get("config_type", "")).strip().lower()
        if config_type in alternative_action_types:
            alternative_actions.append(action)
        else:
            shared_actions.append(action)

    candidate_action_sets: list[list[dict[str, Any]]] = []
    if alternative_actions:
        candidate_action_sets.extend([shared_actions + [action] for action in alternative_actions])
    elif shared_actions:
        candidate_action_sets.extend(
            [shared_actions[:idx] for idx in range(1, len(shared_actions) + 1)]
        )

    attempts: list[dict[str, Any]] = []
    seen_materialized: set[tuple[tuple[str, ...], tuple[tuple[str, str], ...]]] = set()
    base_env = dict(base_env_overrides or {})

    for action_set in candidate_action_sets:
        tuned_cmd, env_overrides, applied_actions, skipped_actions = _build_spawn_execution_plan(
            base_command,
            action_set,
            base_env_overrides=base_env_overrides,
            excluded_config_names=excluded_config_names,
            tunable_whitelist_names=tunable_whitelist_names,
        )
        label = ", ".join(applied_actions) if applied_actions else "(none)"
        key = _command_key(tuned_cmd, env_overrides) if tuned_cmd else None
        materialized_status = "ready"
        materialized_reason = "candidate materialized"

        if not tuned_cmd:
            materialized_status = "rejected"
            materialized_reason = "candidate materialization produced no executable command"
        elif tuned_cmd == base_command and env_overrides == base_env:
            materialized_status = "rejected"
            materialized_reason = "candidate materialized with no effective change"
        elif key in seen_materialized:
            materialized_status = "duplicate"
            materialized_reason = "candidate duplicated an earlier materialized attempt"
        else:
            seen_materialized.add(cast(tuple[tuple[str, ...], tuple[tuple[str, str], ...]], key))

        attempts.append(
            {
                "cmd": tuned_cmd,
                "env_overrides": env_overrides,
                "label": label,
                "skipped_actions": skipped_actions,
                "actions": [dict(action) for action in action_set],
                "materialized_status": materialized_status,
                "materialized_reason": materialized_reason,
            }
        )
        if len(attempts) >= max_attempts:
            break

    return attempts


def _discover_related_processes(root_pid: int, max_targets: int = 4) -> dict[str, Any]:
    if max_targets < 1:
        max_targets = 1

    try:
        root_proc = psutil.Process(root_pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return {
            "root_pid": root_pid,
            "target_pids": [root_pid],
            "primary_pid": root_pid,
            "workers": [],
            "all_processes": [],
        }

    processes = [root_proc]
    try:
        processes.extend(root_proc.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    unique_by_pid: dict[int, psutil.Process] = {}
    for proc in processes:
        unique_by_pid[proc.pid] = proc
    processes = list(unique_by_pid.values())

    for proc in processes:
        try:
            proc.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    time.sleep(0.2)

    snapshots: list[dict[str, Any]] = []
    for proc in processes:
        try:
            with proc.oneshot():
                snapshots.append(
                    {
                        "pid": proc.pid,
                        "ppid": proc.ppid(),
                        "name": proc.name(),
                        "status": proc.status(),
                        "cpu_percent": float(proc.cpu_percent(interval=None)),
                        "rss_mb": round(proc.memory_info().rss / (1024 * 1024), 3),
                        "cmdline": proc.cmdline(),
                    }
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    snapshots.sort(
        key=lambda item: (
            float(item.get("cpu_percent", 0.0)),
            float(item.get("rss_mb", 0.0)),
        ),
        reverse=True,
    )

    worker_snapshots = [item for item in snapshots if int(item.get("pid", -1)) != root_pid]
    heavy_workers = [
        item for item in worker_snapshots if float(item.get("cpu_percent", 0.0)) >= 1.0
    ]
    selected_workers = (
        heavy_workers[: max_targets - 1] if heavy_workers else worker_snapshots[: max_targets - 1]
    )

    target_pids = [root_pid]
    target_pids.extend(int(item.get("pid", root_pid)) for item in selected_workers)

    deduped_targets: list[int] = []
    seen_targets: set[int] = set()
    for target_pid in target_pids:
        if target_pid in seen_targets:
            continue
        seen_targets.add(target_pid)
        deduped_targets.append(target_pid)

    primary_pid = selected_workers[0]["pid"] if selected_workers else root_pid

    return {
        "root_pid": root_pid,
        "target_pids": deduped_targets,
        "primary_pid": int(primary_pid),
        "workers": selected_workers,
        "all_processes": snapshots,
    }


async def _extract_metrics_from_output(
    agent: SysPilotAgent,
    output: str,
    context: str = "",
) -> dict[str, Any]:
    literal_metrics = _extract_literal_numeric_metrics(output)
    prompt = f"""{f"Context: {context}" if context else ""}

Program output:
```
{output[:4000]}
```
"""

    try:
        metric_llm = LLMInterface(_metric_extraction_llm_config(agent.config.llm))
        response = await metric_llm.chat(
            messages=[
                Message(role="system", content=METRIC_EXTRACTION_SYSTEM_PROMPT),
                Message(role="user", content=prompt),
            ]
        )
    except LLMError as exc:
        logger.warning("Output metric extraction skipped after LLM failure: %s", exc)
        if literal_metrics:
            literal_metrics["error"] = str(exc)
            return literal_metrics
        return {"no_metrics": True, "error": str(exc)}

    import re

    json_match = re.search(r"\{[^{}]*\}", response.content, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, dict):
                normalized = _normalize_metric_mapping(parsed)
                if literal_metrics:
                    return {**literal_metrics, **normalized}
                return normalized
        except json.JSONDecodeError:
            pass
    if literal_metrics:
        return literal_metrics
    return {"no_metrics": True}


def _extract_literal_numeric_metrics(output: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    metric_tokens = (
        "per_sec",
        "throughput",
        "latency",
        "runtime",
        "elapsed",
        "wall_time",
        "duration",
        "bandwidth",
        "ops_per_sec",
        "bytes_per_sec",
        "score",
        "loss",
        "accuracy",
    )
    metric_pattern = re.compile(
        r'"(?P<key>[A-Za-z][A-Za-z0-9_]*)"\s*:\s*(?P<value>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'
    )
    for match in metric_pattern.finditer(output):
        key = _normalize_metric_key(match.group("key"))
        if not key:
            continue
        if not any(token in key for token in metric_tokens):
            continue
        try:
            metrics[key] = float(match.group("value"))
        except ValueError:
            continue
    return metrics


def _normalize_metric_key(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    return normalized.strip("_")


def _normalize_metric_mapping(metrics: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_key, value in metrics.items():
        if raw_key in {"no_metrics", "error"}:
            normalized[raw_key] = value
            continue
        normalized_key = _normalize_metric_key(str(raw_key))
        if not normalized_key:
            continue
        normalized[normalized_key] = value
    if metrics.get("no_metrics") and "no_metrics" not in normalized:
        normalized["no_metrics"] = True
    if "error" in metrics and "error" not in normalized:
        normalized["error"] = metrics["error"]
    return normalized


def _infer_primary_objective_mode(metric_key: str) -> str | None:
    maximize_tokens = (
        "throughput",
        "per_sec",
        "ops",
        "qps",
        "rps",
        "fps",
        "bandwidth",
        "speedup",
        "accuracy",
    )
    minimize_tokens = (
        "latency",
        "time",
        "duration",
        "elapsed",
        "wall",
        "p50",
        "p90",
        "p95",
        "p99",
        "rss",
        "memory",
        "error",
        "loss",
    )
    if any(token in metric_key for token in maximize_tokens):
        return "maximize"
    if any(token in metric_key for token in minimize_tokens):
        return "minimize"
    return None


def _metric_intent_aliases(metric_key: str) -> set[str]:
    aliases = {metric_key, metric_key.replace("_", " ")}
    tokens = set(metric_key.split("_"))

    is_time_metric = (
        "elapsed" in tokens
        or "duration" in tokens
        or "runtime" in tokens
        or "latency" in tokens
        or {"wall", "time"}.issubset(tokens)
        or ("time" in tokens and "per" not in tokens)
    )
    if is_time_metric:
        aliases.update({"wall time", "elapsed time", "runtime", "time", "duration", "latency"})

    is_throughput_metric = (
        "throughput" in tokens
        or "bandwidth" in tokens
        or "qps" in tokens
        or "rps" in tokens
        or "fps" in tokens
        or "speedup" in tokens
        or metric_key.endswith("_per_sec")
        or {"per", "sec"}.issubset(tokens)
    )
    if is_throughput_metric:
        aliases.update({"throughput", "rate", "per second", "ops per second", "bandwidth"})
    if "token" in tokens and is_throughput_metric:
        aliases.update({"token throughput", "tokens per second"})
    if "task" in tokens and is_throughput_metric:
        aliases.update({"task throughput", "tasks per second"})
    if "request" in tokens and is_throughput_metric:
        aliases.update({"request throughput", "requests per second"})

    if "memory" in tokens or "rss" in tokens:
        aliases.update({"memory", "memory usage", "rss"})
    if "accuracy" in tokens:
        aliases.update({"accuracy"})
    if "error" in tokens or "loss" in tokens:
        aliases.update({"error", "error rate", "loss"})

    return {alias for alias in aliases if alias}


def _resolve_primary_objective_metric(
    metrics: dict[str, Any],
    primary_objective: str | None,
) -> str | None:
    if not primary_objective or metrics.get("no_metrics"):
        return None

    normalized_query = _normalize_metric_key(primary_objective)
    if not normalized_query:
        return None

    numeric_metric_keys = [
        key
        for key, value in metrics.items()
        if key not in {"no_metrics", "error"}
        and not isinstance(value, bool)
        and isinstance(value, (int, float))
    ]
    if normalized_query in numeric_metric_keys:
        return normalized_query

    query_tokens = {token for token in normalized_query.split("_") if token}
    best_key: str | None = None
    best_score = 0.0

    for metric_key in numeric_metric_keys:
        normalized_metric_key = _normalize_metric_key(metric_key)
        aliases = {
            _normalize_metric_key(alias)
            for alias in _metric_intent_aliases(normalized_metric_key)
        }
        aliases.add(normalized_metric_key)

        exact_alias_match = normalized_query in aliases
        alias_similarity = max(
            difflib.SequenceMatcher(None, normalized_query, alias).ratio() for alias in aliases
        )
        alias_tokens = {
            token
            for alias in aliases
            for token in alias.split("_")
            if token
        }
        token_overlap = 0.0
        if query_tokens and alias_tokens:
            token_overlap = len(query_tokens & alias_tokens) / len(query_tokens | alias_tokens)

        score = alias_similarity + token_overlap
        if exact_alias_match:
            score += 1.0

        if score > best_score:
            best_key = normalized_metric_key
            best_score = score

    if best_key is None or best_score < 1.0:
        return None
    return best_key


def _extract_primary_objective_value(
    metrics: dict[str, Any],
    primary_objective_metric: str | None,
) -> float | None:
    if not primary_objective_metric or metrics.get("no_metrics"):
        return None
    value = metrics.get(_normalize_metric_key(primary_objective_metric))
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _compare_objective_values(
    candidate_value: float,
    reference_value: float,
    objective_mode: str,
) -> int:
    tolerance = 1e-12
    if objective_mode == "maximize":
        if candidate_value > reference_value + tolerance:
            return 1
        if candidate_value < reference_value - tolerance:
            return -1
        return 0
    if candidate_value < reference_value - tolerance:
        return 1
    if candidate_value > reference_value + tolerance:
        return -1
    return 0


def _compute_objective_change_pct(
    reference_value: float,
    candidate_value: float,
    objective_mode: str,
) -> float:
    if reference_value == 0:
        return 0.0
    if objective_mode == "maximize":
        return ((candidate_value - reference_value) / abs(reference_value)) * 100
    return ((reference_value - candidate_value) / abs(reference_value)) * 100


def _compute_objective_speedup_ratio(
    reference_value: float | None,
    candidate_value: float | None,
    objective_mode: str,
) -> float | None:
    if reference_value is None or candidate_value is None:
        return None
    if reference_value <= 0.0 or candidate_value <= 0.0:
        return None
    if objective_mode == "maximize":
        return candidate_value / reference_value
    return reference_value / candidate_value


def _objective_speedup_label(ratio: float | None) -> str:
    if ratio is None:
        return ""
    if abs(ratio - 1.0) <= 1e-9:
        return "same"
    return "better" if ratio > 1.0 else "slower"


def _format_objective_speedup(ratio: float | None, *, unavailable: str = "-") -> str:
    if ratio is None:
        return unavailable
    return f"{ratio:.2f}x ({_objective_speedup_label(ratio)})"


def _objective_display_name(primary_objective: str | None) -> str:
    if not primary_objective:
        return "Primary Objective"
    return primary_objective


def _format_metric_value(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6g}"


def _format_bytes_human(value: int | None) -> str:
    if value is None:
        return "-"
    if value < 1024:
        return f"{value}B"
    units = ["KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        size /= 1024.0
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f}{unit}"
    return f"{value}B"


def _render_effective_memory_envelope(
    profiling_json: dict[str, Any],
    memory_high: str | None,
) -> str:
    evidence = profiling_json.get("cgroup_memory_evidence", {}) if profiling_json else {}
    if not isinstance(evidence, dict) or not evidence.get("control_group"):
        return f"## Spawn MemoryHigh Limit\n{memory_high or '(none)'}"

    memory_events = (
        evidence.get("memory_events", {})
        if isinstance(evidence.get("memory_events"), dict)
        else {}
    )
    memory_pressure = (
        evidence.get("memory_pressure", {})
        if isinstance(evidence.get("memory_pressure"), dict)
        else {}
    )
    memory_stat = (
        evidence.get("memory_stat", {})
        if isinstance(evidence.get("memory_stat"), dict)
        else {}
    )
    pressure_some = (
        memory_pressure.get("some", {})
        if isinstance(memory_pressure.get("some"), dict)
        else {}
    )
    pressure_full = (
        memory_pressure.get("full", {})
        if isinstance(memory_pressure.get("full"), dict)
        else {}
    )

    return (
        "## Effective Memory Capacity\n"
        f"Requested MemoryHigh: {evidence.get('requested_memory_high') or '(none)'}\n"
        f"Control group: {evidence.get('control_group') or '(unknown)'}\n"
        "Effective limit: "
        f"{_format_bytes_human(cast(int | None, evidence.get('effective_limit_bytes')))}\n"
        f"Current usage: {_format_bytes_human(cast(int | None, evidence.get('current_bytes')))}\n"
        f"Peak usage: {_format_bytes_human(cast(int | None, evidence.get('peak_bytes')))}\n"
        "Minimum headroom: "
        f"{_format_bytes_human(cast(int | None, evidence.get('min_headroom_bytes')))}\n"
        f"Sample count: {evidence.get('sample_count', 0)}\n"
        f"memory.events: high={memory_events.get('high', 0)} max={memory_events.get('max', 0)} "
        f"oom={memory_events.get('oom', 0)} oom_kill={memory_events.get('oom_kill', 0)}\n"
        f"memory.pressure.some.total={pressure_some.get('total', 0)} "
        f"memory.pressure.full.total={pressure_full.get('total', 0)}\n"
        f"memory.stat: pgfault={memory_stat.get('pgfault', 0)} "
        f"pgmajfault={memory_stat.get('pgmajfault', 0)} pgscan={memory_stat.get('pgscan', 0)} "
        f"pgsteal={memory_stat.get('pgsteal', 0)} "
        f"workingset_refault_anon={memory_stat.get('workingset_refault_anon', 0)} "
        f"workingset_refault_file={memory_stat.get('workingset_refault_file', 0)}\n"
        "Swap current: "
        f"{_format_bytes_human(cast(int | None, evidence.get('swap_current_bytes')))}\n"
        f"Status: {evidence.get('status', 'unknown')}\n"
        "Rule: When this block is present, reason about memory against this "
        "cgroup-constrained envelope rather than host total RAM or generic "
        "process memory percentages."
    )


async def _run_live_profiling_pipeline(
    agent: SysPilotAgent,
    command_parts: list[str],
    timeout: int,
    workdir: str | None = None,
    memory_high: str | None = None,
    env_overrides: dict[str, str] | None = None,
    hypothesis_profiles: list[dict[str, Any]] | None = None,
    live_output_callback: Callable[[str, str], None] | None = None,
    stop_after_seconds: float | None = None,
    intentional_stop_reason: str = "screen_budget_reached",
    periodic_sample_interval_s: float | None = None,
    skip_profiling: bool = False,
) -> dict[str, Any]:
    if skip_profiling:
        artifact = await execute_unprofiled_spawn_run(
            agent,
            command_parts,
            timeout,
            workdir=workdir,
            memory_high=memory_high,
            env_overrides=env_overrides,
            live_output_callback=live_output_callback,
            wrap_spawn_command_with_memory_high=_wrap_spawn_command_with_memory_high,
            summarize_spawn_runtime_output=_summarize_spawn_runtime_output,
        )
        return artifact.to_dict()

    artifact = await execute_profiled_spawn_run(
        agent,
        command_parts,
        timeout,
        workdir=workdir,
        memory_high=memory_high,
        env_overrides=env_overrides,
        hypothesis_profiles=hypothesis_profiles,
        live_output_callback=live_output_callback,
        build_spawn_profiling_env=_build_spawn_profiling_env,
        wrap_spawn_command_with_memory_high=_wrap_spawn_command_with_memory_high,
        discover_related_processes=_discover_related_processes,
        summarize_spawn_runtime_output=_summarize_spawn_runtime_output,
        collect_contract_report_env=_collect_contract_report_env,
        run_contract_hypothesis_validation=_run_contract_hypothesis_validation,
        stringify_env_value=_stringify_env_value,
        format_bytes_human=_format_bytes_human,
        start_probe_sessions=start_toolbox_probe_sessions,
        finish_probe_sessions=finish_toolbox_probe_sessions,
        sample_probe_sessions=sample_toolbox_probe_sessions,
        stop_after_seconds=stop_after_seconds,
        intentional_stop_reason=intentional_stop_reason,
        periodic_sample_interval_s=periodic_sample_interval_s,
    )
    return artifact.to_dict()


async def _run_spawn_tune_loop_async(
    agent: SysPilotAgent,
    command: str,
    contracts: Sequence[str],
    manual_hints: Sequence[str],
    fixed_budget: int | None,
    fixed_goal: float | None,
    max_iterations: int,
    timeout: int,
    primary_objective: str | None = None,
    primary_objective_mode: str | None = None,
    workdir: str | None = None,
    memory_high: str | None = None,
    profiling_enabled: bool = True,
    sink: OutputSink | None = None,
    event_emitter: _TuneEmitter | None = None,
    structured_context_bundle: dict[str, object] | None = None,
    initial_hypothesis_profiles: list[dict[str, Any]] | None = None,
    excluded_config_names: Sequence[str] | None = None,
    tunable_whitelist_names: Sequence[str] | None = None,
    study_trace: NestedCpuBlasStudyTrace | None = None,
) -> None:
    tune_started_at = time.time()
    best_config_find_wall_time_s = 0.0
    agent.configure_tuning_prompt(
        contract_enabled=bool(contracts),
        manual_enabled=bool(agent.context.loaded_manuals or manual_hints),
        quick_mode=False,
        profiling_enabled=profiling_enabled,
    )

    def _elapsed_payload() -> dict[str, float]:
        return {
            "total_wall_time_s": round(time.time() - tune_started_at, 6),
            "config_find_wall_time_s": round(best_config_find_wall_time_s, 6),
        }

    if sink is None:
        sink = make_output_sink(ui_mode=TuneUIMode.PLAIN, console=console)

    fixed_budget_state = FixedBudgetState(total=fixed_budget)
    if fixed_budget_state.enabled:
        fixed_budget_state.max_no_action_reprompts = min(
            fixed_budget_state.max_no_action_reprompts,
            cast(int, fixed_budget_state.total),
        )

    cmd_parts = shlex.split(command)
    if not cmd_parts:
        _emit_tune_event(
            event_emitter,
            "error",
            {"mode": "spawn", "phase": "spawn.command_parse", "message": "empty command"},
        )
        _emit_tune_event(
            event_emitter,
            "run.finished",
            {
                "mode": "spawn",
                "status": "invalid_input",
                **_elapsed_payload(),
                **_fixed_budget_event_payload(fixed_budget_state),
                **fixed_goal_payload(
                    threshold=fixed_goal,
                    value=None,
                    objective_mode="minimize",
                ),
            },
        )
        sink.print("[red]Spawn command is empty after parsing[/red]")
        return

    cmd_parts, inline_env_overrides = _extract_inline_env_assignments(cmd_parts)
    if not cmd_parts:
        _emit_tune_event(
            event_emitter,
            "error",
            {
                "mode": "spawn",
                "phase": "spawn.command_parse",
                "message": "command contained only inline env assignments",
            },
        )
        _emit_tune_event(
            event_emitter,
            "run.finished",
            {
                "mode": "spawn",
                "status": "invalid_input",
                **_elapsed_payload(),
                **_fixed_budget_event_payload(fixed_budget_state),
                **fixed_goal_payload(
                    threshold=fixed_goal,
                    value=None,
                    objective_mode="minimize",
                ),
            },
        )
        sink.print("[red]Spawn command contained only inline env assignments[/red]")
        return
    if inline_env_overrides:
        sink.print(f"[cyan]Detected inline environment baseline: {inline_env_overrides}[/cyan]")

    loaded_contract_names = list(contracts)
    manual_hint_list = list(manual_hints)
    excluded_config_list = list(excluded_config_names or [])
    tunable_whitelist_list = (
        None if tunable_whitelist_names is None else list(tunable_whitelist_names)
    )
    loaded_manual_names = sorted(agent.context.loaded_manuals.keys())
    manual_context_status = (
        f"Manual context: {len(loaded_manual_names)} loaded, "
        f"{len(manual_hint_list)} hints provided"
    )

    _emit_tune_event(
        event_emitter,
        "run.started",
        {
            "mode": "spawn",
            "command": command,
            "workdir": workdir or "",
            "memory_high": memory_high or "",
            "profiling_enabled": profiling_enabled,
            "max_iterations": max_iterations,
            "fixed_goal_threshold": fixed_goal,
            "timeout_s": timeout,
            "loaded_contracts": loaded_contract_names,
            "loaded_manuals": loaded_manual_names,
            "manual_hints": manual_hint_list,
            "excluded_config_names": excluded_config_list,
            "tunable_whitelist_names": tunable_whitelist_list or [],
            "baseline_env_overrides": dict(inline_env_overrides),
            "startup_hypothesis_profile_count": len(initial_hypothesis_profiles or []),
            "primary_objective_query": primary_objective or "",
            "primary_objective_metric": "",
            "primary_objective_mode": primary_objective_mode or "",
            **_fixed_budget_event_payload(fixed_budget_state),
        },
    )

    candidate_trial_limit_display = (
        f"fixed budget {fixed_budget_state.total} (--max-iterations ignored)"
        if fixed_budget_state.enabled
        else (
            f"deep-trace target {study_trace.unique_execution_target} unique non-baseline "
            f"launches / {study_trace.proposal_cap} proposals (--max-iterations ignored)"
            if study_trace is not None
            else str(max_iterations)
        )
    )
    if tunable_whitelist_list:
        tunable_whitelist_display = ", ".join(tunable_whitelist_list)
    elif tunable_whitelist_list is not None:
        tunable_whitelist_display = "(empty)"
    else:
        tunable_whitelist_display = "(not set)"
    sink.panel(
        f"[bold green]SysPilot Tuning Engine (Spawn Mode)[/bold green]\n"
        f"Command: {command}\n"
        f"Primary objective: {primary_objective or '(wall_time only)'}"
        f"{f' ({primary_objective_mode})' if primary_objective_mode else ''}\n"
        f"Workdir: {workdir or '(current)'}\n"
        f"MemoryHigh: {memory_high or '(none)'}\n"
        f"Profiling: {'enabled' if profiling_enabled else 'disabled (raw output only)'}\n"
        f"Candidate Trial Limit: {candidate_trial_limit_display}\n"
        f"Timeout: {timeout}s\n"
        f"Contracts: {len(loaded_contract_names)} loaded\n"
        f"Tunable whitelist: {tunable_whitelist_display}\n"
        f"{manual_context_status}",
        title="Starting",
    )

    manual_hint_content = _render_spawn_manual_hints(manual_hint_list)
    loaded_manual_content = _render_loaded_manual_context(agent.context.loaded_manuals)
    excluded_config_content = _render_excluded_config_names(excluded_config_list)
    tunable_whitelist_content = (
        _render_tunable_whitelist_names(tunable_whitelist_list)
        if tunable_whitelist_list is not None
        else "(not set; all supported non-immutable controls remain tunable)"
    )

    benchmark_deterministic = _is_env_flag_enabled("SYSPILOT_BENCH_DETERMINISTIC")
    if benchmark_deterministic:
        sink.print(
            "[dim]Benchmark deterministic mode enabled: skipping agent tuning analysis.[/dim]"
        )

    def _plain_live_output_callback(stream_name: str, line: str) -> None:
        text = line.rstrip("\r\n")
        if not text:
            return
        escaped = rich_escape(text)
        if stream_name == "stderr":
            sink.print(f"  [dim]stderr:[/dim] {escaped}")
            return
        sink.print(f"  {escaped}")

    startup_hypothesis_profiles = [
        item for item in (initial_hypothesis_profiles or []) if isinstance(item, dict)
    ]
    generated_hypothesis_content = hypothesis_profiles_yaml_for_prompt(startup_hypothesis_profiles)

    evidence_kind = "profiled" if profiling_enabled else "raw-output"
    run_count_label = "profiled run(s)" if profiling_enabled else "raw-output run(s)"
    if study_trace is not None:
        study_trace.record_baseline_started(
            cmd_parts,
            inline_env_overrides,
            metadata={
                "command": command,
                "workdir": workdir,
                "primary_objective_query": primary_objective,
            },
        )
    sink.print(f"\n[bold cyan]Round 0 Step 0: Initial Baseline Run ({evidence_kind})[/bold cyan]")
    _emit_tune_event(
        event_emitter,
        "phase.started",
        {"mode": "spawn", "phase": "spawn.baseline"},
    )
    _emit_tune_event(
        event_emitter,
        "profile.started",
        {
            "mode": "spawn",
            "phase": "spawn.baseline",
            "run_label": "baseline",
            "profiling_enabled": profiling_enabled,
        },
    )
    _emit_tune_event(
        event_emitter,
        "exec.started",
        {"mode": "spawn", "phase": "spawn.baseline"},
    )
    baseline_artifacts = await _run_live_profiling_pipeline(
        agent,
        cmd_parts,
        timeout,
        workdir=workdir,
        memory_high=memory_high,
        env_overrides=inline_env_overrides,
        hypothesis_profiles=startup_hypothesis_profiles,
        live_output_callback=_plain_live_output_callback,
        skip_profiling=not profiling_enabled,
    )
    baseline_time = baseline_artifacts["wall_time"]
    baseline_stdout = baseline_artifacts.get("stdout", "")
    baseline_stderr = baseline_artifacts.get("stderr", "")
    baseline_rc = baseline_artifacts["returncode"]
    _emit_tune_event(
        event_emitter,
        "exec.finished",
        {
            "mode": "spawn",
            "phase": "spawn.baseline",
            "returncode": baseline_rc,
            "wall_time_s": baseline_time,
        },
    )
    _emit_profile_artifact_events(
        event_emitter,
        baseline_artifacts,
        phase="spawn.baseline",
    )
    baseline_profiling_text, baseline_profiling_json, baseline_contract_validation = (
        _print_profile_artifact_summary(
            sink,
            baseline_artifacts,
            prefix=f"Baseline {evidence_kind} run",
        )
    )
    baseline_timed_out = bool(baseline_artifacts.get("timed_out"))
    if baseline_timed_out:
        baseline_status = "timeout"
    else:
        baseline_status = "success" if baseline_rc == 0 else "failed"
    _emit_tune_event(
        event_emitter,
        "baseline.finished",
        {
            "mode": "spawn",
            "phase": "spawn.baseline",
            "status": baseline_status,
            "returncode": baseline_rc,
            "wall_time_s": baseline_time,
            "profiled_run": profiling_enabled,
            "profiling_enabled": profiling_enabled,
            **_event_output_payload(baseline_stdout, baseline_stderr),
        },
    )
    if baseline_rc != 0 and not baseline_timed_out:
        if study_trace is not None:
            study_trace.record_baseline_finished(
                status="error",
                wall_time_s=float(baseline_time),
                objective_name=None,
                objective_value=None,
                returncode=int(baseline_rc),
            )
        fixed_budget_stop_reason = "hard_failure" if fixed_budget_state.enabled else ""
        _emit_tune_event(
            event_emitter,
            "error",
            {
                "mode": "spawn",
                "phase": "spawn.baseline",
                "message": f"baseline run failed (rc={baseline_rc})",
            },
        )
        _emit_tune_event(
            event_emitter,
            "phase.finished",
            {"mode": "spawn", "phase": "spawn.baseline", "status": "failed"},
        )
        _emit_tune_event(
            event_emitter,
            "run.finished",
            {
                "mode": "spawn",
                "status": "failed",
                "returncode": baseline_rc,
                **_elapsed_payload(),
                **_fixed_budget_event_payload(
                    fixed_budget_state,
                    stop_reason=fixed_budget_stop_reason,
                ),
                **fixed_goal_payload(
                    threshold=fixed_goal,
                    value=None,
                    objective_mode="minimize",
                    stop_reason="hard_failure" if fixed_goal is not None else "",
                ),
            },
        )
        sink.print(f"[red]Baseline run failed (rc={baseline_rc})[/red]")
        sink.print(f"[dim]stderr: {baseline_stderr[:500]}[/dim]")
        return

    if baseline_timed_out:
        _emit_tune_event(
            event_emitter,
            "warning.raised",
            {
                "mode": "spawn",
                "phase": "spawn.baseline",
                "message": "baseline run timed out; continuing with timeout as baseline",
            },
        )
        sink.print("[yellow]Baseline run timed out; continuing with timeout as baseline[/yellow]")

    _emit_tune_event(
        event_emitter,
        "phase.finished",
        {
            "mode": "spawn",
            "phase": "spawn.baseline",
            "status": baseline_status,
            "wall_time_s": baseline_time,
            "cmd_diff": "baseline",
        },
    )

    sink.print(f"  Wall time: {baseline_time:.2f}s")
    sink.print(f"  Output: {len(baseline_stdout)} bytes")

    baseline_metrics: dict[str, Any] = {}
    resolved_primary_objective_metric: str | None = None
    baseline_objective_value: float | None = None
    if primary_objective:
        sink.print("  Parsing baseline output for primary objective...")
        baseline_metrics = await _extract_metrics_from_output(
            agent,
            baseline_stdout,
            f"baseline run without tuning actions; primary objective query: {primary_objective}",
        )
        resolved_primary_objective_metric = _resolve_primary_objective_metric(
            baseline_metrics,
            primary_objective,
        )
        if resolved_primary_objective_metric is None:
            if study_trace is not None:
                study_trace.record_baseline_finished(
                    status="timeout" if baseline_timed_out else "ok",
                    wall_time_s=float(baseline_time),
                    objective_name=None,
                    objective_value=None,
                    returncode=int(baseline_rc),
                )
            fixed_budget_stop_reason = "hard_failure" if fixed_budget_state.enabled else ""
            _emit_tune_event(
                event_emitter,
                "error",
                {
                    "mode": "spawn",
                    "phase": "spawn.baseline",
                    "message": (
                        "baseline output did not resolve primary objective query "
                        f"'{primary_objective}' to a numeric metric"
                    ),
                },
            )
            _emit_tune_event(
                event_emitter,
                "run.finished",
                {
                    "mode": "spawn",
                    "status": "failed",
                    "returncode": 1,
                    "failure_reason": "missing_primary_objective",
                    **_elapsed_payload(),
                    **_fixed_budget_event_payload(
                        fixed_budget_state,
                        stop_reason=fixed_budget_stop_reason,
                    ),
                    **fixed_goal_payload(
                        threshold=fixed_goal,
                        value=None,
                        objective_mode="minimize",
                        stop_reason="hard_failure" if fixed_goal is not None else "",
                    ),
                },
            )
            sink.print(
                "[red]Baseline output did not resolve primary objective query "
                f"'{primary_objective}' to a numeric metric[/red]"
            )
            raise SystemExit(1)
        primary_objective_mode = primary_objective_mode or _infer_primary_objective_mode(
            resolved_primary_objective_metric
        )
        if primary_objective_mode is None:
            if study_trace is not None:
                study_trace.record_baseline_finished(
                    status="timeout" if baseline_timed_out else "ok",
                    wall_time_s=float(baseline_time),
                    objective_name=resolved_primary_objective_metric,
                    objective_value=None,
                    returncode=int(baseline_rc),
                )
            fixed_budget_stop_reason = "hard_failure" if fixed_budget_state.enabled else ""
            _emit_tune_event(
                event_emitter,
                "error",
                {
                    "mode": "spawn",
                    "phase": "spawn.baseline",
                    "message": (
                        "could not infer primary objective mode for resolved metric "
                        f"'{resolved_primary_objective_metric}'"
                    ),
                },
            )
            _emit_tune_event(
                event_emitter,
                "run.finished",
                {
                    "mode": "spawn",
                    "status": "failed",
                    "returncode": 1,
                    "failure_reason": "missing_primary_objective_mode",
                    **_elapsed_payload(),
                    **_fixed_budget_event_payload(
                        fixed_budget_state,
                        stop_reason=fixed_budget_stop_reason,
                    ),
                    **fixed_goal_payload(
                        threshold=fixed_goal,
                        value=None,
                        objective_mode="minimize",
                        stop_reason="hard_failure" if fixed_goal is not None else "",
                    ),
                },
            )
            sink.print(
                "[red]Could not infer --primary-objective-mode after resolving "
                f"'{primary_objective}' to '{resolved_primary_objective_metric}'. "
                "Please specify --primary-objective-mode.[/red]"
            )
            raise SystemExit(1)
        baseline_objective_value = _extract_primary_objective_value(
            baseline_metrics,
            resolved_primary_objective_metric,
        )
        if baseline_objective_value is None:
            if study_trace is not None:
                study_trace.record_baseline_finished(
                    status="timeout" if baseline_timed_out else "ok",
                    wall_time_s=float(baseline_time),
                    objective_name=resolved_primary_objective_metric,
                    objective_value=None,
                    returncode=int(baseline_rc),
                )
            fixed_budget_stop_reason = "hard_failure" if fixed_budget_state.enabled else ""
            _emit_tune_event(
                event_emitter,
                "error",
                {
                    "mode": "spawn",
                    "phase": "spawn.baseline",
                    "message": (
                        "baseline output missing numeric primary objective "
                        f"'{primary_objective}'"
                    ),
                },
            )
            _emit_tune_event(
                event_emitter,
                "run.finished",
                {
                    "mode": "spawn",
                    "status": "failed",
                    "returncode": 1,
                    "failure_reason": "missing_primary_objective",
                    **_elapsed_payload(),
                    **_fixed_budget_event_payload(
                        fixed_budget_state,
                        stop_reason=fixed_budget_stop_reason,
                    ),
                    **fixed_goal_payload(
                        threshold=fixed_goal,
                        value=None,
                        objective_mode="minimize",
                        stop_reason="hard_failure" if fixed_goal is not None else "",
                    ),
                },
            )
            sink.print(
                "[red]Baseline output did not produce numeric primary objective "
                f"'{primary_objective}'[/red]"
            )
            raise SystemExit(1)
        sink.print(
            "  Resolved primary objective metric: "
            f"{resolved_primary_objective_metric} ({primary_objective_mode})"
        )
        sink.print(
            f"  Baseline {_objective_display_name(primary_objective)}: "
            f"{_format_metric_value(baseline_objective_value)}"
        )

    if study_trace is not None:
        study_trace.record_baseline_finished(
            status="timeout" if baseline_timed_out else "ok",
            wall_time_s=float(baseline_time),
            objective_name=resolved_primary_objective_metric,
            objective_value=baseline_objective_value,
            returncode=int(baseline_rc),
        )

    current_command_parts = list(cmd_parts)
    current_env_overrides: dict[str, str] = dict(inline_env_overrides)
    current_time = baseline_time
    current_stdout = baseline_stdout
    current_stderr = baseline_stderr
    current_profiling_text = baseline_profiling_text
    current_profiling_json = baseline_profiling_json
    current_contract_validation = baseline_contract_validation
    current_profile_artifacts = dict(baseline_artifacts)
    current_metrics = dict(baseline_metrics)
    current_objective_value = baseline_objective_value

    best_time = baseline_time
    best_label = "(none)"
    best_command_parts = list(cmd_parts)
    best_env_overrides: dict[str, str] = {}
    best_metrics = dict(baseline_metrics)
    best_objective_value = baseline_objective_value
    best_found_round = 0
    best_found_trial = 0
    best_candidate_trials_to_find = 0
    best_profiled_runs_to_find = 1
    best_config_find_wall_time_s = time.time() - tune_started_at

    seen_trials: set[tuple[tuple[str, ...], tuple[tuple[str, str], ...]]] = {
        _command_key(current_command_parts, current_env_overrides)
    }
    fixed_budget_stop_reason = ""
    fixed_goal_stop_reason = ""
    max_iterations_stop_reason = ""
    fixed_goal_no_action_reprompts_used = 0
    fixed_goal_max_no_action_reprompts = max(0, min(3, max_iterations))
    stagnation_iterations = 0
    convergence_patience = 2
    min_improvement_pct = 1.0
    active_hypothesis_refs: list[str] = []
    total_rounds_executed = 0
    total_candidate_trials_executed = 0
    total_profiled_runs = 1
    candidate_outcomes: list[dict[str, Any]] = []
    previous_round_findings: list[object] = []
    attempted_control_names: set[str] = set()

    def _compact_history_text(value: object, max_chars: int = 420) -> str:
        text = re.sub(r"\s+", " ", str(value)).strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    def _contract_validation_history_text(validation: object) -> str:
        if not isinstance(validation, dict):
            return ""
        profiles = validation.get("profiles")
        if not isinstance(profiles, list) or not profiles:
            return ""
        parts: list[str] = []
        for profile in profiles[:3]:
            if not isinstance(profile, dict):
                continue
            hypothesis_id = profile.get("hypothesis_id") or profile.get("hypothesis_name")
            signal_results = profile.get("signal_results", [])
            observed = 0
            total = 0
            if isinstance(signal_results, list):
                total = len(signal_results)
                observed = sum(
                    1 for item in signal_results if isinstance(item, dict) and item.get("observed")
                )
            if hypothesis_id:
                parts.append(f"{hypothesis_id}: observed {observed}/{total} signal(s)")
        return "; ".join(parts)

    def _profile_history_text(artifacts: dict[str, Any]) -> str:
        profiling_text = _compact_history_text(artifacts.get("profiling_text", ""), 320)
        profiling_json = artifacts.get("profiling_json", {})
        findings: list[str] = []
        if isinstance(profiling_json, dict):
            raw_findings = profiling_json.get("key_findings")
            if isinstance(raw_findings, list):
                findings.extend(_compact_history_text(item, 160) for item in raw_findings[:3])
            contract_text = _contract_validation_history_text(
                profiling_json.get("contract_hypothesis_validation")
            )
            if contract_text:
                findings.append(f"contract validation: {contract_text}")
        if findings:
            return _compact_history_text("; ".join(findings), 500)
        return profiling_text

    def _append_previous_round_finding(finding: object) -> None:
        if isinstance(finding, str):
            finding = finding.strip()
        if not finding:
            return
        previous_round_findings.append(finding)

    def _action_history_entries(actions: Sequence[dict[str, Any]]) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        for idx, action in enumerate(actions, start=1):
            preview = _format_spawn_action_preview(action).strip()
            if not preview:
                continue
            entry: dict[str, object] = {
                "action_index": idx,
                "action": preview,
                "config_type": str(action.get("config_type", "")),
            }
            parameter = action.get("parameter")
            if isinstance(parameter, str) and parameter.strip():
                entry["parameter"] = parameter.strip()
            reason = action.get("reason")
            if isinstance(reason, str) and reason.strip():
                entry["reason"] = _compact_history_text(reason, 360)
            entries.append(entry)
        return entries

    def _candidate_outcome_prompt_entry(outcome: dict[str, Any]) -> dict[str, object]:
        entry: dict[str, object] = {
            "candidate": str(outcome.get("label", "candidate")),
            "verdict": str(outcome.get("verdict", "unknown")),
        }
        wall_time = outcome.get("wall_time_s")
        if wall_time is not None:
            entry["wall_time_s"] = wall_time
        reason = outcome.get("reason")
        if reason:
            entry["reason"] = _compact_history_text(reason, 260)
        profile_summary = outcome.get("profile_summary")
        if profile_summary:
            entry["profile"] = _compact_history_text(profile_summary, 500)
        actions = outcome.get("actions")
        if isinstance(actions, list) and actions:
            entry["materialized_actions"] = [str(item) for item in actions]
        return entry

    def _round_history_entry(
        *,
        round_index: int,
        recommendation_summary: str,
        llm_response: str,
        recommended_actions: Sequence[dict[str, Any]],
        recommendation_evidence: Sequence[str],
        candidate_outcomes_for_round: Sequence[dict[str, Any]],
        result: str,
        current_profile_artifacts: dict[str, Any],
        current_changes: str | None = None,
        current_wall_time_s: float | None = None,
    ) -> dict[str, object]:
        action_entries = _action_history_entries(recommended_actions)
        all_candidate_results = [
            _candidate_outcome_prompt_entry(outcome)
            for outcome in candidate_outcomes_for_round
        ]

        for action_entry in action_entries:
            action_preview = action_entry.get("action")
            if not isinstance(action_preview, str):
                continue
            matching_results = [
                _candidate_outcome_prompt_entry(outcome)
                for outcome in candidate_outcomes_for_round
                if action_preview in [str(item) for item in outcome.get("actions", [])]
            ]
            if matching_results:
                action_entry["candidate_results"] = matching_results

        entry: dict[str, object] = {
            "round": round_index,
            "recommendation": recommendation_summary or "unavailable",
            "llm_response": llm_response or "unavailable",
            "evidence": list(recommendation_evidence) or ["none"],
            "recommended_actions": action_entries or [{"action": "none"}],
            "candidate_results": all_candidate_results or [],
            "result": result,
            "current_profile": _profile_history_text(current_profile_artifacts),
        }
        if current_changes is not None:
            entry["current_changes"] = current_changes
        if current_wall_time_s is not None:
            entry["current_wall_time_s"] = round(current_wall_time_s, 6)
        return entry

    def _command_main_tokens(
        command_parts: Sequence[str],
    ) -> tuple[list[str], list[str], dict[str, str]]:
        normalized_command, inline_env = _extract_inline_env_assignments(list(command_parts))
        base_executable = Path(cmd_parts[0]).name if cmd_parts else ""
        if not normalized_command or not base_executable:
            return [], normalized_command, inline_env
        if Path(normalized_command[0]).name == base_executable:
            return [], normalized_command, inline_env
        for idx, token in enumerate(normalized_command[1:], start=1):
            if Path(token).name == base_executable:
                return normalized_command[:idx], normalized_command[idx:], inline_env
        return [], normalized_command, inline_env

    def _display_option_values(values: list[str] | None) -> str:
        if values is None:
            return "(removed)"
        if not values:
            return "true"
        return ", ".join(value if value else "true" for value in values)

    baseline_change_order = _baseline_control_order(cmd_parts, inline_env_overrides)

    def _baseline_relative_changes(
        command_parts: Sequence[str],
        env_pairs: Sequence[tuple[str, str]],
    ) -> list[dict[str, str]]:
        prefix_tokens, main_command, inline_env = _command_main_tokens(command_parts)
        candidate_env = {str(name): str(value) for name, value in env_pairs}
        candidate_env.update(inline_env)
        base_env = {str(name): str(value) for name, value in inline_env_overrides.items()}
        changes: list[dict[str, str]] = []

        env_names = sorted(
            set(base_env) | set(candidate_env),
            key=lambda name: _control_sort_key(name, baseline_change_order),
        )
        for name in env_names:
            if candidate_env.get(name) != base_env.get(name):
                changes.append(
                    {
                        "name": f"env.{name}",
                        "new_value": candidate_env.get(name, "(unset)"),
                    }
                )

        if prefix_tokens:
            changes.append({"name": "command_prefix", "new_value": " ".join(prefix_tokens)})

        base_options, base_positionals = _parse_cli_command_signature(cmd_parts)
        candidate_options, candidate_positionals = _parse_cli_command_signature(main_command)
        option_names = sorted(
            set(base_options) | set(candidate_options),
            key=lambda name: _control_sort_key(name, baseline_change_order),
        )
        for name in option_names:
            if candidate_options.get(name) != base_options.get(name):
                changes.append(
                    {
                        "name": name,
                        "new_value": _display_option_values(candidate_options.get(name)),
                    }
                )

        if candidate_positionals != base_positionals:
            changes.append(
                {"name": "positionals", "new_value": " ".join(candidate_positionals)}
            )

        return [
            change
            for _, change in sorted(
                enumerate(changes),
                key=lambda item: _control_sort_key(
                    item[1]["name"], baseline_change_order, item[0]
                ),
            )
        ]

    def _tried_config_deltas() -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for command_tuple, env_tuple in seen_trials:
            changes = _baseline_relative_changes(command_tuple, env_tuple)
            if changes:
                entries.append({"changes": changes})
        return sorted(
            entries,
            key=lambda entry: json.dumps(entry["changes"], sort_keys=True),
        )[-12:]

    def _prior_findings_and_tried_changes_text() -> str:
        tried_deltas = _tried_config_deltas()
        tried_text = (
            json.dumps(tried_deltas, indent=2, sort_keys=True)
            if tried_deltas
            else "(baseline only; no changed candidate values tried yet)"
        )
        findings_text = (
            yaml.safe_dump(
                previous_round_findings,
                sort_keys=False,
                allow_unicode=True,
                width=10_000,
            )
            if previous_round_findings
            else "(none)"
        )
        return (
            "### Tried baseline-relative changes\n"
            f"{tried_text}\n\n"
            "### Findings by round\n"
            f"{findings_text}"
        )

    def _current_changes_history_text() -> str:
        changes = _baseline_relative_changes(
            current_command_parts,
            tuple(current_env_overrides.items()),
        )
        if not changes:
            return "baseline"
        return json.dumps(changes, sort_keys=True)

    _append_previous_round_finding(
        {
            "round": 0,
            "result": "baseline",
            "status": baseline_status,
            "wall_time_s": round(current_time, 6),
            "profile": _profile_history_text(current_profile_artifacts),
        }
    )

    def _current_fixed_goal_value() -> float | None:
        if primary_objective:
            return best_objective_value
        return best_time

    def _current_fixed_goal_mode() -> str:
        if primary_objective:
            return primary_objective_mode or "maximize"
        return "minimize"

    def _fixed_goal_is_reached() -> bool:
        return fixed_goal_reached(
            _current_fixed_goal_value(),
            fixed_goal,
            _current_fixed_goal_mode(),
        )

    def _final_stop_reason() -> str:
        return fixed_goal_stop_reason or fixed_budget_stop_reason or max_iterations_stop_reason

    def _candidate_trials_remaining() -> int:
        if fixed_budget_state.enabled:
            return cast(int, fixed_budget_state.remaining)
        if study_trace is not None:
            return max(
                0,
                study_trace.unique_execution_target - study_trace.unique_execution_count,
            )
        return max(0, max_iterations - total_candidate_trials_executed)

    def _fixed_goal_search_active() -> bool:
        return (
            fixed_goal is not None
            and not _fixed_goal_is_reached()
            and _candidate_trials_remaining() > 0
        )

    def _bounded_search_active() -> bool:
        return (
            fixed_budget_state.enabled
            or _fixed_goal_search_active()
            or (study_trace is not None and study_trace.search_active)
        )

    def _reprompt_scope() -> str:
        if fixed_budget_state.enabled:
            return "fixed budget"
        if study_trace is not None:
            return "deep-trace study"
        return "fixed goal"

    def _no_action_reprompts_used() -> int:
        if fixed_budget_state.enabled:
            return fixed_budget_state.no_action_reprompts_used
        return fixed_goal_no_action_reprompts_used

    def _max_no_action_reprompts() -> int:
        if fixed_budget_state.enabled:
            return fixed_budget_state.max_no_action_reprompts
        return fixed_goal_max_no_action_reprompts

    def _record_no_action_reprompt(stall_reprompts_used: int) -> bool:
        nonlocal fixed_goal_no_action_reprompts_used
        if fixed_budget_state.enabled:
            return fixed_budget_state.record_no_action_reprompt(stall_reprompts_used)
        if study_trace is not None:
            if not study_trace.search_active or stall_reprompts_used >= 3:
                return False
            fixed_goal_no_action_reprompts_used += 1
            return True
        if not _fixed_goal_search_active():
            return False
        if stall_reprompts_used >= fixed_goal_max_no_action_reprompts:
            return False
        fixed_goal_no_action_reprompts_used += 1
        return True

    def _capture_action_controls(actions: Sequence[dict[str, Any]]) -> None:
        for action in actions:
            parameter = action.get("parameter")
            if isinstance(parameter, str) and parameter.strip():
                attempted_control_names.add(parameter.strip())

            if action.get("config_type") != "command_prefix":
                continue

            new_value = action.get("new_value")
            if not isinstance(new_value, list):
                continue
            for token in new_value:
                if not isinstance(token, str) or "=" not in token:
                    continue
                name, _sep, _value = token.partition("=")
                if name and name != "env":
                    attempted_control_names.add(name)

    def _append_candidate_outcome(
        label: str,
        verdict: str,
        reason: str,
        *,
        wall_time_s: float | None = None,
        profile_artifacts: dict[str, Any] | None = None,
        actions: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "label": label,
            "verdict": verdict,
            "reason": reason,
        }
        if actions:
            entry["actions"] = [
                preview
                for preview in (_format_spawn_action_preview(action).strip() for action in actions)
                if preview
            ]
        if wall_time_s is not None:
            entry["wall_time_s"] = round(wall_time_s, 6)
        if profile_artifacts:
            profile_summary = _profile_history_text(profile_artifacts)
            if profile_summary:
                entry["profile_summary"] = profile_summary
        candidate_outcomes.append(entry)
        if len(candidate_outcomes) > 12:
            del candidate_outcomes[:-12]

    def _successful_configs_snapshot() -> list[dict[str, Any]]:
        snapshots: list[dict[str, Any]] = [
            {
                "label": "current accepted config",
                "command": current_command_parts,
                "env_overrides": current_env_overrides,
                "wall_time_s": round(current_time, 6),
                "primary_objective_value": current_objective_value,
            }
        ]
        if best_label != "(none)":
            snapshots.append(
                {
                    "label": "best accepted config",
                    "command": best_command_parts,
                    "env_overrides": best_env_overrides,
                    "wall_time_s": round(best_time, 6),
                    "primary_objective_value": best_objective_value,
                }
            )
        return snapshots

    def _build_fixed_budget_no_action_reprompt(
        base_prompt: str,
        reason: str,
        stall_reprompts_used: int,
    ) -> str:
        untouched_controls = sorted(
            (
                set(current_env_overrides.keys())
                if tunable_whitelist_list is None
                else set(tunable_whitelist_list)
            )
            - attempted_control_names
            - set(excluded_config_list)
        )
        remaining_budget = (
            fixed_budget_state.remaining
            if fixed_budget_state.enabled
            else _candidate_trials_remaining()
        )
        if fixed_budget_state.enabled:
            reprompt_heading = "Fixed-Budget"
        elif study_trace is not None:
            reprompt_heading = "Deep-Trace Study"
        else:
            reprompt_heading = "Fixed-Goal"
        whitelist_prompt_display = (
            json.dumps(tunable_whitelist_list)
            if tunable_whitelist_list is not None
            else "(not restricted)"
        )
        return (
            f"{base_prompt}\n\n"
            f"## {reprompt_heading} Re-prompt\n"
            f"Remaining candidate-attempt budget: {remaining_budget}\n"
            f"No-action reason: {reason}\n"
            "No-action re-prompts used for this stall: "
            f"{stall_reprompts_used}/"
            f"{_max_no_action_reprompts()}\n\n"
            "## Candidate History\n"
            "Tried baseline-relative changes: "
            f"{json.dumps(_tried_config_deltas(), indent=2, sort_keys=True)}\n"
            f"Evaluated/rejected candidates: {json.dumps(candidate_outcomes[-8:], indent=2)}\n"
            f"Successful configs: {json.dumps(_successful_configs_snapshot(), indent=2)}\n"
            f"Untouched current controls (where known): {json.dumps(untouched_controls)}\n"
            f"Protected/excluded controls: {json.dumps(excluded_config_list)}\n\n"
            f"Allowed tunable controls: {whitelist_prompt_display}\n\n"
            "## Re-prompt Requirements\n"
            "- Budget is fixed and only materialized concrete candidate attempts consume it.\n"
            "- Return at least one concrete safe action that can materialize into a new executable "
            "candidate; outputting a candidate is mandatory while any allowed, new, safe candidate "
            "can be formed.\n"
            "- For this re-prompt only, you may ignore whether direct evidence supports the "
            "decision, and you do not need to provide evidence to support the candidate.\n"
            "- If you think no untouched action exists, first try different concrete values for "
            "previously tried parameters or environment variables, as long as the resulting "
            "candidate is new, allowed, and safe. Continue exploring value variants until the "
            "allowed value space is exhausted or the budget is exhausted.\n"
            "- Do not repeat previously tried or rejected candidates exactly.\n"
            "- Keep the current command, current environment, immutable controls, and primary "
            "objective constraints in mind.\n"
            "- If no safe concrete candidate can be materialized, return the same JSON schema with "
            '"summary": "STOP:<machine_readable_reason>", empty "candidates", and concise '
            '"evidence" '
            "explaining why no candidate can be formed.\n"
        )

    iteration = 0
    while True:
        if study_trace is not None and study_trace.unique_execution_target_reached:
            break
        if study_trace is not None and study_trace.proposal_cap_reached:
            break
        if fixed_goal is not None and _fixed_goal_is_reached():
            fixed_goal_stop_reason = "fixed_goal_reached"
            break
        if fixed_budget_state.enabled and fixed_budget_state.remaining == 0:
            fixed_budget_stop_reason = "fixed_budget_exhausted"
            break
        if _candidate_trials_remaining() == 0:
            if fixed_goal is not None:
                fixed_goal_stop_reason = "max_iterations_exhausted"
            else:
                max_iterations_stop_reason = "max_iterations_exhausted"
            break

        iteration += 1
        total_rounds_executed = iteration
        _emit_tune_event(
            event_emitter,
            "phase.started",
            {
                "mode": "spawn",
                "phase": "spawn.iteration",
                "iteration": iteration,
            },
        )
        sink.print(f"\n[bold cyan]Round {iteration} Step 1: Agent Tuning Analysis[/bold cyan]")
        profiling_text, profiling_json, contract_validation = _print_profile_artifact_summary(
            sink,
            current_profile_artifacts,
            prefix=f"Current {evidence_kind} run",
        )

        prior_findings_and_tried_changes_text = _prior_findings_and_tried_changes_text()
        candidate_trials_remaining = _candidate_trials_remaining()
        max_candidates_this_response = (
            min(3, study_trace.proposal_cap - study_trace.proposal_count)
            if study_trace is not None
            else 3
        )

        memory_envelope_block = _render_effective_memory_envelope(profiling_json, memory_high)
        immutable_controls_rule = (
            "- Keep only the exact immutable controls listed above unchanged; do not treat "
            "same-library or related controls as immutable unless their exact names are listed."
        )
        tunable_whitelist_rule = (
            "- A tunable whitelist is active: recommend and change only exact controls listed in "
            "the Tunable Whitelist section. Return no actions if no listed control is supported "
            "by evidence."
            if tunable_whitelist_list is not None
            else (
                "- No tunable whitelist is active; supported non-immutable controls may be "
                "changed."
            )
        )
        evidence_heading = (
            "Refined Profiling Evidence" if profiling_enabled else "Raw Runtime Output Evidence"
        )
        evidence_rule = (
            "- You may use profiling evidence, generated contract hypotheses, contract validation "
            "findings, loaded manual context, and application manual hints, but evidence is "
            "optional and a candidate does not need evidence support."
            if profiling_enabled
            else (
                "- Profiling is disabled by --no-profile; you may use raw stdout/stderr, loaded "
                "manual context, generated contract hypotheses, contract validation findings, "
                "and application manual hints, but evidence is optional and a candidate does "
                "not need evidence support."
            )
        )
        profiled_run_rule = (
            "- The current accepted run was already profiled during execution, and candidate "
            "runs in this\n"
            "  iteration will also be profiled during execution."
            if profiling_enabled
            else "- The current accepted run was executed without profiling; candidate runs in "
            "this\n"
            "  iteration will also skip profiling and rely on raw runtime output."
        )

        prompt = f"""Analyze this spawn-mode workload and propose the next optimization actions.

## Current Command
{" ".join(current_command_parts)}

## Current Environment Overrides
{json.dumps(current_env_overrides, indent=2, sort_keys=True) if current_env_overrides else "(none)"}

## Candidate Base State
- Every recommended action is applied relative to the Current Command and Current Environment
  Overrides shown above.
- Previous recommendations, skipped candidates, and rejected candidates are history only; they are
  not active unless their changes appear in the Current Command or Current Environment Overrides.
- If you want to keep an environment or command-parameter value that is not in the Current
  Command/Environment, include it explicitly in this response.

## Immutable Controls
{excluded_config_content}

## Tunable Whitelist
{tunable_whitelist_content}

{memory_envelope_block}

## Current Wall Time
{current_time:.2f}s

## Current Primary Objective
{primary_objective or '(not configured)'}

## Current Primary Objective Value
{_format_metric_value(current_objective_value)}

## Current Program Output (excerpt)
{current_stdout[:2500] if current_stdout else "(no stdout)"}

## Previous Round Findings and Tried Changes
{prior_findings_and_tried_changes_text}

## {evidence_heading}
{profiling_json_for_prompt(profiling_json) if profiling_json else profiling_text[:2500]}

## Generated Contract Hypotheses
{generated_hypothesis_content or "(No generated contract hypotheses)"}

## Contract Hypothesis Validation
{
    profiling_json_for_prompt(contract_validation)
    if contract_validation
    else "(No contract-guided validation findings)"
}

## Loaded Manual Context
{loaded_manual_content if loaded_manual_content else "(No manual content loaded)"}

## Application Manual Hints
{manual_hint_content if manual_hint_content else "(No application manual hints provided)"}

## Rules
{evidence_rule}
- Generated contract hypotheses are the contract-derived plan; you may use their controls,
  signals, recipes, and hypothesis ids together with runtime validation evidence.
- Contract hypothesis validation, when present, is runtime-generated structured evidence.
  Prefer it over inferring raw DSL semantics yourself.
{profiled_run_rule}
- Avoid recommending exact commands already tried.
{immutable_controls_rule}
{tunable_whitelist_rule}
- Treat each object in `candidates` as one complete concrete candidate delta from the Candidate
  Base State, not as a patch on your previous suggestion.
- To test mixed environment-variable and command-parameter changes together, put them in the same
  candidate's `changes` object as exact-name-to-value mappings.
- Keys starting with `-` are command-line parameters; all other keys are environment variables.
- Separate candidate objects are alternatives; they are not sequential edits to combine into one
  command.
- Each materialized candidate is one runnable command plus its environment overrides.
- Keep workload-defining arguments fixed (model/input/token counts/repetitions/output mode)
  so comparisons stay fair.
- Recommend only options that are valid for the target executable.
- Do not reject or omit a candidate only because evidence is absent or insufficient.
- If evidence is absent or insufficient for an untouched control, try a different concrete value
  for a previously tried tunable control when that value has not produced an exact candidate before.
- Return no actions only after no safe new value variant remains within the allowed controls.
- Prefer new concrete values for already tried tunable controls over stopping early when the
  remaining budget can still test safe variants.
- Return JSON only (no markdown, no prose).
- Provide up to {max_candidates_this_response} candidate objects ordered best-first.

## JSON Schema
{{
  "summary": "short decision summary",
  "manual_citation": "optional concrete rule from loaded manual(s), or empty",
  "evidence": ["optional concise evidence item; may be empty", "..."],
  "hypothesis_refs": ["optional hypothesis ids to prioritize next profiling round"],
    "candidates": [
    {{
      "reason": "why this complete candidate should be tried",
      "manual_citation": "optional manual/contract rule supporting this candidate",
      "changes": {{
        "--exact-command-parameter": "concrete value",
        "EXACT_ENV_NAME": "concrete value"
      }}
    }}
  ]
}}

Do not use placeholder keys in real responses. Do not use bundled or alternation syntax such as
`AAA|BBB` anywhere. Values must be concrete scalar target values, not arrays or objects.
"""

        recommendation: dict[str, Any]
        candidate_attempts: list[dict[str, Any]] = []
        candidate_attempt_limit = candidate_trials_remaining
        candidate_hypothesis_profiles = startup_hypothesis_profiles
        stop_due_to_no_action = False
        continue_after_no_candidates = False
        no_action_reason = ""
        no_action_reprompts_this_stall = 0
        prompt_text = prompt
        analysis_unavailable = False
        round_candidate_outcome_start = len(candidate_outcomes)
        round_recommendation_summary = ""
        round_recommendation_evidence: list[str] = []
        round_llm_response_text = ""
        round_recommended_actions: list[dict[str, Any]] = []
        candidate_trace_proposals: dict[int, StudyProposal] = {}

        while True:
            analysis_unavailable = False
            candidate_trace_proposals = {}
            if benchmark_deterministic:
                analysis_wall_time_s = 0.0
                recommendation = {
                    "valid": True,
                    "error": "",
                    "summary": "Deterministic benchmark mode: skipped agent analysis.",
                    "manual_citation": "",
                    "evidence": ["SYSPILOT_BENCH_DETERMINISTIC=1"],
                    "hypothesis_refs": [],
                    "actions": [],
                    "candidates": [],
                    "raw_candidates": [],
                    "skipped_actions": [],
                }
                round_llm_response_text = str(recommendation["summary"])
                sink.print(
                    "\n[dim]Deterministic benchmark mode: no tuning actions generated.[/dim]"
                )
            else:
                analysis_started_at = time.time()
                try:
                    response = await _agent_step_for_prompt(agent, prompt_text)
                except LLMError as exc:
                    analysis_unavailable = True
                    analysis_wall_time_s = time.time() - analysis_started_at
                    logger.warning("Spawn-mode tuning analysis unavailable: %s", exc)
                    sink.print(f"\n[yellow]Agent tuning analysis unavailable: {exc}[/yellow]")
                    recommendation = {
                        "valid": True,
                        "error": "",
                        "summary": f"LLM tuning analysis unavailable: {exc}",
                        "manual_citation": "",
                        "evidence": [],
                        "hypothesis_refs": [],
                        "actions": [],
                        "candidates": [],
                        "raw_candidates": [],
                        "skipped_actions": [],
                    }
                    round_llm_response_text = str(recommendation["summary"])
                else:
                    analysis_wall_time_s = time.time() - analysis_started_at
                    sink.print(f"\n[dim]{response.content}[/dim]")
                    round_llm_response_text = response.content
                    if study_trace is not None:
                        candidate_trace_proposals = study_trace.record_proposals(
                            _extract_raw_spawn_candidates(response.content),
                            current_command_parts,
                            current_env_overrides,
                            metadata={
                                "round": iteration,
                                "current_wall_time_s": current_time,
                                "current_objective_value": current_objective_value,
                                "incumbent_wall_time_s": best_time,
                                "incumbent_objective_value": best_objective_value,
                            },
                        )
                    recommendation = _parse_spawn_actions(
                        response.content,
                        baseline_command_tokens=cmd_parts,
                        baseline_env_overrides=inline_env_overrides,
                    )

            if study_trace is not None:
                valid_source_indices = {
                    int(candidate["source_candidate_index"])
                    for candidate in recommendation.get("candidates", [])
                    if isinstance(candidate, dict)
                    and isinstance(candidate.get("source_candidate_index"), int)
                }
                invalid_reason = "; ".join(
                    str(item) for item in recommendation.get("skipped_actions", [])
                ) or str(recommendation.get("error", "candidate rejected during parsing"))
                for source_index, trace_proposal in candidate_trace_proposals.items():
                    if source_index not in valid_source_indices:
                        study_trace.validate_proposal(
                            trace_proposal,
                            accepted=False,
                            reason=invalid_reason,
                            metadata={"validation_stage": "response_parsing"},
                        )

            if not recommendation["valid"]:
                _emit_tune_event(
                    event_emitter,
                    "warning.raised",
                    {
                        "mode": "spawn",
                        "iteration": iteration,
                        "message": recommendation["error"],
                    },
                )
                if _record_no_action_reprompt(no_action_reprompts_this_stall):
                    no_action_reprompts_this_stall += 1
                    no_action_reason = recommendation["error"]
                    sink.print(
                        "\n[yellow]Agent recommendation was not actionable; re-prompting "
                        f"within {_reprompt_scope()}[/yellow]"
                    )
                    sink.print(f"  Reason: {no_action_reason}")
                    prompt_text = _build_fixed_budget_no_action_reprompt(
                        prompt,
                        no_action_reason,
                        no_action_reprompts_this_stall,
                    )
                    continue

                _emit_tune_event(
                    event_emitter,
                    "phase.finished",
                    {
                        "mode": "spawn",
                        "phase": "spawn.iteration",
                        "iteration": iteration,
                        "status": (
                            "unfillable_no_action"
                            if fixed_budget_state.enabled
                            else (
                                "max_iterations_exhausted"
                                if fixed_goal is not None
                                else "no_actionable_recommendation"
                            )
                        ),
                        **(
                            {"fixed_budget": fixed_budget_state.to_payload()}
                            if fixed_budget_state.enabled
                            else {}
                        ),
                    },
                )
                sink.print("\n[yellow]Agent recommendation was not actionable[/yellow]")
                sink.print(f"  Reason: {recommendation['error']}")
                if fixed_budget_state.enabled:
                    sink.print(
                        "  [yellow]Stopped: no valid action after "
                        f"{fixed_budget_state.no_action_reprompts_used} re-prompts[/yellow]"
                    )
                    fixed_budget_stop_reason = "unfillable_no_action"
                elif fixed_goal is not None:
                    fixed_goal_stop_reason = "max_iterations_exhausted"
                _append_previous_round_finding(
                    _round_history_entry(
                        round_index=iteration,
                        recommendation_summary=f"invalid ({recommendation['error']})",
                        llm_response=round_llm_response_text,
                        recommended_actions=[],
                        recommendation_evidence=[],
                        candidate_outcomes_for_round=[],
                        result="no_actionable_recommendation",
                        current_profile_artifacts=current_profile_artifacts,
                    )
                )
                stop_due_to_no_action = True
                break

            if recommendation["skipped_actions"]:
                sink.print("  Skipped invalid actions:")
                for item in recommendation["skipped_actions"]:
                    sink.print(f"    - {item}")

            summary = recommendation["summary"] or "No actionable tuning recommendation returned."
            next_action_display = _recommendation_next_action_display(recommendation["actions"])
            round_recommendation_summary = summary
            round_recommendation_evidence = [str(item) for item in recommendation["evidence"][:5]]
            round_recommended_actions = [dict(action) for action in recommendation["actions"]]
            recommendation_payload: dict[str, Any] = {
                "mode": "spawn",
                "iteration": iteration,
                "wall_time_s": analysis_wall_time_s,
                "summary": summary,
                "manual_citation": recommendation["manual_citation"],
                "evidence": recommendation["evidence"][:5],
                "candidate_count": len(recommendation["candidates"]),
                "action_count": len(recommendation["actions"]),
            }
            if next_action_display:
                recommendation_payload["next_action"] = next_action_display
            _emit_tune_event(
                event_emitter,
                "recommendation.ready",
                recommendation_payload,
            )
            sink.print(f"  Decision summary: {summary}")
            if next_action_display:
                sink.print(f"  Next action: {next_action_display}")
            if recommendation["manual_citation"]:
                sink.print(f"  Manual citation: {recommendation['manual_citation']}")
            if recommendation["evidence"]:
                for evidence_item in recommendation["evidence"][:5]:
                    sink.print(f"  Evidence: {evidence_item}")
            active_hypothesis_refs = recommendation.get("hypothesis_refs", [])
            if active_hypothesis_refs:
                sink.print(f"  Next hypothesis focus: {active_hypothesis_refs}")

            if active_hypothesis_refs:
                candidate_hypothesis_profiles = _extract_hypothesis_profiles(
                    structured_context_bundle,
                    active_hypothesis_refs,
                )
            else:
                candidate_hypothesis_profiles = startup_hypothesis_profiles

            candidate_specs = recommendation["candidates"]
            if study_trace is not None:
                candidate_specs = [
                    candidate
                    for candidate in candidate_specs
                    if candidate.get("source_candidate_index") in candidate_trace_proposals
                ]
            actions = recommendation["actions"]
            if not candidate_specs:
                if (
                    _bounded_search_active()
                    and not analysis_unavailable
                    and _record_no_action_reprompt(no_action_reprompts_this_stall)
                ):
                    no_action_reprompts_this_stall += 1
                    no_action_reason = summary
                    sink.print(
                        "\n[yellow]Agent recommended no actionable tuning; re-prompting within "
                        f"{_reprompt_scope()}[/yellow]"
                    )
                    prompt_text = _build_fixed_budget_no_action_reprompt(
                        prompt,
                        no_action_reason,
                        no_action_reprompts_this_stall,
                    )
                    continue

                _emit_tune_event(
                    event_emitter,
                    "phase.finished",
                    {
                        "mode": "spawn",
                        "phase": "spawn.iteration",
                        "iteration": iteration,
                        "status": (
                            "unfillable_no_action"
                            if fixed_budget_state.enabled and not analysis_unavailable
                            else (
                                "max_iterations_exhausted"
                                if fixed_goal is not None and not analysis_unavailable
                                else "no_actions"
                            )
                        ),
                        **(
                            {"fixed_budget": fixed_budget_state.to_payload()}
                            if fixed_budget_state.enabled
                            else {}
                        ),
                    },
                )
                sink.print("\n[yellow]Agent recommended no actionable tuning[/yellow]")
                if fixed_budget_state.enabled and not analysis_unavailable:
                    sink.print(
                        "  [yellow]Stopped: no valid action after "
                        f"{fixed_budget_state.no_action_reprompts_used} re-prompts[/yellow]"
                    )
                    fixed_budget_stop_reason = "unfillable_no_action"
                elif fixed_goal is not None and not analysis_unavailable:
                    fixed_goal_stop_reason = "max_iterations_exhausted"
                _append_previous_round_finding(
                    _round_history_entry(
                        round_index=iteration,
                        recommendation_summary=round_recommendation_summary,
                        llm_response=round_llm_response_text,
                        recommended_actions=round_recommended_actions,
                        recommendation_evidence=round_recommendation_evidence,
                        candidate_outcomes_for_round=[],
                        result="no_actions",
                        current_profile_artifacts=current_profile_artifacts,
                    )
                )
                stop_due_to_no_action = True
                break

            _capture_action_controls(actions)

            candidate_attempt_limit = _candidate_trials_remaining()
            if study_trace is not None:
                candidate_attempt_limit = len(candidate_specs)
            if fixed_budget_state.enabled:
                candidate_attempt_limit = min(
                    candidate_attempt_limit,
                    cast(int, fixed_budget_state.remaining),
                )

            candidate_attempts = _materialize_spawn_candidate_attempts_from_candidate_sets(
                current_command_parts,
                candidate_specs,
                max_attempts=candidate_attempt_limit,
                base_env_overrides=current_env_overrides,
                excluded_config_names=excluded_config_list,
                tunable_whitelist_names=tunable_whitelist_list,
            )
            if study_trace is not None:
                attempted_source_indices: set[int] = set()
                for attempt in candidate_attempts:
                    source_index = attempt.get("source_candidate_index")
                    if isinstance(source_index, int):
                        attempted_source_indices.add(source_index)
                        attempt["study_proposal"] = candidate_trace_proposals.get(source_index)
                for candidate_spec in candidate_specs:
                    source_index = candidate_spec.get("source_candidate_index")
                    if (
                        isinstance(source_index, int)
                        and source_index not in attempted_source_indices
                        and source_index in candidate_trace_proposals
                    ):
                        study_trace.validate_proposal(
                            candidate_trace_proposals[source_index],
                            accepted=False,
                            reason="candidate excluded by active candidate-attempt limit",
                            metadata={"validation_stage": "candidate_attempt_limit"},
                        )

            if candidate_attempts:
                break

            _emit_tune_event(
                event_emitter,
                "phase.progress",
                {
                    "mode": "spawn",
                    "phase": "spawn.iteration",
                    "iteration": iteration,
                    "status": "no_new_candidates",
                },
            )
            sink.print(
                "\n[yellow]No new executable tuning actions remained after validation[/yellow]"
            )

            if _record_no_action_reprompt(no_action_reprompts_this_stall):
                no_action_reprompts_this_stall += 1
                no_action_reason = "no_new_candidates_after_validation"
                prompt_text = _build_fixed_budget_no_action_reprompt(
                    prompt,
                    no_action_reason,
                    no_action_reprompts_this_stall,
                )
                continue

            if fixed_budget_state.enabled:
                _emit_tune_event(
                    event_emitter,
                    "phase.finished",
                    {
                        "mode": "spawn",
                        "phase": "spawn.iteration",
                        "iteration": iteration,
                        "status": "candidate_space_exhausted",
                        "fixed_budget": fixed_budget_state.to_payload(),
                    },
                )
                sink.print("  [yellow]Stopped: candidate space exhausted before budget[/yellow]")
                fixed_budget_stop_reason = "candidate_space_exhausted"
                stop_due_to_no_action = True
                break

            if fixed_goal is not None:
                _emit_tune_event(
                    event_emitter,
                    "phase.finished",
                    {
                        "mode": "spawn",
                        "phase": "spawn.iteration",
                        "iteration": iteration,
                        "status": "max_iterations_exhausted",
                    },
                )
                fixed_goal_stop_reason = "max_iterations_exhausted"
                stop_due_to_no_action = True
                break

            _emit_tune_event(
                event_emitter,
                "phase.finished",
                {
                    "mode": "spawn",
                    "phase": "spawn.iteration",
                    "iteration": iteration,
                    "status": "no_new_candidates",
                },
            )
            stagnation_iterations += 1
            if study_trace is None and stagnation_iterations >= convergence_patience:
                _emit_tune_event(
                    event_emitter,
                    "phase.finished",
                    {
                        "mode": "spawn",
                        "phase": "spawn.iteration",
                        "iteration": iteration,
                        "status": "converged",
                    },
                )
                stop_due_to_no_action = True
                break
            continue_after_no_candidates = True
            break

        if continue_after_no_candidates:
            continue
        if stop_due_to_no_action:
            break

        sink.print(
            f"\n[bold cyan]Round {iteration} Step 2: Candidate Search "
            f"({len(candidate_attempts)} candidate attempt(s), max "
            f"{candidate_attempt_limit})[/bold cyan]"
        )

        iteration_best_time = current_time
        iteration_best_stdout = current_stdout
        iteration_best_stderr = current_stderr
        iteration_best_label = "(none)"
        iteration_best_cmd = list(current_command_parts)
        iteration_best_env = dict(current_env_overrides)
        iteration_best_profiling_text = current_profiling_text
        iteration_best_profiling_json = current_profiling_json
        iteration_best_contract_validation = current_contract_validation
        iteration_best_profile_artifacts = dict(current_profile_artifacts)
        iteration_best_metrics = dict(current_metrics)
        iteration_best_objective_value = current_objective_value
        iteration_best_trial = 0
        iteration_best_candidate_trials_to_find = total_candidate_trials_executed
        iteration_best_profiled_runs_to_find = total_profiled_runs
        successful_trials = 0
        non_improving_trials = 0

        for idx, candidate in enumerate(candidate_attempts, start=1):
            if study_trace is not None and study_trace.unique_execution_target_reached:
                for remaining_attempt in candidate_attempts[idx - 1 :]:
                    remaining_proposal = remaining_attempt.get("study_proposal")
                    if isinstance(remaining_proposal, StudyProposal):
                        study_trace.validate_proposal(
                            remaining_proposal,
                            accepted=False,
                            reason="unique execution target reached before candidate launch",
                            metadata={
                                "validation_stage": "unique_execution_target",
                                "round": iteration,
                            },
                        )
                break
            total_candidate_trials_executed += 1
            if fixed_budget_state.enabled:
                fixed_budget_state.consume(
                    "candidate_attempt",
                    str(candidate["label"]),
                    str(candidate.get("materialized_reason", "candidate materialized")),
                )

            trial_plan = candidate
            trial_cmd = trial_plan["cmd"]
            trial_env = trial_plan["env_overrides"]
            trial_label = trial_plan["label"]
            skipped_execution_actions = trial_plan["skipped_actions"]
            trial_actions = trial_plan.get("actions", [])
            study_proposal = trial_plan.get("study_proposal")

            materialized_status = str(trial_plan.get("materialized_status", "ready"))
            if study_trace is not None and isinstance(study_proposal, StudyProposal):
                trace_disposition = study_trace.validate_proposal(
                    study_proposal,
                    accepted=materialized_status == "ready",
                    reason=str(trial_plan.get("materialized_reason", "candidate materialized")),
                    metadata={
                        "validation_stage": "materialization",
                        "round": iteration,
                        "trial": idx,
                    },
                )
                if trace_disposition != "candidate":
                    materialized_status = trace_disposition
                    trial_plan["materialized_reason"] = (
                        "candidate duplicated a previously proposed or baseline configuration"
                        if trace_disposition == "duplicate"
                        else study_proposal.error
                        or str(trial_plan.get("materialized_reason", "candidate rejected"))
                    )
            if materialized_status != "ready":
                _append_candidate_outcome(
                    str(trial_label),
                    materialized_status,
                    str(trial_plan.get("materialized_reason", "candidate rejected")),
                    actions=cast(list[dict[str, Any]], trial_actions),
                )
                evaluated_payload: dict[str, object] = {
                    "mode": "spawn",
                    "iteration": iteration,
                    "candidate_id": idx,
                    "label": trial_label,
                    "verdict": materialized_status,
                    "cmd_diff": trial_label,
                    "reason": str(trial_plan.get("materialized_reason", "candidate rejected")),
                }
                if fixed_budget_state.enabled:
                    evaluated_payload["fixed_budget"] = fixed_budget_state.to_payload()
                _emit_tune_event(
                    event_emitter,
                    "candidate.evaluated",
                    evaluated_payload,
                )
                sink.print(f"  Trial {idx}/{len(candidate_attempts)}: {trial_label}")
                sink.print(
                    "    [yellow]Materialized candidate rejected before execution: "
                    f"{trial_plan.get('materialized_reason', 'candidate rejected')}[/yellow]"
                )
                continue

            trial_key = _command_key(trial_cmd, trial_env)
            if trial_key in seen_trials:
                _append_candidate_outcome(
                    str(trial_label),
                    "duplicate",
                    "candidate duplicated a previously tried command",
                    actions=cast(list[dict[str, Any]], trial_actions),
                )
                evaluated_payload = {
                    "mode": "spawn",
                    "iteration": iteration,
                    "candidate_id": idx,
                    "label": trial_label,
                    "verdict": "duplicate",
                    "cmd_diff": trial_label,
                    "reason": "candidate duplicated a previously tried command",
                }
                if fixed_budget_state.enabled:
                    evaluated_payload["fixed_budget"] = fixed_budget_state.to_payload()
                _emit_tune_event(
                    event_emitter,
                    "candidate.evaluated",
                    evaluated_payload,
                )
                sink.print(f"  Trial {idx}/{len(candidate_attempts)}: {trial_label}")
                sink.print(
                    "    [yellow]Materialized candidate duplicated a previously "
                    "tried command[/yellow]"
                )
                continue

            _emit_tune_event(
                event_emitter,
                "progress.update",
                {
                    "mode": "spawn",
                    "phase": "spawn.candidate_search",
                    "iteration": iteration,
                    "current_trial": idx,
                    "total_trials": len(candidate_attempts),
                    "candidate_label": trial_label,
                },
            )

            study_execution_sequence: int | None = None
            if study_trace is not None and isinstance(study_proposal, StudyProposal):
                study_execution_sequence = study_trace.record_execution_started(
                    study_proposal,
                    metadata={
                        "round": iteration,
                        "trial": idx,
                        "label": trial_label,
                        "current_wall_time_s": current_time,
                        "current_objective_value": current_objective_value,
                        "incumbent_wall_time_s": best_time,
                        "incumbent_objective_value": best_objective_value,
                    },
                )

            while True:
                seen_trials.add(_command_key(trial_cmd, trial_env))
                sink.print(f"  Trial {idx}/{len(candidate_attempts)}: {trial_label}")
                sink.print(f"    Running: {' '.join(trial_cmd)}")
                if trial_env:
                    sink.print(f"    Environment overrides: {trial_env}")
                if skipped_execution_actions:
                    for item in skipped_execution_actions:
                        sink.print(f"    Skipped: {item}")

                _emit_tune_event(
                    event_emitter,
                    "exec.started",
                    {
                        "mode": "spawn",
                        "phase": "spawn.candidate",
                        "iteration": iteration,
                        "candidate_id": idx,
                        "label": trial_label,
                    },
                )
                _emit_tune_event(
                    event_emitter,
                    "profile.started",
                    {
                        "mode": "spawn",
                        "phase": "spawn.candidate",
                        "iteration": iteration,
                        "candidate_id": idx,
                        "candidate_label": trial_label,
                        "run_label": trial_label,
                        "profiling_enabled": profiling_enabled,
                    },
                )
                trial_artifacts = await _run_live_profiling_pipeline(
                    agent,
                    trial_cmd,
                    timeout,
                    workdir=workdir,
                    memory_high=memory_high,
                    env_overrides=trial_env,
                    hypothesis_profiles=candidate_hypothesis_profiles,
                    live_output_callback=_plain_live_output_callback,
                    skip_profiling=not profiling_enabled,
                )
                total_profiled_runs += 1
                trial_time = trial_artifacts["wall_time"]
                trial_stdout = trial_artifacts.get("stdout", "")
                trial_stderr = trial_artifacts.get("stderr", "")
                trial_rc = trial_artifacts["returncode"]
                trial_timed_out = bool(trial_artifacts.get("timed_out"))
                _emit_tune_event(
                    event_emitter,
                    "exec.finished",
                    {
                        "mode": "spawn",
                        "phase": "spawn.candidate",
                        "iteration": iteration,
                        "candidate_id": idx,
                        "label": trial_label,
                        "returncode": trial_rc,
                        "wall_time_s": trial_time,
                    },
                )
                _emit_profile_artifact_events(
                    event_emitter,
                    trial_artifacts,
                    phase="spawn.candidate",
                    iteration=iteration,
                    candidate_id=idx,
                    candidate_label=trial_label,
                )
                _emit_tune_event(
                    event_emitter,
                    "progress.finished",
                    {
                        "mode": "spawn",
                        "phase": "spawn.candidate_search",
                        "iteration": iteration,
                        "current_trial": idx,
                        "total_trials": len(candidate_attempts),
                        "candidate_id": idx,
                        "candidate_label": trial_label,
                        "returncode": trial_rc,
                        "wall_time_s": trial_time,
                        **_event_output_payload(trial_stdout, trial_stderr),
                    },
                )

                if trial_rc == 0:
                    break

                retry_plan, removed_options = _retry_spawn_candidate_without_unsupported_options(
                    current_command_parts,
                    cast(list[dict[str, Any]], trial_actions),
                    base_env_overrides=current_env_overrides,
                    excluded_config_names=excluded_config_list,
                    tunable_whitelist_names=tunable_whitelist_list,
                    unsupported_options=_extract_runtime_unsupported_options(
                        trial_stdout,
                        trial_stderr,
                    ),
                )
                if not retry_plan:
                    break

                retry_key = _command_key(retry_plan["cmd"], retry_plan["env_overrides"])
                if retry_key in seen_trials:
                    break

                if study_trace is not None:
                    if study_execution_sequence is not None:
                        study_trace.record_execution_finished(
                            study_execution_sequence,
                            status="timeout" if trial_timed_out else "error",
                            wall_time_s=float(trial_time),
                            objective_name=None,
                            objective_value=None,
                            returncode=int(trial_rc),
                            timed_out=trial_timed_out,
                            failure_reason="runtime_unsupported_option",
                        )
                        study_execution_sequence = None
                    retry_study_proposal = study_trace.record_runtime_candidate(
                        retry_plan["cmd"],
                        retry_plan["env_overrides"],
                        metadata={
                            "round": iteration,
                            "trial": idx,
                            "source": "unsupported_option_retry",
                            "removed_options": removed_options,
                        },
                    )
                    if retry_study_proposal is None:
                        break
                    retry_disposition = study_trace.validate_proposal(
                        retry_study_proposal,
                        accepted=True,
                        reason="runtime retry after removing unsupported options",
                        metadata={"validation_stage": "runtime_retry"},
                    )
                    if (
                        retry_disposition != "candidate"
                        or study_trace.unique_execution_target_reached
                    ):
                        break
                    study_proposal = retry_study_proposal
                    study_execution_sequence = study_trace.record_execution_started(
                        retry_study_proposal,
                        metadata={
                            "round": iteration,
                            "trial": idx,
                            "label": retry_plan["label"],
                            "source": "unsupported_option_retry",
                        },
                    )

                sink.print(
                    "    [yellow]Retrying without unsupported option(s): "
                    f"{', '.join(removed_options)}[/yellow]"
                )
                trial_plan = retry_plan
                trial_cmd = retry_plan["cmd"]
                trial_env = retry_plan["env_overrides"]
                trial_label = retry_plan["label"]
                skipped_execution_actions = retry_plan["skipped_actions"]
                trial_actions = retry_plan.get("actions", [])

            if trial_rc != 0:
                if study_trace is not None and study_execution_sequence is not None:
                    study_trace.record_execution_finished(
                        study_execution_sequence,
                        status="timeout" if trial_timed_out else "error",
                        wall_time_s=float(trial_time),
                        objective_name=None,
                        objective_value=None,
                        returncode=int(trial_rc),
                        timed_out=trial_timed_out,
                    )
                _append_candidate_outcome(
                    str(trial_label),
                    "failed",
                    f"returncode={trial_rc}",
                    wall_time_s=float(trial_time),
                    profile_artifacts=cast(dict[str, Any], trial_artifacts),
                    actions=cast(list[dict[str, Any]], trial_actions),
                )
                _emit_tune_event(
                    event_emitter,
                    "candidate.evaluated",
                    {
                        "mode": "spawn",
                        "iteration": iteration,
                        "candidate_id": idx,
                        "label": trial_label,
                        "verdict": "failed",
                        "returncode": trial_rc,
                        "wall_time_s": trial_time,
                        "cmd_diff": trial_label,
                    },
                )
                sink.print(f"    [yellow]Trial failed (rc={trial_rc})[/yellow]")
                sink.print(f"    [dim]stderr: {trial_stderr[:240]}[/dim]")
                non_improving_trials += 1
                if (
                    fixed_goal is None
                    and not fixed_budget_state.enabled
                    and study_trace is None
                    and non_improving_trials >= convergence_patience
                    and successful_trials > 0
                ):
                    sink.print("  Candidate search converged after non-improving trials")
                    break
                continue

            trial_metrics: dict[str, Any] = {}
            trial_objective_value: float | None = None
            current_change_pct = (
                ((current_time - trial_time) / current_time) * 100 if current_time > 0 else 0
            )
            current_change_label = "Wall time"
            current_change_value = f"{trial_time:.2f}s"
            trial_is_better_than_iteration_best = trial_time < iteration_best_time

            if primary_objective and resolved_primary_objective_metric:
                trial_metrics = await _extract_metrics_from_output(
                    agent,
                    trial_stdout,
                    "candidate run with actions: "
                    f"{trial_label}; primary objective query: {primary_objective}",
                )
                trial_objective_value = _extract_primary_objective_value(
                    trial_metrics,
                    resolved_primary_objective_metric,
                )
                if trial_objective_value is None:
                    if study_trace is not None and study_execution_sequence is not None:
                        study_trace.record_execution_finished(
                            study_execution_sequence,
                            status="error",
                            wall_time_s=float(trial_time),
                            objective_name=resolved_primary_objective_metric,
                            objective_value=None,
                            returncode=int(trial_rc),
                            timed_out=False,
                            failure_reason="missing_primary_objective",
                        )
                    _append_candidate_outcome(
                        str(trial_label),
                        "missing_primary_objective",
                        f"missing numeric primary objective {resolved_primary_objective_metric}",
                        wall_time_s=float(trial_time),
                        profile_artifacts=cast(dict[str, Any], trial_artifacts),
                        actions=cast(list[dict[str, Any]], trial_actions),
                    )
                    _emit_tune_event(
                        event_emitter,
                        "candidate.evaluated",
                        {
                            "mode": "spawn",
                            "iteration": iteration,
                            "candidate_id": idx,
                            "label": trial_label,
                            "verdict": "missing_primary_objective",
                            "wall_time_s": trial_time,
                            "cmd_diff": trial_label,
                            "primary_objective_query": primary_objective,
                            "primary_objective_metric": resolved_primary_objective_metric,
                            "primary_objective_mode": primary_objective_mode or "",
                        },
                    )
                    sink.print(
                        "    [yellow]Trial missing numeric primary objective "
                        f"'{resolved_primary_objective_metric}' for query "
                        f"'{primary_objective}'; rejected[/yellow]"
                    )
                    non_improving_trials += 1
                    if (
                        fixed_goal is None
                        and not fixed_budget_state.enabled
                        and study_trace is None
                        and non_improving_trials >= convergence_patience
                        and successful_trials > 0
                    ):
                        sink.print("  Candidate search converged after non-improving trials")
                        break
                    continue

                assert primary_objective_mode is not None
                iteration_best_reference_value = (
                    cast(float, iteration_best_objective_value)
                    if iteration_best_objective_value is not None
                    else cast(float, current_objective_value)
                )
                best_order = _compare_objective_values(
                    trial_objective_value,
                    iteration_best_reference_value,
                    primary_objective_mode,
                )
                current_change_pct = _compute_objective_change_pct(
                    cast(float, current_objective_value),
                    trial_objective_value,
                    primary_objective_mode,
                )
                current_change_label = _objective_display_name(primary_objective)
                current_change_value = _format_metric_value(trial_objective_value)
                trial_is_better_than_iteration_best = best_order > 0 or (
                    best_order == 0 and trial_time < iteration_best_time
                )

            if study_trace is not None and study_execution_sequence is not None:
                study_trace.record_execution_finished(
                    study_execution_sequence,
                    status="ok",
                    wall_time_s=float(trial_time),
                    objective_name=resolved_primary_objective_metric,
                    objective_value=trial_objective_value,
                    returncode=int(trial_rc),
                    timed_out=False,
                )

            successful_trials += 1
            _append_candidate_outcome(
                str(trial_label),
                "improved" if trial_is_better_than_iteration_best else "not_improved",
                (
                    "candidate outperformed the current configuration"
                    if trial_is_better_than_iteration_best
                    else "candidate did not outperform the current configuration"
                ),
                wall_time_s=float(trial_time),
                profile_artifacts=cast(dict[str, Any], trial_artifacts),
                actions=cast(list[dict[str, Any]], trial_actions),
            )

            _emit_tune_event(
                event_emitter,
                "candidate.evaluated",
                {
                    "mode": "spawn",
                    "iteration": iteration,
                    "candidate_id": idx,
                    "label": trial_label,
                    "verdict": (
                        "improved" if trial_is_better_than_iteration_best else "not_improved"
                    ),
                    "wall_time_s": trial_time,
                    "change_pct_vs_current": current_change_pct,
                    "cmd_diff": trial_label,
                    "primary_objective_query": primary_objective or "",
                    "primary_objective_metric": resolved_primary_objective_metric or "",
                    "primary_objective_mode": primary_objective_mode or "",
                    "primary_objective_value": trial_objective_value,
                },
            )
            sink.print(
                f"    {current_change_label}: {current_change_value} "
                f"({current_change_pct:+.1f}% vs current)"
            )
            if primary_objective and resolved_primary_objective_metric:
                wall_time_change_pct = (
                    ((current_time - trial_time) / current_time) * 100 if current_time > 0 else 0
                )
                sink.print(
                    f"    Wall time: {trial_time:.2f}s ({wall_time_change_pct:+.1f}% vs current)"
                )

            if trial_is_better_than_iteration_best:
                iteration_best_time = trial_time
                iteration_best_stdout = trial_stdout
                iteration_best_stderr = trial_stderr
                iteration_best_label = trial_label
                iteration_best_cmd = trial_cmd
                iteration_best_env = trial_env
                iteration_best_profiling_text = str(trial_artifacts.get("profiling_text", ""))
                trial_profiling_json = trial_artifacts.get("profiling_json", {})
                iteration_best_profiling_json = (
                    trial_profiling_json if isinstance(trial_profiling_json, dict) else {}
                )
                trial_contract_validation = iteration_best_profiling_json.get(
                    "contract_hypothesis_validation", {}
                )
                iteration_best_contract_validation = (
                    trial_contract_validation if isinstance(trial_contract_validation, dict) else {}
                )
                iteration_best_profile_artifacts = dict(trial_artifacts)
                iteration_best_metrics = dict(trial_metrics)
                iteration_best_objective_value = trial_objective_value
                iteration_best_trial = idx
                iteration_best_candidate_trials_to_find = total_candidate_trials_executed
                iteration_best_profiled_runs_to_find = total_profiled_runs
                if current_change_pct >= min_improvement_pct:
                    non_improving_trials = 0
                else:
                    non_improving_trials += 1
            else:
                non_improving_trials += 1

            if (
                fixed_goal is None
                and not fixed_budget_state.enabled
                and study_trace is None
                and non_improving_trials >= convergence_patience
                and successful_trials > 0
            ):
                sink.print("  Candidate search converged after non-improving trials")
                break

        if successful_trials == 0:
            _emit_tune_event(
                event_emitter,
                "phase.finished",
                {
                    "mode": "spawn",
                    "phase": "spawn.iteration",
                    "iteration": iteration,
                    "status": "no_successful_trials",
                },
            )
            sink.print("  [yellow]No candidate completed successfully in this iteration[/yellow]")
            _append_previous_round_finding(
                _round_history_entry(
                    round_index=iteration,
                    recommendation_summary=round_recommendation_summary,
                    llm_response=round_llm_response_text,
                    recommended_actions=round_recommended_actions,
                    recommendation_evidence=round_recommendation_evidence,
                    candidate_outcomes_for_round=candidate_outcomes[round_candidate_outcome_start:],
                    result="no_successful_trials",
                    current_profile_artifacts=current_profile_artifacts,
                )
            )
            if fixed_budget_state.enabled:
                if fixed_budget_state.remaining == 0:
                    fixed_budget_stop_reason = "fixed_budget_exhausted"
                    break
                continue
            if fixed_goal is not None:
                if _candidate_trials_remaining() == 0:
                    fixed_goal_stop_reason = "max_iterations_exhausted"
                    break
                continue
            stagnation_iterations += 1
            if study_trace is None and stagnation_iterations >= convergence_patience:
                break
            continue

        improvement_pct = (
            ((current_time - iteration_best_time) / current_time) * 100 if current_time > 0 else 0
        )
        iteration_improved = iteration_best_time < current_time
        if primary_objective and primary_objective_mode:
            if iteration_best_objective_value is not None and current_objective_value is not None:
                improvement_pct = _compute_objective_change_pct(
                    cast(float, current_objective_value),
                    cast(float, iteration_best_objective_value),
                    primary_objective_mode,
                )
                objective_order = _compare_objective_values(
                    cast(float, iteration_best_objective_value),
                    cast(float, current_objective_value),
                    primary_objective_mode,
                )
                iteration_improved = objective_order > 0 or (
                    objective_order == 0 and iteration_best_time < current_time
                )

        if iteration_improved:
            current_time = iteration_best_time
            current_stdout = iteration_best_stdout
            current_stderr = iteration_best_stderr
            current_command_parts = list(iteration_best_cmd)
            current_env_overrides = dict(iteration_best_env)
            current_profiling_text = iteration_best_profiling_text
            current_profiling_json = iteration_best_profiling_json
            current_contract_validation = iteration_best_contract_validation
            current_profile_artifacts = dict(iteration_best_profile_artifacts)
            current_metrics = dict(iteration_best_metrics)
            current_objective_value = iteration_best_objective_value
            sink.print(f"  Accepted candidate: {iteration_best_label}")
            if primary_objective and current_objective_value is not None:
                sink.print(
                    f"  New current {_objective_display_name(primary_objective)}: "
                    f"{_format_metric_value(current_objective_value)} "
                    f"({improvement_pct:+.1f}% vs previous)"
                )
                sink.print(f"  Current wall time: {current_time:.2f}s")
            else:
                sink.print(
                    f"  New current wall time: {current_time:.2f}s "
                    f"({improvement_pct:+.1f}% vs previous)"
                )

            best_candidate_wins = current_time < best_time
            if primary_objective and primary_objective_mode:
                if best_objective_value is not None and current_objective_value is not None:
                    best_order = _compare_objective_values(
                        cast(float, current_objective_value),
                        cast(float, best_objective_value),
                        primary_objective_mode,
                    )
                    best_candidate_wins = best_order > 0 or (
                        best_order == 0 and current_time < best_time
                    )

            if best_candidate_wins:
                best_time = current_time
                best_label = iteration_best_label
                best_command_parts = list(current_command_parts)
                best_env_overrides = dict(current_env_overrides)
                best_metrics = dict(current_metrics)
                best_objective_value = current_objective_value
                best_found_round = iteration
                best_found_trial = iteration_best_trial
                best_candidate_trials_to_find = iteration_best_candidate_trials_to_find
                best_profiled_runs_to_find = iteration_best_profiled_runs_to_find
                best_config_find_wall_time_s = time.time() - tune_started_at

            if improvement_pct >= min_improvement_pct:
                stagnation_iterations = 0
            else:
                stagnation_iterations += 1
            _emit_tune_event(
                event_emitter,
                "phase.finished",
                {
                    "mode": "spawn",
                    "phase": "spawn.iteration",
                    "iteration": iteration,
                    "status": "improved",
                    "improvement_pct": improvement_pct,
                },
            )
        else:
            sink.print("  No candidate outperformed the current configuration")
            if fixed_budget_state.enabled and fixed_budget_state.remaining == 0:
                fixed_budget_stop_reason = "fixed_budget_exhausted"
            stagnation_iterations += 1
            _emit_tune_event(
                event_emitter,
                "phase.finished",
                {
                    "mode": "spawn",
                    "phase": "spawn.iteration",
                    "iteration": iteration,
                    "status": (
                        "fixed_budget_exhausted"
                        if fixed_budget_state.enabled and fixed_budget_state.remaining == 0
                        else "no_improvement"
                    ),
                    **(
                        {"fixed_budget": fixed_budget_state.to_payload()}
                        if fixed_budget_state.enabled
                        else {}
                    ),
                },
            )

        _append_previous_round_finding(
            _round_history_entry(
                round_index=iteration,
                recommendation_summary=round_recommendation_summary,
                llm_response=round_llm_response_text,
                recommended_actions=round_recommended_actions,
                recommendation_evidence=round_recommendation_evidence,
                candidate_outcomes_for_round=candidate_outcomes[round_candidate_outcome_start:],
                result="accepted improvement" if iteration_improved else "no improvement",
                current_profile_artifacts=current_profile_artifacts,
                current_changes=_current_changes_history_text(),
                current_wall_time_s=current_time,
            )
        )

        if fixed_goal is not None and _fixed_goal_is_reached():
            sink.print(
                "  Stopped: fixed goal reached "
                f"({_format_metric_value(_current_fixed_goal_value())} "
                f"target, threshold {_format_metric_value(fixed_goal)})"
            )
            fixed_goal_stop_reason = "fixed_goal_reached"
            break

        if fixed_budget_state.enabled and fixed_budget_state.remaining == 0:
            sink.print(
                "  Stopped: fixed budget exhausted after "
                f"{fixed_budget_state.used}/{fixed_budget_state.total} candidate attempts"
            )
            _emit_tune_event(
                event_emitter,
                "phase.progress",
                {
                    "mode": "spawn",
                    "phase": "spawn.iteration",
                    "iteration": iteration,
                    "status": "fixed_budget_exhausted",
                    "fixed_budget": fixed_budget_state.to_payload(),
                },
            )
            fixed_budget_stop_reason = "fixed_budget_exhausted"
            break

        if study_trace is not None and study_trace.unique_execution_target_reached:
            sink.print(
                "  Stopped: deep-trace unique execution target reached after "
                f"{study_trace.unique_execution_count}/"
                f"{study_trace.unique_execution_target} non-baseline launches"
            )
            break

        if study_trace is not None and study_trace.proposal_cap_reached:
            sink.print(
                "  Stopped: deep-trace proposal cap reached after "
                f"{study_trace.proposal_count}/{study_trace.proposal_cap} proposals"
            )
            break

        if _candidate_trials_remaining() == 0:
            if fixed_goal is not None:
                fixed_goal_stop_reason = "max_iterations_exhausted"
                sink.print("  Stopped: max candidate trial limit exhausted before fixed goal")
            else:
                max_iterations_stop_reason = "max_iterations_exhausted"
                sink.print(
                    "  Stopped: max candidate trial limit exhausted after "
                    f"{total_candidate_trials_executed}/{max_iterations} candidate attempts"
                )
            _emit_tune_event(
                event_emitter,
                "phase.progress",
                {
                    "mode": "spawn",
                    "phase": "spawn.iteration",
                    "iteration": iteration,
                    "status": "max_iterations_exhausted",
                    "total_candidate_trials_executed": total_candidate_trials_executed,
                    "max_iterations": max_iterations,
                },
            )
            break

        if (
            fixed_goal is None
            and not fixed_budget_state.enabled
            and study_trace is None
            and stagnation_iterations >= convergence_patience
        ):
            sink.print("  Iterative loop converged after non-improving iterations")
            _emit_tune_event(
                event_emitter,
                "phase.progress",
                {
                    "mode": "spawn",
                    "phase": "spawn.iteration",
                    "iteration": iteration,
                    "status": "converged",
                },
            )
            break

    action_label = best_label
    tuned_time = best_time
    if action_label == "(none)":
        sink.print("  No tuned candidate outperformed baseline; keeping baseline configuration")
        sink.print("  Best found at: baseline (round 0 before candidate trials)")
    else:
        sink.print(f"  Selected best candidate: {action_label}")
        sink.print(
            f"  Best found at: round {best_found_round}, trial {best_found_trial} "
            f"(candidate trial run {best_candidate_trials_to_find})"
        )
        sink.print(f"  Best command: {' '.join(best_command_parts)}")
        if best_env_overrides:
            sink.print(f"  Best environment overrides: {best_env_overrides}")
    sink.print(
        f"  Search to best: {best_found_round} round(s), "
        f"{best_candidate_trials_to_find} candidate trial run(s), "
        f"{best_profiled_runs_to_find} {run_count_label} including baseline/retries"
    )
    sink.print(
        f"  Full tuning search: {total_rounds_executed} round(s), "
        f"{total_candidate_trials_executed} candidate trial run(s), "
        f"{total_profiled_runs} {run_count_label} including baseline/retries"
    )
    if fixed_goal_stop_reason == "max_iterations_exhausted":
        sink.print(
            "  Stopped: max iterations exhausted before fixed goal "
            f"({_format_metric_value(_current_fixed_goal_value())} target, "
            f"threshold {_format_metric_value(fixed_goal)})"
        )
    elif fixed_goal_stop_reason == "fixed_goal_reached":
        sink.print(
            "  Stopped: fixed goal reached "
            f"({_format_metric_value(_current_fixed_goal_value())} target, "
            f"threshold {_format_metric_value(fixed_goal)})"
        )
    elif max_iterations_stop_reason == "max_iterations_exhausted":
        sink.print(
            "  Stopped: max candidate trial limit exhausted after "
            f"{total_candidate_trials_executed}/{max_iterations} candidate attempts"
        )
    sink.print(f"  Total SysPilot wall time: {_elapsed_payload()['total_wall_time_s']:.2f}s")
    sink.print(f"  Config find wall time: {best_config_find_wall_time_s:.2f}s")

    tuned_metrics: dict[str, Any] = dict(best_metrics)
    tuned_objective_value = best_objective_value

    if primary_objective and resolved_primary_objective_metric:
        sink.print("\n[bold cyan]Round final: Reusing Parsed Metrics[/bold cyan]")
        sink.print(f"  Baseline metrics: {baseline_metrics}")
        sink.print(f"  Tuned metrics: {tuned_metrics}")

    sink.print("\n[bold cyan]Round final Step 3: Results Comparison[/bold cyan]")

    speedup = baseline_time / tuned_time if tuned_time > 0 else 0
    wall_time_improvement_pct = (
        ((baseline_time - tuned_time) / baseline_time) * 100 if baseline_time > 0 else 0
    )
    primary_improvement_pct: float | None = None
    primary_speedup_ratio: float | None = None
    primary_speedup_label = ""
    if (
        primary_objective
        and baseline_objective_value is not None
        and tuned_objective_value is not None
    ):
        primary_improvement_pct = _compute_objective_change_pct(
            baseline_objective_value,
            tuned_objective_value,
            cast(str, primary_objective_mode),
        )
        primary_speedup_ratio = _compute_objective_speedup_ratio(
            baseline_objective_value,
            tuned_objective_value,
            cast(str, primary_objective_mode),
        )
        primary_speedup_label = _objective_speedup_label(primary_speedup_ratio)
    reported_improvement_pct = (
        primary_improvement_pct
        if primary_improvement_pct is not None
        else wall_time_improvement_pct
    )
    _emit_tune_event(
        event_emitter,
        "result.comparison",
        {
            "mode": "spawn",
            "baseline_time_s": baseline_time,
            "tuned_time_s": tuned_time,
            "improvement_pct": reported_improvement_pct,
            "speedup": speedup,
            "action_label": action_label,
            "primary_objective_query": primary_objective or "",
            "primary_objective_metric": resolved_primary_objective_metric or "",
            "primary_objective_mode": primary_objective_mode or "",
            "baseline_primary_objective": baseline_objective_value,
            "tuned_primary_objective": tuned_objective_value,
            "wall_time_improvement_pct": wall_time_improvement_pct,
            "primary_objective_improvement_pct": primary_improvement_pct,
            "primary_objective_speedup_ratio": primary_speedup_ratio,
            "primary_objective_speedup_label": primary_speedup_label,
            "best_found_round": best_found_round,
            "best_found_trial": best_found_trial,
            "best_candidate_trials_to_find": best_candidate_trials_to_find,
            "best_profiled_runs_to_find": best_profiled_runs_to_find,
            "total_rounds_executed": total_rounds_executed,
            "total_candidate_trials_executed": total_candidate_trials_executed,
            "total_profiled_runs": total_profiled_runs,
            "stop_reason": _final_stop_reason(),
            "max_iterations_stop_reason": max_iterations_stop_reason,
            **_elapsed_payload(),
            **fixed_goal_payload(
                threshold=fixed_goal,
                value=_current_fixed_goal_value(),
                objective_mode=_current_fixed_goal_mode(),
                stop_reason=fixed_goal_stop_reason,
            ),
            **_fixed_budget_event_payload(
                fixed_budget_state,
                stop_reason=fixed_budget_stop_reason,
            ),
        },
    )

    wall_time_change = f"{wall_time_improvement_pct:+.1f}%"
    if wall_time_improvement_pct > 0 and speedup > 0:
        wall_time_change = f"{speedup:.2f}x speed up ({wall_time_improvement_pct:+.1f}%)"
    objective_speedup_display = _format_objective_speedup(
        primary_speedup_ratio,
        unavailable="ratio unavailable",
    )

    table_rows: list[list[str]] = []

    if (
        primary_objective
        and baseline_objective_value is not None
        and tuned_objective_value is not None
    ):
        table_rows.append(
            [
                _objective_display_name(primary_objective),
                _format_metric_value(baseline_objective_value),
                _format_metric_value(tuned_objective_value),
                _format_objective_speedup(primary_speedup_ratio),
            ]
        )

    table_rows.append(
        [
            "Wall Time",
            f"{baseline_time:.2f}s",
            f"{tuned_time:.2f}s",
            wall_time_change,
        ]
    )

    if primary_objective and baseline_metrics and tuned_metrics:
        if not baseline_metrics.get("no_metrics") and not tuned_metrics.get("no_metrics"):
            all_keys = set(baseline_metrics.keys()) | set(tuned_metrics.keys())
            for key in sorted(all_keys):
                if key in ("no_metrics", "error"):
                    continue
                if resolved_primary_objective_metric and key == resolved_primary_objective_metric:
                    continue
                base_val = baseline_metrics.get(key)
                tuned_val = tuned_metrics.get(key)

                if (
                    isinstance(base_val, (int, float))
                    and isinstance(tuned_val, (int, float))
                    and base_val != 0
                ):
                    metric_mode = _infer_primary_objective_mode(key)
                    if metric_mode is None:
                        change_pct = ((tuned_val - base_val) / abs(base_val)) * 100
                    else:
                        change_pct = _compute_objective_change_pct(
                            float(base_val),
                            float(tuned_val),
                            metric_mode,
                        )
                    change_str = f"{change_pct:+.1f}%"
                else:
                    change_str = ""

                base_str = f"{base_val}" if base_val is not None else "-"
                tuned_str = f"{tuned_val}" if tuned_val is not None else "-"

                display_key = key.replace("_", " ").title()
                table_rows.append([display_key, base_str, tuned_str, change_str])

    table_rows.append(["Applied actions", "(none)", action_label, ""])

    sink.table(
        title="Performance Comparison",
        columns=[
            ("Metric", "cyan"),
            ("Baseline", "white"),
            ("Tuned", "green"),
            ("Change", "yellow"),
        ],
        rows=table_rows,
    )

    if reported_improvement_pct > 5:
        if primary_improvement_pct is not None:
            sink.print(
                "\n[bold green]✓ Success! "
                f"{_objective_display_name(primary_objective)} improved to "
                f"{objective_speedup_display} "
                f"with actions: {action_label}[/bold green]"
            )
        else:
            sink.print(
                f"\n[bold green]✓ Success! {speedup:.2f}x speed up "
                f"({reported_improvement_pct:.1f}% improvement) with actions: "
                f"{action_label}[/bold green]"
            )
        run_status = "improved"
    elif reported_improvement_pct > 0:
        if primary_improvement_pct is not None:
            sink.print(
                "\n[yellow]Modest primary-objective improvement "
                f"({objective_speedup_display})[/yellow]"
            )
        else:
            sink.print(
                f"\n[yellow]Modest improvement ({speedup:.2f}x speed up, "
                f"{reported_improvement_pct:.1f}%)[/yellow]"
            )
        run_status = "modest_improvement"
    else:
        if primary_improvement_pct is not None:
            sink.print(
                "\n[red]No improvement or regression "
                f"({objective_speedup_display})[/red]"
            )
        else:
            sink.print(
                f"\n[red]No improvement or regression ({reported_improvement_pct:.1f}%)[/red]"
            )
        run_status = "no_improvement"

    _emit_tune_event(
        event_emitter,
        "run.finished",
        {
            "mode": "spawn",
            "status": run_status,
            "baseline_time_s": baseline_time,
            "tuned_time_s": tuned_time,
            "improvement_pct": reported_improvement_pct,
            "action_label": action_label,
            "primary_objective_query": primary_objective or "",
            "primary_objective_metric": resolved_primary_objective_metric or "",
            "primary_objective_mode": primary_objective_mode or "",
            "primary_objective_speedup_ratio": primary_speedup_ratio,
            "primary_objective_speedup_label": primary_speedup_label,
            "best_found_round": best_found_round,
            "best_found_trial": best_found_trial,
            "best_candidate_trials_to_find": best_candidate_trials_to_find,
            "best_profiled_runs_to_find": best_profiled_runs_to_find,
            "total_rounds_executed": total_rounds_executed,
            "total_candidate_trials_executed": total_candidate_trials_executed,
            "total_profiled_runs": total_profiled_runs,
            "stop_reason": _final_stop_reason(),
            "max_iterations_stop_reason": max_iterations_stop_reason,
            **_elapsed_payload(),
            **fixed_goal_payload(
                threshold=fixed_goal,
                value=_current_fixed_goal_value(),
                objective_mode=_current_fixed_goal_mode(),
                stop_reason=fixed_goal_stop_reason,
            ),
            **_fixed_budget_event_payload(
                fixed_budget_state,
                stop_reason=fixed_budget_stop_reason,
            ),
        },
    )
    if study_trace is not None:
        study_trace.finish(
            metadata={
                "run_status": run_status,
                "current_wall_time_s": current_time,
                "current_objective_value": current_objective_value,
                "incumbent_wall_time_s": best_time,
                "incumbent_objective_value": best_objective_value,
                "total_candidate_trials_executed": total_candidate_trials_executed,
                "ordinary_stop_reason": _final_stop_reason(),
            }
        )


def _run_spawn_tune_loop(
    agent: SysPilotAgent,
    command: str,
    contracts: Sequence[str],
    manual_hints: Sequence[str],
    fixed_budget: int | None,
    fixed_goal: float | None,
    max_iterations: int,
    timeout: int,
    primary_objective: str | None = None,
    primary_objective_mode: str | None = None,
    workdir: Path | None = None,
    memory_high: str | None = None,
    profiling_enabled: bool = True,
    sink: OutputSink | None = None,
    event_emitter: _TuneEmitter | None = None,
    structured_context_bundle: dict[str, object] | None = None,
    initial_hypothesis_profiles: list[dict[str, Any]] | None = None,
    excluded_config_names: Sequence[str] | None = None,
    tunable_whitelist_names: Sequence[str] | None = None,
    study_trace: NestedCpuBlasStudyTrace | None = None,
) -> None:
    workdir_str = str(workdir) if workdir else None
    try:
        asyncio.run(
            _run_spawn_tune_loop_async(
                agent=agent,
                command=command,
                contracts=contracts,
                manual_hints=manual_hints,
                fixed_budget=fixed_budget,
                fixed_goal=fixed_goal,
                max_iterations=max_iterations,
                timeout=timeout,
                primary_objective=primary_objective,
                primary_objective_mode=primary_objective_mode,
                workdir=workdir_str,
                memory_high=memory_high,
                profiling_enabled=profiling_enabled,
                sink=sink,
                event_emitter=event_emitter,
                structured_context_bundle=structured_context_bundle,
                initial_hypothesis_profiles=initial_hypothesis_profiles,
                excluded_config_names=excluded_config_names,
                tunable_whitelist_names=tunable_whitelist_names,
                study_trace=study_trace,
            )
        )
    finally:
        if study_trace is not None:
            study_trace.finish()


def _run_quick_tune_loop(
    *,
    agent: SysPilotAgent,
    command: str,
    contracts: Sequence[str],
    manual_hints: Sequence[str],
    fixed_budget: int | None,
    fixed_goal: float | None,
    max_iterations: int,
    timeout: int,
    primary_objective: str | None,
    primary_objective_mode: str | None,
    workdir: Path | None,
    memory_high: str | None,
    sink: OutputSink | None,
    event_emitter: _TuneEmitter | None,
    structured_context_bundle: dict[str, object] | None,
    initial_hypothesis_profiles: list[dict[str, Any]] | None,
    defaults: QuickModeDefaults | None,
    excluded_config_names: Sequence[str] | None = None,
    tunable_whitelist_names: Sequence[str] | None = None,
) -> None:
    from syspilot.quick_mode import run_quick_tune_loop

    run_quick_tune_loop(
        agent=agent,
        command=command,
        contracts=contracts,
        manual_hints=manual_hints,
        fixed_budget=fixed_budget,
        fixed_goal=fixed_goal,
        max_iterations=max_iterations,
        timeout=timeout,
        primary_objective=primary_objective,
        primary_objective_mode=primary_objective_mode,
        workdir=workdir,
        memory_high=memory_high,
        sink=sink,
        event_emitter=event_emitter,
        structured_context_bundle=structured_context_bundle,
        initial_hypothesis_profiles=initial_hypothesis_profiles,
        run_executor=_run_live_profiling_pipeline,
        parse_spawn_actions=_parse_spawn_actions,
        build_spawn_candidates=_build_spawn_candidates,
        extract_hypothesis_profiles=_extract_hypothesis_profiles,
        recommendation_next_action_display=_recommendation_next_action_display,
        extract_inline_env_assignments=_extract_inline_env_assignments,
        extract_metrics_from_output=_extract_metrics_from_output,
        resolve_primary_objective_metric=_resolve_primary_objective_metric,
        extract_primary_objective_value=_extract_primary_objective_value,
        infer_primary_objective_mode=_infer_primary_objective_mode,
        compare_objective_values=_compare_objective_values,
        compute_objective_change_pct=_compute_objective_change_pct,
        objective_display_name=_objective_display_name,
        format_metric_value=_format_metric_value,
        defaults=defaults,
        excluded_config_names=excluded_config_names,
        tunable_whitelist_names=tunable_whitelist_names,
    )


def _wrap_spawn_command_with_memory_high(
    command_parts: list[str],
    memory_high: str | None,
    workdir: str | None = None,
) -> list[str]:
    if memory_high is None:
        return list(command_parts)
    normalized_memory_high = memory_high.strip()
    if not normalized_memory_high:
        return list(command_parts)

    if workdir:
        shell_command = f"cd {shlex.quote(workdir)} && exec {shlex.join(command_parts)}"
        return [
            "systemd-run",
            "--user",
            "--scope",
            "-p",
            f"MemoryHigh={normalized_memory_high}",
            "/bin/sh",
            "-lc",
            shell_command,
        ]

    return [
        "systemd-run",
        "--user",
        "--scope",
        "-p",
        f"MemoryHigh={normalized_memory_high}",
        *command_parts,
    ]


@cli.command()
@click.option("--output", "-o", type=click.Path(), default="syspilot.yaml", help="Output path")
def init(output: str) -> None:
    config = SysPilotConfig()
    config.save_yaml(Path(output))
    console.print(f"[green]Created config file: {output}[/green]")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
