from __future__ import annotations

import asyncio
import difflib
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
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
from rich.panel import Panel

from syspilot import __version__
from syspilot.core.config import SysPilotConfig, LLMConfig
from syspilot.core.agent import SysPilotAgent
from syspilot.core.llm import LLMError
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
from syspilot.profiling import ProfilingManager
from syspilot.skills import SkillRegistry
from syspilot.ui.events import is_critical_event
from syspilot.toolbox_runtime import (
    extract_declared_virtual_probes,
    finish_toolbox_probe_sessions,
    start_toolbox_probe_sessions,
)
from syspilot.skills.builtin import (
    DynamicScriptSkill,
    StraceAnalyzerSkill,
    PerfAnalyzerSkill,
    DataProcessorSkill,
)
from syspilot.skills.ebpf import EBPFSkill
from syspilot.tuning import ApplicationTuner
from syspilot.utils import setup_logging
from syspilot.ui.emitter import TuneEventEmitter
from syspilot.ui.output_sink import OutputSink, SwitchableOutputSink, TuneUIMode, make_output_sink
from syspilot.ui.tui_sidecar import TuneTuiOrchestrator, should_wait_for_user_on_close

console = Console()
logger = logging.getLogger(__name__)


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
        "api_key": repo_llm.api_key,
        "base_url": repo_llm.base_url,
        "openai_auth_file": repo_llm.openai_auth_file,
        "temperature": repo_llm.temperature,
        "max_tokens": repo_llm.max_tokens,
        "timeout_seconds": repo_llm.timeout_seconds,
        "max_retries": repo_llm.max_retries,
        "debug_conversation": repo_llm.debug_conversation,
        "debug_markdown_file": repo_llm.debug_markdown_file,
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
        EBPFSkill,
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
            "(sysctl|prlimit|command|env_var|numactl), parameter, current_value, new_value, reason."
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
                                "enum": ["sysctl", "prlimit", "command", "env_var", "numactl"],
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
    if debug_llm or debug_llm_md:
        ctx.obj["config"].llm.debug_conversation = True
    if debug_llm_md:
        ctx.obj["config"].llm.debug_markdown_file = debug_llm_md
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
            "[yellow]No manuals directory found (checked source tree and ~/.syspilot/manual)[/yellow]"
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
    console.print("\n[dim]Use --manual/-m for spawn-mode application hints or agent/manual tooling[/dim]")
    console.print(
        "[dim]Example: syspilot tune --command \"./benchmark\" --contract numactl --manual \"worker pool expands after warmup\"[/dim]"
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
    return _resolve_contract_root(contract_root) / contract_ref / "contract.yaml"


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
        return bundle if _has_structured_context(bundle) else {}
    except KeyError:
        return {}
    except Exception as exc:
        logger.debug("Structured contract context unavailable: %s", exc)
        return {}


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
        EBPFSkill,
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
    help="If set in spawn mode, launch the command via systemd-run --user --scope -p MemoryHigh=...",
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
        "Application-specific natural-language tune hint(s). Repeat the option or use comma-separated "
        "values (escape literal commas as \\,)."
    ),
)
@click.option("--max-iterations", default=5, help="Maximum tuning iterations")
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
@click.pass_context
def tune(
    ctx: click.Context,
    command: str | None,
    ui_mode: str,
    tui_bin: Path | None,
    workdir: Path | None,
    memory_high: str | None,
    contract: tuple[str, ...],
    manual: tuple[str, ...],
    max_iterations: int,
    dry_run: bool,
    timeout: int,
    primary_objective: str | None,
    primary_objective_mode: str | None,
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

    resolved_primary_objective: str | None = None
    resolved_primary_objective_mode: str | None = None
    contract_names = _normalize_multi_value_option(contract)
    manual_hints = _normalize_multi_value_option(manual)
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
    config.tuning.dry_run = dry_run
    config.tuning.max_tuning_iterations = max_iterations

    agent = SysPilotAgent(config)
    prompt_log_path = agent.enable_llm_prompt_logging(_default_tune_prompt_log_path())
    sink.print(f"[cyan]Saving LLM prompts to: {prompt_log_path}[/cyan]")

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

    if contract_names:
        if _has_structured_context(structured_context_bundle):
            sink.print(f"[cyan]Using structured contract context: {', '.join(contract_names)}[/cyan]")
        else:
            sink.print(
                f"[yellow]No structured contract context found for: {', '.join(contract_names)}[/yellow]"
            )

    if manual_hints:
        sink.print(f"[cyan]Using application manual hint(s): {', '.join(manual_hints)}[/cyan]")

    if initial_hypothesis_profiles:
        sink.print(
            f"[cyan]Generated {len(initial_hypothesis_profiles)} contract hypothesis profile(s) at startup[/cyan]"
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
        _run_spawn_tune_loop(
            agent,
            command,
            contract_names,
            manual_hints,
            max_iterations,
            timeout,
            resolved_primary_objective,
            resolved_primary_objective_mode,
            workdir,
            normalized_memory_high,
            sink,
            event_emitter,
            structured_context_bundle if _has_structured_context(structured_context_bundle) else None,
            initial_hypothesis_profiles,
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
            f"limit={_format_bytes_human(cast(int | None, cgroup_memory.get('effective_limit_bytes')))} "
            f"current={_format_bytes_human(cast(int | None, cgroup_memory.get('current_bytes')))} "
            f"peak={_format_bytes_human(cast(int | None, cgroup_memory.get('peak_bytes')))} "
            f"headroom={_format_bytes_human(cast(int | None, cgroup_memory.get('min_headroom_bytes')))} "
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
                f"[extract-evidence] {progress.library_name}: starting {progress.total_chunks} chunks"
            )
        return
    if progress.status == "completed":
        click.echo(
            "[extract-evidence] "
            f"{progress.library_name}: completed {progress.processed_chunks}/{progress.total_chunks} "
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
        f"evidence={progress.evidence_items} elapsed={_format_progress_duration(progress.elapsed_s)} "
        f"eta={_format_progress_duration(progress.eta_s)} finish~{_format_progress_finish_time(progress.eta_s)}"
    )


def _manual_composition_progress_callback(progress: object) -> None:
    from syspilot_manual.integration import ManualCompositionProgress

    if not isinstance(progress, ManualCompositionProgress):
        return
    click.echo(
        "[compose-manuals] "
        f"{progress.app_name}/{progress.manual_id} phase={progress.status} "
        f"libraries={progress.library_count} elapsed={_format_progress_duration(progress.elapsed_s)} "
        f"eta={_format_progress_duration(progress.eta_s)} finish~{_format_progress_finish_time(progress.eta_s)}"
    )


def _contract_synthesis_progress_callback(progress: object) -> None:
    from syspilot_contract.integration import ContractSynthesisProgress

    if not isinstance(progress, ContractSynthesisProgress):
        return
    click.echo(
        "[synthesize-contract] "
        f"{progress.library_name}/{progress.contract_id} phase={progress.status} "
        f"evidence={progress.evidence_items} elapsed={_format_progress_duration(progress.elapsed_s)} "
        f"eta={_format_progress_duration(progress.eta_s)} finish~{_format_progress_finish_time(progress.eta_s)}"
    )


def _contract_composition_progress_callback(progress: object) -> None:
    from syspilot_contract.integration import ContractCompositionProgress

    if not isinstance(progress, ContractCompositionProgress):
        return
    click.echo(
        "[compose-contracts] "
        f"{progress.app_name}/{progress.contract_id} phase={progress.status} "
        f"libraries={progress.library_count} elapsed={_format_progress_duration(progress.elapsed_s)} "
        f"eta={_format_progress_duration(progress.eta_s)} finish~{_format_progress_finish_time(progress.eta_s)}"
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
    terms = [virtual_probe, str(signal.get("metric", "")).strip(), str(signal.get("indicates", "")).strip()]
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
        "runtime_output = profiling.get('runtime_output', {}) if isinstance(profiling, dict) else {}\n"
        "refined_runtime_maps = profiling.get('refined_runtime_maps', {}) if isinstance(profiling, dict) else {}\n"
        "collect_profile_data = profiling.get('collect_profile_data', {}) if isinstance(profiling, dict) else {}\n"
        "probe_results = profiling.get('virtual_probe_results', {}) if isinstance(profiling, dict) else {}\n"
        "cpu_collector = collect_profile_data.get('cpu', {}) if isinstance(collect_profile_data, dict) else {}\n"
        "cpu_data = cpu_collector.get('data', {}) if isinstance(cpu_collector, dict) else {}\n"
        "memory_collector = collect_profile_data.get('memory', {}) if isinstance(collect_profile_data, dict) else {}\n"
        "memory_data = memory_collector.get('data', {}) if isinstance(memory_collector, dict) else {}\n"
        "combined_output = str(payload.get('combined_output', ''))\n"
        "report_env = payload.get('report_env', {}) if isinstance(payload.get('report_env', {}), dict) else {}\n"
        "active_env = payload.get('active_env', {}) if isinstance(payload.get('active_env', {}), dict) else {}\n"
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
        "    resolved_probe = signal_entry.get('resolved_probe', {}) if isinstance(signal_entry, dict) else {}\n"
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
        "    probe_result = probe_results.get(virtual_probe, {}) if isinstance(probe_results, dict) else {}\n"
        "    if isinstance(probe_result, dict) and probe_result:\n"
        "        result['observed'] = bool(probe_result.get('observed', False))\n"
        "        result['source'] = str(probe_result.get('source', 'toolbox_script'))\n"
        "        result['value'] = probe_result.get('value')\n"
        "        excerpt = probe_result.get('excerpt', [])\n"
        "        result['excerpt'] = excerpt[:8] if isinstance(excerpt, list) else []\n"
        "        if result['observed'] or result['source'] in ('unavailable', 'probe_error', 'os_missing', 'runtime_output_missing'):\n"
        "            return result\n"
        "    terms = [virtual_probe.lower(), str(signal.get('metric', '')).lower(), str(signal.get('indicates', '')).lower()]\n"
        "    searchable_terms = [term for term in terms if term]\n"
        "    if isinstance(resolved_probe, dict) and resolved_probe.get('availability') == 'unavailable':\n"
        "        result['source'] = 'unavailable'\n"
        "        return result\n"
        "    matches = [line.strip() for line in lines if any(term in line.lower() for term in searchable_terms)]\n"
        "    result['observed'] = bool(matches)\n"
        "    result['source'] = 'runtime_output' if matches else 'unsupported'\n"
        "    result['value'] = matches[:8] if matches else None\n"
        "    result['excerpt'] = matches[:8]\n"
        "    return result\n"
        "profiles = []\n"
        "for plan in plans:\n"
        "    signals = plan.get('validation_signals', []) if isinstance(plan, dict) else []\n"
        "    signal_results = [_signal_result(signal) for signal in signals if isinstance(signal, dict)]\n"
        "    findings = [f\"{item['ref']} observed via {item['source']}\" for item in signal_results if item.get('observed')]\n"
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
            if config_type not in {"command", "command_prefix", "numactl"}:
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
            if not new_tokens or (config_type == "numactl" and new_tokens == ["numactl"]):
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

    executable_name = Path(base_command[0]).name
    if executable_name == "llama-bench":
        issues.extend(_validate_llama_bench_fairness(base_command, candidate_command))

    return issues


def _parse_spawn_actions(response: str) -> dict[str, Any]:
    parsed = _extract_json_object(response)
    if not parsed:
        return {
            "valid": False,
            "error": "Agent response did not contain a valid JSON object",
            "summary": "",
            "manual_citation": "",
            "evidence": [],
            "actions": [],
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

    actions_raw = parsed.get("actions", [])
    if not isinstance(actions_raw, list):
        return {
            "valid": False,
            "error": "Field 'actions' must be a list",
            "summary": summary,
            "manual_citation": manual_citation,
            "evidence": evidence,
            "hypothesis_refs": hypothesis_refs,
            "actions": [],
            "skipped_actions": [],
        }

    supported_types = {"numactl", "env_var", "command_prefix", "command"}
    normalized_actions: list[dict[str, Any]] = []
    skipped_actions: list[str] = []

    for idx, action in enumerate(actions_raw):
        if not isinstance(action, dict):
            skipped_actions.append(f"action[{idx}] is not an object")
            continue

        config_type = str(action.get("config_type", "")).strip().lower()
        if config_type not in supported_types:
            skipped_actions.append(f"action[{idx}] has unsupported config_type '{config_type}'")
            continue

        parameter = str(action.get("parameter", "")).strip()
        reason = str(action.get("reason", "")).strip()
        action_manual_citation = str(action.get("manual_citation", manual_citation)).strip()

        if "new_value" not in action:
            skipped_actions.append(f"action[{idx}] is missing new_value")
            continue

        if config_type == "env_var" and (not parameter or "=" in parameter or " " in parameter):
            skipped_actions.append(f"action[{idx}] has invalid env var name '{parameter}'")
            continue

        normalized_actions.append(
            {
                "config_type": config_type,
                "parameter": parameter,
                "new_value": action.get("new_value"),
                "reason": reason,
                "manual_citation": action_manual_citation,
            }
        )

    if normalized_actions and (not manual_citation or not evidence):
        return {
            "valid": False,
            "error": "Actionable recommendations require non-empty manual_citation and evidence",
            "summary": summary,
            "manual_citation": manual_citation,
            "evidence": evidence,
            "hypothesis_refs": hypothesis_refs,
            "actions": [],
            "skipped_actions": skipped_actions,
        }

    return {
        "valid": True,
        "error": "",
        "summary": summary,
        "manual_citation": manual_citation,
        "evidence": evidence,
        "hypothesis_refs": hypothesis_refs,
        "actions": normalized_actions,
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
    if config_type == "numactl" and tokens and tokens[0] == "numactl":
        tokens = tokens[1:]
    token_display = " ".join(tokens)
    fallback = "" if new_value is None else str(new_value).strip()
    value_display = token_display or fallback

    if config_type == "command":
        return value_display
    if config_type == "numactl":
        return f"numactl {value_display}".strip()
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


def _build_spawn_execution_plan(
    base_command: list[str],
    actions: list[dict[str, Any]],
    base_env_overrides: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str], list[str], list[str]]:
    if not base_command:
        return [], {}, [], ["base command was empty"]

    prefix_tokens: list[str] = []
    command_tokens = list(base_command)
    command_replaced = False
    env_overrides: dict[str, str] = dict(base_env_overrides or {})
    applied: list[str] = []
    skipped: list[str] = []

    for idx, action in enumerate(actions):
        config_type = str(action.get("config_type", "")).strip().lower()
        parameter = str(action.get("parameter", "")).strip()
        new_value = action.get("new_value")

        if config_type == "env_var":
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

        if config_type == "command":
            if command_replaced:
                skipped.append(f"action[{idx}] additional command rewrite skipped")
                continue

            normalized_tokens, inline_env_overrides = _extract_inline_env_assignments(tokens)
            if not normalized_tokens:
                skipped.append(f"action[{idx}] command rewrite produced no executable tokens")
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

            command_tokens = normalized_tokens
            env_overrides.update(inline_env_overrides)
            command_replaced = True
            applied.append(f"command {' '.join(normalized_tokens)}")
            continue

        if config_type == "numactl":
            if tokens[0] == "numactl":
                tokens = tokens[1:]
            if not tokens:
                skipped.append(f"action[{idx}] numactl flags were empty")
                continue
            prefix_tokens.extend(["numactl", *tokens])
            applied.append(f"numactl {' '.join(tokens)}")
            continue

        if config_type == "command_prefix":
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
) -> list[dict[str, Any]]:
    if max_candidates < 1:
        return []

    alternative_action_types = {"command", "command_prefix", "numactl"}
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
    prompt = f"""Extract ONLY runtime performance outcome metrics from this program output. Return JSON only.

{f"Context: {context}" if context else ""}

Program output:
```
{output[:4000]}
```

Include only metrics that represent measurable performance outcomes, such as:
- throughput/rate (tokens_per_sec, ops_per_sec, req_per_sec, qps, rps, fps)
- latency (ms/us/ns), including p50/p90/p95/p99
- execution time / duration / runtime
- bandwidth

Exclude configuration or static metadata, including:
- model size/params
- thread counts, batch sizes, token counts
- topology/hardware metadata
- any key like n_threads, n_batch, n_prompt_tokens, model_size_bytes, model_params

Return one flat JSON object: {{"metric_name": number, ...}}
Do not return nested JSON, text, or explanations.

If no metrics found, return: {{"no_metrics": true}}

RESPOND WITH ONLY THE JSON OBJECT, NO EXPLANATION:"""

    try:
        response = await agent.step(prompt)
    except LLMError as exc:
        logger.warning("Output metric extraction skipped after LLM failure: %s", exc)
        return {"no_metrics": True, "error": str(exc)}

    import re

    json_match = re.search(r"\{[^{}]*\}", response.content, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, dict):
                return _normalize_metric_mapping(parsed)
        except json.JSONDecodeError:
            pass
    return {"no_metrics": True}


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

    memory_events = evidence.get("memory_events", {}) if isinstance(evidence.get("memory_events"), dict) else {}
    memory_pressure = evidence.get("memory_pressure", {}) if isinstance(evidence.get("memory_pressure"), dict) else {}
    memory_stat = evidence.get("memory_stat", {}) if isinstance(evidence.get("memory_stat"), dict) else {}
    pressure_some = memory_pressure.get("some", {}) if isinstance(memory_pressure.get("some"), dict) else {}
    pressure_full = memory_pressure.get("full", {}) if isinstance(memory_pressure.get("full"), dict) else {}

    return (
        "## Effective Memory Capacity\n"
        f"Requested MemoryHigh: {evidence.get('requested_memory_high') or '(none)'}\n"
        f"Control group: {evidence.get('control_group') or '(unknown)'}\n"
        f"Effective limit: {_format_bytes_human(cast(int | None, evidence.get('effective_limit_bytes')))}\n"
        f"Current usage: {_format_bytes_human(cast(int | None, evidence.get('current_bytes')))}\n"
        f"Peak usage: {_format_bytes_human(cast(int | None, evidence.get('peak_bytes')))}\n"
        f"Minimum headroom: {_format_bytes_human(cast(int | None, evidence.get('min_headroom_bytes')))}\n"
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
        f"Swap current: {_format_bytes_human(cast(int | None, evidence.get('swap_current_bytes')))}\n"
        f"Status: {evidence.get('status', 'unknown')}\n"
        "Rule: When this block is present, reason about memory against this cgroup-constrained envelope rather than host total RAM or generic process memory percentages."
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
) -> dict[str, Any]:
    started_at = time.monotonic()
    selected_hypothesis_profiles = hypothesis_profiles or []
    report_env_overrides = _collect_contract_report_env(selected_hypothesis_profiles)
    env = _build_spawn_profiling_env(env_overrides, report_env_overrides)
    launch_command = _wrap_spawn_command_with_memory_high(command_parts, memory_high, workdir)
    launch_cwd = None if memory_high else workdir

    proc = await asyncio.create_subprocess_exec(
        *launch_command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=launch_cwd,
        env=env,
    )

    async def _drain_stream(stream_name: str, stream: asyncio.StreamReader | None) -> str:
        if stream is None:
            return ""
        chunks: list[str] = []
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace")
            chunks.append(text)
            if live_output_callback is not None:
                live_output_callback(stream_name, text)
        return "".join(chunks)

    stdout_task = asyncio.create_task(_drain_stream("stdout", proc.stdout))
    stderr_task = asyncio.create_task(_drain_stream("stderr", proc.stderr))

    pid = proc.pid
    prev_target_pid = getattr(agent.context, "target_pid", None)
    agent.context.target_pid = pid

    await asyncio.sleep(1.0)

    process_targets = _discover_related_processes(pid, max_targets=4)
    target_pids = process_targets["target_pids"]
    primary_pid = process_targets["primary_pid"]
    requested_virtual_probes = extract_declared_virtual_probes(
        selected_hypothesis_profiles,
        memory_high=memory_high,
    )
    probe_sessions = await start_toolbox_probe_sessions(
        requested_virtual_probes,
        {
            "target_pids": target_pids,
            "primary_pid": primary_pid,
            "memory_high": memory_high,
        },
    )

    tools_used: list[str] = []
    key_findings: list[str] = []
    per_pid_runtime_maps: dict[str, dict[str, Any]] = {}
    per_pid_collect_profile: dict[str, dict[str, Any]] = {}
    collect_summaries: list[str] = []
    agent_tool_calls = 0

    for target_pid in target_pids:
        try:
            collect_result = await agent._execute_tool(
                "collect_profile",
                {
                    "collectors": ["cpu", "memory", "process", "numa"],
                    "pid": target_pid,
                },
            )
            agent_tool_calls += 1
            tools_used.append(f"collect_profile:{target_pid}")

            if isinstance(collect_result, dict):
                per_pid_collect_profile[str(target_pid)] = collect_result
                summaries: list[str] = []
                for collector_name, payload in collect_result.items():
                    if isinstance(payload, dict) and "summary" in payload:
                        summaries.append(f"{collector_name}: {payload['summary']}")
                if summaries:
                    collect_summaries.append(f"pid {target_pid}: " + " | ".join(summaries))
        except Exception as e:
            key_findings.append(f"collect_profile failed for pid {target_pid}: {e}")

    ebpf_summary = ""
    ebpf_data: dict[str, Any] = {}
    try:
        ebpf_check = await agent._execute_tool("skill_ebpf", {"action": "check_availability"})
        agent_tool_calls += 1
        tools_used.append("skill_ebpf.check_availability")
        if isinstance(ebpf_check, dict):
            ebpf_summary = str(ebpf_check.get("summary", ""))
            raw_data = ebpf_check.get("data", {})
            if isinstance(raw_data, dict):
                ebpf_data = raw_data

        template_name = ""
        if ebpf_data.get("bpftrace_available"):
            template_name = "syscall_count"
        elif ebpf_data.get("bcc_available"):
            template_name = "bcc_syscall_latency"

        if template_name:
            ebpf_profile = await agent._execute_tool(
                "skill_ebpf",
                {
                    "action": "run_template",
                    "template_name": template_name,
                    "pid": primary_pid,
                    "duration": 2,
                },
            )
            agent_tool_calls += 1
            tools_used.append(f"skill_ebpf.run_template:{template_name}:{primary_pid}")
            if isinstance(ebpf_profile, dict):
                profile_summary = str(ebpf_profile.get("summary", "")).strip()
                profile_error = str(ebpf_profile.get("error", "")).strip()
                if profile_summary:
                    ebpf_summary = profile_summary
                if profile_error:
                    ebpf_summary = (
                        f"{ebpf_summary} ({profile_error})" if ebpf_summary else profile_error
                    )

                profile_success = bool(ebpf_profile.get("success", False))
                if (
                    not profile_success
                    and template_name == "syscall_count"
                    and ebpf_data.get("bcc_available")
                ):
                    fallback_profile = await agent._execute_tool(
                        "skill_ebpf",
                        {
                            "action": "run_template",
                            "template_name": "bcc_syscall_latency",
                            "pid": primary_pid,
                            "duration": 2,
                        },
                    )
                    agent_tool_calls += 1
                    tools_used.append(f"skill_ebpf.run_template:bcc_syscall_latency:{primary_pid}")
                    if isinstance(fallback_profile, dict):
                        fallback_summary = str(fallback_profile.get("summary", "")).strip()
                        fallback_error = str(fallback_profile.get("error", "")).strip()
                        fallback_text = fallback_summary
                        if fallback_error:
                            fallback_text = (
                                f"{fallback_text} ({fallback_error})"
                                if fallback_text
                                else fallback_error
                            )
                        if fallback_text:
                            ebpf_summary = f"{ebpf_summary} | fallback: {fallback_text}"
    except Exception as e:
        key_findings.append(f"ebpf profiling failed: {e}")

    processor_script = (
        "import json\n"
        "data = {}\n"
        "line_count = 0\n"
        "for line in raw_data.splitlines():\n"
        "    line_count += 1\n"
        "    if ':' not in line:\n"
        "        continue\n"
        "    key, value = line.split(':', 1)\n"
        "    key = key.strip()\n"
        "    value = value.strip()\n"
        "    if key in {'Name', 'State', 'Threads', 'Cpus_allowed_list', 'Mems_allowed_list', 'VmRSS', 'voluntary_ctxt_switches', 'nonvoluntary_ctxt_switches'}:\n"
        "        data[key] = value\n"
        "data['line_count'] = line_count\n"
        "print(json.dumps(data))\n"
    )

    for target_pid in target_pids:
        try:
            capture_result = await agent._execute_tool(
                "skill_data_processor",
                {
                    "action": "capture",
                    "command": f"cat /proc/{target_pid}/status",
                },
            )
            agent_tool_calls += 1
            tools_used.append(f"skill_data_processor.capture:{target_pid}")

            capture_id = None
            if isinstance(capture_result, dict):
                capture_data = capture_result.get("data", {})
                if isinstance(capture_data, dict):
                    capture_id = capture_data.get("capture_id")

            if capture_id:
                process_result = await agent._execute_tool(
                    "skill_data_processor",
                    {
                        "action": "process",
                        "capture_id": capture_id,
                        "script": processor_script,
                    },
                )
                agent_tool_calls += 1
                tools_used.append(f"skill_data_processor.process:{target_pid}")

                if isinstance(process_result, dict):
                    output = process_result.get("output", "")
                    try:
                        refined = json.loads(output.strip()) if output else {}
                        if isinstance(refined, dict):
                            per_pid_runtime_maps[str(target_pid)] = refined
                    except json.JSONDecodeError:
                        key_findings.append(
                            f"status parsing returned invalid JSON for pid {target_pid}"
                        )
        except Exception as e:
            key_findings.append(f"dynamic status pipeline failed for pid {target_pid}: {e}")

    timed_out = False
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
        rc = proc.returncode if proc.returncode is not None else 0
    except asyncio.TimeoutError:
        proc.kill()
        try:
            await proc.wait()
        except ProcessLookupError:
            pass
        rc = -1
        timed_out = True

    stdout = await stdout_task
    stderr = await stderr_task
    if timed_out:
        timeout_message = "Command timed out during profiling pipeline"
        stderr = f"{stderr}\n{timeout_message}".strip() if stderr else timeout_message

    wall = float(timeout) if timed_out else (time.monotonic() - started_at)
    runtime_output = _summarize_spawn_runtime_output(stdout, stderr)
    workers = process_targets.get("workers", [])
    if workers:
        heavy_workers = ", ".join(
            f"pid {worker.get('pid')} cpu={worker.get('cpu_percent')}% rss={worker.get('rss_mb')}MB"
            for worker in workers[:3]
        )
        key_findings.append(f"worker-heavy profiling targets: {heavy_workers}")

    if not key_findings:
        key_findings.append("profiling pipeline executed")

    refined_runtime_maps = per_pid_runtime_maps.get(str(primary_pid), {})
    if not refined_runtime_maps and per_pid_runtime_maps:
        first_key = next(iter(per_pid_runtime_maps))
        refined_runtime_maps = per_pid_runtime_maps[first_key]

    primary_collect_profile = per_pid_collect_profile.get(str(primary_pid), {})
    if not primary_collect_profile and per_pid_collect_profile:
        first_key = next(iter(per_pid_collect_profile))
        primary_collect_profile = per_pid_collect_profile[first_key]

    virtual_probe_results = await finish_toolbox_probe_sessions(
        probe_sessions,
        {
            "target_pids": target_pids,
            "primary_pid": primary_pid,
            "memory_high": memory_high,
            "stdout": stdout,
            "stderr": stderr,
            "runtime_output": runtime_output,
            "refined_runtime_maps": refined_runtime_maps,
            "per_pid_runtime_maps": per_pid_runtime_maps,
            "collect_profile_data": primary_collect_profile,
            "per_pid_collect_profile": per_pid_collect_profile,
            "process_tree_targets": process_targets,
            "command_parts": command_parts,
            "workdir": workdir,
            "launch_env": env,
        },
    )
    for probe_result in virtual_probe_results.values():
        flat = probe_result.get("flat", {}) if isinstance(probe_result, dict) else {}
        if isinstance(flat, dict) and flat:
            refined_runtime_maps = {**refined_runtime_maps, **flat}

    cgroup_probe_result = virtual_probe_results.get("cgroup.memory.events.high", {})
    cgroup_memory_evidence = (
        cgroup_probe_result.get("structured", {})
        if isinstance(cgroup_probe_result, dict)
        else {}
    )
    if isinstance(cgroup_memory_evidence, dict) and cgroup_memory_evidence.get("control_group"):
        key_findings.append(
            "cgroup memory envelope: "
            f"status={cgroup_memory_evidence.get('status', 'unknown')} "
            f"peak={_format_bytes_human(cast(int | None, cgroup_memory_evidence.get('peak_bytes')))} "
            f"headroom={_format_bytes_human(cast(int | None, cgroup_memory_evidence.get('min_headroom_bytes')))}"
        )

    profiling_json: dict[str, Any] = {
        "pipeline_used": True,
        "tools_used": tools_used,
        "target_pids": target_pids,
        "primary_pid": primary_pid,
        "collect_profile_summary": " | ".join(collect_summaries)[:1800],
        "collect_profile_data": primary_collect_profile,
        "ebpf_summary": ebpf_summary,
        "refined_runtime_maps": refined_runtime_maps,
        "per_pid_runtime_maps": per_pid_runtime_maps,
        "cgroup_memory_evidence": cgroup_memory_evidence,
        "virtual_probe_results": virtual_probe_results,
        "runtime_output": runtime_output,
        "process_tree_targets": process_targets,
        "key_findings": key_findings,
    }

    if selected_hypothesis_profiles:
        (
            validation_result,
            validation_tool_calls,
            validation_tools,
        ) = await _run_contract_hypothesis_validation(
            agent,
            selected_hypothesis_profiles,
            profiling_json,
            stdout,
            stderr,
            report_env_overrides,
            {
                **(
                    {key: _stringify_env_value(value) for key, value in env_overrides.items()}
                    if env_overrides
                    else {}
                ),
                **report_env_overrides,
            },
        )
        agent_tool_calls += validation_tool_calls
        tools_used.extend(validation_tools)
        profiling_json["tools_used"] = tools_used
        profiling_json["contract_hypothesis_validation"] = validation_result
        if validation_result.get("profiles"):
            key_findings.append(
                f"contract-guided validation evaluated {len(validation_result.get('profiles', []))} hypothesis profile(s)"
            )
        elif validation_result.get("error"):
            key_findings.append(
                f"contract-guided validation unavailable: {validation_result.get('error')}"
            )
        profiling_json["key_findings"] = key_findings
    profiling_text = json.dumps(profiling_json)

    result = {
        "pid": pid,
        "launch_command": launch_command,
        "wall_time": wall,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": rc,
        "timed_out": timed_out,
        "profiling_text": profiling_text,
        "profiling_json": profiling_json,
        "tool_calls_used": agent_tool_calls,
    }
    agent.context.target_pid = prev_target_pid
    return result


async def _run_spawn_tune_loop_async(
    agent: SysPilotAgent,
    command: str,
    contracts: Sequence[str],
    manual_hints: Sequence[str],
    max_iterations: int,
    timeout: int,
    primary_objective: str | None = None,
    primary_objective_mode: str | None = None,
    workdir: str | None = None,
    memory_high: str | None = None,
    sink: OutputSink | None = None,
    event_emitter: _TuneEmitter | None = None,
    structured_context_bundle: dict[str, object] | None = None,
    initial_hypothesis_profiles: list[dict[str, Any]] | None = None,
) -> None:
    if sink is None:
        sink = make_output_sink(ui_mode=TuneUIMode.PLAIN, console=console)

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
            {"mode": "spawn", "status": "invalid_input"},
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
            {"mode": "spawn", "status": "invalid_input"},
        )
        sink.print("[red]Spawn command contained only inline env assignments[/red]")
        return
    if inline_env_overrides:
        sink.print(f"[cyan]Detected inline environment baseline: {inline_env_overrides}[/cyan]")

    loaded_contract_names = list(contracts)
    manual_hint_list = list(manual_hints)
    loaded_manual_names = sorted(agent.context.loaded_manuals.keys())

    _emit_tune_event(
        event_emitter,
        "run.started",
        {
            "mode": "spawn",
            "command": command,
            "workdir": workdir or "",
            "memory_high": memory_high or "",
            "max_iterations": max_iterations,
            "timeout_s": timeout,
            "loaded_contracts": loaded_contract_names,
            "loaded_manuals": loaded_manual_names,
            "manual_hints": manual_hint_list,
            "baseline_env_overrides": dict(inline_env_overrides),
            "startup_hypothesis_profile_count": len(initial_hypothesis_profiles or []),
            "primary_objective_query": primary_objective or "",
            "primary_objective_metric": "",
            "primary_objective_mode": primary_objective_mode or "",
        },
    )

    sink.panel(
        f"[bold green]SysPilot Tuning Engine (Spawn Mode)[/bold green]\n"
        f"Command: {command}\n"
        f"Primary objective: {primary_objective or '(wall_time only)'}"
        f"{f' ({primary_objective_mode})' if primary_objective_mode else ''}\n"
        f"Workdir: {workdir or '(current)'}\n"
        f"MemoryHigh: {memory_high or '(none)'}\n"
        f"Max Iterations: {max_iterations}\n"
        f"Timeout: {timeout}s\n"
        f"Contracts: {len(loaded_contract_names)} loaded\n"
        f"Manual hints: {len(manual_hint_list)} provided",
        title="Starting",
    )

    contract_content = (
        json.dumps(structured_context_bundle, indent=2)
        if structured_context_bundle
        else ""
    )
    manual_hint_content = _render_spawn_manual_hints(manual_hint_list)

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

    sink.print("\n[bold cyan]Round 0 Step 0: Initial Baseline Run (profiled)[/bold cyan]")
    _emit_tune_event(
        event_emitter,
        "phase.started",
        {"mode": "spawn", "phase": "spawn.baseline"},
    )
    _emit_tune_event(
        event_emitter,
        "profile.started",
        {"mode": "spawn", "phase": "spawn.baseline", "run_label": "baseline"},
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
        _print_profile_artifact_summary(sink, baseline_artifacts, prefix="Baseline profiled run")
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
            "profiled_run": True,
            **_event_output_payload(baseline_stdout, baseline_stderr),
        },
    )
    if baseline_rc != 0 and not baseline_timed_out:
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
            {"mode": "spawn", "status": "failed", "returncode": baseline_rc},
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
    best_stdout = baseline_stdout
    best_stderr = baseline_stderr
    best_label = "(none)"
    best_command_parts = list(cmd_parts)
    best_env_overrides: dict[str, str] = {}
    best_profiling_text = baseline_profiling_text
    best_profiling_json = baseline_profiling_json
    best_contract_validation = baseline_contract_validation
    best_profile_artifacts = dict(baseline_artifacts)
    best_metrics = dict(baseline_metrics)
    best_objective_value = baseline_objective_value

    seen_trials: set[tuple[tuple[str, ...], tuple[tuple[str, str], ...]]] = {
        _command_key(current_command_parts, current_env_overrides)
    }
    iteration_summaries: list[str] = []
    stagnation_iterations = 0
    convergence_patience = 2
    min_improvement_pct = 1.0
    active_hypothesis_refs: list[str] = []

    for iteration in range(1, max_iterations + 1):
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
            prefix="Current profiled run",
        )

        history_text = "\n".join(iteration_summaries[-6:]) if iteration_summaries else "(none)"
        tried_commands = [list(parts) for parts, _ in list(seen_trials)[-12:]]

        memory_envelope_block = _render_effective_memory_envelope(profiling_json, memory_high)

        prompt = f"""Analyze this spawn-mode workload and propose the next optimization actions.

## Current Command
{" ".join(current_command_parts)}

## Current Environment Overrides
{json.dumps(current_env_overrides, indent=2, sort_keys=True) if current_env_overrides else "(none)"}

{memory_envelope_block}

## Current Wall Time
{current_time:.2f}s

## Current Primary Objective
{primary_objective or '(not configured)'}

## Current Primary Objective Value
{_format_metric_value(current_objective_value)}

## Current Program Output (excerpt)
{current_stdout[:2500] if current_stdout else "(no stdout)"}

## Prior Iterations (latest first)
{history_text}

## Already Tried Command Forms
{json.dumps(tried_commands)}

## Refined Profiling Evidence
{json.dumps(profiling_json, indent=2) if profiling_json else profiling_text[:2500]}

## Contract Hypothesis Validation
{json.dumps(contract_validation, indent=2) if contract_validation else "(No contract-guided validation findings)"}

## Loaded Contract Context
{contract_content if contract_content else "(No contract context provided)"}

## Application Manual Hints
{manual_hint_content if manual_hint_content else "(No application manual hints provided)"}

## Rules
- Use only profiling evidence, loaded contract context, and application manual hints.
- Loaded contract context may be a structured JSON context bundle; use the structured bundle first when present.
- Contract hypothesis validation, when present, is runtime-generated structured evidence. Prefer it over inferring raw DSL semantics yourself.
- The current accepted run was already profiled during execution, and candidate runs in this iteration will also be profiled during execution.
- Avoid recommending exact commands already tried.
- Keep workload-defining arguments fixed (model/input/token counts/repetitions/output mode) so comparisons stay fair.
- Recommend only options that are valid for the target executable.
- If evidence is insufficient, return no actions.
- Return JSON only (no markdown, no prose).
- Provide up to {max_iterations} candidates ordered best-first.

## JSON Schema
{{
  "summary": "short decision summary",
  "manual_citation": "one concrete rule from loaded manual(s)",
  "evidence": ["concise evidence item", "..."],
  "hypothesis_refs": ["optional hypothesis ids to prioritize next profiling round"],
  "actions": [
    {{
      "config_type": "numactl | env_var | command_prefix | command",
      "parameter": "required only for env_var",
      "new_value": "string or token list; for command use full rewritten command",
      "reason": "why this action follows from evidence",
      "manual_citation": "manual rule supporting this action"
    }}
  ]
}}
"""

        recommendation: dict[str, Any]
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
                "skipped_actions": [],
            }
            sink.print("\n[dim]Deterministic benchmark mode: no tuning actions generated.[/dim]")
        else:
            analysis_started_at = time.time()
            try:
                response = await agent.step(prompt)
            except LLMError as exc:
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
                    "skipped_actions": [],
                }
            else:
                analysis_wall_time_s = time.time() - analysis_started_at
                sink.print(f"\n[dim]{response.content}[/dim]")
                recommendation = _parse_spawn_actions(response.content)
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
            _emit_tune_event(
                event_emitter,
                "phase.finished",
                {
                    "mode": "spawn",
                    "phase": "spawn.iteration",
                    "iteration": iteration,
                    "status": "no_actionable_recommendation",
                },
            )
            sink.print("\n[yellow]Agent recommendation was not actionable[/yellow]")
            sink.print(f"  Reason: {recommendation['error']}")
            break

        if recommendation["skipped_actions"]:
            sink.print("  Skipped invalid actions:")
            for item in recommendation["skipped_actions"]:
                sink.print(f"    - {item}")

        summary = recommendation["summary"] or "No actionable tuning recommendation returned."
        next_action_display = _recommendation_next_action_display(recommendation["actions"])
        recommendation_payload: dict[str, Any] = {
            "mode": "spawn",
            "iteration": iteration,
            "wall_time_s": analysis_wall_time_s,
            "summary": summary,
            "manual_citation": recommendation["manual_citation"],
            "evidence": recommendation["evidence"][:5],
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

        actions = recommendation["actions"]
        if not actions:
            _emit_tune_event(
                event_emitter,
                "phase.finished",
                {
                    "mode": "spawn",
                    "phase": "spawn.iteration",
                    "iteration": iteration,
                    "status": "no_actions",
                },
            )
            sink.print("\n[yellow]Agent recommended no actionable tuning[/yellow]")
            break

        candidate_plans = _build_spawn_candidates(
            current_command_parts,
            actions,
            max_candidates=max_iterations,
            base_env_overrides=current_env_overrides,
        )
        candidate_plans = [
            candidate
            for candidate in candidate_plans
            if _command_key(candidate["cmd"], candidate["env_overrides"]) not in seen_trials
        ]
        if not candidate_plans:
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
            stagnation_iterations += 1
            if stagnation_iterations >= convergence_patience:
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
            continue

        sink.print(
            f"\n[bold cyan]Round {iteration} Step 2: Candidate Search ({len(candidate_plans)} profiled run(s), max {max_iterations})[/bold cyan]"
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
        successful_trials = 0
        non_improving_trials = 0

        for idx, candidate in enumerate(candidate_plans, start=1):
            trial_plan = candidate
            trial_cmd = trial_plan["cmd"]
            trial_env = trial_plan["env_overrides"]
            trial_label = trial_plan["label"]
            skipped_execution_actions = trial_plan["skipped_actions"]
            trial_actions = trial_plan.get("actions", [])

            _emit_tune_event(
                event_emitter,
                "progress.update",
                {
                    "mode": "spawn",
                    "phase": "spawn.candidate_search",
                    "iteration": iteration,
                    "current_trial": idx,
                    "total_trials": len(candidate_plans),
                    "candidate_label": trial_label,
                },
            )

            while True:
                seen_trials.add(_command_key(trial_cmd, trial_env))
                sink.print(f"  Trial {idx}/{len(candidate_plans)}: {trial_label}")
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
                )
                trial_time = trial_artifacts["wall_time"]
                trial_stdout = trial_artifacts.get("stdout", "")
                trial_stderr = trial_artifacts.get("stderr", "")
                trial_rc = trial_artifacts["returncode"]
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
                        "total_trials": len(candidate_plans),
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
                if non_improving_trials >= convergence_patience and successful_trials > 0:
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
                    f"candidate run with actions: {trial_label}; primary objective query: {primary_objective}",
                )
                trial_objective_value = _extract_primary_objective_value(
                    trial_metrics,
                    resolved_primary_objective_metric,
                )
                if trial_objective_value is None:
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
                        f"'{resolved_primary_objective_metric}' for query '{primary_objective}'; rejected[/yellow]"
                    )
                    non_improving_trials += 1
                    if non_improving_trials >= convergence_patience and successful_trials > 0:
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

            successful_trials += 1

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
                f"    {current_change_label}: {current_change_value} ({current_change_pct:+.1f}% vs current)"
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
                if current_change_pct >= min_improvement_pct:
                    non_improving_trials = 0
                else:
                    non_improving_trials += 1
            else:
                non_improving_trials += 1

            if non_improving_trials >= convergence_patience and successful_trials > 0:
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
            stagnation_iterations += 1
            if stagnation_iterations >= convergence_patience:
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
                    f"{_format_metric_value(current_objective_value)} ({improvement_pct:+.1f}% vs previous)"
                )
                sink.print(f"  Current wall time: {current_time:.2f}s")
            else:
                sink.print(
                    f"  New current wall time: {current_time:.2f}s ({improvement_pct:+.1f}% vs previous)"
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
                best_stdout = current_stdout
                best_stderr = current_stderr
                best_label = iteration_best_label
                best_command_parts = list(current_command_parts)
                best_env_overrides = dict(current_env_overrides)
                best_profiling_text = current_profiling_text
                best_profiling_json = current_profiling_json
                best_contract_validation = current_contract_validation
                best_profile_artifacts = dict(current_profile_artifacts)
                best_metrics = dict(current_metrics)
                best_objective_value = current_objective_value

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
            stagnation_iterations += 1
            _emit_tune_event(
                event_emitter,
                "phase.finished",
                {
                    "mode": "spawn",
                    "phase": "spawn.iteration",
                    "iteration": iteration,
                    "status": "no_improvement",
                },
            )

        iteration_summaries.append(
            f"iter {iteration}: cmd={' '.join(current_command_parts)} "
            f"{primary_objective + '=' if primary_objective else ''}"
            f"{_format_metric_value(current_objective_value) if primary_objective else ''} "
            f"wall={current_time:.2f}s"
        )

        if stagnation_iterations >= convergence_patience:
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
    tuned_stdout = best_stdout
    tuned_stderr = best_stderr
    if action_label == "(none)":
        sink.print("  No tuned candidate outperformed baseline; keeping baseline configuration")
    else:
        sink.print(f"  Selected best candidate: {action_label}")
        sink.print(f"  Best command: {' '.join(best_command_parts)}")
        if best_env_overrides:
            sink.print(f"  Best environment overrides: {best_env_overrides}")

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
    if primary_objective and baseline_objective_value is not None and tuned_objective_value is not None:
        primary_improvement_pct = _compute_objective_change_pct(
            baseline_objective_value,
            tuned_objective_value,
            cast(str, primary_objective_mode),
        )
    reported_improvement_pct = (
        primary_improvement_pct if primary_improvement_pct is not None else wall_time_improvement_pct
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
        },
    )

    wall_time_change = f"{wall_time_improvement_pct:+.1f}%"
    if wall_time_improvement_pct > 0 and speedup > 0:
        wall_time_change = f"{speedup:.2f}x speed up ({wall_time_improvement_pct:+.1f}%)"

    table_rows: list[list[str]] = []

    if primary_objective and baseline_objective_value is not None and tuned_objective_value is not None:
        table_rows.append(
            [
                _objective_display_name(primary_objective),
                _format_metric_value(baseline_objective_value),
                _format_metric_value(tuned_objective_value),
                f"{cast(float, primary_improvement_pct):+.1f}%",
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
                f"\n[bold green]✓ Success! {_objective_display_name(primary_objective)} improved by "
                f"{primary_improvement_pct:.1f}% with actions: {action_label}[/bold green]"
            )
        else:
            sink.print(
                f"\n[bold green]✓ Success! {speedup:.2f}x speed up ({reported_improvement_pct:.1f}% improvement) with actions: {action_label}[/bold green]"
            )
        run_status = "improved"
    elif reported_improvement_pct > 0:
        if primary_improvement_pct is not None:
            sink.print(
                f"\n[yellow]Modest primary-objective improvement ({primary_improvement_pct:.1f}%)[/yellow]"
            )
        else:
            sink.print(
                f"\n[yellow]Modest improvement ({speedup:.2f}x speed up, {reported_improvement_pct:.1f}%)[/yellow]"
            )
        run_status = "modest_improvement"
    else:
        sink.print(f"\n[red]No improvement or regression ({reported_improvement_pct:.1f}%)[/red]")
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
        },
    )


def _run_spawn_tune_loop(
    agent: SysPilotAgent,
    command: str,
    contracts: Sequence[str],
    manual_hints: Sequence[str],
    max_iterations: int,
    timeout: int,
    primary_objective: str | None = None,
    primary_objective_mode: str | None = None,
    workdir: Path | None = None,
    memory_high: str | None = None,
    sink: OutputSink | None = None,
    event_emitter: _TuneEmitter | None = None,
    structured_context_bundle: dict[str, object] | None = None,
    initial_hypothesis_profiles: list[dict[str, Any]] | None = None,
) -> None:
    workdir_str = str(workdir) if workdir else None
    asyncio.run(
        _run_spawn_tune_loop_async(
            agent=agent,
            command=command,
            contracts=contracts,
            manual_hints=manual_hints,
            max_iterations=max_iterations,
            timeout=timeout,
            primary_objective=primary_objective,
            primary_objective_mode=primary_objective_mode,
            workdir=workdir_str,
            memory_high=memory_high,
            sink=sink,
            event_emitter=event_emitter,
            structured_context_bundle=structured_context_bundle,
            initial_hypothesis_profiles=initial_hypothesis_profiles,
        )
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
