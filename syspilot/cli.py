from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import click
import psutil
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from syspilot import __version__
from syspilot.core.config import SysPilotConfig, LLMConfig
from syspilot.core.agent import SysPilotAgent
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
from syspilot.skills.builtin import (
    DynamicScriptSkill,
    StraceAnalyzerSkill,
    PerfAnalyzerSkill,
    DataProcessorSkill,
)
from syspilot.skills.ebpf import EBPFSkill
from syspilot.tuning import ApplicationTuner
from syspilot.tuning.engine import TuningEngine
from syspilot.utils import setup_logging, find_process_by_name
from syspilot.ui.emitter import TuneEventEmitter
from syspilot.ui.output_sink import OutputSink, SwitchableOutputSink, TuneUIMode, make_output_sink
from syspilot.ui.tui_sidecar import TuneTuiOrchestrator

console = Console()


class _TuneEmitter(Protocol):
    def publish(
        self,
        event_name: str,
        payload: dict[str, object] | None = None,
        *,
        event_type: str | None = None,
    ) -> bool: ...


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

    if config:
        ctx.obj["config"] = SysPilotConfig.from_yaml(Path(config))
    else:
        # Load defaults, but try to find LLM config
        sys_config = SysPilotConfig()
        # Try to load credentials from default locations
        sys_config.llm = LLMConfig.from_file()
        ctx.obj["config"] = sys_config

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
@click.option("--manual", is_flag=True, help="Use manual redirect URL paste flow")
@click.option(
    "--timeout", default=180, type=int, show_default=True, help="OAuth callback timeout seconds"
)
@click.option("--auth-file", type=click.Path(), help="Path to save OpenAI auth credentials")
def auth_login(manual: bool, timeout: int, auth_file: str | None) -> None:
    verifier, challenge = create_pkce_verifier_and_challenge()
    state = create_state()
    auth_url = build_authorization_url(state, challenge)

    code: str | None = None
    if not manual:
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
@click.option("--pid", "-p", type=int, help="Target process ID")
@click.option("--name", "-n", help="Target process name")
@click.option("--interval", "-i", default=5.0, help="Monitoring interval (seconds)")
@click.option(
    "--manual", "-m", multiple=True, help="Load specific manuals (e.g., numactl, pytorch)"
)
@click.pass_context
def monitor(
    ctx: click.Context, pid: int | None, name: str | None, interval: float, manual: tuple[str, ...]
) -> None:
    config: SysPilotConfig = ctx.obj["config"]
    config.target_pid = pid
    config.target_name = name
    config.monitoring.interval_seconds = interval

    if name and not pid:
        processes = find_process_by_name(name)
        if not processes:
            console.print(f"[red]No process found matching '{name}'[/red]")
            sys.exit(1)
        if len(processes) > 1:
            console.print(f"[yellow]Multiple processes found:[/yellow]")
            for p in processes:
                console.print(f"  PID {p.pid}: {p.name} - {' '.join(p.cmdline[:3])}")
            console.print("[yellow]Please specify --pid[/yellow]")
            sys.exit(1)
        config.target_pid = processes[0].pid

    agent = SysPilotAgent(config)

    if manual:
        console.print(f"[cyan]Loading manuals: {', '.join(manual)}[/cyan]")
        results = agent.load_manuals(list(manual))
        for manual_name, success in results.items():
            if success:
                console.print(f"  [green]✓ Loaded: {manual_name}[/green]")
            else:
                console.print(f"  [red]✗ Failed: {manual_name}[/red]")

    manager = ProfilingManager()
    tuner = ApplicationTuner(config.tuning)

    _register_default_skills(agent)
    _register_profiling_tools(agent, manager)
    _register_tuning_tools(agent, tuner)

    console.print(
        Panel(
            f"[bold green]SysPilot Monitor[/bold green]\n"
            f"Target: PID {config.target_pid or 'System-wide'}\n"
            f"Interval: {interval}s\n"
            f"Manuals: {len(agent.context.loaded_manuals)} loaded",
            title="Starting",
        )
    )

    try:
        asyncio.run(agent.run_monitoring_loop())
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping monitor...[/yellow]")
        agent.stop()


@cli.command()
@click.argument("pid", type=int)
@click.option(
    "--manual", "-m", multiple=True, help="Load specific manuals (e.g., numactl, pytorch)"
)
@click.pass_context
def analyze(ctx: click.Context, pid: int, manual: tuple[str, ...]) -> None:
    config: SysPilotConfig = ctx.obj["config"]
    config.target_pid = pid

    agent = SysPilotAgent(config)

    if manual:
        console.print(f"[cyan]Loading manuals: {', '.join(manual)}[/cyan]")
        results = agent.load_manuals(list(manual))
        for manual_name, success in results.items():
            if success:
                console.print(f"  [green]✓ Loaded: {manual_name}[/green]")
            else:
                console.print(f"  [red]✗ Failed: {manual_name}[/red]")

    manager = ProfilingManager()
    tuner = ApplicationTuner(config.tuning)

    _register_default_skills(agent)
    _register_profiling_tools(agent, manager)
    _register_tuning_tools(agent, tuner)

    console.print(f"[bold]Analyzing process {pid}...[/bold]")

    result = asyncio.run(agent.analyze_target(pid=pid))
    console.print(Panel(result, title="Analysis Result"))


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
    from pathlib import Path

    # Same discovery logic as Agent
    manual_candidates = [
        Path(__file__).resolve().parents[1] / "manual",
        Path("manual"),
        Path("~/.syspilot/manual").expanduser(),
    ]

    manual_dir = None
    for p in manual_candidates:
        if p.exists() and p.is_dir():
            manual_dir = p
            break

    if not manual_dir:
        console.print(
            "[yellow]No manuals directory found (checked source tree and ~/.syspilot/manual)[/yellow]"
        )
        return

    manuals_list = [p.stem for p in manual_dir.glob("*.md")]

    if not manuals_list:
        console.print("[yellow]No manuals available[/yellow]")
        return

    table = Table(title="Available Tuning Manuals")
    table.add_column("Manual", style="cyan")
    table.add_column("Path", style="green")

    for manual_name in sorted(manuals_list):
        manual_path = manual_dir / f"{manual_name}.md"
        table.add_row(manual_name, str(manual_path))

    console.print(table)
    console.print("\n[dim]Use --manual/-m with monitor, analyze, or tune commands[/dim]")
    console.print(
        "[dim]Examples: syspilot analyze 1234 --manual numactl | syspilot tune --pid 1234 --manual numactl[/dim]"
    )


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
@click.option("--pid", "-p", type=int, help="Target process ID (attach mode)")
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
    "--manual", "-m", multiple=True, help="Load specific manuals (e.g., numactl, pytorch)"
)
@click.option("--max-iterations", default=5, help="Maximum tuning iterations")
@click.option("--dry-run/--no-dry-run", default=True, help="Show changes without applying")
@click.option("--timeout", default=120, help="Timeout per command run in seconds")
@click.option(
    "--parse-output/--no-parse-output",
    default=False,
    help="Use LLM to extract metrics from program output",
)
@click.pass_context
def tune(
    ctx: click.Context,
    pid: int | None,
    command: str | None,
    ui_mode: str,
    tui_bin: Path | None,
    workdir: Path | None,
    manual: tuple[str, ...],
    max_iterations: int,
    dry_run: bool,
    timeout: int,
    parse_output: bool,
) -> None:
    """Iteratively tune a running application.

    Two modes:
      --pid: Attach to existing process and profile it
      --command: Spawn command, profile, get recommendations, respawn with tuning

    Example (spawn mode):
      syspilot tune -c "./app --arg value" --workdir /path/to/app -m numactl
    """
    requested_mode = TuneUIMode(ui_mode.lower())
    plain_sink = make_output_sink(ui_mode=TuneUIMode.PLAIN, console=console)
    sink: OutputSink = plain_sink

    if not pid and not command:
        sink.print("[red]Error: Must specify either --pid or --command[/red]")
        raise SystemExit(1)

    if pid and command:
        sink.print("[red]Error: Cannot specify both --pid and --command[/red]")
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

    if manual:
        sink.print(f"[cyan]Loading manuals: {', '.join(manual)}[/cyan]")
        results = agent.load_manuals(list(manual))
        for manual_name, success in results.items():
            if success:
                sink.print(f"  [green]✓ Loaded: {manual_name}[/green]")
            else:
                sink.print(f"  [red]✗ Failed: {manual_name}[/red]")

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
        if command:
            _run_spawn_tune_loop(
                agent,
                command,
                manual,
                max_iterations,
                timeout,
                parse_output,
                workdir,
                sink,
                event_emitter,
            )
        else:
            assert pid is not None
            _run_attach_tune_loop(
                agent,
                manager,
                tuner,
                pid,
                max_iterations,
                dry_run,
                sink,
                event_emitter,
            )
    finally:
        if logger_forward_handler is not None:
            syspilot_logger.removeHandler(logger_forward_handler)
        if tui_orchestrator is not None:
            tui_orchestrator.close(wait_for_user=True)


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

    def publish(
        self,
        event_name: str,
        payload: dict[str, object] | None = None,
        *,
        event_type: str | None = None,
    ) -> bool:
        ok = self._emitter.publish(event_name, payload, event_type=event_type)
        self._orchestrator.forward_emitter(self._emitter)
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


def _run_attach_tune_loop(
    agent: SysPilotAgent,
    manager: ProfilingManager,
    tuner: ApplicationTuner,
    pid: int,
    max_iterations: int,
    dry_run: bool,
    sink: OutputSink,
    event_emitter: _TuneEmitter | None = None,
) -> None:
    agent.config.target_pid = pid

    engine = TuningEngine(
        profiling_manager=manager,
        tuner=tuner,
        max_iterations=max_iterations,
    )

    sink.panel(
        f"[bold green]SysPilot Tuning Engine (Attach Mode)[/bold green]\n"
        f"Target: PID {pid}\n"
        f"Max Iterations: {max_iterations}\n"
        f"Dry Run: {dry_run}\n"
        f"Manuals: {len(agent.context.loaded_manuals)} loaded",
        title="Starting",
    )

    _emit_tune_event(
        event_emitter,
        "run.started",
        {
            "mode": "attach",
            "pid": pid,
            "max_iterations": max_iterations,
            "dry_run": dry_run,
        },
    )
    _emit_tune_event(
        event_emitter,
        "phase.started",
        {"mode": "attach", "phase": "attach.tuning_loop"},
    )

    try:
        result = asyncio.run(agent.run_tuning_loop(engine, max_iterations))
        sink.panel(result, title="Tuning Results")
        _emit_tune_event(
            event_emitter,
            "phase.finished",
            {"mode": "attach", "phase": "attach.tuning_loop", "status": "success"},
        )
        _emit_tune_event(
            event_emitter,
            "run.finished",
            {"mode": "attach", "status": "success"},
        )
    except KeyboardInterrupt:
        _emit_tune_event(
            event_emitter,
            "error",
            {
                "mode": "attach",
                "phase": "attach.tuning_loop",
                "message": "Keyboard interrupt",
            },
        )
        _emit_tune_event(
            event_emitter,
            "phase.finished",
            {"mode": "attach", "phase": "attach.tuning_loop", "status": "interrupted"},
        )
        _emit_tune_event(
            event_emitter,
            "run.finished",
            {"mode": "attach", "status": "interrupted"},
        )
        sink.print("\n[yellow]Stopping tuning...[/yellow]")
        agent.stop()


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

    actions_raw = parsed.get("actions", [])
    if not isinstance(actions_raw, list):
        return {
            "valid": False,
            "error": "Field 'actions' must be a list",
            "summary": summary,
            "manual_citation": manual_citation,
            "evidence": evidence,
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
            "actions": [],
            "skipped_actions": skipped_actions,
        }

    return {
        "valid": True,
        "error": "",
        "summary": summary,
        "manual_citation": manual_citation,
        "evidence": evidence,
        "actions": normalized_actions,
        "skipped_actions": skipped_actions,
    }


def _build_spawn_execution_plan(
    base_command: list[str],
    actions: list[dict[str, Any]],
) -> tuple[list[str], dict[str, str], list[str], list[str]]:
    if not base_command:
        return [], {}, [], ["base command was empty"]

    prefix_tokens: list[str] = []
    command_tokens = list(base_command)
    command_replaced = False
    env_overrides: dict[str, str] = {}
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
            if Path(tokens[0]).name != Path(base_command[0]).name:
                skipped.append(
                    f"action[{idx}] command replacement must keep executable '{base_command[0]}'"
                )
                continue
            command_tokens = tokens
            command_replaced = True
            applied.append(f"command {' '.join(tokens)}")
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
) -> list[dict[str, Any]]:
    if max_candidates < 1:
        return []

    shared_actions: list[dict[str, Any]] = []
    command_actions: list[dict[str, Any]] = []

    for action in actions:
        if str(action.get("config_type", "")).strip().lower() == "command":
            command_actions.append(action)
        else:
            shared_actions.append(action)

    candidate_action_sets: list[list[dict[str, Any]]] = []
    if command_actions:
        candidate_action_sets.extend([shared_actions + [action] for action in command_actions])
    elif shared_actions:
        candidate_action_sets.append(shared_actions)

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, ...], tuple[tuple[str, str], ...]]] = set()

    for action_set in candidate_action_sets:
        tuned_cmd, env_overrides, applied_actions, skipped_actions = _build_spawn_execution_plan(
            base_command,
            action_set,
        )
        if not tuned_cmd:
            continue

        if tuned_cmd == base_command and not env_overrides:
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

    response = await agent.step(prompt)

    import re

    json_match = re.search(r"\{[^{}]*\}", response.content, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {"no_metrics": True}


async def _run_live_profiling_pipeline(
    agent: SysPilotAgent,
    command_parts: list[str],
    timeout: int,
    workdir: str | None = None,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    started_at = time.monotonic()
    env = None
    if env_overrides:
        env = os.environ.copy()
        env.update({k: str(v) for k, v in env_overrides.items()})

    proc = await asyncio.create_subprocess_exec(
        *command_parts,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=workdir,
        env=env,
    )
    pid = proc.pid
    agent.context.target_pid = pid

    await asyncio.sleep(1.0)

    process_targets = _discover_related_processes(pid, max_targets=4)
    target_pids = process_targets["target_pids"]
    primary_pid = process_targets["primary_pid"]

    tools_used: list[str] = []
    key_findings: list[str] = []
    per_pid_runtime_maps: dict[str, dict[str, Any]] = {}
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

    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        rc = proc.returncode
        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        rc = -1
        stdout = ""
        stderr = "Command timed out during profiling pipeline"

    wall = time.monotonic() - started_at

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

    profiling_json: dict[str, Any] = {
        "pipeline_used": True,
        "tools_used": tools_used,
        "target_pids": target_pids,
        "primary_pid": primary_pid,
        "collect_profile_summary": " | ".join(collect_summaries)[:1800],
        "ebpf_summary": ebpf_summary,
        "refined_runtime_maps": refined_runtime_maps,
        "per_pid_runtime_maps": per_pid_runtime_maps,
        "process_tree_targets": process_targets,
        "key_findings": key_findings,
    }
    profiling_text = json.dumps(profiling_json)

    return {
        "pid": pid,
        "wall_time": wall,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": rc,
        "profiling_text": profiling_text,
        "profiling_json": profiling_json,
        "tool_calls_used": agent_tool_calls,
    }


async def _run_spawn_tune_loop_async(
    agent: SysPilotAgent,
    command: str,
    manuals: tuple[str, ...],
    max_iterations: int,
    timeout: int,
    parse_output: bool = False,
    workdir: str | None = None,
    sink: OutputSink | None = None,
    event_emitter: _TuneEmitter | None = None,
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

    _emit_tune_event(
        event_emitter,
        "run.started",
        {
            "mode": "spawn",
            "command": command,
            "workdir": workdir or "",
            "max_iterations": max_iterations,
            "timeout_s": timeout,
        },
    )

    sink.panel(
        f"[bold green]SysPilot Tuning Engine (Spawn Mode)[/bold green]\n"
        f"Command: {command}\n"
        f"Workdir: {workdir or '(current)'}\n"
        f"Max Iterations: {max_iterations}\n"
        f"Timeout: {timeout}s\n"
        f"Manuals: {len(agent.context.loaded_manuals)} loaded",
        title="Starting",
    )

    manual_content = ""
    if manuals and agent.context.loaded_manuals:
        for manual_name in manuals:
            if manual_name in agent.context.loaded_manuals:
                manual_content += f"\n\n### {manual_name} Manual (excerpt)\n"
                manual_content += agent.context.loaded_manuals[manual_name][:3000]

    sink.print("\n[bold cyan]Step 1: Initial Baseline Run[/bold cyan]")
    _emit_tune_event(
        event_emitter,
        "phase.started",
        {"mode": "spawn", "phase": "spawn.baseline"},
    )
    baseline_time, baseline_stdout, baseline_stderr, baseline_rc = _run_command(
        cmd_parts,
        timeout,
        workdir=workdir,
    )
    if baseline_rc != 0:
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

    _emit_tune_event(
        event_emitter,
        "phase.finished",
        {
            "mode": "spawn",
            "phase": "spawn.baseline",
            "status": "success",
            "wall_time_s": baseline_time,
            "cmd_diff": "baseline",
        },
    )

    sink.print(f"  Wall time: {baseline_time:.2f}s")
    sink.print(f"  Output: {len(baseline_stdout)} bytes")

    current_command_parts = list(cmd_parts)
    current_env_overrides: dict[str, str] = {}
    current_time = baseline_time
    current_stdout = baseline_stdout
    current_stderr = baseline_stderr

    best_time = baseline_time
    best_stdout = baseline_stdout
    best_stderr = baseline_stderr
    best_label = "(none)"
    best_command_parts = list(cmd_parts)
    best_env_overrides: dict[str, str] = {}

    seen_trials: set[tuple[tuple[str, ...], tuple[tuple[str, str], ...]]] = {
        _command_key(current_command_parts, current_env_overrides)
    }
    iteration_summaries: list[str] = []
    stagnation_iterations = 0
    convergence_patience = 2
    min_improvement_pct = 1.0

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
        sink.print(f"\n[bold cyan]Step 2.{iteration}: Iteration Profiling[/bold cyan]")
        profiling_artifacts = await _run_live_profiling_pipeline(
            agent,
            current_command_parts,
            timeout,
            workdir=workdir,
            env_overrides=current_env_overrides,
        )

        sink.print(f"  Profiling run PID: {profiling_artifacts['pid']}")
        sink.print(f"  Profiling run wall time: {profiling_artifacts['wall_time']:.2f}s")
        sink.print(f"  Agent tool calls used: {profiling_artifacts.get('tool_calls_used', 0)}")

        profiling_text = profiling_artifacts.get("profiling_text", "")
        profiling_json = profiling_artifacts.get("profiling_json", {})
        profile_tools: list[str] = []
        profile_json_payload: dict[str, Any] = {}
        if isinstance(profiling_json, dict):
            profile_json_payload = dict(profiling_json)
            raw_tools = profiling_json.get("tools_used")
            if isinstance(raw_tools, list):
                profile_tools = [str(tool) for tool in raw_tools if str(tool).strip()]

        _emit_tune_event(
            event_emitter,
            "profile.snapshot",
            {
                "mode": "spawn",
                "iteration": iteration,
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
        if profiling_artifacts["returncode"] != 0:
            sink.print(
                f"  [yellow]Profiling command return code: {profiling_artifacts['returncode']}[/yellow]"
            )

        if profiling_json:
            sink.print(f"  Refined profiling JSON: {profiling_json}")
        else:
            snippet = profiling_text[:500] + ("..." if len(profiling_text) > 500 else "")
            sink.print(f"  Refined profiling summary: {snippet}")

        target_pids = (
            profiling_json.get("target_pids", []) if isinstance(profiling_json, dict) else []
        )
        primary_pid = (
            profiling_json.get("primary_pid") if isinstance(profiling_json, dict) else None
        )
        if target_pids:
            sink.print(f"  Profiling targets: {target_pids} (primary={primary_pid})")

        history_text = "\n".join(iteration_summaries[-6:]) if iteration_summaries else "(none)"
        tried_commands = [list(parts) for parts, _ in list(seen_trials)[-12:]]

        sink.print(f"\n[bold cyan]Step 3.{iteration}: Agent Tuning Analysis[/bold cyan]")
        prompt = f"""Analyze this spawn-mode workload and propose the next manual-driven optimization actions.

## Current Command
{" ".join(current_command_parts)}

## Current Wall Time
{current_time:.2f}s

## Current Program Output (excerpt)
{current_stdout[:2500] if current_stdout else "(no stdout)"}

## Prior Iterations (latest first)
{history_text}

## Already Tried Command Forms
{json.dumps(tried_commands)}

## Refined Profiling Evidence
{json.dumps(profiling_json, indent=2) if profiling_json else profiling_text[:2500]}

## Loaded Manual Evidence
{manual_content if manual_content else "(No manual content provided)"}

## Rules
- Use only profiling evidence and loaded manuals.
- Re-profile is performed each iteration; propose next-step actions for this iteration.
- Avoid recommending exact commands already tried.
- If evidence is insufficient, return no actions.
- Return JSON only (no markdown, no prose).
- Provide up to {max_iterations} candidates ordered best-first.

## JSON Schema
{{
  "summary": "short decision summary",
  "manual_citation": "one concrete rule from loaded manual(s)",
  "evidence": ["concise evidence item", "..."],
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

        response = await agent.step(prompt)
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
        _emit_tune_event(
            event_emitter,
            "recommendation.ready",
            {
                "mode": "spawn",
                "iteration": iteration,
                "summary": summary,
                "manual_citation": recommendation["manual_citation"],
                "evidence": recommendation["evidence"][:5],
                "action_count": len(recommendation["actions"]),
            },
        )
        sink.print(f"  Decision summary: {summary}")
        if recommendation["manual_citation"]:
            sink.print(f"  Manual citation: {recommendation['manual_citation']}")
        if recommendation["evidence"]:
            for evidence_item in recommendation["evidence"][:5]:
                sink.print(f"  Evidence: {evidence_item}")

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
            f"\n[bold cyan]Step 4.{iteration}: Candidate Search ({len(candidate_plans)} plan(s), max {max_iterations})[/bold cyan]"
        )

        iteration_best_time = current_time
        iteration_best_stdout = current_stdout
        iteration_best_stderr = current_stderr
        iteration_best_label = "(none)"
        iteration_best_cmd = list(current_command_parts)
        iteration_best_env = dict(current_env_overrides)
        successful_trials = 0
        non_improving_trials = 0

        for idx, candidate in enumerate(candidate_plans, start=1):
            trial_cmd = candidate["cmd"]
            trial_env = candidate["env_overrides"]
            trial_label = candidate["label"]
            skipped_execution_actions = candidate["skipped_actions"]

            seen_trials.add(_command_key(trial_cmd, trial_env))
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

            sink.print(f"  Trial {idx}/{len(candidate_plans)}: {trial_label}")
            sink.print(f"    Running: {' '.join(trial_cmd)}")
            if trial_env:
                sink.print(f"    Environment overrides: {trial_env}")
            if skipped_execution_actions:
                for item in skipped_execution_actions:
                    sink.print(f"    Skipped: {item}")

            trial_time, trial_stdout, trial_stderr, trial_rc = _run_command(
                trial_cmd,
                timeout,
                workdir=workdir,
                env_overrides=trial_env,
            )

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

            successful_trials += 1
            current_change_pct = (
                ((current_time - trial_time) / current_time) * 100 if current_time > 0 else 0
            )
            _emit_tune_event(
                event_emitter,
                "candidate.evaluated",
                {
                    "mode": "spawn",
                    "iteration": iteration,
                    "candidate_id": idx,
                    "label": trial_label,
                    "verdict": "improved" if trial_time < iteration_best_time else "not_improved",
                    "wall_time_s": trial_time,
                    "change_pct_vs_current": current_change_pct,
                    "cmd_diff": trial_label,
                },
            )
            sink.print(f"    Wall time: {trial_time:.2f}s ({current_change_pct:+.1f}% vs current)")

            if trial_time < iteration_best_time:
                iteration_best_time = trial_time
                iteration_best_stdout = trial_stdout
                iteration_best_stderr = trial_stderr
                iteration_best_label = trial_label
                iteration_best_cmd = trial_cmd
                iteration_best_env = trial_env
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

        if iteration_best_time < current_time:
            improvement_pct = (
                ((current_time - iteration_best_time) / current_time) * 100
                if current_time > 0
                else 0
            )
            current_time = iteration_best_time
            current_stdout = iteration_best_stdout
            current_stderr = iteration_best_stderr
            current_command_parts = list(iteration_best_cmd)
            current_env_overrides = dict(iteration_best_env)
            sink.print(f"  Accepted candidate: {iteration_best_label}")
            sink.print(
                f"  New current wall time: {current_time:.2f}s ({improvement_pct:+.1f}% vs previous)"
            )

            if current_time < best_time:
                best_time = current_time
                best_stdout = current_stdout
                best_stderr = current_stderr
                best_label = iteration_best_label
                best_command_parts = list(current_command_parts)
                best_env_overrides = dict(current_env_overrides)

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
            f"iter {iteration}: cmd={' '.join(current_command_parts)} wall={current_time:.2f}s"
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

    baseline_metrics: dict[str, Any] = {}
    tuned_metrics: dict[str, Any] = {}

    if parse_output:
        sink.print("\n[bold cyan]Step 5: Extracting Metrics from Output[/bold cyan]")
        sink.print("  Parsing baseline output...")
        baseline_metrics = await _extract_metrics_from_output(
            agent,
            baseline_stdout,
            "baseline run without tuning actions",
        )
        sink.print(f"  Baseline metrics: {baseline_metrics}")

        sink.print("  Parsing tuned output...")
        tuned_metrics = await _extract_metrics_from_output(
            agent,
            tuned_stdout,
            f"tuned run with actions: {action_label}",
        )
        sink.print(f"  Tuned metrics: {tuned_metrics}")

    comparison_step = 6 if parse_output else 5
    sink.print(f"\n[bold cyan]Step {comparison_step}: Results Comparison[/bold cyan]")

    speedup = baseline_time / tuned_time if tuned_time > 0 else 0
    improvement_pct = (
        ((baseline_time - tuned_time) / baseline_time) * 100 if baseline_time > 0 else 0
    )
    _emit_tune_event(
        event_emitter,
        "result.comparison",
        {
            "mode": "spawn",
            "baseline_time_s": baseline_time,
            "tuned_time_s": tuned_time,
            "improvement_pct": improvement_pct,
            "speedup": speedup,
            "action_label": action_label,
        },
    )

    table_rows: list[list[str]] = [
        [
            "Wall Time",
            f"{baseline_time:.2f}s",
            f"{tuned_time:.2f}s",
            f"{improvement_pct:+.1f}% ({speedup:.2f}x)",
        ]
    ]

    if parse_output and baseline_metrics and tuned_metrics:
        if not baseline_metrics.get("no_metrics") and not tuned_metrics.get("no_metrics"):
            all_keys = set(baseline_metrics.keys()) | set(tuned_metrics.keys())
            for key in sorted(all_keys):
                if key in ("no_metrics",):
                    continue
                base_val = baseline_metrics.get(key)
                tuned_val = tuned_metrics.get(key)

                if (
                    isinstance(base_val, (int, float))
                    and isinstance(tuned_val, (int, float))
                    and base_val != 0
                ):
                    change_pct = ((tuned_val - base_val) / abs(base_val)) * 100
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

    if improvement_pct > 5:
        sink.print(
            f"\n[bold green]✓ Success! {improvement_pct:.1f}% improvement with actions: {action_label}[/bold green]"
        )
        run_status = "improved"
    elif improvement_pct > 0:
        sink.print(f"\n[yellow]Modest improvement ({improvement_pct:.1f}%)[/yellow]")
        run_status = "modest_improvement"
    else:
        sink.print(f"\n[red]No improvement or regression ({improvement_pct:.1f}%)[/red]")
        run_status = "no_improvement"

    _emit_tune_event(
        event_emitter,
        "run.finished",
        {
            "mode": "spawn",
            "status": run_status,
            "baseline_time_s": baseline_time,
            "tuned_time_s": tuned_time,
            "improvement_pct": improvement_pct,
            "action_label": action_label,
        },
    )


def _run_spawn_tune_loop(
    agent: SysPilotAgent,
    command: str,
    manuals: tuple[str, ...],
    max_iterations: int,
    timeout: int,
    parse_output: bool = False,
    workdir: Path | None = None,
    sink: OutputSink | None = None,
    event_emitter: _TuneEmitter | None = None,
) -> None:
    workdir_str = str(workdir) if workdir else None
    asyncio.run(
        _run_spawn_tune_loop_async(
            agent=agent,
            command=command,
            manuals=manuals,
            max_iterations=max_iterations,
            timeout=timeout,
            parse_output=parse_output,
            workdir=workdir_str,
            sink=sink,
            event_emitter=event_emitter,
        )
    )


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
