from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from syspilot import __version__
from syspilot.core.config import SysPilotConfig, LLMProvider
from syspilot.core.agent import SysPilotAgent
from syspilot.profiling import ProfilingManager
from syspilot.skills import SkillRegistry
from syspilot.skills.builtin import DynamicScriptSkill, StraceAnalyzerSkill, PerfAnalyzerSkill
from syspilot.skills.ebpf import EBPFSkill
from syspilot.tuning import ApplicationTuner
from syspilot.utils import setup_logging, find_process_by_name

console = Console()


def _register_default_skills(agent: SysPilotAgent) -> None:
    registry = SkillRegistry()
    
    for skill_cls in [DynamicScriptSkill, StraceAnalyzerSkill, PerfAnalyzerSkill, EBPFSkill]:
        skill = skill_cls()
        registry.register(skill)
        
        async def skill_executor(skill_instance: Any = skill, **kwargs: Any) -> dict[str, Any]:
            result = await skill_instance.execute(**kwargs)
            return {
                "success": result.success,
                "output": result.output,
                "summary": result.summary,
                "error": result.error,
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


@click.group()
@click.version_option(version=__version__)
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file path")
@click.option("--log-level", default="INFO", help="Logging level")
@click.pass_context
def cli(ctx: click.Context, config: str | None, log_level: str) -> None:
    ctx.ensure_object(dict)
    
    if config:
        ctx.obj["config"] = SysPilotConfig.from_yaml(Path(config))
    else:
        ctx.obj["config"] = SysPilotConfig()
    
    ctx.obj["config"].log_level = log_level
    setup_logging(log_level)


@cli.command()
@click.option("--pid", "-p", type=int, help="Target process ID")
@click.option("--name", "-n", help="Target process name")
@click.option("--interval", "-i", default=5.0, help="Monitoring interval (seconds)")
@click.option("--manual", "-m", multiple=True, help="Load specific manuals (e.g., numactl, pytorch)")
@click.pass_context
def monitor(ctx: click.Context, pid: int | None, name: str | None, interval: float, manual: tuple[str, ...]) -> None:
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
    
    console.print(Panel(
        f"[bold green]SysPilot Monitor[/bold green]\n"
        f"Target: PID {config.target_pid or 'System-wide'}\n"
        f"Interval: {interval}s\n"
        f"Manuals: {len(agent.context.loaded_manuals)} loaded",
        title="Starting",
    ))
    
    try:
        asyncio.run(agent.run_monitoring_loop())
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping monitor...[/yellow]")
        agent.stop()


@cli.command()
@click.argument("pid", type=int)
@click.option("--manual", "-m", multiple=True, help="Load specific manuals (e.g., numactl, pytorch)")
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
    manual_dir = Path(__file__).parent.parent / "manual"
    
    if not manual_dir.exists():
        console.print("[yellow]No manuals directory found[/yellow]")
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
    console.print("\n[dim]Use --manual/-m flag with monitor or analyze commands to load manuals[/dim]")
    console.print("[dim]Example: syspilot analyze 1234 --manual numactl --manual pytorch[/dim]")


@cli.command()
def skills() -> None:
    registry = SkillRegistry()
    
    for skill_cls in [DynamicScriptSkill, StraceAnalyzerSkill, PerfAnalyzerSkill, EBPFSkill]:
        registry.register(skill_cls())
    
    table = Table(title="Available Skills")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="green")
    
    for skill_info in registry.list_skills():
        table.add_row(skill_info["name"], skill_info["description"])
    
    console.print(table)


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
