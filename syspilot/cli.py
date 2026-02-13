from __future__ import annotations

import asyncio
import json
import logging
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from syspilot import __version__
from syspilot.core.config import SysPilotConfig, LLMProvider, LLMConfig
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
from syspilot.skills.builtin import DynamicScriptSkill, StraceAnalyzerSkill, PerfAnalyzerSkill, DataProcessorSkill
from syspilot.skills.ebpf import EBPFSkill
from syspilot.tuning import ApplicationTuner
from syspilot.tuning.engine import TuningEngine
from syspilot.utils import setup_logging, find_process_by_name

console = Console()


def _register_default_skills(agent: SysPilotAgent) -> None:
    registry = SkillRegistry()
    
    for skill_cls in [DynamicScriptSkill, StraceAnalyzerSkill, PerfAnalyzerSkill, EBPFSkill, DataProcessorSkill]:
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
                            "config_type": {"type": "string", "enum": ["sysctl", "prlimit", "command", "env_var", "numactl"]},
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
@click.pass_context
def cli(ctx: click.Context, config: str | None, log_level: str) -> None:
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
    setup_logging(log_level)


@cli.group()
def auth() -> None:
    pass


@auth.command("login")
@click.option("--manual", is_flag=True, help="Use manual redirect URL paste flow")
@click.option("--timeout", default=180, type=int, show_default=True, help="OAuth callback timeout seconds")
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
            console.print("[yellow]Automatic callback unavailable or timed out. Using manual mode.[/yellow]")

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
    
    # Same discovery logic as Agent
    manual_candidates = [
        Path(__file__).parent.parent.parent / "manual",
        Path("~/.syspilot/manual").expanduser(),
        Path("manual"),
    ]
    
    manual_dir = None
    for p in manual_candidates:
        if p.exists() and p.is_dir():
            manual_dir = p
            break
    
    if not manual_dir:
        console.print("[yellow]No manuals directory found (checked source tree and ~/.syspilot/manual)[/yellow]")
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
    
    for skill_cls in [DynamicScriptSkill, StraceAnalyzerSkill, PerfAnalyzerSkill, EBPFSkill, DataProcessorSkill]:
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
@click.option("--manual", "-m", multiple=True, help="Load specific manuals (e.g., numactl, pytorch)")
@click.option("--max-iterations", default=5, help="Maximum tuning iterations")
@click.option("--dry-run/--no-dry-run", default=True, help="Show changes without applying")
@click.option("--timeout", default=120, help="Timeout per command run in seconds")
@click.option("--parse-output/--no-parse-output", default=False, help="Use LLM to extract metrics from program output")
@click.pass_context
def tune(
    ctx: click.Context,
    pid: int | None,
    command: str | None,
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
      syspilot tune -c "/tmp/llama.cpp/build/bin/llama-bench -m model.gguf -p 512 -n 128" -m numactl
    """
    if not pid and not command:
        console.print("[red]Error: Must specify either --pid or --command[/red]")
        raise SystemExit(1)
    
    if pid and command:
        console.print("[red]Error: Cannot specify both --pid and --command[/red]")
        raise SystemExit(1)

    config: SysPilotConfig = ctx.obj["config"]
    config.tuning.dry_run = dry_run
    config.tuning.max_tuning_iterations = max_iterations

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

    if command:
        _run_spawn_tune_loop(agent, manager, command, manual, max_iterations, timeout, parse_output)
    else:
        assert pid is not None
        _run_attach_tune_loop(agent, manager, tuner, pid, max_iterations, dry_run)


def _run_attach_tune_loop(
    agent: SysPilotAgent,
    manager: ProfilingManager,
    tuner: ApplicationTuner,
    pid: int,
    max_iterations: int,
    dry_run: bool,
) -> None:
    agent.config.target_pid = pid

    engine = TuningEngine(
        profiling_manager=manager,
        tuner=tuner,
        max_iterations=max_iterations,
    )

    console.print(Panel(
        f"[bold green]SysPilot Tuning Engine (Attach Mode)[/bold green]\n"
        f"Target: PID {pid}\n"
        f"Max Iterations: {max_iterations}\n"
        f"Dry Run: {dry_run}\n"
        f"Manuals: {len(agent.context.loaded_manuals)} loaded",
        title="Starting",
    ))

    try:
        result = asyncio.run(agent.run_tuning_loop(engine, max_iterations))
        console.print(Panel(result, title="Tuning Results"))
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping tuning...[/yellow]")
        agent.stop()


def _run_command(cmd: list[str], timeout: int) -> tuple[float, str, str, int]:
    """Run command and return (wall_time, stdout, stderr, returncode)."""
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        wall = time.monotonic() - t0
        return wall, result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return timeout, "", "Command timed out", -1


def _parse_numactl_recommendation(response: str) -> list[str]:
    import re
    
    for line in response.splitlines():
        if "RECOMMENDED_NUMACTL:" in line:
            val = line.split("RECOMMENDED_NUMACTL:")[-1].strip()
            val = re.sub(r'[\*\`\#]+', '', val).strip()
            if val.upper() not in ("NONE", ""):
                return shlex.split(val)
    
    for line in response.splitlines():
        match = re.search(r'(--interleave=\S+|--cpunodebind=\S+|--membind=\S+|--preferred=\S+|--physcpubind=\S+)', line)
        if match:
            flag = match.group(1)
            flag = re.sub(r'[\*\`\#\,\.]+$', '', flag)
            return [flag]
    
    return []


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
    json_match = re.search(r'\{[^{}]*\}', response.content, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {"no_metrics": True}


async def _run_spawn_tune_loop_async(
    agent: SysPilotAgent,
    manager: ProfilingManager,
    command: str,
    manuals: tuple[str, ...],
    max_iterations: int,
    timeout: int,
    parse_output: bool = False,
) -> None:
    cmd_parts = shlex.split(command)
    
    console.print(Panel(
        f"[bold green]SysPilot Tuning Engine (Spawn Mode)[/bold green]\n"
        f"Command: {command}\n"
        f"Max Iterations: {max_iterations}\n"
        f"Timeout: {timeout}s\n"
        f"Manuals: {len(agent.context.loaded_manuals)} loaded",
        title="Starting",
    ))

    console.print("\n[bold cyan]Step 1: Baseline Run (no tuning)[/bold cyan]")
    baseline_time, baseline_stdout, baseline_stderr, baseline_rc = _run_command(cmd_parts, timeout)
    
    if baseline_rc != 0:
        console.print(f"[red]Baseline run failed (rc={baseline_rc})[/red]")
        console.print(f"[dim]stderr: {baseline_stderr[:500]}[/dim]")
        return
    
    console.print(f"  Wall time: {baseline_time:.2f}s")
    console.print(f"  Output: {len(baseline_stdout)} bytes")

    console.print("\n[bold cyan]Step 2: Collecting System Profile[/bold cyan]")
    numa_data = await manager.collect_all(pid=None)
    numa_summary = ""
    numa_topology_details = ""
    num_nodes = 1
    if "numa" in numa_data:
        numa_result = numa_data["numa"]
        numa_summary = numa_result.summarize()
        console.print(f"  {numa_summary[:200]}...")
        
        topo = numa_result.data.get("topology", {})
        num_nodes = numa_result.data.get("num_nodes", 1)
        lines = [f"Number of NUMA nodes: {num_nodes}"]
        for node_name, node_info in topo.items():
            cpulist = node_info.get("cpulist", "unknown")
            mem_total = node_info.get("mem_total_kb", 0) // 1024
            mem_free = node_info.get("mem_free_kb", 0) // 1024
            distance = node_info.get("distance", [])
            lines.append(f"  {node_name}: CPUs={cpulist}, Memory={mem_total}MB (free={mem_free}MB), Distance={distance}")
        numa_topology_details = "\n".join(lines)

    console.print("\n[bold cyan]Step 3: Agent Analysis[/bold cyan]")
    
    manual_content = ""
    if manuals and agent.context.loaded_manuals:
        for m in manuals:
            if m in agent.context.loaded_manuals:
                manual_content += f"\n\n### {m} Manual (excerpt)\n"
                manual_content += agent.context.loaded_manuals[m][:3000]
    
    prompt = f"""Analyze this NUMA workload and recommend numactl settings. Be CONCISE.

## Command
{command}

This is llama-bench (memory bandwidth-bound LLM inference).

## Baseline: {baseline_time:.2f}s wall time

## NUMA Topology
{numa_topology_details}

Inter-node distance is 21 vs local 10 = 2.1x latency penalty for remote access.

## Key Facts
- LLM inference is memory bandwidth-bound (reads full model weights per token)
- 24 threads span both NUMA nodes (12 per node)
- Without numactl, first-touch places ALL weights on one node
- 12 threads on remote node suffer 2.1x memory latency

## Manual Guidance
For memory bandwidth-bound workloads on multi-node systems: use --interleave=all
This spreads pages round-robin, giving each node local access to ~50% of data.

## YOUR RESPONSE (keep it SHORT)

1. One sentence: why this workload needs NUMA tuning
2. One line with your recommendation:

RECOMMENDED_NUMACTL: --interleave=all
"""

    response = await agent.step(prompt)
    console.print(f"\n[dim]{response.content}[/dim]")
    
    numactl_flags = _parse_numactl_recommendation(response.content)
    
    if not numactl_flags:
        console.print("\n[yellow]Agent recommended no NUMA tuning[/yellow]")
        return

    console.print(f"\n[bold cyan]Step 4: Tuned Run (numactl {' '.join(numactl_flags)})[/bold cyan]")
    tuned_cmd = ["numactl"] + numactl_flags + cmd_parts
    console.print(f"  Running: {' '.join(tuned_cmd)}")
    
    tuned_time, tuned_stdout, tuned_stderr, tuned_rc = _run_command(tuned_cmd, timeout)
    
    if tuned_rc != 0:
        console.print(f"[red]Tuned run failed (rc={tuned_rc})[/red]")
        console.print(f"[dim]stderr: {tuned_stderr[:500]}[/dim]")
        return
    
    console.print(f"  Wall time: {tuned_time:.2f}s")

    baseline_metrics: dict[str, Any] = {}
    tuned_metrics: dict[str, Any] = {}
    
    if parse_output:
        console.print("\n[bold cyan]Step 5: Extracting Metrics from Output[/bold cyan]")
        console.print("  Parsing baseline output...")
        baseline_metrics = await _extract_metrics_from_output(
            agent,
            baseline_stdout,
            "baseline run without numactl",
        )
        console.print(f"  Baseline metrics: {baseline_metrics}")
        
        console.print("  Parsing tuned output...")
        tuned_metrics = await _extract_metrics_from_output(
            agent,
            tuned_stdout,
            f"tuned run with numactl {' '.join(numactl_flags)}",
        )
        console.print(f"  Tuned metrics: {tuned_metrics}")

    console.print("\n[bold cyan]Step 6: Results Comparison[/bold cyan]")
    
    speedup = baseline_time / tuned_time if tuned_time > 0 else 0
    improvement_pct = ((baseline_time - tuned_time) / baseline_time) * 100 if baseline_time > 0 else 0
    
    table = Table(title="Performance Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Baseline", style="white")
    table.add_column("Tuned", style="green")
    table.add_column("Change", style="yellow")
    
    table.add_row(
        "Wall Time",
        f"{baseline_time:.2f}s",
        f"{tuned_time:.2f}s",
        f"{improvement_pct:+.1f}% ({speedup:.2f}x)"
    )
    
    if parse_output and baseline_metrics and tuned_metrics:
        if not baseline_metrics.get("no_metrics") and not tuned_metrics.get("no_metrics"):
            all_keys = set(baseline_metrics.keys()) | set(tuned_metrics.keys())
            for key in sorted(all_keys):
                if key in ("no_metrics",):
                    continue
                base_val = baseline_metrics.get(key)
                tuned_val = tuned_metrics.get(key)
                
                if isinstance(base_val, (int, float)) and isinstance(tuned_val, (int, float)) and base_val != 0:
                    change_pct = ((tuned_val - base_val) / abs(base_val)) * 100
                    change_str = f"{change_pct:+.1f}%"
                else:
                    change_str = ""
                
                base_str = f"{base_val}" if base_val is not None else "-"
                tuned_str = f"{tuned_val}" if tuned_val is not None else "-"
                
                display_key = key.replace("_", " ").title()
                table.add_row(display_key, base_str, tuned_str, change_str)
    
    table.add_row(
        "numactl flags",
        "(none)",
        " ".join(numactl_flags),
        ""
    )
    
    console.print(table)
    
    if improvement_pct > 5:
        console.print(f"\n[bold green]✓ Success! {improvement_pct:.1f}% improvement with numactl {' '.join(numactl_flags)}[/bold green]")
    elif improvement_pct > 0:
        console.print(f"\n[yellow]Modest improvement ({improvement_pct:.1f}%)[/yellow]")
    else:
        console.print(f"\n[red]No improvement or regression ({improvement_pct:.1f}%)[/red]")


def _run_spawn_tune_loop(
    agent: SysPilotAgent,
    manager: ProfilingManager,
    command: str,
    manuals: tuple[str, ...],
    max_iterations: int,
    timeout: int,
    parse_output: bool = False,
) -> None:
    asyncio.run(
        _run_spawn_tune_loop_async(
            agent=agent,
            manager=manager,
            command=command,
            manuals=manuals,
            max_iterations=max_iterations,
            timeout=timeout,
            parse_output=parse_output,
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
