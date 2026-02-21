# SysPilot

AI Agent for System Monitoring and Application Performance Tuning

## Overview

SysPilot is an intelligent agent that monitors system resources, profiles applications, and dynamically tunes configurations for optimal performance. It uses LLMs to generate custom profiling scripts and make data-driven tuning decisions.

### Key Features

- **Iterative Tuning Engine**: Profile → Tune → Verify → Iterate loop for finding optimal configurations
- **Dynamic Script Generation**: Agent writes custom scripts to gather specific profiling data
- **eBPF Proficiency**: Low-overhead system profiling using eBPF/bpftrace
- **Context Minimization**: Processes large profiling outputs into essential insights (using LLM parsing)
- **Application Tuning**: Detects misconfigurations (NUMA, CPU affinity, etc.) and recommends optimizations
- **Multi-LLM Support**: Works with OpenAI API keys, OpenAI account OAuth, and Anthropic models

## Architecture

```
syspilot/
├── core/           # Agent framework, LLM interface, configuration
├── profiling/      # System collectors (CPU, memory, process, NUMA)
├── skills/         # Dynamic script execution, eBPF templates
│   ├── builtin/    # Strace analyzer, perf analyzer, data processor
│   └── ebpf/       # eBPF skill and templates
├── tuning/         # Config detection, tuning engine, and application tuner
├── utils/          # Logging, process utilities
└── cli.py          # Command-line interface
```

## Installation

```bash
# Basic installation
pip install -e .

# With eBPF support
pip install -e ".[ebpf]"

# Development
pip install -e ".[dev]"
```

### System Requirements

- Python 3.10+
- Linux (for full eBPF support)
- Optional: bpftrace, bcc-tools for eBPF profiling

```bash
# Ubuntu/Debian
sudo apt install bpftrace bpfcc-tools linux-headers-$(uname -r)

# Fedora
sudo dnf install bpftrace bcc
```

### eBPF Permissions (Linux)

If eBPF profiling fails with `Operation not permitted`, run SysPilot with elevated privileges:

```bash
sudo -E python3 -m syspilot.cli tune --command "..." --manual numactl
```

Runtime behavior notes:
- SysPilot tries direct `bpftrace` first, then `sudo -n bpftrace` if needed.
- If privileges are still insufficient, SysPilot records explicit eBPF errors and continues with non-eBPF profiling evidence.

For stricter hosts, check kernel controls:

```bash
cat /proc/sys/kernel/unprivileged_bpf_disabled
cat /proc/sys/kernel/perf_event_paranoid
```

## Quick Start

```bash
# Initialize configuration
syspilot init

# Monitor a process by PID
syspilot monitor --pid 1234

# Monitor by process name
syspilot monitor --name nginx

# Iteratively tune a running process (Attach Mode)
syspilot tune --pid 1234 --manual numactl

# Launch and tune a command (Spawn Mode)
syspilot tune --command "./benchmark -t 4" --manual numactl --parse-output

# Force plain text mode for tune (optional)
syspilot tune --ui-mode plain --command "./benchmark -t 4" --manual numactl

# Enable full LLM debug logging + markdown transcript
syspilot --debug-llm --debug-llm-md /tmp/syspilot_llm_debug.md tune --command "./benchmark -t 4" --manual numactl

# Analyze a specific process
syspilot analyze 1234

# Profile system resources
syspilot profile 1234

# List available skills
syspilot skills
```

### Tune UI Mode (V1)

- `tune` is TUI-first by default (`--ui-mode tui`).
- Plain output remains available via `--ui-mode plain`.
- Fallback behavior is automatic and non-fatal:
  - non-TTY sessions (CI/headless) fall back to plain mode
  - missing/unavailable TUI binary falls back to plain mode
  - TUI disconnect during a run falls back to plain mode and tuning continues

Optional TUI binary override:

```bash
syspilot tune --ui-mode tui --tui-bin /path/to/syspilot-tui --command "./benchmark -t 4"
# or via env var
SYSPILOT_TUI_BIN=/path/to/syspilot-tui syspilot tune --ui-mode tui --command "./benchmark -t 4"
```

## Tuning Workflows

SysPilot supports two primary tuning workflows via the `tune` command:

### 1. Spawn Mode (Recommended for Benchmarks)

Use this when you want SysPilot to manage the entire lifecycle: launch, baseline, tune, and verify.

```bash
syspilot tune \
  --command "/path/to/app --args" \
  --ui-mode tui \
  --workdir /path/to/working/dir \
  --manual numactl \
  --max-iterations 5 \
  --parse-output
```

**Features:**
- Runs initial baseline benchmark automatically
- Runs iterative loop: re-profile current config -> generate new actions -> run candidate(s) -> accept best improvement
- Uses process-tree-aware profiling (manager + heavy child workers) to avoid parent-only blind spots
- Stops by convergence (non-improving iteration patience) or `--max-iterations`
- Compares baseline vs best-found configuration side-by-side
- `--parse-output` uses LLM to extract app-specific metrics (e.g., tokens/sec) from stdout

### Tune UI Overhead Gate

Use the benchmark harness to compare plain vs TUI mode and enforce the p95 overhead budget:

```bash
python3 scripts/bench_tune_ui_overhead.py \
  --runs 30 \
  --warmup 5 \
  --threshold-p95 3.0 \
  --command "python3 tests/fixtures/bench_stable.py" \
  --json-out .sisyphus/evidence/final-overhead.json
```

If TUI cannot run (for example non-TTY sessions or missing TUI binary), the benchmark exits non-zero and records a clear `failure_reason`/`failure_detail` in the JSON artifact.

The harness enforces deterministic, noninteractive benchmarking by setting:

- `SYSPILOT_BENCH_DETERMINISTIC=1` (skip agent tuning-analysis variance)
- `SYSPILOT_TUI_WAIT_FOR_USER=0` (auto-close sidecar at run end)

## Debugging LLM Conversations

Use debug mode to inspect every LLM request/response and tool-calling turn.

```bash
# Console JSON logs
syspilot --debug-llm tune --command "./benchmark -t 4" --manual numactl

# Human-readable markdown transcript
syspilot --debug-llm-md /tmp/syspilot_llm_debug.md tune --command "./benchmark -t 4" --manual numactl
```

The markdown transcript includes:
- per-exchange request metadata (model/tools/kwargs)
- new messages sent to the model
- full model response, tool calls, and token usage

### 2. Attach Mode (Recommended for Services)

Use this for long-running services (databases, web servers) that are already running.

```bash
syspilot tune --pid 12345 --manual postgresql --dry-run
```

**Features:**
- Connects to existing process
- Profiles live metrics (CPU, memory, NUMA placement)
- Recommends runtime changes (sysctl, prlimit) or restart configurations
- `--dry-run` allows safe inspection of recommended actions

## Demos

- **Interactive Visualization**: Check out `examples/llama_numa_demo.html` to see how SysPilot optimized llama.cpp.
- **Run it yourself**: Execute `python examples/llama_numa_testbench.py` to run the end-to-end llama.cpp tuning benchmark on your machine.

## Configuration

Create `syspilot.yaml`:

```yaml
llm:
  provider: openai  # or openai_auth / anthropic
  model: gpt-4o
  temperature: 0.1
  debug_conversation: false
  # optional: write markdown transcript when debug_conversation=true
  # debug_markdown_file: ~/.syspilot/logs/llm_debug.md

monitoring:
  interval_seconds: 5.0
  cpu_threshold_percent: 80.0
  memory_threshold_percent: 85.0
  enable_ebpf: true

tuning:
  auto_tune: false
  dry_run: true
  backup_configs: true
```

Set your API key:

```bash
export OPENAI_API_KEY="your-key"
# or
export ANTHROPIC_API_KEY="your-key"
```

### OpenAI OAuth (no OpenCode dependency)

If you want to use ChatGPT account OAuth (Free/Plus/Pro) instead of OpenAI API keys:

```bash
syspilot auth login
```

This is implemented directly in SysPilot and does not require installing OpenCode.
If automatic browser launch is unavailable (for example on headless SSH sessions), SysPilot now falls back to manual URL-paste flow immediately instead of waiting for callback timeout.

Then configure SysPilot with provider `openai_auth`:

```yaml
llm:
  provider: openai_auth
  model: gpt-5.3-codex  # supported: gpt-5.2-codex, gpt-5.3-codex
  # Optional override, default is ~/.syspilot/auth/openai.json
  openai_auth_file: ~/.syspilot/auth/openai.json
  # Optional override, default is https://chatgpt.com/backend-api/codex
  # base_url: https://chatgpt.com/backend-api/codex
```

## Usage Examples

### Programmatic Usage

```python
import asyncio
from syspilot import SysPilotAgent, SysPilotConfig
from syspilot.profiling import ProfilingManager

async def main():
    config = SysPilotConfig(target_pid=1234)
    agent = SysPilotAgent(config)
    
    # Analyze target application
    result = await agent.analyze_target(pid=1234)
    print(result)

asyncio.run(main())
```

### Using eBPF Skills

```python
from syspilot.skills.ebpf import EBPFSkill, EBPFTemplates

skill = EBPFSkill()

# Check availability
result = await skill.execute(action="check_availability")

# Run syscall profiling
result = await skill.execute(
    action="run_template",
    template_name="syscall_count",
    pid=1234,
    duration=10,
)
```

### Dynamic Script Generation

The agent can generate custom scripts to process profiling data:

```python
from syspilot.skills.builtin import DynamicScriptSkill

skill = DynamicScriptSkill()

# Agent-generated script to process strace output
script = '''
import re
from collections import Counter
# ... processing logic
'''

result = await skill.execute(
    script_type="python",
    script=script,
)
```

## Available eBPF Templates

| Template | Description |
|----------|-------------|
| `syscall_count` | Count syscalls by type |
| `cpu_profile` | CPU profiling with stack traces |
| `memory_alloc` | Track memory allocations |
| `io_latency` | I/O latency histogram |
| `file_opens` | Track file open operations |
| `network_connections` | Track network connections |
| `context_switches` | Track context switches |
| `page_faults` | Track page faults |
| `numa_page_migrations` | Track NUMA page migrations |
| `cache_misses` | Track hardware cache misses |

## How It Works

1. **Monitoring**: Agent continuously monitors system resources
2. **Detection**: Identifies when target application underutilizes resources
3. **Profiling**: Generates and executes profiling scripts (including eBPF)
4. **Summarization**: Processes large outputs into minimal context
5. **Analysis**: LLM analyzes summarized data for bottlenecks
6. **Tuning**: Recommends or applies configuration changes
7. **Verification**: Tuning Engine runs iterative loops to verify improvements

## Development

```bash
# Run tests
pytest

# Type checking
mypy syspilot

# Linting
ruff check syspilot

# Format
black syspilot
```

## License

MIT
