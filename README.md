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
sudo -E python3 -m syspilot.cli tune --command "..." --contract numactl
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

# Launch and tune a command (Spawn Mode)
syspilot tune --command "./benchmark -t 4" --contract numactl

# Optimize on a workload metric extracted from stdout instead of orchestration wall time
syspilot tune --command "./benchmark -t 4" --contract numactl --primary-objective "throughput"

# Add application-specific tuning hints alongside library contracts
syspilot tune --command "./benchmark -t 4" --contract numactl --manual "parent process forks worker processes after warmup"

# Keep selected controls unchanged while tuning
syspilot tune --command "./benchmark -t 4" --contract numactl --exclude-config OMP_NUM_THREADS

# Optional: constrain spawn-mode runs with a MemoryHigh scope
syspilot tune --command "./benchmark -t 4" --contract numactl --memory-high 1536M

# Run quick screening mode with periodic contract-signal sampling
syspilot tune --command "./benchmark -t 4" --contract numactl --quick-mode

# Force plain text mode for tune (optional)
syspilot tune --ui-mode plain --command "./benchmark -t 4" --contract numactl

# Enable full LLM debug logging + markdown transcript
syspilot --debug-llm --debug-llm-md /tmp/syspilot_llm_debug.md tune --command "./benchmark -t 4" --contract numactl

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

## Tuning Workflow

SysPilot now supports one tuning workflow via the `tune` command:

### Spawn Mode

Use this when you want SysPilot to manage the entire lifecycle: launch, baseline, tune, and verify.

```bash
syspilot tune \
  --command "/path/to/app --args" \
  --ui-mode tui \
  --workdir /path/to/working/dir \
  --contract numactl \
  --max-iterations 5
```

**Features:**
- Runs initial baseline benchmark automatically
- Runs iterative loop: re-profile current config -> generate new actions -> run candidate(s) -> accept best improvement
- `--quick-mode` switches to a fast-screening workflow: one full baseline run with periodic contract-signal sampling, offline abnormal-window detection, short candidate screens that stop at the matched baseline window budget, then capped full validation of the best short screens
- Uses process-tree-aware profiling (manager + heavy child workers) to avoid parent-only blind spots
- Stops by convergence (non-improving iteration patience) or `--max-iterations`
- Compares baseline vs best-found configuration side-by-side
- Final summaries identify the round and per-round trial that found the reported best configuration, plus cumulative candidate trial runs and full-process round/trial/profiled-run totals
- `--contract <name[,name...]>` loads library contract context for spawn-mode tune. Repeat the option or use comma-separated values when multiple libraries are involved.
- When multiple contracts are present, SysPilot synthesizes contract-guided hypotheses and uses them to steer profiling and candidate selection.
- `--manual <hint[,hint...]>` adds application-specific natural-language hints directly to the tuning prompt. Manual hints never generate LPCL hypotheses.
- `--contract` and `--manual` are orthogonal and can be used together in the same tune run.
- `--exclude-config <name[,name...]>` marks environment variables, command-line parameters, or config controls as immutable for the tuning prompt and candidate validation, so SysPilot keeps controls such as `BENCHMARK_BLOSC_CNAME` unchanged even if the LLM recommends modifying them.
- When `--primary-objective` is set, SysPilot uses the LLM to extract workload outcome metrics (for example elapsed time, throughput, latency, or bandwidth) from stdout and rank candidates on the resolved metric first.
- `--primary-objective <query>` is free-form user intent, so phrases like `"wall time"`, `"throughput"`, or even near-miss strings like `"ELAPSED SED"` can resolve onto extracted numeric metrics.
- `--primary-objective-mode {minimize,maximize}` remains available when the resolved metric is real but its optimization direction is not obvious from the metric name.
- When a primary objective is configured, orchestration `wall_time` becomes a secondary tie-breaker instead of the main search objective, so wrapper/setup noise does not silently override a better workload result.
- For JSON-like benchmark output, SysPilot first scans the full captured output for exact numeric objective keys such as `throughput`, `samples_per_sec`, or `steps_per_sec` before using LLM extraction, so long structured summaries do not lose the target metric just because it appears after the prompt slice.
- When a primary objective is configured, the final spawn-mode comparison output now reports the baseline-relative objective result as a ratio like `1.11x (better)` or `0.83x (slower)`, while iterative candidate-search lines stay percent-based (`vs current` / `vs previous`).
- Quick mode supports `--primary-objective` and `--primary-objective-mode` for baseline parsing, promoted full validation, and final winner selection, while still keeping short-run screening strictly signal-based.
- `--memory-high <value>` optionally launches the spawn-mode command through `systemd-run --user --scope -p MemoryHigh=...` so baseline and candidate runs stay inside the same user-scope memory-pressure envelope
- Under `--memory-high`, quick mode can now screen on periodic `cgroup.memory.events.high` samples from that same cgroup envelope instead of treating the signal as unavailable until final post-run evidence.
- If `--command` begins with `env KEY=VALUE ...`, SysPilot now treats those assignments as tunable baseline environment overrides instead of opaque command tokens, so spawn-mode env tuning can adjust the user-specified starting values across candidate trials.
- Candidate command rewrites that include inline `env KEY=VALUE ...` are normalized the same way, so the tuning engine can safely try env-prefixed command variants without rejecting them as different executables.
- Quick-mode artifacts are written under `artifacts/quick_tune_runs/<run_id>/` and include structured baseline/full/short run metadata, per-run raw `run_artifact.json` files, saved stdout/stderr, flattened per-signal sample rows, ranked abnormal windows, promotion decisions, and a final `quick_report.json`.
- Quick mode now filters duplicate semantic signals before ranking and excludes objective-only `role=outcome` metrics such as benchmark elapsed time from the profiling-first quick-mode signal set.

### Quick Mode Workflow

Use quick mode when you have contract signals that are directionally meaningful early in the run and you want to screen more candidates before paying for full validation.

For implementation and maintenance details, see [docs/QUICK_MODE.md](docs/QUICK_MODE.md).

```bash
syspilot tune \
  --command "/path/to/app --args" \
  --contract numactl \
  --quick-mode \
  --max-iterations 6
```

Quick mode currently:

- keeps the default tune workflow unchanged unless `--quick-mode` is explicitly enabled
- runs one full sampled baseline, ranks abnormal windows across useful contract signals offline, and checks them under internal short/full run budgets
- screens candidates only until the exact ranked baseline window end offset for the currently ranked window, so short runs compare the same relative sampled slice
- ranks short runs only on target-signal directional improvement in that matched window
- ranks baseline abnormal windows with a direction-aware burst preference so onset windows beat later plateau-only windows when they describe the same anomaly
- interleaves ranked windows across distinct signals before revisiting a second window for the same signal, so one strong probe cannot consume the early screening budget by itself
- promotes only a capped fraction of short-screen winners to full validation within each checked signal/window group
- uses `--primary-objective` for full validation and final winner selection when configured, otherwise falls back to wall time
- keeps evaluating ranked windows within the configured short/full run budgets instead of stopping at the first validated improvement
- emits distinct quick baseline, short screen, and full validation phases for plain output and event consumers
- suppresses known wrapper footer summaries like `RUN_LABEL=` / `RESULT_JSON=` from the live plain-output stream only for short screens, while still preserving the full captured stdout/stderr inside quick-mode artifacts
- exposes quick-mode budgets and thresholds via `--quick-*` flags when you need to retune sampling cadence, screening budgets, or promotion caps

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
syspilot --debug-llm tune --command "./benchmark -t 4" --contract numactl

# Human-readable markdown transcript
syspilot --debug-llm-md /tmp/syspilot_llm_debug.md tune --command "./benchmark -t 4" --contract numactl
```

The markdown transcript includes:
- per-exchange request metadata (model/tools/kwargs)
- new messages sent to the model
- full model response, tool calls, and token usage

## Demos

- **Interactive Visualization**: Check out `examples/llama_numa_demo.html` to see how SysPilot optimized llama.cpp.
- **Run it yourself**: Execute `python examples/llama_numa_testbench.py` to run the end-to-end llama.cpp tuning benchmark on your machine.

## Configuration

LLM settings are loaded from a single file, `config/llm_config.json`, by default.

Use OpenAI API mode:

```json
{
  "provider": "openai",
  "base_url": "https://your-endpoint/v1",
  "api_key": "sk-...",
  "model": "gpt-5.4",
  "temperature": 0.1,
  "max_tokens": 272000,
  "timeout_seconds": 120,
  "max_retries": 1
}
```

To switch to ChatGPT OAuth mode, edit the same file:

```json
{
  "provider": "openai_auth",
  "model": "gpt-5.3-codex",
  "openai_auth_file": "~/.syspilot/auth/openai.json",
  "base_url": "https://chatgpt.com/backend-api/codex",
  "temperature": 0.1,
  "max_tokens": 4096,
  "timeout_seconds": 120,
  "max_retries": 1
}
```

Create `syspilot.yaml` for the rest of SysPilot settings:

```yaml
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

### OpenAI OAuth (no OpenCode dependency)

If you want to use ChatGPT account OAuth (Free/Plus/Pro) instead of OpenAI API keys:

```bash
syspilot auth login
```

This is implemented directly in SysPilot and does not require installing OpenCode.
`syspilot auth login` uses manual URL-paste flow by default.
Use `syspilot auth login --auto` if you want browser callback mode.
If automatic browser launch is unavailable (for example on headless SSH sessions), SysPilot falls back to manual URL-paste flow immediately instead of waiting for callback timeout.
Manual paste mode accepts callback data from both query strings and URL fragments (for example `.../callback?code=...` and `.../callback#code=...`).

Then switch `config/llm_config.json` to `provider: openai_auth` and set `openai_auth_file` if you do not want the default path.

## Usage Examples

### Programmatic Usage

```python
import asyncio
from syspilot import SysPilotAgent, SysPilotConfig
from syspilot.profiling import ProfilingManager

async def main():
    config = SysPilotConfig()
    agent = SysPilotAgent(config)
    manager = ProfilingManager()
    
    # Collect a profiling snapshot that can inform spawn-mode tuning
    results = await manager.collect_all()
    print(manager.summarize_results(results))

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
