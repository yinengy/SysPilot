# SysPilot

AI Agent for System Monitoring and Application Performance Tuning

## Overview

SysPilot is an intelligent agent that launches Linux workloads, collects structured runtime evidence, and tunes runtime/configuration controls for better performance. The harness gathers evidence with built-in collectors and LPCL contract-declared virtual probes, while the LLM reads that supplied context and recommends the next candidate actions as JSON.

### Key Features

- **Iterative Tuning Engine**: Profile → Tune → Verify → Iterate loop for finding optimal configurations
- **Structured Evidence Collection**: Built-in CPU, memory, process, NUMA, stdout/stderr, and process-tree evidence feeds tune decisions
- **LPCL Virtual Probes**: Contract-declared `virtual_probe` signals map to toolbox scripts under `toolbox/probes/...`
- **Focused LLM Prompting**: Gives the LLM compact runtime evidence, contract findings, manual context, prior attempts, immutable controls, and an explicit JSON action schema
- **Application Tuning**: Detects misconfigurations (NUMA, CPU affinity, etc.) and recommends optimizations
- **Multi-LLM Support**: Works with OpenAI API keys, OpenAI account OAuth, and Anthropic models

## Architecture

```
syspilot/
├── core/           # Agent framework, LLM interface, configuration
├── profiling/      # System collectors (CPU, memory, process, NUMA)
├── skills/         # Bounded helper execution and optional standalone skills
│   ├── builtin/    # Strace analyzer, perf analyzer, data processor
│   └── ebpf/       # Standalone eBPF skill and templates (not loaded by tune)
├── tuning/         # Config detection, tuning engine, and application tuner
├── utils/          # Logging, process utilities
└── cli.py          # Command-line interface
```

## Installation

```bash
# Basic installation
pip install -e .

# Development
pip install -e ".[dev]"
```

### System Requirements

- Python 3.10+
- Linux

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

# Skip profiling and ask the LLM to reason from raw stdout/stderr only
syspilot tune --command "./benchmark -t 4" --no-profile

# Optional: constrain spawn-mode runs with a MemoryHigh scope
syspilot tune --command "./benchmark -t 4" --contract numactl --memory-high 1536M

# Cap candidate search cost at 8 attempted candidate trials, baseline run not included
syspilot tune --command "./benchmark -t 4" --contract numactl --fixed-budget 8

# Continue until the optimization target reaches a fixed goal, or candidate-trial cap is exhausted
syspilot tune --command "./benchmark -t 4" --contract numactl --primary-objective "throughput" --fixed-goal 1231.1 --max-iterations 8

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
  --max-iterations 5 \
  --fixed-budget 12
```

**Features:**
- Runs initial baseline benchmark automatically
- By default, collects profiling evidence for baseline and candidate runs. Use `--no-profile` when you want SysPilot to skip profiler/probe collection and ask the LLM to reason from raw program stdout/stderr instead.
- Runs iterative loop: re-profile current config -> generate new actions -> run candidate(s) -> accept best improvement
- `--quick-mode` switches to a fast-screening workflow: one full baseline run with periodic contract-signal sampling, offline abnormal-window detection, short candidate screens that stop at the matched baseline window budget, then capped full validation of the best short screens
- Uses process-tree-aware profiling (manager + heavy child workers) to avoid parent-only blind spots
- By default, stops by convergence (non-improving iteration patience) or `--max-iterations`; `--max-iterations` defaults to 20 materialized candidate trial attempts and does not count the baseline run.
- `--fixed-budget N` adds an authoritative cap on candidate attempts. It limits search cost, not outcome quality, and does not guarantee the best possible configuration. When fixed budget is set, `--max-iterations` and its default are ignored.
- `--fixed-goal VALUE` keeps tuning until the selected optimization target reaches the requested value, or until the active candidate-trial cap is exhausted. Maximize objectives stop at `>= VALUE`; minimize objectives such as wall time stop at `<= VALUE`.
- Baseline profiling does not consume fixed budget. Concrete candidate attempts do, including duplicate, invalid, failed, or timed-out attempts once they are materialized.
- In fixed-budget mode, SysPilot keeps searching past normal convergence or no-action stops while budget remains. `--fixed-budget` is the active candidate-attempt cap even when `--max-iterations` is also provided.
- No-action or malformed recommendation replies do not spend trial budget. They use a bounded re-prompt allowance that asks for finer-grained or untouched options, and SysPilot stops with an explicit no-valid-action reason if that allowance is exhausted.
- Compares baseline vs best-found configuration side-by-side
- Final summaries identify the round and per-round trial that found the reported best configuration, plus cumulative candidate trial runs and full-process round/trial/profiled-run totals
- Final summaries also report total SysPilot wall time and config-find wall time, so operators can distinguish full tuning cost from the time needed to discover the reported best configuration.
- `--contract <name[,name...]>` loads library contract context for spawn-mode tune. Repeat the option or use comma-separated values when multiple libraries are involved.
- Contract names are resolved through the contract store, so punctuation aliases such as `pytorch-cudnn` and `pytorch_cudnn` resolve to the same contract when the bundle exists.
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
  --max-iterations 6 \
  --fixed-budget 12
```

Quick mode currently:

- keeps the default tune workflow unchanged unless `--quick-mode` is explicitly enabled
- requires profiling/sampling; `--no-profile` is intentionally rejected with `--quick-mode`
- runs one full sampled baseline, ranks abnormal windows across useful contract signals offline, and checks them under internal short/full run budgets
- `--fixed-budget N`, when set, caps total candidate attempts across quick-mode short screens and promoted full validations. It is a search-cost cap, not a promise that quick mode found the best available configuration.
- `--fixed-goal VALUE`, when set, stops quick mode once the best validated target reaches the goal, or reports the current best when the active candidate-trial cap is exhausted first.
- Quick mode keeps its own internal `--quick-max-total-short-runs` and `--quick-max-total-full-runs` caps. Those shape short-screen and promoted-full-validation counts, while `--fixed-budget` is a separate top-level cap on total candidate attempts.
- screens candidates only until the exact ranked baseline window end offset for the currently ranked window, so short runs compare the same relative sampled slice
- ranks short runs only on target-signal directional improvement in that matched window
- ranks baseline abnormal windows with a direction-aware burst preference so onset windows beat later plateau-only windows when they describe the same anomaly
- interleaves ranked windows across distinct signals before revisiting a second window for the same signal, so one strong probe cannot consume the early screening budget by itself
- promotes only a capped fraction of short-screen winners to full validation within each checked signal/window group
- uses `--primary-objective` for full validation and final winner selection when configured, otherwise falls back to wall time
- keeps evaluating ranked windows within the configured short/full run budgets instead of stopping at the first validated improvement
- under `--fixed-budget`, bounded no-action re-prompts can ask for finer-grained or untouched options without consuming trial budget, but exhausted re-prompts still stop the run explicitly
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
  "metric_extraction_model": "gpt-5.3-codex",
  "temperature": 0.1,
  "max_tokens": 272000,
  "timeout_seconds": 120,
  "max_retries": 1
}
```

`model` is used for tuning analysis and candidate recommendations. `metric_extraction_model` is optional and only changes the model used for primary-objective metric extraction from workload stdout/stderr; it reuses the same provider, `base_url`, credentials, timeout, retry, and debug settings.

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
  enable_ebpf: false

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

### Standalone eBPF Skills

The eBPF skill code remains available for direct programmatic use and tests, but it is not loaded by the default `syspilot tune` path. Current tune evidence relies on built-in collectors plus toolbox-backed LPCL `virtual_probe` signals.

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

### Structured Evidence Helpers

SysPilot's primary profiling path is structured collection plus LPCL virtual probes. Bounded helper scripts still exist for deterministic parsing or postprocessing tasks, but LLM-authored custom profiling scripts are not the primary tuning mechanism:

```python
from syspilot.skills.builtin import DynamicScriptSkill

skill = DynamicScriptSkill()

# Bounded helper script to process a known artifact format
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

## Standalone eBPF Templates

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
3. **Profiling**: Collects built-in runtime evidence and LPCL toolbox-backed virtual-probe signals
4. **Validation**: Turns contract hypotheses and collected evidence into structured findings
5. **Analysis**: LLM analyzes compact evidence, loaded contracts/manuals, objective metrics, and prior attempts to recommend JSON candidate actions
6. **Tuning**: Harness materializes and validates recommended runtime/configuration changes
7. **Verification**: Tuning Engine executes candidate loops and compares measured outcomes

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
