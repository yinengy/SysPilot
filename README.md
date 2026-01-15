# SysPilot

AI Agent for System Monitoring and Application Performance Tuning

## Overview

SysPilot is an intelligent agent that monitors system resources, profiles applications, and dynamically tunes configurations for optimal performance. It uses LLMs to generate custom profiling scripts and make data-driven tuning decisions.

### Key Features

- **Dynamic Script Generation**: Agent writes custom scripts to gather specific profiling data
- **eBPF Proficiency**: Low-overhead system profiling using eBPF/bpftrace
- **Context Minimization**: Processes large profiling outputs into essential insights
- **Application Tuning**: Detects misconfigurations and recommends optimizations
- **Multi-LLM Support**: Works with OpenAI and Anthropic models

## Architecture

```
syspilot/
├── core/           # Agent framework, LLM interface, configuration
├── profiling/      # System collectors (CPU, memory, process)
├── skills/         # Dynamic script execution, eBPF templates
│   ├── builtin/    # Strace analyzer, perf analyzer
│   └── ebpf/       # eBPF skill and templates
├── tuning/         # Config detection and application tuning
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

## Quick Start

```bash
# Initialize configuration
syspilot init

# Monitor a process by PID
syspilot monitor --pid 1234

# Monitor by process name
syspilot monitor --name nginx

# Analyze a specific process
syspilot analyze 1234

# Profile system resources
syspilot profile 1234

# List available skills
syspilot skills
```

## Configuration

Create `syspilot.yaml`:

```yaml
llm:
  provider: openai  # or anthropic
  model: gpt-4o
  temperature: 0.1

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

## How It Works

1. **Monitoring**: Agent continuously monitors system resources
2. **Detection**: Identifies when target application underutilizes resources
3. **Profiling**: Generates and executes profiling scripts (including eBPF)
4. **Summarization**: Processes large outputs into minimal context
5. **Analysis**: LLM analyzes summarized data for bottlenecks
6. **Tuning**: Recommends or applies configuration changes

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
