# Manual System - Agent-Readable Tuning Documentation

## Overview

The manual system provides agent-readable documentation for tuning libraries, tools, and frameworks. Manuals are loaded into the agent's context, enabling it to make informed tuning decisions based on expert knowledge.

## Architecture

### Components

1. **Manual Files**: Markdown files in `manual/` directory
2. **Agent Integration**: `SysPilotAgent.load_manual()` method
3. **Context Storage**: `AgentContext.loaded_manuals` dict
4. **System Prompt**: Manuals injected into agent system prompt
5. **CLI Interface**: `--manual` flag for commands

### Manual Format

Each manual (`.md` file) should contain:

1. **TARGET BRIEF**: What the tool/library is, detection methods
2. **CRITICAL SYSTEM RESOURCES**: Key resources and performance impact
3. **TUNING CONFIGURATION**: How to configure via environment, CLI, code
4. **PERFORMANCE DIAGNOSIS**: Symptoms and detection scripts
5. **COMMON TUNING SCENARIOS**: Specific use cases with solutions
6. **TUNING DECISION TREE**: Logic for automated tuning selection
7. **DETECTION HEURISTICS**: Step-by-step agent detection logic

## Usage

### CLI Usage

```bash
# List available manuals
syspilot manuals

# Monitor with manual loaded
syspilot monitor --pid 1234 --manual numactl --manual pytorch

# Analyze with manual loaded
syspilot analyze 1234 --manual numactl

# Spawn-mode tuning with working directory + manual policy
syspilot tune --command "./parallel_vlc ../config/transformer.json 0-11 12-23" --workdir /home/yyan/vlc/examples/Torch/build --manual vlcs --parse-output
```

### Programmatic Usage

```python
from syspilot.core.agent import SysPilotAgent
from syspilot.core.config import SysPilotConfig

# Create agent
agent = SysPilotAgent(SysPilotConfig())

# List available manuals
print(agent.list_available_manuals())

# Load single manual
agent.load_manual("numactl")

# Load multiple manuals
agent.load_manuals(["numactl", "pytorch"])

# Check loaded manuals
print(agent.context.loaded_manuals.keys())
```

### Example: Manual Loading Demo

```bash
python3 examples/manual_loading_demo.py
```

## Available Manuals

### numactl
NUMA control and memory affinity tuning for multi-socket systems.
- Target detection: `numactl` command, libnuma calls
- Key resources: NUMA nodes, CPU affinity, memory binding
- Tuning: CPU/memory binding, interleaving policies

### PyTorch
Deep learning framework optimization for CPU/GPU training.
- Target detection: `torch` imports, `libtorch.so`, CUDA libraries
- Key resources: GPU devices, CPU threads, VRAM, DataLoader workers
- Tuning: OMP_NUM_THREADS, batch size, mixed precision, multi-GPU

### VLCs
Virtual Library Contexts CPU-affinity tuning for manager/worker process workloads.
- Target detection: `parallel_vlc` command and core-range arguments (e.g., `0-11 12-23`)
- Key resources: disjoint core partitions, worker-process hotspots, NUMA locality
- Tuning: iterative core-range rebalancing with process-tree-aware profiling evidence

## Creating New Manuals

### Manual Template

Create `manual/<tool-name>.md`:

```markdown
# Tool Name - Brief Description

## TARGET BRIEF
- What is it?
- How to detect usage?

## CRITICAL SYSTEM RESOURCES
### 1. Resource Name
**Resource**: Description
**Check Command**: `command here`
**Performance Impact**: CRITICAL/HIGH/MEDIUM/LOW
**Key Metrics**: List of metrics

## TUNING CONFIGURATION
### Environment Variables
```bash
export VAR_NAME=value
```

### Command Line
```bash
tool --option value
```

## PERFORMANCE DIAGNOSIS
**Symptom**: Description
```bash
# Detection command
```
**Fix**: Solution

## TUNING DECISION TREE
```
Is condition true?
├─ YES → Action A
└─ NO  → Action B
```

## DETECTION HEURISTICS FOR AGENT
**Step 1**: Check X
**Step 2**: If Y, then Z
**Step 3**: Recommend configuration
```

### Best Practices

1. **Be Specific**: Include exact commands with expected output
2. **Be Actionable**: Provide copy-paste ready configurations
3. **Be Quantitative**: Specify thresholds (e.g., "> 80% GPU util")
4. **Include Examples**: Real-world scenarios with solutions
5. **Agent-Focused**: Write for automated consumption, not humans
6. **Decision Trees**: Provide clear logic for automated tuning

## Implementation Details

### How Manuals Are Loaded

1. User specifies `--manual <name>` or calls `agent.load_manual()`
2. Agent reads `manual/<name>.md`
3. Content stored in `agent.context.loaded_manuals[name]`
4. System prompt rebuilt with manual sections appended
5. Agent references manual content in responses

### System Prompt Structure

```
[Base System Prompt]

## LOADED MANUALS
You have access to the following technical manuals for tuning:

### Manual: numactl
[Full numactl manual content]

### Manual: pytorch
[Full pytorch manual content]
```

### Context Size Management

- Manuals are ~5-15 KB each
- Loaded only when needed (explicit specification)
- Agent uses summaries, not full content, in tool calls
- Future: Auto-discovery will load manuals dynamically

### Spawn-Mode Manual Policy Loop

In spawn mode, manuals are used as policy constraints each iteration:

1. Run current configuration and collect profiling evidence
2. Re-profile process tree (parent + heavy worker children)
3. Ask agent for structured manual-cited actions (`summary`, `manual_citation`, `evidence`, `actions`)
4. Execute bounded candidate search
5. Keep best-so-far config and stop by convergence or max iterations

## Future Enhancements

### Planned Features

1. **Auto-Discovery**: Detect libraries in use and load manuals automatically
   - Scan `/proc/<pid>/maps` for .so files
   - Check Python imports via `lsof`
   - Match against manual library detection patterns

2. **Manual Indexing**: Vector embeddings for RAG-based retrieval
   - Index manual sections
   - Retrieve only relevant sections
   - Reduce context size

3. **Dynamic Manual Updates**: Pull latest tuning guides from repos
   - GitHub-based manual repository
   - Version tracking
   - Community contributions

4. **Manual Validation**: Test detection heuristics
   - Unit tests for detection commands
   - Validation of tuning recommendations
   - Expected improvement benchmarks

## Testing

### Profiling and eBPF Test Strategy

- Keep profiling/eBPF tests deterministic and runnable without root privileges.
- Mock tooling availability checks (`bpftrace`, `bcc`) and executor methods instead of invoking kernel probes in unit tests.
- Cover bpftrace-unavailable paths explicitly (explicit BCC templates and non-BCC failure behavior), plus hard-failure paths (missing PID, invalid script, dangerous script patterns).
- Cover profiling manager edge cases such as unknown collector names, collector failures, and empty result summarization.

```bash
# Focused profiling/eBPF unit tests
pytest tests/test_profiling_manager.py tests/test_ebpf_skill.py
```

```bash
# Test manual loading
python3 -c "
from syspilot.core.agent import SysPilotAgent
from syspilot.core.config import SysPilotConfig
agent = SysPilotAgent(SysPilotConfig())
print('Available:', agent.list_available_manuals())
print('Load numactl:', agent.load_manual('numactl'))
print('Loaded:', list(agent.context.loaded_manuals.keys()))
"

# Run demo
python3 examples/manual_loading_demo.py

# Test CLI
syspilot manuals
```

## Troubleshooting

**Manual not found**:
- Check `manual/<name>.md` exists
- Use `agent.list_available_manuals()` to see available manuals

**Manual not applying**:
- Verify manual is loaded: `agent.context.loaded_manuals`
- Check system prompt includes manual: `agent._message_history[0].content`

**Performance issues with large manuals**:
- Keep manuals focused (< 20 KB recommended)
- Use subsections for different aspects
- Future: Implement RAG-based retrieval
