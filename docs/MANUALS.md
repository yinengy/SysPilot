# Manual System - Agent-Readable Tuning Documentation

## Overview

The manual system provides agent-readable documentation for tuning libraries, tools, and frameworks. Manuals are loaded into the agent's context, enabling it to make informed tuning decisions based on expert knowledge.

## Structured Manual MVP

SysPilot now supports a JSON-first manual bundle for auto-generated manuals while preserving the existing handwritten markdown path.

## LPCL Contract Pipeline

SysPilot now also supports a YAML-based LPCL contract pipeline that can replace the structured manual-generation path operationally while keeping the manual pipeline code intact for ablation.

### Contract bundle layout

For a generated contract named `openmp`, the runtime-compatible outputs are:

- `contracts/openmp/contract.yaml` — LPCL source of truth

The contract schema now prefers a **typed core** over free-form strings:

- controls use canonical scalar/domain types such as `int`, `bool`, `enum`, `cpu_set`, and `numa_node_set`
- demand bounds use host symbols like `host.visible_cpus` instead of prose such as “visible processors”
- signals use direct `virtual_probe` capability names plus strict `role` / `expected_direction_when_mitigated` enums instead of a free-form tool string or nested probe object
- recipes reference control and signal ids directly through `diagnose`, `config_checks`, `outcomes`, `guardrails`, and labeled `actions`
- composed conflicts use `candidate_control_refs`, `candidate_recipe_refs`, and signal/control refs instead of copied action prose

Natural language is still allowed for fields that genuinely need it, such as explanatory `indicates` text, action summaries, and diagnostic questions.

### Enforced LPCL checking

`synthesize-contract` now enforces a bounded verify/repair loop before any bundle is written:

- generated contract JSON is normalized and checked against the LPCL schema
- semantic lint checks verify refs like `controls.affects`, `recipes.collect`, and composed cross-library refs
- if the checker fails, SysPilot asks the LLM to repair the invalid sections and retries with a small bounded budget
- if the contract still fails after the repair budget is exhausted, generation fails explicitly and no partial contract bundle is written

This means `contracts/<id>/contract.yaml` is now a checked artifact, not just a raw first-pass LLM draft.

Artifact/debug outputs are also written under:

- `artifacts/contracts/`
- `artifacts/composed_contracts/`

### Bundle layout

For a generated manual named `openmp`, the runtime-compatible outputs are:

- `manual/openmp/manual.json` — source of truth for storage, querying, and composition
- `manual/openmp/manual.md` — deterministic rendered view of `manual.json`
- `manual/openmp/manual_index.jsonl` — flattened item index for targeted retrieval
- `manual/openmp.md` — compatibility markdown consumed by the existing manual loader

Artifact/debug outputs are also written under:

- `artifacts/raw_sources/`
- `artifacts/chunks/`
- `artifacts/evidence/`
- `artifacts/manuals/`
- `artifacts/composed/`
- `artifacts/logs/`

### Authoritative source packs

For externally sourced generated manuals, keep a small repo-tracked source pack under `manual_sources/<library>/`.

- Store normalized markdown/text snippets instead of raw crawler dumps.
- Keep `metadata.json` with library name, aliases, known knobs, and the authoritative source URLs used.
- Use the source pack as the explicit `syspilot ingest --docs ... --metadata ...` input so manual generation is reproducible and inspectable.

Current source-pack examples:

- `manual_sources/openmp/`
- `manual_sources/openblas/`
- `manual_sources/numactl/`

Current implementation-specific contract-source examples:

- `contract_sources/onemkl/`
- `contract_sources/openmpi/`

### Runtime behavior

- The existing manual runtime still loads `manual/<name>.md` as raw markdown.
- Spawn-mode tuning now prefers a structured LPCL `build_agent_context(...)` bundle when `contracts/<name>/contract.yaml` exists.
- LPCL contracts now use the stricter `lpcl/v1` ontology: components model internal mechanisms, while shared-resource analysis is restricted to `cpu.compute`, `cpu.placement`, `gpu.compute`, `gpu.memory`, `mem.capacity`, `mem.bandwidth`, `io.bandwidth`, and `io.queue` so composition stays conflict-oriented instead of treating thread pools or worker pools as resources.
- Generated `contract.lpcl` artifacts are now compiled statically from validated LPCL YAML/JSON, not LLM-rendered text, so the DSL output stays deterministic, brace/semicolon exact, and golden-testable across runs.
- LPCL compilation hides `evidence = [...]` lines by default for human-readable review output, and callers must pass the explicit `--include-evidence` flag (or equivalent API option) to restore evidence blocks when needed for debugging or papers.
- Contract bundles therefore consist of canonical `contract.yaml` plus compiled `contract.lpcl`; if you need to recompile manually, use `syspilot compile-lpcl --input contracts/<name>/contract.yaml --output contracts/<name>/contract.lpcl`, and add `--include-evidence` only when you explicitly want provenance blocks in the LPCL view.
- A composed contract like `contracts/openmp_openblas_composed/contract.yaml` can now be passed directly as a runtime knowledge input; SysPilot resolves `selected_contracts` plus hypothesis refs back into concrete controls, signals, recipes, assumptions, and shared-resource facts before building the agent context bundle.
- Composed contracts may now include multiple instances of the same library contract. SysPilot writes deterministic `contract_instances` metadata and uses instance-qualified refs like `openmp_1.*` and `openmp_2.*` so duplicate libraries remain distinguishable at both compose time and runtime query time.
- Spawn-mode tuning now derives `hypothesis_profiles` from composed contract conflicts, enables report-style controls for those profiles, and runs contract-guided validation scripts during the profiling phase so each round can feed structured hypothesis evidence back into the next tuning decision.
- Spawn-mode tune now keeps library contracts and application manuals orthogonal: `--contract` loads contract context, while `--manual` passes free-form application hints directly into the tuning prompt.
- Contract-guided hypothesis synthesis is driven only by loaded contracts, not by application manual hints.
- Manual-file loading remains available through agent APIs and manual-related helper commands, while spawn-mode tune uses `--manual` only for free-form application hints.
- Long-running `extract-evidence` runs now report progress + ETA, checkpoint successful chunk results, and resume from remaining chunks after interruption.
- `synthesize-manual` and `compose-manuals` now report phase progress + ETA and fail fast on bounded LLM timeouts instead of hanging indefinitely.

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

# Spawn-mode tuning with working directory + contract context + app hint
syspilot tune --command "./parallel_vlc ../config/transformer.json 0-11 12-23" --workdir /home/yyan/vlc/examples/Torch/build --contract vlcs --manual "manager process spawns workers after warmup"

# Ingest docs into chunked manual-source artifacts
syspilot ingest --library openmp --docs examples/manual_mvp/openmp.md

# Extract structured evidence with the configured LLM
syspilot extract-evidence --library openmp

# Synthesize a JSON-first manual bundle
syspilot synthesize-manual --library openmp

# Synthesize an LPCL contract bundle from the same evidence
syspilot synthesize-contract --library openmp

# Validate an LPCL contract file or contract ID
syspilot check-contract contracts/openmp/contract.yaml

# Re-render compatibility markdown from manual.json
syspilot render-manual --manual openmp

# Rebuild the flattened retrieval index
syspilot build-index --manual openmp

# Compose multiple manuals into an app-level manual
syspilot compose-manuals --app xgb_train --libraries xgboost --libraries openmp

# Compose multiple LPCL contracts into conflict hypotheses
syspilot compose-contracts --app xgb_train --contracts xgboost --contracts openmp

# Compose two instances of the same library contract
syspilot compose-contracts --app openmp_openmp --contracts openmp --contracts openmp --contract-root contracts

# Query targeted manual slices
syspilot query --manual openmp --section observability_map --keywords threads --keywords oversubscription

# Build a focused structured context bundle for the autotuner
syspilot build-agent-context --manuals xgboost --manuals openmp --goal maximize_throughput --symptom throughput_plateau

# Build context directly from a composed LPCL contract
syspilot build-agent-context --manuals openmp_openblas_composed --goal improve_throughput --contract-root contracts

# Build context from a duplicate-instance composed contract
syspilot build-agent-context --manuals openmp_openmp_composed --goal improve_throughput --contract-root contracts

# Generate new OpenMP and OpenBLAS library contracts from authoritative local source packs
syspilot ingest --library openmp --docs manual_sources/openmp --metadata manual_sources/openmp/metadata.json
syspilot extract-evidence --library openmp
syspilot synthesize-manual --library openmp
syspilot synthesize-contract --library openmp --contract-root contracts

syspilot ingest --library openblas --docs manual_sources/openblas --metadata manual_sources/openblas/metadata.json
syspilot extract-evidence --library openblas
syspilot synthesize-manual --library openblas
syspilot synthesize-contract --library openblas --contract-root contracts

# Compose the library contracts for an OpenMP + OpenBLAS workload
syspilot compose-contracts --app openmp_openblas --contracts openmp --contracts openblas --contract-root contracts

# Compose two OpenMP instances when the application loads parallel OpenMP-backed objects
syspilot compose-contracts --app openmp_openmp --contracts openmp --contracts openmp --contract-root contracts
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

### Auto-generation pipeline

The MVP auto-generation path is intentionally simple and inspectable:

1. ingest docs/code hints into normalized source files and chunks
2. extract `evidence.jsonl` with LLM pass 1
3. merge/deduplicate evidence deterministically
4. synthesize `manual.json` with LLM pass 2
5. verify/repair once with schema checks + LLM pass 3
6. render `manual.md` deterministically from `manual.json`
7. flatten `manual_index.jsonl` for targeted retrieval

`manual.json` is the source of truth. Markdown and JSONL are derived artifacts.

### Query API

The structured query layer works over `manual.json` and `manual_index.jsonl` and exposes deterministic methods for:

- `get_manual(manual_id)`
- `list_manuals()`
- `query_section(manual_id, section, keywords=None)`
- `query_by_symptom(manual_ids, symptom)`
- `query_by_goal(manual_ids, goal)`
- `get_observability(manual_ids, keywords=None, symptom=None)`
- `get_interventions(manual_ids, keywords=None, symptom=None)`
- `get_conflicts(manual_ids)`
- `build_agent_context(manual_ids, goal=None, symptom=None, max_items_per_section=5)`

This keeps retrieval explicit and lightweight without adding embeddings or a graph database.

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

Programmatic/manual-loading flows still use manual-file loading:

1. User specifies `--manual <name>` or calls `agent.load_manual()`
2. Agent reads `manual/<name>.md`
3. Content stored in `agent.context.loaded_manuals[name]`
4. System prompt rebuilt with manual sections appended
5. Agent references manual content in responses

Spawn-mode tune is separate and is the only CLI tuning workflow:

1. User specifies `--contract <name>` for library contracts and/or `--manual <hint>` for application hints
2. Contract names are resolved into structured contract context bundles
3. Application manual hints are kept as free-form natural language and appended directly to the tuning prompt
4. Only contracts drive startup hypothesis synthesis and contract-guided profiling
5. UI/event output reports contracts and manual context separately: named manuals loaded
   from disk count as loaded manual context, while unresolved `--manual` values count as
   free-form hints

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

### Spawn-Mode Contract + Manual-Hint Policy Loop

In spawn mode, contracts provide library policy and hypotheses, while manual hints provide application context:

1. Run current configuration and collect profiling evidence
2. Re-profile process tree (parent + heavy worker children), optionally with contract-guided hypothesis validation
3. Ask agent for structured actions using profiling evidence, generated contract hypotheses, contract-validation findings, and application manual hints (`summary`, `manual_citation`, `evidence`, `actions`)
4. Execute bounded candidate search
5. Keep best-so-far config and stop by convergence or the `--max-iterations` materialized candidate-trial cap by default, or by `--fixed-budget N` when an explicit candidate-attempt cap is active. Fixed budget limits search cost only and does not promise an optimal result.

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
