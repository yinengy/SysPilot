# Skill: Map LibContentBench `BENCHMARK_*` Env Overrides into LPCL Library Contracts

Use this playbook when a LibContentBench benchmark exposes real library controls through benchmark-local environment variables, and SysPilot needs LPCL recommendations to manipulate those env vars directly.

## When to use

- The benchmark worker accepts env overrides such as `BENCHMARK_*`.
- Those env vars are the actual fixed-workload tuning surface for one or more libraries.
- Existing LPCL contracts describe native API/config knobs, but SysPilot candidate actions must use the env override names.
- You want to update the original library contracts separately, not create a composed or wrapper contract.

## Core rule learned from `arrow-dask-duckdb-memspill`

`BENCHMARK_*` controls are valid in normal `type: library` LPCL contracts when they are the real action surface exposed to SysPilot. Put each env var into the original library contract that owns the behavior:

- Dask controls go in `contracts/dask/contract.yaml`.
- DuckDB controls go in `contracts/duckdb/contract.yaml`.
- PyArrow controls go in `contracts/pyarrow/contract.yaml`.
- Do not create a separate wrapper/composed contract just to hold the env vars.

## Workflow

### 1. Discover the benchmark env surface

Inspect the benchmark runner and direct-command metadata first. For `arrow-dask-duckdb-memspill`, the authoritative files were:

- `experiments/LibContentBench/arrow-dask-duckdb-memspill/run_arrow_dask_duckdb_case.py`
- `experiments/LibContentBench/libcontentbench/direct_commands.py`
- the benchmark-local `README.md` and matrix YAML for documented/default values

Build a table with:

| Env var | Library owner | Runtime effect | Type/range | Where applied |
|---|---|---|---|---|
| `BENCHMARK_N_WORKERS` | Dask | `LocalCluster(n_workers=...)` | int >= 1 | runner cluster creation |
| `BENCHMARK_THREADS_PER_WORKER` | Dask | `LocalCluster(threads_per_worker=...)` | int >= 1 | runner cluster creation |
| `BENCHMARK_WORKER_MEMORY_LIMIT` | Dask | `LocalCluster(memory_limit=...)` | non-empty string | runner cluster creation |
| `BENCHMARK_ARROW_CPU_COUNT` | PyArrow | `pyarrow.set_cpu_count(...)` | int >= 1 | runner setup |
| `BENCHMARK_ARROW_IO_THREADS` | PyArrow | `pyarrow.set_io_thread_count(...)` | int >= 1 | runner setup |
| `BENCHMARK_DUCKDB_MEMORY_LIMIT` | DuckDB | `SET memory_limit='...'` | non-empty string | DuckDB connection setup |
| `BENCHMARK_DUCKDB_THREADS` | DuckDB | `SET threads=...` | int >= 1 | DuckDB connection setup |
| `BENCHMARK_PATH_LAYOUT` | Dask/DuckDB shared scratch | selects Dask `local_directory` and DuckDB `temp_directory` | `shared` or `isolated` | path construction |

Keep workload-shape knobs separate. If a field changes dataset size, task count, row count, repetitions, model size, etc., it is not a fixed-workload library control unless the benchmark explicitly treats it as a library/runtime knob.

### 2. Update the original library contracts

Edit `contract.yaml`, not `contract.lpcl`. The compiled `.lpcl` file is generated.

For each existing LPCL control, change these fields:

```yaml
name: BENCHMARK_...
kind: env
effect:
  kind: set
  value: <short accepted-value description>
```

Keep the original semantic `target` when possible, because the target describes the library resource being controlled. Example mappings from this task:

```yaml
# contracts/dask/contract.yaml
- id: ctl_n_workers
  name: BENCHMARK_N_WORKERS
  kind: env
  type: int
  target: executor.workers

- id: ctl_threads_per_worker
  name: BENCHMARK_THREADS_PER_WORKER
  kind: env
  type: int
  target: executor.threads_per_worker

- id: ctl_memory_limit
  name: BENCHMARK_WORKER_MEMORY_LIMIT
  kind: env
  type: string
  target: executor.memory_limit

- id: ctl_local_directory
  name: BENCHMARK_PATH_LAYOUT
  kind: env
  type: string
  target: executor.local_directory
```

```yaml
# contracts/duckdb/contract.yaml
- id: ctrl_threads
  name: BENCHMARK_DUCKDB_THREADS
  kind: env
  target: duckdb_executor.workers

- id: ctrl_memory_limit
  name: BENCHMARK_DUCKDB_MEMORY_LIMIT
  kind: env
  target: duckdb_allocator.memory_limit

- id: ctrl_temp_directory
  name: BENCHMARK_PATH_LAYOUT
  kind: env
  target: duckdb_temp_backend.temp_directory
```

```yaml
# contracts/pyarrow/contract.yaml
- id: ctrl_arrow_cpu_count
  name: BENCHMARK_ARROW_CPU_COUNT
  kind: env
  target: arrow_cpu_executor.workers

- id: ctrl_arrow_io_thread_count
  name: BENCHMARK_ARROW_IO_THREADS
  kind: env
  target: arrow_io_executor.workers
```

### 3. Watch validator behavior

The validator now accepts `BENCHMARK_*` in ordinary library contracts. If this regresses, check:

- `syspilot_contract/schemas.py` — `LibraryContract._validate_graph()` must not ban `control.name.startswith("BENCHMARK_")`.
- `tests/test_syspilot_contract_schema.py` — the regression test should assert `BENCHMARK_*` env controls are valid in library contracts.
- `syspilot_contract/prompts/synthesize_contract.txt` and `verify_contract.txt` — prompt policy should say `BENCHMARK_*` is allowed when it is the real fixed-workload action surface.

Worker/thread-like env names still need sensible executor targets. For example, `BENCHMARK_N_WORKERS` should target an executor `.workers` attribute, while `BENCHMARK_THREADS_PER_WORKER` may target `.threads_per_worker` when that attribute exists.

### 4. Regenerate compiled LPCL

Run from repo root:

```bash
python3 -m syspilot.cli compile-lpcl --input contracts/<library>/contract.yaml --output contracts/<library>/contract.lpcl
```

For the Arrow/Dask/DuckDB benchmark, regenerate all three:

```bash
python3 -m syspilot.cli compile-lpcl --input contracts/dask/contract.yaml --output contracts/dask/contract.lpcl
python3 -m syspilot.cli compile-lpcl --input contracts/duckdb/contract.yaml --output contracts/duckdb/contract.lpcl
python3 -m syspilot.cli compile-lpcl --input contracts/pyarrow/contract.yaml --output contracts/pyarrow/contract.lpcl
```

### 5. Validate

Run contract checks:

```bash
python3 -m syspilot.cli check-contract dask
python3 -m syspilot.cli check-contract duckdb
python3 -m syspilot.cli check-contract pyarrow
```

Run targeted tests after validator or compiler changes:

```bash
python3 -m pytest tests/test_syspilot_contract_schema.py tests/test_lpcl_compiler.py
```

Expected result from this task: all three contracts valid and `43 passed` for the two test files.

## Common mistakes

- Do not edit only `contract.lpcl`; it will be overwritten. Edit `contract.yaml` and compile.
- Do not create a new wrapper contract when the user asks to update original library contracts separately.
- Do not mix workload-shape knobs into library controls unless they are explicitly the fixed-workload tuning surface.
- Do not leave prompt/test policy saying `BENCHMARK_*` is wrapper-only after changing validator behavior.
- Do not use `as any`, `@ts-ignore`, or other type suppressions when touching code.

## Quick prompt for a future agent

“Read `experiments/LibContentBench/LPCL_BENCHMARK_ENV_MAPPING_SKILL.md`, then apply the same process to `<benchmark-dir>`: discover its env override surface, map each real library knob into the original `contracts/<library>/contract.yaml`, regenerate `.lpcl`, and run `check-contract` plus targeted LPCL tests.”
