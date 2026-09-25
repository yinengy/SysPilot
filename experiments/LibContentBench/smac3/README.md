# SMAC3 tuning for LibContentBench workflows

This package tunes the five LibContentBench workflows with **SMAC3** (a Bayesian /
random-forest surrogate optimizer) instead of SysPilot's built-in tuner or
OpenTuner. It reuses the suite's single source of truth for command
materialization (`libcontentbench/direct_commands.py`) and each workflow's
`tunable_whitelist.txt`, so SMAC3 only ever varies the library/runtime knobs each
benchmark documents — never the workload shape.

Everything here is designed to come up **from zero on a fresh machine via
scripts** — see [Quick start](#quick-start).

## Scope

| Workflow | Primary metric (maximize) | Contention being tuned | Needs |
|---|---|---|---|
| `nestedcpu-blas` | `throughput` | joblib outer parallelism vs inner BLAS/OpenMP threads | CPU |
| `arrow-dask-duckdb-memspill` | `throughput` | host memory + spill paths across Dask / PyArrow / DuckDB | CPU |
| `hf-datasets-tokenizers-dataloader` | `throughput` (samples/s) | CPU sharing across Torch / `datasets` / `tokenizers` / DataLoader | CPU |
| `torchdata-fsspec-s3-ddp` | `samples_per_sec` | object-store + DDP knobs over `fsspec`/`s3fs` + Gloo DDP | **Docker** (local MinIO) |
| `dali-nvjpeg-pytorch-cudnn` | `score` | single-GPU sharing: DALI/nvJPEG preprocessing vs PyTorch/cuDNN inference under CUDA MPS | **GPU + CUDA + NVIDIA DALI + MPS** |

`score` = `pair_dali_images_per_sec + pair_torch_images_per_sec` from the joint
DALI + PyTorch MPS run (higher is better). The CPU workflows run anywhere;
`torchdata` and `dali` are auto-skipped by the sweep when their dependency
(Docker / GPU) is absent.

The metric per workflow and the parameter/whitelist sources follow
[`docs/fixed_budget_exp_prompt.md`](../../../docs/fixed_budget_exp_prompt.md):
the SMAC3 `--budget` is the total number of benchmark executions, including
the baseline seed, and `--metric`/direction default to the per-benchmark
`--primary-objective` values listed there. SMAC 2.4.0 yields generated Sobol
initial-design configurations first and appends the additional baseline after
them; with budget 30, the 25% cap produces seven Sobol asks and the baseline is
expected at ask position 8.

## Layout

```
smac3/
├── README.md            # this file
├── requirements.txt     # SMAC3 driver dependencies
├── setup.sh             # 0->1 deploy: driver venv + workflow prepare.sh + readiness
├── run_experiment.sh    # ONE command: full fair-comparison sweep + master_report.csv
├── experiment.py        # sweep orchestrator (matrix x sizes x budgets, resumable, aggregates)
├── cli.py               # single tuning run (one workflow x size x budget)
├── workflows.py         # specs: size tiers, baselines, whitelisted tunables, metric (no SMAC import)
├── config_space.py      # WorkflowSpec -> ConfigSpace mapping
├── runner.py            # SMAC target function: build cmd, run, parse metric, write artifacts
├── study_trace.py       # opt-in shared nestedcpu-blas case-study ledger adapter
└── optimize.py          # SMAC3 scenario/facade orchestration + summary/CSV
```

`workflows.py` imports only stdlib + `direct_commands`, so you can inspect the
search spaces and dry-run baseline commands **without SMAC3 installed**.

## Prerequisites

- `python3` (>= 3.8; tested on 3.10) with `venv`.
- Internet access for the first `pip install` and `prepare.sh` downloads.
- **Docker with daemon access** — only for `torchdata-fsspec-s3-ddp` (local MinIO
  container per trial). Your account must actually reach the Docker daemon, not
  just have the `docker` CLI on PATH: be in the `docker` group
  (`sudo usermod -aG docker $USER`, then re-login) or run the tuner under `sudo`.
  Otherwise the runner reports `status=blocked` with `permission denied ...
  docker.sock`, and the sweep auto-skips this workflow with that reason printed.
- **GPU + CUDA + NVIDIA DALI + CUDA MPS** — only for `dali-nvjpeg-pytorch-cudnn`.
  Start the MPS daemon (`nvidia-cuda-mps-control -d`) before running it, or the
  MPS-share knobs are no-ops; its `prepare.sh` installs DALI/Torch on the GPU host.
- The three CPU workflows need none of the above. Integer thread/worker knobs
  auto-scale their upper bound to `min(os.cpu_count(), 64)` and baselines are
  clamped into bounds, so the search space adapts to the machine.

## Quick start (two commands on a fresh machine)

```bash
# from the repo root
sh experiments/LibContentBench/smac3/setup.sh          # 1. deploy 0 -> 1
sh experiments/LibContentBench/smac3/run_experiment.sh # 2. run the full fair-comparison sweep
```

The sweep writes `experiments/LibContentBench/smac3/runs/master_report.csv` —
one row per `(workflow, size, budget)`, directly comparable to SysPilot's
fixed-budget results. That's the whole story; the sections below explain the
knobs.

### Step 1 — Deploy (0 → 1): `setup.sh`

```bash
sh experiments/LibContentBench/smac3/setup.sh
```

This:
1. creates an isolated SMAC3 **driver venv** at `smac3/.venv` and installs SMAC3;
2. runs each workflow's `prepare.sh` (building its own `artifact/venv`);
3. prints a readiness summary.

Subset / toggles:

```bash
sh experiments/LibContentBench/smac3/setup.sh nestedcpu-blas arrow-dask-duckdb-memspill
SKIP_PREPARE=1 sh experiments/LibContentBench/smac3/setup.sh   # driver venv only
SKIP_DRIVER=1  sh experiments/LibContentBench/smac3/setup.sh   # workflow venvs only
```

> If the SMAC3 install fails while building `pyrfr` (rare; only on platforms
> without a prebuilt wheel), install SWIG and re-run:
> `apt-get install -y swig` (or `smac3/.venv/bin/pip install swig`).

### Step 2 — Run the full fair-comparison sweep: `run_experiment.sh`

```bash
sh experiments/LibContentBench/smac3/run_experiment.sh            # full matrix
sh experiments/LibContentBench/smac3/run_experiment.sh --dry-run  # print the plan only
```

This runs **every `(workflow, size, budget)` job sequentially** (fairness — see
below), exporting `PYTHONHASHSEED=0`, and aggregates `runs/master_report.csv`.
The default matrix mirrors `fixed_budget_exp_prompt.md`: **5 workflows ×
{small, medium, large} × budgets {10, 20, 30}** = 45 runs (the `torchdata` and
`dali` rows are auto-skipped when Docker / a GPU is absent). It is **resumable**
(a run whose `summary.json` exists is reused) and writes the master report
incrementally, so you can interrupt and re-run safely.

Scope it down for a quick check or to skip the heavy / special-dependency tiers:

```bash
# fast smoke: one workflow, small, budget 10
sh experiments/LibContentBench/smac3/run_experiment.sh \
  --workflows nestedcpu-blas --sizes small --budgets 10

# only the CPU workflows
sh experiments/LibContentBench/smac3/run_experiment.sh --skip-torchdata --skip-dali

# only the GPU workflow (on a CUDA+DALI+MPS host)
sh experiments/LibContentBench/smac3/run_experiment.sh --workflows dali-nvjpeg-pytorch-cudnn
```

`torchdata-fsspec-s3-ddp` is auto-skipped if Docker is missing, and
`dali-nvjpeg-pytorch-cudnn` if `nvidia-smi` is missing (each with a printed
notice). All extra flags pass through to `experiment.py`
(`--workflows --sizes --budgets --seed --timeout-sec --n-workers --force
--skip-torchdata --skip-dali --output-root`).

> The full matrix is heavy (the `large`/`hf`/`torchdata` tiers run for minutes
> each). Start with `--sizes small --budgets 10` to validate the pipeline, then
> scale up. Wall-clock ≈ sum of all runs since trials are sequential.

### Verify without running anything

```bash
PY=experiments/LibContentBench/smac3/.venv/bin/python   # or any python3
$PY experiments/LibContentBench/smac3/cli.py --list-workflows
$PY experiments/LibContentBench/smac3/cli.py --workflow nestedcpu-blas --size small --dry-run
```

`--dry-run` prints the resolved search space + the exact baseline command and
needs no SMAC3, no workflow venv, and no Docker.

### Run a single tuning job (ad-hoc): `cli.py`

```bash
PY=experiments/LibContentBench/smac3/.venv/bin/python
PYTHONHASHSEED=0 $PY experiments/LibContentBench/smac3/cli.py \
  --workflow nestedcpu-blas \
  --size small \
  --budget 20 \
  --output-dir experiments/LibContentBench/smac3/runs/nestedcpu-blas/small-b20
```

### NestedCPU-BLAS deep-trace study mode

The normalized cross-optimizer case-study ledger is opt-in and restricted to the
pinned `nestedcpu-blas` small workload:

```bash
PYTHONHASHSEED=0 $PY experiments/LibContentBench/smac3/cli.py \
  --workflow nestedcpu-blas \
  --size small \
  --budget 30 \
  --output-dir experiments/LibContentBench/smac3/runs/nestedcpu-blas/deep-trace \
  --study-ledger experiments/LibContentBench/smac3/runs/nestedcpu-blas/deep-trace/case_study.jsonl
```

Study mode uses SMAC 2.4.0's public `ask()` / `tell()` API. It records every
native ask before execution, preserves the seven generated Sobol asks followed
by the added baseline ask at position 8, and continues until 30 unique
non-baseline configurations have launched or 60 total proposals have been
observed. The baseline runs once in the same session but is excluded from the
30-candidate target. Repeated configurations reuse the first result without a
second benchmark launch; failed and timed-out first launches still count once
toward the unique target and are told back to SMAC with the finite crash cost.
If proposal exhaustion prevents 30 unique launches, the ledger ends with
`proposal_cap_reached_with_shortfall` (or another explicit shortfall reason).

Use `--study-proposal-cap` and `--study-unique-execution-target` only for bounded
smoke validation. They require `--study-ledger`; normal runs ignore this study
protocol and keep the existing `--budget` behavior unchanged.

Run each tuning job individually (one workflow × size × budget per command), as
in the experiment protocol. Examples for the other workflows:

```bash
# arrow, medium, 30 trials
$PY .../smac3/cli.py --workflow arrow-dask-duckdb-memspill --size medium --budget 30 \
  --output-dir .../smac3/runs/arrow-dask-duckdb-memspill/medium-b30

# hf, small, 10 trials
$PY .../smac3/cli.py --workflow hf-datasets-tokenizers-dataloader --size small --budget 10 \
  --output-dir .../smac3/runs/hf-datasets-tokenizers-dataloader/small-b10

# torchdata (needs Docker), small, 20 trials
$PY .../smac3/cli.py --workflow torchdata-fsspec-s3-ddp --size small --budget 20 \
  --output-dir .../smac3/runs/torchdata-fsspec-s3-ddp/small-b20
```

## CLI reference

| Flag | Default | Meaning |
|---|---|---|
| `--workflow` | — | one of the four ids (required unless `--list-workflows`) |
| `--size` | `medium` | `small` \| `medium` \| `large` workload tier |
| `--budget` | `20` | number of **total benchmark executions, including the baseline seed**; `--budget 1` runs only the baseline |
| `--metric` | per-workflow primary | metric key parsed from the runner |
| `--metric-mode` | per-workflow (`maximize`) | optimization direction |
| `--output-dir` | — | where artifacts + reports are written (required to tune) |
| `--timeout-sec` | per-size safety value | per-trial subprocess wall timeout |
| `--seed` | `0` | SMAC3 / ConfigSpace seed |
| `--n-workers` | `1` | parallel trial workers — **keep 1** (see below) |
| `--crash-cost` | `1e9` | cost for failed/timed-out trials |
| `--study-ledger` | off | opt in to the normalized nestedcpu-blas deep-trace ledger |
| `--study-proposal-cap` | manifest (`60`) | study-only total native ask cap |
| `--study-unique-execution-target` | manifest (`30`) | study-only unique non-baseline launch target |
| `--dry-run` | off | print search space + baseline command, then exit |
| `--list-workflows` | off | print `id  metric  mode` and exit |

## How it works

For each trial SMAC proposes a configuration over the whitelisted knobs. The
target function (`runner.py`):

1. builds `payload = fixed size-tier shape + tuned knobs + per-trial paths`;
2. materializes the command with `build_direct_command(...)` — identical
   machinery to the suite's direct runners;
3. runs it as a subprocess from the repo root with `PYTHONHASHSEED=0` and an
   outer wall-clock timeout, using a **per-trial `runtime_root`/`case_output_dir`**
   so trials don't share state;
4. parses the metric-only JSON from stdout (falls back to the `--output-json`
   artifact);
5. prints the per-trial wall time and objective value as soon as the trial completes;
6. returns the SMAC *cost* = `-metric` (because every objective is maximize), or
   `--crash-cost` on failure/timeout.

**Failed trials are expected and handled.** Some configurations the optimizer
proposes are legitimately infeasible — e.g. `worker_memory_limit=128MB` for the
arrow workflow makes the Dask workers exceed their memory budget and get killed
(*"N workers died while trying to run it"*), which is exactly the memory/spill
contention the benchmark probes. Such a trial is recorded with `state=ERROR`
(or `TIMEOUT`/`NO_METRIC`) and the configured `crash_cost`, and the search
continues. So a run reporting **fewer OK trials than its total execution budget is normal**, not
a bug; inspect `trials.csv` / `trials/trial_XXXX/trial_artifact.json` for the
per-trial state and stderr.

The baseline (seed) configuration — each workflow's documented default knobs — is
passed as an additional initial-design configuration and consumes one budgeted
execution. SMAC 2.4.0 yields generated Sobol points before additional configurations,
so the baseline follows the capped Sobol points and is used as the speedup denominator.
For budget 30, `n_configs=29` is capped to seven Sobol points and the baseline is
expected at ask position 8.

## Fair comparison with SysPilot (budget, round vs trial)

This harness is meant to be an apple-to-apple alternative optimizer to SysPilot's
LLM-based `tune`. The shared, comparable unit is the **trial = one benchmark
execution = one candidate configuration**. Everything below is aligned with the
SysPilot protocol **except the tunable value ranges** — read the two ⚠️ callouts
just below this section (**Known difference #1 — search-space bounds** and
**#2 — doc pitfall**) before reconciling numbers against SysPilot.

- **Budget includes the baseline.** `--budget N` is the total number of benchmark executions SMAC3 may run. The baseline seed consumes one execution, so `--budget 1` runs only the baseline and `--budget N` leaves at most `N - 1` searched candidates. For budgets above one, generated Sobol initial-design points are yielded before the additional baseline; budget 30 yields seven Sobol points followed by the baseline at expected ask position 8.
- **Round vs trial.** SysPilot has two levels — a *round* is one LLM reasoning
  step that may emit several *candidate trials*; the budget counts candidate
  trials. SMAC evaluates exactly **one configuration per ask/tell step**, so for
  SMAC a *round* == a *trial* == one execution. Reports therefore use **trials**
  as the unit, with `rounds == trials` for SMAC.
- **Single measurement per trial — on purpose.** SysPilot measures each candidate
  once (no in-loop repeats); this harness does the same, so the comparison is
  faithful. (Run-to-run timing noise is real — see Caveats. If you want repeated
  measurement for variance control, apply it to *both* tools, not just one, or the
  comparison stops being apple-to-apple.)
- **Same primary objective + direction, same whitelist, same fixed workload
  shape, same `PYTHONHASHSEED=0`, sequential trials** — all aligned with the
  `fixed_budget_exp_prompt.md` protocol.

### ⚠️ Known difference #1 — search-space bounds (SysPilot has none)

> **The one setting that is *not* identical to SysPilot is the tunable value
> range, and it cannot be — SysPilot does not define one.**
>
> SysPilot's LPCL `Control` schema (`syspilot_contract/schemas.py`) carries only
> `name / kind / type / target / effect` — there is **no `min` / `max` / `range`
> / `domain` / allowed-values field**. In both manual and contract mode SysPilot's
> **LLM proposes values freely** (guided by the knob type, target, profiling, and
> — in contract mode — the effect direction), so it can pick values our search
> never reaches (e.g. `worker_memory_limit=384MB`, `torch_num_threads=200`).
>
> SMAC, by contrast, **requires a bounded `ConfigSpace`**, so this harness
> declares explicit ranges: integer thread/worker knobs `1..min(cpu,64)`,
> `prefetch_factor 1..8`, `ddp_bucket_cap_mb 1..64`, DALI batch `16..256`,
> continuous DALI hardware-decoder load `0.0..1.0`, and categorical option sets
> (`worker_memory_limit ∈ {128MB,256MB,512MB,1GB}`,
> `s3_cache_type ∈ {none,bytes,readahead,mmap}`, …).
> These bounds come from **this repo's OpenTuner integration
> (`opentuner_integration/specs.py`)**, whose parameter ranges and option sets are
> in turn taken from each benchmark's documented values / runner defaults — they are **our
> choice, not inherited from SysPilot**.
>
> Consequence: SMAC and SysPilot do **not** search literally the same space. This
> is the inherent Bayesian-optimization (bounded domain) vs LLM (free proposal)
> difference; it can't be "aligned away" because there is no SysPilot range to
> align to. The knob *set*, metric, workload shape, baseline, and budget **are**
> identical (see table below).

### ⚠️ Known difference #2 — doc pitfall in `fixed_budget_exp_prompt.md`

> **The inline example commands in `docs/fixed_budget_exp_prompt.md` (lines ~42 &
> ~45) do NOT match the authoritative size catalog** — do not copy them verbatim.
>
> That example shows `--matrix-size 128 --task-count 2 --repetitions 3` and
> `MKL_NUM_THREADS=6 BLIS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6`, whereas
> `workflow_size_commands.md` (the source the prompt itself points to, line 6) and
> this harness use the small tier `--matrix-size 320 --task-count 800
> --repetitions 6` with `MKL=1 BLIS=1 NUMEXPR=1`. The inline snippet is a stale
> illustration. **This harness follows `workflow_size_commands.md`**, so its fixed
> workload shape and baseline thread values are byte-for-byte that catalog's — if
> you ever reconcile against SysPilot, compare against `workflow_size_commands.md`,
> not the inline example.

### SysPilot report-item mapping

`summary.json` carries a `report` block keyed to the 9 items in
`fixed_budget_exp_prompt.md`, and `run_report.csv` is a single row (items 2–7)
for easy cross-run concatenation into the SysPilot-style spreadsheet:

| SysPilot report item | `summary.json` field |
|---|---|
| 1. command | `command` (baseline seed command) |
| 2. total trials / rounds | `total_executions` / `n_rounds` (equal) |
| 3. trials / rounds to best | `trials_to_best` / `rounds_to_best` (0 if baseline is best) |
| 4. wall time of optimizer | `wall_time_sec` |
| 5. wall time to best | `wall_time_to_best_sec` |
| 6. baseline metric | `baseline_metric` |
| 7. optimal metric + speedup | `best_metric`, `speedup_vs_baseline` |
| 8. best config | `best_config` |
| 9. per-trial metric + speedup | `trials[]` / `trials.csv` |

## Output artifacts

The sweep (`run_experiment.sh` / `experiment.py`) writes, under `--output-root`
(default `smac3/runs/`):

| Path | Contents |
|---|---|
| `master_report.csv` | **one row per `(workflow, size, budget)`** with mode=`smac3`, trials, trials-to-best, wall times, baseline/best metric, speedup, `status`, and `output_dir` — slot this next to SysPilot's contract/manual rows |
| `<workflow>/<size>-b<budget>-s<seed>/` | a per-run directory (the single-run `--output-dir`, contents below) |

Each single run (`cli.py --output-dir DIR`, also one per sweep cell) writes:

| Path | Contents |
|---|---|
| `summary.json` | budget accounting, the SysPilot-aligned `report` block (items 1–9), baseline/best metric + config, trials-to-best, wall time + wall-time-to-best, speedup, `baseline_is_best`, and a per-trial list |
| `run_report.csv` | single row (SysPilot CSV items 2–7) for cross-run concatenation |
| `best_config.json` | the incumbent knobs + metric + speedup + `baseline_is_best` |
| `trials.csv` | one row per trial: `is_baseline`, state, metric, speedup, timings, and every tuned knob |
| `optimizer_trace.jsonl` | deterministic ask-order trace; every row carries `schema_version`, `optimizer=smac3`, `optimizer_version`, execution order, config fingerprint, raw/normalized origin, baseline identity, result, and incumbent transitions |
| `trace_manifest.json` | trace schema, exact SMAC version check, initial-design settings, baseline fingerprint, and explicit acquisition-score availability |
| `trace_summary.json` | trace counts, origin distribution, baseline position metadata, incomplete asks, and final incumbent identity |
| study ledger path | shared normalized proposal/baseline/execution/duplicate/terminal JSONL events when `--study-ledger` is enabled |
| `trials/trial_XXXX/trial_artifact.json` | full record: command, config, materialized payload, stdout/stderr tails, return code, metric |
| `trials/trial_XXXX/{runtime,case_output,output.json}` | per-trial working dirs + runner output |
| `smac3_runs/<name>/` | SMAC3's own run history / database |

## Reproducibility & determinism

- `setup.sh` rebuilds every environment from scratch; `requirements.txt` pins
  SMAC3 exactly to `2.4.0`; the MinIO image is pinned to a fixed release tag.
- The runner reads the installed distribution version with
  `importlib.metadata.version("smac")` and fails before optimization unless it is
  exactly `2.4.0`.
- The repository callback uses only SMAC 2.4.0 public ask/tell, runhistory, and
  intensifier surfaces. SMAC does not expose acquisition scores through those
  callbacks, so traces record a null score with an explicit unavailability reason
  instead of fabricating one.
- `PYTHONHASHSEED=0` is exported into every trial subprocess (and the CLI sets it
  for the driver too); pass `--seed` to fix SMAC's own RNG. Export
  `PYTHONHASHSEED=0` before launching for end-to-end determinism.
- **Trials run sequentially (`--n-workers 1`).** These benchmarks measure
  cross-library resource contention; running trials in parallel would let them
  contend with each other and corrupt the timing signal. Only raise `--n-workers`
  if you understand and accept that trade-off.

## Fairness rules (inherited from the suite)

- `direct_commands.py` is the only command builder — no bespoke command strings.
- Only knobs authorized by each workflow's `tunable_whitelist.txt` are exposed;
  `workflows.py` validates this at import (and that every baseline encodes to a
  valid default).
- Workload-shape flags (matrix size, dataset rows, file/shard counts, batch size,
  epochs, measurement windows, `world_size`, ...) are **fixed per size tier** and
  never tuned.

## Caveats

- `torchdata-fsspec-s3-ddp` needs Docker and pulls the pinned MinIO image; treat
  its results as single-host/loopback S3 evidence. Each trial starts and tears
  down its own container, so trials must stay sequential.
- The `large` tiers (and `torchdata`/`hf` generally) have longer per-trial
  runtimes; size the budget and `--timeout-sec` accordingly.
- `hf-datasets-tokenizers-dataloader` may propose `num_workers=0`; the runner
  ignores `prefetch_factor`/`persistent_workers` in that case (no crash).
- `dali-nvjpeg-pytorch-cudnn` needs a GPU with CUDA, NVIDIA DALI, and **CUDA MPS
  running** (`nvidia-cuda-mps-control -d`). Without MPS the `*_mps_share` knobs do
  nothing and `score` won't reflect real co-location. The harness materializes a
  **real** run (no `--scaffold-only`); the runner's `--scaffold-only` host-mode is
  not throughput evidence and is not used here. Each trial runs the DALI + Torch
  pair on one GPU, so trials must stay sequential. `--cuda-device` is fixed to `0`.
