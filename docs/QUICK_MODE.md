# Quick Mode Design

This document describes the current implemented design of SysPilot quick mode. It is a design and maintenance reference for the `syspilot tune --quick-mode` workflow, not a roadmap or speculative proposal.

## Purpose

Quick mode exists to screen more candidate configurations before paying for full end-to-end validation on every candidate. It is intended for workloads where contract-backed signals become informative early enough in a run that SysPilot can use those signals as a bounded proxy during screening.

The default spawn-mode tune loop remains the normal workflow. Quick mode is an explicit alternate path selected only when `--quick-mode` is set.

## Scope and Non-Goals

Quick mode is intentionally narrow.

- It does **not** change LPCL or contract authoring.
- It does **not** use online LLM monitoring.
- It does **not** let truncated short-run success count as a final performance claim.
- It does **not** replace the standard iterative tune loop when `--quick-mode` is absent.

Instead, quick mode reuses the same profiled run executor and contract/probe plumbing as the default spawn-mode path, then changes only the candidate-screening strategy.

## Entry Points and Configuration Surface

Quick mode is enabled from the `tune` CLI:

```bash
syspilot tune --command "..." --contract <name> --quick-mode
```

The current quick-mode configuration surface is:

| Flag | Meaning | Default |
|---|---|---:|
| `--quick-mode` | Enable the alternate quick-mode workflow | `false` |
| `--quick-sample-interval-sec` | Periodic contract-probe sampling cadence | `5.0` |
| `--quick-short-measurement-sec` | Target short-screen budget used to define window width | `30.0` |
| `--quick-full-run-ratio` | Fraction of short-screen winners promoted per signal group | `0.20` |
| `--quick-max-full-runs-per-group` | Cap on promoted full validations per checked signal group | `3` |
| `--quick-min-signal-samples` | Minimum baseline samples required for a signal to participate | `4` |
| `--quick-neutral-change-pct` | Threshold below which a signal/objective change is treated as neutral | `2.0` |
| `--quick-significant-improvement-pct` | Full-run improvement threshold for a significant win | `5.0` |
| `--quick-max-windows-per-signal` | Maximum ranked abnormal windows retained per signal | `2` |
| `--quick-max-signal-groups` | Maximum ranked windows checked overall | `6` |
| `--quick-max-total-short-runs` | Global short-screen run budget | `12` |
| `--quick-max-total-full-runs` | Global promoted full-validation budget | `6` |

Quick mode also accepts the top-level tune flags `--max-iterations N`, `--fixed-budget N`, and `--fixed-goal VALUE`. Unlike the `--quick-*` knobs above, `--max-iterations` and fixed budget are candidate-attempt caps shared by standard spawn tuning and quick mode, but fixed budget is authoritative: when `--fixed-budget` is set, `--max-iterations` and its default are ignored. Fixed goal stops once the best validated optimization target reaches the requested value, or reports the current best when the active candidate-trial cap is exhausted first. Quick mode requires profiled signal sampling, so `--no-profile` is rejected when combined with `--quick-mode`.

In quick mode, fixed budget counts concrete candidate attempts after they are materialized:

- short candidate screens count
- promoted full validations count
- duplicate, invalid, failed, or timed-out candidate attempts still count once materialized
- the baseline run does not count

This cap controls search cost only. It does not guarantee that quick mode found the globally best configuration before stopping.

Validation rule: `quick_short_measurement_sec >= 3 * quick_sample_interval_sec`.

Primary objective support remains available in quick mode through `--primary-objective` and `--primary-objective-mode`, but those settings only affect baseline objective parsing, naturally completed short runs, promoted full validations, and final winner selection. They do **not** affect short-run ranking.

## Required Inputs and Preconditions

Quick mode needs contract signals that can support directional short-run screening. In practice, this means SysPilot must be able to extract one or more signal specifications with:

- a numeric `metric`
- a `virtual_probe`
- an `expected_direction_when_mitigated` of `increase` or `decrease`

If no such signal set can be derived from structured contract context and hypothesis profiles, quick mode exits early instead of silently falling back to wall-time-only short screening.

Quick mode also filters out `role=outcome` signals from this profiling-first signal set. Outcome metrics such as benchmark elapsed time remain part of full-run objective selection, but they are not treated as collectable profiling signals for quick-mode baseline ranking.

When `--memory-high` is active and the structured contract context includes cgroup-backed signals such as `cgroup.memory.events.high`, those signals are eligible for quick-mode screening only if the underlying toolbox probe can emit numeric periodic samples. Quick mode now uses the same normalized cgroup-v2 memory evidence path for those periodic samples that standard tune already uses for final post-run memory-envelope evidence.

## Relationship to the Standard Tune Loop

Quick mode changes the orchestration strategy, not the lower-level run machinery.

- The CLI still enters through `syspilot/cli.py`.
- Quick mode dispatches into `syspilot/quick_mode.py`.
- Actual process launch, profiling, process-tree discovery, periodic probe sampling, and artifact collection still go through the shared spawn executor in `syspilot/spawn_runtime.py`.
- Toolbox probe lifecycle remains the same (`begin` / `sample` / `finish`) through `syspilot/toolbox_runtime.py`.

This keeps the default tune loop and quick mode aligned on runtime evidence collection while allowing the decision policy to differ.

## End-to-End Workflow

Quick mode currently runs in five conceptual stages.

### 1. Full Baseline Run With Periodic Sampling

Quick mode always starts with a full profiled baseline run. The baseline is not truncated.

During this run, SysPilot enables periodic toolbox probe sampling using the configured quick-mode sampling interval. Those samples are stored as raw periodic time-series data for later offline analysis.

If a primary objective is configured, SysPilot also parses the baseline stdout after the run completes in order to resolve:

- the concrete metric key
- its optimization mode (`minimize` or `maximize`)
- the baseline objective value

If the primary objective cannot be resolved to a numeric metric, quick mode fails immediately instead of guessing.

### 2. Offline Abnormal-Window Ranking

After the baseline finishes, quick mode analyzes the collected baseline samples offline.

For each eligible signal:

1. extract numeric sample points from `profiling_json.periodic_probe_samples`
2. reject signals that lack enough numeric samples
3. reject flat or otherwise uninformative signals
4. rank candidate windows over the baseline sample stream
5. keep only the most abnormal non-overlapping windows, up to the configured per-signal and global caps

The abnormal window is defined by actual sampled offsets from the baseline run, not by a fresh ad hoc duration guess at screening time.

Quick mode now ranks windows with a direction-aware burst preference before falling back to the older level-based abnormality score. In practice, this means windows that contain the first meaningful increase for `decrease` signals, or the first meaningful decrease for `increase` signals, outrank later plateau-only windows when both point at the same underlying anomaly. If no directional burst is present, the earlier level-based abnormality scoring still decides the winner.

Before ranking, quick mode also collapses semantically duplicate signals that resolve to the same `(metric, virtual_probe, expected_direction_when_mitigated)` tuple, so overlapping contracts do not spend separate ranking budget on the same underlying probe.

After the global abnormal-window sort, quick mode interleaves the retained windows by `signal_ref` before screening them. This makes the first pass spend budget across distinct signals before it revisits a second ranked window from the same signal, so one strong signal cannot crowd out the early screening budget for every other useful signal.

### 3. Per-Window Candidate Proposal

For each ranked abnormal window, SysPilot asks the LLM for candidate actions using:

- the original command
- current inline environment overrides
- the ranked abnormal-window details
- baseline profiling JSON
- loaded structured contract context
- application manual hints

The prompt explicitly constrains the LLM to quick-mode rules:

- short-run ranking must use only target-signal improvement in the matching window
- the LLM must not optimize directly for wall time or the primary objective during short screening
- workload-defining arguments must remain fixed

Candidate actions are then materialized through the existing spawn candidate builder, and duplicate executable plans are removed before execution.

When `--fixed-budget` is enabled, malformed or no-action recommendation replies do not consume candidate budget because they did not produce a concrete candidate. Instead, quick mode uses a bounded same-window re-prompt path that asks for finer-grained actions, untouched controls, or a machine-readable stop reason. If that bounded retry path still cannot materialize a candidate, quick mode stops explicitly instead of pretending budget was spent.

### 4. Short Candidate Screening

Short screens are the core quick-mode optimization step.

Each candidate run:

- uses the shared profiled executor
- still collects periodic probe samples
- is intentionally stopped when it reaches the **exact ranked baseline window end offset** for the window being screened
- suppresses known wrapper footer summary lines such as `RUN_LABEL=` / `RESULT_JSON=` from the live plain-output stream only for short runs, while still preserving the complete raw `stdout.txt` / `stderr.txt` artifacts for debugging

This is the critical apples-to-apples rule: quick mode compares the same relative sampled slice of the candidate run against the same ranked slice from the baseline run.

Short-run evaluation computes target-signal improvement only inside that matched window. A short result can be marked as:

- `screened`
- `neutral`
- `unavailable`

Promotion ordering is based on target-signal improvement first, with wall time only acting as a secondary tie-breaker when signal changes are otherwise equal. The configured primary objective does not participate in short-run ranking.

If a short screen finishes the full workload before the ranked window budget stops it, quick mode now treats that run as validated objective evidence instead of forcing a redundant rerun. This preserves the signal-based screening policy for truncated runs while letting truly completed runs participate in final selection on their measured end-to-end objective.

### 5. Promoted Full Validation and Final Selection

Only a bounded fraction of short-screen winners are promoted to full runs.

Promotions are constrained by both:

- per-window-group cap: `--quick-max-full-runs-per-group`
- global cap: `--quick-max-total-full-runs`

If `--fixed-budget` is also set, both promoted full validations and short screens still spend from that authoritative top-level candidate-attempt cap, and `--max-iterations` does not further limit the run. The quick-mode `--quick-max-total-short-runs` and `--quick-max-total-full-runs` knobs remain separate internal limits. In other words, fixed budget answers "how many candidate attempts can this tune run spend overall", while the quick-mode knobs answer "how many short screens or promoted full runs may quick mode schedule inside that total".

Promoted candidates are rerun as full validations. Those runs are where quick mode returns to end-to-end optimization semantics:

- if `--primary-objective` is configured, full validations are compared on the resolved objective value, with wall time only as a tie-breaker
- otherwise, full validations use orchestration `wall_time` with `minimize` semantics

Operator-facing quick-mode output now reports these full-run comparisons as speedup ratios instead of signed percentages, for example `1.25x (better)` or `0.80x (slower)`, so negative/positive percentage signs do not force the user to mentally invert the meaning of a minimize-style objective.

Final winner selection comes from the validated objective pool: promoted full validations plus any short screens that naturally completed the entire workload before the quick-screen stop budget.

Quick mode no longer stops when the first full validation crosses the significant-improvement threshold. It keeps screening the remaining ranked windows until ranked-window exhaustion or the configured short/full run budgets are reached, then selects the best validated candidate from the full-validation pool.

When fixed budget is active, quick mode also stops when that shared candidate-attempt cap is exhausted, even if some ranked windows or quick internal budgets remain.

## Decision Rules

Quick mode uses three distinct decision layers.

### Signal Usefulness

Signals are first classified based on baseline sample quality and later reclassified after screening/validation. Current statuses include:

- `checking`
- `unavailable`
- `uninformative`
- `unchecked`
- `neutral`
- `predictive`
- `weak`
- `rejected`

This classification is part of the final quick-mode report so operators can see whether the chosen proxy signal was actually predictive of full-run improvement.

### Promotion Rules

Short-screen candidates are promotion-eligible only when they produced numeric signal improvements that were usable for screening. The promotion policy is intentionally bounded and simple:

- sort usable candidates by target-signal improvement
- break exact signal ties with wall time
- keep only the configured fraction/cap
- carry the rest as non-promoted evidence

Short screens that already completed the full workload are not promoted again; they are carried forward as already validated evidence with their parsed objective value.

### Conclusion Rules

Quick mode ends with a conclusion classification rather than a single generic success/fail bit. Current conclusion values are:

- `profiling_unavailable`
- `no_valid_signal_windows`
- `significant_improvement_found`
- `budget_exhausted`
- `signal_improved_but_no_full_speedup`
- `safe_rejection`

This distinction matters because a run can prove that the short-run proxy looked promising while still correctly rejecting all fully validated candidates.

## Artifact Model

Quick mode writes a dedicated artifact tree under:

```text
artifacts/quick_tune_runs/<run_id>/
```

Current top-level artifacts include:

- `metadata.json`
- `abnormal_windows.json`
- `abnormal_windows.jsonl`
- `baseline_run_metadata.jsonl`
- `short_run_metadata.jsonl`
- `full_run_metadata.jsonl`
- `promotion_decisions.jsonl`
- `signal_samples.jsonl`
- `signal_samples.csv`
- `quick_report.json`

Per-run artifacts live under `runs/<config_id>/` and currently include:

- `run_artifact.json`
- `run_metadata.json`
- `stdout.txt`
- `stderr.txt`

The raw periodic samples are intentionally preserved rather than collapsed away so debugging and retrospective signal analysis remain possible. This includes full captured stdout/stderr even when the live plain-output stream suppresses known short-run wrapper footer lines for readability.

## Reporting Surface

Quick mode produces both machine-readable and operator-readable outputs.

- `quick_report.json` is the canonical structured summary.
- Stage-specific JSONL files preserve baseline, short-run, full-run, promotion, and sample evidence incrementally.
- Plain output shows separate quick baseline, short screen, abnormal-window, promotion, full-validation, and signal-usefulness summaries.
- The final quick-mode summary now also groups ranked windows under each signal and shows explicit `window_id` labels in the end-of-run tables, so multiple windows for one signal do not look like duplicated signals.
- The final short-screen summary table also shows each short run's consumed screen time, so operators can see how much runtime budget each screened candidate actually spent.
- Full-validation report entries now also carry `objective_speedup_ratio` and `objective_speedup_label` alongside the raw objective values, so machine-readable artifacts match the ratio-based terminal output.

For short candidate screens in plain mode, quick mode intentionally hides known wrapper footer summaries from the live stream so repeated nested `RUN_LABEL=` / `RESULT_*` blocks do not drown out the higher-level quick-mode progress report. Baseline and promoted full-validation live output still stream normally.

The report includes at least:

- baseline summary
- abnormal windows
- short results
- promotion decisions
- full results
- signal usefulness classification
- final selection basis
- conclusion and conclusion reason

## Event Semantics

Quick mode emits spawn-mode events with `quick_mode: true` so existing consumers can distinguish the alternate path.

Current event naming centers on:

- `spawn.quick_baseline`
- `spawn.quick_screen`
- `spawn.quick_full_validation` (used in candidate evaluation payloads)

The important design point is that quick mode remains observable through the same tune-event system rather than inventing a disconnected reporting channel.

## Design Tradeoffs

Quick mode deliberately trades completeness during screening for bounded throughput:

- **Benefit:** more candidates can be screened before paying for full validation.
- **Protection:** final winner selection still requires full runs.
- **Cost:** useful early-run signals are a prerequisite; workloads without them should stay on the default tune loop.

This is why quick mode is explicit opt-in instead of an automatic replacement for standard spawn-mode tuning.

## Implementation Map

- `syspilot/cli.py`
  - CLI flag surface
  - quick-mode dispatch
- `syspilot/quick_mode.py`
  - quick-mode workflow
  - baseline window ranking
  - screening, promotion, full validation, and reporting
- `syspilot/spawn_runtime.py`
  - shared profiled executor
  - periodic probe sampling and run artifact capture
- `syspilot/toolbox_runtime.py`
  - toolbox probe session lifecycle and periodic `sample()` support
- `tests/test_quick_mode.py`
  - quick-mode behavior and artifact expectations
- `tests/test_cli_tune_ui_contract.py`
  - CLI contract for quick-mode flags

## Maintainer Notes

When changing quick mode, keep these invariants intact unless the design itself is intentionally being revised:

1. quick mode stays opt-in
2. short-run ranking stays signal-based
3. abnormal-window ranking stays offline over collected samples
4. short screens compare the same relative sampled slice as the ranked baseline window
5. final selection stays based on promoted full validations
6. raw time-series artifacts stay preserved for debugging
7. short-run plain-output cleanup must not discard the underlying captured stdout/stderr artifacts
8. quick mode must not stop on the first validated improvement if ranked windows and run budgets remain
