# NestedCPU-BLAS Deep-Trace Analyzer

This directory contains the shared case-study foundation and a deterministic
post-run analyzer for CONCORD/SysPilot, SMAC3, and OpenTuner deep traces. The
shared JSONL ledgers are authoritative. Optional optimizer-native traces add
candidate provenance only after proposal count, order, and identity checks pass.

## Usage

Run from the repository root with one, two, or all three shared ledgers:

```bash
miniconda3/envs/agent/bin/python -m \
  experiments.LibContentBench.nestedcpu-blas.case_study.analyze \
  --concord-ledger runs/concord/case_study.jsonl \
  --smac-ledger runs/smac/case_study.jsonl \
  --smac-native runs/smac/ \
  --opentuner-ledger runs/opentuner/case_study.jsonl \
  --opentuner-native runs/opentuner/ \
  --metric throughput \
  --metric-mode maximize \
  --output-dir runs/deep-trace-analysis
```

`--smac-native` and `--opentuner-native` may name an `optimizer_trace.jsonl`
file or its containing directory. A native trace is optional, but when supplied
it must have exactly the shared ledger's proposal count and matching request
order/identity. For OpenTuner, each native row must contain a mapping-valued raw
`config`: its stored fingerprint must first match that raw config, then the raw
config is canonicalized through the active nestedcpu search-space manifest and
joined to the shared-ledger proposal fingerprint. This dual identity check accepts
OpenTuner aliases such as `--joblib-n-jobs` and `OMP_NUM_THREADS` without treating
their raw fingerprint as the canonical study identity. Malformed shared ledgers,
corrupted native fingerprints, incompatible optimizer names, and native/shared
canonical mismatches fail with an `analysis validation error`.

Partial runs are valid inputs. The analyzer reports pending proposals,
incomplete launches, terminal reason, and target shortfall without requiring all
three optimizers or 30 completed executions.

## Artifacts

- `aggregate_summary.json`: schema/version, exact `2*48^6 = 24,461,180,928`
  domain accounting, protocol, and one trace-specific aggregate row per supplied
  optimizer.
- `trajectory.csv`: baseline plus every proposal in fixed optimizer order
  (CONCORD, SMAC3, OpenTuner) and proposal order. Rows include canonical config,
  baseline-relative delta, disposition/reason, native provenance, launch state,
  metric/wall time, incumbent trajectory, duplicate linkage, reference-line
  membership/distance, and terminal reason.
- `report.md`: paper-oriented human-readable comparison with bounded-coverage
  caveats.
- `table.tex`: LaTeX-ready aggregate `tabular` fragment with escaped text.

JSON uses sorted compact keys, CSV has a fixed column order and 12-significant-
digit float formatting, and every writer uses LF endings and atomic replacement.
Identical inputs therefore produce byte-identical outputs.

## Accounting Semantics

The analyzer distinguishes total proposals, validated proposals, duplicates,
rejections, first unique launches, completed/incomplete launches, and
OK/error/timeout outcomes. The baseline is reported separately and remains
outside unique candidate accounting. Only finite metrics from successful
baseline/candidate executions can become best-so-far.

Coverage is emitted as an exact count ratio, a fixed 12-decimal percentage, and
a scientific fraction. It is bounded trace coverage and never an exhaustive
search or optimality claim.

The 48-point evidenced reference subset is
`(outer_jobs,1,1,1,1,1,threads)`. Membership is exact. Distance is Manhattan
distance to the nearest point in that family: `outer_jobs` is free, each fixed
integer coordinate contributes its absolute difference from 1, and a backend
mismatch contributes 1. The subset is a reference family, not an asserted
optimum.

CONCORD selection provenance comes from the raw candidate `reason` when present,
then the shared validation reason. SMAC provenance uses native `origin` and
`origin_raw`. OpenTuner provenance uses native `requestor` and only emits a
separate `technique` when that field actually exists; the analyzer does not
invent acquisition scores or technique metadata. Native traces never override
shared-ledger metrics, dispositions, or execution accounting.
