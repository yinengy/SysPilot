"""Run SMAC3 Bayesian optimization against one LibContentBench workflow.

Mixed integer + categorical search space, expensive black-box objective ->
:class:`HyperparameterOptimizationFacade` (random-forest surrogate) is the right
default. Trials are run **sequentially** (``n_workers=1``) by default because the
benchmarks measure cross-library resource contention; running trials in parallel
would let trials contend with each other and corrupt the timing signal.

Budget accounting
-----------------
``--budget N`` is the total benchmark-execution budget for SMAC3. The
baseline seed configuration is one of those executions, so ``--budget 1`` runs
only the baseline and performs no searched candidate trial. SMAC evaluates one
configuration per ask/tell step, so for SMAC a "round" == a "trial" == one
benchmark execution.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from smac import HyperparameterOptimizationFacade, Scenario

from experiments.LibContentBench.libcontentbench.common import write_csv, write_json
from experiments.LibContentBench.libcontentbench.direct_commands import build_direct_command
from experiments.LibContentBench.libcontentbench.tuner_trace import config_fingerprint
from experiments.LibContentBench.smac3 import trace_callback
from experiments.LibContentBench.smac3.config_space import (
    baseline_config_values,
    baseline_configuration,
    build_config_space,
)
from experiments.LibContentBench.smac3.runner import TrialEvaluator
from experiments.LibContentBench.smac3.workflows import WorkflowSpec, build_trial_payload


def _select_best(records: list[dict[str, Any]], metric_mode: str) -> dict[str, Any] | None:
    ok = [r for r in records if r["state"] == "OK" and r["metric_value"] is not None]
    if not ok:
        return None
    if metric_mode == "maximize":
        return max(ok, key=lambda r: r["metric_value"])
    return min(ok, key=lambda r: r["metric_value"])


def _speedup(metric: float | None, baseline: float | None, metric_mode: str) -> float | None:
    if metric is None or baseline is None or baseline == 0:
        return None
    if metric_mode == "maximize":
        return metric / baseline
    return baseline / metric


def _baseline_command(spec: WorkflowSpec, size: str, output_dir: Path) -> str:
    """Materialize the baseline (seed-config) command for the run report (item 1)."""
    direct_fields = {f.field_name for f in spec.direct_spec.cli_fields}
    per_trial: dict[str, Any] = {
        "case_id": "baseline",
        "label": f"{spec.workflow_id}-{size}-baseline",
        "mode": "smac3",
        "output_json": str(output_dir / "trials" / "baseline" / "output.json"),
    }
    if "runtime_root" in direct_fields:
        per_trial["runtime_root"] = str(output_dir / "trials" / "baseline" / "runtime")
    if "case_output_dir" in direct_fields:
        per_trial["case_output_dir"] = str(output_dir / "trials" / "baseline" / "case_output")
    payload = build_trial_payload(spec, size, baseline_config_values(spec), per_trial)
    return build_direct_command(spec.workflow_id, payload)


def run_smac(
    *,
    spec: WorkflowSpec,
    size: str,
    budget: int,
    metric_name: str,
    metric_mode: str,
    output_dir: Path,
    timeout_seconds: int,
    seed: int,
    n_workers: int,
    repo_root: Path,
    crash_cost: float,
    study_ledger: Path | None = None,
    study_proposal_cap: int | None = None,
    study_unique_execution_target: int | None = None,
) -> dict[str, Any]:
    smac_version = trace_callback.require_smac_2_4_0()
    os.environ.setdefault("PYTHONHASHSEED", "0")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cs = build_config_space(spec, seed=seed)
    baseline_cfg = baseline_configuration(cs, spec)
    baseline_values = baseline_config_values(spec)
    baseline_fingerprint = config_fingerprint(baseline_values)
    if hasattr(baseline_cfg, "origin"):
        baseline_cfg.origin = "Initial Design: Baseline"

    study_trace = None
    if study_ledger is not None:
        if spec.workflow_id != "nestedcpu-blas" or size != "small":
            raise ValueError("SMAC study mode requires nestedcpu-blas with --size small")
        if n_workers != 1:
            raise ValueError("SMAC study mode requires --n-workers 1")
        from experiments.LibContentBench.smac3.study_trace import SmacStudyTrace

        study_trace = SmacStudyTrace(
            study_ledger,
            proposal_cap=study_proposal_cap,
            unique_execution_target=study_unique_execution_target,
        )

    n_trials = study_trace.proposal_cap if study_trace is not None else budget

    evaluator = TrialEvaluator(
        spec=spec,
        size=size,
        metric_name=metric_name,
        metric_mode=metric_mode,
        timeout_seconds=timeout_seconds,
        output_dir=output_dir,
        repo_root=repo_root,
        crash_cost=crash_cost,
    )

    run_name = f"{spec.workflow_id}-{size}-b{budget}-s{seed}"
    scenario = Scenario(
        cs,
        name=run_name,
        output_directory=output_dir / "smac3_runs",
        deterministic=True,
        objectives="cost",
        crash_cost=crash_cost,
        n_trials=n_trials,
        seed=seed,
        n_workers=n_workers,
    )

    # SMAC's TargetFunctionRunner introspects ``target.__code__`` for its run
    # metadata, so the target must be a plain function, not a callable instance.
    # Wrap the (stateful) evaluator in a closure; ``evaluator.records`` is still
    # populated for the summary. n_workers=1 runs in-process, so the non-picklable
    # closure is fine.
    def target_function(config: Any, seed: int = 0) -> float:
        return evaluator(config, seed)

    initial_design_n_configs = (
        min(7, max(0, study_trace.proposal_cap - 1))
        if study_trace is not None
        else max(0, budget - 1)
    )
    initial_design_max_ratio = 1.0 if study_trace is not None else 0.25
    initial_design = HyperparameterOptimizationFacade.get_initial_design(
        scenario,
        n_configs=initial_design_n_configs,
        max_ratio=initial_design_max_ratio,
        additional_configs=[baseline_cfg],
    )
    decision_trace = trace_callback.SmacDecisionTrace(
        output_dir=output_dir,
        evaluator_records=evaluator.records,
        baseline_fingerprint=baseline_fingerprint,
        budget=budget,
        metric_name=metric_name,
        metric_mode=metric_mode,
        installed_smac_version=smac_version,
        initial_design_n_configs_requested=initial_design_n_configs,
        initial_design_max_ratio=initial_design_max_ratio,
        expected_baseline_ask_order=(
            initial_design_n_configs + 1 if study_trace is not None else None
        ),
        study_proposal_cap=(study_trace.proposal_cap if study_trace is not None else None),
        study_unique_execution_target=(
            study_trace.unique_execution_target if study_trace is not None else None
        ),
    )
    smac = HyperparameterOptimizationFacade(
        scenario,
        target_function,
        initial_design=initial_design,
        callbacks=[decision_trace],
        overwrite=True,
    )

    wall_start = time.time()
    study_summary = None
    try:
        if study_trace is None:
            smac.optimize()
        else:
            stop_reason = _run_study_loop(
                smac=smac,
                evaluator=evaluator,
                decision_trace=decision_trace,
                study_trace=study_trace,
                baseline_fingerprint=baseline_fingerprint,
                metric_name=metric_name,
                crash_cost=crash_cost,
            )
            study_summary = study_trace.finish(
                stop_reason=stop_reason,
                metric_name=metric_name,
                metric_mode=metric_mode,
            )
    except BaseException as exc:
        if study_trace is not None and study_summary is None:
            study_trace.finish(
                stop_reason=f"exception:{type(exc).__name__}",
                metric_name=metric_name,
                metric_mode=metric_mode,
            )
        raise
    finally:
        decision_trace.finalize()
    wall_time_sec = time.time() - wall_start

    records = sorted(evaluator.records, key=lambda r: r["trial_index"])

    # Mark which evaluations were the baseline seed config vs searched configs.
    for r in records:
        r["is_baseline"] = r.get("config_fingerprint") == baseline_fingerprint
        if "config_fingerprint" not in r:
            r["is_baseline"] = config_fingerprint(r["config"]) == baseline_fingerprint

    baseline_record = next(
        (r for r in records if r["state"] == "OK" and r["is_baseline"]),
        None,
    )
    baseline_metric = baseline_record["metric_value"] if baseline_record else None

    best_record = _select_best(records, metric_mode)
    best_metric = best_record["metric_value"] if best_record else None
    best_config = best_record["config"] if best_record else None
    best_trial_index = best_record["trial_index"] if best_record else None
    wall_time_to_best_sec = best_record["cumulative_wall_sec"] if best_record else None

    # Search trials = everything except the baseline seed. trials_to_best = # of
    # searched trials up to & incl. the best; 0 if the baseline itself is best.
    search_records = [r for r in records if not r["is_baseline"]]
    n_search_trials = len(search_records)
    if best_record is None:
        trials_to_best: int | None = None
    elif best_record["is_baseline"]:
        trials_to_best = 0
    else:
        trials_to_best = sum(
            1 for r in search_records if r["trial_index"] <= best_record["trial_index"]
        )

    trial_rows: list[dict[str, Any]] = []
    for r in records:
        row: dict[str, Any] = {
            "trial_index": r["trial_index"],
            "is_baseline": r["is_baseline"],
            "state": r["state"],
            "metric_value": r["metric_value"],
            "cost": r["cost"],
            "speedup_vs_baseline": _speedup(r["metric_value"], baseline_metric, metric_mode),
            "elapsed_sec": round(r["elapsed_sec"], 4),
            "cumulative_wall_sec": round(r["cumulative_wall_sec"], 4),
        }
        for field_name, value in r["config"].items():
            row[f"cfg.{field_name}"] = value
        trial_rows.append(row)

    speedup = _speedup(best_metric, baseline_metric, metric_mode)
    baseline_command = _baseline_command(spec, size, output_dir)

    summary = {
        "workflow": spec.workflow_id,
        "size": size,
        "mode": "smac3",
        # budget accounting: budget is total executions, including the baseline
        "budget": budget,
        "n_search_trials": n_search_trials,
        "total_executions": len(records),
        "metric": metric_name,
        "metric_mode": metric_mode,
        "seed": seed,
        "n_workers": n_workers,
        "timeout_seconds": timeout_seconds,
        "crash_cost": crash_cost,
        "n_trials_ok": sum(1 for r in records if r["state"] == "OK"),
        # ----- SysPilot-aligned report items (see README mapping table) -----
        "report": {
            "1_command": baseline_command,
            "2_total_trials": len(records),
            "2_total_rounds": len(records),  # SMAC: 1 execution/round => rounds == trials
            "3_trials_to_best": trials_to_best,
            "3_rounds_to_best": trials_to_best,
            "4_wall_time_sec": round(wall_time_sec, 4),
            "5_wall_time_to_best_sec": (
                round(wall_time_to_best_sec, 4) if wall_time_to_best_sec is not None else None
            ),
            "6_baseline_metric": baseline_metric,
            "7_best_metric": best_metric,
            "7_speedup_vs_baseline": speedup,
            "8_best_config": best_config,
            "9_per_trial": "see trials[] / trials.csv",
        },
        # flat duplicates for convenience
        "command": baseline_command,
        "n_rounds": len(records),
        "trials_to_best": trials_to_best,
        "rounds_to_best": trials_to_best,
        "wall_time_sec": round(wall_time_sec, 4),
        "wall_time_to_best_sec": (
            round(wall_time_to_best_sec, 4) if wall_time_to_best_sec is not None else None
        ),
        "baseline_metric": baseline_metric,
        "baseline_config": baseline_values,
        "baseline_is_best": bool(best_record and best_record["is_baseline"]),
        "best_metric": best_metric,
        "best_config": best_config,
        "best_trial_index": best_trial_index,
        "speedup_vs_baseline": speedup,
        "round_trial_note": (
            "SMAC evaluates one configuration per ask/tell step, so a 'round' == a "
            "'trial' == one benchmark execution. --budget N = N total executions, "
            "including the baseline seed; searched candidates are at most N - 1."
        ),
        "trials": trial_rows,
    }
    if study_summary is not None:
        assert study_ledger is not None
        summary["study"] = {
            **study_summary,
            "ledger_path": str(Path(study_ledger).resolve()),
        }
        summary["round_trial_note"] = (
            "SMAC study mode records every native ask, excludes the same-session baseline "
            "from unique candidate accounting, reuses repeated configurations without "
            "relaunching, and stops at the unique-execution target or proposal cap."
        )

    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "best_config.json",
        {
            "workflow": spec.workflow_id,
            "size": size,
            "metric": metric_name,
            "metric_mode": metric_mode,
            "best_metric": best_metric,
            "best_config": best_config,
            "best_trial_index": best_trial_index,
            "baseline_is_best": summary["baseline_is_best"],
            "baseline_metric": baseline_metric,
            "speedup_vs_baseline": speedup,
        },
    )

    csv_fields = [
        "trial_index",
        "is_baseline",
        "state",
        "metric_value",
        "cost",
        "speedup_vs_baseline",
        "elapsed_sec",
        "cumulative_wall_sec",
    ] + [f"cfg.{t.field_name}" for t in spec.tunables]
    write_csv(output_dir / "trials.csv", csv_fields, trial_rows)

    # Single-row report (SysPilot CSV items 2-7) for easy cross-run concatenation.
    run_report_fields = [
        "workflow",
        "size",
        "mode",
        "budget",
        "total_trials",
        "total_rounds",
        "trials_to_best",
        "rounds_to_best",
        "wall_time_sec",
        "wall_time_to_best_sec",
        "baseline_metric",
        "best_metric",
        "speedup_vs_baseline",
    ]
    write_csv(
        output_dir / "run_report.csv",
        run_report_fields,
        [
            {
                "workflow": spec.workflow_id,
                "size": size,
                "mode": "smac3",
                "budget": budget,
                "total_trials": len(records),
                "total_rounds": len(records),
                "trials_to_best": trials_to_best,
                "rounds_to_best": trials_to_best,
                "wall_time_sec": round(wall_time_sec, 4),
                "wall_time_to_best_sec": (
                    round(wall_time_to_best_sec, 4) if wall_time_to_best_sec is not None else None
                ),
                "baseline_metric": baseline_metric,
                "best_metric": best_metric,
                "speedup_vs_baseline": speedup,
            }
        ],
    )

    return summary


def _run_study_loop(
    *,
    smac: Any,
    evaluator: TrialEvaluator,
    decision_trace: trace_callback.SmacDecisionTrace,
    study_trace: Any,
    baseline_fingerprint: str,
    metric_name: str,
    crash_cost: float,
) -> str:
    results_by_fingerprint: dict[str, dict[str, Any]] = {}
    first_ask_by_fingerprint: dict[str, int] = {}

    while not study_trace.proposal_cap_reached:
        if study_trace.target_reached and study_trace.baseline_finished:
            return "unique_execution_target_reached"
        try:
            info = smac.ask()
        except StopIteration:
            return "smac_ask_exhausted"

        ask_order = study_trace.proposal_count + 1
        config_values = dict(info.config)
        fingerprint = config_fingerprint(config_values)
        origin_value = getattr(info.config, "origin", None)
        origin = None if origin_value is None else str(origin_value)
        is_baseline = fingerprint == baseline_fingerprint
        study_ask = study_trace.record_ask(
            config_values,
            ask_order=ask_order,
            origin=origin,
            is_baseline=is_baseline,
        )
        prior_record = results_by_fingerprint.get(fingerprint)
        duplicate_of_ask_order = first_ask_by_fingerprint.get(fingerprint)
        first_ask_by_fingerprint.setdefault(fingerprint, ask_order)

        if is_baseline:
            if prior_record is None:
                decision_trace.annotate_latest_ask(info, disposition="baseline")
                evaluator(info.config, info.seed or 0)
                record = evaluator.records[-1]
                results_by_fingerprint[fingerprint] = record
                study_trace.record_baseline_finished(record, metric_name=metric_name)
            else:
                record = prior_record
                decision_trace.annotate_latest_ask(
                    info,
                    disposition="duplicate",
                    reused_record=record,
                    duplicate_of_ask_order=duplicate_of_ask_order,
                )
                study_trace.record_baseline_finished(
                    record,
                    metric_name=metric_name,
                    reused_execution_order=int(record["trial_index"]),
                )
            value = _trial_value_from_record(record, reused=prior_record is not None)
        elif study_ask.disposition == "candidate":
            decision_trace.annotate_latest_ask(info, disposition="candidate")
            execution_sequence = study_trace.record_execution_started(
                study_ask,
                execution_order=len(evaluator.records) + 1,
            )
            evaluator(info.config, info.seed or 0)
            record = evaluator.records[-1]
            results_by_fingerprint[fingerprint] = record
            study_trace.record_execution_finished(
                execution_sequence,
                record,
                metric_name=metric_name,
            )
            value = _trial_value_from_record(record)
        elif study_ask.disposition == "duplicate":
            if prior_record is None:
                raise RuntimeError("duplicate SMAC proposal has no prior execution result")
            decision_trace.annotate_latest_ask(
                info,
                disposition="duplicate",
                reused_record=prior_record,
                duplicate_of_ask_order=duplicate_of_ask_order,
            )
            value = _trial_value_from_record(prior_record, reused=True)
        else:
            decision_trace.annotate_latest_ask(info, disposition="rejected")
            value = _rejected_trial_value(crash_cost)
        smac.tell(info, value)

    return "proposal_cap_reached"


def _trial_value_from_record(record: dict[str, Any], *, reused: bool = False) -> Any:
    from smac.runhistory.dataclasses import TrialValue
    from smac.runhistory.enumerations import StatusType

    state = str(record["state"])
    if state == "OK":
        status = StatusType.SUCCESS
    elif state == "TIMEOUT":
        status = StatusType.TIMEOUT
    else:
        status = StatusType.CRASHED
    return TrialValue(
        cost=float(record["cost"]),
        time=0.0 if reused else float(record["elapsed_sec"]),
        status=status,
        additional_info={
            "metric_value": record.get("metric_value"),
            "result_state": state,
            "result_reused": reused,
            "execution_order": int(record["trial_index"]),
        },
    )


def _rejected_trial_value(crash_cost: float) -> Any:
    from smac.runhistory.dataclasses import TrialValue
    from smac.runhistory.enumerations import StatusType

    return TrialValue(
        cost=float(crash_cost),
        status=StatusType.CRASHED,
        additional_info={
            "result_state": "REJECTED",
            "result_reused": False,
            "reason": "unique_execution_target_reached_before_launch",
        },
    )
