from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from syspilot.cli import (
    _materialize_spawn_candidate_attempts_from_candidate_sets,
    _parse_spawn_actions,
    cli,
)
from syspilot.study_trace import (
    NestedCpuBlasStudyTrace,
    RepositoryStudySourceUnavailableError,
)

BASE_COMMAND = [
    "python",
    "run_nestedcpu_blas_case.py",
    "--joblib-n-jobs",
    "6",
    "--backend-prefer",
    "processes",
]
BASE_ENV = {
    "OMP_NUM_THREADS": "6",
    "OPENBLAS_NUM_THREADS": "6",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@dataclass
class _Response:
    content: str


class _OneCandidateAgent:
    def __init__(self, config: object) -> None:
        self.config = config
        self.context = SimpleNamespace(loaded_manuals={})

    def configure_tuning_prompt(self, **kwargs: object) -> None:
        _ = kwargs

    def load_manuals(self, names: list[str]) -> dict[str, bool]:
        return {name: False for name in names}

    async def step(self, prompt: str) -> _Response:
        _ = prompt
        return _Response(
            '{"summary":"try one","manual_citation":"","evidence":[],"candidates":['
            '{"reason":"reduce nesting","changes":{"OMP_NUM_THREADS":2}},'
            '{"reason":"backup nesting","changes":{"OMP_NUM_THREADS":3}}]}'
        )


def _noop(*args: object, **kwargs: object) -> None:
    _ = args
    _ = kwargs


async def _fake_profile(*args: object, **kwargs: object) -> dict[str, object]:
    _ = args
    _ = kwargs
    return {
        "pid": 123,
        "wall_time": 1.0,
        "stdout": "{}",
        "stderr": "",
        "returncode": 0,
        "timed_out": False,
        "profiling_text": "profile",
        "profiling_json": {},
        "tool_calls_used": 0,
    }


def _events(path: Path) -> list[dict[str, Any]]:
    case_study = importlib.import_module(
        "experiments.LibContentBench.nestedcpu-blas.case_study"
    )
    return case_study.read_events(path)


def test_study_foundation_is_not_imported_until_trace_is_enabled() -> None:
    module_name = "experiments.LibContentBench.nestedcpu-blas.case_study"
    output = subprocess.check_output(
        [
            sys.executable,
            "-c",
            (
                "import sys; import syspilot.study_trace; "
                f"print({module_name!r} in sys.modules)"
            ),
        ],
        text=True,
    )

    assert output.strip() == "False"


def test_tune_help_does_not_load_study_foundation() -> None:
    with patch("syspilot.study_trace.importlib.import_module") as import_module:
        result = CliRunner().invoke(cli, ["tune", "--help"])

    assert result.exit_code == 0
    assert "repository source checkout" in result.output
    import_module.assert_not_called()


def test_missing_study_package_reports_source_checkout_requirement(tmp_path: Path) -> None:
    missing_package = ModuleNotFoundError(
        "No module named 'experiments'",
        name="experiments",
    )

    with patch(
        "syspilot.study_trace.importlib.import_module",
        side_effect=missing_package,
    ):
        result = CliRunner().invoke(
            cli,
            [
                "tune",
                "--command",
                "echo hi",
                "--study-ledger",
                str(tmp_path / "trace.jsonl"),
            ],
        )

    assert result.exit_code == 1
    assert "requires a SysPilot repository source checkout" in result.output
    assert "not included in production wheels" in result.output


@pytest.mark.parametrize(
    "missing_name",
    [
        "experiments",
        "experiments.LibContentBench",
        "experiments.LibContentBench.nestedcpu-blas.case_study",
    ],
)
def test_missing_study_target_or_ancestor_is_relabelled(
    tmp_path: Path,
    missing_name: str,
) -> None:
    missing_package = ModuleNotFoundError(
        f"No module named '{missing_name}'",
        name=missing_name,
    )

    with patch(
        "syspilot.study_trace.importlib.import_module",
        side_effect=missing_package,
    ):
        with pytest.raises(RepositoryStudySourceUnavailableError):
            NestedCpuBlasStudyTrace(tmp_path / "trace.jsonl")


@pytest.mark.parametrize(
    "missing_name",
    [
        "unexpected_dependency",
        "experiments.LibContentBench.nestedcpu-blas.case_study.analysis",
    ],
)
def test_transitive_missing_study_dependency_is_not_relabelled(
    tmp_path: Path,
    missing_name: str,
) -> None:
    missing_dependency = ModuleNotFoundError(
        f"No module named '{missing_name}'",
        name=missing_name,
    )

    with patch(
        "syspilot.study_trace.importlib.import_module",
        side_effect=missing_dependency,
    ):
        with pytest.raises(ModuleNotFoundError) as exc_info:
            NestedCpuBlasStudyTrace(tmp_path / "trace.jsonl")

    assert exc_info.value is missing_dependency


def test_generic_study_import_error_is_not_relabelled(tmp_path: Path) -> None:
    import_error = ImportError("binary extension failed to load")

    with patch(
        "syspilot.study_trace.importlib.import_module",
        side_effect=import_error,
    ):
        with pytest.raises(ImportError) as exc_info:
            NestedCpuBlasStudyTrace(tmp_path / "trace.jsonl")

    assert exc_info.value is import_error


def test_study_budget_options_require_opt_in_ledger() -> None:
    result = CliRunner().invoke(cli, ["tune", "--command", "echo hi", "--study-proposal-cap", "2"])

    assert result.exit_code == 1
    assert "require" in result.output
    assert "--study-ledger" in result.output


def test_parser_preserves_every_raw_model_candidate_before_materialization() -> None:
    parsed = _parse_spawn_actions(
        """{
          "summary": "three proposals",
          "candidates": [
            {"changes": {"OMP_NUM_THREADS": 2}},
            {"changes": []},
            "malformed"
          ]
        }""",
        baseline_command_tokens=BASE_COMMAND,
        baseline_env_overrides=BASE_ENV,
    )

    assert len(parsed["raw_candidates"]) == 3
    assert parsed["candidates"][0]["source_candidate_index"] == 0
    assert any("candidate[1]" in reason for reason in parsed["skipped_actions"])
    assert any("candidate[2]" in reason for reason in parsed["skipped_actions"])


def test_materialization_preserves_proposal_source_for_rejection_and_duplicate() -> None:
    candidate_sets = [
        {
            "source_candidate_index": 4,
            "actions": [
                {
                    "config_type": "env_var",
                    "parameter": "OMP_NUM_THREADS",
                    "new_value": "6",
                }
            ],
        },
        {
            "source_candidate_index": 5,
            "actions": [
                {
                    "config_type": "env_var",
                    "parameter": "OMP_NUM_THREADS",
                    "new_value": "2",
                }
            ],
        },
        {
            "source_candidate_index": 6,
            "actions": [
                {
                    "config_type": "env_var",
                    "parameter": "OMP_NUM_THREADS",
                    "new_value": "2",
                }
            ],
        },
    ]

    attempts = _materialize_spawn_candidate_attempts_from_candidate_sets(
        BASE_COMMAND,
        candidate_sets,
        max_attempts=3,
        base_env_overrides=BASE_ENV,
    )

    assert [attempt["source_candidate_index"] for attempt in attempts] == [4, 5, 6]
    assert [attempt["materialized_status"] for attempt in attempts] == [
        "rejected",
        "ready",
        "duplicate",
    ]


def test_trace_records_baseline_proposals_reuse_and_failed_unique_execution(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    trace = NestedCpuBlasStudyTrace(path, proposal_cap=4, unique_execution_target=2)
    trace.record_baseline_started(BASE_COMMAND, BASE_ENV, metadata={"phase": "baseline"})
    trace.record_baseline_finished(
        status="ok",
        wall_time_s=10.0,
        objective_name="wall_time_sec",
        objective_value=10.0,
        returncode=0,
    )

    proposals = trace.record_proposals(
        [
            {"changes": {"OMP_NUM_THREADS": 2}},
            {"changes": {"OMP_NUM_THREADS": 2}},
            {"changes": {"OMP_NUM_THREADS": 99}},
        ],
        BASE_COMMAND,
        BASE_ENV,
        metadata={"round": 1},
    )
    assert trace.validate_proposal(proposals[0], accepted=True, reason="valid") == "candidate"
    execution_sequence = trace.record_execution_started(
        proposals[0],
        metadata={"current_wall_time_s": 10.0, "incumbent_wall_time_s": 10.0},
    )
    trace.record_execution_finished(
        execution_sequence,
        status="timeout",
        wall_time_s=5.0,
        objective_name=None,
        objective_value=None,
        returncode=-1,
        timed_out=True,
    )
    assert trace.validate_proposal(proposals[1], accepted=True, reason="valid") == "duplicate"
    assert trace.validate_proposal(proposals[2], accepted=True, reason="valid") == "rejected"
    trace.finish(metadata={"ordinary_stop_reason": "candidate_space_exhausted"})

    events = _events(path)
    validations = [event for event in events if event["event_type"] == "proposal.validated"]
    terminal = events[-1]

    assert [event["disposition"] for event in validations] == [
        "candidate",
        "duplicate",
        "rejected",
    ]
    assert validations[1]["duplicate_of_proposal_sequence"] == proposals[0].sequence
    assert terminal["event_type"] == "study.finished"
    assert terminal["reason"] == "terminated_with_shortfall"
    assert terminal["metadata"]["unique_execution_count"] == 1
    assert terminal["metadata"]["baseline_excluded_from_unique_execution_count"] is True
    assert terminal["metadata"]["shortfall"] == 1


def test_trace_terminal_reports_cap_shortfall_and_target_reached(tmp_path: Path) -> None:
    shortfall_path = tmp_path / "shortfall.jsonl"
    shortfall = NestedCpuBlasStudyTrace(
        shortfall_path,
        proposal_cap=2,
        unique_execution_target=2,
    )
    shortfall.record_baseline_started(BASE_COMMAND, BASE_ENV)
    shortfall.record_baseline_finished(
        status="ok",
        wall_time_s=10.0,
        objective_name=None,
        objective_value=None,
        returncode=0,
    )
    proposals = shortfall.record_proposals(
        [
            {"changes": {"OMP_NUM_THREADS": 2}},
            {"changes": {"OMP_NUM_THREADS": 2}},
        ],
        BASE_COMMAND,
        BASE_ENV,
        metadata={},
    )
    shortfall.validate_proposal(proposals[0], accepted=False, reason="fairness rejection")
    shortfall.validate_proposal(proposals[1], accepted=False, reason="duplicate materialization")
    shortfall.finish()
    assert _events(shortfall_path)[-1]["reason"] == "proposal_cap_reached_with_shortfall"

    target_path = tmp_path / "target.jsonl"
    target = NestedCpuBlasStudyTrace(target_path, proposal_cap=2, unique_execution_target=1)
    target.record_baseline_started(BASE_COMMAND, BASE_ENV)
    target.record_baseline_finished(
        status="ok",
        wall_time_s=10.0,
        objective_name=None,
        objective_value=None,
        returncode=0,
    )
    proposal = target.record_proposals(
        [{"changes": {"OMP_NUM_THREADS": 2}}],
        BASE_COMMAND,
        BASE_ENV,
        metadata={},
    )[0]
    target.validate_proposal(proposal, accepted=True, reason="valid")
    execution_sequence = target.record_execution_started(proposal, metadata={})
    target.record_execution_finished(
        execution_sequence,
        status="error",
        wall_time_s=1.0,
        objective_name=None,
        objective_value=None,
        returncode=2,
        timed_out=False,
    )
    target.finish()

    terminal = _events(target_path)[-1]
    assert terminal["reason"] == "unique_execution_target_reached"
    assert terminal["metadata"]["unique_execution_count"] == 1


def test_spawn_tune_trace_executes_target_without_counting_baseline(tmp_path: Path) -> None:
    path = tmp_path / "spawn-trace.jsonl"
    proposal_recorded_before_parse: list[bool] = []
    command = (
        "env OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=1 "
        "BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "
        "echo --joblib-n-jobs 6 --backend-prefer processes"
    )

    def _checking_parse(
        response: str,
        *,
        baseline_command_tokens: list[str] | None = None,
        baseline_env_overrides: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        proposal_recorded_before_parse.append(
            any(event["event_type"] == "proposal.recorded" for event in _events(path))
        )
        return _parse_spawn_actions(
            response,
            baseline_command_tokens=baseline_command_tokens,
            baseline_env_overrides=baseline_env_overrides,
        )

    with (
        patch("syspilot.cli.SysPilotAgent", _OneCandidateAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _fake_profile),
        patch("syspilot.cli._parse_spawn_actions", _checking_parse),
    ):
        result = CliRunner().invoke(
            cli,
            [
                "tune",
                "--ui-mode",
                "plain",
                "--command",
                command,
                "--study-ledger",
                str(path),
                "--study-proposal-cap",
                "2",
                "--study-unique-execution-target",
                "1",
                "--timeout",
                "1",
            ],
        )

    assert result.exit_code == 0, result.output
    events = _events(path)
    assert sum(event["event_type"] == "baseline.started" for event in events) == 1
    assert sum(event["event_type"] == "execution.started" for event in events) == 1
    validations = [event for event in events if event["event_type"] == "proposal.validated"]
    assert [event["disposition"] for event in validations] == ["candidate", "rejected"]
    assert validations[1]["reason"] == "unique execution target reached before candidate launch"
    assert proposal_recorded_before_parse == [True]
    assert events[-1]["event_type"] == "study.finished"
    assert events[-1]["reason"] == "unique_execution_target_reached"
