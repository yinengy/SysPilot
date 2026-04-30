from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from click.testing import CliRunner

from syspilot.cli import (
    cli,
    _load_default_llm_config,
    _build_spawn_profiling_env,
    _collect_contract_report_env,
    _extract_hypothesis_profiles,
    _parse_spawn_actions,
    _summarize_spawn_runtime_output,
)
from syspilot.core.config import SysPilotConfig
from syspilot.core.agent import SysPilotAgent
from syspilot.core.llm import LLMTimeoutError
from syspilot_contract.compose_contracts import compose_contracts
from syspilot_contract.integration import build_agent_context_bundle
from syspilot_contract.integration import write_contract_bundle
from syspilot_contract.schemas import LibraryContract as LibraryContractDSL
from syspilot_manual.integration import write_manual_bundle
from syspilot_manual.schemas import LibraryManual


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "examples" / "manual_mvp"


def _library_manual(manual_id: str, library_name: str) -> LibraryManual:
    return LibraryManual.model_validate(
        {
            "manual_id": manual_id,
            "manual_type": "library",
            "library_name": library_name,
            "library_summary": f"{library_name} tuning guidance.",
            "source_artifacts": [
                {"source_id": f"{manual_id}_doc", "path": f"docs/{manual_id}.md", "kind": "doc"}
            ],
            "feature_graph": [
                {
                    "feature_id": f"{manual_id}_threads",
                    "name": f"{library_name} threading",
                    "description": "Controls worker and thread scheduling.",
                    "resource_domains": ["cpu_threads"],
                    "keywords": ["OMP_NUM_THREADS"],
                    "source_refs": [f"{manual_id}_doc:chunk_001"],
                }
            ],
            "observability_map": [
                {
                    "obs_id": f"{manual_id}_obs",
                    "feature_id": f"{manual_id}_threads",
                    "question": f"How do we detect {library_name} thread contention?",
                    "priority": 1,
                    "symptoms": ["throughput plateau"],
                    "signals": [
                        {
                            "name": "context_switch_rate",
                            "tool": "perf",
                            "reason": "Detect contention",
                        }
                    ],
                    "source_refs": [f"{manual_id}_doc:chunk_001"],
                }
            ],
            "intervention_graph": [
                {
                    "intervention_id": f"{manual_id}_int",
                    "feature_id": f"{manual_id}_threads",
                    "when": ["context switches high"],
                    "actions": [
                        {
                            "kind": "env",
                            "name": "OMP_NUM_THREADS",
                            "suggestion": "set near physical cores",
                        }
                    ],
                    "resource_domains": ["cpu_threads"],
                    "expected_effect": "Reduce thread contention.",
                    "side_effects": ["Too few threads can underutilize cores"],
                    "conflicts_with": ["nthread"],
                    "source_refs": [f"{manual_id}_doc:chunk_001"],
                }
            ],
            "validation_recipes": [
                {
                    "validation_id": f"{manual_id}_val",
                    "for_intervention": f"{manual_id}_int",
                    "reruns": 3,
                    "metrics": ["throughput"],
                    "accept_if": "Median throughput improves.",
                    "rollback_if": "Throughput regresses.",
                    "source_refs": [f"{manual_id}_doc:chunk_001"],
                }
            ],
            "composition_hints": {
                "shared_resources": ["cpu_threads"],
                "conflict_patterns": [
                    {
                        "with_keywords": ["threads"],
                        "reason": "Thread knobs can compete.",
                        "resolution_hint": "Tune one thread family at a time.",
                        "source_refs": [f"{manual_id}_doc:chunk_001"],
                    }
                ],
            },
        }
    )


def _library_contract(
    contract_name: str, control_name: str = "OMP_NUM_THREADS"
) -> LibraryContractDSL:
    prefix = contract_name.lower()
    return LibraryContractDSL.model_validate(
        {
            "contract": contract_name,
            "version": "lpcl/v1",
            "type": "library",
            "components": [
                {
                    "id": f"{prefix}_threads",
                    "kind": "executor",
                    "attrs": {
                        "executor_kind": "thread_pool",
                        "workers": {"min": 1, "max": "host.cpu_slots"},
                    },
                    "evidence": [f"{contract_name}_doc:chunk_001"],
                }
            ],
            "uses": [
                {
                    "id": f"{prefix}_cpu_compute",
                    "resource": "cpu.compute",
                    "by": f"{prefix}_threads",
                    "phase": "compute",
                    "scope": "process",
                    "demand": {
                        "dimension": "workers",
                        "min": 1,
                        "max": "host.cpu_slots",
                    },
                    "share": "shared",
                    "evidence": [f"{contract_name}_doc:chunk_001"],
                }
            ],
            "assumptions": [
                {
                    "id": "solo_compute_pool",
                    "resource": "cpu.compute",
                    "predicate": {"kind": "not_nested"},
                    "evidence": [f"{contract_name}_doc:chunk_001"],
                }
            ],
            "controls": [
                {
                    "id": "omp_num_threads",
                    "name": control_name,
                    "kind": "env",
                    "type": "int",
                    "target": f"{prefix}_threads.workers",
                    "effect": {"kind": "set", "value": "input"},
                    "evidence": [f"{contract_name}_doc:chunk_001"],
                },
                {
                    "id": "omp_display_env",
                    "name": "OMP_DISPLAY_ENV",
                    "kind": "env",
                    "type": "enum",
                    "target": f"{prefix}_threads",
                    "effect": {"kind": "set", "value": "true"},
                    "evidence": [f"{contract_name}_doc:chunk_001"],
                },
                {
                    "id": "omp_display_affinity",
                    "name": "OMP_DISPLAY_AFFINITY",
                    "kind": "env",
                    "type": "bool",
                    "target": f"{prefix}_threads",
                    "effect": {"kind": "set", "value": True},
                    "evidence": [f"{contract_name}_doc:chunk_001"],
                },
            ],
            "signals": [
                {
                    "id": "ctx_switch_rate",
                    "metric": "os.context_switch_rate",
                    "target": "cpu.compute",
                    "indicates": "contention",
                    "probe": {"kind": "perf", "name": "perf_stat"},
                    "evidence": [f"{contract_name}_doc:chunk_001"],
                },
                {
                    "id": "display_env_signal",
                    "metric": "openmp.display_env",
                    "target": f"{prefix}_threads",
                    "indicates": "effective thread settings",
                    "probe": {"kind": "log", "name": "OMP_DISPLAY_ENV"},
                    "evidence": [f"{contract_name}_doc:chunk_001"],
                },
            ],
            "recipes": [
                {
                    "id": "reduce_oversubscription",
                    "for": "shared_cpu_compute_conflict",
                    "collect": ["ctx_switch_rate", "display_env_signal"],
                    "actions": [f"decrease {control_name}"],
                    "accept": "throughput improves",
                    "rollback": "throughput regresses",
                    "evidence": [f"{contract_name}_doc:chunk_001"],
                }
            ],
        }
    )


@dataclass
class _DummyResponse:
    content: str


class _DummyAgent:
    instances: list["_DummyAgent"] = []

    def __init__(self, config: object) -> None:
        self.config = config
        self.context = SimpleNamespace(loaded_manuals={})
        self.prompts: list[str] = []
        self.prompt_log_path: Path | None = None
        _DummyAgent.instances.append(self)

    def enable_llm_prompt_logging(self, path: Path) -> Path:
        self.prompt_log_path = path
        return path

    def load_manuals(self, names: list[str]) -> dict[str, bool]:
        for name in names:
            self.context.loaded_manuals[name] = "RAW MANUAL SHOULD NOT APPEAR"
        return {name: True for name in names}

    async def step(self, prompt: str) -> _DummyResponse:
        self.prompts.append(prompt)
        return _DummyResponse('{"summary":"none","manual_citation":"","evidence":[],"actions":[]}')


class _FailingSpawnAgent(_DummyAgent):
    async def step(self, prompt: str) -> _DummyResponse:
        self.prompts.append(prompt)
        raise LLMTimeoutError("timed out")


async def _raise_llm_timeout(self: object, prompt: str | None = None) -> _DummyResponse:
    _ = self
    _ = prompt
    raise LLMTimeoutError("timed out")


def _noop(*args: object, **kwargs: object) -> None:
    _ = args
    _ = kwargs


def _fake_run_command(*args: object, **kwargs: object) -> tuple[float, str, str, int]:
    _ = args
    _ = kwargs
    return 1.0, "ok", "", 0


async def _fake_live_profile(*args: object, **kwargs: object) -> dict[str, object]:
    _ = args
    _ = kwargs
    return {
        "pid": 9999,
        "wall_time": 0.01,
        "stdout": "",
        "stderr": "",
        "returncode": 0,
        "profiling_text": "context switches high",
        "profiling_json": {},
        "tool_calls_used": 0,
    }


class _FakeManualLLMClient:
    def __init__(self, config: object) -> None:
        self.config = config

    async def complete_prompt(self, prompt_name: str, payload: dict[str, object]) -> str:
        _ = prompt_name
        chunk = payload.get("chunk")
        chunk_id = chunk.get("chunk_id", "chunk") if isinstance(chunk, dict) else "chunk"
        return json.dumps(
            {
                "items": [
                    {
                        "evidence_id": f"ev_{str(chunk_id).replace(':', '_')}",
                        "library_name": payload.get("library_name", "openmp"),
                        "chunk_id": chunk_id,
                        "kind": "feature",
                        "title": "Chunk evidence",
                        "content": "Chunk contains performance-relevant guidance.",
                        "keywords": ["threads"],
                        "resource_domains": ["cpu_threads"],
                        "source_quote": "performance guidance",
                        "source_refs": [chunk_id],
                    }
                ]
            }
        )


class _FakeComposeManualLLMClient:
    def __init__(self, config: object) -> None:
        self.config = config

    async def complete_prompt(self, prompt_name: str, payload: dict[str, object]) -> str:
        if prompt_name == "compose_manuals.txt":
            selected = payload.get("selected_libraries", ["OpenMP", "XGBoost"])
            assert isinstance(selected, list)
            return json.dumps(
                {
                    "manual_id": "ignored",
                    "manual_type": "composed",
                    "app_name": "ignored",
                    "selected_libraries": selected,
                    "app_summary": "Coordinate shared thread resources across libraries.",
                    "shared_resources": [
                        {
                            "resource_domain": "cpu_threads",
                            "libraries": selected,
                            "note": "Both libraries touch CPU threads.",
                            "source_refs": ["openmp_doc:chunk_001", "xgboost_doc:chunk_001"],
                        }
                    ],
                    "conflict_summary": [
                        {
                            "conflict_id": "thread_conflict",
                            "title": "Competing CPU thread knobs",
                            "libraries": selected,
                            "resource_domains": ["cpu_threads"],
                            "reason": "Both libraries expose CPU thread tuning.",
                            "resolution_hint": "Tune one thread family at a time.",
                            "source_refs": ["openmp_doc:chunk_001", "xgboost_doc:chunk_001"],
                        }
                    ],
                    "combined_observability": [
                        {
                            "obs_id": "obs_threads",
                            "library": str(selected[0]),
                            "question": "How do we detect thread contention across libraries?",
                            "priority": 1,
                            "symptoms": ["throughput plateau"],
                            "signals": [
                                {
                                    "name": "context_switch_rate",
                                    "tool": "perf",
                                    "reason": "Detect contention",
                                }
                            ],
                            "source_refs": ["openmp_doc:chunk_001"],
                        }
                    ],
                    "combined_interventions": [
                        {
                            "intervention_id": "coord_threads",
                            "library": str(selected[0]),
                            "when": ["context switches high"],
                            "actions": [
                                {
                                    "kind": "config",
                                    "name": "nthread",
                                    "suggestion": "align thread settings conservatively",
                                }
                            ],
                            "resource_domains": ["cpu_threads"],
                            "expected_effect": "Reduce cross-library contention.",
                            "side_effects": ["Lower peak parallelism if reduced too far"],
                            "conflicts_with": ["OMP_NUM_THREADS"],
                            "source_refs": ["xgboost_doc:chunk_001"],
                        }
                    ],
                    "combined_validation": [
                        {
                            "validation_id": "val_threads",
                            "library": str(selected[0]),
                            "for_intervention": "coord_threads",
                            "reruns": 3,
                            "metrics": ["throughput"],
                            "accept_if": "Throughput improves.",
                            "rollback_if": "Throughput regresses.",
                            "source_refs": ["xgboost_doc:chunk_001"],
                        }
                    ],
                    "recommended_query_order": [
                        "shared_resources",
                        "combined_observability",
                        "conflict_summary",
                        "combined_interventions",
                        "combined_validation",
                    ],
                }
            )

        if prompt_name == "verify_manual.txt":
            manual = payload.get("manual")
            return json.dumps({"valid": True, "issues": [], "repaired_manual": manual})

        raise AssertionError(prompt_name)


def test_ingest_command_writes_chunk_manifest(tmp_path: Path) -> None:
    runner = CliRunner()
    artifact_root = tmp_path / "artifacts"
    result = runner.invoke(
        cli,
        [
            "ingest",
            "--library",
            "openmp",
            "--docs",
            str(FIXTURE_ROOT / "openmp.md"),
            "--artifact-root",
            str(artifact_root),
        ],
    )
    assert result.exit_code == 0
    assert (artifact_root / "chunks" / "openmp" / "chunks.jsonl").exists()


def test_extract_evidence_command_prints_progress_and_eta(tmp_path: Path) -> None:
    runner = CliRunner()
    artifact_root = tmp_path / "artifacts"
    ingest_result = runner.invoke(
        cli,
        [
            "ingest",
            "--library",
            "openmp",
            "--docs",
            str(FIXTURE_ROOT / "openmp.md"),
            "--artifact-root",
            str(artifact_root),
        ],
    )
    assert ingest_result.exit_code == 0

    with patch("syspilot_manual.llm_client.ManualLLMClient", _FakeManualLLMClient):
        result = runner.invoke(
            cli,
            [
                "extract-evidence",
                "--library",
                "openmp",
                "--artifact-root",
                str(artifact_root),
            ],
        )

    assert result.exit_code == 0
    assert "[extract-evidence] openmp: starting" in result.output
    assert "eta=" in result.output
    assert "finish~" in result.output
    assert '"evidence_items"' in result.output


def test_query_and_build_agent_context_commands(tmp_path: Path) -> None:
    runner = CliRunner()
    artifact_root = tmp_path / "artifacts"
    manual_root = tmp_path / "manual"
    write_manual_bundle(_library_manual("openmp", "OpenMP"), artifact_root, manual_root)

    query_result = runner.invoke(
        cli,
        [
            "query",
            "--manual",
            "openmp",
            "--section",
            "observability_map",
            "--keywords",
            "context_switch_rate",
            "--manual-root",
            str(manual_root),
        ],
    )
    assert query_result.exit_code == 0
    assert "How do we detect OpenMP thread contention?" in query_result.output

    context_result = runner.invoke(
        cli,
        [
            "build-agent-context",
            "--manuals",
            "openmp",
            "--goal",
            "improve throughput",
            "--symptom",
            "throughput plateau",
            "--manual-root",
            str(manual_root),
        ],
    )
    assert context_result.exit_code == 0
    payload = json.loads(context_result.output)
    assert payload["profile_first"]
    assert payload["candidate_actions"]


def test_query_command_uses_runtime_manual_discovery_by_default(tmp_path: Path) -> None:
    runner = CliRunner()
    artifact_root = tmp_path / "artifacts"
    manual_root = tmp_path / "manual"
    write_manual_bundle(_library_manual("openmp", "OpenMP"), artifact_root, manual_root)

    with patch("syspilot.cli._discover_manual_root", return_value=manual_root):
        result = runner.invoke(
            cli,
            [
                "query",
                "--manual",
                "openmp",
                "--section",
                "observability_map",
                "--keywords",
                "context_switch_rate",
            ],
        )

    assert result.exit_code == 0
    assert "How do we detect OpenMP thread contention?" in result.output


def test_compose_manuals_command_prints_progress_and_writes_bundle(tmp_path: Path) -> None:
    runner = CliRunner()
    artifact_root = tmp_path / "artifacts"
    manual_root = tmp_path / "manual"
    write_manual_bundle(_library_manual("openmp", "OpenMP"), artifact_root, manual_root)
    write_manual_bundle(_library_manual("xgboost", "XGBoost"), artifact_root, manual_root)

    with patch("syspilot_manual.llm_client.ManualLLMClient", _FakeComposeManualLLMClient):
        result = runner.invoke(
            cli,
            [
                "compose-manuals",
                "--app",
                "xgb_train",
                "--libraries",
                "openmp",
                "--libraries",
                "xgboost",
                "--artifact-root",
                str(artifact_root),
                "--manual-root",
                str(manual_root),
            ],
        )

    assert result.exit_code == 0
    assert "[compose-manuals] xgb_train/xgb_train_composed phase=started" in result.output
    assert "phase=verifying" in result.output
    assert "phase=completed" in result.output
    assert '"manual_id": "xgb_train_composed"' in result.output
    assert (manual_root / "xgb_train_composed" / "manual.json").exists()


def test_spawn_mode_prefers_structured_manual_context(tmp_path: Path) -> None:
    runner = CliRunner()
    artifact_root = tmp_path / "artifacts"
    manual_root = tmp_path / "manual"
    write_manual_bundle(_library_manual("openmp", "OpenMP"), artifact_root, manual_root)
    _DummyAgent.instances.clear()

    with (
        patch("syspilot.cli.SysPilotAgent", _DummyAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _fake_live_profile),
        patch("syspilot.cli._run_command", _fake_run_command),
        patch("syspilot.cli._discover_manual_root", return_value=manual_root),
    ):
        result = runner.invoke(
            cli,
            [
                "tune",
                "--ui-mode",
                "plain",
                "--command",
                "echo hi",
                "--manual",
                "openmp",
                "--max-iterations",
                "1",
                "--timeout",
                "1",
                "--no-parse-output",
            ],
        )

    assert result.exit_code == 0
    assert _DummyAgent.instances
    prompt = _DummyAgent.instances[0].prompts[0]
    assert '"candidate_actions"' in prompt
    assert "RAW MANUAL SHOULD NOT APPEAR" not in prompt


def test_spawn_mode_prefers_contract_context_before_manual(tmp_path: Path) -> None:
    runner = CliRunner()
    artifact_root = tmp_path / "artifacts"
    manual_root = tmp_path / "manual"
    contract_root = tmp_path / "contracts"
    write_contract_bundle("openmp", _library_contract("OpenMP"), artifact_root, contract_root)
    _DummyAgent.instances.clear()

    with (
        patch("syspilot.cli.SysPilotAgent", _DummyAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _fake_live_profile),
        patch("syspilot.cli._run_command", _fake_run_command),
        patch("syspilot.cli._discover_manual_root", return_value=manual_root),
        patch("syspilot.cli._discover_contract_root", return_value=contract_root),
    ):
        result = runner.invoke(
            cli,
            [
                "tune",
                "--ui-mode",
                "plain",
                "--command",
                "echo hi",
                "--manual",
                "openmp",
                "--max-iterations",
                "1",
                "--timeout",
                "1",
                "--no-parse-output",
            ],
        )

    assert result.exit_code == 0
    assert _DummyAgent.instances
    prompt = _DummyAgent.instances[0].prompts[0]
    assert '"candidate_actions"' in prompt
    assert '"os.context_switch_rate"' in prompt
    assert "RAW MANUAL SHOULD NOT APPEAR" not in prompt


def test_spawn_mode_accepts_composed_contract_without_manual_files(tmp_path: Path) -> None:
    runner = CliRunner()
    artifact_root = tmp_path / "artifacts"
    contract_root = tmp_path / "contracts"
    openmp = _library_contract("OpenMP", "OMP_NUM_THREADS")
    openblas = _library_contract("OpenBLAS", "OPENBLAS_NUM_THREADS")
    composed = compose_contracts("openmp_openblas", [openmp, openblas])
    write_contract_bundle("openmp", openmp, artifact_root, contract_root)
    write_contract_bundle("openblas", openblas, artifact_root, contract_root)
    write_contract_bundle("openmp_openblas_composed", composed, artifact_root, contract_root)
    _DummyAgent.instances.clear()

    with (
        patch("syspilot.cli.SysPilotAgent", _DummyAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _fake_live_profile),
        patch("syspilot.cli._run_command", _fake_run_command),
        patch("syspilot.cli._discover_manual_root", return_value=None),
        patch("syspilot.cli._discover_contract_root", return_value=contract_root),
    ):
        result = runner.invoke(
            cli,
            [
                "tune",
                "--ui-mode",
                "plain",
                "--command",
                "echo hi",
                "--manual",
                "openmp_openblas_composed",
                "--max-iterations",
                "1",
                "--timeout",
                "1",
                "--no-parse-output",
            ],
        )

    assert result.exit_code == 0
    assert _DummyAgent.instances
    assert _DummyAgent.instances[0].context.loaded_manuals == {}
    prompt = _DummyAgent.instances[0].prompts[0]
    assert '"candidate_actions"' in prompt
    assert '"conflicts"' in prompt
    assert "RAW MANUAL SHOULD NOT APPEAR" not in prompt


def test_spawn_mode_passes_contract_hypotheses_into_first_profiling_round(tmp_path: Path) -> None:
    runner = CliRunner()
    artifact_root = tmp_path / "artifacts"
    contract_root = tmp_path / "contracts"
    openmp = _library_contract("OpenMP", "OMP_NUM_THREADS")
    openblas = _library_contract("OpenBLAS", "OPENBLAS_NUM_THREADS")
    composed = compose_contracts("openmp_openblas", [openmp, openblas])
    write_contract_bundle("openmp", openmp, artifact_root, contract_root)
    write_contract_bundle("openblas", openblas, artifact_root, contract_root)
    write_contract_bundle("openmp_openblas_composed", composed, artifact_root, contract_root)
    _DummyAgent.instances.clear()

    expected_bundle = build_agent_context_bundle(
        ["openmp_openblas_composed"],
        goal="improve throughput",
        symptom=None,
        contract_root=contract_root,
    )
    expected_profiles = _extract_hypothesis_profiles(
        cast(dict[str, object], expected_bundle),
        max_profiles=None,
    )
    expected_ids = {str(item["hypothesis_id"]) for item in expected_profiles}
    expected_report_env = _collect_contract_report_env(expected_profiles)

    captured_profiles: list[list[dict[str, object]]] = []

    async def _capture_live_profile(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        profiles = cast(list[dict[str, object]], kwargs["hypothesis_profiles"])
        assert profiles
        captured_profiles.append(profiles)
        return {
            "pid": 9999,
            "wall_time": 0.01,
            "stdout": "",
            "stderr": "",
            "returncode": 0,
            "profiling_text": "context switches high",
            "profiling_json": {
                "contract_hypothesis_validation": {
                    "profiles": [
                        {"hypothesis_id": str(item["hypothesis_id"])} for item in profiles
                    ],
                    "report_env": _collect_contract_report_env(
                        cast(list[dict[str, object]], profiles)
                    ),
                }
            },
            "tool_calls_used": 0,
        }

    with (
        patch("syspilot.cli.SysPilotAgent", _DummyAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _capture_live_profile),
        patch("syspilot.cli._run_command", _fake_run_command),
        patch("syspilot.cli._discover_manual_root", return_value=None),
        patch("syspilot.cli._discover_contract_root", return_value=contract_root),
    ):
        result = runner.invoke(
            cli,
            [
                "tune",
                "--ui-mode",
                "plain",
                "--command",
                "echo hi",
                "--manual",
                "openmp_openblas_composed",
                "--max-iterations",
                "1",
                "--timeout",
                "1",
                "--no-parse-output",
            ],
        )

    assert result.exit_code == 0
    assert captured_profiles
    assert {str(item["hypothesis_id"]) for item in captured_profiles[0]} == expected_ids
    assert _collect_contract_report_env(captured_profiles[0]) == expected_report_env
    assert expected_report_env
    prompt = _DummyAgent.instances[0].prompts[0]
    assert "Contract Hypothesis Validation" in prompt
    assert next(iter(expected_ids)) in prompt


def test_build_spawn_profiling_env_enables_openmp_observability() -> None:
    env = _build_spawn_profiling_env({"OMP_NUM_THREADS": "8"})
    assert env is not None
    assert env["OMP_NUM_THREADS"] == "8"
    assert env["OMP_DISPLAY_ENV"] == "true"
    assert env["OMP_DISPLAY_AFFINITY"] == "true"


def test_contract_hypothesis_profiles_drive_report_env(tmp_path: Path) -> None:
    bundle = {
        "hypothesis_profiles": [
            {
                "hypothesis_id": "h_threads",
                "report_controls": [
                    {
                        "control": {
                            "name": "OMP_DISPLAY_ENV",
                            "activation_value": "true",
                        }
                    },
                    {
                        "control": {
                            "name": "OMP_DISPLAY_AFFINITY",
                            "activation_value": True,
                        }
                    },
                ],
            }
        ]
    }
    profiles = _extract_hypothesis_profiles(cast(dict[str, object], bundle))
    report_env = _collect_contract_report_env(profiles)

    assert profiles
    assert report_env == {"OMP_DISPLAY_ENV": "true", "OMP_DISPLAY_AFFINITY": "true"}


def test_parse_spawn_actions_preserves_hypothesis_refs() -> None:
    parsed = _parse_spawn_actions(
        json.dumps(
            {
                "summary": "focus next profiling round",
                "manual_citation": "use hypothesis evidence",
                "evidence": ["validation incomplete"],
                "hypothesis_refs": ["h_shared_cpu_compute_conflict_openblas_openmp_compute"],
                "actions": [],
            }
        )
    )

    assert parsed["valid"] is True
    assert parsed["hypothesis_refs"] == ["h_shared_cpu_compute_conflict_openblas_openmp_compute"]


def test_summarize_spawn_runtime_output_extracts_openmp_signals() -> None:
    summary = _summarize_spawn_runtime_output(
        """
        OPENMP DISPLAY ENVIRONMENT BEGIN
        OMP_PROC_BIND = 'SPREAD'
        OMP_PLACES = '{0},{1}'
        OPENMP DISPLAY ENVIRONMENT END
        Number of Threads requested = 24
        Number of Threads counted = 24
        thread 0 affinity 0
        """,
        "",
    )
    assert summary["openmp_display_env_present"] is True
    assert summary["threads_requested"] == 24
    assert summary["threads_counted"] == 24
    assert "OMP_PLACES" in str(summary["stdout_excerpt"])
    assert "affinity" in str(summary["affinity_excerpt"])


def test_analyze_command_returns_clear_llm_failure_message() -> None:
    runner = CliRunner()

    with (
        patch.object(SysPilotAgent, "step", _raise_llm_timeout),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
    ):
        result = runner.invoke(cli, ["analyze", "1234"])

    assert result.exit_code == 0
    assert "Analysis unavailable:" in result.output
    assert "timed out" in result.output


def test_spawn_mode_degrades_to_no_actions_when_llm_analysis_fails() -> None:
    runner = CliRunner()

    with (
        patch("syspilot.cli.SysPilotAgent", _FailingSpawnAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _fake_live_profile),
        patch("syspilot.cli._run_command", _fake_run_command),
    ):
        result = runner.invoke(
            cli,
            [
                "tune",
                "--ui-mode",
                "plain",
                "--command",
                "echo hi",
                "--max-iterations",
                "1",
                "--timeout",
                "1",
                "--no-parse-output",
            ],
        )

    assert result.exit_code == 0
    assert "Agent tuning analysis unavailable:" in result.output
    assert "recommended no actionable tuning" in result.output.lower()


def test_tune_mode_uses_repo_llm_config_even_with_yaml_config(tmp_path: Path) -> None:
    runner = CliRunner()
    _DummyAgent.instances.clear()
    repo_llm = _load_default_llm_config()
    config_path = tmp_path / "syspilot.yaml"
    config_path.write_text("monitoring:\n  interval_seconds: 1.0\n", encoding="utf-8")

    with (
        patch("syspilot.cli.SysPilotAgent", _DummyAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _fake_live_profile),
        patch("syspilot.cli._run_command", _fake_run_command),
    ):
        result = runner.invoke(
            cli,
            [
                "--config",
                str(config_path),
                "tune",
                "--ui-mode",
                "plain",
                "--command",
                "echo hi",
                "--max-iterations",
                "1",
                "--timeout",
                "1",
                "--no-parse-output",
            ],
        )

    assert result.exit_code == 0
    assert _DummyAgent.instances
    config = cast(SysPilotConfig, _DummyAgent.instances[0].config)
    llm_config = config.llm
    assert llm_config.provider == repo_llm.provider
    assert llm_config.model == repo_llm.model
    assert llm_config.base_url == repo_llm.base_url
    assert llm_config.timeout_seconds == repo_llm.timeout_seconds
    assert llm_config.max_retries == repo_llm.max_retries
