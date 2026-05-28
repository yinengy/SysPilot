from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast
from unittest.mock import patch

from click.testing import CliRunner

from syspilot.cli import (
    _build_spawn_profiling_env,
    _collect_contract_report_env,
    _compute_objective_speedup_ratio,
    _extract_hypothesis_profiles,
    _extract_literal_numeric_metrics,
    _format_objective_speedup,
    _load_default_llm_config,
    _parse_spawn_actions,
    _recommendation_next_action_display,
    _run_live_profiling_pipeline,
    _summarize_spawn_runtime_output,
    cli,
)
from syspilot.core.agent import SysPilotAgent
from syspilot.core.config import SysPilotConfig
from syspilot.core.llm import LLMTimeoutError
from syspilot_contract.compose_contracts import compose_contracts
from syspilot_contract.integration import build_agent_context_bundle, write_contract_bundle
from syspilot_contract.schemas import LibraryContract as LibraryContractDSL
from syspilot_manual.integration import write_manual_bundle
from syspilot_manual.schemas import LibraryManual
from toolbox.shared.cgroup_v2 import (
    flatten_cgroup_memory_evidence as _flatten_cgroup_memory_evidence,
)
from toolbox.shared.cgroup_v2 import (
    normalize_cgroup_memory_evidence as _normalize_cgroup_memory_evidence,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "examples" / "manual_mvp"


def test_primary_objective_speedup_helpers_format_better_and_slower() -> None:
    assert _compute_objective_speedup_ratio(100.0, 90.0, "minimize") == 100.0 / 90.0
    assert _format_objective_speedup(100.0 / 90.0) == "1.11x (better)"
    assert _compute_objective_speedup_ratio(100.0, 125.0, "minimize") == 0.8
    assert _format_objective_speedup(0.8) == "0.80x (slower)"


def test_literal_metric_extraction_reads_full_json_output() -> None:
    long_prefix = "x" * 5000
    output = f'{long_prefix} {{"steps_per_sec": 4.25, "throughput": 136.0, "batch_size": 32}}'

    metrics = _extract_literal_numeric_metrics(output)

    assert metrics["steps_per_sec"] == 4.25
    assert metrics["throughput"] == 136.0
    assert "batch_size" not in metrics


def test_recommendation_next_action_display_formats_first_action_preview() -> None:
    assert (
        _recommendation_next_action_display(
            [
                {
                    "config_type": "env_var",
                    "parameter": "OMP_PROC_BIND",
                    "new_value": "close",
                },
                {
                    "config_type": "env_var",
                    "parameter": "OMP_PLACES",
                    "new_value": "cores",
                },
            ]
        )
        == "OMP_PROC_BIND=close (+1 more)"
    )


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
                    "id": f"{prefix}_not_nested",
                    "resource": "cpu.compute",
                    "predicate": {"kind": "not_nested"},
                    "evidence": [f"{contract_name}_doc:chunk_001"],
                }
            ],
            "controls": [
                {
                    "id": f"{prefix}_threads_control",
                    "name": control_name,
                    "kind": "env",
                    "type": "int",
                    "target": f"{prefix}_threads.workers",
                    "effect": {"kind": "set", "value": "input", "direction": "unknown"},
                    "interacts_with": [f"{prefix}_thread_count"],
                    "evidence": [f"{contract_name}_doc:chunk_001"],
                },
            ],
            "signals": [
                {
                    "id": f"{prefix}_thread_count",
                    "target": f"{prefix}_threads.workers",
                    "role": "config_check",
                    "metric": "active_thread_count",
                    "indicates": "effective_worker_count",
                    "virtual_probe": "process.cpu.thread_count",
                    "expected_direction_when_mitigated": "unknown",
                    "evidence": [f"{contract_name}_doc:chunk_001"],
                },
                {
                    "id": f"{prefix}_ctx_switch_rate",
                    "metric": "os.context_switch_rate",
                    "target": "cpu.compute",
                    "role": "resource_signal",
                    "indicates": "oversubscription",
                    "virtual_probe": "benchmark.context_switch_rate",
                    "expected_direction_when_mitigated": "decrease",
                    "evidence": [f"{contract_name}_doc:chunk_001"],
                },
            ],
            "recipes": [
                {
                    "id": f"{prefix}_recipe",
                    "for": "shared_cpu_compute_conflict",
                    "collect": [f"{prefix}_thread_count", f"{prefix}_ctx_switch_rate"],
                    "diagnose": [
                        {
                            "label": "confirm_runtime_and_contention_signals",
                            "signals": [f"{prefix}_thread_count", f"{prefix}_ctx_switch_rate"],
                        }
                    ],
                    "config_checks": [f"{prefix}_thread_count"],
                    "outcomes": ["throughput_not_worse"],
                    "guardrails": [f"{prefix}_ctx_switch_rate"],
                    "actions": [
                        {
                            "label": f"set_{prefix}_threads_to_one",
                            "summary": f"set {control_name} to 1",
                            "control": f"{prefix}_threads_control",
                            "effect": "set",
                            "expected_changes": [
                                {
                                    "signal": f"{prefix}_ctx_switch_rate",
                                    "expected_direction_when_mitigated": "decrease",
                                }
                            ],
                        }
                    ],
                    "evidence": [f"{contract_name}_doc:chunk_001"],
                }
            ],
        }
    )


@dataclass
class _DummyResponse:
    content: str


class _DummyAgent:
    instances: list[_DummyAgent] = []

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


class _ActionThenNoneSpawnAgent(_DummyAgent):
    def __init__(self, config: object) -> None:
        super().__init__(config)
        self._calls: int = 0

    async def step(self, prompt: str) -> _DummyResponse:
        self.prompts.append(prompt)
        self._calls += 1
        if self._calls == 1:
            return _DummyResponse(
                '{"summary":"try candidate","manual_citation":"rule",'
                '"evidence":["baseline timed out"],'
                '"actions":[{"config_type":"env_var","parameter":"OMP_PROC_BIND","new_value":"close",'
                '"reason":"improve locality","manual_citation":"rule"}]}'
            )
        return _DummyResponse('{"summary":"none","manual_citation":"","evidence":[],"actions":[]}')


class _FixedBudgetRepromptThenActionSpawnAgent(_DummyAgent):
    def __init__(self, config: object) -> None:
        super().__init__(config)
        self._calls = 0

    async def step(self, prompt: str) -> _DummyResponse:
        self.prompts.append(prompt)
        self._calls += 1
        if self._calls == 1:
            return _DummyResponse(
                '{"summary":"STOP:need_concrete_candidate","manual_citation":"","evidence":'
                '["need a concrete safe env-only change"],"actions":[]}'
            )
        return _DummyResponse(
            '{"summary":"apply wait policy","manual_citation":"rule","evidence":'
            '["runtime output shows idle waits"],"actions":[{"config_type":"env_var",'
            '"parameter":"OMP_WAIT_POLICY","new_value":"ACTIVE","reason":"reduce idle waits",'
            '"manual_citation":"rule"}]}'
        )


class _FixedBudgetAlwaysNoActionSpawnAgent(_DummyAgent):
    async def step(self, prompt: str) -> _DummyResponse:
        self.prompts.append(prompt)
        return _DummyResponse(
            '{"summary":"STOP:no_safe_candidate","manual_citation":"","evidence":'
            '["all safe candidates already exhausted"],"actions":[]}'
        )


async def _raise_llm_timeout(self: object, prompt: str | None = None) -> _DummyResponse:
    _ = self
    _ = prompt
    raise LLMTimeoutError("timed out")


def _noop(*args: object, **kwargs: object) -> None:
    _ = args
    _ = kwargs


async def _noop_async(*args: object, **kwargs: object) -> None:
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


async def _fake_timeout_live_profile(*args: object, **kwargs: object) -> dict[str, object]:
    _ = args
    _ = kwargs
    return {
        "pid": 9999,
        "wall_time": 1.0,
        "stdout": "",
        "stderr": "Command timed out during profiling pipeline",
        "returncode": -1,
        "timed_out": True,
        "profiling_text": "timeout profile",
        "profiling_json": {},
        "tool_calls_used": 0,
    }


class _ProfilingToolAgent:
    def __init__(self) -> None:
        self.context = SimpleNamespace(target_pid=777, loaded_manuals={})

    async def _execute_tool(self, tool_name: str, payload: object) -> object:
        _ = tool_name
        _ = payload
        return {}


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


class _CaptureEmitter:
    instances: ClassVar[list[_CaptureEmitter]] = []

    def __init__(
        self,
        *,
        run_id: str | None = None,
        capacity: int = 256,
        critical_reserve: int = 16,
    ) -> None:
        _ = capacity
        _ = critical_reserve
        self.run_id = run_id or "capture-run"
        self.records: list[dict[str, object]] = []
        _CaptureEmitter.instances.append(self)

    def publish(
        self,
        event_name: str,
        payload: dict[str, object] | None = None,
        *,
        event_type: str | None = None,
    ) -> bool:
        self.records.append(
            {
                "event": event_name,
                "payload": payload or {},
                "event_type": event_type or "event",
            }
        )
        return True


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


def test_spawn_mode_uses_contract_context_from_contract_option(tmp_path: Path) -> None:
    runner = CliRunner()
    artifact_root = tmp_path / "artifacts"
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
                "--contract",
                "openmp",
                "--max-iterations",
                "1",
                "--timeout",
                "1",
            ],
        )

    assert result.exit_code == 0
    assert _DummyAgent.instances
    prompt = _DummyAgent.instances[0].prompts[0]
    assert '"candidate_actions"' in prompt
    assert "Loaded Contract Context" in prompt


def test_spawn_mode_supports_contracts_and_manual_hints_together(tmp_path: Path) -> None:
    runner = CliRunner()
    artifact_root = tmp_path / "artifacts"
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
                "--contract",
                "openmp",
                "--manual",
                "parent process launches short-lived workers",
                "--max-iterations",
                "1",
                "--timeout",
                "1",
            ],
        )

    assert result.exit_code == 0
    assert _DummyAgent.instances
    prompt = _DummyAgent.instances[0].prompts[0]
    assert '"candidate_actions"' in prompt
    assert '"os.context_switch_rate"' in prompt
    assert "Application Manual Hints" in prompt
    assert "parent process launches short-lived workers" in prompt


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
                "--contract",
                "openmp_openblas_composed",
                "--max-iterations",
                "1",
                "--timeout",
                "1",
            ],
        )

    assert result.exit_code == 0
    assert _DummyAgent.instances
    assert _DummyAgent.instances[0].context.loaded_manuals == {}
    prompt = _DummyAgent.instances[0].prompts[0]
    assert '"candidate_actions"' in prompt
    assert '"conflicts"' in prompt
    assert "Loaded Contract Context" in prompt


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
                "--contract",
                "openmp_openblas_composed",
                "--max-iterations",
                "1",
                "--timeout",
                "1",
            ],
        )

    assert result.exit_code == 0
    assert captured_profiles
    assert {str(item["hypothesis_id"]) for item in captured_profiles[0]} == expected_ids
    assert _collect_contract_report_env(captured_profiles[0]) == expected_report_env
    prompt = _DummyAgent.instances[0].prompts[0]
    assert "Contract Hypothesis Validation" in prompt
    assert next(iter(expected_ids)) in prompt


def test_spawn_mode_manual_hints_do_not_generate_contract_hypotheses() -> None:
    runner = CliRunner()
    captured_profiles: list[list[dict[str, object]]] = []

    async def _capture_live_profile(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        captured_profiles.append(cast(list[dict[str, object]], kwargs["hypothesis_profiles"]))
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

    with (
        patch("syspilot.cli.SysPilotAgent", _DummyAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _capture_live_profile),
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
                "--manual",
                "manager process forks workers after warmup",
                "--max-iterations",
                "1",
                "--timeout",
                "1",
            ],
        )

    assert result.exit_code == 0
    assert captured_profiles == [[]]
    prompt = _DummyAgent.instances[-1].prompts[0]
    assert "Application Manual Hints" in prompt
    assert "manager process forks workers after warmup" in prompt
    assert "(No contract context provided)" in prompt


def test_spawn_mode_manual_hints_support_comma_splitting_and_escaped_commas() -> None:
    runner = CliRunner()

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
                "tune",
                "--ui-mode",
                "plain",
                "--command",
                "echo hi",
                "--manual",
                "manager process,workers use shared queue\\, backpressure enabled",
                "--manual",
                "latency spikes appear after warmup",
                "--max-iterations",
                "1",
                "--timeout",
                "1",
            ],
        )

    assert result.exit_code == 0
    prompt = _DummyAgent.instances[-1].prompts[0]
    assert "- manager process" in prompt
    assert "- workers use shared queue, backpressure enabled" in prompt
    assert "- latency spikes appear after warmup" in prompt


def test_spawn_mode_prompt_includes_excluded_config_controls() -> None:
    runner = CliRunner()
    _DummyAgent.instances.clear()

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
                "tune",
                "--ui-mode",
                "plain",
                "--command",
                "echo hi",
                "--exclude-config",
                "BENCHMARK_BLOSC_CNAME,BENCHMARK_BLOSC_SHUFFLE",
                "--max-iterations",
                "1",
                "--timeout",
                "1",
            ],
        )

    assert result.exit_code == 0
    prompt = _DummyAgent.instances[-1].prompts[0]
    assert "## Immutable Controls" in prompt
    assert (
        "Keep exactly these controls unchanged: "
        "BENCHMARK_BLOSC_CNAME, BENCHMARK_BLOSC_SHUFFLE"
    ) in prompt
    assert "Only these exact listed environment variables" in prompt
    assert "related controls with different names remain tunable" in prompt
    assert "do not treat same-library or related controls as immutable" in prompt


def test_spawn_mode_prompt_includes_effective_cgroup_memory_envelope() -> None:
    runner = CliRunner()
    _DummyAgent.instances.clear()

    async def _profiled_runs(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        _ = kwargs
        return {
            "pid": 9999,
            "wall_time": 1.0,
            "stdout": "runtime output",
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
            "profiling_text": "profile text",
            "profiling_json": {
                "cgroup_memory_evidence": {
                    "requested_memory_high": "1G",
                    "control_group": "/user.slice/demo.scope",
                    "effective_limit_bytes": 1073741824,
                    "current_bytes": 805306368,
                    "peak_bytes": 966367641,
                    "min_headroom_bytes": 107374183,
                    "sample_count": 8,
                    "memory_events": {"high": 5, "max": 0, "oom": 0, "oom_kill": 0},
                    "memory_pressure": {
                        "some": {"total": 21},
                        "full": {"total": 0},
                    },
                    "memory_stat": {
                        "pgfault": 100,
                        "pgmajfault": 2,
                        "pgscan": 3,
                        "pgsteal": 4,
                        "workingset_refault_anon": 5,
                        "workingset_refault_file": 6,
                    },
                    "swap_current_bytes": 0,
                    "status": "pressured",
                }
            },
            "tool_calls_used": 0,
        }

    with (
        patch("syspilot.cli.SysPilotAgent", _DummyAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _profiled_runs),
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
                "--memory-high",
                "1G",
                "--max-iterations",
                "1",
                "--timeout",
                "1",
            ],
        )

    assert result.exit_code == 0
    prompt = _DummyAgent.instances[-1].prompts[0]
    assert "Effective Memory Capacity" in prompt
    assert "memory.events: high=5" in prompt
    assert "memory.pressure.some.total=21" in prompt
    assert "Rule: When this block is present" in prompt


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


def test_parse_spawn_actions_allows_evidence_without_manual_citation() -> None:
    parsed = _parse_spawn_actions(
        json.dumps(
            {
                "summary": "runtime evidence supports env trial",
                "manual_citation": "",
                "evidence": ["score is DALI-limited and Torch has headroom"],
                "actions": [
                    {
                        "config_type": "command_prefix",
                        "new_value": [
                            "env",
                            "BENCHMARK_DALI_MPS_SHARE=70",
                            "BENCHMARK_TORCH_MPS_SHARE=30",
                        ],
                        "reason": "rebalance reversible env-only GPU share",
                        "manual_citation": "",
                    }
                ],
            }
        )
    )

    assert parsed["valid"] is True
    assert parsed["manual_citation"] == ""
    assert parsed["actions"]


def test_parse_spawn_actions_still_rejects_actions_without_evidence() -> None:
    parsed = _parse_spawn_actions(
        json.dumps(
            {
                "summary": "unsupported guess",
                "manual_citation": "",
                "evidence": [],
                "actions": [
                    {
                        "config_type": "env_var",
                        "parameter": "OMP_PROC_BIND",
                        "new_value": "close",
                        "reason": "guess",
                    }
                ],
            }
        )
    )

    assert parsed["valid"] is False
    assert parsed["error"] == "Actionable recommendations require non-empty evidence"
    assert parsed["actions"] == []


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


def test_normalize_and_flatten_cgroup_memory_evidence() -> None:
    evidence = _normalize_cgroup_memory_evidence(
        requested_memory_high="1G",
        cgroup_context={
            "selected_path": "/user.slice/demo.scope",
            "selected_pid": 4321,
            "selected_target_pids": [4321, 4322],
            "per_pid_paths": {"4321": "/user.slice/demo.scope"},
        },
        sample_result={
            "sampled_peak_bytes": 900,
            "min_headroom_bytes": 124,
            "sample_count": 4,
            "latest_snapshot": {
                "cgroup_root": "/sys/fs/cgroup/user.slice/demo.scope",
                "memory_high_bytes": 1024,
                "memory_max_bytes": None,
                "memory_current_bytes": 800,
                "memory_peak_bytes": 880,
                "memory_events": {"high": 7, "oom": 0, "oom_kill": 0},
                "memory_pressure": {
                    "some": {"avg10": 0.25, "total": 11},
                    "full": {"avg10": 0.0, "total": 0},
                },
                "memory_stat": {"pgfault": 15, "pgsteal": 3},
                "memory_swap_current_bytes": 64,
                "memory_swap_events": {"high": 1},
            },
        },
    )

    assert evidence["status"] == "pressured"
    assert evidence["peak_bytes"] == 900
    assert evidence["effective_limit_bytes"] == 1024
    flat = _flatten_cgroup_memory_evidence(cast(dict[str, object], evidence))
    assert flat["memory.events.high"] == 7
    assert flat["cgroup.memory.events.high"] == 7
    assert flat["memory.pressure.some.total"] == 11
    assert flat["memory.stat.pgfault"] == 15
    assert flat["memory.swap.current"] == 64


def test_normalize_cgroup_memory_evidence_keeps_zero_psi_run_healthy() -> None:
    evidence = _normalize_cgroup_memory_evidence(
        requested_memory_high="1G",
        cgroup_context={
            "selected_path": "/user.slice/demo.scope",
            "selected_pid": 4321,
            "selected_target_pids": [4321],
            "per_pid_paths": {"4321": "/user.slice/demo.scope"},
        },
        sample_result={
            "sampled_peak_bytes": 512,
            "min_headroom_bytes": 512,
            "sample_count": 2,
            "latest_snapshot": {
                "cgroup_root": "/sys/fs/cgroup/user.slice/demo.scope",
                "memory_high_bytes": 1024,
                "memory_max_bytes": None,
                "memory_current_bytes": 512,
                "memory_peak_bytes": 512,
                "memory_events": {"high": 0, "max": 0, "oom": 0, "oom_kill": 0},
                "memory_pressure": {
                    "some": {"avg10": 0.0, "avg60": 0.0, "avg300": 0.0, "total": 0},
                    "full": {"avg10": 0.0, "avg60": 0.0, "avg300": 0.0, "total": 0},
                },
                "memory_stat": {},
                "memory_swap_current_bytes": 0,
                "memory_swap_events": {},
            },
        },
    )

    assert evidence["status"] == "healthy"


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
            ],
        )

    assert result.exit_code == 0
    assert "Agent tuning analysis unavailable:" in result.output
    assert "recommended no actionable tuning" in result.output.lower()


def test_spawn_mode_primary_objective_resolves_throughput_intent() -> None:
    runner = CliRunner()

    async def _profiled_runs(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        _ = kwargs
        return {
            "pid": 9999,
            "wall_time": 1.0,
            "stdout": "ELAPSED_SEC=100\n",
            "stderr": "",
            "returncode": 0,
            "profiling_text": "baseline profile",
            "profiling_json": {},
            "tool_calls_used": 0,
        }

    async def _fake_extract_metrics(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        _ = kwargs
        return {"task_completions_per_sec": 1000.0}

    with (
        patch("syspilot.cli.SysPilotAgent", _DummyAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _profiled_runs),
        patch("syspilot.cli._extract_metrics_from_output", _fake_extract_metrics),
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
                "--primary-objective",
                "throughput",
            ],
        )

    assert result.exit_code == 0
    assert "Resolved primary objective metric: task_completions_per_sec (maximize)" in result.output
    assert "Baseline throughput: 1000" in result.output


def test_spawn_mode_primary_objective_resolves_typo_query_to_metric() -> None:
    runner = CliRunner()

    async def _profiled_runs(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        _ = kwargs
        return {
            "pid": 9999,
            "wall_time": 1.0,
            "stdout": "ELAPSED_SEC=100\n",
            "stderr": "",
            "returncode": 0,
            "profiling_text": "baseline profile",
            "profiling_json": {},
            "tool_calls_used": 0,
        }

    async def _fake_extract_metrics(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        _ = kwargs
        return {"elapsed_sec": 100.0}

    with (
        patch("syspilot.cli.SysPilotAgent", _DummyAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _profiled_runs),
        patch("syspilot.cli._extract_metrics_from_output", _fake_extract_metrics),
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
                "--primary-objective",
                "ELAPSED SED",
            ],
        )

    assert result.exit_code == 0
    assert "Resolved primary objective metric: elapsed_sec (minimize)" in result.output


def test_spawn_mode_primary_objective_requires_explicit_mode_when_resolved_metric_is_unknown(
) -> None:
    runner = CliRunner()

    async def _profiled_runs(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        _ = kwargs
        return {
            "pid": 9999,
            "wall_time": 1.0,
            "stdout": "CUSTOM_METRIC=123\n",
            "stderr": "",
            "returncode": 0,
            "profiling_text": "baseline profile",
            "profiling_json": {},
            "tool_calls_used": 0,
        }

    async def _fake_extract_metrics(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        _ = kwargs
        return {"custom_metric": 123.0}

    with (
        patch("syspilot.cli.SysPilotAgent", _DummyAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _profiled_runs),
        patch("syspilot.cli._extract_metrics_from_output", _fake_extract_metrics),
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
                "--primary-objective",
                "custom metric",
            ],
        )

    assert result.exit_code == 1
    assert (
        "Could not infer --primary-objective-mode after resolving 'custom metric'"
        in result.output
    )
    assert "custom_metric" in result.output


def test_spawn_mode_primary_objective_accepts_better_metric_even_if_wall_time_worse() -> None:
    runner = CliRunner()

    async def _profiled_runs(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        call_index = int(getattr(_profiled_runs, "calls", 0)) + 1
        setattr(_profiled_runs, "calls", call_index)
        if call_index == 1:
            return {
                "pid": 9999,
                "wall_time": 1.0,
                "stdout": "ELAPSED_SEC=100\nTASK_COMPLETIONS_PER_SEC=1000\n",
                "stderr": "",
                "returncode": 0,
                "profiling_text": "baseline profile",
                "profiling_json": {},
                "tool_calls_used": 0,
            }
        return {
            "pid": 10001,
            "wall_time": 1.2,
            "stdout": "ELAPSED_SEC=90\nTASK_COMPLETIONS_PER_SEC=950\n",
            "stderr": "",
            "returncode": 0,
            "profiling_text": "candidate profile",
            "profiling_json": {},
            "tool_calls_used": 0,
        }

    async def _fake_extract_metrics(*args: object, **kwargs: object) -> dict[str, object]:
        output = cast(str, args[1])
        if "ELAPSED_SEC=100" in output:
            return {"elapsed_sec": 100.0, "task_completions_per_sec": 1000.0}
        return {"elapsed_sec": 90.0, "task_completions_per_sec": 950.0}

    with (
        patch("syspilot.cli.SysPilotAgent", _ActionThenNoneSpawnAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _profiled_runs),
        patch("syspilot.cli._extract_metrics_from_output", _fake_extract_metrics),
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
                "--primary-objective",
                "wall time",
            ],
        )

    assert result.exit_code == 0
    assert "Resolved primary objective metric: elapsed_sec (minimize)" in result.output
    assert "New current wall time: 90 (+10.0% vs previous)" in result.output
    assert "Current wall time: 1.20s" in result.output
    assert "✓ Success! wall time improved to 1.11x (better) with actions:" in result.output
    assert "No tuned candidate outperformed baseline" not in result.output


def test_spawn_mode_primary_objective_rejects_candidate_missing_metric() -> None:
    runner = CliRunner()

    async def _profiled_runs(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        call_index = int(getattr(_profiled_runs, "calls", 0)) + 1
        setattr(_profiled_runs, "calls", call_index)
        if call_index == 1:
            return {
                "pid": 9999,
                "wall_time": 1.0,
                "stdout": "ELAPSED_SEC=100\n",
                "stderr": "",
                "returncode": 0,
                "profiling_text": "baseline profile",
                "profiling_json": {},
                "tool_calls_used": 0,
            }
        return {
            "pid": 10001,
            "wall_time": 0.8,
            "stdout": "TASK_COMPLETIONS_PER_SEC=1200\n",
            "stderr": "",
            "returncode": 0,
            "profiling_text": "candidate profile",
            "profiling_json": {},
            "tool_calls_used": 0,
        }

    async def _fake_extract_metrics(*args: object, **kwargs: object) -> dict[str, object]:
        output = cast(str, args[1])
        if "ELAPSED_SEC=100" in output:
            return {"elapsed_sec": 100.0}
        return {"task_completions_per_sec": 1200.0}

    with (
        patch("syspilot.cli.SysPilotAgent", _ActionThenNoneSpawnAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _profiled_runs),
        patch("syspilot.cli._extract_metrics_from_output", _fake_extract_metrics),
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
                "--primary-objective",
                "wall time",
            ],
    )

    assert result.exit_code == 0
    assert "Trial missing numeric primary objective" in result.output
    assert "wall time" in result.output
    assert (
        "No tuned candidate outperformed baseline; keeping baseline configuration"
        in result.output
    )


def test_spawn_mode_primary_objective_fails_when_baseline_metric_missing() -> None:
    runner = CliRunner()

    async def _profiled_runs(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        _ = kwargs
        return {
            "pid": 9999,
            "wall_time": 1.0,
            "stdout": "TASK_COMPLETIONS_PER_SEC=1200\n",
            "stderr": "",
            "returncode": 0,
            "profiling_text": "baseline profile",
            "profiling_json": {},
            "tool_calls_used": 0,
        }

    async def _fake_extract_metrics(*args: object, **kwargs: object) -> dict[str, object]:
        _ = kwargs
        output = cast(str, args[1])
        if "ELAPSED_SEC" in output:
            return {"elapsed_sec": 100.0}
        return {"task_completions_per_sec": 1200.0}

    with (
        patch("syspilot.cli.SysPilotAgent", _DummyAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _profiled_runs),
        patch("syspilot.cli._extract_metrics_from_output", _fake_extract_metrics),
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
                "--primary-objective",
                "memory usage",
            ],
    )

    assert result.exit_code == 1
    assert "Baseline output did not resolve primary objective query 'memory usage'" in result.output
    assert "numeric metric" in result.output


def test_spawn_mode_continues_after_timed_out_baseline() -> None:
    runner = CliRunner()
    _DummyAgent.instances.clear()

    with (
        patch("syspilot.cli.SysPilotAgent", _DummyAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _fake_timeout_live_profile),
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
            ],
        )

    assert result.exit_code == 0
    assert _DummyAgent.instances
    assert len(_DummyAgent.instances[0].prompts) == 1
    assert "Baseline run timed out; continuing with timeout as baseline" in result.output
    assert "Round 1 Step 1: Agent Tuning Analysis" in result.output
    assert "Agent recommended no actionable tuning" in result.output


def test_spawn_mode_runs_candidate_search_after_timed_out_baseline() -> None:
    runner = CliRunner()
    _ActionThenNoneSpawnAgent.instances = []  # type: ignore[attr-defined]
    profile_calls: list[dict[str, object]] = []

    async def _timeout_then_success(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        profile_calls.append(dict(kwargs))
        if len(profile_calls) == 1:
            return {
                "pid": 9999,
                "wall_time": 1.0,
                "stdout": "",
                "stderr": "Command timed out during profiling pipeline",
                "returncode": -1,
                "timed_out": True,
                "profiling_text": "timeout profile",
                "profiling_json": {},
                "tool_calls_used": 0,
            }
        return {
            "pid": 10001,
            "wall_time": 0.8,
            "stdout": "candidate ok",
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
            "profiling_text": "candidate profile",
            "profiling_json": {},
            "tool_calls_used": 0,
        }

    with (
        patch("syspilot.cli.SysPilotAgent", _ActionThenNoneSpawnAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _timeout_then_success),
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
            ],
        )

    assert result.exit_code == 0
    assert len(profile_calls) == 2
    assert "Baseline run timed out; continuing with timeout as baseline" in result.output
    assert "Trial 1/1" in result.output
    assert "Accepted candidate" in result.output


def test_fixed_budget_no_action_reprompts_until_candidate_materializes() -> None:
    _CaptureEmitter.instances.clear()
    _DummyAgent.instances.clear()
    runner = CliRunner()
    profile_calls: list[dict[str, object]] = []

    async def _profiled_runs(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        profile_calls.append(kwargs)
        if len(profile_calls) == 1:
            return {
                "pid": 9999,
                "wall_time": 1.0,
                "stdout": "baseline",
                "stderr": "",
                "returncode": 0,
                "profiling_text": "baseline profile",
                "profiling_json": {},
                "tool_calls_used": 0,
            }
        return {
            "pid": 10001,
            "wall_time": 0.8,
            "stdout": "candidate",
            "stderr": "",
            "returncode": 0,
            "profiling_text": "candidate profile",
            "profiling_json": {},
            "tool_calls_used": 0,
        }

    with (
        patch("syspilot.cli.TuneEventEmitter", _CaptureEmitter),
        patch("syspilot.cli.SysPilotAgent", _FixedBudgetRepromptThenActionSpawnAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _profiled_runs),
        patch("syspilot.cli._run_command", _fake_run_command),
    ):
        result = runner.invoke(
            cli,
            [
                "tune",
                "--ui-mode",
                "plain",
                "--command",
                "env OMP_PROC_BIND=close echo hi",
                "--exclude-config",
                "OMP_NUM_THREADS",
                "--fixed-budget",
                "1",
                "--max-iterations",
                "5",
                "--timeout",
                "1",
            ],
        )

    assert result.exit_code == 0
    assert len(profile_calls) == 2
    assert _DummyAgent.instances
    prompts = _DummyAgent.instances[-1].prompts
    assert len(prompts) == 2
    assert "re-prompting within fixed budget" in result.output
    assert "## Fixed-Budget Re-prompt" in prompts[1]
    assert "Remaining candidate-attempt budget: 1" in prompts[1]
    assert "No-action reason: STOP:need_concrete_candidate" in prompts[1]
    assert 'Protected/excluded controls: ["OMP_NUM_THREADS"]' in prompts[1]
    assert 'Untouched current controls (where known): ["OMP_PROC_BIND"]' in prompts[1]
    assert "Successful configs:" in prompts[1]
    assert "Evaluated/rejected candidates:" in prompts[1]
    comparison_payload = cast(
        dict[str, Any],
        next(
            record["payload"]
            for record in _CaptureEmitter.instances[-1].records
            if record["event"] == "result.comparison"
        ),
    )
    budget_payload = cast(dict[str, Any], comparison_payload["fixed_budget"])
    assert comparison_payload["stop_reason"] == "fixed_budget_exhausted"
    assert budget_payload["used"] == 1
    assert budget_payload["remaining"] == 0
    assert budget_payload["no_action_reprompts_used"] == 1


def test_fixed_budget_unfillable_no_action_stops_without_spending_trials() -> None:
    _CaptureEmitter.instances.clear()
    _DummyAgent.instances.clear()
    runner = CliRunner()

    with (
        patch("syspilot.cli.TuneEventEmitter", _CaptureEmitter),
        patch("syspilot.cli.SysPilotAgent", _FixedBudgetAlwaysNoActionSpawnAgent),
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
                "--fixed-budget",
                "2",
                "--max-iterations",
                "5",
                "--timeout",
                "1",
            ],
        )

    assert result.exit_code == 0
    assert _DummyAgent.instances
    prompts = _DummyAgent.instances[-1].prompts
    assert len(prompts) == 3
    run_payload = cast(
        dict[str, Any],
        next(
            record["payload"]
            for record in _CaptureEmitter.instances[-1].records
            if record["event"] == "run.finished"
        ),
    )
    budget_payload = cast(dict[str, Any], run_payload["fixed_budget"])
    assert run_payload["stop_reason"] == "unfillable_no_action"
    assert run_payload["total_candidate_trials_executed"] == 0
    assert budget_payload["used"] == 0
    assert budget_payload["remaining"] == 2
    assert budget_payload["no_action_reprompts_used"] == 2


def test_fixed_budget_baseline_failure_reports_hard_failure_stop_reason() -> None:
    _CaptureEmitter.instances.clear()
    runner = CliRunner()

    async def _failing_baseline(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        _ = kwargs
        return {
            "pid": 9999,
            "wall_time": 0.01,
            "stdout": "",
            "stderr": "boom",
            "returncode": 3,
            "timed_out": False,
            "profiling_text": "baseline failed",
            "profiling_json": {},
            "tool_calls_used": 0,
        }

    with (
        patch("syspilot.cli.TuneEventEmitter", _CaptureEmitter),
        patch("syspilot.cli.SysPilotAgent", _DummyAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _failing_baseline),
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
                "--fixed-budget",
                "2",
                "--max-iterations",
                "5",
                "--timeout",
                "1",
            ],
        )

    assert result.exit_code == 0
    run_payload = cast(
        dict[str, Any],
        next(
            record["payload"]
            for record in _CaptureEmitter.instances[-1].records
            if record["event"] == "run.finished"
        ),
    )
    assert run_payload["status"] == "failed"
    assert run_payload["fixed_budget_stop_reason"] == "hard_failure"
    budget_payload = cast(dict[str, Any], run_payload["fixed_budget"])
    assert budget_payload["used"] == 0
    assert budget_payload["remaining"] == 2


def test_spawn_mode_streams_candidate_child_output_in_plain_mode() -> None:
    runner = CliRunner()
    _ActionThenNoneSpawnAgent.instances = []  # type: ignore[attr-defined]
    profile_calls: list[dict[str, object]] = []

    async def _streaming_candidate_profile(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        profile_calls.append(dict(kwargs))
        callback = kwargs.get("live_output_callback")
        assert callable(callback)
        if len(profile_calls) == 2:
            callback("stdout", "[memoryhigh-runner] Progress poll 1: completes 100/1,000 (10.0%)\n")
            callback("stderr", "candidate stderr line\n")
        return {
            "pid": 10001,
            "wall_time": 0.8,
            "stdout": "candidate ok",
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
            "profiling_text": "candidate profile",
            "profiling_json": {},
            "tool_calls_used": 0,
        }

    with (
        patch("syspilot.cli.SysPilotAgent", _ActionThenNoneSpawnAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _streaming_candidate_profile),
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
            ],
        )

    assert result.exit_code == 0
    assert len(profile_calls) == 2
    assert "Progress poll 1: completes 100/1,000 (10.0%)" in result.output
    assert "candidate stderr line" in result.output


def test_spawn_mode_streams_baseline_child_output_in_plain_mode() -> None:
    runner = CliRunner()
    _DummyAgent.instances.clear()

    async def _streaming_baseline_profile(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        callback = kwargs.get("live_output_callback")
        assert callable(callback)
        callback("stdout", "baseline live line\n")
        return {
            "pid": 9999,
            "wall_time": 0.01,
            "stdout": "baseline live line\n",
            "stderr": "",
            "returncode": 0,
            "profiling_text": "baseline profile",
            "profiling_json": {},
            "tool_calls_used": 0,
        }

    with (
        patch("syspilot.cli.SysPilotAgent", _DummyAgent),
        patch("syspilot.cli._register_default_skills", _noop),
        patch("syspilot.cli._register_profiling_tools", _noop),
        patch("syspilot.cli._register_tuning_tools", _noop),
        patch("syspilot.cli._run_live_profiling_pipeline", _streaming_baseline_profile),
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
            ],
        )

    assert result.exit_code == 0
    assert "baseline live line" in result.output


def test_live_profiling_pipeline_restores_agent_target_pid() -> None:
    agent = _ProfilingToolAgent()

    line_seen = asyncio.Event()
    callback_lines: list[tuple[str, str]] = []

    class _FakeReader:
        def __init__(self, lines: list[bytes]) -> None:
            self._lines = list(lines)

        async def readline(self) -> bytes:
            if self._lines:
                return self._lines.pop(0)
            return b""

    class _FakeProc:
        pid = 1234
        returncode: int | None = None

        def __init__(self) -> None:
            self.stdout = _FakeReader([b"ok\n"])
            self.stderr = _FakeReader([])

        async def wait(self) -> int:
            await asyncio.wait_for(line_seen.wait(), timeout=0.2)
            self.returncode = 0
            return 0

        def kill(self) -> None:
            return None

    def _live_output_callback(stream_name: str, line: str) -> None:
        callback_lines.append((stream_name, line))
        line_seen.set()

    async def _run() -> None:
        with (
            patch("syspilot.cli.asyncio.create_subprocess_exec", return_value=_FakeProc()),
            patch("syspilot.cli.asyncio.sleep", new=_noop_async),
            patch(
                "syspilot.cli._discover_related_processes",
                return_value={"target_pids": [1234], "primary_pid": 1234, "workers": []},
            ),
        ):
            result = await _run_live_profiling_pipeline(
                cast(SysPilotAgent, cast(object, agent)),
                ["echo", "hi"],
                timeout=1,
                hypothesis_profiles=[],
                live_output_callback=_live_output_callback,
            )

        assert result["returncode"] == 0
        assert result["timed_out"] is False
        assert result["stdout"] == "ok\n"
        assert callback_lines == [("stdout", "ok\n")]
        assert agent.context.target_pid == 777

    asyncio.run(_run())


def test_live_profiling_pipeline_captures_cgroup_memory_evidence() -> None:
    agent = _ProfilingToolAgent()

    class _FakeReader:
        async def readline(self) -> bytes:
            return b""

    class _FakeProc:
        pid = 1234
        returncode: int | None = None

        def __init__(self) -> None:
            self.stdout = _FakeReader()
            self.stderr = _FakeReader()

        async def wait(self) -> int:
            self.returncode = 0
            return 0

        def kill(self) -> None:
            return None

    async def _fake_start_toolbox_probe_sessions(
        virtual_probes: list[str],
        payload: dict[str, object],
    ) -> dict[str, object]:
        _ = payload
        assert "cgroup.memory.events.high" in virtual_probes
        return {"cgroup.memory.events.high": {"probe": "session"}}

    async def _fake_finish_toolbox_probe_sessions(
        sessions: dict[str, object],
        payload: dict[str, object],
    ) -> dict[str, dict[str, object]]:
        _ = sessions
        _ = payload
        return {
            "cgroup.memory.events.high": {
                "observed": True,
                "source": "live_cgroup_v2",
                "value": 7,
                "flat": {
                    "memory.events.high": 7,
                    "memory.pressure.some.total": 11,
                },
                "structured": {
                    "control_group": "/user.slice/demo.scope",
                    "effective_limit_bytes": 1024,
                    "status": "pressured",
                    "peak_bytes": 900,
                    "min_headroom_bytes": 124,
                },
            }
        }

    async def _run() -> None:
        with (
            patch("syspilot.cli.asyncio.create_subprocess_exec", return_value=_FakeProc()),
            patch("syspilot.cli.asyncio.sleep", new=_noop_async),
            patch(
                "syspilot.cli._discover_related_processes",
                return_value={"target_pids": [1234], "primary_pid": 1234, "workers": []},
            ),
            patch("syspilot.cli.start_toolbox_probe_sessions", _fake_start_toolbox_probe_sessions),
            patch(
                "syspilot.cli.finish_toolbox_probe_sessions",
                _fake_finish_toolbox_probe_sessions,
            ),
        ):
            result = await _run_live_profiling_pipeline(
                cast(SysPilotAgent, cast(object, agent)),
                ["echo", "hi"],
                timeout=1,
                memory_high="1G",
                hypothesis_profiles=[],
            )

        profiling_json = cast(dict[str, object], result["profiling_json"])
        cgroup_memory = cast(dict[str, object], profiling_json["cgroup_memory_evidence"])
        refined_runtime_maps = cast(dict[str, object], profiling_json["refined_runtime_maps"])
        assert cgroup_memory["status"] == "pressured"
        assert cgroup_memory["effective_limit_bytes"] == 1024
        assert refined_runtime_maps["memory.events.high"] == 7
        assert refined_runtime_maps["memory.pressure.some.total"] == 11

    asyncio.run(_run())


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
