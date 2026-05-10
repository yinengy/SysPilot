from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

from click.testing import CliRunner

from syspilot.cli import cli
from syspilot_contract.checker import load_contract
from syspilot_contract.integration import write_contract_bundle
from syspilot_contract.schemas import ComposedContract, LibraryContract


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "examples" / "manual_mvp"


def _library_contract(name: str = "OpenMP") -> LibraryContract:
    control_name_map = {
        "OpenMP": "OMP_NUM_THREADS",
        "OpenBLAS": "OPENBLAS_NUM_THREADS",
        "XGBoost": "nthread",
        "scikit-learn": "SKLEARN_NUM_THREADS",
    }
    control_name = control_name_map.get(name, "THREADS")
    prefix = name.lower()
    return LibraryContract.model_validate(
        {
            "version": "lpcl/v1",
            "contract": name,
            "type": "library",
            "components": [
                {
                    "id": f"{prefix}_threads",
                    "kind": "executor",
                    "attrs": {
                        "executor_kind": "thread_pool",
                        "workers": {"min": 1, "max": "host.cpu_slots"},
                    },
                    "evidence": [f"{name}_doc:chunk_001"],
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
                    "evidence": [f"{name}_doc:chunk_001"],
                }
            ],
            "assumptions": [
                {
                    "id": f"{prefix}_not_nested",
                    "resource": "cpu.compute",
                    "predicate": {"kind": "not_nested"},
                    "evidence": [f"{name}_doc:chunk_001"],
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
                    "evidence": [f"{name}_doc:chunk_001"],
                }
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
                    "evidence": [f"{name}_doc:chunk_001"],
                },
                {
                    "id": f"{prefix}_ctx_switch_rate",
                    "target": "cpu.compute",
                    "role": "resource_signal",
                    "metric": "os.context_switch_rate",
                    "indicates": "oversubscription",
                    "virtual_probe": "benchmark.context_switch_rate",
                    "expected_direction_when_mitigated": "decrease",
                    "evidence": [f"{name}_doc:chunk_001"],
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
                    "evidence": [f"{name}_doc:chunk_001"],
                }
            ],
        }
    )


def _xgboost_contract() -> LibraryContract:
    return _library_contract("XGBoost")


def _sklearn_contract() -> LibraryContract:
    return _library_contract("scikit-learn")


class _FakeContractLLMClient:
    def __init__(self, config: object, prompt_dir: Path | None = None) -> None:
        self.config = config
        self.prompt_dir = prompt_dir

    async def complete_prompt(self, prompt_name: str, payload: dict[str, object]) -> str:
        if prompt_name == "extract_evidence.txt":
            chunk = payload.get("chunk")
            chunk_id = chunk.get("chunk_id", "chunk") if isinstance(chunk, dict) else "chunk"
            return json.dumps(
                {
                    "items": [
                        {
                            "evidence_id": f"ev_{str(chunk_id).replace(':', '_')}",
                            "library_name": payload.get("library_name", "OpenMP"),
                            "chunk_id": chunk_id,
                            "kind": "intervention",
                            "title": "Thread count controls oversubscription",
                            "content": "Tune OMP_NUM_THREADS carefully.",
                            "keywords": ["OMP_NUM_THREADS", "threads"],
                            "resource_domains": ["cpu_threads"],
                            "source_quote": "Adjust OMP_NUM_THREADS to match available cores.",
                            "source_refs": [chunk_id],
                        }
                    ]
                }
            )

        if prompt_name == "synthesize_contract.txt":
            return json.dumps(
                _library_contract(str(payload.get("contract", "OpenMP"))).model_dump(
                    mode="json", by_alias=True
                )
            )

        raise AssertionError(prompt_name)


def test_synthesize_contract_and_check_contract_commands(tmp_path: Path) -> None:
    runner = CliRunner()
    artifact_root = tmp_path / "artifacts"
    contract_root = tmp_path / "contracts"

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

    with patch("syspilot_manual.llm_client.ManualLLMClient", _FakeContractLLMClient):
        extract_result = runner.invoke(
            cli,
            [
                "extract-evidence",
                "--library",
                "openmp",
                "--artifact-root",
                str(artifact_root),
            ],
        )
        assert extract_result.exit_code == 0

        synth_result = runner.invoke(
            cli,
            [
                "synthesize-contract",
                "--library",
                "openmp",
                "--artifact-root",
                str(artifact_root),
                "--contract-root",
                str(contract_root),
            ],
        )

    assert synth_result.exit_code == 0
    assert "[synthesize-contract] openmp/openmp phase=started" in synth_result.output
    assert (contract_root / "openmp" / "contract.yaml").exists()
    assert (contract_root / "openmp" / "contract.lpcl").exists()
    lpcl_text = (contract_root / "openmp" / "contract.lpcl").read_text(encoding="utf-8")
    assert 'lpcl "lpcl/v1";' in lpcl_text
    assert "evidence = [" not in lpcl_text
    assert not (contract_root / "openmp" / "contract.md").exists()

    check_result = runner.invoke(
        cli,
        [
            "check-contract",
            str(contract_root / "openmp" / "contract.yaml"),
        ],
    )
    assert check_result.exit_code == 0
    assert '"valid": true' in check_result.output


def test_check_contract_command_reports_invalid_yaml(tmp_path: Path) -> None:
    runner = CliRunner()
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text(
        dedent(
            """
            contract: OpenMP
            contract: duplicate
            version: lpcl/v1
            """
        ),
        encoding="utf-8",
    )

    result = runner.invoke(cli, ["check-contract", str(invalid_path)])

    assert result.exit_code != 0
    assert "duplicate key 'contract'" in result.output


def test_check_contract_rejects_legacy_payloads_without_migration_path(tmp_path: Path) -> None:
    runner = CliRunner()
    legacy_path = tmp_path / "legacy.yaml"
    legacy_path.write_text(
        dedent(
            """
            version: lpcl/v0
            contract: LegacyOpenMP
            contract_type: library
            source_artifacts: []
            uses:
              - id: threads
                resource: cpu.thread
                phase: compute
                scope: process
                role: consume
                sharing: partitionable
                demand:
                  kind: scaled_by
                  knob: OMP_NUM_THREADS
                  lower: 1
                  upper: host.cpu_slots
                source_refs: [doc_001:chunk_001]
            assumptions: []
            controls: []
            signals: []
            recipes: []
            """
        ),
        encoding="utf-8",
    )

    result = runner.invoke(cli, ["check-contract", str(legacy_path)])

    assert result.exit_code != 0
    assert '"valid": false' in result.output


def test_check_contract_rejects_missing_toolbox_probe(tmp_path: Path) -> None:
    runner = CliRunner()
    contract_path = tmp_path / "missing_probe.yaml"
    contract_path.write_text(
        dedent(
            """
            version: lpcl/v1
            contract: MissingProbeDemo
            type: library
            components:
              - id: worker_pool
                kind: executor
                attrs:
                  workers: 4
                evidence: [doc]
            uses:
              - id: cpu_use
                resource: cpu.compute
                by: worker_pool
                phase: compute
                scope: process
                demand:
                  dimension: workers
                  min: 1
                  max: 4
                share: shared
                evidence: [doc]
            assumptions: []
            controls: []
            signals:
              - id: missing_probe_signal
                target: cpu.compute
                role: resource_signal
                metric: demo_metric
                indicates: demo condition
                virtual_probe: totally.missing_probe
                expected_direction_when_mitigated: decrease
                evidence: [doc]
            recipes: []
            """
        ),
        encoding="utf-8",
    )

    result = runner.invoke(cli, ["check-contract", str(contract_path)])

    assert result.exit_code != 0
    assert "totally.missing_probe" in result.output
    assert "missing toolbox script" in result.output


def test_compose_contracts_and_build_agent_context_commands(tmp_path: Path) -> None:
    runner = CliRunner()
    artifact_root = tmp_path / "artifacts"
    contract_root = tmp_path / "contracts"
    write_contract_bundle("openmp", _library_contract("OpenMP"), artifact_root, contract_root)
    write_contract_bundle("openblas", _library_contract("OpenBLAS"), artifact_root, contract_root)

    compose_result = runner.invoke(
        cli,
        [
            "compose-contracts",
            "--app",
            "minife",
            "--contracts",
            "openmp",
            "--contracts",
            "openblas",
            "--artifact-root",
            str(artifact_root),
            "--contract-root",
            str(contract_root),
        ],
    )

    assert compose_result.exit_code == 0
    assert "[compose-contracts] minife/minife_composed phase=started" in compose_result.output
    assert (contract_root / "minife_composed" / "contract.yaml").exists()
    assert (contract_root / "minife_composed" / "contract.lpcl").exists()
    assert "evidence = [" not in (contract_root / "minife_composed" / "contract.lpcl").read_text(
        encoding="utf-8"
    )
    assert not (contract_root / "minife_composed" / "contract.md").exists()

    context_result = runner.invoke(
        cli,
        [
            "build-agent-context",
            "--manuals",
            "openmp",
            "--manuals",
            "openblas",
            "--goal",
            "improve throughput",
            "--contract-root",
            str(contract_root),
        ],
    )
    assert context_result.exit_code == 0
    payload = json.loads(context_result.output)
    assert payload["profile_first"]
    assert payload["candidate_actions"]
    assert payload["conflicts"]
    assert payload["relevant_resources"] == ["cpu.compute"]


def test_build_agent_context_accepts_composed_contract_id(tmp_path: Path) -> None:
    runner = CliRunner()
    artifact_root = tmp_path / "artifacts"
    contract_root = tmp_path / "contracts"
    write_contract_bundle("openmp", _library_contract("OpenMP"), artifact_root, contract_root)
    write_contract_bundle("openblas", _library_contract("OpenBLAS"), artifact_root, contract_root)
    from syspilot_contract.compose_contracts import compose_contracts

    composed = compose_contracts(
        "openmp_openblas",
        [_library_contract("OpenMP"), _library_contract("OpenBLAS")],
    )
    write_contract_bundle("openmp_openblas_composed", composed, artifact_root, contract_root)

    result = runner.invoke(
        cli,
        [
            "build-agent-context",
            "--manuals",
            "openmp_openblas_composed",
            "--goal",
            "improve throughput",
            "--contract-root",
            str(contract_root),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["profile_first"]
    assert payload["candidate_actions"]
    assert payload["conflicts"]
    assert payload["likely_hypotheses"]


def test_compose_contracts_allows_duplicate_contract_refs(tmp_path: Path) -> None:
    runner = CliRunner()
    artifact_root = tmp_path / "artifacts"
    contract_root = tmp_path / "contracts"
    write_contract_bundle("openmp", _library_contract("OpenMP"), artifact_root, contract_root)

    result = runner.invoke(
        cli,
        [
            "compose-contracts",
            "--app",
            "openmp_openmp",
            "--contracts",
            "openmp",
            "--contracts",
            "openmp",
            "--artifact-root",
            str(artifact_root),
            "--contract-root",
            str(contract_root),
        ],
    )

    assert result.exit_code == 0
    contract = load_contract(contract_root / "openmp_openmp_composed" / "contract.yaml")
    assert isinstance(contract, ComposedContract)
    assert [item.instance_id for item in contract.contract_instances] == ["openmp_1", "openmp_2"]
    assert [item.contract_id for item in contract.contract_instances] == ["OpenMP", "OpenMP"]
    assert contract.hypotheses
    assert sorted(contract.hypotheses[0].contracts) == ["openmp_1", "openmp_2"]


def test_compose_contracts_and_build_agent_context_for_xgboost_sklearn_pair(
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    artifact_root = tmp_path / "artifacts"
    contract_root = tmp_path / "contracts"
    write_contract_bundle("xgboost", _xgboost_contract(), artifact_root, contract_root)
    write_contract_bundle("sklearn", _sklearn_contract(), artifact_root, contract_root)

    compose_result = runner.invoke(
        cli,
        [
            "compose-contracts",
            "--app",
            "demo1_xgboost_sklearn",
            "--contracts",
            "xgboost",
            "--contracts",
            "sklearn",
            "--artifact-root",
            str(artifact_root),
            "--contract-root",
            str(contract_root),
        ],
    )

    assert compose_result.exit_code == 0
    assert (
        "[compose-contracts] demo1_xgboost_sklearn/demo1_xgboost_sklearn_composed phase=started"
        in compose_result.output
    )
    assert (contract_root / "demo1_xgboost_sklearn_composed" / "contract.yaml").exists()
    assert (contract_root / "demo1_xgboost_sklearn_composed" / "contract.lpcl").exists()

    context_result = runner.invoke(
        cli,
        [
            "build-agent-context",
            "--manuals",
            "xgboost",
            "--manuals",
            "sklearn",
            "--goal",
            "improve throughput",
            "--contract-root",
            str(contract_root),
        ],
    )
    assert context_result.exit_code == 0
    payload = json.loads(context_result.output)
    assert payload["profile_first"]
    assert payload["candidate_actions"]
    assert payload["conflicts"]


def test_build_agent_context_accepts_xgboost_sklearn_composed_contract_id(tmp_path: Path) -> None:
    runner = CliRunner()
    artifact_root = tmp_path / "artifacts"
    contract_root = tmp_path / "contracts"
    write_contract_bundle("xgboost", _xgboost_contract(), artifact_root, contract_root)
    write_contract_bundle("sklearn", _sklearn_contract(), artifact_root, contract_root)
    from syspilot_contract.compose_contracts import compose_contracts

    composed = compose_contracts(
        "xgboost_sklearn",
        [_xgboost_contract(), _sklearn_contract()],
    )
    write_contract_bundle("xgboost_sklearn_composed", composed, artifact_root, contract_root)

    result = runner.invoke(
        cli,
        [
            "build-agent-context",
            "--manuals",
            "xgboost_sklearn_composed",
            "--goal",
            "improve throughput",
            "--contract-root",
            str(contract_root),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["profile_first"]
    assert payload["candidate_actions"]
    assert payload["conflicts"]
    assert payload["likely_hypotheses"]


def test_compile_lpcl_cli_can_include_evidence(tmp_path: Path) -> None:
    runner = CliRunner()
    input_path = tmp_path / "contract.yaml"
    output_path = tmp_path / "contract.lpcl"
    input_path.write_text(
        _library_contract("OpenMP").model_dump_json(by_alias=True, indent=2), encoding="utf-8"
    )

    result = runner.invoke(
        cli,
        [
            "compile-lpcl",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--include-evidence",
        ],
    )

    assert result.exit_code == 0
    assert "evidence = [" in output_path.read_text(encoding="utf-8")


def test_compile_lpcl_cli_hides_evidence_by_default(tmp_path: Path) -> None:
    runner = CliRunner()
    input_path = tmp_path / "contract.yaml"
    output_path = tmp_path / "contract.lpcl"
    input_path.write_text(
        _library_contract("OpenMP").model_dump_json(by_alias=True, indent=2), encoding="utf-8"
    )

    result = runner.invoke(
        cli,
        [
            "compile-lpcl",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert "evidence = [" not in output_path.read_text(encoding="utf-8")


def test_synthesize_contract_cli_can_include_evidence(tmp_path: Path) -> None:
    runner = CliRunner()
    artifact_root = tmp_path / "artifacts"
    contract_root = tmp_path / "contracts"

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

    with patch("syspilot_manual.llm_client.ManualLLMClient", _FakeContractLLMClient):
        extract_result = runner.invoke(
            cli,
            [
                "extract-evidence",
                "--library",
                "openmp",
                "--artifact-root",
                str(artifact_root),
            ],
        )
        assert extract_result.exit_code == 0

        synth_result = runner.invoke(
            cli,
            [
                "synthesize-contract",
                "--library",
                "openmp",
                "--artifact-root",
                str(artifact_root),
                "--contract-root",
                str(contract_root),
                "--include-evidence",
            ],
        )

    assert synth_result.exit_code == 0
    assert "evidence = [" in (contract_root / "openmp" / "contract.lpcl").read_text(
        encoding="utf-8"
    )
    assert "evidence = [" in (artifact_root / "contracts" / "openmp" / "contract.lpcl").read_text(
        encoding="utf-8"
    )


def test_compose_contracts_cli_can_include_evidence(tmp_path: Path) -> None:
    runner = CliRunner()
    artifact_root = tmp_path / "artifacts"
    contract_root = tmp_path / "contracts"
    write_contract_bundle("openmp", _library_contract("OpenMP"), artifact_root, contract_root)
    write_contract_bundle("openblas", _library_contract("OpenBLAS"), artifact_root, contract_root)

    result = runner.invoke(
        cli,
        [
            "compose-contracts",
            "--app",
            "minife",
            "--contracts",
            "openmp",
            "--contracts",
            "openblas",
            "--artifact-root",
            str(artifact_root),
            "--contract-root",
            str(contract_root),
            "--include-evidence",
        ],
    )

    assert result.exit_code == 0
    assert "evidence = [" in (contract_root / "minife_composed" / "contract.lpcl").read_text(
        encoding="utf-8"
    )
    assert "evidence = [" in (
        artifact_root / "composed_contracts" / "minife_composed" / "contract.lpcl"
    ).read_text(encoding="utf-8")
