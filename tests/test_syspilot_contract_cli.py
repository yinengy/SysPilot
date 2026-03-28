from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

from click.testing import CliRunner

from syspilot.cli import cli
from syspilot_contract.integration import write_contract_bundle
from syspilot_contract.schemas import LibraryContract


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "examples" / "manual_mvp"


def _library_contract(name: str = "OpenMP") -> LibraryContract:
    control_name = "OMP_NUM_THREADS" if name == "OpenMP" else "NUMACTL_THREADS"
    return LibraryContract.model_validate(
        {
            "contract": name,
            "version": "lpcl/v0",
            "contract_type": "library",
            "source_artifacts": [
                {"source_id": f"{name}_doc", "path": f"docs/{name}.md", "kind": "doc"}
            ],
            "uses": [
                {
                    "id": "threads",
                    "resource": "cpu.thread",
                    "phase": "compute",
                    "scope": "process",
                    "role": "consume",
                    "sharing": "partitionable",
                    "demand": {
                        "kind": "scaled_by",
                        "knob": control_name,
                        "lower": 1,
                        "upper": "host.physical_cores",
                    },
                    "source_refs": [f"{name}_doc:chunk_001"],
                }
            ],
            "assumptions": [
                {
                    "id": "solo_compute_pool",
                    "predicate": {
                        "kind": "max_competitors",
                        "resource": "cpu.thread",
                        "phase": "compute",
                        "value": 1,
                    },
                    "source_refs": [f"{name}_doc:chunk_001"],
                }
            ],
            "controls": [
                {
                    "id": f"{name.lower()}_threads",
                    "name": control_name,
                    "kind": "env",
                    "type": "int",
                    "affects": ["threads"],
                    "effect": {"op": "increase", "resource": "cpu.thread", "strength": "strong"},
                    "scope": "process",
                    "source_refs": [f"{name}_doc:chunk_001"],
                }
            ],
            "signals": [
                {
                    "id": f"{name.lower()}_signal",
                    "metric": f"{name.lower()}.runqueue_length",
                    "resource": "cpu.thread",
                    "indicates": "thread contention",
                    "probe": {"kind": "command", "target": "perf"},
                    "source_refs": [f"{name}_doc:chunk_001"],
                }
            ],
            "recipes": [
                {
                    "id": f"{name.lower()}_recipe",
                    "hypothesis_kind": "shared_resource_conflict",
                    "collect": [f"{name.lower()}_signal"],
                    "control_refs": [f"{name.lower()}_threads"],
                    "actions": [f"decrease {control_name}"],
                    "accept_if": "throughput improves",
                    "rollback_if": "throughput regresses",
                    "source_refs": [f"{name}_doc:chunk_001"],
                }
            ],
        }
    )


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
                _library_contract(str(payload.get("contract", "OpenMP"))).model_dump(mode="json")
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
            version: lpcl/v0
            """
        ),
        encoding="utf-8",
    )

    result = runner.invoke(cli, ["check-contract", str(invalid_path)])

    assert result.exit_code != 0
    assert "duplicate key 'contract'" in result.output


def test_compose_contracts_and_build_agent_context_commands(tmp_path: Path) -> None:
    runner = CliRunner()
    artifact_root = tmp_path / "artifacts"
    contract_root = tmp_path / "contracts"
    write_contract_bundle("openmp", _library_contract("OpenMP"), artifact_root, contract_root)
    write_contract_bundle("numactl", _library_contract("numactl"), artifact_root, contract_root)

    compose_result = runner.invoke(
        cli,
        [
            "compose-contracts",
            "--app",
            "minife",
            "--contracts",
            "openmp",
            "--contracts",
            "numactl",
            "--artifact-root",
            str(artifact_root),
            "--contract-root",
            str(contract_root),
        ],
    )

    assert compose_result.exit_code == 0
    assert "[compose-contracts] minife/minife_composed phase=started" in compose_result.output
    assert (contract_root / "minife_composed" / "contract.yaml").exists()
    assert not (contract_root / "minife_composed" / "contract.md").exists()

    context_result = runner.invoke(
        cli,
        [
            "build-agent-context",
            "--manuals",
            "openmp",
            "--manuals",
            "numactl",
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
