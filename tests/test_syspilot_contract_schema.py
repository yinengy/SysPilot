import json
from pathlib import Path
from textwrap import dedent

import pytest

from syspilot_contract.checker import dump_contract_yaml, validate_contract_text
from syspilot_contract.schemas import LibraryContract, SourceArtifact
from syspilot_contract.synthesize_contract import synthesize_library_contract
from syspilot_manual.schemas import EvidenceItem


def _library_contract() -> LibraryContract:
    return LibraryContract.model_validate(
        {
            "contract": "OpenMP",
            "version": "lpcl/v0",
            "contract_type": "library",
            "source_artifacts": [{"source_id": "doc_001", "path": "docs/openmp.md", "kind": "doc"}],
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
                        "knob": "OMP_NUM_THREADS",
                        "lower": 1,
                        "upper": "host.physical_cores",
                    },
                    "source_refs": ["doc_001:chunk_001"],
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
                    "source_refs": ["doc_001:chunk_001"],
                }
            ],
            "controls": [
                {
                    "id": "omp_num_threads",
                    "name": "OMP_NUM_THREADS",
                    "kind": "env",
                    "type": "int",
                    "affects": ["threads"],
                    "effect": {
                        "op": "increase",
                        "resource": "cpu.thread",
                        "strength": "strong",
                    },
                    "scope": "process",
                    "source_refs": ["doc_001:chunk_001"],
                }
            ],
            "signals": [
                {
                    "id": "display_env",
                    "metric": "openmp.display_env",
                    "resource": "cpu.thread",
                    "indicates": "effective thread settings",
                    "probe": {"kind": "runtime_env", "target": "OMP_DISPLAY_ENV"},
                    "source_refs": ["doc_001:chunk_001"],
                }
            ],
            "recipes": [
                {
                    "id": "reduce_oversubscription",
                    "hypothesis_kind": "shared_resource_conflict",
                    "collect": ["display_env"],
                    "control_refs": ["omp_num_threads"],
                    "actions": ["decrease OMP_NUM_THREADS"],
                    "accept_if": "throughput improves",
                    "rollback_if": "throughput regresses",
                    "source_refs": ["doc_001:chunk_001"],
                }
            ],
        }
    )


def test_contract_yaml_round_trip_validates() -> None:
    contract = _library_contract()
    parsed, result = validate_contract_text(dump_contract_yaml(contract))

    assert result.valid is True
    assert parsed is not None
    assert getattr(parsed, "contract", None) == "OpenMP"


def test_checker_rejects_duplicate_yaml_keys() -> None:
    _, result = validate_contract_text(
        dedent(
            """
            contract: OpenMP
            contract: Duplicate
            version: lpcl/v0
            contract_type: library
            source_artifacts: []
            uses: []
            assumptions: []
            controls: []
            signals: []
            recipes: []
            """
        )
    )

    assert result.valid is False
    assert any("duplicate key 'contract'" in issue for issue in result.issues)


def test_checker_reports_semantic_reference_errors() -> None:
    _, result = validate_contract_text(
        dedent(
            """
            version: lpcl/v0
            contract: OpenMP
            contract_type: library
            source_artifacts:
              - source_id: doc_001
                path: docs/openmp.md
                kind: doc
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
                  upper: host.physical_cores
                source_refs: ["doc_001:chunk_001"]
            assumptions: []
            controls:
              - id: omp_num_threads
                name: OMP_NUM_THREADS
                kind: env
                type: int
                affects: [missing_use]
                effect:
                  op: increase
                  resource: cpu.thread
                  strength: strong
                scope: process
                source_refs: ["doc_001:chunk_001"]
            signals:
              - id: display_env
                metric: openmp.display_env
                resource: cpu.thread
                indicates: effective thread settings
                probe:
                  kind: runtime_env
                  target: OMP_DISPLAY_ENV
                source_refs: ["doc_001:chunk_001"]
            recipes:
              - id: reduce_oversubscription
                hypothesis_kind: shared_resource_conflict
                collect: [missing_signal]
                control_refs: [missing_control]
                actions: ["decrease OMP_NUM_THREADS"]
                accept_if: throughput improves
                rollback_if: throughput regresses
                source_refs: ["doc_001:chunk_001"]
            """
        )
    )

    assert result.valid is False
    assert any("affects unknown uses" in issue for issue in result.issues)
    assert any("collects unknown signals" in issue for issue in result.issues)
    assert any("references unknown controls" in issue for issue in result.issues)


class _FakeContractLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    async def complete_prompt(self, prompt_name: str, payload: dict[str, object]) -> str:
        _ = payload
        if prompt_name == "verify_contract.txt":
            return json.dumps(
                {
                    "valid": False,
                    "issues": ["unable to repair"],
                    "repaired_contract": json.loads(self.response),
                }
            )
        return self.response


class _RepairingContractLLM:
    def __init__(self, repaired_contract: dict[str, object]) -> None:
        self.repaired_contract = repaired_contract
        self.prompt_names: list[str] = []

    async def complete_prompt(self, prompt_name: str, payload: dict[str, object]) -> str:
        self.prompt_names.append(prompt_name)
        if prompt_name == "synthesize_contract.txt":
            return json.dumps(
                {
                    "contract": "ignored",
                    "version": "lpcl/v0",
                    "contract_type": "library",
                    "source_artifacts": [
                        {"source_id": "doc_001", "path": "docs/openmp.md", "kind": "doc"}
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
                                "knob": "OMP_NUM_THREADS",
                                "lower": 1,
                                "upper": "visible processors",
                            },
                            "source_refs": ["doc_001:chunk_001"],
                        }
                    ],
                    "assumptions": [],
                    "controls": [
                        {
                            "id": "omp_num_threads",
                            "name": "OMP_NUM_THREADS",
                            "kind": "env",
                            "type": "integer",
                            "affects": ["missing_use"],
                            "effect": {
                                "op": "increase",
                                "resource": "cpu.thread",
                                "strength": "strong",
                            },
                            "scope": "process",
                            "source_refs": ["doc_001:chunk_001"],
                        }
                    ],
                    "signals": [
                        {
                            "id": "display_env",
                            "metric": "openmp.display_env",
                            "resource": "cpu.thread",
                            "indicates": "effective thread settings",
                            "tool": "OMP_DISPLAY_ENV",
                            "source_refs": ["doc_001:chunk_001"],
                        }
                    ],
                    "recipes": [
                        {
                            "id": "repair_me",
                            "hypothesis_kind": "shared_cpu_thread_conflict",
                            "collect": ["missing_signal"],
                            "control_refs": ["missing_control"],
                            "actions": ["decrease OMP_NUM_THREADS"],
                            "accept_if": "throughput improves",
                            "rollback_if": "throughput regresses",
                            "source_refs": ["doc_001:chunk_001"],
                        }
                    ],
                }
            )
        if prompt_name == "verify_contract.txt":
            invalid_sections = payload.get("invalid_sections")
            assert isinstance(invalid_sections, list)
            assert "controls" in invalid_sections or "recipes" in invalid_sections
            return json.dumps(
                {
                    "valid": False,
                    "issues": ["repaired invalid refs"],
                    "repaired_contract": self.repaired_contract,
                }
            )
        raise AssertionError(prompt_name)


@pytest.mark.asyncio
async def test_synthesizer_fails_when_incomplete_recipes_cannot_be_repaired() -> None:
    llm = _FakeContractLLM(
        """
        {
          "contract": "ignored",
          "version": "lpcl/v0",
          "contract_type": "library",
          "source_artifacts": [{"source_id": "doc_001", "path": "docs/openmp.md", "kind": "doc"}],
          "uses": [{
            "id": "threads",
            "resource": "cpu.thread",
            "phase": "compute",
            "scope": "process",
            "role": "consume",
            "sharing": "partitionable",
            "demand": {"kind": "scaled_by", "knob": "OMP_NUM_THREADS", "lower": 1, "upper": "host.physical_cores"},
            "source_refs": ["doc_001:chunk_001"]
          }],
          "assumptions": [],
          "controls": [{
            "id": "omp_num_threads",
            "name": "OMP_NUM_THREADS",
            "kind": "env",
            "type": "int",
            "affects": ["threads"],
            "effect": {"op": "increase", "resource": "cpu.thread", "strength": "strong"},
            "scope": "process",
            "source_refs": ["doc_001:chunk_001"]
          }],
          "signals": [{
            "id": "display_env",
            "metric": "openmp.display_env",
            "resource": "cpu.thread",
            "indicates": "effective thread settings",
            "tool": "OMP_DISPLAY_ENV",
            "source_refs": ["doc_001:chunk_001"]
          }],
          "recipes": [
            {
              "id": "drop_me",
              "hypothesis_kind": "shared_cpu_thread_conflict",
              "collect": [],
              "control_refs": [],
              "actions": ["decrease OMP_NUM_THREADS"],
              "accept_if": "throughput improves",
              "rollback_if": "throughput regresses",
              "source_refs": ["doc_001:chunk_001"]
            },
            {
              "id": "keep_me",
              "hypothesis_kind": "shared_cpu_thread_conflict",
              "collect": ["display_env"],
              "control_refs": [],
              "actions": ["decrease OMP_NUM_THREADS"],
              "accept_if": "throughput improves",
              "rollback_if": "throughput regresses",
              "source_refs": ["doc_001:chunk_001"]
            }
          ]
        }
        """
    )

    with pytest.raises(ValueError, match="Contract verification failed after repair attempts"):
        await synthesize_library_contract(
            "OpenMP",
            [SourceArtifact(source_id="doc_001", path="docs/openmp.md", kind="doc")],
            [
                EvidenceItem.model_validate(
                    {
                        "evidence_id": "ev_001",
                        "library_name": "OpenMP",
                        "chunk_id": "doc_001:chunk_001",
                        "kind": "intervention",
                        "title": "Tune threads",
                        "content": "Set OMP_NUM_THREADS carefully.",
                        "keywords": ["OMP_NUM_THREADS"],
                        "resource_domains": ["cpu_threads"],
                        "source_quote": "Set OMP_NUM_THREADS carefully.",
                        "source_refs": ["doc_001:chunk_001"],
                    }
                )
            ],
            llm,
        )


@pytest.mark.asyncio
async def test_synthesizer_repairs_invalid_contract_via_verify_prompt() -> None:
    repaired_contract = _library_contract().model_dump(mode="json")
    llm = _RepairingContractLLM(repaired_contract)

    contract = await synthesize_library_contract(
        "OpenMP",
        [SourceArtifact(source_id="doc_001", path="docs/openmp.md", kind="doc")],
        [
            EvidenceItem.model_validate(
                {
                    "evidence_id": "ev_001",
                    "library_name": "OpenMP",
                    "chunk_id": "doc_001:chunk_001",
                    "kind": "intervention",
                    "title": "Tune threads",
                    "content": "Set OMP_NUM_THREADS carefully.",
                    "keywords": ["OMP_NUM_THREADS"],
                    "resource_domains": ["cpu_threads"],
                    "source_quote": "Set OMP_NUM_THREADS carefully.",
                    "source_refs": ["doc_001:chunk_001"],
                }
            )
        ],
        llm,
    )

    assert llm.prompt_names == ["synthesize_contract.txt", "verify_contract.txt"]
    assert contract.controls[0].affects == ["threads"]
    assert contract.recipes[0].collect == ["display_env"]
    assert contract.signals[0].probe.target == "OMP_DISPLAY_ENV"
