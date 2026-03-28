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
                    "tool": "runtime",
                    "source_refs": ["doc_001:chunk_001"],
                }
            ],
            "recipes": [
                {
                    "id": "reduce_oversubscription",
                    "hypothesis_kind": "shared_cpu_thread_conflict",
                    "collect": ["display_env"],
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
                tool: runtime
                source_refs: ["doc_001:chunk_001"]
            recipes:
              - id: reduce_oversubscription
                hypothesis_kind: shared_cpu_thread_conflict
                collect: [missing_signal]
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


class _FakeContractLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    async def complete_prompt(self, prompt_name: str, payload: dict[str, object]) -> str:
        _ = prompt_name
        _ = payload
        return self.response


@pytest.mark.asyncio
async def test_synthesizer_drops_incomplete_recipes_before_validation() -> None:
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
            "tool": "runtime",
            "source_refs": ["doc_001:chunk_001"]
          }],
          "recipes": [
            {
              "id": "drop_me",
              "hypothesis_kind": "shared_cpu_thread_conflict",
              "collect": [],
              "actions": ["decrease OMP_NUM_THREADS"],
              "accept_if": "throughput improves",
              "rollback_if": "throughput regresses",
              "source_refs": ["doc_001:chunk_001"]
            },
            {
              "id": "keep_me",
              "hypothesis_kind": "shared_cpu_thread_conflict",
              "collect": ["display_env"],
              "actions": ["decrease OMP_NUM_THREADS"],
              "accept_if": "throughput improves",
              "rollback_if": "throughput regresses",
              "source_refs": ["doc_001:chunk_001"]
            }
          ]
        }
        """
    )

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

    assert [recipe.id for recipe in contract.recipes] == ["keep_me"]
