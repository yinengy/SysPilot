from textwrap import dedent

import pytest

from syspilot_contract.checker import dump_contract_yaml, validate_contract_text
from syspilot_contract.migration import migrate_contract_payload
from syspilot_contract.schemas import LibraryContract, SourceArtifact
from syspilot_contract.synthesize_contract import synthesize_library_contract
from syspilot_manual.schemas import EvidenceItem


def _openblas_contract_payload() -> dict[str, object]:
    return {
        "version": "lpcl/v1",
        "contract": "OpenBLAS",
        "type": "library",
        "components": [
            {
                "id": "blas_threads",
                "kind": "executor",
                "attrs": {
                    "executor_kind": "thread_pool",
                    "backend": "unknown",
                    "workers": {"min": 1, "max": "host.cpu_slots"},
                },
                "evidence": ["doc_001:chunk_001"],
            }
        ],
        "uses": [
            {
                "id": "blas_cpu_compute",
                "resource": "cpu.compute",
                "by": "blas_threads",
                "phase": "compute",
                "scope": "process",
                "demand": {"dimension": "workers", "min": 1, "max": "host.cpu_slots"},
                "share": "shared",
                "evidence": ["doc_001:chunk_001"],
            }
        ],
        "assumptions": [
            {
                "id": "avoid_nested_parallelism",
                "resource": "cpu.compute",
                "predicate": {"kind": "not_nested"},
                "evidence": ["doc_001:chunk_001"],
            }
        ],
        "controls": [
            {
                "id": "openblas_num_threads",
                "name": "OPENBLAS_NUM_THREADS",
                "kind": "env",
                "type": "int",
                "target": "blas_threads.workers",
                "effect": {"kind": "set", "value": "input"},
                "evidence": ["doc_001:chunk_001"],
            }
        ],
        "signals": [
            {
                "id": "openblas_num_threads_signal",
                "target": "blas_threads.workers",
                "metric": "active_thread_count",
                "indicates": "effective_worker_count",
                "probe": {"kind": "api", "name": "openblas_get_num_threads"},
                "evidence": ["doc_001:chunk_001"],
            }
        ],
        "recipes": [
            {
                "id": "avoid_nested_parallel_oversubscription",
                "for": "shared_cpu_compute_conflict",
                "collect": ["openblas_num_threads_signal"],
                "actions": ["set OPENBLAS_NUM_THREADS to 1"],
                "accept": "effective_worker_count == 1 and throughput_not_worse",
                "rollback": "throughput_worse",
                "evidence": ["doc_001:chunk_001"],
            }
        ],
    }


def test_contract_yaml_round_trip_validates() -> None:
    contract = LibraryContract.model_validate(_openblas_contract_payload())
    parsed, result = validate_contract_text(dump_contract_yaml(contract))

    assert result.valid is True
    assert parsed is not None
    assert getattr(parsed, "contract", None) == "OpenBLAS"


def test_checker_rejects_duplicate_yaml_keys() -> None:
    _, result = validate_contract_text(
        dedent(
            """
            contract: OpenBLAS
            contract: Duplicate
            version: lpcl/v1
            type: library
            components: []
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


def test_validator_rejects_worker_pool_resource() -> None:
    _, result = validate_contract_text(
        dedent(
            """
        version: lpcl/v1
        contract: OpenBLAS
        type: library
        components:
          - id: blas_threads
            kind: executor
            attrs: {}
            evidence: [doc_001:chunk_001]
        uses:
          - id: blas_worker_pool
            resource: worker.pool
            by: blas_threads
            phase: compute
            scope: process
            demand:
              dimension: workers
              min: 1
              max: host.cpu_slots
            share: shared
            evidence: [doc_001:chunk_001]
        assumptions: []
        controls: []
        signals: []
        recipes: []
        """
        )
    )

    assert result.valid is False
    assert any("worker.pool is a component, not a resource" in issue for issue in result.issues)


def test_validator_rejects_cpu_thread_resource() -> None:
    _, result = validate_contract_text(
        dedent(
            """
            version: lpcl/v1
            contract: OpenBLAS
            type: library
            components:
              - id: blas_threads
                kind: executor
                attrs: {}
                evidence: [doc_001:chunk_001]
            uses:
              - id: blas_cpu_threads
                resource: cpu.thread
                by: blas_threads
                phase: compute
                scope: process
                demand:
                  dimension: workers
                  min: 1
                  max: host.cpu_slots
                share: shared
                evidence: [doc_001:chunk_001]
            assumptions: []
            controls: []
            signals: []
            recipes: []
            """
        )
    )

    assert result.valid is False
    assert any(
        "Use resource cpu.compute with demand.dimension = workers" in issue
        for issue in result.issues
    )


def test_validator_rejects_missing_evidence() -> None:
    _, result = validate_contract_text(
        dedent(
            """
            version: lpcl/v1
            contract: OpenBLAS
            type: library
            components:
              - id: blas_threads
                kind: executor
                attrs: {}
                evidence: [doc_001:chunk_001]
            uses:
              - id: blas_cpu_compute
                resource: cpu.compute
                by: blas_threads
                phase: compute
                scope: process
                demand:
                  dimension: workers
                  min: 1
                  max: host.cpu_slots
                share: shared
                evidence: []
            assumptions: []
            controls: []
            signals: []
            recipes: []
            """
        )
    )

    assert result.valid is False


def test_validator_rejects_use_referencing_missing_component() -> None:
    _, result = validate_contract_text(
        dedent(
            """
            version: lpcl/v1
            contract: OpenBLAS
            type: library
            components: []
            uses:
              - id: blas_cpu_compute
                resource: cpu.compute
                by: nonexistent_executor
                phase: compute
                scope: process
                demand:
                  dimension: workers
                  min: 1
                  max: host.cpu_slots
                share: shared
                evidence: [doc_001:chunk_001]
            assumptions: []
            controls: []
            signals: []
            recipes: []
            """
        )
    )

    assert result.valid is False
    assert any(
        "references missing component nonexistent_executor" in issue for issue in result.issues
    )


def test_validator_rejects_recipe_collecting_missing_signal() -> None:
    _, result = validate_contract_text(
        dedent(
            """
            version: lpcl/v1
            contract: OpenBLAS
            type: library
            components:
              - id: blas_threads
                kind: executor
                attrs: {}
                evidence: [doc_001:chunk_001]
            uses:
              - id: blas_cpu_compute
                resource: cpu.compute
                by: blas_threads
                phase: compute
                scope: process
                demand:
                  dimension: workers
                  min: 1
                  max: host.cpu_slots
                share: shared
                evidence: [doc_001:chunk_001]
            assumptions: []
            controls: []
            signals: []
            recipes:
              - id: avoid_nested_parallel_oversubscription
                for: shared_cpu_compute_conflict
                collect: [missing_signal]
                actions: [set OPENBLAS_NUM_THREADS to 1]
                accept: effective_worker_count == 1 and throughput_not_worse
                rollback: throughput_worse
                evidence: [doc_001:chunk_001]
            """
        )
    )

    assert result.valid is False
    assert any("collects missing signals" in issue for issue in result.issues)


def test_validator_allows_backend_control_targeting_component_attribute() -> None:
    _, result = validate_contract_text(
        dedent(
            """
            version: lpcl/v1
            contract: OpenMPI
            type: library
            components:
              - id: ompi_comm_backend
                kind: backend
                attrs:
                  backend_name: ucx
                  backend_type: transport
                evidence: [doc_001:chunk_001]
            uses: []
            assumptions: []
            controls:
              - id: select_pml
                name: pml
                kind: cli
                type: string
                target: ompi_comm_backend.backend_name
                effect:
                  kind: set
                  value: input
                evidence: [doc_001:chunk_001]
            signals: []
            recipes: []
            """
        )
    )

    assert result.valid is True


def test_validator_rejects_missing_component_attribute_target() -> None:
    _, result = validate_contract_text(
        dedent(
            """
            version: lpcl/v1
            contract: OpenMPI
            type: library
            components:
              - id: ompi_comm_backend
                kind: backend
                attrs:
                  backend_name: ucx
                  backend_type: transport
                evidence: [doc_001:chunk_001]
            uses: []
            assumptions: []
            controls:
              - id: select_pml
                name: pml
                kind: cli
                type: string
                target: ompi_comm_backend.selectorz
                effect:
                  kind: set
                  value: input
                evidence: [doc_001:chunk_001]
            signals: []
            recipes: []
            """
        )
    )

    assert result.valid is False
    assert any(
        "targets nonexistent ompi_comm_backend.selectorz" in issue for issue in result.issues
    )


def test_old_schema_migration_maps_cpu_thread_to_cpu_compute_workers() -> None:
    migrated = migrate_contract_payload(
        {
            "version": "lpcl/v0",
            "contract": "OpenMP",
            "contract_type": "library",
            "source_artifacts": [],
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
                        "upper": "host.cpu_slots",
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
                    "type": "int",
                    "affects": ["threads"],
                    "effect": {"op": "increase", "resource": "cpu.thread", "strength": "strong"},
                    "scope": "process",
                    "source_refs": ["doc_001:chunk_001"],
                }
            ],
            "signals": [],
            "recipes": [],
        }
    )

    assert migrated["version"] == "lpcl/v1"
    assert migrated["uses"][0]["resource"] == "cpu.compute"  # type: ignore[index]
    assert migrated["uses"][0]["demand"]["dimension"] == "workers"  # type: ignore[index]
    assert str(migrated["controls"][0]["target"]).endswith(".workers")  # type: ignore[index]


class _YamlContractLLM:
    async def complete_prompt(self, prompt_name: str, payload: dict[str, object]) -> str:
        _ = payload
        if prompt_name == "verify_contract.txt":
            raise AssertionError("verify should not be needed")
        return dedent(
            """
            version: lpcl/v1
            contract: ignored
            type: library
            components:
              - id: omp_executor
                kind: executor
                attrs:
                  workers: OMP_NUM_THREADS
                evidence: [doc_001:chunk_001]
            uses:
              - id: omp_cpu_compute
                resource: cpu.compute
                by: omp_executor
                phase: compute
                scope: process
                demand:
                  dimension: workers
                  min: 1
                  max: OMP_NUM_THREADS
                share: shared
                evidence: [doc_001:chunk_001]
            assumptions:
              - id: placement_ok
                resource: cpu.placement
                predicate:
                  kind: placement_compatible
                  source: should_be_removed
                evidence: [doc_001:chunk_001]
            controls:
              - id: ctl_omp_num_threads
                name: OMP_NUM_THREADS
                kind: env
                type: string
                target: omp_executor.attrs.workers
                effect:
                  kind: set
                  value: OMP_NUM_THREADS
                  resource: cpu.compute
                evidence: [doc_001:chunk_001]
            signals:
              - id: sig_omp_display_env
                target: omp_executor
                metric: effective_runtime_settings
                indicates: configured settings
                probe:
                  kind: log
                  name: OMP_DISPLAY_ENV
                evidence: [doc_001:chunk_001]
            recipes:
              - id: recipe_shared_cpu_compute_conflict
                for: shared_cpu_compute_conflict
                collect: [sig_omp_display_env]
                actions: [Record effective settings]
                accept: stable runtime
                rollback: revert change
                evidence: [doc_001:chunk_001]
            """
        )


@pytest.mark.asyncio
async def test_synthesizer_accepts_yaml_and_canonicalizes_v1_fields() -> None:
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
        _YamlContractLLM(),
    )

    assert contract.controls[0].target == "omp_executor.workers"
    assert contract.controls[0].effect.model_dump(mode="json", exclude_none=True) == {
        "kind": "set",
        "value": "OMP_NUM_THREADS",
    }
    assert contract.assumptions[0].predicate == {"kind": "placement_compatible"}
