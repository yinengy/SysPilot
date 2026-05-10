from textwrap import dedent
from pathlib import Path

import pytest

from syspilot_contract.checker import dump_contract_yaml, validate_contract_text
from syspilot_contract.query_store import ContractStore
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
                "effect": {"kind": "set", "value": "input", "direction": "unknown"},
                "interacts_with": ["openblas_num_threads_signal"],
                "evidence": ["doc_001:chunk_001"],
            }
        ],
        "signals": [
            {
                "id": "openblas_num_threads_signal",
                "target": "blas_threads.workers",
                "role": "config_check",
                "metric": "active_thread_count",
                "indicates": "effective_worker_count",
                "virtual_probe": "process.cpu.thread_count",
                "expected_direction_when_mitigated": "unknown",
                "evidence": ["doc_001:chunk_001"],
            }
        ],
        "recipes": [
            {
                "id": "avoid_nested_parallel_oversubscription",
                "for": "shared_cpu_compute_conflict",
                "collect": ["openblas_num_threads_signal"],
                "diagnose": [
                    {
                        "label": "confirm_effective_thread_count",
                        "signals": ["openblas_num_threads_signal"],
                    }
                ],
                "config_checks": ["openblas_num_threads_signal"],
                "outcomes": ["throughput_not_worse"],
                "guardrails": ["throughput_worse"],
                "actions": [
                    {
                        "label": "collapse_openblas_threads",
                        "summary": "set OPENBLAS_NUM_THREADS to 1",
                        "control": "openblas_num_threads",
                        "effect": "set",
                        "expected_changes": [
                            {
                                "signal": "openblas_num_threads_signal",
                                "expected_direction_when_mitigated": "unknown",
                            }
                        ],
                    }
                ],
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


def test_repo_contract_store_loads_all_tracked_bundles() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    contracts_root = repo_root / "contracts"
    tracked_bundles = sorted(
        bundle_dir.name
        for bundle_dir in contracts_root.iterdir()
        if bundle_dir.is_dir() and (bundle_dir / "contract.yaml").exists()
    )

    for bundle_name in tracked_bundles:
        load_result = validate_contract_text(
            (contracts_root / bundle_name / "contract.yaml").read_text(encoding="utf-8")
        )[1]
        assert load_result.valid is True, f"{bundle_name}: {load_result.issues}"

    store = ContractStore(contracts_root)

    assert store.list_contracts() == tracked_bundles


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
                diagnose:
                  - label: confirm_missing_signal
                    signals: [missing_signal]
                config_checks: [missing_signal]
                outcomes: [throughput_not_worse]
                guardrails: [throughput_worse]
                actions:
                  - label: reduce_threads
                    summary: set OPENBLAS_NUM_THREADS to 1
                evidence: [doc_001:chunk_001]
            """
        )
    )

    assert result.valid is False
    assert any("collects missing signals" in issue for issue in result.issues)


def test_validator_rejects_vague_metric_and_missing_virtual_probe() -> None:
    _, result = validate_contract_text(
        dedent(
            """
            version: lpcl/v1
            contract: DemoLib
            type: library
            components:
              - id: demo_executor
                kind: executor
                attrs: {}
                evidence: [doc_001:chunk_001]
            uses:
              - id: demo_cpu_compute
                resource: cpu.compute
                by: demo_executor
                phase: compute
                scope: process
                demand:
                  dimension: workers
                  min: 1
                  max: 2
                share: shared
                evidence: [doc_001:chunk_001]
            assumptions: []
            controls: []
            signals:
              - id: demo_signal
                target: demo_executor
                metric: metric
                indicates: generic status
                role: resource_signal
                virtual_probe: ''
                expected_direction_when_mitigated: decrease
                evidence: [doc_001:chunk_001]
            recipes: []
            """
        )
    )

    assert result.valid is False
    assert any("too vague" in issue for issue in result.issues)
    assert any("virtual_probe" in issue for issue in result.issues)


def test_validator_rejects_benchmark_wrapper_controls_in_library_contract() -> None:
    _, result = validate_contract_text(
        dedent(
            """
            version: lpcl/v1
            contract: kafka
            type: library
            components:
              - id: demo_executor
                kind: executor
                attrs:
                  workers: threads
                evidence: [doc_001:chunk_001]
            uses:
              - id: demo_cpu_compute
                resource: cpu.compute
                by: demo_executor
                phase: compute
                scope: process
                demand:
                  dimension: workers
                  min: 1
                  max: threads
                share: shared
                evidence: [doc_001:chunk_001]
            assumptions: []
            controls:
              - id: ctl_wrapper_threads
                name: BENCHMARK_NUM_THREADS
                kind: env
                type: int
                target: demo_executor.workers
                effect:
                  kind: set
                  value: BENCHMARK_NUM_THREADS
                  direction: unknown
                evidence: [doc_001:chunk_001]
            signals: []
            recipes: []
            """
        )
    )

    assert result.valid is False
    assert any("benchmark-wrapper variable" in issue for issue in result.issues)


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


def test_validator_rejects_legacy_contract_shape_without_migration() -> None:
    _, result = validate_contract_text(
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
        )
    )

    assert result.valid is False
    assert any("Unknown contract type" in issue or "type" in issue for issue in result.issues)


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
                target: omp_executor.workers
                effect:
                  kind: set
                  value: OMP_NUM_THREADS
                  direction: unknown
                evidence: [doc_001:chunk_001]
            signals:
              - id: sig_omp_display_env
                target: omp_executor
                role: config_check
                metric: effective_runtime_settings
                indicates: configured settings
                virtual_probe: runtime.config.startup_properties
                expected_direction_when_mitigated: unknown
                evidence: [doc_001:chunk_001]
            recipes:
              - id: recipe_shared_cpu_compute_conflict
                for: shared_cpu_compute_conflict
                collect: [sig_omp_display_env]
                diagnose:
                  - label: confirm_effective_settings
                    signals: [sig_omp_display_env]
                config_checks: [sig_omp_display_env]
                outcomes: [sig_omp_display_env]
                guardrails: [sig_omp_display_env]
                actions:
                  - label: record_effective_settings
                    summary: Record effective settings before choosing a thread count
                    control: ctl_omp_num_threads
                    effect: set
                    expected_changes:
                      - signal: sig_omp_display_env
                        expected_direction_when_mitigated: unknown
                search_space_hints: [bounded thread-count sweep]
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
        "direction": "unknown",
        "value": "OMP_NUM_THREADS",
    }
    assert contract.assumptions[0].predicate == {"kind": "placement_compatible"}
    assert contract.signals[0].virtual_probe == "runtime.config.startup_properties"
    assert contract.recipes[0].config_checks == ["sig_omp_display_env"]
