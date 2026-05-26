from textwrap import dedent
from pathlib import Path

import pytest

from syspilot_contract.checker import dump_contract_yaml, validate_contract_text
from syspilot_contract.query_store import ContractStore
from syspilot_contract.schemas import LibraryContract, SourceArtifact
from syspilot_contract.synthesize_contract import _invalid_sections, synthesize_library_contract
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


def test_validator_allows_benchmark_env_controls_in_library_contract() -> None:
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

    assert result.valid is True
    assert result.issues == []


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


class _RepairingContractLLM:
    def __init__(self, contract_name: str, control_name: str) -> None:
        self.contract_name = contract_name
        self.control_name = control_name
        self.verify_called = False

    async def complete_prompt(self, prompt_name: str, payload: dict[str, object]) -> str:
        if prompt_name == "synthesize_contract.txt":
            return dedent(
                f"""
                version: lpcl/v1
                contract: ignored
                type: library
                components:
                  - id: {self.contract_name.lower().replace('-', '_')}_threads
                    kind: executor
                    attrs:
                      workers: host.cpu_slots
                    evidence: [doc_001:chunk_001]
                uses:
                  - id: {self.contract_name.lower().replace('-', '_')}_cpu_compute
                    resource: cpu.compute
                    by: {self.contract_name.lower().replace('-', '_')}_threads
                    phase: compute
                    scope: process
                    demand:
                      dimension: workers
                      min: 1
                      max: host.cpu_slots
                    share: shared
                    evidence: [doc_001:chunk_001]
                assumptions:
                  - id: avoid_nested_parallelism
                    resource: cpu.compute
                    predicate:
                      kind: not_nested
                    evidence: [doc_001:chunk_001]
                controls:
                  - id: {self.contract_name.lower().replace('-', '_')}_threads_control
                    name: {self.control_name}
                    kind: api
                    type: int
                    target: {self.contract_name.lower().replace('-', '_')}_threads.workers
                    effect:
                      kind: set
                      value: input
                    evidence: [doc_001:chunk_001]
                signals:
                  - id: runtime_threads
                    target: {self.contract_name.lower().replace('-', '_')}_threads.workers
                    metric: active_thread_count
                    indicates: effective_worker_count
                    probe:
                      kind: os
                      name: process.cpu.thread_count
                    evidence: [doc_001:chunk_001]
                  - id: contention_signal
                    target: cpu.compute
                    metric: os.context_switch_rate
                    indicates: oversubscription
                    probe:
                      kind: benchmark
                      name: benchmark.context_switch_rate
                    evidence: [doc_001:chunk_001]
                recipes:
                  - id: repair_me
                    for: shared_cpu_compute_conflict
                    collect: [runtime_threads, contention_signal]
                    actions:
                      - decrease {self.control_name}
                    evidence: [doc_001:chunk_001]
                """
            )
        if prompt_name == "verify_contract.txt":
            self.verify_called = True
            invalid_sections = payload.get("invalid_sections")
            assert isinstance(invalid_sections, list)
            assert "signals" in invalid_sections
            assert "recipes" in invalid_sections
            return dedent(
                f"""
                valid: false
                issues:
                  - repaired stale probe structure and recipe blocks
                repaired_contract:
                  version: lpcl/v1
                  contract: {self.contract_name}
                  type: library
                  components:
                    - id: {self.contract_name.lower().replace('-', '_')}_threads
                      kind: executor
                      attrs:
                        workers: host.cpu_slots
                      evidence: [doc_001:chunk_001]
                  uses:
                    - id: {self.contract_name.lower().replace('-', '_')}_cpu_compute
                      resource: cpu.compute
                      by: {self.contract_name.lower().replace('-', '_')}_threads
                      phase: compute
                      scope: process
                      demand:
                        dimension: workers
                        min: 1
                        max: host.cpu_slots
                      share: shared
                      evidence: [doc_001:chunk_001]
                  assumptions:
                    - id: avoid_nested_parallelism
                      resource: cpu.compute
                      predicate:
                        kind: not_nested
                      evidence: [doc_001:chunk_001]
                  controls:
                    - id: {self.contract_name.lower().replace('-', '_')}_threads_control
                      name: {self.control_name}
                      kind: api
                      type: int
                      target: {self.contract_name.lower().replace('-', '_')}_threads.workers
                      effect:
                        kind: set
                        value: input
                        direction: unknown
                      interacts_with:
                        - runtime_threads
                      evidence: [doc_001:chunk_001]
                  signals:
                    - id: runtime_threads
                      target: {self.contract_name.lower().replace('-', '_')}_threads.workers
                      role: config_check
                      metric: active_thread_count
                      indicates: effective_worker_count
                      virtual_probe: process.cpu.thread_count
                      expected_direction_when_mitigated: unknown
                      evidence: [doc_001:chunk_001]
                    - id: contention_signal
                      target: cpu.compute
                      role: resource_signal
                      metric: os.context_switch_rate
                      indicates: oversubscription
                      virtual_probe: benchmark.context_switch_rate
                      expected_direction_when_mitigated: decrease
                      evidence: [doc_001:chunk_001]
                  recipes:
                    - id: repair_me
                      for: shared_cpu_compute_conflict
                      collect: [runtime_threads, contention_signal]
                      diagnose:
                        - label: confirm_runtime_and_contention_signals
                          signals: [runtime_threads, contention_signal]
                      config_checks: [runtime_threads]
                      outcomes: [throughput_not_worse]
                      guardrails: [contention_signal]
                      actions:
                        - label: set_threads_lower
                          summary: reduce thread parallelism
                          control: {self.contract_name.lower().replace('-', '_')}_threads_control
                          effect: set
                          expected_changes:
                            - signal: contention_signal
                              expected_direction_when_mitigated: decrease
                      search_space_hints: [bounded thread-count sweep]
                      evidence: [doc_001:chunk_001]
                """
            )
        raise AssertionError(prompt_name)


class _RetryingRepairContractLLM(_RepairingContractLLM):
    def __init__(self, contract_name: str, control_name: str) -> None:
        super().__init__(contract_name, control_name)
        self.verify_attempts = 0

    async def complete_prompt(self, prompt_name: str, payload: dict[str, object]) -> str:
        if prompt_name == "verify_contract.txt":
            self.verify_attempts += 1
            if self.verify_attempts == 1:
                return '{"valid": true, "issues": [], "repaired_contract": null}'
        return await super().complete_prompt(prompt_name, payload)


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


class _AttrsTargetContractLLM:
    async def complete_prompt(self, prompt_name: str, payload: dict[str, object]) -> str:
        _ = prompt_name
        _ = payload
        return dedent(
            """
            version: lpcl/v1
            contract: ignored
            type: library
            components:
              - id: xgboost_core
                kind: executor
                attrs:
                  device: cpu
                  executor:
                    workers: nthread
                evidence: [doc_001:chunk_001]
            uses:
              - id: xgboost_cpu_compute
                resource: cpu.compute
                by: xgboost_core
                phase: compute
                scope: process
                demand:
                  dimension: workers
                  min: 1
                  max: host.cpu_slots
                share: shared
                evidence: [doc_001:chunk_001]
            assumptions:
              - id: avoid_nested_parallelism
                resource: cpu.compute
                predicate:
                  kind: not_nested
                evidence: [doc_001:chunk_001]
            controls:
              - id: ctl_nthread
                name: nthread
                kind: api
                type: int
                target: xgboost_core.attrs.pool.workers
                effect:
                  kind: set
                  value: input
                  direction: unknown
                evidence: [doc_001:chunk_001]
              - id: ctl_device
                name: device
                kind: api
                type: enum
                target: xgboost_core.attrs.device
                effect:
                  kind: set
                  value: input
                  direction: unknown
                evidence: [doc_001:chunk_001]
            signals:
              - id: sig_thread_count
                target: xgboost_core.attrs.executor.workers
                role: config_check
                metric: active_thread_count
                indicates: effective_worker_count
                virtual_probe: process.cpu.thread_count
                expected_direction_when_mitigated: unknown
                evidence: [doc_001:chunk_001]
            recipes:
              - id: recipe_threads
                for: shared_cpu_compute_conflict
                collect: [sig_thread_count]
                diagnose:
                  - label: confirm_runtime_threads
                    signals: [sig_thread_count]
                config_checks: [sig_thread_count]
                outcomes: [throughput_not_worse]
                guardrails: [sig_thread_count]
                actions:
                  - label: set_threads
                    summary: set nthread explicitly
                    control: ctl_nthread
                    effect: set
                    expected_changes:
                      - signal: sig_thread_count
                        expected_direction_when_mitigated: unknown
                evidence: [doc_001:chunk_001]
            """
        )


@pytest.mark.asyncio
async def test_synthesizer_canonicalizes_component_attrs_targets() -> None:
    contract = await synthesize_library_contract(
        "XGBoost",
        [SourceArtifact(source_id="doc_001", path="docs/xgboost.md", kind="doc")],
        [
            EvidenceItem.model_validate(
                {
                    "evidence_id": "ev_001",
                    "library_name": "XGBoost",
                    "chunk_id": "doc_001:chunk_001",
                    "kind": "intervention",
                    "title": "Tune XGBoost thread count",
                    "content": "Set nthread carefully.",
                    "keywords": ["nthread"],
                    "resource_domains": ["cpu_threads"],
                    "source_quote": "Set nthread carefully.",
                    "source_refs": ["doc_001:chunk_001"],
                }
            )
        ],
        _AttrsTargetContractLLM(),
    )

    assert contract.controls[0].target == "xgboost_core.executor.workers"
    assert contract.controls[1].target == "xgboost_core.device"
    assert contract.signals[0].target == "xgboost_core.executor.workers"


class _PrefixedTargetContractLLM:
    async def complete_prompt(self, prompt_name: str, payload: dict[str, object]) -> str:
        _ = prompt_name
        _ = payload
        return dedent(
            """
            version: lpcl/v1
            contract: ignored
            type: library
            components:
              - id: joblib_parallel
                kind: executor
                attrs:
                  default_backend: loky
                  workers: host.cpu_slots
                evidence: [doc_001:chunk_001]
            uses:
              - id: joblib_cpu_compute
                resource: cpu.compute
                by: joblib_parallel
                phase: compute
                scope: process
                demand:
                  dimension: workers
                  min: 1
                  max: host.cpu_slots
                share: shared
                evidence: [doc_001:chunk_001]
            assumptions:
              - id: avoid_nested_parallelism
                resource: cpu.compute
                predicate:
                  kind: not_nested
                evidence: [doc_001:chunk_001]
            controls:
              - id: ctl_n_jobs
                name: n_jobs
                kind: api
                type: int
                target: use.joblib_cpu_compute.demand.max
                effect:
                  kind: set
                  value: input
                  direction: unknown
                evidence: [doc_001:chunk_001]
              - id: ctl_backend
                name: backend
                kind: api
                type: enum
                target: executor.joblib_parallel.default_backend
                effect:
                  kind: set
                  value: loky
                  direction: unknown
                evidence: [doc_001:chunk_001]
            signals:
              - id: sig_config_n_jobs
                target: ctl_n_jobs
                role: config_check
                metric: joblib.config.n_jobs
                indicates: configured outer worker count
                virtual_probe: runtime.config.startup_properties
                expected_direction_when_mitigated: stable
                evidence: [doc_001:chunk_001]
              - id: sig_config_backend
                target: executor.joblib_parallel.default_backend
                role: config_check
                metric: joblib.config.backend
                indicates: configured joblib backend
                virtual_probe: runtime.config.startup_properties
                expected_direction_when_mitigated: stable
                evidence: [doc_001:chunk_001]
            recipes:
              - id: recipe_threads
                for: shared_cpu_compute_conflict
                collect: [sig_config_n_jobs, sig_config_backend]
                diagnose:
                  - label: confirm_joblib_parallelism
                    signals: [sig_config_n_jobs, sig_config_backend]
                config_checks: [sig_config_n_jobs, sig_config_backend]
                outcomes: [sig_config_n_jobs]
                guardrails: [sig_config_backend]
                actions:
                  - label: bound_outer_parallelism
                    summary: set n_jobs explicitly
                    control: ctl_n_jobs
                    effect: set
                    expected_changes:
                      - signal: sig_config_n_jobs
                        expected_direction_when_mitigated: stable
                evidence: [doc_001:chunk_001]
            """
        )


@pytest.mark.asyncio
async def test_synthesizer_canonicalizes_prefixed_component_and_use_targets() -> None:
    contract = await synthesize_library_contract(
        "scikit-learn",
        [SourceArtifact(source_id="doc_001", path="docs/sklearn.md", kind="doc")],
        [
            EvidenceItem.model_validate(
                {
                    "evidence_id": "ev_001",
                    "library_name": "scikit-learn",
                    "chunk_id": "doc_001:chunk_001",
                    "kind": "intervention",
                    "title": "Tune sklearn joblib parallelism",
                    "content": "Set n_jobs and backend intentionally.",
                    "keywords": ["n_jobs", "backend"],
                    "resource_domains": ["cpu_threads"],
                    "source_quote": "Set n_jobs and backend intentionally.",
                    "source_refs": ["doc_001:chunk_001"],
                }
            )
        ],
        _PrefixedTargetContractLLM(),
    )

    assert contract.controls[0].target == "joblib_cpu_compute.demand.max"
    assert contract.controls[1].target == "joblib_parallel.default_backend"
    assert contract.signals[0].target == "joblib_cpu_compute.demand.max"
    assert contract.signals[1].target == "joblib_parallel.default_backend"


@pytest.mark.asyncio
async def test_synthesizer_adds_workers_attr_for_worker_control_targeting_wrong_executor_attr() -> None:
    class _MissingWorkersAttrContractLLM:
        async def complete_prompt(self, prompt_name: str, payload: dict[str, object]) -> str:
            _ = prompt_name
            _ = payload
            return dedent(
                """
                version: lpcl/v1
                contract: ignored
                type: library
                components:
                  - id: hybrid_executor
                    kind: executor
                    attrs:
                      hybrid_runtime: psmp
                    evidence: [doc_001:chunk_001]
                uses:
                  - id: use_cpu_threads_compute
                    resource: cpu.compute
                    by: hybrid_executor
                    phase: compute
                    scope: process
                    demand:
                      dimension: workers
                      min: 1
                      max: host.cpu_slots
                    share: shared
                    evidence: [doc_001:chunk_001]
                assumptions:
                  - id: avoid_nested_parallelism
                    resource: cpu.compute
                    predicate:
                      kind: not_nested
                    evidence: [doc_001:chunk_001]
                controls:
                  - id: ctl_omp_num_threads
                    name: OMP_NUM_THREADS
                    kind: env
                    type: int
                    target: hybrid_executor.hybrid_runtime
                    effect:
                      kind: set
                      value: input
                      direction: unknown
                    evidence: [doc_001:chunk_001]
                signals:
                  - id: sig_thread_count
                    target: ctl_omp_num_threads
                    role: config_check
                    metric: active_thread_count
                    indicates: effective_worker_count
                    virtual_probe: process.cpu.thread_count
                    expected_direction_when_mitigated: unknown
                    evidence: [doc_001:chunk_001]
                recipes:
                  - id: recipe_threads
                    for: shared_cpu_compute_conflict
                    collect: [sig_thread_count]
                    diagnose:
                      - label: confirm_runtime_threads
                        signals: [sig_thread_count]
                    config_checks: [sig_thread_count]
                    outcomes: [throughput_not_worse]
                    guardrails: [sig_thread_count]
                    actions:
                      - label: set_threads
                        summary: set OMP_NUM_THREADS explicitly
                        control: ctl_omp_num_threads
                        effect: set
                        expected_changes:
                          - signal: sig_thread_count
                            expected_direction_when_mitigated: unknown
                    evidence: [doc_001:chunk_001]
                """
            )

    contract = await synthesize_library_contract(
        "CP2K",
        [SourceArtifact(source_id="doc_001", path="docs/cp2k.md", kind="doc")],
        [
            EvidenceItem.model_validate(
                {
                    "evidence_id": "ev_001",
                    "library_name": "CP2K",
                    "chunk_id": "doc_001:chunk_001",
                    "kind": "intervention",
                    "title": "Tune OMP_NUM_THREADS",
                    "content": "Set OMP_NUM_THREADS carefully.",
                    "keywords": ["OMP_NUM_THREADS"],
                    "resource_domains": ["cpu_threads"],
                    "source_quote": "Set OMP_NUM_THREADS carefully.",
                    "source_refs": ["doc_001:chunk_001"],
                }
            )
        ],
        _MissingWorkersAttrContractLLM(),
    )

    assert contract.components[0].attrs["workers"] == "OMP_NUM_THREADS"
    assert contract.controls[0].target == "hybrid_executor.workers"
    assert contract.signals[0].target == "hybrid_executor.workers"


@pytest.mark.asyncio
async def test_synthesizer_canonicalizes_gromacs_cli_worker_targets_to_workers_attr() -> None:
    class _NestedWorkerCliContractLLM:
        async def complete_prompt(self, prompt_name: str, payload: dict[str, object]) -> str:
            _ = prompt_name
            _ = payload
            return dedent(
                """
                version: lpcl/v1
                contract: ignored
                type: library
                components:
                  - id: mdrun_executor
                    kind: executor
                    attrs:
                      workers:
                        total_threads: -nt
                        mpi_ranks: -ntmpi
                        omp_threads: -ntomp
                    evidence: [doc_001:chunk_001]
                uses:
                  - id: use_cpu_threads_compute
                    resource: cpu.compute
                    by: mdrun_executor
                    phase: compute
                    scope: process
                    demand:
                      dimension: workers
                      min: 1
                      max: host.cpu_slots
                    share: shared
                    evidence: [doc_001:chunk_001]
                assumptions:
                  - id: avoid_nested_parallelism
                    resource: cpu.compute
                    predicate:
                      kind: not_nested
                    evidence: [doc_001:chunk_001]
                controls:
                  - id: ctl_ntomp
                    name: -ntomp
                    kind: cli
                    type: int
                    target: mdrun_executor.workers.omp_threads
                    effect:
                      kind: set
                      value: input
                      direction: unknown
                    evidence: [doc_001:chunk_001]
                signals:
                  - id: sig_runtime_threads
                    target: ctl_ntomp
                    role: config_check
                    metric: active_thread_count
                    indicates: effective_worker_count
                    virtual_probe: process.cpu.thread_count
                    expected_direction_when_mitigated: unknown
                    evidence: [doc_001:chunk_001]
                recipes:
                  - id: recipe_threads
                    for: shared_cpu_compute_conflict
                    collect: [sig_runtime_threads]
                    diagnose:
                      - label: confirm_runtime_threads
                        signals: [sig_runtime_threads]
                    config_checks: [sig_runtime_threads]
                    outcomes: [throughput_not_worse]
                    guardrails: [sig_runtime_threads]
                    actions:
                      - label: set_threads
                        summary: set -ntomp explicitly
                        control: ctl_ntomp
                        effect: set
                        expected_changes:
                          - signal: sig_runtime_threads
                            expected_direction_when_mitigated: unknown
                    evidence: [doc_001:chunk_001]
                """
            )

    contract = await synthesize_library_contract(
        "gromacs",
        [SourceArtifact(source_id="doc_001", path="docs/gromacs.md", kind="doc")],
        [
            EvidenceItem.model_validate(
                {
                    "evidence_id": "ev_001",
                    "library_name": "gromacs",
                    "chunk_id": "doc_001:chunk_001",
                    "kind": "intervention",
                    "title": "Tune GROMACS thread flags",
                    "content": "Set -ntomp carefully.",
                    "keywords": ["-ntomp"],
                    "resource_domains": ["cpu_threads"],
                    "source_quote": "Set -ntomp carefully.",
                    "source_refs": ["doc_001:chunk_001"],
                }
            )
        ],
        _NestedWorkerCliContractLLM(),
    )

    assert contract.controls[0].target == "mdrun_executor.workers"
    assert contract.signals[0].target == "mdrun_executor.workers"


def test_invalid_sections_detects_control_and_signal_keywords_from_checker_errors() -> None:
    sections = _invalid_sections(
        [
            "Value error, Control ctl_n_jobs targets nonexistent executor.workers.; Signal sig_cfg_n_jobs targets nonexistent ctl_n_jobs."
        ]
    )

    assert sections == ["controls", "signals"]


@pytest.mark.parametrize(
    ("contract_name", "control_name"),
    [("XGBoost", "nthread"), ("scikit-learn", "n_jobs")],
)
@pytest.mark.asyncio
async def test_synthesizer_repairs_stale_probe_and_recipe_shapes_for_new_libraries(
    contract_name: str,
    control_name: str,
) -> None:
    llm = _RepairingContractLLM(contract_name, control_name)
    contract = await synthesize_library_contract(
        contract_name,
        [SourceArtifact(source_id="doc_001", path=f"docs/{contract_name}.md", kind="doc")],
        [
            EvidenceItem.model_validate(
                {
                    "evidence_id": "ev_001",
                    "library_name": contract_name,
                    "chunk_id": "doc_001:chunk_001",
                    "kind": "intervention",
                    "title": "Coordinate parallel worker counts",
                    "content": "Reduce oversubscription by lowering outer or inner worker counts.",
                    "keywords": [control_name, "threads", "oversubscription"],
                    "resource_domains": ["cpu_threads"],
                    "source_quote": "Reduce oversubscription by lowering worker counts.",
                    "source_refs": ["doc_001:chunk_001"],
                }
            )
        ],
        llm,
    )

    assert llm.verify_called is True
    assert contract.contract == contract_name
    assert all(signal.virtual_probe for signal in contract.signals)
    assert contract.signals[0].virtual_probe == "process.cpu.thread_count"
    assert contract.signals[1].virtual_probe == "benchmark.context_switch_rate"
    assert contract.recipes[0].diagnose
    assert contract.recipes[0].config_checks == ["runtime_threads"]
    assert contract.recipes[0].guardrails == ["contention_signal"]
    assert contract.recipes[0].actions[0].label == "set_threads_lower"


@pytest.mark.asyncio
async def test_synthesizer_retries_when_verifier_claims_valid_without_repair_payload() -> None:
    llm = _RetryingRepairContractLLM("scikit-learn", "n_jobs")
    contract = await synthesize_library_contract(
        "scikit-learn",
        [SourceArtifact(source_id="doc_001", path="docs/scikit-learn.md", kind="doc")],
        [
            EvidenceItem.model_validate(
                {
                    "evidence_id": "ev_001",
                    "library_name": "scikit-learn",
                    "chunk_id": "doc_001:chunk_001",
                    "kind": "intervention",
                    "title": "Coordinate parallel worker counts",
                    "content": "Reduce oversubscription by lowering outer or inner worker counts.",
                    "keywords": ["n_jobs", "threads", "oversubscription"],
                    "resource_domains": ["cpu_threads"],
                    "source_quote": "Reduce oversubscription by lowering worker counts.",
                    "source_refs": ["doc_001:chunk_001"],
                }
            )
        ],
        llm,
    )

    assert llm.verify_attempts == 2
    assert contract.contract == "scikit-learn"


def test_verify_contract_prompt_requires_current_virtual_probe_design() -> None:
    prompt_path = (
        Path(__file__).resolve().parents[1]
        / "syspilot_contract"
        / "prompts"
        / "verify_contract.txt"
    )
    prompt = prompt_path.read_text(encoding="utf-8")

    assert "signal.virtual_probe" in prompt
    assert "signals must use probe.kind + probe.name" not in prompt
    assert "probe.kind" not in prompt
    assert "probe.name" not in prompt
    assert "recipes must include diagnose blocks, config_checks, outcomes, and guardrails" in prompt
    assert "use predicate kind not_nested only with cpu.compute or gpu.compute" in prompt
    assert "use predicate kind placement_compatible only with cpu.placement" in prompt
    assert "config-check signals must target the configured component/use/resource path, never a control id" in prompt
    assert "do not redirect worker-count controls to use demand paths or raw resources just to satisfy existence" in prompt


def test_synthesize_contract_prompt_documents_predicate_resource_constraints() -> None:
    prompt_path = (
        Path(__file__).resolve().parents[1]
        / "syspilot_contract"
        / "prompts"
        / "synthesize_contract.txt"
    )
    prompt = prompt_path.read_text(encoding="utf-8")

    assert "use predicate kind not_nested only with cpu.compute or gpu.compute" in prompt
    assert "use predicate kind placement_compatible only with cpu.placement" in prompt
    assert "if a worker-count control exists, model the controlled runtime as a component kind executor with an attrs.workers field" in prompt
    assert "config-check signals should target the configured component/use/resource path itself, never a control id" in prompt
