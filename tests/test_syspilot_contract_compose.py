from pathlib import Path
from typing import cast

from syspilot_contract.checker import dump_contract_yaml
from syspilot_contract.compose_contracts import compose_contracts, derive_conflict_hypotheses
from syspilot_contract.integration import build_agent_context_bundle, write_contract_bundle
from syspilot_contract.query_store import ContractStore
from syspilot_contract.schemas import LibraryContract


def _compute_contract(
    name: str, control_name: str, signal_prefix: str | None = None
) -> LibraryContract:
    signal_prefix = signal_prefix or name.lower()
    return LibraryContract.model_validate(
        {
            "version": "lpcl/v1",
            "contract": name,
            "type": "library",
            "components": [
                {
                    "id": f"{signal_prefix}_threads",
                    "kind": "executor",
                    "attrs": {
                        "executor_kind": "thread_pool",
                        "workers": {"min": 1, "max": "host.cpu_slots"},
                    },
                    "evidence": [f"{signal_prefix}_doc:chunk_001"],
                }
            ],
            "uses": [
                {
                    "id": f"{signal_prefix}_cpu_compute",
                    "resource": "cpu.compute",
                    "by": f"{signal_prefix}_threads",
                    "phase": "compute",
                    "scope": "process",
                    "demand": {
                        "dimension": "workers",
                        "min": 1,
                        "max": "host.cpu_slots",
                    },
                    "share": "shared",
                    "evidence": [f"{signal_prefix}_doc:chunk_001"],
                }
            ],
            "assumptions": [
                {
                    "id": f"{signal_prefix}_not_nested",
                    "resource": "cpu.compute",
                    "predicate": {"kind": "not_nested"},
                    "evidence": [f"{signal_prefix}_doc:chunk_001"],
                }
            ],
            "controls": [
                {
                    "id": f"{signal_prefix}_threads_control",
                    "name": control_name,
                    "kind": "env",
                    "type": "int",
                    "target": f"{signal_prefix}_threads.workers",
                    "effect": {"kind": "set", "value": "input"},
                    "evidence": [f"{signal_prefix}_doc:chunk_001"],
                }
            ],
            "signals": [
                {
                    "id": f"{signal_prefix}_thread_count",
                    "target": f"{signal_prefix}_threads.workers",
                    "metric": "active_thread_count",
                    "indicates": "effective_worker_count",
                    "probe": {"kind": "api", "name": control_name.lower().replace("-", "_")},
                    "evidence": [f"{signal_prefix}_doc:chunk_001"],
                },
                {
                    "id": f"{signal_prefix}_ctx_switch_rate",
                    "target": "cpu.compute",
                    "metric": "os.context_switch_rate",
                    "indicates": "oversubscription",
                    "probe": {"kind": "perf", "name": "perf_stat"},
                    "evidence": [f"{signal_prefix}_doc:chunk_001"],
                },
            ],
            "recipes": [
                {
                    "id": f"{signal_prefix}_reduce_oversubscription",
                    "for": "shared_cpu_compute_conflict",
                    "collect": [
                        f"{signal_prefix}_thread_count",
                        f"{signal_prefix}_ctx_switch_rate",
                    ],
                    "actions": [f"set {control_name} to 1"],
                    "accept": "effective_worker_count == 1 and throughput_not_worse",
                    "rollback": "throughput_worse",
                    "evidence": [f"{signal_prefix}_doc:chunk_001"],
                }
            ],
        }
    )


def _placement_contract(name: str, bind_name: str, places_name: str) -> LibraryContract:
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
                    "attrs": {"executor_kind": "thread_pool"},
                    "evidence": [f"{prefix}_doc:chunk_001"],
                }
            ],
            "uses": [
                {
                    "id": f"{prefix}_cpu_placement",
                    "resource": "cpu.placement",
                    "by": f"{prefix}_threads",
                    "phase": "compute",
                    "scope": "process",
                    "demand": {"dimension": "core_set", "source": "runtime_affinity_mask"},
                    "share": "partitioned",
                    "evidence": [f"{prefix}_doc:chunk_001"],
                }
            ],
            "assumptions": [
                {
                    "id": f"{prefix}_placement_compatible",
                    "resource": "cpu.placement",
                    "predicate": {"kind": "placement_compatible"},
                    "evidence": [f"{prefix}_doc:chunk_001"],
                }
            ],
            "controls": [
                {
                    "id": f"{prefix}_proc_bind",
                    "name": bind_name,
                    "kind": "env",
                    "type": "enum",
                    "target": f"{prefix}_cpu_placement",
                    "effect": {"kind": "set", "value": "input"},
                    "evidence": [f"{prefix}_doc:chunk_001"],
                },
                {
                    "id": f"{prefix}_places",
                    "name": places_name,
                    "kind": "env",
                    "type": "string",
                    "target": f"{prefix}_cpu_placement",
                    "effect": {"kind": "set", "value": "input"},
                    "evidence": [f"{prefix}_doc:chunk_001"],
                },
            ],
            "signals": [
                {
                    "id": f"{prefix}_placement_signal",
                    "target": "cpu.placement",
                    "metric": "thread_placement",
                    "indicates": "placement_state",
                    "probe": {"kind": "log", "name": "OMP_DISPLAY_AFFINITY"},
                    "evidence": [f"{prefix}_doc:chunk_001"],
                }
            ],
            "recipes": [
                {
                    "id": f"{prefix}_placement_recipe",
                    "for": "cpu_placement_conflict",
                    "collect": [f"{prefix}_placement_signal"],
                    "actions": [f"compare placement with {bind_name} and {places_name}"],
                    "accept": "placement_matches_intent",
                    "rollback": "placement_conflicts",
                    "evidence": [f"{prefix}_doc:chunk_001"],
                }
            ],
        }
    )


def test_compose_order_independent() -> None:
    openmp = _compute_contract("OpenMP", "OMP_NUM_THREADS", "omp")
    openblas = _compute_contract("OpenBLAS", "OPENBLAS_NUM_THREADS", "blas")

    first = compose_contracts("openmp_openblas", [openmp, openblas])
    second = compose_contracts("openmp_openblas", [openblas, openmp])

    assert dump_contract_yaml(first) == dump_contract_yaml(second)


def test_compose_openblas_openmp_derives_shared_cpu_compute_conflict() -> None:
    openmp = _compute_contract("OpenMP", "OMP_NUM_THREADS", "omp")
    openblas = _compute_contract("OpenBLAS", "OPENBLAS_NUM_THREADS", "blas")

    hypotheses = derive_conflict_hypotheses([openmp, openblas])

    assert hypotheses
    hypothesis = hypotheses[0]
    assert hypothesis.kind == "shared_cpu_compute_conflict"
    assert hypothesis.resource == "cpu.compute"
    assert sorted(hypothesis.contracts) == ["OpenBLAS", "OpenMP"]
    assert any(ref.endswith("omp_threads") for ref in hypothesis.components)
    assert any(ref.endswith("blas_threads") for ref in hypothesis.components)
    assert any(
        violation.assumption.endswith("omp_not_nested")
        for violation in hypothesis.basis.assumption_violations
    )
    assert any(action.endswith("omp_threads_control") for action in hypothesis.candidate_actions)
    assert any(action.endswith("blas_threads_control") for action in hypothesis.candidate_actions)
    assert any(ref.endswith("omp_ctx_switch_rate") for ref in hypothesis.validation.collect)
    assert any(ref.endswith("blas_ctx_switch_rate") for ref in hypothesis.validation.collect)


def test_no_fake_worker_pool_hypothesis_or_resource() -> None:
    openmp = _compute_contract("OpenMP", "OMP_NUM_THREADS", "omp")
    openblas = _compute_contract("OpenBLAS", "OPENBLAS_NUM_THREADS", "blas")

    composed = compose_contracts("openmp_openblas", [openmp, openblas])

    assert all(item.resource != "worker.pool" for item in composed.shared_resources)
    assert all(item.resource != "worker.pool" for item in composed.hypotheses)
    assert all(item.kind != "worker.pool" for item in composed.hypotheses)


def test_cpu_placement_conflict() -> None:
    gromacs = _placement_contract("GROMACS", "OMP_PROC_BIND", "OMP_PLACES")
    cp2k = _placement_contract("CP2K", "KMP_AFFINITY", "KMP_HW_SUBSET")

    hypotheses = derive_conflict_hypotheses([gromacs, cp2k])

    assert hypotheses
    assert hypotheses[0].kind == "cpu_placement_conflict"
    assert hypotheses[0].resource == "cpu.placement"


def test_contract_context_bundle_exposes_conflict_oriented_context(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    contract_root = tmp_path / "contracts"
    openmp = _compute_contract("OpenMP", "OMP_NUM_THREADS", "omp")
    openblas = _compute_contract("OpenBLAS", "OPENBLAS_NUM_THREADS", "blas")
    write_contract_bundle("openmp", openmp, artifact_root, contract_root)
    write_contract_bundle("openblas", openblas, artifact_root, contract_root)

    context = build_agent_context_bundle(
        ["openmp", "openblas"],
        goal="improve throughput",
        symptom="thread contention",
        contract_root=contract_root,
        max_items_per_section=3,
    )

    assert context["relevant_components"]
    assert context["relevant_resources"] == ["cpu.compute"]
    assert context["likely_hypotheses"]
    assert context["signals_to_collect"]
    assert context["candidate_controls"]
    assert context["recipes"]
    assert context["candidate_actions"]
    assert context["conflicts"]
    assert context["validation"]


def test_duplicate_library_contracts_get_distinct_instances_and_context(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    contract_root = tmp_path / "contracts"
    openmp = _compute_contract("OpenMP", "OMP_NUM_THREADS", "omp")
    composed = compose_contracts("openmp_openmp", [openmp, openmp.model_copy(deep=True)])

    write_contract_bundle("openmp", openmp, artifact_root, contract_root)
    write_contract_bundle("openmp_openmp_composed", composed, artifact_root, contract_root)

    assert [item.instance_id for item in composed.contract_instances] == ["openmp_1", "openmp_2"]
    assert [item.contract_id for item in composed.contract_instances] == ["OpenMP", "OpenMP"]
    assert composed.hypotheses
    assert sorted(composed.hypotheses[0].contracts) == ["openmp_1", "openmp_2"]
    assert "openmp_1.omp_cpu_compute" in composed.hypotheses[0].uses
    assert "openmp_2.omp_cpu_compute" in composed.hypotheses[0].uses

    context = build_agent_context_bundle(
        ["openmp_openmp_composed"],
        goal="improve throughput",
        symptom="thread contention",
        contract_root=contract_root,
        max_items_per_section=4,
    )

    assert context["shared_resources"]
    conflicts = cast(list[dict[str, object]], context["conflicts"])
    assert conflicts
    contracts = conflicts[0].get("contracts")
    assert isinstance(contracts, list)
    assert sorted(str(item) for item in contracts) == ["openmp_1", "openmp_2"]


def test_query_api_filters_uses_controls_signals_by_resource(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    contract_root = tmp_path / "contracts"
    openmp = _compute_contract("OpenMP", "OMP_NUM_THREADS", "omp")
    placement = _placement_contract("PlacementLib", "OMP_PROC_BIND", "OMP_PLACES")
    write_contract_bundle("openmp", openmp, artifact_root, contract_root)
    write_contract_bundle("placementlib", placement, artifact_root, contract_root)

    store = ContractStore(contract_root)

    cpu_compute_uses = store.list_uses("openmp", resource="cpu.compute")
    cpu_compute_controls = store.list_controls("openmp", resource="cpu.compute")
    cpu_compute_signals = store.list_signals("openmp", resource="cpu.compute")
    cpu_placement_uses = store.list_uses("placementlib", resource="cpu.placement")
    cpu_placement_signals = store.list_signals("placementlib", resource="cpu.placement")

    assert len(cpu_compute_uses) == 1
    assert len(cpu_compute_controls) == 1
    assert len(cpu_compute_signals) == 2
    assert len(cpu_placement_uses) == 1
    assert len(cpu_placement_signals) == 1
