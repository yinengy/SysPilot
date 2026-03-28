from pathlib import Path
from typing import cast

from syspilot_contract.checker import dump_contract_yaml
from syspilot_contract.compose_contracts import compose_contracts, derive_conflict_hypotheses
from syspilot_contract.integration import build_agent_context_bundle, write_contract_bundle
from syspilot_contract.schemas import LibraryContract


def _contract(name: str, control_name: str, role: str = "consume") -> LibraryContract:
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
                    "role": role,
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
                    "id": f"{name}_control",
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
                    "id": f"{name}_signal",
                    "metric": f"{name}.runqueue_length",
                    "resource": "cpu.thread",
                    "indicates": "thread contention",
                    "probe": {"kind": "command", "target": "perf"},
                    "source_refs": [f"{name}_doc:chunk_001"],
                }
            ],
            "recipes": [
                {
                    "id": f"{name}_recipe",
                    "hypothesis_kind": "shared_resource_conflict",
                    "collect": [f"{name}_signal"],
                    "control_refs": [f"{name}_control"],
                    "actions": [f"decrease {control_name}"],
                    "accept_if": "throughput improves",
                    "rollback_if": "throughput regresses",
                    "source_refs": [f"{name}_doc:chunk_001"],
                }
            ],
        }
    )


def test_compose_order_independent() -> None:
    openmp = _contract("OpenMP", "OMP_NUM_THREADS")
    openblas = _contract("OpenBLAS", "OPENBLAS_NUM_THREADS")

    first = compose_contracts("openmp_openblas", [openmp, openblas])
    second = compose_contracts("openmp_openblas", [openblas, openmp])

    assert dump_contract_yaml(first) == dump_contract_yaml(second)


def test_composer_derives_hypotheses_from_shared_resource() -> None:
    openmp = _contract("OpenMP", "OMP_NUM_THREADS")
    openblas = _contract("OpenBLAS", "OPENBLAS_NUM_THREADS", role="reserve")

    hypotheses = derive_conflict_hypotheses([openmp, openblas])

    assert hypotheses
    assert hypotheses[0].resource == "cpu.thread"
    assert "OpenMP.OpenMP_control" in hypotheses[0].basis.colliding_controls
    assert any("solo_compute_pool" in item for item in hypotheses[0].basis.violated_assumptions)
    assert "OpenMP.OpenMP_signal" in hypotheses[0].validation.collect
    assert "OpenMP.OpenMP_recipe" in hypotheses[0].candidate_recipe_refs


def test_contract_context_bundle_exposes_profile_actions_conflicts(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    contract_root = tmp_path / "contracts"
    openmp = _contract("OpenMP", "OMP_NUM_THREADS")
    openblas = _contract("OpenBLAS", "OPENBLAS_NUM_THREADS", role="reserve")
    write_contract_bundle("openmp", openmp, artifact_root, contract_root)
    write_contract_bundle("openblas", openblas, artifact_root, contract_root)

    context = build_agent_context_bundle(
        ["openmp", "openblas"],
        goal="improve throughput",
        symptom="thread contention",
        contract_root=contract_root,
        max_items_per_section=3,
    )

    assert context["profile_first"]
    assert context["candidate_actions"]
    assert context["conflicts"]
    assert context["validation"]


def test_composed_contract_id_hydrates_library_refs_for_context(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    contract_root = tmp_path / "contracts"
    openmp = _contract("OpenMP", "OMP_NUM_THREADS")
    openblas = _contract("OpenBLAS", "OPENBLAS_NUM_THREADS", role="reserve")
    composed = compose_contracts("openmp_openblas", [openmp, openblas])

    write_contract_bundle("openmp", openmp, artifact_root, contract_root)
    write_contract_bundle("openblas", openblas, artifact_root, contract_root)
    write_contract_bundle("openmp_openblas_composed", composed, artifact_root, contract_root)

    context = build_agent_context_bundle(
        ["openmp_openblas_composed"],
        goal="improve throughput",
        symptom="thread contention",
        contract_root=contract_root,
        max_items_per_section=3,
    )

    assert context["profile_first"]
    assert context["candidate_actions"]
    assert context["conflicts"]
    assert context["validation"]
    candidate_titles = {
        item["title"]
        for item in cast(list[dict[str, object]], context["candidate_actions"])
        if "title" in item
    }
    assert "OMP_NUM_THREADS" in candidate_titles or "OPENBLAS_NUM_THREADS" in candidate_titles
