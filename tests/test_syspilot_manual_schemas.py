import pytest
from pydantic import ValidationError

from syspilot_manual.schemas import EvidenceItem, LibraryManual, RESOURCE_DOMAINS


def test_resource_domains_match_mvp_vocabulary() -> None:
    assert RESOURCE_DOMAINS == (
        "cpu_threads",
        "cpu_affinity",
        "memory_capacity",
        "memory_bandwidth",
        "allocator",
        "cache",
        "io_queue",
        "disk_io",
        "network_io",
        "batching",
        "workers",
    )


def test_evidence_requires_supported_resource_domains_and_source_refs() -> None:
    with pytest.raises(ValidationError):
        EvidenceItem.model_validate(
            {
                "evidence_id": "ev_001",
                "library_name": "OpenMP",
                "chunk_id": "doc_001:chunk_001",
                "kind": "intervention",
                "title": "Tune threads",
                "content": "Set OMP_NUM_THREADS carefully.",
                "resource_domains": ["made_up"],
                "source_quote": "Set OMP_NUM_THREADS to avoid oversubscription.",
                "source_refs": [],
            }
        )


def test_library_manual_requires_core_sections() -> None:
    with pytest.raises(ValidationError):
        LibraryManual.model_validate(
            {
                "manual_id": "openmp",
                "manual_type": "library",
                "library_name": "OpenMP",
                "library_summary": "Runtime guidance.",
                "source_artifacts": [{"source_id": "doc_001", "path": "openmp.md", "kind": "doc"}],
                "feature_graph": [],
                "observability_map": [],
                "intervention_graph": [],
                "validation_recipes": [],
                "composition_hints": {
                    "shared_resources": ["cpu_threads"],
                    "conflict_patterns": [],
                },
            }
        )
