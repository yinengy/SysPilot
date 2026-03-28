from pathlib import Path
from typing import cast

import pytest

from syspilot_manual.compose_manuals import (
    compose_manuals,
    derive_conflicts,
    derive_shared_resources,
)
from syspilot_manual.integration import write_manual_bundle
from syspilot_manual.query_store import ManualStore
from syspilot_manual.schemas import LibraryManual


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response

    async def complete_prompt(self, prompt_name: str, payload: dict[str, object]) -> str:
        _ = prompt_name
        _ = payload
        return self.response


def _library_manual(
    manual_id: str, library_name: str, action_name: str, conflict_name: str
) -> LibraryManual:
    return LibraryManual.model_validate(
        {
            "manual_id": manual_id,
            "manual_type": "library",
            "library_name": library_name,
            "library_summary": f"{library_name} tuning guidance.",
            "source_artifacts": [
                {"source_id": f"{manual_id}_doc", "path": f"docs/{manual_id}.md", "kind": "doc"}
            ],
            "feature_graph": [
                {
                    "feature_id": f"{manual_id}_threads",
                    "name": f"{library_name} threading",
                    "description": "Controls worker and thread scheduling.",
                    "resource_domains": ["cpu_threads"],
                    "keywords": [action_name],
                    "source_refs": [f"{manual_id}_doc:chunk_001"],
                }
            ],
            "observability_map": [
                {
                    "obs_id": f"{manual_id}_obs",
                    "feature_id": f"{manual_id}_threads",
                    "question": f"How do we detect {library_name} thread contention?",
                    "priority": 1,
                    "symptoms": ["throughput plateau"],
                    "signals": [
                        {
                            "name": "context_switch_rate",
                            "tool": "perf",
                            "reason": "Detect thread contention",
                        }
                    ],
                    "source_refs": [f"{manual_id}_doc:chunk_001"],
                }
            ],
            "intervention_graph": [
                {
                    "intervention_id": f"{manual_id}_int",
                    "feature_id": f"{manual_id}_threads",
                    "when": ["context switches high"],
                    "actions": [
                        {
                            "kind": "env",
                            "name": action_name,
                            "suggestion": f"tune {action_name} conservatively",
                        }
                    ],
                    "resource_domains": ["cpu_threads"],
                    "expected_effect": "Reduce thread contention.",
                    "side_effects": ["Too few threads can underutilize cores"],
                    "conflicts_with": [conflict_name],
                    "source_refs": [f"{manual_id}_doc:chunk_001"],
                }
            ],
            "validation_recipes": [
                {
                    "validation_id": f"{manual_id}_val",
                    "for_intervention": f"{manual_id}_int",
                    "reruns": 3,
                    "metrics": ["throughput"],
                    "accept_if": "Median throughput improves.",
                    "rollback_if": "Throughput regresses.",
                    "source_refs": [f"{manual_id}_doc:chunk_001"],
                }
            ],
            "composition_hints": {
                "shared_resources": ["cpu_threads"],
                "conflict_patterns": [
                    {
                        "with_keywords": ["threads", "workers"],
                        "reason": "Thread knobs can compete across libraries.",
                        "resolution_hint": "Tune one thread family at a time.",
                        "source_refs": [f"{manual_id}_doc:chunk_001"],
                    }
                ],
            },
        }
    )


@pytest.mark.asyncio
async def test_composition_builds_shared_resources_and_conflicts() -> None:
    openmp = _library_manual("openmp", "OpenMP", "OMP_NUM_THREADS", "nthread")
    xgboost = _library_manual("xgboost", "XGBoost", "nthread", "OMP_NUM_THREADS")

    shared = derive_shared_resources([openmp, xgboost])
    conflicts = derive_conflicts([openmp, xgboost])
    assert shared
    assert conflicts

    llm = FakeLLMClient(
        """
        {
          "manual_id": "ignored",
          "manual_type": "composed",
          "app_name": "ignored",
          "selected_libraries": ["OpenMP", "XGBoost"],
          "app_summary": "Coordinate OpenMP and XGBoost thread knobs around shared CPU resources.",
          "shared_resources": [
            {
              "resource_domain": "cpu_threads",
              "libraries": ["OpenMP", "XGBoost"],
              "note": "Both libraries own CPU thread knobs.",
              "source_refs": ["openmp_doc:chunk_001", "xgboost_doc:chunk_001"]
            }
          ],
          "conflict_summary": [
            {
              "conflict_id": "thread_conflict",
              "title": "Competing CPU thread ownership",
              "libraries": ["OpenMP", "XGBoost"],
              "resource_domains": ["cpu_threads"],
              "reason": "OMP_NUM_THREADS and nthread both control CPU thread fan-out.",
              "resolution_hint": "Change one knob family at a time.",
              "source_refs": ["openmp_doc:chunk_001", "xgboost_doc:chunk_001"]
            }
          ],
          "combined_observability": [
            {
              "obs_id": "obs_cpu_threads",
              "library": "OpenMP",
              "question": "How do we detect shared thread oversubscription?",
              "priority": 1,
              "symptoms": ["throughput plateau"],
              "signals": [
                {"name": "context_switch_rate", "tool": "perf", "reason": "Detect oversubscription"}
              ],
              "source_refs": ["openmp_doc:chunk_001"]
            }
          ],
          "combined_interventions": [
            {
              "intervention_id": "coord_threads",
              "library": "XGBoost",
              "when": ["context switches high"],
              "actions": [
                {"kind": "config", "name": "nthread", "suggestion": "align with OpenMP thread settings"}
              ],
              "resource_domains": ["cpu_threads"],
              "expected_effect": "Reduce cross-library oversubscription.",
              "side_effects": ["Can limit parallelism if set too low"],
              "conflicts_with": ["OMP_NUM_THREADS"],
              "source_refs": ["xgboost_doc:chunk_001"]
            }
          ],
          "combined_validation": [
            {
              "validation_id": "val_threads",
              "library": "XGBoost",
              "for_intervention": "coord_threads",
              "reruns": 3,
              "metrics": ["throughput", "latency_p95"],
              "accept_if": "Throughput improves without p95 regression.",
              "rollback_if": "Throughput regresses or latency worsens.",
              "source_refs": ["xgboost_doc:chunk_001"]
            }
          ],
          "recommended_query_order": [
            "shared_resources",
            "combined_observability",
            "conflict_summary",
            "combined_interventions",
            "combined_validation"
          ]
        }
        """
    )
    manual = await compose_manuals(
        manual_id="xgb_train_composed",
        app_name="xgb_train",
        manuals=[openmp, xgboost],
        llm_client=llm,
    )
    assert manual.app_name == "xgb_train"
    assert manual.shared_resources[0].resource_domain == "cpu_threads"
    assert manual.conflict_summary


def test_query_api_filters_and_builds_agent_context(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    manual_root = tmp_path / "manual"
    openmp = _library_manual("openmp", "OpenMP", "OMP_NUM_THREADS", "nthread")
    xgboost = _library_manual("xgboost", "XGBoost", "nthread", "OMP_NUM_THREADS")
    write_manual_bundle(openmp, artifact_root, manual_root)
    write_manual_bundle(xgboost, artifact_root, manual_root)

    store = ManualStore(manual_root)
    assert store.list_manuals() == ["openmp", "xgboost"]

    interventions = store.get_interventions(["openmp", "xgboost"], keywords=["OMP_NUM_THREADS"])
    assert interventions
    assert interventions[0]["section"] == "intervention_graph"

    observability = store.get_observability(["openmp", "xgboost"], keywords=["context_switch_rate"])
    assert observability
    assert observability[0]["section"] == "observability_map"

    conflicts = store.get_conflicts(["openmp", "xgboost"])
    assert conflicts

    context = store.build_agent_context(
        ["openmp", "xgboost"],
        goal="improve throughput",
        symptom="throughput plateau",
        max_items_per_section=2,
    )
    assert "summary" in context
    assert len(cast(list[dict[str, object]], context["profile_first"])) <= 2
    assert len(cast(list[dict[str, object]], context["candidate_actions"])) <= 2
