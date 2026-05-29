from __future__ import annotations

import json

from syspilot.prompt_budget import (
    CONTRACT_YAML_CONTEXT_KEY,
    PROMPT_JSON_LIMIT_MARKER,
    compact_contract_context_for_prompt,
    contract_context_json_for_prompt,
    hypothesis_profiles_yaml_for_prompt,
    profiling_json_for_prompt,
)


def test_contract_context_prompt_compaction_keeps_actionable_fields() -> None:
    bundle = {
        "summary": "demo context",
        "relevant_resources": ["cpu.compute"],
        "candidate_controls": [
            {
                "ref": "pytorch.ctrl_threads",
                "contract": "pytorch",
                "control": {
                    "name": "BENCHMARK_TORCH_NUM_THREADS",
                    "kind": "env",
                    "type": "int",
                    "target": "torch.thread_pool.workers",
                    "activation_value": 1,
                    "effect": {"kind": "set", "value": "input"},
                    "evidence": ["x" * 20_000],
                },
                "evidence": ["y" * 20_000],
            }
        ],
        "hypothesis_profiles": [
            {
                "hypothesis_id": "cpu-oversubscription",
                "candidate_controls": [
                    {
                        "ref": "h5py.ctrl_rdcc_nbytes",
                        "contract": "h5py",
                        "control": {"name": "BENCHMARK_HDF5_RDCC_NBYTES", "kind": "env"},
                    }
                ],
                "validation_signals": [
                    {
                        "ref": "pytorch.sig_threads",
                        "signal": {
                            "metric": "thread_count",
                            "virtual_probe": "process.cpu.thread_count",
                            "expected_direction_when_mitigated": "decrease",
                        },
                    }
                ],
                "evidence": ["z" * 20_000],
            }
        ],
    }

    compacted = compact_contract_context_for_prompt(bundle)
    rendered = contract_context_json_for_prompt(bundle)

    assert "BENCHMARK_TORCH_NUM_THREADS" in rendered
    assert "BENCHMARK_HDF5_RDCC_NBYTES" in rendered
    assert "process.cpu.thread_count" in rendered
    assert "evidence" not in json.dumps(compacted)
    assert len(rendered) < 10_000


def test_contract_context_prompt_preserves_small_raw_payload() -> None:
    bundle = {
        "summary": "small context",
        "candidate_controls": [
            {
                "control": {"name": "BENCHMARK_TORCH_NUM_THREADS"},
                "evidence": ["small provenance should remain when under budget"],
            }
        ],
    }

    rendered = contract_context_json_for_prompt(bundle)

    assert "small provenance should remain when under budget" in rendered
    assert PROMPT_JSON_LIMIT_MARKER not in rendered


def test_contract_context_prompt_prefers_yaml_context() -> None:
    rendered = contract_context_json_for_prompt(
        {
            CONTRACT_YAML_CONTEXT_KEY: "### Contract YAML: openmp\n```yaml\ncontract: OpenMP\n```",
            "candidate_actions": [{"control": {"name": "OMP_NUM_THREADS"}}],
        }
    )

    assert "Contract YAML: openmp" in rendered
    assert "candidate_actions" not in rendered


def test_hypothesis_profiles_prompt_renders_generated_profiles_as_full_yaml_without_evidence() -> None:
    long_note = "x" * 90_000
    rendered = hypothesis_profiles_yaml_for_prompt(
        [
            {
                "hypothesis_id": "h_shared_cpu_compute_conflict_openmp_openblas_compute",
                "kind": "shared_resource_conflict",
                "resource": "cpu.compute",
                "explanation": long_note,
                "validation_signals": [
                    {"ref": "s1", "signal": {"virtual_probe": "process.cpu.thread_count"}}
                ],
                "candidate_controls": [
                    {"control": {"name": "OMP_NUM_THREADS", "kind": "env"}}
                ],
                "evidence": ["source provenance should not be prompt-facing"],
            }
        ]
    )

    assert "h_shared_cpu_compute_conflict_openmp_openblas_compute" in rendered
    assert "hypothesis_id: h_shared_cpu_compute_conflict_openmp_openblas_compute" in rendered
    assert "process.cpu.thread_count" in rendered
    assert "OMP_NUM_THREADS" in rendered
    assert long_note in rendered
    assert "truncated" not in rendered
    assert PROMPT_JSON_LIMIT_MARKER not in rendered
    assert rendered.lstrip().startswith("hypotheses:\n- hypothesis_id:")
    assert "source provenance should not be prompt-facing" not in rendered


def test_profiling_json_prompt_preserves_small_raw_payload() -> None:
    rendered = profiling_json_for_prompt(
        {
            "periodic_probe_samples": [
                {"time_offset_s": 0.0, "results": {"probe": {"value": 1, "detail": "raw detail"}}}
            ]
        }
    )

    assert "raw detail" in rendered
    assert PROMPT_JSON_LIMIT_MARKER not in rendered


def test_profiling_json_prompt_drops_duplicate_prompt_fields() -> None:
    rendered = profiling_json_for_prompt(
        {
            "collect_profile_summary": "duplicate summary",
            "contract_hypothesis_validation": {"profiles": [{"hypothesis_id": "h1"}]},
            "ebpf_summary": "unused ebpf",
            "collect_profile_data": {"cpu": {"data": {"num_threads": 6}}},
            "key_findings": ["useful finding"],
        }
    )

    assert "duplicate summary" not in rendered
    assert "contract_hypothesis_validation" not in rendered
    assert "unused ebpf" not in rendered
    assert "collect_profile_data" in rendered
    assert "useful finding" in rendered


def test_profiling_json_prompt_keeps_simple_virtual_probe_results() -> None:
    rendered = profiling_json_for_prompt(
        {
            "virtual_probe_results": [
                {"process.cpu.thread_count": 8},
                {"cgroup.memory.events.high": 2},
            ],
        }
    )

    assert "process.cpu.thread_count" in rendered
    assert "cgroup.memory.events.high" in rendered
    assert "resolved_probe" not in rendered
    assert "structured" not in rendered


def test_profiling_json_prompt_rendering_truncates_large_payload() -> None:
    rendered = profiling_json_for_prompt(
        {
            "periodic_probe_samples": [
                {"time_offset_s": i, "results": {"probe": {"value": i, "detail": "x" * 3000}}}
                for i in range(40)
            ]
        }
    )

    assert len(rendered) <= 46_000
    assert PROMPT_JSON_LIMIT_MARKER in rendered or "truncated" in rendered
