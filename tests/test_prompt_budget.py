from __future__ import annotations

import json

from syspilot.prompt_budget import (
    PROMPT_JSON_LIMIT_MARKER,
    compact_contract_context_for_prompt,
    contract_context_json_for_prompt,
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
