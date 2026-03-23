from __future__ import annotations

from collections.abc import Callable
from typing import cast

import syspilot.cli as cli_module


_build_spawn_execution_plan = cast(
    Callable[
        [list[str], list[dict[str, str]]], tuple[list[str], dict[str, str], list[str], list[str]]
    ],
    getattr(cli_module, "_build_spawn_execution_plan"),
)


def _llama_bench_base_command() -> list[str]:
    return [
        "llama-bench",
        "-m",
        "model.gguf",
        "-p",
        "512",
        "-n",
        "128",
        "-r",
        "1",
        "-t",
        "24",
        "-o",
        "json",
    ]


def test_llama_bench_rewrite_rejects_workload_changes() -> None:
    base = _llama_bench_base_command()
    action = {
        "config_type": "command",
        "new_value": "llama-bench -m model.gguf -p 0 -n 128 -r 1 -t 12 -o json",
    }

    command, env_overrides, applied, skipped = _build_spawn_execution_plan(base, [action])

    assert command == base
    assert env_overrides == {}
    assert applied == []
    assert any("must keep 'n_prompt' fixed" in item for item in skipped)


def test_llama_bench_rewrite_rejects_unsupported_option() -> None:
    base = _llama_bench_base_command()
    action = {
        "config_type": "command",
        "new_value": "llama-bench -m model.gguf -p 512 -n 128 -r 1 -t 12 -tb 12 -o json",
    }

    command, env_overrides, applied, skipped = _build_spawn_execution_plan(base, [action])

    assert command == base
    assert env_overrides == {}
    assert applied == []
    assert any("-tb" in item and "unsupported" in item for item in skipped)


def test_llama_bench_rewrite_accepts_runtime_tuning_options() -> None:
    base = _llama_bench_base_command()
    action = {
        "config_type": "command",
        "new_value": "llama-bench -m model.gguf -p 512 -n 128 -r 1 -t 12 -o json -mmp 1",
    }

    command, env_overrides, applied, skipped = _build_spawn_execution_plan(base, [action])

    assert command != base
    assert env_overrides == {}
    assert skipped == []
    assert any(item.startswith("command ") for item in applied)
    assert "-mmp" in command


def test_llama_bench_rewrite_rejects_thread_sweep_values() -> None:
    base = _llama_bench_base_command()
    action = {
        "config_type": "command",
        "new_value": "llama-bench -m model.gguf -p 512 -n 128 -r 1 -t 1,2,4,8 -o json",
    }

    command, env_overrides, applied, skipped = _build_spawn_execution_plan(base, [action])

    assert command == base
    assert env_overrides == {}
    assert applied == []
    assert any("single concrete value for 'threads'" in item for item in skipped)
