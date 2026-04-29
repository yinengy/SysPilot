from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest

import syspilot.cli as cli_module


_build_spawn_execution_plan = cast(
    Callable[..., tuple[list[str], dict[str, str], list[str], list[str]]],
    getattr(cli_module, "_build_spawn_execution_plan"),
)

_build_spawn_candidates = cast(
    Callable[..., list[dict[str, object]]],
    getattr(cli_module, "_build_spawn_candidates"),
)

_extract_inline_env_assignments = cast(
    Callable[[list[str]], tuple[list[str], dict[str, str]]],
    getattr(cli_module, "_extract_inline_env_assignments"),
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


def _gmx_mdrun_base_command() -> list[str]:
    return [
        "gmx_mpi_d",
        "mdrun",
        "-s",
        "topol.tpr",
        "-npme",
        "0",
        "-ntomp",
        "24",
        "-pin",
        "off",
        "-deffnm",
        "run",
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


def test_gmx_mdrun_rewrite_accepts_valid_subcommand_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _gmx_mdrun_base_command()
    action = {
        "config_type": "command",
        "new_value": "gmx_mpi_d mdrun -s topol.tpr -npme 0 -ntomp 12 -pin on -deffnm run",
    }

    def fake_discover_supported_cli_options(help_target: tuple[str, ...]) -> set[str] | None:
        if help_target == ("gmx_mpi_d", "mdrun"):
            return {"-s", "-npme", "-ntomp", "-pin", "-deffnm"}
        if help_target == ("gmx_mpi_d",):
            return {"-h", "--help", "-quiet", "-version"}
        return None

    monkeypatch.setattr(
        cli_module,
        "_discover_supported_cli_options",
        fake_discover_supported_cli_options,
    )

    command, env_overrides, applied, skipped = _build_spawn_execution_plan(base, [action])

    assert command != base
    assert command[:2] == ["gmx_mpi_d", "mdrun"]
    assert env_overrides == {}
    assert skipped == []
    assert any(item.startswith("command ") for item in applied)
    assert "-pin" in command
    assert "on" in command
    assert "12" in command


def test_gmx_mdrun_rewrite_rejects_unsupported_subcommand_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _gmx_mdrun_base_command()
    action = {
        "config_type": "command",
        "new_value": "gmx_mpi_d mdrun -s topol.tpr -npme 0 -ntomp 12 -pin on -badflag 1 -deffnm run",
    }

    def fake_discover_supported_cli_options(help_target: tuple[str, ...]) -> set[str] | None:
        if help_target == ("gmx_mpi_d", "mdrun"):
            return {"-s", "-npme", "-ntomp", "-pin", "-deffnm"}
        if help_target == ("gmx_mpi_d",):
            return {"-h", "--help", "-quiet", "-version"}
        return None

    monkeypatch.setattr(
        cli_module,
        "_discover_supported_cli_options",
        fake_discover_supported_cli_options,
    )

    command, env_overrides, applied, skipped = _build_spawn_execution_plan(base, [action])

    assert command == base
    assert env_overrides == {}
    assert applied == []
    assert any("-badflag" in item and "supported" in item for item in skipped)


def test_extract_inline_env_assignments_from_spawn_command() -> None:
    command = [
        "env",
        "OMP_NUM_THREADS=24",
        "OMP_PROC_BIND=false",
        "MKL_NUM_THREADS=24",
        "gmx_mpi_d",
        "mdrun",
        "-s",
        "topol.tpr",
    ]

    stripped_command, env_overrides = _extract_inline_env_assignments(command)

    assert stripped_command == ["gmx_mpi_d", "mdrun", "-s", "topol.tpr"]
    assert env_overrides == {
        "OMP_NUM_THREADS": "24",
        "OMP_PROC_BIND": "false",
        "MKL_NUM_THREADS": "24",
    }


def test_build_spawn_execution_plan_preserves_inline_env_baseline() -> None:
    base = _gmx_mdrun_base_command()
    action = {
        "config_type": "env_var",
        "parameter": "OMP_PROC_BIND",
        "new_value": "close",
    }

    command, env_overrides, applied, skipped = _build_spawn_execution_plan(
        base,
        [action],
        base_env_overrides={
            "OMP_NUM_THREADS": "24",
            "OMP_PROC_BIND": "false",
            "MKL_NUM_THREADS": "24",
        },
    )

    assert command == base
    assert env_overrides == {
        "OMP_NUM_THREADS": "24",
        "OMP_PROC_BIND": "close",
        "MKL_NUM_THREADS": "24",
    }
    assert skipped == []
    assert applied == ["env OMP_PROC_BIND=close"]


def test_build_spawn_candidates_separates_command_prefix_alternatives() -> None:
    base = ["gmx_mpi_d", "mdrun", "-s", "topol.tpr"]
    actions = [
        {
            "config_type": "command_prefix",
            "new_value": ["env", "OMP_PROC_BIND=spread", "OMP_PLACES=cores"],
        },
        {
            "config_type": "command_prefix",
            "new_value": ["env", "OPENBLAS_NUM_THREADS=1", "OPENBLAS_MAIN_FREE=1"],
        },
    ]

    candidates = _build_spawn_candidates(base, actions, max_candidates=4)

    assert len(candidates) == 2
    commands = [candidate["cmd"] for candidate in candidates]
    assert commands[0] == ["env", "OMP_PROC_BIND=spread", "OMP_PLACES=cores", *base]
    assert commands[1] == ["env", "OPENBLAS_NUM_THREADS=1", "OPENBLAS_MAIN_FREE=1", *base]


def test_build_spawn_candidates_accumulates_env_only_actions_in_order() -> None:
    base = ["gmx_mpi_d", "mdrun", "-s", "topol.tpr"]
    actions = [
        {"config_type": "env_var", "parameter": "OMP_PROC_BIND", "new_value": "close"},
        {"config_type": "env_var", "parameter": "OMP_PLACES", "new_value": "cores"},
        {"config_type": "env_var", "parameter": "OPENBLAS_NUM_THREADS", "new_value": "1"},
    ]

    candidates = _build_spawn_candidates(base, actions, max_candidates=4)

    assert len(candidates) == 3
    assert candidates[0]["env_overrides"] == {"OMP_PROC_BIND": "close"}
    assert candidates[1]["env_overrides"] == {
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
    }
    assert candidates[2]["env_overrides"] == {
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
        "OPENBLAS_NUM_THREADS": "1",
    }
