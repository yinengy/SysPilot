from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, TypedDict, cast


class OverheadMetrics(TypedDict):
    per_run_overhead_pct: list[float]
    median_overhead_pct: float
    p95_overhead_pct: float


class BenchModule(Protocol):
    def percentile(self, values: list[float], pct: float) -> float: ...

    def compute_overhead_metrics(
        self, plain_times_s: list[float], tui_times_s: list[float]
    ) -> OverheadMetrics: ...

    def extract_fallback_messages(self, stdout: str, stderr: str) -> list[str]: ...

    def classify_tui_fallback(self, messages: list[str]) -> str: ...


def _load_module() -> BenchModule:
    path = Path(__file__).resolve().parents[1] / "scripts" / "bench_tune_ui_overhead.py"
    spec = importlib.util.spec_from_file_location("bench_tune_ui_overhead", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    assert isinstance(module, ModuleType)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast(BenchModule, cast(object, module))


def test_percentile_linear_interpolation() -> None:
    module = _load_module()
    values = [1.0, 2.0, 4.0, 8.0]

    assert module.percentile(values, 0.0) == 1.0
    assert module.percentile(values, 50.0) == 3.0
    assert module.percentile(values, 100.0) == 8.0


def test_compute_overhead_metrics_shape_and_values() -> None:
    module = _load_module()

    plain = [1.0, 2.0, 4.0]
    tui = [1.1, 2.2, 4.4]
    result = module.compute_overhead_metrics(plain, tui)

    per_run = result["per_run_overhead_pct"]
    assert len(per_run) == 3
    for value in per_run:
        assert abs(value - 10.0) < 1e-9
    assert abs(result["median_overhead_pct"] - 10.0) < 1e-9
    assert abs(result["p95_overhead_pct"] - 10.0) < 1e-9


def test_compute_overhead_rejects_mismatched_lengths() -> None:
    module = _load_module()
    try:
        _ = module.compute_overhead_metrics([1.0], [1.0, 2.0])
    except ValueError as exc:
        assert "equal length" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_extract_fallback_messages_deduplicates_and_filters() -> None:
    module = _load_module()

    stdout = (
        "setup ok\n"
        "Non-TTY detected; falling back to --ui-mode plain\n"
        "Non-TTY detected; falling back to --ui-mode plain\n"
    )
    stderr = "TUI disconnected; continuing in --ui-mode plain\n"

    result = module.extract_fallback_messages(stdout=stdout, stderr=stderr)
    assert result == [
        "Non-TTY detected; falling back to --ui-mode plain",
        "TUI disconnected; continuing in --ui-mode plain",
    ]


def test_classify_tui_fallback_binary_unavailable() -> None:
    module = _load_module()
    reason = module.classify_tui_fallback(
        ["TUI binary unavailable; falling back to --ui-mode plain"]
    )
    assert reason == "tui_binary_unavailable"
