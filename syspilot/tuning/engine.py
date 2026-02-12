from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from syspilot.profiling.manager import ProfilingManager
from syspilot.tuning.tuner import ApplicationTuner, TuningAction

logger = logging.getLogger(__name__)


@dataclass
class IterationResult:
    iteration: int
    before_metrics: dict[str, Any]
    after_metrics: dict[str, Any]
    actions_applied: list[TuningAction]
    improvement_pct: float
    rolled_back: bool = False


class TuningEngine:
    def __init__(
        self,
        profiling_manager: ProfilingManager,
        tuner: ApplicationTuner,
        max_iterations: int = 5,
        settle_seconds: float = 2.0,
        convergence_threshold: float = 1.0,
        regression_threshold: float = 5.0,
    ) -> None:
        self._profiler = profiling_manager
        self._tuner = tuner
        self.max_iterations = max_iterations
        self.settle_seconds = settle_seconds
        self.convergence_threshold = convergence_threshold
        self.regression_threshold = regression_threshold
        self.history: list[IterationResult] = []

    async def collect_metrics(self, pid: int) -> dict[str, Any]:
        results = await self._profiler.collect_all(pid)
        flat: dict[str, Any] = {}
        for name, collector_result in results.items():
            for k, v in collector_result.data.items():
                flat[f"{name}.{k}"] = v
        return flat

    @staticmethod
    def compute_improvement(
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> float:
        numeric_deltas: list[float] = []
        for key in before:
            if key not in after:
                continue
            b_val = before[key]
            a_val = after[key]
            if isinstance(b_val, (int, float)) and isinstance(a_val, (int, float)) and b_val != 0:
                delta_pct = ((a_val - b_val) / abs(b_val)) * 100
                numeric_deltas.append(delta_pct)
        if not numeric_deltas:
            return 0.0
        return sum(numeric_deltas) / len(numeric_deltas)

    async def run_iteration(
        self,
        pid: int,
        recommendations: list[dict[str, Any]],
        iteration_num: int,
    ) -> IterationResult:
        before = await self.collect_metrics(pid)

        actions = self._tuner.apply_tuning(pid, recommendations)

        await asyncio.sleep(self.settle_seconds)

        after = await self.collect_metrics(pid)

        improvement = self.compute_improvement(before, after)

        rolled_back = False
        if improvement < -self.regression_threshold:
            logger.warning(
                f"Iteration {iteration_num}: regression of {improvement:.1f}% detected, rolling back"
            )
            for action in reversed(actions):
                if action.applied:
                    self._tuner.rollback(action)
            rolled_back = True

        result = IterationResult(
            iteration=iteration_num,
            before_metrics=before,
            after_metrics=after,
            actions_applied=actions,
            improvement_pct=improvement,
            rolled_back=rolled_back,
        )
        self.history.append(result)
        return result

    def is_converged(self, last_n: int = 2) -> bool:
        if len(self.history) < last_n:
            return False
        recent = self.history[-last_n:]
        return all(
            abs(r.improvement_pct) < self.convergence_threshold for r in recent
        )

    def get_summary(self) -> dict[str, Any]:
        return {
            "total_iterations": len(self.history),
            "converged": self.is_converged(),
            "iterations": [
                {
                    "iteration": r.iteration,
                    "improvement_pct": round(r.improvement_pct, 2),
                    "actions_count": len(r.actions_applied),
                    "rolled_back": r.rolled_back,
                }
                for r in self.history
            ],
        }
