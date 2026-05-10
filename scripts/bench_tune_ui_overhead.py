#!/usr/bin/env python3
from __future__ import annotations

import argparse
import errno
import json
import os
import pty
import select
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast


FALLBACK_MARKERS = (
    "falling back to --ui-mode plain",
    "continuing in --ui-mode plain",
)
RUN_OUTPUT_TAIL_CHARS = 1200
PTY_CAPTURE_TAIL_BYTES = 20000
PTY_READ_CHUNK_BYTES = 65536
PTY_READ_POLL_S = 0.005


@dataclass(frozen=True)
class ModeRunResult:
    elapsed_s: float
    returncode: int
    stdout: str
    stderr: str
    fallback_messages: list[str]


@dataclass(frozen=True)
class BenchArgs:
    runs: int
    warmup: int
    threshold_p95: float
    command: str
    json_out: Path
    max_iterations: int
    timeout: int


class OverheadMetrics(TypedDict):
    per_run_overhead_pct: list[float]
    median_overhead_pct: float
    p95_overhead_pct: float


class RunRecord(TypedDict):
    run_index: int
    phase: str
    elapsed_s: float
    returncode: int
    fallback_detected: bool
    fallback_messages: list[str]
    stdout_tail: str
    stderr_tail: str


class ModeBenchmarkResult(TypedDict):
    measured_times_s: list[float]
    returncodes: list[int]
    run_records: list[RunRecord]
    fallback_messages: list[str]
    failed: bool
    failure_detail: str | None


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * (pct / 100.0)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = rank - lower_index
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    return lower * (1.0 - fraction) + upper * fraction


def compute_overhead_metrics(
    plain_times_s: list[float],
    tui_times_s: list[float],
) -> OverheadMetrics:
    if len(plain_times_s) != len(tui_times_s):
        raise ValueError("plain and tui timing lists must have equal length")

    overhead_pct: list[float] = []
    for plain, tui in zip(plain_times_s, tui_times_s):
        if plain <= 0:
            overhead_pct.append(0.0)
            continue
        overhead_pct.append(((tui - plain) / plain) * 100.0)

    return {
        "per_run_overhead_pct": overhead_pct,
        "median_overhead_pct": percentile(overhead_pct, 50.0),
        "p95_overhead_pct": percentile(overhead_pct, 95.0),
    }


def extract_fallback_messages(stdout: str, stderr: str) -> list[str]:
    messages: list[str] = []
    combined = f"{stdout}\n{stderr}"
    for raw_line in combined.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if any(marker in lowered for marker in FALLBACK_MARKERS):
            if line not in messages:
                messages.append(line)
    return messages


def classify_tui_fallback(messages: list[str]) -> str:
    joined = "\n".join(messages).lower()
    if "tui binary unavailable" in joined:
        return "tui_binary_unavailable"
    if "failed to launch tui" in joined:
        return "tui_launch_failed"
    if "non-tty detected" in joined:
        return "non_tty_fallback"
    if "tui disconnected" in joined:
        return "tui_disconnected"
    return "tui_fallback_detected"


def _tail(text: str, *, max_chars: int = RUN_OUTPUT_TAIL_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _coerce_text(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _append_tail_bytes(current: bytes, chunk: bytes, *, max_bytes: int) -> bytes:
    merged = current + chunk
    if len(merged) <= max_bytes:
        return merged
    return merged[-max_bytes:]


def _run_once_with_pty(
    *,
    command: list[str],
    env: dict[str, str],
    process_timeout_s: int,
    ui_mode: str,
) -> tuple[float, str, str, int, list[str]]:
    started = time.monotonic()
    child_pid, master_fd = pty.fork()

    if child_pid == 0:
        try:
            os.execvpe(command[0], command, env)
        except Exception:
            os._exit(127)

    output_tail = b""
    timed_out = False
    child_status: int | None = None

    def poll_child_status() -> int | None:
        nonlocal child_status
        if child_status is not None:
            return child_status
        waited_pid, status = os.waitpid(child_pid, os.WNOHANG)
        if waited_pid == child_pid:
            child_status = status
        return child_status

    try:
        deadline = started + process_timeout_s
        while True:
            now = time.monotonic()
            if now >= deadline and poll_child_status() is None:
                timed_out = True
                try:
                    os.kill(child_pid, 9)
                except ProcessLookupError:
                    pass
                _ = poll_child_status()

            timeout = (
                0.0
                if poll_child_status() is not None
                else min(
                    PTY_READ_POLL_S,
                    max(0.0, deadline - now),
                )
            )

            ready, _, _ = select.select([master_fd], [], [], timeout)
            if ready:
                try:
                    data = os.read(master_fd, PTY_READ_CHUNK_BYTES)
                except OSError as exc:
                    if exc.errno == errno.EIO:
                        data = b""
                    else:
                        raise

                if data:
                    output_tail = _append_tail_bytes(
                        output_tail,
                        data,
                        max_bytes=PTY_CAPTURE_TAIL_BYTES,
                    )

            if poll_child_status() is not None:
                while True:
                    ready, _, _ = select.select([master_fd], [], [], 0)
                    if not ready:
                        break
                    try:
                        data = os.read(master_fd, PTY_READ_CHUNK_BYTES)
                    except OSError as exc:
                        if exc.errno == errno.EIO:
                            break
                        raise
                    if not data:
                        break
                    output_tail = _append_tail_bytes(
                        output_tail,
                        data,
                        max_bytes=PTY_CAPTURE_TAIL_BYTES,
                    )
                break
    finally:
        try:
            os.close(master_fd)
        except OSError:
            pass
        if child_status is None:
            try:
                _, child_status = os.waitpid(child_pid, 0)
            except ChildProcessError:
                child_status = 0

    elapsed = time.monotonic() - started
    if timed_out:
        returncode = 124
        timeout_message = (
            f"Benchmark timeout: mode={ui_mode} exceeded {process_timeout_s}s "
            "while waiting for syspilot tune"
        )
        output_tail = _append_tail_bytes(
            output_tail,
            f"\n{timeout_message}".encode("utf-8", errors="replace"),
            max_bytes=PTY_CAPTURE_TAIL_BYTES,
        )
    else:
        returncode = int(os.waitstatus_to_exitcode(child_status))

    output_tail_text = output_tail.decode(errors="replace")
    fallback_messages = extract_fallback_messages(stdout=output_tail_text, stderr="")
    return elapsed, output_tail_text, "", returncode, fallback_messages


def _build_tune_command(
    *,
    ui_mode: str,
    workload_command: str,
    max_iterations: int,
    timeout_s: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "syspilot.cli",
        "tune",
        "--ui-mode",
        ui_mode,
        "--command",
        workload_command,
        "--max-iterations",
        str(max_iterations),
        "--timeout",
        str(timeout_s),
        "--dry-run",
    ]


def _run_once(
    *,
    ui_mode: str,
    workload_command: str,
    max_iterations: int,
    timeout_s: int,
) -> ModeRunResult:
    command = _build_tune_command(
        ui_mode=ui_mode,
        workload_command=workload_command,
        max_iterations=max_iterations,
        timeout_s=timeout_s,
    )

    started = time.monotonic()
    process_timeout_s = timeout_s + 120
    child_env = os.environ.copy()
    child_env["SYSPILOT_TUI_WAIT_FOR_USER"] = "0"
    child_env["SYSPILOT_BENCH_DETERMINISTIC"] = "1"
    try:
        elapsed, stdout, stderr, returncode, fallback_messages = _run_once_with_pty(
            command=command,
            env=child_env,
            process_timeout_s=process_timeout_s,
            ui_mode=ui_mode,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        stdout = _coerce_text(exc.stdout)
        stderr = _coerce_text(exc.stderr)
        if stderr:
            stderr = f"{stderr}\n"
        stderr = (
            f"{stderr}Benchmark timeout: mode={ui_mode} exceeded {process_timeout_s}s "
            "while waiting for syspilot tune"
        )
        returncode = 124
        fallback_messages = extract_fallback_messages(stdout=stdout, stderr=stderr)
    except Exception:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=process_timeout_s,
                env=child_env,
            )
            elapsed = time.monotonic() - started
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            returncode = completed.returncode
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - started
            stdout = _coerce_text(exc.stdout)
            stderr = _coerce_text(exc.stderr)
            if stderr:
                stderr = f"{stderr}\n"
            stderr = (
                f"{stderr}Benchmark timeout: mode={ui_mode} exceeded {process_timeout_s}s "
                "while waiting for syspilot tune"
            )
            returncode = 124
        fallback_messages = extract_fallback_messages(stdout=stdout, stderr=stderr)

    if not fallback_messages:
        fallback_messages = extract_fallback_messages(stdout=stdout, stderr=stderr)
    return ModeRunResult(
        elapsed_s=elapsed,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        fallback_messages=fallback_messages,
    )


def _run_paired_benchmark(
    *,
    runs: int,
    warmup: int,
    workload_command: str,
    max_iterations: int,
    timeout_s: int,
) -> tuple[ModeBenchmarkResult, ModeBenchmarkResult]:
    total = warmup + runs

    plain: ModeBenchmarkResult = {
        "measured_times_s": [],
        "returncodes": [],
        "run_records": [],
        "fallback_messages": [],
        "failed": False,
        "failure_detail": None,
    }
    tui: ModeBenchmarkResult = {
        "measured_times_s": [],
        "returncodes": [],
        "run_records": [],
        "fallback_messages": [],
        "failed": False,
        "failure_detail": None,
    }

    by_mode = {
        "plain": plain,
        "tui": tui,
    }

    for index in range(total):
        phase = "measured" if index >= warmup else "warmup"
        order = ("plain", "tui") if index % 2 == 0 else ("tui", "plain")

        for ui_mode in order:
            result = _run_once(
                ui_mode=ui_mode,
                workload_command=workload_command,
                max_iterations=max_iterations,
                timeout_s=timeout_s,
            )

            mode_result = by_mode[ui_mode]
            mode_result["returncodes"].append(result.returncode)
            for message in result.fallback_messages:
                if message not in mode_result["fallback_messages"]:
                    mode_result["fallback_messages"].append(message)

            if phase == "measured":
                mode_result["measured_times_s"].append(result.elapsed_s)

            mode_result["run_records"].append(
                {
                    "run_index": index + 1,
                    "phase": phase,
                    "elapsed_s": result.elapsed_s,
                    "returncode": result.returncode,
                    "fallback_detected": bool(result.fallback_messages),
                    "fallback_messages": result.fallback_messages,
                    "stdout_tail": _tail(result.stdout),
                    "stderr_tail": _tail(result.stderr),
                }
            )

            if result.returncode != 0:
                snippet = _tail(result.stderr or result.stdout, max_chars=800)
                mode_result["failed"] = True
                mode_result["failure_detail"] = (
                    f"{ui_mode} run failed (run={index + 1}/{total}, rc={result.returncode}): "
                    f"{snippet}"
                )
                return plain, tui

    return plain, tui


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _require_int(value: object, flag: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"{flag} must be an integer")
    return value


def _require_float(value: object, flag: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{flag} must be a number")
    return float(value)


def _require_str(value: object, flag: str) -> str:
    if not isinstance(value, str):
        raise SystemExit(f"{flag} must be a string")
    return value


def _require_path(value: object, flag: str) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    raise SystemExit(f"{flag} must be a filesystem path")


def parse_args() -> BenchArgs:
    parser = argparse.ArgumentParser(
        description="Benchmark SysPilot tune UI overhead (tui vs plain)."
    )
    _ = parser.add_argument("--runs", type=int, default=30)
    _ = parser.add_argument("--warmup", type=int, default=5)
    _ = parser.add_argument("--threshold-p95", type=float, default=3.0)
    _ = parser.add_argument(
        "--command",
        required=True,
        help="Workload command passed to syspilot tune in both modes",
    )
    _ = parser.add_argument("--json-out", type=Path, required=True)
    _ = parser.add_argument(
        "--max-iterations",
        type=int,
        default=1,
        help="Max tune iterations for both plain/tui trials",
    )
    _ = parser.add_argument("--timeout", type=int, default=30)

    namespace = parser.parse_args()
    return BenchArgs(
        runs=_require_int(cast(object, getattr(namespace, "runs")), "--runs"),
        warmup=_require_int(cast(object, getattr(namespace, "warmup")), "--warmup"),
        threshold_p95=_require_float(
            cast(object, getattr(namespace, "threshold_p95")), "--threshold-p95"
        ),
        command=_require_str(cast(object, getattr(namespace, "command")), "--command"),
        json_out=_require_path(cast(object, getattr(namespace, "json_out")), "--json-out"),
        max_iterations=_require_int(
            cast(object, getattr(namespace, "max_iterations")), "--max-iterations"
        ),
        timeout=_require_int(cast(object, getattr(namespace, "timeout")), "--timeout"),
    )


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if args.threshold_p95 < 0:
        raise SystemExit("--threshold-p95 must be >= 0")
    if args.max_iterations < 1:
        raise SystemExit("--max-iterations must be >= 1")
    if args.timeout < 1:
        raise SystemExit("--timeout must be >= 1")

    payload: dict[str, object] = {
        "schema_version": "1.0",
        "benchmark": "tune_ui_overhead",
        "runs": args.runs,
        "warmup": args.warmup,
        "threshold_p95_pct": args.threshold_p95,
        "command": args.command,
        "max_iterations": args.max_iterations,
        "timeout_s": args.timeout,
        "pass": False,
    }

    try:
        plain_result, tui_result = _run_paired_benchmark(
            runs=args.runs,
            warmup=args.warmup,
            workload_command=args.command,
            max_iterations=args.max_iterations,
            timeout_s=args.timeout,
        )

        payload.update(
            {
                "plain_times_s": plain_result["measured_times_s"],
                "tui_times_s": tui_result["measured_times_s"],
                "plain_returncodes": plain_result["returncodes"],
                "tui_returncodes": tui_result["returncodes"],
                "plain_runs": plain_result["run_records"],
                "tui_runs": tui_result["run_records"],
                "plain_fallback_messages": plain_result["fallback_messages"],
                "tui_fallback_messages": tui_result["fallback_messages"],
            }
        )

        if plain_result["failed"]:
            payload.update(
                {
                    "failure_reason": "plain_mode_run_failed",
                    "failure_detail": plain_result["failure_detail"],
                }
            )
            _write_json(args.json_out, payload)
            print(str(payload["failure_detail"]), file=sys.stderr)
            return 4

        if tui_result["failed"]:
            payload.update(
                {
                    "failure_reason": "tui_mode_run_failed",
                    "failure_detail": tui_result["failure_detail"],
                }
            )
            _write_json(args.json_out, payload)
            print(str(payload["failure_detail"]), file=sys.stderr)
            return 4

        tui_fallback_messages = tui_result["fallback_messages"]
        if tui_fallback_messages:
            failure_reason = classify_tui_fallback(tui_fallback_messages)
            payload.update(
                {
                    "failure_reason": failure_reason,
                    "failure_detail": "; ".join(tui_fallback_messages),
                }
            )
            _write_json(args.json_out, payload)
            print(
                f"Benchmark failed: {failure_reason} ({payload['failure_detail']})",
                file=sys.stderr,
            )
            return 3

        metrics = compute_overhead_metrics(
            plain_result["measured_times_s"],
            tui_result["measured_times_s"],
        )
        p95_overhead = metrics["p95_overhead_pct"]
        passed = p95_overhead <= args.threshold_p95

        payload.update(metrics)
        payload["pass"] = passed

        print(
            " ".join(
                [
                    f"median_overhead_pct={metrics['median_overhead_pct']:.4f}",
                    f"p95_overhead_pct={metrics['p95_overhead_pct']:.4f}",
                    f"threshold_p95_pct={args.threshold_p95:.4f}",
                    f"pass={str(passed).lower()}",
                ]
            )
        )

        if not passed:
            payload.update(
                {
                    "failure_reason": "p95_threshold_exceeded",
                    "failure_detail": (
                        f"p95_overhead_pct={p95_overhead:.4f} "
                        f"threshold_p95_pct={args.threshold_p95:.4f}"
                    ),
                }
            )

        _write_json(args.json_out, payload)
        return 0 if passed else 2
    except Exception as exc:
        payload.update(
            {
                "failure_reason": "benchmark_exception",
                "failure_detail": str(exc),
            }
        )
        _write_json(args.json_out, payload)
        print(f"Benchmark failed unexpectedly: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
