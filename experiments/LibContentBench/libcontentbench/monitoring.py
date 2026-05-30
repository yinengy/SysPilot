from __future__ import annotations

import csv
import importlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_psutil() -> Any | None:
    try:
        return importlib.import_module("psutil")
    except ModuleNotFoundError:  # pragma: no cover - fallback for host test envs
        return None


psutil = _load_psutil()


@dataclass
class Sample:
    ts: float
    processes: int
    threads: int
    rss_bytes: int
    voluntary_ctx: int
    involuntary_ctx: int
    read_count: int
    write_count: int
    read_bytes: int
    write_bytes: int


def _sum_process_tree(root: Any) -> Sample:
    processes = [root]
    try:
        processes.extend(root.children(recursive=True))
    except Exception:
        pass
    alive: list[Any] = []
    for proc in processes:
        try:
            proc.status()
            alive.append(proc)
        except Exception:
            continue

    threads = 0
    rss_bytes = 0
    voluntary_ctx = 0
    involuntary_ctx = 0
    read_count = 0
    write_count = 0
    read_bytes = 0
    write_bytes = 0
    for proc in alive:
        try:
            threads += proc.num_threads()
        except Exception:
            pass
        try:
            rss_bytes += proc.memory_info().rss
        except Exception:
            pass
        try:
            ctx = proc.num_ctx_switches()
            voluntary_ctx += int(ctx.voluntary)
            involuntary_ctx += int(ctx.involuntary)
        except Exception:
            pass
        try:
            io = proc.io_counters()
            read_count += int(io.read_count)
            write_count += int(io.write_count)
            read_bytes += int(io.read_bytes)
            write_bytes += int(io.write_bytes)
        except Exception:
            pass

    return Sample(
        ts=time.time(),
        processes=len(alive),
        threads=threads,
        rss_bytes=rss_bytes,
        voluntary_ctx=voluntary_ctx,
        involuntary_ctx=involuntary_ctx,
        read_count=read_count,
        write_count=write_count,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
    )


class ProcessTreeSampler:
    def __init__(self, root_pid: int, output_csv: Path, interval_seconds: float = 0.2) -> None:
        self._root = psutil.Process(root_pid) if psutil is not None else None
        self._output_csv = output_csv
        self._interval_seconds = interval_seconds
        self._samples: list[Sample] = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        if self._root is None:
            return
        while not self._stop_event.is_set():
            try:
                self._samples.append(_sum_process_tree(self._root))
            except Exception:
                break
            self._stop_event.wait(self._interval_seconds)

    def start(self) -> None:
        if self._root is not None:
            self._thread.start()

    def stop(self) -> dict[str, float | int | None]:
        self._stop_event.set()
        if self._root is not None:
            self._thread.join(timeout=2.0)
        if self._samples:
            self._write_csv()
        return self._summary()

    def _write_csv(self) -> None:
        self._output_csv.parent.mkdir(parents=True, exist_ok=True)
        with self._output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "ts",
                    "processes",
                    "threads",
                    "rss_bytes",
                    "voluntary_ctx",
                    "involuntary_ctx",
                    "read_count",
                    "write_count",
                    "read_bytes",
                    "write_bytes",
                ],
            )
            writer.writeheader()
            for sample in self._samples:
                writer.writerow(sample.__dict__)

    def _summary(self) -> dict[str, float | int | None]:
        if len(self._samples) < 1:
            return {
                "sample_count": 0,
                "peak_rss_bytes": None,
                "peak_threads": None,
                "peak_processes": None,
                "context_switches_per_sec": None,
                "read_count_delta": None,
                "write_count_delta": None,
                "read_bytes_delta": None,
                "write_bytes_delta": None,
            }

        first = self._samples[0]
        last = self._samples[-1]
        duration = max(last.ts - first.ts, 1e-9)
        total_ctx = (last.voluntary_ctx + last.involuntary_ctx) - (
            first.voluntary_ctx + first.involuntary_ctx
        )
        return {
            "sample_count": len(self._samples),
            "peak_rss_bytes": max(sample.rss_bytes for sample in self._samples),
            "peak_threads": max(sample.threads for sample in self._samples),
            "peak_processes": max(sample.processes for sample in self._samples),
            "context_switches_per_sec": total_ctx / duration,
            "read_count_delta": last.read_count - first.read_count,
            "write_count_delta": last.write_count - first.write_count,
            "read_bytes_delta": last.read_bytes - first.read_bytes,
            "write_bytes_delta": last.write_bytes - first.write_bytes,
        }
