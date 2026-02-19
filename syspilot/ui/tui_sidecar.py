from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
from typing import cast
from pathlib import Path

from syspilot.ui.emitter import TuneEventEmitter
from syspilot.ui.events import is_critical_event


DEFAULT_TUI_BIN_ENV = "SYSPILOT_TUI_BIN"


def is_interactive_tty() -> bool:
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except Exception:
        return False


def default_tui_binary_path() -> Path:
    return default_tui_binary_paths()[0]


def default_tui_binary_paths() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    base = repo_root / "ui" / "ratatui" / "target"
    binary_name = "syspilot-tui.exe" if os.name == "nt" else "syspilot-tui"
    return [
        base / "release" / binary_name,
        base / "debug" / binary_name,
    ]


def resolve_tui_binary_path(
    *,
    override: Path | None,
    env_var: str = DEFAULT_TUI_BIN_ENV,
) -> tuple[Path | None, str | None]:
    if override is not None:
        try:
            candidate = override.expanduser()
        except Exception:
            candidate = override
        if candidate.exists() and candidate.is_file():
            return candidate, None
        return None, f"TUI binary from flag not found at: {candidate}"

    env_value = os.environ.get(env_var)
    if env_value:
        candidate = Path(env_value).expanduser()
        if candidate.exists() and candidate.is_file():
            return candidate, None
        return None, f"TUI binary from env:{env_var} not found at: {candidate}"

    default_candidates = default_tui_binary_paths()
    for candidate in default_candidates:
        if candidate.exists() and candidate.is_file():
            return candidate, None

    checked = ", ".join(str(path) for path in default_candidates)
    return (
        None,
        (
            "TUI binary not found. Build with `cargo build --release --manifest-path "
            "ui/ratatui/Cargo.toml` or pass `--tui-bin /path/to/syspilot-tui` "
            f"(or set `SYSPILOT_TUI_BIN`). Checked: {checked}"
        ),
    )


class TuneTuiSidecar:
    def __init__(self, *, binary: Path) -> None:
        self._binary: Path = binary
        self._proc: subprocess.Popen[bytes] | None = None

        self._q: queue.Queue[bytes | None] = queue.Queue(maxsize=512)
        self._writer_thread: threading.Thread | None = None

        self._disconnected: threading.Event = threading.Event()
        self._stream_ended: bool = False

    @property
    def disconnected(self) -> bool:
        if self._disconnected.is_set():
            return True

        proc = self._proc
        if proc is not None:
            try:
                if proc.poll() is not None:
                    self._disconnected.set()
            except Exception:
                pass

        return self._disconnected.is_set()

    def start(self) -> None:
        if self._proc is not None:
            return

        proc: subprocess.Popen[bytes] = subprocess.Popen(
            [str(self._binary)],
            stdin=subprocess.PIPE,
            stdout=None,
            stderr=None,
        )
        if proc.stdin is None:
            proc.terminate()
            raise RuntimeError("failed to open stdin pipe for TUI")

        self._proc = proc

        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="syspilot-tui-writer",
            daemon=True,
        )
        self._writer_thread.start()

    def enqueue_jsonl(self, envelope: dict[str, object]) -> None:
        if self._proc is None or self.disconnected or self._stream_ended:
            return

        try:
            if self._proc.poll() is not None:
                self._disconnected.set()
                return
        except Exception:
            return

        try:
            line = (json.dumps(envelope, separators=(",", ":"), ensure_ascii=True) + "\n").encode(
                "utf-8"
            )
        except Exception:
            return

        try:
            self._q.put_nowait(line)
        except queue.Full:
            event_name = str(envelope.get("event", ""))
            if is_critical_event(event_name):
                try:
                    _ = self._q.get_nowait()
                    self._q.put_nowait(line)
                except Exception:
                    return

    def end_stream(self) -> None:
        if self._proc is None or self._stream_ended:
            return

        self._stream_ended = True
        try:
            self._q.put_nowait(None)
        except queue.Full:
            try:
                _ = self._q.get_nowait()
                self._q.put_nowait(None)
            except Exception:
                return
        except Exception:
            return

    def wait_for_exit(self, *, timeout_s: float | None = None) -> bool:
        if self._proc is None:
            return True

        try:
            _ = self._proc.wait(timeout=timeout_s)
            return True
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False

    def close(self, *, wait_timeout_s: float = 0.5) -> None:
        if self._proc is None:
            return

        self._disconnected.set()
        self.end_stream()
        if self.wait_for_exit(timeout_s=wait_timeout_s):
            return

        try:
            self._proc.terminate()
        except Exception:
            pass

        _ = self.wait_for_exit(timeout_s=wait_timeout_s)

    def _writer_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            return

        while True:
            try:
                item = self._q.get()
            except Exception:
                self._disconnected.set()
                return

            if item is None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                return

            try:
                _ = proc.stdin.write(item)
                proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                self._disconnected.set()
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                return


class TuneTuiOrchestrator:
    def __init__(
        self,
        *,
        requested: str,
        tui_bin_override: Path | None = None,
    ) -> None:
        self.requested: str = requested
        self._override: Path | None = tui_bin_override
        self._sidecar: TuneTuiSidecar | None = None
        self._log_silencer: _SyspilotConsoleLogSilencer = _SyspilotConsoleLogSilencer()

        self._fallback_warned: bool = False
        self._using_tui: bool = False

    @property
    def using_tui(self) -> bool:
        return self._using_tui

    def try_start(self) -> tuple[bool, str | None]:
        if self.requested != "tui":
            self._using_tui = False
            return False, None

        if not is_interactive_tty():
            self._using_tui = False
            return False, "Non-TTY detected; falling back to --ui-mode plain"

        path, err = resolve_tui_binary_path(override=self._override)
        if path is None:
            self._using_tui = False
            return False, (err or "TUI binary unavailable; falling back to --ui-mode plain")

        try:
            sidecar = TuneTuiSidecar(binary=path)
            sidecar.start()
        except Exception as exc:
            self._using_tui = False
            return False, f"Failed to launch TUI ({path}): {exc}; falling back to --ui-mode plain"

        self._sidecar = sidecar
        self._using_tui = True
        self._log_silencer.activate()
        return True, None

    def forward_emitter(self, emitter: TuneEventEmitter) -> None:
        if not self._using_tui or self._sidecar is None:
            return
        if self._sidecar.disconnected:
            return

        try:
            events = emitter.drain()
        except Exception:
            return

        for env in events:
            self._sidecar.enqueue_jsonl(env)

    def disconnected(self) -> bool:
        return bool(self._sidecar is not None and self._sidecar.disconnected)

    def maybe_fallback_warning(self) -> str | None:
        if self._fallback_warned:
            return None
        if not self.disconnected():
            return None
        self._fallback_warned = True
        self._using_tui = False
        self._log_silencer.restore()
        return "TUI disconnected; continuing in --ui-mode plain"

    def close(self, *, wait_for_user: bool = False) -> None:
        sidecar = self._sidecar
        if sidecar is not None and wait_for_user and self._using_tui and not sidecar.disconnected:
            sidecar.end_stream()
            _ = sidecar.wait_for_exit(timeout_s=None)

        self._log_silencer.restore()
        if sidecar is None:
            return
        sidecar.close()


class _SyspilotConsoleLogSilencer:
    def __init__(self) -> None:
        self._logger: logging.Logger = logging.getLogger("syspilot")
        self._removed_stream_handlers: list[logging.Handler] = []
        self._active: bool = False

    def activate(self) -> None:
        if self._active:
            return

        handlers = list(self._logger.handlers)
        for handler in handlers:
            if not isinstance(handler, logging.StreamHandler):
                continue
            if handler in self._logger.handlers:
                typed_handler = cast(logging.Handler, handler)
                self._logger.removeHandler(typed_handler)
                self._removed_stream_handlers.append(typed_handler)

        self._active = True

    def restore(self) -> None:
        if not self._active:
            return

        for handler in self._removed_stream_handlers:
            if handler not in self._logger.handlers:
                self._logger.addHandler(handler)

        self._removed_stream_handlers.clear()
        self._active = False
