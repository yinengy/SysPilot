from __future__ import annotations

from pathlib import Path

from syspilot.ui import tui_sidecar


def test_resolve_default_missing_includes_build_guidance(monkeypatch) -> None:
    monkeypatch.delenv("SYSPILOT_TUI_BIN", raising=False)
    monkeypatch.setattr(
        tui_sidecar,
        "default_tui_binary_paths",
        lambda: [Path("/tmp/missing-release"), Path("/tmp/missing-debug")],
    )

    path, warning = tui_sidecar.resolve_tui_binary_path(override=None)

    assert path is None
    assert warning is not None
    assert "cargo build --release --manifest-path ui/ratatui/Cargo.toml" in warning
    assert "Checked:" in warning


def test_resolve_uses_existing_default_candidate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SYSPILOT_TUI_BIN", raising=False)
    existing = tmp_path / "syspilot-tui"
    existing.write_text("binary")
    monkeypatch.setattr(
        tui_sidecar,
        "default_tui_binary_paths",
        lambda: [tmp_path / "missing", existing],
    )

    path, warning = tui_sidecar.resolve_tui_binary_path(override=None)

    assert path == existing
    assert warning is None
