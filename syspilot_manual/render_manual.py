from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from syspilot_manual.schemas import ComposedManual, LibraryManual


def _environment() -> Environment:
    template_dir = Path(__file__).resolve().parent / "templates"
    return Environment(
        loader=FileSystemLoader(str(template_dir)),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def _template_context(manual: LibraryManual | ComposedManual) -> dict[str, Any]:
    payload = manual.model_dump(mode="json")
    if isinstance(manual, LibraryManual):
        payload["shared_resources_text"] = ", ".join(manual.composition_hints.shared_resources)
    return payload


def render_manual_markdown(manual: LibraryManual | ComposedManual) -> str:
    template_name = "manual.md.j2" if isinstance(manual, LibraryManual) else "composed_manual.md.j2"
    rendered = _environment().get_template(template_name).render(**_template_context(manual))
    return rendered.strip() + "\n"
