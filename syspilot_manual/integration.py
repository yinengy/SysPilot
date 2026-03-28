from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from syspilot_manual.build_index import build_manual_index, render_index_jsonl
from syspilot_manual.chunking import chunk_source
from syspilot_manual.compose_manuals import compose_manuals
from syspilot_manual.extract_evidence import ExtractionProgress, extract_evidence
from syspilot_manual.ingest import (
    CODE_SUFFIXES,
    DOC_SUFFIXES,
    discover_files,
    ingest_paths,
    load_metadata_source,
)
from syspilot_manual.llm_client import ManualLLMProtocol
from syspilot_manual.merge_evidence import merge_evidence
from syspilot_manual.query_store import ManualStore
from syspilot_manual.render_manual import render_manual_markdown
from syspilot_manual.schemas import (
    ComposedManual,
    EvidenceItem,
    IngestedSource,
    LibraryManual,
    SourceArtifact,
    SourceChunk,
)
from syspilot_manual.synthesize_manual import synthesize_library_manual
from syspilot_manual.verify_manual import verify_or_repair_manual


ARTIFACT_SUBDIRS: tuple[str, ...] = (
    "raw_sources",
    "chunks",
    "evidence",
    "manuals",
    "composed",
    "logs",
)


@dataclass(frozen=True)
class ManualSynthesisProgress:
    manual_id: str
    library_name: str
    status: Literal["started", "verifying", "rendering", "completed"]
    evidence_items: int
    elapsed_s: float
    eta_s: float


@dataclass(frozen=True)
class ManualCompositionProgress:
    manual_id: str
    app_name: str
    status: Literal["started", "verifying", "rendering", "completed"]
    library_count: int
    elapsed_s: float
    eta_s: float


def ensure_artifact_dirs(artifact_root: Path) -> None:
    for name in ARTIFACT_SUBDIRS:
        (artifact_root / name).mkdir(parents=True, exist_ok=True)


def default_manual_root() -> Path:
    return Path(__file__).resolve().parents[1] / "manual"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def source_manifest_path(artifact_root: Path, library_name: str) -> Path:
    return artifact_root / "raw_sources" / library_name / "sources.jsonl"


def chunk_manifest_path(artifact_root: Path, library_name: str) -> Path:
    return artifact_root / "chunks" / library_name / "chunks.jsonl"


def evidence_manifest_path(artifact_root: Path, library_name: str) -> Path:
    return artifact_root / "evidence" / library_name / "evidence.jsonl"


def artifact_manual_dir(artifact_root: Path, manual: LibraryManual | ComposedManual) -> Path:
    folder = "manuals" if isinstance(manual, LibraryManual) else "composed"
    return artifact_root / folder / manual.manual_id


def runtime_manual_dir(manual_root: Path, manual_id: str) -> Path:
    return manual_root / manual_id


def write_sources(artifact_root: Path, library_name: str, sources: list[IngestedSource]) -> None:
    _write_jsonl(
        source_manifest_path(artifact_root, library_name),
        [source.model_dump(mode="json") for source in sources],
    )


def load_sources(artifact_root: Path, library_name: str) -> list[IngestedSource]:
    return [
        IngestedSource.model_validate(row)
        for row in _read_jsonl(source_manifest_path(artifact_root, library_name))
    ]


def write_chunks(artifact_root: Path, library_name: str, chunks: list[SourceChunk]) -> None:
    _write_jsonl(
        chunk_manifest_path(artifact_root, library_name),
        [chunk.model_dump(mode="json") for chunk in chunks],
    )


def load_chunks(artifact_root: Path, library_name: str) -> list[SourceChunk]:
    return [
        SourceChunk.model_validate(row)
        for row in _read_jsonl(chunk_manifest_path(artifact_root, library_name))
    ]


def write_evidence(artifact_root: Path, library_name: str, evidence: list[EvidenceItem]) -> None:
    _write_jsonl(
        evidence_manifest_path(artifact_root, library_name),
        [item.model_dump(mode="json") for item in evidence],
    )


def load_evidence(artifact_root: Path, library_name: str) -> list[EvidenceItem]:
    return [
        EvidenceItem.model_validate(row)
        for row in _read_jsonl(evidence_manifest_path(artifact_root, library_name))
    ]


def _estimate_synthesis_eta(evidence_count: int) -> tuple[float, float]:
    synthesize_s = max(20.0, min(300.0, evidence_count * 0.08))
    verify_s = max(8.0, min(120.0, evidence_count * 0.03))
    return synthesize_s, verify_s


def _estimate_composition_eta(library_count: int) -> tuple[float, float]:
    compose_s = max(12.0, min(90.0, library_count * 10.0))
    verify_s = max(6.0, min(45.0, library_count * 4.0))
    return compose_s, verify_s


def write_manual_bundle(
    manual: LibraryManual | ComposedManual,
    artifact_root: Path,
    manual_root: Path | None = None,
) -> Path:
    manual_root = manual_root or default_manual_root()
    ensure_artifact_dirs(artifact_root)
    manual_root.mkdir(parents=True, exist_ok=True)

    markdown = render_manual_markdown(manual)
    rows = build_manual_index(manual)
    index_jsonl = render_index_jsonl(rows)

    artifact_dir = artifact_manual_dir(artifact_root, manual)
    runtime_dir = runtime_manual_dir(manual_root, manual.manual_id)
    for target_dir in (artifact_dir, runtime_dir):
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "manual.json").write_text(manual.model_dump_json(indent=2), encoding="utf-8")
        (target_dir / "manual.md").write_text(markdown, encoding="utf-8")
        (target_dir / "manual_index.jsonl").write_text(index_jsonl, encoding="utf-8")

    (manual_root / f"{manual.manual_id}.md").write_text(markdown, encoding="utf-8")
    return runtime_dir


def load_manual_json(path: Path) -> LibraryManual | ComposedManual:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("manual_type") == "library":
        return LibraryManual.model_validate(payload)
    return ComposedManual.model_validate(payload)


def ingest_library_inputs(
    library_name: str,
    docs_path: Path,
    artifact_root: Path,
    *,
    code_path: Path | None = None,
    metadata_path: Path | None = None,
    max_chars: int = 1600,
) -> list[SourceChunk]:
    ensure_artifact_dirs(artifact_root)
    sources: list[IngestedSource] = []
    sources.extend(ingest_paths(discover_files(docs_path, suffixes=DOC_SUFFIXES), kind="doc"))
    if code_path is not None:
        sources.extend(
            ingest_paths(discover_files(code_path, suffixes=CODE_SUFFIXES), kind="code_hint")
        )
    if metadata_path is not None and metadata_path.exists():
        sources.append(load_metadata_source(metadata_path))
    sources = sorted(sources, key=lambda source: (source.kind, source.source_id, source.path))
    write_sources(artifact_root, library_name, sources)
    chunks = [chunk for source in sources for chunk in chunk_source(source, max_chars=max_chars)]
    write_chunks(artifact_root, library_name, chunks)
    return chunks


async def extract_library_evidence(
    library_name: str,
    artifact_root: Path,
    llm_client: ManualLLMProtocol,
    *,
    max_retries: int = 2,
    max_concurrency: int = 4,
    progress_callback: Callable[[ExtractionProgress], None] | None = None,
) -> list[EvidenceItem]:
    chunks = load_chunks(artifact_root, library_name)
    merged = merge_evidence(
        await extract_evidence(
            library_name,
            chunks,
            llm_client,
            artifact_root,
            max_retries=max_retries,
            max_concurrency=max_concurrency,
            progress_callback=progress_callback,
        )
    )
    write_evidence(artifact_root, library_name, merged)
    return merged


async def synthesize_library_bundle(
    manual_id: str,
    library_name: str,
    artifact_root: Path,
    llm_client: ManualLLMProtocol,
    *,
    manual_root: Path | None = None,
    progress_callback: Callable[[ManualSynthesisProgress], None] | None = None,
) -> LibraryManual:
    sources = load_sources(artifact_root, library_name)
    evidence = load_evidence(artifact_root, library_name)
    synthesize_eta_s, verify_eta_s = _estimate_synthesis_eta(len(evidence))
    started_at = time.monotonic()
    if progress_callback is not None:
        progress_callback(
            ManualSynthesisProgress(
                manual_id=manual_id,
                library_name=library_name,
                status="started",
                evidence_items=len(evidence),
                elapsed_s=0.0,
                eta_s=synthesize_eta_s + verify_eta_s + 2.0,
            )
        )
    source_refs = [
        SourceArtifact(source_id=source.source_id, path=source.path, kind=source.kind)
        for source in sources
    ]
    manual = await synthesize_library_manual(
        manual_id=manual_id,
        library_name=library_name,
        source_artifacts=source_refs,
        evidence_items=evidence,
        llm_client=llm_client,
    )
    if progress_callback is not None:
        progress_callback(
            ManualSynthesisProgress(
                manual_id=manual_id,
                library_name=library_name,
                status="verifying",
                evidence_items=len(evidence),
                elapsed_s=time.monotonic() - started_at,
                eta_s=verify_eta_s + 1.0,
            )
        )
    verified = await verify_or_repair_manual(manual, llm_client=llm_client)
    if not isinstance(verified, LibraryManual):
        raise TypeError("Expected a library manual after verification")
    if progress_callback is not None:
        progress_callback(
            ManualSynthesisProgress(
                manual_id=manual_id,
                library_name=library_name,
                status="rendering",
                evidence_items=len(evidence),
                elapsed_s=time.monotonic() - started_at,
                eta_s=1.0,
            )
        )
    write_manual_bundle(verified, artifact_root, manual_root)
    if progress_callback is not None:
        progress_callback(
            ManualSynthesisProgress(
                manual_id=manual_id,
                library_name=library_name,
                status="completed",
                evidence_items=len(evidence),
                elapsed_s=time.monotonic() - started_at,
                eta_s=0.0,
            )
        )
    return verified


def render_existing_manual(
    manual_id: str,
    artifact_root: Path,
    manual_root: Path | None = None,
) -> Path:
    manual_root = manual_root or default_manual_root()
    manual = load_manual_json(runtime_manual_dir(manual_root, manual_id) / "manual.json")
    return write_manual_bundle(manual, artifact_root, manual_root)


def build_existing_index(
    manual_id: str,
    artifact_root: Path,
    manual_root: Path | None = None,
) -> Path:
    manual_root = manual_root or default_manual_root()
    manual = load_manual_json(runtime_manual_dir(manual_root, manual_id) / "manual.json")
    return write_manual_bundle(manual, artifact_root, manual_root)


def resolve_library_manual_inputs(
    manual_refs: list[str],
    manual_root: Path,
) -> list[LibraryManual]:
    manuals: list[LibraryManual] = []
    store = ManualStore(manual_root)
    for manual_ref in sorted(set(manual_refs)):
        candidate = Path(manual_ref).expanduser()
        manual = load_manual_json(candidate) if candidate.exists() else store.get_manual(manual_ref)
        if not isinstance(manual, LibraryManual):
            raise TypeError(f"Expected library manual for {manual_ref}")
        manuals.append(manual)
    return manuals


async def compose_and_write_manual(
    manual_id: str,
    app_name: str,
    manual_refs: list[str],
    artifact_root: Path,
    llm_client: ManualLLMProtocol,
    *,
    manual_root: Path | None = None,
    objective: str | None = None,
    symptom: str | None = None,
    progress_callback: Callable[[ManualCompositionProgress], None] | None = None,
) -> ComposedManual:
    manual_root = manual_root or default_manual_root()
    manuals = resolve_library_manual_inputs(manual_refs, manual_root)
    compose_eta_s, verify_eta_s = _estimate_composition_eta(len(manuals))
    started_at = time.monotonic()
    if progress_callback is not None:
        progress_callback(
            ManualCompositionProgress(
                manual_id=manual_id,
                app_name=app_name,
                status="started",
                library_count=len(manuals),
                elapsed_s=0.0,
                eta_s=compose_eta_s + verify_eta_s + 2.0,
            )
        )
    composed = await compose_manuals(
        manual_id=manual_id,
        app_name=app_name,
        manuals=manuals,
        llm_client=llm_client,
        objective=objective,
        symptom=symptom,
    )
    if progress_callback is not None:
        progress_callback(
            ManualCompositionProgress(
                manual_id=manual_id,
                app_name=app_name,
                status="verifying",
                library_count=len(manuals),
                elapsed_s=time.monotonic() - started_at,
                eta_s=verify_eta_s + 1.0,
            )
        )
    verified = await verify_or_repair_manual(composed, llm_client=llm_client)
    if not isinstance(verified, ComposedManual):
        raise TypeError("Expected a composed manual after verification")
    if progress_callback is not None:
        progress_callback(
            ManualCompositionProgress(
                manual_id=manual_id,
                app_name=app_name,
                status="rendering",
                library_count=len(manuals),
                elapsed_s=time.monotonic() - started_at,
                eta_s=1.0,
            )
        )
    write_manual_bundle(verified, artifact_root, manual_root)
    if progress_callback is not None:
        progress_callback(
            ManualCompositionProgress(
                manual_id=manual_id,
                app_name=app_name,
                status="completed",
                library_count=len(manuals),
                elapsed_s=time.monotonic() - started_at,
                eta_s=0.0,
            )
        )
    return verified


def load_store(manual_root: Path | None = None) -> ManualStore:
    return ManualStore(manual_root or default_manual_root())


def build_agent_context_bundle(
    manual_ids: list[str],
    *,
    goal: str | None = None,
    symptom: str | None = None,
    max_items_per_section: int = 5,
    manual_root: Path | None = None,
) -> dict[str, object]:
    return load_store(manual_root).build_agent_context(
        manual_ids,
        goal=goal,
        symptom=symptom,
        max_items_per_section=max_items_per_section,
    )
