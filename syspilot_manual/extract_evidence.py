from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import ValidationError

from syspilot_manual.llm_client import ManualLLMProtocol, extract_json_object
from syspilot_manual.schemas import EvidenceBatch, EvidenceItem, SourceChunk


@dataclass(frozen=True)
class ExtractionProgress:
    library_name: str
    status: Literal["started", "running", "completed"]
    total_chunks: int
    resumed_chunks: int
    processed_chunks: int
    succeeded_chunks: int
    failed_chunks: int
    current_chunk_id: str
    current_source_id: str
    current_items: int
    elapsed_s: float
    avg_chunk_s: float
    eta_s: float


def _attempt_dir(artifact_root: Path, library_name: str) -> Path:
    return artifact_root / "logs" / library_name / "extract_evidence"


def _failure_path(artifact_root: Path, library_name: str) -> Path:
    return artifact_root / "evidence" / library_name / "failures.jsonl"


def _checkpoint_state_path(artifact_root: Path, library_name: str) -> Path:
    return artifact_root / "evidence" / library_name / "checkpoint_state.json"


def _checkpoint_evidence_path(artifact_root: Path, library_name: str) -> Path:
    return artifact_root / "evidence" / library_name / "checkpoint_evidence.jsonl"


def _load_checkpoint(
    artifact_root: Path,
    library_name: str,
) -> tuple[list[EvidenceItem], set[str]]:
    state_path = _checkpoint_state_path(artifact_root, library_name)
    evidence_path = _checkpoint_evidence_path(artifact_root, library_name)
    completed_chunk_ids: set[str] = set()
    if state_path.exists():
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        completed_chunk_ids = {
            str(chunk_id) for chunk_id in payload.get("completed_chunk_ids", []) if chunk_id
        }
    if completed_chunk_ids and not evidence_path.exists():
        completed_chunk_ids = set()

    evidence: list[EvidenceItem] = []
    if evidence_path.exists():
        for line in evidence_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            evidence.append(EvidenceItem.model_validate(json.loads(line)))
    return evidence, completed_chunk_ids


def _append_checkpoint_evidence(
    artifact_root: Path,
    library_name: str,
    items: list[EvidenceItem],
) -> None:
    if not items:
        return
    evidence_path = _checkpoint_evidence_path(artifact_root, library_name)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with evidence_path.open("a", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.model_dump(mode="json"), sort_keys=True) + "\n")


def _write_checkpoint_state(
    artifact_root: Path,
    library_name: str,
    total_chunks: int,
    completed_chunk_ids: set[str],
) -> None:
    state_path = _checkpoint_state_path(artifact_root, library_name)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "library_name": library_name,
                "total_chunks": total_chunks,
                "completed_chunk_ids": sorted(completed_chunk_ids),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_attempt_logs(
    artifact_root: Path,
    library_name: str,
    chunk_id: str,
    attempt: int,
    payload: dict[str, Any],
    response_text: str,
) -> None:
    log_dir = _attempt_dir(artifact_root, library_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{chunk_id}.attempt_{attempt}.prompt.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (log_dir / f"{chunk_id}.attempt_{attempt}.response.txt").write_text(
        response_text,
        encoding="utf-8",
    )


def _normalize_items(
    library_name: str,
    chunk: SourceChunk,
    items: list[EvidenceItem],
) -> list[EvidenceItem]:
    normalized: list[EvidenceItem] = []
    for index, item in enumerate(items):
        normalized.append(
            item.model_copy(
                update={
                    "evidence_id": item.evidence_id or f"ev_{chunk.order:03d}_{index:02d}",
                    "library_name": library_name,
                    "chunk_id": chunk.chunk_id,
                    "source_refs": item.source_refs or [chunk.chunk_id],
                }
            )
        )
    return normalized


async def extract_evidence_from_chunk(
    library_name: str,
    chunk: SourceChunk,
    llm_client: ManualLLMProtocol,
    artifact_root: Path,
    *,
    max_retries: int = 2,
) -> list[EvidenceItem]:
    payload: dict[str, Any] = {
        "library_name": library_name,
        "chunk": chunk.model_dump(mode="json"),
        "schema": EvidenceBatch.model_json_schema(),
    }
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        response_text = await llm_client.complete_prompt("extract_evidence.txt", payload)
        _write_attempt_logs(
            artifact_root,
            library_name,
            chunk.chunk_id.replace(":", "_"),
            attempt,
            payload,
            response_text,
        )
        try:
            parsed = extract_json_object(response_text)
            batch = EvidenceBatch.model_validate(parsed)
            return _normalize_items(library_name, chunk, batch.items)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            payload = {
                **payload,
                "retry_feedback": str(exc),
                "retry_instruction": "Return valid JSON only and do not invent unsupported claims.",
            }

    if last_error is None:
        raise RuntimeError(f"Evidence extraction failed for {chunk.chunk_id}")
    raise last_error


async def extract_evidence(
    library_name: str,
    chunks: list[SourceChunk],
    llm_client: ManualLLMProtocol,
    artifact_root: Path,
    *,
    max_retries: int = 2,
    max_concurrency: int = 4,
    progress_callback: Callable[[ExtractionProgress], None] | None = None,
) -> list[EvidenceItem]:
    evidence_dir = artifact_root / "evidence" / library_name
    evidence_dir.mkdir(parents=True, exist_ok=True)
    failure_path = _failure_path(artifact_root, library_name)
    if failure_path.exists():
        failure_path.unlink()

    ordered_chunks = sorted(chunks, key=lambda item: (item.source_id, item.order, item.chunk_id))
    total_chunks = len(ordered_chunks)
    evidence, completed_chunk_ids = _load_checkpoint(artifact_root, library_name)
    remaining_chunks = [
        chunk for chunk in ordered_chunks if chunk.chunk_id not in completed_chunk_ids
    ]
    resumed_chunks = len(completed_chunk_ids)
    started_at = time.monotonic()
    succeeded_chunks = resumed_chunks
    failed_chunks = 0
    concurrency = max(1, min(max_concurrency, len(remaining_chunks) or 1))
    semaphore = asyncio.Semaphore(concurrency)

    async def _extract_one(
        chunk: SourceChunk,
    ) -> tuple[SourceChunk, list[EvidenceItem] | None, Exception | None]:
        try:
            async with semaphore:
                chunk_items = await extract_evidence_from_chunk(
                    library_name,
                    chunk,
                    llm_client,
                    artifact_root,
                    max_retries=max_retries,
                )
            return chunk, chunk_items, None
        except Exception as exc:
            return chunk, None, exc

    if progress_callback is not None:
        progress_callback(
            ExtractionProgress(
                library_name=library_name,
                status="started",
                total_chunks=total_chunks,
                resumed_chunks=resumed_chunks,
                processed_chunks=resumed_chunks,
                succeeded_chunks=succeeded_chunks,
                failed_chunks=0,
                current_chunk_id="",
                current_source_id="",
                current_items=0,
                elapsed_s=0.0,
                avg_chunk_s=0.0,
                eta_s=0.0,
            )
        )

    tasks = [asyncio.create_task(_extract_one(chunk)) for chunk in remaining_chunks]

    session_processed_chunks = 0
    for task in asyncio.as_completed(tasks):
        chunk, chunk_items, error = await task
        session_processed_chunks += 1
        processed_chunks = resumed_chunks + session_processed_chunks
        current_items = 0
        try:
            if error is not None:
                raise error
            assert chunk_items is not None
            evidence.extend(chunk_items)
            current_items = len(chunk_items)
            _append_checkpoint_evidence(artifact_root, library_name, chunk_items)
            completed_chunk_ids.add(chunk.chunk_id)
            _write_checkpoint_state(artifact_root, library_name, total_chunks, completed_chunk_ids)
            succeeded_chunks += 1
        except Exception as exc:
            failed_chunks += 1
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            with failure_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "library_name": library_name,
                            "chunk_id": chunk.chunk_id,
                            "source_id": chunk.source_id,
                            "error": str(exc),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

        if progress_callback is not None:
            elapsed_s = time.monotonic() - started_at
            avg_chunk_s = elapsed_s / session_processed_chunks if session_processed_chunks else 0.0
            eta_s = avg_chunk_s * max(total_chunks - processed_chunks, 0)
            progress_callback(
                ExtractionProgress(
                    library_name=library_name,
                    status="running",
                    total_chunks=total_chunks,
                    resumed_chunks=resumed_chunks,
                    processed_chunks=processed_chunks,
                    succeeded_chunks=succeeded_chunks,
                    failed_chunks=failed_chunks,
                    current_chunk_id=chunk.chunk_id,
                    current_source_id=chunk.source_id,
                    current_items=current_items,
                    elapsed_s=elapsed_s,
                    avg_chunk_s=avg_chunk_s,
                    eta_s=eta_s,
                )
            )

    if progress_callback is not None:
        elapsed_s = time.monotonic() - started_at
        avg_chunk_s = elapsed_s / session_processed_chunks if session_processed_chunks else 0.0
        progress_callback(
            ExtractionProgress(
                library_name=library_name,
                status="completed",
                total_chunks=total_chunks,
                resumed_chunks=resumed_chunks,
                processed_chunks=total_chunks,
                succeeded_chunks=succeeded_chunks,
                failed_chunks=failed_chunks,
                current_chunk_id="",
                current_source_id="",
                current_items=0,
                elapsed_s=elapsed_s,
                avg_chunk_s=avg_chunk_s,
                eta_s=0.0,
            )
        )

    return sorted(evidence, key=lambda item: (item.chunk_id, item.evidence_id, item.title))
