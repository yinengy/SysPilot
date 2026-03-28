from pathlib import Path

import pytest

from syspilot.core.config import LLMConfig, LLMProvider
from syspilot.core.llm import LLMTimeoutError
from syspilot_manual.build_index import build_manual_index, render_index_jsonl
from syspilot_manual.chunking import chunk_source
from syspilot_manual.extract_evidence import extract_evidence, extract_evidence_from_chunk
from syspilot_manual.ingest import ingest_file
from syspilot_manual.integration import write_manual_bundle
from syspilot_manual.llm_client import ManualLLMClient, ManualLLMRequestError
from syspilot_manual.render_manual import render_manual_markdown
from syspilot_manual.schemas import EvidenceItem, SourceArtifact
from syspilot_manual.synthesize_manual import synthesize_library_manual


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "examples" / "manual_mvp"


class FakeLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    async def complete_prompt(self, prompt_name: str, payload: dict[str, object]) -> str:
        _ = prompt_name
        _ = payload
        return self.responses.pop(0)


class _RaisingInterface:
    async def chat(self, messages: list[object]) -> object:
        _ = messages
        raise LLMTimeoutError("timed out")


def test_chunking_preserves_source_and_chunk_ids() -> None:
    source = ingest_file(FIXTURE_ROOT / "openmp.md")
    chunks = chunk_source(source)
    assert chunks
    assert chunks[0].source_id == source.source_id
    assert chunks[0].chunk_id.startswith(f"{source.source_id}:chunk_")


def test_ingest_source_ids_are_unique_for_same_stem_paths(tmp_path: Path) -> None:
    first = tmp_path / "tools" / "README.md"
    second = tmp_path / "docs" / "README.md"
    first.parent.mkdir(parents=True, exist_ok=True)
    second.parent.mkdir(parents=True, exist_ok=True)
    first.write_text("# Tools\n", encoding="utf-8")
    second.write_text("# Docs\n", encoding="utf-8")

    first_source = ingest_file(first)
    second_source = ingest_file(second)

    assert first_source.source_id != second_source.source_id


@pytest.mark.asyncio
async def test_extract_evidence_parses_fake_llm_output(tmp_path: Path) -> None:
    source = ingest_file(FIXTURE_ROOT / "openmp.md")
    chunk = chunk_source(source)[0]
    llm = FakeLLMClient(
        [
            "not json",
            """
            {
              "items": [
                {
                  "evidence_id": "ev_001",
                  "library_name": "OpenMP",
                  "chunk_id": "ignored",
                  "kind": "intervention",
                  "title": "Thread count controls oversubscription",
                  "content": "Tune OMP_NUM_THREADS before other thread knobs.",
                  "keywords": ["OMP_NUM_THREADS", "threads"],
                  "resource_domains": ["cpu_threads"],
                  "source_quote": "Adjust OMP_NUM_THREADS to match available cores.",
                  "source_refs": ["doc_001:chunk_001"]
                }
              ]
            }
            """,
        ]
    )
    items = await extract_evidence_from_chunk("OpenMP", chunk, llm, tmp_path)
    assert len(items) == 1
    assert items[0].library_name == "OpenMP"
    assert items[0].chunk_id == chunk.chunk_id
    assert items[0].resource_domains == ["cpu_threads"]


@pytest.mark.asyncio
async def test_extract_evidence_reports_progress_with_bounded_concurrency(tmp_path: Path) -> None:
    source = ingest_file(FIXTURE_ROOT / "openmp.md")
    chunks = chunk_source(source)[:2]
    llm = FakeLLMClient(
        [
            '{"items":[{"evidence_id":"ev_001","library_name":"OpenMP","chunk_id":"ignored","kind":"feature","title":"Threads","content":"Thread guidance.","keywords":["threads"],"resource_domains":["cpu_threads"],"source_quote":"quote","source_refs":["doc:chunk"]}]}',
            '{"items":[{"evidence_id":"ev_002","library_name":"OpenMP","chunk_id":"ignored","kind":"feature","title":"Affinity","content":"Affinity guidance.","keywords":["affinity"],"resource_domains":["cpu_affinity"],"source_quote":"quote","source_refs":["doc:chunk"]}]}',
        ]
    )
    snapshots: list[tuple[str, int, int]] = []

    def on_progress(progress: object) -> None:
        status = getattr(progress, "status", "")
        processed = int(getattr(progress, "processed_chunks", 0))
        total = int(getattr(progress, "total_chunks", 0))
        snapshots.append((status, processed, total))

    items = await extract_evidence(
        "OpenMP",
        chunks,
        llm,
        tmp_path,
        max_concurrency=2,
        progress_callback=on_progress,
    )

    assert len(items) == 2
    assert snapshots[0] == ("started", 0, 2)
    assert snapshots[-1] == ("completed", 2, 2)
    assert any(status == "running" and processed == 1 for status, processed, _ in snapshots)


@pytest.mark.asyncio
async def test_manual_llm_client_rethrows_bounded_error() -> None:
    client = ManualLLMClient(
        config=LLMConfig(provider=LLMProvider.OPENAI, api_key="test"),
        interface=_RaisingInterface(),
    )

    with pytest.raises(ManualLLMRequestError, match="extract_evidence.txt"):
        await client.complete_prompt("extract_evidence.txt", {"library_name": "OpenMP"})


@pytest.mark.asyncio
async def test_extract_evidence_resumes_from_checkpoint(tmp_path: Path) -> None:
    source = ingest_file(FIXTURE_ROOT / "openmp.md")
    chunks = chunk_source(source)[:2]

    first_llm = FakeLLMClient(
        [
            '{"items":[{"evidence_id":"ev_001","library_name":"OpenMP","chunk_id":"ignored","kind":"feature","title":"Threads","content":"Thread guidance.","keywords":["threads"],"resource_domains":["cpu_threads"],"source_quote":"quote","source_refs":["doc:chunk"]}]}'
        ]
    )
    await extract_evidence("OpenMP", chunks[:1], first_llm, tmp_path, max_concurrency=1)

    second_llm = FakeLLMClient(
        [
            '{"items":[{"evidence_id":"ev_002","library_name":"OpenMP","chunk_id":"ignored","kind":"feature","title":"Affinity","content":"Affinity guidance.","keywords":["affinity"],"resource_domains":["cpu_affinity"],"source_quote":"quote","source_refs":["doc:chunk"]}]}'
        ]
    )
    snapshots: list[tuple[str, int]] = []

    def on_progress(progress: object) -> None:
        snapshots.append(
            (str(getattr(progress, "status", "")), int(getattr(progress, "processed_chunks", 0)))
        )

    items = await extract_evidence(
        "OpenMP",
        chunks,
        second_llm,
        tmp_path,
        max_concurrency=2,
        progress_callback=on_progress,
    )

    assert len(items) == 2
    assert snapshots[0] == ("started", 1)
    assert snapshots[-1] == ("completed", 2)


@pytest.mark.asyncio
async def test_synthesize_render_and_index_generation(tmp_path: Path) -> None:
    llm = FakeLLMClient(
        [
            """
            {
              "manual_id": "ignored",
              "manual_type": "library",
              "library_name": "ignored",
              "library_summary": "OpenMP controls CPU threading behavior for shared-memory parallel work.",
              "source_artifacts": [
                {"source_id": "doc_001", "path": "docs/openmp.md", "kind": "doc"}
              ],
              "feature_graph": [
                {
                  "feature_id": "thread_parallelism",
                  "name": "Thread parallelism",
                  "description": "OpenMP exposes thread-count and affinity behavior.",
                  "resource_domains": ["cpu_threads", "cpu_affinity"],
                  "keywords": ["OMP_NUM_THREADS", "OMP_PROC_BIND"],
                  "source_refs": ["doc_001:chunk_001"]
                }
              ],
              "observability_map": [
                {
                  "obs_id": "obs_threads",
                  "feature_id": "thread_parallelism",
                  "question": "How do we detect OpenMP oversubscription?",
                  "priority": 1,
                  "symptoms": ["throughput plateau", "many runnable threads"],
                  "signals": [
                    {"name": "context_switch_rate", "tool": "perf", "reason": "Detect oversubscription"},
                    {"name": "cpu_utilization", "tool": "metrics", "reason": "Detect saturation"}
                  ],
                  "source_refs": ["doc_001:chunk_001"]
                }
              ],
              "intervention_graph": [
                {
                  "intervention_id": "reduce_thread_count",
                  "feature_id": "thread_parallelism",
                  "when": ["cpu saturated", "context switches high"],
                  "actions": [
                    {"kind": "env", "name": "OMP_NUM_THREADS", "suggestion": "set to physical cores or lower"}
                  ],
                  "resource_domains": ["cpu_threads"],
                  "expected_effect": "Reduce oversubscription and coordination overhead.",
                  "side_effects": ["May reduce single-operation parallelism"],
                  "conflicts_with": ["nthread"],
                  "source_refs": ["doc_001:chunk_001"]
                }
              ],
              "validation_recipes": [
                {
                  "validation_id": "val_reduce_thread_count",
                  "for_intervention": "reduce_thread_count",
                  "reruns": 3,
                  "metrics": ["throughput", "latency_p95"],
                  "accept_if": "Median throughput improves and p95 does not regress materially.",
                  "rollback_if": "Throughput regresses or p95 regresses materially.",
                  "source_refs": ["doc_001:chunk_001"]
                }
              ],
              "composition_hints": {
                "shared_resources": ["cpu_threads"],
                "conflict_patterns": [
                  {
                    "with_keywords": ["threads", "workers", "parallelism"],
                    "reason": "CPU thread ownership can conflict across libraries.",
                    "resolution_hint": "Do not increase all thread knobs at once.",
                    "source_refs": ["doc_001:chunk_001"]
                  }
                ]
              }
            }
            """
        ]
    )
    manual = await synthesize_library_manual(
        manual_id="openmp",
        library_name="OpenMP",
        source_artifacts=[SourceArtifact(source_id="doc_001", path="docs/openmp.md", kind="doc")],
        evidence_items=[
            EvidenceItem.model_validate(
                {
                    "evidence_id": "ev_001",
                    "library_name": "OpenMP",
                    "chunk_id": "doc_001:chunk_001",
                    "kind": "intervention",
                    "title": "Thread count controls oversubscription",
                    "content": "Tune OMP_NUM_THREADS before other thread knobs.",
                    "keywords": ["OMP_NUM_THREADS"],
                    "resource_domains": ["cpu_threads"],
                    "source_quote": "Adjust OMP_NUM_THREADS to match available cores.",
                    "source_refs": ["doc_001:chunk_001"],
                }
            )
        ],
        llm_client=llm,
    )
    markdown = render_manual_markdown(manual)
    rows = build_manual_index(manual)
    jsonl = render_index_jsonl(rows)

    artifact_root = tmp_path / "artifacts"
    manual_root = tmp_path / "manual"
    bundle_dir = write_manual_bundle(manual, artifact_root, manual_root)

    assert manual.manual_id == "openmp"
    assert manual.library_name == "OpenMP"
    assert "## Profiling First: Observability Map" in markdown
    assert rows[0].manual_id == "openmp"
    assert '"manual_id": "openmp"' in jsonl
    assert (bundle_dir / "manual.json").exists()
    assert (manual_root / "openmp.md").exists()
