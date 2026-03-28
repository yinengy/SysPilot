from __future__ import annotations

import json
import re
from pathlib import Path

from syspilot_manual.build_index import SECTION_ALIASES, build_manual_index
from syspilot_manual.compose_manuals import derive_conflicts
from syspilot_manual.schemas import ComposedManual, LibraryManual, ManualIndexEntry


def _tokenize(value: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9_]+", value.lower()) if token]


def _score_row(row: ManualIndexEntry, terms: list[str]) -> int:
    if not terms:
        return 0
    score = 0
    haystacks = {
        5: _tokenize(row.title),
        4: _tokenize(" ".join(row.keywords)),
        3: _tokenize(" ".join(row.symptoms + row.goals)),
        1: _tokenize(row.text),
    }
    for weight, tokens in haystacks.items():
        token_set = set(tokens)
        score += sum(weight for term in terms if term in token_set)
    return score


def _canonical_sections(section: str) -> tuple[str, ...]:
    normalized = section.strip().lower()
    return SECTION_ALIASES.get(normalized, (normalized,))


class ManualStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._manuals: dict[str, LibraryManual | ComposedManual] = {}
        self._indexes: dict[str, list[ManualIndexEntry]] = {}
        self._discover()

    def _discover(self) -> None:
        if not self.root.exists():
            return
        for manual_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            manual_path = manual_dir / "manual.json"
            if not manual_path.exists():
                continue
            payload = json.loads(manual_path.read_text(encoding="utf-8"))
            manual = (
                LibraryManual.model_validate(payload)
                if payload.get("manual_type") == "library"
                else ComposedManual.model_validate(payload)
            )
            self._manuals[manual.manual_id] = manual
            index_path = manual_dir / "manual_index.jsonl"
            if index_path.exists():
                rows = [
                    ManualIndexEntry.model_validate(json.loads(line))
                    for line in index_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            else:
                rows = build_manual_index(manual)
            self._indexes[manual.manual_id] = rows

    def get_manual(self, manual_id: str) -> LibraryManual | ComposedManual:
        return self._manuals[manual_id]

    def list_manuals(self) -> list[str]:
        return sorted(self._manuals)

    def _rows(self, manual_ids: list[str]) -> list[ManualIndexEntry]:
        rows: list[ManualIndexEntry] = []
        for manual_id in sorted(set(manual_ids)):
            rows.extend(self._indexes.get(manual_id, []))
        return rows

    def _rank(self, rows: list[ManualIndexEntry], query_terms: list[str]) -> list[ManualIndexEntry]:
        normalized_terms = []
        for term in query_terms:
            normalized_terms.extend(_tokenize(term))
        scored = [(row, _score_row(row, normalized_terms)) for row in rows]
        scored.sort(
            key=lambda item: (-item[1], item[0].manual_id, item[0].section, item[0].item_id)
        )
        if normalized_terms:
            return [row for row, score in scored if score > 0]
        return [row for row, _ in scored]

    def query_section(
        self,
        manual_id: str,
        section: str,
        keywords: list[str] | None = None,
    ) -> list[dict[str, object]]:
        sections = set(_canonical_sections(section))
        rows = [row for row in self._indexes.get(manual_id, []) if row.section in sections]
        return [row.model_dump(mode="json") for row in self._rank(rows, keywords or [])]

    def query_by_symptom(self, manual_ids: list[str], symptom: str) -> list[dict[str, object]]:
        return [
            row.model_dump(mode="json") for row in self._rank(self._rows(manual_ids), [symptom])
        ]

    def query_by_goal(self, manual_ids: list[str], goal: str) -> list[dict[str, object]]:
        return [row.model_dump(mode="json") for row in self._rank(self._rows(manual_ids), [goal])]

    def get_observability(
        self,
        manual_ids: list[str],
        keywords: list[str] | None = None,
        symptom: str | None = None,
    ) -> list[dict[str, object]]:
        sections = set(_canonical_sections("observability_map"))
        rows = [row for row in self._rows(manual_ids) if row.section in sections]
        terms = list(keywords or []) + ([symptom] if symptom else [])
        return [row.model_dump(mode="json") for row in self._rank(rows, terms)]

    def get_interventions(
        self,
        manual_ids: list[str],
        keywords: list[str] | None = None,
        symptom: str | None = None,
    ) -> list[dict[str, object]]:
        sections = set(_canonical_sections("intervention_graph"))
        rows = [row for row in self._rows(manual_ids) if row.section in sections]
        terms = list(keywords or []) + ([symptom] if symptom else [])
        return [row.model_dump(mode="json") for row in self._rank(rows, terms)]

    def get_conflicts(self, manual_ids: list[str]) -> list[dict[str, object]]:
        conflict_sections = set(_canonical_sections("conflicts"))
        rows = [row for row in self._rows(manual_ids) if row.section in conflict_sections]
        if rows:
            return [row.model_dump(mode="json") for row in self._rank(rows, [])]

        library_manuals = [
            manual
            for manual_id in manual_ids
            for manual in [self._manuals.get(manual_id)]
            if isinstance(manual, LibraryManual)
        ]
        derived = derive_conflicts(library_manuals)
        return [item.model_dump(mode="json") for item in derived]

    def build_agent_context(
        self,
        manual_ids: list[str],
        goal: str | None = None,
        symptom: str | None = None,
        max_items_per_section: int = 5,
    ) -> dict[str, object]:
        observability = self.get_observability(manual_ids, symptom=symptom)[:max_items_per_section]
        if not observability:
            observability = self.get_observability(manual_ids)[:max_items_per_section]
        interventions = self.get_interventions(
            manual_ids,
            keywords=[goal] if goal else None,
            symptom=symptom,
        )[:max_items_per_section]
        if not interventions:
            interventions = self.get_interventions(
                manual_ids,
                keywords=[goal] if goal else None,
            )[:max_items_per_section]
        if not interventions:
            interventions = self.get_interventions(manual_ids)[:max_items_per_section]
        validation_sections = set(_canonical_sections("validation_recipes"))
        validation_rows = [
            row for row in self._rows(manual_ids) if row.section in validation_sections
        ]
        validation = [
            row.model_dump(mode="json")
            for row in self._rank(validation_rows, [value for value in (goal, symptom) if value])[
                :max_items_per_section
            ]
        ]
        if not validation:
            validation = [
                row.model_dump(mode="json")
                for row in self._rank(validation_rows, [])[:max_items_per_section]
            ]
        conflicts = self.get_conflicts(manual_ids)[:max_items_per_section]
        manual_labels = []
        for manual_id in manual_ids:
            manual = self._manuals.get(manual_id)
            if isinstance(manual, LibraryManual):
                manual_labels.append(manual.library_name)
            elif isinstance(manual, ComposedManual):
                manual_labels.append(manual.app_name)
            else:
                manual_labels.append(manual_id)
        summary = (
            f"Structured manual context for {', '.join(sorted(manual_labels))}. "
            f"Goal={goal or 'unspecified'}. Symptom={symptom or 'unspecified'}."
        )
        return {
            "summary": summary,
            "profile_first": observability,
            "candidate_actions": interventions,
            "conflicts": conflicts,
            "validation": validation,
        }
