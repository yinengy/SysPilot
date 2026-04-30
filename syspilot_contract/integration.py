from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from syspilot_contract.checker import (
    assert_valid_contract,
    dump_contract_yaml,
    load_contract,
)
from syspilot_contract.compose_contracts import compose_contracts
from syspilot_contract.query_store import ContractStore
from syspilot_contract.render_contract import render_contract_lpcl
from syspilot_contract.schemas import AnyContract, ComposedContract, LibraryContract, SourceArtifact
from syspilot_contract.synthesize_contract import synthesize_library_contract
from syspilot_manual.integration import (
    ensure_artifact_dirs,
    load_evidence,
    load_sources,
)
from syspilot_manual.llm_client import ManualLLMProtocol


@dataclass(frozen=True)
class ContractSynthesisProgress:
    contract_id: str
    library_name: str
    status: Literal["started", "checking", "writing", "completed"]
    evidence_items: int
    elapsed_s: float
    eta_s: float


@dataclass(frozen=True)
class ContractCompositionProgress:
    contract_id: str
    app_name: str
    status: Literal["started", "checking", "writing", "completed"]
    library_count: int
    elapsed_s: float
    eta_s: float


def default_contract_root() -> Path:
    return Path(__file__).resolve().parents[1] / "contracts"


def artifact_contract_dir(artifact_root: Path, contract_id: str, *, composed: bool) -> Path:
    folder = "composed_contracts" if composed else "contracts"
    return artifact_root / folder / contract_id


def runtime_contract_dir(contract_root: Path, contract_id: str) -> Path:
    return contract_root / contract_id


def contract_markdown_path(runtime_dir: Path) -> Path:
    return runtime_dir / "contract.md"


def contract_yaml_path(runtime_dir: Path) -> Path:
    return runtime_dir / "contract.yaml"


def contract_lpcl_path(runtime_dir: Path) -> Path:
    return runtime_dir / "contract.lpcl"


def write_contract_bundle(
    contract_id: str,
    contract: AnyContract,
    artifact_root: Path,
    contract_root: Path | None = None,
    *,
    include_evidence: bool = False,
) -> Path:
    contract_root = contract_root or default_contract_root()
    ensure_artifact_dirs(artifact_root)
    contract_root.mkdir(parents=True, exist_ok=True)

    runtime_dir = runtime_contract_dir(contract_root, contract_id)
    artifact_dir = artifact_contract_dir(
        artifact_root,
        contract_id,
        composed=isinstance(contract, ComposedContract),
    )
    yaml_text = dump_contract_yaml(contract)
    lpcl_text = render_contract_lpcl(contract, include_evidence=include_evidence)

    for target_dir in (runtime_dir, artifact_dir):
        target_dir.mkdir(parents=True, exist_ok=True)
        contract_yaml_path(target_dir).write_text(yaml_text, encoding="utf-8")
        contract_lpcl_path(target_dir).write_text(lpcl_text, encoding="utf-8")
    return runtime_dir


def load_contract_yaml(path: Path) -> AnyContract:
    return load_contract(path)


def resolve_library_contract_inputs(
    contract_refs: list[str],
    contract_root: Path,
    *,
    allow_migration: bool = False,
) -> list[LibraryContract]:
    contracts: list[LibraryContract] = []
    for contract_ref in contract_refs:
        candidate = Path(contract_ref).expanduser()
        loaded = (
            load_contract(candidate, allow_migration=allow_migration)
            if candidate.exists()
            else load_contract(
                runtime_contract_dir(contract_root, contract_ref) / "contract.yaml",
                allow_migration=allow_migration,
            )
        )
        if not isinstance(loaded, LibraryContract):
            raise TypeError(f"Expected library contract for {contract_ref}")
        contracts.append(loaded)
    return contracts


async def synthesize_library_contract_bundle(
    contract_id: str,
    library_name: str,
    artifact_root: Path,
    llm_client: ManualLLMProtocol,
    *,
    contract_root: Path | None = None,
    include_evidence: bool = False,
    progress_callback: Callable[[ContractSynthesisProgress], None] | None = None,
) -> LibraryContract:
    contract_root = contract_root or default_contract_root()
    sources = load_sources(artifact_root, library_name)
    evidence = load_evidence(artifact_root, library_name)
    started_at = time.monotonic()
    if progress_callback is not None:
        progress_callback(
            ContractSynthesisProgress(
                contract_id=contract_id,
                library_name=library_name,
                status="started",
                evidence_items=len(evidence),
                elapsed_s=0.0,
                eta_s=max(15.0, len(evidence) * 0.1),
            )
        )
    source_artifacts = [
        SourceArtifact(source_id=source.source_id, path=source.path, kind=source.kind)
        for source in sources
    ]
    contract = await synthesize_library_contract(
        library_name, source_artifacts, evidence, llm_client
    )
    if progress_callback is not None:
        progress_callback(
            ContractSynthesisProgress(
                contract_id=contract_id,
                library_name=library_name,
                status="checking",
                evidence_items=len(evidence),
                elapsed_s=time.monotonic() - started_at,
                eta_s=2.0,
            )
        )
    verified = assert_valid_contract(contract)
    if not isinstance(verified, LibraryContract):
        raise TypeError("Expected a library contract after checking")
    if progress_callback is not None:
        progress_callback(
            ContractSynthesisProgress(
                contract_id=contract_id,
                library_name=library_name,
                status="writing",
                evidence_items=len(evidence),
                elapsed_s=time.monotonic() - started_at,
                eta_s=1.0,
            )
        )
    write_contract_bundle(
        contract_id,
        verified,
        artifact_root,
        contract_root,
        include_evidence=include_evidence,
    )
    if progress_callback is not None:
        progress_callback(
            ContractSynthesisProgress(
                contract_id=contract_id,
                library_name=library_name,
                status="completed",
                evidence_items=len(evidence),
                elapsed_s=time.monotonic() - started_at,
                eta_s=0.0,
            )
        )
    return verified


def compose_and_write_contract(
    contract_id: str,
    app_name: str,
    contract_refs: list[str],
    artifact_root: Path,
    *,
    contract_root: Path | None = None,
    allow_migration: bool = False,
    include_evidence: bool = False,
    progress_callback: Callable[[ContractCompositionProgress], None] | None = None,
) -> ComposedContract:
    contract_root = contract_root or default_contract_root()
    started_at = time.monotonic()
    library_contracts = resolve_library_contract_inputs(
        contract_refs,
        contract_root,
        allow_migration=allow_migration,
    )
    if progress_callback is not None:
        progress_callback(
            ContractCompositionProgress(
                contract_id=contract_id,
                app_name=app_name,
                status="started",
                library_count=len(library_contracts),
                elapsed_s=0.0,
                eta_s=max(6.0, len(library_contracts) * 2.0),
            )
        )
    composed = compose_contracts(app_name, library_contracts)
    if progress_callback is not None:
        progress_callback(
            ContractCompositionProgress(
                contract_id=contract_id,
                app_name=app_name,
                status="checking",
                library_count=len(library_contracts),
                elapsed_s=time.monotonic() - started_at,
                eta_s=1.0,
            )
        )
    verified = assert_valid_contract(composed)
    if not isinstance(verified, ComposedContract):
        raise TypeError("Expected a composed contract after checking")
    if progress_callback is not None:
        progress_callback(
            ContractCompositionProgress(
                contract_id=contract_id,
                app_name=app_name,
                status="writing",
                library_count=len(library_contracts),
                elapsed_s=time.monotonic() - started_at,
                eta_s=1.0,
            )
        )
    write_contract_bundle(
        contract_id,
        verified,
        artifact_root,
        contract_root,
        include_evidence=include_evidence,
    )
    if progress_callback is not None:
        progress_callback(
            ContractCompositionProgress(
                contract_id=contract_id,
                app_name=app_name,
                status="completed",
                library_count=len(library_contracts),
                elapsed_s=time.monotonic() - started_at,
                eta_s=0.0,
            )
        )
    return verified


def load_store(contract_root: Path | None = None) -> ContractStore:
    return ContractStore(contract_root or default_contract_root())


def build_agent_context_bundle(
    contract_ids: list[str],
    *,
    goal: str | None = None,
    symptom: str | None = None,
    max_items_per_section: int = 5,
    contract_root: Path | None = None,
) -> dict[str, object]:
    return load_store(contract_root).build_agent_context(
        contract_ids,
        goal=goal,
        symptom=symptom,
        max_items_per_section=max_items_per_section,
    )
