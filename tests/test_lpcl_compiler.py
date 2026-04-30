from __future__ import annotations

import re
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

from click.testing import CliRunner

from syspilot.cli import cli
from syspilot_contract.checker import load_contract
from syspilot_contract.compiler import compile_lpcl, compile_lpcl_file, compile_lpcl_from_yaml_text
from syspilot_contract.compose_contracts import compose_contracts
from syspilot_contract.schemas import LibraryContract


GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


OPENMP_SAMPLE_YAML = (
    dedent(
        """
    version: lpcl/v1
    contract: openmp
    type: library
    components:
      - id: openmp_executor
        kind: executor
        attrs:
          workers: OMP_NUM_THREADS
          placement_controls:
            - OMP_PROC_BIND
            - OMP_PLACES
        evidence:
          - openmp-icv-thread-controls
    uses:
      - id: openmp_cpu_compute
        resource: cpu.compute
        by: openmp_executor
        phase: compute
        scope: process
        demand:
          dimension: workers
          min: 1
          max: OMP_NUM_THREADS
        share: shared
        evidence:
          - openmp-icv-thread-controls
      - id: openmp_cpu_placement
        resource: cpu.placement
        by: openmp_executor
        phase: compute
        scope: process
        demand:
          dimension: core_set
          source: OMP_PLACES|OMP_PROC_BIND|KMP_AFFINITY|GOMP_CPU_AFFINITY|KMP_HW_SUBSET
        share: partitioned
        evidence:
          - openmp-knob-affinity-binding
    assumptions:
      - id: openmp_assume_external_pinning_compatible
        resource: cpu.placement
        predicate:
          kind: placement_compatible
        when: composing OpenMP placement with lower-layer pinning or visibility restrictions
        evidence:
          - openmp_pinning_conflict
    controls:
      - id: control_omp_num_threads
        name: OMP_NUM_THREADS
        kind: env
        type: string
        target: openmp_executor.workers
        effect:
          kind: set
          value: comma-separated positive integers by nested level
        evidence:
          - openmp-omp-num-threads-list
    signals:
      - id: signal_omp_display_env
        target: openmp_executor
        metric: effective OpenMP runtime environment
        indicates: actual worker-count-related runtime settings before tuning or comparison
        probe:
          kind: log
          name: OMP_DISPLAY_ENV
        evidence:
          - openmp-obs-display-env
    recipes:
      - id: recipe_shared_cpu_compute_conflict
        for: shared_cpu_compute_conflict
        collect:
          - signal_omp_display_env
        actions:
          - Print effective OpenMP runtime settings before tuning.
          - Sweep OMP_NUM_THREADS on the same workload.
        accept: Stable or improved workload runtime.
        rollback: Restore prior OMP_NUM_THREADS if performance worsens.
        when: OpenMP throughput or runtime degrades as worker count is increased on the same workload
        evidence:
          - openmp_val_print_effective_env_and_placement
    """
    ).strip()
    + "\n"
)


def _golden(name: str) -> str:
    return (GOLDEN_DIR / name).read_text(encoding="utf-8")


def _library_contract(name: str, control_name: str) -> LibraryContract:
    prefix = name.lower()
    return LibraryContract.model_validate(
        {
            "version": "lpcl/v1",
            "contract": name,
            "type": "library",
            "components": [
                {
                    "id": f"{prefix}_executor",
                    "kind": "executor",
                    "attrs": {"workers": "host.cpu_slots"},
                    "evidence": [f"{prefix}_doc"],
                }
            ],
            "uses": [
                {
                    "id": f"{prefix}_cpu_compute",
                    "resource": "cpu.compute",
                    "by": f"{prefix}_executor",
                    "phase": "compute",
                    "scope": "process",
                    "demand": {"dimension": "workers", "min": 1, "max": "host.cpu_slots"},
                    "share": "shared",
                    "evidence": [f"{prefix}_doc"],
                }
            ],
            "assumptions": [
                {
                    "id": f"{prefix}_not_nested",
                    "resource": "cpu.compute",
                    "predicate": {"kind": "not_nested"},
                    "evidence": [f"{prefix}_doc"],
                }
            ],
            "controls": [
                {
                    "id": f"{prefix}_threads_control",
                    "name": control_name,
                    "kind": "env",
                    "type": "int",
                    "target": f"{prefix}_executor.workers",
                    "effect": {"kind": "set", "value": "input"},
                    "evidence": [f"{prefix}_doc"],
                }
            ],
            "signals": [
                {
                    "id": f"{prefix}_thread_count",
                    "target": f"{prefix}_executor.workers",
                    "metric": "active_thread_count",
                    "indicates": "effective_worker_count",
                    "probe": {"kind": "api", "name": control_name.lower()},
                    "evidence": [f"{prefix}_doc"],
                }
            ],
            "recipes": [
                {
                    "id": f"{prefix}_recipe",
                    "for": "shared_cpu_compute_conflict",
                    "collect": [f"{prefix}_thread_count"],
                    "actions": [f"set {control_name} to 1"],
                    "accept": "effective_worker_count == 1",
                    "rollback": "throughput_worse",
                    "evidence": [f"{prefix}_doc"],
                }
            ],
        }
    )


def test_compiler_does_not_call_llm(tmp_path: Path) -> None:
    runner = CliRunner()
    input_path = tmp_path / "openmp.yaml"
    output_path = tmp_path / "openmp.lpcl"
    input_path.write_text(OPENMP_SAMPLE_YAML, encoding="utf-8")

    class _FailIfConstructed:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("LLM must not be constructed during LPCL compilation")

    with patch("syspilot_manual.llm_client.ManualLLMClient", _FailIfConstructed):
        result = runner.invoke(
            cli,
            [
                "compile-lpcl",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ],
        )

    assert result.exit_code == 0
    assert output_path.exists()


def test_openmp_sample_matches_golden_output() -> None:
    assert compile_lpcl_from_yaml_text(OPENMP_SAMPLE_YAML) == _golden("openmp_sample.lpcl")


def test_openmp_sample_can_include_evidence_when_requested() -> None:
    assert compile_lpcl_from_yaml_text(
        OPENMP_SAMPLE_YAML,
        include_evidence=True,
    ) == _golden("openmp_sample_with_evidence.lpcl")


def test_compiler_does_not_emit_loose_syntax() -> None:
    output = compile_lpcl_from_yaml_text(OPENMP_SAMPLE_YAML)
    bad_patterns = [
        r"^lpcl lpcl/v1$",
        r"^contract \w+$",
        r"^type library$",
        r"^component .* kind .*$",
        r"^use \w+$",
        r"^assumption \w+$",
    ]
    for pattern in bad_patterns:
        assert re.search(pattern, output, re.MULTILINE) is None
    assert output.startswith('lpcl "lpcl/v1";\n\ncontract openmp : library {\n')
    assert "evidence = [" not in output


def test_every_scalar_assignment_ends_with_semicolon() -> None:
    output = compile_lpcl_from_yaml_text(OPENMP_SAMPLE_YAML)
    for line in output.splitlines():
        if " = " in line:
            assert line.rstrip().endswith(";")


def test_compiled_output_has_balanced_braces() -> None:
    output = compile_lpcl_from_yaml_text(OPENMP_SAMPLE_YAML)
    depth = 0
    for char in output:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            assert depth >= 0
    assert depth == 0


def test_compiler_escapes_quotes_and_backslashes() -> None:
    contract = LibraryContract.model_validate(
        {
            "version": "lpcl/v1",
            "contract": "escape_demo",
            "type": "library",
            "components": [
                {
                    "id": "demo_executor",
                    "kind": "executor",
                    "attrs": {"note": 'He said "hello" \\ test'},
                    "evidence": ["doc_001"],
                }
            ],
            "uses": [
                {
                    "id": "demo_use",
                    "resource": "cpu.compute",
                    "by": "demo_executor",
                    "phase": "compute",
                    "scope": "process",
                    "demand": {"dimension": "workers", "min": 1, "max": 1},
                    "share": "shared",
                    "evidence": ["doc_001"],
                }
            ],
            "assumptions": [],
            "controls": [],
            "signals": [],
            "recipes": [],
        }
    )

    output = compile_lpcl(contract)

    assert 'note = "He said \\"hello\\" \\\\ test";' in output


def test_compiler_escapes_newlines_and_tabs() -> None:
    contract = LibraryContract.model_validate(
        {
            "version": "lpcl/v1",
            "contract": "escape_controls",
            "type": "library",
            "components": [
                {
                    "id": "control_executor",
                    "kind": "executor",
                    "attrs": {"note": "line1\n\tline2"},
                    "evidence": ["doc_001"],
                }
            ],
            "uses": [
                {
                    "id": "control_use",
                    "resource": "cpu.compute",
                    "by": "control_executor",
                    "phase": "compute",
                    "scope": "process",
                    "demand": {"dimension": "workers", "min": 1, "max": 1},
                    "share": "shared",
                    "evidence": ["doc_001"],
                }
            ],
            "assumptions": [],
            "controls": [],
            "signals": [],
            "recipes": [],
        }
    )

    output = compile_lpcl(contract)

    assert 'note = "line1\\n\\tline2";' in output


def test_invalid_yaml_is_rejected_before_compile() -> None:
    invalid_yaml = dedent(
        """
        version: lpcl/v1
        contract: bad
        type: library
        components:
          - id: bad_executor
            kind: executor
            attrs: {}
            evidence: [doc_001]
        uses:
          - id: bad_use
            resource: worker.pool
            by: bad_executor
            phase: compute
            scope: process
            demand:
              dimension: workers
              min: 1
              max: host.cpu_slots
            share: shared
            evidence: [doc_001]
        assumptions: []
        controls: []
        signals: []
        recipes: []
        """
    )

    try:
        compile_lpcl_from_yaml_text(invalid_yaml)
    except Exception as exc:
        assert "worker.pool" in str(exc)
    else:
        raise AssertionError("invalid LPCL YAML should fail before compilation")


def test_order_preservation_within_declaration_categories() -> None:
    ordered_yaml = dedent(
        """
        version: lpcl/v1
        contract: ordered
        type: library
        components:
          - id: second_component
            kind: executor
            attrs: {}
            evidence: [doc_001]
          - id: first_component
            kind: executor
            attrs: {}
            evidence: [doc_002]
        uses:
          - id: later_use
            resource: cpu.compute
            by: second_component
            phase: compute
            scope: process
            demand:
              dimension: workers
              min: 1
              max: 2
            share: shared
            evidence: [doc_001]
          - id: earlier_use
            resource: cpu.compute
            by: first_component
            phase: compute
            scope: process
            demand:
              dimension: workers
              min: 1
              max: 1
            share: shared
            evidence: [doc_002]
        assumptions: []
        controls: []
        signals: []
        recipes: []
        """
    )

    output = compile_lpcl_from_yaml_text(ordered_yaml)

    assert output.index("component second_component") < output.index("component first_component")
    assert output.index("use later_use") < output.index("use earlier_use")


def test_invalid_identifier_is_rejected_before_compile() -> None:
    invalid_yaml = dedent(
        """
        version: lpcl/v1
        contract: invalid contract
        type: library
        components:
          - id: good_executor
            kind: executor
            attrs: {}
            evidence: [doc_001]
        uses:
          - id: good_use
            resource: cpu.compute
            by: good_executor
            phase: compute
            scope: process
            demand:
              dimension: workers
              min: 1
              max: 1
            share: shared
            evidence: [doc_001]
        assumptions: []
        controls: []
        signals: []
        recipes: []
        """
    )

    try:
        compile_lpcl_from_yaml_text(invalid_yaml)
    except Exception as exc:
        assert "Invalid LPCL identifier" in str(exc)
    else:
        raise AssertionError("invalid LPCL identifiers should fail before compilation")


def test_optional_when_is_omitted_entirely() -> None:
    output = compile_lpcl_from_yaml_text(OPENMP_SAMPLE_YAML)
    assert "when = null;" not in output
    assert 'when = "";' not in output


def test_nested_attrs_object_renders_as_nested_block() -> None:
    nested_yaml = dedent(
        """
        version: lpcl/v1
        contract: nested
        type: library
        components:
          - id: nested_executor
            kind: executor
            attrs:
              workers:
                min: 1
                max: host.cpu_slots
            evidence: [doc_001]
        uses:
          - id: nested_use
            resource: cpu.compute
            by: nested_executor
            phase: compute
            scope: process
            demand:
              dimension: workers
              min: 1
              max: host.cpu_slots
            share: shared
            evidence: [doc_001]
        assumptions: []
        controls: []
        signals: []
        recipes: []
        """
    )

    output = compile_lpcl_from_yaml_text(nested_yaml)

    assert (
        "attrs {\n      workers {\n        min = 1;\n        max = host.cpu_slots;\n      }\n    }"
        in output
    )


def test_compile_lpcl_file_writes_expected_output(tmp_path: Path) -> None:
    input_path = tmp_path / "openmp.yaml"
    output_path = tmp_path / "openmp.lpcl"
    input_path.write_text(OPENMP_SAMPLE_YAML, encoding="utf-8")

    compile_lpcl_file(input_path, output_path)

    assert output_path.read_text(encoding="utf-8") == _golden("openmp_sample.lpcl")


def test_compile_lpcl_file_can_include_evidence(tmp_path: Path) -> None:
    input_path = tmp_path / "openmp.yaml"
    output_path = tmp_path / "openmp_with_evidence.lpcl"
    input_path.write_text(OPENMP_SAMPLE_YAML, encoding="utf-8")

    compile_lpcl_file(input_path, output_path, include_evidence=True)

    assert output_path.read_text(encoding="utf-8") == _golden("openmp_sample_with_evidence.lpcl")


def test_composed_contract_matches_golden_output() -> None:
    composed = compose_contracts(
        "openmp_openblas",
        [
            _library_contract("OpenMP", "OMP_NUM_THREADS"),
            _library_contract("OpenBLAS", "OPENBLAS_NUM_THREADS"),
        ],
    )

    assert compile_lpcl(composed) == _golden("openmp_openblas_composed.lpcl")


def test_composed_contract_can_include_evidence_when_requested() -> None:
    composed = compose_contracts(
        "openmp_openblas",
        [
            _library_contract("OpenMP", "OMP_NUM_THREADS"),
            _library_contract("OpenBLAS", "OPENBLAS_NUM_THREADS"),
        ],
    )

    assert compile_lpcl(composed, include_evidence=True) == _golden(
        "openmp_openblas_composed_with_evidence.lpcl"
    )


def test_tracked_lpcl_artifacts_match_compiler_output() -> None:
    tracked_roots = [
        Path("contracts"),
        Path("artifacts/contracts"),
        Path("artifacts/composed_contracts"),
    ]

    tracked_contracts = [
        contract_yaml
        for root in tracked_roots
        if root.exists()
        for contract_yaml in sorted(root.glob("*/contract.yaml"))
    ]

    assert tracked_contracts

    for contract_yaml in tracked_contracts:
        contract = load_contract(contract_yaml)
        compiled = compile_lpcl(contract)
        lpcl_path = contract_yaml.with_name("contract.lpcl")
        assert lpcl_path.exists()
        assert compiled == lpcl_path.read_text(encoding="utf-8")
