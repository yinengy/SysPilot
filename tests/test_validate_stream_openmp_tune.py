from typing import cast

from scripts.validate_stream_openmp_tune import parse_tune_output


def test_parse_tune_output_accepts_speedup_first_success_and_table_wording() -> None:
    output = """
Round 0 Step 0: Initial Baseline Run
  baseline details
  Wall time: 678.00s

Performance Comparison
│ Wall Time │ 678.00s │ 146.00s │ 4.64x speed up (+78.5%)

✓ Success! 4.64x speed up (78.5% improvement) with actions: OMP_PROC_BIND=close
"""

    parsed = parse_tune_output(output)

    assert parsed["baseline_wall_s"] == 678.0
    assert parsed["tuned_wall_s"] == 146.0
    assert parsed["improvement_pct"] == 78.5
    assert parsed["actions"] == "OMP_PROC_BIND=close"
    assert parsed["success"] is True


def test_parse_tune_output_falls_back_to_wall_times_when_change_text_varies() -> None:
    output = """
Round 0 Step 0: Initial Baseline Run
  baseline details
  Wall time: 678.00s

Performance Comparison
│ Wall Time │ 678.00s │ 146.00s │ speedup recorded elsewhere
"""

    parsed = parse_tune_output(output)

    assert parsed["baseline_wall_s"] == 678.0
    assert parsed["tuned_wall_s"] == 146.0
    assert abs(cast(float, parsed["improvement_pct"]) - 78.4660766961652) < 1e-9
    assert parsed["actions"] == ""
    assert parsed["success"] is True
