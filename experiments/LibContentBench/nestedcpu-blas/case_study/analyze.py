from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import AnalysisValidationError, analyze_studies, write_analysis_artifacts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze nestedcpu-blas shared case-study ledgers deterministically."
    )
    parser.add_argument("--concord-ledger", type=Path)
    parser.add_argument("--smac-ledger", type=Path)
    parser.add_argument("--opentuner-ledger", type=Path)
    parser.add_argument("--smac-native", type=Path)
    parser.add_argument("--opentuner-native", type=Path)
    parser.add_argument("--metric", default="throughput")
    parser.add_argument("--metric-mode", choices=("maximize", "minimize"), default="maximize")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.smac_native is not None and args.smac_ledger is None:
        parser.error("--smac-native requires --smac-ledger")
    if args.opentuner_native is not None and args.opentuner_ledger is None:
        parser.error("--opentuner-native requires --opentuner-ledger")
    inputs = {}
    if args.concord_ledger is not None:
        inputs["concord"] = (args.concord_ledger, None)
    if args.smac_ledger is not None:
        inputs["smac3"] = (args.smac_ledger, args.smac_native)
    if args.opentuner_ledger is not None:
        inputs["opentuner"] = (args.opentuner_ledger, args.opentuner_native)
    try:
        aggregate, trajectory = analyze_studies(
            inputs, metric_name=args.metric, metric_mode=args.metric_mode
        )
        paths = write_analysis_artifacts(args.output_dir, aggregate, trajectory)
    except (AnalysisValidationError, ValueError) as exc:
        parser.exit(2, f"analysis validation error: {exc}\n")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
