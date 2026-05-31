#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    from experiments.LibContentBench.opentuner_integration.cli import main as integration_main

    return integration_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
