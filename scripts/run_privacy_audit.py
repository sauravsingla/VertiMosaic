# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vertimosaic.privacy import run_privacy_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VertiMosaic empirical privacy leakage baselines"
    )
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("reports/privacy_audit.json"))
    args = parser.parse_args()

    result = run_privacy_audit(rows=args.rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
