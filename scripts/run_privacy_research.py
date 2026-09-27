# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from vertimosaic.privacy import privacy_research_summary_rows, run_privacy_research


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VertiMosaic multi-seed privacy/utility research experiments"
    )
    parser.add_argument("--smoke", action="store_true", help="run the reduced CI experiment grid")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/privacy_research.json"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("reports/privacy_research_summary.csv"),
    )
    args = parser.parse_args()

    result = run_privacy_research(smoke=args.smoke)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    summary_rows = privacy_research_summary_rows(result)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in summary_rows for key in row})
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(args.output)
    print(args.csv)


if __name__ == "__main__":
    main()
