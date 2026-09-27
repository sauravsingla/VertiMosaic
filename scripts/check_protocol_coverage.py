# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CRITICAL_FILES = (
    "vertimosaic/models/vfl_logistic.py",
    "vertimosaic/models/vfl_hist_gbdt.py",
    "vertimosaic/parties/core.py",
    "vertimosaic/parties/coordinator.py",
    "vertimosaic/transport/core.py",
    "vertimosaic/alignment/entity.py",
    "vertimosaic/audit/logging.py",
)


def measured_percentages(report_path: Path) -> dict[str, float]:
    payload: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    files = payload.get("files")
    if not isinstance(files, dict):
        raise ValueError("coverage report does not contain a files mapping")

    percentages: dict[str, float] = {}
    for required in CRITICAL_FILES:
        matches = [
            data
            for filename, data in files.items()
            if str(filename).replace("\\", "/").endswith(required)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"coverage report must contain exactly one entry ending in {required!r}; "
                f"found {len(matches)}"
            )
        summary = matches[0].get("summary")
        if not isinstance(summary, dict) or "percent_covered" not in summary:
            raise ValueError(f"coverage summary is missing percent_covered for {required}")
        percentages[required] = float(summary["percent_covered"])
    return percentages


def enforce_threshold(percentages: dict[str, float], threshold: float) -> None:
    failures: list[str] = []
    for filename in CRITICAL_FILES:
        percent = percentages[filename]
        print(f"{filename}: {percent:.2f}%")
        if percent < threshold:
            failures.append(f"{filename}={percent:.2f}%")
    if failures:
        joined = ", ".join(failures)
        raise SystemExit(f"protocol-critical coverage below {threshold:.2f}%: {joined}")
    print(f"All protocol-critical modules meet the {threshold:.2f}% coverage target.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enforce per-file coverage for VertiMosaic protocol-critical modules."
    )
    parser.add_argument("--coverage", type=Path, default=Path("coverage.json"))
    parser.add_argument("--threshold", type=float, default=90.0)
    args = parser.parse_args()
    if not args.coverage.is_file():
        raise SystemExit(f"coverage report not found: {args.coverage}; run `make coverage` first")
    enforce_threshold(measured_percentages(args.coverage), args.threshold)


if __name__ == "__main__":
    main()
