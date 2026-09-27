# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from vertimosaic.privacy import privacy_research_summary_rows, run_privacy_research

_BOUNDED_METRICS: dict[str, tuple[float, float]] = {
    "membership_roc_auc": (0.0, 1.0),
    "membership_advantage": (0.0, 1.0),
    "label_inference_accuracy": (0.0, 1.0),
    "routing_exposure_fraction": (0.0, 1.0),
    "holdout_roc_auc": (0.0, 1.0),
}


def _apply_metric_support(result: dict[str, Any]) -> None:
    """Keep raw Student-t endpoints while making report endpoints support-aware."""

    experiments = result.get("experiments")
    if not isinstance(experiments, dict):
        return
    for experiment in experiments.values():
        if not isinstance(experiment, dict):
            continue
        summary = experiment.get("summary")
        if not isinstance(summary, list):
            continue
        for row in summary:
            if not isinstance(row, dict):
                continue
            for metric, (lower, upper) in _BOUNDED_METRICS.items():
                interval = row.get(metric)
                if not isinstance(interval, dict):
                    continue
                raw_low = interval.get("ci95_low")
                raw_high = interval.get("ci95_high")
                if not isinstance(raw_low, (int, float)) or not isinstance(raw_high, (int, float)):
                    continue
                interval["ci95_raw_low"] = float(raw_low)
                interval["ci95_raw_high"] = float(raw_high)
                interval["ci95_low"] = max(lower, float(raw_low))
                interval["ci95_high"] = min(upper, float(raw_high))
    result["confidence_interval_reporting"] = (
        "Raw two-sided 95% Student-t endpoints are retained as ci95_raw_low/high; "
        "reported ci95_low/high are clipped only for metrics with known bounded support."
    )


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
    _apply_metric_support(result)
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
