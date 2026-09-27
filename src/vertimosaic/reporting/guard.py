from __future__ import annotations

from pathlib import Path
from typing import Any

from vertimosaic.reporting.report import write_final_report as _write_final_report


class MissingMeasuredResultsError(ValueError):
    """Raised when a final report is requested without measured run outputs."""


def write_final_report(
    payload: dict[str, Any],
    directory: Path = Path("reports"),
    *,
    results_directory: Path = Path("results"),
    benchmark_directory: Path = Path("benchmarks"),
    case_study_path: Path = Path("reports/case_study.json"),
) -> tuple[Path, Path]:
    """Write a final report only when its primary measured payload is present.

    Study CSVs may still be optional, but the primary experiment payload must
    contain at least measured metrics or measured model-comparison output. This
    prevents an absent input file from being transformed into a plausible-looking
    empty report.
    """
    metrics = payload.get("metrics")
    comparisons = payload.get("comparisons")
    has_metrics = isinstance(metrics, dict) and bool(metrics)
    has_comparisons = isinstance(comparisons, list) and bool(comparisons)
    if not has_metrics and not has_comparisons:
        raise MissingMeasuredResultsError(
            "final report requires measured metrics or measured model comparisons; "
            "generate an experiment payload first"
        )
    return _write_final_report(
        payload,
        directory,
        results_directory=results_directory,
        benchmark_directory=benchmark_directory,
        case_study_path=case_study_path,
    )
