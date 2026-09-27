from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from vertimosaic.visualization import (
    save_architecture,
    save_calibration_curve,
    save_category_metric_plot,
    save_pr_curve,
    save_roc_curve,
    save_scaling_plot,
    save_training_loss,
)


def _required(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"required measured artifact is missing: {path}")
    return path


def _metric_interval(metrics: dict[str, Any], name: str) -> str:
    interval = metrics["confidence_intervals"][name]
    return f"[{float(interval['lower']):.6f}, {float(interval['upper']):.6f}]"


def build_main_results_table(
    run_directories: list[Path],
    *,
    output: Path = Path("paper/tables/main_results.csv"),
) -> pd.DataFrame:
    """Build the paper's main result table exclusively from run metrics JSON files."""
    rows: list[dict[str, Any]] = []
    for directory in run_directories:
        metrics = json.loads(_required(directory / "metrics.json").read_text(encoding="utf-8"))
        measured = metrics["metrics"]
        rows.append(
            {
                "Model": metrics["model"],
                "Dataset Mode": metrics["mode"],
                "Parties": "Bank + Telecom + Insurance + Retail",
                "ROC-AUC": measured["roc_auc"],
                "ROC-AUC CI": _metric_interval(metrics, "roc_auc"),
                "PR-AUC": measured["pr_auc"],
                "PR-AUC CI": _metric_interval(metrics, "pr_auc"),
                "Brier": measured["brier"],
                "F1": measured["f1"],
                "Training seconds": metrics["training_seconds"],
                "Peak memory MB": float(metrics["peak_rss_bytes"]) / (1024.0 * 1024.0),
                "Estimated communication MB": float(metrics["estimated_communication_bytes"])
                / (1024.0 * 1024.0),
            }
        )
    frame = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def generate_publication_artifacts(
    primary_run_directory: Path,
    *,
    run_directories: list[Path] | None = None,
    results_directory: Path = Path("results"),
    benchmark_directory: Path = Path("benchmarks"),
    figure_directory: Path = Path("paper/figures"),
    table_output: Path = Path("paper/tables/main_results.csv"),
) -> list[Path]:
    """Generate every required publication figure from measured repository outputs."""
    predictions = pd.read_parquet(_required(primary_run_directory / "predictions.parquet"))
    training = pd.read_csv(_required(primary_run_directory / "training_history.csv"))
    y_true = predictions["target"].to_numpy()
    probabilities = predictions["probability"].to_numpy(dtype=float)

    generated: list[Path] = []
    for paths in (
        save_architecture(figure_directory / "architecture"),
        save_roc_curve(y_true, probabilities, figure_directory / "roc_curve"),
        save_pr_curve(y_true, probabilities, figure_directory / "pr_curve"),
        save_calibration_curve(y_true, probabilities, figure_directory / "calibration_curve"),
        save_training_loss(training, figure_directory / "training_loss"),
    ):
        generated.extend(paths)

    ablation = pd.read_csv(_required(results_directory / "party_ablation.csv"))
    contribution = pd.read_csv(_required(results_directory / "party_contribution.csv"))
    overlap = pd.read_csv(_required(results_directory / "partial_overlap.csv"))
    dropout = pd.read_csv(_required(results_directory / "party_dropout.csv"))
    drift = pd.read_csv(_required(results_directory / "feature_drift.csv"))
    benchmarks = pd.read_csv(_required(benchmark_directory / "results.csv"))

    contribution = contribution.copy()
    contribution["party_method"] = (
        contribution["party"].astype(str) + ":" + contribution["method"].astype(str)
    )
    dropout = dropout.copy()
    dropout["phase_scenario"] = dropout["phase"].astype(str) + ":" + dropout["scenario"].astype(str)

    for paths in (
        save_category_metric_plot(
            ablation,
            category="parties",
            metric="pr_auc",
            title="Party ablation: PR-AUC",
            output_stem=figure_directory / "party_ablation",
        ),
        save_category_metric_plot(
            contribution,
            category="party_method",
            metric="delta",
            title="Predictive party utility",
            output_stem=figure_directory / "party_contribution",
        ),
        save_scaling_plot(
            overlap,
            x="overlap_fraction",
            y="pr_auc",
            title="Performance vs entity overlap",
            output_stem=figure_directory / "performance_vs_overlap",
        ),
        save_category_metric_plot(
            dropout,
            category="phase_scenario",
            metric="pr_auc",
            title="Performance vs party dropout",
            output_stem=figure_directory / "performance_vs_party_dropout",
        ),
        save_category_metric_plot(
            drift,
            category="scenario",
            metric="pr_auc",
            title="Performance vs feature drift",
            output_stem=figure_directory / "performance_vs_drift",
        ),
        save_scaling_plot(
            benchmarks,
            x="rows",
            y="training_seconds",
            title="CPU runtime scaling",
            output_stem=figure_directory / "runtime_scaling",
        ),
        save_scaling_plot(
            benchmarks,
            x="rows",
            y="peak_rss_bytes",
            title="Memory scaling",
            output_stem=figure_directory / "memory_scaling",
        ),
        save_scaling_plot(
            benchmarks,
            x="rows",
            y="estimated_communication_bytes",
            title="Simulated communication payload scaling",
            output_stem=figure_directory / "communication_scaling",
        ),
    ):
        generated.extend(paths)

    build_main_results_table(run_directories or [primary_run_directory], output=table_output)
    generated.append(table_output)
    return generated
