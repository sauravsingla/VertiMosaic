from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from vertimosaic.reporting.interpreter import (
    comparison_observation,
    interpret_calibration,
    interpret_costs,
)


def _read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _overlap_observations(frame: pd.DataFrame) -> list[str]:
    """Describe each measured overlap strategy independently."""
    observations: list[str] = []
    if "method" in frame.columns:
        groups = frame.groupby("method", sort=True)
    else:
        groups = [("intersection_only", frame)]
    for method, group in groups:
        ordered = group.sort_values("overlap_fraction")
        if ordered.empty:
            continue
        low = ordered.iloc[0]
        high = ordered.iloc[-1]
        delta = float(low["pr_auc"]) - float(high["pr_auc"])
        coverage_name = "intersection_coverage" if "intersection_coverage" in ordered else "coverage"
        observations.append(
            f"Partial-overlap study ({method}): PR-AUC changed by {delta:.6f} between "
            f"overlap={float(high['overlap_fraction']):.2f} and "
            f"overlap={float(low['overlap_fraction']):.2f}; "
            f"{coverage_name.replace('_', ' ')} at the latter was "
            f"{float(low[coverage_name]):.6f}."
        )
    return observations


def _dropout_observations(frame: pd.DataFrame) -> list[str]:
    observations: list[str] = []
    for phase, group in frame.groupby("phase", sort=True):
        baseline = group.loc[group["scenario"] == "none"]
        if baseline.empty:
            continue
        baseline_pr = float(baseline.iloc[0]["pr_auc"])
        changed = group.loc[group["scenario"] != "none"]
        if changed.empty:
            continue
        deltas = changed["pr_auc"].astype(float) - baseline_pr
        observations.append(
            f"Party availability ({phase}): PR-AUC deltas relative to the no-dropout "
            f"condition ranged from {float(deltas.min()):.6f} to {float(deltas.max()):.6f}."
        )
    return observations


def _contribution_observation(frame: pd.DataFrame) -> str:
    values = frame["delta"].astype(float)
    return (
        "Predictive party utility study: measured deltas across the reported retraining, "
        f"permutation and Shapley-style methods ranged from {float(values.min()):.6f} "
        f"to {float(values.max()):.6f}; these values are not causal importance."
    )


def _drift_observation(frame: pd.DataFrame) -> str:
    baseline = frame.loc[frame["scenario"] == "baseline"]
    if baseline.empty:
        return "Feature-drift results were present but contained no explicit baseline row."
    baseline_pr = float(baseline.iloc[0]["pr_auc"])
    shifted = frame.loc[frame["scenario"] != "baseline"].copy()
    if shifted.empty:
        return "Feature-drift results contained only the baseline condition."
    shifted["pr_auc_delta"] = shifted["pr_auc"].astype(float) - baseline_pr
    maximum = shifted.loc[shifted["pr_auc_delta"].abs().idxmax()]
    return (
        f"Feature drift: the largest absolute PR-AUC change was "
        f"{float(maximum['pr_auc_delta']):.6f} for scenario {maximum['scenario']}."
    )


def _benchmark_observation(frame: pd.DataFrame) -> str:
    largest = frame.sort_values("rows").iloc[-1]
    memory_mb = float(largest["peak_rss_bytes"]) / (1024.0 * 1024.0)
    communication_mb = float(largest["estimated_communication_bytes"]) / (1024.0 * 1024.0)
    return (
        f"CPU benchmark at {int(largest['rows'])} rows: training took "
        f"{float(largest['training_seconds']):.6f} seconds, peak RSS was "
        f"{memory_mb:.3f} MB, and estimated simulated payload was "
        f"{communication_mb:.3f} MB. These payload bytes are not real network traffic."
    )


def _convergence_observation(frame: pd.DataFrame) -> str:
    """Describe measured loss history without asserting mathematical convergence."""
    for column in ("validation_loss", "loss", "training_loss"):
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        initial = float(values.iloc[0])
        final = float(values.iloc[-1])
        delta = final - initial
        return (
            f"Convergence trace ({column}): measured loss changed from {initial:.6f} "
            f"to {final:.6f} across {len(values)} recorded steps (delta={delta:.6f}). "
            "This is a descriptive training trace, not proof of mathematical convergence."
        )
    return "Training history was present but contained no numeric loss series to interpret."


def build_data_driven_report(
    payload: dict[str, Any],
    *,
    results_directory: Path = Path("results"),
    benchmark_directory: Path = Path("benchmarks"),
    case_study_path: Path = Path("reports/case_study.json"),
) -> dict[str, Any]:
    metrics = payload.get("metrics", {})
    comparisons = payload.get("comparisons", [])
    observations = [comparison_observation(item) for item in comparisons]
    if "pr_auc" in metrics:
        observations.append(f"Held-out test PR-AUC was {float(metrics['pr_auc']):.6f}.")
    if "brier" in metrics and "ece" in metrics:
        observations.append(interpret_calibration(float(metrics["brier"]), float(metrics["ece"])))
    observations.append(
        interpret_costs(
            float(payload["training_seconds"]) if "training_seconds" in payload else None,
            float(payload["estimated_communication_bytes"])
            if "estimated_communication_bytes" in payload
            else None,
        )
    )
    if "training_steps" in payload:
        observations.append(
            f"Training completed with {int(payload['training_steps'])} epochs/trees."
        )

    artifact_status: dict[str, bool] = {}
    run_directory_value = payload.get("run_directory")
    training_history: pd.DataFrame | None = None
    if isinstance(run_directory_value, str) and run_directory_value:
        training_history = _read_csv(Path(run_directory_value) / "training_history.csv")
    artifact_status["training_history"] = training_history is not None
    if training_history is not None and not training_history.empty:
        observations.append(_convergence_observation(training_history))

    overlap = _read_csv(results_directory / "partial_overlap.csv")
    artifact_status["partial_overlap"] = overlap is not None
    if overlap is not None and not overlap.empty:
        observations.extend(_overlap_observations(overlap))
    dropout = _read_csv(results_directory / "party_dropout.csv")
    artifact_status["party_dropout"] = dropout is not None
    if dropout is not None and not dropout.empty and "phase" in dropout:
        observations.extend(_dropout_observations(dropout))
    contribution = _read_csv(results_directory / "party_contribution.csv")
    artifact_status["party_contribution"] = contribution is not None
    if contribution is not None and not contribution.empty:
        observations.append(_contribution_observation(contribution))
    drift = _read_csv(results_directory / "feature_drift.csv")
    artifact_status["feature_drift"] = drift is not None
    if drift is not None and not drift.empty:
        observations.append(_drift_observation(drift))
    benchmark = _read_csv(benchmark_directory / "results.csv")
    artifact_status["cpu_benchmark"] = benchmark is not None
    if benchmark is not None and not benchmark.empty:
        observations.append(_benchmark_observation(benchmark))

    case_study: dict[str, Any] | None = None
    if case_study_path.exists():
        case_study = json.loads(case_study_path.read_text(encoding="utf-8"))
        artifact_status["case_study"] = True
    else:
        artifact_status["case_study"] = False

    return {
        "observations": observations,
        "interpretations": [
            "Conclusions are restricted to measured outputs from the supplied run artifacts.",
            "Negative or uncertain comparisons are retained rather than rewritten as improvements.",
            "Party-contribution results describe predictive utility, not causal importance.",
        ],
        "limitations": [
            "Raw-feature locality does not imply cryptographic confidentiality.",
            (
                "The four-industry external benchmark uses explicitly semi-synthetic "
                "cross-domain linkage."
            ),
            "Simulated payload bytes are not real network-traffic measurements.",
            "Missing study artifacts are not imputed or replaced with fabricated measurements.",
        ],
        "artifact_status": artifact_status,
        "case_study": case_study,
        "source": payload,
    }


def write_final_report(
    payload: dict[str, Any],
    directory: Path = Path("reports"),
    *,
    results_directory: Path = Path("results"),
    benchmark_directory: Path = Path("benchmarks"),
    case_study_path: Path = Path("reports/case_study.json"),
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    report = build_data_driven_report(
        payload,
        results_directory=results_directory,
        benchmark_directory=benchmark_directory,
        case_study_path=case_study_path,
    )
    json_path = directory / "final_report.json"
    md_path = directory / "final_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# VertiMosaic measured-result report", "", "## Observations", ""]
    lines.extend(f"- {item}" for item in report["observations"])
    lines.extend(["", "## Interpretations", ""])
    lines.extend(f"- {item}" for item in report["interpretations"])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    if report["case_study"] is not None:
        case_study = report["case_study"]
        lines.extend(
            [
                "",
                "## Sanitized case study",
                "",
                f"- Entity: `{case_study['entity_id']}`",
                f"- Bank-only risk: {float(case_study['bank_only_risk']):.6f}",
                f"- Four-party VFL risk: {float(case_study['four_party_vfl_risk']):.6f}",
                "- No passive raw feature values are included.",
            ]
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path
