from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def _save(fig: plt.Figure, output_stem: Path) -> tuple[Path, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    fig.tight_layout()
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def save_roc_curve(
    y_true: np.ndarray, probabilities: np.ndarray, output_stem: Path
) -> tuple[Path, Path]:
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, probabilities, ax=ax)
    ax.set_title("ROC curve")
    return _save(fig, output_stem)


def save_pr_curve(
    y_true: np.ndarray, probabilities: np.ndarray, output_stem: Path
) -> tuple[Path, Path]:
    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_true, probabilities, ax=ax)
    ax.set_title("Precision-recall curve")
    return _save(fig, output_stem)


def save_calibration_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    output_stem: Path,
    *,
    bins: int = 10,
) -> tuple[Path, Path]:
    observed, predicted = calibration_curve(y_true, probabilities, n_bins=bins, strategy="uniform")
    fig, ax = plt.subplots()
    ax.plot(predicted, observed, marker="o", label="model")
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", label="ideal")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title("Calibration curve")
    ax.legend()
    return _save(fig, output_stem)


def save_category_metric_plot(
    frame: pd.DataFrame,
    *,
    category: str,
    metric: str,
    title: str,
    output_stem: Path,
) -> tuple[Path, Path]:
    fig, ax = plt.subplots()
    positions = np.arange(len(frame))
    ax.bar(positions, frame[metric].to_numpy(dtype=float))
    ax.set_xticks(positions, frame[category].astype(str), rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title)
    return _save(fig, output_stem)


def save_scaling_plot(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    output_stem: Path,
) -> tuple[Path, Path]:
    fig, ax = plt.subplots()
    ax.plot(frame[x], frame[y], marker="o")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    return _save(fig, output_stem)
