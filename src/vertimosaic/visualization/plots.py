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


def save_architecture(output_stem: Path) -> tuple[Path, Path]:
    """Render the four-party vertical-FL protocol without implying cryptographic security."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_axis_off()
    positions = {
        "Bank\nfeatures + labels": (0.18, 0.68),
        "Telecom\nlocal features": (0.18, 0.38),
        "Insurance\nlocal features": (0.18, 0.18),
        "Retail\nlocal features": (0.18, 0.88),
        "Coordinator /\nactive protocol": (0.68, 0.53),
    }
    for label, (x, y) in positions.items():
        ax.text(x, y, label, ha="center", va="center", bbox={"boxstyle": "round,pad=0.5"})
    for label in (
        "Bank\nfeatures + labels",
        "Telecom\nlocal features",
        "Insurance\nlocal features",
        "Retail\nlocal features",
    ):
        x, y = positions[label]
        cx, cy = positions["Coordinator /\nactive protocol"]
        ax.annotate("", xy=(cx - 0.1, cy), xytext=(x + 0.12, y), arrowprops={"arrowstyle": "->"})
    ax.text(
        0.5,
        0.03,
        "Raw feature matrices remain local; simulated protocol metadata/payload signals only",
        ha="center",
        va="center",
    )
    ax.set_title("VertiMosaic vertical federated learning architecture")
    return _save(fig, output_stem)


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


def save_training_loss(
    frame: pd.DataFrame,
    output_stem: Path,
) -> tuple[Path, Path]:
    """Plot measured training/validation loss from a run history table."""
    fig, ax = plt.subplots()
    if "iteration" in frame and "loss" in frame:
        ax.plot(frame["iteration"], frame["loss"], label="training")
        ax.set_xlabel("iteration")
    elif "tree" in frame and "training_loss" in frame:
        ax.plot(frame["tree"], frame["training_loss"], label="training")
        if "validation_loss" in frame:
            ax.plot(frame["tree"], frame["validation_loss"], label="validation")
        ax.set_xlabel("tree")
    else:
        raise ValueError("training history must contain logistic or GBDT loss columns")
    ax.set_ylabel("loss")
    ax.set_title("Training convergence")
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
    series: str | None = None,
) -> tuple[Path, Path]:
    fig, ax = plt.subplots()
    if series is not None and series in frame.columns:
        for label, group in frame.groupby(series, sort=True):
            ordered = group.sort_values(x)
            ax.plot(ordered[x], ordered[y], marker="o", label=str(label))
        ax.legend(title=series)
    else:
        ordered = frame.sort_values(x)
        ax.plot(ordered[x], ordered[y], marker="o")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    return _save(fig, output_stem)
