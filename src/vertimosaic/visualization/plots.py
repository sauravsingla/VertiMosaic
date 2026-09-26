"""Publication-oriented result plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_training_loss(history: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([row["epoch"] for row in history], [row["loss"] for row in history])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Log loss + regularization")
    ax.set_title("VertiMosaic training convergence")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
