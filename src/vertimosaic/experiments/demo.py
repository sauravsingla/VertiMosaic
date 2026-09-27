from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vertimosaic.experiments.pipeline import run_synthetic_experiment


def run_demo(
    rows: int = 2000,
    seed: int = 42,
    model_name: str = "logistic",
    *,
    runs_root: Path = Path("runs"),
) -> dict[str, Any]:
    """CPU smoke demo with the same reproducibility contract as full experiments."""
    payload = run_synthetic_experiment(
        rows=rows,
        seed=seed,
        model_name=model_name,
        bootstrap_replicates=100,
        write_run=True,
        runs_root=runs_root,
    )
    metrics: dict[str, Any] = dict(payload["metrics"])
    metrics["threshold_selected_on_validation"] = payload["threshold"]
    metrics["estimated_communication_bytes"] = payload["estimated_communication_bytes"]
    metrics["run_directory"] = payload["run_directory"]
    metrics["bootstrap_replicates"] = payload["bootstrap_replicates"]
    return metrics


def write_demo_report(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
