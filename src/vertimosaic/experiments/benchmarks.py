from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import pandas as pd

from vertimosaic.experiments.pipeline import run_synthetic_experiment
from vertimosaic.reproducibility import environment_snapshot


def run_cpu_benchmarks(
    sizes: list[int],
    *,
    seed: int = 42,
    model_name: str = "logistic",
    bootstrap_replicates: int = 100,
    directory: Path = Path("benchmarks"),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for size in sizes:
        result = run_synthetic_experiment(
            rows=size,
            seed=seed,
            model_name=model_name,
            bootstrap_replicates=bootstrap_replicates,
            write_run=False,
        )
        metrics = result["metrics"]
        rows.append(
            {
                "rows": size,
                "model": model_name,
                "training_seconds": result["training_seconds"],
                "inference_seconds": result["inference_seconds"],
                "peak_rss_bytes": result["peak_rss_bytes"],
                "estimated_communication_bytes": result["estimated_communication_bytes"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "brier": metrics["brier"],
                "f1": metrics["f1"],
            }
        )
    frame = pd.DataFrame(rows)
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / "results.csv", index=False)
    environment = environment_snapshot(seed)
    environment["benchmark_model"] = model_name
    environment["platform_python_implementation"] = platform.python_implementation()
    (directory / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return frame
