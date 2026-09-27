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
    bootstrap_replicates: int = 1000,
    directory: Path = Path("benchmarks"),
    write_runs: bool = True,
    runs_root: Path = Path("runs"),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for size in sizes:
        result = run_synthetic_experiment(
            rows=size,
            seed=seed,
            model_name=model_name,
            bootstrap_replicates=bootstrap_replicates,
            write_run=write_runs,
            runs_root=runs_root,
        )
        metrics = result["metrics"]
        communication = result["communication"]
        rows.append(
            {
                "rows": size,
                "model": model_name,
                "bootstrap_replicates": bootstrap_replicates,
                "data_preparation_seconds": result["data_preparation_seconds"],
                "entity_alignment_seconds": result["entity_alignment_seconds"],
                "preprocessing_seconds": result["preprocessing_seconds"],
                "training_seconds": result["training_seconds"],
                "inference_seconds": result["inference_seconds"],
                "training_steps": result["training_steps"],
                "peak_rss_bytes": result["peak_rss_bytes"],
                "estimated_communication_bytes": result["estimated_communication_bytes"],
                "communication_message_count": communication["message_count"],
                "communication_scalar_count": communication["scalar_count"],
                "forward_communication_bytes": communication["forward_estimated_bytes"],
                "backward_communication_bytes": communication["backward_estimated_bytes"],
                "traffic_type": communication["traffic_type"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "brier": metrics["brier"],
                "f1": metrics["f1"],
                "run_directory": result.get("run_directory"),
            }
        )
    frame = pd.DataFrame(rows)
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / "results.csv", index=False)
    environment = environment_snapshot(seed)
    environment["benchmark_model"] = model_name
    environment["bootstrap_replicates"] = bootstrap_replicates
    environment["platform_python_implementation"] = platform.python_implementation()
    environment["traffic_type"] = "SIMULATED PAYLOAD SIZE"
    (directory / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return frame
