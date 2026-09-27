from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import pandas as pd

from vertimosaic.datasets import fetch_external_party
from vertimosaic.experiments.external_run import run_external_experiment
from vertimosaic.experiments.pipeline import run_synthetic_experiment
from vertimosaic.reproducibility import environment_snapshot

_EXTERNAL_MODES = {"observed_target_external", "distributed_signal_external"}


def _benchmark_row(
    result: dict[str, Any],
    *,
    rows: int,
    model_name: str,
    mode: str,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    metrics = result["metrics"]
    communication = result["communication"]
    return {
        "mode": mode,
        "rows": rows,
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


def _write_benchmark_outputs(
    frame: pd.DataFrame,
    *,
    directory: Path,
    seed: int,
    model_name: str,
    bootstrap_replicates: int,
) -> None:
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
    """Run the fully synthetic CPU scaling grid."""
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
        rows.append(
            _benchmark_row(
                result,
                rows=size,
                model_name=model_name,
                mode="synthetic_scale",
                bootstrap_replicates=bootstrap_replicates,
            )
        )
    frame = pd.DataFrame(rows)
    _write_benchmark_outputs(
        frame,
        directory=directory,
        seed=seed,
        model_name=model_name,
        bootstrap_replicates=bootstrap_replicates,
    )
    return frame


def run_external_cpu_benchmark(
    *,
    mode: str = "observed_target_external",
    seed: int = 42,
    model_name: str = "vfl-hist-gbdt",
    bootstrap_replicates: int = 1000,
    cross_party_correlation: float = 0.25,
    insurance_sample_size: int | None = None,
    directory: Path = Path("benchmarks"),
    write_runs: bool = True,
    runs_root: Path = Path("runs"),
    append: bool = True,
) -> pd.DataFrame:
    """Benchmark an external mode once at the actual available Bank anchor size."""
    if mode not in _EXTERNAL_MODES:
        raise ValueError(f"external benchmark mode must be one of {sorted(_EXTERNAL_MODES)}")
    bundles = {
        name: fetch_external_party(
            name,
            insurance_sample_size=insurance_sample_size,
            seed=seed,
        )
        for name in ("bank", "telecom", "insurance", "retail")
    }
    anchor_rows = len(bundles["bank"].features)
    result = run_external_experiment(
        mode=mode,
        model_name=model_name,
        seed=seed,
        cross_party_correlation=cross_party_correlation,
        insurance_sample_size=insurance_sample_size,
        bootstrap_replicates=bootstrap_replicates,
        write_run=write_runs,
        runs_root=runs_root,
        bundles=bundles,
    )
    new_frame = pd.DataFrame(
        [
            _benchmark_row(
                result,
                rows=anchor_rows,
                model_name=model_name,
                mode=mode,
                bootstrap_replicates=bootstrap_replicates,
            )
        ]
    )
    results_path = directory / "results.csv"
    if append and results_path.exists():
        existing = pd.read_csv(results_path)
        frame = pd.concat([existing, new_frame], ignore_index=True, sort=False)
    else:
        frame = new_frame
    _write_benchmark_outputs(
        frame,
        directory=directory,
        seed=seed,
        model_name=model_name,
        bootstrap_replicates=bootstrap_replicates,
    )
    return frame
