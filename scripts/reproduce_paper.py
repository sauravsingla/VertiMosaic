"""Reproduce VertiMosaic measured research artifacts on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vertimosaic.experiments import (
    generate_publication_artifacts,
    run_ablation_study,
    run_contribution_study,
    run_cpu_benchmarks,
    run_distributed_signal_case_study,
    run_drift_study,
    run_dropout_study,
    run_feature_importance_study,
    run_missing_party_methods_study,
    run_overlap_study,
    run_synthetic_experiment,
)
from vertimosaic.reporting import write_final_report


def main(*, smoke: bool = False) -> None:
    bootstrap_replicates = 100 if smoke else 1000
    experiment_rows = 2000 if smoke else 5000
    benchmark_sizes = [1000, 3000] if smoke else [10_000, 30_000, 50_000, 100_000]

    logistic = run_synthetic_experiment(
        rows=experiment_rows,
        seed=42,
        model_name="logistic",
        bootstrap_replicates=bootstrap_replicates,
        write_run=True,
    )
    gbdt = run_synthetic_experiment(
        rows=experiment_rows,
        seed=42,
        model_name="vfl-hist-gbdt",
        bootstrap_replicates=bootstrap_replicates,
        write_run=True,
    )

    run_ablation_study(rows=1200 if smoke else 3000, seed=42)
    run_contribution_study(rows=800 if smoke else 2000, seed=42)
    run_overlap_study(rows=2000 if smoke else 5000, seed=42)
    run_dropout_study(rows=1600 if smoke else 4000, seed=42, max_iter=100 if smoke else 350)
    run_drift_study(rows=1600 if smoke else 4000, seed=42)
    run_missing_party_methods_study(rows=1800 if smoke else 4000, seed=42)
    run_feature_importance_study(rows=1200 if smoke else 3000, seed=42)
    run_cpu_benchmarks(
        benchmark_sizes,
        seed=42,
        model_name="logistic",
        bootstrap_replicates=100,
    )

    if not smoke:
        run_distributed_signal_case_study(seed=42)

    logistic_run = Path(str(logistic["run_directory"]))
    gbdt_run = Path(str(gbdt["run_directory"]))
    generate_publication_artifacts(
        gbdt_run,
        run_directories=[logistic_run, gbdt_run],
    )

    reports = Path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    experiment_payload = reports / "experiment_payload.json"
    experiment_payload.write_text(
        json.dumps(gbdt, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    write_final_report(gbdt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Use smaller CPU-friendly sizes and 100 bootstrap replicates; "
            "skip external case study."
        ),
    )
    arguments = parser.parse_args()
    main(smoke=arguments.smoke)
