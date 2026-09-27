# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from vertimosaic.experiments import (
    run_ablation_study,
    run_drift_study,
    run_dropout_study,
    run_overlap_study,
    run_synthetic_experiment,
)

SEED = 42
CORE_ROWS = 2000
STUDY_ROWS = 1200
BOOTSTRAP_REPLICATES = 200


def _source_sha() -> str:
    configured = os.environ.get("GITHUB_SHA") or os.environ.get("SOURCE_SHA")
    if configured:
        return configured
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def _core_row(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    communication = result["communication"]
    return {
        "mode": result.get("mode"),
        "model": result.get("model"),
        "rows": result.get("rows"),
        "seed": result.get("seed"),
        "bootstrap_replicates": result.get("bootstrap_replicates"),
        "roc_auc": metrics.get("roc_auc"),
        "pr_auc": metrics.get("pr_auc"),
        "f1": metrics.get("f1"),
        "log_loss": metrics.get("log_loss"),
        "brier": metrics.get("brier"),
        "ece": metrics.get("ece"),
        "threshold": result.get("threshold"),
        "training_steps": result.get("training_steps"),
        "training_seconds": result.get("training_seconds"),
        "inference_seconds": result.get("inference_seconds"),
        "peak_rss_bytes": result.get("peak_rss_bytes"),
        "estimated_communication_bytes": result.get("estimated_communication_bytes"),
        "communication_message_count": communication.get("message_count"),
        "communication_scalar_count": communication.get("scalar_count"),
        "forward_communication_bytes": communication.get("forward_estimated_bytes"),
        "backward_communication_bytes": communication.get("backward_estimated_bytes"),
        "traffic_type": communication.get("traffic_type"),
    }


def build(output: Path) -> None:
    source_sha = _source_sha()
    core = [
        _core_row(
            run_synthetic_experiment(
                rows=CORE_ROWS,
                seed=SEED,
                model_name=model,
                bootstrap_replicates=BOOTSTRAP_REPLICATES,
                write_run=False,
            )
        )
        for model in ("logistic", "vfl-hist-gbdt")
    ]

    with tempfile.TemporaryDirectory(prefix="vertimosaic-space-") as temporary:
        temp = Path(temporary)
        ablation = run_ablation_study(
            rows=STUDY_ROWS,
            seed=SEED,
            model_name="logistic",
            output=temp / "party_ablation.csv",
            write_run=False,
        )
        overlap = run_overlap_study(
            rows=STUDY_ROWS,
            seed=SEED,
            model_name="logistic",
            output=temp / "partial_overlap.csv",
            write_run=False,
        )
        dropout = run_dropout_study(
            rows=STUDY_ROWS,
            seed=SEED,
            max_iter=250,
            output=temp / "party_dropout.csv",
            write_run=False,
        )
        drift = run_drift_study(
            rows=STUDY_ROWS,
            seed=SEED,
            output=temp / "feature_drift.csv",
            write_run=False,
        )

    payload = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": source_sha,
        "links": {
            "github": "https://github.com/sauravsingla/VertiMosaic",
            "dataset": "https://huggingface.co/datasets/sauravsingla08/VertiMosaic-VFL-Benchmark",
            "space": "https://huggingface.co/spaces/sauravsingla08/VertiMosaic",
        },
        "benchmark": {
            "seed": SEED,
            "core_rows": CORE_ROWS,
            "study_rows": STUDY_ROWS,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "party_roles": ["bank", "telecom", "insurance", "retail"],
            "raw_feature_tables_pooled": False,
            "source_type": "fully_synthetic",
        },
        "privacy_boundary": (
            "Research raw-feature locality only; no cryptographic privacy guarantee, PSI, MPC, "
            "homomorphic encryption, secure aggregation, collusion resistance, malicious-party "
            "security or formal differential privacy claim."
        ),
        "external_data_boundary": (
            "No UCI, OpenML or IEEE-CIS source rows are redistributed. The separate four-industry "
            "external benchmark is semi-synthetic and its public sources do not represent the same "
            "real individuals."
        ),
        "core_model_comparison": core,
        "party_ablation": _records(ablation),
        "partial_overlap": _records(overlap),
        "party_dropout": _records(dropout),
        "feature_drift": _records(drift),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build VertiMosaic Hugging Face Space evidence")
    parser.add_argument("--output", type=Path, default=Path("hf-space/data.json"))
    args = parser.parse_args()
    build(args.output)
