from pathlib import Path

import pandas as pd
import pytest

from vertimosaic.reproducibility import RunArtifacts


def test_run_artifacts_write_required_files(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    run = RunArtifacts.create(root=tmp_path, run_id="unit")
    output = run.finalize(
        config={"seed": 42},
        seed=42,
        dataset_provenance={"dataset": "synthetic"},
        linkage_manifest={"method": "synthetic"},
        metrics={"roc_auc": 0.5},
        predictions=pd.DataFrame({"target": [0, 1], "probability": [0.4, 0.6]}),
        training_history=pd.DataFrame({"loss": [0.7]}),
        communication=pd.DataFrame({"estimated_bytes": [8]}),
        feature_provenance=pd.DataFrame({"feature": ["x"]}),
    )
    required = {
        "config.yaml",
        "dataset_provenance.json",
        "linkage_manifest.json",
        "environment.json",
        "metrics.json",
        "predictions.parquet",
        "training_history.csv",
        "communication.csv",
        "feature_provenance.csv",
        "run_manifest.json",
    }
    assert required.issubset({item.name for item in output.iterdir()})
