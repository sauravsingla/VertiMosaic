import json
from pathlib import Path

import pandas as pd

from vertimosaic.reproducibility import RunArtifacts


def test_run_manifest_records_hashes_dependencies_and_required_files(tmp_path: Path) -> None:
    run = RunArtifacts.create(root=tmp_path, run_id="test-run")
    directory = run.finalize(
        config={"model": "logistic", "seed": 7},
        seed=7,
        dataset_provenance={"source_type": "fully_synthetic", "dataset_hashes": {"x": "abc"}},
        linkage_manifest={"method": "synthetic", "target_blind": True},
        metrics={"roc_auc": 0.5},
        predictions=pd.DataFrame({"target": [0, 1], "probability": [0.25, 0.75]}),
        training_history=pd.DataFrame({"iteration": [0], "loss": [0.69]}),
        communication=pd.DataFrame({"estimated_bytes": [8]}),
        feature_provenance=pd.DataFrame({"party": ["bank"], "feature": ["feature_0"]}),
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
    assert required.issubset({path.name for path in directory.iterdir()})
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_hashes"] == {"x": "abc"}
    assert len(manifest["configuration_hash"]) == 64
    assert len(manifest["dataset_provenance_hash"]) == 64
    assert "numpy" in manifest["dependency_versions"]
    assert set(manifest["artifact_sha256"]) == required - {"run_manifest.json"}
    assert all(len(value) == 64 for value in manifest["artifact_sha256"].values())
