import json
from pathlib import Path

import numpy as np
import pandas as pd

from vertimosaic.datasets import ExternalDatasetBundle
from vertimosaic.experiments.external_run import run_external_experiment
from vertimosaic.provenance import FeatureProvenance
from vertimosaic.reproducibility import RunArtifacts


def _bundle(
    party: str,
    values: np.ndarray,
    *,
    target: pd.Series | None = None,
) -> ExternalDatasetBundle:
    frame = pd.DataFrame(values, columns=["feature_a", "feature_b"])
    provenance = [
        FeatureProvenance(
            party=party,
            feature=column,
            external_dataset=f"test-{party}",
            source_column=column,
            transformation="identity test fixture",
            source_type="real_external",
            observed_or_derived="observed",
            semi_synthetic=False,
        )
        for column in frame.columns
    ]
    return ExternalDatasetBundle(
        party=party,
        features=frame,
        target=target,
        provenance=provenance,
        metadata={"license": "test-license", "processed_rows": len(frame)},
    )


def test_external_experiment_writes_complete_reproducibility_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(31)
    rows = 240
    bank_values = rng.normal(size=(rows, 2))
    target = pd.Series((bank_values[:, 0] + 0.4 * bank_values[:, 1] > 0.0).astype(float))
    bundles = {
        "bank": _bundle("bank", bank_values, target=target),
        "telecom": _bundle("telecom", rng.normal(size=(rows, 2))),
        "insurance": _bundle("insurance", rng.normal(size=(rows, 2))),
        "retail": _bundle("retail", rng.normal(size=(rows, 2))),
    }

    result = run_external_experiment(
        mode="observed_target_external",
        model_name="logistic",
        seed=31,
        bootstrap_replicates=20,
        runs_root=tmp_path / "runs",
        bundles=bundles,
    )
    run_directory = Path(result["run_directory"])
    expected = {
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
    assert expected <= {path.name for path in run_directory.iterdir()}
    predictions = pd.read_parquet(run_directory / "predictions.parquet")
    assert "entity_id" in predictions
    assert "row_index" not in predictions
    assert predictions["entity_id"].str.len().eq(64).all()
    feature_provenance = pd.read_csv(run_directory / "feature_provenance.csv")
    passive = feature_provenance[feature_provenance["party"] != "bank"]
    assert passive["semi_synthetic"].all()
    metrics = json.loads((run_directory / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["entity_alignment_seconds"] >= 0.0
    assert metrics["data_preparation_seconds"] >= 0.0
    assert metrics["training_steps"] > 0
    assert metrics["communication"]["traffic_type"] == "SIMULATED PAYLOAD SIZE"


def test_run_artifact_ids_do_not_collide_within_same_second(tmp_path: Path) -> None:
    first = RunArtifacts.create(root=tmp_path)
    second = RunArtifacts.create(root=tmp_path)
    assert first.run_id != second.run_id
    assert first.directory.exists()
    assert second.directory.exists()
