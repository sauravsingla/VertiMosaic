import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import vertimosaic.datasets.external as external_datasets
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
    metadata: dict[str, object] = {
        "license": "test-license",
        "processed_rows": len(frame),
        "retrieval_date": "2026-09-27",
    }
    if party == "insurance":
        metadata["source_licenses"] = {
            "insurance_freq": "test-license-frequency",
            "insurance_sev": "test-license-severity",
        }
        metadata["source_raw_rows"] = {
            "insurance_freq": len(frame),
            "insurance_sev": len(frame),
        }
    return ExternalDatasetBundle(
        party=party,
        features=frame,
        target=target,
        provenance=provenance,
        metadata=metadata,
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
    provenance_payload = json.loads(
        (run_directory / "dataset_provenance.json").read_text(encoding="utf-8")
    )
    insurance_sources = provenance_payload["sources"]["insurance"]["sources"]
    assert [source["dataset_id"] for source in insurance_sources] == ["41214", "41215"]
    assert [source["license"] for source in insurance_sources] == [
        "test-license-frequency",
        "test-license-severity",
    ]


def test_retail_fetch_applies_source_time_cutoff_before_customer_aggregation(monkeypatch) -> None:
    transactions = pd.DataFrame(
        {
            "InvoiceNo": ["1", "2", "3", "4"],
            "StockCode": ["A", "B", "C", "D"],
            "Quantity": [1, 2, 1, 100],
            "InvoiceDate": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-02-01"]
            ),
            "UnitPrice": [10.0, 5.0, 3.0, 999.0],
            "CustomerID": [101, 101, 202, 101],
            "Description": ["a", "b", "c", "future"],
            "Country": ["UK", "UK", "UK", "UK"],
        }
    )
    fake = SimpleNamespace(data=SimpleNamespace(original=transactions, features=transactions))
    monkeypatch.setattr(external_datasets, "fetch_ucirepo", lambda id: fake)

    bundle = external_datasets.fetch_retail(feature_cutoff="2020-01-03")

    assert bundle.metadata["future_rows_excluded"] == 1
    assert str(bundle.metadata["feature_cutoff"]).startswith("2020-01-03")
    assert bundle.metadata["temporal_leakage_control"] == (
        "aggregate transactions at or before cutoff before linkage"
    )
    assert float(bundle.features["total_spend"].max()) == 20.0
    assert all("at or before 2020-01-03" in item.transformation for item in bundle.provenance)


def test_run_artifact_ids_do_not_collide_within_same_second(tmp_path: Path) -> None:
    first = RunArtifacts.create(root=tmp_path)
    second = RunArtifacts.create(root=tmp_path)
    assert first.run_id != second.run_id
    assert first.directory.exists()
    assert second.directory.exists()
