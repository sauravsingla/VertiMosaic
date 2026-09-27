import json
from pathlib import Path

import numpy as np
import pandas as pd

from vertimosaic.experiments import run_ieee_cis_experiment


def test_ieee_cis_linked_runs_two_party_vfl_without_exporting_raw_ids(tmp_path: Path) -> None:
    rows = 240
    transaction_ids = np.arange(100_000, 100_000 + rows)
    amount = np.linspace(1.0, 500.0, rows)
    fraud = ((np.arange(rows) % 5) == 0).astype(int)
    transaction = pd.DataFrame(
        {
            "TransactionID": transaction_ids,
            "isFraud": fraud,
            "TransactionAmt": amount,
            "card4": np.where(np.arange(rows) % 2 == 0, "visa", "mastercard"),
        }
    )
    identity = pd.DataFrame(
        {
            "TransactionID": transaction_ids,
            "id_01": np.sin(np.arange(rows) / 11.0),
            "DeviceType": np.where(np.arange(rows) % 3 == 0, "mobile", "desktop"),
        }
    )
    transaction_path = tmp_path / "train_transaction.csv"
    identity_path = tmp_path / "train_identity.csv"
    transaction.to_csv(transaction_path, index=False)
    identity.to_csv(identity_path, index=False)

    payload = run_ieee_cis_experiment(
        transaction_path,
        identity_path,
        model_name="logistic",
        seed=7,
        bootstrap_replicates=20,
        artifact_directory=tmp_path / "artifacts",
        runs_root=tmp_path / "runs",
        output=tmp_path / "ieee_cis_linked.json",
    )

    assert payload["mode"] == "ieee_cis_linked"
    assert payload["parties"] == ["transaction", "identity"]
    assert payload["four_industry_benchmark"] is False
    assert payload["source_data_redistributed"] is False
    assert payload["overlap_rows"] == rows
    assert 0.0 <= payload["metrics"]["roc_auc"] <= 1.0
    assert payload["communication"]["traffic_type"] == "SIMULATED PAYLOAD SIZE"

    run_directory = Path(payload["run_directory"])
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
    assert "TransactionID" not in predictions.columns
    assert predictions["entity_id"].str.len().eq(64).all()
    raw_identifiers = {str(value) for value in transaction_ids}
    assert raw_identifiers.isdisjoint(set(predictions["entity_id"]))

    provenance = json.loads((run_directory / "dataset_provenance.json").read_text(encoding="utf-8"))
    assert provenance["authorization_required"] is True
    assert provenance["source_data_redistributed"] is False
    assert len(provenance["sources"]) == 2
    assert all(len(source["checksum"]) == 64 for source in provenance["sources"])
    assert all(source["redistributed_by_vertimosaic"] is False for source in provenance["sources"])

    linkage = json.loads((run_directory / "linkage_manifest.json").read_text(encoding="utf-8"))
    assert linkage["real_linkage"] is True
    assert linkage["method"].startswith("exact TransactionID intersection")
    assert linkage["raw_source_ids_exported"] is False

    assert (tmp_path / "artifacts" / "ieee_cis_transaction_preprocessor.joblib").exists()
    assert (tmp_path / "artifacts" / "ieee_cis_identity_preprocessor.joblib").exists()
