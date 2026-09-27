from types import SimpleNamespace

import numpy as np
import pandas as pd

import vertimosaic.datasets.provider as provider


def _uci_350_fixture(rows: int = 8) -> pd.DataFrame:
    values = np.arange(rows * 23, dtype=float).reshape(rows, 23) + 1.0
    return pd.DataFrame(values, columns=[f"X{i}" for i in range(1, 24)])


def test_normalize_bank_provider_columns_supports_current_uci_schema() -> None:
    source = _uci_350_fixture()
    normalized = provider.normalize_bank_provider_columns(source)
    assert "LIMIT_BAL" in normalized
    assert "AGE" in normalized
    assert "PAY_0" in normalized
    assert "BILL_AMT1" in normalized
    assert "BILL_AMT6" in normalized
    assert "PAY_AMT1" in normalized
    assert "PAY_AMT6" in normalized
    assert "X1" not in normalized
    assert "X23" not in normalized


def test_provider_bank_fetch_builds_nonempty_features_from_x_schema(monkeypatch) -> None:
    features = _uci_350_fixture()
    targets = pd.DataFrame({"Y": [0, 1, 0, 1, 0, 1, 0, 1]})
    original = pd.concat([features, targets], axis=1)
    fake = SimpleNamespace(
        data=SimpleNamespace(features=features, targets=targets, original=original)
    )
    monkeypatch.setattr(provider, "fetch_ucirepo", lambda id: fake)

    bundle = provider.fetch_bank()

    assert bundle.features.shape[0] == len(features)
    assert bundle.features.shape[1] > 0
    assert {
        "limit_bal",
        "age",
        "mean_bill_amount",
        "mean_payment_amount",
        "repayment_delay_mean",
    } <= set(bundle.features.columns)
    assert bundle.target is not None
    assert bundle.target.tolist() == targets["Y"].tolist()
    aliases = bundle.metadata["provider_column_aliases_applied"]
    assert aliases["X1"] == "LIMIT_BAL"
    assert aliases["X23"] == "PAY_AMT6"
    assert len(bundle.metadata["source_checksums"]["bank"]) == 64


def test_normalize_bank_provider_columns_preserves_descriptive_schema() -> None:
    frame = pd.DataFrame({"LIMIT_BAL": [1000.0], "AGE": [40.0], "BILL_AMT1": [100.0]})
    normalized = provider.normalize_bank_provider_columns(frame)
    assert normalized.equals(frame)
