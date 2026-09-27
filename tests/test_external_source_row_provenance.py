from types import SimpleNamespace

import pandas as pd

from vertimosaic.datasets import external


def test_external_loaders_record_measured_raw_source_rows_and_checksums(monkeypatch) -> None:
    bank_features = pd.DataFrame({"LIMIT_BAL": [1, 2, 3], "AGE": [30, 31, 32]})
    bank_targets = pd.DataFrame({"target": [0, 1, 0]})
    telecom_features = pd.DataFrame({"Age": [20, 21, 22, 23]})
    retail_rows = pd.DataFrame(
        {
            "InvoiceNo": ["1", "2"],
            "StockCode": ["A", "B"],
            "Quantity": [1, 2],
            "InvoiceDate": ["2026-01-01", "2026-01-02"],
            "UnitPrice": [10.0, 20.0],
            "CustomerID": [100, 100],
        }
    )

    def fake_uci(*, id: int):
        if id == 350:
            return SimpleNamespace(
                data=SimpleNamespace(features=bank_features, targets=bank_targets, original=None)
            )
        if id == 563:
            return SimpleNamespace(
                data=SimpleNamespace(features=telecom_features, targets=None, original=None)
            )
        if id == 352:
            return SimpleNamespace(
                data=SimpleNamespace(features=retail_rows, targets=None, original=retail_rows)
            )
        raise AssertionError(id)

    frequency = pd.DataFrame({"IDpol": [1, 2, 3], "ClaimNb": [0, 1, 0]})
    severity = pd.DataFrame({"IDpol": [2, 2], "ClaimAmount": [100.0, 50.0]})

    def fake_openml(*, data_id: int, as_frame: bool, parser: str):
        assert as_frame is True
        assert parser == "auto"
        if data_id == 41214:
            return SimpleNamespace(frame=frequency, details={"licence": "CC0"})
        if data_id == 41215:
            return SimpleNamespace(frame=severity, details={"licence": "CC0"})
        raise AssertionError(data_id)

    monkeypatch.setattr(external, "fetch_ucirepo", fake_uci)
    monkeypatch.setattr(external, "fetch_openml", fake_openml)

    bank = external.fetch_bank()
    telecom = external.fetch_telecom()
    insurance = external.fetch_insurance(sample_size=2, seed=42)
    retail = external.fetch_retail()

    assert bank.metadata["source_raw_rows"] == {"bank": 3}
    assert telecom.metadata["source_raw_rows"] == {"telecom": 4}
    assert insurance.metadata["source_raw_rows"] == {
        "insurance_freq": 3,
        "insurance_sev": 2,
    }
    assert insurance.metadata["raw_rows"] == 3
    assert retail.metadata["source_raw_rows"] == {"retail": 2}
    assert retail.metadata["raw_rows"] == 2

    for bundle, keys in (
        (bank, ("bank",)),
        (telecom, ("telecom",)),
        (insurance, ("insurance_freq", "insurance_sev")),
        (retail, ("retail",)),
    ):
        checksums = bundle.metadata["source_checksums"]
        assert isinstance(checksums, dict)
        assert set(checksums) == set(keys)
        assert all(len(str(checksums[key])) == 64 for key in keys)
        assert bundle.metadata["source_checksum_algorithm"] == "sha256"

    assert insurance.metadata["source_checksums"]["insurance_freq"] != insurance.metadata[
        "source_checksums"
    ]["insurance_sev"]
