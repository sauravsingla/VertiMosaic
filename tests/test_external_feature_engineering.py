import pandas as pd

from vertimosaic.datasets.external import (
    prepare_bank_frame,
    prepare_retail_transactions,
    prepare_telecom_frame,
)


def test_bank_features_use_source_columns_only() -> None:
    frame = pd.DataFrame(
        {
            "BILL_AMT1": [100.0, 200.0],
            "BILL_AMT2": [80.0, 150.0],
            "PAY_AMT1": [20.0, 40.0],
            "PAY_AMT2": [10.0, 30.0],
            "PAY_0": [0, 2],
            "PAY_2": [0, 1],
            "LIMIT_BAL": [1000.0, 2000.0],
            "AGE": [30, 40],
        }
    )
    features, provenance = prepare_bank_frame(frame)
    assert "mean_bill_amount" in features
    assert "credit_utilization_proxy" in features
    assert all(
        item.source_type in {"real_external", "real_external_derived"} for item in provenance
    )


def test_retail_cutoff_prevents_future_data_leakage() -> None:
    frame = pd.DataFrame(
        {
            "InvoiceNo": ["1", "2"],
            "StockCode": ["a", "b"],
            "Quantity": [1, 10],
            "InvoiceDate": ["2020-01-01", "2021-01-01"],
            "UnitPrice": [10.0, 100.0],
            "CustomerID": [1, 1],
        }
    )
    features, _ = prepare_retail_transactions(frame, cutoff=pd.Timestamp("2020-06-01"))
    assert features.loc[0, "total_spend"] == 10.0
    assert features.loc[0, "purchase_count"] == 1


def test_telecom_does_not_invent_sim_swap_fields() -> None:
    frame = pd.DataFrame(
        {
            "Frequency of use": [10, 20],
            "Frequency of SMS": [5, 7],
            "Call Failure": [1, 2],
            "Subscription Length": [12, 24],
            "Customer Value": [50.0, 70.0],
        }
    )
    features, _ = prepare_telecom_frame(frame)
    assert not any("sim" in column.casefold() for column in features.columns)
