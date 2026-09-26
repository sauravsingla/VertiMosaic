from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from ucimlrepo import fetch_ucirepo

from vertimosaic.provenance import FeatureProvenance


@dataclass(frozen=True)
class ExternalDatasetBundle:
    party: str
    features: pd.DataFrame
    target: pd.Series | None
    provenance: list[FeatureProvenance]
    metadata: dict[str, Any]


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return pd.DataFrame(index=frame.index)
    return frame[available].apply(pd.to_numeric, errors="coerce")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return (numerator / denom).replace([np.inf, -np.inf], np.nan)


def _record(
    party: str,
    feature: str,
    dataset: str,
    source: str,
    transformation: str,
    *,
    source_type: str = "real_external_derived",
    observed: bool = False,
) -> FeatureProvenance:
    return FeatureProvenance(
        party=party,
        feature=feature,
        external_dataset=dataset,
        source_column=source,
        transformation=transformation,
        source_type=source_type,
        observed_or_derived="observed" if observed else "derived",
        semi_synthetic=False,
    )


def prepare_bank_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[FeatureProvenance]]:
    """Build documented Bank behaviour features from the UCI source columns."""
    out = pd.DataFrame(index=frame.index)
    provenance: list[FeatureProvenance] = []
    bills = _numeric(frame, [f"BILL_AMT{i}" for i in range(1, 7)])
    payments = _numeric(frame, [f"PAY_AMT{i}" for i in range(1, 7)])
    delays = _numeric(frame, ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"])

    def add(name: str, values: pd.Series, source: str, transformation: str) -> None:
        out[name] = values
        provenance.append(
            _record("bank", name, "UCI Default of Credit Card Clients", source, transformation)
        )

    if not bills.empty:
        add("mean_bill_amount", bills.mean(axis=1), "BILL_AMT1..6", "row mean")
        add("median_bill_amount", bills.median(axis=1), "BILL_AMT1..6", "row median")
        add("max_bill_amount", bills.max(axis=1), "BILL_AMT1..6", "row max")
        add("bill_std", bills.std(axis=1).fillna(0.0), "BILL_AMT1..6", "row std")
        if bills.shape[1] >= 2:
            add(
                "bill_trend",
                bills.iloc[:, 0] - bills.iloc[:, -1],
                "BILL_AMT1..6",
                "recent minus oldest",
            )
            add(
                "recent_bill_change",
                bills.iloc[:, 0] - bills.iloc[:, 1],
                "BILL_AMT1..2",
                "difference",
            )
    if not payments.empty:
        add("mean_payment_amount", payments.mean(axis=1), "PAY_AMT1..6", "row mean")
        add("max_payment_amount", payments.max(axis=1), "PAY_AMT1..6", "row max")
        add("payment_std", payments.std(axis=1).fillna(0.0), "PAY_AMT1..6", "row std")
        add(
            "payment_volatility",
            payments.std(axis=1).fillna(0.0),
            "PAY_AMT1..6",
            "row std",
        )
    if not bills.empty and not payments.empty:
        add(
            "payment_to_bill_ratio",
            _safe_ratio(payments.mean(axis=1), bills.abs().mean(axis=1)),
            "PAY_AMT1..6,BILL_AMT1..6",
            "mean payment / mean absolute bill",
        )
        add(
            "recent_payment_ratio",
            _safe_ratio(payments.iloc[:, 0], bills.iloc[:, 0].abs()),
            "PAY_AMT1,BILL_AMT1",
            "recent payment / recent absolute bill",
        )
    if not delays.empty:
        add("repayment_delay_mean", delays.mean(axis=1), "PAY_0,PAY_2..6", "row mean")
        add("repayment_delay_max", delays.max(axis=1), "PAY_0,PAY_2..6", "row max")
        add(
            "repayment_delay_count",
            (delays > 0).sum(axis=1),
            "PAY_0,PAY_2..6",
            "positive count",
        )
    if "LIMIT_BAL" in frame.columns and not bills.empty:
        limit_balance = pd.to_numeric(frame["LIMIT_BAL"], errors="coerce")
        add(
            "credit_utilization_proxy",
            _safe_ratio(bills.iloc[:, 0], limit_balance),
            "BILL_AMT1,LIMIT_BAL",
            "recent bill / credit limit",
        )
    for observed in ("LIMIT_BAL", "AGE"):
        if observed in frame.columns:
            name = observed.lower()
            out[name] = pd.to_numeric(frame[observed], errors="coerce")
            provenance.append(
                _record(
                    "bank",
                    name,
                    "UCI Default of Credit Card Clients",
                    observed,
                    "numeric coercion",
                    source_type="real_external",
                    observed=True,
                )
            )
    return out, provenance


def _resolve_columns(frame: pd.DataFrame) -> dict[str, str]:
    return {
        str(column).strip().casefold().replace(" ", ""): str(column)
        for column in frame.columns
    }


def prepare_telecom_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[FeatureProvenance]]:
    """Build only features supported by the published Iranian Churn columns."""
    source_map = {
        "call_failure": "Call Failure",
        "complains": "Complains",
        "subscription_length": "Subscription Length",
        "charge_amount": "Charge Amount",
        "seconds_of_use": "Seconds of Use",
        "frequency_of_use": "Frequency of use",
        "frequency_of_sms": "Frequency of SMS",
        "distinct_called_numbers": "Distinct Called Numbers",
        "age": "Age",
        "customer_value": "Customer Value",
    }
    normalized = _resolve_columns(frame)
    values: dict[str, pd.Series] = {}
    for alias, expected in source_map.items():
        key = expected.casefold().replace(" ", "")
        if key in normalized:
            values[alias] = pd.to_numeric(frame[normalized[key]], errors="coerce")
    out = pd.DataFrame(index=frame.index)
    provenance: list[FeatureProvenance] = []

    def add(name: str, value: pd.Series, columns: str, transformation: str) -> None:
        out[name] = value
        provenance.append(_record("telecom", name, "UCI Iranian Churn", columns, transformation))

    usage = values.get("frequency_of_use")
    seconds = values.get("seconds_of_use")
    subscription = values.get("subscription_length")
    sms = values.get("frequency_of_sms")
    failures = values.get("call_failure")
    distinct = values.get("distinct_called_numbers")
    if usage is not None:
        add("usage_intensity", usage, "Frequency of use", "numeric usage intensity")
    if sms is not None:
        add("sms_intensity", sms, "Frequency of SMS", "numeric SMS intensity")
    if failures is not None and usage is not None:
        add(
            "call_failure_intensity",
            _safe_ratio(failures, usage + 1.0),
            "Call Failure,Frequency of use",
            "failure/(usage+1)",
        )
    if distinct is not None and usage is not None:
        add(
            "distinct_contact_intensity",
            _safe_ratio(distinct, usage + 1.0),
            "Distinct Called Numbers,Frequency of use",
            "contacts/(usage+1)",
        )
    if subscription is not None:
        add("service_tenure", subscription, "Subscription Length", "numeric tenure")
    if seconds is not None and subscription is not None:
        add(
            "usage_per_subscription_month",
            _safe_ratio(seconds, subscription + 1.0),
            "Seconds of Use,Subscription Length",
            "seconds/(tenure+1)",
        )
    if sms is not None and usage is not None:
        add(
            "sms_to_call_ratio",
            _safe_ratio(sms, usage + 1.0),
            "Frequency of SMS,Frequency of use",
            "sms/(usage+1)",
        )
    complains = values.get("complains")
    if complains is not None:
        add("complaint_indicator", (complains > 0).astype(float), "Complains", "indicator")
    charge = values.get("charge_amount")
    if charge is not None:
        add("charge_level", charge, "Charge Amount", "numeric charge")
    customer_value = values.get("customer_value")
    if customer_value is not None:
        scale = customer_value.std()
        scale = scale if np.isfinite(scale) and scale > 0 else 1.0
        add(
            "customer_value_normalized",
            (customer_value - customer_value.mean()) / scale,
            "Customer Value",
            "z-score within source dataset",
        )
    if "age" in values:
        out["age"] = values["age"]
        provenance.append(
            _record(
                "telecom",
                "age",
                "UCI Iranian Churn",
                "Age",
                "numeric coercion",
                source_type="real_external",
                observed=True,
            )
        )
    return out, provenance


def prepare_insurance_frames(
    frequency: pd.DataFrame, severity: pd.DataFrame
) -> tuple[pd.DataFrame, list[FeatureProvenance]]:
    """Merge OpenML frequency/severity sources by policy identifier and aggregate claims."""
    sev = severity.copy()
    if "IDpol" not in frequency.columns or "IDpol" not in sev.columns:
        raise ValueError("insurance source requires IDpol")
    if "ClaimAmount" not in sev.columns:
        raise ValueError("severity source requires ClaimAmount")
    sev["ClaimAmount"] = pd.to_numeric(sev["ClaimAmount"], errors="coerce")
    grouped = sev.groupby("IDpol", as_index=False)["ClaimAmount"].agg(
        total_claim_amount="sum",
        mean_claim_amount="mean",
        max_claim_amount="max",
    )
    merged = frequency.merge(grouped, on="IDpol", how="left")
    claim_columns = ["total_claim_amount", "mean_claim_amount", "max_claim_amount"]
    merged[claim_columns] = merged[claim_columns].fillna(0.0)
    out = pd.DataFrame(index=merged.index)
    provenance: list[FeatureProvenance] = []
    numeric_map = {
        "claim_count": "ClaimNb",
        "exposure": "Exposure",
        "bonus_malus": "BonusMalus",
        "vehicle_age": "VehAge",
        "driver_age": "DrivAge",
        "vehicle_power": "VehPower",
        "density": "Density",
    }
    for name, source in numeric_map.items():
        if source in merged.columns:
            out[name] = pd.to_numeric(merged[source], errors="coerce")
            provenance.append(
                _record(
                    "insurance",
                    name,
                    "OpenML freMTPL2freq/freMTPL2sev",
                    source,
                    "numeric coercion",
                )
            )
    if "claim_count" in out:
        out["has_claim"] = (out["claim_count"] > 0).astype(float)
        exposure = out.get("exposure", pd.Series(1.0, index=out.index))
        out["claim_frequency"] = _safe_ratio(out["claim_count"], exposure)
    for source in claim_columns:
        out[source] = pd.to_numeric(merged[source], errors="coerce")
        provenance.append(
            _record(
                "insurance",
                source,
                "OpenML freMTPL2freq/freMTPL2sev",
                "ClaimAmount,IDpol",
                f"claim-level aggregation: {source}",
            )
        )
    for source, name in (
        ("Area", "area"),
        ("Region", "region"),
        ("VehBrand", "vehicle_brand"),
        ("VehGas", "vehicle_fuel"),
    ):
        if source in merged.columns:
            out[name] = merged[source].astype("string")
            provenance.append(
                _record(
                    "insurance",
                    name,
                    "OpenML freMTPL2freq/freMTPL2sev",
                    source,
                    "string category",
                    source_type="real_external",
                    observed=True,
                )
            )
    return out, provenance


def prepare_retail_transactions(
    transactions: pd.DataFrame, *, cutoff: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, list[FeatureProvenance]]:
    """Aggregate transaction rows to one customer row using a temporal feature cutoff."""
    frame = transactions.copy()
    required = {"InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"retail source missing columns: {sorted(missing)}")
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"], errors="coerce")
    frame = frame.dropna(subset=["CustomerID", "InvoiceDate"])
    if cutoff is None:
        cutoff = frame["InvoiceDate"].max()
    frame = frame.loc[frame["InvoiceDate"] <= cutoff].copy()
    frame["Quantity"] = pd.to_numeric(frame["Quantity"], errors="coerce")
    frame["UnitPrice"] = pd.to_numeric(frame["UnitPrice"], errors="coerce")
    frame["line_value"] = frame["Quantity"] * frame["UnitPrice"]
    frame["cancelled"] = (
        frame["InvoiceNo"].astype(str).str.startswith("C") | (frame["Quantity"] < 0)
    )
    grouped = frame.groupby("CustomerID", sort=True)
    output = pd.DataFrame(
        {
            "purchase_count": grouped.size(),
            "invoice_count": grouped["InvoiceNo"].nunique(),
            "total_spend": grouped["line_value"].sum(),
            "average_order_value": grouped["line_value"].mean(),
            "median_order_value": grouped["line_value"].median(),
            "max_order_value": grouped["line_value"].max(),
            "spend_std": grouped["line_value"].std().fillna(0.0),
            "quantity_sum": grouped["Quantity"].sum(),
            "mean_quantity": grouped["Quantity"].mean(),
            "unique_products": grouped["StockCode"].nunique(),
            "active_days": grouped["InvoiceDate"].apply(lambda x: x.dt.normalize().nunique()),
            "cancellation_count": grouped["cancelled"].sum(),
        }
    )
    output["product_diversity"] = _safe_ratio(output["unique_products"], output["purchase_count"])
    output["purchase_frequency"] = _safe_ratio(output["purchase_count"], output["active_days"])
    last_purchase = grouped["InvoiceDate"].max()
    output["recency_days"] = (cutoff - last_purchase).dt.total_seconds() / 86400.0
    output["cancellation_ratio"] = _safe_ratio(
        output["cancellation_count"], output["purchase_count"]
    )
    threshold = frame["line_value"].quantile(0.9)
    output["high_value_purchase_ratio"] = (
        frame.assign(high_value=frame["line_value"] >= threshold)
        .groupby("CustomerID")["high_value"]
        .mean()
    )
    output["recency"] = output["recency_days"]
    output["frequency"] = output["invoice_count"]
    output["monetary"] = output["total_spend"]
    output = output.reset_index(drop=True)
    provenance = [
        _record(
            "retail",
            column,
            "UCI Online Retail",
            "transaction-level source columns",
            f"customer aggregation at or before {cutoff.isoformat()}",
            source_type="real_external_aggregated",
        )
        for column in output.columns
    ]
    return output, provenance


def fetch_bank() -> ExternalDatasetBundle:
    data = fetch_ucirepo(id=350)
    features = data.data.features.copy()
    targets = data.data.targets
    target = pd.to_numeric(targets.iloc[:, 0], errors="coerce") if targets is not None else None
    prepared, provenance = prepare_bank_frame(features)
    return ExternalDatasetBundle(
        "bank",
        prepared,
        target,
        provenance,
        {"dataset_id": 350, "provider": "UCI", "retrieval_date": date.today().isoformat()},
    )


def fetch_telecom() -> ExternalDatasetBundle:
    data = fetch_ucirepo(id=563)
    prepared, provenance = prepare_telecom_frame(data.data.features.copy())
    return ExternalDatasetBundle(
        "telecom",
        prepared,
        None,
        provenance,
        {"dataset_id": 563, "provider": "UCI", "retrieval_date": date.today().isoformat()},
    )


def fetch_insurance(sample_size: int | None = None, seed: int = 42) -> ExternalDatasetBundle:
    freq = fetch_openml(data_id=41214, as_frame=True, parser="auto")
    sev = fetch_openml(data_id=41215, as_frame=True, parser="auto")
    frequency = freq.frame.copy()
    severity = sev.frame.copy()
    if sample_size is not None and sample_size < len(frequency):
        frequency = frequency.sample(n=sample_size, random_state=seed)
        severity = severity[severity["IDpol"].isin(frequency["IDpol"])].copy()
    prepared, provenance = prepare_insurance_frames(frequency, severity)
    details = getattr(freq, "details", {}) or {}
    return ExternalDatasetBundle(
        "insurance",
        prepared,
        None,
        provenance,
        {
            "dataset_id": "41214+41215",
            "provider": "OpenML",
            "retrieval_date": date.today().isoformat(),
            "license": details.get("licence") or details.get("license"),
        },
    )


def fetch_retail() -> ExternalDatasetBundle:
    data = fetch_ucirepo(id=352)
    raw = getattr(data.data, "original", None)
    if raw is None:
        raw = data.data.features
    prepared, provenance = prepare_retail_transactions(raw.copy())
    return ExternalDatasetBundle(
        "retail",
        prepared,
        None,
        provenance,
        {"dataset_id": 352, "provider": "UCI", "retrieval_date": date.today().isoformat()},
    )


def fetch_external_party(
    name: str, *, insurance_sample_size: int | None = None, seed: int = 42
) -> ExternalDatasetBundle:
    if name == "bank":
        return fetch_bank()
    if name == "telecom":
        return fetch_telecom()
    if name == "insurance":
        return fetch_insurance(sample_size=insurance_sample_size, seed=seed)
    if name == "retail":
        return fetch_retail()
    raise ValueError(f"unknown external party: {name}")


def save_bundle(bundle: ExternalDatasetBundle, directory: Path) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    features_path = directory / f"{bundle.party}_features.parquet"
    provenance_path = directory / f"{bundle.party}_feature_provenance.csv"
    metadata_path = directory / f"{bundle.party}_metadata.json"
    bundle.features.to_parquet(features_path, index=False)
    pd.DataFrame([asdict(item) for item in bundle.provenance]).to_csv(
        provenance_path, index=False
    )
    metadata_path.write_text(
        json.dumps(bundle.metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    outputs = {
        "features": str(features_path),
        "feature_provenance": str(provenance_path),
        "metadata": str(metadata_path),
    }
    if bundle.target is not None:
        target_path = directory / f"{bundle.party}_target.parquet"
        pd.DataFrame({"target": bundle.target}).to_parquet(target_path, index=False)
        outputs["target"] = str(target_path)
    return outputs
