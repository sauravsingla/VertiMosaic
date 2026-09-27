from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from ucimlrepo import fetch_ucirepo

from vertimosaic.provenance import FeatureProvenance

_SOURCE_HASH_ALGORITHM = "sha256"
_SOURCE_HASH_SCOPE = "retrieved_dataframe_content_before_transformation"
_INSURANCE_SOURCE_HASH_SCOPE = "retrieved_dataframe_before_sampling_or_transformation"
_RETAIL_SOURCE_HASH_SCOPE = "retrieved_dataframe_before_cutoff_or_transformation"


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


def _frame_sha256(frame: pd.DataFrame) -> str:
    """Hash one retrieved source-frame capture before transformation or sampling."""
    digest = sha256()
    schema = [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]
    digest.update(json.dumps(schema, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    hashed = pd.util.hash_pandas_object(frame, index=True, categorize=True).to_numpy(
        dtype=np.uint64
    )
    digest.update(hashed.tobytes())
    return digest.hexdigest()


def _uci_source_capture(
    features: pd.DataFrame,
    targets: pd.DataFrame | None,
    original: pd.DataFrame | None,
) -> pd.DataFrame:
    if original is not None:
        return original.copy()
    frames = [features.reset_index(drop=True)]
    if targets is not None:
        frames.append(targets.reset_index(drop=True))
    return pd.concat(frames, axis=1)


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
        str(column).strip().casefold().replace(" ", ""): str(column) for column in frame.columns
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
    if "complains" in values:
        add("complaint_indicator", values["complains"], "Complains", "numeric indicator")
    if "charge_amount" in values:
        add("charge_level", values["charge_amount"], "Charge Amount", "numeric charge level")
    if "customer_value" in values:
        value = values["customer_value"]
        std = float(value.std())
        normalized_value = (value - float(value.mean())) / (std if std > 1e-12 else 1.0)
        add(
            "customer_value_normalized",
            normalized_value,
            "Customer Value",
            "z-score within source domain",
        )
    for observed in ("age",):
        if observed in values:
            out[observed] = values[observed]
            provenance.append(
                _record(
                    "telecom",
                    observed,
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
    """Aggregate severity by policy before joining to the frequency table."""
    frequency = frequency.copy()
    severity = severity.copy()
    if "IDpol" not in frequency or "IDpol" not in severity:
        raise ValueError("insurance frames must contain IDpol")
    severity["ClaimAmount"] = pd.to_numeric(severity["ClaimAmount"], errors="coerce")
    severity_agg = severity.groupby("IDpol", as_index=False)["ClaimAmount"].agg(
        total_claim_amount="sum",
        mean_claim_amount="mean",
        max_claim_amount="max",
    )
    merged = frequency.merge(severity_agg, on="IDpol", how="left")
    out = pd.DataFrame(index=merged.index)
    provenance: list[FeatureProvenance] = []
    numeric_map = {
        "ClaimNb": "claim_count",
        "Exposure": "exposure",
        "BonusMalus": "bonus_malus",
        "VehAge": "vehicle_age",
        "DrivAge": "driver_age",
        "VehPower": "vehicle_power",
        "Density": "density",
    }
    for source, name in numeric_map.items():
        if source in merged.columns:
            out[name] = pd.to_numeric(merged[source], errors="coerce")
            provenance.append(
                _record(
                    "insurance",
                    name,
                    "OpenML freMTPL2freq/freMTPL2sev",
                    source,
                    "numeric coercion",
                    source_type="real_external",
                    observed=True,
                )
            )
    if "claim_count" in out:
        out["has_claim"] = (out["claim_count"] > 0).astype(float)
        provenance.append(
            _record(
                "insurance",
                "has_claim",
                "OpenML freMTPL2freq/freMTPL2sev",
                "ClaimNb",
                "ClaimNb > 0",
            )
        )
        if "exposure" in out:
            out["claim_frequency"] = _safe_ratio(out["claim_count"], out["exposure"])
            provenance.append(
                _record(
                    "insurance",
                    "claim_frequency",
                    "OpenML freMTPL2freq/freMTPL2sev",
                    "ClaimNb,Exposure",
                    "ClaimNb / Exposure",
                )
            )
    claim_columns = ["total_claim_amount", "mean_claim_amount", "max_claim_amount"]
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
    frame["cancelled"] = frame["InvoiceNo"].astype(str).str.startswith("C") | (
        frame["Quantity"] < 0
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
    target_frame = targets.copy() if targets is not None else None
    original = getattr(data.data, "original", None)
    source_frame = _uci_source_capture(features, target_frame, original)
    target = pd.to_numeric(targets.iloc[:, 0], errors="coerce") if targets is not None else None
    prepared, provenance = prepare_bank_frame(features)
    raw_rows = len(source_frame)
    return ExternalDatasetBundle(
        "bank",
        prepared,
        target,
        provenance,
        {
            "dataset_id": 350,
            "provider": "UCI",
            "retrieval_date": date.today().isoformat(),
            "raw_rows": raw_rows,
            "source_raw_rows": {"bank": raw_rows},
            "source_checksums": {"bank": _frame_sha256(source_frame)},
            "source_checksum_algorithm": _SOURCE_HASH_ALGORITHM,
            "source_checksum_scope": _SOURCE_HASH_SCOPE,
        },
    )


def fetch_telecom() -> ExternalDatasetBundle:
    data = fetch_ucirepo(id=563)
    features = data.data.features.copy()
    targets = data.data.targets
    target_frame = targets.copy() if targets is not None else None
    original = getattr(data.data, "original", None)
    source_frame = _uci_source_capture(features, target_frame, original)
    prepared, provenance = prepare_telecom_frame(features)
    raw_rows = len(source_frame)
    return ExternalDatasetBundle(
        "telecom",
        prepared,
        None,
        provenance,
        {
            "dataset_id": 563,
            "provider": "UCI",
            "retrieval_date": date.today().isoformat(),
            "raw_rows": raw_rows,
            "source_raw_rows": {"telecom": raw_rows},
            "source_checksums": {"telecom": _frame_sha256(source_frame)},
            "source_checksum_algorithm": _SOURCE_HASH_ALGORITHM,
            "source_checksum_scope": _SOURCE_HASH_SCOPE,
        },
    )


def fetch_insurance(sample_size: int | None = None, seed: int = 42) -> ExternalDatasetBundle:
    freq = fetch_openml(data_id=41214, as_frame=True, parser="auto")
    sev = fetch_openml(data_id=41215, as_frame=True, parser="auto")
    frequency = freq.frame.copy()
    severity = sev.frame.copy()
    source_raw_rows = {
        "insurance_freq": len(frequency),
        "insurance_sev": len(severity),
    }
    source_checksums = {
        "insurance_freq": _frame_sha256(frequency),
        "insurance_sev": _frame_sha256(severity),
    }
    frequency_details = getattr(freq, "details", {}) or {}
    severity_details = getattr(sev, "details", {}) or {}
    source_licenses = {
        "insurance_freq": frequency_details.get("licence") or frequency_details.get("license"),
        "insurance_sev": severity_details.get("licence") or severity_details.get("license"),
    }
    if sample_size is not None and sample_size < len(frequency):
        frequency = frequency.sample(n=sample_size, random_state=seed)
        severity = severity[severity["IDpol"].isin(frequency["IDpol"])].copy()
    prepared, provenance = prepare_insurance_frames(frequency, severity)
    return ExternalDatasetBundle(
        "insurance",
        prepared,
        None,
        provenance,
        {
            "dataset_id": "41214+41215",
            "provider": "OpenML",
            "retrieval_date": date.today().isoformat(),
            "license": source_licenses["insurance_freq"],
            "source_licenses": source_licenses,
            "raw_rows": source_raw_rows["insurance_freq"],
            "source_raw_rows": source_raw_rows,
            "source_checksums": source_checksums,
            "source_checksum_algorithm": _SOURCE_HASH_ALGORITHM,
            "source_checksum_scope": _INSURANCE_SOURCE_HASH_SCOPE,
        },
    )


def _resolve_retail_cutoff(
    transactions: pd.DataFrame,
    *,
    feature_cutoff: pd.Timestamp | str | None,
    cutoff_quantile: float,
) -> tuple[pd.Timestamp, str]:
    if not 0.0 < cutoff_quantile < 1.0:
        raise ValueError("retail cutoff_quantile must be in (0, 1)")
    timestamps = pd.to_datetime(transactions["InvoiceDate"], errors="coerce").dropna()
    if timestamps.empty:
        raise ValueError("retail source contains no valid InvoiceDate values")
    if feature_cutoff is not None:
        cutoff = pd.Timestamp(feature_cutoff)
        if pd.isna(cutoff):
            raise ValueError("retail feature_cutoff must be a valid timestamp")
        return cutoff, "explicit source-time feature cutoff"
    cutoff = pd.Timestamp(timestamps.quantile(cutoff_quantile))
    return cutoff, f"source-time InvoiceDate quantile {cutoff_quantile:.2f}"


def fetch_retail(
    *,
    feature_cutoff: pd.Timestamp | str | None = None,
    cutoff_quantile: float = 0.70,
) -> ExternalDatasetBundle:
    """Fetch Retail and build a leakage-safe customer snapshot before any cross-domain linkage."""
    data = fetch_ucirepo(id=352)
    raw = getattr(data.data, "original", None)
    if raw is None:
        raw = data.data.features
    raw_frame = raw.copy()
    source_checksum = _frame_sha256(raw_frame)
    cutoff, cutoff_policy = _resolve_retail_cutoff(
        raw_frame,
        feature_cutoff=feature_cutoff,
        cutoff_quantile=cutoff_quantile,
    )
    timestamps = pd.to_datetime(raw_frame["InvoiceDate"], errors="coerce")
    future_rows_excluded = int((timestamps > cutoff).sum())
    prepared, provenance = prepare_retail_transactions(raw_frame, cutoff=cutoff)
    raw_rows = len(raw_frame)
    return ExternalDatasetBundle(
        "retail",
        prepared,
        None,
        provenance,
        {
            "dataset_id": 352,
            "provider": "UCI",
            "retrieval_date": date.today().isoformat(),
            "raw_rows": raw_rows,
            "source_raw_rows": {"retail": raw_rows},
            "source_checksums": {"retail": source_checksum},
            "source_checksum_algorithm": _SOURCE_HASH_ALGORITHM,
            "source_checksum_scope": _RETAIL_SOURCE_HASH_SCOPE,
            "feature_cutoff": cutoff.isoformat(),
            "feature_cutoff_policy": cutoff_policy,
            "future_rows_excluded": future_rows_excluded,
            "temporal_leakage_control": "aggregate transactions at or before cutoff before linkage",
        },
    )


def fetch_external_party(
    name: str,
    *,
    insurance_sample_size: int | None = None,
    seed: int = 42,
    retail_feature_cutoff: pd.Timestamp | str | None = None,
    retail_cutoff_quantile: float = 0.70,
) -> ExternalDatasetBundle:
    if name == "bank":
        return fetch_bank()
    if name == "telecom":
        return fetch_telecom()
    if name == "insurance":
        return fetch_insurance(sample_size=insurance_sample_size, seed=seed)
    if name == "retail":
        return fetch_retail(
            feature_cutoff=retail_feature_cutoff,
            cutoff_quantile=retail_cutoff_quantile,
        )
    raise ValueError(f"unknown external party: {name}")


def save_bundle(bundle: ExternalDatasetBundle, directory: Path) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    features_path = directory / f"{bundle.party}_features.parquet"
    provenance_path = directory / f"{bundle.party}_feature_provenance.csv"
    metadata_path = directory / f"{bundle.party}_metadata.json"
    bundle.features.to_parquet(features_path, index=False)
    pd.DataFrame([asdict(item) for item in bundle.provenance]).to_csv(provenance_path, index=False)
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
