"""External-domain feature engineering with provenance-friendly transformations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_bank(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    f = frame.copy()
    target_candidates = [c for c in f.columns if "default" in str(c).lower()]
    if not target_candidates:
        raise ValueError("could not locate bank default target")
    target_col = target_candidates[-1]
    y = pd.to_numeric(f.pop(target_col), errors="coerce").fillna(0).astype(int)
    bill = [c for c in f.columns if str(c).upper().startswith("BILL_AMT")]
    pay_amt = [c for c in f.columns if str(c).upper().startswith("PAY_AMT")]
    pay_status = [c for c in f.columns if str(c).upper() in {"PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"}]
    out = pd.DataFrame(index=f.index)
    for c in [c for c in ["LIMIT_BAL", "AGE"] if c in f.columns]:
        out[c.lower()] = pd.to_numeric(f[c], errors="coerce")
    if bill:
        b = f[bill].apply(pd.to_numeric, errors="coerce")
        out["mean_bill_amount"] = b.mean(axis=1)
        out["max_bill_amount"] = b.max(axis=1)
        out["bill_std"] = b.std(axis=1).fillna(0)
        out["bill_trend"] = b.iloc[:, 0] - b.iloc[:, -1]
    if pay_amt:
        p = f[pay_amt].apply(pd.to_numeric, errors="coerce")
        out["mean_payment_amount"] = p.mean(axis=1)
        out["max_payment_amount"] = p.max(axis=1)
        out["payment_std"] = p.std(axis=1).fillna(0)
        if bill:
            out["payment_to_bill_ratio"] = p.mean(axis=1) / (np.abs(out["mean_bill_amount"]) + 1.0)
    if pay_status:
        s = f[pay_status].apply(pd.to_numeric, errors="coerce")
        out["repayment_delay_mean"] = s.mean(axis=1)
        out["repayment_delay_max"] = s.max(axis=1)
        out["repayment_delay_count"] = (s > 0).sum(axis=1)
    if "LIMIT_BAL" in f.columns and "mean_bill_amount" in out:
        out["credit_utilization_proxy"] = out["mean_bill_amount"] / (pd.to_numeric(f["LIMIT_BAL"], errors="coerce").abs() + 1.0)
    return out, y


def engineer_telecom(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    colmap = {str(c).lower().strip(): c for c in f.columns}
    def num(name: str) -> pd.Series:
        c = colmap.get(name.lower())
        return pd.to_numeric(f[c], errors="coerce") if c is not None else pd.Series(np.nan, index=f.index)
    out = pd.DataFrame(index=f.index)
    out["usage_intensity"] = num("Seconds of Use")
    out["sms_intensity"] = num("Frequency of SMS")
    out["call_failure_intensity"] = num("Call Failure")
    out["distinct_contact_intensity"] = num("Distinct Called Numbers")
    out["service_tenure"] = num("Subscription Length")
    out["usage_per_subscription_month"] = out["usage_intensity"] / (out["service_tenure"].abs() + 1.0)
    out["sms_to_call_ratio"] = out["sms_intensity"] / (num("Frequency of use").abs() + 1.0)
    out["complaint_indicator"] = num("Complains")
    out["charge_level"] = num("Charge Amount")
    out["customer_value_normalized"] = num("Customer Value")
    return out


def engineer_insurance(freq: pd.DataFrame, sev: pd.DataFrame) -> pd.DataFrame:
    f = freq.copy()
    if "IDpol" not in f.columns:
        raise ValueError("freMTPL2freq must contain IDpol")
    s = sev.copy()
    if "IDpol" in s.columns and "ClaimAmount" in s.columns:
        agg = s.groupby("IDpol")["ClaimAmount"].agg(["sum", "mean", "max"]).reset_index()
        agg.columns = ["IDpol", "total_claim_amount", "mean_claim_amount", "max_claim_amount"]
        f = f.merge(agg, how="left", on="IDpol")
    out = pd.DataFrame(index=f.index)
    for source, dest in [
        ("ClaimNb", "claim_count"), ("Exposure", "exposure"), ("BonusMalus", "bonus_malus"),
        ("VehAge", "vehicle_age"), ("DrivAge", "driver_age"), ("VehPower", "vehicle_power"),
        ("Density", "density"), ("total_claim_amount", "total_claim_amount"),
        ("mean_claim_amount", "mean_claim_amount"), ("max_claim_amount", "max_claim_amount"),
    ]:
        if source in f.columns:
            out[dest] = pd.to_numeric(f[source], errors="coerce")
    if "claim_count" in out:
        out["has_claim"] = (out["claim_count"] > 0).astype(int)
        out["claim_frequency"] = out["claim_count"] / (out.get("exposure", 1.0) + 1e-6)
    for c in ["Area", "Region", "VehBrand", "VehGas"]:
        if c in f.columns:
            out[c.lower()] = f[c].astype(str)
    return out


def engineer_retail(frame: pd.DataFrame, cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    required = {"CustomerID", "InvoiceNo", "Quantity", "InvoiceDate", "UnitPrice", "StockCode"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"online retail missing columns: {sorted(missing)}")
    f = frame.copy()
    f["InvoiceDate"] = pd.to_datetime(f["InvoiceDate"], errors="coerce")
    if cutoff is None:
        cutoff = f["InvoiceDate"].max()
    f = f[f["InvoiceDate"] <= cutoff].copy()
    f = f[f["CustomerID"].notna()]
    f["spend"] = pd.to_numeric(f["Quantity"], errors="coerce") * pd.to_numeric(f["UnitPrice"], errors="coerce")
    f["cancelled"] = f["InvoiceNo"].astype(str).str.startswith("C")
    g = f.groupby("CustomerID", observed=True)
    out = pd.DataFrame({
        "purchase_count": g.size(),
        "invoice_count": g["InvoiceNo"].nunique(),
        "total_spend": g["spend"].sum(),
        "average_order_value": g["spend"].mean(),
        "median_order_value": g["spend"].median(),
        "max_order_value": g["spend"].max(),
        "spend_std": g["spend"].std().fillna(0),
        "quantity_sum": g["Quantity"].sum(),
        "mean_quantity": g["Quantity"].mean(),
        "unique_products": g["StockCode"].nunique(),
        "active_days": g["InvoiceDate"].apply(lambda x: x.dt.date.nunique()),
        "cancellation_count": g["cancelled"].sum(),
    }).reset_index(drop=True)
    out["purchase_frequency"] = out["invoice_count"] / out["active_days"].clip(lower=1)
    out["cancellation_ratio"] = out["cancellation_count"] / out["purchase_count"].clip(lower=1)
    return out
