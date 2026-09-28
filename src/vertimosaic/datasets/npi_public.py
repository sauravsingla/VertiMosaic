# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import pandas as pd


def _normalized_npi(series: pd.Series, *, source: str) -> pd.Series:
    values = series.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    valid = values.str.fullmatch(r"\d{10}", na=False)
    if not bool(valid.all()):
        bad = int((~valid).sum())
        raise ValueError(f"{source}: {bad} rows contain invalid/non-10-digit NPI values")
    return values.astype(str)


def _frame_digest(frame: pd.DataFrame) -> str:
    first_column = str(frame.columns[0])
    canonical = frame.sort_index(axis=1).sort_values(first_column).reset_index(drop=True)
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return sha256(payload).hexdigest()


def _select_columns(frame: pd.DataFrame, columns: Sequence[str], *, source: str) -> pd.DataFrame:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source}: missing requested columns: {missing}")
    return frame.loc[:, list(columns)].copy()


@dataclass(frozen=True)
class NPILinkedFrames:
    """Three-source exact-NPI linked public benchmark frames.

    The active frame contains prior-year Open Payments aggregates. ``target`` is derived
    from a later Open Payments period. Passive frames contain NPPES and Care Compare / CMS
    provider-data columns for the exact same ordered NPI entities.
    """

    entity_ids: tuple[str, ...]
    active: pd.DataFrame
    nppes: pd.DataFrame
    care_compare: pd.DataFrame
    target: np.ndarray
    manifest: dict[str, object]


def build_npi_linked_frames(
    prior_payments: pd.DataFrame,
    target_payments: pd.DataFrame,
    nppes: pd.DataFrame,
    care_compare: pd.DataFrame,
    *,
    prior_npi_column: str,
    target_npi_column: str,
    nppes_npi_column: str,
    care_npi_column: str,
    prior_amount_column: str,
    target_amount_column: str,
    nppes_feature_columns: Sequence[str],
    care_feature_columns: Sequence[str],
    target_quantile: float = 0.75,
    min_prior_payment_records: int = 1,
) -> NPILinkedFrames:
    """Build a genuinely exact-linked multi-source provider benchmark using NPI.

    This function intentionally accepts locally downloaded public CMS/NPPES source files
    rather than hard-coding unstable download URLs or schemas. The NPI is an authoritative
    provider identifier, so linkage is exact rather than synthetic or similarity-based.

    The benchmark is *multi-source* and describes real providers. It must not be described
    as evidence of private cross-company data collaboration: Open Payments, NPPES, and Care
    Compare/provider-data are public US government data sources.
    """
    if not 0.0 < target_quantile < 1.0:
        raise ValueError("target_quantile must be in (0, 1)")
    if min_prior_payment_records < 1:
        raise ValueError("min_prior_payment_records must be positive")

    for frame, column, source in (
        (prior_payments, prior_npi_column, "prior Open Payments"),
        (target_payments, target_npi_column, "target Open Payments"),
        (nppes, nppes_npi_column, "NPPES"),
        (care_compare, care_npi_column, "Care Compare/provider-data"),
    ):
        if column not in frame.columns:
            raise ValueError(f"{source}: missing NPI column {column!r}")

    if prior_amount_column not in prior_payments.columns:
        raise ValueError(f"prior Open Payments: missing amount column {prior_amount_column!r}")
    if target_amount_column not in target_payments.columns:
        raise ValueError(f"target Open Payments: missing amount column {target_amount_column!r}")

    prior = prior_payments.copy()
    future = target_payments.copy()
    nppes_local = nppes.copy()
    care_local = care_compare.copy()
    prior["__npi"] = _normalized_npi(prior[prior_npi_column], source="prior Open Payments")
    future["__npi"] = _normalized_npi(future[target_npi_column], source="target Open Payments")
    nppes_local["__npi"] = _normalized_npi(nppes_local[nppes_npi_column], source="NPPES")
    care_local["__npi"] = _normalized_npi(
        care_local[care_npi_column], source="Care Compare/provider-data"
    )

    prior_amount = pd.to_numeric(prior[prior_amount_column], errors="coerce")
    future_amount = pd.to_numeric(future[target_amount_column], errors="coerce")
    if prior_amount.isna().all() or future_amount.isna().all():
        raise ValueError("payment amount columns must contain numeric values")
    prior["__amount"] = prior_amount.fillna(0.0)
    future["__amount"] = future_amount.fillna(0.0)

    prior_agg = (
        prior.groupby("__npi", sort=True)["__amount"]
        .agg(
            prior_payment_count="size",
            prior_payment_sum="sum",
            prior_payment_mean="mean",
        )
        .reset_index()
    )
    prior_agg = prior_agg[prior_agg["prior_payment_count"] >= min_prior_payment_records]
    future_agg = (
        future.groupby("__npi", sort=True)["__amount"]
        .agg(target_payment_count="size", target_payment_sum="sum")
        .reset_index()
    )

    nppes_features = _select_columns(nppes_local, nppes_feature_columns, source="NPPES")
    nppes_features.insert(0, "__npi", nppes_local["__npi"].to_numpy())
    nppes_features = nppes_features.drop_duplicates("__npi", keep="first")
    care_features = _select_columns(
        care_local,
        care_feature_columns,
        source="Care Compare/provider-data",
    )
    care_features.insert(0, "__npi", care_local["__npi"].to_numpy())
    care_features = care_features.drop_duplicates("__npi", keep="first")

    linked = prior_agg.merge(future_agg, on="__npi", how="inner", validate="one_to_one")
    linked = linked.merge(nppes_features, on="__npi", how="inner", validate="one_to_one")
    linked = linked.merge(
        care_features,
        on="__npi",
        how="inner",
        validate="one_to_one",
        suffixes=("__nppes", "__care"),
    )
    linked = linked.sort_values("__npi").reset_index(drop=True)
    if len(linked) < 100:
        raise ValueError(
            "exact linked benchmark contains fewer than 100 entities; check source years, "
            "NPI columns, and feature-source coverage"
        )

    threshold = float(linked["target_payment_sum"].quantile(target_quantile))
    target = (linked["target_payment_sum"].to_numpy(dtype=float) >= threshold).astype(float)
    ids = tuple(linked["__npi"].astype(str).tolist())

    active = linked[["prior_payment_count", "prior_payment_sum", "prior_payment_mean"]].copy()
    nppes_requested = set(nppes_feature_columns)
    care_requested = set(care_feature_columns)
    nppes_names = [column for column in linked.columns if column in nppes_requested]
    care_names = [column for column in linked.columns if column in care_requested]
    if len(nppes_names) != len(nppes_feature_columns):
        nppes_names = [
            f"{column}__nppes" if f"{column}__nppes" in linked.columns else column
            for column in nppes_feature_columns
        ]
    if len(care_names) != len(care_feature_columns):
        care_names = [
            f"{column}__care" if f"{column}__care" in linked.columns else column
            for column in care_feature_columns
        ]
    nppes_party = linked[nppes_names].copy()
    care_party = linked[care_names].copy()

    manifest: dict[str, object] = {
        "benchmark": "public_npi_multisource",
        "linkage": "exact 10-digit National Provider Identifier intersection",
        "entities": len(ids),
        "target": "later-period Open Payments total >= configured quantile threshold",
        "target_quantile": float(target_quantile),
        "target_threshold": threshold,
        "active_source": "Open Payments prior period",
        "passive_sources": ["NPPES", "CMS Care Compare/provider-data"],
        "source_scope": (
            "real public provider records from distinct CMS/NPPES data products; "
            "not evidence of private cross-company collaboration"
        ),
        "digests": {
            "active": _frame_digest(active.assign(__npi=list(ids))),
            "nppes": _frame_digest(nppes_party.assign(__npi=list(ids))),
            "care_compare": _frame_digest(care_party.assign(__npi=list(ids))),
        },
    }
    return NPILinkedFrames(
        entity_ids=ids,
        active=active,
        nppes=nppes_party,
        care_compare=care_party,
        target=target,
        manifest=manifest,
    )
