from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
from ucimlrepo import fetch_ucirepo

from vertimosaic.datasets import external as _external
from vertimosaic.datasets.external import ExternalDatasetBundle

_BANK_UCI_COLUMN_ALIASES = {
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    "X6": "PAY_0",
    "X7": "PAY_2",
    "X8": "PAY_3",
    "X9": "PAY_4",
    "X10": "PAY_5",
    "X11": "PAY_6",
    "X12": "BILL_AMT1",
    "X13": "BILL_AMT2",
    "X14": "BILL_AMT3",
    "X15": "BILL_AMT4",
    "X16": "BILL_AMT5",
    "X17": "BILL_AMT6",
    "X18": "PAY_AMT1",
    "X19": "PAY_AMT2",
    "X20": "PAY_AMT3",
    "X21": "PAY_AMT4",
    "X22": "PAY_AMT5",
    "X23": "PAY_AMT6",
}


def normalize_bank_provider_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize UCI dataset-350 provider columns to the documented feature names.

    UCI's current metadata exposes the physical columns as ``X1`` through ``X23``
    while the variable descriptions carry names such as ``LIMIT_BAL`` and
    ``BILL_AMT1``. Historical/provider variants may already use the descriptive
    names, so only aliases whose destination is absent are renamed.
    """
    rename = {
        source: destination
        for source, destination in _BANK_UCI_COLUMN_ALIASES.items()
        if source in frame.columns and destination not in frame.columns
    }
    return frame.rename(columns=rename).copy()


def fetch_bank() -> ExternalDatasetBundle:
    """Fetch UCI Bank data robustly across current and descriptive schemas."""
    data = fetch_ucirepo(id=350)
    provider_features = data.data.features.copy()
    targets = data.data.targets
    target_frame = targets.copy() if targets is not None else None
    original = getattr(data.data, "original", None)
    source_frame = _external._uci_source_capture(provider_features, target_frame, original)
    normalized_features = normalize_bank_provider_columns(provider_features)
    prepared, provenance = _external.prepare_bank_frame(normalized_features)
    if prepared.shape[1] == 0:
        raise RuntimeError(
            "UCI dataset 350 was retrieved but no supported Bank feature columns were resolved"
        )
    target = pd.to_numeric(targets.iloc[:, 0], errors="coerce") if targets is not None else None
    raw_rows = len(source_frame)
    aliases_applied = {
        source: destination
        for source, destination in _BANK_UCI_COLUMN_ALIASES.items()
        if source in provider_features.columns and destination not in provider_features.columns
    }
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
            "source_checksums": {"bank": _external._frame_sha256(source_frame)},
            "source_checksum_algorithm": "sha256",
            "source_checksum_scope": "retrieved_dataframe_content_before_transformation",
            "provider_column_aliases_applied": aliases_applied,
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
    """Provider-compatible public dispatcher used by experiment and CLI paths."""
    if name == "bank":
        return fetch_bank()
    return _external.fetch_external_party(
        name,
        insurance_sample_size=insurance_sample_size,
        seed=seed,
        retail_feature_cutoff=retail_feature_cutoff,
        retail_cutoff_quantile=retail_cutoff_quantile,
    )


def bank_provider_schema() -> dict[str, Any]:
    """Return the stable current UCI-to-research column mapping for provenance/tests."""
    return dict(_BANK_UCI_COLUMN_ALIASES)
