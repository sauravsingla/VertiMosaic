"""Programmatic external dataset retrieval.

Downloads are explicit user actions. Raw files are not committed to the repository.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

from .registry import DATASETS, DatasetSpec


def _frame_checksum(frame: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _save(frame: pd.DataFrame, spec: DatasetSpec, root: Path, metadata: dict[str, object]) -> Path:
    raw_dir = root / "raw"
    prov_dir = root / "provenance"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prov_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{spec.key}.parquet"
    frame.to_parquet(path, index=False)
    provenance = {
        **spec.to_dict(),
        "retrieval_date": datetime.now(UTC).isoformat(),
        "raw_rows": len(frame),
        "raw_columns": list(frame.columns),
        "checksum_sha256_dataframe": _frame_checksum(frame),
        **metadata,
    }
    (prov_dir / f"{spec.key}.json").write_text(json.dumps(provenance, indent=2, default=str))
    return path


def download_uci(key: str, root: Path = Path("data")) -> Path:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install VertiMosaic with the 'data' extra") from exc
    spec = DATASETS[key]
    ds = fetch_ucirepo(id=int(spec.dataset_id))
    features = ds.data.features.reset_index(drop=True)
    target = ds.data.targets.reset_index(drop=True) if ds.data.targets is not None else None
    frame = pd.concat([features, target], axis=1) if target is not None else features
    metadata = getattr(ds, "metadata", {}) or {}
    return _save(
        frame, spec, root, {"provider_metadata": metadata, "retrieval_method": "ucimlrepo"}
    )


def download_openml(key: str, root: Path = Path("data")) -> Path:
    spec = DATASETS[key]
    bunch = fetch_openml(data_id=int(spec.dataset_id), as_frame=True, parser="auto")
    frame = bunch.frame.copy()
    return _save(
        frame,
        spec,
        root,
        {
            "provider_metadata": {"details": bunch.details},
            "retrieval_method": "sklearn.fetch_openml",
        },
    )


def download_dataset(key: str, root: Path = Path("data")) -> Path:
    if key in {"bank", "telecom", "retail"}:
        return download_uci(key, root)
    if key in {"insurance_freq", "insurance_sev"}:
        return download_openml(key, root)
    raise KeyError(f"unknown dataset key: {key}")
