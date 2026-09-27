from __future__ import annotations

import json
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Any

from vertimosaic.datasets.external import ExternalDatasetBundle
from vertimosaic.datasets.external import save_bundle as _save_bundle
from vertimosaic.datasets.registry import DatasetRegistry


_PARTY_REGISTRY_KEYS: dict[str, tuple[str, ...]] = {
    "bank": ("bank",),
    "telecom": ("telecom",),
    "insurance": ("insurance_freq", "insurance_sev"),
    "retail": ("retail",),
}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_records(bundle: ExternalDatasetBundle) -> list[dict[str, Any]]:
    registry = DatasetRegistry()
    retrieval_date = str(bundle.metadata.get("retrieval_date") or date.today().isoformat())
    runtime_license = bundle.metadata.get("license")
    records: list[dict[str, Any]] = []
    for key in _PARTY_REGISTRY_KEYS.get(bundle.party, ()):
        record = dict(registry.describe(key))
        record["retrieval_date"] = retrieval_date
        record["processed_rows"] = len(bundle.features)
        # Provider APIs used by the loader do not expose a stable raw-file checksum.
        # Keep the field explicit and null rather than fabricating one.
        record["checksum"] = None
        if isinstance(runtime_license, str) and runtime_license.strip():
            record["license"] = runtime_license.strip()
        records.append(record)
    return records


def _combined_metadata(
    bundle: ExternalDatasetBundle,
    *,
    features_path: Path,
) -> dict[str, Any]:
    metadata = dict(bundle.metadata)
    sources = _source_records(bundle)
    retrieval_date = str(metadata.get("retrieval_date") or date.today().isoformat())
    if sources:
        first = sources[0]
        metadata.setdefault("dataset_name", first.get("dataset_name"))
        metadata.setdefault("provider", first.get("provider"))
        metadata.setdefault("dataset_id", first.get("dataset_id"))
        metadata.setdefault("doi", first.get("doi"))
        metadata.setdefault("provider_url", first.get("provider_url"))
        metadata.setdefault("license", first.get("license"))
        metadata.setdefault("license_url", first.get("license_url"))
        metadata.setdefault("citation", first.get("citation"))
        metadata.setdefault("retrieval_method", first.get("retrieval_method"))
    metadata["retrieval_date"] = retrieval_date
    metadata.setdefault("raw_rows", None)
    metadata["processed_rows"] = len(bundle.features)
    metadata["checksum"] = _file_sha256(features_path)
    metadata["checksum_algorithm"] = "sha256"
    metadata["checksum_scope"] = "processed_features_parquet"
    metadata["sources"] = sources
    return metadata


def save_bundle(bundle: ExternalDatasetBundle, directory: Path) -> dict[str, str]:
    """Persist a bundle and enrich its metadata with reproducible provenance.

    The checksum covers the processed feature parquet produced by VertiMosaic. It
    must not be interpreted as a checksum supplied by the original data provider.
    """
    outputs = _save_bundle(bundle, directory)
    features_path = Path(outputs["features"])
    metadata_path = Path(outputs["metadata"])
    metadata = _combined_metadata(bundle, features_path=features_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return outputs
