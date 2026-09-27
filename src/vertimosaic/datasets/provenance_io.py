from __future__ import annotations

import json
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Any

from vertimosaic.datasets.external import (
    ExternalDatasetBundle,
)
from vertimosaic.datasets.external import (
    save_bundle as _save_bundle,
)
from vertimosaic.datasets.registry import DatasetRegistry


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
    source_licenses = bundle.metadata.get("source_licenses", {})
    if not isinstance(source_licenses, dict):
        source_licenses = {}
    raw_rows = bundle.metadata.get("raw_rows")
    source_raw_rows = bundle.metadata.get("source_raw_rows", {})
    if not isinstance(source_raw_rows, dict):
        source_raw_rows = {}
    keys = registry.keys_for_party(bundle.party)
    records: list[dict[str, Any]] = []
    for key in keys:
        record = dict(registry.describe(key))
        record["retrieval_date"] = retrieval_date
        record["processed_rows"] = len(bundle.features)
        # Provider APIs used by the loader do not expose a stable raw-file checksum.
        # Keep the field explicit and null rather than fabricating one.
        record["checksum"] = None
        if key in source_raw_rows:
            record["raw_rows"] = source_raw_rows[key]
        elif len(keys) == 1:
            record["raw_rows"] = raw_rows
        else:
            # A combined multi-source row count cannot safely be assigned to either
            # underlying source, so preserve the unknown value explicitly.
            record["raw_rows"] = None
        source_license = source_licenses.get(key)
        if not isinstance(source_license, str) or not source_license.strip():
            if len(keys) == 1 and isinstance(runtime_license, str) and runtime_license.strip():
                source_license = runtime_license
            elif record.get("license") is None:
                source_license = registry.runtime_license(key)
        if isinstance(source_license, str) and source_license.strip():
            record["license"] = source_license.strip()
        if record.get("license") is None:
            raise RuntimeError(
                f"license metadata could not be verified for source {key}; "
                "provenance persistence stopped conservatively"
            )
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
