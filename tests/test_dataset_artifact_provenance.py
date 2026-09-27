import json
from pathlib import Path

import pandas as pd

from vertimosaic.datasets import ExternalDatasetBundle, save_bundle

_REQUIRED_SOURCE_FIELDS = {
    "dataset_name",
    "provider",
    "dataset_id",
    "doi",
    "provider_url",
    "license",
    "license_url",
    "citation",
    "retrieval_method",
    "retrieval_date",
    "checksum",
    "raw_rows",
    "processed_rows",
}


def test_saved_bundle_metadata_contains_required_provenance(tmp_path: Path) -> None:
    bundle = ExternalDatasetBundle(
        party="bank",
        features=pd.DataFrame({"feature": [1.0, 2.0, 3.0]}),
        target=pd.Series([0.0, 1.0, 0.0]),
        provenance=[],
        metadata={
            "retrieval_date": "2026-09-27",
            "raw_rows": 5,
            "source_raw_rows": {"bank": 5},
            "source_checksums": {"bank": "a" * 64},
            "source_checksum_algorithm": "sha256",
            "source_checksum_scope": "retrieved_dataframe_content_before_transformation",
        },
    )

    outputs = save_bundle(bundle, tmp_path)
    metadata = json.loads(Path(outputs["metadata"]).read_text(encoding="utf-8"))

    assert _REQUIRED_SOURCE_FIELDS.issubset(metadata)
    assert metadata["processed_rows"] == 3
    assert metadata["raw_rows"] == 5
    assert metadata["checksum_algorithm"] == "sha256"
    assert metadata["checksum_scope"] == "processed_features_parquet"
    assert len(metadata["checksum"]) == 64
    assert metadata["sources"][0]["dataset_id"] == "350"
    assert _REQUIRED_SOURCE_FIELDS.issubset(metadata["sources"][0])
    assert metadata["sources"][0]["raw_rows"] == 5
    assert metadata["sources"][0]["processed_rows"] == 3
    assert metadata["sources"][0]["checksum"] == "a" * 64
    assert metadata["sources"][0]["checksum_scope"] == (
        "retrieved_dataframe_content_before_transformation"
    )


def test_runtime_insurance_licenses_and_checksums_are_retained_per_source(tmp_path: Path) -> None:
    bundle = ExternalDatasetBundle(
        party="insurance",
        features=pd.DataFrame({"claim_count": [0.0, 1.0]}),
        target=None,
        provenance=[],
        metadata={
            "retrieval_date": "2026-09-27",
            "license": "frequency-license",
            "source_licenses": {
                "insurance_freq": "frequency-license",
                "insurance_sev": "severity-license",
            },
            "raw_rows": 12,
            "source_raw_rows": {"insurance_freq": 10, "insurance_sev": 2},
            "source_checksums": {
                "insurance_freq": "b" * 64,
                "insurance_sev": "c" * 64,
            },
            "source_checksum_algorithm": "sha256",
            "source_checksum_scope": "retrieved_dataframe_content_before_sampling_or_transformation",
        },
    )

    outputs = save_bundle(bundle, tmp_path)
    metadata = json.loads(Path(outputs["metadata"]).read_text(encoding="utf-8"))

    assert metadata["license"] == "frequency-license"
    assert {source["dataset_id"] for source in metadata["sources"]} == {"41214", "41215"}
    source_licenses = {source["dataset_id"]: source["license"] for source in metadata["sources"]}
    assert source_licenses == {"41214": "frequency-license", "41215": "severity-license"}
    assert all(_REQUIRED_SOURCE_FIELDS.issubset(source) for source in metadata["sources"])
    source_rows = {source["dataset_id"]: source["raw_rows"] for source in metadata["sources"]}
    assert source_rows == {"41214": 10, "41215": 2}
    source_checksums = {source["dataset_id"]: source["checksum"] for source in metadata["sources"]}
    assert source_checksums == {"41214": "b" * 64, "41215": "c" * 64}
