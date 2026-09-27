import json
from pathlib import Path

import pandas as pd

from vertimosaic.datasets import ExternalDatasetBundle, save_bundle


def test_saved_bundle_metadata_contains_required_provenance(tmp_path: Path) -> None:
    bundle = ExternalDatasetBundle(
        party="bank",
        features=pd.DataFrame({"feature": [1.0, 2.0, 3.0]}),
        target=pd.Series([0.0, 1.0, 0.0]),
        provenance=[],
        metadata={"retrieval_date": "2026-09-27"},
    )

    outputs = save_bundle(bundle, tmp_path)
    metadata = json.loads(Path(outputs["metadata"]).read_text(encoding="utf-8"))

    required = {
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
    assert required.issubset(metadata)
    assert metadata["processed_rows"] == 3
    assert metadata["raw_rows"] is None
    assert metadata["checksum_algorithm"] == "sha256"
    assert metadata["checksum_scope"] == "processed_features_parquet"
    assert len(metadata["checksum"]) == 64
    assert metadata["sources"][0]["dataset_id"] == "350"


def test_runtime_insurance_license_is_retained_for_both_source_records(tmp_path: Path) -> None:
    bundle = ExternalDatasetBundle(
        party="insurance",
        features=pd.DataFrame({"claim_count": [0.0, 1.0]}),
        target=None,
        provenance=[],
        metadata={"retrieval_date": "2026-09-27", "license": "provider-runtime-license"},
    )

    outputs = save_bundle(bundle, tmp_path)
    metadata = json.loads(Path(outputs["metadata"]).read_text(encoding="utf-8"))

    assert metadata["license"] == "provider-runtime-license"
    assert {source["dataset_id"] for source in metadata["sources"]} == {"41214", "41215"}
    assert all(source["license"] == "provider-runtime-license" for source in metadata["sources"])
