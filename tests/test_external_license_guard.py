import numpy as np
import pandas as pd
import pytest

from vertimosaic.datasets import ExternalDatasetBundle
from vertimosaic.experiments import external as external_module


def _bundle(party: str, *, license_value: str | None = None) -> ExternalDatasetBundle:
    rows = 12
    target = pd.Series(np.tile([0.0, 1.0], rows // 2)) if party == "bank" else None
    keys = {
        "bank": ("bank",),
        "telecom": ("telecom",),
        "insurance": ("insurance_freq", "insurance_sev"),
        "retail": ("retail",),
    }[party]
    metadata: dict[str, object] = {
        "provider": "test",
        "retrieval_date": "2026-09-27",
        "raw_rows": rows,
        "source_raw_rows": {key: rows for key in keys},
        "source_checksums": {key: f"{index + 1:064x}" for index, key in enumerate(keys)},
        "source_checksum_algorithm": "sha256",
        "source_checksum_scope": "test-retrieved-frame",
    }
    if license_value is not None:
        metadata["license"] = license_value
        if party == "insurance":
            metadata["source_licenses"] = {
                "insurance_freq": f"{license_value}-freq",
                "insurance_sev": f"{license_value}-sev",
            }
    return ExternalDatasetBundle(
        party=party,
        features=pd.DataFrame({"value": np.arange(rows, dtype=float)}),
        target=target,
        provenance=[],
        metadata=metadata,
    )


def test_external_fetch_stops_when_openml_license_is_unverifiable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundles = {
        "bank": _bundle("bank"),
        "telecom": _bundle("telecom"),
        "insurance": _bundle("insurance"),
        "retail": _bundle("retail"),
    }

    def fake_fetch(name: str, **_: object) -> ExternalDatasetBundle:
        return bundles[name]

    monkeypatch.setattr(external_module, "fetch_external_party", fake_fetch)
    monkeypatch.setattr(
        external_module.DatasetRegistry,
        "runtime_license",
        lambda self, name: None if name.startswith("insurance_") else self.get(name).license,
    )
    with pytest.raises(RuntimeError, match="license metadata could not be verified"):
        external_module.prepare_external_benchmark()


def test_external_fetch_accepts_verified_per_source_insurance_licenses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundles = {
        "bank": _bundle("bank"),
        "telecom": _bundle("telecom"),
        "insurance": _bundle("insurance", license_value="ODbL-1.0"),
        "retail": _bundle("retail"),
    }

    def fake_fetch(name: str, **_: object) -> ExternalDatasetBundle:
        return bundles[name]

    monkeypatch.setattr(external_module, "fetch_external_party", fake_fetch)
    benchmark = external_module.prepare_external_benchmark(cross_party_correlation=0.0, seed=3)
    assert benchmark.active.n_rows == 12
    assert [party.name for party in benchmark.passive] == ["telecom", "insurance", "retail"]
    insurance_sources = benchmark.source_metadata["insurance"]["sources"]
    assert [source["license"] for source in insurance_sources] == [
        "ODbL-1.0-freq",
        "ODbL-1.0-sev",
    ]
    assert all(len(source["checksum"]) == 64 for source in insurance_sources)
