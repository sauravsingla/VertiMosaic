import numpy as np
import pandas as pd
import pytest

from vertimosaic.datasets import ExternalDatasetBundle
from vertimosaic.experiments import external as external_module


def _bundle(party: str, *, license_value: str | None = None) -> ExternalDatasetBundle:
    rows = 12
    target = pd.Series(np.tile([0.0, 1.0], rows // 2)) if party == "bank" else None
    metadata: dict[str, object] = {"provider": "test"}
    if license_value is not None:
        metadata["license"] = license_value
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
    with pytest.raises(RuntimeError, match="license metadata could not be verified"):
        external_module.prepare_external_benchmark()


def test_external_fetch_accepts_provider_insurance_license(
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
