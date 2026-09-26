import numpy as np

from vertimosaic.datasets import generate_synthetic


def test_synthetic_four_party_schema() -> None:
    data = generate_synthetic(1000, seed=7)
    assert set(data) == {"bank", "telecom", "insurance", "retail"}
    ids = data["bank"]["entity_id"].tolist()
    for frame in data.values():
        assert frame["entity_id"].tolist() == ids
    assert "target" in data["bank"]
    assert all("target" not in data[p] for p in ("telecom", "insurance", "retail"))
    rate = data["bank"]["target"].mean()
    assert 0.01 < rate < 0.5
    assert np.isfinite(data["bank"].drop(columns=["entity_id", "target"]).to_numpy()).all()
