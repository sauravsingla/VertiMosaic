import numpy as np

from vertimosaic.baselines import fit_party_subset_baselines


def test_centralized_subset_baselines_cover_all_bank_anchored_party_sets() -> None:
    rng = np.random.default_rng(23)
    rows = 80
    labels = np.tile(np.array([0.0, 1.0]), rows // 2)
    train = {
        "bank": rng.normal(size=(rows, 2)),
        "telecom": rng.normal(size=(rows, 2)),
        "insurance": rng.normal(size=(rows, 1)),
        "retail": rng.normal(size=(rows, 2)),
    }
    results = fit_party_subset_baselines(train, labels, train, model="logistic", seed=23)
    assert len(results) == 8
    assert "bank" in results
    assert "bank+insurance+retail+telecom" in results
    assert all(result.label == "NON-FEDERATED BASELINE" for result in results.values())
    assert all(result.probabilities.shape == (rows,) for result in results.values())
