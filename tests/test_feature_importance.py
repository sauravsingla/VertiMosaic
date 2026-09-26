from vertimosaic.datasets import make_vertical_synthetic
from vertimosaic.models import VFLHistGBDT, VFLLogisticRegression
from vertimosaic.reporting import gbdt_local_feature_importance, logistic_local_feature_importance


def test_logistic_importance_is_party_local_and_standardized() -> None:
    active, passive = make_vertical_synthetic(240, seed=13)
    model = VFLLogisticRegression(max_iter=20, learning_rate=0.05)
    model.fit(active, passive)
    frames = logistic_local_feature_importance(model, [active, *passive])
    assert set(frames) == {"bank", "telecom", "insurance", "retail"}
    assert "standardized_coefficient_importance" in frames["telecom"].columns
    assert len(frames["telecom"]) == passive[0].n_features


def test_gbdt_importance_contains_only_aggregate_split_statistics() -> None:
    active, passive = make_vertical_synthetic(300, seed=14)
    model = VFLHistGBDT(n_estimators=4, max_depth=2, min_samples_leaf=15, max_bins=8)
    model.fit(active, passive)
    frames = gbdt_local_feature_importance(model)
    assert frames
    for frame in frames.values():
        assert set(frame.columns) == {"party", "feature", "split_count", "gain_sum", "gain_mean"}
        assert (frame["split_count"] > 0).all()
