from vertimosaic.datasets import make_vertical_synthetic
from vertimosaic.models import VFLHistGBDT
from vertimosaic.reporting import gbdt_local_feature_importance


def test_gbdt_importance_can_be_aggregated_by_split_owning_parties() -> None:
    active, passive = make_vertical_synthetic(320, seed=37)
    model = VFLHistGBDT(n_estimators=4, max_depth=2, min_samples_leaf=12, max_bins=8)
    model.fit(active, passive)
    frames = gbdt_local_feature_importance(model, [active, *passive])
    assert frames
    for party, frame in frames.items():
        assert party in {"bank", "telecom", "insurance", "retail"}
        assert set(frame.columns) == {"party", "feature", "split_count", "gain_sum", "gain_mean"}
        assert (frame["split_count"] > 0).all()
