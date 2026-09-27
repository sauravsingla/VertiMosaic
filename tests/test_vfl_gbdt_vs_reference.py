from vertimosaic.datasets import make_vertical_synthetic
from vertimosaic.models import VFLHistGBDT


def test_vfl_gbdt_fits_and_predicts_probabilities() -> None:
    active, passive = make_vertical_synthetic(300, seed=4)
    model = VFLHistGBDT(n_estimators=3, max_depth=2, min_samples_leaf=15, max_bins=8)
    model.fit(active, passive)
    p = model.predict_proba([active, *passive])[:, 1]
    assert p.shape == (300,)
    assert ((p >= 0.0) & (p <= 1.0)).all()
    assert len(model.trees_) == 3
