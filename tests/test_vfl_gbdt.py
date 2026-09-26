import numpy as np
from sklearn.metrics import roc_auc_score

from vertimosaic.models import VFLHistGBDT
from vertimosaic.parties import ActiveParty, PassiveParty


def test_vfl_hist_gbdt_uses_passive_signal() -> None:
    rng = np.random.default_rng(9)
    n = 800
    xb = rng.normal(size=(n, 1))
    xt = rng.normal(size=(n, 1))
    y = ((xb[:, 0] > 0.5) | (xt[:, 0] > 0.8)).astype(float)
    bank = ActiveParty("bank", xb, ("bank_x",), y)
    telco = PassiveParty("telecom", xt, ("telco_x",))
    model = VFLHistGBDT(n_estimators=15, max_depth=2, max_bins=16, min_samples_leaf=10).fit([bank, telco], bank)
    p = model.predict_proba([bank, telco])[:, 1]
    assert roc_auc_score(y, p) > 0.95
    assert model.feature_importance_["telecom"][0] > 0
    assert any(event.kind == "split_histogram" for event in model.transport.events)
