import numpy as np

from vertimosaic.models import VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty


def test_vfl_logistic_matches_joint_linear_signal() -> None:
    rng = np.random.default_rng(11)
    n = 1500
    xb = rng.normal(size=(n, 2))
    xt = rng.normal(size=(n, 2))
    logits = 1.2 * xb[:, 0] - 0.8 * xb[:, 1] + 1.0 * xt[:, 0] + 0.5 * xt[:, 1]
    y = rng.binomial(1, 1 / (1 + np.exp(-logits))).astype(float)
    bank = ActiveParty("bank", xb, ("b0", "b1"), y)
    telco = PassiveParty("telecom", xt, ("t0", "t1"))
    model = VFLLogisticRegression(learning_rate=0.15, epochs=180, batch_size=n, l2=0.0, patience=180).fit([bank, telco], bank)
    pred = model.predict_proba([bank, telco])[:, 1]
    assert np.corrcoef(pred, 1 / (1 + np.exp(-logits)))[0, 1] > 0.97
    assert np.allclose(model.weights_["bank"], [1.2, -0.8], atol=0.3)
    assert np.allclose(model.weights_["telecom"], [1.0, 0.5], atol=0.3)
