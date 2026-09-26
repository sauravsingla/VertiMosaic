import numpy as np

from vertimosaic.models.vfl_logistic import VFLLogisticRegression, _sigmoid
from vertimosaic.parties import ActiveParty, PassiveParty


def test_logistic_matches_centralized_gradient_descent() -> None:
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=(250, 3)); x2 = rng.normal(size=(250, 2))
    true_w = np.array([0.7, -0.5, 0.4, 0.6, -0.8]); x = np.column_stack([x1, x2])
    p = _sigmoid(x @ true_w - 0.2); y = rng.binomial(1, p).astype(float)
    active = ActiveParty("bank", x1, y); passive = PassiveParty("telecom", x2)
    model = VFLLogisticRegression(learning_rate=0.08, max_iter=350, gradient_clip=None).fit(active, [passive])
    w = np.zeros(x.shape[1]); b = 0.0
    for _ in range(350):
        pred = _sigmoid(x @ w + b); residual = pred - y
        w -= 0.08 * (x.T @ residual / len(y)); b -= 0.08 * residual.mean()
    vfl_w = np.concatenate([model.weights_["bank"], model.weights_["telecom"]])
    assert np.allclose(vfl_w, w, atol=1e-10)
    assert np.isclose(model.intercept_, b, atol=1e-10)
    assert np.allclose(model.predict_proba([active, passive])[:, 1], _sigmoid(x @ w + b), atol=1e-10)
