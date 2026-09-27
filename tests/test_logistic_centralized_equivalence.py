# SPDX-License-Identifier: Apache-2.0
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from vertimosaic.models.vfl_logistic import VFLLogisticRegression, _sigmoid
from vertimosaic.parties import ActiveParty, PassiveParty


def _binary_log_loss(y: np.ndarray, probability: np.ndarray) -> float:
    eps = 1e-12
    return float(
        -np.mean(y * np.log(probability + eps) + (1.0 - y) * np.log(1.0 - probability + eps))
    )


def test_logistic_matches_centralized_gradient_descent() -> None:
    """Verify the VFL objective against the same centralized first-principles update.

    This is deliberately stronger than a probability smoke test: coefficients,
    intercept, raw logits, probabilities, objective loss, ROC-AUC and PR-AUC must all
    agree with a centralized implementation when regularization, clipping, batching and
    scheduling are disabled so the mathematical objectives are identical.
    """
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=(250, 3))
    x2 = rng.normal(size=(250, 2))
    true_w = np.array([0.7, -0.5, 0.4, 0.6, -0.8])
    x = np.column_stack([x1, x2])
    probability = _sigmoid(x @ true_w - 0.2)
    y = rng.binomial(1, probability).astype(float)

    active = ActiveParty("bank", x1, y)
    passive = PassiveParty("telecom", x2)
    model = VFLLogisticRegression(
        learning_rate=0.08,
        max_iter=350,
        gradient_clip=None,
        tolerance=0.0,
    )
    model.fit(active, [passive])

    centralized_w = np.zeros(x.shape[1])
    centralized_intercept = 0.0
    for _ in range(350):
        centralized_logits = x @ centralized_w + centralized_intercept
        centralized_probability = _sigmoid(centralized_logits)
        residual = centralized_probability - y
        centralized_w -= 0.08 * (x.T @ residual / len(y))
        centralized_intercept -= 0.08 * residual.mean()

    centralized_logits = x @ centralized_w + centralized_intercept
    centralized_probability = _sigmoid(centralized_logits)
    centralized_loss = _binary_log_loss(y, centralized_probability)
    centralized_roc_auc = roc_auc_score(y, centralized_probability)
    centralized_pr_auc = average_precision_score(y, centralized_probability)

    vfl_w = np.concatenate([model.weights_["bank"], model.weights_["telecom"]])
    vfl_logits = model.decision_function([active, passive])
    vfl_probability = model.predict_proba([active, passive])[:, 1]
    vfl_loss = model._loss(y, vfl_probability, np.ones_like(y))
    vfl_roc_auc = roc_auc_score(y, vfl_probability)
    vfl_pr_auc = average_precision_score(y, vfl_probability)

    assert np.allclose(vfl_w, centralized_w, atol=1e-10)
    assert np.isclose(model.intercept_, centralized_intercept, atol=1e-10)
    assert np.allclose(vfl_logits, centralized_logits, atol=1e-10)
    assert np.allclose(vfl_probability, centralized_probability, atol=1e-10)
    assert np.isclose(vfl_loss, centralized_loss, atol=1e-12)
    assert np.isclose(vfl_roc_auc, centralized_roc_auc, atol=1e-12)
    assert np.isclose(vfl_pr_auc, centralized_pr_auc, atol=1e-12)
