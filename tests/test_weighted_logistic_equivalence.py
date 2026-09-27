# SPDX-License-Identifier: Apache-2.0
import numpy as np

from vertimosaic.models.vfl_logistic import VFLLogisticRegression, _sigmoid
from vertimosaic.parties import ActiveParty, PassiveParty


def _centralized_weighted_updates(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    coef = np.zeros(x.shape[1], dtype=float)
    intercept = 0.0
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = np.arange(len(y), dtype=int) if batch_size == len(y) else rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            batch = order[start : start + batch_size]
            probability = _sigmoid(x[batch] @ coef + intercept)
            batch_weights = weights[batch]
            denominator = float(batch_weights.sum())
            residual = (probability - y[batch]) * batch_weights
            coef -= learning_rate * (x[batch].T @ residual / denominator)
            intercept -= learning_rate * float(residual.sum() / denominator)
    return coef, intercept


def _assert_equivalence(class_weight: str | dict[int, float], batch_size: int) -> None:
    rng = np.random.default_rng(2026)
    x_bank = rng.normal(size=(211, 3))
    x_telco = rng.normal(size=(211, 2))
    x = np.column_stack([x_bank, x_telco])
    latent = x @ np.array([0.4, -0.7, 0.2, 0.6, -0.3]) - 1.25
    y = rng.binomial(1, _sigmoid(latent)).astype(float)
    active = ActiveParty("bank", x_bank, y)
    passive = PassiveParty("telecom", x_telco)

    model = VFLLogisticRegression(
        learning_rate=0.035,
        max_iter=17,
        class_weight=class_weight,
        batch_size=batch_size,
        gradient_clip=None,
        tolerance=0.0,
        seed=77,
    )
    sample_weight = model._sample_weights(y)
    expected_coef, expected_intercept = _centralized_weighted_updates(
        x,
        y,
        sample_weight,
        learning_rate=0.035,
        epochs=17,
        batch_size=batch_size,
        seed=77,
    )
    model.fit(active, [passive])

    actual_coef = np.concatenate([model.weights_["bank"], model.weights_["telecom"]])
    assert np.allclose(actual_coef, expected_coef, atol=1e-11)
    assert np.isclose(model.intercept_, expected_intercept, atol=1e-11)

    probability = _sigmoid(x @ expected_coef + expected_intercept)
    expected_loss = -np.average(
        y * np.log(probability + 1e-12) + (1.0 - y) * np.log(1.0 - probability + 1e-12),
        weights=sample_weight,
    )
    assert np.isclose(model.loss_history_[-1], expected_loss, atol=1e-12)


def test_dictionary_class_weights_match_weighted_centralized_minibatches() -> None:
    _assert_equivalence({0: 0.65, 1: 4.25}, batch_size=23)


def test_balanced_class_weights_match_weighted_centralized_minibatches() -> None:
    _assert_equivalence("balanced", batch_size=29)


def test_zero_total_class_weights_are_rejected() -> None:
    active = ActiveParty("bank", np.ones((4, 1)), np.array([0.0, 1.0, 0.0, 1.0]))
    model = VFLLogisticRegression(class_weight={0: 0.0, 1: 0.0})
    try:
        model.fit(active, [])
    except ValueError as exc:
        assert "positive total weight" in str(exc)
    else:
        raise AssertionError("zero-total class weights must be rejected")
