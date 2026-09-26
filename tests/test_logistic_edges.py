import numpy as np
import pytest

from vertimosaic.models import VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty


def _parties(seed: int = 1):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=(80, 2))
    x2 = rng.normal(size=(80, 2))
    y = (x1[:, 0] + x2[:, 0] > 0).astype(float)
    return ActiveParty("bank", x1, y), PassiveParty("telecom", x2)


def test_predict_before_fit_raises() -> None:
    active, passive = _parties()
    with pytest.raises(RuntimeError):
        VFLLogisticRegression().predict_proba([active, passive])


def test_fit_rejects_unaligned_rows() -> None:
    active, passive = _parties()
    bad = PassiveParty("telecom", passive._x[:-1])
    with pytest.raises(ValueError):
        VFLLogisticRegression(max_iter=2).fit(active, [bad])


def test_balanced_l1_l2_clip_and_predict() -> None:
    active, passive = _parties(2)
    model = VFLLogisticRegression(
        max_iter=30,
        learning_rate=0.2,
        class_weight="balanced",
        l1=1e-3,
        l2=1e-3,
        gradient_clip=0.05,
    ).fit(active, [passive])
    pred = model.predict([active, passive], threshold=0.4)
    assert set(np.unique(pred)).issubset({0, 1})
    assert len(model.loss_history_) >= 1


def test_decision_requires_parties() -> None:
    active, passive = _parties(3)
    model = VFLLogisticRegression(max_iter=2).fit(active, [passive])
    with pytest.raises(ValueError):
        model.decision_function([])
