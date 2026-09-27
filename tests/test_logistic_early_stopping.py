import numpy as np

from vertimosaic.datasets import make_vertical_synthetic
from vertimosaic.models import VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty


def _slice(
    active: ActiveParty, passive: list[PassiveParty], indices: np.ndarray
) -> tuple[ActiveParty, list[PassiveParty]]:
    return (
        ActiveParty(active.name, active._x[indices], active.labels[indices]),
        [PassiveParty(party.name, party._x[indices]) for party in passive],
    )


def test_logistic_validation_early_stopping_tracks_and_restores_best_state() -> None:
    active, passive = make_vertical_synthetic(360, seed=44)
    train_active, train_passive = _slice(active, passive, np.arange(0, 260))
    val_active, val_passive = _slice(active, passive, np.arange(260, 360))
    model = VFLLogisticRegression(
        learning_rate=0.08,
        max_iter=20,
        l2=1e-3,
        early_stopping_rounds=2,
        seed=7,
    )
    model.fit(train_active, train_passive, val_active, val_passive)
    assert model.validation_loss_history_
    assert model.best_iteration_ is not None
    assert 1 <= model.n_iter_ <= 20
    probability = model.predict_proba([val_active, *val_passive])[:, 1]
    assert np.isfinite(probability).all()
