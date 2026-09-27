import numpy as np
import pytest

from vertimosaic.models import VFLHistGBDT, VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty


def _data(rows: int = 48) -> tuple[ActiveParty, list[PassiveParty]]:
    rng = np.random.default_rng(123)
    labels = np.tile(np.array([0.0, 1.0]), rows // 2)
    active = ActiveParty("bank", rng.normal(size=(rows, 3)), labels)
    passive = [
        PassiveParty("telecom", rng.normal(size=(rows, 2))),
        PassiveParty("insurance", rng.normal(size=(rows, 2))),
        PassiveParty("retail", rng.normal(size=(rows, 2))),
    ]
    return active, passive


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": 0.0},
        {"max_iter": 0},
        {"l1": -1.0},
        {"l2": -1.0},
        {"gradient_clip": 0.0},
        {"batch_size": 0},
        {"learning_rate_schedule": "unsupported"},
        {"early_stopping_rounds": 0},
        {"class_weight": "unsupported"},
    ],
)
def test_logistic_rejects_invalid_hyperparameters(kwargs: dict[str, object]) -> None:
    active, passive = _data()
    model = VFLLogisticRegression(**kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        model.fit(active, passive)


def test_logistic_exercises_weighting_schedules_batches_and_warm_start() -> None:
    active, passive = _data()
    model = VFLLogisticRegression(
        learning_rate=0.05,
        max_iter=3,
        l1=1e-4,
        l2=1e-4,
        class_weight={0: 0.75, 1: 1.25},
        batch_size=12,
        learning_rate_schedule="linear_decay",
        warm_start=True,
        seed=11,
    )
    model.fit(active, passive)
    first_weights = {name: value.copy() for name, value in model.weights_.items()}
    model.fit(active, passive)
    assert model.n_iter_ > 0
    assert set(model.weights_) == set(first_weights)

    inverse = VFLLogisticRegression(
        learning_rate=0.05,
        max_iter=2,
        learning_rate_schedule="inverse_sqrt",
        class_weight="balanced",
        seed=11,
    )
    inverse.fit(active, passive)
    assert inverse.loss_history_


def test_logistic_validation_and_inference_guards() -> None:
    active, passive = _data()
    with pytest.raises(ValueError, match="validation data"):
        VFLLogisticRegression(early_stopping_rounds=2).fit(active, passive)

    validation_active, validation_passive = _data(24)
    model = VFLLogisticRegression(max_iter=3, early_stopping_rounds=1, seed=4)
    model.fit(active, passive, validation_active, validation_passive)
    assert model.validation_loss_history_
    assert model.best_iteration_ is not None

    missing_validation_party = validation_passive[:-1]
    with pytest.raises(ValueError, match="same VFL parties"):
        VFLLogisticRegression(max_iter=1).fit(
            active,
            passive,
            validation_active,
            missing_validation_party,
        )

    bad_validation_passive = [
        PassiveParty("telecom", np.zeros((23, 2))),
        PassiveParty("insurance", np.zeros((24, 2))),
        PassiveParty("retail", np.zeros((24, 2))),
    ]
    with pytest.raises(ValueError, match="align"):
        VFLLogisticRegression(max_iter=1).fit(
            active,
            passive,
            validation_active,
            bad_validation_passive,
        )

    unfitted = VFLLogisticRegression()
    with pytest.raises(RuntimeError, match="not fitted"):
        unfitted.decision_function([active])
    with pytest.raises(ValueError, match="at least one party"):
        model.decision_function([])
    with pytest.raises(ValueError, match="equal row counts"):
        model.decision_function([active, PassiveParty("telecom", np.zeros((47, 2)))])
    with pytest.raises(ValueError, match="unknown inference party"):
        model.decision_function([PassiveParty("unknown", np.zeros((48, 2)))])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_estimators": 0},
        {"learning_rate": 0.0},
        {"max_depth": -1},
        {"max_leaves": 1},
        {"min_samples_leaf": 0},
        {"min_child_weight": -1.0},
        {"l2_leaf_reg": -1.0},
        {"max_bins": 1},
        {"subsample": 0.0},
        {"subsample": 1.1},
        {"feature_subsample": 0.0},
        {"feature_subsample": 1.1},
        {"early_stopping_rounds": 0},
        {"missing_party_policy": "unsupported"},
    ],
)
def test_gbdt_rejects_invalid_hyperparameters(kwargs: dict[str, object]) -> None:
    active, passive = _data()
    model = VFLHistGBDT(**kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        model.fit(active, passive)


def test_gbdt_exercises_subsampling_validation_and_missing_party_paths() -> None:
    active, passive = _data()
    validation_active, validation_passive = _data(24)

    with pytest.raises(ValueError, match="validation data"):
        VFLHistGBDT(early_stopping_rounds=1).fit(active, passive)

    model = VFLHistGBDT(
        n_estimators=4,
        max_depth=2,
        max_leaves=3,
        min_samples_leaf=3,
        subsample=0.7,
        feature_subsample=0.5,
        early_stopping_rounds=2,
        seed=9,
    )
    model.fit(active, passive, validation_active, validation_passive)
    assert model.trees_
    assert model.training_loss_history_
    assert model.validation_loss_history_

    with pytest.raises(ValueError, match="same VFL parties"):
        VFLHistGBDT(n_estimators=1).fit(
            active,
            passive,
            validation_active,
            validation_passive[:-1],
        )

    bad_validation_passive = [
        PassiveParty("telecom", np.zeros((23, 2))),
        PassiveParty("insurance", np.zeros((24, 2))),
        PassiveParty("retail", np.zeros((24, 2))),
    ]
    with pytest.raises(ValueError, match="align"):
        VFLHistGBDT(n_estimators=1).fit(
            active,
            passive,
            validation_active,
            bad_validation_passive,
        )

    with pytest.raises(RuntimeError, match="not fitted"):
        VFLHistGBDT().decision_function([active])
    with pytest.raises(ValueError, match="at least one party"):
        model.decision_function([])
    with pytest.raises(ValueError, match="missing parties"):
        model.decision_function([active])

    zero_policy = VFLHistGBDT(
        n_estimators=2,
        max_depth=1,
        min_samples_leaf=3,
        missing_party_policy="zero_contribution",
        seed=9,
    )
    zero_policy.fit(active, passive)
    probability = zero_policy.predict_proba([active, passive[0]])
    assert probability.shape == (active.n_rows, 2)

    mismatched = PassiveParty("telecom", np.zeros((47, 2)))
    with pytest.raises(ValueError, match="equal row counts"):
        zero_policy.decision_function([active, mismatched])
