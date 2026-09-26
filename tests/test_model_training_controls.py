import numpy as np

from vertimosaic.datasets import make_vertical_synthetic
from vertimosaic.models import VFLHistGBDT, VFLLogisticRegression
from vertimosaic.models.vfl_hist_gbdt import TreeNode
from vertimosaic.parties import ActiveParty, PassiveParty


def _slice(
    active: ActiveParty, passive: list[PassiveParty], indices: np.ndarray
) -> tuple[ActiveParty, list[PassiveParty]]:
    return (
        ActiveParty(active.name, active._x[indices], active.labels[indices]),
        [PassiveParty(party.name, party._x[indices]) for party in passive],
    )


def _leaf_count(node: TreeNode) -> int:
    if node.left is None and node.right is None:
        return 1
    if node.left is None or node.right is None:
        raise AssertionError("non-leaf test node must have both children")
    return _leaf_count(node.left) + _leaf_count(node.right)


def test_logistic_supports_minibatches_schedule_elastic_net_and_warm_start() -> None:
    active, passive = make_vertical_synthetic(240, seed=18)
    model = VFLLogisticRegression(
        learning_rate=0.05,
        max_iter=5,
        l1=1e-4,
        l2=1e-3,
        batch_size=32,
        learning_rate_schedule="inverse_sqrt",
        warm_start=True,
        seed=9,
    )
    model.fit(active, passive)
    first = {name: values.copy() for name, values in model.weights_.items()}
    model.fit(active, passive)
    assert 1 <= model.n_iter_ <= 5
    assert model.loss_history_
    assert any(not np.array_equal(first[name], model.weights_[name]) for name in first)


def test_gbdt_supports_leaf_subsampling_and_validation_early_stopping() -> None:
    active, passive = make_vertical_synthetic(360, seed=21)
    train_idx = np.arange(0, 260)
    validation_idx = np.arange(260, 360)
    train_active, train_passive = _slice(active, passive, train_idx)
    validation_active, validation_passive = _slice(active, passive, validation_idx)
    model = VFLHistGBDT(
        n_estimators=8,
        max_depth=3,
        max_leaves=3,
        min_samples_leaf=12,
        subsample=0.8,
        feature_subsample=0.5,
        early_stopping_rounds=2,
        seed=5,
    )
    model.fit(train_active, train_passive, validation_active, validation_passive)
    assert 1 <= len(model.trees_) <= 8
    assert model.training_loss_history_
    assert model.validation_loss_history_
    assert all(_leaf_count(tree) <= 3 for tree in model.trees_)


def test_gbdt_zero_contribution_policy_is_explicit_for_missing_parties() -> None:
    active, passive = make_vertical_synthetic(260, seed=31)
    model = VFLHistGBDT(
        n_estimators=3,
        max_depth=2,
        min_samples_leaf=15,
        missing_party_policy="zero_contribution",
    )
    model.fit(active, passive)
    probability = model.predict_proba([active])[:, 1]
    assert probability.shape == (260,)
    assert np.isfinite(probability).all()
