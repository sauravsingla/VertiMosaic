# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

from vertimosaic.alignment import bind_entity_ids, ordered_entity_digest
from vertimosaic.models import VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty
from vertimosaic.privacy.backends import ClippedGaussianDPBackend


def _parties() -> tuple[ActiveParty, list[PassiveParty]]:
    ids = np.asarray(["e0", "e1", "e2", "e3", "e4", "e5"])
    active = ActiveParty(
        "bank",
        np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]),
        np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
    )
    telecom = PassiveParty(
        "telecom",
        np.asarray([[5.0], [4.0], [3.0], [2.0], [1.0], [0.0]]),
    )
    insurance = PassiveParty(
        "insurance",
        np.asarray([[1.0], [0.0], [1.0], [0.0], [1.0], [0.0]]),
    )
    for party in [active, telecom, insurance]:
        bind_entity_ids(party, ids)
    return active, [telecom, insurance]


def test_ordered_entity_digest_changes_when_rows_are_permuted() -> None:
    assert ordered_entity_digest(["a", "b", "c"]) != ordered_entity_digest(["b", "a", "c"])


def test_training_rejects_equal_length_but_misaligned_party() -> None:
    active, passive = _parties()
    bind_entity_ids(passive[0], ["e1", "e0", "e2", "e3", "e4", "e5"])
    model = VFLLogisticRegression(max_iter=2, require_entity_ids=True)
    with pytest.raises(ValueError, match="entity order mismatch"):
        model.fit(active, passive)


def test_training_requires_bound_ids_in_strict_mode() -> None:
    active = ActiveParty("bank", np.ones((4, 1)), np.asarray([0.0, 0.0, 1.0, 1.0]))
    passive = [PassiveParty("telecom", np.ones((4, 1)))]
    model = VFLLogisticRegression(max_iter=2, require_entity_ids=True)
    with pytest.raises(ValueError, match="explicit entity identifiers are required"):
        model.fit(active, passive)


def test_inference_rejects_missing_party_by_default() -> None:
    active, passive = _parties()
    model = VFLLogisticRegression(max_iter=3, learning_rate=0.05, require_entity_ids=True)
    model.fit(active, passive)
    with pytest.raises(ValueError, match="missing inference parties"):
        model.predict_proba([active, passive[0]])


def test_zero_contribution_fallback_is_explicit() -> None:
    active, passive = _parties()
    model = VFLLogisticRegression(
        max_iter=3,
        learning_rate=0.05,
        require_entity_ids=True,
        missing_party_policy="zero_contribution",
    )
    model.fit(active, passive)
    probability = model.predict_proba([active, passive[0]])
    assert probability.shape == (active.n_rows, 2)
    assert np.isfinite(probability).all()


def test_clipped_gaussian_backend_is_accounted_on_real_residual_messages() -> None:
    active, passive = _parties()
    backend = ClippedGaussianDPBackend(
        clip_l2_norm=1.0,
        noise_multiplier=1.5,
        adjacency="replace_one",
        seed=7,
    )
    model = VFLLogisticRegression(
        max_iter=2,
        batch_size=3,
        learning_rate=0.05,
        require_entity_ids=True,
        residual_dp_backend=backend,
    )
    model.fit(active, passive)
    report = model.privacy_report(delta=1e-6)
    assert report is not None
    assert int(report["releases"]) > 0
    assert report["protected_message"] == "active-to-passive residual_signal"
