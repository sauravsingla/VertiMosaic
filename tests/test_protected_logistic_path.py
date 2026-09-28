# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from vertimosaic.parties import ActiveParty, PassiveParty
from vertimosaic.privacy.protected_logistic import fit_protected_logistic, psi_align_parties


class ReverseIntersectionPSI:
    def intersection(self, client_ids: list[str], server_ids: list[str]) -> list[str]:
        server = set(server_ids)
        return list(reversed([value for value in client_ids if value in server]))


def test_psi_alignment_restores_active_canonical_order() -> None:
    active = ActiveParty(
        "bank",
        np.asarray([[0.0], [1.0], [2.0], [3.0]]),
        np.asarray([0.0, 0.0, 1.0, 1.0]),
    )
    passive = PassiveParty("telecom", np.asarray([[30.0], [10.0], [20.0], [99.0]]))
    aligned_active, aligned_passive, ids = psi_align_parties(
        active,
        ["a", "b", "c", "d"],
        [passive],
        [["c", "a", "b", "x"]],
        psi_backend=ReverseIntersectionPSI(),
    )
    assert ids == ("a", "b", "c")
    assert aligned_active.n_rows == 3
    assert aligned_passive[0]._x[:, 0].tolist() == [10.0, 20.0, 30.0]


def test_protected_logistic_executes_psi_and_dp_residual_release() -> None:
    active = ActiveParty(
        "bank",
        np.arange(12, dtype=float).reshape(6, 2),
        np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
    )
    passive = PassiveParty(
        "telecom",
        np.asarray([[6.0], [5.0], [4.0], [3.0], [2.0], [1.0]]),
    )
    run = fit_protected_logistic(
        active,
        ["a", "b", "c", "d", "e", "f"],
        [passive],
        [["a", "b", "c", "d", "e", "f"]],
        psi_backend=ReverseIntersectionPSI(),
        clip_l2_norm=1.0,
        noise_multiplier=2.0,
        seed=5,
        model_kwargs={"max_iter": 2, "batch_size": 3, "learning_rate": 0.02},
    )
    report = run.privacy_report(delta=1e-6)
    residual = report["residual_release"]
    assert isinstance(residual, dict)
    assert int(residual["releases"]) > 0
    assert run.model.require_entity_ids is True
    assert run.model.missing_party_policy == "error"
