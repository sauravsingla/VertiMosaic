# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading

import numpy as np

from vertimosaic.models import VFLHistGBDT
from vertimosaic.parties import ActiveParty, PassiveParty, RemotePartyService, RemotePassiveParty
from vertimosaic.transport import RemoteHTTPTransport


def _dataset(rows: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    bank = rng.normal(size=(rows, 2))
    telecom = rng.normal(size=(rows, 3))
    score = 0.9 * bank[:, 0] - 0.6 * bank[:, 1] + 0.8 * telecom[:, 1]
    labels = (score + rng.normal(scale=0.35, size=rows) > 0.0).astype(float)
    return bank, telecom, labels


def test_remote_gbdt_matches_local_and_keeps_histogram_state_server_side() -> None:
    train_bank, train_telco, train_y = _dataset(128, 31)
    validation_bank, validation_telco, validation_y = _dataset(72, 47)

    local_train_passive = PassiveParty("telecom", train_telco.copy())
    local_validation_passive = PassiveParty("telecom", validation_telco.copy())
    service_train_passive = PassiveParty("telecom", train_telco.copy())
    service_validation_passive = PassiveParty("telecom", validation_telco.copy())
    service = RemotePartyService(
        service_train_passive,
        allowed_senders={"bank"},
        partitions={
            "train": service_train_passive,
            "validation": service_validation_passive,
        },
    )
    server = service.make_server(("127.0.0.1", 0), bearer_token="gbdt-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address

    rpc_transport = RemoteHTTPTransport(
        endpoints={"telecom": f"http://{host}:{port}/v1/messages"},
        bearer_tokens={"telecom": "gbdt-token"},
        allow_insecure_http=True,
        timeout_seconds=3.0,
    )
    remote_train = RemotePassiveParty.discover(
        "telecom",
        rpc_transport,
        partition="train",
    )
    remote_validation = RemotePassiveParty.discover(
        "telecom",
        rpc_transport,
        partition="validation",
    )
    assert not hasattr(remote_train, "_x")
    assert not hasattr(remote_validation, "_x")

    local_model = VFLHistGBDT(
        n_estimators=5,
        learning_rate=0.12,
        max_depth=2,
        min_samples_leaf=8,
        l2_leaf_reg=1.0,
        max_bins=8,
        early_stopping_rounds=3,
        seed=13,
    )
    remote_model = VFLHistGBDT(
        n_estimators=5,
        learning_rate=0.12,
        max_depth=2,
        min_samples_leaf=8,
        l2_leaf_reg=1.0,
        max_bins=8,
        early_stopping_rounds=3,
        seed=13,
    )

    local_train_active = ActiveParty("bank", train_bank.copy(), train_y.copy())
    local_validation_active = ActiveParty(
        "bank",
        validation_bank.copy(),
        validation_y.copy(),
    )
    remote_train_active = ActiveParty("bank", train_bank.copy(), train_y.copy())
    remote_validation_active = ActiveParty(
        "bank",
        validation_bank.copy(),
        validation_y.copy(),
    )

    try:
        local_model.fit(
            local_train_active,
            [local_train_passive],
            validation_active=local_validation_active,
            validation_passive=[local_validation_passive],
        )
        remote_model.fit(
            remote_train_active,
            [remote_train],  # type: ignore[list-item]
            validation_active=remote_validation_active,
            validation_passive=[remote_validation],  # type: ignore[list-item]
        )
        local_probability = local_model.predict_proba(
            [local_validation_active, local_validation_passive]
        )[:, 1]
        remote_probability = remote_model.predict_proba(
            [remote_validation_active, remote_validation]  # type: ignore[list-item]
        )[:, 1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3.0)

    assert len(remote_model.trees_) == len(local_model.trees_)
    assert np.allclose(
        remote_model.training_loss_history_,
        local_model.training_loss_history_,
        atol=1e-12,
    )
    assert np.allclose(
        remote_model.validation_loss_history_,
        local_model.validation_loss_history_,
        atol=1e-12,
    )
    assert np.allclose(remote_probability, local_probability, atol=1e-12)

    message_types = [event.message_type for event in rpc_transport.audit_log]
    assert "party.histogram.prepare" in message_types
    assert "party.histogram.set_signals" in message_types
    assert "party.histogram.candidates" in message_types
    assert "party.histogram.route" in message_types
    assert "party.histogram.share_state" in message_types
    assert message_types.count("party.histogram.set_signals") <= len(remote_model.trees_)
