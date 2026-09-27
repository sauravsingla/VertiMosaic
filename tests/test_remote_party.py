# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading

import numpy as np

from vertimosaic.models import VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty, RemotePartyService, RemotePassiveParty
from vertimosaic.transport import RemoteHTTPTransport


def test_remote_passive_party_keeps_features_server_side_and_matches_local_training() -> None:
    rng = np.random.default_rng(19)
    x_bank = rng.normal(size=(96, 2))
    x_telco = rng.normal(size=(96, 3))
    y = (0.7 * x_bank[:, 0] - 0.5 * x_telco[:, 1] > 0).astype(float)

    local_passive = PassiveParty("telecom", x_telco)
    service = RemotePartyService(local_passive, allowed_senders={"bank"})
    server = service.make_server(("127.0.0.1", 0), bearer_token="test-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address

    rpc_transport = RemoteHTTPTransport(
        endpoints={"telecom": f"http://{host}:{port}/v1/messages"},
        bearer_tokens={"telecom": "test-token"},
        allow_insecure_http=True,
        timeout_seconds=2.0,
    )
    remote_passive = RemotePassiveParty.discover("telecom", rpc_transport)
    assert remote_passive.n_rows == 96
    assert remote_passive.n_features == 3
    assert not hasattr(remote_passive, "_x")

    active_local = ActiveParty("bank", x_bank, y)
    active_remote = ActiveParty("bank", x_bank.copy(), y.copy())
    local_model = VFLLogisticRegression(
        learning_rate=0.04,
        max_iter=12,
        tolerance=0.0,
        gradient_clip=None,
        seed=5,
    )
    remote_model = VFLLogisticRegression(
        learning_rate=0.04,
        max_iter=12,
        tolerance=0.0,
        gradient_clip=None,
        seed=5,
    )
    try:
        local_model.fit(active_local, [local_passive])
        remote_model.fit(active_remote, [remote_passive])  # type: ignore[list-item]
        remote_probability = remote_model.predict_proba(
            [active_remote, remote_passive]  # type: ignore[list-item]
        )[:, 1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)

    assert np.allclose(remote_model.weights_["bank"], local_model.weights_["bank"], atol=1e-12)
    assert np.allclose(
        remote_model.weights_["telecom"], local_model.weights_["telecom"], atol=1e-12
    )
    assert np.isclose(remote_model.intercept_, local_model.intercept_, atol=1e-12)
    local_probability = local_model.predict_proba([active_local, local_passive])[:, 1]
    assert np.allclose(remote_probability, local_probability, atol=1e-12)
    assert rpc_transport.audit_log
