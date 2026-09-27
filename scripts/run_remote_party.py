# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vertimosaic.parties import PassiveParty, RemotePartyService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an isolated VertiMosaic passive-party computation service"
    )
    parser.add_argument("--role", required=True)
    parser.add_argument("--features", type=Path, required=True, help="2D .npy feature matrix")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--coordinator-role", default="bank")
    parser.add_argument("--bearer-token")
    parser.add_argument("--server-cert")
    parser.add_argument("--server-key")
    parser.add_argument("--client-ca")
    args = parser.parse_args()

    features = np.load(args.features, allow_pickle=False)
    if features.ndim != 2:
        parser.error("--features must contain a 2D NumPy array")
    party = PassiveParty(args.role, features)
    service = RemotePartyService(party, allowed_senders={args.coordinator_role})
    server = service.make_server(
        (args.host, args.port),
        bearer_token=args.bearer_token,
    )

    tls_values = (args.server_cert, args.server_key, args.client_ca)
    if any(tls_values) and not all(tls_values):
        parser.error("--server-cert, --server-key and --client-ca must be supplied together")
    if all(tls_values):
        server.enable_mtls(
            server_cert=args.server_cert,
            server_key=args.server_key,
            client_ca=args.client_ca,
        )

    print(
        f"VertiMosaic remote party {args.role} listening on {args.host}:{args.port}; "
        f"rows={party.n_rows}, features={party.n_features}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
