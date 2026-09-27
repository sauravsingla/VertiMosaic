# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

from vertimosaic.transport import ReferenceRelayServer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a VertiMosaic reference message relay")
    parser.add_argument("--role", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--bearer-token")
    parser.add_argument("--server-cert")
    parser.add_argument("--server-key")
    parser.add_argument("--client-ca")
    args = parser.parse_args()

    server = ReferenceRelayServer(
        (args.host, args.port),
        receiver_role=args.role,
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
    print(f"VertiMosaic relay for {args.role} listening on {args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
