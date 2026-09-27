# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vertimosaic.parties import PassiveParty, RemotePartyService


def _load_features(path: Path, parser: argparse.ArgumentParser, flag: str) -> np.ndarray:
    features = np.load(path, allow_pickle=False)
    if features.ndim != 2:
        parser.error(f"{flag} must contain a 2D NumPy array")
    return np.asarray(features, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an isolated VertiMosaic passive-party computation service"
    )
    parser.add_argument("--role", required=True)
    parser.add_argument("--features", type=Path, required=True, help="2D training .npy matrix")
    parser.add_argument(
        "--validation-features",
        type=Path,
        help="optional 2D validation .npy matrix owned by the same remote organization",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--coordinator-role", default="bank")
    parser.add_argument("--bearer-token")
    parser.add_argument("--server-cert")
    parser.add_argument("--server-key")
    parser.add_argument("--client-ca")
    args = parser.parse_args()

    train_features = _load_features(args.features, parser, "--features")
    party = PassiveParty(args.role, train_features)
    partitions: dict[str, PassiveParty] = {"train": party}
    if args.validation_features is not None:
        validation_features = _load_features(
            args.validation_features,
            parser,
            "--validation-features",
        )
        if validation_features.shape[1] != train_features.shape[1]:
            parser.error("training and validation feature matrices must have equal width")
        partitions["validation"] = PassiveParty(args.role, validation_features)

    service = RemotePartyService(
        party,
        allowed_senders={args.coordinator_role},
        partitions=partitions,
    )
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

    partition_summary = ", ".join(
        f"{name}:{candidate.n_rows}x{candidate.n_features}"
        for name, candidate in partitions.items()
    )
    print(
        f"VertiMosaic remote party {args.role} listening on {args.host}:{args.port}; "
        f"partitions={partition_summary}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
