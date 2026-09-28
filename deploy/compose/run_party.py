# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vertimosaic.parties import PassiveParty, RemotePartyService


def _features(role: str, rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocks = {
        "bank": rng.normal(size=(rows, 2)),
        "telecom": rng.normal(size=(rows, 3)),
        "insurance": rng.normal(size=(rows, 2)),
        "retail": rng.normal(size=(rows, 2)),
    }
    if role not in blocks:
        raise ValueError(f"unknown role: {role}")
    return blocks[role]


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic mTLS VertiMosaic passive service")
    parser.add_argument("--role", required=True, choices=["telecom", "insurance", "retail"])
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--token", required=True)
    parser.add_argument("--cert-dir", type=Path, default=Path("/certs"))
    args = parser.parse_args()

    party = PassiveParty(args.role, _features(args.role, args.rows, args.seed))
    service = RemotePartyService(party, allowed_senders={"bank"})
    server = service.make_server(("0.0.0.0", args.port), bearer_token=args.token)
    server.enable_mtls(
        server_cert=str(args.cert_dir / f"{args.role}.pem"),
        server_key=str(args.cert_dir / f"{args.role}-key.pem"),
        client_ca=str(args.cert_dir / "ca.pem"),
    )
    print(f"{args.role} ready on 0.0.0.0:{args.port} with mTLS")
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
