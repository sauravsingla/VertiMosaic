# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from vertimosaic.models import VFLLogisticRegression
from vertimosaic.parties import ActiveParty, RemotePassiveParty
from vertimosaic.transport import RemoteHTTPTransport


def _load(path: Path) -> np.ndarray:
    return np.asarray(np.load(path, allow_pickle=False), dtype=float)


def _discover(role: str, transport: RemoteHTTPTransport, timeout: float) -> RemotePassiveParty:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return RemotePassiveParty.discover(role, transport)
        except Exception as error:  # readiness retry for container startup only
            last_error = error
            time.sleep(0.5)
    raise RuntimeError(f"remote party {role!r} did not become ready") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Compose active-party coordinator")
    parser.add_argument("--bank-data", type=Path, default=Path("/bank-data"))
    parser.add_argument("--cert-dir", type=Path, default=Path("/certs"))
    parser.add_argument("--output", type=Path, default=Path("/output/compose-smoke.json"))
    parser.add_argument("--token", default="vertimosaic-demo-token")
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    args = parser.parse_args()

    roles = ("telecom", "insurance", "retail")
    transport = RemoteHTTPTransport(
        endpoints={role: f"https://{role}:8443/v1/messages" for role in roles},
        bearer_tokens={role: args.token for role in roles},
        ca_file=str(args.cert_dir / "ca.pem"),
        client_cert=str(args.cert_dir / "bank-client.pem"),
        client_key=str(args.cert_dir / "bank-client-key.pem"),
        timeout_seconds=5.0,
        max_retries=2,
        array_codec="npy-zlib-base64",
    )
    remote = [_discover(role, transport, args.startup_timeout) for role in roles]
    bank = ActiveParty(
        "bank",
        _load(args.bank_data / "features.npy"),
        _load(args.bank_data / "labels.npy"),
    )

    model = VFLLogisticRegression(
        learning_rate=0.04,
        max_iter=20,
        tolerance=0.0,
        gradient_clip=10.0,
        missing_party_policy="error",
        transport=transport,
        seed=42,
    )
    started = time.perf_counter()
    model.fit(bank, remote)  # type: ignore[arg-type]
    training_seconds = time.perf_counter() - started
    probability = model.predict_proba([bank, *remote])[:, 1]  # type: ignore[list-item]
    digest = hashlib.sha256(np.ascontiguousarray(probability).tobytes()).hexdigest()
    network_bytes = sum(event.measured_application_bytes for event in transport.network_audit_log)
    network_seconds = sum(event.round_trip_seconds for event in transport.network_audit_log)
    payload = {
        "rows": bank.n_rows,
        "seed": 42,
        "parties": [bank.name, *roles],
        "mtls": True,
        "bearer_auth": True,
        "array_codec": transport.array_codec,
        "training_seconds": training_seconds,
        "logical_messages": transport.message_count,
        "logical_payload_bytes": transport.estimated_payload_bytes,
        "measured_application_http_bytes": network_bytes,
        "summed_http_round_trip_seconds": network_seconds,
        "probability_sha256": digest,
        "probability_mean": float(probability.mean()),
        "non_guarantee": (
            "mTLS/authentication protect the transport channel; they do not make VFL "
            "residuals, logits or learned parameters cryptographically private"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
