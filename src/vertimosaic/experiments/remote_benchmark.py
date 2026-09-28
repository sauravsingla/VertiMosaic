# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np

from vertimosaic.transport import ReferenceRelayServer, RemoteHTTPTransport

_SECRET = b"vertimosaic-benchmark-signing-key-32"


def _relay_process(port_queue: Any) -> None:
    server = ReferenceRelayServer(
        ("127.0.0.1", 0),
        receiver_role="passive",
        sender_signing_keys={"active": {"benchmark-v1": _SECRET}},
        require_signed_messages=True,
    )
    host, port = server.server_address
    port_queue.put((host, port))
    server.serve_forever(poll_interval=0.05)


def run_remote_transport_benchmark(
    *,
    messages: int = 20,
    vector_size: int = 4096,
    seed: int = 42,
    array_codec: str = "npy-zlib-base64",
    output: Path = Path("benchmarks/remote_transport.json"),
) -> dict[str, Any]:
    """Measure the real serialized HTTP path using a separate relay process."""
    if messages <= 0 or vector_size <= 0:
        raise ValueError("messages and vector_size must be positive")
    context = mp.get_context("spawn")
    port_queue = context.Queue()
    process = context.Process(target=_relay_process, args=(port_queue,), daemon=True)
    process.start()
    try:
        try:
            host, port = port_queue.get(timeout=10.0)
        except Empty as exc:
            raise RuntimeError("relay process did not publish its listening port") from exc
        endpoint = f"http://{host}:{port}/v1/messages"
        transport = RemoteHTTPTransport(
            endpoints={"passive": endpoint},
            signing_keys={"passive": ("benchmark-v1", _SECRET)},
            allow_insecure_http=True,
            array_codec=array_codec,
            max_retries=0,
        )
        rng = np.random.default_rng(seed)
        started = time.perf_counter()
        for step in range(messages):
            value = rng.normal(size=vector_size).astype(np.float64)
            response = transport.send(
                value,
                message_type="benchmark_vector",
                sender_role="active",
                receiver_role="passive",
                stage="remote_transport_benchmark",
                step=step,
            )
            if not np.array_equal(response, value):
                raise RuntimeError("remote benchmark relay changed payload values")
        elapsed = time.perf_counter() - started
        network = transport.network_totals()
        payload_bytes = int(transport.estimated_payload_bytes)
        result: dict[str, Any] = {
            "benchmark": "separate_process_remote_transport",
            "messages": messages,
            "vector_size": vector_size,
            "dtype": "float64",
            "array_codec": array_codec,
            "elapsed_seconds": elapsed,
            "logical_protocol_payload_bytes": payload_bytes,
            "network": network,
            "process_isolation": "relay runs in a separate spawned process",
            "transport": "loopback HTTP for reproducible measurement",
            "tls_note": (
                "TLS record bytes are not exposed by urllib and are not estimated. "
                "HTTPS/mTLS behavior is covered separately by transport configuration/tests."
            ),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5.0)
        port_queue.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark VertiMosaic's serialized remote path.")
    parser.add_argument("--messages", type=int, default=20)
    parser.add_argument("--vector-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--array-codec",
        choices=["json", "npy-zlib-base64"],
        default="npy-zlib-base64",
    )
    parser.add_argument("--output", type=Path, default=Path("benchmarks/remote_transport.json"))
    args = parser.parse_args()
    result = run_remote_transport_benchmark(
        messages=args.messages,
        vector_size=args.vector_size,
        seed=args.seed,
        array_codec=args.array_codec,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
