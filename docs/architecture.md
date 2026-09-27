# Architecture

VertiMosaic models column-partitioned learning. Bank is the active party and owns the binary target; Telecom, Insurance, and Retail are passive parties that own complementary feature columns for aligned entities. Federated model code requests local computations from party objects rather than concatenating passive raw matrices.

## Transport boundary

All protocol message backends implement the structural `Transport` interface.

`InMemoryTransport` records only message metadata: type, sender/receiver roles, shapes, scalar counts, and estimated bytes. It never retains transmitted array values. It remains the default deterministic research/CI simulator.

`RemoteHTTPTransport` is the reference process/network backend. It provides versioned JSON serialization, receiver-specific endpoints, stable message IDs, retry/backoff, receiver-role validation, bearer authentication, optional client certificates, and HTTPS/mTLS support. `scripts/run_transport_relay.py` can be launched as a separate receiver process for each role. Plain HTTP is rejected by default and exists only as an explicit local-test option.

The remote transport does **not** change the privacy threat model. Residuals, logits, gradients, Hessians, node membership, and routing messages can still leak information unless a stronger privacy mechanism is layered on top.

The reference relay demonstrates a real network/process boundary for protocol messages; it is not a production service mesh, key-management system, or cryptographic VFL protocol.
