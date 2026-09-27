# Architecture

VertiMosaic models column-partitioned learning. Bank is the active party and owns the binary target; Telecom, Insurance, and Retail are passive parties that own complementary feature columns for aligned entities. Federated model code requests local computations from party objects rather than concatenating passive raw matrices.

## Transport boundary

All protocol message backends implement the structural `Transport` interface.

`InMemoryTransport` records only message metadata: type, sender/receiver roles, shapes, scalar counts, and estimated bytes. It never retains transmitted array values. It remains the default deterministic research/CI simulator.

`RemoteHTTPTransport` is the reference process/network backend. It provides versioned JSON serialization, receiver-specific endpoints, stable message IDs, retry/backoff, receiver-role validation, bearer authentication, optional client certificates, and HTTPS/mTLS support. `scripts/run_transport_relay.py` can be launched as a separate receiver process for each role. Plain HTTP is rejected by default and exists only as an explicit local-test option.

## Isolated remote passive party

`RemotePartyService` and `RemotePassiveParty` provide a reference for true passive-party process isolation for the logistic protocol. The service process owns the party's raw feature matrix. The client-side proxy contains only the party name, row/feature counts, and a remote transport. Calls to `local_logits` and `local_gradient` execute inside the remote party service; the raw feature matrix is never returned to the coordinator.

A standalone reference process can be launched with:

```bash
python scripts/run_remote_party.py \
  --role telecom \
  --features /secure/local/telecom_features.npy \
  --port 8443 \
  --bearer-token "$VERTIMOSAIC_TOKEN" \
  --server-cert /secure/tls/server.crt \
  --server-key /secure/tls/server.key \
  --client-ca /secure/tls/client-ca.crt
```

The initial remote-party RPC surface covers the local-logit and local-gradient operations required by `VFLLogisticRegression`. The histogram GBDT implementation continues to use party-local objects plus the transport message abstraction; extending the isolated RPC service to histogram construction/routing is a compatible future extension rather than an implied current guarantee.

The remote transport does **not** change the privacy threat model. Residuals, logits, gradients, Hessians, node membership, and routing messages can still leak information unless a stronger privacy mechanism is layered on top.

Neither the reference relay nor remote-party service is a production service mesh, key-management system, PSI implementation, or cryptographic VFL protocol.
