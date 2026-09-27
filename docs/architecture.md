# Architecture

VertiMosaic models column-partitioned learning. Bank is the active party and owns the binary target; Telecom, Insurance, and Retail are passive parties that own complementary feature columns for aligned entities. Federated model code requests local computations from party objects rather than concatenating passive raw matrices.

## Transport boundary

All protocol message backends implement the structural `Transport` interface.

`InMemoryTransport` records only message metadata: type, sender/receiver roles, shapes, scalar counts, and estimated bytes. It never retains transmitted array values. It remains the default deterministic research/CI simulator.

`RemoteHTTPTransport` is the reference process/network backend. It provides versioned JSON serialization, receiver-specific endpoints, stable message IDs, retry/backoff, receiver-role validation, bearer authentication, optional client certificates, and HTTPS/mTLS support. `scripts/run_transport_relay.py` can be launched as a separate receiver process for each role. Plain HTTP is rejected by default and exists only as an explicit local-test option.

## Isolated remote passive parties

`RemotePartyService` and `RemotePassiveParty` provide a reference process/network boundary for both `VFLLogisticRegression` and `VFLHistGBDT`. The service process owns the party's raw feature matrix. The client-side proxy contains only the party name, partition name, row/feature counts, opaque histogram routing state, and a remote transport. Raw passive feature matrices and numeric tree thresholds are never returned to the coordinator.

The logistic RPC surface executes `local_logits` and `local_gradient` inside the remote service.

The histogram GBDT RPC surface executes all feature-dependent tree operations inside the remote service:

- training-derived quantile-bin preparation;
- opaque routing-state creation/export;
- one-per-round gradient/Hessian registration and server-side retention;
- candidate histogram aggregation over requested entity/feature references;
- aggregate split-statistic return with opaque feature/bin references only;
- selected-split resolution against private numeric thresholds;
- private split routing with only left/right entity indices returned;
- train-to-validation routing-state sharing within the same remote organization; and
- local split-importance aggregation by opaque feature reference.

Gradient/Hessian arrays remain documented target-derived leakage surfaces. To avoid retransmitting them for each node search, the client computes a deterministic round reference and uploads each distinct signal pair once. The service expires the previous cached signal state for that partition when the next round arrives. Candidate queries then refer to the cached round state by opaque reference.

A single service can own multiple row partitions for the same organization, such as `train` and `validation`. Training-derived histogram thresholds may be attached to the validation partition entirely inside that service. This mirrors the local-party invariant that held-out rows reuse training-derived thresholds rather than refitting bins.

A standalone reference process can be launched with:

```bash
python scripts/run_remote_party.py \
  --role telecom \
  --features /secure/local/telecom_train.npy \
  --validation-features /secure/local/telecom_validation.npy \
  --port 8443 \
  --bearer-token "$VERTIMOSAIC_TOKEN" \
  --server-cert /secure/tls/server.crt \
  --server-key /secure/tls/server.key \
  --client-ca /secure/tls/client-ca.crt
```

The same pattern can be used for Insurance and Retail with separate authenticated endpoints. The reference implementation demonstrates physical process/network isolation and protocol behavior; it is not a production service mesh, certificate authority, key-management system, PSI implementation, or cryptographic VFL protocol.

## Privacy boundary

Remote execution strengthens the raw-feature/process-isolation boundary but does **not** make the learning protocol cryptographically private. Residuals, logits, gradients, Hessians, entity membership, aggregate split statistics, opaque split usage, and routing information can still leak information. VertiMosaic therefore measures selected leakage surfaces empirically and continues to document cryptographic confidentiality, malicious-party security, collusion resistance, PSI, and formal differential privacy as non-guarantees unless separately implemented and validated.
